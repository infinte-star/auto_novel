"""Rebuild memory files from Ch1-76 for a clean restart.

Reads all chapter text + prompt.md, calls LLM to regenerate each memory
file with content only from Ch1-76 (no spoilers from deleted chapters).

Usage:
    python tools/rebuild_memory.py --novel tangshuting --apply
    python tools/rebuild_memory.py --novel tangshuting --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _setup_env(novel_name: str):
    novel_dir = ROOT / "novels" / novel_name
    os.environ["NOVEL_CONFIG"] = str(novel_dir / "config.yaml")
    os.environ["NOVEL_PROMPT"] = str(novel_dir / "prompt.md")


def _load_chapters(chapters_dir: Path, end: int = 76) -> str:
    parts = []
    for i in range(1, end + 1):
        f = chapters_dir / f"{i:04d}.md"
        if f.exists():
            parts.append(f.read_text(encoding="utf-8").strip())
    return "\n\n---\n\n".join(parts)


MEMORY_SPECS = {
    "bible.md": {
        "label": "世界观圣经",
        "prompt": (
            "根据以下小说正文（第1-76章）和创作大纲，生成世界观圣经文件。\n\n"
            "要求：\n"
            "1. 只包含第1-76章已确立的事实\n"
            "2. 涵盖：故事背景、地理设定、社会规则、重要组织/机构\n"
            "3. 不包含任何失明/盲杖/冷库酷刑/排爆/追车等内容\n"
            "4. 如果Ch1-76中有轻度犯罪伏笔（TSY编号、铁箱、虚假推广调查），保留但定性为'轻悬疑调味'\n"
            "5. 格式用markdown标题分节，简洁精炼\n"
        ),
    },
    "characters.md": {
        "label": "人物状态",
        "prompt": (
            "根据以下小说正文（第1-76章），生成人物状态文件。\n\n"
            "要求：\n"
            "1. 列出所有重要角色，截止第76章的状态\n"
            "2. 每个角色：姓名、身份、与主角关系、当前情感状态、关键标签\n"
            "3. 特别标注主角团：汤舒婷、陆时砚、以及其他核心角色\n"
            "4. 不包含Ch77后的任何剧情发展\n"
            "5. 不涉及失明/致残/酷刑等情节\n"
        ),
    },
    "timeline.md": {
        "label": "时间线",
        "prompt": (
            "根据以下小说正文（第1-76章），生成时间线文件。\n\n"
            "要求：\n"
            "1. 按时间顺序列出Ch1-76的关键事件\n"
            "2. 格式：## ChN-M: 事件概述，然后列出具体节点\n"
            "3. 标注哪些线索已关闭，哪些仍然开放\n"
            "4. 不包含Ch77后的事件\n"
        ),
    },
    "threads.md": {
        "label": "开放线索",
        "prompt": (
            "根据以下小说正文（第1-76章），生成开放线索文件。\n\n"
            "要求：\n"
            "1. 列出截止Ch76所有未解决的悬念/伏笔/线索\n"
            "2. 每条线索：描述、引入章节、当前状态、预期解决方向\n"
            "3. 将线索分为：主线、支线、角色线\n"
            "4. 标注解决方向时，必须符合女频甜宠言情体裁\n"
            "5. 犯罪/调查线索的解决方向限定为：商业欺诈/合同纠纷/证据调查/法律手段\n"
        ),
    },
    "volume_plan.md": {
        "label": "卷轴规划",
        "prompt": (
            "根据以下创作大纲和前76章剧情进展，重新生成后续卷轴规划。\n\n"
            "要求：\n"
            "1. 确认Ch1-76已覆盖的剧情阶段\n"
            "2. 规划Ch77-200的整体走向\n"
            "3. 核心基调：女频甜宠言情+美食探店\n"
            "4. 硬约束：\n"
            "   - 女主不受永久性身体伤害\n"
            "   - 无追车枪战/排爆/冷库酷刑/绑架劫持\n"
            "   - 暴力限于推搡口角层面\n"
            "   - 犯罪元素以商业欺诈/法律调查为主\n"
            "   - 每5章至少3章以美食/甜宠/修罗场/事业/日常为主场景\n"
            "5. 按10-20章为一卷划分阶段\n"
        ),
    },
}


def rebuild_memory(
    novel_dir: Path, client, paths, config, dry_run: bool = True
):
    from llm import call_llm

    chapters_dir = novel_dir / "chapters"
    memory_dir = novel_dir / "memory"
    prompt_file = novel_dir / "prompt.md"

    prompt_text = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    chapter_text = _load_chapters(chapters_dir, 76)
    total_chars = len(chapter_text)
    print(f"Loaded {total_chars} chars from Ch1-76")

    if total_chars > 200000:
        print("Chapter text is very long, will use abbreviated version for each call")
        abbreviated = chapter_text[:80000] + "\n\n...(中间省略)...\n\n" + chapter_text[-80000:]
    else:
        abbreviated = chapter_text

    for filename, spec in MEMORY_SPECS.items():
        filepath = memory_dir / filename
        print(f"\n--- {spec['label']} ({filename}) ---")

        if dry_run:
            print(f"  [DRY-RUN] Would regenerate {filepath}")
            continue

        system = (
            "你是一位专业的小说编辑助手。根据提供的小说正文和大纲，生成指定的记忆文件。"
            "输出纯markdown格式，不要代码围栏。"
        )
        user = (
            f"{spec['prompt']}\n\n"
            f"## 创作大纲\n{prompt_text[:10000]}\n\n"
            f"## 小说正文（第1-76章）\n{abbreviated}\n"
        )

        try:
            result = call_llm(
                client, paths, config,
                system=system,
                user=user,
                max_tokens=4000,
                temperature=0.3,
                tag="rebuild_memory",
            )
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result
                if result.endswith("```"):
                    result = result[:-3].rstrip()

            backup = filepath.with_suffix(".md.bak")
            if filepath.exists():
                filepath.rename(backup)
                print(f"  Backed up {filename} → {backup.name}")

            filepath.write_text(result, encoding="utf-8")
            print(f"  [DONE] Written {len(result)} chars to {filename}")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    voice_baseline = memory_dir / "voice_baseline.md"
    voice_file = memory_dir / "voice.md"
    if voice_baseline.exists() and not dry_run:
        import shutil
        shutil.copy2(voice_baseline, voice_file)
        print(f"\n  [DONE] Copied voice_baseline.md → voice.md")


def main():
    parser = argparse.ArgumentParser(description="Rebuild memory files from Ch1-76")
    parser.add_argument("--novel", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Error: specify --dry-run or --apply")
        sys.exit(1)

    novel_dir = ROOT / "novels" / args.novel
    if not novel_dir.exists():
        print(f"Error: {novel_dir} not found")
        sys.exit(1)

    if args.apply:
        _setup_env(args.novel)
        sys.path.insert(0, str(ROOT))
        from config import load_config, get_paths, configured_api_endpoints
        from llm import LLMClientPool
        from openai import OpenAI

        config = load_config()
        paths = get_paths(config)
        endpoints, primary_count = configured_api_endpoints(config)
        default_headers = {}
        user_agent = str(config["api"].get("user_agent", "")).strip()
        if user_agent:
            default_headers["User-Agent"] = user_agent
        clients = [
            OpenAI(base_url=base_url, api_key=api_key,
                   default_headers=default_headers or None)
            for base_url, api_key in endpoints
        ]
        client = (
            LLMClientPool(clients, primary_count, endpoints=endpoints)
            if len(clients) > 1 else clients[0]
        )
    else:
        client = paths = config = None

    rebuild_memory(novel_dir, client, paths, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
