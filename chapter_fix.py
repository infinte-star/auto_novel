"""Targeted chapter fixes for finished novels — LLM-based rewrites.

Fixes specific problems in individual chapters: low dialogue, short length,
monotonous endings. Reads from chapters_fixed/ (or chapters_refined/), writes
back in-place. Each chapter is one LLM call with problem-specific instructions.

Usage: called programmatically or via novel.py (future integration).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent


def _read_config(name: str) -> dict[str, Any]:
    from config import load_config
    return load_config()


def _get_client(config: dict[str, Any]):
    from openai import OpenAI
    api = config["api"]
    return OpenAI(
        base_url=api["base_url"],
        api_key=api["api_key"],
        default_headers={"User-Agent": api.get("user_agent", "")},
    )


def _call_llm(client, model: str, system: str, user: str, max_tokens: int = 8192) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True,
            )
            chunks = []
            for chunk in resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
            text = "".join(chunks).strip()
            if len(text) > 200:
                return text
            print(f"    [warn] short response ({len(text)} chars), retry {attempt+1}")
        except Exception as e:
            print(f"    [error] {e}, retry {attempt+1}")
            time.sleep(5)
    return ""


def _dialogue_ratio(text: str) -> float:
    spans = re.findall(r'“([^”]*?)”', text)
    dlg = sum(len(s) for s in spans)
    return dlg / len(text) if text else 0


def _get_context_chapters(chapters_dir: Path, ch_num: int, window: int = 2) -> str:
    """Read surrounding chapters for continuity context."""
    parts = []
    for n in range(max(1, ch_num - window), ch_num):
        p = chapters_dir / f"{n:04d}.md"
        if p.exists():
            t = p.read_text("utf-8", errors="replace")
            parts.append(f"--- 第{n}章（前文摘要，最后500字） ---\n{t[-500:]}")
    for n in range(ch_num + 1, ch_num + window + 1):
        p = chapters_dir / f"{n:04d}.md"
        if p.exists():
            t = p.read_text("utf-8", errors="replace")
            parts.append(f"--- 第{n}章（后文摘要，前300字） ---\n{t[:300]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fix type: dialogue injection
# ---------------------------------------------------------------------------

DIALOGUE_SYSTEM = """你是一位中文网文精修编辑，专长是将叙述过重的章节改写为对话驱动的场景。

核心原则：
1. 保留原文的全部情节、信息传递和人物关系变化，不增不减
2. 将叙述性的心理独白、旁白解释、信息传递改写为角色间的对话交锋
3. 对话必须有交锋感——不是你一句我一句的信息交换，而是有情绪张力、潜台词、话外音
4. 保留原文的叙事声音和文风，不要变成另一种风格
5. 对话使用中文双引号""
6. 输出完整章节（包含章节标题），不要输出分析或说明"""

DIALOGUE_USER = """这一章对话占比仅 {ratio:.0%}，远低于目标 20%。请精修这一章，将部分叙述段落改写为对话场景。

要求：
- 对话占比提升到 15-25%（不要过度，保持叙述与对话的平衡）
- 至少增加2-3组有效对话（每组≥3轮交锋）
- 将"她想到…"/"他意识到…"等内心独白转化为角色间的对话
- 关键信息传递从旁白叙述改为对话中自然展现
- 保持原文字数（±10%），不要大幅缩短或膨胀
- 保留所有原文的情节推进和信息点

{context}

--- 待精修章节 ---
{chapter}"""


# ---------------------------------------------------------------------------
# Fix type: chapter expansion
# ---------------------------------------------------------------------------

EXPAND_SYSTEM = """你是一位中文网文精修编辑，专长是将过于压缩的章节扩展为丰满的场景。

核心原则：
1. 保留原文全部情节和信息，不删不改，只在原有节拍之间填充
2. 扩展方向：场景环境细节、人物微表情/小动作、对话交锋、感官描写
3. 不引入新情节、新人物、新信息；只让已有的内容落地更充分
4. 保留原文的文风和叙事节奏
5. 输出完整章节（包含章节标题），不要输出分析或说明"""

EXPAND_USER = """这一章仅 {length} 字，低于目标 2800 字。请扩写到 2800-3500 字。

扩写方向（按优先级）：
1. 关键场景增加动作细节和环境互动（不是堆砌修辞，而是动作链条）
2. 已有对话前后补充表情/动作/停顿（"他说"→ 具体的说话时动作 + 台词）
3. 信息传递段落展开为场景呈现而非叙述总结
4. 转场处补充过渡感官细节（视觉/听觉/温度/气味）

禁止：
- 不要加入与原文无关的新剧情
- 不要堆砌形容词和比喻
- 不要在已经饱满的段落上硬加描写

{context}

--- 待扩写章节（{length}字）---
{chapter}"""


# ---------------------------------------------------------------------------
# Fix type: ending diversification
# ---------------------------------------------------------------------------

ENDING_SYSTEM = """你是一位中文网文章末钩子专家。
只重写提供的章节末尾段落（约200-400字），将叙述性收尾改写为更有悬念感的结尾。

约束：
- 只输出替换后的新结尾段落（200-400字），不要输出前文
- 新结尾必须与后续章节内容衔接
- 保持与原文相同的人物状态和场景"""

ENDING_USER = """本章以叙述方式收尾，缺乏钩子。请重写下面的结尾段落，改为以下手法之一：
- 对话收尾：以一句意味深长的台词结束
- 疑问悬念：抛出一个具体的、未解决的新问题
- 场景突变：一个打断当前状态的突发事件
- 信息炸弹：一条改变局面的新信息出现

只输出新的结尾段落（200-400字），不要输出完整章节。

{context}

--- 本章前文（最后800字，供理解上下文）---
{head}

--- 需要重写的结尾段落 ---
{tail}"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fix_chapters(
    name: str,
    fix_plan: dict[str, Any],
    batch: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fix chapters according to the plan."""
    import os
    os.environ["NOVEL_CONFIG"] = f"novels/{name}/config.yaml"
    os.environ["NOVEL_PROMPT"] = f"novels/{name}/prompt.md"

    config = _read_config(name)
    client = _get_client(config)
    model = config["api"]["model"]
    novel_dir = PROJECT_DIR / "novels" / name

    chapters_dir = novel_dir / "chapters_fixed"
    if not chapters_dir.exists() or not list(chapters_dir.glob("*.md")):
        chapters_dir = novel_dir / "chapters_refined"
    if not chapters_dir.exists():
        chapters_dir = novel_dir / "chapters"

    print(f"[chapter_fix] source: {chapters_dir}")
    print(f"[chapter_fix] model: {model}")
    print(f"[chapter_fix] batch: {batch}")

    all_fixes = fix_plan.get("all_fixes", {})
    results = {"fixed": 0, "skipped": 0, "failed": 0, "details": {}}

    # Filter by batch
    if batch == "dialogue":
        targets = {ch: f for ch, f in all_fixes.items()
                   if any(x.startswith("dialogue") for x in f)}
    elif batch == "short":
        targets = {ch: f for ch, f in all_fixes.items()
                   if any(x.startswith("short") for x in f)}
    elif batch == "ending":
        targets = {ch: f for ch, f in all_fixes.items()
                   if any(x.startswith("ending") for x in f)}
    elif batch == "multi":
        targets = {ch: f for ch, f in all_fixes.items() if len(f) >= 2}
    else:
        targets = all_fixes

    print(f"[chapter_fix] {len(targets)} chapters to fix")

    for ch_str, fixes in sorted(targets.items(), key=lambda x: int(x[0])):
        ch_num = int(ch_str)
        ch_path = chapters_dir / f"{ch_num:04d}.md"
        if not ch_path.exists():
            print(f"  Ch{ch_num}: SKIP (file not found)")
            results["skipped"] += 1
            continue

        text = ch_path.read_text("utf-8", errors="replace")
        context = _get_context_chapters(chapters_dir, ch_num)
        original_len = len(text)
        original_dlg = _dialogue_ratio(text)

        # Determine primary fix type (prioritize: dialogue > short > ending)
        has_dlg = any(x.startswith("dialogue") for x in fixes)
        has_short = any(x.startswith("short") for x in fixes)
        has_ending = any(x.startswith("ending") for x in fixes)

        # For multi-problem chapters, combine instructions
        if has_dlg and has_short:
            fix_type = "dialogue+expand"
            system = DIALOGUE_SYSTEM
            user = DIALOGUE_USER.format(
                ratio=original_dlg,
                context=f"前后文参考：\n{context}" if context else "",
                chapter=text,
            )
            user += f"\n\n额外要求：本章仅{original_len}字，精修后至少达到2800字。在增加对话的同时补充场景细节。"
        elif has_dlg:
            fix_type = "dialogue"
            system = DIALOGUE_SYSTEM
            user = DIALOGUE_USER.format(
                ratio=original_dlg,
                context=f"前后文参考：\n{context}" if context else "",
                chapter=text,
            )
        elif has_short:
            fix_type = "expand"
            system = EXPAND_SYSTEM
            user = EXPAND_USER.format(
                length=original_len,
                context=f"前后文参考：\n{context}" if context else "",
                chapter=text,
            )
        elif has_ending:
            fix_type = "ending"
            system = ENDING_SYSTEM
            # Split into head + last paragraph for efficient ending rewrite
            split_pos = text.rfind("\n\n")
            if split_pos == -1:
                split_pos = len(text) - 400
            _head_part = text[:split_pos]
            _tail_part = text[split_pos:]
            user = ENDING_USER.format(
                context=f"后续章节参考：\n{context}" if context else "",
                head=_head_part[-800:],
                tail=_tail_part,
            )
        else:
            continue

        print(f"  Ch{ch_num} [{fix_type}] {original_len}字 dlg={original_dlg:.0%} ...", end=" ", flush=True)

        if dry_run:
            print("(dry run)")
            continue

        _max_tok = 2048 if fix_type == "ending" else 12000
        fixed = _call_llm(client, model, system, user, max_tokens=_max_tok)
        if not fixed:
            print("FAILED (empty response)")
            results["failed"] += 1
            continue

        # For ending fixes, stitch the new ending back onto the head
        if fix_type == "ending":
            fixed = _head_part + "\n\n" + fixed.strip()

        new_len = len(fixed)
        new_dlg = _dialogue_ratio(fixed)

        # Validation
        ok = True
        if fix_type in ("expand", "dialogue+expand") and new_len < original_len * 0.9:
            print(f"REJECT (shrunk {original_len}→{new_len})")
            ok = False
        if fix_type == "dialogue" and new_len < original_len * 0.8:
            print(f"REJECT (dialogue fix shrunk {original_len}→{new_len})")
            ok = False
        if fix_type == "ending" and new_len < original_len * 0.7:
            print(f"REJECT (shrunk too much {original_len}→{new_len})")
            ok = False
        if new_len < 500:
            print(f"REJECT (too short {new_len})")
            ok = False

        if ok:
            ch_path.write_text(fixed, encoding="utf-8")
            print(f"OK {original_len}→{new_len}字 dlg={original_dlg:.0%}→{new_dlg:.0%}")
            results["fixed"] += 1
            results["details"][ch_num] = {
                "type": fix_type,
                "len": f"{original_len}→{new_len}",
                "dlg": f"{original_dlg:.1%}→{new_dlg:.1%}",
            }
        else:
            results["failed"] += 1

    print(f"\n[chapter_fix] done: {results['fixed']} fixed, {results['failed']} failed, {results['skipped']} skipped")
    return results


def _rebuild_book(name: str):
    """Rebuild book_fixed.md from chapters_fixed/."""
    chapters_dir = PROJECT_DIR / "novels" / name / "chapters_fixed"
    parts = []
    for p in sorted(chapters_dir.glob("*.md")):
        parts.append(p.read_text("utf-8", errors="replace"))
    book = PROJECT_DIR / "novels" / name / "book_fixed.md"
    book.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"[chapter_fix] rebuilt {book} ({len(parts)} chapters, {book.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tangshuting"
    batch = sys.argv[2] if len(sys.argv) > 2 else "all"
    dry = "--dry-run" in sys.argv

    plan_path = PROJECT_DIR / "novels" / name / "logs" / "_fix_plan.json"
    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found. Run analysis first.")
        sys.exit(1)

    plan = json.loads(plan_path.read_text("utf-8"))
    results = fix_chapters(name, plan, batch=batch, dry_run=dry)

    if not dry and results["fixed"] > 0:
        _rebuild_book(name)
