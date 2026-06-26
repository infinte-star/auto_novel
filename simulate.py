"""Style simulation: analyze reference texts to extract a reusable style profile.

The profile captures prose structure, dialogue patterns, sentence rhythm,
signature devices, and anti-patterns from one or more reference texts. It is
saved as `memory/style_profile.json` in the novel directory and optionally
injected into the writer prompt as a stylistic directive block.

CLI entry points:
    python novel.py import-style <name> <sources...> [--merge]
    python novel.py simulate <name> [--prompt "..."] [--tokens N]
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Paths, log, read_text, write_text

STYLE_ANALYSIS_SYSTEM = """\
你是一位专业的文学风格分析师，擅长从小说样本中提取可复用的写作风格特征。
你的输出是结构化 JSON，供 AI 写手在创作时参照。

分析维度：
1. prose_structure — 段落组织、叙述结构、场景切换方式
2. dialogue_style — 对话风格、动作描写与对话的比例、对话节奏
3. sentence_rhythm — 句子长短节奏、标点使用特征、快慢切换规律
4. pov_and_tense — 人称视角、时态、叙述距离
5. imagery — 意象偏好（体感/视觉/嗅觉等）、比喻风格
6. pacing — 每章事件密度、悬念节奏、高潮分布
7. signature_devices — 标志性手法（如反讽、环境映射、信息差叙事等）
8. anti_patterns — 该风格明确避免的写法
9. hook_types — 章末钩子/悬念的典型类型
10. vocabulary_notes — 用词风格（口语/书面/古风/现代等）

## 强制 JSON 输出格式
```json
{
  "prose_structure": "...",
  "dialogue_style": "...",
  "sentence_rhythm": "...",
  "pov_and_tense": "...",
  "imagery": "...",
  "pacing": "...",
  "signature_devices": ["...", "..."],
  "anti_patterns": ["...", "..."],
  "hook_types": ["...", "..."],
  "vocabulary_notes": "..."
}
```
每个字符串字段 80-200 字，数组字段 3-6 项。只输出 JSON，不要解释。"""

STYLE_MERGE_SYSTEM = """\
你是文学风格分析师。下面给出多份独立的风格分析结果，请合并为一份统一的风格档案。
合并原则：共性特征优先；冲突的特征取出现频率最高的；保留所有 anti_patterns。

## 强制 JSON 输出格式
与输入相同的 JSON 结构。只输出 JSON，不要解释。"""

PROFILE_FILE = "style_profile.json"
CHUNK_SIZE = 8000


def _profile_path(paths: Paths) -> Path:
    return paths.voice.parent / PROFILE_FILE


def analyze_samples(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    sample_paths: list[Path],
) -> dict[str, Any]:
    """Read reference files, analyze each via LLM, merge into a unified profile."""
    from llm import call_llm, load_json_with_repair

    profiles: list[dict[str, Any]] = []
    source_names: list[str] = []

    for sp in sample_paths:
        text = sp.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            log(paths, f"Style analysis: skipping empty file {sp.name}")
            continue
        source_names.append(sp.name)
        if len(text) > CHUNK_SIZE:
            chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            chunk_profiles = []
            for idx, chunk in enumerate(chunks[:5]):
                user = f"分析以下文本片段（{sp.name} 第{idx+1}/{min(len(chunks),5)}段）的写作风格：\n\n{chunk}"
                raw = call_llm(client, paths, config, STYLE_ANALYSIS_SYSTEM, user,
                               temperature=0.3, tag="plan_candidate")
                parsed = load_json_with_repair(client, paths, config, raw, fallback={})
                if parsed:
                    chunk_profiles.append(parsed)
            if len(chunk_profiles) > 1:
                merge_user = "请合并以下风格分析结果：\n\n" + "\n\n---\n\n".join(
                    json.dumps(p, ensure_ascii=False, indent=2) for p in chunk_profiles
                )
                raw = call_llm(client, paths, config, STYLE_MERGE_SYSTEM, merge_user,
                               temperature=0.2, tag="plan_candidate")
                merged = load_json_with_repair(client, paths, config, raw, fallback={})
                profiles.append(merged or chunk_profiles[0])
            elif chunk_profiles:
                profiles.append(chunk_profiles[0])
        else:
            user = f"分析以下文本（{sp.name}）的写作风格：\n\n{text}"
            raw = call_llm(client, paths, config, STYLE_ANALYSIS_SYSTEM, user,
                           temperature=0.3, tag="plan_candidate")
            parsed = load_json_with_repair(client, paths, config, raw, fallback={})
            if parsed:
                profiles.append(parsed)

    if not profiles:
        return {}

    if len(profiles) == 1:
        result = profiles[0]
    else:
        merge_user = "请合并以下多部作品的风格分析结果为统一风格档案：\n\n" + "\n\n---\n\n".join(
            json.dumps(p, ensure_ascii=False, indent=2) for p in profiles
        )
        raw = call_llm(client, paths, config, STYLE_MERGE_SYSTEM, merge_user,
                       temperature=0.2, tag="plan_candidate")
        result = load_json_with_repair(client, paths, config, raw, fallback=profiles[0])

    result["source_files"] = source_names
    result["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
    return result


def save_style_profile(paths: Paths, profile: dict[str, Any]) -> Path:
    out = _profile_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_style_profile(paths: Paths) -> dict[str, Any] | None:
    p = _profile_path(paths)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def style_profile_block(paths: Paths, config: dict[str, Any], max_chars: int | None = None) -> str:
    """Render the style profile as a writer prompt injection block."""
    profile = load_style_profile(paths)
    if not profile:
        return ""
    if max_chars is None:
        max_chars = int(config.get("novel", {}).get("style_profile_max_chars", 2000))

    lines = ["## 风格模拟档案（补充 voice.md，不替代）\n"]
    field_labels = {
        "prose_structure": "散文结构",
        "dialogue_style": "对话风格",
        "sentence_rhythm": "句子节奏",
        "pov_and_tense": "视角与时态",
        "imagery": "意象偏好",
        "pacing": "节奏特征",
        "vocabulary_notes": "用词风格",
    }
    for key, label in field_labels.items():
        val = profile.get(key, "")
        if val:
            lines.append(f"**{label}**: {val}")

    list_labels = {
        "signature_devices": "标志手法",
        "anti_patterns": "明确避免",
        "hook_types": "钩子类型",
    }
    for key, label in list_labels.items():
        items = profile.get(key, [])
        if isinstance(items, list) and items:
            lines.append(f"**{label}**: {'、'.join(str(i) for i in items)}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0] + "\n…[截断]"
    return result


def generate_sample_text(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    prompt: str = "写一段约500字的示范段落，展示该风格特征",
    max_tokens: int = 4000,
) -> str:
    """Generate a sample passage using the loaded style profile."""
    from llm import call_llm

    profile = load_style_profile(paths)
    if not profile:
        return "[错误] 未找到风格档案，请先运行 import-style 命令。"

    profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
    system = (
        "你是一位技艺精湛的小说作者。请严格按照下方风格档案的特征来创作。\n\n"
        f"## 风格档案\n{profile_text}"
    )
    return call_llm(client, paths, config, system, prompt,
                    max_tokens=max_tokens, temperature=0.7, tag="write")
