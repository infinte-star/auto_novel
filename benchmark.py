from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import ROOT, Paths, read_text


PLATFORM_GUIDANCE = {
    "qidian_male": (
        "平台画像：起点男频。读者更接受长线设定、体系成长、伏笔回收与阶段性大高潮；"
        "前三章仍需清晰卖点，但可以保留中长线谜团。每章要有明确推进，避免纯解释设定。"
    ),
    "fanqie_free": (
        "平台画像：免费阅读强留存。读者决策快，开头必须强冲突、强情绪、强可懂卖点；"
        "短周期爽点密度要高，尽量少慢热铺设定，章末问题必须直观。"
    ),
    "jinjiang_female": (
        "平台画像：晋江/女频。关系张力、人设化学反应、情绪递进与角色能动性优先；"
        "冲突要落到人物选择和关系变化，避免只有外部事件推动。"
    ),
    "qimao_free": (
        "平台画像：免费阅读泛用户。节奏要直给，冲突和收益要低门槛可理解；"
        "每章都应有明显情绪收益或悬念推进，减少复杂专名堆叠。"
    ),
    "general": (
        "平台画像：通用网文。优先保证开篇卖点清晰、章节推进稳定、承诺及时兑现、重复模式不过度。"
    ),
}


def platform_guidance(config: dict[str, Any]) -> str:
    preset = str(config.get("novel", {}).get("platform_preset", "general")).strip() or "general"
    return PLATFORM_GUIDANCE.get(preset, PLATFORM_GUIDANCE["general"])


# Platform-specific GOLDEN-FIRST-CHAPTERS rules. The generic OPENING_RULES_BLOCK
# (writing.py) covers craft fundamentals; this layer encodes how each platform's
# audience decides whether to keep reading, which differs sharply (免费流要前几行
# 见冲突；起点男频容忍中线铺垫但要亮体系卖点). Injected into both the opening
# plan prompt and the opening writer prompt for chapter_num <= opening_chapters.
PLATFORM_OPENING = {
    "qidian_male": (
        "## 平台开篇专项（起点男频·黄金三章）\n"
        "- 卖点亮相：前 1/3 必须让读者看清本书的体系/金手指/主角核心反差，可保留中长线谜团但不能让首章只铺设定。\n"
        "- 主角立住：首章用一次可见的选择或冲突展示主角性格与处境，给出明确的长期目标方向。\n"
        "- 钩子：章末留一个能支撑追读的具体悬念或升级预期，而非泛泛的'命运改变'。\n"
        "- 信息克制：世界观边演边给，禁止整段名词/设定倾倒。"
    ),
    "fanqie_free": (
        "## 平台开篇专项（番茄免费·强留存）\n"
        "- 极速入戏：正文前 300 字内必须出现强冲突或强情绪事件，禁止任何回忆/铺垫/设定/天气开场。\n"
        "- 低门槛卖点：本书爽点方向要一句话能懂，主角的处境与诉求直给，不绕弯。\n"
        "- 高密度钩子：首章至少 2 个让人想往下翻的悬念或反转点，章末问题必须直观强烈。\n"
        "- 情绪优先：让读者第一时间产生爽/愤/好奇/揪心中的一种强情绪。"
    ),
    "qimao_free": (
        "## 平台开篇专项（七猫免费·泛用户）\n"
        "- 直给冲突：开篇即抛出低门槛、可秒懂的冲突与利害关系，少用复杂专名。\n"
        "- 即时收益：首章就给读者一个明显的情绪收益或悬念推进，不慢热。\n"
        "- 主角动机清晰：让读者立刻明白主角要什么、被什么逼着走。\n"
        "- 章末强钩：用一个直观的危机或反转收尾。"
    ),
    "jinjiang_female": (
        "## 平台开篇专项（晋江女频·关系驱动）\n"
        "- 关系张力前置：首章即建立男女主（或核心关系）的化学反应或张力锚点，哪怕只是一次有潜台词的交锋。\n"
        "- 人设化学反应：通过具体互动展现人物魅力与反差，而非旁白介绍。\n"
        "- 情绪锚：给读者一个明确的情绪投射点（心动/意难平/好奇/护短欲）。\n"
        "- 能动性：女主在首章要有自己的选择与诉求，不做纯被动工具。"
    ),
    "general": (
        "## 平台开篇专项（通用·黄金三章）\n"
        "- 卖点清晰：首章让读者明白本书核心吸引力与主角处境。\n"
        "- 冲突前置：尽早抛出核心冲突或悬念，避免长铺垫。\n"
        "- 章末留钩：用具体悬念/反转/危机收尾，制造追读冲动。"
    ),
}


def platform_opening(config: dict[str, Any]) -> str:
    """Return the platform-specific golden-first-chapters opening rules."""
    preset = str(config.get("novel", {}).get("platform_preset", "general")).strip() or "general"
    return PLATFORM_OPENING.get(preset, PLATFORM_OPENING["general"])


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^一-鿿A-Za-z0-9]", "", text)
    if len(cleaned) < 2:
        return set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _score(query_tokens: set[str], text: str) -> float:
    tokens = _tokenize(text[:6000])
    if not query_tokens or not tokens:
        return 0.0
    return len(query_tokens & tokens) / max(1, len(query_tokens | tokens))


def _candidate_dirs(config: dict[str, Any]) -> list[Path]:
    novel = config.get("novel", {})
    base = ROOT / str(novel.get("benchmark_dir", "benchmarks"))
    platform = str(novel.get("platform_preset", "general"))
    style = str(novel.get("style_preset", "history"))
    dirs = [base / platform / style, base / platform, base / style, base / "common", base]
    out: list[Path] = []
    for d in dirs:
        if d.exists() and d.is_dir() and d not in out:
            out.append(d)
    return out


def _read_benchmark(path: Path) -> tuple[str, str]:
    title = path.stem
    text = read_text(path)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            title = str(data.get("title") or title)
            chunks = []
            for key in ("summary", "opening", "chapter_1", "chapter_3", "payoff_pattern", "notes"):
                if data.get(key):
                    chunks.append(f"{key}: {data[key]}")
            text = "\n".join(chunks) or text
        except Exception:
            pass
    return title, text.strip()


def benchmark_context(
    paths: Paths,
    config: dict[str, Any],
    query: str,
    max_chars: int | None = None,
) -> str:
    """Return a compact local benchmark block for the current platform/genre.

    The project can drop markdown/txt/json samples under benchmarks/<platform>/<style>/.
    This helper is intentionally dependency-free; if no sample exists it returns
    an empty string and the pipeline behaves as before.
    """
    if not bool(config.get("novel", {}).get("benchmark_enabled", True)):
        return ""
    max_chars = int(max_chars or config.get("novel", {}).get("benchmark_context_chars", 5000))
    query_tokens = _tokenize(query)
    candidates: list[tuple[float, str, str, Path]] = []
    for d in _candidate_dirs(config):
        for path in d.glob("*"):
            if path.suffix.lower() not in {".md", ".txt", ".json"} or not path.is_file():
                continue
            if path.name.lower().startswith("readme."):
                continue
            title, text = _read_benchmark(path)
            if not text:
                continue
            candidates.append((_score(query_tokens, text), title, text, path))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    parts = ["## 爆款样本参照（只学习结构/节奏/读者承诺，不模仿具体表达）"]
    used = len(parts[0])
    for score, title, text, path in candidates[: int(config.get("novel", {}).get("benchmark_top_k", 3))]:
        snippet = text[:1800]
        block = f"### {title} score={score:.3f} source={path.relative_to(ROOT)}\n{snippet}"
        if used + len(block) + 2 > max_chars:
            remaining = max_chars - used - 80
            if remaining > 300:
                parts.append(block[:remaining] + "\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)
