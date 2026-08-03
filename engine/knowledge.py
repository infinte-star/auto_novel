"""Dynamic knowledge-base selector — reads knowledge/*.md at startup and injects
context-relevant writing/planning supplements into prompts.

Design rationale: the knowledge base contains ~15 000 chars across 13 files.
Dumping it all into every prompt wastes context and drowns the model. This module
parses the files into named sections and selects the top-N most relevant ones
based on (chapter_num, tension_level, genre, emotion_target), capped at a
character budget (default 1 500 chars for writing, 800 for planning).

Injection sites (both volatile — never touch the cacheable_prefix):
- Writer: appended to the user message in build_user() after exemplars
- Planner: appended to the system prompt in generate_arc() (arc planning does
  not use the cacheable_prefix)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Section model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnowledgeSection:
    file: str
    heading: str
    body: str
    tags: frozenset[str] = frozenset()

    @property
    def char_len(self) -> int:
        return len(self.body)


# ---------------------------------------------------------------------------
# Markdown parser — splits on ## / ### headings
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)


def _parse_md(text: str, filename: str) -> list[KnowledgeSection]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []
    sections: list[KnowledgeSection] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not body or len(body) < 20:
            continue
        heading = m.group(2).strip()
        sections.append(KnowledgeSection(
            file=filename, heading=heading, body=body,
        ))
    return sections


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CACHE: dict[str, list[KnowledgeSection]] | None = None


def load_knowledge(root: Path) -> dict[str, list[KnowledgeSection]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    knowledge_dir = root / "novel_knowledge"
    if not knowledge_dir.is_dir():
        knowledge_dir = root / "knowledge"
    if not knowledge_dir.is_dir():
        _CACHE = {}
        return _CACHE

    result: dict[str, list[KnowledgeSection]] = {}
    for md in sorted(knowledge_dir.rglob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        fname = md.stem
        sections = _parse_md(text, fname)
        if sections:
            result[fname] = sections

    _CACHE = result
    return _CACHE


def _all_sections(root: Path) -> list[KnowledgeSection]:
    kb = load_knowledge(root)
    out: list[KnowledgeSection] = []
    for secs in kb.values():
        out.extend(secs)
    return out


# ---------------------------------------------------------------------------
# Selection rules — each rule is a (condition, file_stem, heading_substring)
# triple.  When the condition matches the chapter state, sections whose file
# and heading match are boosted to the front.
# ---------------------------------------------------------------------------

_WRITER_RULES: list[tuple[str, str | None, str | None, int]] = [
    # (condition_key, file_stem, heading_substring, priority)
    # condition_key meanings:
    #   "opening"       — chapter_num <= 3
    #   "opening_ext"   — chapter_num <= 7
    #   "high_tension"  — tension_level in ("high", "extreme", "5", "4")
    #   "low_tension"   — tension_level in ("low", "1", "2")
    #   "romance"       — genre contains romance/female
    #   "always"        — every chapter
    #   "foreshadow"    — tension_level low or chapter_num % 5 == 0

    ("opening",      "opening_rules", "第1章",         100),
    ("opening",      "opening_rules", "第2章",         98),
    ("opening",      "opening_rules", "第3章",         96),
    ("opening",      "opening_rules", "平台",          90),
    ("opening",      "opening_rules", "开篇公式",       80),
    ("opening",      "opening_rules", "反模式",         70),
    ("opening_ext",  "opening_rules", "第7章检查点",    60),
    ("opening_ext",  "opening_rules", "前3章",          55),
    ("high_tension", "pacing_templates", "过山车",      85),
    ("high_tension", "hit_rules",     "三级爽点",       75),
    ("high_tension", "emotion_rules", "情绪圆融",       65),
    ("low_tension",  "pacing_templates", "呼吸式",      85),
    ("low_tension",  "foreshadow_rules", "埋设",        75),
    ("low_tension",  "foreshadow_rules", "记忆锚点",    65),
    ("romance",      "romance_rules", None,             70),
    ("romance",      "emotion_rules", "甜虐比",         80),
    ("always",       "hit_rules",     "三种落差",        50),
    ("always",       "hit_rules",     "冲突类型",        40),
    ("always",       "emotion_rules", "情绪递进公式",    45),
    ("foreshadow",   "foreshadow_rules", "回收",        60),
    ("foreshadow",   "hit_rules",     "密度规则",        55),
]

_PLANNER_RULES: list[tuple[str | None, str | None, int]] = [
    # (file_stem, heading_substring, priority)
    ("emotion_rules",     "情绪曲线模板",   100),
    ("emotion_rules",     "甜虐比",         90),
    ("pacing_templates",  "五章过山车",     85),
    ("pacing_templates",  "十章弧线",       80),
    ("pacing_templates",  "张弛交替",       75),
    ("pacing_templates",  "情绪强度标注",   70),
    ("hit_rules",         "三级爽点",       65),
    ("hit_rules",         "打脸升级",       60),
    ("hit_rules",         "三种落差",       55),
    ("foreshadow_rules",  "密度控制",       50),
]


def _condition_matches(
    key: str,
    chapter_num: int,
    tension: str,
    genre: str,
) -> bool:
    if key == "always":
        return True
    if key == "opening":
        return chapter_num <= 3
    if key == "opening_ext":
        return chapter_num <= 7
    if key == "high_tension":
        return tension.lower() in ("high", "extreme", "5", "4")
    if key == "low_tension":
        return tension.lower() in ("low", "1", "2")
    if key == "romance":
        return any(k in genre.lower() for k in ("romance", "female", "言情", "女频"))
    if key == "foreshadow":
        return tension.lower() in ("low", "1", "2") or chapter_num % 5 == 0
    return False


def _find_sections(
    all_secs: list[KnowledgeSection],
    file_stem: str | None,
    heading_sub: str | None,
) -> list[KnowledgeSection]:
    out: list[KnowledgeSection] = []
    for s in all_secs:
        if file_stem and s.file != file_stem:
            continue
        if heading_sub and heading_sub not in s.heading:
            continue
        out.append(s)
    return out


def _budget_pack(
    scored: list[tuple[int, KnowledgeSection]],
    budget: int,
) -> str:
    scored.sort(key=lambda x: -x[0])
    seen_headings: set[str] = set()
    parts: list[str] = []
    used = 0
    for _score, sec in scored:
        key = (sec.file, sec.heading)
        if key in seen_headings:
            continue
        text = f"**{sec.heading}**\n{sec.body}"
        if used + len(text) > budget:
            trimmed = text[:budget - used]
            last_nl = trimmed.rfind("\n")
            if last_nl > 40:
                trimmed = trimmed[:last_nl]
            if trimmed.strip():
                parts.append(trimmed.strip())
                used += len(trimmed)
            break
        parts.append(text)
        used += len(text)
        seen_headings.add(key)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_for_writer(
    root: Path,
    chapter_num: int,
    tension_level: str = "",
    genre: str = "",
    emotion_target: str = "",
    hook_type: str = "",
    budget: int = 1500,
) -> str:
    """Return a compact knowledge supplement for the writer prompt.

    Injected into the user message (volatile) — never into the system prompt
    or cacheable_prefix.
    """
    all_secs = _all_sections(root)
    if not all_secs:
        return ""

    tension = str(tension_level or "medium")
    genre = str(genre or "")

    scored: list[tuple[int, KnowledgeSection]] = []
    for cond_key, file_stem, heading_sub, priority in _WRITER_RULES:
        if not _condition_matches(cond_key, chapter_num, tension, genre):
            continue
        matches = _find_sections(all_secs, file_stem, heading_sub)
        for sec in matches:
            scored.append((priority, sec))

    if not scored:
        return ""

    block = _budget_pack(scored, budget)
    if not block.strip():
        return ""
    return f"<知识库精选>\n{block}\n</知识库精选>"


def select_for_planner(
    root: Path,
    arc_start: int = 1,
    arc_end: int = 10,
    genre: str = "",
    budget: int = 1200,
) -> str:
    """Return a compact knowledge supplement for the arc planner prompt.

    Appended to the system prompt in generate_arc() — arc planning does NOT
    use the cacheable_prefix, so this is safe.
    """
    all_secs = _all_sections(root)
    if not all_secs:
        return ""

    scored: list[tuple[int, KnowledgeSection]] = []
    for file_stem, heading_sub, priority in _PLANNER_RULES:
        matches = _find_sections(all_secs, file_stem, heading_sub)
        for sec in matches:
            scored.append((priority, sec))

    if any(k in str(genre).lower() for k in ("romance", "female", "言情", "女频")):
        for sec in _find_sections(all_secs, "emotion_rules", "女频情感曲线"):
            scored.append((92, sec))

    if not scored:
        return ""

    block = _budget_pack(scored, budget)
    if not block.strip():
        return ""
    return (
        "\n\n## 爆款方法论参考（知识库精选）\n"
        "以下是从爆款网文研究中提炼的节奏/情绪/爽点规律，规划弧线时参考：\n\n"
        + block
    )


def invalidate_cache() -> None:
    """Force re-read on next access (for testing or hot-reload)."""
    global _CACHE
    _CACHE = None
