"""Shared data structures for the engine package.

Extracted from ``engine.loop`` so that ``engine.write`` (and any other module
that needs ``StoryState``, ``ChapterDelta``, or ``Section``) can import them
without creating a circular dependency.  ``engine.loop`` re-exports everything
defined here, so existing ``from engine.loop import StoryState`` still works.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Section / StoryState
# ---------------------------------------------------------------------------

BUDGET: dict[str, int] = {
    # stable
    "brief": 1500,
    "facts": 2000,
    "voice": 1500,
    "route": 600,
    # volatile
    "card": 3000,
    "focus": 800,
    "threads": 2000,
    "recent": 1500,
    "ledger": 1000,
    "rag": 4000,
    "opening": 500,
}

STABLE_SECTIONS: tuple[str, ...] = ("brief", "facts", "voice", "route")
VOLATILE_SECTIONS: tuple[str, ...] = ("card", "focus", "threads", "recent",
                                      "ledger", "rag", "opening")

TITLES: dict[str, str] = {
    "brief": "创作纲要",
    "facts": "世界与人物事实",
    "voice": "叙事声音",
    "route": "作品定位（已采纳开篇路线）",
    "card": "本章卡片",
    "focus": "主角当前处境",
    "threads": "未结伏线",
    "recent": "最近发生",
    "ledger": "已用元素账",
    "rag": "相关历史原文",
    "opening": "开篇执行指令",
}

STABLE_HEADER = "# 稳定参照（可缓存）"
VOLATILE_HEADER = "# 本章上下文"

CLIP_MARK = "〔…截断 {n} 字〕"
DROP_MARK = "〔…另有 {n} 条未列出〕"


@dataclasses.dataclass(frozen=True)
class Section:
    key: str
    body: str
    stable: bool

    @property
    def title(self) -> str:
        return TITLES.get(self.key, self.key)

    def render(self) -> str:
        return f"## {self.title}\n{self.body}" if self.body.strip() else ""


@dataclasses.dataclass(frozen=True)
class StoryState:
    """The projected context for exactly one chapter."""

    chapter: int
    sections: tuple[Section, ...]
    stable_key: str = ""

    def _join(self, stable: bool, header: str) -> str:
        parts = [s.render() for s in self.sections if s.stable is stable]
        parts = [p for p in parts if p]
        return (header + "\n\n" + "\n\n".join(parts)) if parts else ""

    def stable_prefix(self) -> str:
        """The bytes that must not move. Feed this as ``cacheable_prefix``."""
        return self._join(True, STABLE_HEADER)

    def volatile_block(self) -> str:
        return self._join(False, VOLATILE_HEADER)

    def render(self) -> str:
        head, tail = self.stable_prefix(), self.volatile_block()
        return "\n\n".join(p for p in (head, tail) if p)

    def section(self, key: str) -> str:
        for s in self.sections:
            if s.key == key:
                return s.body
        return ""

    def sizes(self) -> dict[str, int]:
        out = {s.key: len(s.body) for s in self.sections if s.body}
        out["_stable"] = len(self.stable_prefix())
        out["_volatile"] = len(self.volatile_block())
        out["_total"] = len(self.render())
        return out

    def digest(self) -> str:
        return hashlib.sha1(self.render().encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# ChapterDelta
# ---------------------------------------------------------------------------

DEFAULT_TAIL_CHARS = 400


@dataclasses.dataclass(frozen=True)
class ChapterDelta:
    """The state change a chapter produced, as the writer reported it."""

    events: tuple[dict[str, Any], ...] = ()
    entities: tuple[dict[str, Any], ...] = ()
    threads: tuple[dict[str, Any], ...] = ()
    protagonist_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    next_directions: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Any) -> "ChapterDelta":
        """Build from the write call's JSON. Never raises; missing is empty."""
        d = payload if isinstance(payload, dict) else {}

        def _dicts(key: str) -> tuple[dict[str, Any], ...]:
            v = d.get(key)
            return tuple(x for x in v if isinstance(x, dict)) if isinstance(v, list) else ()

        nd = d.get("next_12_directions") or d.get("next_directions") or []
        return cls(
            events=_dicts("events"),
            entities=_dicts("entities"),
            threads=_dicts("threads"),
            protagonist_state=d.get("protagonist_state") if isinstance(
                d.get("protagonist_state"), dict) else {},
            next_directions=tuple(str(x).strip() for x in nd
                                  if isinstance(nd, list) and str(x).strip()),
        )

    def as_extraction(self) -> dict[str, Any]:
        """The v1 extraction dict, for the one persistence path."""
        return {
            "events": list(self.events),
            "entities": list(self.entities),
            "threads": list(self.threads),
            "protagonist_state": dict(self.protagonist_state),
            "next_12_directions": list(self.next_directions),
        }

    @property
    def empty(self) -> bool:
        return not (self.events or self.entities or self.threads
                    or self.protagonist_state)


# ---------------------------------------------------------------------------
# AcceptanceReport TypedDict
# ---------------------------------------------------------------------------

class GateResult(TypedDict, total=False):
    metrics: dict[str, Any]
    penalty: float
    flags: list[str]
    directives: list[str]
    block: bool
    level: str


class AcceptanceReport(TypedDict, total=False):
    engine: str
    chapter: int
    accepted: bool
    gate_rejects: list[dict[str, Any]]
    problems: list[str]
    writer_directives_for_next_chapter: list[str]
    block_reasons: list[str]
    style_health: GateResult
    length_band: GateResult
    opening_hook_gate: GateResult
    adjacent_repetition: GateResult
    cross_chapter_repetition: GateResult
    book_fossils: GateResult
    descriptor_frequency: GateResult
    genre_adherence: GateResult
    contract_fulfilment: dict[str, Any]
    card_defects: list[str]
    ai_flavor_health: GateResult
    paragraph_shape_health: GateResult
    hook_tail_repetition: GateResult
    intra_chapter_repetition: GateResult
    prose_texture: GateResult
    dialogue_health: GateResult
    long_span_fatigue: GateResult
    payoff_beat_density: GateResult
    shareable_line: GateResult
    information_density: GateResult
