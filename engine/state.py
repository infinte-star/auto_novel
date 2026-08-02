"""Shared data structures and StoryState projection functions.

Merged from ``engine.types`` (Section, StoryState, ChapterDelta, GateResult,
AcceptanceReport) and the projection block formerly in ``engine.loop`` (all
``project_*`` functions, ``build_story_state``, ``load_story_state``, etc.).

Both original modules re-export everything defined here for backward compat.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence, TypedDict

import engine.store as store


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

DEFAULT_TAIL_CHARS = 600


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


# ===========================================================================
# StoryState projection (inlined from v2/canon.py)
# ===========================================================================

ROUTE_KEEP_STABLE: tuple[str, ...] = ("核心卖点", "差异化", "读者承诺")
ROUTE_KEEP_OPENING: tuple[str, ...] = ("正式连载前修改指令",)

OPENING_ROUTE_SPAN = 3


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def _clip(text: str, cap: int) -> str:
    """Trim to *cap*, on a line boundary where possible, and SAY that it trimmed."""
    text = (text or "").strip()
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    mark = CLIP_MARK.format(n=len(text) - cap)
    room = max(0, cap - len(mark) - 1)
    if room <= 0:
        return ""
    head = text[:room]
    nl = head.rfind("\n")
    if nl > room * 0.6:
        head = head[:nl]
    return head.rstrip() + "\n" + mark


def _clip_items(items: Sequence[str], cap: int) -> str:
    """Join *items* under *cap*, dropping WHOLE items from the end."""
    kept: list[str] = []
    used = 0
    for i, it in enumerate(items):
        it = (it or "").strip()
        if not it:
            continue
        need = len(it) + (1 if kept else 0)
        left = len(items) - i
        reserve = len(DROP_MARK.format(n=left)) + 1 if left > 1 else 0
        if used + need + reserve > cap and kept:
            return "\n".join(kept) + "\n" + DROP_MARK.format(n=len(items) - len(kept))
        kept.append(it)
        used += need
    return "\n".join(kept)


# Section and StoryState live in engine.types; re-exported above.


# ---------------------------------------------------------------------------
# Projections — pure. Data in, string out, no IO, no config lookups.
# ---------------------------------------------------------------------------

def project_brief(brief: str, cap: int = 0) -> str:
    return _clip(brief, cap or BUDGET["brief"])


def project_facts(bible: str, characters: str, contract: str = "",
                  cap: int = 0) -> str:
    cap = cap or BUDGET["facts"]
    contract = _clip((contract or "").strip(), cap // 2)
    room = max(0, cap - (len(contract) + 12 if contract else 0))
    parts = []
    for label, text in (("世界", bible), ("人物", characters)):
        text = (text or "").strip()
        if text:
            parts.append(f"### {label}\n{_clip(text, room // 2)}")
    if contract:
        parts.append(f"### 硬约束\n{contract}")
    return "\n".join(parts).strip()


def project_voice(voice: str, voices: str = "", cap: int = 0) -> str:
    cap = cap or BUDGET["voice"]
    parts = []
    v = (voice or "").strip()
    if v:
        parts.append(_clip(v, int(cap * 0.6)))
    t = (voices or "").strip()
    if t:
        parts.append("### 人物声音\n" + _clip(t, cap - len("\n".join(parts)) - 12))
    return "\n".join(parts).strip()


def _route_blocks(text: str) -> dict[str, str]:
    """Split an ``opening_route.md`` into ``{heading: body}`` by its ``## `` lines."""
    blocks: dict[str, str] = {}
    head: str | None = None
    buf: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("## "):
            if head is not None:
                blocks[head] = "\n".join(buf).strip()
            head, buf = line[3:].strip(), []
        elif head is not None:
            buf.append(line)
    if head is not None:
        blocks[head] = "\n".join(buf).strip()
    return blocks


def _route_pick(text: str, wanted: Sequence[str], cap: int) -> str:
    blocks = _route_blocks(text)
    items = []
    for want in wanted:
        body = next((v for k, v in blocks.items() if want in k and v.strip()), "")
        if body:
            items.append(f"### {want}\n{body.strip()}")
    return _clip_items(items, cap)


def project_route(opening_route: str, cap: int = 0) -> str:
    return _route_pick(opening_route, ROUTE_KEEP_STABLE, cap or BUDGET["route"])


def project_opening(opening_route: str, chapter: int, span: int = OPENING_ROUTE_SPAN,
                    cap: int = 0) -> str:
    if chapter > max(0, span):
        return ""
    return _route_pick(opening_route, ROUTE_KEEP_OPENING, cap or BUDGET["opening"])


def project_card(card: dict[str, Any] | None, arc_note: str = "",
                 cap: int = 0) -> str:
    cap = cap or BUDGET["card"]
    if not isinstance(card, dict) or not card:
        return _clip((arc_note or "").strip(), cap)

    labels = [
        ("where", "地点"), ("who", "在场"), ("wants", "目标"),
        ("blocked_by", "阻力"), ("turn", "转折"), ("payoff", "兑现"),
        ("beats", "节拍"), ("exit_hook", "出章钩子"),
        ("tension_level", "张力目标"), ("hook_type", "钩子类型"),
        ("emotion_target", "情绪目标"),
    ]
    lines: list[str] = []
    for key, label in labels:
        v = card.get(key)
        if isinstance(v, (list, tuple)):
            v = "；".join(str(x).strip() for x in v if str(x).strip())
        v = str(v or "").strip()
        if v:
            lines.append(f"- {label}：{v}")
    body = _clip("\n".join(lines), max(0, cap - 400))

    forbid = card.get("forbid")
    if isinstance(forbid, (list, tuple)):
        items = [str(x).strip() for x in forbid if str(x).strip()]
        if items:
            body += "\n- 本章禁止（逐条硬性）：\n" + "\n".join(
                f"  · {x}" for x in items[:12])
    if arc_note:
        body += "\n\n### 弧线位置\n" + _clip(arc_note, 400)
    return body.strip()


def project_focus(protagonist: Any, cap: int = 0,
                   recent_entity_changes: Iterable[dict[str, Any]] = ()) -> str:
    cap = cap or BUDGET["focus"]
    lines: list[str] = []
    if isinstance(protagonist, dict):
        for k, v in protagonist.items():
            if isinstance(v, (list, tuple)):
                v = "；".join(str(x).strip() for x in v if str(x).strip())
            k, v = str(k).strip(), str(v or "").strip()
            if k and v:
                lines.append(f"- {k}：{v}")
    elif protagonist:
        lines.append(str(protagonist).strip())
    changes = list(recent_entity_changes)
    if changes:
        lines.append("")
        lines.append("近期状态变化：")
        for c in changes[:8]:
            ch = c.get("chapter", "?")
            name = c.get("name", "")
            field = c.get("field", "")
            new_v = c.get("new_value", "")
            old_v = c.get("old_value")
            if old_v:
                lines.append(f"- Ch{ch} {name}.{field}：{old_v} → {new_v}")
            else:
                lines.append(f"- Ch{ch} {name}.{field}：{new_v}")
    return _clip_items(lines, cap)


def _thread_line(t: dict[str, Any]) -> str:
    desc = str(t.get("description", "")).strip()
    if not desc:
        return ""
    due = t.get("due_chapter")
    over = t.get("overdue_by")
    if over:
        return f"- [逾期{over}章] {desc}"
    if due:
        return f"- [第{due}章前] {desc}"
    return f"- {desc}"


def project_threads(open_threads: Iterable[dict[str, Any]],
                    overdue: Iterable[dict[str, Any]] = (),
                    cap: int = 0) -> str:
    cap = cap or BUDGET["threads"]
    seen: set[str] = set()
    items: list[str] = []
    for t in list(overdue) + list(open_threads):
        if not isinstance(t, dict):
            continue
        key = str(t.get("id") or t.get("description", ""))
        if key in seen:
            continue
        seen.add(key)
        line = _thread_line(t)
        if line:
            items.append(line)
    return _clip_items(items, cap)


def project_recent(events: Iterable[dict[str, Any]],
                   metrics: Iterable[dict[str, Any]] = (),
                   cap: int = 0) -> str:
    cap = cap or BUDGET["recent"]
    lines: list[str] = []
    for e in reversed(list(events)):
        if not isinstance(e, dict):
            continue
        pay = e.get("payload")
        if isinstance(pay, dict):
            desc = str(pay.get("description") or pay.get("summary") or "").strip()
        else:
            desc = str(pay or "").strip()
        if desc:
            lines.append(f"- Ch{e.get('chapter', '?')} {desc}")
    body = _clip_items(lines, int(cap * 0.75))

    mrows = [m for m in metrics if isinstance(m, dict)]
    if mrows:
        keys = ("chapter", "chars", "dialogue_ratio", "style_penalty")
        trim = [{k: m[k] for k in keys if k in m} for m in mrows[:5]]
        trim = [t for t in trim if t]
        if trim:
            body += "\n### 近期指标\n" + _clip(
                json.dumps(trim, ensure_ascii=False), cap - len(body) - 16)
    return body.strip()


def project_ledger(used_elements: Iterable[str] = (),
                   constraints: Iterable[dict[str, Any]] = (),
                   cap: int = 0) -> str:
    cap = cap or BUDGET["ledger"]
    lines: list[str] = []
    for c in constraints:
        if isinstance(c, dict):
            txt = str(c.get("constraint") or c.get("description") or "").strip()
        else:
            txt = str(c or "").strip()
        if txt:
            lines.append(f"- [必须] {txt}")
    body = _clip_items(lines, int(cap * 0.6)) if lines else ""

    used = [str(u).strip() for u in used_elements if str(u).strip()]
    if used:
        tail = "### 近期已用（避免重复）\n" + "、".join(used)
        room = cap - len(body) - 2
        body = (body + "\n" + _clip(tail, room)).strip() if room > 40 else body
    return body.strip()


def project_rag(block: str, cap: int = 0) -> str:
    return _clip(block, cap or BUDGET["rag"])


# ---------------------------------------------------------------------------
# StoryState assembly
# ---------------------------------------------------------------------------

def build_story_state(
    chapter: int, *, brief: str = "", bible: str = "", characters: str = "",
    contract: str = "", voice: str = "", voices: str = "",
    card: dict[str, Any] | None = None, arc_note: str = "",
    protagonist: Any = None,
    recent_entity_changes: Iterable[dict[str, Any]] = (),
    open_threads: Iterable[dict[str, Any]] = (),
    overdue: Iterable[dict[str, Any]] = (),
    events: Iterable[dict[str, Any]] = (),
    metrics: Iterable[dict[str, Any]] = (),
    used_elements: Iterable[str] = (),
    constraints: Iterable[dict[str, Any]] = (),
    rag: str = "", opening_route: str = "",
    opening_span: int = OPENING_ROUTE_SPAN,
    sk: str = "",
) -> StoryState:
    """Assemble a StoryState from already-read data. Pure; ``load_story_state`` does the IO."""
    sections = (
        Section("brief", project_brief(brief), True),
        Section("facts", project_facts(bible, characters, contract), True),
        Section("voice", project_voice(voice, voices), True),
        Section("route", project_route(opening_route), True),
        Section("card", project_card(card, arc_note), False),
        Section("focus", project_focus(protagonist,
                                       recent_entity_changes=recent_entity_changes), False),
        Section("threads", project_threads(open_threads, overdue), False),
        Section("recent", project_recent(events, metrics), False),
        Section("ledger", project_ledger(used_elements, constraints), False),
        Section("rag", project_rag(rag), False),
        Section("opening", project_opening(opening_route, chapter, opening_span), False),
    )
    return StoryState(chapter=chapter, sections=sections, stable_key=sk)


def _canon_read(path: Path | None, cap: int = 0) -> str:
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:cap] if cap else text


def opening_route_path(paths: Any) -> Path | None:
    vp = getattr(paths, "volume_plan", None)
    return (Path(vp).parent / "opening_route.md") if vp else None


def stable_key(paths: Any, prompt_file: Path | None = None) -> str:
    """sha1 over the files the stable prefix is built from."""
    h = hashlib.sha1()
    for p in (prompt_file, getattr(paths, "bible", None),
              getattr(paths, "characters", None), getattr(paths, "voice", None),
              getattr(paths, "voices", None), getattr(paths, "contract", None),
              opening_route_path(paths)):
        if p is None:
            continue
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
        try:
            h.update(hashlib.sha1(Path(p).read_bytes()).digest())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:16]


def latest_protagonist(conn: Any) -> Any:
    """The most recent ``protagonist_state`` the writer reported."""
    rows = store.recent_events(conn, limit=1, event_types=("chapter_extraction",))
    if not rows:
        return None
    payload = rows[0].get("payload")
    if not isinstance(payload, dict):
        return None
    return payload.get("protagonist_state") or None


def load_story_state(
    paths: Any, conn: Any, config: dict[str, Any], chapter: int, *,
    card: dict[str, Any] | None = None, arc_note: str = "", rag: str = "",
    prompt_file: Path | None = None,
) -> StoryState:
    """Read the sources and project. The ONLY function here that touches IO."""
    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    contract = _canon_read(getattr(paths, "contract", None), 4000)
    events = _safe(lambda: store.recent_events(conn, limit=12,
                                               event_types=("story_event",)), [])
    used = []
    for ev in events:
        p = ev.get("payload") if isinstance(ev, dict) else None
        if isinstance(p, dict):
            d = str(p.get("description") or p.get("event") or "").strip()
            if d and len(d) <= 60:
                used.append(d)
    rec = _safe(lambda: store.get_recent_entity_changes(conn, max(1, chapter - 5)), [])
    return build_story_state(
        chapter,
        brief=_canon_read(prompt_file, 8000),
        bible=_canon_read(getattr(paths, "bible", None), 20000),
        characters=_canon_read(getattr(paths, "characters", None), 20000),
        contract=contract,
        voice=_canon_read(getattr(paths, "voice", None), 8000),
        voices=_canon_read(getattr(paths, "voices", None), 12000),
        card=card,
        arc_note=arc_note,
        protagonist=_safe(lambda: latest_protagonist(conn), None),
        recent_entity_changes=rec,
        open_threads=_safe(lambda: store.get_open_threads(conn, chapter, limit=12), []),
        overdue=_safe(lambda: store.get_overdue_reader_promises(conn, chapter), []),
        events=events,
        metrics=_safe(lambda: store.recent_metrics(conn, 5), []),
        used_elements=used[:20],
        constraints=_safe(lambda: store.get_active_constraints(conn, chapter), []),
        rag=rag,
        opening_route=_canon_read(opening_route_path(paths), 20000),
        sk=stable_key(paths, prompt_file),
    )


# ---------------------------------------------------------------------------
# ChapterDelta
# ---------------------------------------------------------------------------

# ChapterDelta lives in engine.types; re-exported above.


def apply_delta(paths: Any, conn: Any, chapter: int, delta: ChapterDelta, *,
                review: dict[str, Any] | None = None,
                card: dict[str, Any] | None = None) -> None:
    """Persist a delta through v1's writer — one persistence path."""
    from engine.write import update_structured_state

    update_structured_state(paths, conn, chapter, delta.as_extraction(),
                            review or {}, {}, card or None)
