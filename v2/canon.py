"""StoryState — everything a chapter needs to know, projected once, in ~15k chars.

REDESIGN_V2 §3.4 ③. v1 assembles context four separate times, from four builders
(`cacheable_prefix` 30k, `writing_memory_context` 50k, `memory_context` 4 tiers,
`lite_memory_context`), each with its own budget and its own idea of what matters.
That is where the 520:1 prompt amplification lives, and it is also where the
tier-overlap bug lived (LESSONS §12: tiers 2/3 were prefixes of tier 4, so the
largest prompt in the engine shipped every row twice).

v2 projects ONE state, and splits it by **mutability rather than by topic**:

  stable   — brief / facts / voice.  Byte-identical for as long as the source
             files are, so it is the provider prompt-cache prefix.  ~5k.
  volatile — card / focus / threads / recent / ledger / rag.  Changes every
             chapter. ~11k.

`render()` always emits stable first. That ordering is the entire cache strategy:
a prefix cache hits on a shared *prefix*, so a single volatile byte in the head
would cost the hit on every call for the rest of the book. `stable_key` is a sha1
of the stable sources; `run.py` logs hit/miss off it, the same discipline
`memory.cacheable_prefix` already uses.

Three rules the projections obey, each of them a measured lesson:

1. **A clipped section says so, in the text.**  Head truncation that looks like
   the whole thing is what starved the mid-book for 40 chapters before anyone
   noticed (`volume_plan.md`, LESSONS §6).  `_clip` appends `〔…截断 N 字〕`, and
   `_clip_items` drops whole items from the END and says how many.

2. **Empty and clipped are different facts.**  An empty section emits no header
   at all.  Printing `## 伏线` with nothing under it tells the writer there are no
   open threads, which is a lie whenever the truth is "the budget ate them".

3. **Persistence has one writer.**  `apply_delta` delegates to
   `writing.update_structured_state` instead of re-implementing the entity /
   thread / promise upserts.  A second writer against the same schema is a second
   ruler: it drifts, and the drift is invisible until the two disagree about what
   canon says.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

# Per-section char caps. The sum (~16k) is the whole point of the module, so it
# is stated in one place and asserted by the test suite rather than accumulated by
# accident across five builders.
BUDGET: dict[str, int] = {
    # stable
    "brief": 1500,
    "facts": 2000,
    "voice": 1500,
    # volatile
    "card": 3000,
    "focus": 800,
    "threads": 1000,
    "recent": 1500,
    "ledger": 1000,
    "rag": 4000,
}

STABLE_SECTIONS: tuple[str, ...] = ("brief", "facts", "voice")
VOLATILE_SECTIONS: tuple[str, ...] = ("card", "focus", "threads", "recent",
                                      "ledger", "rag")

TITLES: dict[str, str] = {
    "brief": "创作纲要",
    "facts": "世界与人物事实",
    "voice": "叙事声音",
    "card": "本章卡片",
    "focus": "主角当前处境",
    "threads": "未结伏线",
    "recent": "最近发生",
    "ledger": "已用元素账",
    "rag": "相关历史原文",
}

STABLE_HEADER = "# 稳定参照（可缓存）"
VOLATILE_HEADER = "# 本章上下文"

CLIP_MARK = "〔…截断 {n} 字〕"
DROP_MARK = "〔…另有 {n} 条未列出〕"


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def _clip(text: str, cap: int) -> str:
    """Trim to `cap`, on a line boundary where possible, and SAY that it trimmed.

    `cap <= 0` means NOTHING FITS, not "no limit". The two readings collapse into
    the same integer, and picking the permissive one is how a section that had no
    room left emitted its full 40k anyway (measured on the first run of this
    module: `facts` shipped 105k against a 2k cap because a long contract drove
    the remaining room negative). Callers that mean "use the default" resolve the
    default before calling.
    """
    text = (text or "").strip()
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    mark = CLIP_MARK.format(n=len(text) - cap)
    room = max(0, cap - len(mark) - 1)   # -1 for the newline before the mark
    if room <= 0:
        return ""
    head = text[:room]
    nl = head.rfind("\n")
    if nl > room * 0.6:          # only snap to a line break if it costs little
        head = head[:nl]
    return head.rstrip() + "\n" + mark


def _clip_items(items: Sequence[str], cap: int) -> str:
    """Join `items` under `cap`, dropping WHOLE items from the end.

    Half a thread description is worse than no thread description: it reads as a
    complete fact and it is not one. So items are atomic here, and the count of
    what did not fit is stated.
    """
    kept: list[str] = []
    used = 0
    for i, it in enumerate(items):
        it = (it or "").strip()
        if not it:
            continue
        need = len(it) + (1 if kept else 0)
        left = len(items) - i
        # Reserve room for the drop marker whenever anything might not fit.
        reserve = len(DROP_MARK.format(n=left)) + 1 if left > 1 else 0
        if used + need + reserve > cap and kept:
            return "\n".join(kept) + "\n" + DROP_MARK.format(n=len(items) - len(kept))
        kept.append(it)
        used += need
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Section:
    key: str
    body: str
    stable: bool

    @property
    def title(self) -> str:
        return TITLES.get(self.key, self.key)

    def render(self) -> str:
        # Rule 2: no body, no header.
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
        """The bytes that must not move. Feed this as `cacheable_prefix`."""
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
        """Per-section chars plus totals — what `run.py` logs each chapter."""
        out = {s.key: len(s.body) for s in self.sections if s.body}
        out["_stable"] = len(self.stable_prefix())
        out["_volatile"] = len(self.volatile_block())
        out["_total"] = len(self.render())
        return out

    def digest(self) -> str:
        return hashlib.sha1(self.render().encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Projections — pure. Data in, string out, no IO, no config lookups.
# ---------------------------------------------------------------------------

def project_brief(brief: str, cap: int = 0) -> str:
    return _clip(brief, cap or BUDGET["brief"])


def project_facts(bible: str, characters: str, contract: str = "",
                  cap: int = 0) -> str:
    """World + cast + the hard contract, in that order of volatility.

    The contract goes LAST and is never clipped away first: it is the only part
    of this section that can make a chapter fail acceptance (an ability the brief
    bans is a `contract_violations` hard entry). Clipping it to make room for more
    scenery would be trading a blocking fact for a decorative one.
    """
    cap = cap or BUDGET["facts"]
    # Priority, but bounded: half the section is reserved for world+cast so a
    # runaway contract cannot evict them entirely.
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


def project_card(card: dict[str, Any] | None, arc_note: str = "",
                 cap: int = 0) -> str:
    """The chapter's contract, verbatim and first.

    Rendered as labelled lines rather than raw JSON: the same fields are read
    back by `accept.contract_fulfilment`, and a writer that skims JSON braces
    misses `forbid` — which is exactly the field whose breach is a violation.
    `forbid` is therefore emitted LAST, where the tail-anchor position effect the
    engine already relies on (`writing.fossil_tail_anchor`) puts it in front of a
    weak instruction-follower.
    """
    cap = cap or BUDGET["card"]
    if not isinstance(card, dict) or not card:
        return _clip((arc_note or "").strip(), cap)

    labels = [
        ("where", "地点"), ("who", "在场"), ("wants", "目标"),
        ("blocked_by", "阻力"), ("turn", "转折"), ("payoff", "兑现"),
        ("beats", "节拍"), ("exit_hook", "出章钩子"),
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


def project_focus(protagonist: Any, cap: int = 0) -> str:
    """Where the protagonist stands right now — the one carried-over summary.

    Everything else in the volatile half is a list of *things that happened*.
    This is the only section that says what is currently true of the person the
    reader is following: what they want, what they hold, what is pressing, what
    they have not decided. v1 carried it in `state.md`; v2 reads it back from the
    delta the previous chapter reported, so there is no second file to render.

    v1's sibling field `next_12_directions` is deliberately NOT projected here:
    mid-range steering is what the arc plan and its next-arc skeleton are for
    (`v2/beat.py`), and two sources of "what should happen next" is how a plan
    and a directive list end up contradicting each other.
    """
    cap = cap or BUDGET["focus"]
    if isinstance(protagonist, dict):
        lines = []
        for k, v in protagonist.items():
            if isinstance(v, (list, tuple)):
                v = "；".join(str(x).strip() for x in v if str(x).strip())
            k, v = str(k).strip(), str(v or "").strip()
            if k and v:
                lines.append(f"- {k}：{v}")
        return _clip_items(lines, cap)
    return _clip(str(protagonist or "").strip(), cap)


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
    """Overdue promises first — they are the only threads with a deadline."""
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
    """What just happened, newest last so it reads forward.

    `recent_events` returns newest-first (the ordering that caused the tier
    overlap in v1); reversing here means the writer reads the run-up to this
    chapter in story order instead of backwards.
    """
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
    """Hard constraints first, then the do-not-reuse list.

    A constraint is an obligation for THIS chapter; the used-element ledger is an
    avoidance list. When the budget bites, the obligation is what must survive.
    """
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
# Assembly
# ---------------------------------------------------------------------------

def build(chapter: int, *, brief: str = "", bible: str = "", characters: str = "",
          contract: str = "", voice: str = "", voices: str = "",
          card: dict[str, Any] | None = None, arc_note: str = "",
          protagonist: Any = None,
          open_threads: Iterable[dict[str, Any]] = (),
          overdue: Iterable[dict[str, Any]] = (),
          events: Iterable[dict[str, Any]] = (),
          metrics: Iterable[dict[str, Any]] = (),
          used_elements: Iterable[str] = (),
          constraints: Iterable[dict[str, Any]] = (),
          rag: str = "", stable_key: str = "") -> StoryState:
    """Assemble a StoryState from already-read data. Pure; `load` does the IO."""
    sections = (
        Section("brief", project_brief(brief), True),
        Section("facts", project_facts(bible, characters, contract), True),
        Section("voice", project_voice(voice, voices), True),
        Section("card", project_card(card, arc_note), False),
        Section("focus", project_focus(protagonist), False),
        Section("threads", project_threads(open_threads, overdue), False),
        Section("recent", project_recent(events, metrics), False),
        Section("ledger", project_ledger(used_elements, constraints), False),
        Section("rag", project_rag(rag), False),
    )
    return StoryState(chapter=chapter, sections=sections, stable_key=stable_key)


def _read(path: Path | None, cap: int = 0) -> str:
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:cap] if cap else text


def stable_key(paths: Any, prompt_file: Path | None = None) -> str:
    """sha1 over the files the stable prefix is built from.

    Same key discipline as `memory._files_hash`: a changed key means the prefix
    cache is cold for the rest of the book, so `run.py` logs it and a surprise
    invalidation is visible rather than merely expensive.
    """
    h = hashlib.sha1()
    for p in (prompt_file, getattr(paths, "bible", None),
              getattr(paths, "characters", None), getattr(paths, "voice", None),
              getattr(paths, "voices", None), getattr(paths, "contract", None)):
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
    """The most recent `protagonist_state` the writer reported.

    v1 keeps this in `state.md`, rendered by `writing.update_state_file`; v2 does
    not read markdown it wrote itself, so it goes back to the source. The
    `chapter_extraction` event is where `update_structured_state` archives the
    whole payload — under v2 that payload is the ChapterDelta, so this reads back
    exactly what the previous chapter's write call declared.
    """
    import store

    rows = store.recent_events(conn, limit=1, event_types=("chapter_extraction",))
    if not rows:
        return None
    payload = rows[0].get("payload")
    if not isinstance(payload, dict):
        return None
    return payload.get("protagonist_state") or None


def load(paths: Any, conn: Any, config: dict[str, Any], chapter: int, *,
         card: dict[str, Any] | None = None, arc_note: str = "", rag: str = "",
         prompt_file: Path | None = None) -> StoryState:
    """Read the sources and project. The ONLY function here that touches IO."""
    import store

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    contract = _read(getattr(paths, "contract", None), 4000)
    return build(
        chapter,
        brief=_read(prompt_file, 8000),
        bible=_read(getattr(paths, "bible", None), 20000),
        characters=_read(getattr(paths, "characters", None), 20000),
        contract=contract,
        voice=_read(getattr(paths, "voice", None), 8000),
        voices=_read(getattr(paths, "voices", None), 12000),
        card=card,
        arc_note=arc_note,
        protagonist=_safe(lambda: latest_protagonist(conn), None),
        open_threads=_safe(lambda: store.get_open_threads(conn, chapter, limit=12), []),
        overdue=_safe(lambda: store.get_overdue_reader_promises(conn, chapter), []),
        events=_safe(lambda: store.recent_events(conn, limit=12,
                                                 event_types=("story_event",)), []),
        metrics=_safe(lambda: store.recent_metrics(conn, 5), []),
        constraints=_safe(lambda: store.get_active_constraints(conn, chapter), []),
        rag=rag,
        stable_key=stable_key(paths, prompt_file),
    )


# ---------------------------------------------------------------------------
# ChapterDelta
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ChapterDelta:
    """The state change a chapter produced, as the writer reported it.

    v2's write call returns prose AND this in one response, which is the whole
    saving over v1's separate `extract_events` call. The schema is v1's extraction
    schema unchanged — deliberately, so `apply_delta` can hand it to the existing
    persistence rather than becoming a second writer against the same tables.
    """

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


def apply_delta(paths: Any, conn: Any, chapter: int, delta: ChapterDelta, *,
                review: dict[str, Any] | None = None,
                card: dict[str, Any] | None = None) -> None:
    """Persist a delta through v1's writer — see rule 3 in the module docstring."""
    from writing import update_structured_state

    update_structured_state(paths, conn, chapter, delta.as_extraction(),
                            review or {}, {}, card or None)
