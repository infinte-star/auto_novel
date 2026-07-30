"""v3 loop: the decision table, acceptance, StoryState projection, and repair.

Self-contained engine module that inlines the acceptance set, StoryState canon
projection, and repair ladder alongside the decision table. Everything that
decides *what happens next* is a pure predicate over recorded state — no LLM
ranks the options, no model is asked whether it is done. The three model calls
(arc, write, L1) are **actions** the table dispatches to; they never vote on
the routing.

v3 changes from v2:
- canon check (the one LLM judging call) removed — zero-value path
- next_card_patch carry-forward removed
- 9 advisory-only gates removed from acceptance_report
- v2/accept.py, v2/canon.py, v2/repair.py inlined here
- decision table: 8 rows (was 10)

Three rules hold the design up:

**One ruler.** Every acceptance verdict in this file comes from
``acceptance_report`` / ``block_reasons``, which are
``hard_block_reasons`` — the same function ``tools/fpy_prime.py`` replays.

**Round 0 is the raw first draft, and it is archived before anything touches
it.** ``review_round0.json`` is what FPY' replays. It is written the moment the
first report exists, and never overwritten by a repaired or rescued draft.

**No repair may buy a rewrite.** L0 and L1 sit *above* ``rescue`` in the table,
so the cheap deterministic fixes always get their turn first.

**Nothing latches.** Every member of the acceptance set is ``scope="chapter"`` —
each blocking reason is something *this* chapter's text can turn green.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import engine.store as store
from engine.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    should_resume_existing_chapter,
)
from engine.config import (
    PROMPT_FILE,
    Paths,
    book_reached_target,
    chapter_path,
    count_chars,
    configured_api_endpoints_with_models,
    configured_role_endpoints,
    ensure_project,
    find_last_chapter,
    get_paths,
    load_config,
    log,
    read_text,
    rebuild_book,
    write_text,
)
from engine.config import text_bigrams
from engine import quality
from engine.quality import REGISTRY, hard_block_reasons
from engine.plan import ensure_card, CARD_CHECKPOINT, card_to_plan
from engine.llm import LLMClientPool
from engine.store import db_event, init_db
from engine.types import (  # noqa: F401 — re-exported for backward compat
    BUDGET,
    CLIP_MARK,
    DEFAULT_TAIL_CHARS,
    DROP_MARK,
    STABLE_HEADER,
    STABLE_SECTIONS,
    TITLES,
    VOLATILE_HEADER,
    VOLATILE_SECTIONS,
    AcceptanceReport,
    ChapterDelta,
    GateResult,
    Section,
    StoryState,
)

# ---------------------------------------------------------------------------
# Checkpoints. Three of these names are NOT v2's to choose: `review_round0.json`,
# `final_review.json`, `extraction.json`, `structured_state_done.json` and
# `chapter_completed.json` are what `tools/fpy_prime.py`, `novel.py stats`,
# `compare.py` and `checkpoint.should_resume_existing_chapter` read. A v2 that
# archived its first draft anywhere else would be invisible to the metric that
# settles the A/B.
# ---------------------------------------------------------------------------
DRAFT_CHECKPOINT = "chapter_draft.json"
ROUND0_CHECKPOINT = "review_round0.json"
FINAL_REVIEW_CHECKPOINT = "final_review.json"
EXTRACTION_CHECKPOINT = "extraction.json"
STRUCTURED_DONE_CHECKPOINT = "structured_state_done.json"
STATE_FILE_DONE_CHECKPOINT = "state_file_done.json"
COMPLETED_CHECKPOINT = "chapter_completed.json"
# `tools/fpy_prime.COUNTED_REPLANS` matches `card_replan_attempt*.json` and
# derives the label by splitting on `_attempt`, so this name is load-bearing: it
# is how a v2 chapter that needed a second write pays for it in FPY′, exactly as
# a v1 chapter pays for a plan retry.
RESCUE_CHECKPOINT = "card_replan_attempt1_rescue.json"

# A chapter that cannot settle in this many steps is not converging, and the
# table has no row that spends more than one call, so the cap is generous by
# construction: 1 card + 2 writes + 2 repair layers + 1 canon + bookkeeping.
MAX_STEPS = 24
# A `WriteError` is a refusal or a truncated stream, not a bad draft. Retry the
# call; do not spend a rescue on it.
WRITE_ATTEMPTS = 2
# `rescue` is the only row that buys a second full write. One attempt, then the
# chapter commits with its blocks recorded — because every blocking reason here
# is chapter-scoped, a second rescue would be retrying a rule the text already
# failed to satisfy under the same instructions.
RESCUE_ATTEMPTS = 1


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


def project_focus(protagonist: Any, cap: int = 0) -> str:
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
        Section("focus", project_focus(protagonist), False),
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


# ===========================================================================
# Acceptance set (inlined from v2/accept.py)
# ===========================================================================

ACCEPTANCE_GATES: tuple[str, ...] = (
    "style_health",
    "cross_chapter_repetition",
    "book_wide_fossils",
    "descriptor_frequency",
    "genre_adherence",
    "adjacent_repetition",
    "length_band_check",
    "opening_hook_gate",
)

NATIVE_CHECKS: tuple[str, ...] = ("contract_fulfilment", "citation_check")

NOT_IN_ACCEPTANCE: dict[str, str] = {
    "beat_coverage":
        "superseded by contract_fulfilment, which asks the same question "
        "(did the prose stage what was promised) against the whole card rather "
        "than the `beats` list alone. Running both would double-charge.",
    "dialogue_health":
        "runs as advisory in acceptance_report (directives forwarded to next "
        "chapter) but never blocks — penalty only, no path into "
        "hard_block_reasons.",
    "intra_chapter_repetition":
        "runs as advisory — penalty only, directives forwarded.",
    "hook_tail_repetition":
        "runs as advisory — penalty only, directives forwarded.",
    "scene_similarity": "card-phase; see CARD_GATES.",
    "narrative_pattern_repetition": "card-phase; see CARD_GATES.",
    "plan_visual_payoff_check": "card-phase; see CARD_GATES.",
    "plan_executability_gate": "card-phase; see CARD_GATES.",
    "chapter_ending_strength":
        "L1 repair-only: hook_revise rewrites weak endings but the gate "
        "carries no penalty and never emits reject level, so it has no "
        "path into hard_block_reasons. Blocking is via repair improvement, "
        "not via acceptance rejection.",
}

CARD_GATES: tuple[str, ...] = (
    "scene_similarity",
    "narrative_pattern_repetition",
    "plan_visual_payoff_check",
    "plan_executability_gate",
)


# ---------------------------------------------------------------------------
# CCC -- contract fulfilment
# ---------------------------------------------------------------------------

HARD_FIELDS: tuple[str, ...] = ("where", "turn", "exit_hook")
FORBID_MIN_ANCHOR = 4

OBLIGATION_MARKERS: tuple[str, ...] = ("必须", "务必", "应当", "应该", "需要", "要求")
PROHIBITION_MARKERS: tuple[str, ...] = (
    "禁止", "严禁", "不要", "不得", "不可", "不能", "不准", "避免", "别", "勿", "忌",
)


def _is_misfiled_requirement(entry: str) -> bool:
    """True when this ``forbid`` entry is actually a REQUIREMENT, not a ban."""
    e = str(entry or "")
    if any(m in e for m in PROHIBITION_MARKERS):
        return False
    return any(m in e for m in OBLIGATION_MARKERS)


def _body(text: str) -> str:
    return quality._strip_title_line(str(text or ""))


def _anchors(target: str) -> list[str]:
    return quality._beat_anchor_fragments(str(target or ""))


REQUIRED_FIELDS: tuple[str, ...] = (
    "where", "who", "turn", "payoff", "exit_hook", "beats", "goal", "conflict",
    "title", "opening_type",
)


def _required_text(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    parts: list[str] = []
    for field in REQUIRED_FIELDS:
        value = card.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if isinstance(v, (str, int, float)))
    return "\n".join(parts)


def _hit(target: str, text: str, grams: set[str]) -> tuple[bool, list[str], list[str]]:
    """(hit, anchors, matched). No anchors -> unjudgeable, reported as hit=False
    with an empty anchor list so the caller can drop it from the denominator."""
    anchors = _anchors(target)
    if not anchors:
        return False, [], []
    matched = [a for a in anchors if quality._fragment_hit(a, text, grams)]
    return bool(matched), anchors, matched


def contract_fulfilment(
    card: dict[str, Any] | None,
    text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Did the prose stage what the ChapterCard promised? Zero LLM."""
    cfg = (config or {}).get("novel", {}) if config else {}
    enabled = bool(cfg.get("ccc_enabled", True))
    out: dict[str, Any] = {
        "enabled": enabled, "ccr": 1.0, "judgeable": 0, "fulfilled": 0,
        "unjudgeable": 0, "items": [], "missing": [], "violations": [],
        "forbid_conflicts": [],
        "hard_misses": [], "passed": True, "directives": [],
    }
    if not enabled or not isinstance(card, dict) or not card:
        return out
    body = _body(text)
    if len(body) < 500:
        return out
    grams = text_bigrams(body, strip="none")
    tail = body[-int(cfg.get("ccc_tail_chars", DEFAULT_TAIL_CHARS)):]
    tail_grams = text_bigrams(tail, strip="none")
    hard_fields = set(HARD_FIELDS)

    def add(field: str, target: str, hit: bool, anchors: list[str], matched: list[str]):
        verdict = "hit" if hit else ("miss" if anchors else "unjudgeable")
        item = {"field": field, "target": target, "verdict": verdict,
                "anchors": anchors, "matched": matched,
                "hard": field in hard_fields}
        out["items"].append(item)
        if verdict == "unjudgeable":
            out["unjudgeable"] += 1
            return
        out["judgeable"] += 1
        if hit:
            out["fulfilled"] += 1
        else:
            out["missing"].append(item)
            if item["hard"]:
                out["hard_misses"].append(item)

    for field in ("where", "turn", "payoff"):
        target = str(card.get(field) or "").strip()
        if target:
            add(field, target, *_hit(target, body, grams))

    for raw_name in (card.get("who") or []):
        name = re.sub(r"[（(][^)）]*[)）]", "", str(raw_name or "")).strip()
        if len(name) >= 2:
            got = name in body
            add("who", name, got, [name], [name] if got else [])

    hook = str(card.get("exit_hook") or "").strip()
    if hook:
        add("exit_hook", hook, *_hit(hook, tail, tail_grams))

    for beat in (card.get("beats") or []):
        beat = str(beat or "").strip()
        if beat:
            add("beats", beat, *_hit(beat, body, grams))

    required = _required_text(card)
    for entry in (card.get("forbid") or []):
        entry = str(entry or "").strip()
        if not entry:
            continue

        if _is_misfiled_requirement(entry):
            out["forbid_conflicts"].append(
                {"field": "forbid", "target": entry, "phrase": "",
                 "why": "requirement_misfiled_as_ban"})
            continue

        def charge(phrase: str) -> bool:
            if phrase in required:
                out["forbid_conflicts"].append(
                    {"field": "forbid", "target": entry, "phrase": phrase,
                     "why": "card_requires_the_phrase_it_bans"})
                return True
            out["violations"].append(
                {"field": "forbid", "target": entry, "phrase": phrase})
            return True

        if len(entry) >= FORBID_MIN_ANCHOR and entry in body:
            charge(entry)
            continue
        for anchor in _anchors(entry):
            if len(anchor) >= FORBID_MIN_ANCHOR and anchor in body:
                charge(anchor)
                break

    if out["judgeable"]:
        out["ccr"] = out["fulfilled"] / out["judgeable"]
    out["passed"] = not out["hard_misses"] and not out["violations"]

    for item in out["hard_misses"]:
        out["directives"].append(
            f"本章卡片承诺的【{item['field']}】没有落到页面上："
            f"{item['target']}。必须补写具体动作+具体物，"
            f"至少让「{item['anchors'][0]}」真实出现。")
    for v in out["violations"]:
        out["directives"].append(
            f"本章违反卡片禁令【{v['target']}】：正文里出现了「{v['phrase']}」，必须换掉。")
    return out


# ---------------------------------------------------------------------------
# cite-or-drop
# ---------------------------------------------------------------------------

QUOTE_KEYS = ("quote", "evidence", "excerpt", "原文", "证据", "引用", "locator")

_PUNCT_RE = re.compile("[\\s\u201c\u201d\"'\u2018\u2019\u300c\u300d\u300e\u300f\uff08\uff09()\\[\\]\uff0c,\u3002.\uff01!\uff1f?\uff1b;\uff1a:\u3001\u2026\u2014\\-~\u00b7]+")


def _normalize_quote(s: str) -> str:
    return _PUNCT_RE.sub("", str(s or ""))


def citation_check(
    claims: Iterable[dict[str, Any]] | None,
    text: str,
    min_quote_chars: int = 4,
) -> dict[str, Any]:
    """Drop every claim that cannot point at a substring of the chapter."""
    claims_list = [c for c in (claims or []) if isinstance(c, dict)]
    body = _normalize_quote(_body(text))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for claim in claims_list:
        raw = ""
        for key in QUOTE_KEYS:
            if str(claim.get(key) or "").strip():
                raw = str(claim[key]).strip()
                break
        norm = _normalize_quote(raw)
        if not raw:
            dropped.append({**claim, "_drop_reason": "uncited"})
        elif len(norm) < min_quote_chars:
            dropped.append({**claim, "_drop_reason": "quote_too_short"})
        elif norm not in body:
            dropped.append({**claim, "_drop_reason": "quote_not_in_chapter"})
        else:
            kept.append(claim)
    total = len(claims_list)
    return {"kept": kept, "dropped": dropped, "total": total,
            "drop_rate": (len(dropped) / total) if total else 0.0}


# ---------------------------------------------------------------------------
# The acceptance report
# ---------------------------------------------------------------------------

def _em_history(
    conn: Any,
    chapter_num: int,
    config: dict[str, Any],
) -> list[float] | None:
    """The prior chapters' em-dash density, oldest-first, or None."""
    if conn is None:
        return None
    window = max(int(config.get("novel", {}).get(
        "style_em_dash_trend_window", 5)), 1)
    try:
        rows = store.recent_metrics(conn, window + 1)
    except Exception:
        return None
    seq = sorted(
        (int(r["chapter"]), float(r["em_dash_per_kchar"]))
        for r in rows
        if r.get("chapter") is not None
        and isinstance(r.get("em_dash_per_kchar"), (int, float))
        and int(r["chapter"]) < chapter_num
    )
    return [v for _, v in seq[-window:]] or None


_DETERMINISTIC_PREFIX = "DETERMINISTIC: "


def acceptance_report(
    chapter_num: int,
    text: str,
    card: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    prior_texts: list[str] | None = None,
    prior_texts_long: list[str] | None = None,
    prev_text: str = "",
    book_texts: dict[int, str] | None = None,
    book_scans: Sequence[str] | None = None,
    recent_genre_scores: list[float] | None = None,
    recent_payoff_types: list[str] | None = None,
    conn: Any = None,
    fossil_whitelist: set[str] | None = None,
) -> dict[str, Any]:
    """Run the acceptance set and emit a v1-schema review payload.

    v3 simplification: advisory-only gates removed, canon_claims removed.
    """
    cfg = config.get("novel", {})
    body = str(text or "")
    report: dict[str, Any] = {
        "engine": "v2",
        "chapter": chapter_num,
        "accepted": True,
        "gate_rejects": [],
        "problems": [],
        "writer_directives_for_next_chapter": [],
    }

    def enabled(gate: str) -> bool:
        return REGISTRY.is_enabled(gate, config)

    def scanning(gate: str) -> bool:
        return enabled(gate) and (book_scans is None or gate in book_scans)

    # --- style_health -> style_collapse -----------------------------------
    if enabled("style_health"):
        report["style_health"] = quality.style_health(
            body, config, em_history=_em_history(conn, chapter_num, config))

    # --- length / opening: the ruler reads `.block` -----------------------
    report["length_band"] = quality.length_band_check(body, config)
    if enabled("opening_hook_gate"):
        report["opening_hook_gate"] = quality.opening_hook_gate(body, chapter_num, config)

    # --- adjacent repetition -> adjacent_repeat_block ---------------------
    if prev_text and enabled("adjacent_repetition"):
        ar = quality.adjacent_repetition(body, prev_text, config)
        report["adjacent_repetition"] = ar
        if str(ar.get("level", "")) == "block":
            report["gate_rejects"].append({"gate": "adjacent_repetition",
                                           "level": "block"})

    # --- cross-chapter repetition -> gate_rejects -------------------------
    if enabled("cross_chapter_repetition"):
        cr = quality.cross_chapter_repetition(
            body, prior_texts or [], config,
            prior_texts_long=list(prior_texts_long) if prior_texts_long else None)
        report["cross_chapter_repetition"] = cr
        if str(cr.get("level", "")) == "reject":
            report["gate_rejects"].append({
                "gate": "cross_chapter_repetition", "level": "reject",
                "phrases": [str(p) for p in (cr.get("repeated") or [])][:8]})

    # --- book-wide fossils -> gate_rejects (hard only) --------------------
    if book_texts and scanning("book_wide_fossils"):
        bf = quality.book_wide_fossils(book_texts, config,
                                       whitelist=fossil_whitelist,
                                       current_chapter=chapter_num)
        report["book_fossils"] = bf
        phrases = [str(p) for p in (bf.get("phrases") or [])]
        struct_count = int(cfg.get("book_fossil_struct_count", 10))
        if len(phrases) >= struct_count:
            report["gate_rejects"].append({
                "gate": "book_wide_fossils", "level": "reject",
                "count": len(phrases), "phrases": phrases[:8]})
        hard = bf.get("hard_fossils") or []
        if hard:
            report["gate_rejects"].append({
                "gate": "book_wide_fossils_ratio", "level": "reject",
                "phrases": [str(f.get("phrase")) for f in hard]})

    # --- descriptor frequency -> gate_rejects -----------------------------
    if book_texts and scanning("descriptor_frequency"):
        df = quality.descriptor_frequency(book_texts, config)
        report["descriptor_frequency"] = df
        if str(df.get("level", "")) == "reject":
            report["gate_rejects"].append({"gate": "descriptor_frequency",
                                           "level": "reject"})

    # --- genre adherence -> gate_rejects ----------------------------------
    if enabled("genre_adherence"):
        ga = quality.genre_adherence(body, recent_genre_scores or [], config)
        report["genre_adherence"] = ga
        if str(ga.get("level", "")) == "reject":
            report["gate_rejects"].append({"gate": "genre_adherence",
                                           "level": "reject"})

    # --- CCC: the v2-native member ----------------------------------------
    ccc = contract_fulfilment(card, body, config)
    report["contract_fulfilment"] = ccc
    if ccc["enabled"] and not ccc["passed"]:
        report["gate_rejects"].append({
            "gate": "contract_fulfilment", "level": "reject",
            "phrases": [i["target"] for i in ccc["hard_misses"]][:4],
            "violations": [v["phrase"] for v in ccc["violations"]][:4]})
    if ccc.get("forbid_conflicts"):
        report["card_defects"] = [
            (f"卡片自相矛盾：`forbid` 里这条其实是硬性要求而不是禁令"
             f"（{c['target'][:60]}…），整条豁免"
             if c.get("why") == "requirement_misfiled_as_ban" else
             f"卡片自相矛盾：`forbid` 禁了本卡片自己要求的「{c['phrase']}」"
             f"（出自 {c['target'][:40]}…），已豁免")
            for c in ccc["forbid_conflicts"]]

    # --- advisory gates (zero-LLM, directives only) ------------------------
    if enabled("ai_flavor_health"):
        report["ai_flavor_health"] = quality.ai_flavor_health(body, config)
    if enabled("paragraph_shape_health"):
        report["paragraph_shape_health"] = quality.paragraph_shape_health(body, config)
    if prior_texts and enabled("hook_tail_repetition"):
        report["hook_tail_repetition"] = quality.hook_tail_repetition(
            body, prior_texts, config)
    if enabled("intra_chapter_repetition"):
        report["intra_chapter_repetition"] = quality.intra_chapter_repetition(body, config)
    if enabled("prose_texture"):
        report["prose_texture"] = quality.prose_texture(body, config)
    if enabled("dialogue_health"):
        report["dialogue_health"] = quality.dialogue_health(body, config)
    if conn and enabled("long_span_fatigue"):
        report["long_span_fatigue"] = quality.long_span_fatigue(
            conn, chapter_num, config)
    if enabled("payoff_beat_density"):
        report["payoff_beat_density"] = quality.payoff_beat_density(
            body, recent_payoff_types, config)
    if enabled("shareable_line"):
        report["shareable_line"] = quality.shareable_line(body, config)
    if enabled("information_density"):
        report["information_density"] = quality.information_density(
            body, card, None, config)
    if enabled("chapter_ending_strength"):
        report["chapter_ending_strength"] = quality.chapter_ending_strength(
            body, config)

    # --- directives -------------------------------------------------------
    wd = report["writer_directives_for_next_chapter"]
    for key in ("style_health", "length_band", "opening_hook_gate",
                "adjacent_repetition", "cross_chapter_repetition",
                "book_fossils", "descriptor_frequency", "genre_adherence",
                "contract_fulfilment",
                "ai_flavor_health", "paragraph_shape_health",
                "hook_tail_repetition", "intra_chapter_repetition",
                "prose_texture", "dialogue_health", "long_span_fatigue",
                "payoff_beat_density", "shareable_line", "information_density",
                "chapter_ending_strength"):
        for d in (report.get(key) or {}).get("directives", []):
            if d not in wd:
                wd.append(d)

    # --- finalize verdict -------------------------------------------------
    reasons = hard_block_reasons(report, config)
    report["block_reasons"] = reasons
    report["accepted"] = not reasons
    problems = report.get("problems") or []
    if reasons:
        problems.append(_DETERMINISTIC_PREFIX + "; ".join(reasons))
    report["problems"] = problems
    return report


def block_reasons(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """The one ruler, re-exported so callers never grow a second copy."""
    return hard_block_reasons(report, config)


# ===========================================================================
# Repair ladder (inlined from v2/repair.py)
# ===========================================================================

REPAIR_LAYERS: tuple[str, ...] = ("L0", "L1")

_KIND_RE = re.compile(r"^[a-z_]+")

Recheck = Callable[[str], dict]


def reason_kind(reason: Any) -> str:
    m = _KIND_RE.match(str(reason or ""))
    return m.group(0) if m else str(reason or "")


def reason_kinds(reasons: Iterable[Any]) -> frozenset[str]:
    return frozenset(reason_kind(r) for r in (reasons or ()))


def repair_pending(report: dict[str, Any], config: dict[str, Any],
                   layer: str) -> tuple[str, ...]:
    """The repair actions this layer still has to offer for this report."""
    if not isinstance(report, dict):
        return ()
    try:
        steps = quality.plan_repairs(report, config)
    except Exception:
        return ()
    return tuple(s["action"] for s in steps if s.get("layer") == layer)


@dataclasses.dataclass(frozen=True)
class RepairOutcome:
    """What a repair pass did, and whether the ruler agrees it helped."""

    text: str
    report: dict[str, Any]
    applied: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()
    blocks_before: tuple[str, ...] = ()
    blocks_after: tuple[str, ...] = ()
    reverted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    @property
    def cleared(self) -> bool:
        return not self.blocks_after

    @property
    def improved(self) -> bool:
        return len(self.blocks_after) < len(self.blocks_before)

    def summary(self) -> str:
        bits = [f"blocks {len(self.blocks_before)}->{len(self.blocks_after)}"]
        if self.applied:
            bits.append("applied=" + ",".join(self.applied))
        if self.reverted:
            bits.append("reverted=" + ",".join(self.reverted))
        if self.skipped:
            bits.append("skipped=" + ",".join(self.skipped))
        return " ".join(bits)


def _repair_log(paths: Any, message: str) -> None:
    if paths is None:
        return
    try:
        log(paths, message)
    except Exception:
        pass


def _regressed(before: Sequence[str], after: Sequence[str]) -> str:
    """'' when the repair is safe to keep, else why it is not."""
    if len(after) < len(before):
        return ""
    if len(after) > len(before):
        return f"count {len(before)}->{len(after)}"
    swapped = reason_kinds(after) - reason_kinds(before)
    if swapped:
        return "swapped_for=" + ",".join(sorted(swapped))
    return ""


def run_repair_layer(
    layer: str,
    *,
    text: str,
    report: dict[str, Any],
    config: dict[str, Any],
    chapter_num: int,
    recheck: Recheck,
    client: Any = None,
    paths: Any = None,
) -> RepairOutcome:
    """Run ONE repair layer and re-score it. Never raises, never re-drafts."""
    before = tuple(block_reasons(report, config))
    idle = RepairOutcome(text=text, report=report, blocks_before=before,
                         blocks_after=before)

    actions = repair_pending(report, config, layer)
    if not actions:
        return idle
    if layer == "L1" and client is None:
        _repair_log(paths, f"v2.repair Ch{chapter_num} L1 skipped (no client): "
                    + ",".join(actions))
        return dataclasses.replace(idle, skipped=actions)

    try:
        if layer == "L0":
            new_text, applied = quality.apply_l0(text, report, config, chapter_num)
        else:
            new_text, applied = quality.apply_l1(client, paths, config, chapter_num,
                                                 text, report)
    except Exception as exc:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} failed (non-fatal): {exc}")
        return idle

    if not applied or not new_text or new_text == text:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} kept nothing from "
                    f"[{','.join(actions)}]")
        return idle

    try:
        new_report = recheck(new_text)
    except Exception as exc:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} recheck failed, "
                    f"reverting (non-fatal): {exc}")
        return dataclasses.replace(idle, reverted=tuple(applied))

    after = tuple(block_reasons(new_report, config))
    why = _regressed(before, after)
    if why:
        _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} REVERTED ({why}): "
                    + ",".join(applied))
        return dataclasses.replace(idle, reverted=tuple(applied))

    _repair_log(paths, f"v2.repair Ch{chapter_num} {layer} kept "
                f"[{','.join(applied)}] blocks {len(before)}->{len(after)}")
    return RepairOutcome(text=new_text, report=new_report,
                         applied=tuple(applied), layers=(layer,),
                         blocks_before=before, blocks_after=after)


# ---------------------------------------------------------------------------
# Corpus — the context the acceptance gates read, assembled to match v1 exactly
# ---------------------------------------------------------------------------

def book_scan_gates(config: dict[str, Any], chapter_num: int) -> tuple[str, ...]:
    """Which whole-book scans may run this chapter, on v1's cadence.

    `review.py` runs `book_wide_fossils` and `descriptor_frequency` only every
    Nth chapter, past a minimum. FPY′ judges both arms by whatever their round-0
    payload happens to contain, so a v2 that scanned every chapter would find
    fossils v1 was never asked about and report itself as the worse engine for
    looking harder. The cadence is copied, not improved: improving it is a
    separate experiment with its own control.
    """
    cfg = config.get("novel", {})
    out: list[str] = []
    every = max(1, int(cfg.get("book_fossil_every", 5)))
    if (chapter_num >= int(cfg.get("book_fossil_min_chapters", 6))
            and chapter_num % every == 0):
        out.append("book_wide_fossils")
    d_every = max(1, int(cfg.get("descriptor_freq_every", 5)))
    if (chapter_num >= int(cfg.get("descriptor_freq_min_spread", 15))
            and chapter_num % d_every == 0):
        out.append("descriptor_frequency")
    return tuple(out)


def _chapter_texts(paths: Paths, first: int, last: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for num in range(max(1, first), last + 1):
        p = chapter_path(paths, num)
        if p.exists():
            out[num] = read_text(p)
    return out


def _genre_scores(conn: Any, config: dict[str, Any], chapter_num: int) -> list[float]:
    try:
        if conn is None or isinstance(conn, store.JsonStoryStore):
            return []
        window = int(config["novel"].get("genre_adherence_window", 5))
        cursor = conn.execute(
            "SELECT genre_score FROM chapter_metrics "
            "WHERE chapter < ? AND genre_score IS NOT NULL "
            "ORDER BY chapter DESC LIMIT ?",
            (chapter_num, window),
        )
        return [row[0] for row in cursor.fetchall()][::-1]
    except Exception:
        return []


def _payoff_types(conn: Any, config: dict[str, Any], chapter_num: int) -> list[str]:
    """The recent `payoff_type` cadence, NEWEST-FIRST.

    That order is not cosmetic: `quality.payoff_beat_density` counts its payoff
    drought by walking the list from index 0 and stopping at the first strong type,
    which is what `store.recent_metrics` hands every other metrics-reading gate.
    Fed ascending it counts forward from chapter 1, breaks on the book's first
    strong payoff, and reports a drought of 0 for every chapter after it.

    On v2 this column is populated 30/30 (it comes from the ChapterCard through
    `arc.card_to_plan`, not from a self-score), and it is MORE diverse than v1's
    extraction-derived version over the same positions — 8 distinct values against
    5. The gate reads a signal v2 made better, not one v2 lost.
    """
    try:
        if conn is None or isinstance(conn, store.JsonStoryStore):
            return []
        window = int(config["novel"].get("payoff_density_window", 0))
        if window <= 0:
            # Derived, not a new config key. The gate's drought threshold is
            # `round(1 / payoff_density_min)` — 2 chapters for 爽文, 5 for 历史 — and
            # a window shorter than that truncates the drought it is meant to
            # detect, reporting a healthy cadence because we stopped counting.
            # Two chapters of slack so the flag can exceed the line, not just reach it.
            min_rate = float(config["novel"].get("payoff_density_min", 0.34))
            window = (int(round(1.0 / min_rate)) if min_rate > 0 else 3) + 2
        cursor = conn.execute(
            "SELECT payoff_type FROM chapter_metrics "
            "WHERE chapter < ? AND payoff_type IS NOT NULL "
            "ORDER BY chapter DESC LIMIT ?",
            (chapter_num, window),
        )
        return [str(row[0]) for row in cursor.fetchall()]
    except Exception:
        return []


@dataclasses.dataclass
class Corpus:
    """Everything ``acceptance_report`` needs besides the text itself.

    Read once per chapter. The gates are re-run several times (after L0, after
    L1, after a rescue) and re-globbing the book each time would make the
    deterministic half of the pipeline the slow half.
    """

    prior_texts: list[str] = dataclasses.field(default_factory=list)
    prior_long: list[str] = dataclasses.field(default_factory=list)
    prev_text: str = ""
    book_texts: dict[int, str] = dataclasses.field(default_factory=dict)
    book_scans: tuple[str, ...] = ()
    genre_scores: list[float] = dataclasses.field(default_factory=list)
    payoff_types: list[str] = dataclasses.field(default_factory=list)
    whitelist: set[str] = dataclasses.field(default_factory=set)


def load_corpus(paths: Paths, conn: Any, config: dict[str, Any],
                chapter_num: int) -> Corpus:
    cfg = config["novel"]
    lookback = int(cfg.get("style_cross_repeat_lookback", 6))
    lookback_long = int(cfg.get("style_cross_repeat_lookback_long", 20))
    span = max(lookback, lookback_long)
    all_prior = list(_chapter_texts(paths, chapter_num - span, chapter_num - 1).values())
    scans = book_scan_gates(config, chapter_num)
    # Chapters 1..n from disk — which on a first draft means 1..n-1, because the
    # chapter under review has not been saved yet. That is v1's corpus verbatim,
    # including the consequence that `in_current` (and therefore a hard fossil
    # reject) can only fire on a resume. Feeding the draft in here would be a
    # strictly stricter gate than the arm being compared against.
    book_texts = _chapter_texts(paths, 1, chapter_num) if scans else {}
    try:
        prompt_text = read_text(PROMPT_FILE)
    except Exception:
        prompt_text = ""
    return Corpus(
        prior_texts=all_prior[-lookback:] if len(all_prior) > lookback else all_prior,
        prior_long=all_prior,
        prev_text=all_prior[-1] if all_prior else "",
        book_texts=book_texts,
        book_scans=scans,
        genre_scores=_genre_scores(conn, config, chapter_num),
        payoff_types=_payoff_types(conn, config, chapter_num),
        whitelist=quality.fossil_whitelist(config, prompt_text),
    )


def persist_scan_caches(paths: Paths, report: dict[str, Any]) -> None:
    """Mirror `review.py`'s two avoid-list caches.

    `writing._preflight_negative_list` reads `logs/book_fossils.json` on EVERY
    chapter and v2's writer goes through that same function. Skipping the write
    would quietly hand the v1 arm an avoid-list the v2 arm never gets, which
    would show up as v2 writing more fossils and be read as a v2 defect.
    """
    for key, name, marker in (("book_fossils", "book_fossils.json", "phrases"),
                              ("descriptor_frequency", "descriptor_freq.json", "flagged")):
        data = report.get(key)
        if isinstance(data, dict) and data.get(marker):
            try:
                write_text(paths.logs_dir / name,
                           json.dumps(data, ensure_ascii=False, indent=2))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-chapter state
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Ctx:
    client: Any
    paths: Paths
    conn: Any
    config: dict[str, Any]
    prompt_file: Path | None = None


def _event(ctx: Ctx, chapter: int, event_type: str, payload: dict[str, Any]) -> None:
    """Record an event, and never let recording it be what fails.

    `store.db_event` writes to SQLite unguarded, which is correct for v1 where
    every caller is inside a chapter that is already committing. Here the event
    log is pure observation — a locked database or a missing connection must not
    lose a finished chapter, and it must not be able to abort a run from a step
    whose real work already succeeded.
    """
    try:
        db_event(ctx.conn, chapter, event_type, payload)
    except Exception as exc:
        log(ctx.paths, f"v2.event {event_type} Ch{chapter} not recorded: {exc}")


@dataclasses.dataclass
class ChapterRun:
    """Everything the decision table branches on. Mutable by design: each row's
    action advances exactly one field, and the table re-reads from the top."""

    chapter_num: int
    resume: bool = False

    card: dict[str, Any] | None = None
    plan: dict[str, Any] = dataclasses.field(default_factory=dict)
    decision: dict[str, Any] = dataclasses.field(default_factory=dict)
    card_source: str = ""
    card_degraded: bool = False
    constraints: tuple[str, ...] | None = None

    state: StoryState | None = None
    text: str = ""
    raw_text: str = ""          # the FIRST draft, never overwritten
    delta: ChapterDelta | None = None
    delta_status: str = ""
    title: str = ""

    report: dict[str, Any] | None = None
    round0_saved: bool = False
    layers_run: tuple[str, ...] = ()

    write_attempts: int = 0
    rescue_attempts: int = 0
    steps: int = 0
    trace: list[str] = dataclasses.field(default_factory=list)
    committed: bool = False

    corpus: Corpus | None = None

    @property
    def blocks(self) -> tuple[str, ...]:
        if not isinstance(self.report, dict):
            return ()
        return tuple(self.report.get("block_reasons") or ())

    def summary(self) -> str:
        return (f"Ch{self.chapter_num} card={self.card_source}"
                f"{'*' if self.card_degraded else ''} "
                f"chars={len(self.text)} delta={self.delta_status or '-'} "
                f"layers={'+'.join(self.layers_run) or '-'} "
                f"rescue={self.rescue_attempts} steps={self.steps} "
                f"blocks={','.join(self.blocks) or 'none'}")


# ---------------------------------------------------------------------------
# The one report builder + the recheck it produces
# ---------------------------------------------------------------------------

def build_report(ctx: Ctx, run: ChapterRun, text: str) -> dict[str, Any]:
    corpus = run.corpus or Corpus()
    return acceptance_report(
        run.chapter_num, text, run.card, ctx.config,
        prior_texts=corpus.prior_texts,
        prior_texts_long=corpus.prior_long,
        prev_text=corpus.prev_text,
        book_texts=corpus.book_texts,
        book_scans=corpus.book_scans,
        recent_genre_scores=corpus.genre_scores,
        recent_payoff_types=corpus.payoff_types,
        conn=ctx.conn,
        fossil_whitelist=corpus.whitelist,
    )


def recheck_fn(ctx: Ctx, run: ChapterRun) -> Callable[[str], dict[str, Any]]:
    """The judge handed to ``run_repair_layer``.

    It is the same function that produced the report being repaired, closed over
    the same corpus — so "did this fix help" is answered in the currency the
    chapter ships in, not in the fixer's own metric.
    """
    return lambda text: build_report(ctx, run, text)


# ---------------------------------------------------------------------------
# Actions. Each returns a short label for the trace; each advances exactly one
# thing, so the table can be re-read from the top afterwards.
# ---------------------------------------------------------------------------

def _act_card(ctx: Ctx, run: ChapterRun) -> str:
    state = load_story_state(ctx.paths, ctx.conn, ctx.config, run.chapter_num,
                             prompt_file=ctx.prompt_file)
    result = ensure_card(ctx.client, ctx.paths, ctx.conn, ctx.config,
                         run.chapter_num, state=state)
    run.card, run.plan, run.decision = result.card, result.plan, result.decision
    run.card_source, run.card_degraded = result.source, result.degraded
    if result.degraded:
        _event(ctx, run.chapter_num, "card_degraded",
                 {"source": result.source, "unresolved": list(result.unresolved)})
    return f"card[{result.source}]"


def _act_fold_constraints(ctx: Ctx, run: ChapterRun) -> str:
    """The ``card_invalid`` row, and it costs nothing.

    ``ensure_card`` already owns validate -> repair -> single re-plan and has
    already folded whatever it could not clear into
    ``decision["required_constraints"]`` as writer obligations. What is left is
    genuinely deterministic: gather those obligations plus last chapter's writer
    directives.
    """
    items: list[str] = []

    def add(v: Any) -> None:
        s = str(v).strip()
        if s and s not in items:
            items.append(s)

    for c in (run.decision.get("required_constraints") or []):
        add(c)
    prev = load_checkpoint(ctx.paths, run.chapter_num - 1, FINAL_REVIEW_CHECKPOINT)
    if isinstance(prev, dict):
        for d in (prev.get("writer_directives_for_next_chapter") or [])[:8]:
            add(d)
    run.constraints = tuple(items[:16])
    return f"constraints[{len(run.constraints)}]"


def _act_write(ctx: Ctx, run: ChapterRun) -> str:
    if run.state is None:
        rag = ""
        try:
            from engine.retrieval import retrieval_block

            _threads = store.get_open_threads(ctx.conn, run.chapter_num, limit=12) if ctx.conn else []
            _overdue = store.get_overdue_reader_promises(ctx.conn, run.chapter_num) if ctx.conn else []
            rag = retrieval_block(ctx.paths, ctx.config, run.plan, run.chapter_num,
                                 open_threads=_threads, overdue_promises=_overdue)
        except Exception:
            rag = ""
        run.state = load_story_state(ctx.paths, ctx.conn, ctx.config, run.chapter_num,
                                     card=run.card, rag=rag, prompt_file=ctx.prompt_file)
        log(ctx.paths, f"v2.state Ch{run.chapter_num} {run.state.sizes()}")

    from engine.write import write_chapter, WriteError, backfill_delta  # lazy

    run.write_attempts += 1
    try:
        result = write_chapter(
            ctx.client, ctx.paths, ctx.conn, ctx.config, run.chapter_num,
            run.card or {}, run.state, plan=run.plan,
            constraints=run.constraints or ())
    except WriteError as exc:
        log(ctx.paths, f"v2.write Ch{run.chapter_num} attempt "
                       f"{run.write_attempts}/{WRITE_ATTEMPTS} failed: {exc}")
        if run.write_attempts >= WRITE_ATTEMPTS:
            raise
        return f"write_retry[{run.write_attempts}]"

    run.text = result.text
    run.title = result.title
    run.delta, run.delta_status = result.delta, result.delta_status
    if not run.raw_text:
        run.raw_text = result.text
    if not result.delta_ok:
        # The one fallback call in v2, and it is bought rather than skipped.
        # Skipping looked right while the writer was gemini (2 for 2 compliant);
        # deepseek-v4-pro — which is what the A/B arm writes with — returned
        # 5,015 chars of pure prose and no delta on Ch2 of the smoke run, ok=True
        # and nothing truncated. Committing that silently is not a cheap choice:
        # `load_story_state` builds facts / threads / recent out of what the delta
        # writes, so a book that keeps missing it goes blind a chapter at a time
        # and the A/B ends up measuring a crippled v2 instead of the proposed
        # one. The honest form is to spend a CHEAP call and let it show: it
        # routes to the extraction model, carries its own `delta_backfill` tag
        # into `llm_calls.jsonl`, and is therefore counted in the headline
        # calls/chapter that decides this experiment.
        if bool(ctx.config["novel"].get("v2_delta_backfill_enabled", True)):
            delta, status = backfill_delta(
                ctx.client, ctx.paths, ctx.config, run.chapter_num, result.text)
            if status == "backfilled":
                run.delta, run.delta_status = delta, status
        if run.delta_status not in ("backfilled",):
            log(ctx.paths, f"v2.write Ch{run.chapter_num} delta={result.delta_status}; "
                           f"committing without structured state")
        _event(ctx, run.chapter_num, "delta_missing",
                 {"status": result.delta_status, "recovered": run.delta_status})
    save_checkpoint(ctx.paths, run.chapter_num, DRAFT_CHECKPOINT, {
        "text": result.text, "title": result.title,
        "delta": run.delta.as_extraction(), "delta_status": run.delta_status,
        "prompt_chars": result.prompt_chars, "attempt": run.write_attempts})
    run.report = None
    run.layers_run = ()
    return f"write[{len(result.text)}]"


def _act_report(ctx: Ctx, run: ChapterRun) -> str:
    run.report = build_report(ctx, run, run.text)
    persist_scan_caches(ctx.paths, run.report)
    if not run.round0_saved:
        # The raw first draft's verdict, archived before any repair touches it.
        # This is the payload FPY′ replays; everything after this point may
        # improve the chapter but must not improve its first-pass record.
        save_checkpoint(ctx.paths, run.chapter_num, ROUND0_CHECKPOINT, run.report)
        run.round0_saved = True
    return f"report[{len(run.blocks)}]"


def _act_layer(layer: str) -> Callable[[Ctx, ChapterRun], str]:
    def action(ctx: Ctx, run: ChapterRun) -> str:
        outcome = run_repair_layer(
            layer, text=run.text, report=run.report or {}, config=ctx.config,
            chapter_num=run.chapter_num, recheck=recheck_fn(ctx, run),
            client=ctx.client if layer != "L0" else None, paths=ctx.paths)
        run.layers_run = run.layers_run + (layer,)
        if outcome.changed:
            run.text = outcome.text
            run.report = outcome.report
            persist_scan_caches(ctx.paths, run.report)
            _event(ctx, run.chapter_num, f"v2_repair_{layer.lower()}", {
                "applied": list(outcome.applied), "reverted": list(outcome.reverted),
                "blocks_before": list(outcome.blocks_before),
                "blocks_after": list(outcome.blocks_after)})
        log(ctx.paths, f"v2.repair Ch{run.chapter_num} {layer}: {outcome.summary()}")
        return f"{layer}[{len(outcome.applied)}]"

    return action


def _act_rescue(ctx: Ctx, run: ChapterRun) -> str:
    """Rewrite once, with the surviving blocks as explicit instructions.

    The doc gates this row on `score < 6.5`; v2 has no self-score, and inventing
    one would put the v1 defect back. The substitute is stricter and decidable:
    blocks that survived L0 and L1, i.e. the cheap options are exhausted and the
    text still fails the release rule.

    It is checkpointed as a `card_replan` so `tools/fpy_prime.py` charges the
    chapter for it. A rescue that went unrecorded would be v2 buying a second
    draft off the books while v1 pays for every plan retry it takes.
    """
    run.rescue_attempts += 1
    reasons = list(run.blocks)
    directives = list((run.report or {}).get("writer_directives_for_next_chapter") or [])
    save_checkpoint(ctx.paths, run.chapter_num, RESCUE_CHECKPOINT,
                    {"attempt": run.rescue_attempts, "block_reasons": reasons,
                     "directives": directives[:8]})
    _event(ctx, run.chapter_num, "v2_rescue",
             {"attempt": run.rescue_attempts, "block_reasons": reasons})
    extra = ["上一稿被确定性验收判为不合格，原因：" + "；".join(reasons)]
    extra += [d for d in directives[:6]]
    run.constraints = tuple(list(run.constraints or ()) + extra)[:20]
    run.write_attempts = 0
    run.text = ""
    run.report = None
    run.layers_run = ()
    return f"rescue[{len(reasons)}]"


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------

def _flatten_protagonist(state: Any) -> str:
    """`writing.update_state_file` stringifies whatever it is handed, and v2's
    delta holds a dict — `str(dict)` would render Python repr into state.md and
    the writer would read it every chapter after."""
    if isinstance(state, dict):
        lines = [f"- {k}：{v}" for k, v in state.items()
                 if str(v).strip() and not isinstance(v, (dict, list))]
        for k, v in state.items():
            if isinstance(v, list) and v:
                lines.append(f"- {k}：" + "、".join(str(x) for x in v[:8]))
        return "\n".join(lines)
    return str(state or "").strip()


def _save_text(ctx: Ctx, run: ChapterRun) -> None:
    """Write the chapter, idempotently.

    The resume path can arrive here with the file already on disk. `save_chapter`
    appends to `book.md`, so calling it twice would duplicate the chapter in the
    book while leaving `chapters/` correct — a corruption that no gate reads and
    every reader would.
    """
    from engine.write import save_chapter

    path = chapter_path(ctx.paths, run.chapter_num)
    if path.exists():
        if read_text(path).strip() == run.text.strip():
            log(ctx.paths, f"v2.commit Ch{run.chapter_num}: identical text already "
                           f"on disk; not re-saving")
            return
        write_text(path, run.text)
        rebuild_book(ctx.paths)
        try:
            from engine.retrieval import index_chapter

            index_chapter(ctx.paths, run.chapter_num, run.text)
        except Exception:
            pass
        log(ctx.paths, f"v2.commit Ch{run.chapter_num}: replaced existing text and "
                       f"rebuilt book.md")
        return
    save_chapter(ctx.paths, run.chapter_num, run.text, run.report or {}, run.plan)


def _act_commit(ctx: Ctx, run: ChapterRun) -> str:
    from engine.write import update_state_file

    report = run.report or {}
    save_checkpoint(ctx.paths, run.chapter_num, FINAL_REVIEW_CHECKPOINT, report)
    _save_text(ctx, run)

    delta = run.delta or ChapterDelta()
    extraction = delta.as_extraction()
    extraction["title"] = run.title or (run.card or {}).get("title")
    try:
        apply_delta(ctx.paths, ctx.conn, run.chapter_num, delta,
                    review=report, card=run.card)
    except Exception as exc:
        log(ctx.paths, f"v2.commit Ch{run.chapter_num}: apply_delta failed "
                       f"(non-fatal): {exc}")
    save_checkpoint(ctx.paths, run.chapter_num, EXTRACTION_CHECKPOINT, extraction)
    save_checkpoint(ctx.paths, run.chapter_num, STRUCTURED_DONE_CHECKPOINT,
                    {"done": True})

    try:
        flat = dict(extraction)
        flat["protagonist_state"] = _flatten_protagonist(delta.protagonist_state)
        update_state_file(ctx.client, ctx.paths, ctx.conn, ctx.config,
                          run.chapter_num, run.text, flat)
        save_checkpoint(ctx.paths, run.chapter_num, STATE_FILE_DONE_CHECKPOINT,
                        {"done": True})
    except Exception as exc:
        log(ctx.paths, f"v2.commit Ch{run.chapter_num}: state.md render failed "
                       f"(non-fatal): {exc}")

    if bool(ctx.config["novel"].get("fingerprint_enabled", True)):
        try:
            quality.store_chapter_fingerprint(ctx.conn, run.chapter_num, run.plan)
        except Exception:
            pass

    _event(ctx, run.chapter_num, "chapter_completed", {
        "engine": "v2", "chars": len(run.text), "card_source": run.card_source,
        "layers": list(run.layers_run), "rescues": run.rescue_attempts,
        "delta_status": run.delta_status,
        "block_reasons": list(run.blocks)})
    # SYNCHRONOUS, and last. `should_resume_existing_chapter` reads this file;
    # deferring it is the loop-leak invariant in CLAUDE.md, and it is written
    # after the state writes so a crash between them resumes rather than skips.
    save_checkpoint(ctx.paths, run.chapter_num, COMPLETED_CHECKPOINT, {
        "chapter": run.chapter_num, "chars": len(run.text),
        "accepted": bool(report.get("accepted")),
        "block_reasons": list(run.blocks)})
    run.committed = True
    return "commit"


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

Predicate = Callable[["Ctx", "ChapterRun"], bool]
Action = Callable[["Ctx", "ChapterRun"], str]

DECISIONS: tuple[tuple[str, Predicate, Action], ...] = (
    ("need_card", lambda ctx, r: r.card is None, _act_card),
    ("card_invalid", lambda ctx, r: r.constraints is None, _act_fold_constraints),
    ("need_draft", lambda ctx, r: not r.text, _act_write),
    # Not in the doc's row list, and it has to be: ``repair_pending`` is a pure
    # function OF a report, so "does L0 have anything to do" is unanswerable
    # until one exists. Zero LLM, so it costs the design nothing.
    ("need_report", lambda ctx, r: r.report is None, _act_report),
    # Deliberately NOT conditioned on `r.blocks`. A repair layer answers the
    # gates that FIRED, and most fixable findings never reach a hard block:
    # `length_band_check` is the clearest case -- it is declared `repair="L1"`
    # and its short-side flag has no path into `hard_block_reasons` at all, so
    # under a blocks-gated predicate the expand-to-band fixer is dead code and
    # v2 has no answer to a short chapter (v1 answers it with a score penalty
    # that v2 has banned). v1's `_stage_fix` runs unconditionally for exactly
    # this reason; every fixer is keep-only-if-improved, so an unfired layer
    # costs nothing and a fired one cannot make the text worse by the ruler.
    # ``repair_pending`` is already the "is there anything to do" test, and L1 is
    # capped at ``fix_max_l1_calls``, so this is bounded, not open-ended.
    ("l0_pending", lambda ctx, r: ("L0" not in r.layers_run
                                   and bool(repair_pending(r.report or {}, ctx.config, "L0"))),
     _act_layer("L0")),
    ("l1_pending", lambda ctx, r: ("L1" not in r.layers_run
                                   and bool(repair_pending(r.report or {}, ctx.config, "L1"))),
     _act_layer("L1")),
    ("rescue", lambda ctx, r: bool(r.blocks) and r.rescue_attempts < RESCUE_ATTEMPTS,
     _act_rescue),
    ("commit", lambda ctx, r: True, _act_commit),
)


def run_chapter(ctx: Ctx, chapter_num: int, *, resume: bool = False,
                actions: dict[str, Action] | None = None,
                corpus: Corpus | None = None) -> ChapterRun:
    """Drive one chapter to a commit by re-reading the table from the top.

    `actions` substitutes a row's action by row name — the same dependency
    injection ``write_chapter`` takes as ``call=`` and ``run_repair_layer`` takes
    as ``recheck=``. The routing is the part of this module that has to be right;
    with the four model calls swapped out it is testable offline, which is the
    only way a claim about a zero-LLM decision table can be checked at all.
    """
    run = ChapterRun(chapter_num=chapter_num, resume=resume)
    run.corpus = corpus if corpus is not None else load_corpus(
        ctx.paths, ctx.conn, ctx.config, chapter_num)
    # Unconditional, not `if resume`. `resume` means "a chapter file is already
    # on disk, partially finalized" — but the artifact worth reclaiming is the
    # DRAFT, and a draft exists precisely in the case `resume` is false: the run
    # died after the write call and before `save_chapter`. Re-entering that
    # chapter re-bought the one call in the design that costs real money, which
    # would also inflate v2's measured calls/chapter every time the gateway
    # hiccups mid-A/B. `_restore` is a pure load-what-is-already-paid-for: on a
    # genuinely fresh chapter every checkpoint is absent and it is a no-op.
    _restore(ctx, run)
    table = [(name, pred, (actions or {}).get(name, act))
             for name, pred, act in DECISIONS]

    while not run.committed:
        run.steps += 1
        if run.steps > MAX_STEPS:
            raise RuntimeError(
                f"Ch{chapter_num}: decision table did not converge in {MAX_STEPS} "
                f"steps (trace: {' -> '.join(run.trace)}). This is a routing bug, "
                f"not a quality problem — a row is firing without advancing its "
                f"own precondition.")
        for name, predicate, action in table:
            if not predicate(ctx, run):
                continue
            run.trace.append(action(ctx, run))
            break

    log(ctx.paths, f"v2.chapter {run.summary()} trace={' -> '.join(run.trace)}")
    _event(ctx, chapter_num, "v2_chapter_trace",
             {"trace": run.trace, "steps": run.steps,
              "blocks": list(run.blocks), "card_source": run.card_source})
    return run


def _restore(ctx: Ctx, run: ChapterRun) -> None:
    """Re-enter a chapter that was interrupted, from what is already on disk.

    Only artifacts that cannot be recomputed are restored — the draft, which
    cost a model call. The report is deliberately NOT restored: it is free to
    recompute and recomputing it is what makes the resume path judge the chapter
    by today's gates rather than by whatever the interrupted run had archived.
    """
    n = run.chapter_num
    card = load_checkpoint(ctx.paths, n, CARD_CHECKPOINT)
    if isinstance(card, dict):
        run.card = card
        run.plan, run.decision = card_to_plan(card)
        run.card_source = "stored"

    draft = load_checkpoint(ctx.paths, n, DRAFT_CHECKPOINT)
    text = ""
    if isinstance(draft, dict):
        text = str(draft.get("text") or "")
        run.title = str(draft.get("title") or "")
        run.delta = ChapterDelta.from_payload(draft.get("delta"))
        run.delta_status = str(draft.get("delta_status") or "")
    path = chapter_path(ctx.paths, n)
    if path.exists():
        # The file on disk outranks the draft checkpoint: it is what a reader
        # would see and what `book.md` already contains.
        text = read_text(path)
    if text.strip():
        run.text = text
        run.raw_text = str((draft or {}).get("text") or text) if isinstance(draft, dict) else text
        run.round0_saved = load_checkpoint(ctx.paths, n, ROUND0_CHECKPOINT) is not None

    if run.card or run.text:
        log(ctx.paths, f"v2.resume Ch{n}: card={'y' if run.card else 'n'} "
                       f"text={len(run.text)}")


# ---------------------------------------------------------------------------
# Startup + loop
# ---------------------------------------------------------------------------

def build_client(paths: Paths, config: dict[str, Any]) -> Any:
    """Client pool + per-role routing.

    A near-copy of `pipeline.main`'s setup, kept local rather than factored out
    of `pipeline.py`: the v1 arm's startup path must not change while it is one
    half of a running A/B. Phase D deletes the original, and this becomes the
    only copy.
    """
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: run `pip install -r requirements.txt` "
                           "before generation.") from exc

    endpoints, primary_count, endpoint_models = configured_api_endpoints_with_models(config)
    if not endpoints:
        raise RuntimeError("Missing API key: set api.api_key, api.api_keys, or "
                           "api.api_key_groups in config.yaml")

    import httpx

    connect = int(config["api"].get("client_connect_timeout", 15))
    timeout = httpx.Timeout(connect=connect,
                            read=int(config["api"].get("client_read_timeout", 180)),
                            write=connect, pool=connect)
    headers: dict[str, str] = {}
    ua = str(config["api"].get("user_agent", "")).strip()
    if ua:
        headers["User-Agent"] = ua

    def _clients(pairs: Sequence[tuple[str, str]]) -> list[Any]:
        return [OpenAI(base_url=b, api_key=k, timeout=timeout,
                       default_headers=headers or None) for b, k in pairs]

    primary = _clients(endpoints)
    client: Any = (
        LLMClientPool(primary, primary_count, endpoints=endpoints,
                      log_fn=lambda msg: log(paths, msg),
                      endpoint_models=endpoint_models)
        if len(primary) > 1 else primary[0]
    )
    log(paths, f"v2 LLM client pool keys={len(primary)} primary={primary_count}")

    for role in ("review", "planning", "writing", "extraction"):
        role_endpoints = configured_role_endpoints(config, role)
        if not role_endpoints:
            continue
        role_clients = _clients(role_endpoints)
        role_pool: Any = (
            LLMClientPool(role_clients, endpoints=role_endpoints,
                          log_fn=lambda msg: log(paths, msg))
            if len(role_clients) > 1 else role_clients[0]
        )
        setattr(client, f"{role}_pool", role_pool)
        setattr(client, f"{role}_api", config["api"])
        log(paths, f"v2 {role} pool model={config['api'].get(f'{role}_model')} "
                   f"endpoints={len(role_clients)}")
    return client


def _ensure_bootstrap(client: Any, paths: Paths, conn: Any,
                      config: dict[str, Any]) -> None:
    """v2 still bootstraps through `memory.bootstrap`.

    ``load_story_state`` projects from bible / characters / contract / voice, and those
    files are what bootstrap writes. Replacing it is a separate change with its
    own risk; doing it inside the A/B would mean the two arms started from
    different world state, which is the one thing a fork is for preventing.
    """
    # Same pre-flight as `pipeline.main`: a state.md left behind by an aborted
    # bootstrap is worse than none, because its presence is the "already
    # bootstrapped" signal while its contents are a stub the writer would read
    # every chapter.
    if paths.state.exists():
        try:
            st = read_text(paths.state)
            missing = [p.name for p in (paths.bible, paths.characters, paths.timeline,
                                        paths.threads, paths.volume_plan)
                       if not p.exists() or p.stat().st_size < 100]
            if len(st) < 500 or "待连载补全" in st or missing:
                paths.state.unlink()
                log(paths, f"v2 pre-flight: removed partial state.md (len={len(st)}, "
                           f"missing_or_empty={missing})")
        except Exception:
            pass

    if paths.state.exists() and read_text(paths.state).strip():
        return
    from engine.bootstrap import bootstrap

    try:
        bootstrap(client, paths, conn, config)
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("quota exhausted", "429", "401", "all api keys",
                                  "marked invalid")):
            try:
                if paths.state.exists() and len(read_text(paths.state).strip()) < 500:
                    paths.state.unlink()
            except Exception:
                pass
            raise SystemExit("Bootstrap aborted: API quota/auth exhausted "
                             "(keys 401/429). Rotate keys or wait for quota reset, "
                             "then re-run.") from exc
        raise


def main() -> None:
    config = load_config()
    paths = get_paths(config)
    ensure_project(paths)
    conn = init_db(paths)
    client = build_client(paths, config)
    ctx = Ctx(client=client, paths=paths, conn=conn, config=config,
              prompt_file=PROMPT_FILE)

    _ensure_bootstrap(client, paths, conn, config)
    if not paths.book.exists() and find_last_chapter(paths) > 0:
        rebuild_book(paths)

    target = int(config["novel"]["target_words"])
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    log(paths, f"v2 start target_chars={target} current={count_chars(paths.book)} "
               f"max_chapters={max_chapters or 'none'}")

    # The v2-native circuit breaker. v1's counts consecutive force-accepts below a
    # SCORE floor; v2 has no score, and the honest analogue is stronger anyway:
    # chapters committed with deterministic blocks still outstanding. N of those
    # in a row is a failure mode more tokens will not fix.
    breaker_n = int(config["novel"].get("quality_breaker_consecutive", 2))
    blocked_streak = 0
    halted_by_breaker = False

    while True:
        if book_reached_target(paths.book, target):
            log(paths, "v2 target reached; stopping")
            break
        last = find_last_chapter(paths)
        if max_chapters and last >= max_chapters:
            log(paths, f"v2 reached max_chapters={max_chapters}; stopping")
            break

        resume = should_resume_existing_chapter(paths, last)
        chapter_num = last if resume else last + 1
        if resume:
            log(paths, f"v2 resuming partially finalized Ch{chapter_num}")

        run = run_chapter(ctx, chapter_num, resume=resume)
        total = count_chars(paths.book)
        log(paths, f"v2 progress chars={total}/{target} "
                   f"pct={total / max(target, 1) * 100:.2f}%")

        blocked_streak = blocked_streak + 1 if run.blocks else 0
        if breaker_n > 0 and blocked_streak >= breaker_n:
            log(paths, f"v2 QUALITY BREAKER: {blocked_streak} consecutive chapters "
                       f"committed with unresolved blocks (last: "
                       f"{', '.join(run.blocks)}). Halting so a human decides; "
                       f"re-running resumes cleanly.")
            halted_by_breaker = True
            break

    if halted_by_breaker:
        log(paths, f"v2 halted by quality breaker at total_chars={count_chars(paths.book)}; "
                   f"post-completion passes skipped.")
        return

    log(paths, f"v2 done total_chars={count_chars(paths.book)}")

    # Two config keys `pipeline.main` owned. They are ported verbatim rather than
    # dropped with v1: both default false, but `package_after_complete: true` in an
    # existing config would otherwise stop working with no error — a silent feature
    # loss is worse than the 14 lines. Package runs first so it describes the
    # canonical chapters/book.md; both are best-effort and never touch prose.
    if bool(config["novel"].get("package_after_complete", False)):
        try:
            from commands.package import build_package
            log(paths, "v2 generating book package (titles/intros/synopsis)")
            build_package(client, paths, config)
        except Exception as exc:
            log(paths, f"Package generation failed (non-fatal): {exc}")

    if bool(config["novel"].get("refine_after_complete", False)):
        try:
            from commands.refine import refine_book
            log(paths, "v2 starting post-completion refine pass")
            refine_book(client, paths, conn, config)
        except Exception as exc:
            log(paths, f"Refine pass failed (non-fatal): {exc}")

    log(paths, "v2 book complete")


__all__ = [
    # StoryState (inlined from v2/canon.py)
    "StoryState", "ChapterDelta", "Section", "BUDGET", "STABLE_SECTIONS",
    "VOLATILE_SECTIONS", "load_story_state", "apply_delta", "stable_key",
    # Acceptance (inlined from v2/accept.py)
    "ACCEPTANCE_GATES", "NOT_IN_ACCEPTANCE", "contract_fulfilment",
    "citation_check", "acceptance_report", "block_reasons",
    # Repair (inlined from v2/repair.py)
    "RepairOutcome", "repair_pending", "run_repair_layer",
    # Decision table and loop
    "Ctx", "ChapterRun", "Corpus", "DECISIONS", "MAX_STEPS", "RESCUE_ATTEMPTS",
    "WRITE_ATTEMPTS", "book_scan_gates", "load_corpus", "persist_scan_caches",
    "build_report", "recheck_fn", "run_chapter", "build_client", "main",
]


if __name__ == "__main__":
    main()
