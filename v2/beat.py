"""v2 planning: one arc call every `arc_span` chapters, planned two layers deep.

REDESIGN_V2 §3.2. v1 spends ~6.9 calls per chapter on a five-stage committee to
emit ~800 chars of plan; v2 spends ONE call per arc and projects a ChapterCard
per chapter out of it. The card is the chapter's contract — the same seven fields
`v2.accept.contract_fulfilment` scores and `tools/ccr_baseline.py` reports.

Three things separate this from `arc.py`, which it promotes:

1. **No committee underneath.** `arc.plan_from_arc` returns None on any trouble
   and the caller falls back to the five-stage planner. v2 has no such floor, so
   every failure path here has to end in a real card or an exception — never in a
   fabricated one. A card nobody planned would still score a CCR, and that number
   would be a measurement of nothing.

2. **Two-layer rolling.** Each arc call also emits a one-line skeleton for the
   NEXT arc, which is fed back in when that arc is planned. It costs no extra
   call (same response) and it is what stops arc N+1 from restarting the story
   at a right angle to arc N. A skeleton is a *promise*, not a constraint: the
   next arc may revise it, but the prompt requires it to say so in `arc_intent`.

3. **Context comes from `v2.canon`, not `memory.py`.** The stable half of the
   StoryState is passed as `cacheable_prefix`, so the arc call shares a prompt
   cache prefix with the write calls of every chapter it plans.

The card *vocabulary* (schema prompt, `normalize_card`, `validate_card`,
`card_to_plan`) is imported from `arc.py` rather than copied. There is one
definition of what a card is. When Phase D retires the v1 planner, `arc.py`'s
pure half moves into this file and no consumer import changes.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable

from arc import (
    ARC_SYSTEM,
    CARD_REPAIR_SYSTEM,
    arc_span,
    arc_window,
    card_to_plan,
    load_cards,
    normalize_card,
    save_cards,
    validate_card,
)
from checkpoint import save_checkpoint
from config import Paths, log, read_text
from llm import call_llm, json_prompt, load_json_with_repair, safe_json_loads
# Pure function; `memory.py` is otherwise unused by v2 and this import is the one
# thing Phase D has to relocate (see the module docstring).
from memory import volume_plan_window
from store import db_event, validate_plan_continuity
from v2 import canon

# Where the card this chapter was written against is archived. MUST equal
# `tools.ccr_baseline.CARD_CHECKPOINT` — a v2 arm that writes it anywhere else is
# silently measured against a reconstruction of its own plan instead of the real
# card, and the CCR it reports is a different quantity under the same name.
CARD_CHECKPOINT = "chapter_card.json"

# Appended to `arc.ARC_SYSTEM`, never a second copy of it. The arc rules have one
# home; this adds the second layer to the schema and nothing else.
SKELETON_DELTA = """

## 追加输出：下一弧骨架（双层滚动）
除 `cards` 外，还必须输出 `next_arc`：下一段弧的一句话骨架，一章一行。

{
  "arc_intent": "...",
  "cards": [ ... ],
  "next_arc": {
    "intent": "下一段弧整体要完成的一件事（从什么局面推进到什么局面）",
    "chapters": [
      {"ch": 31, "line": "该章要发生的一件具体事（可拍，一句话）"}
    ]
  }
}

`next_arc.chapters` 的章号必须是本弧最后一章之后的连续 N 章（N 与本弧章数相同，
若已接近全书终章则只写到终章为止）。骨架只写「发生什么」，不必写满卡片字段。

若「请求」里给出了「上一弧留下的骨架」，默认按它推进；确因剧情已经偏离而需要改，
可以改，但必须在 `arc_intent` 里用一句话说明改了哪一条、为什么。"""

ARC_SYSTEM_V2 = ARC_SYSTEM + SKELETON_DELTA


class BeatError(RuntimeError):
    """No usable card could be produced for this chapter.

    Raised rather than papered over: v2 writes against the card, `accept` scores
    against the card, and CCR reports against the card. Writing without one is
    not a degraded v2 — it is v1 with the planning removed, reported as v2.
    """


@dataclasses.dataclass(frozen=True)
class CardResult:
    """A card plus the honest provenance of how it was obtained."""

    card: dict[str, Any]
    plan: dict[str, Any]
    decision: dict[str, Any]
    source: str            # stored | arc | repaired | single
    unresolved: tuple[str, ...] = ()   # problems no attempt cleared
    advisories: tuple[str, ...] = ()   # non-CRITICAL continuity notes

    @property
    def degraded(self) -> bool:
        """True when this card cost more than the arc call that should have
        produced it. `run.py` logs it; the A/B counts it. A v2 arm whose cards are
        mostly `single` is not the design being tested."""
        return self.source in {"repaired", "single"} or bool(self.unresolved)


# ---------------------------------------------------------------------------
# skeleton — pure
# ---------------------------------------------------------------------------

def normalize_skeleton(raw: Any, chapters: list[int]) -> dict[str, Any] | None:
    """Coerce a `next_arc` payload into {"intent": str, "chapters": {"31": line}}.

    Chapter numbers outside `chapters` are dropped rather than renumbered: a
    skeleton line the model attached to the wrong chapter is a guess about the
    wrong chapter, and silently sliding it one slot over would launder that.
    """
    if not isinstance(raw, dict):
        return None
    allowed = {int(c) for c in chapters}
    lines: dict[str, str] = {}
    items = raw.get("chapters")
    if isinstance(items, dict):
        items = [{"ch": k, "line": v} for k, v in items.items()]
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ch = int(item.get("ch", 0))
            except (TypeError, ValueError):
                continue
            line = str(item.get("line") or item.get("text") or "").strip()
            if ch in allowed and line:
                lines[str(ch)] = line
    intent = str(raw.get("intent") or "").strip()
    if not lines and not intent:
        return None
    return {"intent": intent, "chapters": lines}


def skeleton_block(skel: dict[str, Any] | None, chapters: list[int]) -> str:
    """Render a stored skeleton for the chapters actually being planned now."""
    if not isinstance(skel, dict):
        return ""
    wanted = [str(c) for c in chapters]
    lines = [f"- Ch{c}: {skel['chapters'][c]}"
             for c in wanted if c in (skel.get("chapters") or {})]
    if not lines and not skel.get("intent"):
        return ""
    head = f"整体意图：{skel['intent']}" if skel.get("intent") else ""
    return "\n".join(p for p in (head, "\n".join(lines)) if p)


def _previous_skeleton(store: dict[str, Any], start: int, span: int) -> dict[str, Any] | None:
    """The skeleton the preceding arc left for this one, if any."""
    prev = store.get("arcs", {}).get(str(max(1, start - span)))
    if not isinstance(prev, dict):
        return None
    skel = prev.get("next_skeleton")
    return skel if isinstance(skel, dict) else None


def _recent_cards(store: dict[str, Any], chapter_num: int,
                  lookback: int = 5) -> list[dict[str, Any]]:
    out = []
    for n in range(max(1, chapter_num - lookback), chapter_num):
        card = store.get("cards", {}).get(str(n))
        if isinstance(card, dict):
            out.append(card)
    return out


# ---------------------------------------------------------------------------
# the arc call
# ---------------------------------------------------------------------------

def _parse(client: Any, paths: Paths, config: dict[str, Any], raw: str) -> dict[str, Any]:
    try:
        data = safe_json_loads(raw)
    except Exception:
        data = load_json_with_repair(client, paths, config, raw, fallback={})
    return data if isinstance(data, dict) else {}


def _volume_plan(paths: Paths, config: dict[str, Any], start: int, span: int) -> str:
    try:
        return volume_plan_window(
            read_text(paths.volume_plan), start,
            cap=int(config["novel"].get("volume_plan_chars", 12000) or 12000),
            lookahead=span + 1,
        )
    except Exception:
        return ""


def arc_user_prompt(state: canon.StoryState, chapters: list[int], *,
                    volume_plan: str = "", prev_skeleton: str = "",
                    finale_note: str = "") -> str:
    """The volatile half of the arc prompt. Pure, so its shape is testable."""
    parts = [state.volatile_block()]
    if volume_plan:
        parts.append("## 卷纲（本弧窗口）\n" + volume_plan)
    if prev_skeleton:
        parts.append("## 上一弧留下的骨架（默认延续，偏离必须在 arc_intent 里说明）\n"
                     + prev_skeleton)
    parts.append(
        f"## 请求\n为以下章节各生成一张 ChapterCard，共 {len(chapters)} 张，"
        f"`ch` 必须严格等于：{chapters}\n"
        f"这一段是一个完整的弧：先想清楚 arc_intent（从什么局面推进到什么局面），"
        f"再把它拆成 {len(chapters)} 章的连续推进，每一章都要有自己的兑现，"
        f"同时服务于整段的位移。最后再写出 next_arc 骨架。" + finale_note
    )
    return "\n\n".join(p for p in parts if p)


def generate_arc(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    start_ch: int,
    end_ch: int,
    *,
    state: canon.StoryState | None = None,
    prev_skeleton: dict[str, Any] | None = None,
    call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """ONE call producing every card for `start_ch..end_ch` plus the next skeleton.

    `call` is injected in tests; production passes None and gets `llm.call_llm`.
    """
    chapters = list(range(start_ch, end_ch + 1))
    call = call or call_llm
    if state is None:
        state = canon.load(paths, conn, config, start_ch)

    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    finale_note = ""
    if max_chapters and end_ch >= max_chapters:
        finale_note = (
            f"\n\n## 终章提醒\n第 {max_chapters} 章是全书终章：该章必须收束主线、"
            f"给出确定谜底，exit_hook 改为收束余韵，禁止引入任何新人物/新势力/新悬念。"
            f"本弧之后没有下一弧，`next_arc.chapters` 留空数组。"
        )

    span = len(chapters)
    nxt_start = end_ch + 1
    nxt_end = nxt_start + span - 1
    if max_chapters:
        nxt_end = min(nxt_end, max_chapters)
    next_chapters = list(range(nxt_start, nxt_end + 1))

    user = arc_user_prompt(
        state, chapters,
        volume_plan=_volume_plan(paths, config, start_ch, span),
        prev_skeleton=skeleton_block(prev_skeleton, chapters),
        finale_note=finale_note,
    )

    raw = call(
        client, paths, config, ARC_SYSTEM_V2, json_prompt(user),
        max_tokens=int(config["novel"].get("arc_max_tokens", 32000) or 32000),
        temperature=float(config["novel"].get("arc_temperature", 0.75) or 0.75),
        cacheable_prefix=state.stable_prefix(),
        tag="arc_plan",
    )
    data = _parse(client, paths, config, raw)
    cards_raw = data.get("cards")
    if not isinstance(cards_raw, list) or not cards_raw:
        raise BeatError(f"arc planner returned no cards for Ch{start_ch}-{end_ch}")

    by_ch: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(cards_raw):
        ch = chapters[idx] if idx < len(chapters) else None
        if isinstance(item, dict):
            try:
                declared = int(item.get("ch", 0))
            except (TypeError, ValueError):
                declared = 0
            if declared in chapters:
                ch = declared
        if ch is None:
            continue
        card = normalize_card(item, ch)
        if card:
            by_ch[ch] = card
    if not by_ch:
        raise BeatError(f"arc planner produced no usable card for Ch{start_ch}-{end_ch}")

    missing = [c for c in chapters if c not in by_ch]
    if missing:
        # In v1 these chapters fell through to the committee. Here they will each
        # cost a single-chapter arc call, so the warning names the price.
        log(paths, f"[WARN] arc {start_ch}-{end_ch}: no card for {missing}; "
                   f"each will cost its own single-chapter plan call.")
    return {
        "intent": str(data.get("arc_intent") or "").strip(),
        "cards": by_ch,
        "missing": missing,
        "next_skeleton": normalize_skeleton(data.get("next_arc"), next_chapters),
    }


def repair_card(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    card: dict[str, Any],
    problems: list[str],
    chapter_num: int,
    *,
    call: Callable[..., str] | None = None,
) -> dict[str, Any] | None:
    """One cheap call to fix a card that failed pre-write validation."""
    call = call or call_llm
    user = (
        f"## 待修复的卡片（第 {chapter_num} 章）\n"
        + json.dumps(card, ensure_ascii=False, indent=2)
        + "\n\n## 必须消除的问题（逐条修掉，不要回避）\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(problems))
        + "\n\n请返回修复后的完整卡片 JSON（保持 ch 不变，保持 schema 字段齐全）。"
    )
    raw = call(client, paths, config, CARD_REPAIR_SYSTEM, json_prompt(user),
               max_tokens=8000, temperature=0.4, tag="arc_card_repair")
    fixed = _parse(client, paths, config, raw)
    if isinstance(fixed.get("card"), dict):
        fixed = fixed["card"]  # tolerate a {"card": {...}} wrapper
    return normalize_card(fixed, chapter_num)


# ---------------------------------------------------------------------------
# validation inputs (zero LLM)
# ---------------------------------------------------------------------------

def _continuity(paths: Paths, conn: Any, config: dict[str, Any],
                plan: dict[str, Any], chapter_num: int) -> tuple[list[str], list[str]]:
    """(critical, advisory). Same severity policy as v1's `_stage_plan`: only
    CRITICAL forces a repair. Treating advisories as repair triggers fires a call
    on nearly every chapter and eats the whole cost saving."""
    try:
        violations = validate_plan_continuity(conn, plan, chapter_num, config=config)
    except Exception as exc:
        log(paths, f"beat: continuity check failed (non-fatal) Ch{chapter_num}: {exc}")
        return [], []
    return ([v for v in violations if str(v).startswith("CRITICAL")],
            [v for v in violations if not str(v).startswith("CRITICAL")])


def _scene_sim(paths: Paths, config: dict[str, Any], store: dict[str, Any],
               plan: dict[str, Any], chapter_num: int) -> float | None:
    """Max scene-skeleton similarity against recent CARDS.

    v1 dedupes against `plan_arbitration` events, which v2 never writes. Cards
    are v2's record of what was planned, so they are the only honest comparison
    set — and reading the empty event table instead would make the check silently
    disappear rather than fail.
    """
    if not bool(config["novel"].get("scene_dedupe_enabled", True)):
        return None
    try:
        from quality import scene_similarity

        lookback = int(config["novel"].get("scene_dedupe_window", 8))
        recent = [card_to_plan(c)[0] for c in _recent_cards(store, chapter_num, lookback)]
        if not recent:
            return None
        return float(scene_similarity(plan, recent).get("max_sim", 0.0) or 0.0)
    except Exception as exc:
        log(paths, f"beat: scene-dedupe check failed (non-fatal) Ch{chapter_num}: {exc}")
        return None


def _problems(paths: Paths, conn: Any, config: dict[str, Any],
              store: dict[str, Any], card: dict[str, Any],
              chapter_num: int) -> tuple[list[str], list[str]]:
    plan, _ = card_to_plan(card)
    critical, advisories = _continuity(paths, conn, config, plan, chapter_num)
    problems = validate_card(
        card,
        recent_cards=_recent_cards(store, chapter_num),
        continuity_violations=critical,
        scene_sim=_scene_sim(paths, config, store, plan, chapter_num),
        scene_sim_block=float(config["novel"].get("scene_dedupe_sim_block", 0.85)),
    )
    return problems, advisories


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _store_arc(paths: Paths, conn: Any, store: dict[str, Any], arc: dict[str, Any],
               block_id: int, gen_start: int, end: int, chapter_num: int) -> None:
    for ch, card in arc["cards"].items():
        store.setdefault("cards", {})[str(ch)] = card
    store.setdefault("arcs", {})[str(block_id)] = {
        "intent": arc["intent"], "start": gen_start, "end": end,
        "missing": arc["missing"], "next_skeleton": arc["next_skeleton"],
    }
    save_cards(paths, store)
    save_checkpoint(paths, chapter_num, "arc_generated.json",
                    {"start": gen_start, "end": end, "intent": arc["intent"],
                     "chapters": sorted(arc["cards"]),
                     "next_skeleton": arc["next_skeleton"]})
    db_event(conn, chapter_num, "arc_generated",
             {"start": gen_start, "end": end, "intent": arc["intent"],
              "count": len(arc["cards"]), "missing": arc["missing"],
              "skeleton": bool(arc["next_skeleton"])})


def ensure_card(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    *,
    state: canon.StoryState | None = None,
    call: Callable[..., str] | None = None,
) -> CardResult:
    """The card for `chapter_num`, planning an arc if there isn't one yet.

    Cost: 0 calls on 9 chapters out of 10 (the arc is already planned), 1 on the
    tenth. Worst case 3 — arc, repair, single-chapter re-plan — and the result
    says which of those happened, because a v2 arm that silently spends 3 calls a
    chapter has lost the argument it was built to make.

    Raises `BeatError` if no card can be produced at all. Stalling a chapter is
    the correct failure: writing against a card nobody planned would produce a
    CCR that measures nothing.
    """
    store = load_cards(paths)
    card = store.get("cards", {}).get(str(chapter_num))
    source = "stored"

    if not isinstance(card, dict):
        span = arc_span(config)
        max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
        block_id, end = arc_window(chapter_num, span, max_chapters)
        # Never plan chapters that are already written (a fork, or a run resumed
        # mid-block): those cards would be wasted tokens and an invitation to
        # contradict finished prose. The block id stays anchored at `block_id`.
        gen_start = max(block_id, chapter_num)
        log(paths, f"beat: planning arc Ch{gen_start}-{end} (span={span})")
        arc = generate_arc(client, paths, conn, config, gen_start, end,
                           state=state,
                           prev_skeleton=_previous_skeleton(store, block_id, span),
                           call=call)
        _store_arc(paths, conn, store, arc, block_id, gen_start, end, chapter_num)
        card = store["cards"].get(str(chapter_num))
        source = "arc"

    if not isinstance(card, dict):
        # The arc call ran and skipped exactly this chapter. One card, one call.
        card = _single(client, paths, conn, config, store, chapter_num,
                       state=state, call=call)
        source = "single"

    problems, advisories = _problems(paths, conn, config, store, card, chapter_num)
    unresolved: list[str] = []

    if problems:
        log(paths, f"beat: card Ch{chapter_num} failed pre-write validation: {problems}")
        db_event(conn, chapter_num, "card_repair", {"problems": problems, "source": source})
        fixed = repair_card(client, paths, config, card, problems, chapter_num, call=call)
        still = validate_card(fixed, recent_cards=_recent_cards(store, chapter_num)) if fixed else ["修复调用未返回可用卡片"]
        if fixed and not still:
            card, source = fixed, "repaired"
            problems = []
        else:
            log(paths, f"beat: card Ch{chapter_num} still invalid after repair ({still}); "
                       f"re-planning this chapter alone.")
            card = _single(client, paths, conn, config, store, chapter_num,
                           state=state, call=call)
            source = "single"
            problems, advisories = _problems(paths, conn, config, store, card, chapter_num)
            if problems:
                # Third attempt. Do NOT loop: every one of these problems is
                # measured against chapters this attempt cannot rewrite (the
                # neighbour's opening_type, the book's scene history), so a
                # fourth try is a latch, not a fix. Carry them to the writer as
                # obligations and let `accept` judge the prose instead.
                unresolved = list(problems)
                log(paths, f"beat: card Ch{chapter_num} accepted with unresolved "
                           f"problems (carried to the writer): {unresolved}")
                db_event(conn, chapter_num, "card_unresolved", {"problems": unresolved})

    store.setdefault("cards", {})[str(chapter_num)] = card
    save_cards(paths, store)

    plan, decision = card_to_plan(card)
    decision["planner"] = "v2_beat"
    decision["card_source"] = source
    extra = list(advisories) + [f"（规划未消除，写作时请规避）{p}" for p in unresolved]
    if extra:
        decision.setdefault("required_constraints", []).extend(extra)
    # The CCR contract: archive the card the chapter was actually written against.
    save_checkpoint(paths, chapter_num, CARD_CHECKPOINT, card)
    log(paths, f"beat: card Ch{chapter_num} [{source}] {card.get('title')} "
               f"[{card.get('opening_type')}] @ {card.get('where')}")
    return CardResult(card=card, plan=plan, decision=decision, source=source,
                      unresolved=tuple(unresolved), advisories=tuple(advisories))


def _single(client: Any, paths: Paths, conn: Any, config: dict[str, Any],
            store: dict[str, Any], chapter_num: int, *,
            state: canon.StoryState | None = None,
            call: Callable[..., str] | None = None) -> dict[str, Any]:
    """Re-plan one chapter with an arc call of length 1 — the degraded path.

    Honest but poorer: a real planned card, planned without arc-wide vision. It
    is NOT a fabricated card, and the distinction is the whole reason this
    function exists rather than a `dict(where=..., turn=...)` fallback.
    """
    arc = generate_arc(client, paths, conn, config, chapter_num, chapter_num,
                       state=state, call=call)
    card = arc["cards"].get(chapter_num)
    if not isinstance(card, dict):
        raise BeatError(f"no card for Ch{chapter_num} after a single-chapter re-plan")
    store.setdefault("cards", {})[str(chapter_num)] = card
    save_cards(paths, store)
    db_event(conn, chapter_num, "card_single_replan", {"title": card.get("title")})
    return card
