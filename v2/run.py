"""v2 run: the decision table, and the loop that turns it.

REDESIGN_V2 §3.4 ④. Everything that decides *what happens next* in this module
is a pure predicate over recorded state — no LLM ranks the options, no model is
asked whether it is done. The four model calls (arc, write, canon, L1) are
**actions** the table dispatches to; they never vote on the routing.

Why that matters is the whole v1 post-mortem in one line: v1's rework decision
read a self-score with no discrimination, so the engine's most expensive branch
was chosen by its least reliable signal. Here the branch conditions are
`is there a card`, `is there prose`, `does the report block`, `did this layer
run yet` — all of them decidable offline, all of them replayable by
`tools/fpy_prime.py` after the fact.

Four rules hold the design up:

**One ruler.** Every acceptance verdict in this file comes from
`accept.acceptance_report` / `accept.block_reasons`, which are
`quality.hard_block_reasons` — the same function the v1 arm releases on. A v2
that graded itself would win its own A/B.

**Round 0 is the raw first draft, and it is archived before anything touches
it.** `review_round0.json` is what FPY′ replays. It is written the moment the
first report exists, and never overwritten by a repaired or rescued draft. The
one exception is the canon re-fold (see `_act_canon`), which does not overwrite
the verdict so much as recompute it: the canon check runs last, on the text that
will actually ship, and its claims are then re-cited *against the raw draft* so
that a finding whose evidence exists only in the repaired prose is dropped
rather than charged to the first draft. Without that, v2's round 0 would carry
no LLM-derived violations at all while v1's carries its reviewer's, and the two
arms would be measuring different things.

**No repair may buy a rewrite.** L0 and L1 sit *above* `rescue` in the table, so
the cheap deterministic fixes always get their turn first. v1 had to hand-place
`pipeline._repair_fossil_rejects` inside its review loop to get this ordering for
one gate; here it is the default for all of them.

**Nothing latches.** Every member of the acceptance set is `scope="chapter"` —
each blocking reason is something *this* chapter's text can turn green. That is
what makes a bounded rescue safe: if the blocks survive one rewrite, they are
recorded on the committed chapter rather than retried forever.

The loop deliberately has no background pool, no adaptive candidate count, and
no score-keyed circuit breaker. The breaker it does have counts chapters
committed **with blocks outstanding**, which is the v2-native version of the
same distress signal and needs no score to read.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import quality
from checkpoint import (
    load_checkpoint,
    save_checkpoint,
    should_resume_existing_chapter,
)
from config import (
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
from llm import LLMClientPool
from store import db_event, init_db
from v2 import accept, beat, canon, repair, write

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
CANON_CHECKPOINT = "canon_claims.json"
CARD_PATCH_CHECKPOINT = "card_patch.json"

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
        import store as _store

        if conn is None or isinstance(conn, _store.JsonStoryStore):
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
        import store as _store

        if conn is None or isinstance(conn, _store.JsonStoryStore):
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
    """Everything `accept.acceptance_report` needs besides the text itself.

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

    state: canon.StoryState | None = None
    text: str = ""
    raw_text: str = ""          # the FIRST draft, never overwritten
    delta: canon.ChapterDelta | None = None
    delta_status: str = ""
    title: str = ""

    report: dict[str, Any] | None = None
    round0_saved: bool = False
    layers_run: tuple[str, ...] = ()
    canon_claims: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    canon_checked: bool = False
    patched_next: bool = False

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
    return accept.acceptance_report(
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
        canon_claims=run.canon_claims or None,
    )


def recheck_fn(ctx: Ctx, run: ChapterRun) -> Callable[[str], dict[str, Any]]:
    """The judge handed to `repair.run_layer`.

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
    state = canon.load(ctx.paths, ctx.conn, ctx.config, run.chapter_num,
                       prompt_file=ctx.prompt_file)
    result = beat.ensure_card(ctx.client, ctx.paths, ctx.conn, ctx.config,
                              run.chapter_num, state=state)
    run.card, run.plan, run.decision = result.card, result.plan, result.decision
    run.card_source, run.card_degraded = result.source, result.degraded
    if result.degraded:
        _event(ctx, run.chapter_num, "card_degraded",
                 {"source": result.source, "unresolved": list(result.unresolved)})
    return f"card[{result.source}]"


def _act_fold_constraints(ctx: Ctx, run: ChapterRun) -> str:
    """The `card_invalid` row, and it costs nothing.

    `beat.ensure_card` already owns validate → repair → single re-plan and has
    already folded whatever it could not clear into
    `decision["required_constraints"]` as writer obligations. A second validator
    here would be a second ruler for cards, which is the defect this design is
    named after. What is left is genuinely deterministic: gather those
    obligations, last chapter's writer directives, and any canon note the
    previous chapter addressed to this one.
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
    patch = load_checkpoint(ctx.paths, run.chapter_num, CARD_PATCH_CHECKPOINT)
    if isinstance(patch, dict):
        for note in (patch.get("notes") or [])[:6]:
            add(note)
    run.constraints = tuple(items[:16])
    return f"constraints[{len(run.constraints)}]"


def _act_write(ctx: Ctx, run: ChapterRun) -> str:
    if run.state is None:
        rag = ""
        try:
            from retrieval import retrieval_block

            rag = retrieval_block(ctx.paths, ctx.config, run.plan, run.chapter_num)
        except Exception:
            rag = ""
        run.state = canon.load(ctx.paths, ctx.conn, ctx.config, run.chapter_num,
                               card=run.card, rag=rag, prompt_file=ctx.prompt_file)
        log(ctx.paths, f"v2.state Ch{run.chapter_num} {run.state.sizes()}")

    run.write_attempts += 1
    try:
        result = write.write_chapter(
            ctx.client, ctx.paths, ctx.conn, ctx.config, run.chapter_num,
            run.card or {}, run.state, plan=run.plan,
            constraints=run.constraints or ())
    except write.WriteError as exc:
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
        # `canon.load` builds facts / threads / recent out of what the delta
        # writes, so a book that keeps missing it goes blind a chapter at a time
        # and the A/B ends up measuring a crippled v2 instead of the proposed
        # one. The honest form is to spend a CHEAP call and let it show: it
        # routes to the extraction model, carries its own `delta_backfill` tag
        # into `llm_calls.jsonl`, and is therefore counted in the headline
        # calls/chapter that decides this experiment.
        if bool(ctx.config["novel"].get("v2_delta_backfill_enabled", True)):
            delta, status = write.backfill_delta(
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
        outcome = repair.run_layer(
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


def _act_canon(ctx: Ctx, run: ChapterRun) -> str:
    """The one judging call, spent on the text that will actually ship.

    It runs after repair on purpose: fossil rotation rewrites the exact clauses a
    canon finding would quote, so a check run earlier would be citing prose that
    no longer exists. The cost of that ordering is that the archived round-0
    payload predates the claims, which `accept.fold_citations` settles by
    re-citing them against the raw draft — findings whose evidence only appears
    in the repaired text are dropped from round 0 rather than charged to it.
    """
    run.canon_checked = True
    claims = accept.canon_check(ctx.client, ctx.paths, ctx.config, run.chapter_num,
                                run.text, run.state, card=run.card)
    run.canon_claims = list(claims or [])
    save_checkpoint(ctx.paths, run.chapter_num, CANON_CHECKPOINT,
                    {"findings": run.canon_claims})
    if not run.canon_claims:
        return "canon[0]"

    this_chapter = [c for c in run.canon_claims
                    if str(c.get("target", "this_chapter")) != "next_card"]
    run.report = accept.fold_citations(run.report or {}, run.text, this_chapter,
                                       ctx.config)
    round0 = load_checkpoint(ctx.paths, run.chapter_num, ROUND0_CHECKPOINT)
    if isinstance(round0, dict) and run.raw_text:
        save_checkpoint(ctx.paths, run.chapter_num, ROUND0_CHECKPOINT,
                        accept.fold_citations(round0, run.raw_text, this_chapter,
                                              ctx.config))
    cited = (run.report.get("citations") or {})
    log(ctx.paths, f"v2.canon Ch{run.chapter_num}: {len(run.canon_claims)} findings, "
                   f"kept={cited.get('kept')} dropped={cited.get('dropped')}")
    return f"canon[{len(run.canon_claims)}]"


def _act_patch_next(ctx: Ctx, run: ChapterRun) -> str:
    """Carry a canon finding the NEXT chapter has to answer.

    Zero LLM: the finding's own text becomes a writer obligation on chapter n+1,
    picked up by `_act_fold_constraints` there. A finding aimed at the next card
    is not a defect in this chapter — blocking on it would be exactly the latch
    the acceptance set is built to avoid.
    """
    run.patched_next = True
    notes = [str(c.get("detail", "")).strip() for c in run.canon_claims
             if str(c.get("target", "")) == "next_card" and str(c.get("detail", "")).strip()]
    if not notes:
        return "patch_next[0]"
    save_checkpoint(ctx.paths, run.chapter_num + 1, CARD_PATCH_CHECKPOINT,
                    {"from_chapter": run.chapter_num, "notes": notes[:6]})
    log(ctx.paths, f"v2.canon Ch{run.chapter_num}: {len(notes)} note(s) carried to "
                   f"Ch{run.chapter_num + 1}")
    return f"patch_next[{len(notes)}]"


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
    # Deliberately not reset: `canon_checked`. The canon call is spent per
    # chapter, not per draft — re-buying it would make a rescued chapter cost two
    # judging calls, and `fold_citations` already re-cites the existing claims
    # against whatever text ends up shipping.
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
    from writing import save_chapter

    path = chapter_path(ctx.paths, run.chapter_num)
    if path.exists():
        if read_text(path).strip() == run.text.strip():
            log(ctx.paths, f"v2.commit Ch{run.chapter_num}: identical text already "
                           f"on disk; not re-saving")
            return
        write_text(path, run.text)
        rebuild_book(ctx.paths)
        try:
            from retrieval import index_chapter

            index_chapter(ctx.paths, run.chapter_num, run.text)
        except Exception:
            pass
        log(ctx.paths, f"v2.commit Ch{run.chapter_num}: replaced existing text and "
                       f"rebuilt book.md")
        return
    save_chapter(ctx.paths, run.chapter_num, run.text, run.report or {}, run.plan)


def _act_commit(ctx: Ctx, run: ChapterRun) -> str:
    from writing import update_state_file

    report = run.report or {}
    save_checkpoint(ctx.paths, run.chapter_num, FINAL_REVIEW_CHECKPOINT, report)
    _save_text(ctx, run)

    delta = run.delta or canon.ChapterDelta()
    extraction = delta.as_extraction()
    extraction["title"] = run.title or (run.card or {}).get("title")
    try:
        canon.apply_delta(ctx.paths, ctx.conn, run.chapter_num, delta,
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
    # Not in the doc's row list, and it has to be: `repair.pending` is a pure
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
    # `repair.pending` is already the "is there anything to do" test, and L1 is
    # capped at `fix_max_l1_calls`, so this is bounded, not open-ended.
    ("l0_pending", lambda ctx, r: ("L0" not in r.layers_run
                                   and bool(repair.pending(r.report or {}, ctx.config, "L0"))),
     _act_layer("L0")),
    ("l1_pending", lambda ctx, r: ("L1" not in r.layers_run
                                   and bool(repair.pending(r.report or {}, ctx.config, "L1"))),
     _act_layer("L1")),
    ("canon_pending", lambda ctx, r: not r.canon_checked, _act_canon),
    ("next_card_patch", lambda ctx, r: not r.patched_next, _act_patch_next),
    ("rescue", lambda ctx, r: bool(r.blocks) and r.rescue_attempts < RESCUE_ATTEMPTS,
     _act_rescue),
    ("commit", lambda ctx, r: True, _act_commit),
)


def run_chapter(ctx: Ctx, chapter_num: int, *, resume: bool = False,
                actions: dict[str, Action] | None = None,
                corpus: Corpus | None = None) -> ChapterRun:
    """Drive one chapter to a commit by re-reading the table from the top.

    `actions` substitutes a row's action by row name — the same dependency
    injection `write.write_chapter` takes as `call=` and `repair.run_layer` takes
    as `recheck=`. The routing is the part of this module that has to be right;
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

    Only artifacts that cannot be recomputed are restored — the draft and the
    canon claims, both of which cost a model call. The report is deliberately
    NOT restored: it is free to recompute and recomputing it is what makes the
    resume path judge the chapter by today's gates rather than by whatever the
    interrupted run happened to have archived.
    """
    n = run.chapter_num
    card = load_checkpoint(ctx.paths, n, beat.CARD_CHECKPOINT)
    if isinstance(card, dict):
        from arc import card_to_plan

        run.card = card
        run.plan, run.decision = card_to_plan(card)
        run.card_source = "stored"

    draft = load_checkpoint(ctx.paths, n, DRAFT_CHECKPOINT)
    text = ""
    if isinstance(draft, dict):
        text = str(draft.get("text") or "")
        run.title = str(draft.get("title") or "")
        run.delta = canon.ChapterDelta.from_payload(draft.get("delta"))
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

    claims = load_checkpoint(ctx.paths, n, CANON_CHECKPOINT)
    if isinstance(claims, dict):
        run.canon_claims = [c for c in (claims.get("findings") or [])
                            if isinstance(c, dict)]
        run.canon_checked = True
    if run.card or run.text or run.canon_checked:
        log(ctx.paths, f"v2.resume Ch{n}: card={'y' if run.card else 'n'} "
                       f"text={len(run.text)} canon={'y' if run.canon_checked else 'n'}")


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

    `canon.load` projects from bible / characters / contract / voice, and those
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
    from memory import bootstrap

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
            from package import build_package
            log(paths, "v2 generating book package (titles/intros/synopsis)")
            build_package(client, paths, config)
        except Exception as exc:
            log(paths, f"Package generation failed (non-fatal): {exc}")

    if bool(config["novel"].get("refine_after_complete", False)):
        try:
            from refine import refine_book
            log(paths, "v2 starting post-completion refine pass")
            refine_book(client, paths, conn, config)
        except Exception as exc:
            log(paths, f"Refine pass failed (non-fatal): {exc}")

    log(paths, "v2 book complete")


__all__ = [
    "Ctx", "ChapterRun", "Corpus", "DECISIONS", "MAX_STEPS", "RESCUE_ATTEMPTS",
    "WRITE_ATTEMPTS", "book_scan_gates", "load_corpus", "persist_scan_caches",
    "build_report", "recheck_fn", "run_chapter", "build_client", "main",
]


if __name__ == "__main__":
    main()
