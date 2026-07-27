# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Universal multi-novel AI writing framework. The core engine is an automated
long-form Chinese web novel generation pipeline that targets a configurable
character count (`novel.target_words`) by repeatedly running plan → write →
review → revise → extract loops until done. An optional post-completion `refine`
pass (explicit manual step: `python novel.py refine <name>`) rewrites in
5-chapter groups under intensities chosen by a diagnose LLM call.

The architecture follows a "Less is More" MVP model (2026-07 refactor): a
single-candidate plan→write→review pipeline guarded by DETERMINISTIC quality
gates, with multi-candidate breadth spent only when risk signals fire. Modules
that were premature for the current scale (reader_panel, rolling_plan,
scene_breakdown, craft/distill cross-book learning consumption, simulate style
profiles, pairwise judge) were deleted; recover them from git history if the
library ever reaches the scale (≥5 finished books) where they pay for
themselves.

The pipeline itself is **content-agnostic** — it only consumes a creative brief
(`prompt.md`) and a config (`config.yaml`). Each novel lives in its own
directory `novels/<name>/` and runs as an independent OS process, so multiple
novels can be written simultaneously without colliding on the engine's
process-level global state (`config.PROMPT_FILE`, `memory._CACHEABLE_PREFIX_CACHE`).

Every novel lives under `novels/<name>/` and is created and managed through the
unified `novel.py` CLI — there is no other entry point.

## Multi-novel framework (`novel.py`)

`novel.py` is the unified CLI that scaffolds and manages per-novel processes:

```bash
python novel.py create <name>            # scaffold novels/<name>/ from config_template.yaml + prompt_template.md
python novel.py run <name>               # run the pipeline detached (log -> novels/<name>/logs/run.log)
python novel.py run <name> --foreground  # run in the current console
python novel.py list                     # list every novel: chapters / chars / running? / last log line
python novel.py stop <name>              # kill ONLY this novel's process (token-exact `run <name>` match)
python novel.py restart <name>           # stop + relaunch (resumes from checkpoint)
python novel.py stats <name>             # rich per-novel quality+cost report (per-chapter scores/penalties/LLM cost)
python novel.py trial <name>             # generate opening trial variants WITHOUT touching chapters/book (-> logs/opening_trials/)
python novel.py adopt-trial <name> [id]  # adopt a trial's best opening route into memory/opening_route.md
python novel.py benchmark list|add ...   # manage the local 爆款 sample library (structural recall, never copies prose)
python novel.py script --input PATH      # convert ANY novel text file -> 短剧 screenplay (standalone)
python novel.py script <name> --chapters 1-3  # convert chapters 1..3 of novels/<name>/
python novel.py compare <a> <b>          # deterministic side-by-side report (scores/penalties/fossils/cost/config diff) -> experiments/
python novel.py ablate <name> --flip <key> [--chapters N]  # scaffold a chapter-capped copy with ONE config key flipped (starts at Ch1)
python novel.py fork <name> --as <new> [--flip <key>] [--chapters N]  # branch at HEAD for a mid-book A/B (preferred over ablate)
python novel.py refine <name>            # explicit post-completion refine pass (chapters_refined/ + book_refined.md; resumable)
python novel.py fix-fossils <name>       # deterministic fossil replacement (chapters_fixed/ + book_fixed.md)
python novel.py package <name>           # book packaging (titles/intros/tags/synopsis) for a finished novel
python novel.py telemetry backfill       # import every novel's history into telemetry/global.db (idempotent)
python novel.py telemetry stats [--genre G]  # cross-book strategy win-rates + totals
```

`novel.py` is the **only** entry point — there is no longer any root-level
`run.py`/`restart.py` (the README still mentions them, but they have been
removed; ignore that section). All argument parsing and command dispatch lives
in `novel.py` (`cmd_*` functions + an argparse subparser tree).

How it works (no engine changes — pure scaffolding around the existing pipeline):
- `create` copies `config_template.yaml` replacing the `__NOVEL__` placeholder so
  every `paths:` entry points inside `novels/<name>/`, and copies
  `prompt_template.md` to `novels/<name>/prompt.md` for the user to fill in.
- `run` sets `NOVEL_CONFIG`/`NOVEL_PROMPT` env vars **before** importing `pipeline`
  (same ordering constraint described in "Things to be careful with"), since
  `config.py` reads them at import time and `memory.py` captures `PROMPT_FILE` at
  its own import. Detached background launch prefers the project venv
  (`E:\pycharmproject\allvenv\novel\Scripts\python.exe`); override with the
  `NOVEL_PYTHON` env var.
- `stop`/`restart` find the process by the command-line token sequence
  `run <name>` (so `run foo` never matches `run foobar`) confined to this project.

Each novel's `story_state.db`, `logs/`, `checkpoints/`, `memory/` are isolated in
its own directory, so concurrent novels never share SQLite/file writes. All novels
read API keys from their own config's `api:` section — running many in parallel
shares the same keys' RPM/TPM quota unless you give each novel distinct keys.

## Common commands

```bash
pip install -r requirements.txt        # only dependency is openai>=1.0.0

python novel.py create <name>          # scaffold novels/<name>/
python novel.py run <name>             # run detached; resumes from checkpoint
python novel.py run <name> --foreground  # run in the current console
python novel.py list                   # progress + running state for all novels
python novel.py stop|restart <name>    # per-novel process control
```

There is no lint config or build step. Tests: `python -m unittest discover tests` (pure-function tests only, no LLM).

## Configuration

`config.yaml` (and each `novels/<name>/config.yaml`) is parsed by a hand-rolled
YAML-subset reader in `config.py:load_config` (not real YAML — only `section:`
headers and `key: value` pairs, no nested maps, no lists, no anchors). Adding new
keys requires updating the `required` dict in `load_config` if they're mandatory.

`config_template.yaml` is the scaffold copied by `novel.py create`; its `paths:`
section uses the `__NOVEL__` placeholder. Because `config.py:get_paths` joins each
`paths:` value onto `ROOT` (the project dir), a per-novel config simply sets
`paths.book: novels/<name>/book.md` etc. and the whole engine becomes
directory-isolated with zero code changes. `config.py:15-16` reads
`NOVEL_PROMPT`/`NOVEL_CONFIG` from the environment (default: `prompt.md`/`config.yaml` in the project root, used only if the env vars are unset).

Multi-endpoint, multi-key API access is configured via three keys in `api:`:
- `api_key` — single primary key
- `api_keys` — comma/semicolon list of additional keys for the primary `base_url`
- `api_key_groups` — `base_url|key1,key2;base_url2|key3,...` for fallback endpoints

`configured_api_endpoints()` returns `(endpoints, primary_count)`; the
`LLMClientPool` rotates across primary keys round-robin and only falls back to
secondary endpoints when all primaries are dead.

Per-role routing keys (`{planning,writing,extraction,review}_*`) carry two
independent reasoning knobs, because gateways honour one or the other:
- `{role}_thinking_mode` — `disabled` / `auto` / `enabled` (+`{role}_thinking_budget_tokens`),
  emitted as `extra_body["thinking"]` (Anthropic/豆包 style). `auto` omits it.
- `{role}_reasoning_effort` — OpenAI style (`none`/`low`/`medium`/`high`), emitted
  as a top-level body field via `extra_body`. Empty ⇒ never sent.

Live case: `gemini-2.5-pro` behind an nginx-fronted reseller gateway (the
template's writing role) reasons past nginx's ~70s proxy timeout and 504s before
the first byte. Neither `thinking:{"type":"disabled"}` nor omitting the param
helps — only `writing_reasoning_effort: none` does (first byte ~29s, 3.1k chars
in ~170s at the default `max_tokens: 65536`). That endpoint also 504s on any
non-stream request, so `api.stream: true` is mandatory for it.

**But treat both knobs as requests, not guarantees, and never as an A/B
variable.** Measured 2026-07-27 (`tools/probe_reasoning.py`, 8 tiers × 2
`max_tokens`, same gateway but `deepseek-v4-pro`): all 11 calls streamed
5.7k–7.2k reasoning chars — including `thinking:{"type":"disabled"}`,
`reasoning_effort: none`, and sending neither — with no correlation to the tier
and not one 504 (TTFB 1.6–76.7s, two over 70s still succeeded). Across the
library's production logs, reasoning shows up on 0% of calls on 12 days and
47–69% on two others; `yeban_guize` and `guize_guaitan` share a config yet differ
65 points. The gateway routes one model name to several upstreams and decides for
itself. `novel.py stats` prints the measured per-role coverage
(`_reasoning_coverage`) — check that both sides match before trusting any
`novel.py compare`. Actually controlling reasoning requires a different
endpoint/model; config alone will not do it.

## Architecture

### Top-level loop (`pipeline.py:main`)
1. `bootstrap()` once — generates `state.md`, `memory/{bible,characters,timeline,threads,volume_plan}.md` from `prompt.md`
2. Loop: `find_last_chapter()` → `generate_one_chapter()` until `count_chars(book.md) >= target_words`
3. `BackgroundTasks` thread pool runs finalization (extract + structured-state + state.md), stage reviews, memory compression, adaptive replans, and next-chapter plan prefetches off the critical path
4. After completion, optional `refine.refine_book()` if `novel.refine_after_complete: true` (default **false** — run `python novel.py refine <name>` manually instead)

### One chapter (`pipeline.py:generate_one_chapter`)
Strict ordering with a barrier on the previous chapter's `chapter_finalize_ch{n-1}` background label so memory/threads/metrics are fresh before planning:

```
create_plan → validate_plan_continuity → write_chapter_with_candidates
            → review/revise loop (max_revision_rounds, no-improvement early stop)
            → optional revise_hook_only for weak endings
            → save_chapter → extract_events → update_structured_state → update_state_file
```

Critical invariant in `pipeline.py:413-422`: `chapter_completed.json` must be written **synchronously** before submitting the finalize background task. If left for the bg task, the main loop's resume check would re-enter `Resuming partially indexed Ch{n}` and resubmit on every iteration, leaking threads and memory.

### Planning (`planning.py:create_plan`)
1. `generate_candidate_plans` — N candidates (default N=1; see adaptive cost control), each forced into a different strategy (`scene-driven`, `character-driven`, `thread-driven`, `institutional`, `reversal`, `pressure-payoff`) selected by a Thompson-sampling bandit (Beta posterior on arbiter win-rates, `strategy_bandit_explore_frac` forced exploration) over historical `plan_arbitration` events. Candidates whose scene skeleton is ≥ `scene_dedupe_candidate_block` (0.85) similar to a recently selected plan are dropped pre-review (unless all would be dropped)
2. Optional `screen_candidates` (skipped when `plan_skip_screen: true`, or automatically at ≤3 candidates)
3. `review_candidate_plans` — fused 6-axis review (world/character/rhythm/payoff/foreshadowing/reader) per candidate, one LLM call expanded into 6 legacy reports via `_explode_fused_axes` (the only review path — the legacy 6-parallel-calls variant was removed)
4. `arbitrate_plan` — picks `selected_index` and emits a `merged_plan` plus `required_constraints`. Still runs with a single candidate: it merges rhythm diagnostics / recent quality feedback / used-element ledger into the plan

### Arc planner (`arc.py`, `arc_planning_enabled`, default **false**)
An alternative to the five-stage committee above (REDESIGN.md L2), motivated by
the measured finding that `replan` is the #1 rework cause in every novel
(33–68% of chapters): the rework battleground is planning, not writing. ONE
high-reasoning call every `arc_span` (default 10) chapters emits one
**ChapterCard** per chapter of the arc, so 错峰兑现 / 场地轮换 / 开场轮换 /
整段推进 are decided once with whole-arc vision instead of being patched
chapter-by-chapter by gates that can only look backwards.

- Integration is **one seam**: `planning.create_plan` calls `plan_from_arc` right
  after the resume-from-checkpoint block, only when `checkpoint_label ==
  "initial"` and `replan_feedback is None`. Every replan keeps the committee, so
  a bad card still has the old safety net. `pipeline.py` is untouched.
- `card_to_plan` **projects** a card onto the existing plan schema
  (`wants→goal`, `blocked_by→conflict`, `where→location`, `exit_hook→hook`, …)
  so writing/review/quality/store need zero changes. Card-only fields
  (`opening_type`, `forbid`, `turn`) ride along in the plan dict, which
  `writing.py` dumps into the prompt wholesale.
- `decision["scores"]` is **deliberately empty**: there is no arbiter here, and a
  fake score would poison `chapter_metrics.plan_score` and the writer's quality
  contract. `plan_score()` returns 0.0 and `_prewrite_quality_contract` therefore
  suppresses its 大纲仲裁分 line instead of printing `0.0/10`.
- `plan_from_arc` **never raises** — any failure (no card, bad JSON, still-invalid
  card after one `repair_card` call) returns `None` and falls through to the
  committee.
- `validate_card` is the zero-LLM pre-write gate: empty required fields,
  `opening_type` equal to the previous chapter's, same `where` as the previous
  chapter, the same `payoff_type` three chapters running, scene similarity ≥
  `scene_dedupe_sim_block`, plus CRITICAL continuity violations. It follows the
  committee's severity policy exactly (`pipeline._stage_plan`): only CRITICAL
  forces repair; overdue threads and un-cashed setups are advisories appended to
  `required_constraints`. Treating advisories as repair triggers fires a repair
  on nearly every chapter (5 fired on the first live card) and eats the saving.
- Cards persist in `logs/arc_cards.json`; `arc_window` is anchored at Ch1 so it is
  a pure function of the chapter number and a resumed/forked run recomputes the
  same boundaries. Generation clips to `max(start, chapter_num)` so a run that
  starts mid-block never plans already-written chapters.
- Scene-dedupe in the arc arm must union `_recent_selected_plans` with recent
  cards projected through `card_to_plan`: `_recent_selected_plans` reads
  `plan_arbitration` events, which the arc path never emits, so an all-arc run
  would silently lose the check.

### Writing & revision (`writing.py`)
- Writer system prompts use a **shared-base + genre-delta architecture**: `GENRE_PROFILES` dict holds per-genre deltas (role, self_review, core_discipline, structure_template, genre_bans, sensory_dialogue, time_marker_ban, extras), and `_build_write_system()` assembles the system prompt from shared constants (`_SELF_REVIEW_PREAMBLE`, `_OUTPUT_SECTION`, `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT`) + genre deltas + `ANTI_FRAGMENT_BAN` + `ANTI_PITFALL_BLOCK` + aesthetic. Adding a new genre only requires a new dict entry in `GENRE_PROFILES`. Genres: `history`, `xuanhuan_shuang`, `system_stream`, `urban_ability`, `romance_female`, `wanzu_xuanhuan`, `suspense`. (`rule_horror` falls back to `suspense`.)
- `write_chapter_with_candidates` generates `candidate_chapters` parallel drafts at spread temperatures (`base ± 0.08·offset`), reviews each, keeps the highest-scoring
- `write_chapter` injects a RAG `retrieval_block` (see below) into the writer prompt so early concrete facts that summary compression erased are back in context
- `revise_chapter` first tries surgical `apply_review_patches` (replace/insert_after/delete by literal substring locator); only falls back to a full LLM rewrite when fewer than `revise_patch_min_frac` of patches apply cleanly
- `revise_hook_only` rewrites only the last ~400 chars when `hook_strength < hook_strength_min`, copying the head verbatim

### Quality control (`quality.py`, `retrieval.py`, plus checks in `review.py`)
The pipeline's biggest failure mode is **style collapse**: prose drifts into
telegraphic em-dash fragments (`句子——状态——状态`) that the model's own
self-review happily rates 9+, because its voice has drifted with the prose. The
following layers exist specifically because LLM self-assessment can't be trusted
to catch its own degeneration.

- **`quality.py:style_health(text, config)`** — deterministic, non-LLM prose
  metrics: em-dash density per kchar (`style_em_dash_per_kchar_warn`/`_bad`), avg
  sentence length (`style_min_avg_sentence_chars`), fragment-line ratio
  (`style_fragment_line_ratio_max`), dialogue presence. Returns a `penalty`
  (capped at `style_penalty_cap`), `flags`, and writer `directives`. Wired into
  `review.py:review_chapter`, which **subtracts the penalty from the LLM score**,
  blocks accept when penalty ≥ `style_penalty_block`, and injects the directives
  into the next chapter's writer prompt. Gated by `style_health_enabled`.
- **`quality.py:scene_similarity(plan, recent_plans)`** — Jaccard similarity of a
  plan's scene skeleton (conflict/payoff/pressure/goal/beats) vs recent selected
  plans. Three escalation levels in `planning.py:create_plan`: WARN appends
  `required_constraints` at `scene_dedupe_sim_warn`; BLOCK forces a plan retry at
  `scene_dedupe_sim_block` (relaxed to `scene_dedupe_short_novel_block` in
  chapter-capped mode, but no longer disabled there); `scene_dedupe_sim_identical`
  (0.97) is an absolute ceiling that forces retry in EVERY mode (v11 Ch8 shipped a
  max_sim=1.0 plan when short-novel mode disabled the retry). Candidates are also
  pre-filtered at generation time (`scene_dedupe_candidate_block`).
  Gated by `scene_dedupe_enabled`.
- **`quality.py:cross_chapter_repetition`** — detects signature clauses reused
  verbatim across chapters. Returns a `level`: `advise` (penalty + avoid-list
  directive) or `reject` when fossils ≥ `style_cross_repeat_reject_count` (8).
  A `reject` makes `review_chapter` mark the report `accepted=False` with a
  structured `gate_rejects` entry; `pipeline._classify_replan_failure` routes any
  `gate_rejects` straight to STRUCTURAL replan (never wording patches), and
  `_build_replan_feedback` injects the concrete fossil clauses as hard avoid
  evidence into the new plan. Rationale: v11 carried fossils 9–25 for 6 straight
  chapters on advisory directives alone and never recovered.
- **`retrieval.py`** — dependency-free TF-IDF char-bigram RAG (no embeddings — the
  only dependency is `openai`). `index_chapter` is called idempotently from
  `save_chapter` and writes `logs/retrieval_index.json`; `retrieval_block` builds
  a "## 相关历史原文（检索…）" section from the plan's fields for the writer prompt.
  `backfill_index` indexes a finished book. Gated by `rag_enabled` (`rag_top_k`,
  `rag_exclude_recent`).
- **`retrieval.exemplar_block`** (gated by `exemplar_rag_enabled`, default false) —
  quotes the book's own strongest chapters back to the writer as style anchors.
  Selection is **rank-based** (top `exemplar_rag_top_k` by score plus a small
  on-type bonus), NOT an absolute score threshold: 1021 measured chapters
  self-score 7.4–8.7, so the old `exemplar_rag_score_min: 8.8` selected nothing.
  The type bonus is deliberately small (0.3 payoff / 0.15 conflict) because an
  on-type mediocre chapter is a worse model than an off-type excellent one. It
  picks one dialogue-dense and one action/scene-dense exemplar (measured by
  `_dialogue_ratio`) so the two are not redundant, caps and caches the file reads,
  and preserves paragraph indentation — the old newline-flattening was itself
  nudging the prose telegraphic.
- **`review.py:cold_reader_review`** — an independent terminal review run every
  `cold_reader_every` chapters that **deliberately omits the cacheable_prefix**, so
  it cannot ratify the drifting voice the way the main reviewer (which shares the
  drifted context) does. Gated by `cold_reader_enabled`.
- **`review.py:macro_progress_check`** — every `macro_progress_every` chapters
  (from Ch20), measures plot advancement against `volume_plan` anchors and persists
  acceleration directives into `final_review.json` when stalled past
  `macro_progress_stall_threshold`. Gated by `macro_progress_enabled`.
- **`review.py:refresh_voice_anchors`** — anchors to a frozen `voice_baseline.md`
  (captured the first time it runs) instead of re-deriving voice from recent prose,
  and **skips the refresh entirely** when recent prose shows collapse
  (`voice_refresh_skip_penalty`). This closes the voice.md self-feeding loop where
  degraded prose became "the book's voice."
- **`quality.py:dialogue_health`** — deterministic dialogue-ratio gate. Measures
  text inside `"…"` Chinese curly quotes as a fraction of total chars. Penalty
  when ratio < `dialogue_char_ratio_min` (default 0.10), target is
  `dialogue_char_ratio_target` (0.20). Wired into `review.py` like `style_health`.
  The writer prompt also gets a dialogue-ratio warning when recent chapters run low.
  Gated by `dialogue_health_enabled`.
- **Chapter length penalty** — deterministic check in `review.py`: chapters shorter
  than `chapter_min_chars` (default 2800) get a proportional penalty (up to
  `chapter_length_penalty_cap`). The writer prompt injects the floor as a directive.
- **Opening diversity** — `writing.py:_prewrite_quality_contract` reads the first
  line of the last 5 chapters and injects them into the writer prompt with a
  diversity requirement, preventing consecutive same-type openings. Gated by
  `opening_diversity_enabled`.

### Rework trigger + repair ladder (`pipeline._rework_needed`, `fix.py`)
ONE predicate decides whether a draft has to be reworked, and it is called at
four sites: the per-round early break in `_stage_review_revise`, the replan gate
in `_stage_quality_replan`, `_stage_force_accept`, and the resume-authority check
in `generate_one_chapter`. Mode is set by `rework_trigger`:

- `score` (**default**) — the historical behaviour, bit for bit:
  `score < quality_threshold or not accepted`. `tests/test_fix.py` asserts
  point-for-point equality with the old expression over a
  (score × accepted × gate_rejects) grid, so the default config cannot drift.
- `deterministic` — rework only on measured evidence: `_hard_block_reasons`
  (gate_rejects, style collapse ≥ `style_penalty_block`, hard contradictions,
  hard contract violations, `length_band`/`opening_hook_gate` block,
  adjacent-repeat block, ≥ `constraint_violation_block_count` unmet
  constraints), a score below `rework_score_floor` (6.5), or an `accepted=False`
  the threshold cannot explain. Whole-chapter rewrite survives as the
  below-floor rescue path.

Why the second mode exists: `quality_threshold` is 8.0 and the library's 1023
measured self-scores have a median of **exactly 8.00** (34% below 8.0, only 6%
below 6.5), so ~1/3 of chapters enter the rework loop *by construction* — while
that same self-score has no demonstrated discrimination (1021 chapters span
7.4–8.7, 6 rejections). The pipeline currently answers noise with structural
replans, the #1 rework cause in every novel. **Do not "fix" a low FPY by moving
`quality_threshold`; that just relocates the median problem.**

Two things are load-bearing here:
- **`RevisionTracker.record()` cannot deliver this.** It reports `converged`
  only when `score >= threshold AND accepted`, and `accepted` is itself derived
  from `quality_threshold` (`review.py:1486`), so lowering the tracker's
  threshold can never release a 7.6 chapter. The release has to be an explicit
  `_rework_needed` break after each review round.
- **Ledger hygiene.** A 7.x chapter accepted in deterministic mode goes through
  `_accept_without_debt`: it is NOT stamped `force_accepted` and NOT written to
  `quality_debt.json` (it records a `quality_note` db_event instead). Otherwise
  `consecutive_force_accept_limit` × `circuit_breaker_score_floor` and refine
  prioritization would be swamped by noise-band chapters. `rework_score_floor`
  is aligned with `circuit_breaker_score_floor` for the same reason.

`fix.py` is the repair ladder that catches what no longer triggers rework. Layer
membership is declared ON the gate (`@REGISTRY.register(..., repair="L0"|"L1"|
"L2"|"advisory")` in `quality.py`); `fix.ACTION_BY_GATE` only maps a gate to an
action, so declaring a layer is never a promise that a fixer exists.
- **L0** (zero LLM, always runs): em-dash reduction, fragment-line merging,
  bank-only fossil rotation, scenery-opening demotion.
- **L1** (≤ `fix_max_l1_calls` bounded calls, skipped for force-accepted drafts):
  targeted expand-to-band, dialogue injection, em-dash rewording — each
  extracts a handful of passages, rewrites them in one numbered-list call, and
  splices them back. Never a whole-chapter rewrite.
- Every fixer is **keep-only-if-the-metric-improved** (same pattern as
  `_beat_gate_one` and the revision-gate rollback), which is what makes
  `_stage_fix` safe to run unconditionally.
- `_stage_fix` records `style_health_after_fix` rather than overwriting
  `style_health`: the latter is the measurement `score` was computed from, and
  overwriting it would leave score and penalty describing different texts.
- Two gates store their result under a key that is NOT the gate name
  (`length_band_check`→`length_band`, `book_wide_fossils`→`book_fossils`); read
  them via `fix.gate_result`. Also do NOT use `REGISTRY.is_enabled` as a
  "did this gate run" test — `length_band_check`'s config key only controls its
  penalty (default false) while the gate always runs.
- Fossil rotation is **bank-only on purpose**: of 109 distinct fossil phrases
  across 647 archived reviews, `FOSSIL_REPLACEMENTS` covers the generic-cliché
  subset (`声音压得很低`, 49 firings) and the rest are book-specific proper nouns
  (`老市场街七号`, 42) that must never be rotated — swapping those is canon
  corruption, not repair.

Zero-LLM tooling: `python tools/gate_census.py` (per-gate ran/fired/penalty over
archived reviews — the data behind any gate-deletion decision; measured total is
0.42 penalty per review over 651 reviews) and `python tools/replay_l0.py` (replays
L0 over finished chapters; 647 reviews → 29 chapters repaired, 0 made worse, all
length changes within ±2%).

**Read the census's two columns separately.** `fire%` is a *verdict* (penalty,
level, block, or flagged spans); `advise%` is a directive with no verdict behind
it. Conflating them is not cosmetic: the gates do not share one result shape, and
counting any non-empty list as a firing reports `information_density` at 91% when
its actual verdict rate (`low_information`, ≥3 of 4 probes) is 6.8%. Conversely a
penalty-only test scores that same gate a structural 0/649 — it emits no penalty
at all — which reads as "dead gate" when it means "never measured."

**A silent gate is a bug report, not a deletion candidate.** Before deleting one,
compare its threshold against the metric's measured distribution:
- `genre_adherence` showed 0 firings in 215 runs because `review.py` called a
  `store.get_connection` that **does not exist**; the `AttributeError` was
  swallowed by a bare `except: pass`, so `recent_genre_scores` was always empty
  and `low_streak` could never exceed 1 — while tangshuting's own
  `chapter_metrics` holds negative-score streaks of up to 8. Fixed to use the
  `conn` `review_chapter` is already handed.
- Fixing the wiring alone would have been worse than the bug. `genre_score` is a
  signed keyword-density difference whose library-wide median is **exactly
  0.000** (neither keyword list matched — no evidence), so the old
  `genre_drift_threshold: 0.3` scores "no evidence" as "drift": replayed over 357
  real scores it puts **46.8%** of chapters over the reject streak (86% in some
  novels), and a reject forces a STRUCTURAL replan. The threshold must sit
  strictly below zero; `-1.0` replays to warn 4.8% / reject 2.5%. The reject
  branch is additionally gated off by default (`genre_drift_reject_enabled:
  false`) because it has never executed in production.
- `dialogue_pingpong` (threshold 0.50 vs observed max qa_ratio **0.140**) and
  `chapter_ending_quality` (threshold 3 summary markers vs observed max **1**)
  were unreachable by construction, had no entry in `fix.ACTION_BY_GATE`, and
  each duplicated a gate that does fire (`dialogue_health` at 34.8%; the
  `hook_strength`/`revise_hook_only` path). Both deleted.
- `adjacent_repetition` also shows 0/641, but its warn line (0.10 clause overlap)
  sits just above the observed max (0.090) rather than 3× above it, it feeds
  `_hard_block_reasons`, and the repo has a recorded true positive above its
  block line (suspense_v11 Ch3, overlap 0.73). Kept.

### Fossil fix (`fossil_fix.py`)
Post-processing tool: `python novel.py fix-fossils <name>` scans finished
chapters for CJK n-gram fossils (phrases recurring in ≥15% of chapters) and
replaces excess occurrences with rotated synonym variants from
`FOSSIL_REPLACEMENTS`. Keeps `--max-keep` (default 1) per chapter; uses
`--custom-replacements <json>` for user-defined mappings. Reads from
`chapters_refined/` if available, writes to `chapters_fixed/` + `book_fixed.md`.
Zero LLM calls — purely deterministic.

### Adaptive cost control (`planning.py`)
- Inverted cost model: the DEFAULT is cheap (`candidate_plans: 1`, `candidate_chapters: 1`)
  and breadth is spent only on trouble. `_effective_candidate_count` RISK UPSHIFT
  (always on, from Ch3, no warmup) WIDENS the candidate count to
  `risk_upshift_candidates` (default 3) when the last `risk_upshift_window` chapters
  show a score below `risk_upshift_score_floor` or a style penalty ≥
  `risk_upshift_style_penalty`, or when a degradation-recovery directive is active —
  collapse recovery is when plan diversity pays. STABLE DOWNSHIFT (gated by
  `adaptive_downshift_enabled`, only meaningful for multi-candidate bases) drops one
  candidate once quality is stably ≥ `adaptive_downshift_score`. The structural
  replan path independently forces multi-draft sampling (`structural_replan_candidates`).

### Experiment harness (`compare.py`)
- `novel.py compare <a> <b>` — deterministic, zero-LLM side-by-side report
  (per-chapter scores/style penalties, force-accepts, quality-debt/gate-reject
  events, fossil warnings, scene-dedupe hits, LLM cost + planning share, non-secret
  config diff, heuristic verdict). Saved to `experiments/<a>_vs_<b>.md`. Calibrated
  against known ground truth: it must judge v10 over v11.
- `novel.py ablate <name> --flip <key> [--set V] [--chapters N]` — scaffolds
  `novels/<name>__ablate_<key>/` with the same prompt, ONE config key flipped, and
  `max_chapters` capped (default 8). Run it like any novel, then `compare` it
  against the source. Metadata saved under `experiments/ablate_*.json`. Every
  engine change should carry an ablation report instead of a hand-compared full
  rerun.
- `novel.py fork <name> --as <new> [--flip <key>] [--set V] [--chapters N]` —
  **the A/B tool to reach for on anything mid-book.** `ablate` always restarts at
  Ch1, and this repo's own recorded lesson is that short opening runs fabricate
  positive results (score inflation on short chapters, no mid-book problem zone).
  `fork` branches at HEAD instead: it copies `memory/`, `chapters/`, `book.md`,
  `state.md`, `story_state.db` and the RAG index, so both arms start from a
  byte-identical mid-book state. It forks at HEAD **only** — memory markdown and
  the entity/thread tables describe the book as of its last written chapter and
  there is no faithful rollback to an earlier one.
  - `logs/` is deliberately NOT copied (except the RAG index, the one logs
    artifact carrying story content), so FPY / cost / reasoning-coverage describe
    only the chapters the fork writes.
  - Budgeting prefers raising `target_words` over setting `max_chapters`, because
    `max_chapters` switches on the ending-aware machinery and would make the tail
    unrepresentative. When the source already has `max_chapters`, it extends that.
  - Metadata lands in `experiments/fork_<new>.json`. Run both arms, verify
    reasoning coverage matches in `novel.py stats`, then `novel.py compare`.
  - **Check `consecutive_force_accept_limit` against the source's tail scores
    before launching.** The force-accept circuit breaker in
    `pipeline._stage_force_accept` counts backwards through `chapter_metrics`,
    which the fork inherits, so a source whose last chapter scored below
    `circuit_breaker_score_floor` gives every fork a hair trigger: with the
    default limit of 2, one weak first chapter kills the run outright
    (`RuntimeError: Circuit breaker…`, seen killing an arm at Ch26 off a Ch25 of
    4.6). Raise the limit in BOTH arms so the inherited chapter can't decide the
    experiment — and raise it identically, or you have added a second variable.
  - `novel.py run` on Windows does not return until the pipeline child exits
    (the PowerShell `Start-Process` launcher blocks in `check_output`), so launch
    it as a background job and confirm with `grep -c "Start target_chars"` in
    `run.log` rather than waiting on the command. Note also that this venv is a
    **virtualenv** whose `python.exe` is a redirector stub, so every run appears
    twice in a process list (`stub → real interpreter`); that is one pipeline,
    not two, and `logs/run.pid` records the real one.

### Cross-book telemetry (`telemetry.py`)
Each novel runs as an isolated process with its own `story_state.db`.
`telemetry.py` is the ONE shared sink: `telemetry/global.db` (WAL, one fresh
connection per write so N novel processes write concurrently). It is a strict
observer / safe no-op: any failure (db missing, locked, malformed) returns an
empty value and never stalls a chapter. Live double-writes from the pipeline
(`record_chapter_metrics`/`record_event`/`record_arbitration`/`record_revise_pair`)
plus idempotent `backfill_novel` (`novel.py telemetry backfill`).

Telemetry is currently **write-only** (pure logging + `telemetry stats`). The
consumption layers that read it back into generation (distill → craft rules,
cross-book bandit prior, reader_panel) were deleted in the MVP refactor because
the library lacks the ≥5-book sample size where they beat noise; recover them
from git history (commit `9dd1ec0` and earlier) when that scale is reached.

### Memory layers (`memory.py`)
Two distinct context builders feed different LLM calls:
- `cacheable_prefix` — exact-bytes prefix shared across calls (creative brief + voice + bible + characters), keyed by sha1 of source files. Identical bytes ⇒ provider prompt-cache hits. **Whenever you change how this string is assembled, you invalidate the cache for every existing chapter.**
- `writing_memory_context` — small variable portion (state + threads + recent metrics + volume plan head) for write/revise/review hot path
- `memory_context` — full layered context (4 tiers, char-budgeted) for plan generation and event extraction
- `lite_memory_context` — heavily abbreviated for plan-review/screening

`volume_plan.md` is the one memory file that grows linearly with the book (a new
`## 第N卷` per volume, plus `extend_volume_schedule` APPENDING per-chapter schedule
tables), so plain head truncation silently starves the mid-book: a novel at Ch41
saw only 第一卷（Ch1-24）and none of the current volume's 角色高光轮值表 /
爽点兑现节拍表 / 反转排期 / 伏笔兑现 tables — the schedules were never executed
(ensemble cast collapsed to a two-hander, `payoff_deferred`, `tension_flat`).
`memory.py:volume_plan_window(text, chapter_num, cap, lookahead)` replaces head
truncation at all three read sites: it keeps blocks whose header declares no
chapter range or a range containing `chapter_num`, reduces out-of-range volumes to
a header breadcrumb, and inside kept blocks keeps only `| ChN |` rows for
`[chapter_num-1, chapter_num+lookahead]`. Rangeless sub-blocks inherit their
nearest ranged ancestor's decision; if no block covers the chapter, the nearest one
is kept verbatim. Gated by `volume_plan_window_enabled`.
Visibility alone doesn't produce compliance, so `memory.py:chapter_schedule_directive`
quotes this chapter's own schedule rows as a hard obligation into BOTH
`generate_candidate_plans` and `arbitrate_plan` (the arbiter must push the row into
`required_constraints`). Gated by `chapter_schedule_directive_enabled`.

Per-chapter state persistence is a SINGLE LLM call: `extract_events` returns the
extraction JSON **including** `protagonist_state` + `next_12_directions`;
`update_structured_state` (pure DB writes) and `update_state_file` (deterministic
markdown render of those fields) consume it with zero further LLM calls.

`compress_all_memory` consolidates per-chapter `## ChN` sections in bible/characters/timeline/threads when files exceed `memory_max_kb` or every `memory_compress_every` chapters; archives the old sections under `logs/memory_archive/`.

### Shared prompt constants (cross-module deduplication)
Several prompt fragments are shared across modules to eliminate redundancy and
ensure consistency:
- `memory.py:STYLE_HEALTH_GUARDRAILS` — the "健康文风护栏" prose-health guardrails,
  referenced by `VOICE_CHAIN_SYSTEM` (memory.py) and `VOICE_ANCHOR_SYSTEM` (review.py)
- `memory.py:_VOLUME_PLAN_STRUCTURE_SPEC` — the volume plan output format (OKR structure,
  线索兑现表, pacing discipline), referenced by `VOLUME_PLAN_CHAIN_SYSTEM` (memory.py)
  and `REPLAN_SYSTEM` (review.py) so both produce structurally identical volume plans
- `planning.py:_EXECUTABILITY_DOCTRINE` — the executability scoring doctrine (score
  baseline 6.5, "shootable action" definition, reversal structure requirement),
  referenced by `CANDIDATE_PLAN_SYSTEM` and `ARBITER_SYSTEM`

### Persistence (`store.py`)
SQLite (`story_state.db`, WAL mode) is the primary store with tables `events`,
`chapter_metrics`, `entities`, `open_threads`, `agent_reports`, `stage_constraints`,
`causal_links`. If `sqlite3` is unavailable, `JsonStoryStore` writes `logs/story_state.json`
as a fallback — most code branches on `isinstance(conn, JsonStoryStore)` and a few
features (stage constraints, causal links, plan-continuity validation, silent-thread
detection) are SQLite-only.

The RAG index (`logs/retrieval_index.json`) and the frozen voice anchor
(`memory/voice_baseline.md`) are separate per-novel artifacts written outside the
SQLite store; both are safe to delete and will be rebuilt (the index by
`retrieval.backfill_index` / on the next `save_chapter`, the baseline on the next
`refresh_voice_anchors`).

### Checkpoints (`checkpoint.py`)
Every stage in `generate_one_chapter` writes a checkpoint under `logs/checkpoints/ch{NNNN}/`:

```
plan_initial_attempt0_candidates.json
plan_initial_attempt0_reports.json
plan_initial_attempt0_arbitration.json
plan_initial_selected.json
validated_plan.json
chapter_current_v2.md          ← versioned via CHECKPOINT_VERSION
review_round0.json … final_review.json
chapter_saved.json
extraction.json → structured_state_done.json → state_file_done.json → chapter_completed.json
```

Resume detection lives in `should_resume_existing_chapter`: chapter file exists AND checkpoint dir exists AND `chapter_completed.json` does not. Bumping `CHECKPOINT_VERSION` invalidates all `.json` checkpoints from prior versions.

### LLM calls (`llm.py`)
`call_llm` handles streaming with three timeouts (`stream_timeout` total / `stream_idle_startup` / `stream_idle_steady`), salvages partial output past `stream_salvage_min_chars`, falls back to `reasoning_content` when `content` is empty, retries refusals (REFUSAL_PATTERNS), and emergency-truncates user messages by section priority when prompt exceeds `context_window * 1.8` chars.

`load_json_with_repair` calls `safe_json_loads` (which itself runs `_repair_truncated_json` for cut-off streams), and on failure asks the LLM to repair the JSON. It returns `fallback` instead of raising when one is provided. Refusal-prefixed responses skip the repair attempt.

When the JSON contract matters, prompts are wrapped in `json_prompt(user)` which appends the mandatory output contract block. `call_llm` infers JSON mode from the presence of that string and sets `response_format={"type": "json_object"}`, automatically retrying without it when a provider returns a 400/404/422 mentioning `response_format`.

### Refine pass (`refine.py`)
Explicit manual step: `python novel.py refine <name>` (`refine_after_complete`
defaults to false). Reads finished `chapters/*.md` in 5-chapter groups, asks an LLM to assign per-chapter intensity (`polish` / `restructure` / `rewrite`) plus up to 4 anchor chapters from elsewhere in the book. Refined output goes to `chapters_refined/` and `book_refined.md`; `chapters/` and `book.md` are never modified. Per-group checkpoints under `logs/refine/group_NNNN.json` make the pass resumable. Sanity check `_refined_text_acceptable` rejects refines that shrink below `refine_min_keep_ratio` (default 0.6) or grow beyond an intensity-tiered ceiling (`polish` 1.5× via `refine_max_grow_ratio`, `restructure` 2.0×, `rewrite` 2.5×).

Diagnose prompts mirror the writer's shared-base pattern: `DIAGNOSE_CORE` (shared preamble + intensity definitions + common dimensions) + `DIAGNOSE_GENRE_DIMS[preset]` (genre-specific dimensions) + `DIAGNOSE_COMMON_FOOTER` (task steps + JSON schema), assembled by `_build_diagnose_system()`. Adding a genre's diagnose prompt only requires a new entry in `DIAGNOSE_GENRE_DIMS`.

### Screenplay conversion (`screenplay.py`)
Standalone novel-text → 短剧 (vertical-drama) script converter, decoupled from the
generation pipeline. `convert_file(input, out)` / `convert_text(...)` split input on
`第N章` markers (or char-budgeted paragraph packing when there are no markers), then
run **one LLM call per segment** with continuity carry-over (running 第N集 episode
number, last segment's tail) so episode/scene numbering stays monotonic across calls.
Output follows the reference duanju format: `第N集` → `N-N 地点 时段 内/外` → `人物：` →
`△`动作行 → `角色：台词` → `（字幕：…）` / `角色（OS）：旁白` / `（镜头：…）`. Per-segment
checkpoints under `<out>.checkpoints/seg_NNNN.json` make the pass resumable. Default
output goes to a `scripts/` dir: `novels/<name>/scripts/` for per-novel mode, or a
`scripts/` subdir next to the input file in standalone `--input` mode (override with
`--out`). It reuses
the engine's config-driven LLM client only for API keys; with no `--config`/`NOVEL_CONFIG`
it falls back to `config_template.yaml` (the shared keys). Tuned by `script_seg_chars`,
`script_max_tokens`, `script_temperature`. CLI: `python novel.py script --input PATH`
(any file) or `python novel.py script <name> --chapters A-B` / bare `<name>` (book.md).

## Things to be careful with

- **Don't add `cd <project>` before `git` commands** — bash already runs in the project root.
- **`config.yaml` is not real YAML.** Anchors, lists, nested maps will silently fail to parse; values become strings. The parser only understands `section:` and indented `key: value`.
- **`NOVEL_CONFIG`/`NOVEL_PROMPT` must be set before importing `pipeline`/`config`/`memory`.** `config.py` reads them at import time and `memory.py` captures `PROMPT_FILE` at its own import. `novel.py run` relies on this ordering — set the env vars first, import second.
- **Per-novel paths live entirely in each `config.yaml`'s `paths:` section**, joined onto `ROOT`. The engine has no hardcoded knowledge of `novels/`; isolation is purely a path convention. `config_template.yaml`'s `__NOVEL__` placeholder is what makes a new novel directory-isolated.
- **Background-task ordering** is load-bearing. The barriers in `generate_one_chapter` (`wait_label("chapter_finalize_ch{n-1}")` and the prefetch wait) keep memory/threads consistent. Re-ordering them can cause the next plan to see stale state.
- **`save_chapter` refuses to write chapters under 500 chars** (`writing.py:843`). This guards against provider refusals being persisted as legitimate chapters.
- **`cacheable_prefix` content changes invalidate the prompt cache** for every subsequent chapter — only modify it when the cache cost is worth it.
- **`cold_reader_review` must NOT use the cacheable_prefix.** Its entire value is being an independent judge that hasn't been steeped in the (possibly drifted) book context — sharing the prefix would defeat the point and re-introduce the rating inflation it exists to catch.
- **`style_health` is the objective anchor against score inflation.** Don't relax its thresholds to make chapters "pass"; the model's self-review already over-rates fragmented prose. The penalty is meant to fight that, not be tuned away.
- **`voice_baseline.md` is frozen on purpose.** `refresh_voice_anchors` anchors to it rather than re-deriving voice from recent prose; re-deriving from drifted prose is exactly the self-feeding loop that caused style collapse.
- **Live API keys sit in `config.yaml` / `config_template.yaml` / `novels/*/config.yaml`.** All are gitignored — don't echo them into tracked files or logs. New per-novel configs inherit the template's keys, so parallel novels share quota.
- **`config_template.yaml` is gitignored but must exist on disk** for `novel.py create` to work. Don't delete it. It was *also tracked* until 2026-07-28, which made the ignore rule inert (gitignore does not apply to already-tracked files) and left a live key one `git commit -a` away from publication — `git rm --cached` fixed that. The key never reached history; verified with `git log --all -S<key>` and a blob scan. When adding config keys, edit **both** `config_template.yaml` (your working copy, with keys) and the tracked credential-free `config_template.example.yaml`, which is what `novel.py create` falls back to on a fresh clone. `.gitignore` needs the `!config_template.example.yaml` negation because the broad `config_*.yaml` rule would otherwise swallow it.
- **`GENRE_PROFILES` shared constants affect all genres.** Modifying `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT`, `_SELF_REVIEW_PREAMBLE`, or `_OUTPUT_SECTION` in `writing.py` changes every genre's writer prompt at once. Per-genre overrides go in the `GENRE_PROFILES` dict entry (set `sensory_dialogue` or `time_marker_ban` to a non-empty string to override the default). Same applies to `DIAGNOSE_CORE`/`DIAGNOSE_COMMON_FOOTER` in `refine.py` and `_EXECUTABILITY_DOCTRINE` in `planning.py`.
- **Ending awareness (`ending_aware`, default true) only fires when `max_chapters` is set.** In short-novel mode, the final chapter (`chapter_num == max_chapters`) gets a `CLOSING_RULES_BLOCK` (writing.py) + a planning ending directive, skips hook-only-revise (pipeline.py), and refine's diagnose/refine prompts demand closure instead of a cliffhanger. Detection lives in `config.py:is_final_chapter`. Pure char-target long novels (no `max_chapters`) have no deterministic finale, so this is inert there and per-chapter behaviour is unchanged.
