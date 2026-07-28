# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**This file holds the rules and the pipeline you must not break.** Four
companion docs carry the rest — follow the pointers instead of guessing:

| doc | contents |
| --- | --- |
| `docs/LESSONS.md` | the measurements and post-mortems **behind** these rules. Read the cited section before changing a threshold, deleting a gate, or running an A/B. |
| `docs/INTERNALS.md` | mechanical reference: store schema, checkpoint layout, `llm.py` plumbing, shared prompt constants, refine / fossil-fix / screenplay tools, `tools/*` |
| `REDESIGN.md` | quality/FPY redesign roadmap v1 + P1–P4 execution record |
| `docs/REDESIGN_V2.md` | first-principles redesign v2. **Read §0 before adding any quality gate or A/B judged by `score`** — the self-score has no discrimination and `quality_threshold: 8.0` sits exactly on the library's median, so an experiment that changes rework rules cannot be settled with it. |

`README.md` is the user-facing quickstart.

## Overview

Universal multi-novel AI writing framework. The core engine is an automated
long-form Chinese web novel generation pipeline that targets a configurable
character count (`novel.target_words`) by repeatedly running plan → write →
review → revise → extract loops until done.

Architecture follows a "Less is More" MVP model (2026-07 refactor): a
single-candidate plan→write→review pipeline guarded by DETERMINISTIC quality
gates, with multi-candidate breadth spent only when risk signals fire. Modules
premature for the current scale were deleted; the recovery commit and the "bring
them back at ≥5 finished books" rule are in LESSONS §9.

The pipeline is **content-agnostic** — it consumes only a creative brief
(`prompt.md`) and a config (`config.yaml`). Each novel lives in `novels/<name>/`
and runs as an independent OS process, so multiple novels can run simultaneously
without colliding on the engine's process-level global state
(`config.PROMPT_FILE`, `memory._CACHEABLE_PREFIX_CACHE`).

## Entry point: `novel.py`

`novel.py` is the **only** entry point; all parsing and dispatch lives there
(`cmd_*` functions + an argparse subparser tree).

```bash
pip install -r requirements.txt          # only dependency is openai>=1.0.0
python -m unittest discover tests        # pure-function tests only, no LLM. No lint config, no build step.

# lifecycle
python novel.py create <name>            # scaffold novels/<name>/ from config_template.yaml + prompt_template.md
python novel.py run <name>               # detached; resumes from checkpoint (log -> novels/<name>/logs/run.log)
python novel.py run <name> --foreground  # run in the current console
python novel.py list                     # all novels: chapters / chars / running? / last log line
python novel.py stop|restart <name>      # per-novel control, token-exact: `run foo` never matches `run foobar`
python novel.py stats <name>             # per-chapter scores/penalties/LLM cost + per-role reasoning coverage

# openings & samples
python novel.py trial <name>             # opening variants WITHOUT touching chapters/book (-> logs/opening_trials/)
python novel.py adopt-trial <name> [id]  # adopt a trial's opening route into memory/opening_route.md
python novel.py benchmark list|add ...   # local 爆款 sample library (structural recall, never copies prose)

# experiments — read LESSONS §5 before running one
python novel.py compare <a> <b>          # deterministic zero-LLM side-by-side -> experiments/<a>_vs_<b>.md
python novel.py fork <name> --as <new> [--flip <key>] [--set V] [--chapters N]   # branch at HEAD (preferred)
python novel.py ablate <name> --flip <key> [--set V] [--chapters N]              # -> novels/<name>__ablate_<key>/, restarts at Ch1
python novel.py telemetry backfill|stats [--genre G]

# post-completion (details: docs/INTERNALS.md)
python novel.py refine <name>            # refine pass -> chapters_refined/ + book_refined.md (resumable)
python novel.py fix-fossils <name>       # deterministic fossil replacement -> chapters_fixed/ + book_fixed.md
python novel.py package <name>           # titles/intros/tags/synopsis for a finished novel
python novel.py script --input PATH      # ANY novel text file -> 短剧 screenplay (standalone)
python novel.py script <name> --chapters A-B   # or a novel's chapters / bare <name> for book.md
```

Multi-novel scaffolding is pure path convention — no engine changes:
- `create` copies `config_template.yaml` replacing the `__NOVEL__` placeholder so
  every `paths:` entry points inside `novels/<name>/` (e.g.
  `paths.book: novels/<name>/book.md`), and copies `prompt_template.md` to
  `novels/<name>/prompt.md` for the user to fill in.
- `run` sets `NOVEL_CONFIG`/`NOVEL_PROMPT` **before** importing `pipeline`.
  Detached launch prefers the project venv
  (`E:\pycharmproject\allvenv\novel\Scripts\python.exe`); override with
  `NOVEL_PYTHON`.
- Each novel's `story_state.db`, `logs/`, `checkpoints/`, `memory/` are isolated in
  its own directory, so concurrent novels never share SQLite/file writes. Keys come
  from each novel's own config, so parallel runs share RPM/TPM quota unless given
  distinct keys.

## Configuration

`config.yaml` (and each `novels/<name>/config.yaml`) is parsed by a hand-rolled
YAML-**subset** reader in `config.py:load_config` — only `section:` headers and
`key: value` pairs. No nested maps, no lists, no anchors. Mandatory new keys must
be added to the `required` dict in `load_config`.

`config.py:15-16` reads `NOVEL_PROMPT`/`NOVEL_CONFIG` from the environment
(defaults `prompt.md`/`config.yaml` in the project root, used only if unset).
`config.py:get_paths` joins each `paths:` value onto `ROOT`, which is what makes a
per-novel config directory-isolating with zero code changes.

Multi-endpoint, multi-key access — three keys in `api:`: `api_key` (single
primary), `api_keys` (comma/semicolon list for the primary `base_url`),
`api_key_groups` (`base_url|key1,key2;base_url2|key3,...` fallbacks).
`configured_api_endpoints()` returns `(endpoints, primary_count)`; `LLMClientPool`
rotates primary keys round-robin and falls back to secondary endpoints only when
all primaries are dead.

Per-role routing keys (`{planning,writing,extraction,review}_*`) carry two
independent reasoning knobs, because gateways honour one or the other:
- `{role}_thinking_mode` — `disabled`/`auto`/`enabled`
  (+`{role}_thinking_budget_tokens`), emitted as `extra_body["thinking"]`
  (Anthropic/豆包 style). `auto` omits it.
- `{role}_reasoning_effort` — OpenAI style (`none`/`low`/`medium`/`high`), emitted
  as a top-level body field via `extra_body`. Empty ⇒ never sent.

**Both are requests, not guarantees, and must never be an A/B variable** — the
gateway decides whether reasoning happens. Evidence, the two live gateway cases,
and why `api.stream: true` is mandatory for one of them: LESSONS §1. Check
`novel.py stats`'s `_reasoning_coverage` before trusting any comparison.

## Module map

| module | role |
| --- | --- |
| `pipeline.py` | top-level loop, per-chapter stage orchestration, rework/replan routing |
| `planning.py` | candidate plans, fused review, arbitration, adaptive candidate count |
| `arc.py` | alternative arc-level ChapterCard planner (`arc_planning_enabled`, default false) |
| `writing.py` | writer prompts (`GENRE_PROFILES`), multi-draft, revise/patch, `save_chapter` |
| `review.py` | chapter review + cold reader, macro progress, voice anchors, replan feedback |
| `quality.py` | deterministic gates (registry + metrics), zero LLM |
| `fix.py` | repair ladder L0/L1 — fix instead of re-roll |
| `taxonomy.py` | canonical failure-code vocabulary + fix routing (`failure_taxonomy_enabled`) |
| `retrieval.py` | dependency-free TF-IDF RAG + exemplar blocks |
| `memory.py` | context builders, memory files, compression, volume-plan windowing |
| `store.py` | SQLite persistence (`story_state.db`), JSON fallback |
| `checkpoint.py` | per-stage checkpoints + resume detection |
| `llm.py` | streaming, timeouts, salvage, JSON repair, refusal retry |
| `config.py` | YAML-subset config, paths, endpoints, `is_final_chapter` |
| `telemetry.py` | cross-book sink `telemetry/global.db` (write-only observer) |
| `refine.py`, `fossil_fix.py`, `screenplay.py`, `package.py` | post-completion / standalone tools |
| `compare.py` | `compare` / `ablate` / `fork` experiment harness |
| `trial.py`, `benchmark.py` | opening trials, local sample library |
| `tools/*.py` | zero/low-LLM analysis: `fpy_prime` (acceptance metric), `replay_gates` (settles gate-logic changes), `gate_census`, `replay_l0`, `prompt_census`, `pairwise_ab`, `probe_reasoning`, `defossil`, `rebuild_memory` |

## Architecture

### Top-level loop (`pipeline.py:main`)
1. `bootstrap()` once — generates `state.md`, `memory/{bible,characters,timeline,threads,volume_plan}.md` from `prompt.md`
2. Loop: `find_last_chapter()` → `generate_one_chapter()` until `count_chars(book.md) >= target_words`
3. `BackgroundTasks` thread pool runs finalization (extract + structured-state + state.md), stage reviews, memory compression, adaptive replans, and next-chapter plan prefetches off the critical path
4. On completion, optional `refine.refine_book()` if `novel.refine_after_complete: true` (default **false** — run `python novel.py refine <name>` manually instead)

### One chapter (`pipeline.py:generate_one_chapter`)
Strict ordering, with a barrier on the previous chapter's
`chapter_finalize_ch{n-1}` background label so memory/threads/metrics are fresh
before planning:

```
create_plan → validate_plan_continuity → write_chapter_with_candidates
            → review/revise loop (max_revision_rounds, no-improvement early stop)
            → optional revise_hook_only for weak endings
            → save_chapter → extract_events → update_structured_state → update_state_file
```

Critical invariant in `pipeline.py:_stage_finalize` (write site ~`pipeline.py:2038`):
`chapter_completed.json` must be written **synchronously** before submitting the
finalize background task. Deferred, the main loop's resume check re-enters
`Resuming partially indexed Ch{n}` and resubmits on every iteration, leaking
threads and memory.

### Planning (`planning.py:create_plan`)
1. `generate_candidate_plans` — N candidates (default N=1), each forced into a different strategy (`scene-driven`, `character-driven`, `thread-driven`, `institutional`, `reversal`, `pressure-payoff`) chosen by a Thompson-sampling bandit (Beta posterior on arbiter win-rates, `strategy_bandit_explore_frac` forced exploration) over historical `plan_arbitration` events. Candidates ≥ `scene_dedupe_candidate_block` (0.85) similar to a recently selected plan are dropped pre-review (unless all would be dropped)
2. Optional `screen_candidates` (skipped when `plan_skip_screen: true`, or automatically at ≤3 candidates)
3. `review_candidate_plans` — fused 6-axis review (world/character/rhythm/payoff/foreshadowing/reader) per candidate, ONE LLM call expanded into 6 legacy reports via `_explode_fused_axes` (the only review path)
4. `arbitrate_plan` — picks `selected_index`, emits `merged_plan` + `required_constraints`. Still runs with a single candidate: it merges rhythm diagnostics / recent quality feedback / used-element ledger into the plan

**The arbitration dict is untrusted input, keys included.** `llm.load_json_with_repair`
accepts any repair that merely *parses*, so a salvage fragment
(`{"./output.json": "{"}`, `.selected_index` beside empty values) is laundered into a
decision — 16 of 934 archived arbitrations, 14 of them in one book. Three pure
functions guard the readers: `_normalize_decision` (unwrap a single-key wrapper,
rebuild the envelope around a bare score row, repair `.`/quote/path-prefixed key
spellings — never invents a key, never overwrites a well-spelled one),
`_decision_usable` (has `scores` **or** a non-empty `merged_plan`), and
`decision_has_score`. When unusable, `arbitrate_plan` re-asks the ARBITER once
(`arbiter_reask_enabled`) and marks `decision["arbitration_failed"]` if that fails
too — recovery costs 1 call instead of the ~4 a whole plan round was paying.

**A missing measurement is not a low one.** `plan_score` returns 0.0 for an empty
`scores` list (its float contract is load-bearing for `chapter_metrics.plan_score`,
and `arc.py` leaves `scores` empty on purpose), but 0.0 is below every threshold, so
`create_plan`'s score gate read a parse failure as a terrible plan and bought a full
extra plan round — 12 library-wide, 48 archived calls where 11 would do. The gate now
consults `decision_has_score` and logs `plan_score_unavailable` instead. Re-rolling
candidates cannot fix an arbitration parse failure. LESSONS §13.

### Arc planner (`arc.py`, `arc_planning_enabled`, default **false**)
Alternative to the five-stage committee (REDESIGN L2): ONE high-reasoning call
every `arc_span` (10) chapters emits a **ChapterCard** per chapter, so 错峰兑现 /
场地轮换 / 开场轮换 / 整段推进 are decided once with whole-arc vision. Four rules
carry the design — full notes in LESSONS §11:
- **One seam**: `create_plan` calls `plan_from_arc` only when
  `checkpoint_label == "initial"` and `replan_feedback is None`; every replan keeps
  the committee. `pipeline.py` is untouched.
- **`card_to_plan` projects** a card onto the existing plan schema, so
  writing/review/quality/store need zero changes; card-only fields
  (`opening_type`, `forbid`, `turn`) ride along in the plan dict.
- **`plan_from_arc` never raises**, and `decision["scores"]` stays deliberately
  empty (a fake score would poison `chapter_metrics.plan_score`).
- **`validate_card` repairs on CRITICAL only** — advisories go into
  `required_constraints`, and scene-dedupe must union `_recent_selected_plans` with
  recent cards projected through `card_to_plan`.

### Writing & revision (`writing.py`)
- Writer prompts use a **shared-base + genre-delta architecture**: `GENRE_PROFILES`
  holds per-genre deltas (role, self_review, core_discipline, structure_template,
  genre_bans, `sensory_dialogue`, `time_marker_ban`, extras); `_build_write_system()`
  assembles them onto shared constants (`_SELF_REVIEW_PREAMBLE`, `_OUTPUT_SECTION`,
  `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT`) plus
  `ANTI_FRAGMENT_BAN`, `ANTI_PITFALL_BLOCK`, aesthetic. A new genre is one new dict
  entry. Genres: `history`, `xuanhuan_shuang`, `system_stream`, `urban_ability`,
  `romance_female`, `wanzu_xuanhuan`, `suspense` (`rule_horror` → `suspense`).
- `write_chapter_with_candidates` generates `candidate_chapters` parallel drafts at
  spread temperatures (`base ± 0.08·offset`), reviews each, keeps the best.
- `write_chapter` injects a RAG `retrieval_block` so early concrete facts erased by
  summary compression are back in context.
- `revise_chapter` first tries surgical `apply_review_patches`
  (replace/insert_after/delete by literal substring locator), falling back to a full
  LLM rewrite only when fewer than `revise_patch_min_frac` of patches apply.
- `revise_hook_only` rewrites only the last ~400 chars when
  `hook_strength < hook_strength_min`, copying the head verbatim.

### Quality control (`quality.py`, `retrieval.py`, checks in `review.py`)
These layers exist because LLM self-assessment cannot catch its own degeneration —
prose drifts telegraphic and the model rates it 9+. Incident record: LESSONS §3.

| gate / layer | config | behaviour |
| --- | --- | --- |
| `quality.style_health(text, config)` | `style_health_enabled`, `style_em_dash_per_kchar_warn`/`_bad`, `style_min_avg_sentence_chars`, `style_fragment_line_ratio_max`, `style_penalty_cap`, `style_penalty_block` | deterministic prose metrics (em-dash density, avg sentence length, fragment-line ratio, dialogue presence) → `penalty`, `flags`, writer `directives` |
| `quality.scene_similarity(plan, recent_plans)` | `scene_dedupe_enabled`, `scene_dedupe_sim_warn`/`_block`/`_identical`, `scene_dedupe_short_novel_block`, `scene_dedupe_candidate_block` | Jaccard similarity of a plan's scene skeleton (conflict/payoff/pressure/goal/beats) vs recent selected plans |
| `quality.cross_chapter_repetition` | `style_cross_repeat_reject_count` (8) | signature clauses reused verbatim across chapters → a `level` of `advise` or `reject` |
| `quality.dialogue_health` | `dialogue_health_enabled`, `dialogue_char_ratio_min` (0.10), `dialogue_char_ratio_target` (0.20) | dialogue-ratio gate over text inside `"…"`; the writer prompt also warns when recent chapters run low |
| `quality.book_wide_fossils` | `book_fossil_enabled`, `book_fossil_chapter_frac` (0.30), `book_fossil_min_chapters` (6), `book_fossil_hard_ratio`, `book_fossil_struct_count` | 6-char CJK n-grams recurring across a large fraction of the WHOLE book (what `cross_chapter_repetition`'s 6-chapter window structurally misses). **A hard fossil requires `current_chapter` and only indicts a chapter that actually contains the phrase** — the ratio is book-cumulative, so without that the gate latches on and rejects compliant chapters forever. `book_fossil_hard_ratio` is floored at the candidacy fraction (below it, every candidate is automatically hard). LESSONS §13 |
| `quality.chapter_mode_monotony` | `chapter_mode_enabled`, `chapter_mode_baseline`, `chapter_mode_window` (6), `chapter_mode_min_window` (4), `chapter_mode_warn_frac`/`_block_frac` | frequency of the current chapter's coarse form across a window → warn/block (block forces a plan retry). **The frac is counted with the UNBIASED classifier even under a genre baseline**; the biased classifier returns the baseline label unless a chapter clearly breaks form, so under a baseline the frac floors near 1.0 and is not comparable to `_block_frac` at all. The biased label is still what gets reported and named in the directive. LESSONS §13 |
| chapter length | `chapter_min_chars` (2800), `chapter_length_penalty_cap` | proportional penalty below the floor; the floor is also a writer directive |
| opening diversity | `opening_diversity_enabled` | `writing.py:_prewrite_quality_contract` injects the first line of the last 5 chapters + a diversity requirement |
| `retrieval.py` RAG | `rag_enabled`, `rag_top_k`, `rag_exclude_recent` | dependency-free TF-IDF char-bigram index (no embeddings). `index_chapter` is called idempotently from `save_chapter` → `logs/retrieval_index.json`; `retrieval_block` builds the "## 相关历史原文（检索…）" section; `backfill_index` indexes a finished book |
| `retrieval.exemplar_block` | `exemplar_rag_enabled` (false), `exemplar_rag_top_k` | quotes the book's own strongest chapters back as style anchors. **Rank-based, not an absolute score threshold**; picks one dialogue-dense + one action-dense exemplar (LESSONS §7) |
| `review.py:cold_reader_review` | `cold_reader_enabled`, `cold_reader_every` | independent terminal review that **deliberately omits the cacheable_prefix**, so it cannot ratify the drifting voice the main reviewer shares context with |
| `review.py:macro_progress_check` | `macro_progress_enabled`, `macro_progress_every`, `macro_progress_stall_threshold` | from Ch20, measures advancement against `volume_plan` anchors; persists acceleration directives into `final_review.json` when stalled |
| `review.py:refresh_voice_anchors` | `voice_refresh_skip_penalty` | anchors to a frozen `voice_baseline.md` instead of re-deriving voice from recent prose, and skips the refresh entirely when recent prose shows collapse |
| failure taxonomy | `failure_taxonomy_enabled` (true) | `review.py` tags each report with canonical `failure_codes` from `taxonomy.py`; `_classify_replan_failure` prefers `taxonomy.replan_kind`/`dominant_route` over free-text prefixes. Pure, additive, degrades to the legacy path on import failure |
| `quality.fingerprint_avoidance_context` | `fingerprint_enabled` | the 全书结构指纹 block in the plan prompt. **Emits an aggregate (recurring move bigrams/trigrams + payoff/conflict/move frequencies), never one line per chapter** — the per-chapter form was 19.6% of the largest prompt and grew linearly while carrying no signal (194 distinct flows in 200 chapters). `store_chapter_fingerprint` keeps writing `skeleton_tokens` even though nothing reads it: that column is what lets a future recalibration be replayed offline. `fingerprint_warn_threshold` is now **dead** — its only reader, `check_plan_against_fingerprints`, was deleted as unreachable (max 0.448 measured vs 0.65). LESSONS §8 |

Three routing rules inside that table are load-bearing:
- **`style_health`'s penalty is subtracted from the LLM score.** `review.py:review_chapter`
  subtracts it, blocks accept at `style_penalty_block`, and injects the directives
  into the next chapter's writer prompt.
- **Scene dedupe escalates in three steps.** WARN appends `required_constraints`;
  BLOCK forces a plan retry (relaxed to `scene_dedupe_short_novel_block` in
  chapter-capped mode, but **not** disabled there); `scene_dedupe_sim_identical`
  (0.97) is an absolute ceiling that forces retry in EVERY mode.
- **A `cross_chapter_repetition` reject is structural, not cosmetic.** It marks the
  report `accepted=False` with a `gate_rejects` entry;
  `pipeline._classify_replan_failure` routes any `gate_rejects` straight to
  STRUCTURAL replan (never wording patches), and `_build_replan_feedback` injects
  the concrete fossil clauses as hard avoid evidence.

### Rework trigger + repair ladder (`pipeline._rework_needed`, `fix.py`)
ONE predicate decides whether a draft must be reworked, called at four sites: the
per-round early break in `_stage_review_revise`, the replan gate in
`_stage_quality_replan`, `pipeline._stage_force_accept`, and the resume-authority
check in `generate_one_chapter`. Mode is `rework_trigger`:
- `score` (**default**) — historical behaviour bit for bit:
  `score < quality_threshold or not accepted`.
- `deterministic` — rework only on measured evidence: `_hard_block_reasons`
  (gate_rejects, style collapse ≥ `style_penalty_block`, hard contradictions, hard
  contract violations, `length_band`/`opening_hook_gate` block, adjacent-repeat
  block, ≥ `constraint_violation_block_count` unmet constraints), a score below
  `rework_score_floor` (6.5), or an `accepted=False` the threshold cannot explain.
  Whole-chapter rewrite survives as the below-floor rescue path.

**Do not "fix" a low FPY by moving `quality_threshold`.** Why the second mode
exists, why `RevisionTracker` cannot deliver it, and the `_accept_without_debt`
ledger-hygiene requirement: LESSONS §2.

`deterministic` **did not pass its A/B and is not the default** — it came out
*more* expensive (16.25 vs 14.25 calls/chapter). Releasing the 7.x band trips RISK
UPSHIFT on the next chapter, and because a single plan candidate skips fused plan
review entirely, widening 1→3 candidates is not "2 more calls" but "0 reviews → 3
reviews": `plan_candidate`+`plan_review_fused` went 11→25 while `revise` fell 4→2.
`planning._risk_score_floor` now lowers the risk floor to `rework_score_floor` in
this mode so the two rules stop fighting over the same undiscriminating score; that
fix is itself **unmeasured** and inert in `score` mode. Full numbers, the FPY′
re-settlement, and the re-run precondition: REDESIGN §7 "P4 A/B 结论".

`fix.py` is the repair ladder that catches what no longer triggers rework. Layer
membership is declared ON the gate
(`@REGISTRY.register(..., repair="L0"|"L1"|"L2"|"advisory")` in `quality.py`);
`fix.ACTION_BY_GATE` only maps a gate to an action, so declaring a layer is never a
promise that a fixer exists.
- **L0** (zero LLM, always runs): em-dash reduction, fragment-line merging,
  bank-only fossil rotation, scenery-opening demotion.
- **L1** (≤ `fix_max_l1_calls` bounded calls, skipped for force-accepted drafts):
  targeted expand-to-band, dialogue injection, em-dash rewording — each extracts a
  handful of passages, rewrites them in one numbered-list call, splices them back.
  Never a whole-chapter rewrite.
- Every fixer is **keep-only-if-the-metric-improved** (same pattern as
  `_beat_gate_one` and the revision-gate rollback), which is what makes `_stage_fix`
  safe to run unconditionally. **A metric check cannot catch broken Chinese**, so any
  grammar guard must be structural instead: `fossil_fix._safe_alt` refuses a variant
  that would follow an attributive 「的/之」 without sharing the phrase's head noun
  (「顾峥的声音压得很低」 → 「顾峥的压着嗓子」), and keeping the fossil beats emitting
  that.
- **The repair target is per-gate, and getting it wrong makes the fixer a no-op.**
  Density gates (`cross_chapter_repetition`, `descriptor_frequency`) are answered by
  keep-1 rotation; a `book_wide_fossils` hard reject is a book-cumulative ratio and
  can only be cleared by ZERO occurrences in this chapter, so `fix.rotate_fossils`
  runs two passes with two targets. Under the old shared keep-1 target it replaced
  nothing in 10 of 12 real cases (they contain the phrase exactly once). LESSONS §13.
- **A repair that must prevent a rework cannot live in `_stage_fix`.** A fossil
  `gate_rejects` entry routes `_classify_replan_failure` straight to STRUCTURAL, and
  `_stage_fix` runs after that decision. `pipeline._repair_fossil_rejects` therefore
  sits inside the review loop (after the review is archived, before the rework
  decision) and is **verify-then-drop**: the reject is removed only once every phrase
  it named is provably absent from the rotated text — which also makes it idempotent
  on the resume path, where a cached review meets an already-rotated chapter. It
  re-derives `failure_codes` (consulted *before* the gate list) and touches neither
  the archived `review_round{n}.json` nor `score`/`style_health`.
- **Repair is the second half; the first is not writing the fossil at all.** 12 of the
  library's 20 remaining first-draft `gate_rejects` are one entrenched bank phrase per
  book, already at **rank 0** of that chapter's mid-prompt avoid list — writer
  non-compliance, not a missing ban. `writing.fossil_tail_anchor`
  (`fossil_tail_anchor_enabled`) restates the hard-fossil ban at the prompt tail, the
  fourth such anchor for this weak instruction-following writer (ability capsule,
  recovery directive, scene-entry salience). Hard-only and capped at
  `FOSSIL_TAIL_ANCHOR_MAX` — a long tail dilutes the position it exploits. **Forward-only
  and unmeasurable on the archive**, like the `bd577ba` bootstrap-order fix.
- `_stage_fix` records `style_health_after_fix` rather than overwriting
  `style_health`: the latter is the measurement `score` was computed from, and
  overwriting it would leave score and penalty describing different texts.
- Two gates store their result under a key that is NOT the gate name
  (`length_band_check`→`length_band`, `book_wide_fossils`→`book_fossils`); read them
  via `fix.gate_result`. Do NOT use `REGISTRY.is_enabled` as a "did this gate run"
  test — `length_band_check`'s config key only controls its penalty (default false)
  while the gate always runs.
- Fossil rotation is **bank-only on purpose** (rotating book-specific proper nouns
  is canon corruption, not repair): LESSONS §4.

Gate calibration is measured, not guessed: `python tools/gate_census.py` and
`python tools/replay_l0.py`. **A silent gate is a bug report, not a deletion
candidate** — read LESSONS §4, including the `fire%` vs `advise%` distinction,
before deleting or re-thresholding one.

**`python tools/fpy_prime.py [novel…] [--from N --to N]` is the acceptance metric
to settle engine A/Bs with** (zero LLM, read-only). The FPY in `novel.py stats`
counts any rework artifact, and every one of those is produced by a rule keyed on
`quality_threshold` — so it cannot settle an experiment that *changes* that rule
(P4 moved the release line and FPY moved with it in both arms, for free). FPY′
replays `pipeline._hard_block_reasons` over archived `review_round0.json` payloads
with `score` excluded entirely, and counts only pre-write deterministic plan
retries (`plan_initial_attempt[1-9]`, `plan_critical`,
`plan_fossil_catastrophe`) — `plan_quality_replan`/`plan_hard_floor` are excluded
because they are downstream of the release rule. Thresholds are pinned at engine
defaults in `fpy_prime.PINNED` so two arms with divergent configs are still judged
by one ruler. Library-wide it reads **83.3% → 92.0%** after the latching-gate fixes
plus the unmeasured-plan-score fix (vs 12%–63% for "self-score ≥ 8.0"), and it names
the failing gate for every miss; the leading remaining killers are `hard_contract`
(14) and `gate_rejects` (9).

**A payload cannot fail a bucket it has no key for, and that reads as a PASS.**
`hard_block_reasons` fires off `review.get(key)`, so when two engines write
`review_round0.json` the absent keys are silently clean. `v2/accept.py` calls the
same `quality.hard_block_reasons` and writes v1's own key spellings on purpose —
including routing a CCR hard miss into `gate_rejects` and cited canon findings into
`contract_violations` — so exactly one bucket is unshared: v1's LLM factcheck writes
`contradictions`, and v2 has nothing that lands there. `fpy_prime` prints an
**ENGINE MIX** block naming it whenever the novels in scope disagree on
`payload["engine"]`. It reports rather than corrects, because which arm the gap
favours depends on the question.

**The aggregate excludes derivative novel dirs, and that changes answers.**
`fpy_prime.discover_novels` (shared with `replay_gates`) drops provable
derivatives — `__ablate_` names, dirs with `experiments/{fork,ablate}_<name>.json`,
and dirs whose Ch1 is byte-identical to another book's — prints each drop with its
reason, and takes `--all` to include them. Explicit novel names on the command line
are never filtered, so an A/B still reads its own arms. This is not cosmetic: with
`tangshuting_v1_backup` in the pool (200 chapters, Ch1 identical to `tangshuting`,
only 76/200 chapters shared) the same gate fixes measured **+4.0pt instead of
+6.2pt**, and `style_collapse` ranked #2 at 24 misses when 17 of those sat in that
one copy. Two independent runs never produce a byte-identical Ch1 even from the same
brief — `tangshuting_e2e` is the control and is deliberately KEPT — so identical Ch1
is proof of a copy, not a heuristic. LESSONS §13.

**`fpy_prime` cannot settle a change to a gate's LOGIC** — it replays archived
payloads, so a verdict already baked into them is frozen and the tool reports the
old answer forever. Use **`python tools/replay_gates.py [novel…] [--fix A,B,C]
[--detail]`** for that: it recomputes the changed gates from the primary data they
read (chapter texts, archived `plan_initial_attempt0_arbitration.json`, each
novel's own config) and re-runs `_hard_block_reasons` on the corrected payload.
Three traps it encodes, each of which gave a wrong answer first: `scene_dedupe_retry`
is the **generic** `duplicate_blocked` marker shared by three gates, not the
scene-dedupe gate's own event; the plan-gate chain is sequential with
`continue`, so a replan may only be dropped after the gates downstream of the
removed blocker get their first chance to speak; and **a fix that cannot be
isolated must say so** — `--fix C` adds B (both are settled by re-running the chain,
and the chain reads today's `quality.py`) and prints the implication, instead of
reporting B+C under C's name. LESSONS §13.

**Before adding any blocking gate, answer: what can THIS chapter do to turn it
green?** If nothing, the gate latches and every forced retry it buys is a
guaranteed first-pass failure. Three measured instances (fossil hard-rejects on a
frozen book-cumulative ratio, `chapter_mode_monotony` counting a genre label,
`CONTRACT_SYSTEM` fabricating an `ability_whitelist` for an ability-free brief) cost
4.0pt of library FPY′ between them — same defect class as the deleted
`fingerprint_warn_threshold`. The inverse question is just as load-bearing: **can the
signal your gate reads distinguish "bad" from "not measured"?** A sentinel that
doubles as a verdict (`plan_score`'s 0.0 for an empty `scores` list) makes every
parse failure look like the worst possible input. LESSONS §13.

**A retro-replay must normalize engine semantics that have since changed**, or it
reports fixed bugs as live problems. `fpy_prime._normalize` re-stamps two by default
(`--raw` replays payloads verbatim), and each one changes which bucket looks like the
top killer:
- `review.py`'s contract backstop stamped keyword-matched `problems` text as a HARD
  violation until `b54bfd0` downgraded it to SOFT. 22 archived chapters fail their
  first draft on that alone — the whole difference between "`hard_contract` is the #1
  killer at 54" (wrong) and "32, second to gate_rejects" (right), and between
  tunshi_xitong at 85% and its true 98%.
- `style_health`'s em-dash TREND term charged a flat +1.0, which stacked onto the
  static tier's +1.0 to hit `style_penalty_block` (2.0) *exactly*; it is now graduated
  by ratio. 31 archived chapters were scored under the flat rule and 5 block on it
  alone, so `style_collapse` reads as 7 live killers when only 2 are real. Two
  execution rules: only the em-dash terms are recomputed (every other component's
  input is the chapter text, which round0 no longer describes after revision), and a
  penalty already at `style_penalty_cap` is skipped because it is no longer a sum of
  its terms. `_restamp_style_penalty` calls `quality.em_dash_penalty` — the arithmetic
  was extracted into that pure function precisely so the tool cannot drift from the
  engine. LESSONS §13.

Two failure buckets that survive both normalizations are **not** defects to fix:
`hard_contradictions` (5 misses, one per book, each a concrete canon breach the
chapter could have avoided) is a healthy gate, and the remaining `hard_contract`
misses are the already-root-caused `ability_whitelist` fabrication (`bd577ba`).

**Offline tools must not log into the novel they measure.** `call_llm` appends
every call to `paths.logs_dir/llm_calls.jsonl`, which is exactly the file
`compare._llm_totals` reads for calls/chapter — so `tools/pairwise_ab.py` borrowing
an arm's config for its API keys charged that arm 10 calls for the cost of being
measured, inflating its own P4 report from 14.25 to 14.75 calls/chapter. The judge
now redirects to `experiments/pairwise_logs/` via
`dataclasses.replace(paths, logs_dir=…)`, and `compare.OFFLINE_TOOL_TAGS` filters
such rows out of already-written logs, printing the excluded count so a filtered
log is never silently indistinguishable from a clean one. `compare.py` also flags
its own score lines as **circular** when the flipped key is in
`RELEASE_RULE_KEYS`, and points at `fpy_prime`/`pairwise_ab` instead.

### Adaptive cost control (`planning.py`)
Inverted cost model: the DEFAULT is cheap (`candidate_plans: 1`,
`candidate_chapters: 1`) and breadth is spent only on trouble.
`_effective_candidate_count` RISK UPSHIFT (always on, from Ch3, no warmup) widens
to `risk_upshift_candidates` (3) when the last `risk_upshift_window` chapters show a
score below `planning._risk_score_floor` (`risk_upshift_score_floor`, lowered to
`rework_score_floor` under `rework_trigger: deterministic` — see the Rework trigger
section: an accepted chapter must not double as a distress signal) or a style penalty ≥
`risk_upshift_style_penalty`, or when a degradation-recovery directive is active —
collapse recovery is when plan diversity pays. STABLE DOWNSHIFT
(`adaptive_downshift_enabled`, only meaningful for multi-candidate bases) drops one
candidate once quality is stably ≥ `adaptive_downshift_score`. The structural
replan path independently forces multi-draft sampling
(`structural_replan_candidates`). Measured cost shape: LESSONS §8.

### Experiment harness (`compare.py`)
- `compare <a> <b>` — deterministic zero-LLM report (per-chapter scores/style
  penalties, force-accepts, quality-debt/gate-reject events, fossil warnings,
  scene-dedupe hits, LLM cost + planning share, non-secret config diff, heuristic
  verdict) → `experiments/`. Calibrated against known ground truth: it must judge
  v10 over v11.
- `fork <name> --as <new>` — **the A/B tool to reach for on anything mid-book.**
  Branches at HEAD (copies `memory/`, `chapters/`, `book.md`, `state.md`,
  `story_state.db`, RAG index) so both arms start byte-identical; forks at HEAD
  **only**. Metadata → `experiments/fork_<new>.json`.
- `ablate <name> --flip <key>` — chapter-capped copy with ONE key flipped, but it
  restarts at Ch1. Metadata → `experiments/ablate_*.json`.
- `tools/pairwise_ab.py` — blinded pairwise prose judge on the chapters that
  actually differ (the anti-self-dealing half of the P4 criterion).

**Short opening runs fabricate positive results.** The full protocol — one variable,
mid-book fork, matching reasoning coverage, `logs/` deliberately not copied, budget
via `target_words` not `max_chapters`, the `consecutive_force_accept_limit`
circuit-breaker trap, and the Windows launch/venv-stub gotchas — is LESSONS §5.
Every engine change should carry an ablation/fork report instead of a hand-compared
full rerun.

### Cross-book telemetry (`telemetry.py`)
Each novel has its own `story_state.db`; `telemetry.py` is the ONE shared sink
(`telemetry/global.db`, WAL, one fresh connection per write so N processes write
concurrently). Strict observer / safe no-op: any failure returns an empty value and
never stalls a chapter. Live double-writes from the pipeline
(`record_chapter_metrics`/`record_event`/`record_arbitration`/`record_revise_pair`)
plus idempotent `backfill_novel`. Currently **write-only** (logging +
`telemetry stats`) — the consumption layers were deleted; see LESSONS §9.

### Memory layers (`memory.py`)
Four context builders feed different LLM calls:
- `cacheable_prefix` — exact-bytes prefix shared across calls (creative brief +
  voice + bible + characters), keyed by sha1 of source files. Identical bytes ⇒
  provider prompt-cache hits. **Changing how this string is assembled invalidates
  the cache for every existing chapter.**
- `writing_memory_context` — small variable portion (state + threads + recent
  metrics + volume plan head) for the write/revise/review hot path
- `memory_context` — full layered context (4 tiers, char-budgeted) for plan
  generation and event extraction
- `lite_memory_context` — heavily abbreviated for plan-review/screening

`memory_context`'s four tiers **must not overlap**: `recent_metrics`/`recent_events`
return newest-first, so tier2/tier3 are prefixes of tier4, and emitting both in full
shipped every row twice in the engine's largest prompt. tier4 emits only the older
tail (`[5:]`/`[20:]`) under `## 更早的…JSON` headers and is omitted when that tail is
empty. `tests/test_memory_tiers.py` holds the invariant in both directions — no row
twice, no gap. Measurement: LESSONS §12.

`volume_plan.md` is the one memory file that grows linearly with the book, so plain
head truncation silently starves the mid-book (Ch41 case study: LESSONS §6).
`memory.volume_plan_window(text, chapter_num, cap, lookahead)` replaces head
truncation at all three read sites: it keeps blocks whose header declares no chapter
range or a range containing `chapter_num`, reduces out-of-range volumes to a header
breadcrumb, and inside kept blocks keeps only `| ChN |` rows for
`[chapter_num-1, chapter_num+lookahead]`. Rangeless sub-blocks inherit their nearest
ranged ancestor's decision; if no block covers the chapter, the nearest is kept
verbatim. Gated by `volume_plan_window_enabled`. Visibility alone doesn't produce
compliance, so `memory.chapter_schedule_directive` quotes this chapter's own schedule
rows as a hard obligation into BOTH `generate_candidate_plans` and `arbitrate_plan`
(the arbiter must push the row into `required_constraints`); gated by
`chapter_schedule_directive_enabled`.

Per-chapter state persistence is a SINGLE LLM call: `extract_events` returns the
extraction JSON **including** `protagonist_state` + `next_12_directions`;
`update_structured_state` (pure DB writes) and `update_state_file` (deterministic
markdown render) consume it with zero further LLM calls.

`compress_all_memory` consolidates per-chapter `## ChN` sections in
bible/characters/timeline/threads when files exceed `memory_max_kb` or every
`memory_compress_every` chapters; archives old sections under `logs/memory_archive/`.

## Things to be careful with

- **Don't add `cd <project>` before `git` commands** — the shell already runs in the project root.
- **`config.yaml` is not real YAML.** Anchors, lists, nested maps silently fail to parse; values become strings.
- **`NOVEL_CONFIG`/`NOVEL_PROMPT` must be set before importing `pipeline`/`config`/`memory`** — `config.py` reads them at import time and `memory.py` captures `PROMPT_FILE` at its own import.
- **Per-novel paths live entirely in each `config.yaml`'s `paths:` section**, joined onto `ROOT`. The engine has no hardcoded knowledge of `novels/`; isolation is purely a path convention, created by `config_template.yaml`'s `__NOVEL__` placeholder.
- **Background-task ordering is load-bearing.** The barriers in `generate_one_chapter` (`wait_label("chapter_finalize_ch{n-1}")` and the prefetch wait) keep memory/threads consistent; re-ordering them lets the next plan see stale state.
- **`chapter_completed.json` must be written synchronously** in `pipeline._stage_finalize`, never deferred to the background task (loop-leak invariant above).
- **`save_chapter` refuses to write chapters under 500 chars** (`writing.py:2739`), so provider refusals never persist as legitimate chapters.
- **`extract_contract` must run BEFORE `_bootstrap_chain`** in `memory.bootstrap`, and its markdown is passed into the bible + characters calls (`contract_md=`). Those two files are where abilities get declared; generated without the contract, they can invent an ability the brief explicitly bans, and it becomes canon the writer reads every chapter while the reviewer measures against the contract — a contradiction no chapter can resolve. Costs zero extra LLM calls (same call, moved earlier). LESSONS §13.
- **`cacheable_prefix` content changes invalidate the prompt cache** for every subsequent chapter — only modify it when the cache cost is worth it.
- **`cold_reader_review` must NOT use the cacheable_prefix.** Its whole value is being a judge that hasn't been steeped in the (possibly drifted) book context.
- **`style_health` is the objective anchor against score inflation.** Don't relax its thresholds to make chapters "pass"; the penalty exists to fight the model's over-rating of fragmented prose, not to be tuned away.
- **`voice_baseline.md` is frozen on purpose** — re-deriving voice from drifted prose is exactly the self-feeding loop that caused style collapse.
- **Reasoning knobs are not an A/B variable, and `quality_threshold` is not an FPY dial** (LESSONS §1, §2).
- **Live API keys sit in `config.yaml` / `config_template.yaml` / `novels/*/config.yaml`.** All gitignored — never echo them into tracked files or logs. New per-novel configs inherit the template's keys, so parallel novels share quota.
- **`config_template.yaml` is gitignored but must exist on disk** for `novel.py create`; don't delete it. When adding config keys, edit **both** it and the tracked credential-free `config_template.example.yaml` (the fresh-clone fallback). The `!config_template.example.yaml` negation in `.gitignore` is required — the broad `config_*.yaml` rule would swallow it. Incident record: LESSONS §10.
- **`GENRE_PROFILES` shared constants affect all genres.** Modifying `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT`, `_SELF_REVIEW_PREAMBLE`, or `_OUTPUT_SECTION` in `writing.py` changes every genre's writer prompt at once; per-genre overrides go in the `GENRE_PROFILES` entry. Same for `DIAGNOSE_CORE`/`DIAGNOSE_COMMON_FOOTER` in `refine.py` and `_EXECUTABILITY_DOCTRINE` in `planning.py`.
- **Ending awareness (`ending_aware`, default true) only fires when `max_chapters` is set.** In short-novel mode the final chapter (`chapter_num == max_chapters`) gets `CLOSING_RULES_BLOCK` (writing.py) + a planning ending directive, skips hook-only-revise, and refine demands closure instead of a cliffhanger. Detection is `config.py:is_final_chapter`. Pure char-target long novels have no deterministic finale, so this is inert there.
