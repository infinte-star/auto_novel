# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**This file holds the rules and the pipeline you must not break.** Four
companion docs carry the rest — follow the pointers instead of guessing:

| doc | contents |
| --- | --- |
| `docs/REDESIGN_V2.md` | **the live engine's design doc** (v2) + the A/B that settled it (§9.7). Read §0 before adding any quality gate or A/B judged by `score`: the self-score has no discrimination, so an experiment that changes rework rules cannot be settled with it. |
| `docs/LESSONS.md` | the measurements and post-mortems **behind** these rules. Read the cited section before changing a threshold, deleting a gate, or running an A/B. **Historical**: most of it was measured on the v1 engine, deleted 2026-07-28. The lessons hold — the file/function names in them often no longer exist. |
| `docs/INTERNALS.md` | mechanical reference: store schema, checkpoint layout, `llm.py` plumbing, shared prompt constants, refine / fossil-fix / screenplay tools, `tools/*` |
| `REDESIGN.md` | the v1 quality/FPY roadmap + P1–P4 execution record. **Historical**, same caveat — it is the record of what was measured on the engine v2 replaced. |

`README.md` is the user-facing quickstart.

## Overview

Universal multi-novel AI writing framework. The core engine (`v2/`) is an
automated long-form Chinese web novel generation pipeline that targets a
configurable character count (`novel.target_words`).

Architecture is **a deterministic decision table with four LLM actions**
(REDESIGN_V2): plan an arc once every ~10 chapters, write the chapter and its
state delta in ONE call, check canon, and repair. Nothing about *what happens
next* is decided by a model — every routing predicate is a pure function over
recorded state, so every branch is replayable offline. The v1 engine
(`pipeline.py`/`planning.py`/`review.py`/`taxonomy.py`, ~7.3k lines, a five-stage
plan committee plus a self-score-keyed rework loop) was deleted at `95361b9`
after a matched-position A/B; it is recoverable from git history and the
settlement is REDESIGN_V2 §9.7.

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
- `run` sets `NOVEL_CONFIG`/`NOVEL_PROMPT` **before** importing `v2.run`.
  Detached launch prefers the project venv
  (`E:\pycharmproject\allvenv\novel\Scripts\python.exe`); override with
  `NOVEL_PYTHON`.
- Each novel's `story_state.db`, `logs/`, `checkpoints/`, `memory/` are isolated in
  its own directory, so concurrent novels never share SQLite/file writes. Keys come
  from each novel's own config, so parallel runs share RPM/TPM quota unless given
  distinct keys.

**`novel.engine` selects the engine, and an unknown value is an ERROR, not a
guess.** `v2` (or absent) is the only accepted value; `v1` prints what happened
to it and exits 2. The two engines wrote different checkpoint labels and
different `review_round0.json` keys, so silently running a v1 config on v2 would
be a measurement forgery as much as a behaviour change — half the book would be
archived under one ruler and half under the other.

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
| `v2/run.py` | the decision table + the chapter loop; the only orchestrator |
| `v2/beat.py` | one arc call per `arc_span` chapters → a ChapterCard per chapter (two-layer rolling) |
| `v2/write.py` | ONE call returns prose **and** the `ChapterDelta` (no separate extraction call) |
| `v2/accept.py` | the decidable acceptance set, contract fulfilment (CCC), cite-or-drop |
| `v2/repair.py` | runs `fix.py`'s L0/L1 ladder, then re-judges it by the acceptance ruler |
| `v2/canon.py` | `StoryState` — the whole context projected once, ~15k chars, split stable/volatile |
| `v2/anchor.py` | the external anchor: blinded, position-debiased pairwise prose judge |
| `quality.py` | deterministic gate library (registry + metrics), zero LLM |
| `fix.py` | repair ladder L0/L1 implementations — fix instead of re-roll |
| `arc.py` | the ChapterCard *vocabulary* (`ARC_SYSTEM`, `normalize_card`, `validate_card`, `card_to_plan`) that `v2/beat.py` imports |
| `writing.py` | writer prompt doctrine (`GENRE_PROFILES`, `_build_write_system`), `save_chapter`, `update_structured_state`, `update_state_file` |
| `memory.py` | `bootstrap`, `cacheable_prefix`, `memory_context`, volume-plan windowing + transition steer |
| `retrieval.py` | dependency-free TF-IDF RAG + exemplar blocks |
| `store.py` | SQLite persistence (`story_state.db`), JSON fallback, event log readers |
| `checkpoint.py` | per-stage checkpoints + resume detection |
| `llm.py` | streaming, timeouts, salvage, JSON repair, refusal retry |
| `config.py` | YAML-subset config, paths, endpoints, `is_final_chapter` |
| `telemetry.py` | cross-book sink `telemetry/global.db` (write-only observer) |
| `refine.py`, `fossil_fix.py`, `screenplay.py`, `package.py` | post-completion / standalone tools |
| `compare.py` | `compare` / `ablate` / `fork` experiment harness |
| `trial.py`, `benchmark.py` | opening trials, local sample library |
| `tools/*.py` | zero/low-LLM analysis: `fpy_prime` (acceptance metric), `replay_gates` (settles gate-logic changes), `ccr_baseline`, `gate_census`, `orphan_gates` (does a gate still fire), `orphan_defs` (does anything still name it — defs / constants / config keys), `replay_l0`, `prompt_census`, `pairwise_ab`, `probe_reasoning`, `defossil`, `rebuild_memory`, `truncate_arm` |

## Architecture

### Top-level loop (`v2/run.py:main`)
1. `_ensure_bootstrap()` once — still `memory.bootstrap`, which generates `state.md` and `memory/{bible,characters,timeline,threads,volume_plan}.md` from `prompt.md`
2. Loop: `find_last_chapter()` → `run_chapter()` until `book_reached_target()`
3. On completion, optional `package.build_package()` then `refine.refine_book()`, gated on `novel.package_after_complete` / `novel.refine_after_complete` (both default **false**). Both are skipped when the run stopped on the quality breaker — half a book must not be finished as if it were whole.

There is **no background thread pool**. v1 had one because finalization,
stage reviews, memory compression and plan prefetch were all off the critical
path of a ~23-call chapter; v2's chapter is ~2.5 calls and its state write is
part of the write call's own response, so the pool bought nothing and cost the
ordering invariants it needed barriers to restore.

### One chapter (`v2/run.py:DECISIONS` + `run_chapter`)
`run_chapter` re-reads the table **from the top** after every action, so an
action that invalidates an earlier row is re-answered instead of skipped. Rows,
in order:

```
need_card → card_invalid → need_draft → need_report → l0_pending → l1_pending
          → canon_pending → next_card_patch → rescue → commit
```

Four properties of that ordering are load-bearing, each documented at the row:
- **`need_report` exists because `repair.pending` is a pure function OF a report** — "does L0 have anything to do" is unanswerable until one exists. Zero LLM.
- **The repair rows are NOT gated on `r.blocks`.** A repair layer answers the gates that *fired*, and most fixable findings never reach a hard block (`length_band_check`'s short side has no path into `hard_block_reasons` at all). Every fixer is keep-only-if-improved, so an unfired layer costs nothing.
- **Repair sits ABOVE `rescue`**, so the cheap deterministic fixes always get their turn before anything buys a rewrite. v1 had to hand-place `_repair_fossil_rejects` inside its review loop to get this for one gate; here it is the default for all of them.
- **`review_round0.json` is the raw first draft and is written the moment the first report exists** — never overwritten by a repaired or rescued draft. That file is what `tools/fpy_prime.py` replays. The one exception is the canon re-fold (`_act_canon`), which recomputes rather than overwrites: the canon check runs last, on the text that will ship, and its claims are re-cited *against the raw draft* so a finding whose evidence exists only in repaired prose is dropped rather than charged to the first draft.

`RESCUE_ATTEMPTS` bounds the rewrite path. It is safe to bound because
**nothing latches**: every member of the acceptance set is `scope="chapter"`, so
each blocking reason is something *this* chapter's text can turn green. If blocks
survive the rescue, they are recorded on the committed chapter rather than
retried forever, and the breaker (`quality_breaker_consecutive`, default 2)
counts chapters committed **with blocks outstanding** — the v2-native distress
signal, which needs no score to read. A run halted by the breaker skips the
post-completion passes.

`chapter_completed.json` must be written **synchronously** in the commit action.
Deferred, the resume check re-enters the chapter and resubmits on every
iteration (the v1 loop-leak incident).

### Planning (`v2/beat.py:generate_arc`)
ONE high-reasoning call every `arc_span` (10) chapters emits a **ChapterCard**
per chapter, so 错峰兑现 / 场地轮换 / 开场轮换 / 整段推进 are decided once with
whole-arc vision instead of re-argued per chapter. The card is the chapter's
contract — the same seven fields `v2.accept.contract_fulfilment` scores and
`tools/ccr_baseline.py` reports.

Three rules carry it (full notes in the module docstring and LESSONS §11):
- **No committee underneath.** `arc.plan_from_arc` returned None on trouble and let v1 fall back to the five-stage planner; v2 has no such floor, so every failure path must end in a real card or an exception — **never in a fabricated one**. A card nobody planned would still score a CCR, and that number would measure nothing.
- **Two-layer rolling.** Each arc call also emits a one-line skeleton for the NEXT arc, fed back in when that arc is planned. Same response, no extra call. A skeleton is a *promise*, not a constraint: the next arc may revise it but must say so in `arc_intent`.
- **`validate_card` repairs on CRITICAL only** — advisories become `required_constraints`. Card *vocabulary* is imported from `arc.py`, not copied: there is exactly one definition of what a card is.

Context comes from `v2/canon.py`, not from `memory.py`'s four builders. The
stable half of the StoryState is passed as `cacheable_prefix`, so the arc call
shares a prompt-cache prefix with every write call of the chapters it plans.
`memory.volume_transition_directive`'s HARD level is injected for **every**
chapter in the span, because `arc_span` does not align with `volume_plan.md`'s
volume boundaries — checking only `start_ch` misses the boundary.

### Writing (`v2/write.py:write_chapter`)
ONE call returns prose, a sentinel line, then a bare JSON object carrying exactly
`canon.ChapterDelta`'s five fields. v1 spent two calls here and the second
re-read the chapter it had just written to answer questions the writer already
knew. Four decisions are load-bearing:
- **The prose doctrine is v1's, unchanged.** `GENRE_PROFILES`, the shared constants, `ANTI_FRAGMENT_BAN`, the aesthetic presets — all still `writing._build_write_system`. Rewriting style teaching would have made the A/B a test of two prompt libraries instead of two architectures.
- **v1's output section is REPLACED, not appended to.** It ends with 「只输出章节正文…严禁输出…JSON」, the exact opposite of what v2 asks. An appended override leaves two contradictory instructions for a writer already documented as a weak instruction-follower, and it fails *quietly*: the model obeys the older, more emphatic rule, no delta parses, and the engine spends an extraction call per chapter forever — corrupting the very cost number the A/B measures. The swap is an exact-string replacement that **raises** when the string is absent, pinned by `tests/test_v2_write.py`.
- **Prose first, JSON last; a parse failure loses the delta, never the prose.** The exit hook is the most important sentence in a web-novel chapter and must be the last prose written, not something hurried past en route to a JSON object. And a 5000-char chapter inside a JSON string is a 5000-char escaping problem.
- **The acceptance checklist names the literal fragments the gate greps for**, generated from the same `quality._beat_anchor_fragments` call `accept.contract_fulfilment` uses. Telling the writer 「把转折写出来」 and grading on whether 「钥匙」 appears grades a different question than the one asked.

Genres live in `writing.GENRE_PROFILES`: `history`, `xuanhuan_shuang`,
`system_stream`, `urban_ability`, `romance_female`, `wanzu_xuanhuan`, `suspense`
(`rule_horror` → `suspense`).

### Acceptance (`v2/accept.py`)
Three rules govern the module, and all three exist to stop v2 grading itself:
1. **The acceptance set is defined by the ruler, not by taste.** A gate belongs exactly when `quality.hard_block_reasons` — the release rule `tools/fpy_prime.py` replays to settle every engine A/B — can read its output and declare the draft a write-off. `ACCEPTANCE_GATES` is checked against that criterion by `tests/test_v2_accept.py`, and every excluded blocking gate must give its reason in `NOT_IN_ACCEPTANCE`.
2. **Every member is zero-LLM decidable AND actionable.** `quality.REGISTRY.may_block()` enforces the second half: a book-cumulative quantity may advise but never reject, because no rewrite of the current chapter can lower it.
3. **The output is a v1-schema review payload** — `gate_rejects`, `style_health`, `length_band`, `opening_hook_gate`, `adjacent_repetition`, `contract_violations` — so both engines are settled by the same `fpy_prime` invocation with no tool changes. v2-only findings ride along under `contract_fulfilment` / `citations`, which the ruler ignores; they become `gate_rejects` entries explicitly, where the ruler can see them.

The two v2-native checks:
- **`contract_fulfilment(card, text)` — CCC.** Did the prose stage what the card promised? Zero LLM: the card's fields are concrete by construction (`arc.ARC_SYSTEM` rule 2 forbids abstract intent), so their anchors either appear on the page or they do not. This is the measured successor to `quality.beat_coverage`, widened from `beats` to the whole card.
- **`citation_check(claims, text)` — cite-or-drop.** A review finding that cannot point at a substring of the chapter it judges is discarded, not weighed. It is the only defence against a reviewer inventing a violation, and unlike a confidence score it is decidable.

### StoryState (`v2/canon.py`)
ONE projection, ~15k chars, split **by mutability rather than by topic**:
- `stable` — brief / facts / voice / route. Byte-identical for as long as the source files are, so it is the provider prompt-cache prefix. ~5.6k.
- `volatile` — card / focus / threads / recent / ledger / rag / opening. Changes every chapter. ~11.5k.

`render()` always emits stable first, and that ordering *is* the cache strategy:
a prefix cache hits on a shared prefix, so one volatile byte in the head would
cost the hit on every call for the rest of the book. `stable_key` is a sha1 of
the stable sources and `run.py` logs hit/miss off it.

Three rules the projections obey, each a measured lesson:
1. **A clipped section says so, in the text.** Head truncation that looks like the whole thing is what starved the mid-book for 40 chapters before anyone noticed (LESSONS §6). `_clip` appends `〔…截断 N 字〕`; `_clip_items` drops whole items from the END and says how many.
2. **Empty and clipped are different facts.** An empty section emits no header. Printing `## 伏线` with nothing under it tells the writer there are no open threads, which is a lie whenever the truth is "the budget ate them".
3. **Persistence has one writer.** `apply_delta` delegates to `writing.update_structured_state`. A second writer against the same schema is a second ruler: it drifts, and the drift is invisible until the two disagree about what canon says.

### Quality control (`quality.py`, `retrieval.py`)
These layers exist because LLM self-assessment cannot catch its own degeneration —
prose drifts telegraphic and the model rates it 9+. Incident record: LESSONS §3.

`GateRegistry` is a metadata registry, **not a dispatcher** — nothing calls
`REGISTRY.get(name)(...)`. A gate runs only where some module calls it by name,
which is why the inventory below is the ground truth and not the `@register`
decorators.

| gate / layer | config | behaviour |
| --- | --- | --- |
| `quality.style_health(text, config)` | `style_health_enabled`, `style_em_dash_per_kchar_warn`/`_bad`, `style_min_avg_sentence_chars`, `style_fragment_line_ratio_max`, `style_penalty_cap`, `style_penalty_block` | deterministic prose metrics (em-dash density, avg sentence length, fragment-line ratio, dialogue presence) → `penalty`, `flags`, writer `directives`. Blocks at `style_penalty_block` |
| `quality.scene_similarity(plan, recent_plans)` | `scene_dedupe_enabled`, `scene_dedupe_sim_warn`/`_block`/`_identical`, `scene_dedupe_short_novel_block`, `scene_dedupe_candidate_block` | Jaccard similarity of a plan's scene skeleton (conflict/payoff/pressure/goal/beats) vs recent cards projected through `card_to_plan` |
| `quality.cross_chapter_repetition` | `style_cross_repeat_reject_count` (8) | signature clauses reused verbatim across chapters → a `level` of `advise` or `reject` |
| `quality.dialogue_health` | `dialogue_health_enabled`, `dialogue_char_ratio_min` (0.10), `dialogue_char_ratio_target` (0.20) | dialogue-ratio gate over text inside `"…"`; reached via `fix.py`'s L1 dialogue injection |
| `quality.book_wide_fossils` | `book_fossil_enabled`, `book_fossil_chapter_frac` (0.30), `book_fossil_min_chapters` (6), `book_fossil_hard_ratio`, `book_fossil_struct_count` | 6-char CJK n-grams recurring across a large fraction of the WHOLE book (what `cross_chapter_repetition`'s 6-chapter window structurally misses). **A hard fossil requires `current_chapter` and only indicts a chapter that actually contains the phrase** — the ratio is book-cumulative, so without that the gate latches on and rejects compliant chapters forever. `book_fossil_hard_ratio` is floored at the candidacy fraction (below it, every candidate is automatically hard). LESSONS §13 |
| `quality.descriptor_frequency` | `descriptor_frequency_enabled` | over-used descriptors across the book → `gate_rejects`; answered by keep-1 rotation |
| `quality.adjacent_repetition` | `adjacent_repeat_*` | this chapter vs the previous one → `adjacent_repeat_block` |
| `quality.length_band_check` | `chapter_min_chars` (2800), `chapter_max_chars` | band check; the short side is answered by L1 expand-to-band, the long side blocks. **Its config key only controls its penalty (default false) — the gate always runs**, so `REGISTRY.is_enabled` is not a "did this gate run" test |
| `quality.opening_hook_gate` | `opening_hook_gate_enabled` | first-line hook strength → block; L0 demotes a scenery opening |
| `quality.genre_adherence` | `genre_drift_reject_enabled` | genre-signal drift vs recent chapters → `gate_rejects` |
| `quality.plan_executability_gate`, `plan_visual_payoff_check`, `narrative_pattern_repetition` | card-scope | plan/card gates, fixable before a word is written. Currently reached only by `tools/replay_gates.py` — see the wiring gap below |
| `retrieval.py` RAG | `rag_enabled`, `rag_top_k`, `rag_exclude_recent` | dependency-free TF-IDF char-bigram index (no embeddings). `index_chapter` is called idempotently from `save_chapter` → `logs/retrieval_index.json`; `retrieval_block` builds the "## 相关历史原文（检索…）" section; `backfill_index` indexes a finished book |
| `retrieval.exemplar_block` | `exemplar_rag_enabled` (false), `exemplar_rag_top_k` | quotes the book's own strongest chapters back as style anchors. **Rank-based, not an absolute score threshold**; picks one dialogue-dense + one action-dense exemplar (LESSONS §7) |
| `quality.fingerprint_avoidance_context` | `fingerprint_enabled` | the 全书结构指纹 block. **Emits an aggregate (recurring move bigrams/trigrams + payoff/conflict/move frequencies), never one line per chapter** — the per-chapter form was 19.6% of the largest prompt and grew linearly while carrying no signal (194 distinct flows in 200 chapters). `store_chapter_fingerprint` keeps writing `skeleton_tokens` even though nothing reads it: that column is what lets a future recalibration be replayed offline. LESSONS §8 |

Two routing rules inside that table are load-bearing:
- **Scene dedupe escalated in three steps under v1; v2 kept only one.** `arc.validate_card` takes a single `scene_sim_block` (`v2/beat.py:429`, default 0.85) and files a CRITICAL above it. The WARN tier that appended `required_constraints`, the chapter-capped relaxation, and the `scene_dedupe_sim_identical` (0.97) absolute ceiling all lived in the deleted `review.py`/`planning.py` — their config keys (`scene_dedupe_sim_warn`, `scene_dedupe_candidate_block`) now have no reader. Found by `tools/orphan_defs.py --config`; it is a wiring gap to settle per tier, not a redesign that dropped them deliberately.
- **A `cross_chapter_repetition` reject is structural, not cosmetic** — it lands in `gate_rejects`, and the repair rows answer it before `rescue` can buy a rewrite.

#### Wiring gap left by the v1 deletion — settled per gate, by measurement

The deletion of `review.py` orphaned 12 registered gates: `GateRegistry` never
dispatches, so a gate with no caller is simply silent. All 12 are now settled.
`quality.py` registers **23**, of which the chapter loop reaches **19**.

**Nine were WIRED as advisories in `v2/accept.py`** — `ai_flavor_health`,
`paragraph_shape_health`, `prose_texture`, `shareable_line`,
`intra_chapter_repetition`, `hook_tail_repetition`, `payoff_beat_density`,
`information_density`, `long_span_fatigue`. Three of them (`prose_texture`,
`information_density`, `long_span_fatigue`) were **recalibrated first**: each had
a term whose threshold sat outside the metric's measured distribution, so it was
firing on ~44%/~60% of the corpus and carrying no signal. The wiring is safe to
ship without an A/B for one structural reason — **advisory means the result
contributes `directives` and NOTHING else.** No wired gate appends to
`gate_rejects`, so `hard_block_reasons` cannot move and archived FPY′ readings
stay comparable (verified: 365/438 = 83% before and after, byte-identical).

**Two were DELETED**, both for the "can the signal distinguish bad from
not-measured" defect: `emotional_cadence` compared consecutive `emotional_tone`
values for equality, but that column holds free text (median 65 chars, 382/565
distinct) so it could not fire on **any** book, v1 or v2; `flat_chapter_streak`
produced 3 findings in 638 chapters, every one of them already reported by
`payoff_beat_density`, and its one distinguishing input (`emotional_impact`) is
structurally NULL on v2 — v2 has no self-score, so `tension`, `emotional_tone`,
`emotional_impact` and `score` are all 0/30 on v2-written chapters while
`payoff_type`/`conflict_type` are 30/30.

`beat_coverage` remains deliberately unwired — superseded by
`accept.contract_fulfilment`, which widened it from `beats` to the whole card.
**The one real gap left** is the three card gates (`plan_executability_gate`,
`plan_visual_payoff_check`, `narrative_pattern_repetition`): still reached only by
`tools/replay_gates.py`, and blocking there would be cheap because nothing has
been written yet.

**A silent gate stays a bug report, not a deletion candidate** (LESSONS §4, and
the `fire%` vs `advise%` distinction there). Two rulers, and the difference
matters: `python tools/gate_census.py` replays archived verdicts, so it reports an
advisory as permanently silent — nothing archives one. `python
tools/orphan_gates.py` recomputes the nine from primary data (chapter texts,
archived cards, metrics rows) and is what every wired gate's `proof=` string
quotes. Run it after touching any of their thresholds. It also found the census's
own blind spot: `_fired` recognized no key named `repeat`, which made
`hook_tail_repetition`'s only verdict invisible and nearly argued a live gate out
of the tree.

The rest of that deletion's debris is swept (2026-07-28): **49 module-level defs
and constants / ~2.4k lines, plus 133 config keys** whose only consumer was v1
machinery v2 replaced. `python tools/orphan_defs.py [--constants|--config]` is the
census — the companion to `orphan_gates.py`, and the tool to re-run after any
deletion, because a cut cascades (the first sweep exposed 14 more defs that only
the deleted ones had named, and two rounds after that before it converged). Its
docstring carries five predicates that each gave a wrong answer first; the one
worth knowing without reading it is that **a config key read through an f-string
has no literal to grep** — `config.py:512` reads `api.get(f"{role}_base_url")`, so
all 29 role-routing keys look dead and are not.

Four dead keys were **held back on purpose**, because they are dead from a wiring
break rather than from replacement, and deleting the knob would bury the bug
report:
- `style_em_dash_trend_window` — nothing supplies `style_health`'s `em_history`
  (`v2/accept.py:608` passes text and config only), so `recent_mean is None` on
  every call and **the em-dash TREND term never fires in v2**. That is the check
  that caught gudai50_v2 Ch20-24 climbing 6.6→8.8 while the static tier flat-lined
  at +1.0 — and it is the same term `fpy_prime._normalize` re-stamps, so the ruler
  currently models a gate the engine no longer runs. Same for `tech_history`.
- `scene_dedupe_sim_warn`, `scene_dedupe_candidate_block` — see the escalation note
  above: v2 kept only the BLOCK step.
- `telemetry_enabled` — `telemetry.py` reads it nowhere, so telemetry cannot be
  turned off. Harmless (strict observer) but the config file makes a promise it
  does not keep.

`compare.py`'s `RELEASE_RULE_KEYS` still names several deleted keys, and that is
deliberate: it is a *guard* list, not a reader. Every already-written
`novels/*/config.yaml` still carries those keys, so `--flip rework_trigger` on an
old book remains possible and must still be flagged as circular.

**`quality_threshold` is a special case worth knowing about.** It is no longer a
release rule anywhere — but `writing._prewrite_quality_contract` still reads it
to tell the writer what score to aim for. Deleting it silently drops a number out
of the writer prompt; leaving it invites the belief that it still gates anything.
It does not.

**One capability gap left; the other is closed:**
- **`memory/opening_route.md` reaches the v2 writer as of 2026-07-28** — but *projected*, not pasted. The file is a mixture of six blocks (see `trial.py`'s `best_md`) with three different lifetimes, so `canon.build` routes them: 核心卖点/差异化/读者承诺 are book-level positioning → the **stable** `route` section (in `stable_key`, so adopting a route mid-book moves the key and `run.py` prints the miss — that is what makes `adopt-trial` honest rather than merely effective); 正式连载前修改指令 is opening-scoped → the **volatile** `opening` section, empty past `OPENING_ROUTE_SPAN` (3) because at Ch150 it is an instruction about a chapter that shipped 147 chapters ago; 推荐书名/推荐简介 are `package.py`'s job and trial_score/variant_path are bookkeeping → **dropped**. v1 pasted all six at a 5000-char cap into every chapter prompt, so a 200-chapter book shipped ten candidate titles 200 times. **The opening directives must never move into the stable head** — that would either freeze a stale instruction into the cached prefix for the rest of the book, or make the head chapter-dependent while `stable_key` (a hash of *files*) still claimed it hadn't moved, which is the one failure mode that key exists to prevent. `tests/test_v2_canon.py:OpeningRouteTest` pins it.
- **`voice_baseline.md` is no longer produced.** Its writer was `review.refresh_voice_anchors`. v2 has no voice-refresh path at all, so `memory/voice.md` is written once by `bootstrap` and never re-derived — which is the frozen-baseline outcome the rule wanted, reached by having no refresh rather than by skipping one. Nothing is broken; the *file* is simply gone, and `tools/rebuild_memory.py` still looks for it.

### Repair ladder (`v2/repair.py` → `fix.py`)
Layer membership is declared ON the gate
(`@REGISTRY.register(..., repair="L0"|"L1"|"L2"|"advisory")` in `quality.py`);
`fix.ACTION_BY_GATE` only maps a gate to an action, so declaring a layer is never
a promise that a fixer exists.
- **L0** (zero LLM): em-dash reduction, fragment-line merging, bank-only fossil rotation, scenery-opening demotion.
- **L1** (≤ `fix_max_l1_calls` bounded calls): targeted expand-to-band, dialogue injection, em-dash rewording — each extracts a handful of passages, rewrites them in one numbered-list call, splices them back. Never a whole-chapter rewrite.
- Every fixer is **keep-only-if-the-metric-improved**, which is what makes the repair rows safe to run unconditionally. **A metric check cannot catch broken Chinese**, so any grammar guard must be structural instead: `fossil_fix._safe_alt` refuses a variant that would follow an attributive 「的/之」 without sharing the phrase's head noun (「顾峥的声音压得很低」 → 「顾峥的压着嗓子」), and keeping the fossil beats emitting that.
- **A repair is kept only if ACCEPTANCE says so.** Each fixer guards itself against its own metric, but `style_health` is not the currency v2 releases on — `accept.block_reasons` is. A rotation that dodges a fossil and lands on an adjacent-repeat, or an expansion that clears the length floor and blows the ceiling, passes every inner guard and fails the only one that matters. So each layer is re-scored with the same `recheck` the release rule uses and **reverted whole** if it introduced a blocking reason that was not there before. Whole-layer revert is a real cost (one bad rotation discards two good em-dash fixes); `fix.apply_l0` returns one string, so per-action granularity would mean reimplementing the ladder. The log line names it when it happens.
- **The repair target is per-gate, and getting it wrong makes the fixer a no-op.** Density gates (`cross_chapter_repetition`, `descriptor_frequency`) are answered by keep-1 rotation; a `book_wide_fossils` hard reject is a book-cumulative ratio and can only be cleared by ZERO occurrences in this chapter, so `fix.rotate_fossils` runs two passes with two targets. Under the old shared keep-1 target it replaced nothing in 10 of 12 real cases (they contain the phrase exactly once). LESSONS §13.
- **Repair is the second half; the first is not writing the fossil at all.** 12 of the archive's 20 remaining first-draft `gate_rejects` are one entrenched bank phrase per book, already at **rank 0** of that chapter's mid-prompt avoid list — writer non-compliance, not a missing ban. `writing.fossil_tail_anchor` (`fossil_tail_anchor_enabled`) restates the hard-fossil ban at the prompt tail, the fourth such anchor for this weak instruction-following writer (ability capsule, recovery directive, scene-entry salience). Hard-only and capped at `FOSSIL_TAIL_ANCHOR_MAX` — a long tail dilutes the position it exploits.
- Two gates store their result under a key that is NOT the gate name (`length_band_check`→`length_band`, `book_wide_fossils`→`book_fossils`); read them via `fix.gate_result`.
- Fossil rotation is **bank-only on purpose** (rotating book-specific proper nouns is canon corruption, not repair): LESSONS §4.

Gate calibration is measured, not guessed: `python tools/gate_census.py` and
`python tools/replay_l0.py`. **A silent gate is a bug report, not a deletion
candidate** — read LESSONS §4 before deleting or re-thresholding one.

## Measurement discipline

**`python tools/fpy_prime.py [novel…] [--from N --to N]` is the acceptance metric
to settle engine A/Bs with** (zero LLM, read-only). The FPY in `novel.py stats`
counts any rework artifact, and each of those is produced by a rule keyed on a
self-score — so it cannot settle an experiment that *changes* that rule. FPY′
replays `quality.hard_block_reasons` over archived `review_round0.json` payloads
with `score` excluded entirely, and counts only pre-write deterministic plan
retries (`plan_initial_attempt[1-9]`, `plan_critical`,
`plan_fossil_catastrophe`) — `plan_quality_replan`/`plan_hard_floor` are excluded
because they are downstream of the release rule. Thresholds are pinned at engine
defaults in `fpy_prime.PINNED` so two arms with divergent configs are judged by
one ruler. It names the failing gate for every miss.

Current readings (2026-07-28): the archive as frozen reads **83.3%**; replayed
through today's gate logic (`tools/replay_gates.py`) it reads **92.0%** — the gap
*is* the latching-gate and unmeasured-plan-score fixes, which cannot be seen in
frozen payloads. On the 30 matched chapters of the settlement A/B
(`tangshuting` Ch171-200 vs `ts_v2match`), the settled reading is **86.7% → 96.7%**
via `replay_gates`. Raw `fpy_prime` reports 73% for the v1 arm, and that 13pt is
**not** a v2 win: all six of those misses are `book_wide_fossils` hard rejects the
gate cannot produce on a first draft today. Quote the normalized pair, and read
REDESIGN_V2 §9.7 before citing either — the 10pt that survives is entirely
pre-write plan/card retries (3 : 1), not "v2's prose passes gates more easily".

**A payload cannot fail a bucket it has no key for, and that reads as a PASS.**
`hard_block_reasons` fires off `review.get(key)`, so absent keys are silently
clean. `v2/accept.py` writes v1's own key spellings on purpose — including
routing a CCR hard miss into `gate_rejects` and cited canon findings into
`contract_violations` — so exactly one bucket was unshared: v1's LLM factcheck
wrote `contradictions`, and v2 has nothing that lands there. `fpy_prime` prints
an **ENGINE MIX** block naming it whenever the novels in scope disagree on
`payload["engine"]`. It reports rather than corrects, because which arm the gap
favours depends on the question. This still matters after v1's deletion: the
archive is mostly v1-written books.

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
read (chapter texts, archived arbitration/card JSON, each novel's own config) and
re-runs `hard_block_reasons` on the corrected payload. Three traps it encodes,
each of which gave a wrong answer first: `scene_dedupe_retry` is the **generic**
`duplicate_blocked` marker shared by three gates, not the scene-dedupe gate's own
event; the plan-gate chain is sequential with `continue`, so a replan may only be
dropped after the gates downstream of the removed blocker get their first chance
to speak; and **a fix that cannot be isolated must say so** — `--fix C` adds B and
prints the implication, instead of reporting B+C under C's name. LESSONS §13.

**Before adding any blocking gate, answer: what can THIS chapter do to turn it
green?** If nothing, the gate latches and every forced retry it buys is a
guaranteed first-pass failure. Three measured instances (fossil hard-rejects on a
frozen book-cumulative ratio, `chapter_mode_monotony` counting a genre label,
`CONTRACT_SYSTEM` fabricating an `ability_whitelist` for an ability-free brief)
cost 4.0pt of library FPY′ between them — same defect class as the deleted
`fingerprint_warn_threshold`. `GATE_SCOPES` now makes this structural:
`scope="book"` may advise but never reject. The inverse question is just as
load-bearing: **can the signal your gate reads distinguish "bad" from "not
measured"?** A sentinel that doubles as a verdict (a `plan_score` of 0.0 for an
empty `scores` list) makes every parse failure look like the worst possible
input. LESSONS §13.

**A retro-replay must normalize engine semantics that have since changed**, or it
reports fixed bugs as live problems. `fpy_prime._normalize` re-stamps two by
default (`--raw` replays payloads verbatim), and each changes which bucket looks
like the top killer:
- v1's contract backstop stamped keyword-matched `problems` text as a HARD violation until `b54bfd0` downgraded it to SOFT. 22 archived chapters fail their first draft on that alone — the whole difference between "`hard_contract` is the #1 killer at 54" (wrong) and "32, second to gate_rejects" (right), and between tunshi_xitong at 85% and its true 98%.
- `style_health`'s em-dash TREND term charged a flat +1.0, which stacked onto the static tier's +1.0 to hit `style_penalty_block` (2.0) *exactly*; it is now graduated by ratio. 31 archived chapters were scored under the flat rule and 5 block on it alone. Two execution rules: only the em-dash terms are recomputed (every other component's input is the chapter text, which round0 no longer describes after revision), and a penalty already at `style_penalty_cap` is skipped because it is no longer a sum of its terms. `_restamp_style_penalty` calls `quality.em_dash_penalty` — the arithmetic was extracted into that pure function precisely so the tool cannot drift from the engine. LESSONS §13. **Caveat as of 2026-07-28:** the trend term does not run in v2 at all (nothing supplies `em_history`), so this normalization currently models a gate the live engine has stopped enforcing — see the held-back keys under the wiring gap.

Two failure buckets that survive both normalizations are **not** defects to fix:
`hard_contradictions` (5 misses, one per book, each a concrete canon breach the
chapter could have avoided) is a healthy gate, and the remaining `hard_contract`
misses are the already-root-caused `ability_whitelist` fabrication (`bd577ba`).

**Offline tools must not log into the novel they measure.** `call_llm` appends
every call to `paths.logs_dir/llm_calls.jsonl`, which is exactly the file
`compare._llm_totals` reads for calls/chapter — so `tools/pairwise_ab.py`
borrowing an arm's config for its API keys charged that arm 10 calls for the cost
of being measured, inflating its own P4 report from 14.25 to 14.75 calls/chapter.
The judge redirects to `experiments/pairwise_logs/` via
`dataclasses.replace(paths, logs_dir=…)`, and `compare.OFFLINE_TOOL_TAGS` filters
such rows out of already-written logs, printing the excluded count so a filtered
log is never silently indistinguishable from a clean one. `compare.py` also flags
its own score lines as **circular** when the flipped key is in
`RELEASE_RULE_KEYS`, and points at `fpy_prime`/`pairwise_ab` instead.

### The external anchor (`v2/anchor.py`)
The third metric is WR — a chapter's win rate against a reference chapter, judged
by a model told nothing about where either came from. It is the only metric that
reads prose and the only judgement the engine cannot award itself: FPY′ and CCR
both measure whether the engine did what it said it would, which a sufficiently
timid engine maxes out by promising nothing. Four properties, each here because
dropping it produced a number that looked fine and meant nothing:
- **Blind.** Sides are labelled 甲/乙; arm names, engine versions and paths never reach the model.
- **Two-way.** Every pair is judged twice with sides swapped; a win counts only when both orders agree. `tally` reports the flip rate AND its direction — one-sided flips are measured position preference (no resolving power on those pairs), split flips are genuinely close prose. Quoting `n` instead of `n_decisive` overstates the evidence; the v1/v2 settlement had n=30 but n_decisive=5.
- **No cacheable_prefix**, enforced structurally: the module never imports `memory`, so there is no prefix to add.
- **Log isolation** via `judge_paths` (see above).

`anchor_chapters()` reads human-written references from `benchmarks/anchor/`. That
directory does not exist yet, and the only things under `benchmarks/` are pattern
NOTES about 爆款 structure — not prose, and they must never be fed to a prose
judge. So `wr_against_anchor` returns `{"available": False}` with a reason rather
than quietly substituting an arm-vs-arm comparison, which would report an
internal A/B under the name of an external one.

### Experiment harness (`compare.py`)
- `compare <a> <b>` — deterministic zero-LLM report (per-chapter scores/style penalties, force-accepts, gate-reject events, fossil warnings, scene-dedupe hits, LLM cost + planning share, non-secret config diff, heuristic verdict) → `experiments/`.
- `fork <name> --as <new>` — **the A/B tool to reach for on anything mid-book.** Branches at HEAD (copies `memory/`, `chapters/`, `book.md`, `state.md`, `story_state.db`, RAG index) so both arms start byte-identical; forks at HEAD **only**. Metadata → `experiments/fork_<new>.json`.
- `ablate <name> --flip <key>` — chapter-capped copy with ONE key flipped, but it restarts at Ch1. Metadata → `experiments/ablate_*.json`.
- `tools/pairwise_ab.py` — the CLI over `v2/anchor.py` for a two-arm engine A/B. Supports `--b-from` (offset pairing when the arms sit at different outline positions) and `--probe N` (judge chapters against themselves to calibrate position bias).

**Short opening runs fabricate positive results.** The full protocol — one
variable, mid-book fork, matching reasoning coverage, `logs/` deliberately not
copied, budget via `target_words` not `max_chapters`, and the Windows
launch/venv-stub gotchas — is LESSONS §5. Every engine change should carry an
ablation/fork report instead of a hand-compared full rerun.

### Cross-book telemetry (`telemetry.py`)
Each novel has its own `story_state.db`; `telemetry.py` is the ONE shared sink
(`telemetry/global.db`, WAL, one fresh connection per write so N processes write
concurrently). Strict observer / safe no-op: any failure returns an empty value and
never stalls a chapter. Currently **write-only** (logging + `telemetry stats`) —
the consumption layers were deleted; see LESSONS §9.

### Memory (`memory.py`)
v2 gets its per-chapter context from `v2/canon.py`, so `memory.py` survives for
four jobs:
- **`bootstrap`** — the one-time generation of `state.md` + `memory/*.md` from `prompt.md`. **`extract_contract` must run BEFORE `_bootstrap_chain`**, and its markdown is passed into the bible + characters calls (`contract_md=`): those two files are where abilities get declared, and generated without the contract they can invent an ability the brief explicitly bans, which then becomes canon the writer reads every chapter while acceptance measures against the contract — a contradiction no chapter can resolve. Costs zero extra LLM calls (same call, moved earlier). LESSONS §13.
- **`cacheable_prefix` / `memory_context`** — still read by `arc.py`, `trial.py`, `package.py`. **Changing how `cacheable_prefix` is assembled invalidates the prompt cache for every existing chapter.** `memory_context`'s four tiers **must not overlap**: `recent_metrics`/`recent_events` return newest-first, so tier2/tier3 are prefixes of tier4, and emitting both in full shipped every row twice in the largest prompt. tier4 emits only the older tail (`[5:]`/`[20:]`) under `## 更早的…JSON` headers, omitted when that tail is empty. `tests/test_memory_tiers.py` holds the invariant in both directions. LESSONS §12.
- **`volume_plan_window(text, chapter_num, cap, lookahead)`** — `volume_plan.md` is the one memory file that grows linearly with the book, so plain head truncation silently starves the mid-book (Ch41 case study: LESSONS §6). It keeps blocks whose header declares no chapter range or a range containing `chapter_num`, reduces out-of-range volumes to a header breadcrumb, and inside kept blocks keeps only `| ChN |` rows for `[chapter_num-1, chapter_num+lookahead]`. Rangeless sub-blocks inherit their nearest ranged ancestor's decision; if no block covers the chapter, the nearest is kept verbatim. Gated by `volume_plan_window_enabled`.
- **`volume_transition_directive`** — visibility alone doesn't produce compliance, so a volume boundary inside the arc span is restated as a hard obligation in the arc prompt (`v2/beat.py`). HARD level only.

Per-chapter state persistence costs **zero** extra LLM calls: the write call
returns the `ChapterDelta` with the prose, and `canon.apply_delta` →
`writing.update_structured_state` (pure DB writes) plus
`writing.update_state_file` (deterministic markdown render) consume it.

## Things to be careful with

- **Don't add `cd <project>` before `git` commands** — the shell already runs in the project root.
- **`config.yaml` is not real YAML.** Anchors, lists, nested maps silently fail to parse; values become strings.
- **`NOVEL_CONFIG`/`NOVEL_PROMPT` must be set before importing `v2.run`/`config`/`memory`** — `config.py` reads them at import time and `memory.py` captures `PROMPT_FILE` at its own import.
- **Per-novel paths live entirely in each `config.yaml`'s `paths:` section**, joined onto `ROOT`. The engine has no hardcoded knowledge of `novels/`; isolation is purely a path convention, created by `config_template.yaml`'s `__NOVEL__` placeholder.
- **`chapter_completed.json` must be written synchronously** in the commit action, never deferred (loop-leak invariant above).
- **`save_chapter` refuses to write chapters under 500 chars** (`writing.py`), so provider refusals never persist as legitimate chapters.
- **`review_round0.json` is the FPY′ ruler's input.** Never overwrite it with a repaired or rescued draft; a v2 that archived its best attempt as round 0 would win its own A/B.
- **`cacheable_prefix` / `canon.StoryState.stable` content changes invalidate the prompt cache** for every subsequent chapter — only modify them when the cache cost is worth it, and never move a volatile byte into the stable head.
- **The prose judge must NOT use the cacheable_prefix.** Its whole value is being a judge that hasn't been steeped in the (possibly drifted) book context.
- **`style_health` is the objective anchor against score inflation.** Don't relax its thresholds to make chapters "pass"; the penalty exists to fight the model's over-rating of fragmented prose, not to be tuned away.
- **`voice_baseline.md` is frozen on purpose** — re-deriving voice from drifted prose is exactly the self-feeding loop that caused style collapse.
- **Reasoning knobs are not an A/B variable, and a release threshold is not an FPY dial** (LESSONS §1, §2).
- **Live API keys sit in `config.yaml` / `config_template.yaml` / `novels/*/config.yaml`.** All gitignored — never echo them into tracked files or logs. New per-novel configs inherit the template's keys, so parallel novels share quota.
- **`config_template.yaml` is gitignored but must exist on disk** for `novel.py create`; don't delete it. When adding config keys, edit **both** it and the tracked credential-free `config_template.example.yaml` (the fresh-clone fallback). The `!config_template.example.yaml` negation in `.gitignore` is required — the broad `config_*.yaml` rule would swallow it. Incident record: LESSONS §10.
- **`GENRE_PROFILES` shared constants affect all genres.** Modifying `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT`, `_SELF_REVIEW_PREAMBLE`, or `_OUTPUT_SECTION` in `writing.py` changes every genre's writer prompt at once; per-genre overrides go in the `GENRE_PROFILES` entry. **`_OUTPUT_SECTION` is additionally load-bearing for `v2/write.py`**, which replaces it by exact string match and raises if it has drifted. Same shared-constant caution for `DIAGNOSE_CORE`/`DIAGNOSE_COMMON_FOOTER` in `refine.py`.
- **Ending awareness (`ending_aware`, default true) only fires when `max_chapters` is set.** In short-novel mode the final chapter (`chapter_num == max_chapters`) gets `CLOSING_RULES_BLOCK` (writing.py), skips hook-strength pressure, and refine demands closure instead of a cliffhanger. Detection is `config.py:is_final_chapter`. Pure char-target long novels have no deterministic finale, so this is inert there.
