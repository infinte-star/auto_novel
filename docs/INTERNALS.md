# INTERNALS.md — mechanical reference

Companion to `CLAUDE.md`. That file holds the rules and the per-chapter pipeline;
this one holds the plumbing you look up rather than memorize: persistence schema,
checkpoint layout, the LLM call layer, shared prompt constants, and the
post-completion / standalone tools.

`docs/LESSONS.md` holds the measurements and post-mortems behind the rules.

---

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
| `telemetry.py` | cross-book sink `telemetry/global.db` (write-only observer; populated by `telemetry backfill` only, never by a run) |
| `refine.py`, `fossil_fix.py`, `screenplay.py`, `package.py` | post-completion / standalone tools |
| `compare.py` | `compare` / `ablate` / `fork` experiment harness |
| `trial.py`, `benchmark.py` | opening trials, local sample library |
| `tools/*.py` | zero/low-LLM analysis: `fpy_prime` (acceptance metric), `replay_gates` (settles gate-logic changes), `ccr_baseline`, `gate_census`, `orphan_gates` (does a gate still fire), `orphan_defs` (does anything still name it — defs / constants / config keys), `replay_l0`, `prompt_census`, `pairwise_ab`, `probe_reasoning`, `defossil`, `rebuild_memory`, `truncate_arm` |

---

## Quality gates (`quality.py`)

`GateRegistry` is a metadata registry, **not a dispatcher** — nothing calls
`REGISTRY.get(name)(...)`. A gate runs only where some module calls it by name,
which is why the table below is the ground truth and not the `@register`
decorators.

| gate / layer | config | behaviour |
| --- | --- | --- |
| `quality.style_health(text, config, em_history=…)` | `style_health_enabled`, `style_em_dash_per_kchar_warn`/`_bad`, `style_em_dash_trend_window`, `style_min_avg_sentence_chars`, `style_fragment_line_ratio_max`, `style_penalty_cap`, `style_penalty_block` | deterministic prose metrics (em-dash density, avg sentence length, fragment-line ratio, dialogue presence) → `penalty`, `flags`, writer `directives`. Blocks at `style_penalty_block`. **`em_history` is not optional in practice** — without it the em-dash TREND term is silent and the gate is strictly smaller than this row describes; `v2/accept.py:_em_history` supplies it from `chapter_metrics` |
| `quality.scene_similarity(plan, recent_plans)` | `scene_dedupe_enabled`, `scene_dedupe_window` (8), `scene_dedupe_sim_block` (0.82) | Jaccard similarity of a plan's scene skeleton (conflict/payoff/pressure/goal/beats) vs recent cards projected through `card_to_plan`. **Blind to `where` and `turn`** — mutating either leaves the similarity at 1.000; those are `validate_card`'s neighbour rules. ONE tier, and the other four are deleted rather than dormant (see the escalation note below; REDESIGN_V2 §9.11.3) |
| `quality.cross_chapter_repetition` | `style_cross_repeat_reject_count` (8) | signature clauses reused verbatim across chapters → a `level` of `advise` or `reject` |
| `quality.dialogue_health` | `dialogue_health_enabled`, `dialogue_char_ratio_min` (0.10), `dialogue_char_ratio_target` (0.20) | dialogue-ratio gate over text inside `"…"`; reached via `fix.py`'s L1 dialogue injection |
| `quality.book_wide_fossils` | `book_fossil_enabled`, `book_fossil_chapter_frac` (0.30), `book_fossil_min_chapters` (6), `book_fossil_hard_ratio`, `book_fossil_struct_count` | 6-char CJK n-grams recurring across a large fraction of the WHOLE book (what `cross_chapter_repetition`'s 6-chapter window structurally misses). **A hard fossil requires `current_chapter` and only indicts a chapter that actually contains the phrase** — the ratio is book-cumulative, so without that the gate latches on and rejects compliant chapters forever. `book_fossil_hard_ratio` is floored at the candidacy fraction (below it, every candidate is automatically hard). LESSONS §13 |
| `quality.descriptor_frequency` | `descriptor_frequency_enabled` | over-used descriptors across the book → `gate_rejects`; answered by keep-1 rotation |
| `quality.adjacent_repetition` | `adjacent_repeat_*` | this chapter vs the previous one → `adjacent_repeat_block` |
| `quality.length_band_check` | `chapter_min_chars` (2800), `chapter_max_chars` | band check; the short side is answered by L1 expand-to-band, the long side blocks. **Its config key only controls its penalty (default false) — the gate always runs**, so `REGISTRY.is_enabled` is not a "did this gate run" test |
| `quality.opening_hook_gate` | `opening_hook_gate_enabled` | first-line hook strength → block; L0 demotes a scenery opening |
| `quality.genre_adherence` | `genre_drift_reject_enabled` | genre-signal drift vs recent chapters → `gate_rejects` |
| `quality.plan_executability_gate`, `plan_visual_payoff_check`, `narrative_pattern_repetition` | card-scope | plan/card gates, fixable before a word is written. Currently reached only by `tools/replay_gates.py` — see CLAUDE.md's wiring-gap note |
| `retrieval.py` RAG | `rag_enabled`, `rag_top_k`, `rag_exclude_recent` | dependency-free TF-IDF char-bigram index (no embeddings). `index_chapter` is called idempotently from `save_chapter` → `logs/retrieval_index.json`; `retrieval_block` builds the "## 相关历史原文（检索…）" section; `backfill_index` indexes a finished book |
| `retrieval.exemplar_block` | `exemplar_rag_enabled` (false), `exemplar_rag_top_k` | quotes the book's own strongest chapters back as style anchors. **Rank-based, not an absolute score threshold**; picks one dialogue-dense + one action-dense exemplar (LESSONS §7) |
| `quality.fingerprint_avoidance_context` | `fingerprint_enabled` | the 全书结构指纹 block, consumed by **`v2/beat.py:_fingerprints` — once per arc, not once per chapter**. **Emits an aggregate (recurring move bigrams/trigrams + payoff/conflict/move frequencies), never one line per chapter** — the per-chapter form grew linearly while carrying no signal; the aggregate amortizes to ~123 chars/chapter and does not grow with the book (LESSONS §8, REDESIGN_V2 §9.11.1). **It returns the literal string `"None"`, not `""`, when it has nothing to say** (a v1 template convention), which `_fingerprints` filters — a header promising overused patterns with the word "None" under it is worse than no header. `store_chapter_fingerprint` keeps writing `skeleton_tokens` even though nothing reads it: that column is what lets a future recalibration be replayed offline. |

Two routing rules: **scene dedupe is ONE tier**, the missing WARN/short-novel/
0.97-ceiling/`force_retry` tiers were deleted on measurement, not restored
(REDESIGN_V2 §9.11.3); a **`cross_chapter_repetition` reject is structural**,
not cosmetic — it lands in `gate_rejects` before `rescue` can buy a rewrite.

`quality.py` registers 23 gates, of which the chapter loop reaches 19 (CLAUDE.md
names the wiring-gap disposition of the other 12). Gate calibration is measured,
not guessed: `python tools/gate_census.py` and `python tools/replay_l0.py`.

---

## StoryState (`v2/canon.py`)

ONE projection, ~15k chars, split by mutability: `stable` (~5.6k, the
prompt-cache prefix) vs `volatile` (~11.5k). Three rules the projections obey:
1. **A clipped section says so, in the text.** `_clip` appends
   `〔…截断 N 字〕`; `_clip_items` drops whole items from the END and says how
   many. Silent head truncation is what starved the mid-book for 40 chapters
   before anyone noticed (LESSONS §6).
2. **Empty and clipped are different facts.** An empty section emits no
   header — printing `## 伏线` with nothing under it lies whenever the truth
   is "the budget ate them".
3. **Persistence has one writer.** `apply_delta` delegates to
   `writing.update_structured_state`; a second writer against the same schema
   is a second ruler that drifts invisibly.

---

## Memory (`memory.py`)

v2 gets its per-chapter context from `v2/canon.py`, so `memory.py` survives for
four jobs:
- **`bootstrap`** — one-time generation of `state.md` + `memory/*.md` from
  `prompt.md`. `extract_contract` runs BEFORE `_bootstrap_chain` and its
  markdown feeds the bible + characters calls (`contract_md=`).
- **`cacheable_prefix` / `memory_context`** — still read by `arc.py`,
  `trial.py`, `package.py`. Four tiers must not overlap (tier2/tier3 are
  prefixes of tier4; tier4 emits only the older tail).
- **`volume_plan_window(text, chapter_num, cap, lookahead)`** — keeps blocks
  whose header covers `chapter_num`, reduces out-of-range volumes to a header
  breadcrumb, and inside kept blocks keeps only `| ChN |` rows for
  `[chapter_num-1, chapter_num+lookahead]`.
- **`volume_transition_directive`** — restates a volume boundary inside the
  arc span as a hard obligation in the arc prompt (HARD level only).

Per-chapter state persistence costs zero extra LLM calls: the write call
returns the `ChapterDelta`, and `canon.apply_delta` → pure DB writes +
deterministic markdown render consume it.

---

## Cross-book telemetry (`telemetry.py`)

Each novel has its own `story_state.db`; `telemetry.py` is the ONE shared sink
(`telemetry/global.db`, WAL, one fresh connection per write so N processes
write concurrently). Strict observer / safe no-op: any failure returns an
empty value and never stalls a chapter. Currently write-only — the
consumption layers were deleted (LESSONS §9).

Nothing writes it during a run — the only populator is the user-typed
`novel.py telemetry backfill`, reconstructing from each book's own
`story_state.db` + checkpoints. Which events reach the sink is
`telemetry.IMPORTED_EVENT_TYPES`, not "all of them".

---

## Experiment harness CLI (`compare.py`)

- `compare <a> <b>` — deterministic zero-LLM report → `experiments/`.
- `fork <name> --as <new>` — the A/B tool for anything mid-book. Branches at
  HEAD (copies `memory/`, `chapters/`, `book.md`, `state.md`,
  `story_state.db`, RAG index) so both arms start byte-identical.
- `ablate <name> --flip <key>` — chapter-capped copy with ONE key flipped,
  restarts at Ch1.
- `tools/pairwise_ab.py` — CLI over `v2/anchor.py`. `--b-from` offsets pairing
  for arms at different outline positions; `--probe N` calibrates position
  bias; `--anchor` judges one arm against `benchmarks/anchor/` and rejects
  `--b`/`--b-from`/`--probe`/`--all`.

---

## Persistence (`store.py`)

SQLite (`story_state.db`, WAL) is the primary store. Tables: `events`,
`chapter_metrics`, `entities`, `open_threads`, `agent_reports`,
`stage_constraints`, `causal_links`.

If `sqlite3` is unavailable, `JsonStoryStore` writes `logs/story_state.json`
instead — most code branches on `isinstance(conn, JsonStoryStore)`, and a few
features are SQLite-only: stage constraints, causal links, and the two `arc.py`
readers `v2/beat.py` depends on (`validate_plan_continuity`,
`get_silent_threads`). `stage_constraints` and `causal_links` are written but
currently have no reader in the live engine.

Two per-novel artifacts live **outside** the store and are both safe to delete:
- `logs/retrieval_index.json` — rebuilt by `retrieval.backfill_index` or the next
  `save_chapter`
- `logs/arc_cards.json` — the ChapterCard archive; re-derived by the next arc call
  (the card for an already-written chapter is not, so deleting it loses the record
  of what past chapters promised, which `tools/ccr_baseline.py` reads)

`memory/voice.md` is written **once, by `memory.bootstrap`, and never re-derived**.
v1 refreshed it per-chapter from recent prose and needed a frozen
`voice_baseline.md` plus a skip-on-collapse rule to stop the self-feeding loop
(LESSONS §3); v2 has no refresh path, so the baseline is the file itself. A v2 book
has no `voice_baseline.md`, and `tools/rebuild_memory.py` still restores one from an
older book if it finds it.

---

## Checkpoints (`checkpoint.py`)

Every action in `v2/run.py`'s decision table writes under
`logs/checkpoints/ch{NNNN}/`:

```
arc_generated.json             ← whole-arc response; only on the span's first chapter
chapter_card.json              ← this chapter's card (after normalize + validate)
chapter_draft.json             ← the draft: text, title, delta, prompt_chars, attempt
review_round0.json             ← the RAW first draft's report. Never overwritten by a
                                 repaired draft — this is what tools/fpy_prime.py replays
canon_claims.json              ← canon findings, already cite-or-drop filtered
card_patch.json                ← written by chapter N-1 FOR chapter N, and only when
                                 that chapter produced a `target: next_card` finding
                                 (0 files in a clean 30-chapter run)
card_replan_attempt1_rescue.json ← only when blocks survived to the rescue row
final_review.json              ← the shipped text's report; also read by the NEXT
                                 chapter's _act_patch_next
extraction.json → structured_state_done.json → state_file_done.json → chapter_completed.json
```

There is no `chapter_saved.json` and no `chapter_current_v2.md`: v1 wrote the draft
text as a bare `.md` and confirmed the save separately, while v2 keeps the draft
inside `chapter_draft.json` and treats `chapters/NNNN.md` itself as the commit
record. `CHECKPOINT_VERSION` still versions the layout.

`extraction.json` is the `ChapterDelta` the write call returned, not a separate
extraction call's output — the file name is kept so `store`/`telemetry` backfill and
the offline tools keep reading one spelling across both engines' archives.

Resume detection is `should_resume_existing_chapter`: chapter file exists AND
checkpoint dir exists AND `chapter_completed.json` does not. Bumping
`CHECKPOINT_VERSION` invalidates all `.json` checkpoints from prior versions.

`v2/run.py:_restore` rebuilds a `ChapterRun` from these files, which is why the
decision table's predicates are all "is this artifact present/absent" rather than
"which step did we reach": a resumed chapter re-derives its position instead of
trusting a stored cursor.

Refine groups live in `logs/refine/group_NNNN.json`; screenplay segments in
`<out>.checkpoints/seg_NNNN.json`.

---

## LLM calls (`llm.py`)

`call_llm`:
- streams with three timeouts — `stream_timeout` (total), `stream_idle_startup`,
  `stream_idle_steady`
- salvages partial output past `stream_salvage_min_chars`
- falls back to `reasoning_content` when `content` comes back empty
- retries refusals (`REFUSAL_PATTERNS`)
- emergency-truncates user messages by section priority when the prompt exceeds
  `context_window * 1.8` chars

JSON contracts:
- `json_prompt(user)` appends the mandatory output-contract block. `call_llm`
  infers JSON mode from that string's presence and sets
  `response_format={"type": "json_object"}`, retrying without it when a provider
  returns 400/404/422 mentioning `response_format`.
- `load_json_with_repair` calls `safe_json_loads` (which runs
  `_repair_truncated_json` for cut-off streams) and, on failure, asks the LLM to
  repair the JSON. It returns `fallback` instead of raising when one is given.
  Refusal-prefixed responses skip the repair attempt.

---

## Shared prompt constants (cross-module deduplication)

Editing any of these changes every consumer at once.

| constant | defined in | consumed by |
| --- | --- | --- |
| `STYLE_HEALTH_GUARDRAILS` (健康文风护栏) | `memory.py` | `VOICE_CHAIN_SYSTEM` (memory.py). v1's `VOICE_ANCHOR_SYSTEM` was its second consumer and went with `review.py` |
| `_VOLUME_PLAN_STRUCTURE_SPEC` (OKR structure, 线索兑现表, pacing discipline) | `memory.py` | `VOLUME_PLAN_CHAIN_SYSTEM` (memory.py). v1's `REPLAN_SYSTEM` was its second consumer; v2 does not regenerate volume plans mid-book |
| `ARC_SYSTEM`, `normalize_card`, `validate_card`, `card_to_plan` (what a ChapterCard *is*) | `arc.py` | `v2/beat.py` — imported, never copied, so there is exactly one definition of the card schema `v2/accept.py:contract_fulfilment` scores |
| `_SELF_REVIEW_PREAMBLE`, `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT` | `writing.py` | every genre's writer prompt via `_build_write_system()` |
| `_OUTPUT_SECTION` | `writing.py` | same, **plus `v2/write.py`, which replaces it by exact string match and raises if it has drifted** — v1's version ends with 「严禁输出…JSON」, the exact opposite of what v2 asks for, and an appended override would fail silently (the writer obeys the older rule, no delta parses, and the engine quietly costs an extra call per chapter) |
| `DIAGNOSE_CORE`, `DIAGNOSE_COMMON_FOOTER` | `refine.py` | every genre's diagnose prompt via `_build_diagnose_system()` |

v1's `_EXECUTABILITY_DOCTRINE` (score baseline 6.5, "shootable action", reversal
requirement) went with `planning.py`. Its content did not vanish: the concreteness
requirement it enforced by prose instruction is now `arc.ARC_SYSTEM`'s rule 2, which
is what makes a card's fields greppable and therefore makes CCC decidable without an
LLM.

---

## Refine pass (`refine.py`)

`python novel.py refine <name>` — explicit manual step
(`refine_after_complete` defaults to **false**).

Reads finished `chapters/*.md` in 5-chapter groups and asks an LLM for per-chapter
intensity (`polish` / `restructure` / `rewrite`) plus up to 4 anchor chapters from
elsewhere in the book. Output goes to `chapters_refined/` + `book_refined.md`;
`chapters/` and `book.md` are **never** modified. Per-group checkpoints under
`logs/refine/group_NNNN.json` make the pass resumable.

`_refined_text_acceptable` rejects a refine that shrinks below
`refine_min_keep_ratio` (0.6) or grows past an intensity-tiered ceiling:
`polish` 1.5× (via `refine_max_grow_ratio`), `restructure` 2.0×, `rewrite` 2.5×.

Diagnose prompts mirror the writer's shared-base pattern: `DIAGNOSE_CORE` +
`DIAGNOSE_GENRE_DIMS[preset]` + `DIAGNOSE_COMMON_FOOTER`, assembled by
`_build_diagnose_system()`. Adding a genre's diagnose prompt is one new dict entry.

---

## Fossil fix (`fossil_fix.py`)

`python novel.py fix-fossils <name>` — zero LLM calls, purely deterministic.

Scans finished chapters for CJK n-gram fossils (phrases recurring in ≥15% of
chapters) and replaces excess occurrences with rotated synonym variants from
`FOSSIL_REPLACEMENTS`. Keeps `--max-keep` (default 1) per chapter;
`--custom-replacements <json>` supplies user-defined `{phrase: [alternatives]}`
mappings. Reads from `chapters_refined/` when available, writes `chapters_fixed/` +
`book_fixed.md`.

Why the replacement bank is deliberately small (rotating book-specific proper nouns
would be canon corruption): `docs/LESSONS.md` §4.

## Screenplay conversion (`screenplay.py`)

Standalone novel-text → 短剧 (vertical-drama) converter, decoupled from the
generation pipeline. CLI: `python novel.py script --input PATH` (any file), or
`python novel.py script <name> --chapters A-B` / bare `<name>` (book.md).

`convert_file(input, out)` / `convert_text(...)` split input on `第N章` markers (or
char-budgeted paragraph packing when there are no markers), then run **one LLM call
per segment** with continuity carry-over (running 第N集 episode number, last
segment's tail) so episode/scene numbering stays monotonic across calls.

Output follows the reference duanju format:

```
第N集
N-N 地点 时段 内/外
人物：…
△动作行
角色：台词
（字幕：…） / 角色（OS）：旁白 / （镜头：…）
```

Per-segment checkpoints under `<out>.checkpoints/seg_NNNN.json` make the pass
resumable. Default output is a `scripts/` dir: `novels/<name>/scripts/` in
per-novel mode, or a `scripts/` subdir next to the input file in standalone
`--input` mode (override with `--out`).

It reuses the engine's config-driven LLM client only for API keys; with no
`--config`/`NOVEL_CONFIG` it falls back to `config_template.yaml` (the shared
keys). Tuned by `script_seg_chars`, `script_max_tokens`, `script_temperature`.

---

## Zero/low-LLM analysis tools (`tools/`)

| tool | what it does |
| --- | --- |
| `fpy_prime.py` | **the acceptance metric** — replays `quality.hard_block_reasons` over archived `review_round0.json` with `score` excluded, thresholds pinned, derivative novel dirs dropped. Names the failing gate for every miss. Read CLAUDE.md's measurement-discipline section before quoting a number from it |
| `replay_gates.py` | the tool for a change to a gate's **logic** — recomputes the changed gates from primary data (chapter texts, archived cards/arbitrations, each novel's own config) and re-runs the release rule on the corrected payload. `fpy_prime` cannot do this: it replays frozen verdicts |
| `ccr_baseline.py` | contract-fulfilment rate per chapter. On v2 books it reads real `chapter_card.json`; on v1 archives it uses the arbitration's `merged_plan` as a **proxy** card and says so |
| `replay_ccc.py` | replays `accept.contract_fulfilment` over an archive after the checker itself changes (the CCC counterpart of `replay_gates`) |
| `gate_census.py` | per-gate ran/fired/penalty over archived reviews — the data behind any gate-deletion decision, and the tool that decides wire-vs-delete for the 12 gates the v1 deletion orphaned. Read `fire%` and `advise%` separately (LESSONS §4) |
| `replay_l0.py` | replays the L0 fixers over finished chapters (647 reviews → 29 chapters repaired, 0 made worse, all length changes within ±2%) |
| `prompt_census.py` | where the context budget goes, by call tag, from `novels/*/logs/llm_calls.jsonl` |
| `window_cost.py` | cost/context accounting over a chapter window |
| `pairwise_ab.py` | blinded pairwise prose judge (the CLI over `v2/anchor.py`) on the chapters where two A/B arms actually differ. `--b-from` pairs arms sitting at different outline positions; `--probe N` judges chapters against themselves to calibrate position bias. Quote `n_decisive`, not `n` |
| `truncate_arm.py` | trims an A/B arm back to a chapter so both arms cover the same window |
| `probe_reasoning.py` | probes whether a gateway honours the reasoning knobs (LESSONS §1) |
| `defossil.py` | LLM fossil-phrase replacement + descriptor thinning on a chapter range |
| `rebuild_memory.py` | regenerate memory files from a chapter range (spoiler-free restart) |

**Any tool here that calls an LLM must redirect its own logs.** `call_llm` appends to
`paths.logs_dir/llm_calls.jsonl`, which is the file `compare._llm_totals` reads for
calls/chapter — so a tool borrowing an arm's config for API keys charges that arm for
the cost of being measured. Use `dataclasses.replace(paths, logs_dir=…)`; the
incident record is in CLAUDE.md.
