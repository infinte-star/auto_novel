# INTERNALS.md — mechanical reference

Companion to `CLAUDE.md`. That file holds the rules and the per-chapter pipeline;
this one holds the plumbing you look up rather than memorize: persistence schema,
checkpoint layout, the LLM call layer, shared prompt constants, and the
post-completion / standalone tools.

`docs/LESSONS.md` holds the measurements and post-mortems behind the rules.

---

## Persistence (`store.py`)

SQLite (`story_state.db`, WAL) is the primary store. Tables: `events`,
`chapter_metrics`, `entities`, `open_threads`, `agent_reports`,
`stage_constraints`, `causal_links`.

If `sqlite3` is unavailable, `JsonStoryStore` writes `logs/story_state.json`
instead — most code branches on `isinstance(conn, JsonStoryStore)`, and a few
features are SQLite-only: stage constraints, causal links, plan-continuity
validation, silent-thread detection.

Two per-novel artifacts live **outside** the store and are both safe to delete:
- `logs/retrieval_index.json` — rebuilt by `retrieval.backfill_index` or the next
  `save_chapter`
- `memory/voice_baseline.md` — rebuilt on the next `refresh_voice_anchors` (but see
  the "frozen on purpose" rule in `CLAUDE.md`: deleting it re-captures the baseline
  from *current* prose)

---

## Checkpoints (`checkpoint.py`)

Every stage in `generate_one_chapter` writes under `logs/checkpoints/ch{NNNN}/`:

```
plan_initial_attempt0_candidates.json / _reports.json / _arbitration.json
plan_initial_selected.json → validated_plan.json
chapter_current_v2.md          ← versioned via CHECKPOINT_VERSION
review_round0.json … final_review.json
chapter_saved.json
extraction.json → structured_state_done.json → state_file_done.json → chapter_completed.json
```

Resume detection is `should_resume_existing_chapter`: chapter file exists AND
checkpoint dir exists AND `chapter_completed.json` does not. Bumping
`CHECKPOINT_VERSION` invalidates all `.json` checkpoints from prior versions.

Arc cards persist separately in `logs/arc_cards.json`; refine groups in
`logs/refine/group_NNNN.json`; screenplay segments in
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
| `STYLE_HEALTH_GUARDRAILS` (健康文风护栏) | `memory.py` | `VOICE_CHAIN_SYSTEM` (memory.py), `VOICE_ANCHOR_SYSTEM` (review.py) |
| `_VOLUME_PLAN_STRUCTURE_SPEC` (OKR structure, 线索兑现表, pacing discipline) | `memory.py` | `VOLUME_PLAN_CHAIN_SYSTEM` (memory.py), `REPLAN_SYSTEM` (review.py) — so both emit structurally identical volume plans |
| `_EXECUTABILITY_DOCTRINE` (score baseline 6.5, "shootable action", reversal requirement) | `planning.py` | `CANDIDATE_PLAN_SYSTEM`, `ARBITER_SYSTEM` |
| `_SELF_REVIEW_PREAMBLE`, `_OUTPUT_SECTION`, `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT` | `writing.py` | every genre's writer prompt via `_build_write_system()` |
| `DIAGNOSE_CORE`, `DIAGNOSE_COMMON_FOOTER` | `refine.py` | every genre's diagnose prompt via `_build_diagnose_system()` |

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

## Targeted chapter fixes (`chapter_fix.py`)

LLM-based per-chapter rewrites for finished books — low dialogue, short length,
monotonous endings. Reads `chapters_fixed/` (or `chapters_refined/`) and writes back
in place, one LLM call per chapter with problem-specific instructions. **Not wired
to the `novel.py` CLI**; call it programmatically.

---

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
| `gate_census.py` | per-gate ran/fired/penalty over archived reviews — the data behind any gate-deletion decision. Read `fire%` and `advise%` separately (LESSONS §4) |
| `replay_l0.py` | replays the L0 fixers over finished chapters (647 reviews → 29 chapters repaired, 0 made worse, all length changes within ±2%) |
| `prompt_census.py` | where the context budget goes, by call tag, from `novels/*/logs/llm_calls.jsonl` |
| `pairwise_ab.py` | blinded pairwise prose judge on the chapters where two A/B arms actually differ |
| `probe_reasoning.py` | probes whether a gateway honours the reasoning knobs (LESSONS §1) |
| `defossil.py` | LLM fossil-phrase replacement + descriptor thinning on a chapter range |
| `rebuild_memory.py` | regenerate memory files from a chapter range (spoiler-free restart) |
