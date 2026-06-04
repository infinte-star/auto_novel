# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Universal multi-novel AI writing framework. The core engine is an automated
long-form Chinese web novel generation pipeline that targets a configurable
character count (`novel.target_words`) by repeatedly running plan → write →
review → revise → extract loops until done. Optional post-completion `refine`
pass rewrites in 5-chapter groups under intensities chosen by a diagnose LLM call.

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
python novel.py script --input PATH      # convert ANY novel text file -> 短剧 screenplay (standalone)
python novel.py script <name> --chapters 1-3  # convert chapters 1..3 of novels/<name>/
```

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

There is no test suite, lint config, or build step.

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

## Architecture

### Top-level loop (`pipeline.py:main`)
1. `bootstrap()` once — generates `state.md`, `memory/{bible,characters,timeline,threads,volume_plan}.md` from `prompt.md`
2. Loop: `find_last_chapter()` → `generate_one_chapter()` until `count_chars(book.md) >= target_words`
3. `BackgroundTasks` thread pool runs finalization (extract + structured-state + state.md), stage reviews, memory compression, adaptive replans, and next-chapter plan prefetches off the critical path
4. After completion, optional `refine.refine_book()` if `novel.refine_after_complete: true`

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
1. `generate_candidate_plans` — N parallel candidates, each forced into a different strategy (`scene-driven`, `character-driven`, `thread-driven`, `institutional`, `reversal`, `pressure-payoff`) selected by an epsilon-greedy bandit over historical `plan_arbitration` events
2. Optional `screen_candidates` (skipped when `plan_skip_screen: true`)
3. `review_candidate_plans` — fused 6-axis review (world/character/rhythm/payoff/foreshadowing/reader) per candidate, one LLM call expanded into 6 legacy reports via `_explode_fused_axes`. Toggle with `fused_plan_review` (true) — the legacy 6-parallel-calls path is still in the codebase
4. `arbitrate_plan` — picks `selected_index` and emits a `merged_plan` plus `required_constraints`

### Writing & revision (`writing.py`)
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
  plans. `planning.py:create_plan` warns and appends `required_constraints` when
  similarity exceeds `scene_dedupe_sim_warn`, stopping the engine from endlessly
  slicing one micro-scene. Gated by `scene_dedupe_enabled`.
- **`retrieval.py`** — dependency-free TF-IDF char-bigram RAG (no embeddings — the
  only dependency is `openai`). `index_chapter` is called idempotently from
  `save_chapter` and writes `logs/retrieval_index.json`; `retrieval_block` builds
  a "## 相关历史原文（检索…）" section from the plan's fields for the writer prompt.
  `backfill_index` indexes a finished book. Gated by `rag_enabled` (`rag_top_k`,
  `rag_exclude_recent`).
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

### Adaptive cost control (`planning.py`)
- `_effective_candidate_count` downshifts the number of candidate plans (overriding
  `candidate_plans`) once quality is stably high — min score ≥ `adaptive_downshift_score`
  over an `adaptive_downshift_window`-chapter window, after `adaptive_downshift_warmup`
  chapters. Gated by `adaptive_downshift_enabled`.

### Memory layers (`memory.py`)
Two distinct context builders feed different LLM calls:
- `cacheable_prefix` — exact-bytes prefix shared across calls (creative brief + voice + bible + characters), keyed by sha1 of source files. Identical bytes ⇒ provider prompt-cache hits. **Whenever you change how this string is assembled, you invalidate the cache for every existing chapter.**
- `writing_memory_context` — small variable portion (state + threads + recent metrics + volume plan head) for write/revise/review hot path
- `memory_context` — full layered context (4 tiers, char-budgeted) for plan generation and event extraction
- `lite_memory_context` — heavily abbreviated for plan-review/screening

`compress_all_memory` consolidates per-chapter `## ChN` sections in bible/characters/timeline/threads when files exceed `memory_max_kb` or every `memory_compress_every` chapters; archives the old sections under `logs/memory_archive/`.

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
Reads finished `chapters/*.md` in 5-chapter groups, asks an LLM to assign per-chapter intensity (`polish` / `restructure` / `rewrite`) plus up to 4 anchor chapters from elsewhere in the book. Refined output goes to `chapters_refined/` and `book_refined.md`; `chapters/` and `book.md` are never modified. Per-group checkpoints under `logs/refine/group_NNNN.json` make the pass resumable. Sanity check `_refined_text_acceptable` rejects refines that shrink below `refine_min_keep_ratio` (default 0.6) or grow beyond 3× original.

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
- **API keys are committed in `config.yaml` / `config_template.yaml`.** Both are gitignored, but they hold live keys — don't echo them into tracked files or logs. New per-novel configs inherit the template's keys, so parallel novels share quota.
- **`config_template.yaml` is gitignored but must exist on disk** for `novel.py create` to work. Don't delete it.
- **Ending awareness (`ending_aware`, default true) only fires when `max_chapters` is set.** In short-novel mode, the final chapter (`chapter_num == max_chapters`) gets a `CLOSING_RULES_BLOCK` (writing.py) + a planning ending directive, skips hook-only-revise (pipeline.py), and refine's diagnose/refine prompts demand closure instead of a cliffhanger. Detection lives in `config.py:is_final_chapter`. Pure char-target long novels (no `max_chapters`) have no deterministic finale, so this is inert there and per-chapter behaviour is unchanged.
