# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Automated long-form Chinese web novel generation pipeline. Targets 2M+ characters
(`novel.target_words` in `config.yaml`) by repeatedly running plan → write → review →
revise → extract loops until done. Optional post-completion `refine` pass rewrites in
5-chapter groups under intensities chosen by a diagnose LLM call.

The novel content (history-rewrite of late-Ming Chongzhen reign) is described in
`prompt.md`; the pipeline itself is content-agnostic and only consumes that brief.

## Common commands

```bash
pip install -r requirements.txt        # only dependency is openai>=1.0.0
python run.py                          # main entry; resumes automatically from last checkpoint
python restart.py                      # kill any running pipeline + relaunch detached
python restart.py --foreground         # attach restart to current console
python restart.py --kill-only          # stop the pipeline without relaunching
python repair_stub_chapters.py         # regenerate provider-refusal stubs (run while pipeline is stopped)
python check_token_plan_keys.py --keys-file keys.txt   # validate which api keys still work
```

Windows shortcuts: `start_pipeline.bat` (foreground) and `restart.bat` (uses
`E:\pycharmproject\allvenv\novel\Scripts\python.exe` if present, else PATH `python`).

There is no test suite, lint config, or build step.

## Configuration

`config.yaml` is parsed by a hand-rolled YAML-subset reader in `config.py:load_config`
(not real YAML — only `section:` headers and `key: value` pairs, no nested maps,
no lists, no anchors). Adding new keys requires updating the `required` dict in
`load_config` if they're mandatory.

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
- `revise_chapter` first tries surgical `apply_review_patches` (replace/insert_after/delete by literal substring locator); only falls back to a full LLM rewrite when fewer than `revise_patch_min_frac` of patches apply cleanly
- `revise_hook_only` rewrites only the last ~400 chars when `hook_strength < hook_strength_min`, copying the head verbatim

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

## Things to be careful with

- **Don't add `cd <project>` before `git` commands** — bash already runs in the project root.
- **`config.yaml` is not real YAML.** Anchors, lists, nested maps will silently fail to parse; values become strings. The parser only understands `section:` and indented `key: value`.
- **Background-task ordering** is load-bearing. The barriers in `generate_one_chapter` (`wait_label("chapter_finalize_ch{n-1}")` and the prefetch wait) keep memory/threads consistent. Re-ordering them can cause the next plan to see stale state.
- **`save_chapter` refuses to write chapters under 500 chars** (`writing.py:732`). This guards against provider refusals being persisted as legitimate chapters. `repair_stub_chapters.py` exists to fix older stubs (Ch29/Ch43) that pre-date this guard.
- **`cacheable_prefix` content changes invalidate the prompt cache** for every subsequent chapter — only modify it when the cache cost is worth it.
- **API keys are committed in `config.yaml`.** Do not assume new keys can be added there silently — they're visible in git history.
