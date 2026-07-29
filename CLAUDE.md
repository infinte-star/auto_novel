# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**This file holds the rules and the pipeline you must not break.** Four companion docs carry the rest — follow the pointers instead of guessing:

| doc | contents |
| --- | --- |
| `docs/REDESIGN_V2.md` | **the live engine's design doc** (v2) + the A/B that settled it (§9.7) + the load-bearing *why* behind every invariant below (§9.12). Read §0 before adding any quality gate or A/B judged by `score`. |
| `docs/LESSONS.md` | the measurements and post-mortems **behind** these rules. Read the cited section before changing a threshold, deleting a gate, or running an A/B. **Historical**: most of it was measured on the v1 engine, deleted 2026-07-28. |
| `docs/INTERNALS.md` | mechanical reference: module map, quality-gate table, store schema, checkpoint layout, `llm.py` plumbing, shared prompt constants, StoryState/memory/telemetry mechanics, refine / fossil-fix / screenplay tools, `tools/*`, experiment-harness CLI |
| `REDESIGN.md` | the v1 quality/FPY roadmap + P1–P4 execution record. **Historical**, same caveat. |

`README.md` is the user-facing quickstart.

## Overview

Universal multi-novel AI writing framework. The core engine (`engine/`) is an automated long-form Chinese web novel generation pipeline that targets a configurable character count (`novel.target_words`). CLI subcommands live in `commands/`.

Architecture is **a deterministic decision table with four LLM actions** (REDESIGN_V2): plan an arc once every ~10 chapters, write the chapter and its state delta in ONE call, check canon, and repair. Every routing predicate is a pure function over recorded state, so every branch is replayable offline. The v1 engine (~7.3k lines, a five-stage plan committee plus a self-score-keyed rework loop) was deleted at `95361b9` after a matched-position A/B (settlement: REDESIGN_V2 §9.7); recoverable from git history.

The pipeline is **content-agnostic** — it consumes only a creative brief (`prompt.md`) and a config (`config.yaml`). Each novel lives in `novels/<name>/` and runs as an independent OS process, isolated on the engine's process-level global state (`config.PROMPT_FILE`, `memory._CACHEABLE_PREFIX_CACHE`).

## Entry point: `novel.py`

The **only** entry point; all parsing and dispatch lives there (`cmd_*` + argparse subparser tree).

```bash
pip install -r requirements.txt          # only dependency is openai>=1.0.0
python -m unittest discover tests        # pure-function tests only, no LLM

# lifecycle
python novel.py create <name>            # scaffold from config_template.yaml + prompt_template.md
python novel.py run <name>               # detached; resumes from checkpoint
python novel.py run <name> --foreground
python novel.py list
python novel.py stop|restart <name>      # token-exact: `run foo` never matches `run foobar`
python novel.py stats <name>             # scores/penalties/cost + per-role reasoning coverage

# openings & samples
python novel.py trial <name>             # opening variants, no chapters/book touched
python novel.py adopt-trial <name> [id]
python novel.py benchmark list|add ...

# experiments — read LESSONS §5 before running one
python novel.py compare <a> <b>
python novel.py fork <name> --as <new> [--flip <key>] [--set V] [--chapters N]   # preferred
python novel.py ablate <name> --flip <key> [--set V] [--chapters N]              # restarts at Ch1
python novel.py telemetry backfill|stats [--genre G]

# post-completion (details: docs/INTERNALS.md)
python novel.py refine <name>
python novel.py fix-fossils <name>
python novel.py package <name>
python novel.py script --input PATH | <name> --chapters A-B | <name>
```

Multi-novel isolation is pure path convention (`config_template.yaml`'s `__NOVEL__` placeholder → every `paths:` entry inside `novels/<name>/`); no engine changes. `run` sets `NOVEL_CONFIG`/`NOVEL_PROMPT` **before** importing `engine.loop`. Parallel novels share RPM/TPM quota unless given distinct keys.

**`novel.engine` selects the engine, and an unknown value is an ERROR, not a guess.** `v2` (or absent) is the only accepted value; `v1` prints what happened to it and exits 2 — the two engines wrote different checkpoint labels and `review_round0.json` keys, so silently running a v1 config on v2 would be a measurement forgery.

## Configuration

- `config.yaml` is a hand-rolled YAML-**subset** (`engine/config.py:load_config`) — only `section:`/`key: value`, no nested maps/lists/anchors. Mandatory keys go in the `required` dict.
- `NOVEL_PROMPT`/`NOVEL_CONFIG` env vars (read at import time) select `prompt.md`/`config.yaml`; `get_paths` joins `paths:` onto `ROOT`, which is what makes a per-novel config directory-isolating with zero code changes.
- Multi-key/endpoint: `api_key` / `api_keys` (comma list) / `api_key_groups` (`base_url|k1,k2;base_url2|k3,...` fallback). `LLMClientPool` round-robins primaries, falls back to secondaries only when all primaries are dead.
- Per-role (`{planning,writing,extraction,review}_*`) reasoning knobs — `{role}_thinking_mode` (Anthropic/豆包 style) and `{role}_reasoning_effort` (OpenAI style) — exist because gateways honour one or the other. **Both are requests, not guarantees, and must never be an A/B variable** (LESSONS §1). Check `novel.py stats`'s `_reasoning_coverage` before trusting any comparison.

## Architecture invariants

Full module table + quality-gate table + mechanical detail: `docs/INTERNALS.md`. Full design rationale behind every bullet here: `docs/REDESIGN_V2.md` §9.12 (and the section numbers inline below).

- **Top-level loop** (`engine/loop.py:main`): bootstrap once → loop `find_last_chapter`/`run_chapter` until target reached → optional package/refine, both gated off (default false) and both **skipped when the run halted on the quality breaker**. No background thread pool by design.
- **One chapter** (`engine/loop.py:DECISIONS`): the table is re-read **from the top** after every action. Row order is load-bearing — repair sits above `rescue`, repair rows are not gated on `r.blocks`, `review_round0.json` is written once and never overwritten by a repaired/rescued draft (it's what `fpy_prime` replays). `chapter_completed.json` **must** be written synchronously in commit (loop-leak invariant). `RESCUE_ATTEMPTS` is safe to bound because every acceptance member is `scope="chapter"` — nothing latches across chapters.
- **Planning** (`engine/plan.py:generate_arc`): one high-reasoning call per `arc_span` (10) emits a ChapterCard per chapter. **No committee fallback** — every failure path ends in a real card or an exception, never a fabricated one. Context comes from `engine/canon.py`, not `memory.py`'s builders.
- **Writing** (`engine/write.py:write_chapter`): ONE call returns prose, a sentinel line, then the `ChapterDelta` JSON. `writing._OUTPUT_SECTION` is **replaced by exact-string match, not appended to** — `engine/write.py` raises if it has drifted (pinned by `tests/test_write.py`). Prose first, JSON last, so a parse failure loses the delta, never the prose.
- **Acceptance** (`engine/accept.py`): the acceptance set is exactly what `quality.hard_block_reasons` can read and declare a write-off; every member is zero-LLM decidable AND actionable (a book-cumulative quantity may advise, never reject — `GATE_SCOPES`). Output is a v1-schema payload so one `fpy_prime` invocation settles both engines. `contract_fulfilment` (CCC) and `citation_check` (cite-or-drop) are the two v2-native checks.
- **StoryState** (`engine/canon.py`): stable/volatile split; `render()` always emits stable first (the prompt-cache prefix). Never move a volatile byte into the stable head. Three projection rules → `docs/INTERNALS.md`.
- **Quality gates** (`engine/quality.py`, `engine/retrieval.py`): `GateRegistry` is metadata only, **not a dispatcher** — a gate runs only where some module calls it by name. Full 14-row table → `docs/INTERNALS.md`. `style_health` is the objective anchor against score inflation; don't relax it to make chapters "pass".
- **Repair ladder** (`engine/repair.py` → `engine/quality.py`): L0 is zero-LLM, L1 is bounded (`fix_max_l1_calls`). Every fixer is keep-only-if-improved, and a layer is **reverted whole** if it introduces a blocking reason acceptance didn't already have — `style_health` improving is not enough. Fossil rotation is bank-only (rotating book-specific nouns is canon corruption).
- **Measurement discipline**: `tools/fpy_prime.py` is the acceptance metric (score excluded entirely, thresholds pinned) — quote it for any engine A/B. `tools/replay_gates.py` is the only tool that can settle a change to a gate's *logic* (fpy_prime replays frozen verdicts). Current readings (2026-07-28): frozen archive 83.3%, replayed through today's gate logic 92.0% — **always quote the normalized pair**. Before adding any blocking gate: what can THIS chapter do to turn it green? If nothing, it latches (`GATE_SCOPES` makes `scope="book"` advisory-only, structurally).
- **External anchor** (`engine/anchor.py`): WR, the only metric that reads prose and the only one the engine can't award itself. Blind (甲/乙, no names), two-way (swap sides, count only agreeing wins — quote `n_decisive` not `n`), and **structurally has no `cacheable_prefix`** (the module never imports `memory`). `benchmarks/anchor/` doesn't exist yet, so `wr_against_anchor` reports unavailable rather than silently substituting an internal A/B.
- **Experiment harness** (`commands/compare.py`): `fork` (branches at HEAD, preferred for mid-book A/Bs) / `ablate` (chapter-capped, restarts at Ch1) / `tools/pairwise_ab.py`. CLI reference → `docs/INTERNALS.md`. **Short opening runs fabricate positive results** — full protocol is LESSONS §5.
- **Telemetry** (`commands/telemetry.py`): one shared `telemetry/global.db` sink, write-only, populated **only** by the user-typed `telemetry backfill` — never by a run. Mechanics → `docs/INTERNALS.md`.
- **Memory** (`memory.py`): survives for four jobs (bootstrap, cacheable_prefix/memory_context, volume_plan_window, volume_transition_directive) now that `v2/canon.py` owns per-chapter context. Changing `cacheable_prefix` assembly invalidates the prompt cache for every existing chapter. Detail → `docs/INTERNALS.md`.

## Things to be careful with

- **Don't add `cd <project>` before `git` commands** — the shell already runs in the project root.
- **`config.yaml` is not real YAML.** Anchors, lists, nested maps silently fail to parse; values become strings.
- **`NOVEL_CONFIG`/`NOVEL_PROMPT` must be set before importing `engine.loop`/`engine.config`/`memory`** — `engine/config.py` reads them at import time and `memory.py` captures `PROMPT_FILE` at its own import.
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
- **`config_template.yaml` is gitignored but must exist on disk** for `novel.py create`; don't delete it. When adding config keys, edit **both** it and the tracked credential-free `config_template.example.yaml`. The `!config_template.example.yaml` negation in `.gitignore` is required. Incident record: LESSONS §10.
- **`GENRE_PROFILES` shared constants affect all genres.** Modifying `_SENSORY_DIALOGUE_DEFAULT`, `_TIME_MARKER_BAN_DEFAULT`, `_SELF_REVIEW_PREAMBLE`, or `_OUTPUT_SECTION` in `writing.py` changes every genre's writer prompt at once; per-genre overrides go in the `GENRE_PROFILES` entry. **`_OUTPUT_SECTION` is additionally load-bearing for `engine/write.py`**, which replaces it by exact string match and raises if it has drifted. Same shared-constant caution for `DIAGNOSE_CORE`/`DIAGNOSE_COMMON_FOOTER` in `commands/refine.py`.
- **Ending awareness (`ending_aware`, default true) only fires when `max_chapters` is set.** In short-novel mode the final chapter (`chapter_num == max_chapters`) gets `CLOSING_RULES_BLOCK` (writing.py), skips hook-strength pressure, and refine demands closure instead of a cliffhanger. Detection is `engine/config.py:is_final_chapter`. Pure char-target long novels have no deterministic finale, so this is inert there.
