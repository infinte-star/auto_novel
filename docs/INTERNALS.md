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
