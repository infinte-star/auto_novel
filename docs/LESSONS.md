# LESSONS.md — measured evidence behind the engine's design

Companion to `CLAUDE.md`. `CLAUDE.md` states the **rules**; this file holds the
**measurements and post-mortems** that produced them, so the rules can be
re-derived (or overturned) instead of cargo-culted.

Read the relevant section before you change a threshold, delete a gate, or run an
A/B. Every claim here has a date and a sample size; if you re-measure and get
something else, update the section rather than quietly acting on the new number.

> **Historical file names, live lessons.** Most of what follows was measured on the
> v1 engine, deleted 2026-07-28 (`95361b9`, disposition in `docs/REDESIGN_V2.md`
> §9.8). The *rules* still hold — they are why v2 is shaped the way it is — but the
> modules many sections name (`pipeline.py`, `planning.py`, `review.py`,
> `taxonomy.py`) no longer exist, and functions like `_hard_block_reasons` moved
> (now `quality.hard_block_reasons`). Sections are **not** rewritten to pretend v1
> never happened: a post-mortem edited to match today's code stops being evidence.
> When a section names a file, check `CLAUDE.md`'s module map for where that
> responsibility lives now.

Related docs: `docs/REDESIGN_V2.md` (the live engine's design + the A/B that
settled it), `REDESIGN.md` (the v1 quality/FPY roadmap and its P1–P4 execution
record — historical, same caveat), `README.md` (user-facing quickstart).

---

## 1. The gateway decides whether reasoning happens — config does not

Per-role reasoning knobs (`{role}_thinking_mode`, `{role}_reasoning_effort`)
exist because different gateways honour one or the other. **Treat both as
requests, not guarantees, and never as an A/B variable.**

Measured 2026-07-27 (`tools/probe_reasoning.py`, 8 tiers × 2 `max_tokens`, one
gateway, `deepseek-v4-pro`): all 11 calls streamed 5.7k–7.2k reasoning chars —
including `thinking:{"type":"disabled"}`, `reasoning_effort: none`, and sending
neither — with no correlation to the tier and not one 504 (TTFB 1.6–76.7s, two
over 70s still succeeded).

Across production logs, reasoning appears on 0% of calls on 12 days and 47–69%
on two others. `yeban_guize` and `guize_guaitan` share a config yet differ 65
points. The gateway routes one model name to several upstreams and decides for
itself.

Consequences:
- `novel.py stats` prints measured per-role coverage (`_reasoning_coverage`).
  **Check that both arms match before trusting any `novel.py compare`.**
- Actually controlling reasoning requires a different endpoint/model.

Counter-case worth keeping (different failure, same knobs): `gemini-2.5-pro`
behind an nginx-fronted reseller gateway reasons past nginx's ~70s proxy timeout
and 504s before the first byte. Neither `thinking:{"type":"disabled"}` nor
omitting the param helped — only `writing_reasoning_effort: none` did (first byte
~29s, 3.1k chars in ~170s at `max_tokens: 65536`). That endpoint also 504s on any
non-stream request, so `api.stream: true` is mandatory for it.

---

## 2. The self-score has no discrimination — and 8.0 sits exactly on its median

`quality_threshold` is 8.0. The library's 1023 measured self-scores have a median
of **exactly 8.00** (34% below 8.0, only 6% below 6.5). So under
`rework_trigger: score`, ~1/3 of chapters enter the rework loop *by
construction* — while that same self-score spans only 7.4–8.7 across 1021
chapters with 6 rejections, i.e. no demonstrated discrimination.

The pipeline answers this noise with structural replans, which are the #1 rework
cause in every novel (33–68% of chapters).

**Do not "fix" a low FPY by moving `quality_threshold`** — that only relocates
the median problem. `rework_trigger: deterministic` exists to rework on measured
evidence instead.

Two load-bearing details of that mode:
- **`RevisionTracker.record()` cannot deliver it.** It reports `converged` only
  when `score >= threshold AND accepted`, and `accepted` is itself derived from
  `quality_threshold` (`review.py:1468`), so lowering the tracker's threshold can
  never release a 7.6 chapter. The release has to be an explicit
  `_rework_needed` break after each review round.
- **Ledger hygiene.** A 7.x chapter accepted in deterministic mode goes through
  `_accept_without_debt`: not stamped `force_accepted`, not written to
  `quality_debt.json` (it records a `quality_note` db_event instead). Otherwise
  `consecutive_force_accept_limit` × `circuit_breaker_score_floor` and refine
  prioritization would be swamped by noise-band chapters. `rework_score_floor`
  is aligned with `circuit_breaker_score_floor` for the same reason.

`tests/test_fix.py` asserts point-for-point equality between `score` mode and the
historical `score < quality_threshold or not accepted` expression over a
(score × accepted × gate_rejects) grid, so the default config cannot drift.

### The plan self-score has the same shape — and the engine already knows it (2026-07-28)

`min_plan_score` is also 8.0, and the plan self-score is the same kind of number:
426 archived attempt0 plans, **median exactly 8.00**, mean 7.899, and only **20
distinct values** — 90 plans at exactly 7.0, 103 at 8.5, i.e. a coarse 0.5-step
rating, not a measurement. 41.3% sit below 8.0.

The engine does **not** retry those. `planning.create_plan` retries only below
`plan_retry_score` (default `min_plan_score - 1.5` = 6.5, 4.0% of plans), accepting
the whole 6.5–8.0 band "to save tokens" unless `cost_savings_disabled` (the 收尾
zone) takes the shortcut away. Measured retry rate: **24 of 431 chapters**. So the
token-saving shortcut is the thing keeping the plan side off the median — which
means **do not "tighten" it**; that would recreate §2's defect one stage earlier.

And a retry buys nothing on the score it is judged by: excluding the 4 gate-driven
retries (attempt0 score absent), the 19 score-driven retries move the plan
self-score by **+0.07 mean** (deltas: 5×+0.5, 2×0.0, 2×−0.5, and a scatter of
±0.2–1.0). The chapters that got a retried plan score *worse* (6.66 vs 7.33 mean),
which is confounded — they were the troubled ones — but there is no direction in
which this rule pays.

**Fifth measurement pollution, in the tool that had just been hardened against the
fourth.** `replay_gates.plan_chain_block` read `plan_retry_score_threshold`, a key
that **does not exist**, so it fell back to `min_plan_score` and reported
`low_plan_score` for every plan under 8.0 — the tool was applying the very
threshold rule the engine avoids, and it also ignored `cost_savings_disabled`.
Fixed to mirror the engine (and the corrected combined gate-fix number is +6.2pt,
not +6.0pt: one more `tangshuting_e2e` chapter clears). `tests/test_latching_gates.py::ReplayPlanScoreFidelityTest`
pins the mid-band-accept, below-floor-block, and ending-zone cases so the tool
cannot silently disagree with the engine again. Lesson generalized: **a replay tool
must read the same config key the engine reads, and a `.get(k, default)` on a
misspelled key is indistinguishable from a working one** — that is why the tool
now has fidelity tests rather than only output tests.

---

## 3. Style collapse is the pipeline's biggest failure mode

Prose drifts into telegraphic em-dash fragments (`句子——状态——状态`) that the
model's own self-review happily rates 9+, because its voice has drifted with the
prose. Every deterministic prose gate exists because LLM self-assessment cannot
catch its own degeneration.

Recorded incidents:
- **v11 fossils.** `suspense_v11` carried 9–25 repeated signature clauses for 6
  straight chapters on advisory directives alone and never recovered. That is why
  `cross_chapter_repetition` has a `reject` level (`style_cross_repeat_reject_count`,
  8) that routes to STRUCTURAL replan with the concrete fossil clauses injected as
  hard avoid evidence — not another wording patch.
- **v11 Ch8 identical plan.** Short-novel mode used to disable the scene-dedupe
  retry, and a `max_sim=1.0` plan shipped. Hence `scene_dedupe_sim_identical`
  (0.97), an absolute ceiling that forces retry in EVERY mode.
- **Library-wide "instrument-report voice" collapse.** See the
  `overwriting_collapse_anchor` note: the three newer `style_health` checks were
  calibrated against this, plus a replay tool to validate thresholds offline.
- **The voice.md self-feeding loop.** `refresh_voice_anchors` used to re-derive
  voice from recent prose, so degraded prose became "the book's voice." It now
  anchors to a frozen `voice_baseline.md` and skips the refresh entirely when
  recent prose shows collapse (`voice_refresh_skip_penalty`).
- **Flattened exemplars.** `exemplar_block` used to flatten newlines, which was
  itself nudging the prose telegraphic. It now preserves paragraph indentation.

Rules that follow: do not relax `style_health` thresholds to make chapters pass;
do not let `cold_reader_review` share the `cacheable_prefix`; do not unfreeze
`voice_baseline.md`.

---

## 4. Gate archaeology — a silent gate is a bug report, not a deletion candidate

Zero-LLM tooling: `python tools/gate_census.py` (per-gate ran/fired/penalty over
archived reviews; measured total is 0.42 penalty per review over 651 reviews) and
`python tools/replay_l0.py` (replays L0 over finished chapters; 647 reviews → 29
chapters repaired, 0 made worse, all length changes within ±2%).

**Read the census's two columns separately.** `fire%` is a *verdict* (penalty,
level, block, or flagged spans); `advise%` is a directive with no verdict behind
it. Conflating them is not cosmetic: the gates do not share one result shape, and
counting any non-empty list as a firing reports `information_density` at 91% when
its actual verdict rate (`low_information`, ≥3 of 4 probes) is 6.8%. Conversely a
penalty-only test scores that same gate a structural 0/649 — it emits no penalty
at all — which reads as "dead gate" when it means "never measured."

Before deleting a silent gate, compare its threshold against the metric's
measured distribution:

- **`genre_adherence` — wiring bug, not a dead gate.** 0 firings in 215 runs
  because `review.py` called a `store.get_connection` that **does not exist**;
  the `AttributeError` was swallowed by a bare `except: pass`, so
  `recent_genre_scores` was always empty and `low_streak` could never exceed 1 —
  while tangshuting's own `chapter_metrics` holds negative-score streaks of up to
  8. Fixed to use the `conn` `review_chapter` is already handed.
- **…and fixing the wiring alone would have been worse than the bug.**
  `genre_score` is a signed keyword-density difference whose library-wide median
  is **exactly 0.000** (neither keyword list matched — no evidence), so the old
  `genre_drift_threshold: 0.3` scores "no evidence" as "drift": replayed over 357
  real scores it puts **46.8%** of chapters over the reject streak (86% in some
  novels), and a reject forces a STRUCTURAL replan. The threshold must sit
  strictly below zero; `-1.0` replays to warn 4.8% / reject 2.5%. The reject
  branch is additionally gated off by default
  (`genre_drift_reject_enabled: false`) because it has never executed in
  production.
- **`dialogue_pingpong` and `chapter_ending_quality` — deleted.** Thresholds were
  unreachable by construction (0.50 vs observed max qa_ratio **0.140**; 3 summary
  markers vs observed max **1**), neither had an entry in `fix.ACTION_BY_GATE`,
  and each duplicated a gate that does fire (`dialogue_health` at 34.8%; the
  `hook_strength`/`revise_hook_only` path).
- **`adjacent_repetition` — kept despite 0/641.** Its warn line (0.10 clause
  overlap) sits just above the observed max (0.090) rather than 3× above it, it
  feeds `_hard_block_reasons`, and the repo has a recorded true positive above
  its block line (suspense_v11 Ch3, overlap 0.73).

**Fossil rotation is bank-only on purpose.** Of 109 distinct fossil phrases
across 647 archived reviews, `FOSSIL_REPLACEMENTS` covers the generic-cliché
subset (`声音压得很低`, 49 firings); the rest are book-specific proper nouns
(`老市场街七号`, 42) that must never be rotated — swapping those is canon
corruption, not repair.

---

## 5. A/B methodology — short opening runs fabricate positive results

`ablate` always restarts at Ch1, and this repo's own recorded lesson is that
short opening runs fabricate positive results: score inflation on short chapters,
no mid-book problem zone. Two engine changes (lightweight planning, trimmed
context) were wrongly judged positive this way and later deleted.

Requirements for a trustworthy arm comparison:
1. **One variable.** Including `consecutive_force_accept_limit` (see below).
2. **Equal-length chapters, mid-book problem zone.** Prefer `novel.py fork
   <name> --as <new>` — it branches at HEAD, copying `memory/`, `chapters/`,
   `book.md`, `state.md`, `story_state.db` and the RAG index so both arms start
   byte-identical. It forks at HEAD **only**: memory markdown and the
   entity/thread tables describe the book as of its last written chapter and
   there is no faithful rollback to an earlier one.
3. **Matching reasoning coverage** in `novel.py stats` (§1).
4. **Prose read, not just counters** — `tools/pairwise_ab.py` is the
   anti-self-dealing half of the P4 criterion (REDESIGN §7): P4 *widens* the
   definition of "no rework," so FPY and call count improve mechanically whether
   or not the chapters got worse. It takes the chapters where arm A spent
   strictly more rework than arm B and asks a judge which is better, blind.

Operational traps:
- **`logs/` is deliberately not copied** by `fork` (except the RAG index, the one
  logs artifact carrying story content), so FPY / cost / reasoning coverage
  describe only the chapters the fork writes.
- **Budget with `target_words`, not `max_chapters`** — `max_chapters` switches on
  the ending-aware machinery and would make the tail unrepresentative. When the
  source already has `max_chapters`, `fork` extends that.
- **Check `consecutive_force_accept_limit` against the source's tail scores
  before launching.** The force-accept circuit breaker counts backwards through
  `chapter_metrics`, which the fork inherits, so a source whose last chapter
  scored below `circuit_breaker_score_floor` gives every fork a hair trigger:
  with the default limit of 2, one weak first chapter kills the run outright
  (`RuntimeError: Circuit breaker…`, seen killing an arm at Ch26 off a Ch25 of
  4.6). Raise the limit in **both** arms, identically, or you have added a second
  variable.
- **Windows launch.** `novel.py run` does not return until the pipeline child
  exits (the PowerShell `Start-Process` launcher blocks in `check_output`), so
  launch it as a background job and confirm with `grep -c "Start target_chars"`
  in `run.log` rather than waiting on the command. This venv is a **virtualenv**
  whose `python.exe` is a redirector stub, so every run appears twice in a
  process list (`stub → real interpreter`); that is one pipeline, not two, and
  `logs/run.pid` records the real one.

`compare.py` is calibrated against known ground truth: it must judge v10 over
v11.

---

## 6. `volume_plan.md` head truncation starved the mid-book

`volume_plan.md` is the one memory file that grows linearly with the book (a new
`## 第N卷` per volume, plus `extend_volume_schedule` APPENDING per-chapter
schedule tables), so plain head truncation silently starves the mid-book.

A novel at Ch41 saw only 第一卷（Ch1-24）and none of the current volume's
角色高光轮值表 / 爽点兑现节拍表 / 反转排期 / 伏笔兑现 tables. The schedules were
never executed: the ensemble cast collapsed to a two-hander, plus
`payoff_deferred` and `tension_flat`.

Fix: `memory.volume_plan_window(text, chapter_num, cap, lookahead)` replaces head
truncation at all three read sites (`volume_plan_window_enabled`). Visibility
alone doesn't produce compliance, so `memory.chapter_schedule_directive` quotes
this chapter's own schedule rows as a hard obligation into BOTH
`generate_candidate_plans` and `arbitrate_plan` (the arbiter must push the row
into `required_constraints`); gated by `chapter_schedule_directive_enabled`.

---

## 7. Rank-based exemplar selection, not absolute score thresholds

`retrieval.exemplar_block` quotes the book's own strongest chapters back to the
writer as style anchors. Selection is **rank-based** (top `exemplar_rag_top_k` by
score plus a small on-type bonus), NOT an absolute threshold: 1021 measured
chapters self-score 7.4–8.7, so the old `exemplar_rag_score_min: 8.8` selected
nothing.

The type bonus is deliberately small (0.3 payoff / 0.15 conflict) because an
on-type mediocre chapter is a worse model than an off-type excellent one. It
picks one dialogue-dense and one action/scene-dense exemplar (measured by
`_dialogue_ratio`) so the two are not redundant, and caps/caches the file reads.

---

## 8. Cost shape: planning is the budget, not writing

Audited 2026-07: **22.7 LLM calls per chapter**, **520:1 prompt amplification**,
planning ≈ **53%** of cost, FPY median 38%, and the per-chapter self-review has
no discriminative power (§2). `tools/prompt_census.py` re-measures this from
`novels/*/logs/llm_calls.jsonl` with zero LLM calls (14,976 calls / 991M prompt
chars at the time of writing) — and it refuted the intuition that "the writer's
context is too big": the writer is not where the budget goes.

This is the data floor under `REDESIGN.md`. The MVP inversion follows from it:
the default is cheap (`candidate_plans: 1`, `candidate_chapters: 1`) and breadth
is spent only on trouble.

### Where the `plan_candidate` prompt actually goes (2026-07-28)

`plan_candidate` is 33.7% of all prompt volume and its prompt is *larger* than
the writer's (median 131,872 vs 81,152), so it is the only entry point worth
attacking. Captured one real prompt at Ch201 (patch `planning.call_llm`, let the
first candidate raise, split the user message on `^##` headers):

| block | chars | share |
| --- | --- | --- |
| `fingerprint_block` 全书结构指纹库 | **22,813** | **19.6%** |
| `dedupe_block` 近期已用过的场景骨架 | 7,775 | 6.7% |
| `mem` sections (characters / metrics / voice / volume_plan …) | ~60,000 (capped) | ~51% |
| 132 sections under 300 chars each | 18,062 | 15.5% |
| exact duplicate sections (same header AND body twice) | 4,023 | 3.5% |

Two consequences, both counter-intuitive:

* **`mem` is capped, so shrinking anything inside it saves nothing.** Every novel
  config sets `plan_memory_chars: 60000`; at that budget only tier1 plus a
  truncated tier2 fit, and `memory_context` returns early — tier3/tier4 never
  reach this prompt at all. The §12 tier dedupe is real, but it lands on
  `extract` and other uncapped consumers, **not** on the largest prompt. Trim
  inside `mem` and the truncation point simply moves; the prompt stays 60k.
* **The saving has to come from the blocks `planning.py` adds after `mem`**, which
  is where `fingerprint_block` sat.

**`fingerprint_block` was the largest block in the largest prompt and carried no
signal.** It enumerated one line per completed chapter (`Ch7: enter_space→…`),
growing linearly with the book forever, to deliver what its own header asks for:
「特别是那些高频出现的流程组合」. Measured, whole-flow repetition does not exist —
tangshuting's 200 chapters hold **194 distinct flows** (3 flows repeat, covering
8 chapters); huangliang 97/100. Two hundred all-but-unique strings is noise by
construction.

The repetition is real one level down, in the 11-token move vocabulary:
`collect_evidence→deduce_conclusion` ×29, `enter_space→collect_evidence` ×26,
**91 of 106 bigrams recur ≥3 times**, 50 trigrams do. So the block now emits the
aggregate (recurring bigrams/trigrams + payoff/conflict/move frequencies) and is
O(1) in book length:

```
Ch201  22,701 -> 1,240 chars      whole plan prompt 116,592 -> 95,157  (-18.4%)
Ch50    5,710 ->   908
```

Nothing was lost that was ever used. The recent chapters are still quoted verbatim
by `narrative_pattern_block`, and plan-skeleton duplication is judged by
`scene_similarity`, which fires.

**Correction to the first version of this section and to commit `adfb545`'s
message:** both claimed the discriminating job was still done by
`check_plan_against_fingerprints` ("按候选确定性比对全部指纹"). That was wrong on two
counts, and grepping instead of asserting is what found it — the function was
referenced only by `quality.py` and its own tests, **never called in production**.
Replayed over 437 real chapters in 6 novels, its composite similarity peaked at
**0.448** against `fingerprint_warn_threshold: 0.65`:

```
tangshuting  n=199 median=0.273 p90=0.340 max=0.448   >=0.65: 0
huangliang   n= 99 median=0.278 p90=0.325 max=0.434   >=0.65: 0
tunshi_xitong n=51 median=0.310 p90=0.368 max=0.386   >=0.65: 0
p4_score     n= 49 median=0.224 p90=0.271 max=0.299   >=0.65: 0
yeban_guize  n= 24 median=0.230 p90=0.277 max=0.312   >=0.65: 0
guize_guaitan n=15 median=0.221 p90=0.298 max=0.308   >=0.65: 0
```

Unreachable by construction — the same defect as the deleted `dialogue_pingpong` /
`chapter_ending_quality` gates (§ the silent-gate rule). The only test that ever
saw it exceed 0.65 compared a plan **to itself**. So it was deleted, together with
its `fingerprint_warn_threshold` config key's last reader. `store_chapter_fingerprint`
still writes `skeleton_tokens` — that column is what made the replay above possible
offline, so keep it (`tests/test_pure_functions.py:ChapterFingerprintTests` now
guards the write path for exactly that reason).

The read path is held by `tests/test_fingerprint_context.py`, including a
20-vs-400-chapter size test, so the linear growth cannot come back.

`dedupe_block` was left alone deliberately: its fields must match
`_plan_skeleton_tokens` exactly or the generator is steered on one set of
dimensions while `scene_similarity` judges duplication on another — and that gate
does fire.

---

## 9. What the MVP refactor deleted, and when to bring it back

The 2026-07 refactor deleted modules that were premature for the current scale:
`reader_panel`, `rolling_plan`, `scene_breakdown`, craft/distill cross-book
learning consumption, simulate style profiles, pairwise judge. Telemetry's
consumption layers (distill → craft rules, cross-book bandit prior) went with
them.

Recovery point: git commit `9dd1ec0` and earlier. Bring them back when the
library reaches the scale where they beat noise (**≥5 finished books**), not
before. Telemetry itself stays: it is write-only today (logging +
`telemetry stats`), which is cheap and keeps the sample accumulating.

The 3 BeatCoverage test failures observed during that refactor are historical
leftovers, not regressions from it.

---

## 10. Credential incident: `config_template.yaml`

`config_template.yaml` is gitignored but **must exist on disk** for `novel.py
create`. It was *also tracked* until 2026-07-28, which made the ignore rule inert
(gitignore does not apply to already-tracked files) and left a live key one
`git commit -a` away from publication. `git rm --cached` fixed that. The key never
reached history — verified with `git log --all -S<key>` and a blob scan.

Standing rule: when adding config keys, edit **both** `config_template.yaml`
(your working copy, with keys) and the tracked credential-free
`config_template.example.yaml`, which is what `novel.py create` falls back to on
a fresh clone. `.gitignore` needs the `!config_template.example.yaml` negation
because the broad `config_*.yaml` rule would otherwise swallow it.

---

## 11. Arc planner (`arc.py`) — design notes

Default **off** (`arc_planning_enabled: false`). Motivation: `replan` is the #1
rework cause in every novel (33–68% of chapters), so the rework battleground is
planning, not writing. ONE high-reasoning call every `arc_span` (10) chapters
emits one **ChapterCard** per chapter, so 错峰兑现 / 场地轮换 / 开场轮换 /
整段推进 are decided once with whole-arc vision instead of being patched
chapter-by-chapter by gates that can only look backwards (REDESIGN L2).

Design constraints, each of which cost something to learn:

- **One seam only.** `planning.create_plan` calls `plan_from_arc` right after the
  resume-from-checkpoint block, and only when `checkpoint_label == "initial"` and
  `replan_feedback is None`. Every replan keeps the committee, so a bad card still
  has the old safety net. `pipeline.py` is untouched.
- **`card_to_plan` projects, it does not replace.** A card maps onto the existing
  plan schema (`wants→goal`, `blocked_by→conflict`, `where→location`,
  `exit_hook→hook`, …) so writing/review/quality/store need zero changes.
  Card-only fields (`opening_type`, `forbid`, `turn`) ride along in the plan dict,
  which `writing.py` dumps into the prompt wholesale.
- **`decision["scores"]` is deliberately empty.** There is no arbiter here, and a
  fake score would poison `chapter_metrics.plan_score` and the writer's quality
  contract. `plan_score()` returns 0.0 and `_prewrite_quality_contract` suppresses
  its 大纲仲裁分 line instead of printing `0.0/10`.
- **`plan_from_arc` never raises.** No card, bad JSON, or still-invalid card after
  one `repair_card` → return `None` and fall through to the committee.
- **Only CRITICAL forces repair.** `validate_card` (zero LLM, pre-write) checks
  empty required fields, `opening_type` equal to the previous chapter's, same
  `where` as the previous chapter, the same `payoff_type` three chapters running,
  scene similarity ≥ `scene_dedupe_sim_block`, plus CRITICAL continuity
  violations — and follows the committee's severity policy exactly
  (`pipeline._stage_plan`). Overdue threads and un-cashed setups are advisories
  appended to `required_constraints`. Treating advisories as repair triggers fires
  a repair on nearly every chapter (5 fired on the first live card) and eats the
  entire saving.
- **Deterministic windows.** Cards persist in `logs/arc_cards.json`; `arc_window`
  is anchored at Ch1 so it is a pure function of the chapter number and a
  resumed/forked run recomputes the same boundaries. Generation clips to
  `max(start, chapter_num)` so a run starting mid-block never plans
  already-written chapters.
- **Scene-dedupe must union both sources.** `_recent_selected_plans` reads
  `plan_arbitration` events, which the arc path never emits, so the arc arm has to
  union it with recent cards projected through `card_to_plan` or an all-arc run
  silently loses the check.

---

## 12. `memory_context`'s four tiers must not overlap

`recent_metrics`/`recent_events` return newest-first, so tier2's
`recent_metrics(5)` is a *prefix* of tier4's `recent_metrics(fatigue_window)` and
tier3's `recent_events(20)` is a prefix of tier4's `recent_events(40)`. Emitting
both in full shipped the same rows twice in every plan/extract prompt — measured
on a live Ch49 novel: **11,064 of 121,617 chars (9%)** inside `plan_candidate`,
which is the largest prompt the engine sends (median 131,872 chars over 2,501
library calls).

tier4 now emits only the older tail (`[5:]` / `[20:]`) under `## 更早的…JSON`
headers, and omits the section entirely when the tail is empty
(`fatigue_window <= 5`). Slicing is safe because the tiers assemble in order and
any short budget returns early, so tier4 can only appear when tier2/tier3 went in
whole. `tests/test_memory_tiers.py` holds the invariant in both directions — no
row twice, and no gap between the tiers.

---

## 13. Latching gates: a block conditioned on state the attempt cannot change

The single largest FPY′ defect class found so far, and the one to check FIRST when
a book's first-pass rate is low. Shape:

> a BLOCKING gate measures a property the current attempt has no way to move, so
> the block is unactionable, the forced retry provably fails, and every one of
> those retries is counted as a first-draft failure.

It is not a threshold-tuning problem — re-thresholding a latched gate just moves
where it latches. Three instances, all measured, plus the already-deleted
`fingerprint_warn_threshold` (§8) as prior art:

**`book_wide_fossils` hard rejects — cumulative book property charged to one
chapter.** `hard_fossils` routes to STRUCTURAL replan, and its input was
`frac = chapters_containing(phrase) / chapters_total`. Once 「声音压得很低」sat in
82/199 tangshuting chapters the numerator was frozen and the denominator grew by
1/chapter, so **274 more clean chapters** would have been needed to get back under
the line. The gate latched ON and rejected Ch95–Ch120 six consecutive times for a
phrase those chapters did not contain — punishing the writer for complying. Of 34
archived (chapter, phrase) hard flags across three novels, **22 were false**. Fix:
a fossil may only turn hard when the chapter under review actually uses it
(`in_current`). The phrases still ship as `phrases` + `directives` on every scan,
so avoidance pressure is unchanged; only the *verdict* narrowed.

Second defect in the same function: `book_fossil_hard_ratio` (0.20) was
**unreachable from below**. Candidacy already requires `frac >=
book_fossil_chapter_frac` (0.30) — and `>= min_ch/total`, itself ≥ 0.30 at every
book size — so every fossil was automatically hard and the two-tier design
collapsed into "reject on any fossil at all". It is now
`max(hard_ratio, candidacy_frac)`, so the key describes what the code does.

**`chapter_mode_monotony` — counting a genre label instead of a form.** The frac
was measured with the genre-BIASED classifier, which by design returns the
baseline label unless a chapter *clearly* breaks form. Under `baseline:
"reasoning"` (config.py's default for suspense) **38/41** tangshuting_e2e plans
and **13/13** guize_guaitan plans classify as `reasoning`, so frac ≥ 0.93 by
construction against a 0.80 block line — permanently on. It blocked 18
tangshuting_e2e chapters and **16 (89%) were still blocked after the forced
retry**, because re-rolling a plan cannot change the genre, while buying a re-roll
of the engine's most expensive call (~132k prompt). The frac now uses the unbiased
classifier (same books read 30/41 and 11/13 — monotony that is real, local, and
escapable); the biased label is still what gets reported and named in the
directive, because it is the better *description*, it just cannot be *counted*.

**`CONTRACT_SYSTEM`'s forced `ability_whitelist` — a red line invented at
bootstrap.** The extractor prompt demanded the core 金手指 appear in
`ability_whitelist` "even if the brief only strongly implies it… 宁可粗略也不能缺席".
tangshuting's brief is a 20.9k-char realistic 都市甜宠/美食探店 plan with **no
ability system at all** (`grep -c` for both invented names: 0). Forced to produce
one, bootstrap wrote 「错字食谱暗码解密」into contract.md and a *different*,
contradictory 「味觉共情」into bible.md. The whitelist is a per-chapter HARD
acceptance rule, so from Ch1 onward every scene where she tastes something was
`ability_out_of_scope` — **13 first-draft failures** across tangshuting +
tangshuting_v1_backup, none of them clearable by any rewrite. The prompt now
branches on whether the brief actually establishes a special ability and requires
an empty array for realistic briefs; `_contract_to_markdown` already omits the
whole section on an empty list, so the reviewer then has no clause to cite
(`tests/test_latching_gates.py` pins both halves).

That prompt fix stops the contract from inventing a red line, but not the *bible*
from inventing an ability — and those are two separate bugs with one symptom. A
census of every archived HARD contract violation (35 across 31 chapters) found
**30 of them in the tangshuting family alone** (11 tangshuting + 19 its
`_v1_backup` copy), 2 in an ablation fork, 2 in yeban_guize: `hard_contract` is
*not* the library-wide killer an earlier reading of FPY′ made it look like — it
is one book, one contradiction, counted once per derivative copy. The brief bans
the invented ability **in so many words** (`prompt.md:309`, 能力边界：没有神奇味觉，
不是天才厨师), and the contract extractor got that right (whitelist omits it,
blacklist says 不能凭空知道未亲自品尝的食物味道). The bible said the opposite
because `extract_contract` ran **last in `bootstrap()`** — after bible and
characters were already written and conditioned on the brief alone. So the one
artifact that had correctly parsed the author's red lines could not constrain the
two artifacts that declare abilities, and 味觉共情 became canon in the two files
the writer reads every chapter (11 + 22 mentions) while the reviewer measured
against the contract. No chapter can resolve that: obeying the contract means
contradicting characters.md.

Fix: extract the contract FIRST and inject it into the bible + characters calls as
a top-priority constraint (`_bootstrap_chain(contract_md=…)`). **Zero added LLM
calls** — the same call, moved earlier. Forward-only, as bootstrap runs once per
book; an empty/disabled contract leaves the chain exactly as it was.

A deterministic "bible declared an off-contract ability" detector was also built
and **rejected on measurement**: flagging bolded terms on ability-declaring lines
that appear nowhere in contract.md fires on **8/8 books**, surfacing 味觉共情×33
buried among character names (沈兰梅×280), volume headers (卷四×12) and generic
words (身份, 弱点, 动机). No discrimination ⇒ it would only have been one more
noisy advisory. Prefer constraining generation over detecting the mess afterwards
when the generator is the thing you control.

### Measuring a gate fix: `tools/replay_gates.py`

`fpy_prime` replays *archived payloads*, so a gate verdict baked into those
payloads is frozen and **a gate-logic change cannot show up in FPY′ at all** —
the tool keeps reporting the old verdict forever. `tools/replay_gates.py`
recomputes the changed gates from the primary data they read (chapter texts,
archived `plan_initial_attempt0_arbitration.json`, each novel's own config) and
re-runs `_hard_block_reasons` on the corrected payload, so the fix is settled by
the same ruler as everything else.

Two traps it exists to avoid, both of which produced a wrong answer first:

- **`scene_dedupe_retry` is NOT the scene-dedupe gate's event.** It is the generic
  `duplicate_blocked` retry marker shared by scene_similarity + narrative_pattern
  + chapter_mode (`planning.py:2061`), written only when a further attempt follows
  (`attempt < max_attempts - 1`). Counting it as a gate made chapter_mode look
  like it co-fired with a second independent blocker on 18/18 chapters — a
  faithful scene_similarity replay of those same plans reads **max_sim 0.041–0.068
  against a 0.82 block line**. Never infer causation from a replan event name.
- **The plan-gate chain is sequential with `continue`.** While chapter_mode was
  blocking, `visual_payoff` / `executability` / plan-score downstream of it never
  ran. Dropping a replan because one gate went quiet, without giving the rest
  their first chance to speak, fabricates a gain — so the replay re-runs the
  *whole* chain in engine order and reports a survivor histogram (7 survived:
  4 `low_plan_score`, 2 `chapter_mode`, 1 `visual_payoff`).

Measured, `--fix` isolating each arm over the 435 archived chapters of the six
non-derivative books (see "第三次测量污染" below — the first run of this table said
81.8% → 85.8% (+4.0) over 643 chapters, and was diluted by a 200-chapter copy):

| arm | library FPY′ | notes |
| --- | --- | --- |
| baseline | 363/435 = **83.4%** | 4 books < 80%. Was 359 = 82.5% before the em-dash re-stamp (see 「扁平的趋势罚分」 below) |
| A fossil `in_current` only | 376/435 = 86.4% (+3.0) | tangshuting 78.4→83.9, yeban 76.9→84.6 |
| B chapter_mode frac only | 375/435 = 86.2% (+2.8) | tangshuting_e2e 62.2→88.9 |
| A+B | 390/435 = 89.7% (+6.2) | e2e →93.3 (+31.1, super-additive) |
| C unmeasured plan score | 376/435 = 86.4% (+3.0) | **= B + 1 chapter**; C cannot be replayed without B |
| **A+B+C** | 391/435 = **89.9%** (+6.4) | guize_guaitan 69.2→76.9 is C's whole FPY′ effect |

Every row moved +4 chapters when `fpy_prime._normalize` learned the second frozen
semantic; the *deltas* are unchanged, because a normalization lifts both sides of
each arm equally. A row copied forward from an earlier run is how a stale 85.1%
survived here once already — re-run the whole table, never one row.

A+B is super-additive because both gates were blocking the *same* chapters: fixing
one leaves the chapter failing on the other, so neither arm alone can show the full
gain. **Isolate arms to attribute, but decide on the combined number.** C is the
opposite shape — worth 1 chapter on this metric and 37 LLM calls off it (see
「没测出来」被当成「测出来很差」 below); FPY′ counts only `plan_initial`, and 8 of
the 12 rounds C removes are `quality_replan`.

### 第三次测量污染：派生书目稀释了每一个「全库」数字

CLAUDE.md already records two instances of a measurement charging the thing it
measures (`pairwise_ab` billing an arm 10 calls for being judged; `compare.py`'s
circular score lines). This is the third, and it was found by trying to act on a
library aggregate: every top remaining killer pointed at the same book family, so
I claimed `tangshuting_v1_backup` was "a copy, counted twice". **It isn't a pure
copy** — only 76 of its 200 chapters are byte-identical to `tangshuting`. Measuring
that before designing the fix is what kept the fix honest.

What both replay tools were doing: `sorted(p.name for p in (ROOT/"novels").iterdir()
if (p/"logs"/"checkpoints").is_dir())` — one vote per *directory*. Three of the
eight dirs were derivatives, and they were 208 of 643 chapters (32%). Consequences,
all measured:

- the gate fixes above read **+4.0pt instead of +6.0pt** — the 200-chapter copy
  contributed +0.0pt and diluted the mean;
- `style_collapse` ranked as the #2 remaining first-draft killer at **24 misses**;
  **17 of those 24 were inside the excluded copy**, leaving 7 library-wide. It had
  already been written down as the next engine target on that basis;
- `hard_contract` read 27–31 and now reads 13.

Detection must be evidence, not a name guess (`_v1_backup` / `_e2e` are ad-hoc human
names; the next one will be spelled differently):

1. `__ablate_` in the dir name, or `experiments/{ablate,fork}_<name>.json` — the
   engine's OWN conventions, so they cannot drift out of sync with a rename.
2. **Ch1 byte-identical to another book's Ch1.** Two independent runs never produce
   the same first chapter even from the same brief; `tangshuting_e2e` is the control
   (same story concept, 46 chapters, different Ch1 → kept). Of a copy pair the longer
   book is canonical, ties alphabetical.

Two design decisions worth keeping: **`prompt.md` hashing was tried first and
rejected** — grouping by brief sounds obviously right, but all eight briefs hash
differently (tangshuting 56974 bytes vs `_v1_backup` 55277 vs `_e2e` 24020), because
the brief keeps being edited between runs. It would have grouped nothing. And an
ablation's Ch1 is NOT identical to its parent's (ablate restarts at Ch1 and
regenerates), so signal 1 is not redundant with signal 2 — each catches a case the
other misses. `discover_novels` prints every drop with its reason and takes `--all`,
per "No silent caps": a silently narrowed corpus reads exactly like a clean one.
Explicit names on the command line bypass the filter entirely, or an A/B could no
longer read its own forked arms. `tests/test_corpus_discovery.py` pins all of it.

### 声明了修复层 ≠ 修复真的能生效（2026-07-28）

`gate_rejects` 是 FPY′ 的头号剩余杀手（全库 20 次）。追到底，**每一本都只有一条**
入库短语，而且 11/12 是同一条 bank 短语 `声音压得很低`。它在写作时**已经**排在
writer avoid-list 的 rank 0（`tools/_fossil_probe.py` 逐章重算过 1..N-1 的
avoid-list 确认），所以这不是闸门锁死，是写手不合规——本章完全有办法把它变绿。

`quality.py` 用 `@REGISTRY.register(..., repair="L0")` 给这个门声明了确定性修复，
`fix.rotate_fossils` 也确实存在。但**声明不等于能生效**，三个缺陷叠在一起让这条修复
在真实数据上一次都没成功过：

1. **修复目标错了。** `rotate_fossils` 对所有短语共用 `fix_fossil_max_keep: 1`，而
   12 章里有 10 章**只出现一次**——`kept = min(1, 1)` ⇒ 一次替换都不做。而
   `book_wide_fossils` 的 hard reject 是**全书累计比率**，只能靠本章**零出现**转绿。
   密度门（`cross_chapter_repetition`/`descriptor_frequency`）保留 1 次是对的，比率门
   必须清零：现在 `rotate_fossils` 分两组、两个目标各跑一次 `fix_chapter`。
2. **缺语法护栏。** Ch195 的原文是「顾峥的声音压得很低」；bank 的头两个变体是动词短语，
   直接换进去会写出「顾峥的压着嗓子」。`keep-only-if-metric-improved` 抓不到这个——
   化石计数确实降了。护栏必须是**结构性**的：`fossil_fix._safe_alt` 在前一个字是
   「的/之」时只允许与原短语同中心词的变体，没有就宁可留着化石。
3. **修复跑在返工判决之后。** `_stage_fix` 排在 `_stage_quality_replan` 后面，所以为这个
   门声明的免费修复永远来不及阻止它触发的重做。新增 `pipeline._repair_fossil_rejects`
   插在 review 归档之后、返工判决之前，**先证后销**：只有当 reject 点名的短语在改写后
   的正文里确实一个都不剩，才把这条 reject 摘掉（对拿到的是缓存 review 的 resume 路径
   天然幂等）。归档的 `review_round{n}.json` 与 `style_health` 一律不动。

**实测收益要说实话。** 归档 29 章带化石 reject：11 章零 LLM 转绿（长度变化
−0.12%…+0.93%），16 章是 `current_chapter` 修复前的锁死残留（现在按 `stale` 单独记账，
不让闸门 bug 藏在修复里），2 章是书内专名（bank-only，正确地不动）。但
`_classify_replan_failure` 的结构性判定在 11 章里**只翻转了 1 章**——其余 10 章因为维度
分独立偏低，摘掉 reject 后照样走 structural。而且 `book_fossils` 的 `penalty` 全库都是
`None`，所以把化石提前轮换掉也**不会**抬分（这直接否掉了「把轮换挪到 review 之前」的
方案）。所以这次改动的价值是：**一条声明过但从未生效的修复终于能生效**、11 章入库正文
免费变干净（且不再把这条化石写进全书累计比率去污染后面每一章）、以及在
`rework_trigger: deterministic` 下少一个硬闸——**不是**省调用。FPY′ 不动也是对的：首稿
确实带着化石。

**治本的那一半：位置，不是措辞（2026-07-28）。** 两个闸门修完后全库剩 20 次首稿
`gate_rejects`，其中 **12 次是同一本书里同一条 bank 短语**。把每一章的 1..N-1 避免清单重新
扫一遍，这条短语在中段清单里**每次都排 rank 0**——写手看得见禁令，照用。这不是「禁令写得
不够狠」，是弱指令跟随，和能力白名单（v4 在 6 章里违约 5 章）、恢复指令（城中村段句长
23→30.6 仍照写）完全同一个失效模式，所以用同一个修法：`writing.fossil_tail_anchor` 在
prompt **末尾**把硬化石禁令复述一遍（hard-only、上限 5 条——长尾会稀释这个位置本身的价
值），`fossil_tail_anchor_enabled` 可关。这是**前向措施，归档上无法测**（同
`bd577ba` 的 bootstrap 顺序修复）：正文里出现过的化石不会因为改了 prompt 而消失。
`fix.rotate_fossils` 负责漏网的收尾，这一半负责别写进去。

**测量污染第四例：`--fix` 未知 token 被当成「什么都不修」。** `tools/replay_gates.py`
默认全开，我手敲 `--fix book_wide_fossils,chapter_mode_monotony`（当时只认 `A`/`B`），两个
token 都没匹配上，工具照样打了一份表头回显着这两个名字、结论写着 `+0.0pt` 的报告——而实
测是 **+6.2pt**（82.5% → 88.7%）。一个测量工具的默认行为绝不能是「静默降级成 no-op 再输出
一个看起来权威的数字」：现在描述性门名是合法别名，未知 token 直接 `return 2`。同类前三例
见本节与 §5（`pairwise_ab` 给被测臂记账、`compare` 的循环判据、`fpy_prime` 未归一化旧语义）。

### 「没测出来」被当成「测出来很差」：一份 JSON 残骸买走整轮规划（2026-07-28）

同一个缺陷类的第四例，但方向反过来：**不是门闸死，而是缺失的测量被读成了最低分。**

`planning.plan_score` 对空 `scores` 返回 `0.0`——这个契约本身是对的（`chapter_metrics.plan_score`
要的是 float，`arc.py` 也故意留空 `scores` 以免伪造分数污染指标）。问题是 `create_plan`
的重试闸把这个 0.0 当判决读：0.0 低于任何门槛，于是**一份解析失败的仲裁 JSON 会强制引擎多跑
一整轮规划**，而重掷候选根本治不了「仲裁没解析出来」。

根因在 `llm.load_json_with_repair`：修复调用的产物只要**能解析**就被接受，于是
`{"./output.json": "{"}` 这样的残骸被洗成了一份合法 decision。全库 934 次归档仲裁的实测：

| 形态 | 次数 |
| --- | --- |
| 不可用仲裁（`scores` 与 `merged_plan` 皆空/缺） | **16 / 934** |
| ——其中 guize_guaitan（共 34 次仲裁） | **14** |
| ——其他每本 ≤1（huangliang 1、tangshuting 1，其余 0） | 2 |
| 被键名/形态修复救回（离线，0 调用） | 1 |
| 需要重问仲裁 | 15 |

被弄坏的键名形态：`.selected_index`、`''`、`./output.json`、`./merged_plan.json`、
`./response.json`、`./scores`、`/assistant`。

两点必须写下来，因为它们各自纠正了我先前的一个错判：

- **键名修复单独救不回任何一例。** 14 份 `.selected_index` payload 旁边的
  `scores: []`、`merged_plan: {}`、`required_constraints: []` 全是空的——丢的是内容不是标签。
  键名修复照样保留（免费，且 `_decision_usable` 必须判内容而不是判拼写），但**真正的恢复手段是
  在仲裁处重问一次**。我一开始按「`merged_plan` 完好」写进了 docstring，是读数据才发现写错的。
- **唯一真能离线救回的形态是一行 `scores` 被提到了顶层**（tangshuting Ch175：
  `{index, score, pros, cons}`）——那就是测量本身，只是少了信封。补上信封后它的分是 5.5，
  低于 `plan_retry_score`，于是那一轮重规划变成**合法**的了：治本之后该书那一章不是缺陷而是差计划。

成本（按归档 checkpoint 里 attempt1 的调用产物逐个数，不是估的）：

```
12 轮被强制的额外规划   48 次调用  ->  11 次重问 + 1 次离线形态修复  =  11 次   省 37
其中 3 轮的 attempt1 本身也是残骸（guize 的评审端点问题），重问同样可能失败 ->
  标 decision["arbitration_failed"]，不再拿整轮规划当恢复手段
```

FPY′ 侧只 **+0.2pt（386→387）**，因为 FPY′ 按设计只计 `plan_initial`，而 12 轮里 8 轮是
`quality_replan`（在发布规则下游，被排除）。真正的收益是调用成本，加上一件账目问题：这 16 次里
仲裁的 `merged_plan` 与 `required_constraints` 是**静默丢失**的——章节按未仲裁的候选写出来、
没有任何硬约束，而日志里什么都没说。修法：`_normalize_decision` / `_decision_usable` /
`decision_has_score` 三个纯函数 + `arbitrate_plan` 里一次有界重问
（`arbiter_reask_enabled`）+ `create_plan` 在「压根没测量」时跳过按分重试并记
`plan_score_unavailable`。测量工具侧是 `tools/replay_gates.py --fix C`。

**`load_json_with_repair` 的普遍问题没有一并修**：它对所有调用点都是「能解析即接受」，
`screen_candidates` / `review_candidate_plans` 走的是同一条路。但除 guize_guaitan 外全库干净
（≤1 例/本），按「先测量再推广」的规矩，加一个通用 `expect_keys` 契约要等有第二本书出问题再说。

**`--fix C` 不能单独回放。** C 与 B 都靠重跑 plan-gate 链来结算，而链读的是**今天的**
`quality.py`，所以 B 的修复无法被工具关掉。`parse_fixes` 因此在请求 C 时显式补上 B **并打印
出来**——否则就是第四例测量污染（静默 no-op）的反向版本：把 B+C 的收益记在 C 名下。

### 扁平的趋势罚分：两个视角对同一个指标各收一次费，和差 0.1 就等于塌缩（2026-07-28）

把三个闸门修完后剩下的 12 次首稿失败里，`style_collapse` 有 7 次、`hard_contradictions`
有 5 次。**逐一读证据之后，两个桶的结论完全相反**，而这正是「先诊断再动手」的价值：

- **`hard_contradictions` 5 次全是真的**，而且分布在 5 本不同的书里，每本 1 次。证据具体到
  可执行：「第七块砖」被写成「第三块砖」、副官遗言与第 76 章刻字不符、转诊记录 2008-01-15
  被写成 01-14、林越左眼已失明却又用朱砂瞳看见了隐规则。这是一个**健康**的闸门——命中率
  1.1%、跨书均匀、每一条本章都完全有办法改绿。**不动它。**
- **`style_collapse` 7 次里 5 次是冻结的旧语义**，今天的引擎压根不会判它塌缩。

第二条的机理值得写清楚，因为它是本节缺陷类的又一个变体：**同一个指标被两个视角各收一次费，
两笔之和恰好压在阻断线上。** `style_health` 的破折号项有三段——静态档（≥6.0/千字 +1.0）、
趋势档（相对近章均值上升）、平台档（章与均值双双超阈 +1.0）。趋势档曾经是**扁平的 +1.0**，
而 `style_penalty_block` 正好是 **2.0**：于是一章只要密度刚过 6.0 且比自己的近章均值高
1.9 倍，静态 1.0 + 趋势 1.0 = 2.0，判定「文体塌缩」、`accepted=False`、返工。趋势档后来改成
按 ratio 分级（1.9x→0.3、2.1x→0.5、2.6x→0.8、≥3.0x→1.0），同一章今天只收 1.3。

全库归档实测（93 章有风格罚分）：

| | 章数 |
| --- | --- |
| 罚分在今天的算术下会变（全部**下降**，无一上升） | 31 |
| ——其中 **仅靠这一笔** 就跨过 2.0 阻断线的 | **5** |
| 真塌缩，修完仍然阻断 | 2 |

5 章是 tangshuting Ch6/9/41/70/142（重算 1.3–1.8）。留下的 2 章是真的：tangshuting Ch37
（2.5 = 静态 1.0 + 趋势 0.5 + `dialogue_starved` 1.0）与 yeban_guize Ch9（7.18/k vs 均值
2.20/k = 3.3x，本来就是分级后的判决）。

所以这是 CLAUDE.md 那条规矩的第二个实例——**回放必须归一化已经改过的引擎语义**，否则它把修好的
bug 报成活的问题。落地方式与合约 HARD→SOFT 那一半同形，`fpy_prime._normalize` 里加
`_restamp_style_penalty`，`--raw` 保留逐字回放。两个执行细节是必须的：

- **只重算破折号那几项**，其余分项（句长/断行/对话）的输入是正文，而 round0 的正文在修订之后
  已经不存在了；从归档总分里减掉旧的破折号项、加上新的，是唯一忠实的调整。
- **罚分已经触到 `style_penalty_cap` 的章直接跳过**——那个数不再是各分项之和，按项调整会算错。

**并且不许在工具里重写这套算术。** 把它从 `style_health` 里提成纯函数
`quality.em_dash_penalty(密度, 近章均值, config)`（零行为变化，`tests/test_style_overwriting.py`
逐点比对两者），回放工具调它。否则阈值一改，工具就开始和引擎静默不一致——就是
`ReplayPlanScoreFidelityTest` 记下的那个失效模式。

代价对比值得记：`style_collapse` 从「7 次活的首稿杀手」变成「2 次」，全库基线 FPY′
82.5% → 83.4%，三闸全开 89.0% → **89.9%**。如果不做这一步，下一步就会去「修」一个五分之三
是幻影的桶——那才是真正的浪费。

### Two fixes measured and rejected before writing code

Per "无效的删除" — a candidate that cannot show a gain offline gets deleted at the
design stage, not merged and watched:

- **Canon-membership fossil whitelist** (auto-whitelist n-grams that appear in
  bible/characters as proper nouns). Replayed payoff **≤2 chapters (+0.3pt)**, and
  it would have whitelisted the generic clause 「有什么东西在」. Rejected: the
  existing quoted-name whitelist already covers the real cases.
- **D2 containment** ("an ability-whitelist rule violated from Ch1 that is never
  once satisfied ⇒ downgrade to SOFT"). Principled and offline-replayable, but it
  is containment for exactly the case the `CONTRACT_SYSTEM` fix prevents, and
  adding a general gate-relaxation mechanism on one book's evidence is the
  over-engineering the design principles forbid. Rejected in favour of the root
  cause.

### Also found, not worth a fix

`writing.py:879-909` splits the cached `logs/book_fossils.json` into hard/soft at
`frac >= 0.20`. Across all 8 novels **every** cached fossil has `frac >= 0.31`, so
the `soft_fossils` branch is dead library-wide. Harmless (it is a prompt-assembly
branch, not a verdict) and it would come alive if `book_fossil_chapter_frac` were
ever lowered — left in place, recorded here so the next reader is not surprised by
a branch that never renders.
