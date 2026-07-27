# LESSONS.md — measured evidence behind the engine's design

Companion to `CLAUDE.md`. `CLAUDE.md` states the **rules**; this file holds the
**measurements and post-mortems** that produced them, so the rules can be
re-derived (or overturned) instead of cargo-culted.

Read the relevant section before you change a threshold, delete a gate, or run an
A/B. Every claim here has a date and a sample size; if you re-measure and get
something else, update the section rather than quietly acting on the new number.

Related docs: `REDESIGN.md` (the quality/FPY-oriented redesign roadmap and its
P1–P4 execution record), `README.md` (user-facing quickstart).

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
