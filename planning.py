from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint, save_checkpoint
from config import Paths, log, safe_score
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, lite_memory_context, memory_context, rhythm_diagnostics, structural_repetition_analysis
from store import JsonStoryStore, db_event, get_active_constraints, get_silent_threads, recent_quality_feedback

if TYPE_CHECKING:
    from openai import OpenAI

CANDIDATE_PLAN_SYSTEM = """You are a chapter-planning agent in an industrial long-form fiction engine.
Return exactly one valid JSON object and no other text. Create one candidate plan for the requested chapter.
Schema:
{
  "title": "...",
  "goal": "...",
  "conflict": "...",
  "conflict_type": "court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
  "payoff": "...",
  "payoff_type": "court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
  "pressure": "what suppresses the protagonist/readers before payoff",
  "beats": ["5-9 concrete beats"],
  "character_focus": ["characters who get agency or emotional movement"],
  "world_state_changes": ["state changes if this chapter happens"],
  "thread_actions": ["foreshadowing opened/advanced/recovered"],
  "hook": "chapter-end reader question",
  "risk": "main continuity or repetition risk"
}
The chapter must advance long-term causality, not merely create local excitement.
Every plan must:
- Convert at least one stale review problem into a concrete on-page scene.
- Include causal bridges for travel time, message delivery, money movement, and surveillance if they matter.
- Specify visible actions, sensory anchors, and dialogue pressure, not only analysis or summary.
- Avoid reusing the recent chapter-ending device, analysis posture, or emotional beat."""

SCREEN_SYSTEM = """You are the fast screening layer for a long-form fiction engine.
You receive multiple candidate chapter plans. Rank them by overall quality considering:
- Causal coherence and continuity with established state
- Novelty relative to recent chapters (avoid repetition)
- Reader anticipation and hook strength
- Character agency and cost visibility
- Thread advancement and payoff freshness

Return exactly one valid JSON object and no other text:
{"ranking": [{"index": 0, "brief": "one-line reason"}, ...]}
Order from best to worst. Be decisive — ties waste downstream resources."""

ARBITER_SYSTEM = """You are the arbitration layer for a long-form fiction engine.
Evaluate candidate plans against global state, recent metrics, repetition risk, causal value, character consistency,
payoff freshness, and reader anticipation.
Return exactly one valid JSON object and no other text:
{
  "selected_index": 0,
  "scores": [{"index": 0, "score": 1-10, "pros": [], "cons": []}],
  "merged_plan": {same schema as candidate, improved if needed},
  "required_constraints": ["hard constraints the writer must obey"],
  "reader_expectation_delta": "why this improves or hurts follow-up desire"
}
Reject or downgrade plans that keep known review problems abstract, rely on off-page resolution,
repeat the same physical staging, or contain unresolved timeline/logistics gaps. Improve the merged plan
so the writer has concrete scene obligations rather than vague intentions."""

FUSED_PLAN_REVIEW_SYSTEM = """You are the multi-axis plan reviewer for a Chinese historical/fantasy web novel.
Evaluate the candidate plan along 6 INDEPENDENT axes. Do NOT let a strong axis lift a weak one — each axis is scored on its own.

## Scoring philosophy (CHANGED — use the FULL 1-10 range, do not artificially cap at 8)
Each axis score must reflect HONEST quality on a 1-10 scale. The "score_caps_triggered" field below is now informational —
it records which soft penalties fire, but does NOT clip the axis score to a ceiling. Reviewers historically defaulted to 7-8
across the board, suppressing useful signal. A plan with strong execution on its strongest axes can earn 9 or 10 on those axes
even if other axes are weak.

Use this rubric per axis:
- 10: exemplary, no risks, novel angle, clear long-range payoff
- 9: very strong, only minor cosmetic concerns
- 8: solid, one or two specific fixable issues
- 7: workable, needs targeted revision on a known weakness
- 6: significant concern that could damage reader experience
- <=5: structural problem requiring redesign

When a "score cap" condition below is met, record it in score_caps_triggered AND DEDUCT a soft penalty
from the axis score (typically -1.0 to -1.5 per trigger), but do not clamp the result.

Axes and what they check:

1. world — Geography, travel time (京城到江南需数日), power system canon, institution/title accuracy, resource conservation, calendar/season consistency.
   Soft penalties: -1.5 if geography/travel violated; -2.0 if power system contradicted; -2.5 if institutional procedures anachronistic.

2. character — Each character acts on goals (not plot convenience); has agency with visible cost; uses only knowledge they possess; growth is incremental; dialogue voice fits status.
   Soft penalties: -2.0 if a character acts on info they shouldn't have; -1.0 if protagonist has no meaningful choice/cost; -1.5 if any character is out-of-character without justification.

3. rhythm — Variety vs. recent chapters in opening/closing device, scene count (≥2 distinct scenes), compression-then-release, balanced action/dialogue/reflection.
   Soft penalties: -1.0 if ending device repeats the previous chapter; -1.5 if the chapter is one extended scene; -1.0 if pacing is monotone.

4. payoff — Pressure→payoff causality; payoff_type fresh vs last 3 chapters; visible cost; earned resolution (no coincidence/deus-ex-machina); distinct emotional texture.
   Soft penalties: -2.0 if payoff is coincidence/luck; -1.0 if payoff_type repeats prior 2 chapters; -1.0 if there's no visible cost.

5. foreshadowing — Advances at least one open thread; recovers dropped threads when natural; new threads have realistic due_chapter; total active threads ≤ 8.
   Soft penalties: -1.0 if overdue threads (>20 chapters) ignored; -1.5 if no thread advanced/recovered; -1.0 if opening a 9th+ concurrent thread.

6. reader — A serial reader finishing the chapter has clear next-chapter questions, gets at least one satisfaction moment, isn't lost if 2 chapters were skipped, isn't asked to juggle too many threads, has at least one empathy moment.
   Soft penalties: -1.0 if no clear "next" hook; -1.5 if pure setup with zero payoff moments; -1.0 if reader must recall >5 prior plot points.

Return exactly one valid JSON object and no other text:
{
  "axes": {
    "world":         {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "character":     {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "rhythm":        {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "payoff":        {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "foreshadowing": {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "reader":        {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[],"follow_next_reason":"..."}
  },
  "overall_score": 1-10,
  "merged_required_fixes": []
}

Rules:
- score_caps_triggered records which soft-penalty conditions matched, for diagnostics. Apply the deductions but do NOT clamp.
- overall_score = average of the 6 axis scores, rounded to the nearest 0.5.
- Be decisive — vague risks waste downstream tokens. Each axis's risks/required_fixes must be specific and actionable."""

AGENT_REVIEW_SYSTEMS = {
    "world": """You are World Agent for a Chinese historical/fantasy web novel.
Check the chapter plan against established world rules. Specifically verify:
1. Geography & travel: distances, routes, travel time consistency (京城到江南需数日, not instant)
2. Power system: cultivation/combat/political power rules match established canon
3. Institutions: official titles, bureaucratic procedures, hierarchy are period-appropriate
4. Resources: money, materials, troops obey conservation (no unexplained refills)
5. Calendar & seasons: dates align with established timeline, seasonal details consistent

Use the full 1-10 range. Strong plans can score 9+. Apply soft penalties (deduct, do not clamp):
- -1.5 if geography/travel time is violated
- -2.0 if power system contradicts established rules
- -2.5 if institutional procedures are anachronistic or impossible

Return exactly one valid JSON object and no other text.
Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "character": """You are Character Agent for a Chinese historical/fantasy web novel.
Check the chapter plan for character consistency and growth. Specifically verify:
1. Goals & motivations: each character acts from established goals, not plot convenience
2. Agency: characters make active choices with visible cost, not passive observers
3. Relationships: interactions reflect established dynamics (allies, enemies, debts)
4. Secrets & knowledge: characters only act on information they actually possess
5. Growth arc: protagonist shows incremental change, not sudden personality shifts
6. Dialogue voice: each character's speech pattern matches their background and status

Use the full 1-10 range. Strong plans can score 9+. Apply soft penalties (deduct, do not clamp):
- -2.0 if a character acts on information they shouldn't have
- -1.0 if protagonist has no meaningful choice or cost in this chapter
- -1.5 if a character behaves out-of-character without justification

Return exactly one valid JSON object and no other text.
Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "rhythm": """You are Rhythm Agent for a Chinese historical/fantasy web novel.
Check pacing and structural variety against recent chapters. Specifically verify:
1. Scene structure: does this chapter use a different opening/closing device than the last 3?
2. Compression/release: is there both tension buildup AND a release moment?
3. Scene count & variety: at least 2 distinct scenes with different settings or dynamics
4. Ending device: not the same type as the previous 2 chapters (cliffhanger/revelation/quiet)
5. Information density: balanced between action, dialogue, and reflection (no 1000+ char monologues)

Use the full 1-10 range. Strong plans can score 9+. Apply soft penalties (deduct, do not clamp):
- -1.0 if chapter ending repeats the same device as the previous chapter
- -1.5 if the entire chapter is a single extended scene with no shift
- -1.0 if pacing is monotone (all high tension or all low tension)

Return exactly one valid JSON object and no other text.
Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "payoff": """You are Payoff Agent for a Chinese historical/fantasy web novel.
Check emotional payoff quality and pressure-payoff balance. Specifically verify:
1. Pressure buildup: is there meaningful resistance/obstacle before the payoff?
2. Payoff freshness: is the payoff_type different from the last 3 chapters?
3. Cost visibility: does the payoff come with visible cost or trade-off?
4. Earned resolution: is the resolution causally earned (not coincidence or deus ex machina)?
5. Emotional texture: does the chapter evoke a distinct emotion, not generic tension?

Use the full 1-10 range. Strong plans can score 9+. Apply soft penalties (deduct, do not clamp):
- -2.0 if payoff relies on coincidence or unexplained luck
- -1.0 if payoff_type repeats the same as the previous 2 chapters
- -1.0 if there's no visible cost or trade-off for the protagonist

Return exactly one valid JSON object and no other text.
Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "foreshadowing": """You are Foreshadowing Agent for a Chinese historical/fantasy web novel.
Check thread management and long-term promise fulfillment. Specifically verify:
1. Overdue threads: flag any open thread introduced >15 chapters ago that isn't advanced here
2. Thread advancement: does this chapter advance at least one existing thread?
3. New thread introduction: if opening a new thread, is the due_chapter realistic?
4. Recovery opportunities: are there dropped threads that could be naturally recovered here?
5. Promise density: not too many open threads (>8 active = reader confusion risk)

Use the full 1-10 range. Strong plans can score 9+. Apply soft penalties (deduct, do not clamp):
- -1.0 if there are overdue threads (>20 chapters) that could be addressed but aren't
- -1.5 if no existing thread is advanced or recovered
- -1.0 if opening a 9th+ concurrent thread without closing one

Return exactly one valid JSON object and no other text.
Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "reader": """You are Reader-Simulation Agent for a Chinese historical/fantasy web novel.
Simulate a serial reader finishing this chapter plan. Specifically evaluate:
1. Follow-next desire: after this chapter, what 3 questions would make the reader click "next"?
2. Satisfaction: does the chapter deliver at least one moment of satisfaction (not all setup)?
3. Confusion risk: would a reader who skipped 2 chapters still follow the main thread?
4. Fatigue signals: is the reader being asked to track too many simultaneous threads?
5. Emotional hook: is there a character moment that creates empathy or investment?

Use the full 1-10 range. Strong plans can score 9+. Apply soft penalties (deduct, do not clamp):
- -1.0 if there's no clear "next chapter" question for the reader
- -1.5 if the chapter is pure setup with zero payoff moments
- -1.0 if the reader needs to remember >5 prior plot points to understand this chapter

Return exactly one valid JSON object and no other text.
Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"follow_next_reason":"..."}""",
}

def _carried_over_risks_from_prev(paths: Paths, chapter_num: int) -> list[str]:
    """Extract continuity/rhythm/fatigue risks from the previous chapter's final review.

    Returns a deduplicated list of risk strings that the next plan should explicitly address.
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    risks: list[str] = []
    for key in ("final_review.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        for field in ("continuity_risks", "rhythm_risks", "reader_fatigue_risks", "problems"):
            for item in data.get(field, []) or []:
                text = str(item).strip()
                if text and text not in risks:
                    risks.append(text)
        if risks:
            break
    return risks[:8]


def _strategy_history(conn: Any, lookback: int = 60) -> dict[str, dict[str, float]]:
    """Aggregate per-strategy stats from past plan_arbitration events.

    Returns {strategy_name: {"trials": N, "score_sum": X, "wins": K}}.
    "wins" counts how often a candidate with that strategy was the
    arbiter-selected one.
    """
    if isinstance(conn, JsonStoryStore):
        events = conn.recent_events(lookback)
    else:
        try:
            rows = conn.execute(
                "SELECT payload FROM events WHERE event_type='plan_arbitration' "
                "ORDER BY id DESC LIMIT ?",
                (lookback,),
            ).fetchall()
            events = [{"payload": json.loads(r["payload"])} for r in rows]
        except Exception:
            return {}
    stats: dict[str, dict[str, float]] = {}
    for ev in events:
        payload = ev.get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        # plan_arbitration payload shape: {"decision": {...}, "plans": [...]}
        decision = payload.get("decision") or {}
        plans = payload.get("plans") or []
        if not plans:
            continue
        sel_idx = int(decision.get("selected_index", 0))
        scores = decision.get("scores") or []
        score_map = {int(s.get("index", -1)): safe_score(s.get("score", 0)) for s in scores}
        for i, plan in enumerate(plans):
            if not isinstance(plan, dict):
                continue
            strat = str(plan.get("strategy") or "").strip()
            if not strat:
                continue
            entry = stats.setdefault(strat, {"trials": 0.0, "score_sum": 0.0, "wins": 0.0})
            entry["trials"] += 1
            entry["score_sum"] += float(score_map.get(i, 5.0))
            if i == sel_idx:
                entry["wins"] += 1
    return stats


def _select_strategies_bandit(
    conn: Any,
    config: dict[str, Any],
    strategies: list[tuple[str, str]],
    n: int,
    paths: Paths,
) -> list[tuple[str, str]]:
    """Epsilon-greedy selection of n strategies from the candidate pool.

    Score per strategy = mean(score) + 0.5 * win_rate. Strategies with
    fewer than 3 trials are treated as "exploration" and always included
    in the pool. Picks top-n by composite score with ε probability of a
    random swap to keep exploring.
    """
    import random as _random

    bandit_enabled = bool(config["novel"].get("strategy_bandit", True))
    if not bandit_enabled or n <= 0:
        return [strategies[i % len(strategies)] for i in range(n)]

    lookback = int(config["novel"].get("strategy_bandit_lookback", 60))
    epsilon = float(config["novel"].get("strategy_bandit_epsilon", 0.2))
    stats = _strategy_history(conn, lookback=lookback)

    scored: list[tuple[float, int, tuple[str, str]]] = []
    for idx, strat in enumerate(strategies):
        name = strat[0]
        s = stats.get(name)
        if not s or s["trials"] < 3:
            # Boost under-explored strategies so they get picked sometimes.
            composite = 9.0 + _random.random() * 0.5
        else:
            mean_score = s["score_sum"] / s["trials"]
            win_rate = s["wins"] / s["trials"]
            composite = mean_score + 0.5 * win_rate
        scored.append((composite, idx, strat))

    # Sort by composite desc, stable on original idx.
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [item[2] for item in scored[:n]]

    # With probability epsilon, swap one of the picked with a random un-picked.
    if epsilon > 0 and len(strategies) > n and _random.random() < epsilon:
        picked_names = {p[0] for p in picked}
        leftovers = [s for s in strategies if s[0] not in picked_names]
        if leftovers:
            swap_in = _random.choice(leftovers)
            swap_out_idx = _random.randrange(len(picked))
            picked[swap_out_idx] = swap_in

    try:
        log(paths, f"Strategy bandit picked: {[p[0] for p in picked]}")
    except Exception:
        pass
    return picked


def generate_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
    cached_memory: str | None = None,
) -> list[dict[str, Any]]:
    diagnostics = rhythm_diagnostics(conn, config)
    structural = structural_repetition_analysis(conn, config)
    constraints = get_active_constraints(conn, chapter_num)
    quality_feedback = recent_quality_feedback(paths)
    silence_threshold = int(config["novel"].get("thread_silence_threshold", 10))
    silent_threads = get_silent_threads(conn, chapter_num, silence_threshold=silence_threshold)
    carried_over_risks = _carried_over_risks_from_prev(paths, chapter_num)
    mem = cached_memory or memory_context(paths, conn, config)
    base_user = f"""## Memory
{mem}

## Rhythm Diagnostics JSON
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural, ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON (MUST REPAIR, DO NOT REPEAT)
{json.dumps(quality_feedback, ensure_ascii=False, indent=2) if quality_feedback else "None"}

## Active Stage Constraints (MUST OBEY)
{json.dumps(constraints, ensure_ascii=False, indent=2) if constraints else "None"}

## Silent Threads (HARD REQUIREMENT: advance at least one of these on page if narratively viable)
{json.dumps(silent_threads, ensure_ascii=False, indent=2) if silent_threads else "None"}

## Carried-over Risks from Ch{chapter_num - 1} (MUST address at least 2 of these on page)
{json.dumps(carried_over_risks, ensure_ascii=False, indent=2) if carried_over_risks else "None"}

## Previous Chapter Tail
{tail[-2000:]}

## Request
Create candidate plan for chapter {chapter_num}.
Avoid recent repetition. Preserve causal debt. Increase reader follow-up desire.
If silent threads exist above, the plan MUST either advance one in beats/thread_actions, or explicitly note in "risk" why none are viable this chapter."""
    num_candidates = int(config["novel"]["candidate_plans"])
    max_workers = int(config["novel"].get("max_parallel_workers", 5))

    # Explicit differentiation strategies — each candidate is told to attack the
    # chapter from a distinct angle so the arbiter sees a real choice, not 5
    # near-identical variants.
    candidate_strategies = [
        ("scene-driven",
         "以单个高密度场景为核心：物理空间高度具象，让冲突在一个房间/一段路途/一桌对峙中爆发；最少3次场地切换。"),
        ("character-driven",
         "以主角或核心配角的内心两难为核心：本章的胜负来自角色的关键选择与可见代价；选择必须在 beats 里明示。"),
        ("thread-driven",
         "以推进 2 条以上 open thread 为核心：必须在 thread_actions 显式列出推进的具体 thread id 与下一步具体动作。"),
        ("institutional",
         "以制度/程序/官僚摩擦为核心：本章必须呈现一次具体的衙门程序（如送文、批红、查证、回禀），用程序细节制造张力。"),
        ("reversal",
         "以认知反转为核心：本章某个既定事实被推翻，原信任的来源被证伪，导致主角不得不修正策略；hook 必须基于反转。"),
        ("pressure-payoff",
         "以挤压-释放节奏为核心：前 60% 持续施压（资源/时间/信任三轴中至少 2 轴），后 40% 给一个小而可信的释放点。"),
    ]

    # Strategy bandit: pick which strategies to use this chapter based on
    # historical plan_score in the agent_reports/plan_arbitration events table.
    # Falls back to round-robin when there's not enough data.
    chosen_strategies = _select_strategies_bandit(
        conn, config, candidate_strategies, num_candidates, paths,
    )

    def gen_one(idx: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        strategy_name, strategy_desc = chosen_strategies[idx % len(chosen_strategies)]
        strategy_block = (
            f"\n\n## Candidate Strategy (MANDATORY)\n"
            f"Candidate index: {idx}\n"
            f"Strategy: {strategy_name}\n"
            f"Definition: {strategy_desc}\n"
            f"You MUST design this candidate plan around this strategy. "
            f"Other candidates use different strategies — do not converge."
        )
        for retry in range(2):
            try:
                raw = call_llm(
                    client,
                    paths,
                    config,
                    CANDIDATE_PLAN_SYSTEM,
                    json_prompt(base_user + strategy_block),
                    max_tokens=16000,
                    temperature=0.65 + idx * 0.05,
                    cacheable_prefix=cacheable_prefix(paths, config),
                )
                plan = load_json_with_repair(client, paths, config, raw)
                plan["candidate_index"] = idx
                plan["strategy"] = strategy_name
                return plan
            except Exception as exc:
                last_exc = exc
                log(paths, f"Candidate plan {idx} ({strategy_name}) attempt failed retry={retry}: {exc}")
        log(paths, f"Candidate plan {idx} ({strategy_name}) discarded after retries: {last_exc}")
        return {}

    plans: list[dict[str, Any]] = [{}] * num_candidates
    with ThreadPoolExecutor(max_workers=min(max_workers, num_candidates)) as executor:
        futures = {executor.submit(gen_one, idx): idx for idx in range(num_candidates)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                plans[idx] = future.result()
            except Exception as exc:
                log(paths, f"Candidate plan {idx} thread failed: {exc}")
                plans[idx] = {}
    valid = [p for p in plans if p]
    if not valid:
        raise RuntimeError(f"All {num_candidates} candidate plans failed for chapter")
    return valid

def screen_candidates(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    top_n: int = 2,
    cached_memory: str | None = None,
) -> list[int]:
    if len(plans) <= top_n:
        return list(range(len(plans)))
    mem = lite_memory_context(paths, conn, config)
    user = f"""## Memory (abbreviated)
{mem}

## Candidate Plans JSON
{json.dumps(plans, ensure_ascii=False, indent=2)}

Rank all {len(plans)} candidates for chapter {chapter_num}."""
    raw = call_llm(
        client, paths, config, SCREEN_SYSTEM, json_prompt(user),
        max_tokens=12000, temperature=0.2, cacheable_prefix=cacheable_prefix(paths, config),
    )
    result = load_json_with_repair(
        client, paths, config, raw, fallback={"ranking": [{"index": i} for i in range(len(plans))]}
    )
    ranking = result.get("ranking", [])
    indices = []
    for entry in ranking:
        idx = int(entry.get("index", 0))
        if 0 <= idx < len(plans) and idx not in indices:
            indices.append(idx)
    if len(indices) < top_n:
        for i in range(len(plans)):
            if i not in indices:
                indices.append(i)
            if len(indices) >= top_n:
                break
    return indices[:top_n]


def _explode_fused_axes(fused: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a fused-review JSON object into the legacy per-agent reports list.

    Downstream code (arbitrate_plan, agent_reports table, plan_score) expects a
    list of {"agent","score","risks","required_fixes","state_patch"} dicts. This
    expands the fused {"axes":{...}} payload into that shape.
    """
    axes = fused.get("axes") or {}
    reports: list[dict[str, Any]] = []
    for axis_name in ("world", "character", "rhythm", "payoff", "foreshadowing", "reader"):
        axis = axes.get(axis_name) or {}
        report = {
            "agent": axis_name,
            "score": safe_score(axis.get("score", 5)),
            "risks": axis.get("risks") or [],
            "required_fixes": axis.get("required_fixes") or [],
            "state_patch": axis.get("state_patch") or [],
            "score_caps_triggered": axis.get("score_caps_triggered") or [],
        }
        if axis_name == "reader" and axis.get("follow_next_reason"):
            report["follow_next_reason"] = axis["follow_next_reason"]
        reports.append(report)
    return reports


def _fused_review_one_plan(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    user: str,
    plan_index_for_log: int | None = None,
) -> list[dict[str, Any]]:
    """Run one fused plan-review call and return 6 legacy-format reports."""
    fallback_axes = {
        name: {"score": 5, "risks": [], "required_fixes": [], "state_patch": [], "score_caps_triggered": []}
        for name in ("world", "character", "rhythm", "payoff", "foreshadowing", "reader")
    }
    fallback = {"axes": fallback_axes, "overall_score": 5, "merged_required_fixes": []}
    for retry in range(2):
        try:
            raw = call_llm(
                client,
                paths,
                config,
                FUSED_PLAN_REVIEW_SYSTEM,
                json_prompt(user),
                max_tokens=12000,
                temperature=0.2,
                cacheable_prefix=cacheable_prefix(paths, config),
            )
            fused = load_json_with_repair(client, paths, config, raw, fallback=fallback)
            if not isinstance(fused.get("axes"), dict):
                raise ValueError("fused review missing 'axes' dict")
            return _explode_fused_axes(fused)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            tag = f"plan={plan_index_for_log}" if plan_index_for_log is not None else "single"
            log(paths, f"Fused plan review parse failed {tag} retry={retry}: {exc}")
    return _explode_fused_axes(fallback)


def agent_review_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    user = f"""## Memory
{lite_memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Candidate Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

Review chapter {chapter_num} plan."""

    fused_enabled = bool(config["novel"].get("fused_plan_review", True))
    if fused_enabled:
        reports = _fused_review_one_plan(client, paths, config, user)
    else:
        max_workers = int(config["novel"].get("max_parallel_workers", 5))

        def review_one(agent: str, system: str) -> dict[str, Any]:
            for retry in range(2):
                try:
                    raw = call_llm(
                        client, paths, config, system, json_prompt(user),
                        max_tokens=12000, temperature=0.2,
                        cacheable_prefix=cacheable_prefix(paths, config),
                    )
                    report = load_json_with_repair(
                        client,
                        paths,
                        config,
                        raw,
                        fallback={"score": 5, "risks": [], "required_fixes": [], "state_patch": []},
                    )
                    report["agent"] = agent
                    return report
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    log(paths, f"Agent {agent} review parse failed retry={retry}: {exc}")
            return {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}

        reports: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(review_one, agent, system): agent
                for agent, system in AGENT_REVIEW_SYSTEMS.items()
            }
            for future in as_completed(futures):
                reports.append(future.result())

    for report in reports:
        agent = report["agent"]
        if isinstance(conn, JsonStoryStore):
            conn.add_agent_report(chapter_num, agent, report)
        else:
            conn.execute(
                "INSERT INTO agent_reports(chapter, agent, score, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    chapter_num,
                    agent,
                    safe_score(report.get("score", 0)),
                    json.dumps(report, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    if not isinstance(conn, JsonStoryStore):
        conn.commit()
    return reports

def review_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    cached_memory: str | None = None,
) -> list[list[dict[str, Any]]]:
    plan_users = []
    diagnostics_json = json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)
    memory = lite_memory_context(paths, conn, config)
    for plan in plans:
        plan_users.append(
            f"""## Memory
{memory}

## Rhythm Diagnostics JSON
{diagnostics_json}

## Candidate Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

Review chapter {chapter_num} plan."""
        )

    max_workers = int(config["novel"].get("max_parallel_workers", 5))
    reports_by_plan: list[list[dict[str, Any]]] = [[] for _ in plans]
    fused_enabled = bool(config["novel"].get("fused_plan_review", True))

    if fused_enabled:
        # One fused LLM call per candidate plan; expands to 6 axis reports each.
        def fused_one(plan_index: int) -> tuple[int, list[dict[str, Any]]]:
            return plan_index, _fused_review_one_plan(
                client, paths, config, plan_users[plan_index], plan_index_for_log=plan_index
            )

        with ThreadPoolExecutor(max_workers=min(max_workers, len(plans))) as executor:
            futures = {executor.submit(fused_one, i): i for i in range(len(plans))}
            for future in as_completed(futures):
                plan_index = futures[future]
                try:
                    _, reports = future.result()
                    reports_by_plan[plan_index] = reports
                except Exception as exc:
                    log(paths, f"Fused review thread failed plan={plan_index}: {exc}")
                    reports_by_plan[plan_index] = _explode_fused_axes(
                        {"axes": {
                            name: {"score": 5, "risks": [], "required_fixes": [], "state_patch": [], "score_caps_triggered": []}
                            for name in ("world", "character", "rhythm", "payoff", "foreshadowing", "reader")
                        }}
                    )
    else:
        def review_one(plan_index: int, agent: str, system: str) -> dict[str, Any]:
            user = plan_users[plan_index]
            for retry in range(2):
                try:
                    raw = call_llm(
                        client, paths, config, system, json_prompt(user),
                        max_tokens=12000, temperature=0.2,
                        cacheable_prefix=cacheable_prefix(paths, config),
                    )
                    report = load_json_with_repair(
                        client,
                        paths,
                        config,
                        raw,
                        fallback={"score": 5, "risks": [], "required_fixes": [], "state_patch": []},
                    )
                    report["agent"] = agent
                    return report
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    log(paths, f"Agent {agent} review parse failed plan={plan_index} retry={retry}: {exc}")
            return {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(review_one, plan_index, agent, system): (plan_index, agent)
                for plan_index in range(len(plans))
                for agent, system in AGENT_REVIEW_SYSTEMS.items()
            }
            for future in as_completed(futures):
                plan_index, agent = futures[future]
                try:
                    reports_by_plan[plan_index].append(future.result())
                except Exception as exc:
                    log(paths, f"Agent {agent} review thread failed plan={plan_index}: {exc}")
                    reports_by_plan[plan_index].append(
                        {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}
                    )

    for reports in reports_by_plan:
        for report in reports:
            agent = report["agent"]
            if isinstance(conn, JsonStoryStore):
                conn.add_agent_report(chapter_num, agent, report)
            else:
                conn.execute(
                    "INSERT INTO agent_reports(chapter, agent, score, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        chapter_num,
                        agent,
                        safe_score(report.get("score", 0)),
                        json.dumps(report, ensure_ascii=False),
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
    if not isinstance(conn, JsonStoryStore):
        conn.commit()

    return reports_by_plan

def arbitrate_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    reports_by_plan: list[list[dict[str, Any]]],
    cached_memory: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mem = cached_memory or memory_context(paths, conn, config)
    user = f"""## Memory
{mem}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON (MUST REPAIR, DO NOT REPEAT)
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Candidate Plans JSON
{json.dumps(plans, ensure_ascii=False, indent=2)}

## Agent Reports JSON
{json.dumps(reports_by_plan, ensure_ascii=False, indent=2)}

Select and improve the best plan for chapter {chapter_num}."""
    raw = call_llm(
        client, paths, config, ARBITER_SYSTEM, json_prompt(user),
        max_tokens=12000, temperature=0.25, cacheable_prefix=cacheable_prefix(paths, config),
    )
    decision = load_json_with_repair(client, paths, config, raw)
    plan = decision.get("merged_plan") or plans[int(decision.get("selected_index", 0))]
    db_event(conn, chapter_num, "plan_arbitration", {"decision": decision, "plans": plans})
    return plan, decision

def plan_score(decision: dict[str, Any], selected_index: int | None = None) -> float:
    scores = decision.get("scores") or []
    if not scores:
        return 0.0
    if selected_index is None:
        selected_index = int(decision.get("selected_index", 0))
    for score in scores:
        if int(score.get("index", -1)) == selected_index:
            return safe_score(score.get("score", 0))
    return safe_score(scores[0].get("score", 0))

def create_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
    checkpoint_label: str = "initial",
    cached_memory: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = load_checkpoint(paths, chapter_num, f"plan_{checkpoint_label}_selected.json")
    if isinstance(cached, dict) and cached.get("plan") and cached.get("decision"):
        log(paths, f"Resuming cached {checkpoint_label} plan Ch{chapter_num}")
        return cached["plan"], cached["decision"]

    mem = cached_memory or memory_context(paths, conn, config)
    best_plan: dict[str, Any] | None = None
    best_decision: dict[str, Any] | None = None
    best_score = -1.0
    min_score = float(config["novel"]["min_plan_score"])
    retry_score = float(config["novel"].get("plan_retry_score", min_score - 1.5))
    max_attempts = int(config["novel"].get("plan_max_attempts", 2))
    for attempt in range(max_attempts):
        log(paths, f"Generating candidate plans Ch{chapter_num} attempt={attempt}")
        plans_key = f"plan_{checkpoint_label}_attempt{attempt}_candidates.json"
        reports_key = f"plan_{checkpoint_label}_attempt{attempt}_reports.json"
        arbitration_key = f"plan_{checkpoint_label}_attempt{attempt}_arbitration.json"

        plans = load_checkpoint(paths, chapter_num, plans_key)
        if isinstance(plans, list) and plans:
            log(paths, f"Resuming cached candidate plans Ch{chapter_num} attempt={attempt}")
        else:
            plans = generate_candidate_plans(client, paths, conn, config, chapter_num, tail, cached_memory=mem)
            save_checkpoint(paths, chapter_num, plans_key, plans)

        screen_key = f"plan_{checkpoint_label}_attempt{attempt}_screen.json"
        cached_screen = load_checkpoint(paths, chapter_num, screen_key)
        skip_screen = bool(config["novel"].get("plan_skip_screen", False))
        if isinstance(cached_screen, list) and cached_screen:
            top_indices = cached_screen
            log(paths, f"Resuming cached screening Ch{chapter_num} attempt={attempt} top={top_indices}")
        elif skip_screen:
            top_indices = list(range(len(plans)))
            save_checkpoint(paths, chapter_num, screen_key, top_indices)
            log(paths, f"Skipping screen Ch{chapter_num}: all {len(plans)} candidates go to agent review")
        else:
            top_indices = screen_candidates(client, paths, conn, config, chapter_num, plans, cached_memory=mem)
            save_checkpoint(paths, chapter_num, screen_key, top_indices)
            log(paths, f"Screened Ch{chapter_num} candidates: top={top_indices} from {len(plans)}")

        screened_plans = [plans[i] for i in top_indices if i < len(plans)]

        reports = load_checkpoint(paths, chapter_num, reports_key)
        if isinstance(reports, list) and reports:
            log(paths, f"Resuming cached agent reports Ch{chapter_num} attempt={attempt}")
        else:
            reports = review_candidate_plans(client, paths, conn, config, chapter_num, screened_plans, cached_memory=mem)
            save_checkpoint(paths, chapter_num, reports_key, reports)

        arbitration = load_checkpoint(paths, chapter_num, arbitration_key)
        if isinstance(arbitration, dict) and arbitration.get("plan") and arbitration.get("decision"):
            log(paths, f"Resuming cached arbitration Ch{chapter_num} attempt={attempt}")
            plan = arbitration["plan"]
            decision = arbitration["decision"]
        else:
            plan, decision = arbitrate_plan(client, paths, conn, config, chapter_num, screened_plans, reports, cached_memory=mem)
            save_checkpoint(paths, chapter_num, arbitration_key, {"plan": plan, "decision": decision})

        score = plan_score(decision)
        log(paths, f"Arbiter selected Ch{chapter_num} plan score={score}")
        if score > best_score:
            best_plan, best_decision, best_score = plan, decision, score
        if score >= min_score:
            break
        if score >= retry_score:
            log(
                paths,
                f"Ch{chapter_num} plan score={score} below min={min_score} but above retry_threshold={retry_score}; "
                f"accepting without retry to save tokens.",
            )
            break
        db_event(conn, chapter_num, "low_plan_score_retry", {"score": score, "decision": decision})
    assert best_plan is not None and best_decision is not None
    save_checkpoint(
        paths,
        chapter_num,
        f"plan_{checkpoint_label}_selected.json",
        {"plan": best_plan, "decision": best_decision},
    )
    return best_plan, best_decision
