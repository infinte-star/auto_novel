from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

from config import Paths, log, safe_score
from llm import call_llm, json_prompt, load_json_with_repair
from memory import memory_context, rhythm_diagnostics, structural_repetition_analysis
from store import JsonStoryStore, db_event, get_active_constraints, recent_quality_feedback

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

AGENT_REVIEW_SYSTEMS = {
    "world": """You are World Agent. Check world rules, power system, geography, institutions, and resource logic.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "character": """You are Character Agent. Check character goals, agency, relationships, secrets, trauma, and OOC drift.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "rhythm": """You are Rhythm Agent. Check pacing, compression/release cycle, scene variety, and reader fatigue.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "payoff": """You are Payoff Agent. Check emotional payoff, pressure-payoff ratio, hook strength, and novelty.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "foreshadowing": """You are Foreshadowing Agent. Check opened/advanced/recovered threads and long-term promises.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}.""",
    "reader": """You are Reader-Simulation Agent. Simulate a serial reader after this chapter plan.
Return exactly one valid JSON object and no other text. Schema: {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"follow_next_reason":"..."}.""",
}

def generate_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
) -> list[dict[str, Any]]:
    diagnostics = rhythm_diagnostics(conn, config)
    structural = structural_repetition_analysis(conn, config)
    constraints = get_active_constraints(conn, chapter_num)
    quality_feedback = recent_quality_feedback(paths)
    base_user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural, ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON (MUST REPAIR, DO NOT REPEAT)
{json.dumps(quality_feedback, ensure_ascii=False, indent=2) if quality_feedback else "None"}

## Active Stage Constraints (MUST OBEY)
{json.dumps(constraints, ensure_ascii=False, indent=2) if constraints else "None"}

## Previous Chapter Tail
{tail[-2000:]}

## Request
Create candidate plan for chapter {chapter_num}.
Avoid recent repetition. Preserve causal debt. Increase reader follow-up desire."""
    num_candidates = int(config["novel"]["candidate_plans"])
    max_workers = int(config["novel"].get("max_parallel_workers", 5))

    def gen_one(idx: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        for retry in range(2):
            try:
                raw = call_llm(
                    client,
                    paths,
                    config,
                    CANDIDATE_PLAN_SYSTEM,
                    json_prompt(base_user + f"\n\nCandidate index: {idx}. Use a distinct strategy."),
                    max_tokens=16000,
                    temperature=0.65 + idx * 0.05,
                )
                plan = load_json_with_repair(client, paths, config, raw)
                plan["candidate_index"] = idx
                return plan
            except Exception as exc:
                last_exc = exc
                log(paths, f"Candidate plan {idx} attempt failed retry={retry}: {exc}")
        log(paths, f"Candidate plan {idx} discarded after retries: {last_exc}")
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

def agent_review_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Candidate Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

Review chapter {chapter_num} plan."""
    max_workers = int(config["novel"].get("max_parallel_workers", 5))

    def review_one(agent: str, system: str) -> dict[str, Any]:
        for retry in range(2):
            try:
                raw = call_llm(client, paths, config, system, json_prompt(user), max_tokens=16000, temperature=0.2)
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
) -> list[list[dict[str, Any]]]:
    plan_users = []
    diagnostics_json = json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)
    memory = memory_context(paths, conn, config)
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

    def review_one(plan_index: int, agent: str, system: str) -> dict[str, Any]:
        user = plan_users[plan_index]
        for retry in range(2):
            try:
                raw = call_llm(client, paths, config, system, json_prompt(user), max_tokens=16000, temperature=0.2)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON (MUST REPAIR, DO NOT REPEAT)
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Candidate Plans JSON
{json.dumps(plans, ensure_ascii=False, indent=2)}

## Agent Reports JSON
{json.dumps(reports_by_plan, ensure_ascii=False, indent=2)}

Select and improve the best plan for chapter {chapter_num}."""
    raw = call_llm(client, paths, config, ARBITER_SYSTEM, json_prompt(user), max_tokens=8000, temperature=0.25)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = load_checkpoint(paths, chapter_num, f"plan_{checkpoint_label}_selected.json")
    if isinstance(cached, dict) and cached.get("plan") and cached.get("decision"):
        log(paths, f"Resuming cached {checkpoint_label} plan Ch{chapter_num}")
        return cached["plan"], cached["decision"]

    best_plan: dict[str, Any] | None = None
    best_decision: dict[str, Any] | None = None
    min_score = float(config["novel"]["min_plan_score"])
    for attempt in range(2):
        log(paths, f"Generating candidate plans Ch{chapter_num} attempt={attempt}")
        plans_key = f"plan_{checkpoint_label}_attempt{attempt}_candidates.json"
        reports_key = f"plan_{checkpoint_label}_attempt{attempt}_reports.json"
        arbitration_key = f"plan_{checkpoint_label}_attempt{attempt}_arbitration.json"

        plans = load_checkpoint(paths, chapter_num, plans_key)
        if isinstance(plans, list) and plans:
            log(paths, f"Resuming cached candidate plans Ch{chapter_num} attempt={attempt}")
        else:
            plans = generate_candidate_plans(client, paths, conn, config, chapter_num, tail)
            save_checkpoint(paths, chapter_num, plans_key, plans)

        reports = load_checkpoint(paths, chapter_num, reports_key)
        if isinstance(reports, list) and reports:
            log(paths, f"Resuming cached agent reports Ch{chapter_num} attempt={attempt}")
        else:
            reports = review_candidate_plans(client, paths, conn, config, chapter_num, plans)
            save_checkpoint(paths, chapter_num, reports_key, reports)

        arbitration = load_checkpoint(paths, chapter_num, arbitration_key)
        if isinstance(arbitration, dict) and arbitration.get("plan") and arbitration.get("decision"):
            log(paths, f"Resuming cached arbitration Ch{chapter_num} attempt={attempt}")
            plan = arbitration["plan"]
            decision = arbitration["decision"]
        else:
            plan, decision = arbitrate_plan(client, paths, conn, config, chapter_num, plans, reports)
            save_checkpoint(paths, chapter_num, arbitration_key, {"plan": plan, "decision": decision})

        score = plan_score(decision)
        log(paths, f"Arbiter selected Ch{chapter_num} plan score={score}")
        best_plan, best_decision = plan, decision
        if score >= min_score:
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
