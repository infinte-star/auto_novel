from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from config import Paths, append_text, chapter_path, log, normalize_text, read_text, safe_score, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import memory_context, rhythm_diagnostics, structural_repetition_analysis
from store import (
    db_event,
    get_active_constraints,
    get_open_causal_requirements,
    get_silent_threads,
    recent_metrics,
    recent_quality_feedback,
    store_stage_constraints,
)

if TYPE_CHECKING:
    from openai import OpenAI

REVIEW_SYSTEM = """You are a strict final editor for serialized Chinese web fiction.
Return exactly one valid JSON object and no other text:
{
  "score": 1-10,
  "accepted": true,
  "problems": [],
  "fixes": [],
  "continuity_risks": [],
  "rhythm_risks": [],
  "reader_fatigue_risks": [],
  "beats_audit": [{"beat":"...", "status":"realized|partial|absent", "evidence":"quote or note"}]
}
Scoring rules:
- Cap score at 8 if important selected-plan beats are missing from the chapter text.
- Cap score at 8 if a timeline, money movement, message route, surveillance source, or procedure is hand-waved.
- Cap score at 8 if the chapter repeats a recent scene shape or ending device without a clear new function.
- Cap score at 7 if continuity risks from recent reviews are ignored again.
- Cap score at 7 if more than 30% of plan beats are "partial" or "absent" in beats_audit.
- Cap score at 7 if a silent thread (>10 chapters without advancement) listed in the plan context could be advanced naturally but is ignored.
- Award 9+ only when the chapter solves prior feedback on page while preserving tension and follow-up desire.

Plan Beats Audit (REQUIRED):
For each beat in the plan's "beats" array, add an entry to beats_audit:
- "realized": the beat is fully executed on page with visible action
- "partial": the beat is referenced but lacks concrete scene or sensory detail
- "absent": the beat is missing or only implied off-page"""

STAGE_REVIEW_SYSTEM = """You are the long-cycle quality evaluator for serialized Chinese web fiction.
Return exactly one valid JSON object and no other text:
{
  "quality_trend": "summary of recent score and engagement trajectory",
  "continuity_risks": ["specific continuity issues spanning multiple chapters"],
  "rhythm_payoff_risks": ["pacing or pressure-payoff problems across the window"],
  "repetition_risks": ["repeated structures, payoffs, or staging"],
  "next_20_chapters_replan": ["concrete plan adjustments for the next 20 chapters"],
  "threads_to_recover_or_upgrade": ["open threads that need attention or elevation"],
  "constraints": [
    {"type": "avoid|require|replan|recover_thread", "description": "...", "priority": 1-10, "expires_in_chapters": 20}
  ]
}"""

REPLAN_SYSTEM = """You are the strategic replanner for a long-form fiction engine.
The current volume plan has degraded in quality metrics. Analyze the current state,
recent trajectory, open threads, and repetition patterns.
Produce a revised plan for the NEXT 40-60 chapters that:
- Resolves stale or overdue threads
- Introduces new conflict dimensions not seen in recent chapters
- Shifts character dynamics and power relationships
- Avoids patterns flagged in repetition analysis
- Maintains causal consistency with established events
- Increases reader anticipation and follow-up desire
Return the full revised volume_plan markdown only, no explanation."""

def review_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    chapter: str,
    tail: str,
    cached_memory: str | None = None,
) -> dict[str, Any]:
    mem = cached_memory or memory_context(paths, conn, config)
    silence_threshold = int(config["novel"].get("thread_silence_threshold", 10))
    silent_threads = get_silent_threads(conn, chapter_num, silence_threshold=silence_threshold)
    user = f"""## Memory
{mem}

## Previous Tail
{tail[-1500:]}

## Recent Quality Feedback JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Silent Threads JSON (silent >{silence_threshold} chapters; check whether the chapter advances any of these or has good reason to skip)
{json.dumps(silent_threads, ensure_ascii=False, indent=2) if silent_threads else "None"}

## Selected Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Chapter Text
{chapter[:12000]}"""
    raw = call_llm(client, paths, config, REVIEW_SYSTEM, json_prompt(user), max_tokens=65000, temperature=0.2)
    report = load_json_with_repair(
        client,
        paths,
        config,
        raw,
        fallback={
            "score": 5,
            "accepted": False,
            "problems": ["JSON parsing failed; conservative review fallback used."],
            "fixes": [],
            "continuity_risks": [],
            "rhythm_risks": [],
            "reader_fatigue_risks": [],
        },
    )
    report["score"] = safe_score(report.get("score", 0))
    report.setdefault("accepted", report["score"] >= float(config["novel"]["quality_threshold"]))
    return report

def stage_review(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> None:
    start = max(1, chapter_num - int(config["novel"]["stage_review_every"]) + 1)
    recent = []
    for num in range(start, chapter_num + 1):
        text = read_text(chapter_path(paths, num))
        if text:
            recent.append(f"## Ch{num}\n{text[:1600]}")
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural_repetition_analysis(conn, config), ensure_ascii=False, indent=2)}

## Recent Chapters
{chr(10).join(recent)}

Review long-cycle quality through chapter {chapter_num}."""
    raw = call_llm(client, paths, config, STAGE_REVIEW_SYSTEM, json_prompt(user), max_tokens=65000, temperature=0.3)
    data = load_json_with_repair(
        client,
        paths,
        config,
        raw,
        fallback={
            "quality_trend": "JSON parse failed; fallback used.",
            "continuity_risks": [],
            "rhythm_payoff_risks": [],
            "repetition_risks": [],
            "next_20_chapters_replan": [],
            "threads_to_recover_or_upgrade": [],
            "constraints": [],
        },
    )

    def render_section(title: str, content: Any) -> str:
        if isinstance(content, list):
            if not content:
                return f"## {title}\n_(none)_\n"
            return f"## {title}\n" + "\n".join(f"- {item}" for item in content) + "\n"
        return f"## {title}\n{content}\n"

    markdown = (
        render_section("Quality Trend", data.get("quality_trend", ""))
        + render_section("Continuity Risks", data.get("continuity_risks", []))
        + render_section("Rhythm and Payoff Risks", data.get("rhythm_payoff_risks", []))
        + render_section("Repetition Risks", data.get("repetition_risks", []))
        + render_section("Next 20 Chapters Replan", data.get("next_20_chapters_replan", []))
        + render_section("Threads to Recover or Upgrade", data.get("threads_to_recover_or_upgrade", []))
    )
    append_text(paths.logs_dir / "stage_reviews.md", f"\n\n# Ch{chapter_num} Stage Review\n\n{markdown}\n")
    db_event(conn, chapter_num, "stage_review", {"review": data})

    constraints = data.get("constraints") or []
    if constraints:
        store_stage_constraints(conn, chapter_num, constraints)
        log(paths, f"Stored {len(constraints)} stage constraints from Ch{chapter_num} review")

def should_replan(conn: Any, config: dict[str, Any]) -> bool:
    rows = recent_metrics(conn, 20)
    if len(rows) < 15:
        return False
    threshold_score = float(config["novel"].get("replan_score_threshold", 6.5))
    threshold_novelty = float(config["novel"].get("replan_novelty_threshold", 5.5))
    triggers = 0
    scores = [safe_score(r.get("score", 7)) for r in rows if r.get("score") is not None]
    novelties = [int(r.get("novelty", 7)) for r in rows if r.get("novelty") is not None]
    if scores and sum(scores) / len(scores) < threshold_score:
        triggers += 1
    if novelties and sum(novelties) / len(novelties) < threshold_novelty:
        triggers += 1
    structural = structural_repetition_analysis(conn, config)
    if len(structural.get("warnings", [])) >= 3:
        triggers += 1
    return triggers >= 2

def adaptive_replan(
    client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any], chapter_num: int
) -> None:
    shutil.copy2(paths.volume_plan, paths.volume_plan.with_suffix(".md.bak"))
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Rhythm Diagnostics JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## Structural Repetition Analysis JSON
{json.dumps(structural_repetition_analysis(conn, config), ensure_ascii=False, indent=2)}

## Open Causal Requirements JSON
{json.dumps(get_open_causal_requirements(conn), ensure_ascii=False, indent=2)}

## Active Constraints JSON
{json.dumps(get_active_constraints(conn, chapter_num), ensure_ascii=False, indent=2)}

Current chapter: {chapter_num}. Replan the next 40-60 chapters."""
    new_plan = call_llm(client, paths, config, REPLAN_SYSTEM, user, max_tokens=65000, temperature=0.5)
    write_text(paths.volume_plan, normalize_text(new_plan) + "\n")
    db_event(conn, chapter_num, "adaptive_replan", {"reason": "metrics_degradation"})
