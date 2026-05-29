from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from config import Paths, append_text, chapter_path, log, normalize_text, read_text, safe_score, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, memory_context, rhythm_diagnostics, structural_repetition_analysis, writing_memory_context
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
  "hook_strength": 1-10,
  "beats_audit": [{"beat":"...", "status":"realized|partial|absent", "evidence":"quote or note"}],
  "patches": [
    {"op":"replace", "locator":"quote 8-20 chars from current text", "before":"exact substring to replace", "after":"replacement text", "reason":"why"},
    {"op":"insert_after", "locator":"quote 8-20 chars after which to insert", "insert":"new text", "reason":"why"},
    {"op":"delete", "locator":"quote 8-20 chars identifying the segment", "before":"exact substring to delete", "reason":"why"}
  ],
  "writer_directives_for_next_chapter": [
    "3-6 imperative directives the NEXT chapter writer must follow",
    "Each directive must be concrete execution guidance, not abstract advice",
    "Examples: '下一章必须用反转结构，最近 3 章都是 pressure-payoff', '户部官僚程序需要落到至少一段对话上', '主角必须在场景 2 做一次有可见代价的选择'"
  ]
}

## Scoring philosophy (CHANGED — DO NOT artificially cap at 8)
The score must be a HONEST quality estimate on a 1-10 scale. Use the full range.
A chapter that fully realizes its plan, is causally sound, varied in shape, and gives the reader fresh follow-up
desire should score 9 or 10. Caps below were historically used as hard ceilings; they are now SOFT penalties
that DEDUCT from a base score, but a strong chapter can still earn 9+.

Start from a base score reflecting raw craft (writing quality, scene specificity, dialogue, emotional payoff).
Then DEDUCT according to the following soft penalties (do NOT clip to 8; just subtract):
- Missing important plan beats: -1.0 per fully absent beat; -0.5 per partial beat.
- Hand-waved timeline/money/route/procedure: -1.0 per occurrence.
- Repeats recent scene shape or ending device without new function: -1.0.
- Ignores continuity risks called out in recent reviews: -1.0 per ignored risk (cap total at -2.0).
- More than 30% of plan beats partial/absent: additional -0.5 (on top of per-beat deductions).
- Silent thread (>10 chapters silent) listed in plan context could be advanced but ignored: -0.7.

After deductions, apply bonuses (additive, max +1.5 total):
- All plan beats realized with concrete on-page action: +0.5
- Solves prior feedback on page while preserving tension and follow-up desire: +0.7
- Distinct scene staging and ending device vs. last 3 chapters: +0.3
- Visible cost / agency moment for protagonist with emotional texture: +0.3

Clamp final score to [1.0, 10.0]. Score 9.0+ is reserved for chapters with no critical deductions.

Plan Beats Audit (REQUIRED):
For each beat in the plan's "beats" array, add an entry to beats_audit:
- "realized": the beat is fully executed on page with visible action
- "partial": the beat is referenced but lacks concrete scene or sensory detail
- "absent": the beat is missing or only implied off-page

Patches (REQUIRED when score < 9 OR there are any "partial"/"absent" beats):
- Output 1-8 surgical patches that, when applied, would lift the chapter at least one band.
- Each patch's insert/after content MUST be SHORT (<= 200 Chinese chars) and SELF-CONTAINED.
- Prefer insert_after for adding missing scenes/details; prefer replace for fixing concrete wording.
- Each patch's locator/before MUST quote text that exists verbatim in the chapter (a contiguous substring, 8-20 chars).
- Each patch must be INDEPENDENT — applying any subset (or all) in any order must still produce valid prose.
- Do NOT chain dependent patches; if you need a long insertion, split it into multiple independent inserts at different locators.
- If the chapter is already at 9+ with no partial/absent beats, you MAY return "patches": [].

Writer directives (REQUIRED): output 3-6 imperative directives the NEXT chapter writer must follow.
- Be execution-level concrete (a specific scene type, structural choice, or character action), not abstract.
- Each directive should be a single short Chinese sentence.
- Prefer directives that REPAIR specific problems found in THIS chapter or compensate for recent repetition.

Hook strength (REQUIRED): rate the chapter's ENDING-hook strength 1-10 independently.
- 9-10: ending raises a sharp, specific question the reader will click "next" for.
- 6-8: workable hook but generic or already used in recent chapters.
- <=5: weak/summary-style ending — DO NOT use vague "他知道，一切才刚刚开始" style endings."""

STAGE_REVIEW_SYSTEM = """You are the long-cycle quality evaluator for serialized Chinese web fiction.
Return exactly one valid JSON object and no other text:
{
  "quality_trend": "summary of recent score and engagement trajectory",
  "continuity_risks": ["specific continuity issues spanning multiple chapters"],
  "rhythm_payoff_risks": ["pacing or pressure-payoff problems across the window"],
  "repetition_risks": ["repeated structures, payoffs, or staging"],
  "next_20_chapters_replan": ["concrete plan adjustments for the next 20 chapters"],
  "threads_to_recover_or_upgrade": ["open threads that need attention or elevation"],
  "writer_directives_for_next_chapter": ["3-6 imperative concrete directives the immediate next chapter writer must follow"],
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

VOICE_ANCHOR_SYSTEM = """You maintain the narrative voice anchor for a long-form serialized novel.
You will receive: the current voice.md, the previous N chapters of actual prose, and a brief on what
shifted in the storyline. Your job: produce an updated voice.md that PRESERVES at least 70% of
existing constraints while INCORPORATING 1-3 new style features that the recent prose has stabilized
on (e.g. new recurring imagery, new sentence-rhythm habits, new character speech patterns).

Rules:
- Output the FULL replacement voice.md in Chinese, markdown only.
- Do not weaken existing forbidden-patterns list; you may add to it.
- Keep sections: 时态/视角, 句长节奏, 词汇调性, 感官锚, 心境呈现, 章节结构惯例, 节奏禁忌.
- Add a brief change log at the bottom: `## 修订日志\\n- Ch{chapter_num}: <one-line summary>`."""

VOICES_TABLE_SYSTEM = """You maintain the character voice table for a long-form novel.
You will receive: the current voices.md plus recent chapters featuring named characters.
Update the voices.md to:
- Refine each existing character's voice fingerprint based on what actually appeared in recent prose.
- Add 1-2 NEW named characters who appeared in recent chapters and lack an entry.
- Keep all existing characters; refine rather than delete.
Output the full updated voices.md in Chinese, markdown only. Same section structure as input."""

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
    mem = cached_memory or writing_memory_context(paths, conn, config)
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
    raw = call_llm(
        client, paths, config, REVIEW_SYSTEM, json_prompt(user),
        max_tokens=32000, temperature=0.2, cacheable_prefix=cacheable_prefix(paths, config),
    )
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
            "hook_strength": 6,
            "writer_directives_for_next_chapter": [],
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
    raw = call_llm(client, paths, config, STAGE_REVIEW_SYSTEM, json_prompt(user), max_tokens=12000, temperature=0.3)
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
            "writer_directives_for_next_chapter": [],
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
        + render_section("Writer Directives For Next Chapter", data.get("writer_directives_for_next_chapter", []))
    )
    append_text(paths.logs_dir / "stage_reviews.md", f"\n\n# Ch{chapter_num} Stage Review\n\n{markdown}\n")
    db_event(conn, chapter_num, "stage_review", {"review": data})

    # Persist stage-level writer_directives onto the most-recent chapter's
    # final_review.json so the NEXT chapter writer prompt picks them up via
    # writer_directives_for_chapter(). We append to the existing list (chapter
    # review directives take precedence — we only add if the stage layer has
    # surfaced something not already listed).
    stage_directives = data.get("writer_directives_for_next_chapter") or []
    if stage_directives:
        try:
            from checkpoint import load_checkpoint as _load, save_checkpoint as _save
            existing = _load(paths, chapter_num, "final_review.json")
            if isinstance(existing, dict):
                merged = list(existing.get("writer_directives_for_next_chapter") or [])
                for d in stage_directives:
                    s = str(d).strip()
                    if s and s not in merged:
                        merged.append(s)
                existing["writer_directives_for_next_chapter"] = merged[:10]
                _save(paths, chapter_num, "final_review.json", existing)
                log(paths, f"Merged {len(stage_directives)} stage directives into Ch{chapter_num} review")
        except Exception as exc:
            log(paths, f"Failed to merge stage directives into Ch{chapter_num} review: {exc}")

    constraints = data.get("constraints") or []
    if constraints:
        store_stage_constraints(conn, chapter_num, constraints)
        log(paths, f"Stored {len(constraints)} stage constraints from Ch{chapter_num} review")

    # Refresh narrative voice anchors using the recent prose window.
    try:
        refresh_voice_anchors(client, paths, conn, config, chapter_num, recent_text="\n\n".join(recent))
    except Exception as exc:
        log(paths, f"Voice anchor refresh failed at Ch{chapter_num}: {exc}")


def refresh_voice_anchors(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    recent_text: str,
) -> None:
    """Update memory/voice.md and memory/voices.md based on actual recent prose.

    Called from stage_review (every N chapters). Each refresh is best-effort:
    if the LLM call fails, the existing files remain unchanged.
    """
    if not recent_text.strip():
        return

    current_voice = read_text(paths.voice)
    voice_user = f"""## Current voice.md
{current_voice if current_voice.strip() else "(empty — generate from prose)"}

## Recent Chapters Prose
{recent_text[:18000]}

Refresh voice.md for Ch{chapter_num}."""
    new_voice = call_llm(client, paths, config, VOICE_ANCHOR_SYSTEM, voice_user, max_tokens=8000, temperature=0.3)
    new_voice = normalize_text(new_voice).strip()
    if new_voice:
        write_text(paths.voice, new_voice + "\n")
        log(paths, f"Updated voice.md at Ch{chapter_num} (len={len(new_voice)})")

    current_voices = read_text(paths.voices)
    voices_user = f"""## Current voices.md
{current_voices if current_voices.strip() else "(empty — generate from prose)"}

## Recent Chapters Prose
{recent_text[:18000]}

Refresh voices.md for Ch{chapter_num}."""
    new_voices = call_llm(client, paths, config, VOICES_TABLE_SYSTEM, voices_user, max_tokens=8000, temperature=0.3)
    new_voices = normalize_text(new_voices).strip()
    if new_voices:
        write_text(paths.voices, new_voices + "\n")
        log(paths, f"Updated voices.md at Ch{chapter_num} (len={len(new_voices)})")

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
    new_plan = call_llm(client, paths, config, REPLAN_SYSTEM, user, max_tokens=16000, temperature=0.5)
    write_text(paths.volume_plan, normalize_text(new_plan) + "\n")
    db_event(conn, chapter_num, "adaptive_replan", {"reason": "metrics_degradation"})
