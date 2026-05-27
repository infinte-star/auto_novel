from __future__ import annotations

import json
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

from config import Paths, append_text, chapter_path, count_chars, normalize_chapter, normalize_text, read_text, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import memory_context
from planning import plan_score
from store import JsonStoryStore, db_event, recent_quality_feedback, store_causal_links

if TYPE_CHECKING:
    from openai import OpenAI

WRITE_SYSTEM = """You are a professional Chinese long-form web novel author.
Write the chapter in Chinese.
Requirements:
- Around {chapter_words} Chinese characters.
- Start exactly with: 第{chapter_num}章 {title}
- Execute the selected plan and all constraints.
- Put the high-risk plan beats directly on page; do not leave important operations only implied.
- Repair the recent quality feedback explicitly through scenes, choices, cost, and consequences.
- Vary scene staging, chapter ending, emotional texture, and reasoning posture from recent chapters.
- When logistics matter, show the time, route, handler, procedure, and institutional friction.
- Keep causality, character agency, pressure-payoff rhythm, and hook strength.
- Avoid summary-like prose, repetitive shock reactions, and cheap coincidence.
- Output the chapter only, no explanation."""

REVISE_SYSTEM = """You are a Chinese web novel revision writer.
Revise the full chapter according to the final editor report.
Keep the title and core events. Do not introduce new continuity risks.
Prefer targeted structural repair over cosmetic rewriting:
- Add missing causal bridges and concrete scenes.
- Replace repeated staging or chapter endings.
- Make plan beats visible on page.
- Strengthen character agency, procedural friction, and pressure-payoff rhythm.
Output the revised chapter only."""

EXTRACT_SYSTEM = """You are the event-sourcing extractor for a long-form fiction engine.
Return exactly one valid JSON object and no other text:
{
  "title": "...",
  "events": [{"type":"plot|world|character|force|thread|item|battle|relationship","summary":"...","effects":[]}],
  "entities": [{"entity_type":"character|force|place|item|rule","name":"...","state_patch":{}}],
  "threads": [{"id":"stable-id","description":"...","status":"open|advanced|recovered|dropped","introduced_chapter":1,"due_chapter":20,"payload":{}}],
  "causal_links": [{"from_event":"source event summary","to_event":"expected future event or consequence","link_type":"causes|enables|blocks|requires","description":"why this causal link exists"}],
  "metrics": {
    "payoff_type":"court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
    "conflict_type":"court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
    "tension":1-10,
    "novelty":1-10,
    "hook_strength":1-10,
    "emotional_tone":"..."
  },
  "memory_updates": {
    "bible": [],
    "characters": [],
    "timeline": [],
    "threads": []
  }
}"""

STATE_UPDATE_SYSTEM = """You maintain the short working state for a 2M+ novel.
Return markdown only, no explanation.
Requirements:
- <=5000 Chinese characters.
- Include current progress, volume/stage goal, protagonist state, key conflicts, next 12 chapter directions.
- Keep recent chapter summaries compact.
- Preserve hard continuity constraints."""

def write_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    tail: str,
) -> str:
    title = str(plan.get("title") or f"Chapter {chapter_num}").strip()
    system = WRITE_SYSTEM.format(
        chapter_words=int(config["novel"]["chapter_words"]),
        chapter_num=chapter_num,
        title=title,
    )
    user = f"""## Memory
{memory_context(paths, conn, config)}

## Previous Tail
{tail[-int(config["novel"]["recent_tail_chars"]):]}

## Recent Quality Feedback JSON (MUST REPAIR IN THIS CHAPTER)
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Selected Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Arbitration Constraints JSON
{json.dumps(decision.get("required_constraints", []), ensure_ascii=False, indent=2)}

Write chapter {chapter_num}."""
    raw = call_llm(client, paths, config, system, user, temperature=float(config["api"]["temperature"]))
    return normalize_chapter(raw)

def revise_chapter(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter: str,
    review: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    user = f"""## Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Editor Report JSON
{json.dumps(review, ensure_ascii=False, indent=2)}

## Original Chapter
{chapter}

Revise the full chapter."""
    raw = call_llm(client, paths, config, REVISE_SYSTEM, user, temperature=0.45)
    return normalize_chapter(raw)

def extract_events(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
) -> dict[str, Any]:
    user = f"""## Memory Before Chapter
{memory_context(paths, conn, config)}

## Chapter {chapter_num}
{chapter[:12000]}

Extract durable state changes."""
    raw = call_llm(client, paths, config, EXTRACT_SYSTEM, max_tokens=8000, user=json_prompt(user), temperature=0.2)
    return load_json_with_repair(client, paths, config, raw)

def update_structured_state(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    review: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    db_event(conn, chapter_num, "chapter_extraction", extraction)

    for event in extraction.get("events", []):
        db_event(conn, chapter_num, "story_event", event)

    for entity in extraction.get("entities", []):
        entity_type = str(entity.get("entity_type", "unknown"))
        name = str(entity.get("name", "unknown"))
        if isinstance(conn, JsonStoryStore):
            state = conn.get_entity_state(entity_type, name)
        else:
            old = conn.execute(
                "SELECT state_json FROM entities WHERE entity_type=? AND name=?",
                (entity_type, name),
            ).fetchone()
            state = json.loads(old["state_json"]) if old else {}
        patch = entity.get("state_patch") or {}
        if isinstance(patch, dict):
            state.update(patch)
        else:
            state["note"] = str(patch)
        if isinstance(conn, JsonStoryStore):
            conn.upsert_entity(entity_type, name, state, chapter_num)
        else:
            conn.execute(
                """
                INSERT INTO entities(entity_type, name, state_json, updated_chapter)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_type, name)
                DO UPDATE SET state_json=excluded.state_json, updated_chapter=excluded.updated_chapter
                """,
                (entity_type, name, json.dumps(state, ensure_ascii=False), chapter_num),
            )

    for thread in extraction.get("threads", []):
        thread_id = str(thread.get("id") or f"ch{chapter_num}-{abs(hash(json.dumps(thread, ensure_ascii=False))) % 100000}")
        if isinstance(conn, JsonStoryStore):
            conn.upsert_thread(thread_id, thread, chapter_num)
        else:
            conn.execute(
                """
                INSERT INTO open_threads(id, description, status, introduced_chapter, due_chapter, updated_chapter, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id)
                DO UPDATE SET description=excluded.description, status=excluded.status,
                              due_chapter=excluded.due_chapter, updated_chapter=excluded.updated_chapter,
                              payload_json=excluded.payload_json
                """,
                (
                    thread_id,
                    str(thread.get("description", "")),
                    str(thread.get("status", "open")),
                    thread.get("introduced_chapter"),
                    thread.get("due_chapter"),
                    chapter_num,
                    json.dumps(thread.get("payload", {}), ensure_ascii=False),
                ),
            )

    metrics = extraction.get("metrics") or {}
    metrics_row = {
        "chapter": chapter_num,
        "title": extraction.get("title"),
        "score": safe_score(review.get("score", 0)),
        "plan_score": plan_score(decision),
        "payoff_type": metrics.get("payoff_type"),
        "conflict_type": metrics.get("conflict_type"),
        "tension": metrics.get("tension"),
        "novelty": metrics.get("novelty"),
        "hook_strength": metrics.get("hook_strength"),
        "emotional_tone": metrics.get("emotional_tone"),
        "accepted": 1 if review.get("accepted") else 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(conn, JsonStoryStore):
        conn.upsert_metrics(chapter_num, metrics_row)
    else:
        conn.execute(
            """
            INSERT INTO chapter_metrics(
                chapter, title, score, plan_score, payoff_type, conflict_type, tension,
                novelty, hook_strength, emotional_tone, accepted, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter) DO UPDATE SET
                title=excluded.title, score=excluded.score, plan_score=excluded.plan_score,
                payoff_type=excluded.payoff_type, conflict_type=excluded.conflict_type,
                tension=excluded.tension, novelty=excluded.novelty, hook_strength=excluded.hook_strength,
                emotional_tone=excluded.emotional_tone, accepted=excluded.accepted
            """,
            (
                metrics_row["chapter"],
                metrics_row["title"],
                metrics_row["score"],
                metrics_row["plan_score"],
                metrics_row["payoff_type"],
                metrics_row["conflict_type"],
                metrics_row["tension"],
                metrics_row["novelty"],
                metrics_row["hook_strength"],
                metrics_row["emotional_tone"],
                metrics_row["accepted"],
                metrics_row["created_at"],
            ),
        )
        conn.commit()

    updates = extraction.get("memory_updates") or {}
    append_memory(paths.bible, chapter_num, updates.get("bible") or [])
    append_memory(paths.characters, chapter_num, updates.get("characters") or [])
    append_memory(paths.timeline, chapter_num, updates.get("timeline") or [])
    append_memory(paths.threads, chapter_num, updates.get("threads") or [])

    store_causal_links(conn, chapter_num, extraction.get("causal_links") or [])

def append_memory(path: Path, chapter_num: int, items: list[Any]) -> None:
    if not items:
        return
    append_text(path, f"\n\n## Ch{chapter_num}\n" + "\n".join(f"- {item}" for item in items) + "\n")

def update_state_file(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    extraction: dict[str, Any],
) -> None:
    if paths.state.exists():
        shutil.copy2(paths.state, paths.state.with_suffix(".md.bak"))
    user = f"""## Current State
{read_text(paths.state)}

## Memory Context
{memory_context(paths, conn, config)}

## Extraction JSON
{json.dumps(extraction, ensure_ascii=False, indent=2)}

## Current Total Characters
{count_chars(paths.book)}

## Recent Chapter Text
{chapter[:5000]}

Update state.md after chapter {chapter_num}."""
    new_state = call_llm(client, paths, config, STATE_UPDATE_SYSTEM, user, max_tokens=8000, temperature=0.25)
    write_text(paths.state, normalize_text(new_state) + "\n")

def save_chapter(paths: Paths, chapter_num: int, chapter: str, review: dict[str, Any], plan: dict[str, Any]) -> None:
    chapter = normalize_chapter(chapter)
    write_text(chapter_path(paths, chapter_num), chapter)
    append_text(paths.book, "\n\n" + chapter)
    append_text(
        paths.logs_dir / "reviews.jsonl",
        json.dumps(
            {
                "chapter": chapter_num,
                "score": review.get("score"),
                "accepted": review.get("accepted"),
                "problems": review.get("problems", []),
                "continuity_risks": review.get("continuity_risks", []),
                "plan_title": plan.get("title"),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )
