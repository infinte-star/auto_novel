from __future__ import annotations

import json
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint
from config import (
    Paths,
    append_text,
    chapter_path,
    count_chars,
    normalize_chapter,
    normalize_text,
    read_text,
    safe_score,
    write_text,
)
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, memory_context, writing_memory_context
from planning import plan_score
from store import JsonStoryStore, db_event, recent_quality_feedback, store_causal_links

if TYPE_CHECKING:
    from openai import OpenAI

WRITE_SYSTEM = """You are a professional Chinese long-form web novel author.
Write the chapter in Chinese.

## Internal self-critique protocol (MANDATORY before writing)
Before producing the chapter, run this critique inside reasoning_content:
1. List the 3 highest risks specific to this chapter:
   - Repetition risk: which recent scene/staging/ending device this chapter might inadvertently copy.
   - Shallow execution risk: which plan beat is most likely to become summary instead of scene.
   - Hollow payoff risk: where the protagonist might get a win without visible cost.
2. For each risk, write one concrete avoidance commitment (e.g. "use 茶寮 not 文渊阁", "show 户部 procedure on page", "let 朱由检 lose a piece of leverage").
3. Sketch 2 candidate openings (1 sentence each). Pick the stronger one and justify in one phrase.
4. Only after steps 1-3 begin the actual chapter output.

Do NOT include the critique in the final chapter output. It belongs in reasoning only.

## Output requirements
- Around {chapter_words} Chinese characters.
- Start exactly with: 第{chapter_num}章 {title}
- Execute the selected plan and all constraints.
- Put the high-risk plan beats directly on page; do not leave important operations only implied.
- Repair the recent quality feedback explicitly through scenes, choices, cost, and consequences.
- Vary scene staging, chapter ending, emotional texture, and reasoning posture from recent chapters.
- When logistics matter, show the time, route, handler, procedure, and institutional friction.
- Keep causality, character agency, pressure-payoff rhythm, and hook strength.
- Avoid summary-like prose, repetitive shock reactions, and cheap coincidence.
- Output the chapter only, no explanation.

Structure template:
- Opening hook (200-400字): 紧接上章末尾，建立本章核心问题或悬念
- Scene 1 (1000-1500字): 主要冲突场景，含具体动作、对话与环境描写
- Scene 2 (800-1200字): 转折或揭示场景，推进plan中的关键beats
- Scene 3 (600-1000字): 决定或代价场景，呈现选择后果
- Closing hook (200-400字): 制造下章悬念，不要用总结式收尾

Sensory discipline:
- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）
- 用具体细节代替抽象描述（"墨迹未干的公文" 而非 "重要文件"）

Dialogue ratio:
- 对话占全章30-50%，避免连续500字以上无对白的段落
- 每个角色的语气、用词应反映其身份和性格

Forbidden patterns:
- 禁止"他突然意识到/恍然大悟"式的廉价顿悟
- 禁止角色长段心理独白超过200字
- 禁止用解释性叙述代替戏剧化呈现（show don't tell）
- 禁止章末用"他知道，一切才刚刚开始"之类的空洞总结"""

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

def carried_over_partial_beats(paths: Paths, chapter_num: int, limit: int = 6) -> list[dict[str, Any]]:
    """Return the previous chapter's partial/absent beats so the next writer can repair them.

    Reads final_review.json -> review_round0.json -> review_round1.json in order
    of preference, and returns up to `limit` entries containing
    {"beat": str, "status": "partial|absent", "evidence": str}.
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    for key in ("final_review.json", "review_round1.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        beats = data.get("beats_audit") or []
        partial: list[dict[str, Any]] = []
        for entry in beats:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).lower()
            if status not in ("partial", "absent"):
                continue
            partial.append({
                "beat": str(entry.get("beat", ""))[:300],
                "status": status,
                "evidence": str(entry.get("evidence", ""))[:200],
            })
            if len(partial) >= limit:
                break
        if partial:
            return partial
    return []


def writer_directives_for_chapter(paths: Paths, chapter_num: int, limit: int = 6) -> list[str]:
    """Return directives carried from the previous chapter's review.

    Reads the previous chapter's review (final_review.json preferred) and
    extracts a flat list of imperative strings to inject at the top of the
    current chapter's write prompt. This forms a review->writer feedback loop
    that is more concrete than plan-level required_constraints (it speaks in
    terms of execution, not strategy).
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    directives: list[str] = []
    for key in ("final_review.json", "review_round1.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        for field in ("writer_directives_for_next_chapter", "writer_directives"):
            for item in data.get(field, []) or []:
                text = str(item).strip()
                if text and text not in directives:
                    directives.append(text)
                if len(directives) >= limit:
                    return directives
        if directives:
            return directives
    return directives


HOOK_REVISE_SYSTEM = """You are a Chinese web novel chapter-ending specialist.
Rewrite ONLY the final 300-500 Chinese characters of the chapter so the ending hook is sharp, specific,
and creates a clear next-chapter question for the reader.

Constraints:
- Do NOT change ANYTHING before the rewrite point. Output the FULL chapter with the original opening and middle preserved verbatim, and only the closing segment replaced.
- The new ending must avoid the forbidden patterns: 廉价顿悟 ("他突然意识到"), 总结式收尾 ("一切才刚刚开始"), abstract foreshadowing.
- The new ending should raise a specific, concrete question or set up a concrete obstacle that the next chapter must address.
- Match the established narrative voice; do not introduce new characters or facts that aren't already established.
- The replacement should be roughly the same length as the original ending (within 20%)."""


def revise_hook_only(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter: str,
    plan: dict[str, Any],
    review: dict[str, Any],
    tail_to_revise_chars: int = 400,
) -> str:
    """Rewrite only the last ~300-500 chars of the chapter to fix a weak ending hook.

    This is a much cheaper alternative to a full revise: a single small LLM call
    that the writer copies the head verbatim and only mutates the tail. Returns
    the new full chapter text.
    """
    chapter = normalize_chapter(chapter)
    n = len(chapter)
    cut = max(0, n - tail_to_revise_chars)
    # Snap cut point to a paragraph boundary if possible (look back up to 200 chars
    # for double-newline; otherwise single newline).
    snap_window = chapter[max(0, cut - 200): cut + 200]
    for marker in ("\n\n", "\n"):
        idx = snap_window.find(marker)
        if idx >= 0:
            cut = max(0, cut - 200) + idx + len(marker)
            break
    head = chapter[:cut]
    original_tail = chapter[cut:]
    user = f"""## Plan JSON (for context)
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Reviewer Feedback (why the hook is weak)
{json.dumps({
    "hook_strength": review.get("hook_strength"),
    "rhythm_risks": review.get("rhythm_risks", []),
    "writer_directives": review.get("writer_directives_for_next_chapter", []),
}, ensure_ascii=False, indent=2)}

## Chapter Head (DO NOT MODIFY — copy verbatim)
{head}

## Current Tail to Rewrite (length {len(original_tail)} chars)
{original_tail}

Rewrite the chapter. Copy the head verbatim, then replace the tail with a sharper ending
of similar length. Output the FULL chapter only."""
    raw = call_llm(
        client, paths, config, HOOK_REVISE_SYSTEM, user,
        max_tokens=8000, temperature=0.55,
        cacheable_prefix=cacheable_prefix(paths, config),
    )
    new_chapter = normalize_chapter(raw)
    # Safety: if the model failed to preserve the head (e.g., truncated or
    # rewrote opening), fall back to head + new tail by splicing.
    if not new_chapter.startswith(head[: min(len(head), 200)]):
        # Try to recover by extracting the model's "new tail" — assume it's
        # the last paragraph in its output.
        from config import log as _log
        _log(paths, "hook revise: head verification failed; splicing head + model_tail")
        model_tail = new_chapter.rsplit("\n\n", 1)[-1] if "\n\n" in new_chapter else new_chapter[-tail_to_revise_chars * 2:]
        new_chapter = normalize_chapter(head.rstrip() + "\n\n" + model_tail.strip())
    return new_chapter


def write_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
    temperature: float | None = None,
) -> str:
    title = str(plan.get("title") or f"Chapter {chapter_num}").strip()
    system = WRITE_SYSTEM.format(
        chapter_words=int(config["novel"]["chapter_words"]),
        chapter_num=chapter_num,
        title=title,
    )
    mem = cached_memory or writing_memory_context(paths, conn, config)
    partial_beats = carried_over_partial_beats(paths, chapter_num)
    directives = writer_directives_for_chapter(paths, chapter_num)
    carryover_block = ""
    if partial_beats:
        carryover_block += (
            f"\n## CRITICAL CARRYOVER FROM CH{chapter_num - 1} (MUST address on page)\n"
            f"The following beats were marked partial/absent in the previous chapter's review. "
            f"You MUST realize these on page in this chapter when narratively viable. "
            f"Do NOT leave them implied or off-page.\n"
            f"{json.dumps(partial_beats, ensure_ascii=False, indent=2)}\n"
        )
    if directives:
        carryover_block += (
            f"\n## REVIEWER DIRECTIVES FOR CH{chapter_num} (MUST obey)\n"
            f"These execution-level directives come from the previous chapter's reviewer. "
            f"They override generic guidelines when in conflict.\n"
            f"{json.dumps(directives, ensure_ascii=False, indent=2)}\n"
        )
    user = f"""## Memory
{mem}
{carryover_block}
## Previous Tail
{tail[-int(config["novel"]["recent_tail_chars"]):]}

## Recent Quality Feedback JSON (MUST REPAIR IN THIS CHAPTER)
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Selected Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Arbitration Constraints JSON
{json.dumps(decision.get("required_constraints", []), ensure_ascii=False, indent=2)}

Write chapter {chapter_num}."""
    temp = float(config["api"]["temperature"]) if temperature is None else temperature
    prefix = cacheable_prefix(paths, config)
    raw = call_llm(client, paths, config, system, user, temperature=temp, cacheable_prefix=prefix)
    return normalize_chapter(raw)

def apply_review_patches(chapter: str, patches: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Apply review-provided patches to chapter text in-place.

    Returns (new_chapter, applied_patches_with_status). Each patch entry gets an
    "applied" boolean and optionally an "error" reason if it could not be applied.
    Patches are applied in input order, each operating on the current text.
    Locators that no longer match (because an earlier patch removed/replaced the
    surrounding region) are skipped with applied=False.
    """
    text = chapter
    results: list[dict[str, Any]] = []
    for raw_patch in patches or []:
        if not isinstance(raw_patch, dict):
            results.append({"applied": False, "error": "non-dict patch", "patch": raw_patch})
            continue
        op = str(raw_patch.get("op", "")).strip().lower()
        locator = str(raw_patch.get("locator", "")).strip()
        before = str(raw_patch.get("before", "") or "").strip()
        after = str(raw_patch.get("after", "") or "")
        insert_text = str(raw_patch.get("insert", "") or "")
        entry = {**raw_patch, "applied": False}
        try:
            if op == "replace":
                target = before or locator
                if not target:
                    entry["error"] = "empty before/locator for replace"
                elif target not in text:
                    entry["error"] = "before/locator not found in chapter"
                else:
                    text = text.replace(target, after, 1)
                    entry["applied"] = True
            elif op == "insert_after":
                if not locator or locator not in text:
                    entry["error"] = "locator not found for insert_after"
                else:
                    idx = text.find(locator) + len(locator)
                    glue_before = "" if text[idx:idx+1] in {"\n", ""} else "\n\n"
                    glue_after = "" if text[idx:idx+2] == "\n\n" else "\n\n"
                    text = text[:idx] + glue_before + insert_text + glue_after + text[idx:]
                    entry["applied"] = True
            elif op == "delete":
                target = before or locator
                if not target or target not in text:
                    entry["error"] = "before/locator not found for delete"
                else:
                    text = text.replace(target, "", 1)
                    entry["applied"] = True
            else:
                entry["error"] = f"unknown op: {op!r}"
        except Exception as exc:
            entry["error"] = f"exception: {exc}"
        results.append(entry)
    return text, results


def revise_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter: str,
    review: dict[str, Any],
    plan: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
) -> str:
    # Fast path: try applying review patches directly without a full LLM rewrite.
    # Only fall back to LLM revision when patches are missing, incomplete, or fail.
    patches = review.get("patches") if isinstance(review, dict) else None
    use_patch_path = bool(config["novel"].get("revise_use_patches", True))
    if use_patch_path and isinstance(patches, list) and patches:
        patched, results = apply_review_patches(chapter, patches)
        applied = sum(1 for r in results if r.get("applied"))
        total = len(results)
        # Relaxed threshold: 1/2 of patches applied counts as success.
        # Surgical patch path is much faster than a full rewrite and the unapplied
        # patches typically address minor issues; the next review round will pick
        # up anything material that remains.
        min_apply_frac = float(config["novel"].get("revise_patch_min_frac", 0.5))
        threshold_hit = max(1, int(total * min_apply_frac + 0.999))
        if applied >= threshold_hit:
            from config import log as _log
            _log(paths, f"Revise via patches applied={applied}/{total} (>= {threshold_hit}); skipping full rewrite")
            return normalize_chapter(patched)
        else:
            from config import log as _log
            _log(paths, f"Revise patches too few hit ({applied}/{total} < {threshold_hit}); falling back to LLM rewrite")

    mem = cached_memory or writing_memory_context(paths, conn, config)
    user = f"""## Memory
{mem}

## Previous Tail
{tail[-1500:]}

## Plan JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## Recent Quality Feedback JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## Editor Report JSON
{json.dumps(review, ensure_ascii=False, indent=2)}

## Original Chapter
{chapter}

Revise the full chapter."""
    raw = call_llm(
        client, paths, config, REVISE_SYSTEM, user,
        temperature=0.45, cacheable_prefix=cacheable_prefix(paths, config),
    )
    return normalize_chapter(raw)

def extract_events(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    cached_memory: str | None = None,
) -> dict[str, Any]:
    mem = cached_memory or memory_context(paths, conn, config)
    user = f"""## Memory Before Chapter
{mem}

## Chapter {chapter_num}
{chapter[:8000]}

Extract durable state changes."""
    raw = call_llm(client, paths, config, EXTRACT_SYSTEM, max_tokens=12000, user=json_prompt(user), temperature=0.2)
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
    existing = read_text(path)
    section_header = f"## Ch{chapter_num}"
    if section_header in existing:
        return
    existing_bullets = set()
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            existing_bullets.add(stripped[2:].strip())
    fresh = []
    for item in items:
        text = str(item).strip()
        if not text or text in existing_bullets:
            continue
        fresh.append(text)
        existing_bullets.add(text)
    if not fresh:
        return
    append_text(path, f"\n\n{section_header}\n" + "\n".join(f"- {t}" for t in fresh) + "\n")

STATE_DYNAMIC_SECTIONS_SYSTEM = """You generate ONLY the two dynamic sections of a long-form novel's working state.
Return exactly one valid JSON object and no other text:
{
  "protagonist_state": "<=600 Chinese chars markdown: protagonist's current goals, resources, fears, secrets, ongoing pressure, and key decisions still pending. Reflect changes from this chapter.>",
  "next_12_directions": ["10-12 concrete directives for upcoming chapters; each a single Chinese sentence specifying what concretely must happen, NOT abstract themes"]
}
Constraints:
- protagonist_state must be self-sufficient (a new reader could pick up). Avoid vague phrases.
- next_12_directions must be SPECIFIC executable directives, not plot themes."""


def _render_state_md_template(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    protagonist_state: str,
    next_directions: list[str],
) -> str:
    """Compose the new state.md deterministically.

    The structure follows what readers expect: progress meta, recent chapter
    summaries (5), key entity states, active threads (open), and the LLM-only
    sections (protagonist_state, next_12_directions).
    """
    from store import recent_events, recent_metrics

    total_chars = count_chars(paths.book)
    metrics = recent_metrics(conn, 5)
    threads_text = read_text(paths.threads).strip()

    # Last 5 chapter title+key payoff
    summary_lines: list[str] = []
    for m in metrics:
        ch = m.get("chapter")
        title = m.get("title") or ""
        score = m.get("score")
        tone = m.get("emotional_tone") or ""
        payoff = m.get("payoff_type") or ""
        summary_lines.append(f"- Ch{ch} 「{title}」 score={score} payoff={payoff} tone={tone}")

    # Pull events from this chapter's extraction
    this_chapter_events: list[str] = []
    for ev in extraction.get("events", [])[:8]:
        s = str(ev.get("summary", "")).strip()
        if s:
            this_chapter_events.append(f"- {s[:200]}")

    next_dir_lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(next_directions[:12]))

    parts: list[str] = [
        f"# State Snapshot after Ch{chapter_num}",
        f"\n## Progress\n- Total chars: {total_chars}\n- Last chapter: Ch{chapter_num} 「{extraction.get('title', '')}」",
        "\n## Recent Chapters (newest first)\n" + ("\n".join(summary_lines) if summary_lines else "_(none)_"),
        "\n## Latest Chapter Key Events\n" + ("\n".join(this_chapter_events) if this_chapter_events else "_(none)_"),
        "\n## Protagonist State\n" + (protagonist_state.strip() or "_(empty)_"),
        "\n## Next 12 Chapter Directions\n" + (next_dir_lines or "_(none)_"),
        "\n## Active Threads\n" + (threads_text[:4000] if threads_text else "_(none)_"),
    ]
    return "\n".join(parts) + "\n"


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

    template_mode = bool(config["novel"].get("state_template_mode", True))
    if not template_mode:
        # Legacy path: full LLM regeneration (kept as fallback).
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
        new_state = call_llm(client, paths, config, STATE_UPDATE_SYSTEM, user, max_tokens=12000, temperature=0.25)
        write_text(paths.state, normalize_text(new_state) + "\n")
        return

    # Template mode: only ask LLM for the 2 dynamic sections, then deterministically
    # render the full state.md. This drops LLM output from ~12K tokens to ~2-3K.
    current_state_excerpt = read_text(paths.state)
    if len(current_state_excerpt) > 3000:
        current_state_excerpt = current_state_excerpt[:3000] + "\n...[truncated]"
    user = f"""## Previous Protagonist State (for continuity)
{current_state_excerpt}

## Extraction From Chapter {chapter_num}
{json.dumps(extraction, ensure_ascii=False, indent=2)}

## Latest Chapter Tail (last 2500 chars for fresh detail)
{chapter[-2500:]}

Produce ONLY the JSON with protagonist_state and next_12_directions."""
    try:
        raw = call_llm(
            client, paths, config, STATE_DYNAMIC_SECTIONS_SYSTEM, json_prompt(user),
            max_tokens=4000, temperature=0.25,
            cacheable_prefix=cacheable_prefix(paths, config),
        )
        data = load_json_with_repair(
            client, paths, config, raw,
            fallback={"protagonist_state": "", "next_12_directions": []},
        )
    except Exception as exc:
        from config import log as _log
        _log(paths, f"State dynamic sections LLM failed (non-fatal); using empty fallback: {exc}")
        data = {"protagonist_state": "", "next_12_directions": []}

    protagonist_state = str(data.get("protagonist_state", "")).strip()
    next_directions = [str(d).strip() for d in (data.get("next_12_directions") or []) if str(d).strip()]
    new_state = _render_state_md_template(
        paths, conn, chapter_num, extraction, protagonist_state, next_directions
    )
    write_text(paths.state, new_state)

def save_chapter(paths: Paths, chapter_num: int, chapter: str, review: dict[str, Any], plan: dict[str, Any]) -> None:
    chapter = normalize_chapter(chapter)
    if len(chapter.strip()) < 500:
        raise RuntimeError(
            f"Refusing to save Ch{chapter_num}: only {len(chapter.strip())} chars "
            f"(likely provider refusal or empty response). Preview: {chapter[:200]!r}"
        )
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
