from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from config import PROMPT_FILE, Paths, log, normalize_text, read_text, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from store import db_event, recent_events, recent_metrics

if TYPE_CHECKING:
    from openai import OpenAI

BOOTSTRAP_SYSTEM = """You are the chief architect for a 2M+ Chinese web novel.
Return exactly one valid JSON object and no other text. Keys:
{
  "state": "short current-state markdown, <=5000 Chinese chars",
  "bible": "world rules, power system, social order, hard constraints",
  "characters": "major character state machines: goal, fear, resources, relationships, secrets",
  "timeline": "initial chronology and planned historical pressure",
  "threads": "open foreshadowing ledger with introduced/due/status",
  "volume_plan": "at least 3 volumes, 60-80 chapters each, with major event anchors"
}
Create original material. Do not imitate existing works. Optimize for long-term causality and reader anticipation."""

MEMORY_COMPRESS_SYSTEM = """You compress memory entries for a long-form fiction engine.
Input: a memory file with per-chapter entries (## ChN sections).
Output: a consolidated markdown that preserves:
- All entity names and their CURRENT state (not historical intermediate states)
- All unresolved constraints and open threads
- All causal dependencies still relevant to future chapters
- Key turning points and irreversible changes
Remove: superseded states, routine confirmations, resolved items, redundant updates.
Keep output under {max_chars} Chinese characters.
Output the consolidated content only, no explanation."""

def bootstrap(client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any]) -> None:
    log(paths, "Bootstrapping layered memory")
    raw = call_llm(client, paths, config, BOOTSTRAP_SYSTEM, json_prompt(read_text(PROMPT_FILE)), temperature=0.7)
    data = load_json_with_repair(client, paths, config, raw)
    write_text(paths.state, data["state"].strip() + "\n")
    write_text(paths.bible, data["bible"].strip() + "\n")
    write_text(paths.characters, data["characters"].strip() + "\n")
    write_text(paths.timeline, data["timeline"].strip() + "\n")
    write_text(paths.threads, data["threads"].strip() + "\n")
    write_text(paths.volume_plan, data["volume_plan"].strip() + "\n")
    db_event(conn, 0, "bootstrap", data)

def estimate_chars_budget(config: dict[str, Any]) -> int:
    context_window = int(config["api"].get("context_window", 1000000))
    reserve = int(config["novel"].get("context_budget_reserve_chars", 40000))
    return max(context_window - reserve, 50000)

def truncate_section(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

def memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    budget = estimate_chars_budget(config)
    fatigue_window = int(config["novel"]["fatigue_window"])

    creative_brief = read_text(PROMPT_FILE).strip()
    current_state = read_text(paths.state).strip()
    tier1 = "## Creative Brief\n" + creative_brief + "\n\n## Current State\n" + current_state

    volume_plan = read_text(paths.volume_plan).strip()
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)
    threads_text = read_text(paths.threads).strip()
    tier2 = "## Volume Plan\n" + volume_plan + "\n\n## Key Metrics JSON\n" + metrics_5 + "\n\n## Threads\n" + threads_text

    characters = read_text(paths.characters).strip()
    bible = read_text(paths.bible).strip()
    events_20 = json.dumps(recent_events(conn, 20), ensure_ascii=False, indent=2)
    tier3 = "## Characters\n" + characters + "\n\n## World Bible\n" + bible + "\n\n## Recent Events JSON\n" + events_20

    timeline = read_text(paths.timeline).strip()
    metrics_full = json.dumps(recent_metrics(conn, fatigue_window), ensure_ascii=False, indent=2)
    events_full = json.dumps(recent_events(conn, 40), ensure_ascii=False, indent=2)
    tier4 = "## Timeline\n" + timeline + "\n\n## Full Metrics JSON\n" + metrics_full + "\n\n## Full Events JSON\n" + events_full

    assembled = tier1
    remaining = budget - len(assembled)

    if remaining > len(tier2):
        assembled += "\n\n" + tier2
        remaining = budget - len(assembled)
    else:
        assembled += "\n\n" + truncate_section(tier2, max(remaining - 100, 0))
        return assembled

    if remaining > len(tier3):
        assembled += "\n\n" + tier3
        remaining = budget - len(assembled)
    else:
        assembled += "\n\n" + truncate_section(tier3, max(remaining - 100, 0))
        return assembled

    if remaining > len(tier4):
        assembled += "\n\n" + tier4
    elif remaining > 2000:
        assembled += "\n\n" + truncate_section(tier4, max(remaining - 100, 0))

    return assembled

def should_compress_memory(paths: Paths, config: dict[str, Any], chapter_num: int) -> bool:
    compress_every = int(config["novel"].get("memory_compress_every", 30))
    max_kb = int(config["novel"].get("memory_max_kb", 15))
    if chapter_num > 0 and chapter_num % compress_every == 0:
        return True
    for p in [paths.bible, paths.characters, paths.timeline, paths.threads]:
        if p.exists() and p.stat().st_size > max_kb * 1024:
            return True
    return False

def compress_memory_file(
    client: OpenAI, paths: Paths, config: dict[str, Any], file_path: Path, keep_recent: int = 30
) -> None:
    content = read_text(file_path)
    if not content.strip():
        return
    sections = re.split(r"(?=^## Ch\d+)", content, flags=re.MULTILINE)
    if len(sections) <= 2:
        return
    header = sections[0]
    chapter_sections = sections[1:]
    if len(chapter_sections) <= keep_recent:
        return
    old_sections = chapter_sections[:-keep_recent]
    recent_sections = chapter_sections[-keep_recent:]
    archive_dir = paths.logs_dir / "memory_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    write_text(archive_path, "".join(old_sections))
    old_text = "".join(old_sections)
    max_chars = 3000
    system = MEMORY_COMPRESS_SYSTEM.format(max_chars=max_chars)
    compressed = call_llm(client, paths, config, system, old_text, max_tokens=65000, temperature=0.2)
    compressed = normalize_text(compressed)
    new_content = header.rstrip() + "\n\n## Consolidated\n" + compressed + "\n\n" + "".join(recent_sections)
    write_text(file_path, new_content)

def compress_all_memory(client: OpenAI, paths: Paths, config: dict[str, Any]) -> None:
    for file_path in [paths.bible, paths.characters, paths.timeline, paths.threads]:
        if file_path.exists() and read_text(file_path).strip():
            compress_memory_file(client, paths, config, file_path)

def rhythm_diagnostics(conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    window = int(config["novel"]["repeat_window"])
    rows = recent_metrics(conn, window)
    if not rows:
        return {
            "warnings": [],
            "payoff_counts": {},
            "conflict_counts": {},
            "avg_tension": None,
            "avg_novelty": None,
            "avg_hook": None,
        }

    payoff_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    tensions = []
    novelties = []
    hooks = []
    for row in rows:
        payoff_counts[row.get("payoff_type") or "unknown"] = payoff_counts.get(row.get("payoff_type") or "unknown", 0) + 1
        conflict_counts[row.get("conflict_type") or "unknown"] = conflict_counts.get(row.get("conflict_type") or "unknown", 0) + 1
        if row.get("tension") is not None:
            tensions.append(int(row["tension"]))
        if row.get("novelty") is not None:
            novelties.append(int(row["novelty"]))
        if row.get("hook_strength") is not None:
            hooks.append(int(row["hook_strength"]))

    warnings = []
    dominant_payoff = max(payoff_counts.items(), key=lambda x: x[1])
    dominant_conflict = max(conflict_counts.items(), key=lambda x: x[1])
    if dominant_payoff[1] >= max(4, window // 3):
        warnings.append(f"Payoff repetition risk: {dominant_payoff[0]} used {dominant_payoff[1]} times recently.")
    if dominant_conflict[1] >= max(4, window // 3):
        warnings.append(f"Conflict repetition risk: {dominant_conflict[0]} used {dominant_conflict[1]} times recently.")
    avg_novelty = sum(novelties) / len(novelties) if novelties else None
    avg_hook = sum(hooks) / len(hooks) if hooks else None
    if avg_novelty is not None and avg_novelty < 6:
        warnings.append("Novelty is low across recent chapters.")
    if avg_hook is not None and avg_hook < 6:
        warnings.append("Hook strength is low across recent chapters.")

    return {
        "warnings": warnings,
        "payoff_counts": payoff_counts,
        "conflict_counts": conflict_counts,
        "avg_tension": sum(tensions) / len(tensions) if tensions else None,
        "avg_novelty": avg_novelty,
        "avg_hook": avg_hook,
    }

def structural_repetition_analysis(conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    window = int(config["novel"]["repeat_window"])
    rows = recent_metrics(conn, window)
    result: dict[str, Any] = {"warnings": [], "repeated_patterns": [], "tension_shape": "unknown"}
    if len(rows) < 6:
        return result

    sequence = [
        (r.get("conflict_type", ""), r.get("payoff_type", ""), r.get("emotional_tone", ""))
        for r in reversed(rows)
    ]

    # Sliding window pattern detection (window size 3)
    seen_patterns: dict[str, int] = {}
    for i in range(len(sequence) - 2):
        pattern_key = "|".join(f"{s[0]},{s[1]}" for s in sequence[i : i + 3])
        seen_patterns[pattern_key] = seen_patterns.get(pattern_key, 0) + 1
    repeated = [(k, v) for k, v in seen_patterns.items() if v >= 2]
    if repeated:
        result["repeated_patterns"] = [k for k, _ in repeated]
        result["warnings"].append(f"Repeated arc patterns detected: {len(repeated)} patterns appear 2+ times")

    # Tension curve shape analysis
    tensions = [int(r.get("tension", 5)) for r in reversed(rows) if r.get("tension") is not None]
    if len(tensions) >= 6:
        diffs = [tensions[i + 1] - tensions[i] for i in range(len(tensions) - 1)]
        flat_count = sum(1 for d in diffs if abs(d) <= 1)
        if flat_count > len(diffs) * 0.7:
            result["tension_shape"] = "flat"
            result["warnings"].append("Tension curve is flat — lacking dramatic variation")
        else:
            rises = sum(1 for d in diffs if d > 0)
            falls = sum(1 for d in diffs if d < 0)
            if rises > len(diffs) * 0.7:
                result["tension_shape"] = "monotone_rise"
            elif falls > len(diffs) * 0.7:
                result["tension_shape"] = "monotone_fall"
                result["warnings"].append("Tension is monotonically falling — reader engagement at risk")
            else:
                result["tension_shape"] = "varied"

    # Resolution monotony: check if emotional_tone repeats
    tones = [r.get("emotional_tone", "") for r in reversed(rows) if r.get("emotional_tone")]
    if len(tones) >= 5:
        tone_counts: dict[str, int] = {}
        for t in tones:
            tone_counts[t] = tone_counts.get(t, 0) + 1
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])
        if dominant_tone[1] >= len(tones) * 0.6:
            result["warnings"].append(f"Emotional monotony: '{dominant_tone[0]}' dominates {dominant_tone[1]}/{len(tones)} chapters")

    return result
