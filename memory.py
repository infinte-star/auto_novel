from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
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
    voice_anchor = read_text(paths.voice).strip()
    voices_table = read_text(paths.voices).strip()
    style_block = ""
    if voice_anchor:
        style_block += "\n\n## Narrative Voice Anchor (MUST follow)\n" + voice_anchor
    if voices_table:
        style_block += "\n\n## Character Voices (MUST follow)\n" + voices_table
    tier1 = "## Creative Brief\n" + creative_brief + "\n\n## Current State\n" + current_state + style_block

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

# Module-level cache for the cacheable prefix so that subsequent calls in the
# same process re-use the EXACT same string (byte-for-byte) when the underlying
# files are unchanged. The cache key is a sha1 of the source file contents +
# budget; when any source changes, the cache is rebuilt and a new prefix string
# is returned (so prefix cache invalidation matches content change).
#
# This also implements task #9 (memory hash skip): the hash is computed over
# bible/characters/voice/voices/prompt content; if all are unchanged since
# last call, the cached string is returned in O(1) (no re-read, no re-format,
# no truncation). Provider prefix caches see identical bytes -> ~free prefill.
_CACHEABLE_PREFIX_CACHE: dict[str, tuple[str, str]] = {}
_CACHEABLE_PREFIX_STATS = {"hits": 0, "misses": 0}


def _files_hash(paths_list: list[Path]) -> str:
    hasher = hashlib.sha1()
    for p in paths_list:
        try:
            data = p.read_bytes() if p.exists() else b""
        except OSError:
            data = b""
        hasher.update(str(p).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(hashlib.sha1(data).digest())
    return hasher.hexdigest()


def file_hash_short(path: Path) -> str:
    """Short sha1 (12 hex chars) of file content; '' if missing."""
    try:
        if not path.exists():
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    return hashlib.sha1(data).hexdigest()[:12]


def cacheable_prefix(
    paths: Paths,
    config: dict[str, Any],
    log_fn: Any = None,
) -> str:
    """Build the EXACT-same-bytes prompt prefix shared across calls.

    This prefix is included verbatim at the top of each LLM call's user message
    (via call_llm's cacheable_prefix arg). Provider-side prefix caches will hit
    as long as the bytes are identical, so we return the same cached string
    when the source files have not changed. On change, the cache key changes
    and downstream invocations naturally invalidate.
    """
    budget = int(config["novel"].get("cacheable_prefix_chars", 30000))
    sources = [PROMPT_FILE, paths.voice, paths.voices, paths.bible, paths.characters]
    key = f"{_files_hash(sources)}:{budget}"

    cached = _CACHEABLE_PREFIX_CACHE.get("active")
    if cached and cached[0] == key:
        _CACHEABLE_PREFIX_STATS["hits"] += 1
        return cached[1]
    _CACHEABLE_PREFIX_STATS["misses"] += 1

    creative_brief = read_text(PROMPT_FILE).strip()
    voice_anchor = read_text(paths.voice).strip()
    voices_table = read_text(paths.voices).strip()
    bible = read_text(paths.bible).strip()
    characters = read_text(paths.characters).strip()

    sections: list[tuple[str, str, int]] = [
        ("Creative Brief", creative_brief, 4000),
        ("Narrative Voice Anchor", voice_anchor, 5000),
        ("Character Voices", voices_table, 7000),
        ("World Bible", bible, 7000),
        ("Characters", characters, 7000),
    ]
    parts: list[str] = ["# Stable Reference (cacheable)"]
    used = len(parts[0])
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > budget:
            remaining = budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    text = "\n\n".join(parts)
    _CACHEABLE_PREFIX_CACHE["active"] = (key, text)
    if log_fn is not None:
        try:
            stats = _CACHEABLE_PREFIX_STATS
            total = stats["hits"] + stats["misses"]
            hit_rate = (stats["hits"] / total * 100.0) if total else 0.0
            log_fn(
                f"cacheable_prefix rebuilt chars={len(text)} key={key[:12]} "
                f"hits={stats['hits']} misses={stats['misses']} hit_rate={hit_rate:.1f}%"
            )
        except Exception:
            pass
    return text


def cacheable_prefix_hit_rate() -> tuple[int, int]:
    """Return (hits, misses) for diagnostics."""
    return _CACHEABLE_PREFIX_STATS["hits"], _CACHEABLE_PREFIX_STATS["misses"]


def writing_memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    """Compact memory context for chapter writing.

    Excludes the content that is already shipped via cacheable_prefix() (creative
    brief, voice anchors, bible, characters). This keeps the variable portion
    small so prefix cache hits more, and avoids duplication.

    Sections (capped):
    - Current State (full state.md)
    - Threads (open)
    - Recent Metrics
    - Volume Plan (small)
    """
    char_budget = int(config["novel"].get("writing_memory_chars", 50000))

    current_state = read_text(paths.state).strip()
    threads_text = read_text(paths.threads).strip()
    volume_plan = read_text(paths.volume_plan).strip()
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)

    sections: list[tuple[str, str, int]] = [
        ("Current State", current_state, 10000),
        ("Threads", threads_text, 8000),
        ("Recent Metrics JSON", metrics_5, 2500),
        ("Volume Plan (head)", volume_plan, 6000),
    ]
    parts: list[str] = []
    used = 0
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > char_budget:
            remaining = char_budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _legacy_writing_memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    # Retained for reference only; not used after cacheable_prefix split.
    return ""


def lite_memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    """Slim memory context for plan-review and screening calls.

    Drops timeline, full events list, voices table, and recent_events from the
    full memory_context. Keeps the creative brief, current state, voice anchor,
    bible (capped), characters (capped), threads (capped), recent metrics 5 rows.
    """
    char_budget = int(config["novel"].get("plan_review_memory_chars", 10000))
    creative_brief = read_text(PROMPT_FILE).strip()
    current_state = read_text(paths.state).strip()
    voice_anchor = read_text(paths.voice).strip()
    bible = read_text(paths.bible).strip()
    characters = read_text(paths.characters).strip()
    threads_text = read_text(paths.threads).strip()
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)

    sections: list[tuple[str, str, int]] = [
        ("Creative Brief", creative_brief, 1500),
        ("Current State", current_state, 2500),
        ("Narrative Voice Anchor", voice_anchor, 1200),
        ("Recent Metrics JSON", metrics_5, 1200),
        ("Threads", threads_text, 1500),
        ("Characters", characters, 1500),
        ("World Bible", bible, 1200),
    ]
    parts: list[str] = []
    used = 0
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > char_budget:
            remaining = char_budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)

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
    compressed = call_llm(client, paths, config, system, old_text, max_tokens=12000, temperature=0.2)
    compressed = normalize_text(compressed)
    new_content = header.rstrip() + "\n\n## Consolidated\n" + compressed + "\n\n" + "".join(recent_sections)
    write_text(file_path, new_content)

def compress_all_memory(client: OpenAI, paths: Paths, config: dict[str, Any]) -> None:
    targets = [
        fp for fp in (paths.bible, paths.characters, paths.timeline, paths.threads)
        if fp.exists() and read_text(fp).strip()
    ]
    if not targets:
        return
    max_workers = int(config["novel"].get("max_parallel_workers", 8))

    def run_one(file_path: Path) -> tuple[Path, Exception | None]:
        try:
            compress_memory_file(client, paths, config, file_path)
            return file_path, None
        except Exception as exc:
            return file_path, exc

    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as executor:
        futures = {executor.submit(run_one, fp): fp for fp in targets}
        for future in as_completed(futures):
            fp, err = future.result()
            if err is not None:
                log(paths, f"compress_memory_file failed for {fp.name}: {err}")

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
