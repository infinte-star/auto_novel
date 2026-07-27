"""Experiment harness: deterministic cross-novel comparison + single-gate ablation.

The engine evolved through 9 hand-compared versions (suspense_v3..v11); each
round cost ~4h of LLM time plus hours of human reading just to answer "did this
change help?". This module turns that into one command, using ONLY data the
pipeline already persists (story_state.db, logs/run.log, logs/llm_calls.jsonl,
config.yaml) — zero LLM calls, zero new dependencies.

    python novel.py compare <a> <b>             # side-by-side report -> experiments/
    python novel.py ablate <name> --flip <key>  # scaffold novels/<name>__ablate_<key>/
                                                #   with one config key flipped

Design constraints:
  * Read-only over novel directories; never touches book.md/chapters/.
  * Degrades gracefully: missing db/log/jsonl just leaves a section empty.
  * Report is plain markdown written to experiments/ AND printed.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
NOVELS_DIR = PROJECT_DIR / "novels"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"


# ----------------------------------------------------------------------------
# data loaders (all tolerant of missing files)
# ----------------------------------------------------------------------------
def _load_chapter_metrics(nd: Path) -> list[dict[str, Any]]:
    db = nd / "story_state.db"
    if not db.exists():
        return []
    try:
        import sqlite3
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM chapter_metrics ORDER BY chapter"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _load_events(nd: Path, types: tuple[str, ...]) -> dict[str, int]:
    """Count events of the given types from the novel's own store."""
    counts = {t: 0 for t in types}
    db = nd / "story_state.db"
    if not db.exists():
        return counts
    try:
        import sqlite3
        conn = sqlite3.connect(str(db))
        for t in types:
            try:
                counts[t] = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type=?", (t,)
                ).fetchone()[0]
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
    return counts


_LOG_PATTERNS = {
    "force_accept": re.compile(r"Accepting anyway to avoid pipeline halt"),
    "fossil_hits": re.compile(r"cross_chapter_fossils\((\d+)\)"),
    "scene_dedupe_warn": re.compile(r"Scene-dedupe WARN"),
    "scene_dedupe_block": re.compile(r"Scene-dedupe BLOCK"),
    "adjacent_block": re.compile(r"Adjacent-(?:repeat|duplicate)"),
    "gate_reject": re.compile(r"GATE-REJECT"),
    "style_collapse": re.compile(r"prose-health collapse|Style-health .* penalty"),
    "json_repair": re.compile(r"json_repair|JSON repair"),
}


def _scan_run_log(nd: Path) -> dict[str, Any]:
    log_path = nd / "logs" / "run.log"
    out: dict[str, Any] = {k: 0 for k in _LOG_PATTERNS}
    out["max_fossils"] = 0
    if not log_path.exists():
        return out
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        for key, pat in _LOG_PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            out[key] += 1
            if key == "fossil_hits" and m.groups():
                try:
                    out["max_fossils"] = max(out["max_fossils"], int(m.group(1)))
                except ValueError:
                    pass
    return out


# Tags emitted by offline `tools/` scripts, not by the pipeline. They land in the
# novel's own llm_calls.jsonl because those tools borrow an arm's config for its API
# keys, and counting them would charge an arm for the cost of being *measured*.
# Measured 2026-07-28: 10 `pairwise_ab` rows in novels/p4_score/logs inflated that
# arm's calls/chapter from 14.25 to 14.75 in its own P4 report. `pairwise_ab.py` no
# longer writes here, but existing logs still carry the rows.
OFFLINE_TOOL_TAGS = ("pairwise_ab",)

# Config keys that decide when a draft is released. Flipping one of these makes the
# self-score a measurement of the flip rather than of the prose, so the report says so
# instead of ranking the arms by it. See the verdict section.
RELEASE_RULE_KEYS = frozenset({
    "rework_trigger", "rework_score_floor", "quality_threshold",
    "max_revision_rounds", "consecutive_force_accept_limit",
    "circuit_breaker_score_floor",
})


def _llm_totals(nd: Path) -> dict[str, float]:
    """Total calls / seconds / output chars, plus the planning-stage share.

    `excluded` counts offline-tool rows skipped (see OFFLINE_TOOL_TAGS); the report
    prints it so a filtered log is never silently indistinguishable from a clean one.
    """
    path = nd / "logs" / "llm_calls.jsonl"
    tot = {"calls": 0.0, "elapsed": 0.0, "output": 0.0, "plan_elapsed": 0.0,
           "fail": 0.0, "excluded": 0.0}
    if not path.exists():
        return tot
    plan_tags = ("plan_candidate", "plan_review_fused", "plan_arbitrate", "plan_screen")
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("tag") or "") in OFFLINE_TOOL_TAGS:
                tot["excluded"] += 1
                continue
            el = float(row.get("elapsed") or 0.0)
            tot["calls"] += 1
            tot["elapsed"] += el
            tot["output"] += float(row.get("output_chars") or 0.0)
            if not row.get("ok", True):
                tot["fail"] += 1
            if str(row.get("tag") or "") in plan_tags:
                tot["plan_elapsed"] += el
    except OSError:
        pass
    return tot


def _book_chars(nd: Path) -> int:
    book = nd / "book.md"
    if not book.exists():
        return 0
    try:
        return len(book.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return 0


def _rework_signals(nd: Path, chapter: int) -> dict[str, Any]:
    """Per-chapter rework evidence, read from the checkpoint directory's file
    NAMES only (no JSON parsing, so this stays fast over a whole book).

    `chapter_metrics` records the outcome of a chapter, not how much work it
    took to get there. The checkpoint filenames do: a structural replan writes
    `plan_quality_replan_attempt*`, a blocked plan writes extra
    `plan_initial_attempt*`, each review round writes `review_round*`. This is
    the signal an A/B on the rework trigger lives or dies by.
    """
    d = nd / "logs" / "checkpoints" / f"ch{chapter:04d}"
    out = {"replan": 0, "plan_retry": 0, "revise": 0, "local_fix": 0, "debt": False}
    if not d.is_dir():
        return out
    try:
        names = [p.name for p in d.iterdir()]
    except OSError:
        return out
    out["replan"] = sum(1 for n in names if n.startswith("plan_quality_replan_attempt")
                        and n.endswith("_candidates.json"))
    out["plan_retry"] = max(0, sum(1 for n in names if n.startswith("plan_initial_attempt")
                                   and n.endswith("_candidates.json")) - 1)
    out["revise"] = max(0, sum(1 for n in names if n.startswith("review_round")) - 1)
    out["local_fix"] = sum(1 for n in names if n.startswith("local_fix_round"))
    out["debt"] = any(n.startswith("quality_debt") for n in names)
    return out


def _read_config_lines(nd: Path) -> dict[str, str]:
    """Flatten `section.key: value` pairs, skipping secrets and paths."""
    cfg = nd / "config.yaml"
    out: dict[str, str] = {}
    if not cfg.exists():
        return out
    section = ""
    try:
        for raw in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith((" ", "\t")) and line.endswith(":"):
                section = line[:-1].strip()
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if "api_key" in key or section == "paths":
                    continue
                out[f"{section}.{key}"] = val
    except OSError:
        pass
    return out


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------
def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def compare_novels(name_a: str, name_b: str,
                   ch_from: int = 0, ch_to: int = 0) -> str:
    """Build a markdown comparison report for two novels; returns the report.

    Entirely deterministic / zero-LLM.

    `ch_from`/`ch_to` (0 = unbounded) restrict every chapter-derived metric to a
    range. This is mandatory for a `fork`-based A/B: both arms inherit the
    source's whole `chapter_metrics` table, so a whole-novel average is diluted
    by the dozens of chapters the two arms share byte-for-byte. Log- and
    jsonl-derived totals are NOT filtered — `fork` leaves `logs/` behind, so
    they already describe only the chapters the fork wrote.
    """
    nd_a, nd_b = NOVELS_DIR / name_a, NOVELS_DIR / name_b
    for nd in (nd_a, nd_b):
        if not nd.exists():
            raise SystemExit(f"[compare] novel directory not found: {nd}")

    def in_range(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not (ch_from or ch_to):
            return rows
        return [r for r in rows
                if (not ch_from or int(r.get("chapter", 0)) >= ch_from)
                and (not ch_to or int(r.get("chapter", 0)) <= ch_to)]

    m_a, m_b = in_range(_load_chapter_metrics(nd_a)), in_range(_load_chapter_metrics(nd_b))
    log_a, log_b = _scan_run_log(nd_a), _scan_run_log(nd_b)
    llm_a, llm_b = _llm_totals(nd_a), _llm_totals(nd_b)
    ev_types = ("quality_debt", "gate_reject", "scene_dedupe_retry", "visual_payoff_retry")
    ev_a, ev_b = _load_events(nd_a, ev_types), _load_events(nd_b, ev_types)

    def scores(ms: list[dict[str, Any]]) -> list[float]:
        return [float(r["score"]) for r in ms if r.get("score") is not None]

    s_a, s_b = scores(m_a), scores(m_b)

    lines: list[str] = []
    lines.append(f"# Compare: {name_a} vs {name_b}")
    if ch_from or ch_to:
        lines.append(f"Chapter range: {ch_from or 1}..{ch_to or '∞'} "
                     f"(chapter metrics only; log/jsonl totals cover the whole run)")
    lines.append(f"Generated {datetime.now().isoformat(timespec='seconds')}\n")

    # --- headline table ---
    def row(label: str, va: Any, vb: Any) -> str:
        return f"| {label} | {_fmt(va)} | {_fmt(vb)} |"

    lines.append(f"| metric | {name_a} | {name_b} |")
    lines.append("|---|---|---|")
    lines.append(row("chapters scored", len(s_a), len(s_b)))
    lines.append(row("avg score", (sum(s_a) / len(s_a)) if s_a else None,
                     (sum(s_b) / len(s_b)) if s_b else None))
    lines.append(row("min score", min(s_a) if s_a else None, min(s_b) if s_b else None))
    lines.append(row("max score", max(s_a) if s_a else None, max(s_b) if s_b else None))
    lines.append(row("chapters < 7.0",
                     sum(1 for s in s_a if s < 7.0), sum(1 for s in s_b if s < 7.0)))
    lines.append(row("book chars", _book_chars(nd_a), _book_chars(nd_b)))

    # Rework is the P4 A/B's primary metric, so it belongs in the headline and
    # has to come from the checkpoints, not from the score column.
    def rework_totals(nd: Path, ms: list[dict[str, Any]]) -> dict[str, int]:
        rws = [_rework_signals(nd, int(r["chapter"])) for r in ms]
        return {
            "replan_ch": sum(1 for r in rws if r["replan"]),
            "revise_rounds": sum(r["revise"] for r in rws),
            "clean": sum(1 for r in rws if not any(
                (r["replan"], r["plan_retry"], r["revise"], r["local_fix"], r["debt"]))),
        }

    rw_a, rw_b = rework_totals(nd_a, m_a), rework_totals(nd_b, m_b)
    lines.append(row("chapters with a replan", rw_a["replan_ch"], rw_b["replan_ch"]))
    lines.append(row("total revise rounds", rw_a["revise_rounds"], rw_b["revise_rounds"]))
    lines.append(row("first-pass-clean chapters", rw_a["clean"], rw_b["clean"]))
    lines.append(row("force-accepts (log)", log_a["force_accept"], log_b["force_accept"]))
    lines.append(row("quality_debt events", ev_a["quality_debt"], ev_b["quality_debt"]))
    lines.append(row("gate_reject events", ev_a["gate_reject"], ev_b["gate_reject"]))
    lines.append(row("fossil warnings (log)", log_a["fossil_hits"], log_b["fossil_hits"]))
    lines.append(row("max fossils in one hit", log_a["max_fossils"], log_b["max_fossils"]))
    lines.append(row("scene-dedupe WARN", log_a["scene_dedupe_warn"], log_b["scene_dedupe_warn"]))
    lines.append(row("scene-dedupe BLOCK", log_a["scene_dedupe_block"], log_b["scene_dedupe_block"]))
    lines.append(row("LLM calls", int(llm_a["calls"]), int(llm_b["calls"])))
    lines.append(row("LLM calls / scored chapter",
                     (llm_a["calls"] / len(s_a)) if s_a else None,
                     (llm_b["calls"] / len(s_b)) if s_b else None))
    if llm_a["excluded"] or llm_b["excluded"]:
        lines.append(row("offline-tool rows excluded",
                         int(llm_a["excluded"]), int(llm_b["excluded"])))
    lines.append(row("LLM total minutes", llm_a["elapsed"] / 60, llm_b["elapsed"] / 60))
    lines.append(row("planning share of LLM time",
                     (llm_a["plan_elapsed"] / llm_a["elapsed"]) if llm_a["elapsed"] else None,
                     (llm_b["plan_elapsed"] / llm_b["elapsed"]) if llm_b["elapsed"] else None))
    lines.append(row("LLM minutes / scored chapter",
                     (llm_a["elapsed"] / 60 / len(s_a)) if s_a else None,
                     (llm_b["elapsed"] / 60 / len(s_b)) if s_b else None))
    lines.append("")

    # --- per-chapter score curves + rework evidence ---
    lines.append("## Per-chapter scores and rework")
    lines.append(f"| ch | {name_a} score | pen | rework | {name_b} score | pen | rework |")
    lines.append("|---|---|---|---|---|---|---|")
    by_a = {int(r["chapter"]): r for r in m_a}
    by_b = {int(r["chapter"]): r for r in m_b}

    def rework_cell(nd: Path, ch: int, present: bool) -> str:
        if not present:
            return "-"
        rw = _rework_signals(nd, ch)
        bits = []
        if rw["replan"]:
            bits.append(f"replan×{rw['replan']}")
        if rw["plan_retry"]:
            bits.append(f"retry×{rw['plan_retry']}")
        if rw["revise"]:
            bits.append(f"revise×{rw['revise']}")
        if rw["local_fix"]:
            bits.append(f"fix×{rw['local_fix']}")
        if rw["debt"]:
            bits.append("debt")
        return " ".join(bits) if bits else "clean"

    for ch in sorted(set(by_a) | set(by_b)):
        ra, rb = by_a.get(ch), by_b.get(ch)
        lines.append(
            f"| {ch} | {_fmt(ra.get('score') if ra else None, 1)} "
            f"| {_fmt(ra.get('style_penalty') if ra else None, 1)} "
            f"| {rework_cell(nd_a, ch, ra is not None)} "
            f"| {_fmt(rb.get('score') if rb else None, 1)} "
            f"| {_fmt(rb.get('style_penalty') if rb else None, 1)} "
            f"| {rework_cell(nd_b, ch, rb is not None)} |"
        )
    lines.append("")

    # --- config diff (non-secret, non-path keys) ---
    cfg_a, cfg_b = _read_config_lines(nd_a), _read_config_lines(nd_b)
    diffs = []
    circular = []
    for key in sorted(set(cfg_a) | set(cfg_b)):
        va, vb = cfg_a.get(key, "<absent>"), cfg_b.get(key, "<absent>")
        if va != vb:
            diffs.append(f"| {key} | {va} | {vb} |")
            if key.split(".")[-1] in RELEASE_RULE_KEYS:
                circular.append(key)
    lines.append("## Config differences")
    if diffs:
        lines.append(f"| key | {name_a} | {name_b} |")
        lines.append("|---|---|---|")
        lines.extend(diffs)
    else:
        lines.append("(identical apart from paths/keys)")
    lines.append("")

    # --- verdict heuristics ---
    lines.append("## Heuristic verdict")
    verdict: list[str] = []
    if circular:
        # The self-score is the release rule's own output: `accepted` is derived from
        # `quality_threshold` (review.py) and every score below it is revised until it
        # rises. So when the flipped key IS the release rule, "avg score" and
        # "sub-7.0 chapters" measure the flip, not the prose — that is how the P4 A/B
        # produced a 0.53 score gap that meant nothing (REDESIGN §7).
        lines.append(
            f"> **Score lines below are circular for this pair**: {', '.join(circular)} "
            f"changes the release rule the score is produced by. Settle it with "
            f"`python tools/fpy_prime.py {name_a} {name_b}` (self-score excluded) plus "
            f"`python tools/pairwise_ab.py --a {name_a} --b {name_b}`.")
        lines.append("")
    if s_a and s_b:
        avg_a, avg_b = sum(s_a) / len(s_a), sum(s_b) / len(s_b)
        d = avg_a - avg_b
        if abs(d) >= 0.3:
            better = name_a if d > 0 else name_b
            verdict.append(f"- avg score favors **{better}** by {abs(d):.2f}")
        low_a = sum(1 for s in s_a if s < 7.0)
        low_b = sum(1 for s in s_b if s < 7.0)
        if low_a != low_b:
            better = name_a if low_a < low_b else name_b
            verdict.append(f"- fewer sub-7.0 chapters: **{better}** ({min(low_a, low_b)} vs {max(low_a, low_b)})")
    if log_a["force_accept"] != log_b["force_accept"]:
        better = name_a if log_a["force_accept"] < log_b["force_accept"] else name_b
        verdict.append(f"- fewer force-accepts: **{better}**")
    if s_a and s_b and llm_a["elapsed"] and llm_b["elapsed"]:
        eff_a = llm_a["elapsed"] / max(len(s_a), 1)
        eff_b = llm_b["elapsed"] / max(len(s_b), 1)
        if abs(eff_a - eff_b) / max(eff_a, eff_b) > 0.15:
            better = name_a if eff_a < eff_b else name_b
            verdict.append(f"- cheaper per chapter: **{better}** ({min(eff_a, eff_b)/60:.0f}m vs {max(eff_a, eff_b)/60:.0f}m)")
    lines.extend(verdict if verdict else ["- no decisive deterministic difference"])
    lines.append("")
    return "\n".join(lines)


def cmd_compare(name_a: str, name_b: str, ch_from: int = 0, ch_to: int = 0) -> int:
    report = compare_novels(name_a, name_b, ch_from=ch_from, ch_to=ch_to)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    suffix = f"_ch{ch_from or 1}-{ch_to}" if (ch_from or ch_to) else ""
    out = EXPERIMENTS_DIR / f"{name_a}_vs_{name_b}{suffix}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[compare] report saved -> {out}")
    return 0


# ----------------------------------------------------------------------------
# ablate
# ----------------------------------------------------------------------------
def _flip_value(val: str) -> str:
    v = val.strip().lower()
    if v == "true":
        return "false"
    if v == "false":
        return "true"
    raise SystemExit(
        f"[ablate] value {val!r} is not a boolean; pass --set <value> to override explicitly."
    )


def _set_config_key(text: str, bare_key: str, new_val: Any, *, section: str = "novel") -> str:
    """Set `  key: value` in the flat hand-rolled config subset, inserting under
    ``section:`` when absent. Comments on the line are preserved."""
    pat = re.compile(rf"^(\s+{re.escape(bare_key)}:\s*)(.+?)(\s*(?:#.*)?)$", re.M)
    if pat.search(text):
        return pat.sub(lambda mm: f"{mm.group(1)}{new_val}{mm.group(3)}", text, count=1)
    return re.sub(rf"^({re.escape(section)}:\s*)$", rf"\g<1>\n  {bare_key}: {new_val}",
                  text, count=1, flags=re.M)


def _get_config_key(text: str, bare_key: str) -> str | None:
    m = re.search(rf"^\s+{re.escape(bare_key)}:\s*(.+?)\s*(?:#.*)?$", text, re.M)
    return m.group(1).strip() if m else None


def cmd_fork(name: str, as_name: str, flip_key: str | None, set_value: str | None,
             chapters: int) -> int:
    """Fork a novel AT ITS CURRENT HEAD into a new novel that continues from there.

    Why this exists: ``ablate`` starts every arm at Ch1, and this repo's own
    recorded lesson (memory ``ab_short_chapter_score_inflation``) is that short
    opening runs fake positive results — the failure modes worth measuring
    (mid-book collapse, fossil accumulation, replan storms) only appear past
    ~Ch20. A fork gives both arms an identical mid-book starting state, so the
    only difference between them is the flipped key.

    Fork is at HEAD, not at an arbitrary chapter: the memory markdown files and
    the entity/thread tables describe the book as of its last written chapter,
    and there is no faithful way to roll them back to Ch N. Fork from a novel
    that is already sitting where you want to start.

    ``logs/`` is deliberately NOT copied (except the RAG index): FPY, cost and
    reasoning-coverage stats then describe ONLY the chapters this fork writes,
    with no contamination from the source's history.
    """
    import shutil

    src = NOVELS_DIR / name
    if not src.exists():
        raise SystemExit(f"[fork] novel not found: {src}")
    cfg_path = src / "config.yaml"
    prompt_path = src / "prompt.md"
    if not cfg_path.exists() or not prompt_path.exists():
        raise SystemExit(f"[fork] {name} missing config.yaml or prompt.md")
    target = NOVELS_DIR / as_name
    if target.exists():
        raise SystemExit(f"[fork] {target} already exists; delete it first.")

    src_chapters = sorted((src / "chapters").glob("*.md")) if (src / "chapters").is_dir() else []
    head = len(src_chapters)
    if head == 0:
        raise SystemExit(f"[fork] {name} has no chapters to fork from.")

    text = cfg_path.read_text(encoding="utf-8")

    flip_note = ""
    if flip_key:
        bare = flip_key.split(".")[-1]
        old_val = _get_config_key(text, bare)
        if old_val is None:
            raise SystemExit(f"[fork] key {bare!r} not found in {cfg_path}")
        new_val = set_value if set_value is not None else _flip_value(old_val)
        text = _set_config_key(text, bare, new_val)
        flip_note = f"{flip_key}: {old_val} -> {new_val}"

    # Budget the run to `chapters` MORE chapters. Prefer raising target_words
    # over setting max_chapters: max_chapters switches on the ending-aware
    # machinery (CLOSING_RULES_BLOCK, ending-zone planning, hook-revise skip),
    # which would make the tail of the run unrepresentative of normal mid-book
    # behaviour. When the source already has max_chapters, extend that instead.
    chapter_words = int(_get_config_key(text, "chapter_words") or 3000)
    book_chars = len((src / "book.md").read_text(encoding="utf-8", errors="replace")) \
        if (src / "book.md").exists() else head * chapter_words
    src_max_ch = _get_config_key(text, "max_chapters")
    if src_max_ch and str(src_max_ch).strip() not in ("", "0"):
        budget_note = f"max_chapters: {src_max_ch} -> {head + chapters}"
        text = _set_config_key(text, "max_chapters", head + chapters)
    else:
        new_target = book_chars + chapters * chapter_words
        budget_note = f"target_words: {_get_config_key(text, 'target_words')} -> {new_target}"
        text = _set_config_key(text, "target_words", new_target)

    text = text.replace(f"novels/{name}/", f"novels/{as_name}/")

    target.mkdir(parents=True)
    (target / "config.yaml").write_text(text, encoding="utf-8")
    (target / "prompt.md").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
    for sub_dir in ("memory", "chapters"):
        if (src / sub_dir).is_dir():
            shutil.copytree(src / sub_dir, target / sub_dir)
        else:
            (target / sub_dir).mkdir()
    for fname in ("book.md", "state.md", "title.md", "story_state.db"):
        if (src / fname).exists():
            shutil.copy2(src / fname, target / fname)
    (target / "logs").mkdir()
    # The RAG index is the one logs/ artifact that carries story content rather
    # than run history — without it the fork's writer loses retrieval over
    # Ch1..head. Everything else in logs/ stays behind on purpose.
    # Current layout is sharded (retrieval_index/ch*.json + _df.json); older
    # novels still carry the monolithic file. Copy whichever exists.
    if (src / "logs" / "retrieval_index").is_dir():
        shutil.copytree(src / "logs" / "retrieval_index", target / "logs" / "retrieval_index")
    if (src / "logs" / "retrieval_index.json").exists():
        shutil.copy2(src / "logs" / "retrieval_index.json", target / "logs" / "retrieval_index.json")

    meta = {
        "source": name,
        "fork_at_chapter": head,
        "chapters_to_write": chapters,
        "flip": flip_note or None,
        "budget": budget_note,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    (EXPERIMENTS_DIR / f"fork_{as_name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[fork] {name} @ Ch{head} -> {target}")
    if flip_note:
        print(f"[fork]   {flip_note}")
    print(f"[fork]   {budget_note}   (writes Ch{head + 1}..Ch{head + chapters})")
    print(f"[fork] next steps:")
    print(f"[fork]   python novel.py run {as_name}")
    print(f"[fork]   python novel.py stats {as_name}        # check FPY + reasoning coverage")
    return 0


def cmd_ablate(name: str, flip_key: str, set_value: str | None, chapters: int) -> int:
    """Scaffold an ablation copy of a novel: same prompt, ONE config key changed,
    chapter-capped so the run is cheap. The copy is a normal novel directory —
    run it with `novel.py run <name>__ablate_<key>` and evaluate with
    `novel.py compare <name> <name>__ablate_<key>`.
    """
    src = NOVELS_DIR / name
    if not src.exists():
        raise SystemExit(f"[ablate] novel not found: {src}")
    cfg_path = src / "config.yaml"
    prompt_path = src / "prompt.md"
    if not cfg_path.exists() or not prompt_path.exists():
        raise SystemExit(f"[ablate] {name} missing config.yaml or prompt.md")

    safe_key = flip_key.replace(".", "_")
    ab_name = f"{name}__ablate_{safe_key}"
    target = NOVELS_DIR / ab_name
    if target.exists():
        raise SystemExit(f"[ablate] {target} already exists; delete it first.")

    text = cfg_path.read_text(encoding="utf-8")
    # Match `  key: value` once (config.yaml is the flat hand-rolled subset).
    bare_key = flip_key.split(".")[-1]
    pat = re.compile(rf"^(\s+{re.escape(bare_key)}:\s*)(.+?)(\s*(?:#.*)?)$", re.M)
    m = pat.search(text)
    if not m:
        raise SystemExit(f"[ablate] key {bare_key!r} not found in {cfg_path}")
    old_val = m.group(2).strip()
    new_val = set_value if set_value is not None else _flip_value(old_val)
    text = pat.sub(lambda mm: f"{mm.group(1)}{new_val}{mm.group(3)}", text, count=1)

    # Cap chapters so the ablation run is cheap and deterministic in length.
    if re.search(r"^\s+max_chapters:", text, re.M):
        text = re.sub(r"^(\s+max_chapters:\s*).+$", rf"\g<1>{chapters}", text, count=1, flags=re.M)
    else:
        text = re.sub(r"^(novel:\s*)$", rf"\g<1>\n  max_chapters: {chapters}", text, count=1, flags=re.M)

    # Re-point every paths: entry into the ablation directory.
    text = text.replace(f"novels/{name}/", f"novels/{ab_name}/")

    target.mkdir(parents=True)
    (target / "memory").mkdir()
    (target / "chapters").mkdir()
    (target / "logs").mkdir()
    (target / "config.yaml").write_text(text, encoding="utf-8")
    (target / "prompt.md").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")

    meta = {
        "source": name,
        "flip_key": flip_key,
        "old_value": old_val,
        "new_value": new_val,
        "max_chapters": chapters,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    (EXPERIMENTS_DIR / f"ablate_{ab_name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ablate] scaffolded {target}")
    print(f"[ablate]   {flip_key}: {old_val} -> {new_val}   (max_chapters={chapters})")
    print(f"[ablate] next steps:")
    print(f"[ablate]   python novel.py run {ab_name}")
    print(f"[ablate]   python novel.py compare {name} {ab_name}")
    return 0
