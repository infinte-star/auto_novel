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


def _load_panel_series(nd: Path) -> list[tuple[int, float, float]]:
    """Read reader-panel reports (chapter, excitement, drop_rate) from the
    novel's own store — the retention signal for P0. Prefers genre-weighted
    values (weighted_excitement/weighted_drop_rate) when present, else raw.
    Oldest-first; tolerant of missing db/rows."""
    db = nd / "story_state.db"
    if not db.exists():
        return []
    out: list[tuple[int, float, float]] = []
    try:
        import sqlite3
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT chapter, payload FROM events WHERE event_type='panel_report' ORDER BY chapter"
        ).fetchall()
        conn.close()
    except Exception:
        return []
    for ch, payload in rows:
        try:
            p = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        ex = p.get("weighted_excitement", p.get("avg_excitement"))
        dr = p.get("weighted_drop_rate", p.get("drop_rate"))
        try:
            out.append((int(p.get("chapter", ch)), float(ex if ex is not None else 5.0),
                        float(dr if dr is not None else 0.0)))
        except (TypeError, ValueError):
            continue
    return out


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


def _llm_totals(nd: Path) -> dict[str, float]:
    """Total calls / seconds / output chars, plus the planning-stage share."""
    path = nd / "logs" / "llm_calls.jsonl"
    tot = {"calls": 0.0, "elapsed": 0.0, "output": 0.0, "plan_elapsed": 0.0, "fail": 0.0}
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


def compare_novels(name_a: str, name_b: str, *, judge: bool = False,
                   client: Any = None, paths: Any = None, config: Any = None) -> str:
    """Build a markdown comparison report for two novels; returns the report.

    When judge=True (and a client/paths/config are supplied), also runs a blind
    pairwise LLM judge over matched chapters and appends its verdict. The default
    (judge=False) path stays entirely deterministic / zero-LLM.
    """
    nd_a, nd_b = NOVELS_DIR / name_a, NOVELS_DIR / name_b
    for nd in (nd_a, nd_b):
        if not nd.exists():
            raise SystemExit(f"[compare] novel directory not found: {nd}")

    m_a, m_b = _load_chapter_metrics(nd_a), _load_chapter_metrics(nd_b)
    log_a, log_b = _scan_run_log(nd_a), _scan_run_log(nd_b)
    llm_a, llm_b = _llm_totals(nd_a), _llm_totals(nd_b)
    ev_types = ("quality_debt", "gate_reject", "scene_dedupe_retry", "visual_payoff_retry")
    ev_a, ev_b = _load_events(nd_a, ev_types), _load_events(nd_b, ev_types)

    # Retention (P0): the reader-panel-derived objective the per-chapter score is
    # blind to. Summarized deterministically via the shared retention module.
    from retention import summarize_retention, retention_by_block
    ps_a, ps_b = _load_panel_series(nd_a), _load_panel_series(nd_b)
    ret_a = summarize_retention([s[1] for s in ps_a], [s[2] for s in ps_a])
    ret_b = summarize_retention([s[1] for s in ps_b], [s[2] for s in ps_b])

    def scores(ms: list[dict[str, Any]]) -> list[float]:
        return [float(r["score"]) for r in ms if r.get("score") is not None]

    s_a, s_b = scores(m_a), scores(m_b)

    lines: list[str] = []
    lines.append(f"# Compare: {name_a} vs {name_b}")
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
    lines.append(row("retention index (0-10)", ret_a["retention_index"], ret_b["retention_index"]))
    lines.append(row("panel mean excitement", ret_a["mean_excitement"], ret_b["mean_excitement"]))
    lines.append(row("panel mean drop_rate", ret_a["mean_drop"], ret_b["mean_drop"]))
    lines.append(row("excitement troughs (<4)", ret_a["trough_count"], ret_b["trough_count"]))
    lines.append(row("force-accepts (log)", log_a["force_accept"], log_b["force_accept"]))
    lines.append(row("quality_debt events", ev_a["quality_debt"], ev_b["quality_debt"]))
    lines.append(row("gate_reject events", ev_a["gate_reject"], ev_b["gate_reject"]))
    lines.append(row("fossil warnings (log)", log_a["fossil_hits"], log_b["fossil_hits"]))
    lines.append(row("max fossils in one hit", log_a["max_fossils"], log_b["max_fossils"]))
    lines.append(row("scene-dedupe WARN", log_a["scene_dedupe_warn"], log_b["scene_dedupe_warn"]))
    lines.append(row("scene-dedupe BLOCK", log_a["scene_dedupe_block"], log_b["scene_dedupe_block"]))
    lines.append(row("LLM calls", int(llm_a["calls"]), int(llm_b["calls"])))
    lines.append(row("LLM total minutes", llm_a["elapsed"] / 60, llm_b["elapsed"] / 60))
    lines.append(row("planning share of LLM time",
                     (llm_a["plan_elapsed"] / llm_a["elapsed"]) if llm_a["elapsed"] else None,
                     (llm_b["plan_elapsed"] / llm_b["elapsed"]) if llm_b["elapsed"] else None))
    lines.append(row("LLM minutes / scored chapter",
                     (llm_a["elapsed"] / 60 / len(s_a)) if s_a else None,
                     (llm_b["elapsed"] / 60 / len(s_b)) if s_b else None))
    lines.append("")

    # --- per-chapter score curves ---
    lines.append("## Per-chapter scores")
    lines.append(f"| ch | {name_a} | style_pen | {name_b} | style_pen |")
    lines.append("|---|---|---|---|---|")
    by_a = {int(r["chapter"]): r for r in m_a}
    by_b = {int(r["chapter"]): r for r in m_b}
    for ch in sorted(set(by_a) | set(by_b)):
        ra, rb = by_a.get(ch), by_b.get(ch)
        lines.append(
            f"| {ch} | {_fmt(ra.get('score') if ra else None, 1)} "
            f"| {_fmt(ra.get('style_penalty') if ra else None, 1)} "
            f"| {_fmt(rb.get('score') if rb else None, 1)} "
            f"| {_fmt(rb.get('style_penalty') if rb else None, 1)} |"
        )
    lines.append("")

    # --- retention curve (per-10-chapter block) ---
    blk_a = {b["ch_from"]: b for b in retention_by_block(ps_a, 10)}
    blk_b = {b["ch_from"]: b for b in retention_by_block(ps_b, 10)}
    if blk_a or blk_b:
        lines.append("## Retention curve (per 10 ch: index / excitement / drop)")
        lines.append(f"| chapters | {name_a} | {name_b} |")
        lines.append("|---|---|---|")
        for cf in sorted(set(blk_a) | set(blk_b)):
            ba, bb = blk_a.get(cf), blk_b.get(cf)
            def cell(b: dict[str, Any] | None) -> str:
                if not b:
                    return "-"
                return f"{_fmt(b['retention_index'],1)} / {_fmt(b['mean_excitement'],1)} / {_fmt(b['mean_drop'],2)}"
            lines.append(f"| {cf}-{cf+9} | {cell(ba)} | {cell(bb)} |")
        lines.append("")

    # --- config diff (non-secret, non-path keys) ---
    cfg_a, cfg_b = _read_config_lines(nd_a), _read_config_lines(nd_b)
    diffs = []
    for key in sorted(set(cfg_a) | set(cfg_b)):
        va, vb = cfg_a.get(key, "<absent>"), cfg_b.get(key, "<absent>")
        if va != vb:
            diffs.append(f"| {key} | {va} | {vb} |")
    lines.append("## Config differences")
    if diffs:
        lines.append(f"| key | {name_a} | {name_b} |")
        lines.append("|---|---|---|")
        lines.extend(diffs)
    else:
        lines.append("(identical apart from paths/keys)")
    lines.append("")

    # --- optional blind pairwise LLM judge (only when --judge + client) ---
    judge_result: dict[str, Any] | None = None
    if judge and client is not None and paths is not None and config is not None:
        try:
            import judge as judge_mod
            judge_result = judge_mod.judge_ablation(client, paths, config, name_a, name_b)
            lines.append("## Pairwise judge (blind, order-swapped)")
            if judge_result.get("error"):
                lines.append(f"(judge skipped: {judge_result['error']})")
            else:
                jw, jl, jt = judge_result["a_wins"], judge_result["b_wins"], judge_result["ties"]
                lines.append(f"Judged {judge_result['judged']} matched chapters (A={name_a}, B={name_b}).")
                lines.append(f"| winner | {name_a} | {name_b} | tie |")
                lines.append("|---|---|---|---|")
                lines.append(f"| chapters | {jw} | {jl} | {jt} |")
                lines.append("")
                lines.append("| ch | winner | reason |")
                lines.append("|---|---|---|")
                for r in judge_result["per_chapter"]:
                    who = {"a": name_a, "b": name_b, "tie": "tie"}.get(r["winner"], "tie")
                    lines.append(f"| {r['chapter']} | {who} | {str(r.get('reason',''))[:70]} |")
            lines.append("")
        except Exception as exc:  # never let the judge crash the deterministic report
            lines.append("## Pairwise judge (blind, order-swapped)")
            lines.append(f"(judge failed: {exc}; see deterministic metrics above)")
            lines.append("")

    # --- verdict heuristics ---
    lines.append("## Heuristic verdict")
    verdict: list[str] = []
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
    ri_a, ri_b = ret_a["retention_index"], ret_b["retention_index"]
    if ri_a is not None and ri_b is not None and abs(ri_a - ri_b) >= 0.5:
        better = name_a if ri_a > ri_b else name_b
        verdict.append(f"- higher retention index: **{better}** ({max(ri_a, ri_b):.1f} vs {min(ri_a, ri_b):.1f})")
    if judge_result and not judge_result.get("error"):
        jw, jl = judge_result["a_wins"], judge_result["b_wins"]
        if jw != jl:
            better = name_a if jw > jl else name_b
            verdict.append(f"- blind pairwise judge favors **{better}** ({max(jw, jl)}-{min(jw, jl)}-{judge_result['ties']} W-L-T)")
        else:
            verdict.append(f"- blind pairwise judge: tie ({jw}-{jl}-{judge_result['ties']})")
    if s_a and s_b and llm_a["elapsed"] and llm_b["elapsed"]:
        eff_a = llm_a["elapsed"] / max(len(s_a), 1)
        eff_b = llm_b["elapsed"] / max(len(s_b), 1)
        if abs(eff_a - eff_b) / max(eff_a, eff_b) > 0.15:
            better = name_a if eff_a < eff_b else name_b
            verdict.append(f"- cheaper per chapter: **{better}** ({min(eff_a, eff_b)/60:.0f}m vs {max(eff_a, eff_b)/60:.0f}m)")
    lines.extend(verdict if verdict else ["- no decisive deterministic difference; consider a blind pairwise judge run"])
    lines.append("")
    return "\n".join(lines)


def cmd_compare(name_a: str, name_b: str, *, judge: bool = False,
                client: Any = None, paths: Any = None, config: Any = None) -> int:
    report = compare_novels(name_a, name_b, judge=judge, client=client, paths=paths, config=config)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / f"{name_a}_vs_{name_b}.md"
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
