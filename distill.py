"""P1-2: Cross-book experience distillation (跨书经验蒸馏).

Scans all novels' story_state.db + agent_reports for failure→fix patterns
and synthesizes global craft rules consumable by the planning/writing pipeline.

Usage:
    python -m distill --output craft_rules.json [--genre <genre>] [--min-novels 3]

Output schema:
{
  "rules": [
    {
      "id": "unique-rule-id",
      "category": "beat_execution|payoff_setup|hook_technique|character_consistency|world_logic|style|other",
      "pattern": "具体的失败模式（从多本书中反复出现）",
      "fix": "对应的修复策略",
      "evidence_count": N,
      "avg_score_before": X.X,
      "avg_score_after": Y.Y,
      "source_novels": ["novel_a", "novel_b", ...],
      "confidence": 0.0-1.0
    }
  ],
  "meta": {
    "generated_at": "ISO timestamp",
    "genre": "genre filter or '_all'",
    "novels_scanned": N,
    "total_chapters": M
  }
}

Integration points:
- planning.py: inject high-confidence rules as required_constraints hints
- writing.py: inject style/beat rules as writer_directives
- review.py: use rules to calibrate scoring thresholds per category
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent


def scan_novels(genre_filter: str | None = None, min_novels: int = 3) -> dict[str, Any]:
    """Scan all novels/<name>/story_state.db for failure→fix patterns.

    Returns aggregated patterns with evidence counts and score deltas.
    """
    novels_dir = ROOT / "novels"
    if not novels_dir.exists():
        return {"rules": [], "meta": {"novels_scanned": 0, "total_chapters": 0}}

    novel_dirs = [d for d in novels_dir.iterdir() if d.is_dir()]
    scanned_count = 0
    total_chapters = 0

    # Aggregate patterns: key = (category, pattern_text), value = evidence list
    pattern_evidence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for novel_dir in novel_dirs:
        db_path = novel_dir / "story_state.db"
        if not db_path.exists():
            continue

        # Optional genre filter
        config_path = novel_dir / "config.yaml"
        if genre_filter and config_path.exists():
            try:
                config_text = config_path.read_text(encoding="utf-8")
                genre_match = re.search(r"^\s*genre:\s*(.+)$", config_text, re.MULTILINE)
                novel_genre = genre_match.group(1).strip() if genre_match else "_default"
                if genre_filter != "_all" and novel_genre != genre_filter:
                    continue
            except Exception:
                pass

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Load per-chapter scores so distilled rules can carry a REAL
            # before/after delta instead of a 0.0 placeholder. A rule's "fix"
            # is logged at chapter C; the chapter that actually applies the fix
            # is C+1. We treat score(C) as "before" and score(C+1) as "after".
            chapter_score: dict[int, float] = {}
            try:
                for mrow in conn.execute(
                    "SELECT chapter, score FROM chapter_metrics WHERE score IS NOT NULL"
                ).fetchall():
                    try:
                        chapter_score[int(mrow["chapter"])] = float(mrow["score"])
                    except (TypeError, ValueError):
                        continue
            except sqlite3.OperationalError:
                chapter_score = {}

            # Scan agent_reports for gate_rejects and fixes
            try:
                rows = conn.execute(
                    "SELECT chapter, report_type, content FROM agent_reports WHERE report_type='review' ORDER BY chapter"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

            for row in rows:
                chapter = row["chapter"]
                content_str = row["content"]
                try:
                    content = json.loads(content_str)
                except json.JSONDecodeError:
                    continue
                try:
                    _ch_int = int(chapter)
                except (TypeError, ValueError):
                    _ch_int = None
                score_before = chapter_score.get(_ch_int) if _ch_int is not None else None
                score_after = chapter_score.get(_ch_int + 1) if _ch_int is not None else None

                total_chapters += 1

                # Extract gate_rejects
                gate_rejects = content.get("gate_rejects", [])
                if isinstance(gate_rejects, list):
                    for gr in gate_rejects:
                        if not isinstance(gr, dict):
                            continue
                        gate = gr.get("gate", "")
                        directives = gr.get("directives", [])
                        evidence_data = gr.get("evidence", {})

                        # Map gate to category
                        category = "other"
                        if "repeat" in gate.lower() or "fossil" in gate.lower():
                            category = "style"
                        elif "beat" in gate.lower():
                            category = "beat_execution"
                        elif "adjacent" in gate.lower():
                            category = "style"

                        pattern = f"gate_reject: {gate}"
                        for directive in directives[:2]:
                            key = (category, pattern)
                            pattern_evidence[key].append({
                                "novel": novel_dir.name,
                                "chapter": chapter,
                                "fix": str(directive),
                                "evidence": evidence_data,
                                "score_before": score_before,
                                "score_after": score_after,
                            })

                # Extract problems with fixes
                problems = content.get("problems", [])
                fixes = content.get("fixes", [])
                if isinstance(problems, list) and isinstance(fixes, list):
                    for prob, fix in zip(problems[:3], fixes[:3]):
                        prob_text = str(prob).strip()
                        fix_text = str(fix).strip()
                        if not prob_text or not fix_text:
                            continue

                        # Infer category from problem text
                        category = "other"
                        if any(kw in prob_text for kw in ["钩子", "hook", "章末"]):
                            category = "hook_technique"
                        elif any(kw in prob_text for kw in ["兑现", "payoff", "爽点"]):
                            category = "payoff_setup"
                        elif any(kw in prob_text for kw in ["人物", "角色", "character"]):
                            category = "character_consistency"
                        elif any(kw in prob_text for kw in ["世界观", "逻辑", "world"]):
                            category = "world_logic"
                        elif any(kw in prob_text for kw in ["beat", "节拍", "场景"]):
                            category = "beat_execution"
                        elif any(kw in prob_text for kw in ["文体", "style", "破折号", "句式"]):
                            category = "style"

                        key = (category, prob_text[:100])
                        pattern_evidence[key].append({
                            "novel": novel_dir.name,
                            "chapter": chapter,
                            "fix": fix_text[:200],
                            "evidence": {},
                            "score_before": score_before,
                            "score_after": score_after,
                        })

            conn.close()
            scanned_count += 1

        except Exception:
            continue

    # Synthesize rules from patterns with sufficient evidence
    rules: list[dict[str, Any]] = []
    for (category, pattern_text), evidences in pattern_evidence.items():
        if len(evidences) < min_novels:
            continue

        # Deduplicate source novels
        source_novels = list(set(e["novel"] for e in evidences))
        if len(source_novels) < min_novels:
            continue

        # Aggregate fixes
        fix_counts: dict[str, int] = defaultdict(int)
        for e in evidences:
            fix_counts[e["fix"]] += 1

        # Pick most common fix
        most_common_fix = max(fix_counts.items(), key=lambda x: x[1])[0] if fix_counts else ""

        # Confidence: based on evidence count and novel diversity
        confidence = min(1.0, len(evidences) / 10.0 * len(source_novels) / max(scanned_count, 1))

        rule_id = f"{category}_{len(rules) + 1}"
        befores = [e["score_before"] for e in evidences if isinstance(e.get("score_before"), (int, float))]
        afters = [e["score_after"] for e in evidences if isinstance(e.get("score_after"), (int, float))]
        avg_before = round(sum(befores) / len(befores), 2) if befores else 0.0
        avg_after = round(sum(afters) / len(afters), 2) if afters else 0.0
        rules.append({
            "id": rule_id,
            "category": category,
            "pattern": pattern_text,
            "fix": most_common_fix,
            "evidence_count": len(evidences),
            "avg_score_before": avg_before,
            "avg_score_after": avg_after,
            "source_novels": source_novels[:10],
            "confidence": round(confidence, 2),
        })

    # Sort by evidence_count desc
    rules.sort(key=lambda r: r["evidence_count"], reverse=True)

    return {
        "rules": rules,
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "genre": genre_filter or "_all",
            "novels_scanned": scanned_count,
            "total_chapters": total_chapters,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="P1-2: Distill cross-book craft rules")
    parser.add_argument("--output", default="craft_rules.json", help="Output JSON file")
    parser.add_argument("--genre", default=None, help="Filter by genre (or '_all' for no filter)")
    parser.add_argument("--min-novels", type=int, default=3, help="Minimum novels for a rule")
    args = parser.parse_args()

    print(f"Scanning novels directory: {ROOT / 'novels'}")
    print(f"Genre filter: {args.genre or 'none'}")
    print(f"Minimum novels per rule: {args.min_novels}")

    result = scan_novels(genre_filter=args.genre, min_novels=args.min_novels)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ Distilled {len(result['rules'])} rules from {result['meta']['novels_scanned']} novels")
    print(f"  Total chapters scanned: {result['meta']['total_chapters']}")
    print(f"  Output written to: {output_path}")

    # Print top 5 rules
    print("\nTop 5 rules by evidence:")
    for rule in result["rules"][:5]:
        print(f"  [{rule['category']}] {rule['pattern'][:60]}...")
        print(f"    → {rule['fix'][:60]}... (confidence={rule['confidence']}, evidence={rule['evidence_count']})")


if __name__ == "__main__":
    main()
