"""Offline replay of the L0 repair ladder over already-written chapters.

Zero LLM calls, read-only. Pairs each archived `final_review.json` payload (the
REAL gate output for that chapter) with the chapter text on disk, runs
`fix.apply_l0`, and reports the before/after deltas of the metrics L0 targets.

The point is to see whether L0 actually moves the metrics — and whether it
damages the text — before it is allowed anywhere near a live run.

    python tools/replay_l0.py [--novel NAME] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.quality as fix  # noqa: E402
from engine.quality import style_health  # noqa: E402

CONFIG = {"novel": {}}  # gate defaults only: the replay must not depend on a novel's tuning


def _metrics(text: str) -> dict:
    m = style_health(text, CONFIG)
    return {
        "penalty": float(m.get("penalty", 0.0) or 0.0),
        "em_dash": float(m.get("metrics", {}).get("em_dash_per_kchar", 0.0) or 0.0),
        "frag": float(m.get("metrics", {}).get("fragment_line_ratio", 0.0) or 0.0),
        "chars": len(text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel", default="*")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    paths = sorted((ROOT / "novels").glob(f"{args.novel}/logs/checkpoints/ch*/final_review.json"))
    for rp in paths:
        novel_dir = rp.parents[3]
        ch = rp.parent.name.replace("ch", "")
        chapter_file = novel_dir / "chapters" / f"{ch}.md"
        if not chapter_file.exists():
            continue
        try:
            review = json.loads(rp.read_text(encoding="utf-8")).get("payload") or {}
            text = chapter_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if not text.strip():
            continue
        planned = fix.plan_repairs(review, CONFIG)
        if not any(s["layer"] == "L0" for s in planned):
            continue
        before = _metrics(text)
        fixed, applied = fix.apply_l0(text, review, CONFIG, chapter_num=int(ch))
        if not applied:
            continue
        after = _metrics(fixed)
        rows.append((novel_dir.name, int(ch), before, after, applied))
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("no chapter had an L0-repairable gate firing")
        return 0

    print(f"{'novel':26} {'ch':>4}  {'penalty':>14} {'em_dash/k':>16} {'frag_ratio':>14} "
          f"{'len%':>7}  actions")
    worse = 0
    length_out_of_bound = 0
    for novel, ch, b, a, applied in rows:
        dlen = (a["chars"] - b["chars"]) / max(1, b["chars"]) * 100
        if a["penalty"] > b["penalty"]:
            worse += 1
        if abs(dlen) > 2.0:
            length_out_of_bound += 1
        print(f"{novel:26} {ch:>4}  {b['penalty']:6.2f}->{a['penalty']:5.2f} "
              f"{b['em_dash']:7.2f}->{a['em_dash']:6.2f} "
              f"{b['frag']:6.2f}->{a['frag']:6.2f} {dlen:+6.2f}%  {','.join(applied)}")

    n = len(rows)
    print(f"\nchapters repaired: {n}")
    for key, label in (("penalty", "style penalty"), ("em_dash", "em-dash /kchar"), ("frag", "fragment ratio")):
        b = sum(r[2][key] for r in rows) / n
        a = sum(r[3][key] for r in rows) / n
        print(f"  mean {label:16} {b:7.3f} -> {a:7.3f}  ({a - b:+.3f})")
    print(f"  chapters made WORSE by L0:      {worse}")
    print(f"  chapters with |len delta| > 2%: {length_out_of_bound}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
