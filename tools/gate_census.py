"""Gate census: which deterministic gates actually fire, and how much they cost.

Zero LLM calls, read-only. Walks every archived
`novels/*/logs/checkpoints/ch*/final_review.json` payload and counts, per gate:
how often it ran, how often it produced a non-zero penalty (or a block/reject),
and the total/average penalty it contributed.

This exists so the "35 gates -> 8 gates" deletion decision in REDESIGN.md §2 L6
is made from measured data instead of by eye. A gate that ran 600+ times and
never fired is a deletion candidate; a gate that never ran in this sample is NOT
(it may simply be disabled in every novel's config, or genre-specific).

    python tools/gate_census.py [--novel NAME] [--min-ran N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quality import REGISTRY  # noqa: E402

# Gates whose report key in the review payload differs from the gate name.
REPORT_KEY = {"book_wide_fossils": "book_fossils", "length_band_check": "length_band"}
# A gate "fired" if it said ANYTHING. The gates do not share one result shape:
# some carry a penalty, some only a level, and the advisory ones carry neither --
# `information_density` returns {low_information, signals, directives} and would
# score a structural 0/649 against a penalty-only test, which reads as "dead gate"
# when it actually means "never measured". Cover every shape in use.
#
# `repeat` was MISSING until 2026-07-28, which made `hook_tail_repetition`'s only
# verdict invisible: it returns {repeat, ratio, repeated_clauses} and read a
# structural 0/638 while it had in fact fired once. That is the same blind spot
# `emotional_cadence`'s `monotony` had, and it is dangerous in one direction only —
# a gate reported silent is a deletion candidate, so a missing key here can argue a
# live gate out of the codebase. When adding a gate, add its verdict key.
VERDICT_LIST_KEYS = ("flags", "phrases", "fossils", "flagged", "repeats", "template_fossils")
VERDICT_BOOL_KEYS = ("block", "blocked", "low_information", "stalled", "collapsed",
                     "repeat")
VERDICT_LEVELS = ("advise", "reject", "warn", "block")


def _fired(result) -> bool:
    """The gate reached a VERDICT: a penalty, a level, a block, or flagged spans.

    Deliberately NOT counting `directives` or `signals`. `signals` is a diagnostic
    list (`information_density` records all four probes and only calls the chapter
    low-information at >=3 of them, 6.8% of the time), and a directive with no
    penalty is an advisory nudge, not a finding. Conflating them reads 91% where
    the real firing rate is 7%.
    """
    if isinstance(result, list):
        return bool(result)
    if not isinstance(result, dict):
        return bool(result)
    if float(result.get("penalty", 0.0) or 0.0) > 0:
        return True
    if any(result.get(k) for k in VERDICT_BOOL_KEYS):
        return True
    if str(result.get("level", "")).strip() in VERDICT_LEVELS:
        return True
    return any(result.get(k) for k in VERDICT_LIST_KEYS)


def _advised(result) -> bool:
    """The gate injected a writer directive without reaching a verdict."""
    return isinstance(result, dict) and bool(result.get("directives"))


def _penalty(result) -> float:
    if isinstance(result, dict):
        try:
            return float(result.get("penalty", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel", default="*")
    ap.add_argument("--min-ran", type=int, default=1,
                    help="hide gates that ran fewer than N times (0-run gates are listed separately)")
    args = ap.parse_args()

    gates = list(REGISTRY.list_gates())
    stats = {g: {"ran": 0, "fired": 0, "advised": 0, "pen": 0.0} for g in gates}
    reviews = 0

    pattern = f"{args.novel}/logs/checkpoints/ch*/final_review.json"
    for rp in sorted((ROOT / "novels").glob(pattern)):
        try:
            payload = json.loads(rp.read_text(encoding="utf-8")).get("payload") or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        reviews += 1
        for gate in gates:
            result = payload.get(REPORT_KEY.get(gate, gate))
            if result is None:
                continue
            s = stats[gate]
            s["ran"] += 1
            if _fired(result):
                s["fired"] += 1
            if _advised(result):
                s["advised"] += 1
            s["pen"] += _penalty(result)

    if not reviews:
        print("no archived final_review.json payloads found")
        return 0

    rows = [(g, s) for g, s in stats.items() if s["ran"] >= args.min_ran]
    rows.sort(key=lambda r: (-r[1]["pen"], -r[1]["fired"], r[0]))

    print(f"reviews scanned: {reviews}\n")
    print(f"{'gate':32} {'layer':9} {'ran':>6} {'fired':>6} {'fire%':>6} "
          f"{'advise%':>7} {'tot_pen':>9} {'avg_pen':>8}")
    for gate, s in rows:
        ran = s["ran"]
        pct = (s["fired"] / ran * 100) if ran else 0.0
        avg = (s["pen"] / ran) if ran else 0.0
        adv = (s["advised"] / ran * 100) if ran else 0.0
        print(f"{gate:32} {REGISTRY.repair(gate):9} {ran:>6} {s['fired']:>6} "
              f"{pct:>5.1f}% {adv:>6.1f}% {s['pen']:>9.1f} {avg:>8.3f}")

    total_pen = sum(s["pen"] for _, s in rows)
    print(f"\ntotal penalty across all gates: {total_pen:.1f} "
          f"({total_pen / reviews:.3f} per review)")

    never = [g for g, s in rows if s["ran"] > 0 and s["fired"] == 0 and s["advised"] == 0]
    unseen = [g for g, s in stats.items() if s["ran"] == 0]
    if never:
        print(f"\nran but NEVER fired ({len(never)}) -- deletion candidates:")
        for g in never:
            print(f"  {g}  (ran {stats[g]['ran']})")
    if unseen:
        print(f"\nnever ran in this sample ({len(unseen)}) -- NOT deletion candidates "
              f"(disabled by config or genre-specific):")
        print("  " + ", ".join(sorted(unseen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
