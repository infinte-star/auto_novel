"""Which side of the length band did the repair planner actually route to L1?

Kept, not thrown away: it is the evidence behind giving `plan_repairs` a side
predicate (`fix._length_band_needs_expand`) instead of just adding the two log
lines the ticket asked for (REDESIGN_V2 §9.11.5, 2026-07-28). Re-run it before
changing how a two-sided gate reaches a one-sided fixer.

Zero LLM, read-only. Three independent passes, because each answers a question the
others cannot:

1. **Side census.** Replays `fix.plan_repairs` over every archived
   `review_round0.json` and splits the planned `expand_to_band` steps by the flag
   the gate actually emitted. `expand_to_band` handles the SHORT side only, so
   every long-side hit is a step that could never have changed a byte.
2. **Displacement.** `fix_max_l1_calls` is enforced inside `plan_repairs`, so a
   no-op planned first can evict a real fixer. This pass diffs the plan with and
   without the guard and reports only the chapters where the SET of L1 actions
   changed — the cases where the bug cost something rather than merely wasting a
   slot that nobody else wanted.
3. **Ledger reconciliation.** `llm_calls.jsonl` (append-only) minus the
   `v2_repair_l1` event log gives the number of paid L1 calls whose result was
   discarded, with no reliance on run.log — which is truncated on every launch.
   This is why the fix is a log line rather than a new event type: the COUNT was
   already derivable, only the REASON was missing.

Reading as of 2026-07-28 (before the fix): 195 planned expands, 86 short-side,
**109 (56%) long-side** — 285/113/172 (60%) with `--all`; 2 real displacements;
13 `fix_expand` calls against 7 `v2_repair_l1` events (`ts_v2arm` 7/5,
`ts_v2match` 6/2).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("NOVEL_CONFIG", "config.yaml")

import engine.quality as fix  # noqa: E402
from fpy_prime import _payload, discover_novels, print_exclusions  # noqa: E402

# The engine defaults, pinned: two arms with divergent configs must be judged by
# one ruler, exactly as `fpy_prime.PINNED` does for the release rule.
CONFIG = {"novel": {"fix_max_l1_calls": 2, "quality_threshold": 8.0}}


def _side(review: dict) -> str:
    """Which side of the band fired, read off the gate's own flag vocabulary."""
    result = fix.gate_result(review, "length_band_check")
    flags = [str(f) for f in ((result.get("flags") or []) if isinstance(result, dict) else [])]
    long_ = any(f.startswith("chapter_too_long") for f in flags)
    short = any(f.startswith("chapter_too_short") for f in flags)
    if long_ and not short:
        return "long"
    if short and not long_:
        return "short"
    return "unknown"  # both, or a vocabulary this script does not know


def _l1(review: dict, *, guard: bool) -> list[str]:
    """The L1 actions `plan_repairs` yields, with the side guard on or off."""
    real = fix._length_band_needs_expand
    if not guard:
        fix._length_band_needs_expand = lambda _r: True
    try:
        return [s["action"] for s in fix.plan_repairs(review, CONFIG)
                if s.get("layer") == "L1"]
    finally:
        fix._length_band_needs_expand = real


def _chapters(novel: Path):
    ck = novel / "logs" / "checkpoints"
    if not ck.is_dir():
        return
    for d in sorted(ck.iterdir(), key=lambda p: p.name):
        payload = _payload(d / "review_round0.json")
        if payload:
            yield d.name, payload


INCLUDE_ALL = "--all" in sys.argv

# Derivative dirs are dropped by default, same rule and same reason as the
# acceptance metric: with the forks in the pool one arm's chapters get counted
# twice and the long-side share moves 56% -> 60%. Pass `--all` for that reading.
names, dropped = discover_novels([], include_all=INCLUDE_ALL)
print_exclusions(dropped)

sides: dict[str, list[str]] = {"short": [], "long": [], "unknown": []}
displaced: list[tuple[str, str, list[str], list[str]]] = []

for name in names:
    novel = ROOT / "novels" / name
    for ch, review in _chapters(novel):
        planned = _l1(review, guard=False)
        if "expand_to_band" not in planned:
            continue
        sides[_side(review)].append(f"{name} {ch}")
        guarded = _l1(review, guard=True)
        # A no-op that merely wasted an unwanted slot is not a cost; a plan whose
        # OTHER actions changed is. Compare the sets, not the lists.
        if set(planned) - {"expand_to_band"} != set(guarded) - {"expand_to_band"}:
            displaced.append((name, ch, planned, guarded))

total = sum(len(v) for v in sides.values())
print(f"\n--- side census: {total} chapters planned expand_to_band ---")
for side, label in (("short", "short side (a real fixer)"),
                    ("long", "LONG side (guaranteed no-op)"),
                    ("unknown", "side not decidable from flags")):
    hits = sides[side]
    pct = f"{100 * len(hits) / total:.0f}%" if total else "-"
    print(f"  {label:>32}: {len(hits):>4}  {pct:>4}")
    if hits:
        print(f"{'':>34}  e.g. {', '.join(hits[:4])}")

print(f"\n--- displacement: {len(displaced)} chapter(s) lost a real fixer to the cap ---")
for name, ch, planned, guarded in displaced:
    print(f"  {name} {ch}\n    without guard: {planned}\n    with guard:    {guarded}")

# Pass 3. Two append-only ledgers, no run.log: the calls are in `llm_calls.jsonl`,
# the keeps are events. Their difference is the number of paid calls thrown away.
print("\n--- ledger: paid L1 expands vs kept L1 repairs ---")
for name in sorted({*names, *(p.parent.parent.name
                              for p in ROOT.glob("novels/*/logs/llm_calls.jsonl"))}):
    calls_f = ROOT / "novels" / name / "logs" / "llm_calls.jsonl"
    if not calls_f.exists():
        continue
    calls = 0
    for line in calls_f.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            if str(json.loads(line).get("tag", "")) == "fix_expand":
                calls += 1
        except Exception:
            continue
    if not calls:
        continue
    # Read-only, and deliberately NOT through `store.init_db`: that runs schema
    # migrations and flips WAL on, which is a write against a novel this script
    # is only measuring.
    events = 0
    try:
        import sqlite3

        db = ROOT / "novels" / name / "story_state.db"
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'v2_repair_l1'").fetchone()[0]
        conn.close()
    except Exception as exc:
        print(f"  {name}: {calls} fix_expand calls, events unreadable ({exc})")
        continue
    print(f"  {name:>14}: {calls:>3} fix_expand calls  {events:>3} v2_repair_l1 events"
          f"  -> {max(0, calls - events):>3} discarded with no record of why")
