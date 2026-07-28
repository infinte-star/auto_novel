"""CCR — did the prose stage what the plan promised? Zero LLM, read-only.

The second of v2's three metrics (FPY' / CCR / WR). FPY' asks whether a draft
carried a measured defect; CCR asks a question no gate in the engine currently
asks at all: **the plan named a place, a turn, a hook and a cast — did any of
them make it onto the page?** A chapter can clear every gate in `quality.py` and
still be about something else entirely, and today nothing would notice.

    python tools/ccr_baseline.py                     # every non-derivative novel
    python tools/ccr_baseline.py tangshuting --detail
    python tools/ccr_baseline.py ts_v1arm ts_v2arm   # the A/B read

TWO CAVEATS, both structural, both printed in the report so a number lifted out
of it carries them:

1. **PROXY CARDS.** Archived books ran the v1 committee, which has no
   ChapterCard (`arc_planning_enabled` is false everywhere). The proxy is the
   arbiter's `merged_plan` from `plan_initial_attempt0_arbitration.json`, mapped
   field-for-field below. It has no `turn` and no `forbid` column, so those two
   contract items are simply not measured on archived books — the proxy CCR is
   computed over `where / who / payoff / exit_hook / beats` only, and the hard
   set shrinks from {where, turn, exit_hook} to {where, exit_hook}. A v2 arm
   carries real cards and is measured on all seven; the report labels each novel
   PROXY or REAL and refuses to print a combined average across the two, because
   averaging a 5-item contract with a 7-item one produces a number that is not a
   rate of anything.

2. **SHIPPED TEXT, NOT FIRST DRAFT.** `chapters/NNNN.md` is what survived
   revision. Round-0 drafts are not archived as text. So this is an upper bound
   on first-draft CCR — revision can only have helped it.

Neither caveat is fixable by cleverness here, and both would be invisible in a
bare percentage, which is the whole reason they are in the header of every run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.fpy_prime import discover_novels, print_exclusions  # noqa: E402
from v2.accept import HARD_FIELDS, contract_fulfilment  # noqa: E402

# merged_plan -> ChapterCard. Only the fields `contract_fulfilment` reads.
#
# `turn` and `forbid` have NO merged_plan counterpart and are deliberately left
# unmapped rather than approximated: `conflict` is a situation, not a turning
# point, and scoring "did the prose contain the conflict description" under the
# name `turn` would report a made-up quantity under a real one. `goal` and
# `conflict` are likewise dropped -- they map to card `wants` / `blocked_by`,
# which the CCC does not judge (an intent is not stageable).
PROXY_MAP = {
    "where": "location",
    "payoff": "payoff",
    "exit_hook": "hook",
    "who": "character_focus",
    "beats": "beats",
}
PROXY_FIELDS = tuple(PROXY_MAP)
REAL_FIELDS = ("where", "turn", "payoff", "who", "exit_hook", "beats", "forbid")

# Where a v2 run archives the card it actually wrote against. `v2/run.py` must
# write this; until it does, every novel reads as PROXY. Keeping the contract in
# the measurement tool rather than only in the engine means a v2 arm that forgets
# to archive its cards shows up as PROXY in the report instead of silently being
# measured against a reconstruction of itself.
CARD_CHECKPOINT = "chapter_card.json"


def _payload(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return raw.get("payload") if isinstance(raw.get("payload"), dict) else raw


def real_card(ch_dir: Path) -> dict | None:
    pay = _payload(ch_dir / CARD_CHECKPOINT)
    if not pay:
        return None
    card = pay.get("card") if isinstance(pay.get("card"), dict) else pay
    return card if isinstance(card, dict) and card else None


def proxy_card(ch_dir: Path) -> dict | None:
    """The arbiter's attempt0 merged_plan, projected onto the card schema."""
    pay = _payload(ch_dir / "plan_initial_attempt0_arbitration.json")
    if not pay:
        return None
    plan = (pay.get("decision") or {}).get("merged_plan") or pay.get("plan")
    if not isinstance(plan, dict):
        return None
    card: dict = {}
    for card_field, plan_field in PROXY_MAP.items():
        v = plan.get(plan_field)
        if isinstance(v, list):
            v = [str(x).strip() for x in v if str(x).strip()]
            if v:
                card[card_field] = v
        elif isinstance(v, str) and v.strip():
            card[card_field] = v.strip()
    return card or None


def chapter_text(novel: Path, ch: int) -> str:
    f = novel / "chapters" / f"{ch:04d}.md"
    try:
        return f.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def measure(name: str, lo: int, hi: int) -> dict:
    novel = ROOT / "novels" / name
    base = novel / "logs" / "checkpoints"
    rows: list[dict] = []
    kinds: set[str] = set()
    if not base.is_dir():
        return {"name": name, "rows": rows, "kind": "none"}

    for ch_dir in sorted(base.glob("ch*")):
        if not ch_dir.is_dir():
            continue
        ch = int(re.sub(r"\D", "", ch_dir.name) or 0)
        if not (lo <= ch <= hi):
            continue
        card, kind = real_card(ch_dir), "REAL"
        if card is None:
            card, kind = proxy_card(ch_dir), "PROXY"
        if card is None:
            continue
        text = chapter_text(novel, ch)
        if len(text) < 500:
            continue
        r = contract_fulfilment(card, text, None)
        if not r["judgeable"]:
            continue
        kinds.add(kind)
        rows.append({"ch": ch, "kind": kind, **r})

    kind = kinds.pop() if len(kinds) == 1 else ("MIXED" if kinds else "none")
    return {"name": name, "rows": rows, "kind": kind}


def field_stats(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """field -> (hits, judgeable). Unjudgeable items are excluded, as in the CCR."""
    out: dict[str, list[int]] = {}
    for r in rows:
        for it in r["items"]:
            if it["verdict"] == "unjudgeable":
                continue
            slot = out.setdefault(it["field"], [0, 0])
            slot[1] += 1
            slot[0] += int(it["verdict"] == "hit")
    return {k: (v[0], v[1]) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("novels", nargs="*")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=10 ** 9)
    ap.add_argument("--detail", action="store_true", help="per-chapter misses")
    ap.add_argument("--all", action="store_true",
                    help="include derivative dirs excluded from the aggregate by default")
    args = ap.parse_args()

    names, dropped = discover_novels(args.novels, include_all=args.all)
    if not names:
        print("no novels with checkpoints found")
        return 2

    print("CCR = fulfilled / judgeable contract items, per chapter, zero LLM.")
    print("PROXY = merged_plan stood in for a ChapterCard: no `turn`, no `forbid`,")
    print("        hard set is {where, exit_hook} instead of "
          + "{" + ", ".join(HARD_FIELDS) + "}.")
    print("Measured on SHIPPED text (post-revision) -> upper bound on first-draft CCR.")
    print()
    print_exclusions(dropped)

    results = [measure(n, args.lo, args.hi) for n in names]
    results = [r for r in results if r["rows"]]
    if not results:
        print("no chapters with both a card/plan and a chapter text")
        return 2

    w = max(len(r["name"]) for r in results)
    print(f"{'novel':<{w}}  {'kind':<5}  {'n':>4}  {'CCR':>6}  {'pass':>11}  "
          f"{'hard miss':>10}  {'forbid':>7}")
    for r in results:
        rows = r["rows"]
        n = len(rows)
        ccr = sum(x["ccr"] for x in rows) / n
        passed = sum(1 for x in rows if x["passed"])
        hard = sum(len(x["hard_misses"]) for x in rows)
        viol = sum(len(x["violations"]) for x in rows)
        print(f"{r['name']:<{w}}  {r['kind']:<5}  {n:>4}  {ccr:>5.1%}  "
              f"{passed:>4}/{n:<3} {100.0 * passed / n:>3.0f}%  {hard:>10}  {viol:>7}")
        if args.detail:
            for x in rows:
                if x["passed"] and x["ccr"] >= 0.999:
                    continue
                miss = ", ".join(f"{m['field']}:{m['target'][:26]}" for m in x["missing"][:4])
                print(f"{'':<{w}}    Ch{x['ch']:<4} ccr={x['ccr']:.0%} "
                      f"{'HARD ' if x['hard_misses'] else ''}{miss}")

    kinds = {r["kind"] for r in results}
    all_rows = [x for r in results for x in r["rows"]]
    if len(kinds) == 1:
        n = len(all_rows)
        ccr = sum(x["ccr"] for x in all_rows) / n
        passed = sum(1 for x in all_rows if x["passed"])
        # `next(iter(...))`, not `.pop()` -- popping empties the set the per-field
        # loop below iterates, so the breakdown silently disappeared in exactly the
        # common case (one kind across the whole corpus).
        print(f"\n{'LIBRARY':<{w}}  {next(iter(kinds)):<5}  {n:>4}  {ccr:>5.1%}  "
              f"{passed:>4}/{n:<3} {100.0 * passed / n:>3.0f}%")
    else:
        # Averaging a 5-item contract with a 7-item one is not a rate of anything.
        print(f"\nNo library average: novels were measured against different "
              f"contracts ({', '.join(sorted(kinds))}). Compare within a kind.")

    for kind in sorted(kinds):
        rows = [x for x in all_rows if x["kind"] == kind]
        stats = field_stats(rows)
        expected = REAL_FIELDS if kind == "REAL" else PROXY_FIELDS
        print(f"\nper-field hit rate ({kind}, n={len(rows)} chapters):")
        for f in expected:
            hits, tot = stats.get(f, (0, 0))
            mark = "  <- hard" if f in HARD_FIELDS else ""
            if not tot:
                print(f"   {f:<10} {'never judgeable':>18}{mark}")
            else:
                print(f"   {f:<10} {hits:>5}/{tot:<5} {100.0 * hits / tot:>5.1f}%{mark}")
        extra = set(stats) - set(expected)
        for f in sorted(extra):
            hits, tot = stats[f]
            print(f"   {f:<10} {hits:>5}/{tot:<5} {100.0 * hits / tot:>5.1f}%  (unexpected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
