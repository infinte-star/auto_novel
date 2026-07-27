"""FPY' — first-pass yield measured WITHOUT the self-score. Zero LLM, read-only.

Why this exists: the FPY in `novel.py stats` counts a chapter as reworked when it
went through revise / replan / force-accept, and all three of those are triggered
by `score < quality_threshold`. `quality_threshold` is 8.0 and the library's 1023
self-scores have a median of exactly 8.00, so FPY is roughly "which side of the
median did the noise land on". Worse, it cannot settle an experiment that *changes
the rework rule*: the P4 A/B moved the release line and FPY moved with it
mechanically, in both arms, for free (REDESIGN §7).

FPY' asks a question the self-score cannot answer:

    did the FIRST draft, off the FIRST plan, carry any *measured* defect?

- "measured defect" is `pipeline._hard_block_reasons` — the engine's own list of
  checks that set `accepted=False` on evidence rather than on a threshold
  (gate rejects, style collapse, hard contradictions, hard contract violations,
  length_band / opening_hook_gate block, adjacent-repeat block, unmet-constraint
  pile-up). Not one of them reads `score`.
- "first plan" is `plan_initial_attempt0_*` with no `attempt1`: a plan retry means
  the specification was rejected before a word was written.
- `review_round0.json` is the first draft's review. `final_review.json` describes
  whatever survived revision, so it cannot answer a first-pass question.

The self-score is still printed, but only as an observable — it never decides
anything here. Chapters whose round-0 review is missing are reported as `n/a`
rather than silently counted as passes.

    python tools/fpy_prime.py                       # every novel
    python tools/fpy_prime.py p4_score p4_det --from 47 --to 50
    python tools/fpy_prime.py tangshuting --bands 40   # FPY' vs chapter position
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import _hard_block_reasons  # noqa: E402

# `_hard_block_reasons` reads a handful of `config["novel"]` thresholds. Using each
# novel's own config would make two arms incomparable the moment their configs
# differ, which is the whole failure mode this tool exists to avoid -- so the
# thresholds are pinned here, at the engine defaults, for every novel.
PINNED = {
    "novel": {
        "style_penalty_block": 2.0,
        "factcheck_hard_blocks_accept": True,
        "contract_blocks_accept": True,
        "constraint_violation_block_count": 3,
    }
}


def _payload(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(raw, dict):
        return raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
    return None


# Which replan checkpoint labels count as a first-pass failure.
#
# COUNTED — all three are pre-review and decided by measured evidence, never by the
# self-score, so including them cannot be gamed by moving the release line:
#   plan_initial_attempt[1-9]  scene-dedupe / executability retry inside create_plan
#   plan_critical              validate_plan_continuity found a CRITICAL violation
#   plan_fossil_catastrophe    every candidate draft was blocked by the fossil pre-screen
#
# EXCLUDED — `plan_quality_replan` and `plan_hard_floor` are downstream of the rework
# decision itself (pipeline.py:1537 / 1679). Counting them would put the release rule
# back into the criterion, which is exactly the circularity FPY' exists to remove.
COUNTED_REPLANS = ("plan_initial_attempt[1-9]*_candidates.json",
                   "plan_critical_attempt*_candidates.json",
                   "plan_fossil_catastrophe_attempt*_candidates.json")

# Archived payloads encode the engine semantics of the day they were written, so a
# retroactive replay reports fixed bugs as live problems unless it normalizes.
#
# `review.py`'s contract backstop synthesizes a violation when the reviewer describes
# an ability/modality breach in free-text `problems` instead of the structured field.
# It originally stamped those `severity: "hard"`, which `_hard_block_reasons` blocks
# on; commit b54bfd0 downgraded them to `soft`. 22 archived chapters fail FPY' on a
# backstop-synthesized hard violation ALONE and would pass under current code — which
# is the difference between "hard_contract is the library's #1 first-draft killer, 54
# misses" (wrong) and "32 misses, second to gate_rejects" (right). Normalized by
# default; `--raw` replays the payloads verbatim.
SUPERSEDED_CONTRACT_RULE = "由 problems 文本回填"


def _normalize(review: dict) -> dict:
    """Re-stamp verdicts whose severity the engine has since changed."""
    cvs = review.get("contract_violations")
    if not isinstance(cvs, list):
        return review
    fixed, changed = [], False
    for c in cvs:
        if (isinstance(c, dict)
                and str(c.get("severity", "")).lower() == "hard"
                and SUPERSEDED_CONTRACT_RULE in str(c.get("rule", ""))):
            c, changed = {**c, "severity": "soft"}, True
        fixed.append(c)
    if not changed:
        return review
    return {**review, "contract_violations": fixed}


def chapter_verdict(ch_dir: Path, *, raw: bool = False) -> dict:
    """Decide one chapter. Returns {ch, ok, reasons, score, replans, missing}."""
    ch = int(re.sub(r"\D", "", ch_dir.name) or 0)
    r0 = _payload(ch_dir / "review_round0.json")
    if r0 is not None and not raw:
        r0 = _normalize(r0)
    labels = []
    for pat in COUNTED_REPLANS:
        labels += [p.name.split("_attempt")[0] for p in ch_dir.glob(pat)]
    replans = sorted(set(labels))
    if r0 is None:
        return {"ch": ch, "ok": None, "reasons": ["no_round0_review"],
                "score": None, "replans": replans, "missing": True}
    reasons = _hard_block_reasons(r0, PINNED)
    reasons = [f"replanned:{lbl.replace('plan_', '')}" for lbl in replans] + reasons
    score = r0.get("score")
    return {"ch": ch, "ok": not reasons, "reasons": reasons,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "replans": replans, "missing": False}


def novel_verdicts(name: str, lo: int, hi: int, *, raw: bool = False) -> list[dict]:
    base = ROOT / "novels" / name / "logs" / "checkpoints"
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.glob("ch*")):
        if not d.is_dir():
            continue
        v = chapter_verdict(d, raw=raw)
        if lo <= v["ch"] <= hi:
            out.append(v)
    return out


def _rate(vs: list[dict]) -> tuple[int, int, float | None]:
    scored = [v for v in vs if v["ok"] is not None]
    ok = sum(1 for v in scored if v["ok"])
    pct = (100.0 * ok / len(scored)) if scored else None
    return ok, len(scored), pct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("novels", nargs="*", help="novel names (default: all)")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=10**9)
    ap.add_argument("--bands", type=int, default=0,
                    help="also print FPY' per band of N chapters (mid-book drift)")
    ap.add_argument("--reasons", action="store_true", help="per-chapter detail")
    ap.add_argument("--raw", action="store_true",
                    help="replay payloads verbatim, without normalizing verdicts whose "
                         "severity the engine has since changed (SUPERSEDED_CONTRACT_RULE)")
    args = ap.parse_args()

    names = args.novels or sorted(
        p.name for p in (ROOT / "novels").iterdir()
        if (p / "logs" / "checkpoints").is_dir())
    if not names:
        print("no novels with checkpoints found")
        return 2

    width = max(len(n) for n in names)
    tally: dict[str, int] = {}
    head = "FPY'"
    print(f"{'novel':<{width}}  {head:>12}  {'score>=8.0':>17}  {'self-score':>10}")
    for name in names:
        vs = novel_verdicts(name, args.lo, args.hi, raw=args.raw)
        if not vs:
            print(f"{name:<{width}}  {'no chapters in range':>12}")
            continue
        ok, n, pct = _rate(vs)
        scores = [v["score"] for v in vs if v["score"] is not None]
        # The score-based comparison point, computed on the SAME chapters:
        # how many would a 8.0 threshold have called clean?
        thr_ok = sum(1 for s in scores if s >= 8.0)
        avg = sum(scores) / len(scores) if scores else float("nan")
        miss = sum(1 for v in vs if v["missing"])
        line = (f"{name:<{width}}  {ok:>4}/{n:<3} {pct:>4.0f}%  "
                f"{thr_ok:>7}/{len(scores):<3} {100.0*thr_ok/len(scores) if scores else 0:>4.0f}%  "
                f"{avg:>10.2f}")
        if miss:
            line += f"   ({miss} n/a)"
        print(line)
        for v in vs:
            for r in v["reasons"]:
                tally[r.split("=")[0].split("×")[0]] = tally.get(r.split("=")[0].split("×")[0], 0) + 1
        if args.bands:
            for start in range(args.lo, max(v["ch"] for v in vs) + 1, args.bands):
                band = [v for v in vs if start <= v["ch"] < start + args.bands]
                if band:
                    b_ok, b_n, b_pct = _rate(band)
                    if b_n:
                        print(f"{'':<{width}}    Ch{start}-{start + args.bands - 1}: "
                              f"{b_ok}/{b_n} {b_pct:.0f}%")
        if args.reasons:
            for v in vs:
                if v["ok"] is not True:
                    print(f"{'':<{width}}    Ch{v['ch']}: {', '.join(v['reasons'])}")

    if tally:
        print("\nwhy first drafts failed (all chapters in range):")
        for k, c in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<28} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
