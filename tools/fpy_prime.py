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

Archived payloads freeze the engine semantics of the day they were written, so
`_normalize` re-stamps the two verdicts whose severity has since changed — the
contract backstop's HARD→SOFT downgrade (b54bfd0, 22 chapters) and the em-dash
trend term's flat +1.0 → graduated charge (31 chapters, 5 of which blocked on the
flat rule alone). Without them the tool reports fixed bugs as live problems: the
same corpus reads `hard_contract` 54 vs 32 and `style_collapse` 7 vs 2. `--raw`
replays the payloads verbatim.

    python tools/fpy_prime.py                       # every novel
    python tools/fpy_prime.py p4_score p4_det --from 47 --to 50
    python tools/fpy_prime.py tangshuting --bands 40   # FPY' vs chapter position
"""
from __future__ import annotations

import argparse
import hashlib
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


def _chapter_files(novel: Path) -> list[Path]:
    return sorted((novel / "chapters").glob("[0-9][0-9][0-9][0-9].md"))


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except Exception:
        return None


def discover_novels(explicit: list[str], *, include_all: bool = False,
                    root: Path | None = None
                    ) -> tuple[list[str], list[tuple[str, str]]]:
    """Novels to aggregate, plus the derivatives dropped and why.

    A library aggregate over every dir under `novels/` counts derivative runs as
    independent books. Measured on the current corpus: `tangshuting` and
    `tangshuting_v1_backup` share 76 byte-identical chapters, and together with
    `tangshuting_e2e` they are 446 of the library's 643 chapters -- so "LIBRARY
    85.8%" was largely one book weighted three times, and every "the library's #1
    killer is X" claim inherited that weighting. Same measurement-pollution class
    as `pairwise_ab` charging an arm for being measured (CLAUDE.md).

    Only PROVABLE derivatives are dropped, each by evidence rather than by a name
    guess:

      * `__ablate_` in the dir name, or `experiments/ablate_<name>.json` -- the
        engine's own naming for an ablation copy.
      * `experiments/fork_<name>.json` -- the engine's own fork metadata.
      * Ch1 byte-identical to another novel's Ch1. Two independent runs never
        produce the same first chapter even from the same brief (`tangshuting_e2e`
        is the control: same story concept, different Ch1, so it is NOT dropped),
        which makes an identical Ch1 proof of a copy/fork at genesis. Of the pair
        the longer book is canonical; ties break alphabetically.

    Explicit names are NEVER filtered -- naming two arms is how an A/B is read.
    Callers must print the returned drop list: a silently narrowed corpus reads
    exactly like a clean one.

    `root` overrides the project root (tests only).
    """
    base = root or ROOT
    novels = base / "novels"
    all_names = sorted(p.name for p in novels.iterdir()
                       if (p / "logs" / "checkpoints").is_dir())
    if explicit:
        return list(explicit), []
    if include_all:
        return all_names, []

    dropped: list[tuple[str, str]] = []
    kept: list[str] = []
    ch1: dict[str, str] = {}
    for name in all_names:
        files = _chapter_files(novels / name)
        h = _sha(files[0]) if files else None
        if h:
            ch1[name] = h
    # Longer books win the canonical slot; sorted() already fixes the tie order.
    order = sorted(all_names, key=lambda n: (-len(_chapter_files(novels / n)), n))
    canonical: dict[str, str] = {}
    for name in order:
        h = ch1.get(name)
        if h and h not in canonical:
            canonical[h] = name

    for name in all_names:
        if "__ablate_" in name or (base / "experiments" / f"ablate_{name}.json").exists():
            dropped.append((name, "ablation copy (engine-generated)"))
            continue
        if (base / "experiments" / f"fork_{name}.json").exists():
            dropped.append((name, "fork copy (engine-generated)"))
            continue
        h = ch1.get(name)
        src = canonical.get(h or "")
        if src and src != name:
            mine = {p.stem: _sha(p) for p in _chapter_files(novels / name)}
            theirs = {p.stem: _sha(p) for p in _chapter_files(novels / src)}
            common = set(mine) & set(theirs)
            same = sum(1 for k in common if mine[k] == theirs[k])
            dropped.append((name, f"Ch1 identical to {src}"
                                  f" ({same}/{len(common)} chapters byte-identical)"))
            continue
        kept.append(name)
    return kept, dropped


def print_exclusions(dropped: list[tuple[str, str]], flag: str = "--all") -> None:
    if not dropped:
        return
    print(f"excluded from the aggregate ({len(dropped)}; pass {flag} to include):")
    for name, why in dropped:
        print(f"   - {name}: {why}")
    print()


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

# Second superseded semantic, same class. `style_health`'s em-dash TREND term used to
# charge a flat +1.0; it is now graduated by ratio (0.3 / 0.5 / 0.8 / 1.0), because a
# flat charge stacked onto the static tier's +1.0 hits `style_penalty_block` (2.0)
# exactly — so a chapter merely 1.9× above its own recent mean was called a collapse.
# 31 archived chapters were penalized under the flat rule and 5 of them BLOCK on it
# alone (tangshuting Ch6/9/41/70/142, recomputed 1.3-1.8 today). Left un-normalized,
# `style_collapse` reads as 7 live first-draft killers when only 2 are real
# (tangshuting Ch37 at 2.5, yeban_guize Ch9 at ratio 3.3×).
#
# The recompute calls `quality.em_dash_penalty` rather than re-implementing it: the
# arithmetic must not be able to drift away from the engine's (the failure mode
# `tests/test_latching_gates.py:ReplayPlanScoreFidelityTest` records). Only the
# em-dash terms are re-stamped -- every other component of the penalty is taken from
# the archived total, since its inputs are the chapter text, which round0 no longer
# has once the draft was revised.


def _restamp_style_penalty(sh: dict) -> dict | None:
    """Recompute `style_health.penalty` under today's em-dash rule, or None.

    Returns None when nothing changes, when the archived metrics are missing, or
    when a `style_penalty_cap` may have clipped the total (in which case the
    archived number is not a sum of its terms and cannot be adjusted term-wise).
    """
    metrics = sh.get("metrics")
    if not isinstance(metrics, dict):
        return None
    em = metrics.get("em_dash_per_kchar")
    old_total = sh.get("penalty")
    if not isinstance(em, (int, float)) or not isinstance(old_total, (int, float)):
        return None
    if float(old_total) >= float(PINNED["novel"].get("style_penalty_cap", 4.0)):
        return None
    flags = [str(f) for f in (sh.get("flags") or [])]
    # Decoding what the ARCHIVED flags were charged at write time. The static
    # tiers below are literals rather than a call into `quality`, because they
    # must describe the past: they happen to equal today's values only because
    # the static tiers never changed, and the graduation touched the trend term
    # alone. If a static tier is ever re-tuned, subtract the OLD number here (the
    # `pen=` field is the durable fix — flags written since the graduation carry
    # their own charge, so future re-tunings need no literal at all).
    old_em = 0.0
    for f in flags:
        if f.startswith("em_dash_overload"):
            old_em += 2.0
        elif f.startswith(("em_dash_high", "em_dash_sustained")):
            old_em += 1.0
        elif f.startswith("em_dash_trend_rise"):
            m = re.search(r"pen=([\d.]+)", f)
            # No `pen=` in the flag ⇒ written before the graduation ⇒ flat +1.0.
            old_em += float(m.group(1)) if m else 1.0
    if not old_em:
        return None
    try:
        from quality import em_dash_penalty
    except Exception:
        return None
    base = metrics.get("em_dash_recent_mean")
    new_em, new_flags, _ = em_dash_penalty(
        float(em), float(base) if isinstance(base, (int, float)) else None, PINNED)
    new_total = round(float(old_total) - old_em + new_em, 2)
    if abs(new_total - float(old_total)) < 0.01:
        return None
    keep = [f for f in flags if not f.startswith("em_dash_")]
    return {**sh, "penalty": new_total, "flags": keep + new_flags}


def _normalize(review: dict) -> dict:
    """Re-stamp verdicts whose severity the engine has since changed."""
    out, changed = review, False
    cvs = review.get("contract_violations")
    if isinstance(cvs, list):
        fixed = []
        for c in cvs:
            if (isinstance(c, dict)
                    and str(c.get("severity", "")).lower() == "hard"
                    and SUPERSEDED_CONTRACT_RULE in str(c.get("rule", ""))):
                c, changed = {**c, "severity": "soft"}, True
            fixed.append(c)
        if changed:
            out = {**out, "contract_violations": fixed}
    sh = review.get("style_health")
    if isinstance(sh, dict):
        restamped = _restamp_style_penalty(sh)
        if restamped is not None:
            out = {**out, "style_health": restamped}
    return out


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
                         "severity the engine has since changed (the contract backstop's "
                         "HARD→SOFT downgrade and the flat em-dash trend charge)")
    ap.add_argument("--all", action="store_true",
                    help="include derivative dirs (ablations / forks / copies) that are "
                         "excluded from the aggregate by default")
    args = ap.parse_args()

    names, dropped = discover_novels(args.novels, include_all=args.all)
    if not names:
        print("no novels with checkpoints found")
        return 2
    print_exclusions(dropped)

    width = max(len(n) for n in names)
    tally: dict[str, int] = {}
    t_ok = t_n = 0
    head = "FPY'"
    print(f"{'novel':<{width}}  {head:>12}  {'score>=8.0':>17}  {'self-score':>10}")
    for name in names:
        vs = novel_verdicts(name, args.lo, args.hi, raw=args.raw)
        if not vs:
            print(f"{name:<{width}}  {'no chapters in range':>12}")
            continue
        ok, n, pct = _rate(vs)
        t_ok, t_n = t_ok + ok, t_n + n
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

    if t_n:
        print(f"\n{'LIBRARY':<{width}}  {t_ok:>4}/{t_n:<3} {100.0 * t_ok / t_n:>4.0f}%")
    if tally:
        print("\nwhy first drafts failed (all chapters in range):")
        for k, c in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<28} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
