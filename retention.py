"""Deterministic retention metric — the pipeline's missing objective function.

The reviewer scores each chapter in isolation (prose/continuity/beats) and
systematically over-rates it; the ONLY signal that tracks real reader retention
is the reader-panel (reader_panel.py), and historically it was advisory-only.
This module turns the panel's per-chapter excitement / drop_rate into a single
deterministic, comparable "retention" summary so it can (a) show up in the
experiment harness (compare.py, P0), (b) gate the composite score (review.py,
P2), and (c) steer arc re-planning (rolling_plan.py, P3).

Everything here is PURE (no I/O, no LLM) so it is unit-testable and cheap to
call anywhere. Callers supply already-loaded excitement/drop series; the store /
compare layers own the actual reads.

Design note — why excitement×stay-rate: a chapter that thrills the readers who
remain but bleeds 80% of them is NOT a good chapter; a chapter that keeps
everyone mildly interested is better for a serialized web novel. So the index
multiplies mean excitement by the stay-rate (1 - drop) and then penalizes deep
troughs, because a single excitement=1.6 crater is a concrete reader exit point
that an average hides.
"""
from __future__ import annotations

from typing import Any


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def summarize_retention(
    excitements: list[float],
    drops: list[float] | None = None,
    *,
    low_excitement: float = 4.0,
    trough_penalty: float = 2.0,
) -> dict[str, Any]:
    """Reduce a panel excitement (1-10) + drop_rate (0-1) series to a summary.

    The two lists are aligned newest-order-agnostic (order doesn't affect the
    aggregate). `drops` may be omitted (treated as 0 = nobody quit) so callers
    with excitement-only data still get a usable index.

    Returns a dict with the raw aggregates plus a single `retention_index`
    (0-10, higher = better) suitable for thresholding and side-by-side compare.
    """
    ex = [float(e) for e in excitements if e is not None]
    n = len(ex)
    if n == 0:
        return {
            "panels": 0,
            "mean_excitement": None,
            "min_excitement": None,
            "mean_drop": None,
            "max_drop": None,
            "trough_count": 0,
            "trough_frac": 0.0,
            "retention_index": None,
        }
    dr = [float(d) for d in (drops or []) if d is not None]
    mean_ex = sum(ex) / n
    min_ex = min(ex)
    mean_drop = (sum(dr) / len(dr)) if dr else 0.0
    max_drop = max(dr) if dr else 0.0
    troughs = sum(1 for e in ex if e < low_excitement)
    trough_frac = troughs / n

    # excitement (1-10) discounted by stay-rate, minus a deep-trough penalty.
    stay_rate = clamp(1.0 - mean_drop, 0.0, 1.0)
    index = mean_ex * stay_rate - trough_penalty * trough_frac
    index = clamp(index, 0.0, 10.0)

    return {
        "panels": n,
        "mean_excitement": round(mean_ex, 2),
        "min_excitement": round(min_ex, 2),
        "mean_drop": round(mean_drop, 3),
        "max_drop": round(max_drop, 3),
        "trough_count": troughs,
        "trough_frac": round(trough_frac, 3),
        "retention_index": round(index, 2),
    }


def retention_by_block(
    series: list[tuple[int, float, float]],
    block: int = 10,
    *,
    low_excitement: float = 4.0,
) -> list[dict[str, Any]]:
    """Bucket a (chapter, excitement, drop) series into N-chapter blocks and
    summarize each — this is the "retention curve" that exposes a mid-book sag
    an overall average hides. `series` need not be sorted or dense.
    """
    if not series:
        return []
    buckets: dict[int, list[tuple[float, float]]] = {}
    for ch, ex, dr in series:
        b = (int(ch) - 1) // block
        buckets.setdefault(b, []).append((ex, dr))
    out: list[dict[str, Any]] = []
    for b in sorted(buckets):
        rows = buckets[b]
        summ = summarize_retention(
            [r[0] for r in rows], [r[1] for r in rows], low_excitement=low_excitement
        )
        summ["ch_from"] = b * block + 1
        summ["ch_to"] = b * block + block
        out.append(summ)
    return out


def weighted_aggregate(
    per_persona: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Weighted drop_rate / pay_rate / excitement over per-persona verdicts.

    `per_persona` rows carry keys: persona, continue_reading(bool),
    would_pay(bool), excitement(float). When `weights` is None/empty the result
    is the plain unweighted mean (identical to the legacy aggregate), so this is
    a safe no-op when persona weighting is disabled. A persona missing from
    `weights` defaults to weight 1.0.
    """
    rows = [r for r in per_persona if isinstance(r, dict)]
    if not rows:
        return {"drop_rate": None, "pay_rate": None, "avg_excitement": None}
    def w(r: dict[str, Any]) -> float:
        if not weights:
            return 1.0
        return float(weights.get(str(r.get("persona", "")), 1.0))
    tw = sum(w(r) for r in rows) or 1.0
    drop = sum(w(r) for r in rows if not r.get("continue_reading", True)) / tw
    pay = sum(w(r) for r in rows if r.get("would_pay", False)) / tw
    exc = sum(w(r) * float(r.get("excitement", 5) or 5) for r in rows) / tw
    return {
        "drop_rate": round(drop, 3),
        "pay_rate": round(pay, 3),
        "avg_excitement": round(exc, 2),
    }
