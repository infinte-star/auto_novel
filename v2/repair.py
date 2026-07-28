"""v2 repair: run v1's fix ladder, then judge it by the acceptance ruler.

REDESIGN_V2 §3.4. The fixers themselves are `fix.py`'s, unchanged and
unwrapped — `apply_l0` (zero LLM: em-dash reduction, fragment merging, fossil
rotation, scenery-opening demotion) and `apply_l1` (bounded splice-back calls:
expand-to-band, dialogue injection, em-dash rewording). Rewriting them would be
rewriting the one part of v1 that is already measured and already
keep-only-if-improved.

What this module adds is the half v1 cannot have, for a structural reason:

**v1 repairs after the verdict; v2 repairs before it.** In v1 `_stage_fix` runs
*after* `_classify_replan_failure` has already decided the draft's fate, which is
why a fossil `gate_rejects` needed its own hand-placed escape hatch
(`pipeline._repair_fossil_rejects`, verify-then-drop, inside the review loop) to
avoid buying a structural replan for something rotation would have cleared. In
v2 the decision table asks `l0_pending` / `l1_pending` *before* `need_draft`, so
the ordering that v1 patched around is the default. No special case survives:
`book_wide_fossils`'s hard reject only indicts a chapter that actually contains
the phrase, so once rotation removes it, a plain recompute clears the reject.

**A repair is kept only if acceptance says so.** Each fixer already guards itself
against its own metric — `apply_l0` keeps a transform only when `style_health`
did not worsen. But `style_health` is not the currency v2 releases on;
`accept.block_reasons` is. A rotation that dodges a fossil and lands on an
adjacent-repeat, or an expansion that clears the length floor and blows the
ceiling, passes every inner guard and fails the only one that matters. So each
layer is re-scored with the same `recheck` the release rule uses, and reverted
whole if it introduced a blocking reason that was not there before.

The revert is whole-layer, not per-action, and that is a real cost: one bad
rotation discards two good em-dash fixes in the same pass. `fix.apply_l0` runs
its actions internally and returns one string, so per-action granularity would
mean reimplementing the ladder — which is exactly the duplication this module
exists not to create. The inner guards make the case rare; the log line names it
when it happens.

`recheck` is injected rather than built here. It closes over the corpus context
`accept.acceptance_report` needs (prior chapters, the book text map, the card),
which is `run.py`'s to own — and injecting it makes every branch in here
testable with no database and no API.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Iterable, Sequence

import fix
from v2.accept import block_reasons

LAYERS: tuple[str, ...] = ("L0", "L1")

# A blocking reason is a label plus its evidence: `gate_rejects=a,b`,
# `style_collapse(penalty=2.3)`, `hard_contract=3`. Comparing the raw strings
# would read a penalty falling 2.5 -> 2.3 as "a different problem", so the
# comparison is on the label alone.
_KIND_RE = re.compile(r"^[a-z_]+")

Recheck = Callable[[str], dict]


def reason_kind(reason: Any) -> str:
    m = _KIND_RE.match(str(reason or ""))
    return m.group(0) if m else str(reason or "")


def reason_kinds(reasons: Iterable[Any]) -> frozenset[str]:
    return frozenset(reason_kind(r) for r in (reasons or ()))


def pending(report: dict[str, Any], config: dict[str, Any],
            layer: str) -> tuple[str, ...]:
    """The repair actions this layer still has to offer for this report.

    The decision table's `l0_pending` / `l1_pending` predicates, and the reason
    they are zero-LLM: `fix.plan_repairs` is a pure function of the gate results
    already in the report.

    Deliberately stateless — it does NOT know whether the layer has already run.
    `run.py` tracks that in the step checkpoint, because a layer that ran and
    did not clear its gate must not be offered again: re-offering it is the latch
    this codebase keeps rediscovering (CLAUDE.md: "what can THIS chapter do to
    turn it green?"). A fixer that failed once on the same text will fail again.
    """
    if not isinstance(report, dict):
        return ()
    try:
        steps = fix.plan_repairs(report, config)
    except Exception:
        return ()
    return tuple(s["action"] for s in steps if s.get("layer") == layer)


@dataclasses.dataclass(frozen=True)
class RepairOutcome:
    """What a repair pass did, and whether the ruler agrees it helped."""

    text: str
    report: dict[str, Any]
    applied: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()
    blocks_before: tuple[str, ...] = ()
    blocks_after: tuple[str, ...] = ()
    reverted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    @property
    def cleared(self) -> bool:
        """No deterministic reason left to reject this draft."""
        return not self.blocks_after

    @property
    def improved(self) -> bool:
        return len(self.blocks_after) < len(self.blocks_before)

    def summary(self) -> str:
        bits = [f"blocks {len(self.blocks_before)}->{len(self.blocks_after)}"]
        if self.applied:
            bits.append("applied=" + ",".join(self.applied))
        if self.reverted:
            bits.append("reverted=" + ",".join(self.reverted))
        if self.skipped:
            bits.append("skipped=" + ",".join(self.skipped))
        return " ".join(bits)


def _log(paths: Any, message: str) -> None:
    if paths is None:
        return
    try:
        from config import log

        log(paths, message)
    except Exception:
        pass


def _regressed(before: Sequence[str], after: Sequence[str]) -> str:
    """'' when the repair is safe to keep, else why it is not.

    Three cases, in order, and the ordering is the whole rule:

    - **Strictly fewer blocking reasons — keep, whatever they are.** The ruler is
      the ruler: a draft with two blocks left is closer to release than one with
      three, and refusing a net win because the remaining problem has a new name
      would be scoring the repair on something other than the release rule.
    - **More blocking reasons — revert.** Unambiguous damage.
    - **The same number, but a kind that was not there before — revert.** This is
      the case the inner guards structurally cannot see: rotation dodges its own
      fossil gate and lands on an adjacent-repeat; expansion clears the length
      floor and blows the ceiling. Each fixer scored itself on the metric it was
      aiming at and saw an improvement. A one-for-one swap is not one.

    Worsening *within* a kind (two fossil rejects where there was one) is not
    caught, on purpose: the counts live inside the reason string, and every fixer
    already guards its own metric. This guard is for cross-metric collateral.
    """
    if len(after) < len(before):
        return ""
    if len(after) > len(before):
        return f"count {len(before)}->{len(after)}"
    swapped = reason_kinds(after) - reason_kinds(before)
    if swapped:
        return "swapped_for=" + ",".join(sorted(swapped))
    return ""


def run_layer(
    layer: str,
    *,
    text: str,
    report: dict[str, Any],
    config: dict[str, Any],
    chapter_num: int,
    recheck: Recheck,
    client: Any = None,
    paths: Any = None,
) -> RepairOutcome:
    """Run ONE repair layer and re-score it. Never raises, never re-drafts.

    Returns the original text and the original report untouched when the layer
    has nothing to offer, when it changed nothing, or when its result was
    reverted — so the caller can treat the outcome as authoritative without
    checking which of those happened.
    """
    before = tuple(block_reasons(report, config))
    idle = RepairOutcome(text=text, report=report, blocks_before=before,
                         blocks_after=before)

    actions = pending(report, config, layer)
    if not actions:
        return idle
    if layer == "L1" and client is None:
        # Not a silent skip: an L1 layer that never ran because no client was
        # threaded through would otherwise look exactly like an L1 layer with
        # nothing to do, and the A/B's call count would be quietly wrong.
        _log(paths, f"v2.repair Ch{chapter_num} L1 skipped (no client): "
                    + ",".join(actions))
        return dataclasses.replace(idle, skipped=actions)

    try:
        if layer == "L0":
            new_text, applied = fix.apply_l0(text, report, config, chapter_num)
        else:
            new_text, applied = fix.apply_l1(client, paths, config, chapter_num,
                                             text, report)
    except Exception as exc:  # a repair must never cost a written chapter
        _log(paths, f"v2.repair Ch{chapter_num} {layer} failed (non-fatal): {exc}")
        return idle

    if not applied or not new_text or new_text == text:
        # A layer that RAN and kept nothing is a different event from a layer
        # with nothing to run, and for L1 the difference is a paid call. Both
        # printed `blocks N->N` and nothing else, which is how this stayed
        # invisible: across the settlement A/B's two v2 arms, 13 `fix_expand`
        # calls produced 7 `v2_repair_l1` events, and the missing 6 left no
        # record of what they bought. The COUNT is recoverable (the calls are in
        # `llm_calls.jsonl`, the keeps in the event log); the reason only ever
        # existed here, and each fixer now names it on the line above.
        _log(paths, f"v2.repair Ch{chapter_num} {layer} kept nothing from "
                    f"[{','.join(actions)}]")
        return idle

    try:
        new_report = recheck(new_text)
    except Exception as exc:
        _log(paths, f"v2.repair Ch{chapter_num} {layer} recheck failed, "
                    f"reverting (non-fatal): {exc}")
        return dataclasses.replace(idle, reverted=tuple(applied))

    after = tuple(block_reasons(new_report, config))
    why = _regressed(before, after)
    if why:
        _log(paths, f"v2.repair Ch{chapter_num} {layer} REVERTED ({why}): "
                    + ",".join(applied))
        return dataclasses.replace(idle, reverted=tuple(applied))

    _log(paths, f"v2.repair Ch{chapter_num} {layer} kept "
                f"[{','.join(applied)}] blocks {len(before)}->{len(after)}")
    return RepairOutcome(text=new_text, report=new_report,
                         applied=tuple(applied), layers=(layer,),
                         blocks_before=before, blocks_after=after)


__all__ = [
    "LAYERS", "RepairOutcome", "pending", "run_layer", "repair_draft",
    "reason_kind", "reason_kinds",
]
