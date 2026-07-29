"""Re-settle a change to CCC's LOGIC on an already-written v2 arm. Zero LLM.

`tools/fpy_prime.py` replays archived `review_round0.json` payloads, so a verdict
already baked into one is frozen: change `v2/accept.contract_fulfilment` and
fpy_prime keeps reporting the old answer forever. This is the `replay_gates.py`
equivalent for CCC.

**It re-decides from the archived payload, not from the chapter text, and that is
the only sound way to do it here.** The first version of this tool recomputed CCC
from `chapter_draft.json` and reported +4 chapters recovered. That was wrong.
`v2/run.py:_act_write` overwrites `DRAFT_CHECKPOINT` on *every* write while
`ROUND0_CHECKPOINT` is written once (`run.round0_saved`), so on any rescued
chapter the archived draft is the **post-rescue** draft while the archived verdict
describes the first one. Recomputing the gate on one text and patching it into the
other text's payload reports a rescue as a first-pass success — precisely the
"repairs counted as first-pass" trap the docstring claimed to be guarding, applied
to the wrong artifact. And the chapters that lose their first draft are exactly the
ones that failed, i.e. the only ones a fix could move.

The `forbid` guards are decidable without the prose: the archived violation records
the `forbid` entry verbatim in `target` and the matched anchor in `phrase`, and both
`_is_misfiled_requirement(target)` and `phrase in _required_text(card)` are pure
functions of the CARD. So the re-decision reads the same functions the engine reads
(never a copy of their logic) and needs no text at all.

Where the text *does* survive — a chapter with no rescue — the tool recomputes CCC
from it as a **cross-check** on the payload-level re-decision, and reports any
disagreement. A mismatch would mean the payload no longer preserves enough to
re-decide, which is the thing that must not be assumed.

A v2 `card_replan` is downstream of the block: the archived
`card_replan_attempt*_rescue.json` records the `block_reasons` that caused it, so a
replan whose every reason the fix clears disappears with the block. A replan that
names some OTHER reason survives.

    python tools/replay_ccc.py ts_v2arm --from 201 --to 230
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _payload(p: Path):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return d.get("payload") if isinstance(d, dict) and "payload" in d else d


def _redecide(ccc0: dict, card, accept) -> dict:
    """Apply today's `forbid` guards to an archived CCC result.

    Reads `accept._is_misfiled_requirement` / `accept._required_text` rather than
    reimplementing them, so this tool cannot drift from the engine.
    """
    required = accept._required_text(card)
    kept, waived = [], list(ccc0.get("forbid_conflicts") or [])
    for v in ccc0.get("violations") or []:
        target, phrase = str(v.get("target") or ""), str(v.get("phrase") or "")
        if v.get("field") == "forbid" and accept._is_misfiled_requirement(target):
            waived.append({**v, "why": "requirement_misfiled_as_ban", "phrase": ""})
            continue
        if v.get("field") == "forbid" and phrase and phrase in required:
            waived.append({**v, "why": "card_requires_the_phrase_it_bans"})
            continue
        kept.append(v)
    out = dict(ccc0)
    out["violations"] = kept
    out["forbid_conflicts"] = waived
    out["passed"] = not out.get("hard_misses") and not kept
    return out


def _mismatch(a: dict, b: dict) -> str:
    """Compare a re-decided CCC against one recomputed from the surviving text."""
    ka = (bool(a.get("passed")), sorted(i.get("field") for i in a.get("hard_misses") or []),
          sorted(str(v.get("phrase")) for v in a.get("violations") or []))
    kb = (bool(b.get("passed")), sorted(i.get("field") for i in b.get("hard_misses") or []),
          sorted(str(v.get("phrase")) for v in b.get("violations") or []))
    return "" if ka == kb else f"payload={ka} text={kb}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("novel")
    ap.add_argument("--from", dest="lo", type=int, required=True)
    ap.add_argument("--to", dest="hi", type=int, required=True)
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    nd = ROOT / "novels" / args.novel
    if not nd.is_dir():
        raise SystemExit(f"no such novel: {args.novel}")
    os.environ["NOVEL_CONFIG"] = str((nd / "config.yaml").relative_to(ROOT))
    os.environ.setdefault("NOVEL_PROMPT", str((nd / "prompt.md").relative_to(ROOT)))

    import engine.config as _config
    import engine.quality as quality
    import engine.loop as accept

    cfg = _config.load_config()

    ck = nd / "logs" / "checkpoints"
    rows, missing, mismatches = [], [], []
    for ch in range(args.lo, args.hi + 1):
        d = ck / f"ch{ch:04d}"
        card = _payload(d / "chapter_card.json")
        r0 = _payload(d / "review_round0.json")
        if not isinstance(r0, dict):
            missing.append(ch)
            continue
        ccc0 = r0.get("contract_fulfilment")
        if not isinstance(ccc0, dict):
            missing.append(ch)
            continue

        before = list(quality.hard_block_reasons(r0, cfg) or [])
        ccc = _redecide(ccc0, card, accept)

        # Patch ONLY the changed gate, then rebuild its gate_rejects entry.
        patched = dict(r0)
        patched["contract_fulfilment"] = ccc
        rejects = [g for g in (r0.get("gate_rejects") or [])
                   if g.get("gate") != "contract_fulfilment"]
        if ccc.get("enabled") and not ccc.get("passed"):
            rejects.append({
                "gate": "contract_fulfilment", "level": "reject",
                "phrases": [i["target"] for i in ccc["hard_misses"]][:4],
                "violations": [v["phrase"] for v in ccc["violations"]][:4]})
        patched["gate_rejects"] = rejects
        patched["accepted"] = not rejects and bool(r0.get("accepted", True))
        after = list(quality.hard_block_reasons(patched, cfg) or [])

        # A replan survives only if it names a reason the fix did not clear.
        replans = sorted(d.glob("card_replan_attempt*_rescue.json"))
        replan_before = len(replans)
        replan_after = 0
        for rp in replans:
            p = _payload(rp) or {}
            named = [str(x) for x in (p.get("block_reasons") or [])]
            if not named or any(n in after for n in named):
                replan_after += 1

        # Cross-check only where the first draft survives. A rescue overwrote it
        # everywhere else, so recomputing there would judge the second draft.
        draft = _payload(d / "chapter_draft.json")
        checked = False
        if not replans and isinstance(draft, dict):
            text = str(draft.get("text") or "")
            if text:
                why = _mismatch(ccc, accept.contract_fulfilment(
                    card, accept._body(text), cfg))
                checked = True
                if why:
                    mismatches.append(f"Ch{ch}: {why}")

        rows.append({"ch": ch, "before": before, "after": after,
                     "rb": replan_before, "ra": replan_after,
                     "ccr": ccc0.get("ccr"), "checked": checked,
                     "draft_gone": bool(replans),
                     "waived": [c.get("why") for c in ccc.get("forbid_conflicts") or []]})

    def fpy(key_r, key_p):
        return len([r for r in rows if not r[key_r] and not r[key_p]]), len(rows)

    ok_b, n = fpy("before", "rb")
    ok_a, _ = fpy("after", "ra")
    print(f"{args.novel} Ch{args.lo}-{args.hi}   n={n}"
          + (f"   (no payload: {missing})" if missing else ""))
    if not n:
        print("  no rows")
        return 0
    print(f"  FPY' as archived : {ok_b}/{n}  {ok_b / n * 100:.0f}%")
    print(f"  FPY' recomputed  : {ok_a}/{n}  {ok_a / n * 100:.0f}%"
          f"   ({ok_a - ok_b:+d} chapters)")
    ccrs = [r["ccr"] for r in rows if isinstance(r["ccr"], (int, float))]
    if ccrs:
        print(f"  CCR (round-0, as archived) : {sum(ccrs) / len(ccrs) * 100:.1f}%"
              f"   chapters clean after fix: {sum(1 for r in rows if not r['after'])}/{n}")
    waived = sum(len(r["waived"]) for r in rows)
    if waived:
        print(f"  card defects waived: {waived} forbid entr(y/ies) across the window")

    checked = sum(1 for r in rows if r["checked"])
    gone = sum(1 for r in rows if r["draft_gone"])
    print(f"  cross-check: {checked}/{n} chapters had a surviving first draft and "
          f"agreed" if not mismatches else
          f"  cross-check: {len(mismatches)} DISAGREEMENT(S) out of {checked} checked")
    for m in mismatches:
        print("    " + m)
    if gone:
        print(f"  {gone} chapter(s) were rescued, so `chapter_draft.json` holds the "
              f"SECOND draft and was not used; those rows are re-decided from the "
              f"archived payload only (sound for a card-only fix).")

    changed = [r for r in rows
               if (bool(r["before"]) or r["rb"]) != (bool(r["after"]) or r["ra"])]
    if changed:
        print("\n  chapters the fix changes:")
        for r in changed:
            print(f"    Ch{r['ch']}: {r['before']}+replan{r['rb']}"
                  f"  ->  {r['after'] or 'clean'}+replan{r['ra']}")
    if args.detail:
        print("\n  all rows:")
        for r in rows:
            print(f"    Ch{r['ch']}: ccr={r['ccr']:.3f} after={r['after'] or 'clean'}"
                  f" replan {r['rb']}->{r['ra']} waived={r['waived']}"
                  f"{' [draft gone]' if r['draft_gone'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
