"""Replay changed BLOCKING gates over the archived corpus and re-settle FPY'.

Why this exists: `tools/fpy_prime.py` replays `pipeline._hard_block_reasons` over
archived `review_round0.json` payloads, so any gate verdict already baked into
those payloads is frozen. Changing a gate's *logic* therefore cannot show up in
FPY' -- the tool keeps reporting the old verdict forever. Fine while gate logic is
stable, useless the moment you fix a gate.

This tool recomputes the changed gates from the PRIMARY data they read (chapter
texts, archived attempt0 plans) and re-runs `_hard_block_reasons` on the corrected
payload, so a gate fix is settled by the same ruler as everything else.

    python tools/replay_gates.py                      # all novels, all fixes on
    python tools/replay_gates.py --fix A               # isolate one fix
    python tools/replay_gates.py tangshuting_e2e --detail

Fixes (`--fix`, comma-separated, default all):

  A  book_wide_fossils_ratio now requires the chapter under review to actually
     CONTAIN the entrenched phrase. Replayed by dropping archived gate_rejects
     whose recorded phrases appear nowhere in the chapter text.

  B  chapter_mode_monotony now measures the monotony FRACTION with the unbiased
     classifier. Replayed by re-running the WHOLE pre-write plan-gate chain on
     the archived attempt0 plan -- not just chapter_mode. That is mandatory: the
     chain is sequential with `continue`, so while chapter_mode was blocking, the
     visual-payoff and executability gates downstream of it never ran. Dropping a
     replan because chapter_mode went quiet, without giving those gates their
     first chance to speak, would fabricate a gain.

Read before trusting any "which gate caused this retry" claim: `scene_dedupe_retry`
is NOT the scene-dedupe gate's event. It is the GENERIC `duplicate_blocked` retry
marker shared by scene_similarity + narrative_pattern + chapter_mode, and it is
written only when a further attempt follows (`attempt < max_attempts - 1`).
Counting it as a gate makes chapter_mode look like it co-fires with a second
independent blocker on 18/18 chapters, when a faithful scene_similarity replay of
those same plans reads max_sim 0.04-0.07 against a 0.82 block line.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as config_mod  # noqa: E402
from pipeline import _hard_block_reasons  # noqa: E402
from tools.fpy_prime import (COUNTED_REPLANS, PINNED, _normalize,  # noqa: E402
                             _payload, discover_novels, print_exclusions)

# `plan_initial` is the only replan label produced by the pre-write gate chain;
# plan_critical / plan_fossil_catastrophe come from other code paths and are left
# exactly as archived.
GATED_REPLAN = "plan_initial"


def novel_config(name: str) -> dict:
    """Each novel's OWN config, genre profile applied.

    Deliberately not pinned to engine defaults the way `fpy_prime.PINNED` is:
    fpy_prime pins because it compares different novels, whereas this tool compares
    old vs new logic on ONE novel, where holding its real config fixed is the
    faithful A/B. The review-side thresholds still come from PINNED.
    """
    saved = config_mod.CONFIG_FILE
    try:
        config_mod.CONFIG_FILE = ROOT / "novels" / name / "config.yaml"
        return config_mod.load_config()
    finally:
        config_mod.CONFIG_FILE = saved


def _chapter_text(novel: Path, ch: int) -> str:
    for pat in (f"*{ch:04d}*", f"*{ch}_*", f"*_{ch}.*"):
        for p in sorted((novel / "chapters").glob(pat)):
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""
    return ""


def _plans(novel: Path) -> dict[int, dict]:
    """Chapter -> the arbiter-selected plan of the FIRST attempt.

    attempt0 is the one the gate chain judged before any retry, i.e. exactly the
    plan whose fate FPY' is scoring.
    """
    out: dict[int, dict] = {}
    for p in sorted((novel / "logs" / "checkpoints").glob(
            "ch*/plan_initial_attempt0_arbitration.json")):
        pay = _payload(p) or {}
        ch = int(re.sub(r"\D", "", p.parent.name) or 0)
        plan = (pay.get("decision") or {}).get("merged_plan") or pay.get("plan")
        if isinstance(plan, dict):
            out[ch] = plan
    return out


def _plan_score(novel: Path, ch: int) -> float | None:
    pay = _payload(novel / "logs" / "checkpoints" / f"ch{ch:04d}"
                   / "plan_initial_attempt0_arbitration.json") or {}
    try:
        from planning import plan_score
        return float(plan_score(pay.get("decision") or {}))
    except Exception:
        return None


def plan_chain_block(plan: dict, recent: list[dict], score: float | None,
                     config: dict) -> str | None:
    """Re-run the pre-write plan-gate chain in ENGINE ORDER; return the first blocker.

    Order matters and is copied from `planning.create_plan`: duplicate gates
    (scene_similarity / narrative_pattern / chapter_mode) -> visual payoff ->
    executability -> plan score. Each blocker `continue`s, so only the first one
    is ever observable in a live run.
    """
    nv = config["novel"]
    try:
        from quality import (chapter_mode_monotony, narrative_pattern_repetition,
                             plan_executability_gate, plan_visual_payoff_check,
                             scene_similarity)
    except Exception:
        return "import_failed"

    if bool(nv.get("scene_dedupe_enabled", True)):
        try:
            sim = float(scene_similarity(plan, recent).get("max_sim", 0.0) or 0.0)
            block = float(nv.get("scene_dedupe_sim_block", 0.82))
            max_ch = nv.get("max_chapters")
            if max_ch and int(max_ch) <= int(nv.get("scene_dedupe_short_novel_chapters", 8)):
                block = max(block, float(nv.get("scene_dedupe_short_novel_block", 0.92)))
            identical = float(nv.get("scene_dedupe_sim_identical", 0.97))
            if (bool(nv.get("scene_dedupe_force_retry", True)) and sim >= block) or sim >= identical:
                return "scene_similarity"
        except Exception:
            pass
    if bool(nv.get("narrative_pattern_enabled", True)):
        try:
            if narrative_pattern_repetition(plan, recent, config).get("level") == "block":
                return "narrative_pattern"
        except Exception:
            pass
    if bool(nv.get("chapter_mode_enabled", True)):
        try:
            if chapter_mode_monotony(plan, recent, config).get("level") == "block":
                return "chapter_mode"
        except Exception:
            pass
    visual_blocks = bool(nv.get("visual_payoff_blocks_plan", True))
    if config_mod.narrative_mode(config) == "serial" and "visual_payoff_blocks_plan" not in nv:
        visual_blocks = False
    if bool(nv.get("visual_payoff_check_enabled", True)) and visual_blocks:
        try:
            if plan_visual_payoff_check(plan, config).get("blocked"):
                return "visual_payoff"
        except Exception:
            pass
    try:
        if plan_executability_gate(plan, config).get("blocked"):
            return "executability"
    except Exception:
        pass
    if score is not None:
        min_score = float(nv.get("min_plan_score", 8.0))
        retry_score = float(nv.get("plan_retry_score_threshold", min_score))
        if score < min_score and score < retry_score:
            return "low_plan_score"
    return None


def replay_novel(name: str, lo: int, hi: int, fixes: set[str], detail: bool):
    novel = ROOT / "novels" / name
    base = novel / "logs" / "checkpoints"
    if not base.is_dir():
        return 0, 0, 0, [], {}, {}
    try:
        cfg = novel_config(name)
    except Exception as exc:
        print(f"  ! {name}: config unreadable ({exc}); skipped")
        return 0, 0, 0, [], {}, {}
    plans = _plans(novel) if "B" in fixes else {}
    window = max(int(cfg["novel"].get("scene_dedupe_window", 8)),
                 int(cfg["novel"].get("chapter_mode_window", 6)))

    old_ok = new_ok = n = 0
    lines: list[str] = []
    still: dict[str, int] = {}
    left: dict[str, int] = {}
    for chdir in sorted(base.glob("ch*")):
        if not chdir.is_dir():
            continue
        ch = int(re.sub(r"\D", "", chdir.name) or 0)
        if not (lo <= ch <= hi):
            continue
        r0 = _payload(chdir / "review_round0.json")
        if r0 is None:
            continue
        r0 = _normalize(r0)
        n += 1

        replans = set()
        for pat in COUNTED_REPLANS:
            replans |= {p.name.split("_attempt")[0] for p in chdir.glob(pat)}

        def reasons(rep: set[str], payload: dict) -> list[str]:
            return [f"replanned:{l.replace('plan_', '')}" for l in sorted(rep)] \
                + _hard_block_reasons(payload, PINNED)

        old = reasons(replans, r0)

        patched = dict(r0)
        new_replans = set(replans)
        why: list[str] = []

        if "A" in fixes:
            text = _chapter_text(novel, ch)
            keep = []
            for g in (r0.get("gate_rejects") or []):
                if isinstance(g, dict) and g.get("gate") == "book_wide_fossils_ratio":
                    phrases = [str(p) for p in (g.get("phrases") or []) if p]
                    # No recorded evidence -> cannot prove absence, keep the reject.
                    if phrases and not any(p in text for p in phrases):
                        why.append("A:fossil_not_in_chapter")
                        continue
                keep.append(g)
            patched["gate_rejects"] = keep

        if "B" in fixes and GATED_REPLAN in replans:
            plan = plans.get(ch)
            if plan:
                recent = [plans[c] for c in
                          sorted((c for c in plans if c < ch), reverse=True)[:window]]
                blocker = plan_chain_block(plan, recent, _plan_score(novel, ch), cfg)
                if blocker is None:
                    new_replans.discard(GATED_REPLAN)
                    why.append("B:plan_chain_clear")
                else:
                    still[blocker] = still.get(blocker, 0) + 1

        new = reasons(new_replans, patched)
        old_ok += not old
        new_ok += not new
        for r in new:
            k = r.split("=")[0].split("×")[0]
            left[k] = left.get(k, 0) + 1
        if detail and old != new:
            lines.append(f"    Ch{ch:>3}: {', '.join(old) or 'PASS'}"
                         f"  ->  {', '.join(new) or 'PASS'}   [{','.join(why)}]")
    return old_ok, new_ok, n, lines, still, left


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("novels", nargs="*")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=10 ** 9)
    ap.add_argument("--fix", default="A,B", help="comma-separated subset of A,B")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="include derivative dirs excluded from the aggregate by default")
    args = ap.parse_args()

    fixes = {f.strip().upper() for f in args.fix.split(",") if f.strip()}
    names, dropped = discover_novels(args.novels, include_all=args.all)
    if not names:
        print("no novels with checkpoints found")
        return 2

    w = max(len(x) for x in names)
    print(f"fixes active: {','.join(sorted(fixes)) or 'none'}\n")
    print_exclusions(dropped)
    print(f"{'novel':<{w}}  {'FPY before':>13}  {'FPY after':>13}   delta")
    t_old = t_new = t_n = 0
    still_all: dict[str, int] = {}
    left_all: dict[str, int] = {}
    for name in names:
        old, new, n, lines, still, left = replay_novel(name, args.lo, args.hi, fixes, args.detail)
        if not n:
            print(f"{name:<{w}}  {'no chapters':>13}")
            continue
        t_old, t_new, t_n = t_old + old, t_new + new, t_n + n
        for k, v in still.items():
            still_all[k] = still_all.get(k, 0) + v
        for k, v in left.items():
            left_all[k] = left_all.get(k, 0) + v
        print(f"{name:<{w}}  {old:>4}/{n:<3} {100.0*old/n:>5.1f}%  "
              f"{new:>4}/{n:<3} {100.0*new/n:>5.1f}%   {100.0*(new-old)/n:>+5.1f}pt"
              f"{'   <80%' if 100.0*new/n < 80 else ''}")
        for ln in lines:
            print(ln)
    if t_n:
        print(f"\n{'LIBRARY':<{w}}  {t_old:>4}/{t_n:<3} {100.0*t_old/t_n:>5.1f}%  "
              f"{t_new:>4}/{t_n:<3} {100.0*t_new/t_n:>5.1f}%   "
              f"{100.0*(t_new-t_old)/t_n:>+5.1f}pt")
    if still_all:
        print("\nplan-chain replans that SURVIVE the fix, by first blocking gate:")
        for k, v in sorted(still_all.items(), key=lambda kv: -kv[1]):
            print(f"   {v:>4}  {k}")
    if left_all:
        print("\nremaining first-draft failures AFTER the fix, by reason:")
        for k, v in sorted(left_all.items(), key=lambda kv: -kv[1]):
            print(f"   {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
