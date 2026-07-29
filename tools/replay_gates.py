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
    python tools/replay_gates.py --fix book_wide_fossils   # same, by name
    python tools/replay_gates.py tangshuting_e2e --detail

Fixes (`--fix`, comma-separated, by id or gate name, default all). An unknown token
is an ERROR, not a silent no-op: it used to print an authoritative `+0.0pt` for a
fix that actually buys +6.0pt.

  A  book_wide_fossils_ratio now requires the chapter under review to actually
     CONTAIN the entrenched phrase. Replayed by RECOMPUTING the gate against the
     corpus the live engine passes on a first draft (Ch1..n-1 — the chapter under
     review is not saved yet), and dropping the reject when that yields no hard
     fossil. A text-presence proxy is the fallback for when the recompute is
     unavailable; it is strictly weaker, because the corpus rule means a hard
     reject cannot fire on a first draft AT ALL. See `_live_hard_fossils`.

  B  chapter_mode_monotony now measures the monotony FRACTION with the unbiased
     classifier. Replayed by re-running the WHOLE pre-write plan-gate chain on
     the archived attempt0 plan -- not just chapter_mode. That is mandatory: the
     chain is sequential with `continue`, so while chapter_mode was blocking, the
     visual-payoff and executability gates downstream of it never ran. Dropping a
     replan because chapter_mode went quiet, without giving those gates their
     first chance to speak, would fabricate a gain.

  C  a MISSING plan measurement no longer reads as a score of 0.0. `plan_score`
     returns 0.0 for an empty `scores` list, so a vacuous / salvaged arbitration
     used to sit below every threshold and force a full plan round. Replayed by
     re-normalizing the archived decision (`planning._normalize_decision`) and
     passing `None` instead of 0.0 when it still carries no measurement.

     **C implies B**: both are settled by re-running the plan-gate chain, and the
     chain reads today's `quality.py`, so B's fix cannot be un-applied by this
     tool. `--fix C` therefore adds B and says so, rather than quietly reporting
     B+C under C's name.

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

import engine.config as config_mod  # noqa: E402
from engine.quality import hard_block_reasons as _hard_block_reasons  # noqa: E402
from tools.fpy_prime import (COUNTED_REPLANS, PINNED, _normalize,  # noqa: E402
                             _payload, discover_novels, print_exclusions)

# An unknown `--fix` token used to mean "apply nothing", and the report still
# printed an authoritative-looking `+0.0pt` under a header echoing the bad token
# back -- `--fix book_wide_fossils` read as "the fossil fix buys nothing" when it
# actually buys +6.0pt. A measurement tool must fail loudly instead, so the ids
# are declared here and the descriptive gate names are accepted as aliases.
FIX_ALIASES: dict[str, str] = {
    "A": "book_wide_fossils",
    "B": "chapter_mode_monotony",
    "C": "unmeasured_plan_score",
}
FIX_IDS = set(FIX_ALIASES)
_BY_NAME = {v.upper(): k for k, v in FIX_ALIASES.items()}
# Fixes that can only be replayed together (see the C entry in the module docstring).
IMPLIES: dict[str, set[str]] = {"C": {"B"}}
# The subset settled by re-running the pre-write plan-gate chain.
CHAIN_FIXES = {"B", "C"}


def parse_fixes(spec: str) -> tuple[set[str], set[str]]:
    """Parse a `--fix` spec. Returns (fixes, implied). ValueError on unknown token.

    `implied` is reported by the caller so a fix that cannot be isolated never gets
    measured under a name that overstates it.
    """
    out: set[str] = set()
    for tok in (t.strip().upper() for t in str(spec).split(",")):
        if not tok:
            continue
        if tok in FIX_IDS:
            out.add(tok)
        elif tok in _BY_NAME:
            out.add(_BY_NAME[tok])
        else:
            raise ValueError(
                f"unknown --fix token {tok!r}; known: "
                + ", ".join(f"{k}/{v}" for k, v in sorted(FIX_ALIASES.items())))
    implied = {d for f in out for d in IMPLIES.get(f, ())} - out
    return out | implied, implied


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


def _live_hard_fossils(novel: Path, ch: int, cfg: dict) -> list[str] | None:
    """`book_wide_fossils`'s hard set as the LIVE engine would compute it at Ch`ch`.

    The corpus is the load-bearing detail, and a text-presence proxy gets it wrong.
    `review.py` builds `texts` over `range(1, chapter_num + 1)` but guards each entry
    with `p.exists()`, and on a first draft the chapter under review has NOT been
    saved yet (`save_chapter` runs after the review), so the corpus is Ch1..n-1.
    `in_current` is therefore False for every candidate, and a hard reject cannot
    fire on a first draft at all — it needs a resume, where the file is already on
    disk. `v2/run.py:load_corpus` passes the same Ch1..n-1 slice on purpose, so the
    two engines agree.

    Measured on tangshuting Ch185/195/200: corpus Ch1..n-1 yields NO hard fossil,
    while including the chapter yields 「声音压得很低」 at frac 0.41. All three carry an
    archived `book_wide_fossils_ratio` reject, so the proxy — "is the phrase in this
    chapter's text?" — keeps three rejects the current engine cannot produce.

    Returns None when the answer cannot be computed (missing chapters, gate raised),
    which the caller must treat as "keep the archived reject" rather than as a pass.
    """
    try:
        import engine.quality as quality
        texts: dict[int, str] = {}
        for n in range(1, ch):
            t = _chapter_text(novel, n)
            if t:
                texts[n] = t
        if not texts:
            return None
        prompt = novel / "prompt.md"
        wl = quality.fossil_whitelist(
            cfg, prompt.read_text(encoding="utf-8") if prompt.exists() else "")
        bf = quality.book_wide_fossils(texts, cfg, whitelist=wl, current_chapter=ch)
        return [str(h.get("phrase")) for h in (bf.get("hard_fossils") or [])]
    except Exception:
        return None


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


def _plan_score(novel: Path, ch: int, fix_unmeasured: bool = False) -> float | None:
    """The arbiter's self-score for attempt0, or None when it never measured one.

    `fix_unmeasured` replays fix C: normalize the decision's shape first (which
    recovers a bare score row), then return None -- not 0.0 -- when the arbiter
    still produced no `scores`. 0.0 is below every threshold, so reading a missing
    measurement as one fabricated a `low_plan_score` block on 12 archived rounds.
    """
    pay = _payload(novel / "logs" / "checkpoints" / f"ch{ch:04d}"
                   / "plan_initial_attempt0_arbitration.json") or {}
    try:
        from engine.quality import _normalize_decision, decision_has_score, plan_score
        dec = pay.get("decision") or {}
        if fix_unmeasured:
            dec = _normalize_decision(dec)
            if not decision_has_score(dec):
                return None
        return float(plan_score(dec))
    except Exception:
        return None


def plan_chain_block(plan: dict, recent: list[dict], score: float | None,
                     config: dict, chapter_num: int = 0) -> str | None:
    """Re-run the pre-write plan-gate chain in ENGINE ORDER; return the first blocker.

    Order matters and is copied from `planning.create_plan`: duplicate gates
    (scene_similarity / narrative_pattern / chapter_mode) -> visual payoff ->
    executability -> plan score. Each blocker `continue`s, so only the first one
    is ever observable in a live run.
    """
    nv = config["novel"]
    try:
        from engine.quality import (chapter_mode_monotony, narrative_pattern_repetition,
                             plan_executability_gate, plan_visual_payoff_check,
                             scene_similarity)
    except Exception:
        return "import_failed"

    if bool(nv.get("scene_dedupe_enabled", True)):
        try:
            # ONE tier, matching the live engine (`v2/beat.py:_problems` ->
            # `arc.validate_card`). This used to replay v1's three: a short-novel
            # relaxation to 0.92, a 0.97 absolute ceiling, and a
            # `scene_dedupe_force_retry` escape hatch. Dropping them is
            # output-neutral on the archive and provably so -- the highest max_sim
            # in 692 real plans/cards is 0.393, so no branch of either rule was
            # ever reachable (experiments/replay_scene_dedupe.py). Keeping them
            # would have left this tool answering "would TODAY's logic pass this"
            # with yesterday's rule.
            if float(scene_similarity(plan, recent).get("max_sim", 0.0) or 0.0) \
                    >= float(nv.get("scene_dedupe_sim_block", 0.82)):
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
        # Mirror `planning.create_plan`'s real condition, which is NOT "below
        # min_plan_score". A plan in [plan_retry_score, min_plan_score) is accepted
        # WITHOUT a retry to save tokens; only the 收尾 zone (`cost_savings_disabled`)
        # takes that shortcut away. An earlier version of this tool read a key that
        # does not exist (`plan_retry_score_threshold`), so it fell back to
        # min_score and reported `low_plan_score` for every plan under 8.0 -- 41.3%
        # of the library by self-score median, against a true retry rate of 4.0%.
        min_score = float(nv.get("min_plan_score", 8.0))
        retry_score = float(nv.get("plan_retry_score", min_score - 1.5))
        if score < min_score and (score < retry_score
                                  or config_mod.cost_savings_disabled(config, chapter_num)):
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
    plans = _plans(novel) if fixes & CHAIN_FIXES else {}
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
            keep = []
            for g in (r0.get("gate_rejects") or []):
                if isinstance(g, dict) and g.get("gate") == "book_wide_fossils_ratio":
                    live = _live_hard_fossils(novel, ch, cfg)
                    if live is not None and not live:
                        why.append("A:no_hard_fossil_live")
                        continue
                    phrases = [str(p) for p in (g.get("phrases") or []) if p]
                    # Recompute unavailable (or still hard): fall back to the weaker
                    # proxy, which can only ever drop a subset of what the recompute
                    # would. No recorded evidence -> cannot prove absence, keep it.
                    text = _chapter_text(novel, ch)
                    if phrases and not any(p in text for p in phrases):
                        why.append("A:fossil_not_in_chapter")
                        continue
                keep.append(g)
            patched["gate_rejects"] = keep

        if fixes & CHAIN_FIXES and GATED_REPLAN in replans:
            plan = plans.get(ch)
            if plan:
                recent = [plans[c] for c in
                          sorted((c for c in plans if c < ch), reverse=True)[:window]]
                score = _plan_score(novel, ch, fix_unmeasured="C" in fixes)
                blocker = plan_chain_block(plan, recent, score, cfg, ch)
                if blocker is None:
                    new_replans.discard(GATED_REPLAN)
                    why.append("+".join(sorted(fixes & CHAIN_FIXES)) + ":plan_chain_clear")
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
    ap.add_argument("--fix", default=",".join(sorted(FIX_IDS)),
                    help="comma-separated subset of " + ", ".join(
                        f"{k} ({v})" for k, v in sorted(FIX_ALIASES.items())))
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="include derivative dirs excluded from the aggregate by default")
    args = ap.parse_args()

    try:
        fixes, implied = parse_fixes(args.fix)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    names, dropped = discover_novels(args.novels, include_all=args.all)
    if not names:
        print("no novels with checkpoints found")
        return 2

    w = max(len(x) for x in names)
    print("fixes active: " + (", ".join(f"{f}/{FIX_ALIASES[f]}" for f in sorted(fixes))
                              or "none"))
    if implied:
        print("  (" + ", ".join(sorted(implied)) + " added: not separable from the "
              "requested fix -- see the module docstring)")
    print()
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
