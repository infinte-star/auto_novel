"""Blinded pairwise judge for a two-arm A/B, on the chapters that actually differ.

This is the anti-self-dealing half of the P4 pass criterion (REDESIGN §7). The
cheap half — calls per chapter — is in `novel.py compare`. The problem with
reading that alone is that P4 *widens the definition of "no rework"*, so FPY and
call count improve mechanically whether or not the chapters got worse. The only
way to know is to read the prose.

So: take the chapters where arm A spent strictly more rework than arm B (that is
exactly where the flipped trigger changed the engine's behaviour), and ask a
judge which chapter is better, blind.

Three things make the verdict worth reading — all three now live in
`v2/anchor.py`, which this script is a CLI over:

* **Blind.** The chapters are labelled 甲/乙. The arm names never reach the model.
* **Position-debiased.** Every pair is judged twice with the sides swapped, and a
  win is only counted when both orders agree. LLM judges have a well-documented
  position preference; a single-order run measures that as much as the prose.
* **No cacheable_prefix.** Same reason `cold_reader_review` omits it — a judge
  steeped in the book's own context ratifies the book's own drift.

The rubric and the position-bias arithmetic are imported rather than kept here so
exactly one copy exists: two experiments settled by two rubrics are not
comparable, and a rubric that drifts silently invalidates every earlier number.
What stays in this file is what is specific to an ENGINE A/B — selecting the
chapters where the arms actually diverged, and the report.

    python tools/pairwise_ab.py --a p4_score --b p4_det --from 47 --to 54
    python tools/pairwise_ab.py --a ts_v2match --from 171 --to 180 --anchor

`--anchor` swaps the reference: instead of the other arm, the chapters are judged
against the frozen human reference set in `benchmarks/anchor/`. That is the only
measurement in the project the engine cannot award itself — an arm-vs-arm WR of
50% says the two engines are indistinguishable from each other, which a pair of
equally bad engines also achieves. **There is no anchor set on disk yet**, so the
mode's first job is to say so and spend nothing: it prints
`anchor_chapters`'s reason and exits 2 before a client is built. A missing
measurement is not a low one (CLAUDE.md), and the whole reason
`wr_against_anchor` returns `available: False` instead of a number is to keep an
absent WR out of a pass/fail table.

Exit code is 0 on a readable verdict, 2 if there is nothing comparable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v2 import anchor  # noqa: E402


def _load_arm(name: str) -> Path:
    d = ROOT / "novels" / name
    if not d.is_dir():
        raise SystemExit(f"no such novel: {name}")
    return d


def _chapter_text(nd: Path, ch: int) -> str:
    f = nd / "chapters" / f"{ch:04d}.md"
    if not f.exists():
        return ""
    return f.read_text(encoding="utf-8", errors="replace").strip()


def _rework_cost(sig: dict) -> int:
    """How much rework this chapter cost, in LLM-call-ish units.

    A plain boolean is too coarse to select on. Measured on the live P4 arms,
    BOTH arms revise on most chapters -- the score arm because a 7.76 is under
    `quality_threshold` 8.0, the det arm because a 6.00 is under
    `rework_score_floor` 6.5. A boolean calls that a match and throws away the
    chapter, when what actually differed is that only the score arm went on to a
    structural replan plus a local fix round.

    So the selector is "A spent strictly more than B", using the same checkpoint
    evidence `novel.py compare` prints. Weights are call counts, roughly: a
    structural replan re-runs the whole planning committee, a plan retry re-runs
    candidate generation, a revise round is one write plus one review.
    """
    return (sig["replan"] * 4 + sig["plan_retry"] * 3
            + sig["revise"] * 2 + sig["local_fix"] + int(sig["debt"]))


def _anchor_mode(args, nd_a: Path, cfg: dict, paths, build_client) -> int:
    """WR of one arm's chapters against the frozen human reference set.

    **The availability check comes before the client**, for the same reason
    `--probe` sits before pair selection: with no anchor set on disk there is
    nothing this run can measure, and spending calls (or requiring a live key) to
    discover that would be pure cost. Today that is the expected path —
    `benchmarks/anchor/` does not exist, and `benchmarks/` holds NOTES about 爆款
    structure, which must never be handed to a prose judge.

    The report states the anchor fingerprint, because two WR numbers measured
    against different anchor sets are different measurements wearing one name.
    """
    anchors, why = anchor.anchor_chapters(cfg, ROOT)
    if not anchors:
        print(f"WR against a human anchor is UNMEASURED: {why}")
        print("Nothing was judged and no LLM call was made. Put real chapter-length "
              "reference prose in benchmarks/anchor/ and re-run; do NOT read an "
              "arm-vs-arm WR as this number.")
        return 2

    chapters = [(str(ch), _chapter_text(nd_a, ch))
                for ch in range(args.ch_from, args.ch_to + 1)]
    missing = [k for k, t in chapters if not t]
    chapters = [(k, t) for k, t in chapters if t]
    if not chapters:
        print(f"no chapter text in {args.a} for Ch{args.ch_from}-{args.ch_to}")
        return 2

    client = build_client(cfg, paths)
    call = anchor.llm_caller(client, paths, cfg, tag="pairwise_anchor")
    print(f"{len(chapters)} chapters x {len(anchors)} anchors x 2 orders = "
          f"{len(chapters) * len(anchors) * 2} calls")
    r = anchor.wr_against_anchor(
        chapters, call=call, config=cfg, root=ROOT,
        on_verdict=lambda v: print(f"{v.key}: order1={v.orders[0]} "
                                   f"order2={v.orders[1]} -> {v.winner}"))

    out = Path(args.out) if args.out else \
        ROOT / "experiments" / f"anchor_wr_{args.a}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Anchor WR: {args.a} Ch{args.ch_from}-{args.ch_to}",
        "",
        f"参考集：{', '.join(r['anchors'])}（指纹 `{r['anchor_fingerprint']}`）。",
        "每章对每个参考章判两次并交换甲乙位置，两次一致才计胜。"
        "判官不知道任何一侧的来源，也不带 cacheable_prefix。",
        "",
        f"- **本书胜率（含平局折半）：{r['win_rate']:.0f}%**",
        f"- 胜 {r['wins_a']} / 负 {r['wins_b']} / 平 {r['ties']}，n={r['n']}，"
        f"决定性 n_decisive=**{r['n_decisive']}**，未测到 {r['unmeasured']}",
        f"- 交换后翻转 {r['flips']}/{r['n']}（倒向先读 "
        f"{r['flips_first_position']}、后读 {r['flips_second_position']}）",
        f"- 可解读（interpretable）：**{r['interpretable']}**",
        "",
        "| 对 | 胜方 | 两次顺序 | 理由（首次） |",
        "|---|---|---|---|",
    ]
    for v in r["verdicts"]:
        reason = v.reasons[0][:70] if v.reasons else ""
        if v.flipped:
            reason = f"（翻转，说法不成立）{reason}"
        lines.append(f"| {v.key} | {v.winner} | {v.orders[0]}/{v.orders[1]} "
                     f"| {reason} |")
    if missing:
        lines += ["", f"未参评（无正文）：{', '.join('Ch' + m for m in missing)}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWR {r['win_rate']:.0f}%  n={r['n']}  n_decisive={r['n_decisive']}"
          f"  flips={r['flips']}  interpretable={r['interpretable']}")
    print(f"report -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm whose rework we expect (e.g. the score arm)")
    ap.add_argument("--b", default="", help="arm we expect to skip it (e.g. the det arm); "
                                           "omitted in --anchor mode")
    ap.add_argument("--anchor", action="store_true",
                    help="judge --a's chapters against the frozen human reference "
                         "set in benchmarks/anchor/ instead of against arm B. The "
                         "only WR the engine cannot award itself")
    ap.add_argument("--from", dest="ch_from", type=int, required=True)
    ap.add_argument("--to", dest="ch_to", type=int, required=True)
    ap.add_argument("--all", action="store_true",
                    help="judge every chapter in range, not only the mismatched ones")
    ap.add_argument("--b-from", dest="b_from", type=int, default=0,
                    help="pair A's Ch<from..to> against B's chapters starting here "
                         "(UNMATCHED mode: the arms sit at different outline "
                         "positions, so the rework-cost selector is meaningless and "
                         "the judge is told the texts are different chapters)")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="calibrate the judge instead of the arms: judge N of arm "
                         "B's chapters against THEMSELVES under both premises. "
                         "Identical text can only be a tie, so any non-tie answer "
                         "is the judge's position preference, measured with no "
                         "reference to either arm. Costs N*4 calls, judges nothing")
    ap.add_argument("--out", default="", help="write the report here (default: experiments/pairwise_<a>_vs_<b>.md)")
    # `argv` is injected by the flag-validation tests: reaching the validation
    # through the real parser is the only way to know the flags are actually
    # rejected rather than merely documented as unsupported.
    args = ap.parse_args(argv)

    if args.anchor:
        # Every arm-vs-arm switch is meaningless here and would be silently
        # ignored, which is how a report ends up describing a run that never
        # happened. `--probe` self-compares arm B's chapters and there is no arm B.
        bad = [f for f, v in (("--b", args.b), ("--b-from", args.b_from),
                              ("--probe", args.probe), ("--all", args.all)) if v]
        if bad:
            ap.error(f"--anchor judges against benchmarks/anchor/, not another "
                     f"arm; drop {', '.join(bad)}")
    elif not args.b:
        ap.error("--b is required unless --anchor is given")

    nd_a = _load_arm(args.a)
    nd_b = _load_arm(args.b) if args.b else nd_a
    # Unmatched mode exists because an A/B can end up with arms at different
    # positions: when the v1 side is a FINISHED book and the v2 side regenerates
    # the chapters after it, the matched set is however many chapters they happen
    # to share -- 2, in the v1/v2 settlement, both of which flipped on side-swap.
    # n=2 measures the judge's position preference, not the prose. Pairing by
    # offset trades outline-matching (a real loss) for an n large enough to
    # estimate the flip rate at all.
    unmatched = args.b_from > 0
    system = anchor.JUDGE_SYSTEM_UNMATCHED if unmatched else anchor.JUDGE_SYSTEM

    # config/paths must be resolved from an arm's own config, and the env var has
    # to be set before `config` is imported (see CLAUDE.md).
    os.environ["NOVEL_CONFIG"] = str((nd_a / "config.yaml").relative_to(ROOT))
    os.environ.setdefault("NOVEL_PROMPT", str((nd_a / "prompt.md").relative_to(ROOT)))
    import config as _config  # noqa: E402
    from compare import _rework_signals  # noqa: E402
    from llm import build_client  # noqa: E402

    cfg = _config.load_config()
    # The judge borrows arm A's config for its API keys, but it must NOT log into
    # arm A's directory: `call_llm` appends every call to `paths.logs_dir/
    # llm_calls.jsonl`, which is the same file `compare.py` reads to compute
    # calls/chapter. Measured 2026-07-28: 10 `pairwise_ab` rows landed in
    # novels/p4_score/logs and inflated that arm's measured cost by ~0.5 calls/ch
    # in the very report the judge existed to complete.
    paths = anchor.judge_paths(_config.get_paths(cfg), ROOT)

    if args.anchor:
        return _anchor_mode(args, nd_a, cfg, paths, build_client)

    pairs: list[tuple[int, str, str, dict, dict]] = []
    skipped: list[str] = []

    if args.probe:
        # Calibration run. Deliberately before pair selection: if the judge cannot
        # tie a chapter with itself, no amount of pairing fixes the verdict, and
        # spending the arm-judging calls first would just produce a number nobody
        # may read.
        lo = args.b_from or args.ch_from
        texts = [(str(ch), _chapter_text(nd_b, ch)) for ch in range(lo, lo + args.probe)]
        texts = [(k, t) for k, t in texts if t]
        if not texts:
            print("no chapters to probe")
            return 2
        client = build_client(cfg, paths)
        call = anchor.llm_caller(client, paths, cfg, tag="pairwise_probe")
        for label, sys_prompt in (("matched", anchor.JUDGE_SYSTEM),
                                  ("unmatched", anchor.JUDGE_SYSTEM_UNMATCHED)):
            r = anchor.null_pair_probe(texts, call=call, system=sys_prompt)
            print(f"{label:>9} premise: ties {r['ties']}/{r['calls'] - r['unmeasured']}"
                  f"  first-position {r['first_position']}  second {r['second_position']}"
                  f"  unmeasured {r['unmeasured']}"
                  f"   first-position rate {r['first_position_rate'] * 100:.0f}%"
                  f"   usable={r['usable']}")
        print("\nA judge that cannot call identical text a tie is measuring position, "
              "not prose; its win rate is not evidence either way.")
        return 0

    for i, ch in enumerate(range(args.ch_from, args.ch_to + 1)):
        ch_b = (args.b_from + i) if unmatched else ch
        ta, tb = _chapter_text(nd_a, ch), _chapter_text(nd_b, ch_b)
        if not ta or not tb:
            skipped.append(f"Ch{ch}: missing text in "
                           f"{args.a if not ta else f'{args.b} (Ch{ch_b})'}")
            continue
        sa, sb = _rework_signals(nd_a, ch), _rework_signals(nd_b, ch_b)
        if not unmatched and not args.all and _rework_cost(sa) <= _rework_cost(sb):
            skipped.append(f"Ch{ch}: A did not spend more (cost {_rework_cost(sa)} vs {_rework_cost(sb)})")
            continue
        pairs.append((ch, ta, tb, sa, sb))

    if not pairs:
        print("no comparable chapters; nothing to judge")
        for s in skipped:
            print("  " + s)
        return 2

    client = build_client(cfg, paths)
    call = anchor.llm_caller(client, paths, cfg, tag="pairwise_ab")

    verdicts = anchor.judge_series(
        [(str(ch), ta, tb) for ch, ta, tb, _, _ in pairs],
        call=call,
        system=system,
        on_verdict=lambda v: print(
            f"Ch{v.key}: order1={v.orders[0]} order2={v.orders[1]} -> {v.winner}"),
    )
    by_ch = {v.key: v for v in verdicts}
    rows = [{"ch": ch, "verdict": by_ch[str(ch)], "sig_a": sa, "sig_b": sb,
             "chars": (len(ta), len(tb))} for ch, ta, tb, sa, sb in pairs]

    # The criterion is stated on arm B (the flipped one): it may not be worse.
    # Ties count for B, because the null hypothesis under test is "removing the
    # rework made the chapter worse" -- an indistinguishable chapter did not.
    t = anchor.tally(verdicts, arm="b")
    wins_a, wins_b, ties = t["wins_a"], t["wins_b"], t["ties"]
    rate_b, flips = t["win_rate"], t["flips"]
    raw_a, raw_b = t["raw_votes"]["a"], t["raw_votes"]["b"]
    n = t["n"]
    # Flip DIRECTION, which separates the two causes the win rate cannot. See
    # `Verdict.flip_side`: one-sided flips are a position preference (the judge had
    # no resolving power on those pairs), split flips are genuinely close prose.
    f1, f2 = t["flips_first_position"], t["flips_second_position"]
    n_dec = t["n_decisive"]

    out = Path(args.out) if args.out else ROOT / "experiments" / f"pairwise_{args.a}_vs_{args.b}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Pairwise: {args.a} (A) vs {args.b} (B)  Ch{args.ch_from}-{args.ch_to}"
        + (f" vs B Ch{args.b_from}-{args.b_from + (args.ch_to - args.ch_from)}"
           if unmatched else ""),
        "",
    ]
    if unmatched:
        lines += [
            "**位置不匹配（offset 配对）。** 两臂不在同一大纲位置，因此：判官被明确告知"
            "这是两个不同章节、不要比较剧情重要性；返工开销筛选器不适用（全判）。"
            "代价是失去了同章号对照，换来的是 n 足够大、翻转率才有估计意义。",
            "",
        ]
    lines += [
        f"判据口径：{'全部章（位置不匹配）' if unmatched else ('全部章' if args.all else '只判 A 的返工开销严格大于 B 的章')}，"
        f"n={n}（判官答上的对数），未测到 {t['unmeasured']} 对。",
        "每对判两次并交换甲乙位置，两次一致才计胜，否则计平。judge 不带 cacheable_prefix。",
        "",
        f"- A（{args.a}）胜：**{wins_a}**",
        f"- B（{args.b}）胜：**{wins_b}**",
        f"- 平：**{ties}**",
        f"- **B 的不劣率（含平局折半）：{rate_b:.0f}%**（判据线 ≥50%）",
        f"- 决定性对数 n_decisive=**{n_dec}**（两次顺序一致的对；余下 {n - n_dec} 对按平计入）",
        f"- 可解读（interpretable）：**{t['interpretable']}**",
        "",
        f"判定稳定性：{flips}/{n} 章在交换甲乙后翻转（判为平），其中"
        f"**倒向先读的那一侧 {f1} 对、倒向后读的 {f2} 对**"
        + (f"（单侧率 {t['flip_bias'] * 100:.0f}%）" if flips else "")
        + "；"
        f"不去重的单次票数 A={raw_a} B={raw_b}（共 {n * 2} 票）。"
        + ("**翻转全部（或几乎全部）倒向同一个位置——这是被测出来的位置偏好，"
           "不是「两臂难分」。这些对上判官没有分辨力，胜负只由 "
           f"n_decisive={n_dec} 对承担，引用 n={n} 会把证据量夸大 "
           f"{(n / n_dec):.0f} 倍。注意：`--probe` 用同一章自比，identical text 上"
           "没有可供合理化的差别，所以校准通过并不能豁免这一条。**"
           if flips and t["flip_bias"] >= 0.8 else "")
        + (f"**翻转过半且两个方向都有（{f1}/{f2}）——这一半是真的难分，按平计入即可，"
           f"但 n_decisive={n_dec} 才是样本量。**"
           if flips * 2 > n and t["flip_bias"] < 0.8 else "")
        + (f"**有 {t['unmeasured']} 对判官没答上，活下来的那半是另一个测量，不劣率不可当作"
           f"请求的那个测量读。**" if t["unmeasured"] else ""),
        "",
        "| 章 | 胜方 | 两次顺序 | A 返工（开销） | B 返工（开销） | 字数 A/B | 理由（首次） |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sa, sb, v = r["sig_a"], r["sig_b"], r["verdict"]
        label = (f"A{r['ch']}/B{args.b_from + (r['ch'] - args.ch_from)}"
                 if unmatched else f"Ch{r['ch']}")
        # On a flipped row the reason is the losing half of a self-contradiction:
        # the same judge argued the other way on side-swap. Printing it unmarked
        # invites mining 25 pro-甲 rationalizations as if they were findings.
        reason = v.reasons[0][:70]
        if v.flipped:
            reason = f"（翻转，仅列首次顺序的说法，不成立）{reason}"
        lines.append(
            f"| {label} | {v.winner} | {v.orders[0]}/{v.orders[1]} | "
            f"replan={sa['replan']} revise={sa['revise']} fix={sa['local_fix']} ({_rework_cost(sa)}) | "
            f"replan={sb['replan']} revise={sb['revise']} fix={sb['local_fix']} ({_rework_cost(sb)}) | "
            f"{r['chars'][0]}/{r['chars'][1]} | {reason} |"
        )
    if skipped:
        lines += ["", "未参评：", ""] + [f"- {s}" for s in skipped]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nA={wins_a}  B={wins_b}  tie={ties}   B non-inferiority {rate_b:.0f}%"
          f"   position-flips {flips}/{n} (first={f1} second={f2})"
          f"   n_decisive={n_dec}   unmeasured {t['unmeasured']}"
          f"   raw votes A={raw_a} B={raw_b}   interpretable={t['interpretable']}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
