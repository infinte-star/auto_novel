"""Blinded pairwise judge for a two-arm A/B, on the chapters that actually differ.

This is the anti-self-dealing half of the P4 pass criterion (REDESIGN §7). The
cheap half — calls per chapter — is in `novel.py compare`. The problem with
reading that alone is that P4 *widens the definition of "no rework"*, so FPY and
call count improve mechanically whether or not the chapters got worse. The only
way to know is to read the prose.

So: take the chapters where arm A spent strictly more rework than arm B (that is
exactly where the flipped trigger changed the engine's behaviour), and ask a
judge which chapter is better, blind.

Three things make the verdict worth reading:

* **Blind.** The chapters are labelled 甲/乙. The arm names never reach the model.
* **Position-debiased.** Every pair is judged twice with the sides swapped, and a
  win is only counted when both orders agree. LLM judges have a well-documented
  position preference; a single-order run measures that as much as the prose.
* **No cacheable_prefix.** Same reason `cold_reader_review` omits it — a judge
  steeped in the book's own context ratifies the book's own drift.

    python tools/pairwise_ab.py --a p4_score --b p4_det --from 47 --to 54

Exit code is 0 on a readable verdict, 2 if there is nothing comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JUDGE_SYSTEM = """你是一位挑剔的网文读者，不是编辑，也不认识作者。
你会看到同一章的两个版本（甲、乙），它们出自同一部书、同一章号、同一大纲位置。

只回答一个问题：**作为读者，你更想继续往下读哪一个？**

判断依据，按重要性排序：
1. 阅读牵引力：读完是否想看下一章；有没有具体的悬念，而不是含糊的"气氛"
2. 具体性：场景、动作、物件是否可拍摄；有没有用抽象状态词代替事件
3. 冲突推进：这一章是否真的发生了事情，还是只在铺垫和内心戏里打转
4. 文字健康：句子是否成句；有没有大量破折号断句、电报体、状态罗列
5. 收尾：结尾是否落在一个具体的、让人不安或好奇的画面上

明确不看：篇幅长短本身、辞藻华丽程度、是否"文学性"。

如果两者差距在你自己也说不清的范围内，就判 tie——不要为了给结论而给结论。"""

JUDGE_USER = """【甲】
{first}

────────────────────

【乙】
{second}

────────────────────

输出 JSON：{{"winner": "甲" | "乙" | "tie", "reason": "一句话，指出决定性的具体差别"}}"""


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm whose rework we expect (e.g. the score arm)")
    ap.add_argument("--b", required=True, help="arm we expect to skip it (e.g. the det arm)")
    ap.add_argument("--from", dest="ch_from", type=int, required=True)
    ap.add_argument("--to", dest="ch_to", type=int, required=True)
    ap.add_argument("--all", action="store_true",
                    help="judge every chapter in range, not only the mismatched ones")
    ap.add_argument("--out", default="", help="write the report here (default: experiments/pairwise_<a>_vs_<b>.md)")
    args = ap.parse_args()

    nd_a, nd_b = _load_arm(args.a), _load_arm(args.b)

    # config/paths must be resolved from an arm's own config, and the env var has
    # to be set before `config` is imported (see CLAUDE.md).
    os.environ["NOVEL_CONFIG"] = str((nd_a / "config.yaml").relative_to(ROOT))
    os.environ.setdefault("NOVEL_PROMPT", str((nd_a / "prompt.md").relative_to(ROOT)))
    import config as _config  # noqa: E402
    from compare import _rework_signals  # noqa: E402
    from llm import call_llm  # noqa: E402
    from screenplay import _build_client  # noqa: E402

    cfg = _config.load_config()
    paths = _config.get_paths(cfg)

    pairs: list[tuple[int, str, str, dict, dict]] = []
    skipped: list[str] = []
    for ch in range(args.ch_from, args.ch_to + 1):
        ta, tb = _chapter_text(nd_a, ch), _chapter_text(nd_b, ch)
        if not ta or not tb:
            skipped.append(f"Ch{ch}: missing text in {args.a if not ta else args.b}")
            continue
        sa, sb = _rework_signals(nd_a, ch), _rework_signals(nd_b, ch)
        if not args.all and _rework_cost(sa) <= _rework_cost(sb):
            skipped.append(f"Ch{ch}: A did not spend more (cost {_rework_cost(sa)} vs {_rework_cost(sb)})")
            continue
        pairs.append((ch, ta, tb, sa, sb))

    if not pairs:
        print("no comparable chapters; nothing to judge")
        for s in skipped:
            print("  " + s)
        return 2

    client = _build_client(cfg, paths)
    rows = []
    for ch, ta, tb, sa, sb in pairs:
        verdicts = []
        # Two runs, sides swapped. `sides` records which arm sat in slot 甲.
        for sides, first, second in ((("a", "b"), ta, tb), (("b", "a"), tb, ta)):
            raw = call_llm(
                client, paths, cfg,
                JUDGE_SYSTEM,
                JUDGE_USER.format(first=first, second=second),
                temperature=0.0,
                json_mode=True,
                tag="pairwise_ab",
            )
            try:
                obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            except Exception:
                verdicts.append(("tie", "unparseable judge output"))
                continue
            w = str(obj.get("winner", "tie")).strip()
            arm = sides[0] if w == "甲" else sides[1] if w == "乙" else "tie"
            verdicts.append((arm, str(obj.get("reason", "")).strip()))

        first_arm, second_arm = verdicts[0][0], verdicts[1][0]
        # Agreement across both orders, or it is a tie. A judge that flips with
        # position has told us about its position bias, not about the prose.
        winner = first_arm if first_arm == second_arm else "tie"
        rows.append({
            "ch": ch, "winner": winner,
            "orders": [first_arm, second_arm],
            "reasons": [verdicts[0][1], verdicts[1][1]],
            "sig_a": sa, "sig_b": sb,
            "chars": (len(ta), len(tb)),
        })
        print(f"Ch{ch}: order1={first_arm} order2={second_arm} -> {winner}")

    wins_a = sum(1 for r in rows if r["winner"] == "a")
    wins_b = sum(1 for r in rows if r["winner"] == "b")
    ties = len(rows) - wins_a - wins_b
    # The criterion is stated on arm B (the flipped one): it may not be worse.
    # Ties count for B, because the null hypothesis under test is "removing the
    # rework made the chapter worse" -- an indistinguishable chapter did not.
    rate_b = (wins_b + ties * 0.5) / len(rows) * 100
    # How many ties are "the judge said the same thing twice and it was a tie"
    # versus "the judge changed its mind when we swapped the sides". The second
    # kind is judge indecision, not measured parity, and a run that is mostly
    # flips has not measured anything -- say so instead of reporting a clean 50%.
    flips = sum(1 for r in rows if r["orders"][0] != r["orders"][1]
                and "tie" not in r["orders"])
    raw_a = sum(1 for r in rows for o in r["orders"] if o == "a")
    raw_b = sum(1 for r in rows for o in r["orders"] if o == "b")

    out = Path(args.out) if args.out else ROOT / "experiments" / f"pairwise_{args.a}_vs_{args.b}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Pairwise: {args.a} (A) vs {args.b} (B)  Ch{args.ch_from}-{args.ch_to}",
        "",
        f"判据口径：{'全部章' if args.all else '只判 A 的返工开销严格大于 B 的章'}，n={len(rows)}。",
        "每对判两次并交换甲乙位置，两次一致才计胜，否则计平。judge 不带 cacheable_prefix。",
        "",
        f"- A（{args.a}）胜：**{wins_a}**",
        f"- B（{args.b}）胜：**{wins_b}**",
        f"- 平：**{ties}**",
        f"- **B 的不劣率（含平局折半）：{rate_b:.0f}%**（判据线 ≥50%）",
        "",
        f"判定稳定性：{flips}/{len(rows)} 章在交换甲乙后翻转（判为平）；"
        f"不去重的单次票数 A={raw_a} B={raw_b}（共 {len(rows) * 2} 票）。"
        + ("**翻转率过半，这一轮主要测到的是评委的位置偏好，不是文本差异——不劣率不可当作证据读。**"
           if flips * 2 > len(rows) else ""),
        "",
        "| 章 | 胜方 | 两次顺序 | A 返工（开销） | B 返工（开销） | 字数 A/B | 理由（首次） |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sa, sb = r["sig_a"], r["sig_b"]
        lines.append(
            f"| Ch{r['ch']} | {r['winner']} | {r['orders'][0]}/{r['orders'][1]} | "
            f"replan={sa['replan']} revise={sa['revise']} fix={sa['local_fix']} ({_rework_cost(sa)}) | "
            f"replan={sb['replan']} revise={sb['revise']} fix={sb['local_fix']} ({_rework_cost(sb)}) | "
            f"{r['chars'][0]}/{r['chars'][1]} | {r['reasons'][0][:70]} |"
        )
    if skipped:
        lines += ["", "未参评：", ""] + [f"- {s}" for s in skipped]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nA={wins_a}  B={wins_b}  tie={ties}   B non-inferiority {rate_b:.0f}%"
          f"   position-flips {flips}/{len(rows)}   raw votes A={raw_a} B={raw_b}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
