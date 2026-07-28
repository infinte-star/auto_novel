"""The external anchor: a blinded pairwise prose judge, as a resident component.

v2's third metric is WR — the win rate of a chapter against a reference chapter,
judged by a model that is told nothing about where either came from. It is the
only one of the three metrics that reads prose, and the only reason it is worth
anything is that it is the one judgement the engine cannot award itself: FPY' and
CCR both measure whether the engine did what it said it would, which a
sufficiently timid engine can max out by promising nothing.

This module is the promotion of `tools/pairwise_ab.py` from a one-off P4
settlement script into something `v2/run.py` and the A/B harness both call. Four
properties are load-bearing and each one is here because dropping it produced a
number that looked fine and meant nothing:

* **Blind.** Sides are labelled 甲/乙. Arm names, engine versions, and file paths
  never reach the model.
* **Two-way.** Every pair is judged twice with the sides swapped, and a win counts
  only when both orders agree. LLM judges have a documented position preference;
  a single-order run measures that as much as the prose. `tally` reports the flip
  rate so a run that mostly measured position bias says so out loud.
* **No cacheable_prefix.** Same reason `review.cold_reader_review` omits it — a
  judge steeped in the book's own context ratifies the book's own drift. Enforced
  structurally: this module never imports `memory`, so there is no prefix to add.
* **Log isolation.** `judge_paths` redirects `logs_dir`, because `llm.call_llm`
  appends to `paths.logs_dir/llm_calls.jsonl` and that is the exact file
  `compare._llm_totals` reads for calls/chapter. Measured 2026-07-28: 10 judge
  calls landed in an arm's log and inflated its cost by ~0.5 calls/ch inside the
  very report the judge existed to produce (CLAUDE.md, "Offline tools must not log
  into the novel they measure").

The LLM call is INJECTED (`call=`), not imported. That keeps the whole module
importable and unit-testable with zero API access, and keeps the judging logic —
which is where the position-bias arithmetic lives — separable from the plumbing.

Anchor sets: `anchor_chapters()` reads human-written reference chapters from
`benchmarks/anchor/`. As of 2026-07-28 that directory does not exist; the only
things under `benchmarks/` are pattern NOTES about 爆款 structure, which are not
prose and must never be fed to a prose judge. So `wr_against_anchor` returns
`{"available": False}` with a reason rather than quietly substituting an arm-vs-arm
comparison, which would report an internal A/B under the name of an external one.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# The rubric. ONE copy: `tools/pairwise_ab.py` imports these rather than keeping
# its own, for the same reason `hard_block_reasons` lives in exactly one place.
# Two arms judged by two rubrics are not comparable, and a rubric that drifts
# between experiments quietly invalidates every earlier WR number.
# --------------------------------------------------------------------------
# The one sentence that states what the two texts ARE. Parameterized because it
# is the only part of the rubric that can be FALSE: matched pairs really are two
# versions of one chapter, but an A/B whose arms sit at different outline
# positions (v1's finished Ch171-200 vs v2's regenerated Ch201-230) is not, and
# telling the judge otherwise asks it to explain away a plot discontinuity that
# does not exist -- while inviting it to score whichever excerpt happens to sit
# nearer a climax. The five scoring criteria are IDENTICAL under both premises, so
# a WR measured either way is measured by one ruler; only the framing moves.
# `JUDGE_SYSTEM` is the matched build and is byte-identical to the original, so
# every existing caller's prompt (and its provider cache entry) is unchanged.
PREMISE_MATCHED = ("你会看到同一章的两个版本（甲、乙），"
                   "它们出自同一部书、同一章号、同一大纲位置。")
PREMISE_UNMATCHED = (
    "你会看到同一部书里两段不同章节的正文（甲、乙），剧情位置不同。\n"
    "因此**不要**比较哪一段剧情更重要、更关键、更靠近高潮——那是大纲安排的，不是写作质量。\n"
    "只比较：把这一章单独递给一个陌生读者，哪一章更能让他读下去。")

JUDGE_SYSTEM_TEMPLATE = """你是一位挑剔的网文读者，不是编辑，也不认识作者。
{premise}

只回答一个问题：**作为读者，你更想继续往下读哪一个？**

判断依据，按重要性排序：
1. 阅读牵引力：读完是否想看下一章；有没有具体的悬念，而不是含糊的"气氛"
2. 具体性：场景、动作、物件是否可拍摄；有没有用抽象状态词代替事件
3. 冲突推进：这一章是否真的发生了事情，还是只在铺垫和内心戏里打转
4. 文字健康：句子是否成句；有没有大量破折号断句、电报体、状态罗列
5. 收尾：结尾是否落在一个具体的、让人不安或好奇的画面上

明确不看：篇幅长短本身、辞藻华丽程度、是否"文学性"。

如果两者差距在你自己也说不清的范围内，就判 tie——不要为了给结论而给结论。"""

JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(premise=PREMISE_MATCHED)
JUDGE_SYSTEM_UNMATCHED = JUDGE_SYSTEM_TEMPLATE.format(premise=PREMISE_UNMATCHED)

JUDGE_USER = """【甲】
{first}

────────────────────

【乙】
{second}

────────────────────

输出 JSON：{{"winner": "甲" | "乙" | "tie", "reason": "一句话，指出决定性的具体差别"}}"""

SIDE_FIRST = "甲"
SIDE_SECOND = "乙"

# Both orders in which one pair is judged. The tuple is (which arm sits in 甲,
# which sits in 乙); the judge's answer is mapped back through it.
ORDERS: tuple[tuple[str, str], ...] = (("a", "b"), ("b", "a"))

# A judge fed a truncated chapter is judging truncation. 12000 chars is ~2x the
# longest chapter the library has produced, so this only ever catches a caller
# passing a whole book by mistake.
MAX_CHAPTER_CHARS = 12000

# An anchor file shorter than this is a note, not a chapter.
ANCHOR_MIN_CHARS = 1200
DEFAULT_ANCHOR_DIR = "benchmarks/anchor"

CallFn = Callable[[str, str], str]


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------
# The verdict of a pair the judge never actually answered. NOT a tie: `tally`
# scores a tie as half a win, so laundering failures into ties drags the win rate
# toward exactly 50% and makes a dead gateway look like "the two arms are
# indistinguishable". Measured: six consecutive gateway failures on the ts_v1arm
# vs ts_v2arm pair reported `WR=50.0% interpretable=True` off zero evidence.
UNMEASURED = "error"


@dataclasses.dataclass(frozen=True)
class Verdict:
    """One pair, judged in both orders.

    `winner` is already position-debiased: it is "a"/"b" only when both orders
    agreed, "tie" otherwise. `orders` keeps the two raw answers so a caller can
    tell an honest tie ("the judge said tie twice") from indecision ("the judge
    picked whichever side went first"), which are very different evidence.

    A pair the judge failed to answer in EITHER order gets `winner == UNMEASURED`
    and is dropped from `tally`'s denominator. Half a measurement is not a
    measurement: with one order missing, the `picks[0] == picks[1]` test can only
    ever say "tie", so keeping it would be a manufactured tie.
    """
    key: str
    winner: str
    orders: tuple[str, str]
    reasons: tuple[str, str]

    @property
    def flipped(self) -> bool:
        """The judge picked a different arm in each order — position bias."""
        return (self.orders[0] != self.orders[1]
                and "tie" not in self.orders
                and UNMEASURED not in self.orders)

    @property
    def flip_side(self) -> str:
        """WHICH position a flip went to: "first" | "second" | "" (not a flip).

        `flipped` alone cannot separate the two causes `tally`'s docstring names.
        The direction can. `ORDERS` is (("a","b"), ("b","a")), so a flip that
        answered "a" then "b" picked whichever text was printed FIRST both times;
        "b" then "a" picked the second both times. A run whose flips are all one
        side is a measured position preference; flips split between the two sides
        are the judge genuinely wavering on near-equal prose.

        This is strictly stronger evidence than `null_pair_probe`, which can only
        show bias on IDENTICAL text — the easy case, where there is nothing to
        rationalize. Measured on the v1/v2 matched settlement: the probe reported
        a 0% first-position rate, yet 25 of 30 real pairs flipped and ALL 25 went
        to the first position (0 to the second). Position preference appears only
        when the texts differ, so a passing probe does not license reading flips
        as "the two arms are indistinguishable".
        """
        if not self.flipped:
            return ""
        return "first" if self.orders[0] == ORDERS[0][0] else "second"

    @property
    def measured(self) -> bool:
        return UNMEASURED not in self.orders

    @property
    def decisive(self) -> bool:
        return self.winner in ("a", "b")


def parse_verdict(raw: str, sides: Sequence[str]) -> tuple[str, str]:
    """Map one raw judge reply onto (arm, reason). Pure; never raises.

    `sides` is the (甲-arm, 乙-arm) assignment for this run. Three outcomes, and
    the distinction between the last two is load-bearing:

    - a side won;
    - the judge answered and called it even -> "tie", real evidence of similarity;
    - the reply could not be read at all -> `UNMEASURED`, no evidence either way.

    An unreadable reply used to return "tie", which `tally` then counted as half a
    win for the arm under test.
    """
    text = str(raw or "")
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return UNMEASURED, "unparseable judge output"
    if not isinstance(obj, dict):
        return UNMEASURED, "unparseable judge output"
    w = str(obj.get("winner", "")).strip()
    reason = str(obj.get("reason", "")).strip()
    if w.startswith(SIDE_FIRST):
        return sides[0], reason
    if w.startswith(SIDE_SECOND):
        return sides[1], reason
    if not w:
        # A dict with no `winner` key is a malformed reply, not a considered tie.
        return UNMEASURED, reason or "judge reply had no winner field"
    return "tie", reason


def judge_pair(text_a: str, text_b: str, *, call: CallFn, key: str = "",
               system: str = JUDGE_SYSTEM) -> Verdict:
    """Judge one pair in both orders. `call(system, user) -> str`.

    Two calls, always — the caller does not get to skip the second one to save a
    call, because a one-order verdict is not the same measurement and mixing the
    two in one tally would be silent.

    `system` defaults to the matched-pair rubric. Pass `JUDGE_SYSTEM_UNMATCHED`
    when the two texts are different chapters; the scoring criteria are the same
    either way, only the premise differs.
    """
    a = str(text_a or "").strip()[:MAX_CHAPTER_CHARS]
    b = str(text_b or "").strip()[:MAX_CHAPTER_CHARS]
    if not a or not b:
        return Verdict(key, UNMEASURED, (UNMEASURED, UNMEASURED),
                       ("missing text", "missing text"))

    picks: list[str] = []
    reasons: list[str] = []
    for sides in ORDERS:
        first, second = (a, b) if sides[0] == "a" else (b, a)
        try:
            raw = call(system, JUDGE_USER.format(first=first, second=second))
        except Exception as exc:  # a dead judge is no measurement, never a win
            picks.append(UNMEASURED)
            reasons.append(f"judge call failed: {exc}")
            continue
        arm, reason = parse_verdict(raw, sides)
        picks.append(arm)
        reasons.append(reason)

    if UNMEASURED in picks:
        winner = UNMEASURED
    else:
        winner = picks[0] if picks[0] == picks[1] else "tie"
    return Verdict(key, winner, (picks[0], picks[1]), (reasons[0], reasons[1]))


def judge_series(pairs: Iterable[tuple[str, str, str]], *, call: CallFn,
                 system: str = JUDGE_SYSTEM,
                 on_verdict: Callable[[Verdict], None] | None = None) -> list[Verdict]:
    """Judge many (key, text_a, text_b) triples. `on_verdict` streams progress."""
    out: list[Verdict] = []
    for key, ta, tb in pairs:
        v = judge_pair(ta, tb, call=call, key=str(key), system=system)
        out.append(v)
        if on_verdict:
            on_verdict(v)
    return out


def null_pair_probe(texts: Sequence[tuple[str, str]], *, call: CallFn,
                    system: str = JUDGE_SYSTEM) -> dict:
    """Judge each text against ITSELF. Calibrates the judge, not the arms.

    A run where most pairs flip on side-swap has two possible causes that the win
    rate cannot separate: the judge prefers a position, or the two arms really are
    indistinguishable. A null pair separates them, because the correct answer is
    known — identical text can only be a tie. Every non-tie answer here is pure
    position preference, measured with no reference to any arm.

    Reported as `first_position_rate` over CALLS rather than over pairs, because
    `judge_pair` already folds a 甲-both-times answer into `tie`: the bias is
    visible in `Verdict.orders`, not in `Verdict.winner`, so counting winners
    would hide exactly what is being probed.

    `system` is the rubric under test, not a fixed one. Probing with the premise
    the experiment actually used is the point — including `JUDGE_SYSTEM_UNMATCHED`,
    where a null pair is a mild lie (it says the texts are different chapters) but
    still the prompt whose bias is in question. Run both premises when the answer
    matters: a bias that appears under only one of them is caused by the premise.
    """
    calls = ties = first = second = unmeasured = 0
    rows: list[dict] = []
    for key, text in texts:
        v = judge_pair(text, text, call=call, key=str(key), system=system)
        for sides, pick in zip(ORDERS, v.orders):
            calls += 1
            if pick == UNMEASURED:
                unmeasured += 1
            elif pick == "tie":
                ties += 1
            elif pick == sides[0]:
                first += 1
            else:
                second += 1
        rows.append({"key": str(key), "orders": list(v.orders)})
    answered = calls - unmeasured
    return {
        "pairs": len(rows), "calls": calls, "ties": ties, "unmeasured": unmeasured,
        "first_position": first, "second_position": second,
        "first_position_rate": (first / answered) if answered else 0.0,
        "tie_rate": (ties / answered) if answered else 0.0,
        # The judge is usable only if it can say "tie" when tie is the truth.
        "usable": bool(answered) and ties * 2 > answered,
        "rows": rows,
    }


def tally(verdicts: Sequence[Verdict], *, arm: str = "b") -> dict:
    """Aggregate. `arm` is the side the pass line is stated on.

    `win_rate` counts ties as half, because the hypothesis under test is almost
    always "the change made the prose worse" and an indistinguishable chapter did
    not. `flip_rate` is reported beside it and is not cosmetic: when more than
    half the pairs flip on side-swap, the run measured the judge's position
    preference and the win rate must not be read as evidence. `interpretable`
    says so in one boolean so a caller cannot forget to check.

    Pairs the judge never answered are counted in `unmeasured` and excluded from
    `n` entirely, so they cannot pull `win_rate` toward 50%. `interpretable` is
    False while ANY pair is unmeasured: a win rate over the surviving half of a
    run is a different measurement from the one that was requested, and the
    difference is invisible in the number itself.

    `n_decisive` is the sample size that actually carries the verdict — flipped
    pairs are folded into `ties`, so a run with 25 flips and 5 decisive pairs has
    `n=30` and `n_decisive=5`, and quoting the 30 overstates the evidence by 6x.
    `flip_bias` says whether the flips were one-sided (a position preference, so
    the instrument had no resolving power on those pairs) or split (the arms were
    genuinely close there); the two readings share a handling — count as tie — but
    not an interpretation.
    """
    measured = [v for v in verdicts if v.measured]
    unmeasured = len(verdicts) - len(measured)
    n = len(measured)
    wins_a = sum(1 for v in measured if v.winner == "a")
    wins_b = sum(1 for v in measured if v.winner == "b")
    ties = n - wins_a - wins_b
    flips = sum(1 for v in measured if v.flipped)
    flips_first = sum(1 for v in measured if v.flip_side == "first")
    flips_second = sum(1 for v in measured if v.flip_side == "second")
    raw_a = sum(1 for v in measured for o in v.orders if o == "a")
    raw_b = sum(1 for v in measured for o in v.orders if o == "b")
    mine = wins_b if arm == "b" else wins_a
    return {
        "n": n,
        "arm": arm,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "unmeasured": unmeasured,
        "flips": flips,
        "flip_rate": (flips / n) if n else 0.0,
        "flips_first_position": flips_first,
        "flips_second_position": flips_second,
        "flip_bias": (max(flips_first, flips_second) / flips) if flips else 0.0,
        "n_decisive": wins_a + wins_b,
        "raw_votes": {"a": raw_a, "b": raw_b, "total": n * 2},
        "win_rate": ((mine + ties * 0.5) / n * 100.0) if n else 0.0,
        "interpretable": bool(n) and flips * 2 <= n and unmeasured == 0,
    }


# --------------------------------------------------------------------------
# Plumbing the caller needs to get right
# --------------------------------------------------------------------------
def judge_paths(paths, root: Path | None = None):
    """A `Paths` whose `logs_dir` is a scratch dir, never a measured novel's.

    Call this on EVERY judging path. Borrowing an arm's config for its API keys is
    fine; borrowing its log file is measurement contamination — the judge's own
    calls get counted as that arm's cost by `compare._llm_totals`.
    """
    d = (root or ROOT) / "experiments" / "pairwise_logs"
    d.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(paths, logs_dir=d)


def llm_caller(client, paths, config, *, tag: str = "anchor_judge",
               temperature: float = 0.0) -> CallFn:
    """Bind `llm.call_llm` into the `call(system, user)` shape `judge_pair` wants.

    Imported lazily so `import v2.anchor` stays free of the LLM stack: the tests
    for the position-bias arithmetic must not need an API key, and `tally` is the
    part most likely to be wrong.

    `paths` must already have been through `judge_paths`; passing a live novel's
    paths here is the contamination bug, so it is refused rather than trusted.
    """
    if Path(getattr(paths, "logs_dir", "")).name != "pairwise_logs":
        raise ValueError(
            "anchor.llm_caller was handed a live novel's paths — wrap them in "
            "anchor.judge_paths() first, or the judge's own calls get counted as "
            "that novel's cost (CLAUDE.md: offline tools must not log into the "
            "novel they measure)")
    from llm import call_llm  # noqa: PLC0415 — deliberately lazy, see docstring

    def _call(system: str, user: str) -> str:
        return call_llm(client, paths, config, system, user,
                        temperature=temperature, json_mode=True, tag=tag)

    return _call


# --------------------------------------------------------------------------
# The frozen external anchor
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AnchorText:
    name: str
    text: str

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:12]


def anchor_chapters(config: dict | None = None, root: Path | None = None
                    ) -> tuple[list[AnchorText], str]:
    """Human reference chapters, plus a reason string when there are none.

    Deliberately NOT `benchmark.py`'s sample library. Those files are notes ABOUT
    爆款 structure ("开篇 300 字内必须给出反常"), not prose, and handing one to a
    prose judge asks it to compare a chapter against an essay. The anchor set is a
    separate directory that must contain actual chapters.
    """
    base = root or ROOT
    nv = (config or {}).get("novel", {}) if isinstance(config, dict) else {}
    d = base / str(nv.get("anchor_dir", DEFAULT_ANCHOR_DIR))
    if not d.is_dir():
        return [], (f"no anchor set: {d.relative_to(base) if d.is_absolute() else d} "
                    f"does not exist. WR against a human anchor is UNMEASURED — do "
                    f"not substitute an arm-vs-arm comparison for it.")
    out: list[AnchorText] = []
    short: list[str] = []
    for p in sorted(d.glob("*")):
        if p.suffix.lower() not in (".md", ".txt") or not p.is_file():
            continue
        if p.name.lower().startswith("readme."):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if len(re.sub(r"\s", "", text)) < ANCHOR_MIN_CHARS:
            short.append(p.name)
            continue
        out.append(AnchorText(p.stem, text))
    if not out:
        why = f"anchor dir {d.name}/ holds no chapter-length file"
        if short:
            why += f" (too short, treated as notes: {', '.join(short)})"
        return [], why
    return out, ""


def anchor_fingerprint(anchors: Sequence[AnchorText]) -> str:
    """Identity of an anchor SET.

    Two WR numbers measured against different anchor sets are different
    measurements wearing the same name. Recording this beside a WR is what lets a
    later reader tell whether a regression is the engine or the ruler.
    """
    h = hashlib.sha1()
    for a in sorted(anchors, key=lambda x: x.name):
        h.update(a.name.encode("utf-8"))
        h.update(a.digest.encode("ascii"))
    return h.hexdigest()[:12]


def wr_against_anchor(chapters: Sequence[tuple[str, str]], *, call: CallFn,
                      config: dict | None = None, root: Path | None = None,
                      on_verdict: Callable[[Verdict], None] | None = None) -> dict:
    """WR of `chapters` (key, text) against the frozen anchor set.

    Every chapter is judged against every anchor, so the denominator is
    len(chapters) × len(anchors) × 2 calls — keep both sides small.

    Returns `{"available": False, "reason": ...}` when there is no anchor set.
    A missing measurement is not a low one (CLAUDE.md), and the caller must print
    the reason rather than fold an absent WR into a pass/fail table.
    """
    anchors, why = anchor_chapters(config, root)
    if not anchors:
        return {"available": False, "reason": why, "n": 0}

    # The generated chapter is always arm "a"; the human anchor is "b". So the
    # headline number is stated on the ENGINE, not on the reference.
    pairs = [(f"{k}~{a.name}", text, a.text) for k, text in chapters for a in anchors]
    verdicts = judge_series(pairs, call=call, on_verdict=on_verdict)
    out = tally(verdicts, arm="a")
    out.update({
        "available": True,
        "anchors": [a.name for a in anchors],
        "anchor_fingerprint": anchor_fingerprint(anchors),
        "verdicts": verdicts,
    })
    return out
