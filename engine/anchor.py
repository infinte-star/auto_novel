"""The external anchor: a blinded pairwise prose judge.

v3's third metric is WR — the win rate of a chapter against a reference chapter,
judged by a model that is told nothing about where either came from. It is the
only one of the three metrics that reads prose, and the only reason it is worth
anything is that it is the one judgement the engine cannot award itself.

Four properties are load-bearing:
* Blind — sides are labelled 甲/乙, no arm names or paths reach the model.
* Two-way — every pair judged twice with sides swapped; only same-order wins count.
* No cacheable_prefix — the judge must not be steeped in the book's own context.
* Log isolation — judge calls go to experiments/pairwise_logs/, not into a novel's log.

Anchor sets: `anchor_chapters()` reads human-written reference chapters from
`benchmarks/anchor/`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent

PREMISE_MATCHED = ("你会看到同一章的两个版本（甲、乙），"
                   "它们出自同一部书、同一章号、同一大纲位置。")
PREMISE_UNMATCHED = (
    "你会看到同一部书里两段不同章节的正文（甲、乙），剧情位置不同。\n"
    "因此**不要**比较哪一段剧情更重要、更关键、更靠近高潮——那是大纲安排的，不是写作质量。\n"
    "只比较：把这一章单独递给一个陌生读者，哪一章更能让他读下去。")
PREMISE_ANCHOR = (
    "你会看到两段互不相关的网文正文（甲、乙），出自不同的书，人物和世界都不同。\n"
    "因此**不要**比较哪一段的设定更宏大、剧情更关键、更靠近高潮——那是各自大纲的安排，"
    "不是写作质量。\n"
    "只比较：把这一段单独递给一个陌生读者，哪一段更能让他读下去。")

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
JUDGE_SYSTEM_ANCHOR = JUDGE_SYSTEM_TEMPLATE.format(premise=PREMISE_ANCHOR)

JUDGE_USER = """【甲】
{first}

────────────────────

【乙】
{second}

────────────────────

输出 JSON：{{"winner": "甲" | "乙" | "tie", "reason": "一句话，指出决定性的具体差别"}}"""

SIDE_FIRST = "甲"
SIDE_SECOND = "乙"

ORDERS: tuple[tuple[str, str], ...] = (("a", "b"), ("b", "a"))

MAX_CHAPTER_CHARS = 12000
ANCHOR_MIN_CHARS = 1200
DEFAULT_ANCHOR_DIR = "benchmarks/anchor"

CallFn = Callable[[str, str], str]

UNMEASURED = "error"


@dataclasses.dataclass(frozen=True)
class Verdict:
    key: str
    winner: str
    orders: tuple[str, str]
    reasons: tuple[str, str]

    @property
    def flipped(self) -> bool:
        return (self.orders[0] != self.orders[1]
                and "tie" not in self.orders
                and UNMEASURED not in self.orders)

    @property
    def flip_side(self) -> str:
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
        return UNMEASURED, reason or "judge reply had no winner field"
    return "tie", reason


def judge_pair(text_a: str, text_b: str, *, call: CallFn, key: str = "",
               system: str = JUDGE_SYSTEM) -> Verdict:
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
        except Exception as exc:
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
    out: list[Verdict] = []
    for key, ta, tb in pairs:
        v = judge_pair(ta, tb, call=call, key=str(key), system=system)
        out.append(v)
        if on_verdict:
            on_verdict(v)
    return out


def null_pair_probe(texts: Sequence[tuple[str, str]], *, call: CallFn,
                    system: str = JUDGE_SYSTEM) -> dict:
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
        "usable": bool(answered) and ties * 2 > answered,
        "rows": rows,
    }


def tally(verdicts: Sequence[Verdict], *, arm: str = "b") -> dict:
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


def judge_paths(paths, root: Path | None = None):
    d = (root or ROOT) / "experiments" / "pairwise_logs"
    d.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(paths, logs_dir=d)


def llm_caller(client, paths, config, *, tag: str = "anchor_judge",
               temperature: float = 0.0) -> CallFn:
    if Path(getattr(paths, "logs_dir", "")).name != "pairwise_logs":
        raise ValueError(
            "anchor.llm_caller was handed a live novel's paths — wrap them in "
            "anchor.judge_paths() first, or the judge's own calls get counted as "
            "that novel's cost")
    from engine.llm import call_llm

    def _call(system: str, user: str) -> str:
        return call_llm(client, paths, config, system, user,
                        temperature=temperature, json_mode=True, tag=tag)

    return _call


@dataclasses.dataclass(frozen=True)
class AnchorText:
    name: str
    text: str

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:12]


def anchor_chapters(config: dict | None = None, root: Path | None = None
                    ) -> tuple[list[AnchorText], str]:
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
    h = hashlib.sha1()
    for a in sorted(anchors, key=lambda x: x.name):
        h.update(a.name.encode("utf-8"))
        h.update(a.digest.encode("ascii"))
    return h.hexdigest()[:12]


def wr_against_anchor(chapters: Sequence[tuple[str, str]], *, call: CallFn,
                      config: dict | None = None, root: Path | None = None,
                      on_verdict: Callable[[Verdict], None] | None = None) -> dict:
    anchors, why = anchor_chapters(config, root)
    if not anchors:
        return {"available": False, "reason": why, "n": 0}

    pairs = [(f"{k}~{a.name}", text, a.text) for k, text in chapters for a in anchors]
    verdicts = judge_series(pairs, call=call, system=JUDGE_SYSTEM_ANCHOR,
                            on_verdict=on_verdict)
    out = tally(verdicts, arm="a")
    out.update({
        "available": True,
        "anchors": [a.name for a in anchors],
        "anchor_fingerprint": anchor_fingerprint(anchors),
        "verdicts": verdicts,
    })
    return out
