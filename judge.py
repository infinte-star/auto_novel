"""Agent-native pairwise LLM judge — adapted from writing-harness's pairwise
judge idea, specialized for validating engine changes on serialized fiction.

`compare.py` is deterministic (metrics diff incl. the retention index) but cannot
answer "did the prose/retention actually get *better*". This module adds a blind
pairwise judge for the ablation workflow: same prompt, one config flipped → the
two novels are the same story, so chapter N vs chapter N is directly comparable.

Design (mirrors review.cold_reader_review):
  * NEVER passes cacheable_prefix — the judge must read the two versions as a
    stranger, with none of the writer's drifted context.
  * Position-bias control: every pair is judged twice with A/B swapped; a winner
    is only declared when BOTH orderings agree, else "tie" (LLM judges have a
    well-known first-position bias).
  * Fully non-fatal: any failure degrades to "tie"/empty so `compare --judge`
    never crashes the report.

Entry points:
  judge_pair(...)      — one bias-controlled A-vs-B verdict over two texts
  judge_ablation(...)  — sample matched chapters of two novels, aggregate wins
  judge_vs_gold(...)   — advisory: novel chapters vs a benchmark gold sample
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Paths, log, safe_score
from llm import call_llm, json_prompt, load_json_with_repair

PROJECT_DIR = Path(__file__).resolve().parent
NOVELS_DIR = PROJECT_DIR / "novels"

JUDGE_SYSTEM = """你是一名严格的网文主编，正在对**同一部小说、同一章的两个版本**做盲评。
你不知道哪个版本来自哪种配置，也没读过本书前文——只依据这两段文字本身，从"读者会不会追读下去"的角度判断哪个更好。
重点看：开篇是否抓人、爽点/情绪是否兑现、章末钩子是否有力、文字是否好读不注水、剧情是否有实质推进、有无复读原地打转。

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "winner": "A|B|tie",       // 综合更好的版本
  "hook": "A|B|tie",         // 开篇/章末钩子更抓人
  "retention": "A|B|tie",    // 你更想追读下一章的是哪个
  "prose": "A|B|tie",        // 文字更好读、更少注水
  "payoff": "A|B|tie",       // 爽点/情绪兑现更强
  "reason": "<=80字，判定的核心理由>"
}
果断。可判 tie，但能分出高下时不要偷懒。"""

_FRAME_A = {"A": "a", "B": "b", "tie": "tie"}   # when text_a was shown as "A"
_FRAME_B = {"A": "b", "B": "a", "tie": "tie"}   # when text_b was shown as "A" (swapped)


def _one_judge(
    client: Any, paths: Paths, config: dict[str, Any],
    left: str, right: str, context: str, cap: int, max_tokens: int, temperature: float,
) -> dict[str, Any]:
    user = (
        (f"{context}\n\n" if context else "")
        + f"## 版本 A\n{left[:cap]}\n\n## 版本 B\n{right[:cap]}\n\n请盲评这两个版本。"
    )
    raw = call_llm(
        client, paths, config, JUDGE_SYSTEM, json_prompt(user),
        max_tokens=max_tokens, temperature=temperature,  # NOTE: deliberately no cacheable_prefix
        tag="pairwise_judge",
    )
    return load_json_with_repair(
        client, paths, config, raw,
        fallback={"winner": "tie", "hook": "tie", "retention": "tie",
                  "prose": "tie", "payoff": "tie", "reason": "judge 解析失败"},
    )


def _resolve(v1: str, v2: str) -> str:
    """Bias-controlled resolution: decisive only when both orderings agree."""
    return v1 if v1 == v2 else "tie"


def judge_pair(
    client: Any, paths: Paths, config: dict[str, Any],
    text_a: str, text_b: str, *, context: str = "",
) -> dict[str, Any]:
    """One bias-controlled verdict. Returns winners in the a/b/tie frame."""
    cfg = config.get("novel", {})
    cap = int(cfg.get("judge_chapter_chars", 8000))
    mt = int(cfg.get("judge_max_tokens", 1500))
    temp = float(cfg.get("judge_temperature", 0.2))
    try:
        r1 = _one_judge(client, paths, config, text_a, text_b, context, cap, mt, temp)  # A=text_a
        r2 = _one_judge(client, paths, config, text_b, text_a, context, cap, mt, temp)  # A=text_b (swapped)
    except Exception as exc:
        log(paths, f"Pairwise judge failed (non-fatal): {exc}")
        return {"winner": "tie", "dims": {}, "reason": "judge 调用失败"}
    out: dict[str, Any] = {"reason": str(r1.get("reason", ""))[:120]}
    dims = {}
    for field in ("winner", "hook", "retention", "prose", "payoff"):
        v1 = _FRAME_A.get(str(r1.get(field, "tie")), "tie")
        v2 = _FRAME_B.get(str(r2.get(field, "tie")), "tie")
        resolved = _resolve(v1, v2)
        if field == "winner":
            out["winner"] = resolved
            out["agreed"] = (v1 == v2)
        else:
            dims[field] = resolved
    out["dims"] = dims
    return out


def _chapter_path(name: str, ch: int) -> Path:
    return NOVELS_DIR / name / "chapters" / f"{ch:04d}.md"


def _matched_chapters(name_a: str, name_b: str) -> list[int]:
    da = NOVELS_DIR / name_a / "chapters"
    db = NOVELS_DIR / name_b / "chapters"
    if not da.exists() or not db.exists():
        return []
    a = {int(p.stem) for p in da.glob("[0-9]" * 4 + ".md") if p.stem.isdigit()}
    b = {int(p.stem) for p in db.glob("[0-9]" * 4 + ".md") if p.stem.isdigit()}
    return sorted(a & b)


def _sample_evenly(items: list[int], k: int) -> list[int]:
    if k <= 0 or len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def judge_ablation(
    client: Any, paths: Paths, config: dict[str, Any],
    name_a: str, name_b: str, *, sample: int | None = None,
) -> dict[str, Any]:
    """Blind pairwise judge over matched chapters of two same-story novels.
    Returns aggregate wins (a/b/tie) + per-chapter verdicts. name_a is treated
    as the baseline, name_b as the variant."""
    chapters = _matched_chapters(name_a, name_b)
    if not chapters:
        return {"error": "no matched chapters", "a_wins": 0, "b_wins": 0, "ties": 0, "per_chapter": []}
    k = int(sample if sample is not None else config.get("novel", {}).get("judge_sample_chapters", 8))
    picks = _sample_evenly(chapters, k)
    a_wins = b_wins = ties = 0
    per: list[dict[str, Any]] = []
    for ch in picks:
        try:
            ta = _chapter_path(name_a, ch).read_text(encoding="utf-8", errors="replace")
            tb = _chapter_path(name_b, ch).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        v = judge_pair(client, paths, config, ta, tb, context=f"（这是第 {ch} 章的两个版本）")
        w = v.get("winner", "tie")
        if w == "a":
            a_wins += 1
        elif w == "b":
            b_wins += 1
        else:
            ties += 1
        per.append({"chapter": ch, "winner": w, "dims": v.get("dims", {}), "reason": v.get("reason", "")})
        log(paths, f"Judge Ch{ch}: winner={w} dims={v.get('dims', {})}")
    return {
        "name_a": name_a, "name_b": name_b,
        "judged": len(per), "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
        "per_chapter": per,
    }


def judge_vs_gold(
    client: Any, paths: Paths, config: dict[str, Any],
    name: str, gold_query: str, *, sample: int | None = None,
) -> dict[str, Any]:
    """Advisory: judge a novel's sampled chapters against a benchmark gold sample.
    Non-fatal; returns {} when no gold sample is found."""
    try:
        from benchmark import benchmark_context
        gold = benchmark_context(paths, config, gold_query,
                                 max_chars=int(config.get("novel", {}).get("judge_chapter_chars", 8000)))
    except Exception:
        gold = ""
    if not gold or not gold.strip():
        return {}
    da = NOVELS_DIR / name / "chapters"
    chs = sorted(int(p.stem) for p in da.glob("[0-9]" * 4 + ".md") if p.stem.isdigit()) if da.exists() else []
    if not chs:
        return {}
    picks = _sample_evenly(chs, int(sample if sample is not None else 4))
    wins = ties = losses = 0
    for ch in picks:
        try:
            txt = _chapter_path(name, ch).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        v = judge_pair(client, paths, config, txt, gold, context="（A=本书章节，B=爆款参照样本）")
        w = v.get("winner", "tie")
        wins += (w == "a"); ties += (w == "tie"); losses += (w == "b")
    return {"name": name, "gold_query": gold_query, "chapter_wins_vs_gold": wins, "ties": ties, "losses": losses}
