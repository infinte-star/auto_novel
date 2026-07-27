"""Arc-level chapter planning — one plan call per arc instead of five per chapter.

REDESIGN.md L2 (+L5). The committee path in ``planning.create_plan``
(策略老虎机 → N 候选 → 预筛 → 6 轴融合评审 → 仲裁, then seven post-arbitration
gates that can force a whole retry) spends ~6.9 LLM calls and ~827k prompt chars
per chapter to emit ~800 chars of plan, and `replan` is the #1 rework cause in
every novel in the library (33%–68% of chapters — see `novel.py stats`). The
consensus-mud it produces is the problem: committee steps grind off the
specificity the writer needs, the writer fills the gap with invention, invention
is variance, variance is rework.

This module replaces that with ONE call every ``arc_span`` chapters that emits
one ``ChapterCard`` per chapter. A card is deliberately concrete — 场景/阻碍/
反转/钩子 are shootable events, not intentions — plus a ``forbid`` list drawn
from the used-element ledger and an ``opening_type`` that must differ from the
neighbours.

Integration is intentionally shallow so the A/B stays single-variable:

* Cards are projected onto the EXISTING plan schema by :func:`card_to_plan`, so
  writing/review/quality/store are untouched.
* ``planning.create_plan`` consults :func:`plan_from_arc` only for the initial
  plan of a chapter (``checkpoint_label == "initial"``, no replan feedback) and
  only when ``arc_planning_enabled`` is on. Every replan path keeps the
  committee, so a bad card still gets the old safety net.
* Any failure returns ``None`` and the caller falls back to the committee.

FPY comparability: the arc path writes no ``plan_initial_attempt0_candidates.json``
and ``novel._fpy_stats`` keys off marker *presence*, so an arc chapter reads as
first-pass unless something actually reworked it.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint, save_checkpoint
from config import Paths, log, read_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, memory_context, volume_plan_window
from store import (
    db_event,
    get_open_threads,
    get_overdue_reader_promises,
    get_silent_threads,
    recent_metrics,
    validate_plan_continuity,
)

if TYPE_CHECKING:
    from openai import OpenAI

# Opening types the card planner must rotate through. Consecutive chapters
# opening the same way is the cheapest-to-detect form of the sameness that
# `opening_diversity_enabled` currently fights at write time — catching it on
# the card costs zero LLM calls instead of a revise round.
OPENING_TYPES: tuple[str, ...] = (
    "physical_action",   # 角色正在做一件具体的事
    "dialogue",          # 从一句话切入
    "sensory_scene",     # 从环境的一个感官细节切入
    "conflict_inmedias", # 从冲突的中段切入
    "object_detail",     # 从一件具体物件切入
    "aftermath",         # 从某事刚发生完的残局切入
)

# Card fields that must be non-empty for the card to be usable at all.
_REQUIRED_CARD_FIELDS = ("where", "who", "wants", "blocked_by", "turn", "payoff", "exit_hook", "beats")

ARC_SYSTEM = """你是工业化长篇小说引擎中的「弧级规划 agent」。
你一次性为连续若干章生成章节卡片（ChapterCard），而不是逐章拍脑袋。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。

## 输出 schema
{
  "arc_intent": "这一段弧整体要完成的一件事（一句话：从什么局面推进到什么局面）",
  "cards": [
    {
      "ch": 27,
      "title": "章节标题",
      "where": "具体物理场地 + 时间 + 光线/环境条件（例：'县医院三楼旧档案室，凌晨两点，只有应急灯'）",
      "who": ["本章在场且有能动性的人物"],
      "pov_character": "本章主视角人物",
      "wants": "主角本章要拿到的具体东西/达成的具体目标（可验收，不是心情）",
      "blocked_by": "阻挡它的具体障碍——必须是一个能被看见的事实或事件，不是'困难重重'",
      "pressure": "在兑现之前用什么持续压制主角（资源/时间/信任三轴择一或多）",
      "turn": "本章的转折：一句具体事件，读者能一眼看出局面变了",
      "payoff": "读者本章拿到的兑现（可拍：谁用什么动作对什么东西做了什么，产生什么可见结果）",
      "payoff_type": "court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
      "conflict_type": "court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
      "beats": ["5-8 个节拍，每个是完整主谓宾句，含一个可见动作；其中必须有一拍就是上面的 turn"],
      "info_source": "本章推进真相/剧情依赖的主要信息来源",
      "thread_actions": ["本章开启/推进/回收的伏线，写清具体动作"],
      "world_state_changes": ["本章结束后世界/关系/资源发生的可验证变化"],
      "exit_hook": "章末抛给读者的具体悬念（一个事件，不是一句感叹）",
      "forbid": ["本章明令禁止使用的套路/意象/句式，来自下方已用元素台账"],
      "opening_type": "physical_action|dialogue|sensory_scene|conflict_inmedias|object_detail|aftermath"
    }
  ]
}

## 硬性规则
1. `cards` 的长度和 `ch` 必须与「请求」里给定的章号列表完全一致，一章一张，不多不少。
2. 每张卡片的 `where` / `blocked_by` / `turn` / `payoff` / `exit_hook` 都必须是**具体事件或具体物**，
   不得是抽象意图。凡出现"推导出/意识到/想通/完成/还原/引导/反应过来"而没有具体动作+具体物体+可见结果的，
   一律改写成可拍句子。这是本引擎历史上首稿写崩的头号成因。
3. 相邻章节的 `opening_type` 必须不同；任意连续 5 章内同一 `opening_type` 不得出现两次以上。
4. 相邻章节的 `where`（场地）不得连续三章相同；`payoff_type` 不得连续三章相同。
5. 这一段弧必须有整体推进：最后一张卡片结束时的局面，必须明显不同于第一张卡片开始时的局面。
   禁止把 N 章写成同一件事的 N 次重复演示。
6. 伏线错峰：一章最多回收 1 条主线索。逾期未收的伏线必须在本弧内排进具体某一章，不要平均分摊。
7. `forbid` 必须从下方「已用元素台账」里挑真实存在的条目，不要编造空泛禁令。
8. 若给定章号包含全书终章，该章 `exit_hook` 改为收束/余韵，不得抛新悬念，且不得引入新人物/新势力。"""

CARD_REPAIR_SYSTEM = """你是章节卡片的定点修复器。
你会收到一张 ChapterCard 和一份「必须消除的问题清单」。
请只修改与问题相关的字段，其余字段原样保留（包括 ch）。
只返回恰好一个合法的 JSON 对象，即修复后的完整卡片，不要输出其它任何内容。
修复原则：
- 具体化优先：把抽象意图改写成"某角色用具体动作操作具体物体、产生可见结果"的可拍句子。
- 换掉重复的场地/开场/兑现方式时，必须同时改写 beats 让新选择真正落地，不要只改标签。
- 与既有设定冲突时，服从既有设定改卡片，不要试图改设定。"""


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def arc_enabled(config: dict[str, Any]) -> bool:
    return bool(config["novel"].get("arc_planning_enabled", False))


def arc_span(config: dict[str, Any]) -> int:
    """Chapters per arc call. Clamped: <3 loses the arc's whole point, >20
    makes the last cards guess too far past the state they'll actually meet."""
    try:
        span = int(config["novel"].get("arc_span", 10) or 10)
    except (TypeError, ValueError):
        span = 10
    return max(3, min(20, span))


def arc_window(chapter_num: int, span: int, max_chapters: int = 0) -> tuple[int, int]:
    """Inclusive chapter range of the arc block containing ``chapter_num``.

    Blocks are anchored at chapter 1 so the window is a pure function of the
    chapter number — a resumed or forked run recomputes the same boundaries
    without needing to remember when arc planning was switched on.
    """
    span = max(1, span)
    start = ((max(1, chapter_num) - 1) // span) * span + 1
    end = start + span - 1
    if max_chapters > 0:
        end = min(end, max_chapters)
    return start, max(start, end)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    return [text] if text else []


_PUNCT_RE = re.compile(r"[\s，。、；：,.;:!?！？「」“”\"'‘’（）()—…·]+")


def _bigrams(text: str) -> set[str]:
    t = _PUNCT_RE.sub("", text or "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _turn_covered_by_beats(turn: str, beats: list[str], cut: float = 0.6) -> bool:
    """Is the turn already told by the beats?

    A prefix match is not enough: the arc model habitually writes the turn as
    one beat's action plus the next beat's result ("刮开门底的腐蚀缝" /
    "勾出电报纸和铜钥匙"), phrased just differently enough that the first dozen
    characters diverge. Folding the turn in on top of that hands the writer the
    same moment three times. Compare character-bigram containment against each
    beat AND each adjacent pair, which catches the split case.
    """
    tb = _bigrams(turn)
    if not tb:
        return True
    windows = list(beats) + [beats[i] + beats[i + 1] for i in range(len(beats) - 1)]
    return any(len(tb & _bigrams(w)) / len(tb) >= cut for w in windows)


def normalize_card(raw: Any, chapter_num: int) -> dict[str, Any] | None:
    """Coerce one raw LLM card into the canonical shape, or None if unusable.

    The model reliably produces the right keys but not always the right types
    (a string where a list belongs, a missing beat for the turn). Fixing that
    here is free; letting it through costs a repair call or a bad chapter.
    """
    if not isinstance(raw, dict):
        return None
    card: dict[str, Any] = {
        "ch": chapter_num,
        "title": str(raw.get("title") or "").strip(),
        "where": str(raw.get("where") or "").strip(),
        "who": _as_list(raw.get("who")),
        "pov_character": str(raw.get("pov_character") or "").strip(),
        "wants": str(raw.get("wants") or "").strip(),
        "blocked_by": str(raw.get("blocked_by") or "").strip(),
        "pressure": str(raw.get("pressure") or "").strip(),
        "turn": str(raw.get("turn") or "").strip(),
        "payoff": str(raw.get("payoff") or "").strip(),
        "payoff_type": str(raw.get("payoff_type") or "").strip(),
        "conflict_type": str(raw.get("conflict_type") or "other").strip(),
        "beats": _as_list(raw.get("beats")),
        "info_source": str(raw.get("info_source") or "").strip(),
        "thread_actions": _as_list(raw.get("thread_actions")),
        "world_state_changes": _as_list(raw.get("world_state_changes")),
        "exit_hook": str(raw.get("exit_hook") or "").strip(),
        "forbid": _as_list(raw.get("forbid")),
        "opening_type": str(raw.get("opening_type") or "").strip(),
    }
    if not card["pressure"]:
        card["pressure"] = card["blocked_by"]
    # Check completeness BEFORE folding the turn in, so a card that arrived with
    # no beats at all can't be rescued into "one beat, which is the turn".
    missing = [f for f in _REQUIRED_CARD_FIELDS if not card.get(f)]
    if missing:
        return None
    # The turn IS the chapter's spine; if the model listed it only in `turn`,
    # the writer would never see it as an obligation. Fold it into the beats.
    turn = card["turn"]
    if turn and not _turn_covered_by_beats(turn, card["beats"]):
        insert_at = max(0, len(card["beats"]) * 2 // 3)
        card["beats"].insert(insert_at, turn)
    if card["opening_type"] not in OPENING_TYPES:
        card["opening_type"] = ""
    return card


def card_to_plan(card: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project a ChapterCard onto the plan/decision pair the pipeline expects.

    Downstream only ever reads ``decision["required_constraints"]`` (writing.py,
    review.py) — everything else in `decision` is checkpoint payload. We leave
    ``scores`` empty on purpose: there is no arbiter here, and inventing a score
    would feed a fake number into `chapter_metrics.plan_score` and the writer's
    "当前大纲仲裁分" line. ``planning.plan_score`` returns 0.0 for an empty list
    and every consumer already treats 0 as "no plan score" (see the
    plan-vs-realized calibration, which skips such rows).
    """
    plan: dict[str, Any] = {
        "title": card.get("title", ""),
        "goal": card.get("wants", ""),
        "conflict": card.get("blocked_by", ""),
        "conflict_type": card.get("conflict_type", "other"),
        "payoff": card.get("payoff", ""),
        "payoff_type": card.get("payoff_type", ""),
        "pressure": card.get("pressure", ""),
        "beats": list(card.get("beats") or []),
        "character_focus": list(card.get("who") or []),
        "pov_character": card.get("pov_character", ""),
        "location": card.get("where", ""),
        "info_source": card.get("info_source", ""),
        "world_state_changes": list(card.get("world_state_changes") or []),
        "thread_actions": list(card.get("thread_actions") or []),
        "hook": card.get("exit_hook", ""),
        "risk": "；".join(card.get("forbid") or []) or "无",
        # Card-only fields. writing.py dumps the whole plan into the prompt, so
        # the writer sees these directly — that's the point of carrying them.
        "opening_type": card.get("opening_type", ""),
        "forbid": list(card.get("forbid") or []),
        "turn": card.get("turn", ""),
        "source": "arc_card",
    }
    constraints: list[dict[str, Any]] = []
    if card.get("turn"):
        constraints.append({
            "id": "card_turn",
            "type": "beat_fidelity",
            "constraint": f"本章转折必须在正文中实演：{card['turn']}",
            "check_method": "action",
            "target": card["turn"],
        })
    if card.get("payoff"):
        constraints.append({
            "id": "card_payoff",
            "type": "payoff_delivery",
            "constraint": f"本章兑现必须可见地发生：{card['payoff']}",
            "check_method": "action",
            "target": card["payoff"],
        })
    if card.get("where"):
        constraints.append({
            "id": "card_location",
            "type": "world_logic",
            "constraint": f"本章主场景必须是：{card['where']}",
            "check_method": "location",
            "target": card["where"],
        })
    if card.get("exit_hook"):
        constraints.append({
            "id": "card_hook",
            "type": "hook_setup",
            "constraint": f"章末钩子必须落在：{card['exit_hook']}",
            "check_method": "keyword",
            "target": card["exit_hook"],
        })
    for item in card.get("forbid") or []:
        constraints.append({
            "id": f"card_forbid_{len(constraints)}",
            "type": "other",
            "constraint": f"本章禁止使用：{item}",
            "check_method": "keyword",
            "target": str(item),
        })
    decision: dict[str, Any] = {
        "selected_index": 0,
        "scores": [],
        "merged_plan": plan,
        "required_constraints": constraints,
        "reader_expectation_delta": card.get("turn", ""),
        "planner": "arc",
        "arc_card": card,
    }
    return plan, decision


def validate_card(
    card: dict[str, Any],
    *,
    recent_cards: list[dict[str, Any]],
    continuity_violations: list[str] | None = None,
    scene_sim: float | None = None,
    scene_sim_block: float = 0.85,
) -> list[str]:
    """Pre-write, zero-LLM card checks (REDESIGN.md L5).

    Repairing a card costs ~2k tokens; discovering the same defect after the
    chapter is written costs a write + a review + a replan. ``recent_cards`` are
    the immediately preceding cards, newest last.
    """
    problems: list[str] = []
    for field in _REQUIRED_CARD_FIELDS:
        if not card.get(field):
            problems.append(f"字段 `{field}` 为空，必须补齐一个具体内容。")

    prev = recent_cards[-1] if recent_cards else None
    if prev:
        if card.get("opening_type") and card["opening_type"] == prev.get("opening_type"):
            problems.append(
                f"opening_type 与上一章重复（都是 {card['opening_type']}）；"
                f"必须换成 {', '.join(t for t in OPENING_TYPES if t != card['opening_type'])} 之一，"
                f"并相应改写第一拍。"
            )
        if card.get("where") and card["where"] == prev.get("where"):
            problems.append(f"场地 `{card['where']}` 与上一章完全相同；换一个具体场地并改写相关 beats。")
    last3 = [c.get("payoff_type") for c in recent_cards[-2:]]
    if card.get("payoff_type") and len(last3) == 2 and all(t == card["payoff_type"] for t in last3):
        problems.append(f"payoff_type `{card['payoff_type']}` 已连续三章相同；换一种兑现方式。")

    if scene_sim is not None and scene_sim >= scene_sim_block:
        problems.append(
            f"场景骨架与近期已选章节相似度 {scene_sim:.2f} ≥ {scene_sim_block:.2f}（近乎重写上一章）；"
            f"必须更换 conflict/payoff/pressure 中至少两项的具体内容。"
        )
    for v in continuity_violations or []:
        problems.append(f"与既有设定冲突：{v}")
    return problems


# ---------------------------------------------------------------------------
# card store
# ---------------------------------------------------------------------------

def _cards_path(paths: Paths):
    return paths.logs_dir / "arc_cards.json"


def load_cards(paths: Paths) -> dict[str, Any]:
    """{"cards": {"27": card, ...}, "arcs": {"21": {"intent": ...}}}"""
    path = _cards_path(paths)
    if not path.exists():
        return {"cards": {}, "arcs": {}}
    try:
        data = json.loads(read_text(path))
    except (OSError, ValueError):
        return {"cards": {}, "arcs": {}}
    if not isinstance(data, dict):
        return {"cards": {}, "arcs": {}}
    data.setdefault("cards", {})
    data.setdefault("arcs", {})
    return data


def save_cards(paths: Paths, data: dict[str, Any]) -> None:
    path = _cards_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _recent_cards(store: dict[str, Any], chapter_num: int, lookback: int = 5) -> list[dict[str, Any]]:
    out = []
    for n in range(max(1, chapter_num - lookback), chapter_num):
        card = store.get("cards", {}).get(str(n))
        if isinstance(card, dict):
            out.append(card)
    return out


# ---------------------------------------------------------------------------
# LLM stages
# ---------------------------------------------------------------------------

def _ledger_block(conn: Any, chapter_num: int, limit: int = 12) -> str:
    """Used-element ledger: what the last chapters already spent, so `forbid`
    can name real repeats instead of inventing generic bans."""
    rows = recent_metrics(conn, limit)
    if not rows:
        return "（暂无历史）"
    lines = []
    for r in reversed(rows):
        lines.append(
            f"- Ch{r.get('chapter')}: payoff_type={r.get('payoff_type') or '?'} "
            f"conflict_type={r.get('conflict_type') or '?'} tension={r.get('tension') or '?'} "
            f"score={r.get('score') or '?'}"
        )
    return "\n".join(lines)


def _threads_block(conn: Any, chapter_num: int, config: dict[str, Any]) -> str:
    parts = []
    try:
        silence = int(config["novel"].get("thread_silence_threshold", 10))
        silent = get_silent_threads(conn, chapter_num, silence_threshold=silence)
        if silent:
            parts.append("### 沉默伏线（本弧内必须各自排进某一章回收或推进）\n"
                         + json.dumps(silent, ensure_ascii=False, indent=2))
    except Exception:
        pass
    try:
        grace = int(config["novel"].get("reader_promise_overdue_grace", 15))
        overdue = get_overdue_reader_promises(conn, chapter_num, grace=grace)
        if overdue:
            parts.append("### 逾期未兑现的读者承诺（错峰安排，一章最多收 1 条）\n"
                         + json.dumps(overdue, ensure_ascii=False, indent=2))
    except Exception:
        pass
    try:
        openv = get_open_threads(conn, chapter_num, limit=12)
        if openv:
            parts.append("### 开放伏线\n" + json.dumps(openv, ensure_ascii=False, indent=2))
    except Exception:
        pass
    return "\n\n".join(parts) or "（暂无）"


def generate_arc(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    start_ch: int,
    end_ch: int,
    cached_memory: str | None = None,
) -> dict[str, Any]:
    """ONE call producing every card for chapters ``start_ch..end_ch``.

    This is the call the whole design saves its budget for: it sees the arc as a
    unit, so 错峰兑现 / 场地轮换 / 开场轮换 / 整段推进 are decided once with全局
    视野 instead of being patched chapter-by-chapter by gates that can only see
    backwards.
    """
    chapters = list(range(start_ch, end_ch + 1))
    mem = cached_memory or memory_context(
        paths, conn, config,
        max_chars=int(config["novel"].get("plan_memory_chars", 60000) or 0),
    )
    vp_text = ""
    try:
        vp_text = volume_plan_window(
            read_text(paths.volume_plan), start_ch,
            cap=int(config["novel"].get("volume_plan_chars", 12000) or 12000),
            lookahead=len(chapters) + 1,
        )
    except Exception:
        pass
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    finale_note = ""
    if max_chapters and end_ch >= max_chapters:
        finale_note = (
            f"\n\n## 终章提醒\n第 {max_chapters} 章是全书终章：该章必须收束主线、给出确定谜底，"
            f"exit_hook 改为收束余韵，禁止引入任何新人物/新势力/新悬念。"
        )
    user = f"""## 全局记忆
{mem}

## 卷纲（本弧窗口）
{vp_text or "（暂无）"}

## 伏线状态
{_threads_block(conn, start_ch, config)}

## 已用元素台账（forbid 从这里挑真实重复项）
{_ledger_block(conn, start_ch)}

## 请求
为以下章节各生成一张 ChapterCard，共 {len(chapters)} 张，`ch` 必须严格等于：{chapters}
这一段是一个完整的弧：请先想清楚 arc_intent（从什么局面推进到什么局面），
再把它拆成 {len(chapters)} 章的连续推进，每一章都要有自己的兑现，同时服务于整段的位移。{finale_note}"""

    raw = call_llm(
        client, paths, config, ARC_SYSTEM, json_prompt(user),
        max_tokens=int(config["novel"].get("arc_max_tokens", 32000) or 32000),
        temperature=float(config["novel"].get("arc_temperature", 0.75) or 0.75),
        cacheable_prefix=cacheable_prefix(paths, config),
        tag="arc_plan",
    )
    data = load_json_with_repair(client, paths, config, raw)
    cards_raw = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards_raw, list) or not cards_raw:
        raise RuntimeError("arc planner returned no cards")

    by_ch: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(cards_raw):
        ch = chapters[idx] if idx < len(chapters) else None
        if isinstance(item, dict):
            try:
                declared = int(item.get("ch", 0))
            except (TypeError, ValueError):
                declared = 0
            if declared in chapters:
                ch = declared
        if ch is None:
            continue
        card = normalize_card(item, ch)
        if card:
            by_ch[ch] = card
    if not by_ch:
        raise RuntimeError("arc planner produced no usable card")
    missing = [c for c in chapters if c not in by_ch]
    if missing:
        log(paths, f"[WARN] arc {start_ch}-{end_ch}: no card for chapters {missing}; "
                   f"those chapters will fall back to the committee planner.")
    return {
        "intent": str(data.get("arc_intent") or "").strip() if isinstance(data, dict) else "",
        "cards": by_ch,
        "missing": missing,
    }


def repair_card(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    card: dict[str, Any],
    problems: list[str],
    chapter_num: int,
) -> dict[str, Any] | None:
    """One cheap call to fix a card that failed pre-write validation."""
    user = f"""## 待修复的卡片（第 {chapter_num} 章）
{json.dumps(card, ensure_ascii=False, indent=2)}

## 必须消除的问题（逐条修掉，不要回避）
""" + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(problems)) + """

请返回修复后的完整卡片 JSON（保持 ch 不变，保持 schema 字段齐全）。"""
    raw = call_llm(
        client, paths, config, CARD_REPAIR_SYSTEM, json_prompt(user),
        max_tokens=8000, temperature=0.4, tag="arc_card_repair",
    )
    fixed = load_json_with_repair(client, paths, config, raw, fallback={})
    if isinstance(fixed, dict) and fixed.get("card") and isinstance(fixed["card"], dict):
        fixed = fixed["card"]  # tolerate a {"card": {...}} wrapper
    return normalize_card(fixed, chapter_num)


# ---------------------------------------------------------------------------
# entry point used by planning.create_plan
# ---------------------------------------------------------------------------

def plan_from_arc(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    cached_memory: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return (plan, decision) for ``chapter_num`` from its arc card, or None.

    None means "the caller should run the committee" — a missing card, a failed
    arc call, or a card that stayed broken after one repair attempt. Never
    raises: an arc bug must not be able to stall a run.
    """
    try:
        store = load_cards(paths)
        card = store.get("cards", {}).get(str(chapter_num))

        if not isinstance(card, dict):
            span = arc_span(config)
            max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
            start, end = arc_window(chapter_num, span, max_chapters)
            # Never plan chapters that are already written. This happens on the
            # first arc call of a run that starts mid-block (a fork, or arc
            # planning switched on mid-book): the block containing Ch26 with
            # span 10 is Ch21-30, and cards for Ch21-25 would be both wasted
            # tokens and an invitation to contradict finished prose. The block
            # id stays `start` so windows remain anchored.
            gen_start = max(start, chapter_num)
            log(paths, f"Arc planner: generating cards for Ch{gen_start}-{end} (span={span})")
            arc = generate_arc(client, paths, conn, config, gen_start, end, cached_memory=cached_memory)
            for ch, c in arc["cards"].items():
                store.setdefault("cards", {})[str(ch)] = c
            store.setdefault("arcs", {})[str(start)] = {
                "intent": arc["intent"], "start": gen_start, "end": end,
                "missing": arc["missing"],
            }
            save_cards(paths, store)
            save_checkpoint(paths, chapter_num, "arc_generated.json",
                            {"start": gen_start, "end": end, "intent": arc["intent"],
                             "chapters": sorted(arc["cards"])})
            db_event(conn, chapter_num, "arc_generated",
                     {"start": gen_start, "end": end, "intent": arc["intent"],
                      "count": len(arc["cards"]), "missing": arc["missing"]})
            card = store["cards"].get(str(chapter_num))

        if not isinstance(card, dict):
            log(paths, f"Arc planner: no card for Ch{chapter_num}; falling back to committee.")
            return None

        # --- pre-write validation (zero LLM) -------------------------------
        recent = _recent_cards(store, chapter_num)
        plan, _ = card_to_plan(card)
        violations: list[str] = []
        try:
            violations = validate_plan_continuity(conn, plan, chapter_num, config=config)
        except Exception as exc:
            log(paths, f"Arc planner: continuity check failed (non-fatal) Ch{chapter_num}: {exc}")
        sim = None
        if bool(config["novel"].get("scene_dedupe_enabled", True)):
            try:
                from planning import _recent_selected_plans
                from quality import scene_similarity

                # `_recent_selected_plans` reads `plan_arbitration` events, which
                # the arc path never emits — in an all-arc run it returns nothing
                # and the dedupe check would silently disappear. Cards are the
                # arc arm's record of what was actually planned, so project them
                # too and dedupe against the union.
                lookback = int(config["novel"].get("scene_dedupe_window", 8))
                recent_plans = _recent_selected_plans(
                    conn, lookback=lookback, exclude_chapter=chapter_num,
                )
                recent_plans += [
                    card_to_plan(c)[0] for c in _recent_cards(store, chapter_num, lookback)
                ]
                if recent_plans:
                    sim = float(scene_similarity(plan, recent_plans).get("max_sim", 0.0) or 0.0)
            except Exception as exc:
                log(paths, f"Arc planner: scene-dedupe check failed (non-fatal) Ch{chapter_num}: {exc}")
        # Match the committee's severity policy exactly (pipeline._stage_plan):
        # only CRITICAL violations force a re-plan; everything else — overdue
        # threads, un-cashed setups, "action requires explanation" — is an
        # advisory that rides into required_constraints. Treating advisories as
        # repair triggers would fire a repair call on nearly every chapter
        # (5 fired on the very first arc card), which both eats the cost saving
        # and makes the A/B a two-variable experiment.
        critical = [v for v in violations if v.startswith("CRITICAL")]
        advisories = [v for v in violations if not v.startswith("CRITICAL")]
        problems = validate_card(
            card, recent_cards=recent, continuity_violations=critical, scene_sim=sim,
            scene_sim_block=float(config["novel"].get("scene_dedupe_sim_block", 0.85)),
        )
        if problems:
            log(paths, f"Arc card Ch{chapter_num} failed pre-write validation: {problems}")
            db_event(conn, chapter_num, "arc_card_repair", {"problems": problems, "card": card})
            fixed = repair_card(client, paths, config, card, problems, chapter_num)
            if fixed:
                still = validate_card(fixed, recent_cards=recent)
                if still:
                    log(paths, f"Arc card Ch{chapter_num} still invalid after repair {still}; "
                               f"falling back to committee.")
                    return None
                card = fixed
                store.setdefault("cards", {})[str(chapter_num)] = card
                save_cards(paths, store)
            else:
                log(paths, f"Arc card repair failed Ch{chapter_num}; falling back to committee.")
                return None

        plan, decision = card_to_plan(card)
        if advisories:
            log(paths, f"Continuity advisories Ch{chapter_num}: {advisories}")
            decision.setdefault("required_constraints", []).extend(advisories)
        save_checkpoint(paths, chapter_num, "arc_card.json", card)
        log(paths, f"Arc card Ch{chapter_num} accepted: {card.get('title')} "
                   f"[{card.get('opening_type')}] @ {card.get('where')}")
        return plan, decision
    except Exception as exc:
        log(paths, f"Arc planner failed Ch{chapter_num} ({exc}); falling back to committee.")
        return None
