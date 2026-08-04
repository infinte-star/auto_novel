"""v3 arc-level planning — card vocabulary, validation, and the arc call.

Merged from v2/beat.py + arc.py + volume functions from memory.py.
One arc call every `arc_span` chapters produces a ChapterCard per chapter.
The card is the chapter's contract — the same seven fields acceptance scores.
"""
from __future__ import annotations

import dataclasses
import copy
import json
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from engine import loop as canon

from engine.checkpoint import save_checkpoint
from engine.config import Paths, chapter_path, log, read_text, text_bigrams
from engine.llm import call_llm, json_prompt, load_json_with_repair, safe_json_loads
from engine.story_spine import (
    story_spine_adherence,
    story_spine_entry,
    story_spine_window,
)
from engine.store import db_event, validate_plan_continuity

# ---------------------------------------------------------------------------
# Card vocabulary (inlined from arc.py)
# ---------------------------------------------------------------------------

OPENING_TYPES: tuple[str, ...] = (
    "physical_action",
    "dialogue",
    "sensory_scene",
    "conflict_inmedias",
    "object_detail",
    "aftermath",
)

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
      "payoff_reaction": "爽点发生后谁产生外部反应（对手名+反应动作，或围观者+态度变化）——必须是他人视角，不是主角内心",
      "exit_hook": "章末抛给读者的具体悬念（一个事件，不是一句感叹）",
      "forbid": ["本章明令禁止使用的套路/意象/句式，来自下方已用元素台账"],
      "opening_type": "physical_action|dialogue|sensory_scene|conflict_inmedias|object_detail|aftermath",
      "tension_level": "high|medium|low — 本章情绪张力目标（可选）",
      "hook_type": "悬念|反转|情绪|信息投放|威胁倒计时|温馨治愈 — exit_hook 的类型（可选，用于轮换检查）",
      "emotion_target": "热血|紧张|温馨|虐心|爽快|敬畏|释然|好奇 等 — 本章要唤起的主导情绪（可选）",
      "payoff_level": "small|medium|large — 本章爽点级别：small=路人反应/小胜利/信息揭示，medium=完整打脸/关系突破/实力跃升，large=多伏笔回收/boss级反转/身份曝光"
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
8. 若给定章号包含全书终章，该章 `exit_hook` 改为收束/余韵，不得抛新悬念，且不得引入新人物/新势力。
9. 弧内节奏必须形成可见波形：不得连续3章同为 high，也不得连续3章同为 low；最后2章至少1章为 high。
10. 兑现按 `small / medium / large` 分级：每章至少 small，弧内至少一次 medium 和一次 large；medium/large 前必须有明确 pressure 铺垫，兑现后必须写 `payoff_reaction`。
11. `hook_type` 必须轮换，连续3章不得相同；不得用同一事件换措辞充当不同章节的 payoff 或 exit_hook。
12. 若输入含「故事脊柱」，它是作者原始简报的不可改写排期。每张卡必须逐字保留该章的硬锚点，并落实其人物、地点、任务等级、关键对象与结局功能；卷纲、世界观或候选创意与之冲突时一律服从故事脊柱。

## 情绪与节奏编排（爆款方法论）
- 用“压迫→选择→兑现→外部余波”组织中、大兑现；各阶段必须是页面上可演的事件。
- 情绪波形服务因果推进，不为凑节奏插入无状态变化的过渡章。
- 弧末先完成本弧位移，再用一个具体后果连接下一弧；终章例外，以收束余韵结束。"""

CARD_REPAIR_SYSTEM = """你是章节卡片的定点修复器。
你会收到一张 ChapterCard 和一份「必须消除的问题清单」。
请只修改与问题相关的字段，其余字段原样保留（包括 ch）。
只返回恰好一个合法的 JSON 对象，即修复后的完整卡片，不要输出其它任何内容。
修复原则：
- 具体化优先：把抽象意图改写成"某角色用具体动作操作具体物体、产生可见结果"的可拍句子。
- 换掉重复的场地/开场/兑现方式时，必须同时改写 beats 让新选择真正落地，不要只改标签。
- 与既有设定冲突时，服从既有设定改卡片，不要试图改设定。
- 问题若以 `story_spine:` 开头，必须把列出的原始简报锚点逐字写回卡片的 title/where/who/turn/payoff/beats/exit_hook；不得用卷纲新增设定替代。"""


def _ground_card_to_story_spine(
    card: dict[str, Any] | None, spine: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Copy author-supplied chapter facts back into a generated card.

    This is not a creative fallback: every appended line is either a literal
    requirement from ``prompt.md`` or a structured relation deterministically
    extracted from it.  Prose is still judged independently, so metadata here
    cannot self-certify a bad chapter.
    """
    if not isinstance(card, dict) or not isinstance(spine, dict) or not spine:
        return card
    grounded = copy.deepcopy(card)
    beats = [str(x).strip() for x in (grounded.get("beats") or []) if str(x).strip()]

    def add_beat(value: str) -> None:
        value = str(value or "").strip()
        if value and value not in beats:
            beats.append(value)

    for requirement in (spine.get("requirements") or []):
        if str(requirement).strip():
            add_beat("原始简报钉死事件：" + str(requirement).strip())

    grade = str(spine.get("expected_grade") or "").strip()
    if grade:
        add_beat(f"系统面板必须明确显示：本章是{grade}。")

    recipient = str(spine.get("named_recipient") or "").strip()
    if recipient:
        card_text = json.dumps(grounded, ensure_ascii=False)
        assignments: list[str] = []
        for match in re.finditer(
            r"(?:目标)?收件人\s*(?:(?:必须|正是|就是|为|是)\s*)?[：:=]?\s*"
            r"[\"“「『【]?\s*([^\"”」』】，,。；;\n]{1,20})",
            card_text,
        ):
            value = re.sub(r"(?:本人|本身)$", "", match.group(1)).strip()
            if value:
                assignments.append(value)
        # Do not launder an explicit contradiction such as “收件人是父亲”.
        # Let the normal relation gate reject it and buy a real re-plan.
        if not assignments or all(recipient in value for value in assignments):
            add_beat(f"系统面板必须逐字明确：收件人是{recipient}。")
            who = [str(x).strip() for x in (grounded.get("who") or []) if str(x).strip()]
            if recipient not in who:
                who.append(recipient)
            grounded["who"] = who

    grounded["beats"] = beats
    return grounded


def arc_span(config: dict[str, Any]) -> int:
    try:
        span = int(config["novel"].get("arc_span", 10) or 10)
    except (TypeError, ValueError):
        span = 10
    return max(3, min(20, span))


def arc_window(chapter_num: int, span: int, max_chapters: int = 0) -> tuple[int, int]:
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


def _turn_covered_by_beats(turn: str, beats: list[str], cut: float = 0.6) -> bool:
    tb = text_bigrams(turn, strip="punct")
    if not tb:
        return True
    windows = list(beats) + [beats[i] + beats[i + 1] for i in range(len(beats) - 1)]
    return any(len(tb & text_bigrams(w, strip="punct")) / len(tb) >= cut for w in windows)


def normalize_card(raw: Any, chapter_num: int) -> dict[str, Any] | None:
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
        "payoff_reaction": str(raw.get("payoff_reaction") or "").strip(),
    }
    card["tension_level"] = str(raw.get("tension_level") or "").strip()
    card["hook_type"] = str(raw.get("hook_type") or "").strip()
    card["emotion_target"] = str(raw.get("emotion_target") or "").strip()
    pl = str(raw.get("payoff_level") or "").strip()
    card["payoff_level"] = pl if pl in ("small", "medium", "large") else ""
    if not card["pressure"]:
        card["pressure"] = card["blocked_by"]
    missing = [f for f in _REQUIRED_CARD_FIELDS if not card.get(f)]
    if missing:
        return None
    turn = card["turn"]
    if turn and not _turn_covered_by_beats(turn, card["beats"]):
        insert_at = max(0, len(card["beats"]) * 2 // 3)
        card["beats"].insert(insert_at, turn)
    if card["opening_type"] not in OPENING_TYPES:
        card["opening_type"] = ""
    if card["tension_level"] and card["tension_level"] not in ("high", "medium", "low"):
        card["tension_level"] = ""
    return card


def card_to_plan(card: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "opening_type": card.get("opening_type", ""),
        "tension_level": card.get("tension_level", ""),
        "hook_type": card.get("hook_type", ""),
        "emotion_target": card.get("emotion_target", ""),
        "forbid": list(card.get("forbid") or []),
        "turn": card.get("turn", ""),
        "payoff_reaction": card.get("payoff_reaction", ""),
        "payoff_level": card.get("payoff_level", ""),
        "source": "arc_card",
    }
    constraints: list[dict[str, Any]] = []
    if card.get("turn"):
        constraints.append({
            "id": "card_turn", "type": "beat_fidelity",
            "constraint": f"本章转折必须在正文中实演：{card['turn']}",
            "check_method": "action", "target": card["turn"],
        })
    if card.get("payoff"):
        constraints.append({
            "id": "card_payoff", "type": "payoff_delivery",
            "constraint": f"本章兑现必须可见地发生：{card['payoff']}",
            "check_method": "action", "target": card["payoff"],
        })
    if card.get("payoff_reaction"):
        constraints.append({
            "id": "card_payoff_reaction", "type": "payoff_delivery",
            "constraint": f"爽点外化反应必须写到页面上：{card['payoff_reaction']}",
            "check_method": "action", "target": card["payoff_reaction"],
        })
    if card.get("where"):
        constraints.append({
            "id": "card_location", "type": "world_logic",
            "constraint": f"本章主场景必须是：{card['where']}",
            "check_method": "location", "target": card["where"],
        })
    if card.get("exit_hook"):
        constraints.append({
            "id": "card_hook", "type": "hook_setup",
            "constraint": f"章末钩子必须落在：{card['exit_hook']}",
            "check_method": "keyword", "target": card["exit_hook"],
        })
    for item in card.get("forbid") or []:
        constraints.append({
            "id": f"card_forbid_{len(constraints)}", "type": "other",
            "constraint": f"本章禁止使用：{item}",
            "check_method": "keyword", "target": str(item),
        })
    decision: dict[str, Any] = {
        "selected_index": 0, "scores": [],
        "merged_plan": plan, "required_constraints": constraints,
        "reader_expectation_delta": card.get("turn", ""),
        "planner": "arc", "arc_card": card,
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
    tl = card.get("tension_level")
    if tl and recent_cards:
        last_t = recent_cards[-1].get("tension_level")
        if last_t == tl and tl in ("high", "low"):
            label = "读者疲劳" if tl == "high" else "读者弃书"
            alts = "medium 或 low" if tl == "high" else "medium 或 high"
            problems.append(
                f"tension_level `{tl}` 连续两章——{label}风险。"
                f"本章换成 {alts} 张力。")
    last_tensions = [c.get("tension_level") for c in recent_cards[-2:]]
    if card.get("tension_level") and len(last_tensions) == 2 and all(t == card["tension_level"] for t in last_tensions):
        problems.append(
            f"tension_level `{card['tension_level']}` 已连续三章相同；"
            f"调整张力节奏——高→中→低的波浪形更佳。")
    last_hooks = [c.get("hook_type") for c in recent_cards[-2:]]
    if card.get("hook_type") and len(last_hooks) == 2 and all(t == card["hook_type"] for t in last_hooks):
        problems.append(
            f"hook_type `{card['hook_type']}` 已连续三章相同；"
            f"轮换钩子类型以维持新鲜感。")
    last_payoff_levels = [c.get("payoff_level") for c in recent_cards[-3:]]
    filled_pls = [p for p in last_payoff_levels if p]
    if card.get("payoff_level") == "small" and len(filled_pls) >= 3 and all(p == "small" for p in filled_pls):
        problems.append(
            "payoff_level 连续 4 章 small——追读断崖风险。"
            "安排一次 medium 或 large 级爽点（打脸/反转/突破）。")
    if scene_sim is not None and scene_sim >= scene_sim_block:
        problems.append(
            f"场景骨架与近期已选章节相似度 {scene_sim:.2f} ≥ {scene_sim_block:.2f}（近乎重写上一章）；"
            f"必须更换 conflict/payoff/pressure 中至少两项的具体内容。"
        )
    for v in continuity_violations or []:
        problems.append(f"与既有设定冲突：{v}")
    return problems


def check_arc_end_acceleration(cards: list[dict[str, Any]]) -> str | None:
    """Return a problem string if the last 2 cards of an arc both have low tension."""
    if len(cards) < 3:
        return None
    last_two = [c.get("tension_level", "") for c in cards[-2:]]
    filled = [t for t in last_two if t]
    if filled and all(t == "low" for t in filled):
        return (
            f"弧线最后两章 (ch {cards[-2].get('ch')}, {cards[-1].get('ch')}) "
            f"张力均为 low，缺少高潮加速——至少有一章应为 high"
        )
    return None


def check_arc_tension_distribution(cards: list[dict[str, Any]]) -> list[str]:
    """Check overall tension distribution across an arc for wave pattern."""
    if len(cards) < 3:
        return []
    problems = []
    levels = [c.get("tension_level", "") for c in cards if c.get("tension_level")]
    if not levels:
        return []
    from collections import Counter
    counts = Counter(levels)
    total = len(levels)
    if "high" not in counts:
        problems.append(
            f"整弧 {total} 章无 high 张力章——缺少高潮，读者无爽点。"
            f"至少安排 1-2 章 high 张力（弧中段或末尾）。"
        )
    for level, cnt in counts.items():
        if cnt / total > 0.6:
            problems.append(
                f"tension_level `{level}` 占弧内 {cnt}/{total}={cnt/total:.0%}，"
                f"分布过于单调。高→中→低波浪形交替更佳。"
            )
    return problems


def check_arc_payoff_distribution(cards: list[dict[str, Any]]) -> list[str]:
    """Check payoff_level distribution across an arc for density rules."""
    if len(cards) < 3:
        return []
    problems = []
    levels = [c.get("payoff_level", "") for c in cards if c.get("payoff_level")]
    if not levels:
        return []
    if "large" not in levels:
        problems.append(
            f"整弧 {len(cards)} 章无 large 级爽点——缺少弧级高潮。"
            f"至少安排 1 章 large 级爽点（多伏笔回收/boss级反转/身份曝光）。"
        )
    runs = 0
    for pl in levels:
        runs = runs + 1 if pl == "small" else 0
        if runs >= 4:
            problems.append(
                "payoff_level 连续 4+ 章 small——追读断崖风险。"
                "穿插 medium/large 级爽点打破平淡。"
            )
            break
    return problems


# ---------------------------------------------------------------------------
# Card store
# ---------------------------------------------------------------------------

def _cards_path(paths: Paths):
    return paths.logs_dir / "arc_cards.json"


def load_cards(paths: Paths) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# JIT card refinement (v3 Phase 2)
# ---------------------------------------------------------------------------

def refine_card(
    card: dict[str, Any],
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    recent_cards: list[dict[str, Any]] | None = None,
    feed_forward: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Zero-LLM deterministic card refinement.

    Adapts an arc card to actual story state just before writing.
    Returns (patched_card, patch_log) where patch_log lists every change made.
    """
    from engine.store import entity_state_as_of, get_open_threads

    card = dict(card)
    patches: list[str] = []

    # 1. Character availability — drop dead/imprisoned characters from who
    who = list(card.get("who") or [])
    surviving: list[str] = []
    for name in who:
        try:
            st = entity_state_as_of(conn, "character", name, chapter_num)
        except Exception:
            st = {}
        status = str(st.get("status") or st.get("状态") or "").lower()
        if status in ("dead", "死亡", "imprisoned", "被囚", "消失"):
            patches.append(f"移除不可用角色 {name}（状态={status}）")
        else:
            surviving.append(name)
    if len(surviving) != len(who):
        card["who"] = surviving if surviving else who[:1]

    # 2. Thread freshness — check payoff/thread_actions still open
    try:
        open_threads = get_open_threads(conn, chapter_num, limit=50)
    except Exception:
        open_threads = []
    open_names = {str(t.get("thread_name") or t.get("name") or "") for t in open_threads}

    payoff = card.get("payoff", "")
    stale_actions: list[str] = []
    for ta in list(card.get("thread_actions") or []):
        parts = ta.split(":", 1)
        action_thread = parts[0].strip() if parts else ta.strip()
        if action_thread and action_thread not in open_names:
            is_stale = not any(action_thread in n for n in open_names)
            if is_stale:
                stale_actions.append(ta)
                patches.append(f"thread_action '{ta}' 引用的线索已关闭→移除")
    if stale_actions and "thread_actions" in card:
        card["thread_actions"] = [
            ta for ta in card["thread_actions"] if ta not in stale_actions
        ]

    # 3. Neighbor dedup — opening_type / tension / hook vs recent cards
    prev = (recent_cards or [])
    if prev:
        last = prev[-1] if prev else {}
        ot = card.get("opening_type", "")
        if ot and ot == last.get("opening_type"):
            alts = [t for t in OPENING_TYPES if t != ot]
            if alts:
                card["opening_type"] = alts[chapter_num % len(alts)]
                patches.append(f"opening_type '{ot}' 与上章重复→'{card['opening_type']}'")

        tl = card.get("tension_level", "")
        arc_span = int((config or {}).get("novel", {}).get("arc_span", 10))
        arc_pos = (chapter_num - 1) % arc_span if arc_span > 0 else 0
        prev1_t = prev[-1].get("tension_level") if prev else None
        prev2 = [c.get("tension_level") for c in prev[-2:]]
        if tl and prev1_t == tl and tl in ("high", "low"):
            if tl == "high":
                new_tl = "medium" if arc_pos >= arc_span * 0.7 else "medium"
            else:
                new_tl = "high" if arc_span > 0 and 0.3 <= arc_pos / arc_span <= 0.7 else "medium"
            card["tension_level"] = new_tl
            patches.append(f"tension_level '{tl}' 连续两章→'{new_tl}'(弧位{arc_pos}/{arc_span})")
        elif tl and len(prev2) == 2 and all(t == tl for t in prev2):
            rotate = {"high": "medium", "medium": "low", "low": "high"}
            card["tension_level"] = rotate.get(tl, tl)
            patches.append(f"tension_level '{tl}' 连续三章→'{card['tension_level']}'")

        ht = card.get("hook_type", "")
        prev2h = [c.get("hook_type") for c in prev[-2:]]
        if ht and len(prev2h) == 2 and all(t == ht for t in prev2h):
            patches.append(f"hook_type '{ht}' 连续三章重复")

    # 4. Feed-forward injection
    for directive in (feed_forward or []):
        d = directive.strip()
        if not d:
            continue
        forbid = list(card.get("forbid") or [])
        if d not in forbid:
            forbid.append(d)
            card["forbid"] = forbid
            patches.append(f"注入 feed-forward: {d[:40]}")

    return card, patches


# ---------------------------------------------------------------------------
# Volume plan windowing (inlined from memory.py)
# ---------------------------------------------------------------------------

# `volume_plan_window` (+ its `_header_range` / `_filter_schedule_rows` helpers)
# has exactly one definition, in engine.bootstrap: that module is the other live
# caller (`_read_volume_plan` → memory_context / cacheable_prefix). Re-exported
# here because the arc prompt below windows the same file, and because the
# `memory` / `arc` compat shims import the name from engine.plan.
from engine.bootstrap import volume_plan_window  # noqa: E402


def parse_volume_ranges(volume_plan_text: str) -> list[dict[str, Any]]:
    """Parse '## 第N卷：<name>（第A-B章）' headers from a volume_plan. Deterministic.

    Returns [{label, name, start, end, pos}] in document order. Tolerant of
    full/half-width colons and parens.
    """
    ranges: list[dict[str, Any]] = []
    if not volume_plan_text:
        return ranges
    pat = re.compile(
        r"##\s*第\s*([0-9一二三四五六七八九十]+)\s*卷\s*[：:]\s*(.*?)\s*[（(]\s*第\s*(\d+)\s*[-–—~]\s*(\d+)\s*章"
    )
    for m in pat.finditer(volume_plan_text):
        ranges.append({
            "label": m.group(1), "name": m.group(2).strip(),
            "start": int(m.group(3)), "end": int(m.group(4)), "pos": m.start(),
        })
    return ranges


def _volume_goal_head(volume_plan_text: str, vol: dict[str, Any], ranges: list[dict[str, Any]], limit: int = 220) -> str:
    """Extract a volume section's '### 卷目标(O)' text (up to the next volume)."""
    start = int(vol.get("pos", 0))
    later = [r["pos"] for r in ranges if r["pos"] > start]
    section = volume_plan_text[start:(min(later) if later else len(volume_plan_text))]
    m = re.search(r"###\s*卷目标\(?O?\)?\s*\n+(.+?)(?:\n#|\Z)", section, re.S)
    return " ".join(m.group(1).split())[:limit] if m else ""


def volume_transition_directive(chapter_num: int, volume_plan_text: str, config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic volume/arc boundary steer (治本 for arc overstay).

    Parses the volume_plan's 第N卷（第A-B章）ranges. When chapter_num sits in the
    opening `volume_transition_grace` window of a volume (other than the first),
    emits a HARD transition block telling the planner to close the previous
    volume and switch scene/form to this volume's goal — automating the manual
    pivot yeban_guize needed (it overstayed the 城中村 arc to Ch28 because nothing
    enforced the planned Ch21 → 卷二 transition). Mid-volume, emits a light
    context note so the planner stays volume-aware and rotates form. Pure
    parse+inject, no LLM; degrades to an empty block on any failure.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    res: dict[str, Any] = {"level": "ok", "block": "", "volume": None, "is_transition": False}
    if not bool(cfg.get("volume_transition_enabled", True)):
        return res
    ranges = parse_volume_ranges(volume_plan_text)
    if not ranges:
        return res
    cur = next((r for r in ranges if r["start"] <= chapter_num <= r["end"]), None)
    if cur is None:
        cur = max(ranges, key=lambda r: r["end"])  # past all ranges → last (finale) volume
    res["volume"] = f"第{cur['label']}卷 {cur['name']}".strip()
    goal = _volume_goal_head(volume_plan_text, cur, ranges)
    grace = max(int(cfg.get("volume_transition_grace", 2)), 1)
    is_first = cur["start"] <= min(r["start"] for r in ranges)
    in_open_window = 0 <= (chapter_num - cur["start"]) < grace
    if in_open_window and not is_first:
        res["level"] = "transition"
        res["is_transition"] = True
        res["block"] = (
            f"## ⚠ 卷务转场（最高优先级）\n"
            f"本章（第{chapter_num}章）进入【{res['volume']}】开篇转场区。务必：\n"
            f"1. 收束上一卷的场景与悬念——不要延续上一卷的地点/机制/套路继续磨；\n"
            f"2. 把场景与章型切换到本卷设定，推进本卷主线。\n"
            + (f"本卷目标：{goal}\n" if goal else "")
        )
    else:
        res["level"] = "context"
        res["block"] = (
            f"## 本卷定位\n本章属【{res['volume']}】。"
            + (f"本卷目标：{goal}" if goal else "")
            + "\n推进本卷主线，并与近几章的章型/形态错开，避免同型连发。\n"
        )
    return res


# ---------------------------------------------------------------------------
# v2 beat.py code (arc call, ensure_card, etc.)
# ---------------------------------------------------------------------------

# Where the card this chapter was written against is archived. MUST equal
# `tools.ccr_baseline.CARD_CHECKPOINT` — a v2 arm that writes it anywhere else is
# silently measured against a reconstruction of its own plan instead of the real
# card, and the CCR it reports is a different quantity under the same name.
CARD_CHECKPOINT = "chapter_card.json"

# Appended to `arc.ARC_SYSTEM`, never a second copy of it. The arc rules have one
# home; this adds the second layer to the schema and nothing else.
SKELETON_DELTA = """

## 追加输出：下一弧骨架（双层滚动）
除 `cards` 外，还必须输出 `next_arc`：下一段弧的一句话骨架，一章一行。

{
  "arc_intent": "...",
  "cards": [ ... ],
  "next_arc": {
    "intent": "下一段弧整体要完成的一件事（从什么局面推进到什么局面）",
    "chapters": [
      {"ch": 31, "line": "该章要发生的一件具体事（可拍，一句话）"}
    ]
  }
}

`next_arc.chapters` 的章号必须是本弧最后一章之后的连续 N 章（N 与本弧章数相同，
若已接近全书终章则只写到终章为止）。骨架只写「发生什么」，不必写满卡片字段。

若「请求」里给出了「上一弧留下的骨架」，默认按它推进；确因剧情已经偏离而需要改，
可以改，但必须在 `arc_intent` 里用一句话说明改了哪一条、为什么。"""

ARC_SYSTEM_V2 = ARC_SYSTEM + SKELETON_DELTA


class BeatError(RuntimeError):
    """No usable card could be produced for this chapter.

    Raised rather than papered over: v2 writes against the card, `accept` scores
    against the card, and CCR reports against the card. Writing without one is
    not a degraded v2 — it is v1 with the planning removed, reported as v2.
    """


@dataclasses.dataclass(frozen=True)
class CardResult:
    """A card plus the honest provenance of how it was obtained."""

    card: dict[str, Any]
    plan: dict[str, Any]
    decision: dict[str, Any]
    source: str            # stored | arc | repaired | single
    unresolved: tuple[str, ...] = ()   # problems no attempt cleared
    advisories: tuple[str, ...] = ()   # non-CRITICAL continuity notes

    @property
    def degraded(self) -> bool:
        """True when this card cost more than the arc call that should have
        produced it. `run.py` logs it; the A/B counts it. A v2 arm whose cards are
        mostly `single` is not the design being tested."""
        return self.source in {"repaired", "single"} or bool(self.unresolved)


# ---------------------------------------------------------------------------
# skeleton — pure
# ---------------------------------------------------------------------------

def normalize_skeleton(raw: Any, chapters: list[int]) -> dict[str, Any] | None:
    """Coerce a `next_arc` payload into {"intent": str, "chapters": {"31": line}}.

    Chapter numbers outside `chapters` are dropped rather than renumbered: a
    skeleton line the model attached to the wrong chapter is a guess about the
    wrong chapter, and silently sliding it one slot over would launder that.
    """
    if not isinstance(raw, dict):
        return None
    allowed = {int(c) for c in chapters}
    lines: dict[str, str] = {}
    items = raw.get("chapters")
    if isinstance(items, dict):
        items = [{"ch": k, "line": v} for k, v in items.items()]
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ch = int(item.get("ch", 0))
            except (TypeError, ValueError):
                continue
            line = str(item.get("line") or item.get("text") or "").strip()
            if ch in allowed and line:
                lines[str(ch)] = line
    intent = str(raw.get("intent") or "").strip()
    if not lines and not intent:
        return None
    return {"intent": intent, "chapters": lines}


def skeleton_block(skel: dict[str, Any] | None, chapters: list[int]) -> str:
    """Render a stored skeleton for the chapters actually being planned now."""
    if not isinstance(skel, dict):
        return ""
    wanted = [str(c) for c in chapters]
    lines = [f"- Ch{c}: {skel['chapters'][c]}"
             for c in wanted if c in (skel.get("chapters") or {})]
    if not lines and not skel.get("intent"):
        return ""
    head = f"整体意图：{skel['intent']}" if skel.get("intent") else ""
    return "\n".join(p for p in (head, "\n".join(lines)) if p)


def _previous_skeleton(store: dict[str, Any], start: int, span: int) -> dict[str, Any] | None:
    """The skeleton the preceding arc left for this one, if any."""
    prev = store.get("arcs", {}).get(str(max(1, start - span)))
    if not isinstance(prev, dict):
        return None
    skel = prev.get("next_skeleton")
    return skel if isinstance(skel, dict) else None


def _recent_cards(store: dict[str, Any], chapter_num: int,
                  lookback: int = 5) -> list[dict[str, Any]]:
    out = []
    for n in range(max(1, chapter_num - lookback), chapter_num):
        card = store.get("cards", {}).get(str(n))
        if isinstance(card, dict):
            out.append(card)
    return out


# ---------------------------------------------------------------------------
# the arc call
# ---------------------------------------------------------------------------

def _parse(client: Any, paths: Paths, config: dict[str, Any], raw: str) -> dict[str, Any]:
    try:
        data = safe_json_loads(raw)
    except Exception:
        data = load_json_with_repair(client, paths, config, raw, fallback={})
    return data if isinstance(data, dict) else {}


def _volume_plan(paths: Paths, config: dict[str, Any], start: int, span: int) -> str:
    try:
        return volume_plan_window(
            read_text(paths.volume_plan), start,
            cap=int(config["novel"].get("volume_plan_chars", 12000) or 12000),
            lookahead=span + 1,
        )
    except Exception:
        return ""


def _volume_transition(paths: Paths, config: dict[str, Any], chapters: list[int]) -> str:
    """The volume-boundary steer for any chapter in this arc, or "".

    Ported from v1's plan call (`planning.generate_candidate_plans`) rather than
    dropped with it: a volume boundary is a deterministic fact in `volume_plan.md`,
    and the yeban_guize 城中村 overstay-to-Ch28 is what happens when nothing
    enforces the planned Ch21 transition. The arc call is the right consumer
    because an `arc_span` of 10 does NOT align with volume ranges — a boundary
    usually lands mid-arc, and the one call planning across it is this one.

    Only the HARD `transition` level is injected. v1 also emitted a mid-volume
    「本卷定位」 context note; the arc prompt already carries the volume-plan window
    verbatim, so here that note would only restate it.
    """
    try:
        text = read_text(paths.volume_plan)
    except Exception:
        return ""
    if not text:
        return ""
    blocks = []
    for ch in chapters:
        try:
            vt = volume_transition_directive(ch, text, config)
        except Exception:
            continue
        if vt.get("is_transition") and vt.get("block"):
            blocks.append(vt["block"])
    return "\n".join(blocks)


def _fingerprints(conn: Any, config: dict[str, Any]) -> str:
    """The 全书结构指纹 aggregate, or "" — the READ side of what v2 already writes.

    `v2/run.py:705` has been calling `store_chapter_fingerprint` every chapter
    since v2 shipped, and nothing read the table back: the one reader
    (`quality.fingerprint_avoidance_context`) lost its call site with `review.py`.
    A write-only fingerprint library is the most expensive kind of dead code,
    because it looks like a working feature from the schema.

    The arc call is the right consumer, and the only affordable one. v1 pasted
    this into EVERY chapter's plan prompt at ~1.2k chars; here it is read once per
    `arc_span` (10) chapters, so it amortizes to ~120 chars/chapter, and the
    aggregate does not grow with the book (the per-chapter form was 19.6% of the
    largest prompt and grew linearly — LESSONS §8).

    **The function returns the literal string "None", not "", when there is
    nothing to say** — a v1 template convention. Pasting that through would tell
    the planner the word "None" under a header promising overused patterns, so it
    is filtered here rather than trusted.
    """
    if not bool(config["novel"].get("fingerprint_enabled", True)):
        return ""
    try:
        from engine.quality import fingerprint_avoidance_context

        text = (fingerprint_avoidance_context(conn, config) or "").strip()
    except Exception:
        return ""
    return "" if text in ("", "None") else text


def arc_user_prompt(state: canon.StoryState, chapters: list[int], *,
                    volume_plan: str = "", prev_skeleton: str = "",
                    finale_note: str = "", volume_transition: str = "",
                    fingerprints: str = "", story_spine: str = "") -> str:
    """The volatile half of the arc prompt. Pure, so its shape is testable."""
    parts = [state.volatile_block()]
    if volume_plan:
        parts.append("## 卷纲（本弧窗口）\n" + volume_plan)
    if story_spine:
        parts.append("## 故事脊柱（原始简报硬排期，覆盖卷纲与候选创意）\n" + story_spine)
    if volume_transition:
        parts.append(volume_transition)
    if prev_skeleton:
        parts.append("## 上一弧留下的骨架（默认延续，偏离必须在 arc_intent 里说明）\n"
                     + prev_skeleton)
    if fingerprints:
        # Immediately before the request: an avoid-list is actionable only next to
        # the ask it constrains.
        parts.append("## 全书结构指纹（已用滥的推进形状，本弧请避开）\n" + fingerprints)
    parts.append(
        f"## 请求\n为以下章节各生成一张 ChapterCard，共 {len(chapters)} 张，"
        f"`ch` 必须严格等于：{chapters}\n"
        f"这一段是一个完整的弧：先想清楚 arc_intent（从什么局面推进到什么局面），"
        f"再把它拆成 {len(chapters)} 章的连续推进，每一章都要有自己的兑现，"
        f"同时服务于整段的位移。最后再写出 next_arc 骨架。" + finale_note
    )
    if story_spine:
        parts.append(
            "## 输出前机械自检（不得省略）\n"
            "1. 对每章逐项检查故事脊柱中的‘必须落地的原文锚点’，每个锚点都必须作为完整连续字符串出现在该章卡片 JSON 中。\n"
            "2. 若有任务等级，卡片必须明确写成‘S级任务/C级任务’等原值。\n"
            "3. 若有命名收件人，必须在 beats 中明确写成‘收件人是X’；其他人物只能作为真相、外观或关系线索，不得替换收件人。\n"
            "4. 终章不得用新订单、新危机或待定周期作 exit_hook。未逐项满足时不要输出。"
        )
    return "\n\n".join(p for p in parts if p)


def generate_arc(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    start_ch: int,
    end_ch: int,
    *,
    state: canon.StoryState | None = None,
    prev_skeleton: dict[str, Any] | None = None,
    call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """ONE call producing every card for `start_ch..end_ch` plus the next skeleton.

    `call` is injected in tests; production passes None and gets `llm.call_llm`.
    """
    chapters = list(range(start_ch, end_ch + 1))
    call = call or call_llm
    if state is None:
        from engine.loop import load_story_state as _canon_load
        state = _canon_load(paths, conn, config, start_ch)

    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    finale_note = ""
    if max_chapters and end_ch >= max_chapters:
        finale_note = (
            f"\n\n## 终章提醒\n第 {max_chapters} 章是全书终章：该章必须收束主线、"
            f"给出确定谜底，exit_hook 改为收束余韵，禁止引入任何新人物/新势力/新悬念。"
            f"本弧之后没有下一弧，`next_arc.chapters` 留空数组。"
        )

    span = len(chapters)
    nxt_start = end_ch + 1
    nxt_end = nxt_start + span - 1
    if max_chapters:
        nxt_end = min(nxt_end, max_chapters)
    next_chapters = list(range(nxt_start, nxt_end + 1))

    # --- arc-intent fulfilment: check if previous arc delivered its promise ---
    arc_unfulfilled_note = ""
    if start_ch > 1:
        try:
            from engine.quality_advisory import arc_intent_fulfilment
            from engine.quality import REGISTRY
            if REGISTRY.is_enabled("arc_intent_fulfilment", config):
                _span = arc_span(config)
                _prev_block = max(1, start_ch - _span)
                _arc_store = load_cards(paths)
                _prev_arc = (_arc_store.get("arcs") or {}).get(str(_prev_block))
                if isinstance(_prev_arc, dict) and _prev_arc.get("intent"):
                    _prev_start = int(_prev_arc.get("start", _prev_block))
                    _prev_end = int(_prev_arc.get("end", start_ch - 1))
                    _texts = []
                    for _n in range(_prev_start, _prev_end + 1):
                        try:
                            _texts.append(read_text(chapter_path(paths, _n)))
                        except Exception:
                            pass
                    if _texts:
                        _aif = arc_intent_fulfilment(
                            _prev_arc["intent"], _texts, config)
                        for _d in (_aif.get("directives") or []):
                            arc_unfulfilled_note += f"\n\n## 上弧未兑现警告\n{_d}"
        except Exception:
            pass

    user = arc_user_prompt(
        state, chapters,
        volume_plan=_volume_plan(paths, config, start_ch, span),
        prev_skeleton=skeleton_block(prev_skeleton, chapters),
        finale_note=finale_note + arc_unfulfilled_note,
        volume_transition=_volume_transition(paths, config, chapters),
        fingerprints=_fingerprints(conn, config),
        story_spine=story_spine_window(paths, chapters),
    )

    arc_system = ARC_SYSTEM_V2
    try:
        from engine.knowledge import select_for_planner
        from engine.config import ROOT
        genre = str(config.get("novel", {}).get("genre", ""))
        kb_block = select_for_planner(ROOT, start_ch, end_ch, genre=genre)
        if kb_block:
            arc_system = arc_system + kb_block
    except Exception:
        pass

    raw = call(
        client, paths, config, arc_system, json_prompt(user),
        max_tokens=int(config["novel"].get("arc_max_tokens", 32000) or 32000),
        temperature=float(config["novel"].get("arc_temperature", 0.75) or 0.75),
        cacheable_prefix=state.stable_prefix(),
        tag="arc_plan",
    )
    data = _parse(client, paths, config, raw)
    cards_raw = data.get("cards")
    if not isinstance(cards_raw, list) or not cards_raw:
        raise BeatError(f"arc planner returned no cards for Ch{start_ch}-{end_ch}")

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
        raise BeatError(f"arc planner produced no usable card for Ch{start_ch}-{end_ch}")

    missing = [c for c in chapters if c not in by_ch]
    if missing:
        log(paths, f"[WARN] arc {start_ch}-{end_ch}: no card for {missing}; "
                   f"each will cost its own single-chapter plan call.")

    # --- cross-card pre-validation: catch repetition problems BEFORE storing ---
    sorted_chs = sorted(by_ch)
    for i, ch in enumerate(sorted_chs):
        recent = [by_ch[c] for c in sorted_chs[:i]]
        problems = validate_card(by_ch[ch], recent_cards=recent)
        if problems:
            log(paths, f"beat: arc pre-validate Ch{ch}: {problems}")
            fixed = repair_card(client, paths, config, by_ch[ch], problems, ch, call=call)
            if fixed:
                by_ch[ch] = fixed

    # --- arc-end acceleration: last 2 cards must include at least one "high" ---
    arc_end_problem = check_arc_end_acceleration(
        [by_ch[c] for c in sorted_chs],
    )
    if arc_end_problem:
        penult_ch = sorted_chs[-2]
        log(paths, f"beat: arc-end acceleration: {arc_end_problem}")
        fixed = repair_card(
            client, paths, config, by_ch[penult_ch],
            [arc_end_problem], penult_ch, call=call,
        )
        if fixed:
            by_ch[penult_ch] = fixed

    dist_problems = check_arc_tension_distribution(
        [by_ch[c] for c in sorted_chs],
    )
    for dp in dist_problems:
        log(paths, f"beat: arc tension distribution: {dp}")

    payoff_problems = check_arc_payoff_distribution(
        [by_ch[c] for c in sorted_chs],
    )
    for pp in payoff_problems:
        log(paths, f"beat: arc payoff distribution: {pp}")

    return {
        "intent": str(data.get("arc_intent") or "").strip(),
        "cards": by_ch,
        "missing": missing,
        "next_skeleton": normalize_skeleton(data.get("next_arc"), next_chapters),
    }


def repair_card(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    card: dict[str, Any],
    problems: list[str],
    chapter_num: int,
    *,
    call: Callable[..., str] | None = None,
) -> dict[str, Any] | None:
    """One cheap call to fix a card that failed pre-write validation."""
    call = call or call_llm
    spine = story_spine_window(paths, [chapter_num])
    user = (
        f"## 待修复的卡片（第 {chapter_num} 章）\n"
        + json.dumps(card, ensure_ascii=False, indent=2)
        + "\n\n## 必须消除的问题（逐条修掉，不要回避）\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(problems))
        + ("\n\n## 原始简报当前章故事脊柱（最高优先级）\n" + spine if spine else "")
        + ("\n\n## 修复后机械验收\n"
           "- 每个‘必须落地的原文锚点’都须作为完整连续字符串出现在 JSON 中。\n"
           "- 命名收件人须在 beats 中逐字写成‘收件人是X’，不得只放进 who 或写成其外观像某人。\n"
           "- 任务等级须逐字写明；父亲等真相对象不得替换命名收件人。")
        + "\n\n请返回修复后的完整卡片 JSON（保持 ch 不变，保持 schema 字段齐全）。"
    )
    raw = call(client, paths, config, CARD_REPAIR_SYSTEM, json_prompt(user),
               max_tokens=8000, temperature=0.4, tag="arc_card_repair")
    fixed = _parse(client, paths, config, raw)
    if isinstance(fixed.get("card"), dict):
        fixed = fixed["card"]  # tolerate a {"card": {...}} wrapper
    return normalize_card(fixed, chapter_num)


# ---------------------------------------------------------------------------
# validation inputs (zero LLM)
# ---------------------------------------------------------------------------

def _continuity(paths: Paths, conn: Any, config: dict[str, Any],
                plan: dict[str, Any], chapter_num: int) -> tuple[list[str], list[str]]:
    """(critical, advisory). Same severity policy as v1's `_stage_plan`: only
    CRITICAL forces a repair. Treating advisories as repair triggers fires a call
    on nearly every chapter and eats the whole cost saving."""
    try:
        violations = validate_plan_continuity(conn, plan, chapter_num, config=config)
    except Exception as exc:
        log(paths, f"beat: continuity check failed (non-fatal) Ch{chapter_num}: {exc}")
        return [], []
    return ([v for v in violations if str(v).startswith("CRITICAL")],
            [v for v in violations if not str(v).startswith("CRITICAL")])


def _scene_sim(paths: Paths, config: dict[str, Any], store: dict[str, Any],
               plan: dict[str, Any], chapter_num: int) -> float | None:
    """Max scene-skeleton similarity against recent CARDS.

    v1 dedupes against `plan_arbitration` events, which v2 never writes. Cards
    are v2's record of what was planned, so they are the only honest comparison
    set — and reading the empty event table instead would make the check silently
    disappear rather than fail.
    """
    if not bool(config["novel"].get("scene_dedupe_enabled", True)):
        return None
    try:
        from engine.quality import scene_similarity

        lookback = int(config["novel"].get("scene_dedupe_window", 8))
        recent = [card_to_plan(c)[0] for c in _recent_cards(store, chapter_num, lookback)]
        if not recent:
            return None
        return float(scene_similarity(plan, recent).get("max_sim", 0.0) or 0.0)
    except Exception as exc:
        log(paths, f"beat: scene-dedupe check failed (non-fatal) Ch{chapter_num}: {exc}")
        return None


def _problems(paths: Paths, conn: Any, config: dict[str, Any],
              store: dict[str, Any], card: dict[str, Any],
              chapter_num: int) -> tuple[list[str], list[str]]:
    plan, _ = card_to_plan(card)
    critical, advisories = _continuity(paths, conn, config, plan, chapter_num)
    problems = validate_card(
        card,
        recent_cards=_recent_cards(store, chapter_num),
        continuity_violations=critical,
        scene_sim=_scene_sim(paths, config, store, plan, chapter_num),
        # 0.82, not 0.85: every shipped config says 0.82 and `tools/replay_gates.py`
        # replays 0.82, so a book whose config predates the key would otherwise be
        # judged by a stricter line than the tool that settles it.
        scene_sim_block=float(config["novel"].get("scene_dedupe_sim_block", 0.82)),
    )

    # --- Planning gates (CARD_GATES) — were registered but never wired. ---
    from engine.quality import (
        plan_executability_gate, plan_visual_payoff_check,
        narrative_pattern_repetition, REGISTRY,
    )

    if REGISTRY.is_enabled("plan_executability_gate", config):
        try:
            peg = plan_executability_gate(plan, config)
            if peg.get("blocked"):
                problems.append(
                    f"plan_executability: payoff过于抽象，缺少具体动作"
                    f"（{str(peg.get('evidence', ''))[:80]}）")
        except Exception:
            pass

    if REGISTRY.is_enabled("plan_visual_payoff_check", config):
        try:
            pvp = plan_visual_payoff_check(plan, config)
            if pvp.get("blocked"):
                for d in (pvp.get("directives") or [])[:2]:
                    problems.append(f"visual_payoff: {d}")
        except Exception:
            pass

    if REGISTRY.is_enabled("narrative_pattern_repetition", config):
        try:
            recent = [card_to_plan(c)[0]
                      for c in _recent_cards(store, chapter_num)]
            npr = narrative_pattern_repetition(plan, recent, config)
            if str(npr.get("level", "")) == "block":
                for d in (npr.get("directives") or [])[:2]:
                    problems.append(f"pattern_repetition: {d}")
        except Exception:
            pass

    # The generated card is not allowed to redefine the author's brief.  This
    # check is independent of volume_plan and of the card itself, so a coherent
    # but off-brief arc cannot self-certify and reach the writer.
    spine = story_spine_entry(paths, chapter_num)
    if spine:
        spine_check = story_spine_adherence(
            spine, json.dumps(card, ensure_ascii=False, sort_keys=True),
            min_anchor_coverage=0.80,
        )
        if not spine_check.get("passed", True):
            missing = spine_check.get("missing_anchors") or []
            reqs = spine_check.get("requirements") or []
            detail = "、".join(f"「{x}」" for x in missing[:6])
            if not detail:
                detail = "；".join(str(x) for x in reqs[:2])
            problems.append(
                f"story_spine: Ch{chapter_num} 偏离原始简报硬排期，必须覆盖 {detail}；"
                "只可扩写过程，不得替换事件、对象或结局功能"
            )
        for relation in (spine_check.get("relation_failures") or []):
            if relation.get("type") == "named_recipient":
                problems.append(
                    f"story_spine: Ch{chapter_num} 收件人必须是「{relation.get('expected')}」；"
                    "不得改成父亲、主角、系统意志或其他对象"
                )
            elif relation.get("type") == "grade":
                problems.append(
                    f"story_spine: Ch{chapter_num} 任务等级必须是「{relation.get('expected')}」"
                )
            elif relation.get("type") == "final_closure":
                hooks = "、".join(str(x) for x in (relation.get("open_hooks") or [])[:4])
                problems.append(
                    f"story_spine: Ch{chapter_num} 是全书终章，必须封闭收束；"
                    f"删除续集钩子{('「' + hooks + '」') if hooks else ''}，"
                    "不得新增订单、下一任务、待定周期、空缺席位或后续危机"
                )

    return problems, advisories


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _store_arc(paths: Paths, conn: Any, store: dict[str, Any], arc: dict[str, Any],
               block_id: int, gen_start: int, end: int, chapter_num: int) -> None:
    for ch, card in arc["cards"].items():
        store.setdefault("cards", {})[str(ch)] = card
    store.setdefault("arcs", {})[str(block_id)] = {
        "intent": arc["intent"], "start": gen_start, "end": end,
        "missing": arc["missing"], "next_skeleton": arc["next_skeleton"],
    }
    save_cards(paths, store)
    save_checkpoint(paths, chapter_num, "arc_generated.json",
                    {"start": gen_start, "end": end, "intent": arc["intent"],
                     "chapters": sorted(arc["cards"]),
                     "next_skeleton": arc["next_skeleton"]})
    db_event(conn, chapter_num, "arc_generated",
             {"start": gen_start, "end": end, "intent": arc["intent"],
              "count": len(arc["cards"]), "missing": arc["missing"],
              "skeleton": bool(arc["next_skeleton"])})


def ensure_card(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    *,
    state: canon.StoryState | None = None,
    call: Callable[..., str] | None = None,
) -> CardResult:
    """The card for `chapter_num`, planning an arc if there isn't one yet.

    Cost: 0 calls on 9 chapters out of 10 (the arc is already planned), 1 on the
    tenth. Worst case 3 — arc, repair, single-chapter re-plan — and the result
    says which of those happened, because a v2 arm that silently spends 3 calls a
    chapter has lost the argument it was built to make.

    Raises `BeatError` if no card can be produced at all. Stalling a chapter is
    the correct failure: writing against a card nobody planned would produce a
    CCR that measures nothing.
    """
    store = load_cards(paths)
    card = store.get("cards", {}).get(str(chapter_num))
    source = "stored"

    if not isinstance(card, dict):
        span = arc_span(config)
        max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
        block_id, end = arc_window(chapter_num, span, max_chapters)
        # Never plan chapters that are already written (a fork, or a run resumed
        # mid-block): those cards would be wasted tokens and an invitation to
        # contradict finished prose. The block id stays anchored at `block_id`.
        gen_start = max(block_id, chapter_num)
        log(paths, f"beat: planning arc Ch{gen_start}-{end} (span={span})")
        arc = generate_arc(client, paths, conn, config, gen_start, end,
                           state=state,
                           prev_skeleton=_previous_skeleton(store, block_id, span),
                           call=call)
        _store_arc(paths, conn, store, arc, block_id, gen_start, end, chapter_num)
        card = store["cards"].get(str(chapter_num))
        source = "arc"

    if not isinstance(card, dict):
        # The arc call ran and skipped exactly this chapter. One card, one call.
        card = _single(client, paths, conn, config, store, chapter_num,
                       state=state, call=call)
        source = "single"

    spine = story_spine_entry(paths, chapter_num)
    card = _ground_card_to_story_spine(card, spine)

    problems, advisories = _problems(paths, conn, config, store, card, chapter_num)
    unresolved: list[str] = []

    if problems:
        log(paths, f"beat: card Ch{chapter_num} failed pre-write validation: {problems}")
        db_event(conn, chapter_num, "card_repair", {"problems": problems, "source": source})
        try:
            fixed = repair_card(client, paths, config, card, problems, chapter_num, call=call)
        except Exception as exc:
            log(paths, f"beat: card repair call failed Ch{chapter_num} (non-fatal): {exc}")
            fixed = None
        # Re-judged by the SAME ruler that rejected it. The three-argument
        # `validate_card` that used to sit here runs a strictly SMALLER check than
        # the one that fired: continuity CRITICALs and the scene-dedupe similarity
        # are added by `_problems`, so a repair that ignored either came back with
        # `still` empty and was filed as fixed — the omitted-argument defect, where
        # "not measured" reads exactly like "clean". Zero LLM either way: both
        # halves are pure functions over the card plus one DB read.
        #
        # The advisories are re-read too. They are handed to the writer as
        # `required_constraints`, and the pre-repair card's advisories describe a
        # card the writer will never see.
        fixed = _ground_card_to_story_spine(fixed, spine)
        still, fixed_advisories = (
            _problems(paths, conn, config, store, fixed, chapter_num) if fixed
            else (["修复调用未返回可用卡片"], []))
        if fixed and not still:
            card, source = fixed, "repaired"
            problems = []
            advisories = fixed_advisories
        else:
            log(paths, f"beat: card Ch{chapter_num} still invalid after repair ({still}); "
                       f"re-planning this chapter alone.")
            card = _single(client, paths, conn, config, store, chapter_num,
                           state=state, call=call)
            card = _ground_card_to_story_spine(card, spine)
            source = "single"
            problems, advisories = _problems(paths, conn, config, store, card, chapter_num)
            if problems:
                # Third attempt. Do NOT loop: every one of these problems is
                # measured against chapters this attempt cannot rewrite (the
                # neighbour's opening_type, the book's scene history), so a
                # fourth try is a latch, not a fix. Carry them to the writer as
                # obligations and let `accept` judge the prose instead.
                if any(str(p).startswith("story_spine:") for p in problems):
                    raise BeatError(
                        f"Ch{chapter_num}: story spine still violated after card repair "
                        f"and single-chapter re-plan: {problems}"
                    )
                unresolved = list(problems)
                log(paths, f"beat: card Ch{chapter_num} accepted with unresolved "
                           f"problems (carried to the writer): {unresolved}")
                db_event(conn, chapter_num, "card_unresolved", {"problems": unresolved})

    store.setdefault("cards", {})[str(chapter_num)] = card
    save_cards(paths, store)

    plan, decision = card_to_plan(card)
    decision["planner"] = "v2_beat"
    decision["card_source"] = source
    extra = list(advisories) + [f"（规划未消除，写作时请规避）{p}" for p in unresolved]
    if extra:
        decision.setdefault("required_constraints", []).extend(extra)
    # The CCR contract: archive the card the chapter was actually written against.
    save_checkpoint(paths, chapter_num, CARD_CHECKPOINT, card)
    log(paths, f"beat: card Ch{chapter_num} [{source}] {card.get('title')} "
               f"[{card.get('opening_type')}] @ {card.get('where')}")
    return CardResult(card=card, plan=plan, decision=decision, source=source,
                      unresolved=tuple(unresolved), advisories=tuple(advisories))


def _single(client: Any, paths: Paths, conn: Any, config: dict[str, Any],
            store: dict[str, Any], chapter_num: int, *,
            state: canon.StoryState | None = None,
            call: Callable[..., str] | None = None) -> dict[str, Any]:
    """Re-plan one chapter with an arc call of length 1 — the degraded path.

    Honest but poorer: a real planned card, planned without arc-wide vision. It
    is NOT a fabricated card, and the distinction is the whole reason this
    function exists rather than a `dict(where=..., turn=...)` fallback.
    """
    arc = generate_arc(client, paths, conn, config, chapter_num, chapter_num,
                       state=state, call=call)
    card = arc["cards"].get(chapter_num)
    if not isinstance(card, dict):
        raise BeatError(f"no card for Ch{chapter_num} after a single-chapter re-plan")
    store.setdefault("cards", {})[str(chapter_num)] = card
    save_cards(paths, store)
    db_event(conn, chapter_num, "card_single_replan", {"title": card.get("title")})
    return card
