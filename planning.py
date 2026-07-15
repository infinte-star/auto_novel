from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint, save_checkpoint
from config import Paths, log, safe_score
from llm import call_llm, json_prompt, load_json_with_repair
from memory import beat_directive, cacheable_prefix, lite_memory_context, memory_context, rhythm_diagnostics, structural_repetition_analysis
from store import db_event, db_lock, get_active_constraints, get_overdue_reader_promises, get_reader_promises, get_silent_threads, recent_metrics, recent_quality_feedback

if TYPE_CHECKING:
    from openai import OpenAI

CANDIDATE_PLAN_SYSTEM = """你是工业化长篇小说引擎中的章节规划 agent。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。为所请求的章节生成一份候选大纲。
schema：
{
  "title": "...",
  "goal": "...",
  "conflict": "...",
  "conflict_type": "court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
  "payoff": "...",
  "payoff_type": "court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
  "pressure": "在兑现之前用什么来压制主角/读者",
  "beats": ["5-9 个具体节拍；每个节拍必须是一句完整的主谓宾句，含一个可见的动作，禁止用破折号堆叠状态短语"],
  "character_focus": ["在本章获得能动性或情感推进的人物"],
  "location": "本章主要发生的物理场地/空间（简短名词，如'灯塔二层走廊''渔民小屋'）",
  "info_source": "本章推进真相/剧情所依赖的主要信息来源（如'死者临终画面''匿名信''某人证词'）",
  "world_state_changes": ["若本章发生，会带来哪些状态变化"],
  "thread_actions": ["开启/推进/找回的伏线"],
  "hook": "章末抛给读者的问题",
  "risk": "主要的连续性或重复风险"
}
本章必须推进长期因果，而不只是制造局部的兴奋点。
每一份大纲都必须：
- 把至少一个遗留的审校问题转化为一个具体的、落在页面上的场景。
- 当旅行时间、消息送达、资金流动、监视等要素重要时，给出对应的因果桥梁。
- 给出可见的动作、感官锚点与对话张力，而非只有分析或概括。
- 避免复用最近章节的章末手法、分析姿态或情感节拍。

## 可落地性（最高优先级——本大纲会被一个"按可落地性而非野心打分"的仲裁层评审）
你的大纲不是用来"听起来雄心勃勃"的，而是要让一个写手能照着 beats 在 3500 字内写出读者一眼可见的画面与物证。历史数据反复证明：抽象意图型大纲被首稿写崩（仲裁 8 分→成稿 5-6 分）。请在生成时就规避：
- payoff 与高潮 beat 必须写成"某角色用具体动作操作具体物体、产生读者一眼可见的结果"的可拍句子（例：'沈澜把两张验尸单并排压在桌沿，用铅笔尖在两处伤口位置各画一道弧线对齐'）。严禁停留在只有结论没有画面的抽象意图——凡核心 payoff/高潮 beat 的动词是"推导出/意识到/想通/完成/还原/引导/心算/反应过来"而无具体动作+具体物体+可见结果，即视为写崩，必须改写成可拍动作。
- 高潮场景禁止被压缩成一句概括，或用纸条/口头转述（如"断电前用纸条告知方法"）：必须原子化为多个连续的可见动作 beat，让读者跟着角色一步步看到过程。
- payoff_type 不得伪装：没有挣来的兑现就不要标 reveal/reversal，必须在 pressure 与 beats 里补足代价与铺垫。
- 若采用反转，须按 setup→misdirect→overturn 组织：先建立并强化一个被相信的事实/信任源，再推翻它；禁止凭空冒出真相的突兀反转。
- 悬疑/推理的核心 payoff 禁止只停留在"逻辑上可推出"这类抽象判断；必须设计成读者一眼能复盘的视觉矛盾：有/无、左/右、正/反、死前/死后、照片/现场、倒影/实体、物件/身体状态冲突。

## 文体（生成端就要锁死，别留给正文重写）
- 每个 beat 都必须是完整的主谓宾长句叙事；严禁用破折号把状态短语堆叠成"句子——状态——状态"的碎片节奏。
- beat 不能只是意图清单（"她决定查清真相"），必须含可见行动、对话交锋、信息变化或资源代价之一。"""

SCREEN_SYSTEM = """你是长篇小说引擎中快速初筛的一层。
你会收到多份候选章节大纲。请按整体质量排序，考量以下方面：
- 与既有状态的因果一致性和连续性
- 相对近期章节的新颖度（避免重复）
- 读者的期待感与钩子强度
- 人物的能动性与代价的可见性
- 伏线推进与兑现的新鲜度

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{"ranking": [{"index": 0, "brief": "一句话理由"}, ...]}
从最好到最差排序。要果断——并列会浪费下游资源。"""

ARBITER_SYSTEM = """你是长篇小说引擎中的仲裁层。
请结合全局状态、近期指标、重复风险、因果价值、人物一致性、兑现新鲜度与读者期待，评估各份候选大纲。

## 评分理念（最高优先级：按"可落地性"打分，不要按"意图的雄心"打分）
你给每份候选的 score（1-10）衡量的是：**一个写手在 3500 字内、能否照这份 beats 真正写出读者一眼可见的画面与物证**，而不是这份大纲"听起来多有野心/题材多独特"。
历史数据反复证明：意图雄心的大纲被你打 8.0，成稿却跌到 5-6，因为高潮/兑现停留在"推导出真相/意识到矛盾/完成闭合"这类没有可拍动作的抽象意图。请直接对这种计划降分。
- 默认从 7.0 起步，逐项核验后只有确实可落地的候选才上浮到 8+。
- 硬上限：若某候选的核心 payoff/高潮 beat 停留在抽象意图（动词是"推导出/意识到/想通/完成/还原/引导"而无具体动作+具体物体+可见结果），或高潮被压缩成一句概括/纸条/口头转述，该候选 score 不得高于 7.0，并在 cons 中点名是哪一条 beat。
- 只有当 payoff/高潮 beat 已写成"角色用具体动作操作具体物体、产生读者可见结果"的可拍句子时，才允许给 8+。
- 后期重复坍缩（最高优先级降分项）：若某候选的核心能力使用方式或核心物证与"近期已用金手指用法/物证"参照表雷同——即让主角用同一套动作作用同一物体、或继续围绕同一件物证演示同一结论而无新信息增量——该候选 score 不得高于 7.0，并在 cons 中点名是哪个物证/哪套用法在重复。这是为防止短篇后期把上一章近乎逐字翻写。
- 选出的 merged_plan 也必须满足上述可落地标准；若候选都不达标，你必须在改写 merged_plan 时把抽象 beat 改写成可拍动作，并据此打分。

请结合全局状态、近期指标、重复风险、因果价值、人物一致性、兑现新鲜度与读者期待综合评估。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "selected_index": 0,
  "scores": [{"index": 0, "score": 1-10, "pros": [], "cons": []}],
  "merged_plan": {
    "title": "...", "goal": "...", "conflict": "...", "conflict_type": "...",
    "payoff": "...", "payoff_type": "...", "pressure": "...",
    "beats": ["..."], "character_focus": ["..."], "world_state_changes": ["..."],
    "location": "...", "info_source": "...",
    "thread_actions": ["..."], "hook": "...", "risk": "..."
  },
  "required_constraints": [
    {
      "id": "唯一标识符（如 beat_X_concrete, payoff_visible, character_Y_motive）",
      "type": "beat_fidelity|character_consistency|world_logic|payoff_delivery|hook_setup|other",
      "constraint": "具体的、可验证的约束条款（一句话，必须陈述可检查的事实）",
      "check_method": "keyword|character_name|location|object|action|dialogue|logic",
      "target": "关键词/人名/地点/物件/动作/对话片段/逻辑关系（供评审验证）"
    }
  ],
  "reader_expectation_delta": "为何这样能提升或损害读者的追读欲"
}
merged_plan 必须包含上述全部键，不得缺字段。改写 beats 时，每个 beat 仍须是完整主谓宾句子，禁止破折号状态短语堆叠。
对以下大纲予以否决或降分：把已知审校问题停留在抽象层面、依赖在页面之外解决、重复相同的物理调度、或留有未解决的时间线/物流漏洞。
若候选采用 "reversal"（反转）策略，当其反转没有铺垫（事先建立并强化过一个事实/信任源，再将其推翻）时降分——
没有铺垫的反转只是突兀的转折，而非兑现。请改进 merged_plan，让作者拿到的是具体的场景任务，而非含糊的意图。

## required_constraints 结构化要求（P0-4）
每条约束必须包含：
- id: 唯一标识（如 beat_3_location, character_zhao_motive, payoff_object_visible）
- type: 约束类型（beat_fidelity=beat 必须落地、character_consistency=人物一致性、world_logic=世界观逻辑、payoff_delivery=兑现交付、hook_setup=钩子铺垫、other）
- constraint: 具体的验收条款，必须可验证（例如："beat 3 提到的'药箱搭扣'必须在正文中出现角色触摸搭扣的动作"）
- check_method: 验证方法（keyword=关键词检查、character_name=人名出现、location=地点提及、object=物件出现、action=动作实演、dialogue=对话内容、logic=逻辑关系）
- target: 验证目标（关键词/人名/物件名/动作描述/对话片段，供评审时机械检查）

示例：
{"id": "beat_2_object", "type": "beat_fidelity", "constraint": "beat 2 中的'灯塔二层走廊'必须在正文中明确出现", "check_method": "location", "target": "灯塔二层走廊"}
{"id": "payoff_visual", "type": "payoff_delivery", "constraint": "核心 payoff 必须通过可见物证（照片 vs 现场）展示，而非口头推理", "check_method": "object", "target": "照片"}
{"id": "character_motive", "type": "character_consistency", "constraint": "祝寒做出选择前必须展示内心挣扎（不少于 50 字独白或动作犹豫）", "check_method": "keyword", "target": "犹豫|挣扎|迟疑|踟蹰"}

最后再次确认：你的 score 是"可落地性分"——payoff/高潮 beat 若仍是抽象意图（无具体动作+物体+可见结果），score 必须 ≤7.0。"""

FUSED_PLAN_REVIEW_SYSTEM = """你是一部中国历史/玄幻网文的多维度大纲审校者。
请沿 6 个相互独立的维度评估候选大纲。不要让某个强项拉高弱项——每个维度都单独评分。

## 评分理念（诚实分布——拒绝分数通胀）
每个维度的分数都必须诚实反映其在 1-10 区间内的真实质量。**默认假设各维度有缺陷**：从 6.5 起步，逐项检查后只有确实通过的维度才上浮。
如果你的维度分长期聚集在 8 附近，说明你在通胀，这会让候选区分失效。"score_caps_triggered" 字段记录哪些软性惩罚被触发。
9+ 必须只保留给"几乎无缺陷"的维度，全书各维度拿到 9+ 的比例应当很低；不要因为某个维度执行力强就让它拉高其它弱维度。

每个维度的评分标准：
- 9.5-10：典范，无风险，角度新颖，长线兑现清晰
- 8.5-9：很强，仅有轻微的表面问题
- 7-8.5：扎实可用，有一两个具体且可修复的问题（最常见区间）
- 5.5-7：存在可能损害读者体验的明显隐患
- <=5：存在需要重新设计的结构性问题

当下文某个 "score cap"（评分上限）条件被触发时，将其记入 score_caps_triggered，并从该维度分数中扣除一项软性惩罚
（通常每次触发 -1.0 到 -1.5），但不要把结果钳制到上限。

各维度及检查要点：

1. world（世界）— 地理、旅行时间（京城到江南需数日）、力量体系设定、机构/官职准确性、资源守恒、历法/季节一致性。
   软性惩罚：违反地理/旅行 -1.5；与力量体系矛盾 -2.0；机构程序时代错置 -2.5。

2. character（人物）— 每个人物都依目标行动（而非剧情便利）；有能动性且代价可见；只使用其所拥有的知识；成长是渐进的；对话口吻契合身份。
   软性惩罚：人物依据其不应拥有的信息行动 -2.0；主角没有有意义的选择/代价 -1.0；任何人物无理由地脱离人设 -1.5。

3. rhythm（节奏）— 相对近期章节在开场/收场手法上的变化、场景数量（≥2 个不同场景）、先压缩后释放、动作/对话/反思的平衡。
   软性惩罚：章末手法与上一章重复 -1.0；整章只是一个拉长的场景 -1.5；节奏单调 -1.0。

4. payoff（兑现）— 压迫→兑现的因果；payoff_type 相对最近 3 章是否新鲜；代价可见；兑现是挣来的（无巧合/天降救星）；情感质地有区分度。
   软性惩罚：兑现靠巧合/运气 -2.0；payoff_type 与前 2 章重复 -1.0；没有可见代价 -1.0。

5. foreshadowing（伏线）— 至少推进一条已开启的伏线；自然时找回被丢弃的伏线；新伏线设有现实的 due_chapter；活跃伏线总数 ≤ 8。
   软性惩罚：忽视逾期伏线（>20 章） -1.0；未推进/找回任何伏线 -1.5；在不闭合旧伏线的情况下开启第 9 条及以上并发伏线 -1.0。
   反转结构检查：若候选策略为 "reversal"，须确认大纲在推翻之前先建立了铺垫（一个被相信的事实/信任源）——反转的推翻若没有页面上或在先的铺垫则属突兀：-1.0。

6. reader（读者）— 一名连载读者读完本章后有清晰的下一章问题、至少获得一个满足时刻、即使跳读 2 章也不会迷失、不被要求同时记忆过多伏线、至少有一个共情时刻。
   软性惩罚：没有清晰的"下一章"钩子 -1.0；纯铺垫、零兑现时刻 -1.5；读者需回忆 >5 个先前情节点 -1.0。

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "axes": {
    "world":         {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "character":     {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "rhythm":        {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "payoff":        {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "foreshadowing": {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[]},
    "reader":        {"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"score_caps_triggered":[],"follow_next_reason":"..."}
  },
  "overall_score": 0,
  "merged_required_fixes": []
}

规则：
- score_caps_triggered 记录哪些软性惩罚条件被命中，用于诊断。请施加扣分，但不要钳制（clamp）。
- overall_score 由引擎根据 6 个维度分数自动计算，你可以填 0 或省略，无需自己做算术。
- 要果断——含糊的风险只会浪费下游 token。每个维度的 risks/required_fixes 都必须具体、可执行。"""

def _carried_over_risks_from_prev(paths: Paths, chapter_num: int) -> list[str]:
    """Extract continuity/rhythm/fatigue risks from the previous chapter's final review.

    Returns a deduplicated list of risk strings that the next plan should explicitly address.
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    risks: list[str] = []
    for key in ("final_review.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        for field in ("continuity_risks", "rhythm_risks", "reader_fatigue_risks", "problems"):
            for item in data.get(field, []) or []:
                text = str(item).strip()
                if text and text not in risks:
                    risks.append(text)
        if risks:
            break
    return risks[:8]


def _strategy_history(conn: Any, lookback: int = 60) -> dict[str, dict[str, float]]:
    """Aggregate per-strategy stats from past plan_arbitration events.

    Returns {strategy_name: {"trials": N, "score_sum": X, "wins": K}}.

    P1-1: "wins" now counts terminal chapter quality (final score from chapter_metrics)
    rather than arbiter selection. This upgrades the bandit's reward signal from
    "仲裁者认为哪个大纲好" to "哪个大纲真正写出了高质量成稿", closing the plan→execution gap.
    A score >= 8.0 counts as a full win; scores 5-8 get partial credit linearly scaled.
    """
    try:
        with db_lock():
            rows = conn.execute(
                "SELECT chapter, payload FROM events WHERE event_type='plan_arbitration' "
                "ORDER BY id DESC LIMIT ?",
                (lookback,),
            ).fetchall()
        events = [{"chapter": r["chapter"], "payload": json.loads(r["payload"])} for r in rows]
    except Exception:
        return {}

    # Load terminal quality scores (chapter_metrics)
    terminal_scores: dict[int, float] = {}
    try:
        with db_lock():
            metric_rows = conn.execute(
                "SELECT chapter, score FROM chapter_metrics ORDER BY chapter DESC LIMIT ?",
                (lookback,)
            ).fetchall()
        for row in metric_rows:
            ch = row["chapter"]
            score = row["score"]
            if ch and score is not None:
                terminal_scores[int(ch)] = float(score)
    except Exception:
        pass

    stats: dict[str, dict[str, float]] = {}
    for ev in events:
        chapter = ev.get("chapter")
        payload = ev.get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        # plan_arbitration payload shape: {"decision": {...}, "plans": [...]}
        decision = payload.get("decision") or {}
        plans = payload.get("plans") or []
        if not plans:
            continue
        # selected_index / scores[].index come from LLM-produced arbitration JSON that
        # is persisted verbatim; a malformed record (e.g. a critique note landing in the
        # index field) must NOT crash the bandit and take down all future planning.
        try:
            sel_idx = int(decision.get("selected_index", 0))
        except (ValueError, TypeError):
            sel_idx = 0
        scores = decision.get("scores") or []
        score_map = {}
        for s in scores:
            if not isinstance(s, dict):
                continue
            try:
                score_map[int(s.get("index", -1))] = safe_score(s.get("score", 0))
            except (ValueError, TypeError):
                continue

        # Get terminal quality for this chapter (if available)
        terminal_score = terminal_scores.get(int(chapter), None) if chapter else None

        for i, plan in enumerate(plans):
            if not isinstance(plan, dict):
                continue
            strat = str(plan.get("strategy") or "").strip()
            if not strat:
                continue
            entry = stats.setdefault(strat, {"trials": 0.0, "score_sum": 0.0, "wins": 0.0})
            entry["trials"] += 1
            entry["score_sum"] += float(score_map.get(i, 5.0))

            # P1-1: Count wins based on terminal quality (selected plans only)
            if i == sel_idx:
                if terminal_score is not None:
                    # A score >= 8.0 counts as a full win
                    win_threshold = 8.0
                    if terminal_score >= win_threshold:
                        entry["wins"] += 1.0
                    else:
                        # Partial credit: linearly scale from 0 (score=5) to 1 (score=8)
                        entry["wins"] += max(0.0, (terminal_score - 5.0) / (win_threshold - 5.0))
                else:
                    # Fallback: arbiter selection (when metrics not yet written)
                    entry["wins"] += 1.0
    return stats


def _select_strategies_bandit(
    conn: Any,
    config: dict[str, Any],
    strategies: list[tuple[str, str]],
    n: int,
    paths: Paths,
) -> list[tuple[str, str]]:
    """Strategy selection for candidate plans using Thompson sampling.

    Beta-posterior Thompson sampling over arbiter win-rates. Each strategy's
    selection count is a Bernoulli success; we sample win_rate ~ Beta(wins+1,
    losses+1) per strategy and keep the top-n samples. This naturally concentrates
    draws on winners while preserving principled exploration: an under-observed
    strategy has a wide posterior and still wins some draws. A small floor of
    forced exploration (`strategy_bandit_explore_frac`, default 0.1) guards
    against posterior lock-in over a long book.

    Cross-book prior (gated by `cross_book_prior_enabled`): global telemetry
    wins/trials are blended in as pseudo-counts, so a brand-new book starts
    from the library's accumulated win-rates. Any telemetry failure silently
    degrades to local-only.
    """
    import random as _random

    bandit_enabled = bool(config["novel"].get("strategy_bandit", True))
    if not bandit_enabled or n <= 0:
        return [strategies[i % len(strategies)] for i in range(n)]

    lookback = int(config["novel"].get("strategy_bandit_lookback", 60))
    stats = _strategy_history(conn, lookback=lookback)

    global_stats: dict[str, dict[str, float]] = {}
    if bool(config["novel"].get("cross_book_prior_enabled", False)):
        try:
            import telemetry as _telemetry
            genre = str(config["novel"].get("genre", "_default") or "_default")
            novel_name = paths.logs_dir.parent.name
            global_stats = _telemetry.global_strategy_history(genre, exclude_novel=novel_name)
        except Exception:
            global_stats = {}
    prior_weight = float(config["novel"].get("cross_book_prior_weight", 0.3))

    used_prior = False
    sampled: list[tuple[float, int, tuple[str, str]]] = []
    for idx, strat in enumerate(strategies):
        name = strat[0]
        s = stats.get(name)
        g = global_stats.get(name)
        wins = float(s["wins"]) if s else 0.0
        trials = float(s["trials"]) if s else 0.0
        # Global prior enters as capped pseudo-counts; local evidence dominates as it accumulates.
        if g and float(g.get("trials", 0)) >= 3 and prior_weight > 0:
            k = prior_weight * min(float(g["trials"]), 20.0)
            g_win_rate = float(g["wins"]) / float(g["trials"])
            wins += k * g_win_rate
            trials += k
            used_prior = True
        losses = max(0.0, trials - wins)
        sampled.append((_random.betavariate(wins + 1.0, losses + 1.0), idx, strat))
    sampled.sort(key=lambda x: (-x[0], x[1]))
    picked = [item[2] for item in sampled[:n]]
    # Exploration floor: with small probability force one slot to a strategy
    # outside the top draws, so a temporarily-unlucky arm keeps getting data.
    explore_frac = float(config["novel"].get("strategy_bandit_explore_frac", 0.1))
    if explore_frac > 0 and len(strategies) > n and _random.random() < explore_frac:
        picked_names = {p[0] for p in picked}
        leftovers = [s for s in strategies if s[0] not in picked_names]
        if leftovers:
            picked[_random.randrange(len(picked))] = _random.choice(leftovers)
    try:
        suffix = " (thompson, with cross-book prior)" if used_prior else " (thompson)"
        log(paths, f"Strategy bandit picked: {[p[0] for p in picked]}{suffix}")
    except Exception:
        pass
    return picked


def _recent_selected_plans(
    conn: Any, lookback: int = 8, exclude_chapter: int | None = None
) -> list[dict[str, Any]]:
    """Return the most recent arbiter-selected (merged) plans, newest first.

    Used for scene-skeleton dedupe: the candidate generator is told to avoid
    re-running the same conflict/payoff/beats that recent chapters already used,
    which is the engine's main defense against "infinitely slicing one scene".

    ``exclude_chapter`` MUST be passed with the chapter currently being planned.
    Otherwise this returns the chapter's OWN just-written ``plan_arbitration``
    event (arbitrate_plan persists it before the dedupe check reads it back),
    so scene_similarity compares the plan against itself and ``max_sim`` is
    pinned at 1.0 on every chapter — silently turning the dedupe BLOCK into a
    guaranteed false positive that wastes a full extra plan-generation round.
    """
    # Over-fetch so that excluding the current chapter's own (possibly multiple)
    # arbitration rows still leaves up to ``lookback`` genuine prior chapters.
    fetch = lookback + 6
    try:
        with db_lock():
            if exclude_chapter is not None:
                rows = conn.execute(
                    "SELECT chapter, payload FROM events WHERE event_type='plan_arbitration' "
                    "AND chapter != ? ORDER BY id DESC LIMIT ?",
                    (int(exclude_chapter), fetch),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT chapter, payload FROM events WHERE event_type='plan_arbitration' "
                    "ORDER BY id DESC LIMIT ?",
                    (fetch,),
                ).fetchall()
        events = [{"chapter": r["chapter"], "payload": json.loads(r["payload"])} for r in rows]
    except Exception:
        return []
    plans: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if exclude_chapter is not None and ev.get("chapter") == int(exclude_chapter):
            continue
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue
        decision = payload.get("decision") or {}
        merged = decision.get("merged_plan")
        cand = payload.get("plans") or []
        if isinstance(merged, dict) and merged:
            plans.append(merged)
        elif cand:
            try:
                sel = int(decision.get("selected_index", 0))
            except (ValueError, TypeError):
                sel = 0
            if 0 <= sel < len(cand) and isinstance(cand[sel], dict):
                plans.append(cand[sel])
        if len(plans) >= lookback:
            break
    return plans


# ---------------------------------------------------------------------------
# Used-element ledger: the single data source that prevents late-chapter
# "repetition collapse" — the dominant quality+cost sink across suspense_v5..v11.
#
# Root cause (confirmed in logs): the candidate generator, arbiter, and writer
# could not SEE which concrete devices/evidence/payoffs prior chapters already
# used (long context dilutes them, and scene_similarity only matches字面 Jaccard
# of conflict/payoff/beats). So once a 6-章 mystery exhausts its fresh scenes,
# Ch7 re-narrates Ch6 near-verbatim (cross_repeat fossils=12), novelty drops
# below floor, the score is capped, structural replan fires and burns ~60% of
# wall-time re-rolling plans that re-commit the SAME repetition.
#
# This函数 mines the recent selected plans for three classes of already-used
# concrete elements and feeds them, as an explicit avoid-list, to ALL THREE of
# generation / arbitration / writing — moving differentiation from after-the-fact
# penalty to up-front prevention. Pure regex + frequency counting; NO LLM call,
# NO cacheable_prefix impact (it lands in the variable user-message segment).
# ---------------------------------------------------------------------------

# Golden-finger / ability verbs. Kept generic so it works cross-genre (触痕/辨隙/
# 临终视像/声纹...). The neighbouring concrete object is captured to form
# "深读门把手"-style usage signatures the writer/planner must vary.
_DEVICE_VERBS = (
    "深读", "凝神", "残力", "残影", "触痕", "按指腹", "指腹", "摸", "触",
    "辨隙", "深听", "聆听", "听出", "回放", "视像", "读取", "读出", "读到",
)

# High-signal evidence nouns recur as fossils ("门把手"/"硬币"/"提手"...). Generic
# household/crime物件 so the same regex serves any closed-room mystery without a
# per-novel list. Falls back gracefully when nothing matches.
_EVIDENCE_NOUNS = (
    "门把手", "把手", "栏杆", "提手", "硬币", "箱扣", "钥匙环", "钥匙", "铝牌",
    "淤伤", "凹痕", "金属粉末", "粉末", "磁带", "录音机", "纽扣", "袖口",
    "鞋印", "压痕", "伤口", "表带", "链节", "镜子", "倒影", "照片", "锁",
    "绳", "血迹", "指甲", "螺丝", "扳手", "刀", "玻璃", "窗", "门",
)


def used_element_ledger(
    conn: Any, config: dict[str, Any], chapter_num: int, lookback: int = 6
) -> dict[str, list[str]]:
    """Mine recently-used concrete devices / evidence / payoff_types so the
    planner, arbiter and writer can be forced to vary them this chapter.

    Returns {"device_usage": [...], "evidence": [...], "payoff_types": [...]}.
    Each list is the top-N most-frequently-reused items across the last
    ``lookback`` selected plans (newest first). No LLM call; safe to disable by
    ignoring the result. Never raises — returns empty lists on any failure.
    """
    try:
        recent = _recent_selected_plans(conn, lookback=lookback, exclude_chapter=chapter_num)
    except Exception:
        return {"device_usage": [], "evidence": [], "payoff_types": []}
    device: list[str] = []
    evidence: list[str] = []
    ptypes: list[str] = []
    verb_alt = "|".join(re.escape(v) for v in _DEVICE_VERBS)
    noun_alt = "|".join(re.escape(n) for n in _EVIDENCE_NOUNS)
    # device usage = ability verb followed (within ~6 chars) by a concrete object
    usage_re = re.compile(rf"(?:{verb_alt})[^，。；、\s]{{0,6}}?({noun_alt})")
    noun_re = re.compile(noun_alt)
    for rp in recent:
        if not isinstance(rp, dict):
            continue
        pt = rp.get("payoff_type")
        if pt:
            ptypes.append(str(pt)[:30])
        blob_parts = [str(rp.get(k, "")) for k in ("payoff", "conflict", "info_source", "goal", "pressure")]
        beats = rp.get("beats")
        if isinstance(beats, list):
            blob_parts.extend(str(b) for b in beats[:8])
        blob = " ".join(blob_parts)
        for m in usage_re.finditer(blob):
            device.append(m.group(0)[:20])
        for m in noun_re.finditer(blob):
            evidence.append(m.group(0))

    def _topn(xs: list[str], n: int = 8) -> list[str]:
        return [w for w, _ in Counter(x for x in xs if x).most_common(n)]

    return {
        "device_usage": _topn(device),
        "evidence": _topn(evidence),
        "payoff_types": _topn(ptypes),
    }


def _serial_milestone_block(config: dict[str, Any], chapter_num: int) -> str:
    """Return a planning constraint block if this chapter is a serialization milestone."""
    serial_every = int(config["novel"].get("serial_milestone_every", 10))
    free_to_paid = int(config["novel"].get("serial_free_to_paid_chapter", 0))
    is_milestone = serial_every > 0 and chapter_num > 0 and chapter_num % serial_every == 0
    is_paywall = free_to_paid > 0 and chapter_num == free_to_paid
    if not is_milestone and not is_paywall:
        return ""
    label = "付费转化章" if is_paywall else f"连载里程碑（每{serial_every}章）"
    return (
        f"## 连载节奏：{label}（硬性——本章是读者留存的关键节点）\n"
        f"- 本章必须包含至少一个强力爽点/反转/揭示/情感高潮，不得纯铺垫。\n"
        f"- hook 字段必须是全书级别的强悬念，让读者非追读不可。\n"
        f"- payoff_type 不得为 strategic_setup；必须给读者一个实质性的兑现。\n"
        f"- 章末的 hook_strength 目标 ≥ 8（比普通章节更高）。\n"
        f"{'- 这是免费→付费转化章：读者在此决定是否花钱。必须让这一章成为全书到目前为止最精彩的一章。' + chr(10) if is_paywall else ''}\n"
    )


def generate_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
    cached_memory: str | None = None,
    num_candidates_override: int | None = None,
    replan_feedback: str | None = None,
) -> list[dict[str, Any]]:
    diagnostics = rhythm_diagnostics(conn, config)
    structural = structural_repetition_analysis(conn, config)
    constraints = get_active_constraints(conn, chapter_num)
    quality_feedback = recent_quality_feedback(paths)
    silence_threshold = int(config["novel"].get("thread_silence_threshold", 10))
    silent_threads = get_silent_threads(conn, chapter_num, silence_threshold=silence_threshold)
    promise_grace = int(config["novel"].get("reader_promise_overdue_grace", 15))
    overdue_promises = get_overdue_reader_promises(conn, chapter_num, grace=promise_grace)
    active_promises = get_reader_promises(conn, chapter_num, limit=12)
    carried_over_risks = _carried_over_risks_from_prev(paths, chapter_num)
    # Character relationships, info revelations, emotional cadence, reader panel alerts
    relationships_block = "None"
    stale_rels_block = "None"
    try:
        from store import get_relationships, get_stale_relationships
        rels = get_relationships(conn, limit=12)
        if rels:
            relationships_block = json.dumps(
                [{k: v for k, v in r.items() if k != "history"} for r in rels],
                ensure_ascii=False, indent=2)
        stale = get_stale_relationships(conn, chapter_num, stale_threshold=8)
        if stale:
            stale_rels_block = json.dumps(stale, ensure_ascii=False, indent=2)
    except Exception:
        pass
    info_rev_block = "None"
    overdue_rev_block = "None"
    try:
        from store import get_pending_revelations, get_overdue_revelations
        pending = get_pending_revelations(conn, chapter_num, limit=8)
        if pending:
            info_rev_block = json.dumps(pending, ensure_ascii=False, indent=2)
        overdue = get_overdue_revelations(conn, chapter_num, grace=5)
        if overdue:
            overdue_rev_block = json.dumps(overdue, ensure_ascii=False, indent=2)
    except Exception:
        pass
    emotional_cadence_block = ""
    try:
        from quality import emotional_cadence as _ec
        tone_rows = recent_metrics(conn, 6)
        tones = [str(r.get("emotional_tone", "")) for r in reversed(tone_rows) if r.get("emotional_tone")]
        ec = _ec(tones, config)
        if ec.get("directives"):
            emotional_cadence_block = "\n".join(ec["directives"])
    except Exception:
        pass
    panel_alert_block = ""
    try:
        from checkpoint import load_checkpoint as _lc
        if chapter_num > 1:
            alert = _lc(paths, chapter_num - 1, "panel_alert.json")
            if isinstance(alert, dict) and alert.get("drop_rate", 0) > 0:
                reasons = "；".join(alert.get("drop_reasons", [])[:3])
                panel_alert_block = (
                    f"## 读者面板紧急警报（上一章 Ch{chapter_num - 1}）\n"
                    f"弃读率：{alert['drop_rate']:.0%}，严重等级：{alert.get('severity', 'high')}\n"
                    f"弃书原因：{reasons}\n"
                    f"本章必须做出根本性调整以挽回读者——不要延续上章的节奏/模式/情感基调。\n"
                    f"必须包含：强爽点 或 情感冲击 或 关键揭示 或 关系突破。\n"
                )
    except Exception:
        pass
    mem = cached_memory or memory_context(paths, conn, config)
    # Whole-book beat scheduler: locate this chapter against the volume_plan
    # milestones so the planner is told which payoff/高潮 should land around now.
    # Pure parse+inject, no LLM call; degrades to "" on any parse failure.
    beat_block = ""
    try:
        from config import read_text as _read_text

        max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
        if max_chapters:
            est_total = max_chapters
        else:
            cw = int(config["novel"].get("chapter_words", 3000) or 3000)
            est_total = int(config["novel"].get("target_words", 0) or 0) // max(cw, 1)
        beat_block, _eff_gap = beat_directive(
            _read_text(paths.volume_plan),
            chapter_num,
            est_total,
            diagnostics.get("chapters_since_payoff"),
            int(diagnostics.get("payoff_max_gap", config["novel"].get("payoff_max_gap", 99))),
            config,
        )
    except Exception:
        beat_block = ""
    platform_block = ""
    benchmark_block = ""
    try:
        from benchmark import benchmark_context, platform_guidance

        platform_block = platform_guidance(config)
        benchmark_block = benchmark_context(paths, config, tail + "\n" + json.dumps(quality_feedback, ensure_ascii=False))
    except Exception:
        pass
    # Scene-skeleton dedupe: list the conflict/payoff/beats of recent selected
    # plans so the generator avoids re-running the same micro-scene.
    dedupe_block = "None"
    if bool(config["novel"].get("scene_dedupe_enabled", True)):
        try:
            window = int(config["novel"].get("scene_dedupe_window", 8))
            recent_sel = _recent_selected_plans(conn, lookback=window, exclude_chapter=chapter_num)
            skeletons = []
            for rp in recent_sel:
                beats = rp.get("beats")
                beats_list = [str(b)[:60] for b in beats[:5]] if isinstance(beats, list) else []
                # Fields here MUST match _plan_skeleton_tokens() in quality.py
                # (conflict/payoff/pressure/goal/beats); otherwise the generator
                # is told to avoid one set of dimensions while the detector judges
                # duplication on another, so a reworded conflict still trips BLOCK.
                skeletons.append({
                    "goal": str(rp.get("goal", ""))[:120],
                    "conflict": str(rp.get("conflict", ""))[:120],
                    "pressure": str(rp.get("pressure", ""))[:120],
                    "payoff": str(rp.get("payoff", ""))[:120],
                    "payoff_type": rp.get("payoff_type", ""),
                    "beats": beats_list,
                })
            if skeletons:
                dedupe_block = json.dumps(skeletons, ensure_ascii=False, indent=2)
        except Exception:
            dedupe_block = "None"
    # Used-location/setting blacklist: collect the physical settings & information
    # sources recent chapters already leaned on so the generator is forced to open
    # a NEW space or info source rather than re-staging the same room. This is the
    # positive-pressure counterpart to the after-the-fact scene_similarity WARN —
    # the WARN only fires once a near-dup is already chosen; this steers generation
    # away up front. (Diagnosis of suspense_v3: Ch2 max_sim=0.807, the whole book
    # circled the航海镜/渔民小屋穿衣镜.)
    used_locations_block = "None"
    if bool(config["novel"].get("scene_dedupe_enabled", True)):
        try:
            window = int(config["novel"].get("scene_dedupe_window", 8))
            recent_sel = _recent_selected_plans(conn, lookback=window, exclude_chapter=chapter_num)
            locs: list[str] = []
            for rp in recent_sel:
                for key in ("location", "setting", "scene", "place", "info_source"):
                    v = rp.get(key)
                    if isinstance(v, str) and v.strip() and v.strip() not in locs:
                        locs.append(v.strip()[:40])
                # Mine beats/conflict free-text for repeated concrete nouns is noisy;
                # rely on structured fields the planner emits + the skeleton beats.
                beats = rp.get("beats")
                if isinstance(beats, list) and beats:
                    head = str(beats[0])[:40]
                    if head and head not in locs:
                        locs.append(head)
            if locs:
                used_locations_block = json.dumps(locs[:12], ensure_ascii=False)
        except Exception:
            used_locations_block = "None"
    # Used-element ledger (P0 anti-collapse): explicit list of devices/evidence/
    # payoff_types prior chapters已用, so the generator must vary them up front
    # rather than letting cross_repeat penalise the fossil after the fact.
    used_element_block = "None"
    if bool(config["novel"].get("used_element_ledger_enabled", True)):
        led = used_element_ledger(
            conn, config, chapter_num,
            lookback=int(config["novel"].get("scene_dedupe_window", 8)),
        )
        if led.get("device_usage") or led.get("evidence"):
            used_element_block = json.dumps(led, ensure_ascii=False)
    # Narrative-pattern ledger: the ordered abstract move-sequence of recent
    # chapters, so the generator can SEE the procedural骨架 it must break instead
    # of unknowingly repeating "enter→collect→compare→deduce" a sixth time. This
    # is the up-front, positive-pressure counterpart to narrative_pattern_repetition
    # (which only fires after a near-dup plan is already chosen).
    narrative_pattern_block = "None"
    if bool(config["novel"].get("narrative_pattern_enabled", True)):
        try:
            from quality import _narrative_pattern_sequence

            window = int(config["novel"].get("narrative_pattern_window", 3))
            recent_np = _recent_selected_plans(conn, lookback=window, exclude_chapter=chapter_num)
            seqs = []
            for rp in recent_np:
                seq = _narrative_pattern_sequence(rp)
                if seq:
                    seqs.append("→".join(seq))
            if seqs:
                narrative_pattern_block = json.dumps(seqs, ensure_ascii=False)
        except Exception:
            narrative_pattern_block = "None"
    fingerprint_block = "None"
    if bool(config["novel"].get("fingerprint_enabled", True)):
        try:
            from quality import fingerprint_avoidance_context
            fingerprint_block = fingerprint_avoidance_context(conn, config)
        except Exception:
            fingerprint_block = "None"
    quality_threshold = float(config["novel"].get("quality_threshold", 8.0))
    dimension_floor = float(config["novel"].get("prewrite_dimension_floor", max(7.2, quality_threshold - 0.3)))
    base_user = f"""## 记忆
{mem}

## 平台/读者画像
{platform_block or "通用网文读者：开篇卖点清晰、章节推进稳定、承诺及时兑现。"}

{benchmark_block}

## 节奏诊断JSON
{json.dumps(diagnostics, ensure_ascii=False, indent=2)}

## 结构重复分析JSON
{json.dumps(structural, ensure_ascii=False, indent=2)}

## 近期质量反馈JSON（必须修复，不得重复）
{json.dumps(quality_feedback, ensure_ascii=False, indent=2) if quality_feedback else "None"}

{_serial_milestone_block(config, chapter_num)}## 首稿质量前置门槛（必须落实到本章大纲，而不是留给正文重写）
- 目标：后续首稿总分达到 {quality_threshold:.1f}+；readthrough/payoff/novelty/prose/continuity/emotional_impact 六个维度都不得低于 {dimension_floor:.1f}。把它们逐一落到字段上：readthrough→hook 要让读者非看下一章不可；payoff→必须是挣来的、可拍的兑现；novelty→location/info_source/payoff_type 相对近期要换新；prose→beats 用完整长句、控破折号；continuity→risk 字段写明本章如何不踩时间线/物流/能力越界；emotional_impact→本章必须有至少一个让读者产生真实情感反应的场景（不是形容词堆砌，而是通过具体行为和选择挣来的情感时刻）。
- goal/conflict/payoff/hook 必须分别回答：本章推进什么、谁阻碍、读者得到什么、为什么要看下一章。
- 至少一个 beat 必须正面修复"近期质量反馈JSON"里的具体问题；若反馈为空，也必须避免重复近期场景骨架。
- （system 已说明可落地性与文体硬约束；下面是本书题材的专项落地要求）
- 悬疑/推理章节的核心 payoff 禁止只停留在"光源方向不对、阴影角度矛盾、逻辑上可推出"这类抽象判断；必须设计成读者一眼能复盘的视觉矛盾：有/无、左/右、反/正、死前/死后、照片/现场、倒影/实体、物件/身体状态冲突。
- 如果使用镜子、照片、倒影、阴影，beats 必须同时给出"画面里看到什么"和"现实中对照什么"，否则视为爽点未落地。
- 若本章使用主角的核心能力（如临终视像读取），该次使用的流程/条件/代价/解读路径必须与最近各章的能力使用方式有可见区别，禁止原样复用同一套"读取→看画面→报结论"的流程；info_source 即便相同，beats 里的能力使用方式也必须翻新。
- 锁定关键嫌疑/真凶时，beats 不得让单一物证一步定罪；必须把画面物证与此前已在前文铺垫的行为模式/逻辑缺口结合成推理闭环。

## 当前生效的阶段约束（必须遵守）
{json.dumps(constraints, ensure_ascii=False, indent=2) if constraints else "None"}

## 沉默伏线（硬性要求：若叙事上可行，至少在页面上推进其中之一）
{json.dumps(silent_threads, ensure_ascii=False, indent=2) if silent_threads else "None"}

## 逾期的读者承诺（硬性：本章在页面上兑现其中之一，或在 "risk" 中说明延迟理由）
{json.dumps(overdue_promises, ensure_ascii=False, indent=2) if overdue_promises else "None"}

## 活跃读者承诺账本（追读心理核心：不要只开不还）
{json.dumps(active_promises, ensure_ascii=False, indent=2) if active_promises else "None"}

## 从 Ch{chapter_num - 1} 遗留的风险（必须在页面上处理其中至少 2 项）
{json.dumps(carried_over_risks, ensure_ascii=False, indent=2) if carried_over_risks else "None"}

## 近期已用过的场景骨架（硬性：本章的 conflict / payoff 不得与下列任何一条实质重复，必须推进到新的局面，禁止把同一个僵局/同一份公文/同一场对峙再切一刀）
{dedupe_block}

## 近期已反复使用的场地 / 信息来源（硬性：本章至少更换其一——开辟一个新的物理空间，或引入一个新的信息来源/对手，不得继续在下列地点原地打转）
{used_locations_block}

## 近期已反复使用的金手指用法 / 物证 / 兑现类型（硬性·防后期重复坍缩：device_usage=已用过的能力使用方式，evidence=已反复出现的物证，payoff_types=已用兑现类型。本章若再次使用核心能力，其"动作+作用物体+解读路径"必须与 device_usage 列表里的写法有可见区别；核心物证不得继续锁定在 evidence 列表的同一件上反复演示同一结论；payoff_type 不得与列表近项重复。除非剧情确有必要追踪同一物件，否则换新的具体物证/新的能力用法）
{used_element_block}

## 近期叙事流程骨架（硬性·防审美疲劳：下列是最近几章的抽象推进流程，如 enter_space→collect_evidence→compare_data→deduce_conclusion。本章绝对不能再走一遍同样形状的线性流程——哪怕换了取证对象，读者仍会感到"上一章看过了"。必须改变章节的叙事驱动力：用人物对峙/外部威胁/时间压力来推进，或调整信息揭示顺序——先抛结论再倒查、让对手先行动、把推理拆散到对话里，而不是静态地"进入→取证→比对→推理"）
{narrative_pattern_block}

## 全书结构指纹库（硬性·防全局重复：下列是本书所有已完成章节的叙事流程指纹。与"近期叙事流程骨架"仅覆盖最近3章不同，这里是全书累积记录。本章必须与全部历史章节在叙事驱动力和信息揭示顺序上有可见区别——特别是那些高频出现的流程组合）
{fingerprint_block}

## 角色关系状态（pair_key=角色对，stage=阶段，intensity=亲密度/紧张度0-10）
{relationships_block}

## 停滞的角色关系（这些关系长期未推进，本章应让其中至少一对有可见变化——哪怕是一个微妙的态度转变或一次有新信息量的对话）
{stale_rels_block}

## 信息/秘密/谜团揭示计划（status: planted=已埋→hinted=已给线索→revealed=已揭晓。本章可以推进其中 1-2 条的状态）
{info_rev_block}

## 逾期未揭示的信息（硬性：这些谜团/秘密已过了计划揭示时间，本章必须至少推进或揭示其中一条）
{overdue_rev_block}

{panel_alert_block}{f"## 情感节奏警告{chr(10)}{emotional_cadence_block}{chr(10)}" if emotional_cadence_block else ""}
{beat_block}

## 上章结尾
{tail[-2000:]}

## 请求
为第 {chapter_num} 章生成候选大纲。
避免近期重复。保留因果债务。提升读者追读欲。
若上方存在沉默伏线，大纲必须在 beats/thread_actions 中推进其中之一，或在 "risk" 中明确说明为何本章均不可行。
若 "节奏诊断JSON" 报告了爽点拖欠警告（chapters_since_payoff >= payoff_max_gap），本章的 payoff_type 必须是一个具体的读者兑现（court_breakthrough/policy_payoff/military_victory/reveal/reversal/personnel_payoff/institutional_fix），而非 strategic_setup 或 emotional。
若上方有停滞的角色关系，本章的 beats 中至少安排一个推动关系变化的具体场景（不是旁白交代，而是通过对话/行为/冲突让关系发生可见转变）。
若上方有逾期未揭示的信息/谜团，本章必须至少推进一条的状态（给出新线索/部分揭示/全面揭示）。
必须在大纲中包含 "relationship_beats" 字段：列出本章要推进的角色关系对和目标变化方向。
必须在大纲中包含 "info_reveals" 字段：列出本章计划揭示/推进的信息条目id。"""
    from config import is_final_chapter, ending_zone_distance
    _ending_remaining = ending_zone_distance(config, chapter_num)
    if is_final_chapter(config, chapter_num):
        base_user += """

## 终章要求（硬性：这是全书最后一章，必须规划成结局而非过渡章）
- 本章 payoff_type 必须是一个真正的读者兑现（court_breakthrough/policy_payoff/military_victory/reveal/reversal/personnel_payoff/institutional_fix），严禁 strategic_setup。
- goal/payoff 必须正面解决全书主线矛盾，把已开启的关键伏线在本章收束。
- 悬疑/推理类必须在本章给出确定的谜底（凶手/真相/核心谜题答案），不得含糊或留作开放。
- 必须收束所有尚未了结的关键悬念线（open threads 逐一收束或明确交代去向）。
- 严禁引入任何新人物、新势力、新案件、新悬念、新危机、新反转钩子；终章只收束已有元素。
- "hook" 字段不再是抛给读者的新悬念，而是一句收束/余韵/主题升华；严禁以全新未解决危机作结。
- beats 的最后 1-2 拍必须落在"结局兑现 + 情绪落点"，而非开启新冲突。"""
    elif _ending_remaining is not None:
        base_user += f"""

## 收束区要求（距全书结局仅剩 {_ending_remaining} 章，规划硬性约束）
- 停止开新坑：本章 beats/thread_actions 不得引入新的重大线索、新势力、新谜团；只允许推进与兑现已有伏线。
- 汇流主线：本章必须正面推进或兑现至少 1 条关键 open thread，向全书核心矛盾的总爆发汇聚。
- payoff_type 倾向收束型兑现（reveal/reversal/personnel_payoff/institutional_fix），避免 strategic_setup。
- relationship_beats / info_reveals 优先选择"接近收尾"的关系对与谜团，而非新铺设。
- hook 转为"既有矛盾收紧/摊牌临近"的收口张力，而非抛出新危机。"""
    else:
        from config import narrative_mode
        mode = narrative_mode(config)
        if mode == "reasoning":
            base_user += """

## 叙事模式：单密室·精密推理（规划硬性约束）
- 场景收敛：本章核心场景应收束在一个封闭/半封闭空间，减少场景跳转，让推理在受限空间逐步逼出。
- payoff 必须绑定 2 个以上可触摸/可观察的具体物件或身体状态（压痕、链节、血迹方向、齿痕、反光），核心爽点尽量是读者一眼能懂的视觉矛盾，而非抽象"角度/逻辑不对"。
- 公平线索：关键揭示必须可在前文找到伏笔；本章至少推进或收束1条已有疑点，少开新悬念。"""
        elif mode == "serial":
            base_user += """

## 叙事模式：强钩子·情绪外放·可连载（规划硬性约束）
- 强钩子前置：beats 的第1拍即抛出强冲突/强悬念/强反差，开篇禁止铺垫与设定倾倒。
- 情绪兑现：本章须有明确的情绪兑现或小高潮（揭晓/打脸/反转/关系推进），payoff_type 优先 reveal/reversal/emotional，避免纯 strategic_setup。
- 章末强钩：hook 字段必须是一个让读者必须追读的强悬念/反转/危机/承诺。"""
    # Cold-start block: in the first few chapters the strategy bandit, scene
    # dedupe, used-locations blacklist and quality-feedback loops are all empty
    # (no history yet), so the generator gets almost no steering and tends to
    # spend the opening on world-info dump + slow setup. These early chapters
    # decide whether a reader keeps going, so inject opening-specific craft
    # constraints when we're early AND there's genuinely little to dedupe against.
    cold_start_n = int(config["novel"].get("cold_start_chapters", 3))
    history_thin = (dedupe_block == "None" and used_locations_block == "None")
    if chapter_num <= cold_start_n and history_thin and not is_final_chapter(config, chapter_num):
        base_user += f"""

## 开篇章节要求（硬性：这是第 {chapter_num} 章，处于决定读者去留的开篇区，历史去重数据尚空，必须靠本章自身立住）
- 卖点前置：本章必须在前 1/3 就让读者看到本书的核心钩子/金手指/独特设定的一次具体运作，而不是铺垫几千字背景后才出现；第一个 beat 就要有动作或冲突，禁止以大段世界观/履历介绍开场。
- 信息克制：禁止信息倾倒（一次性抛设定、地名、人物关系表）；世界观只在角色行动中按需带出。risk 字段须自查是否存在开篇信息过载。
- 钩子强度：hook 必须是一个读者会主动追问的具体悬念或诱惑（谁、为什么、接下来怎么办），而非"故事就此展开"这类空泛收束。
- 主角立人设：第一章须让主角通过一次可见的选择/反应展示其性格与处境，让读者迅速产生代入或好奇。
- 即便没有历史去重数据，仍要为后续留出空间：location 与 info_source 选择具体、有延展性的，不要把最大的爽点在开篇一次性烧光。"""
    # Platform-tuned opening rules: fire for ALL opening chapters (not gated on
    # history_thin), since the golden-3-chapters bar differs sharply by platform
    # and these chapters决定 sign-off/retention. Additive to the craft block above.
    opening_chapters = int(config["novel"].get("opening_chapters", 3))
    if chapter_num <= opening_chapters and not is_final_chapter(config, chapter_num):
        try:
            from benchmark import platform_opening
            po = platform_opening(config)
            if po:
                base_user += "\n\n" + po
        except Exception:
            pass
    # When this is a quality-replan (the previous version of THIS chapter scored
    # below threshold), inject exactly WHY it failed — the reviewer's concrete
    # problems + the deterministic style metrics — so the regenerated plan
    # attacks the real defect instead of being a blind retry. Without this the
    # replan reuses the same lite memory and produces a near-identical candidate
    # set (suspense_v3: replans震荡在 5.5~7.8, 净收益≈0).
    if replan_feedback:
        base_user += f"""

## 上一版本章未达标的具体原因（最高优先级：本次大纲必须正面消除下列每一条，不得回避）
{replan_feedback}
- 不要只换措辞或换场景名；要针对上面的每条缺陷，在 goal/conflict/payoff/beats 里给出可见的修复动作。
- 若上一版因文体碎片化/破折号堆砌失分，本章 beats 必须明确要求用完整主谓宾长句叙事、控制破折号。
- 若上一版因能力越界/视角越界失分，本章必须在 risk 字段写明如何严守能力模态与限制视角。"""
    # Cross-book craft hints (distillation loop): inject library-wide structural
    # lessons so a new book inherits accumulated planning experience from Ch1.
    # Silent no-op when craft_rules.json is absent or yields no qualifying rules.
    if bool(config["novel"].get("craft_rules_enabled", True)):
        try:
            from craft import craft_planner_hints

            ch = craft_planner_hints(config)
            if ch:
                base_user += "\n\n" + ch
        except Exception:
            pass
    # Content register (platform moderation compliance): steer the plan away from
    # mandating graphic gore/death/body-horror scenes so the WRITTEN chapter can pass
    # a content-moderation gateway. Same gate as the writer-side block.
    if bool(config["novel"].get("sensitive_word_avoidance", False)):
        try:
            from writing import SENSITIVE_WORD_AVOIDANCE_BLOCK
            base_user += (
                "\n\n## 内容分级约束（规划层·硬性）\n"
                "本作发布渠道带内容审核。规划 goal/conflict/payoff/beats/scenes 时，"
                "不要设计依赖血腥身体损伤、遗骸解剖、进食血肉脏器等露骨生理细节才能成立的场景；"
                "把黑暗、恐怖、伤亡、"
                "「吞噬变强」都设计成可用氛围、心理、后果、能量/本源汲取来实演的形态——"
                "冲突与爽点强度不减，但呈现方式必须可被克制、含蓄地写出来。\n"
                + SENSITIVE_WORD_AVOIDANCE_BLOCK
            )
        except Exception:
            pass
    # Platform golden-finger constraints (免费流偏好简单/有代价的金手指；Gap-5).
    # Silent no-op (returns "") for non-free presets, so long-line体系文 unaffected.
    try:
        from benchmark import platform_golden_finger
        gf = platform_golden_finger(config)
        if gf:
            base_user += "\n\n" + gf
    except Exception:
        pass
    num_candidates = int(num_candidates_override) if num_candidates_override else int(config["novel"]["candidate_plans"])
    max_workers = int(config["novel"].get("max_parallel_workers", 5))
    # Temperature spread across candidates. A wider spread is the cheapest lever
    # against candidate convergence: the low-temp candidates anchor a safe plan
    # while high-temp ones explore. Default base/step give 0.6,0.72,0.84,... so
    # even candidate 0 and 1 are meaningfully apart (the old 0.65+0.05*idx put
    # the first two at 0.65/0.70 — near-identical drafts). Clamped to <=1.1.
    temp_base = float(config["novel"].get("plan_candidate_temp_base", 0.6))
    temp_step = float(config["novel"].get("plan_candidate_temp_step", 0.12))
    # Explicit differentiation strategies — each candidate is told to attack the
    # chapter from a distinct angle so the arbiter sees a real choice, not 5
    # near-identical variants.
    candidate_strategies = [
        ("scene-driven",
         "以单个高密度场景为核心：物理空间高度具象，让冲突在一个房间/一段路途/一桌对峙中爆发；最少3次场地切换。"),
        ("character-driven",
         "以主角或核心配角的内心两难为核心：本章的胜负来自角色的关键选择与可见代价；选择必须在 beats 里明示。"),
        ("thread-driven",
         "以推进 2 条以上 open thread 为核心：必须在 thread_actions 显式列出推进的具体 thread id 与下一步具体动作。"),
        ("reversal",
         "以认知反转为核心，必须按 setup→misdirect→overturn 三段结构组织，并在 beats 中显式标注每一段："
         "①setup：先建立一个被广泛相信的'事实/信任源/判断'；②misdirect：通过证据或他人之口强化它，让读者也信以为真；"
         "③overturn：推翻它，证伪原信任源，给主角带来必须修正策略的新筹码或新危机。hook 必须基于 overturn。"
         "禁止无 setup 的突兀反转（凭空冒出真相）。"),
        ("pressure-payoff",
         "以挤压-释放节奏为核心：前 60% 持续施压（资源/时间/信任三轴中至少 2 轴），后 40% 给一个小而可信的释放点。"),
    ]

    # Strategy bandit: pick which strategies to use this chapter based on
    # historical plan_score in the agent_reports/plan_arbitration events table.
    # Falls back to round-robin when there's not enough data.
    chosen_strategies = _select_strategies_bandit(
        conn, config, candidate_strategies, num_candidates, paths,
    )

    def gen_one(idx: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        strategy_name, strategy_desc = chosen_strategies[idx % len(chosen_strategies)]
        strategy_block = (
            f"\n\n## 候选策略（强制）\n"
            f"候选编号：{idx}\n"
            f"策略：{strategy_name}\n"
            f"定义：{strategy_desc}\n"
            f"你必须围绕这一策略来设计本候选大纲，让本策略的特征在 goal/conflict/beats 里清晰可辨。\n"
            f"## 反趋同（硬性）\n"
            f"其它候选会采用不同策略来攻同一章——你必须真正岔开，不要产出只是换措辞的安全答案：\n"
            f"- 选一个与上方'近期已反复使用的场地/信息来源'不同、且最贴合本策略的 location 与 info_source；\n"
            f"- 你的 payoff_type 要服务于本策略的兑现方式，不要默认套用最稳妥的那一种；\n"
            f"- 宁可让本候选带有鲜明的策略棱角（可被仲裁层挑出独特的 pros），也不要为了'四平八稳'把棱角磨平。"
        )
        for retry in range(2):
            try:
                raw = call_llm(
                    client,
                    paths,
                    config,
                    CANDIDATE_PLAN_SYSTEM,
                    json_prompt(base_user + strategy_block),
                    max_tokens=16000,
                    temperature=min(1.1, temp_base + idx * temp_step),
                    cacheable_prefix=cacheable_prefix(paths, config),
                    tag="plan_candidate",
                )
                plan = load_json_with_repair(client, paths, config, raw)
                plan["candidate_index"] = idx
                plan["strategy"] = strategy_name
                return plan
            except Exception as exc:
                last_exc = exc
                log(paths, f"Candidate plan {idx} ({strategy_name}) attempt failed retry={retry}: {exc}")
        log(paths, f"Candidate plan {idx} ({strategy_name}) discarded after retries: {last_exc}")
        return {}

    plans: list[dict[str, Any]] = [{}] * num_candidates
    with ThreadPoolExecutor(max_workers=min(max_workers, num_candidates)) as executor:
        futures = {executor.submit(gen_one, idx): idx for idx in range(num_candidates)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                plans[idx] = future.result()
            except Exception as exc:
                log(paths, f"Candidate plan {idx} thread failed: {exc}")
                plans[idx] = {}
    valid = [p for p in plans if p]
    if not valid:
        raise RuntimeError(f"All {num_candidates} candidate plans failed for chapter")
    return valid

def screen_candidates(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    top_n: int = 2,
    cached_memory: str | None = None,
) -> list[int]:
    if len(plans) <= top_n:
        return list(range(len(plans)))
    mem = lite_memory_context(paths, conn, config)
    user = f"""## 记忆（节选）
{mem}

## 候选大纲JSON
{json.dumps(plans, ensure_ascii=False, indent=2)}

为第 {chapter_num} 章的全部 {len(plans)} 份候选排序。"""
    raw = call_llm(
        client, paths, config, SCREEN_SYSTEM, json_prompt(user),
        max_tokens=12000, temperature=0.2, cacheable_prefix=cacheable_prefix(paths, config),
        tag="plan_screen",
    )
    result = load_json_with_repair(
        client, paths, config, raw, fallback={"ranking": [{"index": i} for i in range(len(plans))]}
    )
    ranking = result.get("ranking", [])
    indices = []
    for entry in ranking:
        idx = int(entry.get("index", 0))
        if 0 <= idx < len(plans) and idx not in indices:
            indices.append(idx)
    if len(indices) < top_n:
        for i in range(len(plans)):
            if i not in indices:
                indices.append(i)
            if len(indices) >= top_n:
                break
    return indices[:top_n]


def _explode_fused_axes(fused: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a fused-review JSON object into the legacy per-agent reports list.

    Downstream code (arbitrate_plan, agent_reports table, plan_score) expects a
    list of {"agent","score","risks","required_fixes","state_patch"} dicts. This
    expands the fused {"axes":{...}} payload into that shape.
    """
    axes = fused.get("axes") or {}
    reports: list[dict[str, Any]] = []
    axis_scores: list[float] = []
    for axis_name in ("world", "character", "rhythm", "payoff", "foreshadowing", "reader"):
        axis = axes.get(axis_name) or {}
        sc = safe_score(axis.get("score", 5))
        axis_scores.append(sc)
        report = {
            "agent": axis_name,
            "score": sc,
            "risks": axis.get("risks") or [],
            "required_fixes": axis.get("required_fixes") or [],
            "state_patch": axis.get("state_patch") or [],
            "score_caps_triggered": axis.get("score_caps_triggered") or [],
        }
        if axis_name == "reader" and axis.get("follow_next_reason"):
            report["follow_next_reason"] = axis["follow_next_reason"]
        reports.append(report)
    # Compute overall_score in Python (mean of axes) instead of asking the model
    # to do arithmetic it is unreliable at. Written back onto the fused dict so
    # checkpoints stay meaningful.
    if axis_scores:
        fused["overall_score"] = round(sum(axis_scores) / len(axis_scores) * 2) / 2
    return reports


def _fused_review_one_plan(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    user: str,
    plan_index_for_log: int | None = None,
) -> list[dict[str, Any]]:
    """Run one fused plan-review call and return 6 legacy-format reports."""
    fallback_axes = {
        name: {"score": 5, "risks": [], "required_fixes": [], "state_patch": [], "score_caps_triggered": []}
        for name in ("world", "character", "rhythm", "payoff", "foreshadowing", "reader")
    }
    fallback = {"axes": fallback_axes, "overall_score": 0, "merged_required_fixes": []}
    for retry in range(2):
        try:
            raw = call_llm(
                client,
                paths,
                config,
                FUSED_PLAN_REVIEW_SYSTEM,
                json_prompt(user),
                max_tokens=12000,
                temperature=0.2,
                cacheable_prefix=cacheable_prefix(paths, config),
                tag="plan_review_fused",
            )
            fused = load_json_with_repair(client, paths, config, raw, fallback=fallback)
            if not isinstance(fused.get("axes"), dict):
                raise ValueError("fused review missing 'axes' dict")
            return _explode_fused_axes(fused)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            tag = f"plan={plan_index_for_log}" if plan_index_for_log is not None else "single"
            log(paths, f"Fused plan review parse failed {tag} retry={retry}: {exc}")
    return _explode_fused_axes(fallback)


def agent_review_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    user = f"""## 记忆
{lite_memory_context(paths, conn, config)}

## 节奏诊断JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## 候选大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

审校第 {chapter_num} 章大纲。"""

    fused_enabled = bool(config["novel"].get("fused_plan_review", True))
    if fused_enabled:
        reports = _fused_review_one_plan(client, paths, config, user)
    else:
        reports = _fused_review_one_plan(client, paths, config, user)
        log(paths, "Note: fused_plan_review=false ignored; always using fused review")

    for report in reports:
        agent = report["agent"]
        with db_lock():
            conn.execute(
                "INSERT INTO agent_reports(chapter, agent, score, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    chapter_num,
                    agent,
                    safe_score(report.get("score", 0)),
                    json.dumps(report, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    with db_lock():
        conn.commit()
    return reports

def review_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    cached_memory: str | None = None,
) -> list[list[dict[str, Any]]]:
    plan_users = []
    diagnostics_json = json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)
    memory = lite_memory_context(paths, conn, config)
    for plan in plans:
        plan_users.append(
            f"""## 记忆
{memory}

## 节奏诊断JSON
{diagnostics_json}

## 候选大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

审校第 {chapter_num} 章大纲。"""
        )

    max_workers = int(config["novel"].get("max_parallel_workers", 5))
    reports_by_plan: list[list[dict[str, Any]]] = [[] for _ in plans]
    # Always use fused review (one LLM call per plan, expands to 6 axis reports).
    # The legacy unfused 6-parallel-calls path has been removed.

    def fused_one(plan_index: int) -> tuple[int, list[dict[str, Any]]]:
        return plan_index, _fused_review_one_plan(
            client, paths, config, plan_users[plan_index], plan_index_for_log=plan_index
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(plans))) as executor:
        futures = {executor.submit(fused_one, i): i for i in range(len(plans))}
        for future in as_completed(futures):
            plan_index = futures[future]
            try:
                _, reports = future.result()
                reports_by_plan[plan_index] = reports
            except Exception as exc:
                log(paths, f"Fused review thread failed plan={plan_index}: {exc}")
                reports_by_plan[plan_index] = _explode_fused_axes(
                    {"axes": {
                        name: {"score": 5, "risks": [], "required_fixes": [], "state_patch": [], "score_caps_triggered": []}
                        for name in ("world", "character", "rhythm", "payoff", "foreshadowing", "reader")
                    }}
                )

    for reports in reports_by_plan:
        for report in reports:
            agent = report["agent"]
            with db_lock():
                conn.execute(
                    "INSERT INTO agent_reports(chapter, agent, score, report_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        chapter_num,
                        agent,
                        safe_score(report.get("score", 0)),
                        json.dumps(report, ensure_ascii=False),
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
    with db_lock():
        conn.commit()

    return reports_by_plan

def plan_calibration_hint(conn: Any, config: dict[str, Any], lookback: int = 12) -> str:
    """Build a calibration directive from the historical plan→realized gap.

    The arbiter chronically rates merged plans ~8.5 (a ceiling), while the
    chapters they produce realize ~7.5 — an open loop: the arbiter never sees
    that its 8.5s become 7.5s, so it keeps minting 8.5s. This reads the last
    `lookback` chapters' (plan_score, realized score) pairs from chapter_metrics
    and, when a systematic positive gap exists, tells the arbiter to deflate its
    score by the observed mean gap and to score the *executability* of beats
    rather than the ambition of intent. Returns "" when there isn't enough data.
    """
    rows = recent_metrics(conn, lookback)
    pairs: list[tuple[float, float]] = []
    for r in rows:
        ps = r.get("plan_score")
        sc = r.get("score")
        if ps is None or sc is None:
            continue
        ps_f = safe_score(ps)
        sc_f = safe_score(sc)
        # Only count chapters where both scores are real (>0); a plan_score of 0
        # means the field was never populated (legacy rows / JSON fallback).
        if ps_f <= 0 or sc_f <= 0:
            continue
        pairs.append((ps_f, sc_f))
    if len(pairs) < 3:
        return ""
    gaps = [ps - sc for ps, sc in pairs]
    mean_gap = sum(gaps) / len(gaps)
    mean_plan = sum(ps for ps, _ in pairs) / len(pairs)
    mean_real = sum(sc for _, sc in pairs) / len(pairs)
    # Only fire when the arbiter is systematically over-scoring (gap is the
    # bias we want to correct; a near-zero or negative gap means it's calibrated
    # and we should not nag it).
    min_gap = float(config["novel"].get("plan_calibration_min_gap", 0.6))
    if mean_gap < min_gap:
        return ""
    return (
        f"## 仲裁分校准（必须读，基于最近 {len(pairs)} 章的实测回归）\n"
        f"历史上你给大纲的平均仲裁分 {mean_plan:.1f}，但这些大纲成稿后的实测综合分平均只有 "
        f"{mean_real:.1f}（系统性高估 {mean_gap:.1f} 分）。这说明你在为「意图的雄心」打分，"
        f"而成稿是按「节拍的可执行性」被评判的。本次评分务必：\n"
        f"1) 先按你的直觉给分，再减去 {mean_gap:.1f} 分作为校准基线，"
        f"只有当某候选的 beats 是具体到可直接落地成场景的动作（谁、在哪、做了什么、读者看到什么反转/兑现）时，才把分数加回去；\n"
        f"2) 对停留在抽象意图、把兑现推到页面之外、或 beats 只是状态描述而非可见动作的候选，"
        f"分数必须显著低于 {mean_plan:.1f}；\n"
        f"3) 不要因为「看起来雄心勃勃」而给 8.5+ —— 8.5+ 只应留给那些你确信成稿能实测到 8.5+ 的、"
        f"每个 beat 都自带画面与冲突落点的大纲。"
    )


STRUCTURAL_DIAGNOSE_SYSTEM = """你是长篇小说引擎中的"重写前诊断官"。
一章刚刚被判定为结构性失败（场景设计层面的问题，光改措辞无效，必须重做场景）。
在引擎重新生成大纲之前，你要先精确定位：到底是哪一个 beat / 哪一个维度塌了，
这样重写才有靶子，而不是又生成一份"看起来不一样、实则同样落不了地"的大纲。

你会收到：本章原计划（含 beats）、失败评审（含各维度分、问题、beats_audit）、本章正文节选。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "root_cause": "一句话点出结构性失败的根因（哪个维度/哪个 beat 没落地，为什么）",
  "failed_beats": ["原计划里没有真正在正文兑现的 beat 原文（逐条）"],
  "weakest_dimension": "novelty|payoff|readthrough|hook|continuity|prose",
  "scene_fix": "重写时场景设计必须做出的具体改变（换信息来源/换冲突落点/把抽象兑现落到一个可见动作上等），要具体到可直接执行",
  "must_dramatize": ["重写稿必须在正文里真正演出来（而非梗概/暗示）的 2-4 个具体画面/动作"]
}
只诊断、不重写。判定从严：beats_audit 里 status 为 absent/partial 的，几乎都属于 failed_beats。"""


def diagnose_structural_failure(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    review: dict[str, Any],
    chapter_text: str,
) -> dict[str, Any]:
    """Locate the failed beat/dimension before a structural replan.

    A structural replan regenerates the whole plan from scratch; historically it
    often came back "did not improve" because the new plan repeated the SAME
    open-loop mistake (abstract intent, off-page payoff) with different surface
    wording. This is a cheap, targeted diagnose pass that pins down *what* failed
    — which beats never landed, which dimension is weakest, what the scene must
    change — so the regenerated plan's required_constraints carry a concrete
    target rather than a vague "do better". Returns {} on failure (caller treats
    a missing diagnosis as "no extra constraints").
    """
    beats = plan.get("beats") or []
    audit = review.get("beats_audit") or []
    user = f"""## 本章原计划（节选）
title: {plan.get('title','')}
goal: {plan.get('goal','')}
payoff: {plan.get('payoff','')}
beats:
{json.dumps(beats, ensure_ascii=False, indent=2)}

## 失败评审摘要
score: {review.get('score')}
novelty/payoff/readthrough/hook/continuity/prose: {review.get('novelty_score')}/{review.get('payoff_score')}/{review.get('readthrough_score')}/{review.get('hook_score', review.get('hook_strength'))}/{review.get('continuity_score')}/{review.get('prose_score', review.get('aesthetic_score'))}
problems: {json.dumps((review.get('problems') or [])[:6], ensure_ascii=False)}
beats_audit: {json.dumps(audit[:12], ensure_ascii=False)}

## 本章正文节选
{(chapter_text or '')[:6000]}

诊断本章结构性失败的根因，定位没落地的 beat 与最弱维度。"""
    try:
        raw = call_llm(
            client, paths, config, STRUCTURAL_DIAGNOSE_SYSTEM, json_prompt(user),
            max_tokens=2000, temperature=0.2, tag="structural_diagnose",
        )
        data = load_json_with_repair(client, paths, config, raw, fallback={})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def arbitrate_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plans: list[dict[str, Any]],
    reports_by_plan: list[list[dict[str, Any]]],
    cached_memory: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mem = cached_memory or memory_context(paths, conn, config)
    calib = plan_calibration_hint(conn, config)
    calib_block = f"\n{calib}\n" if calib else ""
    # Used-element ledger so the arbiter can PENALISE a candidate whose ability
    # usage / core evidence just re-runs the last few chapters' (P0 anti-collapse;
    # the writer otherwise produces a near-verbatim repeat that only cross_repeat
    # catches after the fact).
    ledger_block = ""
    if bool(config["novel"].get("used_element_ledger_enabled", True)):
        led = used_element_ledger(
            conn, config, chapter_num,
            lookback=int(config["novel"].get("scene_dedupe_window", 8)),
        )
        if led.get("device_usage") or led.get("evidence"):
            ledger_block = (
                "\n## 近期已用金手指用法/物证/兑现类型（评分参照）\n"
                + json.dumps(led, ensure_ascii=False)
                + "\n规则：若某候选的核心能力使用方式或核心物证与上表雷同（同一动作作用同一物体、"
                "或继续围绕同一物证演示同一结论），判定为后期重复风险，该候选 score 不得高于 7.0，"
                "并在 cons 中点名。仲裁改写 merged_plan 时也必须避开上表已用项。\n"
            )
    user = f"""## 记忆
{mem}
{calib_block}{ledger_block}

## 节奏诊断JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## 近期质量反馈JSON（必须修复，不得重复）
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## 候选大纲JSON
{json.dumps(plans, ensure_ascii=False, indent=2)}

## Agent 报告JSON
{json.dumps(reports_by_plan, ensure_ascii=False, indent=2)}

为第 {chapter_num} 章选出并改进最佳大纲。"""
    raw = call_llm(
        client, paths, config, ARBITER_SYSTEM, json_prompt(user),
        max_tokens=12000, temperature=0.25, cacheable_prefix=cacheable_prefix(paths, config),
        tag="plan_arbitrate",
    )
    decision = load_json_with_repair(client, paths, config, raw)
    plan = decision.get("merged_plan") or plans[int(decision.get("selected_index", 0))]
    db_event(conn, chapter_num, "plan_arbitration", {"decision": decision, "plans": plans})
    # Cross-book telemetry double-write (observer; silently degrades). Done at
    # the source because this is the only point where the full candidate list
    # plus the arbiter decision coexist.
    if bool(config["novel"].get("telemetry_enabled", True)):
        try:
            import telemetry as _telemetry
            _telemetry.record_arbitration(
                paths.logs_dir.parent.name,
                str(config["novel"].get("genre", "_default") or "_default"),
                chapter_num, decision, plans,
            )
        except Exception:
            pass
    return plan, decision

def plan_score(decision: dict[str, Any], selected_index: int | None = None) -> float:
    scores = decision.get("scores") or []
    if not scores:
        return 0.0
    if selected_index is None:
        selected_index = int(decision.get("selected_index", 0))
    for score in scores:
        if int(score.get("index", -1)) == selected_index:
            return safe_score(score.get("score", 0))
    return safe_score(scores[0].get("score", 0))

def _recovery_active(paths: Paths, chapter_num: int) -> bool:
    """True when a mid-book degradation recovery directive (written by
    pipeline._detect_quality_degradation) is still in force for this chapter."""
    try:
        from config import read_text as _read_text
        cache = paths.logs_dir / "recovery_directive.json"
        if not cache.exists():
            return False
        data = json.loads(_read_text(cache))
        return chapter_num <= int(data.get("active_until", 0))
    except Exception:
        return False


def _effective_candidate_count(conn: Any, config: dict[str, Any], chapter_num: int, paths: Paths) -> int:
    """Risk-adaptive candidate-plan count.

    Two forces, applied in order:
      1. RISK UPSHIFT — recent chapters show trouble (style/repeat penalties,
         gate rejects, falling scores, force-accepts): restore full breadth even
         if a downshift would otherwise apply. Recovering from a collapse is
         exactly when plan diversity pays for itself.
      2. STABLE DOWNSHIFT — quality stably high after a warm-up: drop one
         candidate to save the (plan_candidate + fused_review + arbitrate)
         overhead, which measures ~55% of total LLM seconds per chapter.

    Never returns below 1; chapter 1-2 always keep full breadth.
    """
    base = int(config["novel"]["candidate_plans"])
    if not bool(config["novel"].get("adaptive_downshift_enabled", True)):
        return base
    if base <= 1 or chapter_num <= 2:
        return base

    # Recovery mode (mid-book degradation alert): force full candidate breadth
    # for the recovery window — plan diversity is exactly what breaks a slide.
    if _recovery_active(paths, chapter_num):
        full = int(config["novel"]["candidate_plans"])
        log(paths, f"Recovery upshift Ch{chapter_num}: degradation recovery mode "
            f"active — keeping full candidate breadth ({full}).")
        return full

    window = int(config["novel"].get("adaptive_downshift_window", 10))
    rows = recent_metrics(conn, window)

    # --- 1. Risk upshift (acts from Ch3, no warmup: collapse won't wait) ---
    risk_window = int(config["novel"].get("risk_upshift_window", 3))
    recent = rows[:risk_window]  # newest-first
    risky = False
    reasons: list[str] = []
    if recent:
        scores_recent = [safe_score(r.get("score", 0)) for r in recent if r.get("score") is not None]
        risk_floor = float(config["novel"].get("risk_upshift_score_floor", 7.0))
        if scores_recent and min(scores_recent) < risk_floor:
            risky = True
            reasons.append(f"min_recent_score={min(scores_recent):.1f}<{risk_floor}")
        pen_cut = float(config["novel"].get("risk_upshift_style_penalty", 1.0))
        pens = [float(r.get("style_penalty") or 0.0) for r in recent]
        if pens and max(pens) >= pen_cut:
            risky = True
            reasons.append(f"max_style_penalty={max(pens):.1f}>={pen_cut}")
    # Reader-panel retention proxy (Gap-7): a high recent simulated drop_rate is a
    # collapse signal — restore full candidate breadth, plan diversity recovers it.
    if bool(config["novel"].get("reader_panel_enabled", False)):
        try:
            from store import recent_panel_drop_rate
            drop = recent_panel_drop_rate(conn, int(config["novel"].get("reader_panel_replan_window", 3)))
            if drop is not None and drop >= float(config["novel"].get("reader_panel_upshift_drop", 0.5)):
                risky = True
                reasons.append(f"panel_drop_rate={drop:.2f}")
        except Exception:
            pass
    if risky:
        if base < int(config["novel"]["candidate_plans"]):
            base = int(config["novel"]["candidate_plans"])
        log(
            paths,
            f"Risk upshift Ch{chapter_num}: keeping full candidate breadth ({base}) — {', '.join(reasons)}",
        )
        return base

    # --- 2. Stable downshift (original behaviour, warmup-gated) ---
    warmup = int(config["novel"].get("adaptive_downshift_warmup", 60))
    if chapter_num < warmup:
        return base
    if len(rows) < window:
        return base
    score_floor = float(config["novel"].get("adaptive_downshift_score", 8.5))
    scores = [safe_score(r.get("score", 0)) for r in rows if r.get("score") is not None]
    plan_scores = [safe_score(r.get("plan_score", 0)) for r in rows if r.get("plan_score") is not None]
    if not scores:
        return base
    stable = (
        min(scores) >= score_floor
        and (not plan_scores or min(plan_scores) >= score_floor)
    )
    if stable:
        reduced = max(1, base - 1)
        if reduced != base:
            log(
                paths,
                f"Adaptive downshift Ch{chapter_num}: quality stable "
                f"(min score={min(scores):.1f}≥{score_floor}); candidate_plans {base}->{reduced}",
            )
        return reduced
    return base


def create_plan(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
    checkpoint_label: str = "initial",
    cached_memory: str | None = None,
    replan_feedback: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = load_checkpoint(paths, chapter_num, f"plan_{checkpoint_label}_selected.json")
    if isinstance(cached, dict) and cached.get("plan") and cached.get("decision"):
        log(paths, f"Resuming cached {checkpoint_label} plan Ch{chapter_num}")
        return cached["plan"], cached["decision"]

    mem = cached_memory or memory_context(paths, conn, config)
    best_plan: dict[str, Any] | None = None
    best_decision: dict[str, Any] | None = None
    best_score = -1.0
    min_score = float(config["novel"]["min_plan_score"])
    retry_score = float(config["novel"].get("plan_retry_score", min_score - 1.5))
    max_attempts = int(config["novel"].get("plan_max_attempts", 2))
    for attempt in range(max_attempts):
        from config import log as _log
        _log(paths, f"Generating candidate plans Ch{chapter_num} attempt={attempt}")
        plans_key = f"plan_{checkpoint_label}_attempt{attempt}_candidates.json"
        reports_key = f"plan_{checkpoint_label}_attempt{attempt}_reports.json"
        arbitration_key = f"plan_{checkpoint_label}_attempt{attempt}_arbitration.json"

        plans = load_checkpoint(paths, chapter_num, plans_key)
        if isinstance(plans, list) and plans:
            log(paths, f"Resuming cached candidate plans Ch{chapter_num} attempt={attempt}")
        else:
            _log(paths, f"Calling generate_candidate_plans Ch{chapter_num}...")
            n_cand = _effective_candidate_count(conn, config, chapter_num, paths)
            plans = generate_candidate_plans(
                client, paths, conn, config, chapter_num, tail, cached_memory=mem,
                num_candidates_override=n_cand, replan_feedback=replan_feedback,
            )
            _log(paths, f"Got {len(plans)} candidate plans, saving...")
            # Candidate-level scene-dedupe pre-filter: a candidate whose scene
            # skeleton is near-identical to a recently SELECTED plan is dead on
            # arrival — reviewing/arbitrating it wastes LLM calls, and worse, the
            # arbiter regularly picks it (it reads as "consistent with the book").
            # Drop such candidates here unless that would leave none.
            if bool(config["novel"].get("scene_dedupe_enabled", True)) and len(plans) > 1:
                try:
                    from quality import scene_similarity as _scene_sim
                    _recent = _recent_selected_plans(
                        conn,
                        lookback=int(config["novel"].get("scene_dedupe_window", 8)),
                        exclude_chapter=chapter_num,
                    )
                    if _recent:
                        _cut = float(config["novel"].get("scene_dedupe_candidate_block", 0.85))
                        kept, dropped = [], []
                        for p in plans:
                            s = _scene_sim(p, _recent)
                            if float(s.get("max_sim", 0.0) or 0.0) >= _cut:
                                dropped.append((str(p.get("strategy") or "?"), s.get("max_sim")))
                            else:
                                kept.append(p)
                        if dropped and kept:
                            plans = kept
                            log(
                                paths,
                                f"Scene-dedupe candidate filter Ch{chapter_num}: dropped "
                                f"{len(dropped)} near-duplicate candidate(s) {dropped} "
                                f"(sim>={_cut}); {len(plans)} remain.",
                            )
                        elif dropped and not kept:
                            log(
                                paths,
                                f"Scene-dedupe candidate filter Ch{chapter_num}: ALL candidates "
                                f"near-duplicate {dropped}; keeping them (arbitration-stage block will judge).",
                            )
                except Exception as exc:
                    log(paths, f"Scene-dedupe candidate filter failed (non-fatal) Ch{chapter_num}: {exc}")
            save_checkpoint(paths, chapter_num, plans_key, plans)
            _log(paths, f"Saved candidates checkpoint Ch{chapter_num}")

        screen_key = f"plan_{checkpoint_label}_attempt{attempt}_screen.json"
        cached_screen = load_checkpoint(paths, chapter_num, screen_key)
        _n_plans = len(plans)
        _skip_default = _n_plans <= 3
        skip_screen = bool(config["novel"].get("plan_skip_screen", _skip_default))
        if isinstance(cached_screen, list) and cached_screen:
            top_indices = cached_screen
            log(paths, f"Resuming cached screening Ch{chapter_num} attempt={attempt} top={top_indices}")
        elif skip_screen:
            top_indices = list(range(len(plans)))
            save_checkpoint(paths, chapter_num, screen_key, top_indices)
            log(paths, f"Skipping screen Ch{chapter_num}: all {len(plans)} candidates go to agent review")
        else:
            top_indices = screen_candidates(client, paths, conn, config, chapter_num, plans, cached_memory=mem)
            save_checkpoint(paths, chapter_num, screen_key, top_indices)
            log(paths, f"Screened Ch{chapter_num} candidates: top={top_indices} from {len(plans)}")

        screened_plans = [plans[i] for i in top_indices if i < len(plans)]

        # Two-stage planning: when screening was actually run (not skipped) and
        # produced a ranked order for 4+ candidates, pre-eliminate the bottom
        # candidates before the expensive full review to save LLM calls.
        # Keep ceil(67%) of candidates, minimum 2.
        if not skip_screen and len(screened_plans) >= 4:
            keep_n = max(2, math.ceil(len(screened_plans) * 0.67))
            if keep_n < len(screened_plans):
                screened_plans = screened_plans[:keep_n]
                log(
                    paths,
                    f"Two-stage plan knockout Ch{chapter_num}: reduced {len(top_indices)} screened → {keep_n} for full review",
                )

        reports = load_checkpoint(paths, chapter_num, reports_key)
        if isinstance(reports, list) and reports:
            log(paths, f"Resuming cached agent reports Ch{chapter_num} attempt={attempt}")
        else:
            reports = review_candidate_plans(client, paths, conn, config, chapter_num, screened_plans, cached_memory=mem)
            save_checkpoint(paths, chapter_num, reports_key, reports)

        arbitration = load_checkpoint(paths, chapter_num, arbitration_key)
        if isinstance(arbitration, dict) and arbitration.get("plan") and arbitration.get("decision"):
            log(paths, f"Resuming cached arbitration Ch{chapter_num} attempt={attempt}")
            plan = arbitration["plan"]
            decision = arbitration["decision"]
        else:
            from config import log as _log
            _log(paths, f"Calling arbitrate_plan Ch{chapter_num}...")
            plan, decision = arbitrate_plan(client, paths, conn, config, chapter_num, screened_plans, reports, cached_memory=mem)
            _log(paths, f"Got arbitration result, saving Ch{chapter_num}...")
            save_checkpoint(paths, chapter_num, arbitration_key, {"plan": plan, "decision": decision})
            _log(paths, f"Arbitration checkpoint saved Ch{chapter_num}")

        score = plan_score(decision)
        log(paths, f"Arbiter selected Ch{chapter_num} plan score={score}")
        duplicate_blocked = False
        # Scene-skeleton similarity diagnostic: warn (and nudge a retry) when the
        # selected plan is near-identical to a recent one — the signature of the
        # engine slicing the same micro-scene over and over.
        if bool(config["novel"].get("scene_dedupe_enabled", True)):
            try:
                from quality import scene_similarity

                # exclude_chapter is load-bearing: arbitrate_plan persists this
                # chapter's plan_arbitration BEFORE we get here, so without the
                # filter scene_similarity compares the plan against itself and
                # max_sim is pinned at 1.0, forcing a guaranteed false BLOCK.
                recent_sel = _recent_selected_plans(
                    conn,
                    lookback=int(config["novel"].get("scene_dedupe_window", 8)),
                    exclude_chapter=chapter_num,
                )
                sim = scene_similarity(plan, recent_sel)
                warn_threshold = float(config["novel"].get("scene_dedupe_sim_warn", 0.6))
                block_threshold = float(config["novel"].get("scene_dedupe_sim_block", 0.82))
                force_retry = bool(config["novel"].get("scene_dedupe_force_retry", True))
                # Short / chapter-capped novels (暴风雪山庄、密室、单一场景悬疑) reuse
                # the same physical space and cast by design, so a high skeleton
                # overlap is expected, not a bug. Relax the BLOCK there to avoid
                # burning a whole extra plan round chasing differentiation that
                # the premise can't supply; the WARN nudge still fires.
                max_ch = config["novel"].get("max_chapters")
                if max_ch and int(max_ch) <= int(
                    config["novel"].get("scene_dedupe_short_novel_chapters", 8)
                ):
                    block_threshold = max(
                        block_threshold,
                        float(config["novel"].get("scene_dedupe_short_novel_block", 0.92)),
                    )
                    # NOTE: force_retry used to be disabled entirely here, which is
                    # how suspense_v11 Ch8 sailed through with max_sim=1.0 (a plan
                    # literally identical to a recent one). Short-novel mode keeps
                    # the RELAXED threshold (0.92) but retains the hard retry: a
                    # premise can justify reusing the venue/cast, never an
                    # identical conflict/payoff/beats skeleton.
                # Absolute ceiling: a near-identical skeleton is a planning bug in
                # ANY mode and must never be written. Overrides force_retry=false.
                identical_threshold = float(
                    config["novel"].get("scene_dedupe_sim_identical", 0.97)
                )
                if sim.get("max_sim", 0.0) >= warn_threshold:
                    log(
                        paths,
                        f"Scene-dedupe WARN Ch{chapter_num}: selected plan max_sim={sim['max_sim']} "
                        f"vs recent — likely re-slicing the same micro-scene.",
                    )
                    decision.setdefault("required_constraints", []).append(
                        "本章场景骨架与近期高度雷同，必须切换到不同的冲突场景/推进到新的局面，不得继续纠缠同一僵局。"
                    )
                max_sim_val = float(sim.get("max_sim", 0.0) or 0.0)
                if (force_retry and max_sim_val >= block_threshold) or (
                    max_sim_val >= identical_threshold
                ):
                    duplicate_blocked = True
                    decision.setdefault("required_constraints", []).append(
                        "硬性重规划：上一版大纲与近期场景骨架重复度过高。本章必须更换信息来源、物理场地、冲突参与者或兑现类型中的至少两项。"
                    )
            except Exception:
                pass
        # Narrative-pattern dedupe (abstract flow骨架), complementary to the
        #字面-Jaccard scene_similarity above. scene_similarity is blind to "same
        # procedural flow, different subject" (新故事→换水位 share no tokens) — the
        # documented cause of suspense_10ch's Ch3(8.0)→Ch8(6.5) decline. This
        # folds into the same duplicate_blocked retry path.
        if bool(config["novel"].get("narrative_pattern_enabled", True)):
            try:
                from quality import narrative_pattern_repetition

                recent_seq = _recent_selected_plans(
                    conn,
                    lookback=int(config["novel"].get("narrative_pattern_window", 3)),
                    exclude_chapter=chapter_num,
                )
                npr = narrative_pattern_repetition(plan, recent_seq, config)
                decision["narrative_pattern"] = npr
                if npr.get("flags"):
                    log(
                        paths,
                        f"Narrative-pattern Ch{chapter_num}: level={npr.get('level')} "
                        f"max_sim={npr.get('max_sim')} streak={npr.get('consecutive')} "
                        f"seq={'→'.join(npr.get('sequence', []))}",
                    )
                    for directive in npr.get("directives", []):
                        if directive not in decision.setdefault("required_constraints", []):
                            decision["required_constraints"].append(directive)
                if npr.get("level") == "block":
                    duplicate_blocked = True
                    db_event(
                        conn,
                        chapter_num,
                        "narrative_pattern_retry",
                        {"score": score, "narrative_pattern": npr, "plan": plan},
                    )
            except Exception as exc:
                log(paths, f"narrative_pattern check failed (non-fatal) Ch{chapter_num}: {exc}")
        if duplicate_blocked and attempt < max_attempts - 1:
            db_event(
                conn,
                chapter_num,
                "scene_dedupe_retry",
                {"score": score, "decision": decision, "plan": plan},
            )
            log(
                paths,
                f"Scene-dedupe BLOCK Ch{chapter_num}: retrying plan attempt {attempt + 1}/{max_attempts - 1}",
            )
            continue
        visual_blocked = False
        # The visual-payoff gate enforces concrete物证/视觉矛盾 reveals — exactly
        # the spine of "单密室·精密推理" mode. In "serial" (strong-hook/emotional)
        # mode the payoff is often a relational/emotional beat, not a visual
        # contradiction, so forcing the template would mis-steer; allow opting it
        # down to non-blocking advisory there unless explicitly overridden.
        from config import narrative_mode as _nm
        _mode = _nm(config)
        _visual_enabled = bool(config["novel"].get("visual_payoff_check_enabled", True))
        _visual_blocks = bool(config["novel"].get("visual_payoff_blocks_plan", True))
        if _mode == "serial" and "visual_payoff_blocks_plan" not in config["novel"]:
            _visual_blocks = False
        if _visual_enabled:
            try:
                from quality import plan_visual_payoff_check

                visual = plan_visual_payoff_check(plan, config)
                decision["visual_payoff_check"] = visual
                if visual.get("flags"):
                    log(
                        paths,
                        f"Visual-payoff check Ch{chapter_num}: score={visual.get('score')} "
                        f"flags={visual.get('flags')} templates={visual.get('template_hits')}",
                    )
                    for directive in visual.get("directives", []):
                        if directive not in decision.setdefault("required_constraints", []):
                            decision["required_constraints"].append(directive)
                if visual.get("blocked") and _visual_blocks:
                    visual_blocked = True
                    decision.setdefault("required_constraints", []).append(
                        "硬性重规划：本章核心推理爽点过抽象，必须改成具体视觉矛盾模板（画面A vs 现实B），并在 beats 中写出读者可见的物证对照。"
                    )
                    db_event(
                        conn,
                        chapter_num,
                        "visual_payoff_retry",
                        {"score": score, "visual": visual, "decision": decision, "plan": plan},
                    )
            except Exception as exc:
                log(paths, f"Visual-payoff check failed (non-fatal) Ch{chapter_num}: {exc}")
        if visual_blocked and attempt < max_attempts - 1:
            log(
                paths,
                f"Visual-payoff BLOCK Ch{chapter_num}: retrying plan attempt {attempt + 1}/{max_attempts - 1}",
            )
            continue
        # Executability gate (deterministic): the merged_plan's payoff/climax must
        # be a shootable action, not abstract realization. Mechanizes the arbiter's
        # own stated <=7.0 cap, which it ignores under honour-system prompting.
        exec_blocked = False
        try:
            from quality import plan_executability_gate

            execg = plan_executability_gate(plan, config)
            decision["executability_gate"] = execg
            if execg.get("blocked"):
                decision.setdefault("required_constraints", []).append(
                    "硬性重规划：核心 payoff/高潮 beat 停留在抽象领悟（推导出/意识到/想通…），"
                    "无具体动作+具体物体+可见结果。必须改写成'角色用具体动作操作具体物体、"
                    f"产生读者一眼可见结果'的可拍句子。问题句：{execg.get('evidence', '')}"
                )
                db_event(
                    conn,
                    chapter_num,
                    "executability_retry",
                    {"score": score, "gate": execg, "decision": decision, "plan": plan},
                )
                if attempt < max_attempts - 1:
                    exec_blocked = True
        except Exception as exc:
            log(paths, f"Executability gate failed (non-fatal) Ch{chapter_num}: {exc}")
        if exec_blocked:
            log(
                paths,
                f"Executability BLOCK Ch{chapter_num}: abstract payoff, retrying plan "
                f"attempt {attempt + 1}/{max_attempts - 1}",
            )
            continue
        if score > best_score:
            best_plan, best_decision, best_score = plan, decision, score
        if score >= min_score:
            break
        if score >= retry_score:
            log(
                paths,
                f"Ch{chapter_num} plan score={score} below min={min_score} but above retry_threshold={retry_score}; "
                f"accepting without retry to save tokens.",
            )
            break
        db_event(conn, chapter_num, "low_plan_score_retry", {"score": score, "decision": decision})
    assert best_plan is not None and best_decision is not None
    save_checkpoint(
        paths,
        chapter_num,
        f"plan_{checkpoint_label}_selected.json",
        {"plan": best_plan, "decision": best_decision},
    )
    return best_plan, best_decision
