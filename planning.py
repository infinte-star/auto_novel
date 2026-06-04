from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint, save_checkpoint
from config import Paths, log, safe_score
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, lite_memory_context, memory_context, rhythm_diagnostics, structural_repetition_analysis
from store import JsonStoryStore, db_event, db_lock, get_active_constraints, get_overdue_reader_promises, get_reader_promises, get_silent_threads, recent_metrics, recent_quality_feedback

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
- 避免复用最近章节的章末手法、分析姿态或情感节拍。"""

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
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "selected_index": 0,
  "scores": [{"index": 0, "score": 1-10, "pros": [], "cons": []}],
  "merged_plan": {
    "title": "...", "goal": "...", "conflict": "...", "conflict_type": "...",
    "payoff": "...", "payoff_type": "...", "pressure": "...",
    "beats": ["..."], "character_focus": ["..."], "world_state_changes": ["..."],
    "thread_actions": ["..."], "hook": "...", "risk": "..."
  },
  "required_constraints": ["作者必须遵守的硬性约束"],
  "reader_expectation_delta": "为何这样能提升或损害读者的追读欲"
}
merged_plan 必须包含上述全部键，不得缺字段。改写 beats 时，每个 beat 仍须是完整主谓宾句子，禁止破折号状态短语堆叠。
对以下大纲予以否决或降分：把已知审校问题停留在抽象层面、依赖在页面之外解决、重复相同的物理调度、或留有未解决的时间线/物流漏洞。
若候选采用 "reversal"（反转）策略，当其反转没有铺垫（事先建立并强化过一个事实/信任源，再将其推翻）时降分——
没有铺垫的反转只是突兀的转折，而非兑现。请改进 merged_plan，让作者拿到的是具体的场景任务，而非含糊的意图。"""

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

AGENT_REVIEW_SYSTEMS = {
    "world": """你是一部中国历史/玄幻网文的「世界 Agent」。
请对照既定世界规则审查章节大纲。具体核查：
1. 地理与旅行：距离、路线、旅行时间是否一致（京城到江南需数日，而非瞬移）
2. 力量体系：修炼/战斗/政治权力规则是否符合既定设定
3. 机构：官职、官僚程序、等级是否符合时代
4. 资源：金钱、物资、兵力是否守恒（无无故补充）
5. 历法与季节：日期是否与既定时间线对齐，季节细节是否一致

使用完整的 1-10 区间；默认从 6.5 起步，9+ 仅保留给几乎无缺陷的维度，不可滥发。施加软性惩罚（扣分，不钳制）：
- 违反地理/旅行时间 -1.5
- 与既定规则矛盾的力量体系 -2.0
- 机构程序时代错置或不可能 -2.5

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
schema：{"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "character": """你是一部中国历史/玄幻网文的「人物 Agent」。
请审查章节大纲的人物一致性与成长。具体核查：
1. 目标与动机：每个人物都依据既定目标行动，而非剧情便利
2. 能动性：人物做出有可见代价的主动选择，而非被动旁观
3. 关系：互动反映既定的人物关系（盟友、敌人、人情债）
4. 秘密与知识：人物只依据其确实拥有的信息行动
5. 成长弧线：主角呈现渐进式变化，而非突然的性格突变
6. 对话口吻：每个人物的说话方式契合其出身与身份

使用完整的 1-10 区间；默认从 6.5 起步，9+ 仅保留给几乎无缺陷的维度，不可滥发。施加软性惩罚（扣分，不钳制）：
- 人物依据其不应拥有的信息行动 -2.0
- 主角在本章没有有意义的选择或代价 -1.0
- 人物无理由地脱离人设 -1.5

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
schema：{"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "rhythm": """你是一部中国历史/玄幻网文的「节奏 Agent」。
请对照近期章节审查节奏与结构变化。具体核查：
1. 场景结构：本章的开场/收场手法是否不同于最近 3 章？
2. 压缩/释放：是否既有张力积累又有释放时刻？
3. 场景数量与变化：至少 2 个设定或动态不同的场景
4. 章末手法：不与前 2 章相同类型（悬念/揭示/平静收尾）
5. 信息密度：动作、对话与反思之间是否平衡（无 1000 字以上的独白）

使用完整的 1-10 区间；默认从 6.5 起步，9+ 仅保留给几乎无缺陷的维度，不可滥发。施加软性惩罚（扣分，不钳制）：
- 章末与上一章重复同一手法 -1.0
- 整章是单一拉长场景、毫无切换 -1.5
- 节奏单调（全程高张力或全程低张力） -1.0

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
schema：{"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "payoff": """你是一部中国历史/玄幻网文的「兑现 Agent」。
请审查情感兑现质量与压迫-兑现的平衡。具体核查：
1. 压迫积累：兑现之前是否有有意义的阻力/障碍？
2. 兑现新鲜度：payoff_type 是否不同于最近 3 章？
3. 代价可见：兑现是否伴随可见的代价或取舍？
4. 挣来的解决：解决是否由因果挣来（而非巧合或天降救星）？
5. 情感质地：本章是否唤起一种有区分度的情感，而非泛泛的紧张？

使用完整的 1-10 区间；默认从 6.5 起步，9+ 仅保留给几乎无缺陷的维度，不可滥发。施加软性惩罚（扣分，不钳制）：
- 兑现依赖巧合或无解释的运气 -2.0
- payoff_type 与前 2 章相同 -1.0
- 主角没有可见的代价或取舍 -1.0

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
schema：{"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "foreshadowing": """你是一部中国历史/玄幻网文的「伏线 Agent」。
请审查伏线管理与长线承诺的兑现。具体核查：
1. 逾期伏线：标记任何 >20 章前引入、却未在此推进的已开启伏线
2. 伏线推进：本章是否至少推进一条已有伏线？
3. 新伏线引入：若开启新伏线，其 due_chapter 是否现实？
4. 找回机会：是否有被丢弃、可在此自然找回的伏线？
5. 承诺密度：已开启伏线不要过多（>8 条活跃 = 读者混乱风险）

使用完整的 1-10 区间；默认从 6.5 起步，9+ 仅保留给几乎无缺陷的维度，不可滥发。施加软性惩罚（扣分，不钳制）：
- 存在可处理却未处理的逾期伏线（>20 章） -1.0
- 没有推进或找回任何已有伏线 -1.5
- 在不闭合旧伏线的情况下开启第 9 条及以上并发伏线 -1.0

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
schema：{"score":1-10,"risks":[],"required_fixes":[],"state_patch":[]}""",

    "reader": """你是一部中国历史/玄幻网文的「读者模拟 Agent」。
请模拟一名连载读者读完本章大纲。具体评估：
1. 追读欲：读完本章后，哪 3 个问题会让读者点击"下一章"？
2. 满足感：本章是否提供至少一个满足时刻（而非全是铺垫）？
3. 混乱风险：跳读了 2 章的读者是否仍能跟上主线？
4. 疲劳信号：读者是否被要求同时追踪过多伏线？
5. 情感钩子：是否有一个能引发共情或投入的人物时刻？

使用完整的 1-10 区间；默认从 6.5 起步，9+ 仅保留给几乎无缺陷的维度，不可滥发。施加软性惩罚（扣分，不钳制）：
- 没有清晰的"下一章"问题 -1.0
- 本章是纯铺垫、零兑现时刻 -1.5
- 读者需要记住 >5 个先前情节点才能看懂本章 -1.0

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。
schema：{"score":1-10,"risks":[],"required_fixes":[],"state_patch":[],"follow_next_reason":"..."}""",
}

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
    "wins" counts how often a candidate with that strategy was the
    arbiter-selected one.
    """
    if isinstance(conn, JsonStoryStore):
        events = conn.recent_events(lookback)
    else:
        try:
            with db_lock():
                rows = conn.execute(
                    "SELECT payload FROM events WHERE event_type='plan_arbitration' "
                    "ORDER BY id DESC LIMIT ?",
                    (lookback,),
                ).fetchall()
            events = [{"payload": json.loads(r["payload"])} for r in rows]
        except Exception:
            return {}
    stats: dict[str, dict[str, float]] = {}
    for ev in events:
        payload = ev.get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        # plan_arbitration payload shape: {"decision": {...}, "plans": [...]}
        decision = payload.get("decision") or {}
        plans = payload.get("plans") or []
        if not plans:
            continue
        sel_idx = int(decision.get("selected_index", 0))
        scores = decision.get("scores") or []
        score_map = {int(s.get("index", -1)): safe_score(s.get("score", 0)) for s in scores}
        for i, plan in enumerate(plans):
            if not isinstance(plan, dict):
                continue
            strat = str(plan.get("strategy") or "").strip()
            if not strat:
                continue
            entry = stats.setdefault(strat, {"trials": 0.0, "score_sum": 0.0, "wins": 0.0})
            entry["trials"] += 1
            entry["score_sum"] += float(score_map.get(i, 5.0))
            if i == sel_idx:
                entry["wins"] += 1
    return stats


def _select_strategies_bandit(
    conn: Any,
    config: dict[str, Any],
    strategies: list[tuple[str, str]],
    n: int,
    paths: Paths,
) -> list[tuple[str, str]]:
    """Epsilon-greedy selection of n strategies from the candidate pool.

    Score per strategy = mean(score) + 0.5 * win_rate. Strategies with
    fewer than 3 trials are treated as "exploration" and always included
    in the pool. Picks top-n by composite score with ε probability of a
    random swap to keep exploring.
    """
    import random as _random

    bandit_enabled = bool(config["novel"].get("strategy_bandit", True))
    if not bandit_enabled or n <= 0:
        return [strategies[i % len(strategies)] for i in range(n)]

    lookback = int(config["novel"].get("strategy_bandit_lookback", 60))
    epsilon = float(config["novel"].get("strategy_bandit_epsilon", 0.2))
    stats = _strategy_history(conn, lookback=lookback)

    scored: list[tuple[float, int, tuple[str, str]]] = []
    for idx, strat in enumerate(strategies):
        name = strat[0]
        s = stats.get(name)
        if not s or s["trials"] < 3:
            # Boost under-explored strategies so they get picked sometimes.
            composite = 9.0 + _random.random() * 0.5
        else:
            mean_score = s["score_sum"] / s["trials"]
            win_rate = s["wins"] / s["trials"]
            composite = mean_score + 0.5 * win_rate
        scored.append((composite, idx, strat))

    # Sort by composite desc, stable on original idx.
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [item[2] for item in scored[:n]]

    # With probability epsilon, swap one of the picked with a random un-picked.
    if epsilon > 0 and len(strategies) > n and _random.random() < epsilon:
        picked_names = {p[0] for p in picked}
        leftovers = [s for s in strategies if s[0] not in picked_names]
        if leftovers:
            swap_in = _random.choice(leftovers)
            swap_out_idx = _random.randrange(len(picked))
            picked[swap_out_idx] = swap_in

    try:
        log(paths, f"Strategy bandit picked: {[p[0] for p in picked]}")
    except Exception:
        pass
    return picked


def _recent_selected_plans(conn: Any, lookback: int = 8) -> list[dict[str, Any]]:
    """Return the most recent arbiter-selected (merged) plans, newest first.

    Used for scene-skeleton dedupe: the candidate generator is told to avoid
    re-running the same conflict/payoff/beats that recent chapters already used,
    which is the engine's main defense against "infinitely slicing one scene".
    """
    if isinstance(conn, JsonStoryStore):
        events = conn.recent_events(lookback * 3)
    else:
        try:
            with db_lock():
                rows = conn.execute(
                    "SELECT payload FROM events WHERE event_type='plan_arbitration' "
                    "ORDER BY id DESC LIMIT ?",
                    (lookback,),
                ).fetchall()
            events = [{"payload": json.loads(r["payload"])} for r in rows]
        except Exception:
            return []
    plans: list[dict[str, Any]] = []
    for ev in events:
        payload = ev.get("payload") if isinstance(ev, dict) else None
        if not isinstance(payload, dict):
            continue
        decision = payload.get("decision") or {}
        merged = decision.get("merged_plan")
        cand = payload.get("plans") or []
        if isinstance(merged, dict) and merged:
            plans.append(merged)
        elif cand:
            sel = int(decision.get("selected_index", 0))
            if 0 <= sel < len(cand) and isinstance(cand[sel], dict):
                plans.append(cand[sel])
        if len(plans) >= lookback:
            break
    return plans


def generate_candidate_plans(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    tail: str,
    cached_memory: str | None = None,
    num_candidates_override: int | None = None,
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
    mem = cached_memory or memory_context(paths, conn, config)
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
            recent_sel = _recent_selected_plans(conn, lookback=window)
            skeletons = []
            for rp in recent_sel:
                skeletons.append({
                    "conflict": str(rp.get("conflict", ""))[:120],
                    "payoff": str(rp.get("payoff", ""))[:120],
                    "payoff_type": rp.get("payoff_type", ""),
                })
            if skeletons:
                dedupe_block = json.dumps(skeletons, ensure_ascii=False, indent=2)
        except Exception:
            dedupe_block = "None"
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

## 上章结尾
{tail[-2000:]}

## 请求
为第 {chapter_num} 章生成候选大纲。
避免近期重复。保留因果债务。提升读者追读欲。
若上方存在沉默伏线，大纲必须在 beats/thread_actions 中推进其中之一，或在 "risk" 中明确说明为何本章均不可行。
若 "节奏诊断JSON" 报告了爽点拖欠警告（chapters_since_payoff >= payoff_max_gap），本章的 payoff_type 必须是一个具体的读者兑现（court_breakthrough/policy_payoff/military_victory/reveal/reversal/personnel_payoff/institutional_fix），而非 strategic_setup 或 emotional。"""
    from config import is_final_chapter
    if is_final_chapter(config, chapter_num):
        base_user += """

## 终章要求（硬性：这是全书最后一章，必须规划成结局而非过渡章）
- 本章 payoff_type 必须是一个真正的读者兑现（court_breakthrough/policy_payoff/military_victory/reveal/reversal/personnel_payoff/institutional_fix），严禁 strategic_setup。
- goal/payoff 必须正面解决全书主线矛盾，把已开启的关键伏线在本章收束。
- "hook" 字段不再是抛给读者的新悬念，而是一句收束/余韵/主题升华；严禁以全新未解决危机作结。
- beats 的最后 1-2 拍必须落在"结局兑现 + 情绪落点"，而非开启新冲突。"""
    num_candidates = int(num_candidates_override) if num_candidates_override else int(config["novel"]["candidate_plans"])
    max_workers = int(config["novel"].get("max_parallel_workers", 5))

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
        ("institutional",
         "以制度/程序/官僚摩擦为核心：本章必须呈现一次具体的衙门程序（如送文、批红、查证、回禀），用程序细节制造张力。"),
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
            f"你必须围绕这一策略来设计本候选大纲。"
            f"其它候选采用不同策略——不要趋同。"
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
                    temperature=0.65 + idx * 0.05,
                    cacheable_prefix=cacheable_prefix(paths, config),
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
        max_workers = int(config["novel"].get("max_parallel_workers", 5))

        def review_one(agent: str, system: str) -> dict[str, Any]:
            for retry in range(2):
                try:
                    raw = call_llm(
                        client, paths, config, system, json_prompt(user),
                        max_tokens=12000, temperature=0.2,
                        cacheable_prefix=cacheable_prefix(paths, config),
                    )
                    report = load_json_with_repair(
                        client,
                        paths,
                        config,
                        raw,
                        fallback={"score": 5, "risks": [], "required_fixes": [], "state_patch": []},
                    )
                    report["agent"] = agent
                    return report
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    log(paths, f"Agent {agent} review parse failed retry={retry}: {exc}")
            return {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}

        reports: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(review_one, agent, system): agent
                for agent, system in AGENT_REVIEW_SYSTEMS.items()
            }
            for future in as_completed(futures):
                reports.append(future.result())

    for report in reports:
        agent = report["agent"]
        if isinstance(conn, JsonStoryStore):
            conn.add_agent_report(chapter_num, agent, report)
        else:
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
    if not isinstance(conn, JsonStoryStore):
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
    fused_enabled = bool(config["novel"].get("fused_plan_review", True))

    if fused_enabled:
        # One fused LLM call per candidate plan; expands to 6 axis reports each.
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
    else:
        def review_one(plan_index: int, agent: str, system: str) -> dict[str, Any]:
            user = plan_users[plan_index]
            for retry in range(2):
                try:
                    raw = call_llm(
                        client, paths, config, system, json_prompt(user),
                        max_tokens=12000, temperature=0.2,
                        cacheable_prefix=cacheable_prefix(paths, config),
                    )
                    report = load_json_with_repair(
                        client,
                        paths,
                        config,
                        raw,
                        fallback={"score": 5, "risks": [], "required_fixes": [], "state_patch": []},
                    )
                    report["agent"] = agent
                    return report
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    log(paths, f"Agent {agent} review parse failed plan={plan_index} retry={retry}: {exc}")
            return {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(review_one, plan_index, agent, system): (plan_index, agent)
                for plan_index in range(len(plans))
                for agent, system in AGENT_REVIEW_SYSTEMS.items()
            }
            for future in as_completed(futures):
                plan_index, agent = futures[future]
                try:
                    reports_by_plan[plan_index].append(future.result())
                except Exception as exc:
                    log(paths, f"Agent {agent} review thread failed plan={plan_index}: {exc}")
                    reports_by_plan[plan_index].append(
                        {"agent": agent, "score": 5, "risks": [], "required_fixes": [], "state_patch": []}
                    )

    for reports in reports_by_plan:
        for report in reports:
            agent = report["agent"]
            if isinstance(conn, JsonStoryStore):
                conn.add_agent_report(chapter_num, agent, report)
            else:
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
    if not isinstance(conn, JsonStoryStore):
        with db_lock():
            conn.commit()

    return reports_by_plan

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
    user = f"""## 记忆
{mem}

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
    )
    decision = load_json_with_repair(client, paths, config, raw)
    plan = decision.get("merged_plan") or plans[int(decision.get("selected_index", 0))]
    db_event(conn, chapter_num, "plan_arbitration", {"decision": decision, "plans": plans})
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

def _effective_candidate_count(conn: Any, config: dict[str, Any], chapter_num: int, paths: Paths) -> int:
    """Adaptively reduce candidate-plan count when quality is stable + strategy
    bandit has converged, to save tokens on the long tail of a book.

    Returns the number of candidate plans to generate this chapter. Never goes
    below 1; only kicks in after a warm-up so early chapters keep full breadth.
    """
    base = int(config["novel"]["candidate_plans"])
    if not bool(config["novel"].get("adaptive_downshift_enabled", True)):
        return base
    warmup = int(config["novel"].get("adaptive_downshift_warmup", 60))
    if chapter_num < warmup or base <= 1:
        return base
    window = int(config["novel"].get("adaptive_downshift_window", 10))
    rows = recent_metrics(conn, window)
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
                num_candidates_override=n_cand,
            )
            _log(paths, f"Got {len(plans)} candidate plans, saving...")
            save_checkpoint(paths, chapter_num, plans_key, plans)
            _log(paths, f"Saved candidates checkpoint Ch{chapter_num}")

        screen_key = f"plan_{checkpoint_label}_attempt{attempt}_screen.json"
        cached_screen = load_checkpoint(paths, chapter_num, screen_key)
        skip_screen = bool(config["novel"].get("plan_skip_screen", False))
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

                recent_sel = _recent_selected_plans(conn, lookback=int(config["novel"].get("scene_dedupe_window", 8)))
                sim = scene_similarity(plan, recent_sel)
                if sim.get("max_sim", 0.0) >= float(config["novel"].get("scene_dedupe_sim_warn", 0.6)):
                    log(
                        paths,
                        f"Scene-dedupe WARN Ch{chapter_num}: selected plan max_sim={sim['max_sim']} "
                        f"vs recent — likely re-slicing the same micro-scene.",
                    )
                    decision.setdefault("required_constraints", []).append(
                        "本章场景骨架与近期高度雷同，必须切换到不同的冲突场景/推进到新的局面，不得继续纠缠同一僵局。"
                    )
                block_threshold = float(config["novel"].get("scene_dedupe_sim_block", 0.82))
                if (
                    bool(config["novel"].get("scene_dedupe_force_retry", True))
                    and sim.get("max_sim", 0.0) >= block_threshold
                ):
                    duplicate_blocked = True
                    decision.setdefault("required_constraints", []).append(
                        "硬性重规划：上一版大纲与近期场景骨架重复度过高。本章必须更换信息来源、物理场地、冲突参与者或兑现类型中的至少两项。"
                    )
            except Exception:
                pass
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
