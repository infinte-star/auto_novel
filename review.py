from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from config import Paths, append_text, chapter_path, log, normalize_text, read_text, safe_score, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import STYLE_HEALTH_GUARDRAILS, _VOLUME_PLAN_STRUCTURE_SPEC, cacheable_prefix, contract_block, memory_context, rhythm_diagnostics, structural_repetition_analysis, writing_memory_context
from quality import REGISTRY
from store import (
    db_event,
    entity_state_as_of,
    get_active_constraints,
    get_character_voice_notes,
    get_open_causal_requirements,
    get_overdue_reader_promises,
    get_silent_threads,
    recent_metrics,
    recent_quality_feedback,
    store_stage_constraints,
)

if TYPE_CHECKING:
    from openai import OpenAI

REVIEW_SYSTEM = """你是连载中文网文的严格终审编辑。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "score": 1-10,
  "readthrough_score": 1-10,
  "hook_score": 1-10,
  "payoff_score": 1-10,
  "novelty_score": 1-10,
  "prose_score": 1-10,
  "continuity_score": 1-10,
  "emotional_impact": 1-10,
  "accepted": true,
  "problems": [],
  "fixes": [],
  "continuity_risks": [],
  "rhythm_risks": [],
  "reader_fatigue_risks": [],
  "hook_strength": 1-10,
  "aesthetic_score": 1-10,
  "style_audit": {"em_dash_per_kchar": 0.0, "fragment_line_ratio": 0.0, "has_full_dialogue": true},
  "beats_audit": [{"beat":"...", "status":"realized|partial|absent", "evidence":"引文或备注"}],
  "contradictions": [{"fact":"被违背的既定事实", "prose":"引用章节中违背它的 6-20 字原文", "severity":"hard|soft"}],
  "contract_violations": [{"rule":"被违反的契约条款（引用契约里的能力白/黑名单/禁止套路/必守设定的具体一条）", "prose":"引用章节中违反它的 6-30 字原文", "type":"ability_out_of_scope|ability_modality_drift|blacklist|banned_trope|must_hold", "severity":"hard|soft"}],
  "hallucinated_entities": ["在章节中被当作已确立、但不在既定事实中、且此前也未被引入的人名/地名/物品/势力"],
  "character_voice_drift": [{"name":"立场/口吻与基线矛盾的人物", "prose":"引用 6-20 字体现偏移的原文", "note":"与基线立场如何冲突"}],
  "patches": [
    {"op":"replace", "locator":"引用当前文本中 8-20 字", "before":"待替换的精确子串", "after":"替换文本", "reason":"原因"},
    {"op":"insert_after", "locator":"在其后插入的 8-20 字定位串", "insert":"新增文本", "reason":"原因"},
    {"op":"delete", "locator":"标识待删除段落的 8-20 字", "before":"待删除的精确子串", "reason":"原因"}
  ],
  "writer_directives_for_next_chapter": [
    "3-6 条下一章作者必须遵守的祈使指令",
    "每条都必须是具体的执行级指导，而非抽象建议",
    "示例：'下一章必须用反转结构，最近 3 章都是 pressure-payoff'、'户部官僚程序需要落到至少一段对话上'、'主角必须在场景 2 做一次有可见代价的选择'"
  ]
}

## 评分理念（诚实分布——拒绝分数通胀）
分数必须是 1-10 区间上诚实的质量评估。**默认假设本章存在缺陷**：从基础 6.5 起步，逐项检查后只有确实通过检查的维度才上浮，发现问题就下扣。
绝不要因为"读起来还行"就给 8+。如果你给出的分数长期聚集在 8-9 附近，说明你在通胀——这会让修订循环失效。
评分锚点：
- 9.5-10：典范级。因果严密、形态多变、文风健康可读、有挣来的兑现与犀利钩子，且**找不到任何明显缺陷**。
- 8.5-9：很强。仅有 1-2 个轻微表面问题。
- 7-8.5：扎实可用，但有具体可改进处。
- 5.5-7：可读但有明显短板（节拍缺失、兑现空洞、文体偏弱）。
- <=5：存在结构性或可读性问题，需要重写。

## 爆款/追读拆分评分（必须先于总分）
除总分 score 外，必须单独给出以下 5 个维度，后续流程会用它们判断是否需要重规划，而不是只相信综合分：
- readthrough_score：读者读完本章后继续点下一章的欲望；看具体未解问题、情绪悬念、下一章承诺。
- payoff_score：本章是否给了清晰、挣来的兑现/爽点/情绪收益，而不是纯铺垫。
- novelty_score：相对最近章节的新鲜增量（对照前章判定，不是题材是否常见）。锚点：<=6.5=与前章高度雷同 7=复用前章骨架仅换措辞（惩罚分） 8+=至少1项实质翻新（信息源/冲突类型/章末手法/能力用法）
- prose_score：正文可读性、语言质感、对话、意象、节奏；不要把设定正确当作文笔好。
- continuity_score：事实、时间线、人物知识、资源流转、因果闭合程度。
hook_score 与 hook_strength 可以相同，但若章末问题笼统或近期重复，hook_score 必须低于 7。
- emotional_impact：读者是否产生真实情感反应（不是"写了悲伤"而是"读者会心痛"）。锚点：9+=强烈生理反应且情感由事件挣来 7-8=有触动但未"被击中" 5-6=平淡处理 <=4=纯功能推进无感受
score 是综合质量，不得掩盖 readthrough/payoff/novelty 的短板；若任一追读相关维度低于 6.5，score 原则上不应超过 8。

style_audit 如实填写即可（引擎独立测量并据此扣分，你的评分注意力放在剧情因果/兑现/人物/节奏/连续性/审美）。

## 审美评估（必填 aesthetic_score，独立于 score 单列 1-10）
评估维度：动词精准度、意象与潜台词、克制留白、比喻新鲜度、长短句节奏、叙事腔调辨识度。
锚点：9-10 文笔出众有记忆点；7-8 干净有质感但不惊艳；5-6 通顺但平庸、套话偏多；<=4 文笔粗糙、陈词滥调成堆。

文风扣分由引擎确定性门禁执行，你只负责内容维度（因果/人物/情节）的评分；观察到文体问题写入 problems 即可，不要在 score 里扣。
从原始功力（场景具体度、对话、情感兑现）基础分起步，按以下扣分：
- 缺席大纲节拍 -1.0/个，部分实现 -0.5/个；>30%缺席额外 -0.5
- 含糊带过时间线/金钱/路线/程序 -1.0/处
- 重复近期场景形态或章末手法 -1.0
- 忽视连续性风险 -1.0/个（上限 -2.0）；沉默伏线可推进却忽视 -0.7
- HARD矛盾 -2.0 / SOFT矛盾 -0.5（逐条记入 contradictions）
- HARD契约违约 -2.0 / SOFT违约 -0.7（逐条记入 contract_violations；能力越界/模态漂移默认 HARD）
- 幻觉实体 -0.7/个；人物口吻偏移 -0.5/个（上限 -1.0）
- 审美贫乏（陈词滥调/贴标签/句式单调） -0.3~-1.0（上限 -1.5）
加分（上限 +1.5）：全节拍具体实现 +0.5 | 解决反馈且保持张力 +0.7 | 场景手法有区分度 +0.3 | 主角有代价+能动 +0.3 | 文笔出众(aesthetic≥8.5) +0.5

最终分数钳制到 [1.0, 10.0]。9.0+ 仅保留给没有关键扣分项的章节。

大纲节拍审计（必填）：
对大纲 "beats" 数组中的每个节拍，向 beats_audit 添加一条：
- "realized"：该节拍以可见动作在页面上充分实现
- "partial"：该节拍被提及，但缺乏具体场景或感官细节
- "absent"：该节拍缺失或仅在页面之外被暗示

补丁（score < 9 或有 partial/absent 节拍时必填）：
- 1-8 个独立的外科式补丁，locator/before 逐字引用章节原文（8-20字连续子串）。
- insert/after <= 200 中文字符且自足；优先 insert_after 补缺失场景、replace 修措辞。
- 补丁间无依赖，可任意顺序应用。9+ 且无缺失节拍时可返回 []。

作者指令（必填）：输出 3-6 条下一章作者必须遵守的祈使指令。
- 要执行级具体（具体的场景类型、结构选择或人物动作），不要抽象。
- 每条为一句简短中文。
- 优先给出能修复本章具体问题、或弥补近期重复的指令。

钩子强度（必填）：独立地为本章的结尾钩子打 1-10 分。
- 9-10：结尾抛出一个犀利、具体、令读者点击"下一章"的问题。
- 6-8：可用但通用、或近期已用过的钩子。
- <=5：弱/总结式结尾——不要用含糊的"他知道，一切才刚刚开始"式收尾。

矛盾与幻觉检查（当提供了 "## 既定事实" 时必填）：
- 将章节正文逐条对照每个既定事实。"hard" 矛盾是对既述事实的直接推翻（状态/位置/持有/关系）。要保守：只标记你能用逐字原文引出的矛盾。章节自身在页面上演出的合理新发展不算矛盾。
- 若不存在矛盾，返回 "contradictions": [] 与 "hallucinated_entities": []。

契约校验（当提供了 "## 创作契约" 时必填）：
- 逐条对照能力白/黑名单、禁止套路、必守硬设定。重点核对能力越界/模态漂移（如把文本记忆悄悄当听觉用 = ability_modality_drift）。
- 要保守：只标记能用逐字原文引出的违约。**强制路由**：所有违约必须写进 contract_violations（不能只在 problems 里）。自洽不等于合规。
- 模态漂移默认 hard。无违约返回 []。"""

def established_facts_for_chapter(
    conn: Any,
    plan: dict[str, Any],
    chapter_num: int,
    budget_chars: int = 3000,
    promise_grace: int = 15,
) -> str:
    """Compact, budget-limited block of established facts the chapter must not
    contradict: current state of plan-focused characters/entities, relevant open
    threads, and overdue reader promises. Reuses store query helpers."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    def add_entity(etype: str, name: str) -> None:
        key = (etype, str(name))
        if not name or key in seen:
            return
        seen.add(key)
        try:
            state = entity_state_as_of(conn, etype, str(name), chapter_num)
        except Exception:
            state = {}
        if state:
            keep = {k: state[k] for k in list(state)[:6]}
            lines.append(f"- [{etype}] {name}: {json.dumps(keep, ensure_ascii=False)}")

    for char in plan.get("character_focus", []) or []:
        add_entity("character", str(char))
    for force in plan.get("forces", []) or []:
        add_entity("force", str(force))

    # Open threads referenced by the plan's thread_actions, plus overdue promises.
    try:
        from store import db_lock
        with db_lock():
            rows = conn.execute(
                "SELECT id, description, thread_type FROM open_threads WHERE status='open' ORDER BY updated_chapter DESC LIMIT 12",
            ).fetchall()
        for r in rows:
            lines.append(f"- [thread:{r['thread_type']}] {r['id']}: {r['description']}")
    except Exception:
        pass

    try:
        promises = get_overdue_reader_promises(conn, chapter_num, grace=promise_grace)
        for p in promises:
            lines.append(f"- [overdue_promise] {p['id']} (due Ch{p['due_chapter']}): {p['description']}")
    except Exception:
        pass

    if not lines:
        return "None"
    block = "\n".join(lines)
    if len(block) > budget_chars:
        block = block[:budget_chars] + "\n…(truncated)"
    return block

STAGE_REVIEW_SYSTEM = """你是连载中文网文的长周期质量评估者。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "quality_trend": "近期分数与吸引力走势的概括",
  "continuity_risks": ["跨多章的具体连续性问题"],
  "rhythm_payoff_risks": ["该窗口内的节奏或压迫-兑现问题"],
  "repetition_risks": ["重复的结构、兑现或调度"],
  "next_20_chapters_replan": ["接下来 20 章的具体规划调整"],
  "threads_to_recover_or_upgrade": ["需要关注或提升的已开启伏线"],
  "writer_directives_for_next_chapter": ["3-6 条紧接的下一章作者必须遵守的具体祈使指令"],
  "constraints": [
    {"type": "avoid|require|replan|recover_thread", "description": "...", "priority": 1-10, "expires_in_chapters": 20}
  ]
}"""

PACK_REVIEW_SYSTEM = """你是连载网文的 10 章包追读编辑。
只返回恰好一个合法 JSON 对象，不要输出其它内容。
你评估的不是单章文笔，而是这个窗口是否形成可持续追读。

schema:
{
  "window_summary": "这组章节的读者体验概括",
  "readthrough_curve": "追读曲线：哪里上升、哪里掉速",
  "payoff_ledger": {
    "opened_promises": ["新开的读者承诺"],
    "paid_off_promises": ["已经兑现的承诺"],
    "overdue_promises": ["拖欠或快拖欠的承诺"]
  },
  "repetition_patterns": ["重复的场景/信息源/章末手法/解决方式"],
  "drop_off_risks": ["会导致读者弃书的具体风险"],
  "next_10_directives": ["接下来10章必须执行的具体指令"],
  "constraints": [
    {"type": "avoid|require|replan|recover_thread", "description": "...", "priority": 1-10, "expires_in_chapters": 10}
  ]
}

评审重点：
- 这10章是否每 2-3 章至少有一次明确情绪收益/爽点兑现。
- 是否出现只开承诺、不兑现承诺。
- 是否同一场景、同一信息源、同一章末手法反复使用。
- 接下来10章应该关闭哪些账、升级哪些冲突、换哪些场景。"""

REPLAN_SYSTEM = """你是长篇小说引擎的战略重规划者。
当前卷纲在质量指标上已经下滑。请分析当前状态、近期走势、已开启伏线与重复模式。
为接下来的 40-60 章产出一份修订后的 volume_plan，要求：
- 解决陈旧或逾期的伏线
- 引入近期章节未见的新冲突维度
- 转变人物关系与权力格局
- 避开重复分析中被标记的模式
- 与既定事件保持因果一致
- 提升读者的期待感与追读欲

## 输出要求（与初始卷纲同构，必须结构化）
- 保留「当前 Volume Plan 全文」中、已写章节（<= 当前章节号）对应的、已成事实的卷纲部分，不得与既成事实冲突；只重规划当前章节及之后。
- 用 markdown。
""" + _VOLUME_PLAN_STRUCTURE_SPEC + """
- 保持卷间因果递进：上一卷遗留危机 = 下一卷核心矛盾来源。
只返回修订后的完整 volume_plan markdown，不要解释。"""

VOICE_ANCHOR_SYSTEM = """你负责维护一部长篇连载小说的叙事声音锚。
你会收到：原始声音基线（前若干章稳定下来的健康文风）、当前的 voice.md、近期实际正文。
你的任务：产出更新后的 voice.md，让它**始终描述一种人类可读的、句子完整的小说文风**，
而不是被近期正文里出现的任何文体退化所同化。

## 防退化纪律（最高优先级，凌驾于"吸收近期特征"之上）
近期正文可能已经发生**文体塌缩**——句子被拆成单词短句、用大量破折号（——）把碎片粘连、
通篇舞台提示式断行、缺乏完整对话。这是缺陷，不是风格。你**绝不能**把这些特征写进 voice.md。
voice.md 必须显式包含以下"健康文风护栏"：
""" + STYLE_HEALTH_GUARDRAILS + """

## 更新规则
- 以"原始声音基线"为锚：保留其至少 80% 的约束。
- 只吸收近期正文中**健康且正面**的新特征（如新意象、新人物口吻），明确拒绝吸收上述退化特征。
- 不要削弱既有禁忌清单；可增补。
- 输出完整替换版 voice.md，用中文，仅 markdown。
- 保留小节：时态/视角、句长节奏、词汇调性、感官锚、心境呈现、章节结构惯例、节奏禁忌、**健康文风护栏**。
- 在底部加一段简短修订日志：`## 修订日志\\n- Ch{chapter_num}: <一句话概括>`。"""

VOICES_TABLE_SYSTEM = """你负责维护一部长篇小说的人物声音表。
你会收到：当前的 voices.md，以及近期出现具名人物的章节。
请更新 voices.md：
- 依据近期正文中的实际表现，细化每个既有人物的声音指纹。
- 为近期出现、但尚无条目的 1-2 个具名新人物添加条目。
- 保留所有既有人物；细化而非删除。
输出完整更新版 voices.md，用中文，仅 markdown。小节结构与输入一致。"""


def _platform_guidance(config: dict[str, Any]) -> str:
    try:
        from benchmark import platform_guidance

        return platform_guidance(config)
    except Exception:
        return "通用网文读者：开篇卖点清晰、章节推进稳定、承诺及时兑现、重复模式不过度。"


def build_chapter_aux_cache(paths: Paths, conn: Any, config: dict[str, Any], chapter_num: int) -> dict[str, Any]:
    """Pre-build the per-chapter auxiliary context used by review_chapter.

    Called once before parallel candidate reviews so each thread reuses the
    same pre-fetched DB/file results instead of re-querying N times.
    """
    cache: dict[str, Any] = {}
    try:
        cache["writing_memory"] = writing_memory_context(paths, conn, config)
    except Exception:
        pass
    try:
        cache["rhythm_diagnostics"] = rhythm_diagnostics(conn, config)
    except Exception:
        pass
    try:
        cache["recent_quality_feedback"] = recent_quality_feedback(paths)
    except Exception:
        pass
    return cache


_SETTING_RULE_MARKERS = (
    "铁律", "法则", "规则", "禁止", "不能", "不可", "无法", "限制", "边界",
    "上限", "代价", "只能", "必须", "受限", "体系", "等级", "能力", "副作用",
)


def _setting_rules_block(paths: Paths, config: dict[str, Any]) -> str:
    """Extract world-rule / power-system hard lines from bible.md so the reviewer
    can cross-check setting violations. Conservative: picks rule-marker lines,
    caps total length, returns '' if nothing rule-like is found."""
    try:
        text = read_text(paths.bible)
    except Exception:
        return ""
    if not text:
        return ""
    cap = int(config["novel"].get("setting_rules_chars", 1500))
    picked: list[str] = []
    total = 0
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 4 or line.startswith("#"):
            continue
        if any(m in line for m in _SETTING_RULE_MARKERS):
            line = line.lstrip("-*0123456789. 　")
            if line and line not in picked:
                picked.append(line)
                total += len(line)
                if total >= cap:
                    break
    if not picked:
        return ""
    body = "\n".join(f"- {p}" for p in picked)
    return (
        "## 世界观/力量体系铁律（交叉校验，违反记入 contract_violations，type=must_hold）\n"
        "以下是本书设定中的硬规则，主角能力与世界运行不得越界；若本章出现越界/违背，"
        "必须登记为 contract_violations。\n" + body
    )


def review_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    chapter: str,
    tail: str,
    cached_memory: str | None = None,
    chapter_aux_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if chapter_aux_cache is not None and "writing_memory" in chapter_aux_cache:
        mem = chapter_aux_cache["writing_memory"]
    else:
        mem = cached_memory or writing_memory_context(paths, conn, config)
    silence_threshold = int(config["novel"].get("thread_silence_threshold", 10))
    silent_threads = get_silent_threads(conn, chapter_num, silence_threshold=silence_threshold)
    preset = str(config["novel"].get("style_preset", "history"))
    preset_hint = {
        "xuanhuan_shuang": "本作为穿越爽文：payoff 维度应额外考量本章是否有明确的爽点兑现（兑现/打脸/翻盘/掌权），节奏是否够紧；但爽点须有铺垫与代价，无脑碾压应扣分。若下方 Rhythm Diagnostics 报告了爽点拖欠（chapters_since_payoff >= payoff_max_gap）而本章仍未给出兑现类 payoff，额外 -0.5。",
        "history": "本作为历史厚重题材：重视制度细节、政治博弈的真实约束与因果链的严谨。",
        "system_stream": "本作为系统流：payoff 维度应额外考量本章是否有可见的系统反馈（面板/任务/奖励/数值升级/解锁），成长是否有节奏感与成就感；同时审查系统能力是否有代价与限制，是否出现无脑刷数值或金手指降智解题，若有应扣分。若本章完全没有任何系统侧反馈，hook 与 payoff 维度各 -0.5。",
        "urban_ability": "本作为都市异能/重生题材：payoff 维度应额外考量本章是否有打脸/资源碾压/身份反差的爽点兑现，且打脸须有铺垫与对手的合理反应；若对手或配角降智捧哏、爽点凭空降临、缺乏代入感，应扣分。",
        "romance_female": "本作为女频言情/宠文：核心是情绪张力与关系弧推进（拉近/误会/和解/甜虐节奏）。审查男女主对手戏是否有潜台词与化学反应，情绪是否由具体事件支撑而非悬浮；若出现工具人配角、情绪空转或关系毫无推进，应扣分。",
        "wanzu_xuanhuan": "本作为现代玄幻/万族争锋：审查境界/战力体系是否清晰可预期，斗法是否有画面感与天骄争锋的张力；力量解题须正比于此前规则铺垫（Sanderson 第一/二定律），凭空开挂或体系自相矛盾应扣分。",
        "suspense": "本作为悬疑/心理惊悚：核心是限制视角下的信息差与公平线索。审查（1）是否守住限制视角，有无写出视角人物当下不可能知道的真相/他人内心/未到场之事（视角越界应重扣）；（2）关键揭示是否有前文公平铺垫，反转能否在前文找到伏笔，凭空掉落的关键信息应扣分；（3）恐惧/诡异是否靠反常的具体细节与留白营造，而非'恐怖''惊悚'式贴标签形容词；（4）悬念是否只开不收、疑点无限堆积（本章应至少推进或收束1条旧悬念）。foreshadowing 与 reader 维度应重点据此评估。",
    }.get(preset, "")
    factcheck_enabled = bool(config["novel"].get("factcheck_enabled", True))
    if factcheck_enabled:
        facts_block = established_facts_for_chapter(
            conn,
            plan,
            chapter_num,
            budget_chars=int(config["novel"].get("factcheck_facts_chars", 3000)),
            promise_grace=int(config["novel"].get("reader_promise_overdue_grace", 15)),
        )
    else:
        facts_block = "None"
    contract_text_block = contract_block(paths, config) or "None"
    # P2-9: setting/power-system hard rules mined from bible.md, appended to the
    # contract so the reviewer cross-checks world-rule violations (ability over-
    # reach, broken铁律) the same way it checks the explicit contract. Reuses the
    # existing contract_violations channel (no new gate).
    if bool(config["novel"].get("setting_rules_check_enabled", True)):
        try:
            srules = _setting_rules_block(paths, config)
            if srules:
                base = "" if contract_text_block == "None" else contract_text_block
                contract_text_block = (base + "\n\n" if base else "") + srules
        except Exception:
            pass
    # Character voice baseline: cross-chapter stance/voice consistency check.
    # Enabled by default for the 爽文 preset; long novel opts in via config to
    # avoid false positives until the signal is validated.
    voice_check_default = preset in {"xuanhuan_shuang", "romance_female", "urban_ability"}
    voice_check = bool(config["novel"].get("character_voice_check", voice_check_default))
    voice_block = "None"
    if voice_check:
        try:
            focus = [str(c) for c in (plan.get("character_focus") or []) if c]
            notes = get_character_voice_notes(conn, focus, limit=6)
            if notes:
                voice_block = json.dumps(notes, ensure_ascii=False, indent=2)
        except Exception:
            voice_block = "None"
    opening_chapters = int(config["novel"].get("opening_chapters", 3))
    opening_strict = bool(config["novel"].get("opening_review_strict", True))
    opening_block = ""
    if opening_strict and chapter_num <= opening_chapters:
        opening_block = (
            "## 开篇专项评审（黄金三章，弃书率最高）\n"
            "本章属于开篇前几章，请额外按以下要点严格评审，并把不足写入 problems / writer_directives_for_next_chapter：\n"
            "- 钩子是否够快够强：核心冲突或悬念是否在开篇极短篇幅内抛出，而非缓慢铺垫。\n"
            "- 金手指/主角核心反差是否已经亮相或强烈预示，读者能否立刻感知本书卖点。\n"
            "- 代入感：主角目标、处境、情绪是否清晰，读者是否有理由继续读。\n"
            "- 信息密度：是否在抓人的同时高效给出信息，而非堆砌世界观设定拖慢节奏。\n"
            "- 若钩子偏弱、金手指迟迟不亮相、或开篇大段铺设定，hook_strength 应明显压低并在 problems 指出。\n\n"
        )
    _review_chars = int(config.get("novel", {}).get("review_chapter_chars", 16000))
    _rhythm = chapter_aux_cache.get("rhythm_diagnostics") if chapter_aux_cache else None
    if _rhythm is None:
        _rhythm = rhythm_diagnostics(conn, config)
    _quality_fb = chapter_aux_cache.get("recent_quality_feedback") if chapter_aux_cache else None
    if _quality_fb is None:
        _quality_fb = recent_quality_feedback(paths)

    # Deterministic entity-drift shortlist. The reviewer's hallucinated_entities
    # field is otherwise LLM-only; this surfaces proper-noun surface forms that
    # appear in THIS chapter but never in prior indexed chapters, so the reviewer
    # has a concrete list to verify against established facts. A legitimately new
    # character will also appear here, so it is presented as "suspects to check",
    # not a penalty.
    entity_suspect_block = "None"
    if bool(config["novel"].get("entity_drift_check", True)) and chapter_num > int(
        config["novel"].get("entity_drift_warmup", 3)
    ):
        try:
            from retrieval import candidate_new_entities

            suspects = candidate_new_entities(
                paths,
                chapter,
                limit=int(config["novel"].get("entity_drift_limit", 12)),
            )
            if suspects:
                entity_suspect_block = json.dumps(suspects, ensure_ascii=False)
        except Exception:
            entity_suspect_block = "None"

    # Reviewer de-contamination: the shared cacheable_prefix carries the CURRENT
    # voice.md, which refresh_voice_anchors mutates from recent (possibly drifted)
    # prose — so the main reviewer can normalize the very drift it should catch.
    # Inject the FROZEN voice_baseline.md as the authoritative anchor and tell the
    # reviewer it overrides any voice reference in the cached prefix. Falls back to
    # current voice.md before the baseline is first captured. Gated by
    # review_use_frozen_voice (default true).
    frozen_voice_block = ""
    if bool(config["novel"].get("review_use_frozen_voice", True)):
        try:
            baseline_path = paths.voice.with_name("voice_baseline.md")
            bl = read_text(baseline_path) if baseline_path.exists() else read_text(paths.voice)
            bl = (bl or "").strip()
            if bl:
                frozen_voice_block = (
                    "## 冻结文风基线（权威——优先级高于上文缓存中的任何声音参照）\n"
                    "下面是本书最初确立的健康文风基线。请以它为准绳评判本章是否发生文体塌缩/漂移，"
                    "不要把近期正文已漂移的特征当作'本书风格'而放行。\n"
                    f"{bl[:4000]}\n\n"
                )
        except Exception:
            frozen_voice_block = ""

    user = f"""## 风格预设：{preset}
{preset_hint}

{frozen_voice_block}{opening_block}## 平台/读者画像
{_platform_guidance(config)}

## 记忆
{mem}

## 既定事实（不得违背——任何冲突记入 "contradictions"）
{facts_block}

## 创作契约（作者钉死的硬约束——任何违反记入 "contract_violations"）
{contract_text_block}

## 人物声音基线（跨章立场/口吻；冲突记入 "character_voice_drift"）
{voice_block}

## 疑似新出现的专有名词（确定性检索：以下名称在本章出现、但此前章节几乎从未出现。请逐一核对：是本章合理新引入的人物/地点/物品，还是与既定事实矛盾的幻觉实体？属于后者记入 "hallucinated_entities"）
{entity_suspect_block}

## 上章结尾
{tail[-1500:]}

## 近期质量反馈JSON
{json.dumps(_quality_fb, ensure_ascii=False, indent=2)}

## 沉默伏线JSON（沉默 >{silence_threshold} 章；核查本章是否推进了其中任何一条，或有充分理由跳过）
{json.dumps(silent_threads, ensure_ascii=False, indent=2) if silent_threads else "None"}

## 节奏诊断JSON（留意 chapters_since_payoff 与 payoff_max_gap 以判断爽点拖欠）
{json.dumps(_rhythm, ensure_ascii=False, indent=2)}

## 选定大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 章节正文
{chapter[:_review_chars]}"""
    # ②(b) diversify：评审温度可调（默认 0.2，低温=更挑剔，行为不变）。真正的"生成≠评审"
    # 多样性来自把 api.review_base_url/review_model 设成与写作不同的模型——评审就不会与写手
    # "同流合污"（试跑反复出现 style-audit MISMATCH=评审自报文风失真）。见 config_template 注释。
    _review_temp = float(config["novel"].get("review_temperature", 0.2))
    raw = call_llm(
        client, paths, config, REVIEW_SYSTEM, json_prompt(user),
        max_tokens=32000, temperature=_review_temp, cacheable_prefix=cacheable_prefix(paths, config),
        tag="review",
    )
    report = load_json_with_repair(
        client,
        paths,
        config,
        raw,
        fallback={
            "score": 5,
            "accepted": False,
            "problems": ["JSON parsing failed; conservative review fallback used."],
            "fixes": [],
            "continuity_risks": [],
            "rhythm_risks": [],
            "reader_fatigue_risks": [],
            "hook_strength": 6,
            "readthrough_score": 5,
            "hook_score": 6,
            "payoff_score": 5,
            "novelty_score": 5,
            "prose_score": 5,
            "continuity_score": 5,
            "aesthetic_score": 6,
            "style_audit": {"em_dash_per_kchar": 0.0, "fragment_line_ratio": 0.0, "has_full_dialogue": True},
            "contradictions": [],
            "contract_violations": [],
            "hallucinated_entities": [],
            "character_voice_drift": [],
            "writer_directives_for_next_chapter": [],
        },
    )
    report["score"] = safe_score(report.get("score", 0))
    report["aesthetic_score"] = safe_score(report.get("aesthetic_score", report["score"]))
    report["hook_strength"] = safe_score(report.get("hook_strength", 0))
    report["readthrough_score"] = safe_score(report.get("readthrough_score", report.get("reader_score", report["score"])))
    report["hook_score"] = safe_score(report.get("hook_score", report.get("hook_strength", report["score"])))
    report["payoff_score"] = safe_score(report.get("payoff_score", report["score"]))
    report["novelty_score"] = safe_score(report.get("novelty_score", report.get("novelty", report["score"])))
    report["prose_score"] = safe_score(report.get("prose_score", report.get("aesthetic_score", report["score"])))
    report["continuity_score"] = safe_score(report.get("continuity_score", report["score"]))
    report["emotional_impact"] = safe_score(report.get("emotional_impact", report.get("prose_score", report["score"])))
    report.setdefault("contradictions", [])
    report.setdefault("contract_violations", [])
    report.setdefault("hallucinated_entities", [])
    report.setdefault("character_voice_drift", [])

    # ------------------------------------------------------------------
    # Unified score adjustment. All penalties/caps are accumulated here and
    # applied in ONE final clamp, so the order of MARKET-floor vs style-health
    # vs other penalties no longer changes the result. Previously MARKET cap
    # ran before the style penalty was subtracted, so a low-readthrough chapter
    # and a style-collapsed chapter took different effective hits. Now:
    #   final = clamp( min(raw_score, *caps) - sum(penalties) , 1.0, 10.0 )
    # ------------------------------------------------------------------
    raw_score = report["score"]
    caps: list[float] = [10.0]
    penalties: float = 0.0

    # DIMENSION DE-INFLATION: hook_strength/readthrough drift to a permanent
    # ceiling (verify_v8: hook=10.0 in 41/50 chapters, readthrough avg 9.6), so
    # they lose all discrimination and can no longer signal a weak chapter. We
    # (a) feed a discounted value into the MARKET floor + milestone hook check so
    # a saturated dim can't silently prop up the ceiling, and (b) when the sheet
    # is saturated yet a deterministic gate disagrees (style penalty / fossils),
    # add a small composite penalty — same philosophy as style_audit_mismatch:
    # don't trust an all-max scoresheet the objective metrics contradict. The
    # stored dimension scores stay untouched (observation truth).
    mf_readthrough = report["readthrough_score"]
    mf_hook = report["hook_score"]
    dim_saturated_count = 0  # used again before the final clamp (contradiction penalty)
    if bool(config["novel"].get("dimension_inflation_enabled", True)):
        try:
            from store import recent_dimension_scores
            win = int(config["novel"].get("dim_inflation_window", 8))
            sat = float(config["novel"].get("dim_inflation_saturate", 9.3))
            disc = float(config["novel"].get("dim_inflation_discount", 0.5))
            for dim, cur in (("hook_score", mf_hook), ("readthrough_score", mf_readthrough)):
                hist = recent_dimension_scores(conn, dim, win, before_chapter=chapter_num)
                if len(hist) >= max(3, win // 2):
                    avg = sum(hist) / len(hist)
                    if avg >= sat and cur >= avg - 0.1:
                        dim_saturated_count += 1
                        adj = max(1.0, cur - disc)
                        if dim == "hook_score":
                            mf_hook = adj
                        else:
                            mf_readthrough = adj
                        report.setdefault("calibration", []).append(
                            f"{dim} 去通胀 {cur:.1f}→{adj:.1f}（MARKET/里程碑判定用）："
                            f"近{len(hist)}章均值{avg:.1f}≥{sat} 已饱和失去区分度")
        except Exception:
            pass

    # MARKET cap: a strong-prose chapter with weak readthrough/payoff/novelty
    # must not score like a hit. Caps the ceiling rather than subtracting.
    market_floor = min(
        mf_readthrough,
        mf_hook,
        report["payoff_score"],
        report["novelty_score"],
    )
    if market_floor < 6.5:
        caps.append(8.0)
        report.setdefault("problems", []).append(
            "MARKET: 追读/兑现/新鲜度存在短板，综合分按爆款维度上限压低。"
        )

    # SERIALIZATION MILESTONE: at key chapter positions (every N chapters),
    # raise the hook strength bar. These are reader retention inflection points
    # (free-to-paid, weekly milestone). A weak hook at a milestone caps the score.
    serial_every = int(config["novel"].get("serial_milestone_every", 10))
    serial_hook_boost = float(config["novel"].get("serial_milestone_hook_boost", 1.0))
    serial_free_to_paid = int(config["novel"].get("serial_free_to_paid_chapter", 0))
    is_milestone = (serial_every > 0 and chapter_num > 0 and chapter_num % serial_every == 0)
    is_paywall = (serial_free_to_paid > 0 and chapter_num == serial_free_to_paid)
    if is_milestone or is_paywall:
        hook = mf_hook  # de-inflated value, so a chronically-saturated hook can't auto-pass
        min_hook = float(config["novel"].get("hook_strength_min", 6.0)) + serial_hook_boost
        if hook < min_hook:
            caps.append(8.0)
            label = "付费转化章" if is_paywall else f"连载里程碑（每{serial_every}章）"
            report.setdefault("problems", []).append(
                f"SERIAL: {label}，钩子强度{hook:.1f}低于里程碑要求{min_hook:.1f}。"
                f"关键节点必须有强力悬念/揭示/反转来留住读者。"
            )

    # Emotional impact floor: chapters that score well on plot/prose but have
    # zero emotional resonance shouldn't score like hits.
    ei_floor_enabled = bool(config["novel"].get("emotional_impact_floor_enabled", True))
    if ei_floor_enabled:
        ei = report.get("emotional_impact", 10.0)
        ei_min = float(config["novel"].get("emotional_impact_floor", 5.0))
        if ei < ei_min:
            caps.append(8.5)
            report.setdefault("problems", []).append(
                f"EMOTION: 情感冲击力{ei:.1f}低于地板{ei_min:.1f}——"
                f"本章缺少让读者产生真实情感反应的时刻。"
            )

    # RETENTION GATE (P2): the reader-panel excitement is the pipeline's only
    # KEY-DIMENSION HARD FLOOR: novelty/payoff are the two dimensions self-review
    # systematically over-passes — across suspense_v4 every chapter scored
    # novelty 7.0–8.0 (3 of 6 stuck at exactly 7.0) yet the composite still read
    # ~7.8, so "题材常见/金手指变体不够野" never drove a rewrite. A weak key
    # dimension must CAP the composite (not merely nudge it) so the chapter is
    # routed into the revise/replan loop instead of being force-accepted at 7.8.
    # Gated by key_dimension_floor_enabled; thresholds tunable per dimension.
    if bool(config["novel"].get("key_dimension_floor_enabled", True)):
        nov_floor = float(config["novel"].get("novelty_floor", 7.0))
        pay_floor = float(config["novel"].get("payoff_floor", 7.0))
        floor_cap = float(config["novel"].get("key_dimension_floor_cap", 7.5))
        nov = report["novelty_score"]
        pay = report["payoff_score"]
        if (nov and nov < nov_floor) or (pay and pay < pay_floor):
            caps.append(floor_cap)
            weak_dims = []
            if nov and nov < nov_floor:
                weak_dims.append(f"新鲜度{nov:.1f}<{nov_floor:.1f}")
            if pay and pay < pay_floor:
                weak_dims.append(f"兑现{pay:.1f}<{pay_floor:.1f}")
            report.setdefault("problems", []).append(
                f"KEY-DIM: 关键维度未达硬地板（{', '.join(weak_dims)}），"
                f"综合分封顶 {floor_cap:.1f}，必须重做场景而非修辞修补："
                f"提升金手指用法的新颖度与本章爽点的实质兑现。"
            )

    # Objective, non-LLM style-health gate. Self-review cannot detect that the
    # prose has collapsed into telegraphic em-dash fragments because the model's
    # own voice has drifted with it. Apply a deterministic penalty and feed the
    # fixes to the next chapter's writer + this chapter's revise loop.
    if REGISTRY.is_enabled("style_health", config):
        try:
            from quality import style_health

            # Trend term: feed the recent chapters' em-dash density (oldest→
            # newest) so a slow upward drift is penalized even below the
            # absolute warn threshold (the 0.94→4.15 monotonic-creep miss).
            em_history: list[float] = []
            tech_history: list[float] = []
            try:
                from store import recent_metrics

                trend_window = int(config["novel"].get("style_em_dash_trend_window", 5))
                rows = recent_metrics(conn, max(trend_window, 1))
                seq = []
                for r in rows:
                    try:
                        if int(r.get("chapter", 0)) >= chapter_num:
                            continue
                    except (TypeError, ValueError):
                        continue
                    v = r.get("em_dash_per_kchar")
                    if v is None:
                        continue
                    try:
                        tv = r.get("tech_per_kchar")
                        seq.append((int(r.get("chapter", 0)), float(v),
                                    float(tv) if tv is not None else None))
                    except (TypeError, ValueError):
                        continue
                seq.sort(key=lambda x: x[0])
                em_history = [v for _, v, _ in seq[-trend_window:]]
                tech_history = [t for _, _, t in seq[-trend_window:] if t is not None]
            except Exception:
                em_history = []
                tech_history = []

            sh = style_health(
                chapter, config,
                em_history=em_history or None,
                tech_history=tech_history or None,
            )
            penalty = REGISTRY.accumulate(report, sh, "style_health", REGISTRY.tag_prefix("style_health"))
            if penalty > 0:
                penalties += penalty
                log(paths, f"Style-health Ch{chapter_num} penalty={penalty} "
                    f"flags={sh.get('flags')} metrics={sh.get('metrics')}")
                # A hard collapse must not be accepted on quality grounds alone.
                if penalty >= float(config["novel"].get("style_penalty_block", 2.0)):
                    report["accepted"] = False
                    report.setdefault("problems", []).append(
                        "STYLE: prose-health collapse detected "
                        "(em-dash fragments / telegraphic lines / overwriting-instrument-report register)."
                    )

            if bool(config["novel"].get("prose_calibration_enabled", True)):
                cur_prose = report.get("prose_score")
                if cur_prose is not None:
                    if penalty == 0 and cur_prose < 6.0:
                        report.setdefault("calibration", []).append(
                            f"prose_score raised {cur_prose:.1f}→6.0: style_health clean")
                        report["prose_score"] = 6.0
                    elif penalty >= 1.0 and cur_prose > 7.5:
                        report.setdefault("calibration", []).append(
                            f"prose_score lowered {cur_prose:.1f}→7.5: style_health penalty={penalty:.1f}")
                        report["prose_score"] = 7.5

            # Cross-check: the LLM self-reported style_audit vs the deterministic
            # measurement. A large gap means the reviewer is mis-reporting (often
            # because its own voice has drifted with the prose). We don't re-penalize
            # the chapter for it, but we tag it so score inflation is visible and the
            # reviewer's other judgments can be trusted less downstream.
            try:
                audit = report.get("style_audit") or {}
                m = sh.get("metrics") or {}
                det_em = float(m.get("em_dash_per_kchar", 0.0) or 0.0)
                det_frag = float(m.get("fragment_line_ratio", 0.0) or 0.0)
                rep_em = float(audit.get("em_dash_per_kchar", det_em) or 0.0)
                rep_frag = float(audit.get("fragment_line_ratio", det_frag) or 0.0)
                em_tol = float(config["novel"].get("style_audit_em_tol", 3.0))
                frag_tol = float(config["novel"].get("style_audit_frag_tol", 0.15))
                mismatch = abs(det_em - rep_em) > em_tol or abs(det_frag - rep_frag) > frag_tol
                if mismatch:
                    report["style_audit_mismatch"] = {
                        "reported": {"em_dash_per_kchar": rep_em, "fragment_line_ratio": rep_frag},
                        "measured": {"em_dash_per_kchar": round(det_em, 2), "fragment_line_ratio": round(det_frag, 2)},
                    }
                    log(
                        paths,
                        f"Style-audit MISMATCH Ch{chapter_num}: reviewer reported "
                        f"em={rep_em}/frag={rep_frag} but measured em={det_em:.2f}/frag={det_frag:.2f} "
                        f"(reviewer self-report unreliable).",
                    )
                    mm_penalty = float(config["novel"].get("style_audit_mismatch_penalty", 0.5))
                    if mm_penalty > 0:
                        penalties += mm_penalty
                        report.setdefault("calibration", []).append(
                            f"mismatch penalty +{mm_penalty:.1f}: "
                            f"reviewer reported em={rep_em:.1f} but measured {det_em:.1f}")
            except Exception:
                pass
        except Exception as exc:
            log(paths, f"style_health check failed (non-fatal) Ch{chapter_num}: {exc}")

    # AI flavor gate: deterministic detection of AI-typical writing patterns
    # (clichés, metaphor spam, tell-not-show, degree-adverb inflation, summary
    # narration, paragraph monotony). Like style_health, the model's self-review
    # is blind to these because it generated them in the first place.
    if REGISTRY.is_enabled("ai_flavor_health", config):
        try:
            from quality import ai_flavor_health
            af = ai_flavor_health(chapter, config)
            af_pen = REGISTRY.accumulate(report, af, "ai_flavor_health", REGISTRY.tag_prefix("ai_flavor_health"))
            if af_pen > 0:
                penalties += af_pen
                log(paths, f"AI-flavor Ch{chapter_num} penalty={af_pen} flags={af.get('flags')} metrics={af.get('metrics')}")
        except Exception as exc:
            log(paths, f"ai_flavor_health failed (non-fatal) Ch{chapter_num}: {exc}")

    # 黄金三句开篇闸门：前 opening_chapters 章，确定性拦截"景物/时段/设定铺垫"开场。
    # LLM 自评对文学性氛围开场打分偏高，这是它抓不到的下沉/追读病灶的确定性兜底。
    if REGISTRY.is_enabled("opening_hook_gate", config):
        try:
            from quality import opening_hook_gate
            og = opening_hook_gate(chapter, chapter_num, config)
            if og.get("flags"):
                report["opening_hook_gate"] = og
                opening_pen = float(og.get("penalty", 0.0))
                if opening_pen > 0:
                    penalties += opening_pen
                    report.setdefault("rhythm_risks", []).append(
                        f"opening:{','.join(og['flags'])}")
                wd = report.setdefault("writer_directives_for_next_chapter", [])
                for d in og.get("directives", []):
                    if d not in wd:
                        wd.append(d)
                log(paths, f"Opening gate Ch{chapter_num}: penalty={opening_pen} flags={og['flags']}")
                if og.get("block"):
                    report["accepted"] = False
                    report.setdefault("problems", []).append(
                        "OPENING: 黄金三句开篇未达标（开局铺垫而非进行中的危机）。")
        except Exception as exc:
            log(paths, f"opening_hook_gate failed (non-fatal) Ch{chapter_num}: {exc}")

    # Prose texture: quantitative vs poetic balance — inject variation directives
    if REGISTRY.is_enabled("prose_texture", config):
        try:
            from quality import prose_texture
            pt = prose_texture(chapter, config)
            pt_pen = REGISTRY.accumulate(report, pt, "prose_texture", REGISTRY.tag_prefix("prose_texture"))
            if pt.get("directives"):
                log(paths, f"Prose-texture Ch{chapter_num}: balance={pt['balance']} metrics={pt['metrics']}")
            if pt_pen > 0:
                penalties += pt_pen
                log(paths, f"Prose-texture Ch{chapter_num} penalty={pt_pen} "
                    f"(over_poetic, poetic_density={pt['metrics'].get('poetic_density')})")
        except Exception as exc:
            log(paths, f"prose_texture failed (non-fatal) Ch{chapter_num}: {exc}")

    # Dialogue ratio gate: deterministic check that chapters contain enough
    # character dialogue (vs pure narration/internal monologue). Low dialogue
    # kills pacing and reader engagement in 都市/言情/悬疑 genres.
    if REGISTRY.is_enabled("dialogue_health", config):
        try:
            from quality import dialogue_health
            dh = dialogue_health(chapter, config)
            dh_pen = REGISTRY.accumulate(report, dh, "dialogue_health", REGISTRY.tag_prefix("dialogue_health"))
            if dh_pen > 0:
                penalties += dh_pen
                log(paths, f"Dialogue-health Ch{chapter_num} penalty={dh_pen} "
                    f"ratio={dh.get('metrics', {}).get('dialogue_char_ratio', 0):.2%}")
        except Exception as exc:
            log(paths, f"dialogue_health failed (non-fatal) Ch{chapter_num}: {exc}")

    if REGISTRY.is_enabled("paragraph_shape_health", config):
        try:
            from quality import paragraph_shape_health
            ps = paragraph_shape_health(chapter, config)
            ps_pen = REGISTRY.accumulate(report, ps, "paragraph_shape", REGISTRY.tag_prefix("paragraph_shape_health"))
            if ps_pen > 0:
                penalties += ps_pen
                log(paths, f"Paragraph-shape Ch{chapter_num} penalty={ps_pen} "
                    f"cv={ps.get('metrics', {}).get('paragraph_length_cv', '-')} "
                    f"hedge={ps.get('metrics', {}).get('hedge_per_kchar', '-')}/k")
        except Exception as exc:
            log(paths, f"paragraph_shape failed (non-fatal) Ch{chapter_num}: {exc}")

    if REGISTRY.is_enabled("dialogue_pingpong", config):
        try:
            from quality import dialogue_pingpong
            dp = dialogue_pingpong(chapter, config)
            dp_pen = REGISTRY.accumulate(report, dp, "dialogue_pingpong", REGISTRY.tag_prefix("dialogue_pingpong"))
            if dp_pen > 0:
                penalties += dp_pen
                log(paths, f"Dialogue-pingpong Ch{chapter_num} penalty={dp_pen} "
                    f"qa_ratio={dp.get('metrics', {}).get('qa_ratio', '-')}")
        except Exception as exc:
            log(paths, f"dialogue_pingpong failed (non-fatal) Ch{chapter_num}: {exc}")

    if REGISTRY.is_enabled("chapter_ending_quality", config):
        try:
            from quality import chapter_ending_quality
            ceq = chapter_ending_quality(chapter, config)
            ceq_pen = REGISTRY.accumulate(report, ceq, "chapter_ending_quality", REGISTRY.tag_prefix("chapter_ending_quality"))
            if ceq_pen > 0:
                penalties += ceq_pen
                log(paths, f"Chapter-ending Ch{chapter_num} penalty={ceq_pen} "
                    f"markers={ceq.get('metrics', {}).get('ending_summary_markers', '-')}")
        except Exception as exc:
            log(paths, f"chapter_ending_quality failed (non-fatal) Ch{chapter_num}: {exc}")

    if REGISTRY.is_enabled("long_span_fatigue", config):
        try:
            from quality import long_span_fatigue
            lsf = long_span_fatigue(conn, chapter_num, config)
            lsf_pen = REGISTRY.accumulate(report, lsf, "long_span_fatigue", REGISTRY.tag_prefix("long_span_fatigue"))
            if lsf_pen > 0:
                penalties += lsf_pen
                log(paths, f"Long-span-fatigue Ch{chapter_num} penalty={lsf_pen} flags={lsf.get('flags', [])}")
        except Exception as exc:
            log(paths, f"long_span_fatigue failed (non-fatal) Ch{chapter_num}: {exc}")

    # Chapter length penalty: deterministic check for mid-book shrinkage.
    try:
        _ch_min = int(config["novel"].get("chapter_min_chars", 2800))
        if _ch_min > 0:
            _ch_len = len(chapter or "")
            if _ch_len < _ch_min:
                _len_pen = min(float(config["novel"].get("chapter_length_penalty_cap", 1.0)),
                               (_ch_min - _ch_len) / 500.0)
                if _len_pen > 0:
                    penalties += _len_pen
                    report.setdefault("rhythm_risks", []).append("short_chapter")
                    report.setdefault("writer_directives_for_next_chapter", []).append(
                        "本章仅%d字，低于下限%d字。下章必须充分展开场景细节和对话。" % (_ch_len, _ch_min))
                    log(paths, f"Chapter length Ch{chapter_num}: {_ch_len}<{_ch_min} penalty={_len_pen:.1f}")
    except Exception:
        pass

    # Book-wide micro-fossil scan: the sliding-window cross_chapter_repetition
    # misses 4-6 char action stubs that recur across the WHOLE book (e.g.
    # '陆知白用左手' in 42/50 chapters). Runs every book_fossil_every chapters,
    # caches the avoid-list to logs/book_fossils.json so the writer can inject it
    # on EVERY subsequent chapter (not just the one chapter after this review).
    if REGISTRY.is_enabled("book_wide_fossils", config):
        try:
            every = max(1, int(config["novel"].get("book_fossil_every", 5)))
            if chapter_num >= int(config["novel"].get("book_fossil_min_chapters", 6)) \
                    and chapter_num % every == 0:
                from quality import book_wide_fossils
                import json as _json
                texts: dict[int, str] = {}
                for num in range(1, chapter_num + 1):
                    p = chapter_path(paths, num)
                    if p.exists():
                        texts[num] = read_text(p)
                _wl: set[str] = set()
                _wl_str = str(config["novel"].get("book_fossil_whitelist", "")).strip()
                for _w in _wl_str.split(","):
                    _w = _w.strip()
                    if len(_w) >= 2:
                        _wl.add(_w)
                try:
                    import re as _re
                    _prompt_text = read_text(paths.prompt_file)
                    for _m in _re.findall(r'[《「]([^》」]{2,10})[》」]', _prompt_text):
                        _wl.add(_m)
                except Exception:
                    pass
                bf = book_wide_fossils(texts, config, whitelist=_wl)
                report["book_fossils"] = bf
                if bf.get("phrases"):
                    try:
                        write_text(
                            paths.logs_dir / "book_fossils.json",
                            _json.dumps(bf, ensure_ascii=False, indent=2),
                        )
                    except Exception:
                        pass
                    struct_count = int(config["novel"].get("book_fossil_struct_count", 10))
                    log(paths, f"Book fossils Ch{chapter_num}: {len(bf['phrases'])} phrases "
                        f"(threshold={bf['metrics'].get('threshold_chapters')}ch) "
                        f"examples={bf['phrases'][:5]}")
                    if bf.get("directives"):
                        wd = report.setdefault("writer_directives_for_next_chapter", [])
                        for d in bf["directives"]:
                            if d not in wd:
                                wd.append(d)
                    # Many entrenched book-wide fossils = structural monotony, not a
                    # one-chapter tic. Surface as a gate-reject-style marker so
                    # pipeline._classify_replan_failure routes to STRUCTURAL replan.
                    if len(bf["phrases"]) >= struct_count:
                        report.setdefault("gate_rejects", []).append({
                            "gate": "book_wide_fossils",
                            "count": len(bf["phrases"]),
                            "phrases": bf["phrases"][:8],
                        })
                    # A SINGLE phrase recurring in >= hard_ratio of the book (e.g.
                    # tangshuting「老市场街七号」65/199≈33%) is a structural fossil in
                    # its own right even when the DISTINCT-phrase count stays below
                    # struct_count. Route it to STRUCTURAL replan via the same gate.
                    hard = bf.get("hard_fossils") or []
                    if hard:
                        report.setdefault("gate_rejects", []).append({
                            "gate": "book_wide_fossils_ratio",
                            "ratio_threshold": float(config["novel"].get("book_fossil_hard_ratio", 0.20)),
                            "phrases": [f["phrase"] for f in hard[:8]],
                            "fracs": [f["frac"] for f in hard[:8]],
                        })
        except Exception as exc:
            log(paths, f"book_wide_fossils check failed (non-fatal) Ch{chapter_num}: {exc}")

    # Descriptor-frequency gate: catch short (3-6 char) overused descriptors
    # that evade the clause min_len and ngram window. Runs on the same schedule
    # as book_wide_fossils to share the chapter-loading cost.
    if REGISTRY.is_enabled("descriptor_frequency", config):
        try:
            d_every = max(1, int(config["novel"].get("descriptor_freq_every", 5)))
            if chapter_num >= int(config["novel"].get("descriptor_freq_min_spread", 15)) \
                    and chapter_num % d_every == 0:
                from quality import descriptor_frequency
                import json as _json
                texts: dict[int, str] = {}
                for num in range(1, chapter_num + 1):
                    p = chapter_path(paths, num)
                    if p.exists():
                        texts[num] = read_text(p)
                df = descriptor_frequency(texts, config)
                report["descriptor_frequency"] = df
                if df.get("flagged"):
                    try:
                        write_text(
                            paths.logs_dir / "descriptor_freq.json",
                            _json.dumps(df, ensure_ascii=False, indent=2),
                        )
                    except Exception:
                        pass
                    log(paths, f"Descriptor freq Ch{chapter_num}: "
                        f"{df['metrics'].get('descriptor_flagged_count')} flagged "
                        f"examples={[f['phrase'] for f in df['flagged'][:5]]}")
                    if df.get("directives"):
                        wd = report.setdefault("writer_directives_for_next_chapter", [])
                        for d in df["directives"]:
                            if d not in wd:
                                wd.append(d)
                    df_penalty = float(df.get("penalty", 0.0))
                    if df_penalty > 0:
                        penalties += df_penalty
                    if str(df.get("level", "")) == "reject":
                        report.setdefault("gate_rejects", []).append({
                            "gate": "descriptor_frequency",
                            "phrases": [f["phrase"] for f in df["flagged"][:8]],
                        })
        except Exception as exc:
            log(paths, f"descriptor_frequency check failed (non-fatal) Ch{chapter_num}: {exc}")

    # Genre-adherence gate: deterministic keyword check that chapter content
    # matches the declared style_preset.
    if REGISTRY.is_enabled("genre_adherence", config):
        try:
            from quality import genre_adherence
            recent_genre_scores: list[float] = []
            try:
                import store as _store
                conn = _store.get_connection(str(paths.db))
                if conn and not isinstance(conn, _store.JsonStoryStore):
                    cursor = conn.execute(
                        "SELECT genre_score FROM chapter_metrics "
                        "WHERE chapter < ? AND genre_score IS NOT NULL "
                        "ORDER BY chapter DESC LIMIT ?",
                        (chapter_num, int(config["novel"].get("genre_adherence_window", 5))),
                    )
                    recent_genre_scores = [row[0] for row in cursor.fetchall()][::-1]
            except Exception:
                pass
            ga = genre_adherence(chapter, recent_genre_scores, config)
            ga_pen = REGISTRY.accumulate(report, ga, "genre_adherence", REGISTRY.tag_prefix("genre_adherence"))
            if ga_pen > 0:
                penalties += ga_pen
                log(paths, f"Genre adherence Ch{chapter_num} score={ga['genre_score']} "
                    f"penalty={ga_pen} flags={ga.get('flags')}")
            if str(ga.get("level", "")) == "reject":
                report["accepted"] = False
                report.setdefault("gate_rejects", []).append({
                    "gate": "genre_adherence",
                    "genre_score": ga["genre_score"],
                    "streak": ga["metrics"].get("low_streak", 0),
                    "directives": ga.get("directives", []),
                })
                report.setdefault("problems", []).append(
                    "GATE: 体裁严重偏移（确定性检测），"
                    "章节内容与声明体裁不符，必须重新规划。"
                )
        except Exception as exc:
            log(paths, f"genre_adherence failed (non-fatal) Ch{chapter_num}: {exc}")

    # Multi-lead character-service scan: in multi-男主/multi-lead books a secondary
    # lead can go silent for dozens of chapters while the contract's "非官配需完整
    # 成长线" rule has no executable metric. Every character_service_every chapters,
    # measure each principal-cast name's appearance rate over the recent window and
    # persist an under-served alert; planning.create_plan reads the latest alert and
    # injects a soft high-light directive (WARN only — never blocks/replans).
    if bool(config["novel"].get("character_service_enabled", True)):
        try:
            every = max(1, int(config["novel"].get("character_service_every", 5)))
            window = int(config["novel"].get("character_service_window", 15))
            if chapter_num >= every and chapter_num % every == 0:
                from quality import character_names_from_md, character_appearance_rate
                names = character_names_from_md(read_text(paths.characters)) if paths.characters.exists() else []
                if names:
                    lo = max(1, chapter_num - window + 1)
                    texts_cs: dict[int, str] = {}
                    for num in range(lo, chapter_num + 1):
                        p = chapter_path(paths, num)
                        if p.exists():
                            texts_cs[num] = read_text(p)
                    car = character_appearance_rate(
                        names, texts_cs, window=window,
                        floor=float(config["novel"].get("character_service_rate_floor", 0.15)),
                    )
                    report["character_service"] = car
                    if car.get("under_served"):
                        log(paths, f"Character service Ch{chapter_num}: under-served="
                            f"{[u['name'] for u in car['under_served']]}")
                        db_event(conn, chapter_num, "character_service_alert", car)
        except Exception as exc:
            log(paths, f"character_service check failed (non-fatal) Ch{chapter_num}: {exc}")

    # Chapter length band: a guard beyond save_chapter's 500-char hard floor.
    # 番茄短章高频钩子 = 2.5-3k 字/章；超长章把"每章一钩子"稀释成"每5章一钩子"。
    # Always emits a next-chapter directive (advisory); adds a SCORE PENALTY only
    # when length_band_penalty_enabled is on (existing novels keep advisory-only).
    try:
        from quality import length_band_check
        lb = length_band_check(chapter, config)
        report["length_band"] = lb
        for f in lb.get("flags", []):
            report.setdefault("style_flags", []).append(f)
        wd = report.setdefault("writer_directives_for_next_chapter", [])
        for d in lb.get("directives", []):
            if d not in wd:
                wd.append(d)
        lb_pen = float(lb.get("penalty", 0.0))
        if lb_pen > 0:
            penalties += lb_pen
            log(paths, f"Length band Ch{chapter_num}: chars={lb.get('chars')} penalty={lb_pen} flags={lb.get('flags')}")
        elif lb.get("flags"):
            log(paths, f"Length band Ch{chapter_num}: chars={lb.get('chars')} flags={lb.get('flags')} (advisory)")
        if lb.get("block"):
            report["accepted"] = False
            report.setdefault("problems", []).append(
                f"LENGTH: chapter grossly out of band ({lb.get('chars')} chars).")
    except Exception as exc:
        log(paths, f"length_band_check failed (non-fatal) Ch{chapter_num}: {exc}")

    # P2: 爽点 density — inject a payoff directive when the recent window has gone
    # too long without a strong reader payoff.
    if REGISTRY.is_enabled("payoff_beat_density", config):
        try:
            from quality import payoff_beat_density
            from store import recent_metrics as _rm
            rows = _rm(conn, 6)  # newest-first
            recent_ptypes = [str(r.get("payoff_type", "")) for r in rows if r.get("payoff_type")]
            pd = payoff_beat_density(chapter, recent_ptypes, config)
            REGISTRY.accumulate(report, pd, "payoff_density", REGISTRY.tag_prefix("payoff_beat_density"))
            if pd.get("directives"):
                log(paths, f"Payoff density Ch{chapter_num}: {pd['metrics']}")
        except Exception as exc:
            log(paths, f"payoff_beat_density failed (non-fatal) Ch{chapter_num}: {exc}")

    # 连续平路闸门：连续 N 章无强爽点且情绪冲击偏低 → 扣分 + 强制本章给中爽点/情绪高峰。
    if REGISTRY.is_enabled("flat_chapter_streak", config):
        try:
            from quality import flat_chapter_streak
            from store import recent_metrics as _rm2
            rows2 = _rm2(conn, int(config["novel"].get("flat_chapters_max_consecutive", 3)) + 2)
            fs = flat_chapter_streak(rows2, config)
            # Custom rhythm_risk format (flat_streak:{N}) — cannot use accumulate for flags.
            report["flat_streak"] = fs
            fs_pen = float(fs.get("penalty", 0.0))
            if fs_pen > 0:
                penalties += fs_pen
                report.setdefault("rhythm_risks", []).append(f"flat_streak:{fs.get('streak')}")
            wd = report.setdefault("writer_directives_for_next_chapter", [])
            for d in fs.get("directives", []):
                if d not in wd:
                    wd.append(d)
            if fs.get("flags"):
                log(paths, f"Flat streak Ch{chapter_num}: streak={fs.get('streak')} penalty={fs_pen}")
        except Exception as exc:
            log(paths, f"flat_chapter_streak failed (non-fatal) Ch{chapter_num}: {exc}")

    # Gap-9: shareable golden-line (可截图金句/传播性). Advisory — no penalty;
    # nudges the next chapter to plant a传播性金句 when this one had none.
    if REGISTRY.is_enabled("shareable_line", config):
        try:
            from quality import shareable_line
            sl = shareable_line(chapter, config)
            REGISTRY.accumulate(report, sl, "shareable_line", REGISTRY.tag_prefix("shareable_line"))
            if sl.get("directives"):
                log(paths, f"Shareable line Ch{chapter_num}: none found (best_score={sl['metrics'].get('best_score')})")
        except Exception as exc:
            log(paths, f"shareable_line failed (non-fatal) Ch{chapter_num}: {exc}")

    # P2: information density — flag a near-pure transition chapter (no payoff,
    # no new info, no realized beats) and demand推进 next chapter.
    if REGISTRY.is_enabled("information_density", config):
        try:
            from quality import information_density
            idr = information_density(chapter, plan, report, config)
            report["information_density"] = idr
            if idr.get("low_information"):
                report.setdefault("style_flags", []).append(
                    f"low_information_chapter({len(idr.get('signals', []))})")
                wd = report.setdefault("writer_directives_for_next_chapter", [])
                for d in idr.get("directives", []):
                    if d not in wd:
                        wd.append(d)
                log(paths, f"Info density Ch{chapter_num}: low_information "
                    f"signals={idr.get('signals')}")
        except Exception as exc:
            log(paths, f"information_density check failed (non-fatal) Ch{chapter_num}: {exc}")

    # P0-4: Structured constraint verification (required_constraints from arbitrate_plan)
    # Each constraint carries an id/type/constraint/check_method/target; verify each one
    # mechanically and report violations with concrete evidence.
    if bool(config["novel"].get("constraint_verification_enabled", True)):
        try:
            from checkpoint import load_checkpoint
            decision = load_checkpoint(paths, chapter_num, "plan_initial_attempt0_arbitration.json")
            if not decision:
                decision = load_checkpoint(paths, chapter_num, "plan_initial_selected.json")
            constraints = decision.get("required_constraints", []) if isinstance(decision, dict) else []

            if isinstance(constraints, list) and constraints:
                failed_constraints: list[dict[str, Any]] = []
                for c in constraints:
                    if not isinstance(c, dict):
                        continue

                    c_id = str(c.get("id", "")).strip()
                    c_type = str(c.get("type", "")).strip()
                    c_constraint = str(c.get("constraint", "")).strip()
                    c_method = str(c.get("check_method", "")).strip()
                    c_target = str(c.get("target", "")).strip()

                    if not c_id or not c_method or not c_target:
                        continue

                    passed = False
                    # Mechanical verification based on check_method
                    if c_method == "keyword":
                        # Split target by | for OR keywords
                        keywords = [kw.strip() for kw in c_target.split("|")]
                        passed = any(kw in chapter for kw in keywords if kw)
                    elif c_method in ("character_name", "location", "object"):
                        passed = c_target in chapter
                    elif c_method == "action":
                        # Action descriptors should appear as part of prose
                        passed = c_target in chapter or any(word in chapter for word in c_target.split() if len(word) >= 2)
                    elif c_method == "dialogue":
                        # Check if dialogue fragment appears (approximate)
                        passed = c_target in chapter
                    elif c_method == "logic":
                        # Logic checks require LLM; skip mechanical check, rely on LLM report
                        passed = True

                    if not passed:
                        failed_constraints.append({
                            "id": c_id,
                            "type": c_type,
                            "constraint": c_constraint,
                            "check_method": c_method,
                            "target": c_target,
                        })

                if failed_constraints:
                    # Add penalty and surface violations
                    constraint_penalty = min(
                        len(failed_constraints) * float(config["novel"].get("constraint_violation_penalty_each", 0.5)),
                        float(config["novel"].get("constraint_violation_penalty_cap", 2.0))
                    )
                    penalties += constraint_penalty

                    report["constraint_violations_structured"] = failed_constraints
                    report.setdefault("problems", []).append(
                        f"CONSTRAINT: {len(failed_constraints)} 条仲裁契约未兑现（{', '.join(c['id'] for c in failed_constraints[:3])}...）"
                    )

                    # Feed violations back to writer for next chapter
                    wd = report.setdefault("writer_directives_for_next_chapter", [])
                    for fc in failed_constraints[:3]:
                        directive = f"修复契约 {fc['id']}: {fc['constraint']}"
                        if directive not in wd:
                            wd.append(directive)

                    log(
                        paths,
                        f"Constraint violations Ch{chapter_num}: {len(failed_constraints)} failed, penalty={constraint_penalty:.2f}"
                    )

                    # Block accept if too many critical constraints failed
                    block_threshold = int(config["novel"].get("constraint_violation_block_count", 3))
                    if len(failed_constraints) >= block_threshold:
                        report["accepted"] = False
                        log(paths, f"Constraint block Ch{chapter_num}: {len(failed_constraints)} >= {block_threshold}")
        except Exception as exc:
            log(paths, f"Constraint verification failed (non-fatal) Ch{chapter_num}: {exc}")

    # Cross-chapter repetition: signature clauses/metaphors reused verbatim across
    # chapters ("像一颗心脏在缓慢地跳动", "不是暂时的，是永久的", "锁扣声每N秒一次")
    # become tics that self-review treats as motif. Deterministically penalize the
    # chapter and feed an avoid-list directive to the next writer prompt.
    if REGISTRY.is_enabled("cross_chapter_repetition", config):
        try:
            from quality import cross_chapter_repetition

            lookback = int(config["novel"].get("style_cross_repeat_lookback", 6))
            lookback_long = int(config["novel"].get("style_cross_repeat_lookback_long", 20))
            effective_lookback = max(lookback, lookback_long)
            all_prior: list[str] = []
            for num in range(max(1, chapter_num - effective_lookback), chapter_num):
                p = chapter_path(paths, num)
                if p.exists():
                    all_prior.append(read_text(p))
            prior_texts: list[str] = all_prior[-lookback:] if len(all_prior) > lookback else all_prior
            prior_texts_long: list[str] = all_prior
            cr = cross_chapter_repetition(chapter, prior_texts, config,
                                          prior_texts_long=prior_texts_long)
            cr_pen = REGISTRY.accumulate(report, cr, "cross_chapter_repetition", REGISTRY.tag_prefix("cross_chapter_repetition"))
            if cr_pen > 0:
                penalties += cr_pen
                log(paths, f"Cross-repeat Ch{chapter_num} penalty={cr_pen} "
                    f"flags={cr.get('flags')} fossils={cr.get('metrics', {}).get('cross_repeat_fossils')}")
            # L2 escalation: entrenched-fossil collapse is a regenerate-not-revise
            # condition. Mark the report with a structured gate_reject so the
            # pipeline can route this chapter into a forced replan with the fossil
            # list as hard evidence (advisory directives alone demonstrably failed:
            # suspense_v11 carried fossils 9-25 for 6 straight chapters).
            if str(cr.get("level", "")) == "reject":
                report["accepted"] = False
                gr = report.setdefault("gate_rejects", [])
                gr.append({
                    "gate": "cross_chapter_repetition",
                    "evidence": {
                        "fossils": cr.get("metrics", {}).get("cross_repeat_fossils"),
                        "examples": [r.get("clause") for r in (cr.get("repeats") or [])[:6]],
                    },
                    "directives": cr.get("directives", []),
                })
                report.setdefault("problems", []).append(
                    "GATE: 文体化石句复读已达坍塌级（确定性检测），本稿必须作废重做，"
                    "禁止沿用本稿的句式与比喻。"
                )
                log(paths, f"Cross-repeat GATE-REJECT Ch{chapter_num}: fossil collapse "
                    f"({cr.get('metrics', {}).get('cross_repeat_fossils')} fossils) — chapter must be regenerated.")
        except Exception as exc:
            log(paths, f"cross_chapter_repetition failed (non-fatal) Ch{chapter_num}: {exc}")

    # O1: adjacent-chapter duplication gate. The deadliest observed failure is a
    # chapter that re-narrates the previous chapter's ending near-verbatim
    # (suspense_v11 Ch3 clause-overlap 0.73 / Ch8 0.33, suspense_v8 Ch6 0.81 vs
    # 0.00-0.07 for healthy chapters) — and the LLM reviewer, scoring each
    # chapter in isolation, rated those identical hooks 9/10. Deterministic
    # measurement against the previous chapter's actual text: warn-level adds a
    # penalty + avoid-list directive, block-level caps the score AND rejects so
    # the chapter is driven into rewrite instead of force-accept.
    if REGISTRY.is_enabled("adjacent_repetition", config) and chapter_num > 1:
        try:
            from quality import adjacent_repetition

            prev_text = read_text(chapter_path(paths, chapter_num - 1))
            ar = adjacent_repetition(chapter, prev_text, config)
            ar_pen = REGISTRY.accumulate(report, ar, "adjacent_repetition", REGISTRY.tag_prefix("adjacent_repetition"))
            if ar_pen > 0:
                penalties += ar_pen
                log(paths, f"Adjacent-repeat Ch{chapter_num} level={ar.get('level')} "
                    f"penalty={ar_pen} metrics={ar.get('metrics')}")
            if ar.get("level") == "block":
                caps.append(float(config["novel"].get("adjacent_repeat_score_cap", 5.0)))
                report["accepted"] = False
                report.setdefault("problems", []).append(
                    "REPEAT: 本章大量逐字复述上一章内容（确定性检测，clause_overlap="
                    f"{ar.get('metrics', {}).get('clause_overlap')}）。"
                    "必须从上一章结尾之后的【新】事件重写，前章场景只许一笔带过。"
                )
                gr = report.setdefault("gate_rejects", [])
                gr.append({
                    "gate": "adjacent_repetition",
                    "evidence": {"metrics": ar.get("metrics", {})},
                    "directives": ar.get("directives", []),
                })
        except Exception as exc:
            log(paths, f"adjacent_repetition failed (non-fatal) Ch{chapter_num}: {exc}")

    # O1b: intra-chapter self-repetition gate. A chapter that ends with a
    # zero-增量 summary paragraph re-stating its own earlier reasoning (observed:
    # suspense_10ch Ch7, mimo, tail recapped the body's deduction) drags
    # readthrough without adding anything. warn adds penalty + directive; block
    # caps the score so the chapter is driven into a rewrite of its ending.
    if REGISTRY.is_enabled("intra_chapter_repetition", config):
        try:
            from quality import intra_chapter_repetition

            ir = intra_chapter_repetition(chapter, config)
            ir_pen = REGISTRY.accumulate(report, ir, "intra_chapter_repetition", REGISTRY.tag_prefix("intra_chapter_repetition"))
            if ir_pen > 0:
                penalties += ir_pen
                log(paths, f"Intra-repeat Ch{chapter_num} level={ir.get('level')} "
                    f"penalty={ir_pen} metrics={ir.get('metrics')}")
            if ir.get("level") == "block":
                caps.append(float(config["novel"].get("intra_repeat_score_cap", 6.0)))
                report.setdefault("problems", []).append(
                    "RECAP: 本章结尾大段复述正文已给出的信息（确定性检测，tail_recap_ratio="
                    f"{ir.get('metrics', {}).get('tail_recap_ratio')}）。"
                    "章末必须改写成推动剧情的新钩子，删去零增量的总结复述。"
                )
        except Exception as exc:
            log(paths, f"intra_chapter_repetition failed (non-fatal) Ch{chapter_num}: {exc}")

    # Creative-contract violations: author-declared hard rules (ability whitelist/
    # modality, blacklist, banned tropes, must-hold). This is the layer that
    # catches "self-consistent but off-contract" drift (e.g. a text-only memory
    # ability quietly used as auditory analysis) which the "is it good prose?"
    # axes happily pass. HARD -2.0, SOFT -0.7 each, folded into the unified clamp.
    cv = [c for c in report.get("contract_violations", []) if isinstance(c, dict)]
    # Deterministic backstop: the reviewer sometimes RECOGNISES an ability/modality
    # breach but routes the description into "problems" (free text) instead of the
    # structured "contract_violations" field that drives the HARD block. If a
    # contract was provided and a problem clearly names an ability-boundary issue,
    # synthesize a structured violation so the block still fires. Only triggers when
    # the model itself flagged it — we never invent a breach from nothing.
    if contract_text_block and contract_text_block != "None":
        kw = ("越界", "模态", "听觉", "视觉", "白名单", "违背设定", "能力范围",
              "out of scope", "modality", "off-contract", "契约")
        already = {str(c.get("prose") or c.get("rule") or "").strip() for c in cv}
        for p in report.get("problems", []):
            ptxt = str(p)
            if ptxt.startswith("CONTRACT:"):
                continue
            if any(k in ptxt for k in kw) and ptxt.strip() not in already:
                cv.append({
                    "rule": "能力白名单/模态（由 problems 文本回填）",
                    "prose": ptxt[:60],
                    "type": "ability_modality_drift",
                    "severity": "soft",
                })
                log(paths, f"Contract backstop Ch{chapter_num}: lifted problem -> hard violation")
        report["contract_violations"] = cv
    contract_hard = [c for c in cv if str(c.get("severity", "")).lower() == "hard"]
    contract_soft = [c for c in cv if str(c.get("severity", "")).lower() != "hard"]
    if cv:
        penalties += 2.0 * len(contract_hard) + 0.7 * len(contract_soft)
        rules = "; ".join(str(c.get("rule") or c.get("type") or "?") for c in cv[:4])
        report.setdefault("problems", []).append(
            f"CONTRACT: {len(contract_hard)} hard / {len(contract_soft)} soft 违约（{rules}）。"
        )
        log(
            paths,
            f"Contract-violation Ch{chapter_num}: hard={len(contract_hard)} "
            f"soft={len(contract_soft)} rules={rules}",
        )

    sh_data = report.get("style_health") or {}
    sh_penalty_val = float(sh_data.get("penalty", 0.0))
    det_floor = float(config["novel"].get("deterministic_score_floor", 5.0))
    if sh_penalty_val == 0 and raw_score < det_floor:
        report.setdefault("calibration", []).append(
            f"raw_score floored {raw_score:.1f}→{det_floor:.1f}: "
            f"style_health penalty=0 (prose objectively healthy)")
        log(paths, f"Deterministic floor Ch{chapter_num}: "
            f"raw={raw_score:.1f}→{det_floor:.1f} (style_health clean)")
        raw_score = det_floor

    # DIMENSION DE-INFLATION (contradiction penalty): hook/readthrough have been
    # saturated for the whole window (reviewer lost discrimination) AND this
    # chapter's deterministic gates disagree (style/repetition penalties ≥ 1.0).
    # Don't trust an all-max scoresheet the objective metrics contradict — same
    # philosophy as style_audit_mismatch. Applied here so `penalties` is complete.
    if dim_saturated_count >= 2 and penalties >= 1.0:
        _disc = float(config["novel"].get("dim_inflation_discount", 0.5))
        penalties += _disc
        report.setdefault("calibration", []).append(
            f"维度通胀惩罚 +{_disc:.1f}：hook/readthrough 长期饱和但本章确定性指标已扣分"
            f"（penalties={penalties:.1f}），不采信满分评审")

    report["score"] = max(1.0, min(min(caps), raw_score) - penalties)

    report.setdefault("accepted", report["score"] >= float(config["novel"]["quality_threshold"]))
    # Block acceptance on a HARD creative-contract violation (ability out of
    # scope/modality drift, blacklist, banned trope, broken must-hold). Gated by
    # contract_blocks_accept (default true) so the chapter is driven into the
    # revise/replan loop instead of being persisted as a contract breach.
    if bool(config["novel"].get("contract_blocks_accept", True)) and contract_hard:
        report["accepted"] = False
        report.setdefault("problems", []).append(
            f"CONTRACT: {len(contract_hard)} 条硬违约（能力越界/模态漂移/黑名单/禁止套路/破坏必守设定）必须修复。"
        )
        # A hard contract breach is a PLAN-level contradiction (the chapter is doing
        # something the ability/setting forbids), not a wording tic. Surface it as a
        # gate-reject so pipeline._classify_replan_failure routes to STRUCTURAL
        # replan (regenerate the plan) instead of burning revision rounds on futile
        # wording patches and then force-accepting the breach (tangshuting Ch21/23).
        report.setdefault("gate_rejects", []).append({
            "gate": "contract_hard",
            "count": len(contract_hard),
            "rules": [str(c.get("rule", "")) for c in contract_hard[:8]],
            "evidence": [str(c.get("prose", "")) for c in contract_hard[:8]],
        })
    # Optionally block acceptance when a HARD contradiction is detected, so the
    if bool(config["novel"].get("factcheck_hard_blocks_accept", True)):
        hard = [c for c in report.get("contradictions", []) if isinstance(c, dict) and str(c.get("severity", "")).lower() == "hard"]
        if hard:
            report["accepted"] = False
            report.setdefault("problems", []).append(
                f"FACTCHECK: {len(hard)} hard contradiction(s) with established facts must be fixed."
            )

    # Fiction failure taxonomy (additive): derive canonical `failure_codes` from
    # the existing problems prefixes + gate_rejects gates so replan-routing
    # (pipeline._classify_replan_failure) and cross-book distillation read a
    # structured vocabulary instead of free text. Purely additive — no score,
    # message, or accept decision changes. Inert if disabled or import fails.
    if bool(config["novel"].get("failure_taxonomy_enabled", True)):
        try:
            import taxonomy
            codes = taxonomy.codes_from_review(report)
            if codes:
                report["failure_codes"] = codes
        except Exception:
            pass
    return report

def stage_review(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> None:
    start = max(1, chapter_num - int(config["novel"]["stage_review_every"]) + 1)
    recent = []
    for num in range(start, chapter_num + 1):
        text = read_text(chapter_path(paths, num))
        if text:
            recent.append(f"## Ch{num}\n{text[:1600]}")
    user = f"""## 记忆
{memory_context(paths, conn, config)}

## 节奏诊断JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## 结构重复分析JSON
{json.dumps(structural_repetition_analysis(conn, config), ensure_ascii=False, indent=2)}

## 近期章节
{chr(10).join(recent)}

审校截至第 {chapter_num} 章的长周期质量。"""
    raw = call_llm(client, paths, config, STAGE_REVIEW_SYSTEM, json_prompt(user), max_tokens=12000, temperature=0.3, tag="stage_review")
    data = load_json_with_repair(
        client,
        paths,
        config,
        raw,
        fallback={
            "quality_trend": "JSON parse failed; fallback used.",
            "continuity_risks": [],
            "rhythm_payoff_risks": [],
            "repetition_risks": [],
            "next_20_chapters_replan": [],
            "threads_to_recover_or_upgrade": [],
            "writer_directives_for_next_chapter": [],
            "constraints": [],
        },
    )

    def render_section(title: str, content: Any) -> str:
        if isinstance(content, list):
            if not content:
                return f"## {title}\n_(none)_\n"
            return f"## {title}\n" + "\n".join(f"- {item}" for item in content) + "\n"
        return f"## {title}\n{content}\n"

    markdown = (
        render_section("质量走势", data.get("quality_trend", ""))
        + render_section("连续性风险", data.get("continuity_risks", []))
        + render_section("节奏与兑现风险", data.get("rhythm_payoff_risks", []))
        + render_section("重复风险", data.get("repetition_risks", []))
        + render_section("接下来20章重规划", data.get("next_20_chapters_replan", []))
        + render_section("待找回或提升的伏线", data.get("threads_to_recover_or_upgrade", []))
        + render_section("给下一章的作者指令", data.get("writer_directives_for_next_chapter", []))
    )
    append_text(paths.logs_dir / "stage_reviews.md", f"\n\n# 第{chapter_num}章 阶段审校\n\n{markdown}\n")
    db_event(conn, chapter_num, "stage_review", {"review": data})

    # Persist stage-level writer_directives onto the most-recent chapter's
    # final_review.json so the NEXT chapter writer prompt picks them up via
    # writer_directives_for_chapter(). We append to the existing list (chapter
    # review directives take precedence — we only add if the stage layer has
    # surfaced something not already listed).
    stage_directives = data.get("writer_directives_for_next_chapter") or []
    if stage_directives:
        try:
            from checkpoint import load_checkpoint as _load, save_checkpoint as _save
            existing = _load(paths, chapter_num, "final_review.json")
            if isinstance(existing, dict):
                merged = list(existing.get("writer_directives_for_next_chapter") or [])
                for d in stage_directives:
                    s = str(d).strip()
                    if s and s not in merged:
                        merged.append(s)
                existing["writer_directives_for_next_chapter"] = merged[:10]
                _save(paths, chapter_num, "final_review.json", existing)
                log(paths, f"Merged {len(stage_directives)} stage directives into Ch{chapter_num} review")
        except Exception as exc:
            log(paths, f"Failed to merge stage directives into Ch{chapter_num} review: {exc}")

    constraints = data.get("constraints") or []
    if constraints:
        store_stage_constraints(conn, chapter_num, constraints)
        log(paths, f"Stored {len(constraints)} stage constraints from Ch{chapter_num} review")

    # Refresh narrative voice anchors using the recent prose window.
    try:
        refresh_voice_anchors(client, paths, conn, config, chapter_num, recent_text="\n\n".join(recent))
    except Exception as exc:
        log(paths, f"Voice anchor refresh failed at Ch{chapter_num}: {exc}")

    # Refresh the proper-noun glossary from the same recent prose window so
    #专有名词/硬设定 stay consistent across chapters. Best-effort.
    try:
        refresh_glossary(client, paths, conn, config, chapter_num, recent_text="\n\n".join(recent))
    except Exception as exc:
        log(paths, f"Glossary refresh failed at Ch{chapter_num}: {exc}")


def pack_review(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> None:
    pack_size = int(config["novel"].get("pack_review_every", 10))
    start = max(1, chapter_num - pack_size + 1)
    recent = []
    for num in range(start, chapter_num + 1):
        text = read_text(chapter_path(paths, num))
        if text:
            recent.append(f"## Ch{num}\n{text[:2200]}")
    if not recent:
        return
    try:
        from store import get_reader_promises

        promises = get_reader_promises(conn, chapter_num, limit=20)
    except Exception:
        promises = []
    user = f"""## 平台/读者画像
{_platform_guidance(config)}

## 记忆
{memory_context(paths, conn, config)}

## 读者承诺账本
{json.dumps(promises, ensure_ascii=False, indent=2) if promises else "None"}

## 近期指标
{json.dumps(recent_metrics(conn, pack_size), ensure_ascii=False, indent=2)}

## 待评审章节 Ch{start}-{chapter_num}
{chr(10).join(recent)}
"""
    raw = call_llm(client, paths, config, PACK_REVIEW_SYSTEM, json_prompt(user), max_tokens=16000, temperature=0.25, tag="pack_review")
    data = load_json_with_repair(client, paths, config, raw, fallback={})
    append_text(
        paths.logs_dir / "pack_reviews.md",
        f"\n\n# 第{start}-{chapter_num}章 10章包追读审校\n\n{json.dumps(data, ensure_ascii=False, indent=2)}\n",
    )
    db_event(conn, chapter_num, "pack_review", {"review": data})
    directives = data.get("next_10_directives") if isinstance(data, dict) else []
    if isinstance(directives, list) and directives:
        try:
            from checkpoint import load_checkpoint as _load, save_checkpoint as _save

            existing = _load(paths, chapter_num, "final_review.json")
            if isinstance(existing, dict):
                merged = list(existing.get("writer_directives_for_next_chapter") or [])
                for d in directives:
                    s = str(d).strip()
                    if s and s not in merged:
                        merged.append(s)
                existing["writer_directives_for_next_chapter"] = merged[:12]
                _save(paths, chapter_num, "final_review.json", existing)
                log(paths, f"Merged {len(directives)} pack directives into Ch{chapter_num} review")
        except Exception as exc:
            log(paths, f"Failed to merge pack directives into Ch{chapter_num} review: {exc}")
    constraints = data.get("constraints") if isinstance(data, dict) else []
    if isinstance(constraints, list) and constraints:
        store_stage_constraints(conn, chapter_num, constraints)
        log(paths, f"Stored {len(constraints)} pack constraints from Ch{chapter_num} review")


def refresh_voice_anchors(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    recent_text: str,
) -> None:
    """Update memory/voice.md and memory/voices.md based on actual recent prose.

    Called from stage_review (every N chapters). Each refresh is best-effort:
    if the LLM call fails, the existing files remain unchanged.

    Anti-collapse design: a one-time baseline snapshot of voice.md is frozen the
    first time this runs (voice_baseline.md). Every subsequent refresh is anchored
    to that baseline so the prose style cannot self-reinforce into degeneration.
    If the recent prose window itself shows a style-health collapse, the refresh
    is SKIPPED entirely (we never feed a degraded sample back into the anchor).
    """
    if not recent_text.strip():
        return

    # Guard: never absorb a degraded prose sample into the voice anchor.
    if bool(config["novel"].get("style_health_enabled", True)):
        try:
            from quality import style_health

            sh = style_health(recent_text, config)
            if float(sh.get("penalty", 0.0)) >= float(config["novel"].get("voice_refresh_skip_penalty", 1.0)):
                log(
                    paths,
                    f"Voice refresh SKIPPED at Ch{chapter_num}: recent prose shows style collapse "
                    f"flags={sh.get('flags')}; keeping existing voice.md to avoid reinforcing it.",
                )
                return
        except Exception:
            pass

    # Freeze a baseline the first time, then always anchor to it.
    baseline_path = paths.voice.with_name("voice_baseline.md")
    current_voice = read_text(paths.voice)
    if not read_text(baseline_path).strip() and current_voice.strip():
        write_text(baseline_path, current_voice)
        log(paths, f"Froze voice baseline at Ch{chapter_num} (len={len(current_voice)})")
    baseline_voice = read_text(baseline_path) or current_voice

    voice_user = f"""## 原始声音基线（健康文风锚——保留至少 80%）
{baseline_voice if baseline_voice.strip() else "(空)"}

## 当前 voice.md
{current_voice if current_voice.strip() else "(空——从基线生成)"}

## 近期章节正文（仅吸收其中健康的正面特征，拒绝任何文体退化）
{recent_text[:18000]}

为第 {chapter_num} 章刷新 voice.md。"""
    new_voice = call_llm(client, paths, config, VOICE_ANCHOR_SYSTEM, voice_user, max_tokens=8000, temperature=0.3, tag="voice_anchor")
    new_voice = normalize_text(new_voice).strip()
    if new_voice:
        write_text(paths.voice, new_voice + "\n")
        log(paths, f"Updated voice.md at Ch{chapter_num} (len={len(new_voice)})")

    current_voices = read_text(paths.voices)
    voices_user = f"""## 当前 voices.md
{current_voices if current_voices.strip() else "(空——从正文生成)"}

## 近期章节正文
{recent_text[:18000]}

为第 {chapter_num} 章刷新 voices.md。"""
    new_voices = call_llm(client, paths, config, VOICES_TABLE_SYSTEM, voices_user, max_tokens=8000, temperature=0.3, tag="voices_table")
    new_voices = normalize_text(new_voices).strip()
    if new_voices:
        write_text(paths.voices, new_voices + "\n")
        log(paths, f"Updated voices.md at Ch{chapter_num} (len={len(new_voices)})")


GLOSSARY_SYSTEM = """你是本书的设定词条/名词表维护编辑。
你的职责是维护一份【可被作者随手查阅的名词表(glossary)】，确保全书的专有名词
(人名/地名/组织/功法/物品/称号/术语)在每一章里写法一致、设定不漂移。

只返回恰好一个合法 JSON 对象，不要输出其它内容：
{
  "entries": [
    {
      "term": "<规范名词(唯一写法)>",
      "type": "person|place|org|item|skill|title|term|other",
      "canonical": "<本书认定的唯一正确写法/称呼>",
      "aliases": ["允许的别称/简称"],
      "definition": "<一句话设定，含关键不可变属性，如阵营/能力边界/与主角关系>",
      "do_not": "<最容易写错/写漂的点，如'勿写成另一个相似名''其能力不含X'，可留空>"
    }
  ],
  "contradiction_warnings": ["近期章节中疑似出现的名词不一致/设定矛盾(若无则空数组)"]
}

要求：
- 在【现有词条表】基础上增量更新：保留既有正确条目，只新增本批新出现的名词，并修正明显漂移。
- canonical 必须唯一；同一实体的多种写法收进 aliases，不要拆成多条。
- definition 只记跨章不可变的硬设定，不要写剧情进展。
- 名词总数控制在合理范围(优先保留主角/高频/易混名词)，不要堆砌一次性龙套。"""


def glossary_block(paths: Paths, config: dict[str, Any]) -> str:
    """Render memory/glossary.md as a compact writer-prompt injection block.

    Read-only and best-effort: returns "" when the glossary is missing/empty or
    the feature is disabled. Rides in the writer's variable carryover section —
    it is NOT part of cacheable_prefix, so updating it never invalidates the
    prompt cache for prior chapters.
    """
    if not bool(config["novel"].get("glossary_enabled", True)):
        return ""
    try:
        text = read_text(paths.glossary).strip()
    except Exception:
        return ""
    # Skip when empty or only a scaffold heading with no real entries.
    if len(text) < 40:
        return ""
    budget = int(config["novel"].get("glossary_inject_chars", 1800) or 1800)
    snippet = text[:budget]
    return (
        "## 名词表 / 设定一致性(写作时严格遵守，勿改写专有名词)\n"
        "以下是本书已确立的专有名词与硬设定。本章涉及这些名词时，必须使用其 canonical 写法，"
        "不得擅自改名、改设定或赋予白名单外的能力；如需引入全新名词，确保与下列不冲突。\n"
        f"{snippet}"
    )


def refresh_glossary(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    recent_text: str,
) -> None:
    """Incrementally update memory/glossary.md from recent prose.

    Called from stage_review alongside refresh_voice_anchors (every N chapters).
    Feeds the LLM (a) the existing glossary and (b) a deterministic shortlist of
    candidate NEW proper-noun surface forms harvested from the recent window, so
    the model focuses on genuinely new/edge terms rather than re-deriving the
    whole table. Best-effort: any failure leaves the existing file untouched.
    """
    if not bool(config["novel"].get("glossary_enabled", True)):
        return
    if not recent_text.strip():
        return
    try:
        existing = read_text(paths.glossary).strip()
        suspects: list[str] = []
        try:
            from retrieval import candidate_new_entities
            suspects = candidate_new_entities(
                paths, recent_text,
                limit=int(config["novel"].get("glossary_candidate_limit", 20)),
            )
        except Exception:
            suspects = []
        user = f"""## 现有词条表(glossary.md，可能为空)
{existing if existing else "(空——请从近期正文新建)"}

## 确定性挑出的疑似新名词(供参考，非全部都需收录)
{json.dumps(suspects, ensure_ascii=False)}

## 近期章节正文
{recent_text[:16000]}

更新截至第 {chapter_num} 章的名词表。"""
        max_tokens = int(config["novel"].get("glossary_max_tokens", 4000) or 4000)
        raw = call_llm(
            client, paths, config, GLOSSARY_SYSTEM, json_prompt(user),
            max_tokens=max_tokens, temperature=0.2, tag="glossary",
        )
        data = load_json_with_repair(client, paths, config, raw, fallback={})
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            log(paths, f"Glossary refresh Ch{chapter_num}: no entries returned; keeping existing.")
            return
        # Render a stable, human-readable markdown table the writer block reads.
        lines: list[str] = [f"# 名词表 / Glossary（截至第{chapter_num}章）", ""]
        for e in entries:
            if not isinstance(e, dict):
                continue
            term = str(e.get("canonical") or e.get("term") or "").strip()
            if not term:
                continue
            typ = str(e.get("type", "")).strip()
            aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()]
            definition = str(e.get("definition", "")).strip()
            do_not = str(e.get("do_not", "")).strip()
            head = f"## {term}" + (f" [{typ}]" if typ else "")
            lines.append(head)
            if aliases:
                lines.append(f"- 别称：{'、'.join(aliases)}")
            if definition:
                lines.append(f"- 设定：{definition}")
            if do_not:
                lines.append(f"- 勿写错：{do_not}")
            lines.append("")
        warns = data.get("contradiction_warnings") or []
        if isinstance(warns, list) and warns:
            lines.append("## 一致性警告")
            for w in warns:
                if str(w).strip():
                    lines.append(f"- {str(w).strip()}")
            lines.append("")
        write_text(paths.glossary, "\n".join(lines).rstrip() + "\n")
        log(paths, f"Updated glossary.md at Ch{chapter_num} ({len(entries)} entries, suspects={len(suspects)})")
    except Exception as exc:
        log(paths, f"Glossary refresh failed (non-fatal) Ch{chapter_num}: {exc}")


COLD_READER_SYSTEM = """你是一名**没有读过本书前文**、第一次拿到这一章的挑剔读者兼资深编辑。
你不知道作者的任何设定、声音锚或写作意图——你只看这一章的文字本身。
请用"陌生人视角"诚实判断这一章作为小说是否好读，重点抓两类毛病：
1. 文体是否畸形——两种病都要抓：
   (a) 碎句化：大量破折号（——）把句子切成碎片、通篇单词短句、像电报或舞台提示而不像小说；
   (b) 过度书写：超长句一逗到底、堆砌技术名词与伪精确测量值（如"零点三毫米""每分钟七十二次""频率/脉冲/共振"），整章几乎没有人物对话，读起来像仪器报告或实验记录而不像小说。
2. 剧情是否原地打转：这一章是否在反复咀嚼同一个微观场景/同一件事，几乎没有实质推进？读完后你是否觉得"什么都没真正发生"？

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "readable_prose": 1-10,        // 作为人类可读小说的流畅度，畸形文体给低分
  "plot_progression": 1-10,      // 本章是否有实质剧情推进，原地打转给低分
  "overall_impression": "<=80字，一名陌生读者读完的真实感受>",
  "worst_problem": "<=60字，最该修的一个问题>",
  "verdict": "good|mediocre|broken"
}
诚实、果断。畸形文体或原地打转都应给 broken。"""


def cold_reader_review(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
) -> dict[str, Any]:
    """Independent 'cold reader' pass with NO shared cacheable_prefix and no memory.

    Because it shares none of the writer's context (voice anchors, bible, prior
    chapters), it judges the prose as a stranger would — which is exactly what
    catches style collapse and in-place spinning that self-review ratifies.
    """
    _review_chars = int(config.get("novel", {}).get("review_chapter_chars", 16000))
    user = f"""## 这一章的全文（你对本书一无所知）
{chapter[:_review_chars]}

请以陌生读者视角评估这一章。"""
    raw = call_llm(
        client, paths, config, COLD_READER_SYSTEM, json_prompt(user),
        max_tokens=2000, temperature=0.3,  # NOTE: deliberately no cacheable_prefix
        tag="cold_reader",
    )
    data = load_json_with_repair(
        client, paths, config, raw,
        fallback={"readable_prose": 4, "plot_progression": 4, "overall_impression": "", "worst_problem": "冷读者评审解析失败，按最坏情况处理", "verdict": "broken"},
    )
    data["readable_prose"] = safe_score(data.get("readable_prose", 4))
    data["plot_progression"] = safe_score(data.get("plot_progression", 4))
    return data


MACRO_PROGRESS_SYSTEM = """你是一部长篇连载小说的宏观叙事审阅者。
你会收到：全书卷纲（含每卷的"大事件锚点"），以及最近若干章的标题与梗概。
请判断：从最近这些章节看，主线剧情是否在以合理速度向卷纲的大事件锚点推进，
还是在反复纠缠同一个微观局部（同一场对话/同一份公文/同一个僵局）而原地踏步。

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "advancing": true,                 // 主线是否在推进
  "chapters_on_same_microbeat": 0,   // 估计已有多少章纠缠在同一微观事件上
  "next_anchor": "<卷纲中下一个应当触及的大事件锚点的简述>",
  "kr_status": [
    {"kr": "关键成果描述", "progress": "not_started|in_progress|completed", "evidence": "章节中的具体证据（若有）"}
  ],
  "directives": ["3-5 条具体指令，强制后续章节跳出微观僵局、推进到 next_anchor"]
}
若卷纲包含"卷目标(O)"和"关键成果(KR)"，请逐条评估 KR 完成状态填入 kr_status。若某 KR 在该卷已过半章节时仍为 not_started，必须在 directives 中加入加速指令。"""


def macro_progress_check(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> dict[str, Any] | None:
    """Detect macro plot stagnation against volume_plan anchors and inject directives.

    When the story has spun on the same micro-beat for too long, push acceleration
    directives onto the latest chapter's review so the next writer breaks out.
    """
    window = int(config["novel"].get("macro_progress_every", 10))
    rows = recent_metrics(conn, window)
    if len(rows) < max(5, window // 2):
        return None
    recent_lines = []
    for r in rows:
        recent_lines.append(
            f"- Ch{r.get('chapter')} 「{r.get('title','')}」 payoff={r.get('payoff_type','')} tone={str(r.get('emotional_tone',''))[:40]}"
        )
    volume_plan = read_text(paths.volume_plan).strip()
    user = f"""## 全书卷纲（含大事件锚点）
{volume_plan[:12000]}

## 最近 {len(rows)} 章（最新在前）
{chr(10).join(recent_lines)}

当前已写到第 {chapter_num} 章。判断主线是否在推进，还是原地打转。"""
    raw = call_llm(
        client, paths, config, MACRO_PROGRESS_SYSTEM, json_prompt(user),
        max_tokens=2500, temperature=0.3, tag="macro_progress",
    )
    data = load_json_with_repair(
        client, paths, config, raw,
        fallback={"advancing": False, "chapters_on_same_microbeat": 0, "next_anchor": "", "directives": []},
    )
    db_event(conn, chapter_num, "macro_progress", data)
    stall = int(config["novel"].get("macro_progress_stall_threshold", 12))
    spinning = (not data.get("advancing", True)) or int(safe_score(data.get("chapters_on_same_microbeat", 0))) >= stall
    if spinning:
        directives = [str(d).strip() for d in (data.get("directives") or []) if str(d).strip()]
        if data.get("next_anchor"):
            directives.insert(0, f"主线已停滞，本章必须推进到卷纲锚点：{data['next_anchor']}")
        log(
            paths,
            f"Macro stagnation at Ch{chapter_num}: same_microbeat={data.get('chapters_on_same_microbeat')} "
            f"next_anchor={data.get('next_anchor')!r}; injecting {len(directives)} acceleration directives.",
        )
        # Persist onto latest chapter review so the next writer reads them.
        try:
            from checkpoint import load_checkpoint as _load, save_checkpoint as _save

            existing = _load(paths, chapter_num, "final_review.json")
            if isinstance(existing, dict):
                merged = list(existing.get("writer_directives_for_next_chapter") or [])
                for d in directives:
                    if d not in merged:
                        merged.append(d)
                existing["writer_directives_for_next_chapter"] = merged[:12]
                _save(paths, chapter_num, "final_review.json", existing)
        except Exception as exc:
            log(paths, f"Failed to persist macro directives Ch{chapter_num}: {exc}")
    return data


def horizon_review(
    client: "OpenAI",
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
) -> None:
    """Unified periodic review fired every `cold_reader_every` chapters (default 10).

    Combines cold_reader_review + pack_review + macro_progress_check into a single
    background task, reducing the number of independent submit() calls while
    preserving every detection dimension.

    Every trigger: cold_reader (no cacheable_prefix) + pack_review
    chapter_num >= 20: additionally runs macro_progress_check
    chapter_num % stage_review_every == 0: stage_review fires separately (unchanged)
    """
    # --- 1. Cold reader (always, no cacheable_prefix) ---
    try:
        cr = cold_reader_review(client, paths, config, chapter_num, chapter)
        log(
            paths,
            f"Cold-reader Ch{chapter_num} prose={cr.get('readable_prose')}/10 "
            f"progression={cr.get('plot_progression')}/10 verdict={cr.get('verdict')} "
            f"problem={str(cr.get('worst_problem'))[:80]!r}",
        )
        db_event(conn, chapter_num, "cold_reader", cr)
        bad = (
            str(cr.get("verdict")) == "broken"
            or safe_score(cr.get("readable_prose", 10)) < 6
            or safe_score(cr.get("plot_progression", 10)) < 5
        )
        if bad:
            try:
                from checkpoint import load_checkpoint as _load, save_checkpoint as _save
                existing = _load(paths, chapter_num, "final_review.json")
                if isinstance(existing, dict):
                    wd = list(existing.get("writer_directives_for_next_chapter") or [])
                    msg = f"冷读者警示（陌生读者视角）：{cr.get('worst_problem','')}。下一章必须针对性修正。"
                    if msg not in wd:
                        wd.append(msg)
                    existing["writer_directives_for_next_chapter"] = wd[:12]
                    _save(paths, chapter_num, "final_review.json", existing)
            except Exception as exc:
                log(paths, f"Failed to persist cold-reader directive Ch{chapter_num}: {exc}")
    except Exception as exc:
        log(paths, f"Cold-reader review failed (non-fatal) Ch{chapter_num}: {exc}")

    # --- 2. Pack review (always) ---
    try:
        pack_review(client, paths, conn, config, chapter_num)
        log(paths, f"Completed pack review Ch{chapter_num}")
    except Exception as exc:
        log(paths, f"Pack review failed (non-fatal) Ch{chapter_num}: {exc}")

    # --- 3. Macro progress check (only from Ch20 onwards) ---
    macro_every = int(config["novel"].get("macro_progress_every", 10))
    if (
        bool(config["novel"].get("macro_progress_enabled", True))
        and macro_every > 0
        and chapter_num >= 20
    ):
        try:
            macro_progress_check(client, paths, conn, config, chapter_num)
        except Exception as exc:
            log(paths, f"Macro-progress check failed (non-fatal) Ch{chapter_num}: {exc}")


ANCHOR_GATE_SYSTEM = """你是一部连载小说的"完成度审计员"。
你会收到全书卷纲（含每卷的"大事件锚点"与"本卷兑现"）、终章结尾的正文原文、以及迄今所有章节的标题与梗概。
你的唯一任务是判断：本书在叙事上是否已经把卷纲承诺的**必达大事件锚点**都真正落地兑现了，
而不是字数/章数到了就草草收尾、把关键锚点（真相揭露、主线对峙、核心代价兑现）漏在页面之外。

注意：本书可能设有总章数上限（见下方"章数上限"）。当卷纲是按更长篇幅规划、其锚点章号或卷数超出该上限时，
**以章数上限为准**：只审计那些能合理压缩进剩余章数的**核心收束锚点**（真相、对峙、主角核心代价），
不要因为卷纲里那些超出上限的、本就不该在本书兑现的远期锚点而判未兑现。当核心收束锚点已落地时，应判 all_anchors_realized=true。

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "all_anchors_realized": true,
  "unrealized_anchors": [
    {"anchor": "卷纲里被漏掉或只是口头提及、未在正文落地的必达锚点简述",
     "why": "为什么判定它尚未兑现（缺了哪个揭露/对峙/代价）",
     "new_scene": "加写一章将新增的、终章结尾里【完全没有出现过】的新场景/新信息/新动作（必须具体到谁、在哪、做什么）。如果想不出终章里没有的新内容，说明锚点其实已兑现，不要列入本数组",
     "directive": "若要补齐，下一章必须落地的具体场景任务（谁、在哪、做了什么、读者看到什么）"}
  ]
}
判定规则（按优先级）：
1. **终章结尾原文是首要证据**：凡是结尾原文里已经发生的揭露/对峙/代价/签字/指认，一律判 realized——哪怕章节列表的梗概里没提。
2. 只有当某锚点对应的**具体场景**确实没有出现在任何一章正文里（真相没被当面说破、对峙没有发生、代价没有落到人物身上），才算 unrealized；仅在梗概/状态里"提到""暗示""计划"不算兑现。
3. **加写一章必须带来新内容**：每个 unrealized 锚点必须给出 new_scene——一个终章结尾里完全没有的新画面。若加写章只会把终章结尾换一种说法重演一遍，那不是未兑现，是已兑现；判 realized。
误判加写的代价极高（每多写一章都在复述前章、稀释结局），宁可判已兑现，也不要为"再演一遍"开绿灯。"""


def anchor_completion_gate(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
) -> dict[str, Any]:
    """Audit whether the volume_plan's must-hit anchors are actually realized.

    The termination condition was purely quantitative (char target OR
    max_chapters), so the engine could stop with the climactic anchors —
    真相揭露 / 主线对峙 / 核心代价兑现 — still unfulfilled (the failure mode
    where the finale gets summarized away rather than dramatized). This is an
    LLM audit of `volume_plan.md`'s 大事件锚点 + 本卷兑现 against the actual
    chapter梗概 history. Returns:
      {"all_anchors_realized": bool,
       "unrealized_anchors": [{"anchor","why","directive"}],
       "directives": [str]}   # flattened imperative fixes for the next writer

    Only meaningful in short-novel mode (max_chapters set); the caller gates the
    invocation, so a pure char-target long novel (no deterministic finale) never
    pays for this call.
    """
    volume_plan = read_text(paths.volume_plan).strip()
    if not volume_plan:
        return {"all_anchors_realized": True, "unrealized_anchors": [], "directives": []}
    # When bootstrap never produced a real volume plan (placeholder only) there
    # are NO concrete anchors to audit against. The LLM would otherwise keep
    # reporting "卷纲未提供" as an unrealized anchor on every termination check,
    # extending the novel by +1 chapter each pass up to anchor_gate_max_extra
    # for nothing. Treat an anchorless plan as "nothing to enforce" so the
    # quantitative termination (char target / max_chapters) stands.
    plan_probe = volume_plan.replace(" ", "").replace("　", "")
    placeholder_markers = ("bootstrap未生成", "待连载补全", "待补全", "未生成")
    has_anchor_section = ("锚点" in volume_plan) or ("大事件" in volume_plan)
    if (not has_anchor_section) or any(m in plan_probe for m in placeholder_markers):
        return {"all_anchors_realized": True, "unrealized_anchors": [], "directives": []}
    rows = recent_metrics(conn, max(chapter_num, 1))
    rows = sorted(rows, key=lambda r: int(r.get("chapter", 0)))
    history_lines = [
        f"- Ch{r.get('chapter')} 「{r.get('title','')}」 payoff={r.get('payoff_type','')} "
        f"tone={str(r.get('emotional_tone',''))[:40]}"
        for r in rows
    ]
    # O2: the auditor previously judged anchors against one-line metric history
    # only, so an anchor that WAS dramatized in the final chapter's ending still
    # got audited as "未兑现" (observed: suspense_v11 was extended +3 chapters,
    # each a near-verbatim re-narration of the previous ending, scores 3.5-5.5).
    # Give the auditor the actual final-chapter ending text so "already on the
    # page" is verifiable, and require it to name the NEW content an extra
    # chapter would add — no new content, no extension.
    final_tail = ""
    try:
        final_tail = read_text(chapter_path(paths, chapter_num))[-4000:]
    except Exception:
        final_tail = ""
    final_tail_block = (
        f"\n## 终章（第 {chapter_num} 章）结尾原文（判定的首要依据）\n{final_tail}\n"
        if final_tail.strip()
        else ""
    )
    # Short-novel cap context: when the volume_plan was written for a longer book
    # than max_chapters allows (e.g. a 60-70 章 plan in a 6-章 novel), its远期
    # anchors sit beyond the cap and would be audited as永远"未兑现", dragging the
    # book past its end. Tell the auditor the hard cap so it only enforces the
    # core closing anchors that can land within the remaining chapters.
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    cap_block = ""
    if max_chapters:
        cap_block = (
            f"\n## 章数上限（硬约束，优先于卷纲篇幅）\n"
            f"全书总章数上限为 {max_chapters} 章，本书必须在第 {max_chapters} 章收束完结。"
            f"卷纲若按更长篇幅规划，其超出第 {max_chapters} 章的锚点不属于本书必达范围——"
            f"只审计能压缩进前 {max_chapters} 章的核心收束锚点（真相 / 主线对峙 / 主角核心代价）。\n"
        )
    user = f"""## 全书卷纲（含必达大事件锚点与本卷兑现）
{volume_plan[:16000]}
{cap_block}{final_tail_block}
## 迄今全部章节（最旧在前，共 {len(history_lines)} 章）
{chr(10).join(history_lines)}

当前已写到第 {chapter_num} 章，引擎即将判断是否可以收尾。审计必达锚点是否都已在正文落地兑现。
注意：上方"终章结尾原文"是正文实拍——凡是其中已经发生的揭露/对峙/代价，一律判 realized，不要因为章节列表里没写细节就误判未兑现。"""
    raw = call_llm(
        client, paths, config, ANCHOR_GATE_SYSTEM, json_prompt(user),
        max_tokens=2500, temperature=0.2, tag="anchor_gate",
    )
    data = load_json_with_repair(
        client, paths, config, raw,
        fallback={"all_anchors_realized": True, "unrealized_anchors": []},
    )
    unrealized = [u for u in (data.get("unrealized_anchors") or []) if isinstance(u, dict)]
    # O2: an extra chapter must bring NEW content. An unrealized entry that
    # cannot name a concrete new scene (one absent from the final ending) is the
    # exact misfire that produced 3 consecutive re-narration chapters — drop it.
    kept: list[dict[str, Any]] = []
    for u in unrealized:
        ns = str(u.get("new_scene", "")).strip()
        if len(ns) < 10:
            log(
                paths,
                f"Anchor gate Ch{chapter_num}: dropping unrealized anchor without a concrete "
                f"new_scene ({str(u.get('anchor',''))[:40]!r}) — no new content to dramatize.",
            )
            continue
        kept.append(u)
    unrealized = kept
    directives: list[str] = []
    for u in unrealized:
        d = str(u.get("directive", "")).strip()
        anchor = str(u.get("anchor", "")).strip()
        ns = str(u.get("new_scene", "")).strip()
        if d:
            directives.append(d if (anchor and anchor in d) or not anchor else f"必达锚点未兑现：{anchor}。{d}")
        elif anchor:
            directives.append(f"必达锚点未兑现，本章必须落地：{anchor}")
        if ns:
            directives.append(f"加写章必须包含终章里没有的新场景：{ns}。严禁复述前一章已发生的场景与对话。")
    result = {
        "all_anchors_realized": bool(data.get("all_anchors_realized", True)) and not unrealized,
        "unrealized_anchors": unrealized,
        "directives": directives[:6],
    }
    db_event(conn, chapter_num, "anchor_gate", result)
    return result


def should_replan(conn: Any, config: dict[str, Any]) -> bool:
    window = int(config["novel"].get("repeat_window", 24))
    rows = recent_metrics(conn, 20)
    if len(rows) < max(8, window // 2):
        return False
    threshold_score = float(config["novel"].get("replan_score_threshold", 6.5))
    threshold_novelty = float(config["novel"].get("replan_novelty_threshold", 5.5))
    triggers = 0
    scores = [safe_score(r.get("score", 7)) for r in rows if r.get("score") is not None]
    novelties = [int(r.get("novelty", 7)) for r in rows if r.get("novelty") is not None]
    if scores and sum(scores) / len(scores) < threshold_score:
        triggers += 1
    if novelties and sum(novelties) / len(novelties) < threshold_novelty:
        triggers += 1
    structural = structural_repetition_analysis(conn, config)
    if len(structural.get("warnings", [])) >= 3:
        triggers += 1
    # Emotional fatigue: a flat or monotonically-falling tension curve is its own
    # replan trigger — readers disengage when intensity never varies or only sags.
    if structural.get("tension_shape") in {"flat", "monotone_fall"}:
        triggers += 1
    return triggers >= 2

def adaptive_replan(
    client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any], chapter_num: int
) -> None:
    shutil.copy2(paths.volume_plan, paths.volume_plan.with_suffix(".md.bak"))
    user = f"""## 记忆
{memory_context(paths, conn, config)}

## 当前卷纲（全文——保留已写部分，从当前章节起重规划）
{read_text(paths.volume_plan).strip()}

## 节奏诊断JSON
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## 结构重复分析JSON
{json.dumps(structural_repetition_analysis(conn, config), ensure_ascii=False, indent=2)}

## 未决因果需求JSON
{json.dumps(get_open_causal_requirements(conn), ensure_ascii=False, indent=2)}

## 生效约束JSON
{json.dumps(get_active_constraints(conn, chapter_num), ensure_ascii=False, indent=2)}

当前章节：{chapter_num}。从 Ch{chapter_num} 起重规划接下来的 40-60 章，保持分卷的结构化格式。不要改写过去。"""
    new_plan = call_llm(client, paths, config, REPLAN_SYSTEM, user, max_tokens=16000, temperature=0.5, tag="replan")
    write_text(paths.volume_plan, normalize_text(new_plan) + "\n")
    db_event(conn, chapter_num, "adaptive_replan", {"reason": "metrics_degradation"})
