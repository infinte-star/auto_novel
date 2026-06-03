from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from config import Paths, append_text, chapter_path, log, normalize_text, read_text, safe_score, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, memory_context, rhythm_diagnostics, structural_repetition_analysis, writing_memory_context
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
  "accepted": true,
  "problems": [],
  "fixes": [],
  "continuity_risks": [],
  "rhythm_risks": [],
  "reader_fatigue_risks": [],
  "hook_strength": 1-10,
  "style_audit": {"em_dash_per_kchar": 0.0, "fragment_line_ratio": 0.0, "has_full_dialogue": true},
  "beats_audit": [{"beat":"...", "status":"realized|partial|absent", "evidence":"引文或备注"}],
  "contradictions": [{"fact":"被违背的既定事实", "prose":"引用章节中违背它的 6-20 字原文", "severity":"hard|soft"}],
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

## 文风客观自检（必填，且必须先于评分）
在打分之前，先填写 "style_audit"：逐字统计本章正文，给出
- em_dash_per_kchar：破折号（——）出现次数 ÷（正文字数/1000），保留一位小数；
- fragment_line_ratio：不含主谓宾的碎片化短句/断行 占全部句行的比例（0-1）；
- has_full_dialogue：本章是否含有正常成句的人物对话（true/false）。
**硬上限规则（不可通过任何加分绕过）**：若正文出现"句子——状态——状态"式破折号串联碎句、单词短句堆叠、无标点舞台提示式断行，或 em_dash_per_kchar≥6，或 fragment_line_ratio≥0.35，或 has_full_dialogue=false，则本章 base 分上限为 5.5，且最终 score 禁止给出 7 以上。这是严重的"非小说"文风缺陷，不是风格。

先从一个反映原始功力（写作质量、场景具体度、对话、情感兑现、**文风是否流畅可读**）的基础分起步。
然后按以下软性惩罚扣分：
- 缺失重要大纲节拍：每个完全缺席节拍 -1.0；每个部分实现节拍 -0.5。
- 含糊带过时间线/金钱/路线/程序：每处 -1.0。
- 在没有新功能的情况下重复近期场景形态或章末手法：-1.0。
- 忽视近期审校点名的连续性风险：每个被忽视风险 -1.0（总计上限 -2.0）。
- 超过 30% 的大纲节拍为部分/缺席：额外 -0.5（在逐节拍扣分之上）。
- 大纲上下文中列出的沉默伏线（沉默 >10 章）本可推进却被忽视：-0.7。
- 与既定事实（见下方 "## 既定事实"）矛盾：每个 HARD 矛盾 -2.0（既述事实被推翻：已死人物行动、已知地点出错、已失去的资源重新出现），每个 SOFT 矛盾 -0.5（与既定状态的张力/基调不符）。逐条记入 "contradictions"。
- 幻觉实体（被当作已确立、却不在既定事实中、且本章也未合理引入的人名/地名/物品）：每个 -0.7；记入 "hallucinated_entities"。
- 人物口吻/立场偏移（仅当提供了 "## 人物声音基线" 区块时）：某个焦点人物在无页面理由的情况下其行为或言语与基线立场/口吻/目标矛盾：每个 -0.5（总计上限 -1.0）；逐条记入 "character_voice_drift"。未提供基线区块时，"character_voice_drift" 留空。
- 文体退化：正文出现破折号（——）串联碎句、单词短句堆叠、无标点舞台提示式断行、几乎无完整对话等"非小说"文风：每类 -1.0（总计上限 -3.0）。这是严重缺陷，不是风格。

扣分后再施加加分（叠加，总计上限 +1.5）：
- 所有大纲节拍都以具体的页面动作实现：+0.5
- 在页面上解决了先前反馈、同时保持张力与追读欲：+0.7
- 场景调度与章末手法相对最近 3 章有区分度：+0.3
- 主角有可见代价/能动时刻且带情感质地：+0.3

最终分数钳制到 [1.0, 10.0]。9.0+ 仅保留给没有关键扣分项的章节。

大纲节拍审计（必填）：
对大纲 "beats" 数组中的每个节拍，向 beats_audit 添加一条：
- "realized"：该节拍以可见动作在页面上充分实现
- "partial"：该节拍被提及，但缺乏具体场景或感官细节
- "absent"：该节拍缺失或仅在页面之外被暗示

补丁（当 score < 9 或存在任何 "partial"/"absent" 节拍时必填）：
- 输出 1-8 个外科式补丁，使其在被应用后能把章节至少提升一个档位。
- 每个补丁的 insert/after 内容必须简短（<= 200 个中文字符）且自足。
- 优先用 insert_after 添加缺失的场景/细节；优先用 replace 修正具体措辞。
- 每个补丁的 locator/before 必须逐字引用章节中确实存在的文本（连续子串，8-20 字）。
- 每个补丁必须相互独立——以任意顺序应用任意子集（或全部）都仍能产出合法文段。
- 不要链式依赖的补丁；若需要长插入，拆成多个独立 insert，定位到不同的 locator。
- 若章节已达 9+ 且无 partial/absent 节拍，可返回 "patches": []。

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
- 若不存在矛盾，返回 "contradictions": [] 与 "hallucinated_entities": []。"""

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
    from store import JsonStoryStore  # local import to avoid cycle at module load

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
        if not isinstance(conn, JsonStoryStore):
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
- 用 markdown。每一卷用 `## 第N卷：<卷名>（第X-Y章）` 作标题，章节区间明确。
- 每卷必须含小节：**卷主题 / 核心矛盾 / 阶段高潮（每15-25章一个）/ 大事件锚点（≥2-3个具体事件）/ 本卷兑现 / 重大代价 / 遗留危机**。
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
- 以完整的主谓宾句子叙事；破折号每千字不超过 3 个，不得用破折号串联碎片。
- 平均句长保持在正常小说水平（约 15-40 字），不得通篇单词短句。
- 段落是连贯成句的叙事，不是无标点断行的舞台提示。
- 保留有潜台词的人物对话。

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
) -> dict[str, Any]:
    mem = cached_memory or writing_memory_context(paths, conn, config)
    silence_threshold = int(config["novel"].get("thread_silence_threshold", 10))
    silent_threads = get_silent_threads(conn, chapter_num, silence_threshold=silence_threshold)
    preset = str(config["novel"].get("style_preset", "history"))
    preset_hint = {
        "xuanhuan_shuang": "本作为穿越爽文：payoff 维度应额外考量本章是否有明确的爽点兑现（兑现/打脸/翻盘/掌权），节奏是否够紧；但爽点须有铺垫与代价，无脑碾压应扣分。若下方 Rhythm Diagnostics 报告了爽点拖欠（chapters_since_payoff >= payoff_max_gap）而本章仍未给出兑现类 payoff，额外 -0.5。",
        "history": "本作为历史厚重题材：重视制度细节、政治博弈的真实约束与因果链的严谨。",
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
    # Character voice baseline: cross-chapter stance/voice consistency check.
    # Enabled by default for the 爽文 preset; long novel opts in via config to
    # avoid false positives until the signal is validated.
    voice_check_default = preset == "xuanhuan_shuang"
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
    user = f"""## 风格预设：{preset}
{preset_hint}

## 记忆
{mem}

## 既定事实（不得违背——任何冲突记入 "contradictions"）
{facts_block}

## 人物声音基线（跨章立场/口吻；冲突记入 "character_voice_drift"）
{voice_block}

## 上章结尾
{tail[-1500:]}

## 近期质量反馈JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## 沉默伏线JSON（沉默 >{silence_threshold} 章；核查本章是否推进了其中任何一条，或有充分理由跳过）
{json.dumps(silent_threads, ensure_ascii=False, indent=2) if silent_threads else "None"}

## 节奏诊断JSON（留意 chapters_since_payoff 与 payoff_max_gap 以判断爽点拖欠）
{json.dumps(rhythm_diagnostics(conn, config), ensure_ascii=False, indent=2)}

## 选定大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 章节正文
{chapter[:12000]}"""
    raw = call_llm(
        client, paths, config, REVIEW_SYSTEM, json_prompt(user),
        max_tokens=32000, temperature=0.2, cacheable_prefix=cacheable_prefix(paths, config),
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
            "style_audit": {"em_dash_per_kchar": 0.0, "fragment_line_ratio": 0.0, "has_full_dialogue": True},
            "contradictions": [],
            "hallucinated_entities": [],
            "character_voice_drift": [],
            "writer_directives_for_next_chapter": [],
        },
    )
    report["score"] = safe_score(report.get("score", 0))
    report.setdefault("contradictions", [])
    report.setdefault("hallucinated_entities", [])
    report.setdefault("character_voice_drift", [])

    # Objective, non-LLM style-health gate. Self-review cannot detect that the
    # prose has collapsed into telegraphic em-dash fragments because the model's
    # own voice has drifted with it. Apply a deterministic penalty and feed the
    # fixes to the next chapter's writer + this chapter's revise loop.
    if bool(config["novel"].get("style_health_enabled", True)):
        try:
            from quality import style_health

            sh = style_health(chapter, config)
            report["style_health"] = sh
            penalty = float(sh.get("penalty", 0.0))
            if penalty > 0:
                report["score"] = max(1.0, report["score"] - penalty)
                # Surface the fixes into the channels the pipeline already reads.
                rr = report.setdefault("rhythm_risks", [])
                for f in sh.get("flags", []):
                    tag = f"style:{f}"
                    if tag not in rr:
                        rr.append(tag)
                wd = report.setdefault("writer_directives_for_next_chapter", [])
                for d in sh.get("directives", []):
                    if d not in wd:
                        wd.append(d)
                log(
                    paths,
                    f"Style-health Ch{chapter_num} penalty={penalty} "
                    f"flags={sh.get('flags')} metrics={sh.get('metrics')}",
                )
                # A hard collapse must not be accepted on quality grounds alone.
                if penalty >= float(config["novel"].get("style_penalty_block", 2.0)):
                    report["accepted"] = False
                    report.setdefault("problems", []).append(
                        "STYLE: prose-health collapse detected (em-dash fragments / telegraphic lines)."
                    )
        except Exception as exc:
            log(paths, f"style_health check failed (non-fatal) Ch{chapter_num}: {exc}")

    report.setdefault("accepted", report["score"] >= float(config["novel"]["quality_threshold"]))
    # Optionally block acceptance when a HARD contradiction is detected, so the
    if bool(config["novel"].get("factcheck_hard_blocks_accept", False)):
        hard = [c for c in report.get("contradictions", []) if isinstance(c, dict) and str(c.get("severity", "")).lower() == "hard"]
        if hard:
            report["accepted"] = False
            report.setdefault("problems", []).append(
                f"FACTCHECK: {len(hard)} hard contradiction(s) with established facts must be fixed."
            )
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
    raw = call_llm(client, paths, config, STAGE_REVIEW_SYSTEM, json_prompt(user), max_tokens=12000, temperature=0.3)
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
    new_voice = call_llm(client, paths, config, VOICE_ANCHOR_SYSTEM, voice_user, max_tokens=8000, temperature=0.3)
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
    new_voices = call_llm(client, paths, config, VOICES_TABLE_SYSTEM, voices_user, max_tokens=8000, temperature=0.3)
    new_voices = normalize_text(new_voices).strip()
    if new_voices:
        write_text(paths.voices, new_voices + "\n")
        log(paths, f"Updated voices.md at Ch{chapter_num} (len={len(new_voices)})")

COLD_READER_SYSTEM = """你是一名**没有读过本书前文**、第一次拿到这一章的挑剔读者兼资深编辑。
你不知道作者的任何设定、声音锚或写作意图——你只看这一章的文字本身。
请用"陌生人视角"诚实判断这一章作为小说是否好读，重点抓两类毛病：
1. 文体是否畸形：是否大量用破折号（——）把句子切成碎片、通篇单词短句、像电报或舞台提示而不像小说？是否几乎没有正常的完整句子和对话？
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
    user = f"""## 这一章的全文（你对本书一无所知）
{chapter[:12000]}

请以陌生读者视角评估这一章。"""
    raw = call_llm(
        client, paths, config, COLD_READER_SYSTEM, json_prompt(user),
        max_tokens=2000, temperature=0.3,  # NOTE: deliberately no cacheable_prefix
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
  "directives": ["3-5 条具体指令，强制后续章节跳出微观僵局、推进到 next_anchor"]
}"""


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
        max_tokens=2500, temperature=0.3,
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
    new_plan = call_llm(client, paths, config, REPLAN_SYSTEM, user, max_tokens=16000, temperature=0.5)
    write_text(paths.volume_plan, normalize_text(new_plan) + "\n")
    db_event(conn, chapter_num, "adaptive_replan", {"reason": "metrics_degradation"})
