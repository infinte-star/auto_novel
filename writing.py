from __future__ import annotations

import json
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

from checkpoint import load_checkpoint
from config import (
    Paths,
    append_text,
    chapter_path,
    count_chars,
    normalize_chapter,
    normalize_text,
    read_text,
    safe_score,
    write_text,
)
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, memory_context, writing_memory_context
from planning import plan_score
from store import JsonStoryStore, db_event, recent_quality_feedback, store_causal_links

if TYPE_CHECKING:
    from openai import OpenAI

WRITE_SYSTEM_HISTORY = """你是一位擅长中国历史题材的长篇网文作家，风格厚重克制，兼具网文可读性。
用中文写作本章。

## 写前自我审查（必须在reasoning_content中完成，不得出现在正文）
1. 识别本章三项最高风险：
   - 重复风险：本章可能无意间复制哪个近期场景/开场方式/结尾手法？
   - 浅层执行风险：plan中哪个beat最可能变成"叙述概括"而非"戏剧化呈现"？
   - 空洞兑现风险：主角在哪里可能轻松获胜却没有代价？
2. 针对每项风险，写一条具体规避承诺（如"用茶寮而非文渊阁"、"在页面上呈现户部程序"、"让主角失去一张底牌"）。
3. 拟2个开场候选句（各一句），选出更强的一个并简要说明理由。
4. 完成1-3后才开始正式写作。

## 输出要求
- 约{chapter_words}个中文字符。
- 第一行固定格式：第{chapter_num}章 {title}
- 执行选定的plan及所有约束条件。
- plan中的高风险beats必须直接在页面上演出，不得仅作暗示或留白。
- 通过场景、选择、代价和后果，明确修复最近的质量反馈。
- 与近期章节的场景舞台、结尾方式、情感质地、推演姿态进行差异化处理。
- 保持因果关系、人物主动性、压迫-兑现节奏和悬念强度。
- 避免概括性叙事、重复的震惊反应和廉价巧合。
- 只输出章节正文，不要任何解释。

## 结构模板
- 开场钩（200-400字）：紧接上章末尾，建立本章核心问题或悬念，禁止用时间词（"翌日清晨"/"这天傍晚"等）作开场
- 场景一（1000-1500字）：主要冲突场景，含具体动作、对话与环境描写
- 场景二（800-1200字）：转折或揭示场景，推进plan中的关键beats
- 场景三（600-1000字）：决定或代价场景，呈现选择后果
- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾

## 感官纪律
- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）
- 用具体细节代替抽象描述（"墨迹未干的公文" 而非 "重要文件"）
- 季节、天气、光线作为情绪衬托，不作章节进度的计时器

## 对话比例
- 对话占全章30-50%，避免连续500字以上无对白的段落
- 每个角色的语气、用词必须反映其身份、立场和当下心理

## 时间标记禁令（核心问题）
- 严禁以"翌日清晨""这天晚上""次日黄昏""午后""深夜"等时间副词切换场景或开启段落
- 时间流逝必须通过情节动作和因果链条体现，而非显式时间标记
- 每章最多出现2个时间词，且必须与具体情节行为紧密绑定（如"赶在衙门散班前"而非单纯"傍晚"）

## 人物塑造要求
- 每个登场角色必须有具体的立场逻辑和利益驱动，不得无缘无故表忠心或反派
- 主角的成长必须来自挫败、情报或他人的推演，不能突然"顿悟"
- 对话必须含潜台词和话术攻防，不能只喊口号和表态
- 官场人物的措辞必须符合其政治处境（得势者与失势者说话方式不同）

## 情节逻辑要求
- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果
- 如有伏笔，必须在本章或后续章节可查的文本中有对应的"收线"
- 不得出现"某人神秘地笑了"类的模糊悬念代替真实信息

## 禁止模式
- 禁止"他突然意识到/恍然大悟"式的廉价顿悟
- 禁止角色连续心理独白超过150字
- 禁止用解释性叙述代替戏剧化呈现（show don't tell）
- 禁止章末用"他知道，一切才刚刚开始"之类的空洞总结
- 禁止同一章内出现超过3次相同的动作描写（如"皱眉""沉默""点头"）
- 禁止开场连续两段是环境描写，必须在第一段内有人物动作或对话
- 【文风塌缩禁令·最高优先级】禁止"句子——状态——状态"式破折号短句链；破折号每千字不超过3处，且只用于正常插入语，不得用来粘连碎片；每段至少含2个有主谓宾的完整句子，禁止整段单词短句堆叠或无标点的舞台提示式断行；句子长短交替，避免连续3句字数相近的短句。

## 本章必须满足的质量硬指标（写完前自检）
- 显性代价：主角本章至少有一次**可见的资源/政治/情感代价**（失去一张底牌、得罪一方势力、付出信任或人情），不得轻松全胜。
- 对白潜台词：本章至少有一处关键对话含**话术攻防/言外之意**（如表面奏对、暗里递价；正例："臣不敢妄言"实指"陛下先表态臣才敢接"），不得只喊口号表态。
- 差异化：禁止复用最近3章已用过的开场方式与章末钩子手法；若雷同，必须换一种结构（场景驱动↔反转↔压迫-兑现等）。"""


WRITE_SYSTEM_SHUANG = """你是一位擅长穿越爽文的中文网文作家，节奏明快、爽点密集、画面感强、读者代入感极强。
用中文写作本章。

## 写前自我审查（必须在reasoning_content中完成，不得出现在正文）
1. 识别本章三项最高风险：
   - 重复风险：本章可能无意间复制哪个近期场景/开场方式/结尾手法？
   - 浅层执行风险：plan中哪个beat最可能变成"叙述概括"而非"戏剧化呈现"？
   - 无脑碾压风险：主角在哪里可能毫无铺垫地轻松获胜、缺乏代价或对手反应？
2. 针对每项风险，写一条具体规避承诺（如"用一次失败的试探换信任"、"让赵高当场反将一军"、"现代知识落到一个具体器物/制度细节上"）。
3. 拟2个开场候选句（各一句），选出更强的一个并简要说明理由。
4. 明确本章的"爽点高潮"是哪一段（兑现/打脸/翻盘/掌权之一），它如何被前文铺垫和压迫衬托。
5. 完成1-4后才开始正式写作。

## 输出要求
- 约{chapter_words}个中文字符。
- 第一行固定格式：第{chapter_num}章 {title}
- 执行选定的plan及所有约束条件。
- plan中的高风险beats必须直接在页面上演出，不得仅作暗示或留白。
- 通过场景、选择、代价和后果，明确修复最近的质量反馈。
- 与近期章节的场景舞台、结尾方式、情感质地进行差异化处理。
- 只输出章节正文，不要任何解释。

## 爽点纪律（本类型核心）
- 本章必须有**至少1个明确的爽点高潮**：兑现、打脸、翻盘、识破阴谋或掌权之一，且落到具体动作与对手反应上。
- 压迫—兑现节奏要紧：铺垫不拖沓，先制造压迫/轻视/危机，再在高潮处一举兑现，让读者有"出了一口气"的快感。
- 主角靠"现代灵魂的先知与见识"做出超越时代的判断，但每次施展**必须有铺垫与代价**（被猜忌、暴露底牌、消耗人情），不得无脑全知全能。
- 章末必须留一个让读者想立刻看下一章的强钩子。

## 结构模板
- 开场钩（200-400字）：紧接上章末尾，立刻抛出本章核心冲突或压迫，禁止用时间词（"翌日清晨"/"这天傍晚"等）作开场
- 场景一（1000-1500字）：主要冲突/压迫场景，含具体动作、对话与环境描写
- 场景二（800-1200字）：转折或主角施展见识的场景，推进plan关键beats，埋下爽点引信
- 场景三（600-1000字）：爽点兑现/打脸/翻盘场景，呈现选择的后果与代价
- 结尾钩（200-400字）：制造下章悬念，不用总结式收尾

## 感官纪律
- 每个场景至少包含2种感官锚点（视觉/听觉/触觉/嗅觉/味觉）
- 用具体细节代替抽象描述（"竹简上未干的朱批" 而非 "重要文书"）
- 季节、天气、光线作为情绪衬托，不作章节进度的计时器

## 对话比例
- 对话占全章30-50%，避免连续500字以上无对白的段落
- 每个角色的语气、用词必须反映其身份、立场和当下心理
- 关键对话需含潜台词与话术攻防，不只喊口号

## 时间标记禁令（核心问题）
- 严禁以"翌日清晨""这天晚上""次日黄昏""午后""深夜"等时间副词切换场景或开启段落
- 时间流逝必须通过情节动作和因果链条体现，而非显式时间标记
- 每章最多出现2个时间词，且必须与具体情节行为紧密绑定

## 人物塑造要求
- 每个登场角色必须有具体的立场逻辑和利益驱动，不得无缘无故表忠心或当反派
- 主角（现代灵魂）的判断与成长必须来自现代见识、情报或挫败的推演，不能突然"顿悟"
- 对手（赵高、李斯等）要聪明、有手段、有反应，不能是任主角宰割的纸片人
- 秦制背景下的措辞与礼仪需大体得体，不出现现代名词穿帮

## 情节逻辑要求
- 每个场景的因果链条必须闭合：A发生→B感知→C决策→D行动→E后果
- 如有伏笔，必须在本章或后续章节可查的文本中有对应的"收线"
- 不得出现"某人神秘地笑了"类的模糊悬念代替真实信息

## 禁止模式
- 禁止"他突然意识到/恍然大悟"式的廉价顿悟
- 禁止角色连续心理独白超过150字
- 禁止用解释性叙述代替戏剧化呈现（show don't tell）
- 禁止章末用"一切才刚刚开始"之类的空洞总结
- 禁止同一章内出现超过3次相同的动作描写
- 禁止开场连续两段是环境描写，必须在第一段内有人物动作或对话
- 禁止主角无铺垫、无代价地碾压全场（爽要爽得有逻辑）
- 【文风塌缩禁令·最高优先级】禁止"句子——状态——状态"式破折号短句链；破折号每千字不超过3处，且只用于正常插入语，不得用来粘连碎片；每段至少含2个有主谓宾的完整句子，禁止整段单词短句堆叠或无标点的舞台提示式断行；句子长短交替，避免连续3句字数相近的短句。"""


WRITE_SYSTEM_PRESETS = {
    "history": WRITE_SYSTEM_HISTORY,
    "xuanhuan_shuang": WRITE_SYSTEM_SHUANG,
}

REVISE_SYSTEM = """你是一位中文网文修订作者。
请根据终审编辑报告修订整章。
保留标题与核心事件。不要引入新的连续性风险。
优先做有针对性的结构性修复，而非表面润色：
- 补全缺失的因果桥梁与具体场景。
- 替换重复的场景调度或章末手法。
- 让大纲节拍在页面上可见。
- 强化人物能动性、程序摩擦感与压迫-兑现节奏。
- 【文风塌缩禁令】禁止把叙述压成"句子——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。修订不得引入比原文更碎片化的文风。
只输出修订后的章节。"""

EXTRACT_SYSTEM = """你是长篇小说引擎中的事件溯源抽取器。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "title": "...",
  "events": [{"type":"plot|world|character|force|thread|item|battle|relationship","summary":"...","effects":[]}],
  "entities": [{"entity_type":"character|force|place|item|rule","name":"...","state_patch":{}}],
  "threads": [{"id":"stable-id","description":"...","status":"open|advanced|recovered|dropped","thread_type":"plot|reader_promise|character_arc|world_rule|relationship","introduced_chapter":1,"due_chapter":20,"payload":{}}],
  "causal_links": [{"from_event":"来源事件概括","to_event":"预期的未来事件或后果","link_type":"causes|enables|blocks|requires","description":"该因果关联为何存在"}],
  "metrics": {
    "payoff_type":"court_breakthrough|policy_payoff|military_victory|reveal|reversal|personnel_payoff|institutional_fix|strategic_setup|emotional",
    "conflict_type":"court|finance|military|border|famine|faction|intelligence|personnel|institution|diplomacy|civil_unrest|logistics|other",
    "tension":1-10,
    "novelty":1-10,
    "hook_strength":1-10,
    "emotional_tone":"..."
  },
  "memory_updates": {
    "bible": [],
    "characters": [],
    "timeline": [],
    "threads": []
  }
}

为每条 thread 设定 "thread_type"：
- "reader_promise"：对读者做出的、必须兑现的明确钩子/承诺（被预告的对决、立下的复仇、被埋的揭示，"他日必报此仇"式的债）。
- "character_arc"：某个人物的个人成长/转变弧线。
- "world_rule"：关于世界的、后续章节必须遵守的规则/约束。
- "relationship"：两方之间演变中的关系。
- "plot"：任何普通情节伏线（默认）。
拿不准时用 "plot"。"""

STATE_UPDATE_SYSTEM = """你负责维护一部 200 万字以上小说的简短工作状态。
只输出 markdown，不要任何解释。
要求：
- <=5000 个中文字符。
- 包含当前进度、卷/阶段目标、主角状态、关键冲突、接下来 12 章的方向。
- 近期章节摘要保持紧凑。
- 保留硬性连续性约束。"""

def carried_over_partial_beats(paths: Paths, chapter_num: int, limit: int = 6) -> list[dict[str, Any]]:
    """Return the previous chapter's partial/absent beats so the next writer can repair them.

    Reads final_review.json -> review_round0.json -> review_round1.json in order
    of preference, and returns up to `limit` entries containing
    {"beat": str, "status": "partial|absent", "evidence": str}.
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    for key in ("final_review.json", "review_round1.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        beats = data.get("beats_audit") or []
        partial: list[dict[str, Any]] = []
        for entry in beats:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).lower()
            if status not in ("partial", "absent"):
                continue
            partial.append({
                "beat": str(entry.get("beat", ""))[:300],
                "status": status,
                "evidence": str(entry.get("evidence", ""))[:200],
            })
            if len(partial) >= limit:
                break
        if partial:
            return partial
    return []


def writer_directives_for_chapter(paths: Paths, chapter_num: int, limit: int = 6) -> list[str]:
    """Return directives carried from the previous chapter's review.

    Reads the previous chapter's review (final_review.json preferred) and
    extracts a flat list of imperative strings to inject at the top of the
    current chapter's write prompt. This forms a review->writer feedback loop
    that is more concrete than plan-level required_constraints (it speaks in
    terms of execution, not strategy).
    """
    if chapter_num <= 1:
        return []
    prev = chapter_num - 1
    directives: list[str] = []
    for key in ("final_review.json", "review_round1.json", "review_round0.json"):
        data = load_checkpoint(paths, prev, key)
        if not isinstance(data, dict):
            continue
        for field in ("writer_directives_for_next_chapter", "writer_directives"):
            for item in data.get(field, []) or []:
                text = str(item).strip()
                if text and text not in directives:
                    directives.append(text)
                if len(directives) >= limit:
                    return directives
        if directives:
            return directives
    return directives


HOOK_REVISE_SYSTEM = """你是一位中文网文章末钩子专家。
只重写本章最后一段（结尾段），让结尾钩子犀利、具体，并为读者制造清晰的下一章问题。

约束：
- 不要改动重写点之前的任何内容。输出完整章节，开头与中段逐字保留原样，仅替换结尾段。
- 新结尾必须避免以下禁忌：廉价顿悟（"他突然意识到"）、总结式收尾（"一切才刚刚开始"）、抽象的伏笔。
- 新结尾应抛出一个具体、明确的问题，或设置一个下一章必须应对的具体障碍。
- 用完整句收束，禁止用破折号串联碎句或单词短句堆叠。
- 契合既定叙事声音；不要引入尚未确立的新人物或新事实。
- 替换段长度与原结尾大致相当（以用户给出的原结尾字数为准，误差 20% 以内）。"""


def revise_hook_only(
    client: OpenAI,
    paths: Paths,
    config: dict[str, Any],
    chapter: str,
    plan: dict[str, Any],
    review: dict[str, Any],
    tail_to_revise_chars: int = 400,
) -> str:
    """Rewrite only the last ~300-500 chars of the chapter to fix a weak ending hook.

    This is a much cheaper alternative to a full revise: a single small LLM call
    that the writer copies the head verbatim and only mutates the tail. Returns
    the new full chapter text.
    """
    chapter = normalize_chapter(chapter)
    n = len(chapter)
    cut = max(0, n - tail_to_revise_chars)
    # Snap cut point to a paragraph boundary if possible (look back up to 200 chars
    # for double-newline; otherwise single newline).
    snap_window = chapter[max(0, cut - 200): cut + 200]
    for marker in ("\n\n", "\n"):
        idx = snap_window.find(marker)
        if idx >= 0:
            cut = max(0, cut - 200) + idx + len(marker)
            break
    head = chapter[:cut]
    original_tail = chapter[cut:]
    user = f"""## 大纲JSON（供参考）
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 审校反馈（钩子为何偏弱）
{json.dumps({
    "hook_strength": review.get("hook_strength"),
    "rhythm_risks": review.get("rhythm_risks", []),
    "writer_directives": review.get("writer_directives_for_next_chapter", []),
}, ensure_ascii=False, indent=2)}

## 章节开头（不要改动——逐字复制）
{head}

## 待重写的当前结尾（长度 {len(original_tail)} 字）
{original_tail}

重写本章。逐字复制开头，再用一个长度与原结尾相当（约 {len(original_tail)} 字，误差 20% 以内）、更犀利的结尾替换结尾段。只输出完整章节。"""
    raw = call_llm(
        client, paths, config, HOOK_REVISE_SYSTEM, user,
        max_tokens=8000, temperature=0.55,
        cacheable_prefix=cacheable_prefix(paths, config),
    )
    new_chapter = normalize_chapter(raw)
    # Safety: if the model failed to preserve the head (e.g., truncated or
    # rewrote opening), fall back to head + new tail by splicing.
    if not new_chapter.startswith(head[: min(len(head), 200)]):
        # Try to recover by extracting the model's "new tail" — assume it's
        # the last paragraph in its output.
        from config import log as _log
        _log(paths, "hook revise: head verification failed; splicing head + model_tail")
        model_tail = new_chapter.rsplit("\n\n", 1)[-1] if "\n\n" in new_chapter else new_chapter[-tail_to_revise_chars * 2:]
        new_chapter = normalize_chapter(head.rstrip() + "\n\n" + model_tail.strip())
    return new_chapter


def write_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    decision: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
    temperature: float | None = None,
) -> str:
    title = str(plan.get("title") or f"Chapter {chapter_num}").strip()
    preset = str(config["novel"].get("style_preset", "history"))
    write_system = WRITE_SYSTEM_PRESETS.get(preset, WRITE_SYSTEM_HISTORY)
    system = write_system.format(
        chapter_words=int(config["novel"]["chapter_words"]),
        chapter_num=chapter_num,
        title=title,
    )
    mem = cached_memory or writing_memory_context(paths, conn, config)
    partial_beats = carried_over_partial_beats(paths, chapter_num)
    directives = writer_directives_for_chapter(paths, chapter_num)
    carryover_block = ""
    if partial_beats:
        carryover_block += (
            f"\n## 来自 CH{chapter_num - 1} 的关键遗留（必须在页面上处理）\n"
            f"以下节拍在上一章审校中被标记为 partial/absent。"
            f"当叙事上可行时，你必须在本章把它们落到页面上实演，"
            f"不要让其停留在暗示或页面之外。\n"
            f"{json.dumps(partial_beats, ensure_ascii=False, indent=2)}\n"
        )
    if directives:
        carryover_block += (
            f"\n## 给 CH{chapter_num} 的审校指令（必须遵守）\n"
            f"以下执行级指令来自上一章的审校者。"
            f"当与通用准则冲突时，以这些指令为准。\n"
            f"{json.dumps(directives, ensure_ascii=False, indent=2)}\n"
        )
    # Retrieval-augmented context: surface specific older facts the layered
    # memory summaries have compressed away, so long-range names/numbers/places
    # stay consistent.
    rag_block = ""
    if bool(config["novel"].get("rag_enabled", True)):
        try:
            from retrieval import retrieval_block

            rag_block = retrieval_block(paths, config, plan, chapter_num)
        except Exception as exc:
            from config import log as _log
            _log(paths, f"RAG block build failed (non-fatal) Ch{chapter_num}: {exc}")
    if rag_block:
        carryover_block += "\n" + rag_block + "\n"
    user = f"""## 记忆（事实与设定参照，不要模仿其行文风格）
{mem}
{carryover_block}
## 上章结尾
{tail[-int(config["novel"]["recent_tail_chars"]):]}

## 近期质量反馈JSON（本章必须修复；仅作修复目标，不要模仿其文风或照抄措辞）
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## 选定大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 仲裁约束JSON
{json.dumps(decision.get("required_constraints", []), ensure_ascii=False, indent=2)}

写第 {chapter_num} 章。"""
    temp = float(config["api"]["temperature"]) if temperature is None else temperature
    prefix = cacheable_prefix(paths, config)
    from config import log
    log(paths, f"write_chapter Ch{chapter_num} calling LLM with temp={temp:.2f} user_len={len(user)} system_len={len(system)}")
    raw = call_llm(client, paths, config, system, user, temperature=temp, cacheable_prefix=prefix)
    log(paths, f"write_chapter Ch{chapter_num} LLM returned {len(raw)} chars")
    return normalize_chapter(raw)

def apply_review_patches(chapter: str, patches: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Apply review-provided patches to chapter text in-place.

    Returns (new_chapter, applied_patches_with_status). Each patch entry gets an
    "applied" boolean and optionally an "error" reason if it could not be applied.
    Patches are applied in input order, each operating on the current text.
    Locators that no longer match (because an earlier patch removed/replaced the
    surrounding region) are skipped with applied=False.
    """
    text = chapter
    results: list[dict[str, Any]] = []
    for raw_patch in patches or []:
        if not isinstance(raw_patch, dict):
            results.append({"applied": False, "error": "non-dict patch", "patch": raw_patch})
            continue
        op = str(raw_patch.get("op", "")).strip().lower()
        locator = str(raw_patch.get("locator", "")).strip()
        before = str(raw_patch.get("before", "") or "").strip()
        after = str(raw_patch.get("after", "") or "")
        insert_text = str(raw_patch.get("insert", "") or "")
        entry = {**raw_patch, "applied": False}
        try:
            if op == "replace":
                target = before or locator
                if not target:
                    entry["error"] = "empty before/locator for replace"
                elif target not in text:
                    entry["error"] = "before/locator not found in chapter"
                else:
                    text = text.replace(target, after, 1)
                    entry["applied"] = True
            elif op == "insert_after":
                if not locator or locator not in text:
                    entry["error"] = "locator not found for insert_after"
                else:
                    idx = text.find(locator) + len(locator)
                    glue_before = "" if text[idx:idx+1] in {"\n", ""} else "\n\n"
                    glue_after = "" if text[idx:idx+2] == "\n\n" else "\n\n"
                    text = text[:idx] + glue_before + insert_text + glue_after + text[idx:]
                    entry["applied"] = True
            elif op == "delete":
                target = before or locator
                if not target or target not in text:
                    entry["error"] = "before/locator not found for delete"
                else:
                    text = text.replace(target, "", 1)
                    entry["applied"] = True
            else:
                entry["error"] = f"unknown op: {op!r}"
        except Exception as exc:
            entry["error"] = f"exception: {exc}"
        results.append(entry)
    return text, results


def revise_chapter(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter: str,
    review: dict[str, Any],
    plan: dict[str, Any],
    tail: str,
    cached_memory: str | None = None,
) -> str:
    # Fast path: try applying review patches directly without a full LLM rewrite.
    # Only fall back to LLM revision when patches are missing, incomplete, or fail.
    patches = review.get("patches") if isinstance(review, dict) else None
    use_patch_path = bool(config["novel"].get("revise_use_patches", True))
    if use_patch_path and isinstance(patches, list) and patches:
        patched, results = apply_review_patches(chapter, patches)
        applied = sum(1 for r in results if r.get("applied"))
        total = len(results)
        # Relaxed threshold: 1/2 of patches applied counts as success.
        # Surgical patch path is much faster than a full rewrite and the unapplied
        # patches typically address minor issues; the next review round will pick
        # up anything material that remains.
        min_apply_frac = float(config["novel"].get("revise_patch_min_frac", 0.5))
        threshold_hit = max(1, int(total * min_apply_frac + 0.999))
        if applied >= threshold_hit:
            from config import log as _log
            _log(paths, f"Revise via patches applied={applied}/{total} (>= {threshold_hit}); skipping full rewrite")
            return normalize_chapter(patched)
        else:
            from config import log as _log
            _log(paths, f"Revise patches too few hit ({applied}/{total} < {threshold_hit}); falling back to LLM rewrite")

    mem = cached_memory or writing_memory_context(paths, conn, config)
    user = f"""## 记忆
{mem}

## 上章结尾
{tail[-1500:]}

## 大纲JSON
{json.dumps(plan, ensure_ascii=False, indent=2)}

## 近期质量反馈JSON
{json.dumps(recent_quality_feedback(paths), ensure_ascii=False, indent=2)}

## 编辑报告JSON
{json.dumps(review, ensure_ascii=False, indent=2)}

## 原始章节
{chapter}

修订整章。"""
    raw = call_llm(
        client, paths, config, REVISE_SYSTEM, user,
        temperature=0.45, cacheable_prefix=cacheable_prefix(paths, config),
    )
    return normalize_chapter(raw)

def extract_events(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    cached_memory: str | None = None,
) -> dict[str, Any]:
    mem = cached_memory or memory_context(paths, conn, config)
    user = f"""## 章节前记忆
{mem}

## 第 {chapter_num} 章
{chapter[:8000]}

抽取持久的状态变化。"""
    raw = call_llm(client, paths, config, EXTRACT_SYSTEM, max_tokens=12000, user=json_prompt(user), temperature=0.2)
    return load_json_with_repair(client, paths, config, raw)

def update_structured_state(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    review: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    db_event(conn, chapter_num, "chapter_extraction", extraction)

    for event in extraction.get("events", []):
        db_event(conn, chapter_num, "story_event", event)

    for entity in extraction.get("entities", []):
        entity_type = str(entity.get("entity_type", "unknown"))
        name = str(entity.get("name", "unknown"))
        if isinstance(conn, JsonStoryStore):
            state = conn.get_entity_state(entity_type, name)
        else:
            old = conn.execute(
                "SELECT state_json FROM entities WHERE entity_type=? AND name=?",
                (entity_type, name),
            ).fetchone()
            state = json.loads(old["state_json"]) if old else {}
        patch = entity.get("state_patch") or {}
        if isinstance(patch, dict):
            state.update(patch)
        else:
            state["note"] = str(patch)
        if isinstance(conn, JsonStoryStore):
            conn.upsert_entity(entity_type, name, state, chapter_num)
        else:
            conn.execute(
                """
                INSERT INTO entities(entity_type, name, state_json, updated_chapter)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_type, name)
                DO UPDATE SET state_json=excluded.state_json, updated_chapter=excluded.updated_chapter
                """,
                (entity_type, name, json.dumps(state, ensure_ascii=False), chapter_num),
            )

    for thread in extraction.get("threads", []):
        thread_id = str(thread.get("id") or f"ch{chapter_num}-{abs(hash(json.dumps(thread, ensure_ascii=False))) % 100000}")
        if isinstance(conn, JsonStoryStore):
            conn.upsert_thread(thread_id, thread, chapter_num)
        else:
            conn.execute(
                """
                INSERT INTO open_threads(id, description, status, thread_type, introduced_chapter, due_chapter, updated_chapter, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id)
                DO UPDATE SET description=excluded.description, status=excluded.status,
                              thread_type=excluded.thread_type,
                              due_chapter=excluded.due_chapter, updated_chapter=excluded.updated_chapter,
                              payload_json=excluded.payload_json
                """,
                (
                    thread_id,
                    str(thread.get("description", "")),
                    str(thread.get("status", "open")),
                    str(thread.get("thread_type", "plot")),
                    thread.get("introduced_chapter"),
                    thread.get("due_chapter"),
                    chapter_num,
                    json.dumps(thread.get("payload", {}), ensure_ascii=False),
                ),
            )

    metrics = extraction.get("metrics") or {}
    metrics_row = {
        "chapter": chapter_num,
        "title": extraction.get("title"),
        "score": safe_score(review.get("score", 0)),
        "plan_score": plan_score(decision),
        "payoff_type": metrics.get("payoff_type"),
        "conflict_type": metrics.get("conflict_type"),
        "tension": metrics.get("tension"),
        "novelty": metrics.get("novelty"),
        "hook_strength": metrics.get("hook_strength"),
        "emotional_tone": metrics.get("emotional_tone"),
        "accepted": 1 if review.get("accepted") else 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(conn, JsonStoryStore):
        conn.upsert_metrics(chapter_num, metrics_row)
    else:
        conn.execute(
            """
            INSERT INTO chapter_metrics(
                chapter, title, score, plan_score, payoff_type, conflict_type, tension,
                novelty, hook_strength, emotional_tone, accepted, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter) DO UPDATE SET
                title=excluded.title, score=excluded.score, plan_score=excluded.plan_score,
                payoff_type=excluded.payoff_type, conflict_type=excluded.conflict_type,
                tension=excluded.tension, novelty=excluded.novelty, hook_strength=excluded.hook_strength,
                emotional_tone=excluded.emotional_tone, accepted=excluded.accepted
            """,
            (
                metrics_row["chapter"],
                metrics_row["title"],
                metrics_row["score"],
                metrics_row["plan_score"],
                metrics_row["payoff_type"],
                metrics_row["conflict_type"],
                metrics_row["tension"],
                metrics_row["novelty"],
                metrics_row["hook_strength"],
                metrics_row["emotional_tone"],
                metrics_row["accepted"],
                metrics_row["created_at"],
            ),
        )
        conn.commit()

    updates = extraction.get("memory_updates") or {}
    append_memory(paths.bible, chapter_num, updates.get("bible") or [])
    append_memory(paths.characters, chapter_num, updates.get("characters") or [])
    append_memory(paths.timeline, chapter_num, updates.get("timeline") or [])
    append_memory(paths.threads, chapter_num, updates.get("threads") or [])

    store_causal_links(conn, chapter_num, extraction.get("causal_links") or [])

def append_memory(path: Path, chapter_num: int, items: list[Any]) -> None:
    if not items:
        return
    existing = read_text(path)
    section_header = f"## Ch{chapter_num}"
    if section_header in existing:
        return
    existing_bullets = set()
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            existing_bullets.add(stripped[2:].strip())
    fresh = []
    for item in items:
        text = str(item).strip()
        if not text or text in existing_bullets:
            continue
        fresh.append(text)
        existing_bullets.add(text)
    if not fresh:
        return
    append_text(path, f"\n\n{section_header}\n" + "\n".join(f"- {t}" for t in fresh) + "\n")

STATE_DYNAMIC_SECTIONS_SYSTEM = """你只生成一部长篇小说工作状态中的两个动态小节。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容：
{
  "protagonist_state": "<=600 个中文字符 markdown：主角当前的目标、资源、恐惧、秘密、持续的压力，以及尚未决断的关键决定。须反映本章带来的变化。>",
  "next_12_directions": ["10-12 条针对后续章节的具体指令；每条一句中文，明确指出具体必须发生什么，而非抽象主题"]
}
约束：
- protagonist_state 须自足（新读者可据此接续）。避免含糊措辞。
- next_12_directions 必须是具体可执行的指令，而非情节主题。"""


def _render_state_md_template(
    paths: Paths,
    conn: Any,
    chapter_num: int,
    extraction: dict[str, Any],
    protagonist_state: str,
    next_directions: list[str],
) -> str:
    """Compose the new state.md deterministically.

    The structure follows what readers expect: progress meta, recent chapter
    summaries (5), key entity states, active threads (open), and the LLM-only
    sections (protagonist_state, next_12_directions).
    """
    from store import recent_events, recent_metrics

    total_chars = count_chars(paths.book)
    metrics = recent_metrics(conn, 5)
    threads_text = read_text(paths.threads).strip()

    # Last 5 chapter title+key payoff
    summary_lines: list[str] = []
    for m in metrics:
        ch = m.get("chapter")
        title = m.get("title") or ""
        score = m.get("score")
        tone = m.get("emotional_tone") or ""
        payoff = m.get("payoff_type") or ""
        summary_lines.append(f"- Ch{ch} 「{title}」 score={score} payoff={payoff} tone={tone}")

    # Pull events from this chapter's extraction
    this_chapter_events: list[str] = []
    for ev in extraction.get("events", [])[:8]:
        s = str(ev.get("summary", "")).strip()
        if s:
            this_chapter_events.append(f"- {s[:200]}")

    next_dir_lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(next_directions[:12]))

    parts: list[str] = [
        f"# 第{chapter_num}章后状态快照",
        f"\n## 进度\n- 总字数：{total_chars}\n- 最新章节：Ch{chapter_num} 「{extraction.get('title', '')}」",
        "\n## 近期章节（最新在前）\n" + ("\n".join(summary_lines) if summary_lines else "_(无)_"),
        "\n## 最新章节关键事件\n" + ("\n".join(this_chapter_events) if this_chapter_events else "_(无)_"),
        "\n## 主角状态\n" + (protagonist_state.strip() or "_(空)_"),
        "\n## 接下来12章方向\n" + (next_dir_lines or "_(无)_"),
        "\n## 活跃伏线\n" + (threads_text[:4000] if threads_text else "_(无)_"),
    ]
    return "\n".join(parts) + "\n"


def update_state_file(
    client: OpenAI,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
    chapter_num: int,
    chapter: str,
    extraction: dict[str, Any],
) -> None:
    if paths.state.exists():
        shutil.copy2(paths.state, paths.state.with_suffix(".md.bak"))

    template_mode = bool(config["novel"].get("state_template_mode", True))
    if not template_mode:
        # Legacy path: full LLM regeneration (kept as fallback).
        user = f"""## 当前状态
{read_text(paths.state)}

## 记忆上下文
{memory_context(paths, conn, config)}

## 抽取JSON
{json.dumps(extraction, ensure_ascii=False, indent=2)}

## 当前总字数
{count_chars(paths.book)}

## 近期章节正文
{chapter[:5000]}

在第 {chapter_num} 章之后更新 state.md。"""
        new_state = call_llm(client, paths, config, STATE_UPDATE_SYSTEM, user, max_tokens=12000, temperature=0.25)
        write_text(paths.state, normalize_text(new_state) + "\n")
        return

    # Template mode: only ask LLM for the 2 dynamic sections, then deterministically
    # render the full state.md. This drops LLM output from ~12K tokens to ~2-3K.
    current_state_excerpt = read_text(paths.state)
    if len(current_state_excerpt) > 3000:
        current_state_excerpt = current_state_excerpt[:3000] + "\n...[truncated]"
    user = f"""## 上一版主角状态（用于连贯）
{current_state_excerpt}

## 来自第 {chapter_num} 章的抽取
{json.dumps(extraction, ensure_ascii=False, indent=2)}

## 最新章节结尾（最后 2500 字，提供新鲜细节）
{chapter[-2500:]}

只输出含 protagonist_state 与 next_12_directions 的 JSON。"""
    try:
        raw = call_llm(
            client, paths, config, STATE_DYNAMIC_SECTIONS_SYSTEM, json_prompt(user),
            max_tokens=4000, temperature=0.25,
            cacheable_prefix=cacheable_prefix(paths, config),
        )
        data = load_json_with_repair(
            client, paths, config, raw,
            fallback={"protagonist_state": "", "next_12_directions": []},
        )
    except Exception as exc:
        from config import log as _log
        _log(paths, f"State dynamic sections LLM failed (non-fatal); using empty fallback: {exc}")
        data = {"protagonist_state": "", "next_12_directions": []}

    protagonist_state = str(data.get("protagonist_state", "")).strip()
    next_directions = [str(d).strip() for d in (data.get("next_12_directions") or []) if str(d).strip()]
    new_state = _render_state_md_template(
        paths, conn, chapter_num, extraction, protagonist_state, next_directions
    )
    write_text(paths.state, new_state)

def save_chapter(paths: Paths, chapter_num: int, chapter: str, review: dict[str, Any], plan: dict[str, Any]) -> None:
    chapter = normalize_chapter(chapter)
    if len(chapter.strip()) < 500:
        raise RuntimeError(
            f"Refusing to save Ch{chapter_num}: only {len(chapter.strip())} chars "
            f"(likely provider refusal or empty response). Preview: {chapter[:200]!r}"
        )
    write_text(chapter_path(paths, chapter_num), chapter)
    append_text(paths.book, "\n\n" + chapter)
    # Incrementally index the saved chapter for retrieval (RAG). Best-effort.
    try:
        from retrieval import index_chapter

        index_chapter(paths, chapter_num, chapter)
    except Exception:
        pass
    append_text(
        paths.logs_dir / "reviews.jsonl",
        json.dumps(
            {
                "chapter": chapter_num,
                "score": review.get("score"),
                "accepted": review.get("accepted"),
                "problems": review.get("problems", []),
                "continuity_risks": review.get("continuity_risks", []),
                "plan_title": plan.get("title"),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        )
        + "\n",
    )
