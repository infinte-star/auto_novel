"""Post-completion refine pass.

After the main pipeline reaches `target_words`, this module re-reads the
already-written chapters in 5-chapter groups, lets the LLM diagnose what
each group needs (light polish / medium restructure / deep rewrite), and
emits refined chapters into a parallel `chapters_refined/` directory plus
a consolidated `book_refined.md`.

The original `chapters/` directory and `book.md` are NEVER modified.

Design:
- Diagnose stage: LLM reads the 5 chapters + bible/characters/threads,
  decides per-group refine intensity, AND chooses which extra anchor
  chapters from elsewhere in the book to pull in for context (capped).
- Refine stage: LLM rewrites each chapter under the chosen intensity,
  returning the full chapter text. Original chapter is preserved.
- Checkpoint per group so the pass is resumable.
- All output under `chapters_refined/` + `book_refined.md`; nothing else
  is mutated.

Entry point: refine_book(client, paths, conn, config). Called from
pipeline.main() after target_words is reached, gated by config flag
`novel.refine_after_complete` (default off).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    Paths,
    chapter_path,
    find_last_chapter,
    log,
    normalize_chapter,
    read_text,
    safe_score,
    write_text,
)
from llm import call_llm, json_prompt, load_json_with_repair


REFINED_DIR_NAME = "chapters_refined"
REFINED_BOOK = "book_refined.md"
REFINE_LOG_NAME = "refine.log.jsonl"

GROUP_SIZE = 5
DEFAULT_MAX_EXTRA_ANCHORS = 4
DEFAULT_DIAGNOSE_MAX_TOKENS = 4000
DEFAULT_REFINE_MAX_TOKENS = 16000
DEFAULT_MIN_KEEP_RATIO = 0.6  # refined chapter cannot shrink below 60% of original


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def refined_dir(paths: Paths) -> Path:
    # Derive the refined-chapters dir from the configured chapters dir so that a
    # separate novel (e.g. chapters_fusu/) refines into its own sibling
    # (chapters_fusu_refined/) instead of clobbering the default chapters_refined/.
    base = paths.chapters_dir
    return base.parent / f"{base.name}_refined"

def refined_chapter_path(paths: Paths, chapter_num: int) -> Path:
    return refined_dir(paths) / f"{chapter_num:04d}.md"

def refined_book_path(paths: Paths) -> Path:
    # Derive the refined book name from the configured book file so a separate
    # novel (book_fusu.md) refines into book_fusu_refined.md rather than the
    # default book_refined.md.
    book = paths.book
    return book.parent / f"{book.stem}_refined{book.suffix}"

def refine_checkpoint_dir(paths: Paths) -> Path:
    return paths.logs_dir / "refine"

def group_checkpoint_path(paths: Paths, group_start: int) -> Path:
    return refine_checkpoint_dir(paths) / f"group_{group_start:04d}.json"

def refine_log_path(paths: Paths) -> Path:
    return paths.logs_dir / REFINE_LOG_NAME


# ---------------------------------------------------------------------------
# Group iteration
# ---------------------------------------------------------------------------

def iter_groups(last_chapter: int, group_size: int = GROUP_SIZE) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive chapter ranges, e.g. [(1,5), (6,10), ...].
    The final group may be smaller than group_size."""
    if last_chapter < 1:
        return []
    groups: list[tuple[int, int]] = []
    for start in range(1, last_chapter + 1, group_size):
        end = min(start + group_size - 1, last_chapter)
        groups.append((start, end))
    return groups


def load_group_text(paths: Paths, start: int, end: int) -> list[tuple[int, str]]:
    """Load chapters [start, end] from `chapters/`. Skip missing files (which
    would indicate a partial run)."""
    out: list[tuple[int, str]] = []
    for n in range(start, end + 1):
        p = chapter_path(paths, n)
        if p.exists():
            out.append((n, read_text(p)))
    return out


# ---------------------------------------------------------------------------
# Diagnose stage
# ---------------------------------------------------------------------------

DIAGNOSE_SYSTEM_HISTORY = """你是一位资深历史题材网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用"翌日清晨""这天晚上""午后""次日黄昏"等时间词切换场景？是否每章出现3次以上时间标记？时间词是否脱离情节逻辑单独使用？
2. **文风塌缩（破折号碎片化）**：是否大量出现"句子——状态——状态"式破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
3. **文笔和叙事风格**：是否过于白话或缺乏历史感？对话是否缺乏潜台词和话术攻防？是否有"show don't tell"问题？
4. **情节逻辑**：因果链条是否完整？是否有情节跳跃或交代不清的转场？伏笔是否有对应收线？
5. **人物塑造**：人物行为是否符合其立场和利益？主角的成长是否来自具体的挫败或推演？配角是否有独立的行动逻辑？
6. **节奏问题**：章节是否过于碎片化（每次跳转都用时间词）？压迫与兑现的比例是否失衡？

你需要：
1. 阅读这一组章节，针对上述5个维度识别具体问题。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点；指明章节号即可，最多4个。

只输出JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1（注明维度和具体位置）", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（50字内，必须具体指出问题）"}
  ],
  "extra_anchors": [<int>, ...]  // 可空数组，最多4个章节号
}
"""

DIAGNOSE_SYSTEM_SHUANG = """你是一位资深穿越爽文网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理与动作描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用"翌日清晨""这天晚上""午后""次日黄昏"等时间词切换场景？是否每章出现3次以上时间标记？
2. **文风塌缩（破折号碎片化）**：是否大量出现"句子——状态——状态"式破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
3. **爽点密度与兑现**：本章是否有明确的爽点高潮（兑现/打脸/翻盘/识破阴谋/掌权）？爽点是否落到具体动作与对手反应上、还是流于概括？压迫—兑现的节奏是否够紧？
4. **无脑碾压风险**：主角的胜利是否有铺垫与代价（被猜忌、暴露底牌、消耗人情）？是否存在毫无铺垫的全知全能或对手沦为纸片人？
5. **现代灵魂代入**：主角的判断是否来自现代见识/情报/挫败的推演，而非"突然顿悟"？是否出现现代名词穿帮、破坏代入感？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有让读者想立刻看下一章的强钩子？

你需要：
1. 阅读这一组章节，针对上述5个维度识别具体问题。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点；指明章节号即可，最多4个。

只输出JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1（注明维度和具体位置）", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（50字内，必须具体指出问题）"}
  ],
  "extra_anchors": [<int>, ...]  // 可空数组，最多4个章节号
}
"""

DIAGNOSE_SYSTEM_SYSTEM_STREAM = """你是一位资深系统流网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理与动作描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用时间词切换场景、每章出现3次以上时间标记？
2. **文风塌缩（破折号碎片化）**：是否大量出现破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
3. **系统反馈节奏**：本章是否有可见的系统反馈（面板/任务/奖励/数值升级/解锁）？升级与解锁是否有节奏感和成就感、还是流于概括？
4. **金手指代价与平衡**：系统能力是否有代价、冷却或限制？是否出现无脑刷数值、金手指降智解题、成长毫无张力？
5. **代入与目标**：主角的目标与成长动机是否清晰，读者是否有持续追读的动力？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有强钩子？

你需要：
1. 阅读这一组章节，针对上述维度识别具体问题。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点；指明章节号即可，最多4个。

只输出JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1（注明维度和具体位置）", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（50字内，必须具体指出问题）"}
  ],
  "extra_anchors": [<int>, ...]
}
"""

DIAGNOSE_SYSTEM_URBAN_ABILITY = """你是一位资深都市异能/重生题材网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理与动作描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用时间词切换场景、每章出现3次以上时间标记？
2. **文风塌缩（破折号碎片化）**：是否大量出现破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
3. **打脸与碾压兑现**：本章打脸/资源碾压/身份反差的爽点是否落到具体动作与对手反应上？压迫—兑现节奏是否够紧？
4. **对手智商与铺垫**：打脸是否有铺垫，对手反应是否合理？是否出现配角降智捧哏、爽点凭空降临？
5. **代入感**：主角的先知/重生优势是否自然融入推演，而非全知全能？情绪与处境是否清晰？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有强钩子？

你需要：
1. 阅读这一组章节，针对上述维度识别具体问题。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点；指明章节号即可，最多4个。

只输出JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1（注明维度和具体位置）", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（50字内，必须具体指出问题）"}
  ],
  "extra_anchors": [<int>, ...]
}
"""

DIAGNOSE_SYSTEM_ROMANCE_FEMALE = """你是一位资深女频言情/宠文网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理与情绪描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用时间词切换场景、每章出现3次以上时间标记？
2. **文风塌缩（破折号碎片化）**：是否大量出现破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
3. **情绪张力与关系弧**：本章关系是否有实质推进（拉近/误会/和解）？甜虐节奏是否得当？情绪是否由具体事件支撑而非凭空悬浮？
4. **对手戏化学反应**：男女主互动是否有潜台词与张力？是否流于直白或工具化？
5. **配角与代入**：配角是否沦为工具人？女主（或主角）的处境、动机、情绪是否清晰可代入？
6. **节奏与钩子**：章节是否拖沓或情绪空转？章末是否有让人追读的情绪钩子？

你需要：
1. 阅读这一组章节，针对上述维度识别具体问题。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点；指明章节号即可，最多4个。

只输出JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1（注明维度和具体位置）", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（50字内，必须具体指出问题）"}
  ],
  "extra_anchors": [<int>, ...]
}
"""

DIAGNOSE_SYSTEM_WANZU_XUANHUAN = """你是一位资深现代玄幻/万族争锋题材网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理与动作描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用时间词切换场景、每章出现3次以上时间标记？
2. **文风塌缩（破折号碎片化）**：是否大量出现破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
3. **境界/战力体系**：境界与战力是否清晰可预期、前后一致？战力跨度是否失控、自相矛盾？
4. **斗法画面与张力**：斗法/天骄争锋/境界突破是否有画面感和热血张力，还是流于概括陈述？
5. **力量解题合理性**：主角的取胜是否正比于此前规则铺垫（Sanderson 第一/二定律）？是否凭空开挂？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有强钩子？

你需要：
1. 阅读这一组章节，针对上述维度识别具体问题。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点；指明章节号即可，最多4个。

只输出JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1（注明维度和具体位置）", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（50字内，必须具体指出问题）"}
  ],
  "extra_anchors": [<int>, ...]
}
"""

DIAGNOSE_SYSTEM_PRESETS = {
    "history": DIAGNOSE_SYSTEM_HISTORY,
    "xuanhuan_shuang": DIAGNOSE_SYSTEM_SHUANG,
    "system_stream": DIAGNOSE_SYSTEM_SYSTEM_STREAM,
    "urban_ability": DIAGNOSE_SYSTEM_URBAN_ABILITY,
    "romance_female": DIAGNOSE_SYSTEM_ROMANCE_FEMALE,
    "wanzu_xuanhuan": DIAGNOSE_SYSTEM_WANZU_XUANHUAN,
}

REFINE_SYSTEM_BASE_HISTORY = """你是一位精调历史题材中文网文的资深编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持markdown风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。

## 精调核心原则
1. **消除时间词依赖**：删除或替换所有纯粹用于切换场景的时间词（"翌日清晨""这天晚上"等），改用情节动作和因果链条体现时间流逝。
2. **强化文笔质感**：增加历史氛围感，对话必须有潜台词和话术攻防，避免直白说教。
3. **修复人物逻辑**：每个角色的行动必须有具体的立场和利益驱动；主角的判断必须来自具体的信息或挫败，不能"突然顿悟"。
4. **补全因果链条**：A→B→C的因果链必须在页面上清晰呈现，不得有跳跃式转场。
5. **健康文风**：禁止把叙述压成"短语——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。
6. **保持叙事密度**：不得大幅压缩原章节字数（保留80%以上），修改后的篇幅可适当扩展但不超过原文1.5倍。
"""

REFINE_SYSTEM_BASE_SHUANG = """你是一位精调穿越爽文的资深网文编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持markdown风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。

## 精调核心原则
1. **消除时间词依赖**：删除或替换所有纯粹用于切换场景的时间词（"翌日清晨""这天晚上"等），改用情节动作和因果链条体现时间流逝。
2. **强化爽点兑现**：让本章的爽点高潮（兑现/打脸/翻盘/掌权）落到具体动作与对手反应上，压迫—兑现节奏要紧，铺垫不拖沓，给读者"出一口气"的快感。
3. **修复无脑碾压**：主角每次施展现代见识都要有铺垫与代价；对手要聪明、有反应，不能是任人宰割的纸片人。
4. **保住代入感**：主角的判断必须来自现代见识/情报/挫败的推演，不能"突然顿悟"；不得出现现代名词穿帮，秦制背景措辞需大体得体。
5. **补全因果链条**：A→B→C的因果链必须在页面上清晰呈现，不得有跳跃式转场。
6. **健康文风**：禁止把叙述压成"短语——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。
7. **保持叙事密度**：不得大幅压缩原章节字数（保留80%以上），修改后的篇幅可适当扩展但不超过原文1.5倍。
"""

REFINE_SYSTEM_BASE_SYSTEM_STREAM = """你是一位精调系统流网文的资深编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持markdown风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。

## 精调核心原则
1. **消除时间词依赖**：删除或替换所有纯粹用于切换场景的时间词，改用情节动作和因果链条体现时间流逝。
2. **强化系统反馈**：让本章的系统反馈（面板/任务/奖励/数值升级/解锁）落到具体场景，升级与解锁要有节奏感和成就感，避免流于概括。
3. **平衡金手指**：系统能力须有代价、冷却或限制；删改无脑刷数值、金手指降智解题的段落，让成长有张力。
4. **保住代入感**：主角目标与成长动机清晰，读者有持续追读的动力。
5. **补全因果链条**：A→B→C的因果链必须在页面上清晰呈现，不得有跳跃式转场。
6. **健康文风**：禁止把叙述压成"短语——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。
7. **保持叙事密度**：不得大幅压缩原章节字数（保留80%以上），修改后的篇幅可适当扩展但不超过原文1.5倍。
"""

REFINE_SYSTEM_BASE_URBAN_ABILITY = """你是一位精调都市异能/重生题材网文的资深编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持markdown风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。

## 精调核心原则
1. **消除时间词依赖**：删除或替换所有纯粹用于切换场景的时间词，改用情节动作和因果链条体现时间流逝。
2. **强化打脸兑现**：让本章打脸/资源碾压/身份反差的爽点落到具体动作与对手反应上，压迫—兑现节奏要紧，给读者出气的快感。
3. **修复对手降智**：打脸要有铺垫，对手要聪明、有合理反应，不能是任人宰割的纸片人或凭空降临的爽点。
4. **保住代入感**：主角的先知/重生优势要自然融入推演，而非全知全能；情绪与处境清晰。
5. **补全因果链条**：A→B→C的因果链必须在页面上清晰呈现，不得有跳跃式转场。
6. **健康文风**：禁止把叙述压成"短语——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。
7. **保持叙事密度**：不得大幅压缩原章节字数（保留80%以上），修改后的篇幅可适当扩展但不超过原文1.5倍。
"""

REFINE_SYSTEM_BASE_ROMANCE_FEMALE = """你是一位精调女频言情/宠文的资深编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持markdown风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。

## 精调核心原则
1. **消除时间词依赖**：删除或替换所有纯粹用于切换场景的时间词，改用情节动作和情绪推进体现时间流逝。
2. **强化情绪张力**：让本章关系弧有实质推进（拉近/误会/和解），甜虐节奏得当，情绪由具体事件支撑而非凭空悬浮。
3. **打磨对手戏**：男女主互动要有潜台词与化学反应，避免直白或工具化的对白。
4. **去工具人**：让配角有独立动机；主角处境、动机、情绪清晰可代入。
5. **补全因果链条**：情绪转折与关系变化必须有页面上可见的事件支撑，不得有跳跃式转场。
6. **健康文风**：禁止把叙述压成"短语——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。
7. **保持叙事密度**：不得大幅压缩原章节字数（保留80%以上），修改后的篇幅可适当扩展但不超过原文1.5倍。
"""

REFINE_SYSTEM_BASE_WANZU_XUANHUAN = """你是一位精调现代玄幻/万族争锋题材网文的资深编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持markdown风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。

## 精调核心原则
1. **消除时间词依赖**：删除或替换所有纯粹用于切换场景的时间词，改用情节动作和因果链条体现时间流逝。
2. **梳理境界体系**：确保境界与战力清晰可预期、前后一致，修复战力跨度失控或自相矛盾之处。
3. **强化斗法画面**：让斗法/天骄争锋/境界突破有画面感和热血张力，避免流于概括陈述。
4. **力量解题合理**：主角取胜须正比于此前规则铺垫（Sanderson 第一/二定律），删改凭空开挂的段落。
5. **补全因果链条**：A→B→C的因果链必须在页面上清晰呈现，不得有跳跃式转场。
6. **健康文风**：禁止把叙述压成"短语——状态——状态"式破折号碎片链；破折号每千字不超过3处；以完整主谓宾句子和有潜台词的对话为主，禁止单词短句堆叠或舞台提示式断行。
7. **保持叙事密度**：不得大幅压缩原章节字数（保留80%以上），修改后的篇幅可适当扩展但不超过原文1.5倍。
"""

REFINE_SYSTEM_BASE_PRESETS = {
    "history": REFINE_SYSTEM_BASE_HISTORY,
    "xuanhuan_shuang": REFINE_SYSTEM_BASE_SHUANG,
    "system_stream": REFINE_SYSTEM_BASE_SYSTEM_STREAM,
    "urban_ability": REFINE_SYSTEM_BASE_URBAN_ABILITY,
    "romance_female": REFINE_SYSTEM_BASE_ROMANCE_FEMALE,
    "wanzu_xuanhuan": REFINE_SYSTEM_BASE_WANZU_XUANHUAN,
}

INTENSITY_INSTRUCTIONS = {
    "polish": (
        "本轮精调强度=polish：修润错别字、重复词、口癖、不通顺句、连贯性瑕疵，"
        "重点消除多余的时间词（翌日、傍晚等纯场景切换用法）。"
        "保留原情节、对话顺序、段落结构。不得删减超过10%的字数。"
    ),
    "restructure": (
        "本轮精调强度=restructure：允许重写段落、合并/拆分场景、调整节奏；"
        "重点消除时间词切换、补强对话潜台词、修复人物行动逻辑。"
        "不得改变本章标题、关键情节点、人物决策、章末状态。"
    ),
    "rewrite": (
        "本轮精调强度=rewrite：允许重新设计场景顺序、改写大段叙事、补充心理与环境描写。"
        "重点：1)消除所有时间词场景切换；2)重写对话使其有话术攻防；"
        "3)为人物行动补充利益驱动；4)修复因果链断裂。"
        "保持每章的核心目标和章末状态不变。"
    ),
}


def diagnose_group(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    group_chapters: list[tuple[int, str]],
    last_chapter: int,
) -> dict[str, Any]:
    """Ask the LLM to diagnose a group and pick per-chapter intensity + anchors."""
    bible = read_text(paths.bible)
    characters = read_text(paths.characters)
    threads = read_text(paths.threads)

    chapter_blocks = []
    for num, text in group_chapters:
        chapter_blocks.append(f"### Ch{num}\n{text.strip()}")
    chapters_text = "\n\n".join(chapter_blocks)

    nums = [n for n, _ in group_chapters]
    start, end = nums[0], nums[-1]

    ending_note = ""
    if bool(config["novel"].get("ending_aware", True)) and last_chapter in nums:
        ending_note = f"""
## 终章提示
Ch{last_chapter} 是全书最后一章，应作为结局：必须收束主线、给完成感，不得以未解决的新悬念/新危机作结。诊断时若发现终章是开放式悬念结尾（如以新急报/新敌人/新危机收尾），应判为问题，并在该章 focus 中要求改为收束式结尾。
"""

    user = f"""## 任务
诊断小说第 Ch{start}-Ch{end} 这一组章节的问题，决定每章精调强度，并指出还需要哪些其他章节作为锚点。
本小说共 {last_chapter} 章，可选锚点章节号范围 1..{last_chapter}（不在本组之内的章节）。
{ending_note}
## 世界观 Bible
{bible[:8000]}

## 主要人物
{characters[:8000]}

## 主线 Threads
{threads[:6000]}

## 待诊断章节
{chapters_text}
"""
    preset = str(config["novel"].get("style_preset", "history"))
    diagnose_system = DIAGNOSE_SYSTEM_PRESETS.get(preset, DIAGNOSE_SYSTEM_HISTORY)
    raw = call_llm(
        client,
        paths,
        config,
        diagnose_system,
        json_prompt(user),
        max_tokens=int(config["novel"].get("refine_diagnose_max_tokens", DEFAULT_DIAGNOSE_MAX_TOKENS)),
        temperature=0.3,
    )
    data = load_json_with_repair(client, paths, config, raw, fallback={
        "group_summary": "",
        "issues": [],
        "per_chapter": [{"chapter": n, "intensity": "polish", "focus": ""} for n in nums],
        "extra_anchors": [],
    })
    # Validate / clamp.
    valid_intensities = {"polish", "restructure", "rewrite"}
    per_chapter = data.get("per_chapter") or []
    cleaned_per_chapter: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in per_chapter:
        if not isinstance(item, dict):
            continue
        try:
            ch = int(item.get("chapter"))
        except (TypeError, ValueError):
            continue
        if ch not in nums or ch in seen:
            continue
        seen.add(ch)
        intensity = str(item.get("intensity", "polish")).strip().lower()
        if intensity not in valid_intensities:
            intensity = "polish"
        focus = str(item.get("focus", "")).strip()[:200]
        cleaned_per_chapter.append({"chapter": ch, "intensity": intensity, "focus": focus})
    # Ensure every chapter in the group has an entry.
    for n in nums:
        if n not in seen:
            cleaned_per_chapter.append({"chapter": n, "intensity": "polish", "focus": ""})
    cleaned_per_chapter.sort(key=lambda x: x["chapter"])

    max_anchors = int(config["novel"].get("refine_max_extra_anchors", DEFAULT_MAX_EXTRA_ANCHORS))
    extra = data.get("extra_anchors") or []
    anchors: list[int] = []
    if isinstance(extra, list):
        for v in extra:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= iv <= last_chapter and iv not in nums and iv not in anchors:
                anchors.append(iv)
            if len(anchors) >= max_anchors:
                break

    return {
        "group_summary": str(data.get("group_summary", "")).strip()[:500],
        "issues": [str(i) for i in (data.get("issues") or [])][:20],
        "per_chapter": cleaned_per_chapter,
        "extra_anchors": anchors,
    }


# ---------------------------------------------------------------------------
# Refine stage (one chapter at a time)
# ---------------------------------------------------------------------------

def _summarize_chapter(text: str, target_chars: int = 600) -> str:
    """Cheap local summary: take first + last few hundred chars of a chapter as anchor."""
    body = text.strip()
    if len(body) <= target_chars:
        return body
    head = body[: target_chars // 2]
    tail = body[-target_chars // 2 :]
    return f"{head}\n...（中间省略）...\n{tail}"


def refine_one_chapter(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    intensity: str,
    focus: str,
    group_chapters: list[tuple[int, str]],
    extra_anchors: list[tuple[int, str]],
    diagnosis: dict[str, Any],
    is_finale: bool = False,
) -> str:
    """Ask the LLM to refine a single chapter at the chosen intensity."""
    original = ""
    for num, text in group_chapters:
        if num == chapter_num:
            original = text
            break
    if not original.strip():
        raise RuntimeError(f"Refine: chapter Ch{chapter_num} text missing")

    bible = read_text(paths.bible)
    characters = read_text(paths.characters)
    voice = read_text(paths.voice).strip()

    # Neighbours within group, summarised. The target chapter itself stays full.
    neighbour_blocks: list[str] = []
    for num, text in group_chapters:
        if num == chapter_num:
            continue
        rel = "上文" if num < chapter_num else "下文"
        neighbour_blocks.append(f"### Ch{num} ({rel}, 摘要)\n{_summarize_chapter(text)}")

    anchor_blocks: list[str] = []
    for num, text in extra_anchors:
        anchor_blocks.append(f"### Ch{num} (远端锚点, 摘要)\n{_summarize_chapter(text)}")

    intensity_instr = INTENSITY_INSTRUCTIONS.get(intensity, INTENSITY_INSTRUCTIONS["polish"])
    issues_text = "\n".join(f"- {i}" for i in diagnosis.get("issues", [])[:10]) or "（无）"
    group_summary = diagnosis.get("group_summary", "")

    ending_block = ""
    if is_finale:
        ending_block = """
## 终章收束要求（硬性）
本章是全书最后一章。精调时若结尾是抛给读者的新悬念/新危机（如新急报、新敌人、未解决的反转），必须改写为收束式结局：正面兑现主线、给情绪落点与主题呼应；可保留一句远景余韵，但不得制造"必须看下一章"的新问题。
"""

    user = f"""## 精调任务
请精调小说第 Ch{chapter_num} 章。

{intensity_instr}
{ending_block}
## 本章焦点
{focus or "（按编辑判断处理）"}

## 本组整体诊断
{group_summary}

## 本组识别问题
{issues_text}

## 世界观 Bible（节选）
{bible[:5000]}

## 主要人物（节选）
{characters[:5000]}

## 叙事声音基线（必须遵守的健康文风护栏；精调后的文风须符合此基线，不得引入破折号碎片）
{voice[:3000] if voice else "（无——以完整成句的小说文风为准，禁止破折号碎片链）"}

{chr(10).join(neighbour_blocks)}

{chr(10).join(anchor_blocks)}

## 待精调的原章节 Ch{chapter_num}
{original.strip()}

## 输出要求
- 输出修改后的完整章节正文（中文）。
- 第一行保留章节标题（如原文有）。
- 不要解释，不要 JSON，不要 markdown 围栏。
"""
    preset = str(config["novel"].get("style_preset", "history"))
    refine_system = REFINE_SYSTEM_BASE_PRESETS.get(preset, REFINE_SYSTEM_BASE_HISTORY)
    refined = call_llm(
        client,
        paths,
        config,
        refine_system,
        user,
        max_tokens=int(config["novel"].get("refine_chapter_max_tokens", DEFAULT_REFINE_MAX_TOKENS)),
        temperature=float(config["novel"].get("refine_temperature", 0.5)),
    )
    refined = normalize_chapter(refined)
    return refined


def _refined_text_acceptable(
    original: str, refined: str, config: dict[str, Any], intensity: str = "restructure"
) -> tuple[bool, str]:
    """Sanity-check the refined output. Returns (ok, reason_if_not).

    The keep-ratio floor is intensity-aware: a "polish" pass must not drop more
    than ~10% (matching its prompt contract), while "rewrite" may restructure
    more aggressively but still cannot shrink below the global floor. The upper
    bound matches the prompt's "<=1.5x original" instruction so the gate and the
    prompt no longer disagree.
    """
    if len(refined.strip()) < 500:
        return False, f"too short: {len(refined.strip())} chars"
    global_floor = float(config["novel"].get("refine_min_keep_ratio", DEFAULT_MIN_KEEP_RATIO))
    floor_by_intensity = {
        "polish": max(0.9, global_floor),
        "restructure": max(0.75, global_floor),
        "rewrite": global_floor,
    }
    min_keep = floor_by_intensity.get(intensity, global_floor)
    if len(refined) < len(original) * min_keep:
        return False, f"shrank below {int(min_keep * 100)}% of original ({len(refined)}/{len(original)}) at intensity={intensity}"
    max_grow = float(config["novel"].get("refine_max_grow_ratio", 1.5))
    if len(refined) > len(original) * max_grow:
        return False, f"grew beyond {max_grow:g}x of original ({len(refined)}/{len(original)})"
    return True, ""


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_group_checkpoint(paths: Paths, start: int) -> dict[str, Any] | None:
    p = group_checkpoint_path(paths, start)
    if not p.exists():
        return None
    try:
        return json.loads(read_text(p))
    except Exception:
        return None


def save_group_checkpoint(paths: Paths, start: int, payload: dict[str, Any]) -> None:
    p = group_checkpoint_path(paths, start)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_text(p, json.dumps(payload, ensure_ascii=False, indent=2))


def append_refine_log(paths: Paths, entry: dict[str, Any]) -> None:
    p = refine_log_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Book assembly
# ---------------------------------------------------------------------------

def rebuild_refined_book(paths: Paths, last_chapter: int) -> None:
    """Concat all chapters into book_refined.md. Prefer refined version,
    fall back to the original for chapters not yet refined."""
    out_path = refined_book_path(paths)
    chunks: list[str] = []
    rdir = refined_dir(paths)
    for n in range(1, last_chapter + 1):
        rp = rdir / f"{n:04d}.md"
        op = chapter_path(paths, n)
        if rp.exists():
            text = read_text(rp).strip()
        elif op.exists():
            text = read_text(op).strip()
        else:
            continue
        if text:
            chunks.append(text)
    if chunks:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_text(out_path, "\n\n".join(chunks) + "\n")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def refine_book(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
) -> None:
    """Run the post-completion refine pass over all chapters in 5-chapter groups."""
    last_chapter = find_last_chapter(paths)
    if last_chapter < 1:
        log(paths, "Refine: no chapters to refine")
        return

    groups = iter_groups(last_chapter, group_size=GROUP_SIZE)
    log(paths, f"Refine start last_chapter={last_chapter} groups={len(groups)}")

    refined_dir(paths).mkdir(parents=True, exist_ok=True)
    refine_checkpoint_dir(paths).mkdir(parents=True, exist_ok=True)

    for start, end in groups:
        group_chapters = load_group_text(paths, start, end)
        if not group_chapters:
            log(paths, f"Refine group Ch{start}-{end}: no chapters found, skipping")
            continue

        ckpt = load_group_checkpoint(paths, start)
        if ckpt and ckpt.get("completed"):
            log(paths, f"Refine group Ch{start}-{end}: already completed, skipping")
            continue

        # Diagnose (cached in checkpoint to avoid re-paying)
        diagnosis = (ckpt or {}).get("diagnosis")
        if not diagnosis:
            log(paths, f"Refine diagnose Ch{start}-{end} ...")
            try:
                diagnosis = diagnose_group(client, paths, config, group_chapters, last_chapter)
            except Exception as exc:
                log(paths, f"Refine diagnose Ch{start}-{end} failed: {exc}; defaulting to polish")
                diagnosis = {
                    "group_summary": "",
                    "issues": [],
                    "per_chapter": [
                        {"chapter": n, "intensity": "polish", "focus": ""}
                        for n, _ in group_chapters
                    ],
                    "extra_anchors": [],
                }
            save_group_checkpoint(paths, start, {"diagnosis": diagnosis, "completed": False, "refined_chapters": []})

        # Resolve extra anchors once for the whole group.
        anchor_nums = [int(x) for x in (diagnosis.get("extra_anchors") or [])]
        extra_anchors: list[tuple[int, str]] = []
        for n in anchor_nums:
            p = chapter_path(paths, n)
            if p.exists():
                extra_anchors.append((n, read_text(p)))

        already_refined = set((ckpt or {}).get("refined_chapters", []) or [])

        for item in diagnosis.get("per_chapter", []):
            ch = int(item.get("chapter"))
            if ch in already_refined and refined_chapter_path(paths, ch).exists():
                log(paths, f"Refine Ch{ch}: already done, skipping")
                continue
            intensity = item.get("intensity", "polish")
            focus = item.get("focus", "")
            log(paths, f"Refine Ch{ch} intensity={intensity} focus={focus[:60]!r}")
            original = next((t for n, t in group_chapters if n == ch), "")
            if not original:
                log(paths, f"Refine Ch{ch}: original missing, skip")
                continue
            try:
                refined = refine_one_chapter(
                    client, paths, config, ch, intensity, focus,
                    group_chapters, extra_anchors, diagnosis,
                    is_finale=(bool(config["novel"].get("ending_aware", True)) and ch == last_chapter),
                )
            except Exception as exc:
                log(paths, f"Refine Ch{ch} failed: {exc}; keeping original")
                continue
            ok, reason = _refined_text_acceptable(original, refined, config, intensity)
            if not ok:
                log(paths, f"Refine Ch{ch} rejected ({reason}); keeping original")
                continue
            write_text(refined_chapter_path(paths, ch), refined)
            already_refined.add(ch)
            save_group_checkpoint(paths, start, {
                "diagnosis": diagnosis,
                "completed": False,
                "refined_chapters": sorted(already_refined),
            })
            append_refine_log(paths, {
                "chapter": ch,
                "intensity": intensity,
                "focus": focus,
                "original_chars": len(original),
                "refined_chars": len(refined),
                "time": datetime.now().isoformat(timespec="seconds"),
            })

        save_group_checkpoint(paths, start, {
            "diagnosis": diagnosis,
            "completed": True,
            "refined_chapters": sorted(already_refined),
        })
        rebuild_refined_book(paths, last_chapter)
        log(paths, f"Refine group Ch{start}-{end} done; refined_chapters={sorted(already_refined)}")

    rebuild_refined_book(paths, last_chapter)
    log(paths, f"Refine complete. Output: {refined_book_path(paths)}")
