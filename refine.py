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
from quality import text_similarity


REFINED_DIR_NAME = "chapters_refined"
REFINED_BOOK = "book_refined.md"
REFINE_LOG_NAME = "refine.log.jsonl"

GROUP_SIZE = 5
DEFAULT_MAX_EXTRA_ANCHORS = 4
DEFAULT_DIAGNOSE_MAX_TOKENS = 4000
DEFAULT_REFINE_MAX_TOKENS = 16000
DEFAULT_MIN_KEEP_RATIO = 0.6  # refined chapter cannot shrink below 60% of original

# Intensity ordering for "bump" comparisons: a quality-debt chapter must not be
# refined at a LOWER intensity than its debt severity demands.
_INTENSITY_RANK = {"polish": 0, "restructure": 1, "rewrite": 2}


def _load_quality_debt(paths: Paths, chapter_num: int) -> dict[str, Any] | None:
    """Read the quality_debt.json marker pipeline writes when a chapter is
    force-accepted below threshold. Returns None when the chapter met threshold."""
    from checkpoint import load_checkpoint as _lc
    debt = _lc(paths, chapter_num, "quality_debt.json")
    return debt if isinstance(debt, dict) else None


def _debt_min_intensity(debt: dict[str, Any], config: dict[str, Any]) -> str:
    """Map a quality-debt record to the minimum refine intensity it warrants.

    Heavy style collapse or a contract violation => rewrite; a meaningful miss
    => restructure; otherwise leave the diagnosed intensity alone (polish floor).
    """
    try:
        em = float(debt.get("em_dash_per_kchar") or 0.0)
    except (TypeError, ValueError):
        em = 0.0
    try:
        score = float(debt.get("score") or 10.0)
    except (TypeError, ValueError):
        score = 10.0
    em_bad = float(config["novel"].get("style_em_dash_per_kchar_bad", 12.0))
    if debt.get("had_contract_violation") or em >= em_bad or score < 6.0:
        return "rewrite"
    if em >= float(config["novel"].get("style_em_dash_per_kchar_warn", 6.0)) or score < 7.5:
        return "restructure"
    return "polish"



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

DIAGNOSE_CORE = """你是一位资深{genre_label}网文编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理与动作描写；保持每章核心目标和章末状态。

强度选择规则（必须遵守）：默认优先选 polish；仅当本章确有节奏/逻辑/连贯问题但情节本身可用时升到 restructure；仅当场景顺序混乱或因果链断裂严重、非重写无法修复时才升到 rewrite。不要轻易升级强度。

## 重点诊断维度（必须逐一检查）
1. **时间词滥用**：是否用"翌日清晨""这天晚上""午后""次日黄昏"等时间词切换场景？是否每章出现3次以上时间标记？
2. **文风塌缩（破折号碎片化）**：是否大量出现"句子——状态——状态"式破折号短句链、单词短句堆叠、无标点舞台提示式断行？是否缺乏完整成句的叙事与正常对话？这是最严重的缺陷。
"""

DIAGNOSE_COMMON_FOOTER = """
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
  "extra_anchors": [<int>, ...]  // 可空数组，最多4个章节号
}
"""

DIAGNOSE_GENRE_DIMS = {
    "history": """3. **文笔和叙事风格**：是否过于白话或缺乏历史感？对话是否缺乏潜台词和话术攻防？是否有"show don't tell"问题？
4. **情节逻辑**：因果链条是否完整？是否有情节跳跃或交代不清的转场？伏笔是否有对应收线？
5. **人物塑造**：人物行为是否符合其立场和利益？主角的成长是否来自具体的挫败或推演？配角是否有独立的行动逻辑？
6. **节奏问题**：章节是否过于碎片化（每次跳转都用时间词）？压迫与兑现的比例是否失衡？""",
    "xuanhuan_shuang": """3. **爽点密度与兑现**：本章是否有明确的爽点高潮（兑现/打脸/翻盘/识破阴谋/掌权）？爽点是否落到具体动作与对手反应上、还是流于概括？压迫—兑现的节奏是否够紧？
4. **无脑碾压风险**：主角的胜利是否有铺垫与代价（被猜忌、暴露底牌、消耗人情）？是否存在毫无铺垫的全知全能或对手沦为纸片人？
5. **现代灵魂代入**：主角的判断是否来自现代见识/情报/挫败的推演，而非"突然顿悟"？是否出现现代名词穿帮、破坏代入感？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有让读者想立刻看下一章的强钩子？""",
    "system_stream": """3. **系统反馈节奏**：本章是否有可见的系统反馈（面板/任务/奖励/数值升级/解锁）？升级与解锁是否有节奏感和成就感、还是流于概括？
4. **金手指代价与平衡**：系统能力是否有代价、冷却或限制？是否出现无脑刷数值、金手指降智解题、成长毫无张力？
5. **代入与目标**：主角的目标与成长动机是否清晰，读者是否有持续追读的动力？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有强钩子？""",
    "urban_ability": """3. **打脸与碾压兑现**：本章打脸/资源碾压/身份反差的爽点是否落到具体动作与对手反应上？压迫—兑现节奏是否够紧？
4. **对手智商与铺垫**：打脸是否有铺垫，对手反应是否合理？是否出现配角降智捧哏、爽点凭空降临？
5. **代入感**：主角的先知/重生优势是否自然融入推演，而非全知全能？情绪与处境是否清晰？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有强钩子？""",
    "romance_female": """3. **情绪张力与关系弧**：本章关系是否有实质推进（拉近/误会/和解）？甜虐节奏是否得当？情绪是否由具体事件支撑而非凭空悬浮？
4. **对手戏化学反应**：男女主互动是否有潜台词与张力？是否流于直白或工具化？
5. **配角与代入**：配角是否沦为工具人？女主（或主角）的处境、动机、情绪是否清晰可代入？
6. **节奏与钩子**：章节是否拖沓或情绪空转？章末是否有让人追读的情绪钩子？""",
    "wanzu_xuanhuan": """3. **境界/战力体系**：境界与战力是否清晰可预期、前后一致？战力跨度是否失控、自相矛盾？
4. **斗法画面与张力**：斗法/天骄争锋/境界突破是否有画面感和热血张力，还是流于概括陈述？
5. **力量解题合理性**：主角的取胜是否正比于此前规则铺垫（Sanderson 第一/二定律）？是否凭空开挂？
6. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有强钩子？""",
    "suspense": """3. **视角越界**：是否写出视角人物当下不可能知道的真相、他人内心、未到场之事？限制视角是否被破坏（这是本类型的致命伤）？
4. **线索公平性**：关键揭示/反转是否有前文公平铺垫，能否在前文找到伏笔？是否存在凭空掉落的关键信息？
5. **悬念账本**：悬念是否只开不收、疑点无限堆积？每章是否至少推进或收束1条旧悬念？
6. **氛围与留白**：诡异/恐惧是否靠反常的具体细节与留白营造，还是靠"恐怖""惊悚"式贴标签形容词？
7. **节奏与钩子**：章节是否拖沓或碎片化？章末是否有让人追读的悬念钩子？""",
}

_DIAGNOSE_GENRE_LABELS = {
    "history": "历史题材",
    "xuanhuan_shuang": "穿越爽文",
    "system_stream": "系统流",
    "urban_ability": "都市异能/重生题材",
    "romance_female": "女频言情/宠文",
    "wanzu_xuanhuan": "现代玄幻/万族争锋题材",
    "suspense": "悬疑/心理惊悚题材",
}


def _build_diagnose_system(preset: str) -> str:
    """Assemble diagnose system prompt from shared core + genre-specific dimensions + footer."""
    genre_label = _DIAGNOSE_GENRE_LABELS.get(preset, "历史题材")
    genre_dims = DIAGNOSE_GENRE_DIMS.get(preset, DIAGNOSE_GENRE_DIMS["history"])
    return DIAGNOSE_CORE.format(genre_label=genre_label) + genre_dims + DIAGNOSE_COMMON_FOOTER


DIAGNOSE_SYSTEM_PRESETS = {key: _build_diagnose_system(key) for key in DIAGNOSE_GENRE_DIMS}

# ── Refine system prompt = 通用精修核心(REFINE_CORE,所有题材共用,只维护一处) + 题材增量
# (REFINE_SYSTEM_BASE_*,只写该题材独有的精修重点)。组合见 refine_one_chapter。
# 拆分动机：原来 7 个题材各自平铺~7条规则，其中 输出格式/去时间词/因果链/健康文风/叙事密度
# 5 条逐字重复×7（改一处要改7份且易漂移），且缺 去AI腔/画面沉浸/对话见人 等爆款硬能力。
REFINE_CORE = """你是一位精调中文网文的资深编辑兼作家（具体题材见下方「本题材精修重点」）。
你的任务：在**不改变**标题、关键情节点、人物决策、章末状态的前提下，把正文打磨到商业网文可直接连载的水准。这是打磨，不是重写。

## 输出格式（严格）
- 直接输出修改后的完整章节正文（中文，保持 markdown 风格），不要任何元注释、解释、JSON、代码围栏。
- 保留章节首行标题（如有），保留章末状态与下一章的连贯。
- 字数变动幅度以本轮「精调强度」为准（见下方指令），不另设伸缩比例。

## 通用精修准则（所有题材通用，按优先级从高到低）
1. **去AI腔与模板化**：删改翻译腔/说明书腔、空泛大词（震惊/不可思议/仿佛/宛如/不禁）、万能句式与重复口癖；同一情绪或动作每次都换新鲜、具体的写法，杜绝批量润色的机械感。
2. **对话见人**：每句台词都贴合该人物的身份、性格、当下处境；有潜台词与话术攻防，不直白说教、不工具化，避免所有人一个腔调。
3. **画面感与沉浸**：把抽象概括改成看得见的具体动作、感官细节、环境互动（展示而非告知）；情绪由具体事件和身体反应支撑，禁止贴"恐怖/愤怒/震撼"这类标签词。
4. **节奏与爽点**：收紧压迫—兑现节奏、删冗余铺垫；本章该给的爽点/情绪高潮落到具体动作与对手的真实反应上；章末留住读者继续读的钩子。
5. **因果与人物逻辑**：A→B→C 因果链在页面上清晰可见、无跳跃式转场；人物每个决定都有可见的动机/信息/挫败支撑，不许"突然顿悟"。
6. **一致性与衔接**：人物口吻称谓、世界规则、能力/境界边界前后一致；与上下文衔接自然，保持全书文风统一。
7. **健康句法**：破折号每千字≤3处且只作正常插入语；以完整主谓宾句子和有潜台词的对话为主，禁止"短语——状态——状态"碎片链、单词短句堆叠、舞台提示式断行。
8. **时间流转**：删除纯用于切换场景的时间词（翌日清晨/这天晚上），改用情节动作与因果体现时间流逝。"""

# 各题材只写"额外/独有"的精修重点；通用准则已在 REFINE_CORE，不再重复。
REFINE_SYSTEM_BASE_HISTORY = """## 本题材精修重点（历史）
- 增强历史氛围与时代质感；措辞、称谓、器物合乎时代，杜绝现代名词穿帮。
- 权谋对话重潜台词与话术攻防。"""

REFINE_SYSTEM_BASE_SHUANG = """## 本题材精修重点（穿越爽文）
- 爽点高潮（兑现/打脸/翻盘/掌权）落到具体动作与对手反应上，压迫—兑现节奏要紧、给读者出气的快感。
- 主角施展现代见识要有铺垫与代价，对手聪明有反应（非纸片人）；古代背景措辞得体、不穿帮。"""

REFINE_SYSTEM_BASE_SYSTEM_STREAM = """## 本题材精修重点（系统流）
- 系统反馈（面板/任务/奖励/数值/解锁）落到具体场景，升级解锁有节奏感与成就感，不流于概括。
- 金手指须有代价/冷却/限制；删改无脑刷数值、金手指降智解题，让成长有张力。"""

REFINE_SYSTEM_BASE_URBAN_ABILITY = """## 本题材精修重点（都市异能/重生）
- 打脸/资源碾压/身份反差的爽点落到具体动作与对手反应上，节奏紧、给读者出气感。
- 打脸要有铺垫、对手不降智；主角先知/重生优势自然融入推演，非全知全能。"""

REFINE_SYSTEM_BASE_ROMANCE_FEMALE = """## 本题材精修重点（女频言情/宠文）
- 本章关系弧有实质推进（拉近/误会/和解），甜虐节奏得当，情绪由具体事件支撑而非悬浮。
- 男女主互动有潜台词与化学反应；配角有独立动机，不做工具人。"""

REFINE_SYSTEM_BASE_WANZU_XUANHUAN = """## 本题材精修重点（现代玄幻/万族争锋）
- 境界与战力清晰可预期、前后一致，修复战力跨度失控或自相矛盾。
- 斗法/天骄争锋/突破有画面感与热血张力；主角取胜正比于此前规则铺垫（Sanderson 第一/二定律），删凭空开挂。"""

REFINE_SYSTEM_BASE_SUSPENSE = """## 本题材精修重点（悬疑/心理惊悚）
- 守住限制视角：删改视角人物当下不可能知道的真相/他人内心/未到场之事，让悬念回归信息差。
- 关键揭示/反转在前文有公平伏笔；恐怖靠反常的具体细节与留白，不靠贴标签形容词。
- 每章至少推进或收束一条已有悬念，避免疑点只开不收。"""

REFINE_SYSTEM_BASE_RULE_HORROR = """## 本题材精修重点（规则怪谈/民俗无限流）
- 本副本生存规则以清晰编号条目呈现；破局落到"读懂规则真意+民俗常识+人性判断"，非蛮力。
- 金手指（残卷/提示类）只被动给残缺信息、每用付代价，不直接给通关答案。
- 恐怖靠反常的具体细节与留白（多出的一双鞋、被叫名字的回音），不靠贴标签形容词；每章推进或收束一条悬念。"""

# 各题材增量；组合时前置 REFINE_CORE（见 refine_one_chapter）。未命中回退 history。
REFINE_SYSTEM_BASE_PRESETS = {
    "history": REFINE_SYSTEM_BASE_HISTORY,
    "xuanhuan_shuang": REFINE_SYSTEM_BASE_SHUANG,
    "system_stream": REFINE_SYSTEM_BASE_SYSTEM_STREAM,
    "urban_ability": REFINE_SYSTEM_BASE_URBAN_ABILITY,
    "romance_female": REFINE_SYSTEM_BASE_ROMANCE_FEMALE,
    "wanzu_xuanhuan": REFINE_SYSTEM_BASE_WANZU_XUANHUAN,
    "suspense": REFINE_SYSTEM_BASE_SUSPENSE,
    "rule_horror": REFINE_SYSTEM_BASE_RULE_HORROR,
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
    diagnose_system = DIAGNOSE_SYSTEM_PRESETS.get(preset, DIAGNOSE_SYSTEM_PRESETS["history"])
    raw = call_llm(
        client,
        paths,
        config,
        diagnose_system,
        json_prompt(user),
        max_tokens=int(config["novel"].get("refine_diagnose_max_tokens", DEFAULT_DIAGNOSE_MAX_TOKENS)),
        temperature=0.3,
        tag="refine_diagnose",
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
    anti_dup_note: str = "",
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

    fossil_block = ""
    try:
        _fc = paths.logs_dir / "book_fossils.json"
        if _fc.exists():
            _bf = json.loads(read_text(_fc))
            _fl = []
            for _f in (_bf.get("fossils") or [])[:10]:
                _ph = _f.get("phrase", "")
                if _ph:
                    _fl.append("- 「%s」%d章 (%.0f%%)" % (_ph, _f.get("chapter_count", 0), _f.get("frac", 0) * 100))
            if _fl:
                fossil_block = (
                    "## 全书高频化石短语（精修时必须替换）\n"
                    "以下短语在全书中过度重复，已成为机械口癖。\n"
                    "精修本章时，若原文包含这些短语，必须用不同的动作/感官/句式替换：\n"
                    + "\n".join(_fl)
                )
    except Exception:
        pass

    intensity_instr = INTENSITY_INSTRUCTIONS.get(intensity, INTENSITY_INSTRUCTIONS["polish"])
    issues_text = "\n".join(f"- {i}" for i in diagnosis.get("issues", [])[:10]) or "（无）"
    group_summary = diagnosis.get("group_summary", "")

    ending_block = ""
    if is_finale:
        ending_block = """
## 终章收束要求（硬性）
本章是全书最后一章，必须作为真正的结局收束，逐条满足：
1. 正面兑现并解答主线核心悬念（凶手/真相/谜底必须明确给出，不得含糊或留作开放）。
2. 清算尚未了结的关键悬念线（threads 中标为 open 的伏笔逐一收束或明确交代去向）。
3. 不得引入任何新人物、新势力、新案件、新悬念、新危机、新反转钩子。
4. 给情绪落点与主题呼应；可保留一句克制的远景余韵，但不得制造"必须看下一章"的新问题。
若原文结尾是开放式悬念/新急报/新敌人，必须改写为上述收束式结局。
"""

    anti_dup_block = ""
    if anti_dup_note:
        anti_dup_block = f"""
## 去重要求（硬性）
{anti_dup_note}
本章正文必须与相邻章节内容明显不同：不得复述上一章已写过的同一场景、同一段对话或同一组动作。请聚焦本章自身的情节推进与独有信息，与上下文形成区分。
"""

    mode_block = ""
    from config import narrative_mode
    _mode = narrative_mode(config)
    if not is_finale and _mode == "reasoning":
        mode_block = """
## 叙事模式：单密室·精密推理（精调硬性）
精调时强化：场景向封闭/半封闭空间收敛；怀疑与揭示挂在可触摸的具体物证/身体状态上；关键揭示在前文有公平铺垫；核心爽点尽量改写为读者一眼能懂的视觉矛盾，而非抽象推断。
"""
    elif not is_finale and _mode == "serial":
        mode_block = """
## 叙事模式：强钩子·情绪外放·可连载（精调硬性）
精调时强化：开场钩更前置更强；人物情绪敢于外显并落到具体动作与对白；节奏更紧凑、每场景有推进或翻转；章末保留强追读钩子。不要把情绪压得过度内敛。
"""

    user = f"""## 精调任务
请精调小说第 Ch{chapter_num} 章。

{intensity_instr}
{ending_block}
{anti_dup_block}
{mode_block}
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

{fossil_block}

## 待精调的原章节 Ch{chapter_num}
{original.strip()}

## 输出要求
- 输出修改后的完整章节正文（中文）。
- 第一行保留章节标题（如原文有）。
- 不要解释，不要 JSON，不要 markdown 围栏。
"""
    preset = str(config["novel"].get("style_preset", "history"))
    refine_system = REFINE_CORE + "\n\n" + REFINE_SYSTEM_BASE_PRESETS.get(preset, REFINE_SYSTEM_BASE_HISTORY)
    refined = call_llm(
        client,
        paths,
        config,
        refine_system,
        user,
        max_tokens=int(config["novel"].get("refine_chapter_max_tokens", DEFAULT_REFINE_MAX_TOKENS)),
        temperature=float(config["novel"].get("refine_temperature", 0.5)),
        tag="refine_rewrite",
    )
    refined = normalize_chapter(refined)
    return refined


def _refined_text_acceptable(
    original: str, refined: str, config: dict[str, Any], intensity: str = "restructure"
) -> tuple[bool, str]:
    """Sanity-check the refined output. Returns (ok, reason_if_not).

    Both bounds are intensity-aware: a "polish" pass must not drop more than ~10%
    (matching its prompt contract) and stays near 1x on the upper side, while
    "restructure"/"rewrite" may both shrink and expand more aggressively (a thin
    chapter diagnosed for rewrite legitimately needs room to grow). Floors and
    ceilings both fall back to the global config keys when the intensity is unknown.
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
    # Grow ceiling is intensity-aware too: a "polish" pass must stay near 1x, but a
    # "rewrite" legitimately needs room to expand (thin原章 diagnosed for rewrite
    # were being silently dropped by a flat 1.5x ceiling — see tangshuting Ch23).
    global_ceil = float(config["novel"].get("refine_max_grow_ratio", 1.5))
    ceil_by_intensity = {
        "polish": global_ceil,
        "restructure": max(global_ceil, float(config["novel"].get("refine_max_grow_ratio_restructure", 2.0))),
        "rewrite": max(global_ceil, float(config["novel"].get("refine_max_grow_ratio_rewrite", 2.5))),
    }
    max_grow = ceil_by_intensity.get(intensity, global_ceil)
    if len(refined) > len(original) * max_grow:
        return False, f"grew beyond {max_grow:g}x of original ({len(refined)}/{len(original)}) at intensity={intensity}"
    return True, ""


def _adjacent_duplicate(
    refined: str,
    prev_refined: str,
    prev_original: str,
    config: dict[str, Any],
) -> tuple[bool, float]:
    """Detect that a refined chapter is a near-duplicate of its predecessor.

    Compares against BOTH the previously-refined neighbour and that neighbour's
    original text (the duplication may have existed in the source and survived
    refine). Returns (is_duplicate, max_similarity).
    """
    if not bool(config["novel"].get("refine_adjacent_dedupe_enabled", True)):
        return False, 0.0
    threshold = float(config["novel"].get("refine_adjacent_sim_max", 0.7))
    sim = 0.0
    if prev_refined.strip():
        sim = max(sim, text_similarity(refined, prev_refined))
    if prev_original.strip():
        sim = max(sim, text_similarity(refined, prev_original))
    return (sim >= threshold), round(sim, 3)


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

        # Track the immediately-preceding chapter's text (refined if available,
        # else original) to detect adjacent-chapter duplicate generation. Seed
        # from the previous group's last chapter so the gate also catches
        # duplicates that straddle a group boundary (e.g. Ch5≈Ch6).
        prev_refined_text = ""
        prev_original_text = ""
        if start > 1:
            prev_orig_p = chapter_path(paths, start - 1)
            if prev_orig_p.exists():
                prev_original_text = read_text(prev_orig_p)
            prev_ref_p = refined_chapter_path(paths, start - 1)
            if prev_ref_p.exists():
                prev_refined_text = read_text(prev_ref_p)

        for item in diagnosis.get("per_chapter", []):
            ch = int(item.get("chapter"))
            original = next((t for n, t in group_chapters if n == ch), "")
            if ch in already_refined and refined_chapter_path(paths, ch).exists():
                log(paths, f"Refine Ch{ch}: already done, skipping")
                # Keep the rolling-neighbour window correct on resume.
                prev_refined_text = read_text(refined_chapter_path(paths, ch))
                prev_original_text = original
                continue
            intensity = item.get("intensity", "polish")
            focus = item.get("focus", "")
            # Quality-debt priority: if pipeline force-accepted this chapter below
            # the quality threshold, the diagnose LLM (which sees the same drifted
            # prose) may still rate it "polish". Bump the intensity to at least what
            # the recorded debt severity demands, and fold the debt's concrete
            # problems into the focus so the refine pass attacks the real defect.
            debt = _load_quality_debt(paths, ch)
            if debt:
                min_intensity = _debt_min_intensity(debt, config)
                if _INTENSITY_RANK.get(min_intensity, 0) > _INTENSITY_RANK.get(intensity, 0):
                    log(
                        paths,
                        f"Refine Ch{ch}: quality-debt bump intensity {intensity}->{min_intensity} "
                        f"(score={debt.get('score')}, em={debt.get('em_dash_per_kchar')})",
                    )
                    intensity = min_intensity
                debt_problems = "；".join(str(p) for p in (debt.get("problems") or [])[:3])
                if debt_problems:
                    focus = (focus + " ｜ 欠债项(必须修复): " + debt_problems)[:300]
            log(paths, f"Refine Ch{ch} intensity={intensity} focus={focus[:60]!r}")
            if not original:
                log(paths, f"Refine Ch{ch}: original missing, skip")
                continue
            max_dup_retries = int(config["novel"].get("refine_adjacent_dedupe_retries", 1))
            anti_dup_note = ""
            refined = ""
            accepted = False
            for attempt in range(max_dup_retries + 1):
                try:
                    refined = refine_one_chapter(
                        client, paths, config, ch, intensity, focus,
                        group_chapters, extra_anchors, diagnosis,
                        is_finale=(bool(config["novel"].get("ending_aware", True)) and ch == last_chapter),
                        anti_dup_note=anti_dup_note,
                    )
                except Exception as exc:
                    log(paths, f"Refine Ch{ch} failed: {exc}; keeping original")
                    break
                ok, reason = _refined_text_acceptable(original, refined, config, intensity)
                if not ok:
                    log(paths, f"Refine Ch{ch} rejected ({reason}); keeping original")
                    break
                is_dup, sim = _adjacent_duplicate(
                    refined, prev_refined_text, prev_original_text, config
                )
                if is_dup:
                    if attempt < max_dup_retries:
                        log(
                            paths,
                            f"Refine Ch{ch}: adjacent duplicate (sim={sim}); "
                            f"regenerating with anti-dup directive (attempt {attempt + 1})",
                        )
                        anti_dup_note = (
                            f"检测到本章与上一章相似度过高(sim={sim})，疑似重复生成同一场景。"
                        )
                        intensity = "rewrite"  # force a structural rework, not a re-polish
                        continue
                    # Out of retries: keep original rather than persist a duplicate.
                    log(
                        paths,
                        f"Refine Ch{ch}: still duplicate after retries (sim={sim}); "
                        "keeping original to avoid emitting a duplicate chapter",
                    )
                    break
                accepted = True
                break
            if not accepted:
                # Roll the neighbour window forward using the ORIGINAL text so the
                # next chapter is still compared against real adjacent content.
                prev_refined_text = original
                prev_original_text = original
                continue
            write_text(refined_chapter_path(paths, ch), refined)
            already_refined.add(ch)
            prev_refined_text = refined
            prev_original_text = original
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
