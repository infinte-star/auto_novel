"""v3 bootstrap — novel initialization and memory context.

One-time generation of state.md + memory/*.md from prompt.md, plus the
memory-context builders that trial.py, package.py, and the planning module
still use.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engine.config import (
    PROMPT_FILE,
    Paths,
    log,
    normalize_text,
    read_text,
    text_bigrams,
    write_text,
)
from engine.llm import call_llm, json_prompt, load_json_with_repair
from engine.story_spine import build_story_spine, render_story_spine, save_story_spine
from engine.store import db_event, recent_events, recent_metrics

if TYPE_CHECKING:
    from openai import OpenAI

# Event types that represent actual STORY/plot beats, as opposed to the bulky
# diagnostic/telemetry dumps the same `events` table also stores (review +
# arbitration JSON, prescreen/dedupe retries, etc.). `memory_context` injects
# recent events into the planner prompt purely for plot continuity, so it must
# pull ONLY these — otherwise the multi-KB diagnostic payloads accumulate and
# blow up the prompt size (~50K/chapter → ~300K by Ch5 → empty LLM responses).
PLOT_EVENT_TYPES = ("story_event",)

BOOTSTRAP_SYSTEM = """你是一部 200 万字以上中文网文的总设计师。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。键名如下：
{
  "title": "一个原创中文书名，<=15字，契合类型与核心命题，不照搬现有作品",
  "state": "简短的当前状态 markdown，<=5000 个中文字符",
  "voice": "叙事声音宪章 markdown，<=2500 个中文字符，见下述强制内容",
  "bible": "世界规则、力量体系、社会秩序、硬性约束，<=6000 个中文字符",
  "characters": "主要人物的状态机：目标、恐惧、资源、关系、秘密，<=6000 个中文字符（每个主要人物必须含一个 **人设记忆点** 子条目，见下述强制内容）",
  "timeline": "初始时间线与计划中的历史压力，<=3000 个中文字符",
  "threads": "已开启的伏线台账，含 introduced/due/status，<=3000 个中文字符",
  "volume_plan": "结构化卷纲，详见下述强制结构（默认长篇至少 3 卷、每卷 60-80 章；但若下方创作简报/约束限定了总章数上限，必须改按该上限规划，不得套用 60-80 章模板）"
}

## voice 强制内容（这是全书文风基线，奠定整本书的句子质感，必须健康可读）
用 markdown 输出，必须显式包含以下"健康文风护栏"，且给出 2-3 段示范正文片段：
- 以完整的主谓宾句子叙事；破折号（——）每千字不超过 3 个，只用作正常插入语，绝不用来粘连碎片。
- 平均句长保持在正常小说水平（约 15-40 字），不得通篇单词短句，禁止"句子——状态——状态"式破折号短句链。
- 段落是连贯成句的叙事，不是无标点断行的舞台提示。
- 保留有潜台词、有话术攻防的人物对话。
- 另列：时态/视角、词汇调性、感官锚使用习惯、章节结构惯例。

## characters 强制内容（人设记忆点——读者辨识度与追读留存的关键）
characters markdown 中，每个主要人物（至少主角与 2-3 个核心配角）都必须显式包含一个 `**人设记忆点**` 子条目，列 1-3 个具体、可在正文反复复现、人物之间彼此区分的标志性记忆点，类型可为：
- 口头禅 / 说话习惯（具体到一句或一种句式，而非"爱说话"）；
- 标志性动作或小习惯（如"紧张时数手指""总把钥匙在掌心转三圈"）；
- 独特的身体/外形/穿着细节或外号；
- 反差萌或鲜明的性格反差（如"凶悍外表下怕黑"）。
要求：必须具体可演（能写进动作与对白），禁止"善良""聪明""坚强"这类空泛形容词；同一记忆点不得多个角色共用。这些记忆点将随人物档案长期下发，须在全书反复、自然地复现以建立角色辨识度。

## volume_plan 强制结构（这是本书的长期大纲，必须详尽且可执行）
用 markdown 输出。每一卷用 `## 第N卷：<卷名>（第X-Y章）` 作标题，章节区间必须明确（如 第1-70章）。
每卷内部必须包含以下小节，缺一不可：
- **卷主题**：本卷在讲什么、读者情绪主轴。
- **核心矛盾**：本卷要解决的那一组主要矛盾（与上一卷的遗留危机衔接）。
- **阶段高潮**：每 15-25 章一个，列出本卷的 2-4 个阶段高潮及其触发条件。
- **大事件锚点**：至少 2-3 个不可回避的剧情锚点（具体事件，不是抽象目标）。
- **本卷兑现**：本卷解决的主要矛盾 / 给读者的核心爽点兑现。
- **重大代价**：本卷重大胜利必须付出的可见代价（资源/关系/信任/身份等）。
- **遗留危机**：卷末开启的、比本卷更高层级的新危机，作为下一卷钩子。
- **线索兑现表**：本卷每条主线索标注在第几章兑现；兑现章号必须错峰分散（间隔≥2-3章一条），严禁多条线索的 payoff 挤在同一章或卷末相邻几章（挤兑会让该章过载、模型写不动、质量崩塌）。开线不得快于兑现（少埋多兑）。
全书层面要求：保持卷与卷之间的因果递进（上一卷遗留危机 = 下一卷核心矛盾的来源），
主角能力边界逐卷扩张但始终有约束，不得出现「主角一开始就全知全能」。
卷长按题材节奏定（快节奏/短章一卷约15-30章，慢热长篇40-80章）；无论长短都遵守上面的错峰兑现纪律。
为控制 token，先详写前 2 卷，其余各卷给 1 段概要即可。

创作原创素材，不要模仿现有作品。以长期因果与读者期待为优化目标。

## ⚠ 章数上限（最高优先级，覆盖上面的卷数/章数默认值）
如果下方创作简报或附加约束中给出了明确的总章数上限（例如"全书 N 章封闭收束""限 N 章完结""max_chapters=N"等），
则 volume_plan 必须严格按这个上限来规划，**不得**沿用"3 卷 / 每卷 60-80 章"的长篇模板：
- 全书就是 N 章，章节区间不得超过 N（如"第1-6章"，禁止出现"第8-10章""第50章"等超出 N 的锚点）。
- 大事件锚点必须压缩到这 N 章之内，每个锚点标注它落在第几章（且 ≤ N），且最后一个高潮/真相锚点必须落在第 N 章或之前。
- 短篇不必强行分 3 卷；可只写 1 卷（或不分卷），把开局→升级→高潮→收束安排进 N 章。
- 这是硬约束：anchor 完成门会拿这些锚点逐章审计，锚点的章号若超出 N 会导致永远"未兑现"而错误地拖长全书。"""

# ── Chain-mode bootstrap: generate the foundation as a DEPENDENCY-ORDERED chain
# (bible → characters → voice → volume_plan → frame) instead of one JSON object.
# Single-shot forced 8 interdependent artifacts to compete for one output budget
# (a rich volume_plan starved characters), with no grounding (characters not
# conditioned on a finished bible) and no verification. Each section below is one
# markdown completion conditioned on the already-finished upstream sections.
BIBLE_CHAIN_SYSTEM = """你是一部 200 万字以上中文网文的世界观总设计师。
只输出 markdown 正文，不要 JSON、不要代码块包裹、不要任何解释性前言。
根据下方创作简报，写出本书的「世界观圣经」，这是全书每一章的事实地基，务必厚实、自洽、可被逐章引用：
- 力量/能力体系：运作规则、边界、**代价**（用什么资源/精力/风险换取效果）、升级路径；规则要像科学定律一样清晰、可被读者预期。
- 社会结构 / 权力秩序 / 阵营关系 / 重要机构与职官（若适用）。
- 资源稀缺、时间压力、地理与旅行时间等硬约束。
- 硬性禁止项（明令不可违反的红线）。
- 关键名词表：重要的专有名词/地名/物品，给出统一称谓，避免后续章节漂移。
要求：具体到可演可查，避免空泛口号。篇幅充分（建议 4000-9000 字），宁详勿略——这是地基，越厚后续越不易崩。"""

CHARACTERS_CHAIN_SYSTEM = """你是一部 200 万字以上中文网文的人物总设计师。
只输出 markdown 正文，不要 JSON、不要代码块包裹、不要任何前言。
**必须在下方给定的「世界观圣经」之上**设计人物：每个人物的能力边界、资源、立场都要与世界规则一致，不得发明世界观里没有的力量。
为主角与 2-4 个核心配角各写一个状态机档案，每人包含：
- 目标 / 恐惧 / 资源 / 关系 / 秘密；
- 能力边界与**代价**（与世界圣经的力量体系对齐，强调"做不到什么"）；
- `**人设记忆点**` 子条目：1-3 个具体、可在正文反复复现、彼此区分的标志性记忆点（口头禅/标志动作/外形细节/反差），必须具体可演，禁止"善良/聪明/坚强"这类空泛词，同一记忆点不得多人共用。
重要反派必须有自洽动机与能力上限，严禁反派降智。

## 篇幅纪律（严格执行）
- 全文总计 3000-7000 字，每个人物 800-1500 字。
- 「能力边界与代价」只写**清单条目**（每条一行，说明"做不到什么"即可），**不得重复世界观圣经中已有的规则细节、数值、等级描述**——那些内容已在圣经里，这里写"参见世界圣经"然后只列本人物独有的限制。
- 「目标/恐惧/资源/关系/秘密」每条 1-3 句话说清即可，不展开场景描写或内心独白。"""

VOICES_CHAIN_SYSTEM = """你是一部 200 万字以上中文网文的人物声音区分度设计师。
只输出 markdown 正文，不要 JSON、不要代码块包裹、不要任何前言。
基于下方「人物档案」与「叙事声音宪章」，为每个已定义的主要人物（主角、核心配角、重要反派）各写一个 `## 人物名` 小节，内容：
1. **声音指纹**（3-5 条）：此人独有的说话节奏、句式偏好、口头禅、语气词、断句习惯。必须具体到可直接写进对话，禁止"说话干脆""温柔"这类空泛形容。
2. **标志性言行**（2-3 条）：此人在特定情绪下的标志动作或话术模式（如"生气时不提高音量而是放慢语速""撒谎时先摸后颈"），强调与其他人物的差异。
3. **行为锚点**（1-2 条）：反复出现、读者可预期的行为习惯（如"永远最后一个离开""喝茶前先闻杯口"），作为人设记忆点的外化载体。
篇幅纪律：每人 300-600 字，全文总计不超过 5000 字。宁可少写几个次要角色也不要超长——这份表会被每章下发，过长会挤占上下文。
人物之间的声音指纹必须彼此区分：不允许两个人物共用同一口头禅或同一标志动作。"""

VOICE_CHAIN_SYSTEM = '你是一部 200 万字以上中文网文的文风总监。\n只输出 markdown 正文，不要 JSON、不要代码块包裹、不要任何前言。\n基于下方「世界观圣经」与「人物档案」，写出本书的「叙事声音宪章」——这是全书文风基线，奠定整本书的句子质感，必须健康可读。\n必须显式包含以下“健康文风护栏”，并给出 2-3 段**示范正文片段**（真正的小说叙事，不是说明）：\n- 以完整的主谓宾句子叙事；破折号（——）每千字不超过 3 个，只用作正常插入语，绝不用来粘连碎片。\n- 平均句长保持在正常小说水平（约 15-40 字），不得通篇单词短句，禁止“句子——状态——状态”式破折号短句链。\n- 段落是连贯成句的叙事，不是无标点断行的舞台提示。\n- 保留有潜台词、有话术攻防的人物对话。\n另列：时态/视角、词汇调性、感官锚使用习惯、章节结构惯例。\n示范片段必须亲自示范上述健康文风，因为它会成为全书声音锚被反复下发。'

_VOLUME_PLAN_STRUCTURE_SPEC = """每一卷用 `## 第N卷：<卷名>（第X-Y章）` 作标题，章节区间必须明确。
每卷内部必须包含以下小节，缺一不可：
- **卷目标(O)**：一句话可验证的终态（如"主角从杂役晋升为外门弟子并查明师门暗杀线索"）。这是本卷所有章节的北极星。
- **关键成果(KR)**：3 条可观测的子成果，支撑卷目标。每条 KR 应可在 3-5 章内完成。例：KR1=拿下药园管事权，KR2=与灵安峰结盟，KR3=获得暗杀实证。
- **卷主题**：本卷在讲什么、读者情绪主轴。
- **核心矛盾**：本卷要解决的那一组主要矛盾（与上一卷遗留危机衔接）。
- **阶段高潮**：每 15-25 章一个，列出本卷 2-4 个阶段高潮及触发条件。
- **大事件锚点**：至少 2-3 个不可回避的剧情锚点（具体事件，且必须点名涉及哪些已定人物）。
- **本卷兑现**：本卷解决的主要矛盾 / 核心爽点兑现。
- **重大代价**：本卷重大胜利付出的可见代价（资源/关系/信任/身份）。
- **遗留危机**：卷末开启的更高层级新危机，作为下一卷钩子。
- **线索兑现表**：本卷开启/推进的每条主线索，逐条标注在第几章兑现（阶段兑现或收束）。兑现章号必须错峰分散，间隔至少 2-3 章一条。

## 伏笔兑现节奏纪律（最高优先级，防"多线挤同章"过载——违反会让该章过载、模型写不动、质量崩塌）
- 每个「阶段高潮」「大事件锚点」原则上**只兑现一条主线索**；绝不把两条以上主线索的 payoff 塞进同一章或相邻两章。
- **开线不得快于兑现**（少埋多兑）：每关一条再开一条；严禁前期猛开一堆线索、却把兑现全堆到卷末某个"收束点"。
- **章节区间贴合实际写作节奏**：引擎常按快节奏逐章生成，一卷实际有效长度可能远短于模板默认；宁可把卷切短、让每章只扛一个 payoff，也不要为凑长度把多个兑现压进一个收束章。
- 卷末闭环那一章也只做"最后一条主线索的收束＋带出/代价"，此前的支线兑现必须已在前面各章分散消化完。"""

VOLUME_PLAN_CHAIN_SYSTEM = """你是一部 200 万字以上中文网文的卷纲总设计师。
只输出 markdown 正文，不要 JSON、不要代码块包裹、不要任何前言。
基于下方「世界观圣经」与「人物档案」，写出结构化卷纲。""" + _VOLUME_PLAN_STRUCTURE_SPEC + """

全书要求：卷与卷因果递进（上一卷遗留危机=下一卷核心矛盾来源），主角能力逐卷扩张但始终有约束。
默认卷长按题材节奏定：快节奏/短章题材一卷约 15-30 章，慢热长篇一卷 40-80 章；无论长短都必须遵守上面的错峰兑现纪律。
**每卷都完整输出全部结构小节**（卷目标/KR/主题/矛盾/高潮/锚点/兑现/代价/危机/线索兑现表），不要缩写、不要只给概要——逐章排期表（角色高光轮值表、关系线、反转排期、心动破防节拍表、爽点兑现节拍表）将在后续步骤分卷生成，此处**不需要输出**。
⚠ 若下方给出明确的总章数上限（max_chapters=N），则必须严格按 N 章规划：章节区间与所有锚点章号都不得超过 N，最后一个高潮/真相锚点必须落在第 N 章或之前，短篇可只写 1 卷，禁止套用 60-80 章模板。"""

VOLUME_PLAN_CHAIN_SYSTEM += """

## 故事脊柱优先级
若输入包含「故事脊柱」，它来自作者原始简报，优先级高于世界观圣经、人物档案与任何候选创意。卷纲必须逐章保留其人物、地点、任务等级、关键对象、事件功能与终章结局；只允许扩写过程和细节，不得换题、换章、换收件人或把终章改成续作钩子。"""

# ── 群像/多主角增量（仅当 config novel.ensemble_cast 为真时注入）───────────────
# 沿用 writing.py:GENRE_PROFILES 的「共享基座 + 体裁增量」架构：这些 delta 追加到
# CHARACTERS_CHAIN_SYSTEM / VOLUME_PLAN_CHAIN_SYSTEM 之后，让地基生成把「多追求者/群像」
# 特有的结构（人物差异化 + 高光轮值 + 关系线争宠升级 + 反转排期 + 心动破防节拍）产出并保留，
# 而不是套用单主角+金手指模板时被丢弃。单 CP / 单主线书不注入，行为完全不变。
_ENSEMBLE_CHARACTERS_DELTA = """

## ⚠ 群像/多主角增量要求（本书为多追求者/群像结构，最高优先级，覆盖上面「2-4 个核心配角」的默认数量）
不要只写主角 + 2-4 个配角。**简报里出现的每一位主要追求者/群像成员都必须各出一档完整档案，一个都不能省**（哪怕有七八位）。每位群像成员的档案除通用状态机外，必须显式包含：
- **关系定位 / 追求方式**：他与主角的关系锚（合伙 / 守护 / 治愈 / 直球 / 灵魂知己 / 高智占有 / 青梅归属……）以及他独有的靠近方式；不同成员的方式**绝不能重复**。
- **反差与缺口**：人前 vs 人后的反差，以及他自身必须克服的成长缺口（不是完美工具人）。
- **入场符号**：一个专属道具 / 动作 / 台词习惯，读者一眼能把他和别人区分开（如虎口旧疤、粉色头盔、含薄荷糖、口头一个"姐"）。
- **专属名场面（种子）**：一个只属于他、最能让读者上头的破防/高光瞬间雏形。
- **一句话声音辨识**：用一句话钉死他说话的质感（如"短而稳""贫而正""柔而清""冷而锐""温而狠""熟而暖"），使全书七人开口即可辨。
去同质化铁律：多位成员**不得做同一种好、同一件事、说同一类话**；每个人的价值必须由其职业、性格与共同历史唯一决定。反派同样要有自洽动机与能力上限，不得降智。"""

# ── 分卷表格生成 prompt（phase 2 of two-phase volume_plan generation）─────────
# 骨架调用（phase 1）产出全部卷的结构小节但不含逐章表格；detail 调用为每一卷
# 单独生成 ensemble/shuang 要求的逐章排期表，避免单次输出预算被第一卷表格耗尽
# 导致后续卷截断或缺失。行粒度约束防止 LLM 把表格行写成微型剧本。
_VOLUME_DETAIL_SYSTEM = """你是卷纲逐章排期表的生成器。
只输出 markdown 表格，不要 JSON、不要代码块包裹、不要任何前言，不要重复骨架中已有的结构小节（卷目标/KR/主题/高潮/锚点等）。
基于下方「卷纲骨架」和「人物档案」，为**指定的一卷**生成逐章排期表。

## 表格格式规则
- 每张表以加粗小节名开头（如 `- **角色高光轮值表**：`），然后紧跟 markdown 表格。
- 表格用 `| ChN | 内容 |` 格式，章号必须连续覆盖指定卷的全部章。
- **每个单元格 ≤ 40 字**：只写「谁 + 做什么 + 类型标签」，禁止写对话台词、生理反应描写、场景细节。
  示例 OK：「程叙默修厨房不留名（暗守型）；裴执偷拍颠锅（观察型）」
  示例 NG：「他端着保温桶站在门口看了她三秒，把汤放下说"顺路"，她低头时耳朵红了——这是他第三次说顺路」
- 超过 25 章的卷，允许每 2-3 章合并一行（如 `Ch5-7`），但不得跳章。"""

_VOLUME_DETAIL_ENSEMBLE = """
## 需要生成的表格（群像/多追求者）
- **角色高光轮值表**：逐章列出哪些群像成员获得独立高光瞬间及类型。硬约束：①每章至少 3 位成员有展示；②核心成员不得连续两章隐形；③同章不同成员不得做同质化的事。
- **关系线 / 争宠升级曲线**：逐章标注关系推进的强度档位，必须逐章升级，严禁三章停在同一水位。
- **反转做成名场面排期**：每个身份/关系/命运反转标注引爆章号，错峰分散、一章最多一个重量级反转。
- **心动/情绪破防节拍表**：逐章排布每章 1 处心动破防瞬间，用类型标签标注（如「耳红嘴硬型」「沉默守护型」），逐章变化不重复。"""

_VOLUME_DETAIL_SHUANG = """
## 需要生成的表格（爽点节拍）
- **爽点兑现节拍表**：逐章标注主爽点类型（打脸反杀/逆袭翻盘/装逼扮猪吃虎/身份反转苏爆/实力碾压/打脸回响…），相邻章不得同味连用。每章一行，格式：`| ChN | 主爽点类型 | 憋（一句话） | 炸（一句话） | 余韵（一句话） |`，每格 ≤ 40 字。"""


def _insert_volume_tables(skeleton: str, vol_tables: list[tuple[int, str]]) -> str:
    """Insert per-volume tables into the skeleton, processing from last to first."""
    from engine.plan import parse_volume_ranges
    for vol_idx, tables in sorted(vol_tables, key=lambda x: x[0], reverse=True):
        ranges = parse_volume_ranges(skeleton)
        if vol_idx >= len(ranges):
            continue
        if vol_idx + 1 < len(ranges):
            insert_pos = ranges[vol_idx + 1]["pos"]
        else:
            insert_pos = len(skeleton)
        before = skeleton[:insert_pos].rstrip()
        after = skeleton[insert_pos:]
        skeleton = before + "\n\n" + tables.strip() + "\n\n" + after
    return skeleton


FRAME_CHAIN_SYSTEM = """你是长篇小说引擎的开篇定稿器。基于下方世界观/人物/卷纲，产出本书启动所需的几个简短字段。
只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。键名如下：
{
  "title": "一个原创中文书名，<=15字，契合类型与核心命题，不照搬现有作品",
  "state": "简短的当前状态 markdown（第1章开篇时的局面），<=3000 个中文字符",
  "timeline": "初始时间线与计划中的历史压力，<=2500 个中文字符",
  "threads": "已开启的伏线台账（每条含 introduced/due/status），<=2500 个中文字符；核心长线控制在 3-5 条，少埋多兑"
}"""

CREATIVE_BOOST_SYSTEM = """你是一位顶尖网文创意策划，擅长跨题材、跨领域联想，为创作简报提出有记忆点、有差异化的候选增强方案。
读取下方创作简报，结合多领域知识（历史、科技、神话、社会学、游戏机制、商业、悬疑结构等）做一次创意增强。
要求：具体、可执行、避免烂大街套路；不偏离简报的题材与核心设定，只在其留白处提出新意。
这些输出只是**候选创意**，不是作者硬设定：不得修改原简报已经指定的能力边界、人物、章位、事件、结局、禁止项或故事脊柱；与原简报冲突时必须舍弃候选创意。

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。键名如下：
{
  "golden_finger": "新颖的金手指/核心能力机制：一句话点明它与同类套路的不同，并给出其代价或限制",
  "character_hooks": ["主角及关键人物的记忆点人设梗（反差、怪癖、信念、隐秘动机），3-5 条，每条具体可演"],
  "opening_hook": "差异化的开篇钩子：用一句话描述第一章如何在极短篇幅内抛出核心冲突/悬念并展示卖点",
  "world_novelty": ["世界观或设定上的新意点，2-4 条，避免常见模板"],
  "differentiation": "与同类热门作品的核心区隔点：读者为什么要读这一本而不是别的（一两句话）"
}

强调：每一条都要具体到能直接落地写作，禁止空泛口号和万能套话。"""

CONTRACT_SYSTEM = """你是长篇小说引擎的「创作契约」抽取器。你的任务是把用户创作简报里**作者明令钉死、跨全书不可违反的硬规则**，抽成机器可逐章校验的结构化契约。

你不是在做总结或润色，而是在提取「验收红线」。只抽取简报中**明确表述或强烈暗示为硬约束**的内容，不要发明简报里没有的规则。

特别注意主角与关键人物的**能力**：要区分能力的「模态」（modality）——即这个能力作用在什么感官/通道上。例如「过目不忘的文本记忆」属于 text（只对读到/记录过的文字生效），不等于「超强听觉辨音」(audio) 或「超强观察」(visual)。把能力的边界、模态、代价分别钉死，是防止后续章节把一种能力悄悄泛化成另一种的关键。

只返回恰好一个合法的 JSON 对象，不要输出其它任何内容。键名如下：
{
  "protagonist": "主角姓名（若简报未明确则留空字符串）",
  "iron_rules": ["最多3条『每章开写前必检』的最高优先级硬规则——从黄金三章/招牌设定/最易违约点里挑最要命的（如：本副本的生存规则必须在早段以编号清单逐条明示；主角零战力、不得亲自打斗/驱邪）。每条须短、可逐章判定；宁缺毋滥，没有就留空数组"],
  "ability_whitelist": [
    {"name": "能力名", "modality": "text|audio|visual|physical|cognitive|supernatural|other", "scope": "这个能力具体能做什么（边界）", "cost": "使用代价/限制（没有则写 none）"}
  ],
  "ability_blacklist": ["主角/关键人物明确做不到、不许做的事，每条一句话（如：不能凭空知道未亲自记录过的内容；记忆不能当法律证据；不能打斗）"],
  "banned_tropes": ["简报里明令禁止的套路，每条一句话（如：反派降智；主角全知全能；靠巧合/天降救兵解决主线；重大胜利零代价；用恐怖等贴标签词代替细节）；若简报中的核心能力/金手指可被反复使用，追加一条“同一能力的使用流程不得逐章原样复用，每次须在机制/代价/约束上有新变化”"],
  "must_hold": ["其它必须全程维持的硬设定，每条一句话（如：限制视角，只写视角人物当下能感知/推断的；关键揭示必须前文公平出现；终章必须收束不留新危机）"],
  "story_spine": [
    {"ch": 1, "requirement": "只复述简报明确指定给本章的核心事件，不得新增或改写", "hard_anchors": ["从该章原文逐字复制的2-6个短锚点"]}
  ]
}

抽取纪律：
- `iron_rules`（开写铁律）只放最高优先级、每章开写都要自检的 1-3 条——它们会被放到写手提示词注意力最强的末尾锚点。挑「历史/本能最易被漏且违反即毁章」的（题材招牌规则、主角能力硬边界）；与 must_hold 的区别是"每章必检的头等红线"而非泛化硬设定；没有合适的就留空。
- 宁缺毋滥：只收作者真正钉死的红线；模糊的、探索留白的、风格偏好类内容不要收进来。
- 能力白名单只列主角及对剧情有关键作用的人物的**核心**能力，不要把普通技能（会开车、会做饭）也列上。
- **先判断简报到底有没有金手指。** 只有当简报明确建立了超出常人的特殊能力/异能/系统/规则外挂，
  才写 `ability_whitelist`；仅凭“聪明、专业、经验丰富”或模糊暗示不得推断能力。能力若被明确建立，
  才记录其名称、模态、作用边界和代价；不确定时留空并交给后续 canon 处理。
- **写实题材必须让 `ability_whitelist` 留空。** 现代都市/言情/职场/美食/商战这类"主角只是个能力出众的
  普通人"的简报，没有任何异能可抽——此时白名单留空数组，把她的专业本事（味觉灵敏、镜头感、法律常识）
  写进 `must_hold` 或干脆不写。绝对不要因为这个字段存在就发明一个能力：白名单是**逐章 HARD 校验的
  验收红线**，凭空捏一条出来，等于给全书每一章预埋一条永远无法满足的违约——审校每章都会判"能力越界"，
  写手改任何一个字都清不掉，返工全部白烧。白名单为空时不要输出能力越界警告。
- 每条都要短、具体、可判定（一个审校者读完能直接判断某一章有没有违反）。
- `story_spine` 只提取简报明确写出章号的排期（如 Ch3/第3章）；没有明确章号就不要猜。`hard_anchors` 必须逐字复制该章原文中的人物、地点、任务等级、关键对象或结局词，禁止同义改写，禁止从其它章节串线。
- 若简报几乎没有可抽取的硬约束，相应数组留空即可，不要硬凑。"""

def _as_markdown(value: Any) -> str:
    """Coerce a bootstrap field to markdown text.

    The model is asked for markdown strings, but occasionally returns a list
    (one entry per character/thread) or a dict. Flatten those to text instead
    of crashing on .strip().
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, (dict, list)):
                parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            else:
                parts.append(str(item))
        return "\n\n".join(p.strip() for p in parts if p and p.strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value or "").strip()

def creative_boost(client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    """One-time AI creative enhancement of the brief, run before bootstrap.

    Reads prompt.md and asks the LLM for novel golden-finger/character/opening
    ideas, returned as a markdown block to inject into the bootstrap user message.
    Fail-degrades to "" so it can never block book creation.
    """
    if not bool(config["novel"].get("creative_boost_enabled", True)):
        return ""
    try:
        raw = call_llm(
            client, paths, config, CREATIVE_BOOST_SYSTEM,
            json_prompt(read_text(PROMPT_FILE)), temperature=0.9, tag="creative_boost",
        )
        boost = load_json_with_repair(client, paths, config, raw, fallback={})
        if not isinstance(boost, dict) or not boost:
            return ""
        db_event(conn, 0, "creative_boost", boost)
        lines = [
            "## 候选创意（仅填补留白；低于原始简报、创作契约与故事脊柱）",
            "> 以下内容可以选用、改造或舍弃。不得据此新增硬规则，不得替换原简报指定的人物、能力、章位、事件与结局。",
        ]
        gf = _as_markdown(boost.get("golden_finger"))
        if gf:
            lines.append(f"- 金手指/核心机制：{gf}")
        hooks = boost.get("character_hooks") or []
        if isinstance(hooks, list) and hooks:
            lines.append("- 人物记忆点：")
            for h in hooks:
                t = _as_markdown(h)
                if t:
                    lines.append(f"  - {t}")
        oh = _as_markdown(boost.get("opening_hook"))
        if oh:
            lines.append(f"- 开篇钩子：{oh}")
        wn = boost.get("world_novelty") or []
        if isinstance(wn, list) and wn:
            lines.append("- 世界观新意：")
            for w in wn:
                t = _as_markdown(w)
                if t:
                    lines.append(f"  - {t}")
        diff = _as_markdown(boost.get("differentiation"))
        if diff:
            lines.append(f"- 差异化区隔：{diff}")
        if len(lines) <= 2:
            return ""
        return "\n".join(lines)
    except Exception as e:  # never block bootstrap
        log(paths, f"creative_boost skipped: {e}")
        return ""


def _contract_to_markdown(contract: dict[str, Any]) -> str:
    """Render the structured creative contract as human/LLM-readable markdown."""
    if not isinstance(contract, dict):
        return ""
    lines = ["# 创作契约（硬约束，跨全书不可违反；逐章校验）"]
    prot = str(contract.get("protagonist") or "").strip()
    if prot:
        lines.append(f"\n**主角**：{prot}")
    ir = contract.get("iron_rules") or []
    if isinstance(ir, list) and ir:
        # Highest-priority per-chapter rules, rendered at the TOP of the contract so
        # they sit high in contract_block AND are captured first by contract_capsule
        # (the writer-prompt tail recency anchor). See contract_capsule's `wanted`.
        lines.append("\n## 开写铁律（每章开写前逐条自检，最高优先级）")
        lines.extend(f"- {x}" for x in ir if x)
    wl = contract.get("ability_whitelist") or []
    if isinstance(wl, list) and wl:
        lines.append("\n## 能力白名单（主角/关键人物只允许使用以下能力；超出即越界）")
        for a in wl:
            if isinstance(a, dict):
                name = str(a.get("name") or "").strip()
                mod = str(a.get("modality") or "").strip()
                scope = str(a.get("scope") or "").strip()
                cost = str(a.get("cost") or "").strip()
                seg = f"- **{name}**"
                if mod:
                    seg += f"（模态：{mod}）"
                if scope:
                    seg += f"：{scope}"
                if cost and cost.lower() != "none":
                    seg += f"｜代价：{cost}"
                lines.append(seg)
            elif a:
                lines.append(f"- {a}")
    bl = contract.get("ability_blacklist") or []
    if isinstance(bl, list) and bl:
        lines.append("\n## 能力黑名单（明令做不到/不许做）")
        lines.extend(f"- {x}" for x in bl if x)
    bt = contract.get("banned_tropes") or []
    if isinstance(bt, list) and bt:
        lines.append("\n## 禁止套路")
        lines.extend(f"- {x}" for x in bt if x)
    mh = contract.get("must_hold") or []
    if isinstance(mh, list) and mh:
        lines.append("\n## 必须全程维持的硬设定")
        lines.extend(f"- {x}" for x in mh if x)
    return "\n".join(lines) if len(lines) > 1 else ""


def _grounded_contract_text(value: Any, source: str, threshold: float = 0.38) -> bool:
    """Whether an extracted rule has lexical evidence in the original brief."""
    text = str(value or "").strip()
    if not text:
        return False
    if text in source:
        return True
    grams = text_bigrams(text, strip="punct")
    if not grams:
        return False
    source_grams = text_bigrams(source, strip="punct")
    return len(grams & source_grams) / len(grams) >= threshold


def _sanitize_contract(contract: dict[str, Any], source: str) -> dict[str, Any]:
    """Drop boost-like inventions that cannot be grounded in ``prompt.md``."""
    if not isinstance(contract, dict):
        return {}
    out: dict[str, Any] = {
        "protagonist": str(contract.get("protagonist") or "").strip(),
        "iron_rules": [],
        "ability_whitelist": [],
        "ability_blacklist": [],
        "banned_tropes": [],
        "must_hold": [],
        # Kept for the separate story-spine normalizer, which independently
        # restricts chapter numbers and anchors to literal chapter snippets.
        "story_spine": contract.get("story_spine") or [],
    }
    if out["protagonist"] and out["protagonist"] not in source:
        out["protagonist"] = ""
    for key in ("iron_rules", "ability_blacklist", "banned_tropes", "must_hold"):
        values = contract.get(key) or []
        if isinstance(values, list):
            out[key] = [
                str(v).strip() for v in values
                if str(v).strip() and _grounded_contract_text(v, source)
            ][:24]

    abilities = contract.get("ability_whitelist") or []
    if isinstance(abilities, list):
        for raw in abilities:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name or name not in source:
                continue
            scope = str(raw.get("scope") or "").strip()
            cost = str(raw.get("cost") or "").strip()
            out["ability_whitelist"].append({
                "name": name,
                "modality": str(raw.get("modality") or "other").strip() or "other",
                "scope": scope if _grounded_contract_text(scope, source, 0.30) else "",
                "cost": cost if _grounded_contract_text(cost, source, 0.30) else "none",
            })
    return out


def _brief_section(brief: str, *keywords: str) -> str:
    """Return markdown sections whose heading contains one of ``keywords``."""
    lines = str(brief or "").splitlines()
    chunks: list[str] = []
    active = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            active = any(k in stripped for k in keywords)
            continue
        if active and stripped:
            chunks.append(stripped)
    return "\n".join(chunks)


def _fallback_contract_from_brief(brief: str, max_chapters: int = 0) -> dict[str, Any]:
    """Deterministic fail-closed contract for provider/quota failures.

    It intentionally extracts less than the LLM path, never more.  The important
    property is that a failed API call cannot erase author-declared iron rules and
    bans for the whole book.
    """
    protagonist = ""
    m = re.search(r"(?:^|\n)\s*[-*]?\s*主角\s*[：:]\s*([^，,。\n]{2,20})", brief)
    if m:
        protagonist = m.group(1).strip()

    def items(section: str) -> list[str]:
        out: list[str] = []
        for raw in section.splitlines():
            line = re.sub(r"^\s*(?:[-*]|[①-⑳]|\d+[.、)])\s*", "", raw).strip()
            if line and line not in out:
                out.append(line[:260])
        return out

    iron = items(_brief_section(brief, "开写铁律", "写作铁律"))[:6]
    bans = items(_brief_section(brief, "硬性禁止", "禁止项", "禁区"))[:16]
    must_hold = items(_brief_section(brief, "世界规则", "必须全程", "硬设定"))[:16]
    for label in ("世界规则", "核心机制形态"):
        inline = re.search(
            rf"(?:^|\n)\s*[-*]?\s*{re.escape(label)}\s*[：:]\s*([^\n]+)", brief,
        )
        if inline:
            value = inline.group(1).strip()
            if value and value not in must_hold:
                must_hold.append(value[:520])
    if max_chapters:
        must_hold.append(f"全书总章数上限为 {max_chapters} 章，必须在第 {max_chapters} 章或之前完结收束")

    abilities: list[dict[str, str]] = []
    ability_match = re.search(
        r"(?:金手指|核心机制|能力机制)\s*[：:]\s*([^\n]+)", brief, flags=re.IGNORECASE,
    )
    if ability_match:
        raw = ability_match.group(1).strip()
        quoted = re.search(r"[「『\"]([^」』\"]{2,30})[」』\"]", raw)
        name = quoted.group(1).strip() if quoted else raw.split("——", 1)[0].strip(" -")[:30]
        if name:
            scope = raw
            cost = "none"
            cost_match = re.search(r"代价\s*[：:]\s*(.*?)(?=限制\s*[：:]|$)", raw)
            if cost_match:
                cost = cost_match.group(1).strip(" ；;") or "none"
                scope = raw[:cost_match.start()].strip(" ；;")
            abilities.append({
                "name": name, "modality": "supernatural",
                "scope": scope[:260], "cost": cost[:260],
            })

    return {
        "protagonist": protagonist,
        "iron_rules": iron,
        "ability_whitelist": abilities,
        "ability_blacklist": bans,
        "banned_tropes": list(bans),
        "must_hold": must_hold,
        "story_spine": [],
    }


def _merge_contract_fallback(
    contract: dict[str, Any], fallback: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(contract or {})
    if not merged.get("protagonist") and fallback.get("protagonist"):
        merged["protagonist"] = fallback["protagonist"]

    def dedupe_rules(values: list[Any]) -> list[Any]:
        out: list[Any] = []
        seen: set[str] = set()
        for value in values:
            key = re.sub(r"[\s\W_]+", "", str(value), flags=re.UNICODE).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(value)
        return out

    for key in ("iron_rules", "ability_blacklist", "banned_tropes", "must_hold"):
        left = merged.get(key) if isinstance(merged.get(key), list) else []
        right = fallback.get(key) if isinstance(fallback.get(key), list) else []
        merged[key] = dedupe_rules([*left, *right])
    left_abilities = merged.get("ability_whitelist")
    if not isinstance(left_abilities, list):
        left_abilities = []
    right_abilities = fallback.get("ability_whitelist")
    if not isinstance(right_abilities, list):
        right_abilities = []
    seen_names = {
        str(item.get("name") or "").strip()
        for item in left_abilities if isinstance(item, dict)
    }
    merged["ability_whitelist"] = list(left_abilities)
    merged["ability_whitelist"].extend(
        item for item in right_abilities
        if isinstance(item, dict) and str(item.get("name") or "").strip() not in seen_names
    )
    merged.setdefault("story_spine", contract.get("story_spine", []) if contract else [])
    return merged


def extract_contract(
    client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any], brief: str | None = None,
    *, max_chapters: int = 0,
) -> dict[str, Any]:
    """Extract the machine-checkable creative contract from the creative brief.

    Writes memory/contract.md and persists a `contract` event so the per-chapter
    write/review path can enforce author-declared hard rules (ability whitelist/
    blacklist, banned tropes, must-hold settings).  The source is always the
    ORIGINAL brief. Creative-boost candidates are deliberately invisible here.

    Provider failures retry, then fall back to a conservative deterministic
    extractor. Returning an empty contract while contract enforcement is enabled
    is forbidden: bootstrap must never fail open.
    """
    if not bool(config["novel"].get("contract_enabled", True)):
        return {}
    source = brief if brief is not None else read_text(PROMPT_FILE)
    attempts = max(1, int(config["novel"].get("contract_extract_attempts", 2) or 2))
    extracted: dict[str, Any] = {}
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            raw = call_llm(
                client, paths, config, CONTRACT_SYSTEM,
                json_prompt(source), temperature=0.2, tag="contract",
            )
            parsed = load_json_with_repair(client, paths, config, raw, fallback={})
            extracted = _sanitize_contract(parsed if isinstance(parsed, dict) else {}, source)
            if _contract_to_markdown(extracted):
                break
            last_error = "empty or ungrounded contract"
        except Exception as exc:
            last_error = str(exc)
        if attempt < attempts:
            log(paths, f"extract_contract attempt {attempt}/{attempts} failed; retrying: {last_error}")

    fallback = _fallback_contract_from_brief(source, max_chapters)
    contract = _merge_contract_fallback(extracted, fallback)
    md = _contract_to_markdown(contract)
    if not md:
        raise RuntimeError(
            "Creative contract extraction and deterministic fallback both returned empty; "
            "refusing unconstrained bootstrap."
        )
    if last_error and not _contract_to_markdown(extracted):
        log(paths, f"extract_contract using deterministic fallback after provider failure: {last_error}")
    write_text(paths.contract, md + "\n")
    db_event(conn, 0, "contract", contract)
    return contract


def contract_capsule(paths: Paths, config: dict[str, Any], cap: int = 1200) -> str:
    """A compact ability-boundary reminder for the END of the writer prompt.

    The full contract sits high in the prompt inside the cacheable prefix region,
    where ~50k chars of writing context dilute it — across suspense_v4 the model
    breached the ability whitelist/modality in 5 of 6 chapters despite the
    contract being present. LLM attention is strongest at the very tail of the
    prompt (recency), so we re-state ONLY the hard ability boundaries (whitelist
    names+modality, blacklist, banned tropes) as the last thing the writer reads
    before generating. This is a focused recency anchor, not the whole contract.

    Reads the structured `contract` event (preferred) and falls back to slicing
    the relevant sections out of contract.md. Returns "" when disabled/empty.
    """
    if not bool(config["novel"].get("contract_enabled", True)):
        return ""
    if not bool(config["novel"].get("contract_capsule_enabled", True)):
        return ""
    try:
        text = read_text(paths.contract).strip()
        if not text:
            return ""
        # Keep only the ability/blacklist/banned-tropes headings (drop must-hold
        # world settings, which are less prone to per-chapter drift) and cap hard.
        wanted = ("开写铁律", "能力白名单", "能力黑名单", "禁止套路", "必须全程维持的硬设定")
        lines = text.splitlines()
        kept: list[str] = []
        emit = False
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("#"):
                emit = any(w in stripped for w in wanted)
                if emit:
                    kept.append(ln)
                continue
            if emit and stripped:
                kept.append(ln)
        body = "\n".join(kept).strip()
        if not body:
            # Fall back to the whole contract head if our headings weren't found.
            body = text[:cap]
        if len(body) > cap:
            body = body[:cap] + "…"
        return body
    except Exception:
        return ""



def _gen_md_section(
    client: OpenAI, paths: Paths, config: dict[str, Any], system: str, user: str, tag: str,
    max_tokens: int = 16000,
) -> str:
    """One markdown-section completion (plain prose, not JSON). Returns '' on failure."""
    try:
        out = call_llm(
            client, paths, config, system, user, temperature=0.7, tag=tag, max_tokens=max_tokens,
        )
        out = (out or "").strip()
        # Strip an accidental ```markdown fence if the model wrapped its output.
        if out.startswith("```"):
            out = re.sub(r"^```[a-zA-Z]*\n?", "", out)
            out = re.sub(r"\n?```$", "", out).strip()
        return out
    except Exception as exc:
        log(paths, f"bootstrap section '{tag}' failed (non-fatal): {exc}")
        return ""


def _bootstrap_chain(
    client: OpenAI, paths: Paths, config: dict[str, Any], brief: str, max_chapters: int,
    contract_md: str = "", story_spine_md: str = "",
) -> dict[str, Any]:
    """Dependency-ordered foundation with parallel fan-out where the graph allows.

    Serial:  bible → characters
    Parallel step 3:  voice ‖ volume_plan_skeleton
    Parallel step 4:  voices ‖ detail_tables (‖ frame when no details)
    Serial tail (ensemble only):  assemble volume_plan → frame

    Each section is conditioned on the finished upstream sections, so characters are
    grounded in a real bible, voice in both, and the volume_plan in all of them —
    instead of 8 artifacts competing for one JSON budget with no grounding. Returns
    the same dict shape the single-shot path produced, so the caller is unchanged.

    `contract_md` is the already-extracted creative contract (empty when contract
    extraction is disabled or failed, in which case this behaves exactly as before).
    It is injected into the bible and characters calls because those two files are
    where abilities get DECLARED, and an ability invented here becomes canon the
    writer reads every chapter while the reviewer measures against the contract —
    a contradiction no individual chapter can resolve. Measured cost of not doing
    this: LESSONS §13, the `CONTRACT_SYSTEM` entry (tangshuting's 味觉共情 — 30 of
    the library's 35 archived HARD contract violations sit in that one book family).
    """
    # Ensemble / multi-lead books (novel.ensemble_cast) need the full cast profiled
    # (not just 2-4 side characters) and the volume plan to carry spotlight-rotation /
    # rival-escalation / reversal-schedule / heart-flutter tables. Single-CP or
    # single-protagonist books leave this off and behave exactly as before.
    ensemble = bool(config["novel"].get("ensemble_cast", False))
    shuang = bool(config["novel"].get("shuang_pacing", False))
    if ensemble:
        log(paths, "Bootstrap chain: ensemble_cast ON — injecting group-cast deltas")
    if shuang:
        log(paths, "Bootstrap chain: shuang_pacing ON — injecting payoff-cadence delta")
    characters_system = CHARACTERS_CHAIN_SYSTEM + (_ENSEMBLE_CHARACTERS_DELTA if ensemble else "")

    # The contract is the brief's own red lines, already machine-extracted. Handing it
    # to the two ability-declaring calls costs zero extra LLM calls and closes the
    # only bootstrap contradiction the per-chapter loop cannot repair.
    contract_constraint = ""
    if contract_md:
        contract_constraint = (
            "\n\n## 创作契约（已从本简报抽取，最高优先级，本节必须服从）\n"
            f"{contract_md}\n\n"
            "硬性要求：**不得发明「能力白名单」之外的超常能力/异能/天赋**，也不得让人物做到"
            "「能力黑名单」明令做不到的事。白名单与黑名单就是简报作者钉死的红线——与之冲突的设定"
            "一律不许写进世界观与人物，哪怕它更好看。若简报明确写了某种能力「没有」，就必须真的没有。"
        )

    spine_constraint = ""
    if story_spine_md:
        spine_constraint = (
            "\n\n## 故事脊柱（作者原始排期，最高优先级，不得改写）\n"
            f"{story_spine_md}\n\n"
            "卷纲与章节卡只能补全过程和细节，不得替换章位、关键人物、地点、任务等级、对象或终章功能。"
        )

    detail_note = ""
    if ensemble or shuang:
        detail_note = " → volume_detail×N"
    log(paths, "Bootstrap chain: bible → characters → [voice ‖ volume_plan]"
               + detail_note + " → [voices ‖ frame]"
               + (" (contract-constrained)" if contract_constraint else ""))
    data: dict[str, Any] = {}

    # ── Step 1-2: bible → characters (serial, strict dependency) ───────────
    bible = _gen_md_section(
        client, paths, config, BIBLE_CHAIN_SYSTEM,
        f"## 创作简报\n{brief}{contract_constraint}", tag="bootstrap_bible", max_tokens=20000,
    )
    data["bible"] = bible

    characters = _gen_md_section(
        client, paths, config, characters_system,
        f"## 创作简报\n{brief}\n\n## 世界观圣经（必须在此之上设计人物）\n{bible}"
        f"{contract_constraint}",
        tag="bootstrap_characters", max_tokens=12000,
    )
    data["characters"] = characters

    # ── Step 3+: parallel where the dependency graph allows ──────────────
    # voice and volume_plan_skeleton both depend only on bible + characters,
    # not on each other. After both land, voices (needs voice) and detail
    # tables / frame (need skeleton) can fan out again.
    voice_user = f"## 创作简报\n{brief}\n\n## 世界观圣经\n{bible[:4000]}\n\n## 人物档案\n{characters[:4000]}"
    vp_user = (
        f"## 创作简报\n{brief}\n\n## 世界观圣经\n{bible[:5000]}\n\n"
        f"## 人物档案\n{characters[:5000]}{spine_constraint}"
    )

    with ThreadPoolExecutor(max_workers=3) as pool:
        # Step 3: voice ‖ volume_plan_skeleton
        f_voice = pool.submit(
            _gen_md_section, client, paths, config, VOICE_CHAIN_SYSTEM,
            voice_user, tag="bootstrap_voice", max_tokens=8000,
        )
        f_skeleton = pool.submit(
            _gen_md_section, client, paths, config, VOLUME_PLAN_CHAIN_SYSTEM,
            vp_user, tag="bootstrap_volume_plan", max_tokens=16000,
        )
        voice = f_voice.result()
        volume_plan = f_skeleton.result()
        data["voice"] = voice

        # Step 4: voices ‖ detail_tables (‖ frame when no details needed)
        # voices depends on voice (ready), details depend on skeleton (ready).
        f_voices = pool.submit(
            _gen_md_section, client, paths, config, VOICES_CHAIN_SYSTEM,
            f"## 人物档案\n{characters}\n\n## 叙事声音宪章\n{voice[:3000]}",
            tag="bootstrap_voices", max_tokens=8000,
        )

        detail_futures: list[tuple[int, "concurrent.futures.Future[str]"]] = []
        if (ensemble or shuang) and volume_plan:
            from engine.plan import parse_volume_ranges
            detail_system = _VOLUME_DETAIL_SYSTEM
            if ensemble:
                detail_system += _VOLUME_DETAIL_ENSEMBLE
            if shuang:
                detail_system += _VOLUME_DETAIL_SHUANG
            ranges = parse_volume_ranges(volume_plan)
            detail_n = min(
                len(ranges),
                int(config["novel"].get("volume_plan_detail_volumes", 2)),
            )
            for i, vol in enumerate(ranges[:detail_n]):
                label = f"第{vol['label']}卷（第{vol['start']}-{vol['end']}章）"
                log(paths, f"Bootstrap chain: generating detail tables for {label}")
                f = pool.submit(
                    _gen_md_section, client, paths, config, detail_system,
                    f"## 卷纲骨架\n{volume_plan}\n\n"
                    f"## 人物档案\n{characters[:4000]}\n\n"
                    f"## 请为以下卷生成逐章排期表\n{label}"
                    f"{spine_constraint}",
                    tag=f"bootstrap_volume_detail_v{vol['label']}", max_tokens=16000,
                )
                detail_futures.append((i, f))

        if not detail_futures:
            # Non-ensemble: frame can run parallel with voices (both inputs ready).
            frame_user = (
                f"## 创作简报\n{brief}\n\n## 世界观圣经\n{bible[:3000]}\n\n"
                f"## 人物档案\n{characters[:3000]}\n\n## 卷纲\n{volume_plan[:3000]}"
                f"{spine_constraint}"
            )
            f_frame = pool.submit(
                call_llm, client, paths, config, FRAME_CHAIN_SYSTEM,
                json_prompt(frame_user), temperature=0.6, tag="bootstrap_frame",
                max_tokens=8000,
            )

        voices = f_voices.result()
        data["voices"] = voices

        if detail_futures:
            vol_tables: list[tuple[int, str]] = []
            for i, f in detail_futures:
                tables = f.result()
                if tables:
                    vol_tables.append((i, tables))
            if vol_tables:
                volume_plan = _insert_volume_tables(volume_plan, vol_tables)
                log(paths, f"Bootstrap chain: inserted detail tables for {len(vol_tables)} volume(s)")

    data["volume_plan"] = volume_plan

    # ── Frame ────────────────────────────────────────────────────────────
    # Non-ensemble: already submitted in the pool above.
    # Ensemble: volume_plan was just assembled with detail tables, so frame
    # must run after assembly (serial).
    if detail_futures:
        frame_user = (
            f"## 创作简报\n{brief}\n\n## 世界观圣经\n{bible[:3000]}\n\n"
            f"## 人物档案\n{characters[:3000]}\n\n## 卷纲\n{volume_plan[:3000]}"
            f"{spine_constraint}"
        )
        try:
            raw = call_llm(
                client, paths, config, FRAME_CHAIN_SYSTEM, json_prompt(frame_user),
                temperature=0.6, tag="bootstrap_frame", max_tokens=8000,
            )
            frame = load_json_with_repair(client, paths, config, raw, fallback={})
            if isinstance(frame, dict):
                for k in ("title", "state", "timeline", "threads"):
                    if frame.get(k):
                        data[k] = frame[k]
        except Exception as exc:
            log(paths, f"bootstrap frame failed (non-fatal): {exc}")
    else:
        try:
            raw = f_frame.result()  # type: ignore[possibly-undefined]
            frame = load_json_with_repair(client, paths, config, raw, fallback={})
            if isinstance(frame, dict):
                for k in ("title", "state", "timeline", "threads"):
                    if frame.get(k):
                        data[k] = frame[k]
        except Exception as exc:
            log(paths, f"bootstrap frame failed (non-fatal): {exc}")
    return data


def _verify_bootstrap(
    client: OpenAI, paths: Paths, config: dict[str, Any], data: dict[str, Any]
) -> None:
    """Foundation verification: voice style-health (with one repair) + anchor audit.

    Weak bootstrap = weak every chapter, and these defects are otherwise only
    caught (if ever) chapters later. Voice is repaired in place; the rest is
    advisory (logged) so verification can never block the run.
    """
    if not bool(config["novel"].get("bootstrap_verify_enabled", True)):
        return
    # 1. Voice must itself pass the deterministic style gate — its sample prose is
    #    the frozen anchor for the whole book, so a fragmented sample seeds collapse.
    #
    # It is held to the number ITS OWN TEXT preaches (`em_dash_reduce_target_per_kchar`,
    # 3/k — the line VOICE_CHAIN_SYSTEM demands and every writer prompt repeats), not
    # merely to the chapter release line (`style_em_dash_per_kchar_warn`, 6.0). Those
    # are two different rulers and the gap between them was silent: measured across the
    # library's 15 voice anchors, **14 exceeded 3/k and only 1 carried any
    # `style_health` penalty at all** (chaosheng_dangan 5.86, tangshuting_e2e 5.32,
    # huangliang 4.93…). A charter whose demo prose runs at twice the density it
    # prescribes teaches the whole book the density, not the rule — and because
    # `voice.md` is written once and never re-derived, nothing downstream ever revisits
    # it.
    #
    # The ladder is the repair ladder's, in its order and for its reason: L0 is
    # deterministic and free, so the LLM rewrite is spent ONLY when the deterministic
    # gate actually fired (penalty > 0 = real style collapse). Being over 3/k with a
    # clean penalty is answered by `fix.reduce_em_dash_if_needed`, which already targets
    # exactly this number — zero extra calls on the common case.
    try:
        from engine.quality import style_health, reduce_em_dash_if_needed

        voice = _as_markdown(data.get("voice"))
        if voice and len(voice) > 200:
            em_target = float(
                config["novel"].get("em_dash_reduce_target_per_kchar", 3.0)
            )

            def _score(text: str) -> tuple[float, float, list[str]]:
                """(penalty, em/kchar, flags) — the two numbers the anchor is held to."""
                h = style_health(text, config)
                return (
                    float(h.get("penalty", 0) or 0),
                    float((h.get("metrics") or {}).get("em_dash_per_kchar", 0) or 0),
                    list(h.get("flags") or []),
                )

            pen0, em0, flags0 = _score(voice)
            best, best_pen, best_em = voice, pen0, em0

            # L1 (one bounded LLM call) — only when style_health itself fired.
            if pen0 > 0:
                log(
                    paths,
                    f"Bootstrap voice failed style_health (penalty={pen0}, "
                    f"em={em0:.2f}/k, flags={flags0}); regenerating once with "
                    "strict directives.",
                )
                strict_user = (
                    "你上一版叙事声音宪章的示范片段触发了文体塌缩检测"
                    f"（penalty={pen0}, flags={flags0}）。"
                    f"请重写，示范片段必须是完整成句的健康小说叙事：破折号每千字≤{em_target:g}、"
                    "平均句长 15-40 字、禁止碎片短句堆叠、保留有潜台词的对话。\n\n"
                    f"## 世界观与人物（参照）\n{_as_markdown(data.get('bible'))[:2000]}"
                )
                fixed = _gen_md_section(
                    client, paths, config, VOICE_CHAIN_SYSTEM, strict_user,
                    tag="bootstrap_voice_repair", max_tokens=8000,
                )
                if fixed and len(fixed) > 200:
                    pen1, em1, _ = _score(fixed)
                    # Penalty first, then density — a rewrite that only ties on both
                    # is NOT an improvement and must not be adopted while the log
                    # claims it was.
                    if (pen1, em1) < (best_pen, best_em):
                        best, best_pen, best_em = fixed, pen1, em1

            # L0 (zero LLM): hold the anchor to the density its own text prescribes.
            if best_em > em_target:
                reduced = reduce_em_dash_if_needed(best, config)
                if reduced and reduced != best:
                    pen2, em2, _ = _score(reduced)
                    if (pen2, em2) <= (best_pen, best_em):
                        best, best_pen, best_em = reduced, pen2, em2

            if best != voice:
                data["voice"] = best
                write_text(paths.voice, best + "\n")
                log(
                    paths,
                    f"Bootstrap voice rewritten: penalty {pen0}->{best_pen}, "
                    f"em {em0:.2f}->{best_em:.2f}/k (charter target ≤{em_target:g}/k).",
                )
            elif pen0 > 0 or em0 > em_target:
                log(
                    paths,
                    f"WARNING: bootstrap voice still over its own charter "
                    f"(penalty={best_pen}, em={best_em:.2f}/k > {em_target:g}/k) — the "
                    "frozen anchor demonstrates worse prose than it prescribes.",
                )
    except Exception as exc:
        log(paths, f"Bootstrap voice verification skipped (non-fatal): {exc}")

    # 2. Advisory: does the volume_plan reference real character names? A plan whose
    #    anchors name nobody from the character file is a grounding break.
    try:
        chars = _as_markdown(data.get("characters"))
        vp = _as_markdown(data.get("volume_plan"))
        if chars and vp:
            names = re.findall(r"(?:^|\n)#+\s*([^\n#（(]{2,12})", chars)
            names += re.findall(r"\*\*([^\*]{2,12})\*\*", chars)
            names = [n.strip() for n in names if n.strip() and "人设记忆点" not in n]
            if names and not any(n in vp for n in names):
                log(
                    paths,
                    "WARNING: bootstrap volume_plan references none of the character "
                    f"names ({names[:5]}…) — possible grounding break between卷纲 and 人物.",
                )
    except Exception:
        pass


def bootstrap(client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any]) -> None:
    log(paths, "Bootstrapping layered memory")
    original_brief = read_text(PROMPT_FILE)
    brief = original_brief
    # Short-novel mode: surface the hard chapter cap to the bootstrap LLM so the
    # volume_plan is planned WITHIN N chapters instead of defaulting to the
    # "3 卷 / 每卷 60-80 章" long-novel template. Without this, a 6-chapter
    # novel got a 60-70 章 volume_plan whose anchors (Ch8-10 / Ch50-53) sit
    # beyond max_chapters, so the anchor-completion gate audits them forever as
    # "未兑现" and drags the book past its cap with degraded tail chapters.
    max_chapters = int(config["novel"].get("max_chapters", 0) or 0)
    if max_chapters:
        brief = (
            brief
            + f"\n\n## 附加硬约束（最高优先级）\n"
            + f"- max_chapters={max_chapters}：全书总章数上限为 {max_chapters} 章，必须在第 {max_chapters} 章或之前完结收束。\n"
            + f"- volume_plan 必须严格按 {max_chapters} 章规划：章节区间与所有大事件锚点的章号都不得超过 {max_chapters}；"
            + f"最后一个高潮/真相/代价锚点必须落在第 {max_chapters} 章或之前。禁止套用 60-80 章/多卷长篇模板。"
            + f"\n- 卷纲必须保证每一章对核心能力/金手指的使用在「机制 / 代价 / 约束 / 解读路径」上彼此可区分；"
            + f"若该 premise 无法支撑 {max_chapters} 次实质不同的能力使用，请主动把卷纲压缩到更少章数并相应下调高潮锚点章号，"
            + f"宁可短而完整，不要靠重复同一套用法凑章数。"
        )
    def _section(key: str, heading: str, data: dict[str, Any]) -> str:
        val = _as_markdown(data.get(key))
        return val if val else f"# {heading}\n\n（bootstrap 未生成，待连载补全）"

    # P0 ordering: the hard contract and story spine are extracted from the
    # ORIGINAL author brief (the contract additionally sees the engine's explicit
    # max-chapter cap) before
    # any creative expansion exists. Candidate ideas can fill blanks later, but
    # can never be promoted into author-declared law by construction.
    contract = extract_contract(
        client, paths, conn, config, brief=brief, max_chapters=max_chapters,
    )
    if contract:
        log(paths, "Extracted creative contract -> memory/contract.md")

    spine = build_story_spine(original_brief, contract, max_chapters=max_chapters)
    save_story_spine(paths, spine)
    story_spine_md = render_story_spine(spine)
    if story_spine_md:
        log(paths, f"Extracted original-brief story spine -> {len(spine.get('chapters', {}))} chapter slots")

    boost_block = creative_boost(client, paths, conn, config)
    if boost_block:
        brief = brief + "\n\n" + boost_block

    if bool(config["novel"].get("bootstrap_chain_enabled", True)):
        # Render from the dict rather than reusing `contract_capsule`: that helper is
        # gated on `contract_capsule_enabled`, a WRITER-prompt toggle, so borrowing it
        # would let someone disabling the writer's tail anchor silently also disable
        # this bootstrap constraint. Two purposes, two switches.
        contract_md = _contract_to_markdown(contract) if contract else ""
        if len(contract_md) > 6000:
            contract_md = contract_md[:6000] + "\n…（契约过长已截断）"
        data = _bootstrap_chain(
            client, paths, config, brief, max_chapters,
            contract_md=contract_md, story_spine_md=story_spine_md,
        )
    else:
        # Legacy single-shot path: one JSON completion for all 8 artifacts.
        legacy_brief = brief
        if story_spine_md:
            legacy_brief += "\n\n## 故事脊柱（最高优先级，不得改写）\n" + story_spine_md
        raw = call_llm(client, paths, config, BOOTSTRAP_SYSTEM, json_prompt(legacy_brief), temperature=0.7, tag="bootstrap")
        data = load_json_with_repair(client, paths, config, raw)

    title = str(data.get("title") or "").strip()
    if not title:
        # Fallback to the novel directory name (parent of state.md), else placeholder.
        title = paths.state.parent.name or "未命名"
    write_text(paths.title, title + "\n")
    # The bootstrap LLM occasionally omits a key (e.g. "timeline"); never let a
    # single missing field crash the whole bootstrap and leave a half-written
    # state.md that blocks re-bootstrap. Degrade to a labelled placeholder so the
    # pipeline can proceed; the per-chapter loop will populate these going forward.
    write_text(paths.state, _section("state", "当前状态", data) + "\n")
    write_text(paths.bible, _section("bible", "世界观圣经", data) + "\n")
    write_text(paths.characters, _section("characters", "人物", data) + "\n")
    write_text(paths.timeline, _section("timeline", "时间线", data) + "\n")
    write_text(paths.threads, _section("threads", "伏笔与线索", data) + "\n")
    write_text(paths.volume_plan, _section("volume_plan", "卷纲", data) + "\n")
    # Narrative-voice charter: this is the strongest anti-style-collapse anchor and
    # must exist from chapter 1. Only write it when the model produced one; an empty
    # value falls back to the placeholder created by ensure_project().
    voice_charter = _as_markdown(data.get("voice"))
    if voice_charter:
        write_text(paths.voice, voice_charter + "\n")
    voices_text = _as_markdown(data.get("voices"))
    if voices_text:
        write_text(paths.voices, voices_text + "\n")
    db_event(conn, 0, "bootstrap", data)
    # Verification pass: surface (and, for voice, repair) foundation defects before
    # they propagate into every chapter. Advisory/log-only except voice regen.
    _verify_bootstrap(client, paths, config, data)
    # 吸量包（Gap-4）：开写前先用番茄书名/三段式简介公式产出候选，吸量是流量漏斗第一层。
    # 纯顾问产物（hook_package.md），不进 cacheable_prefix，失败静默不阻塞 bootstrap。
    if bool(config["novel"].get("hook_package_enabled", True)):
        try:
            from commands.package import build_hook_package
            pkg = build_hook_package(client, paths, conn, config)
            # 吸量评分/排序 + 赛道评估 + 采纳最优书名（独立评判，点击率优先）。
            if pkg and bool(config["novel"].get("hook_package_scoring_enabled", True)):
                try:
                    from commands.package import score_hook_package
                    score_hook_package(client, paths, config, pkg)
                except Exception as exc:
                    log(paths, f"Hook package scoring step failed (non-fatal): {exc}")
        except Exception as exc:
            log(paths, f"Hook package bootstrap step failed (non-fatal): {exc}")

def estimate_chars_budget(config: dict[str, Any]) -> int:
    context_window = int(config["api"].get("context_window", 1000000))
    reserve = int(config["novel"].get("context_budget_reserve_chars", 40000))
    budget = max(context_window - reserve, 50000)
    # context_window is often set aspirationally (e.g. 1,000,000) while the real
    # model rejects or empties out on prompts far smaller than that. Without a
    # cap, memory_context can assemble an arbitrarily large block and overflow
    # the model's true limit. memory_context_max_chars is a realistic hard
    # ceiling on the assembled layered context, independent of context_window;
    # the per-section memory_*_chars caps already bound each tier well below it,
    # so this only bites pathological growth. 0/absent disables the cap.
    hard_cap = int(config["novel"].get("memory_context_max_chars", 150000) or 0)
    if hard_cap > 0:
        budget = min(budget, hard_cap)
    return budget

def truncate_section(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _read_memory_file(path: Path, cap: int) -> str:
    text = read_text(path).strip()
    if cap > 0 and len(text) > cap:
        return text[:cap] + "\n...[truncated]"
    return text


_HEADER_RE = re.compile(r"^#{2,6}\s")
# 「（第25-48章）」「（Ch31-48，修复中段塌陷追加）」「（Ch49-72）」都要能解析
_RANGE_RE = re.compile(r"(?:第|Ch|ch|CH)\s*(\d{1,4})\s*[-–—~至到]\s*(?:第|Ch|ch|CH)?\s*(\d{1,4})")
_ROW_CHAPTER_RE = re.compile(r"^\|\s*(?:第|Ch|ch|CH)?\s*(\d{1,4})\s*章?\s*\|")


def _header_range(line: str) -> tuple[int, int] | None:
    m = _RANGE_RE.search(line)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo, hi) if lo <= hi else (hi, lo)


def _filter_schedule_rows(body: str, lo: int, hi: int) -> str:
    """Drop per-chapter table rows outside [lo, hi]. Table headers / separators /
    rows without a chapter number are kept, so a filtered table stays readable."""
    out: list[str] = []
    for line in body.split("\n"):
        s = line.lstrip()
        if not s.startswith("|"):
            out.append(line)
            continue
        m = _ROW_CHAPTER_RE.match(s)
        if not m:  # 表头 / 分隔行 / 无章号的行
            out.append(line)
            continue
        if lo <= int(m.group(1)) <= hi:
            out.append(line)
    return "\n".join(out)


def volume_plan_window(text: str, chapter_num: int, cap: int, lookahead: int = 2) -> str:
    """Chapter-relevant view of volume_plan.md, replacing naive head truncation.

    Root cause it fixes: volume_plan grows per volume (and `extend_volume_schedule`
    APPENDS more), so `text[:cap]` means a book at Ch41 shows the writer only
    第一卷（Ch1-24）— the current volume's 角色高光轮值表 / 爽点兑现节拍表 /
    反转排期 / 伏笔兑现表 all sit beyond the window and were never executed
    (ensemble cast collapse, payoff_deferred, tension_flat).

    Strategy: keep the preamble; keep in full every `##`/`###`/`####` block whose
    header declares no chapter range OR whose range contains `chapter_num`; reduce
    out-of-range blocks to their header line as a breadcrumb; inside kept blocks
    keep only `| ChN |` schedule rows for chapters in
    [chapter_num-1, chapter_num+lookahead]. Pure function — no IO.
    """
    text = (text or "").strip()
    if not text or chapter_num <= 0:
        return text if cap <= 0 or len(text) <= cap else text[:cap] + "\n...[truncated]"

    lines = text.split("\n")
    # (header_line | None, body_lines)
    blocks: list[tuple[str | None, list[str]]] = []
    preamble: list[str] = []
    for line in lines:
        if _HEADER_RE.match(line):
            blocks.append((line, []))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            preamble.append(line)

    lo, hi = max(1, chapter_num - 1), chapter_num + max(0, lookahead)
    ranges = [(_header_range(h or "")) for h, _ in blocks]
    hit_any = any(r and r[0] <= chapter_num <= r[1] for r in ranges)
    # Safety net: a plan that never reaches this chapter (e.g. 卷纲 stops at Ch24
    # while the book runs to Ch41) must not collapse to headers only — keep the
    # ranged block closest to the current chapter verbatim (rows included).
    nearest = -1
    if not hit_any:
        best = None
        for i, r in enumerate(ranges):
            if not r:
                continue
            dist = min(abs(chapter_num - r[0]), abs(chapter_num - r[1]))
            if best is None or dist < best:
                best, nearest = dist, i

    parts: list[str] = []
    pre = "\n".join(preamble).strip()
    if pre:
        parts.append(pre)
    # Rangeless sub-blocks (e.g. `#### **角色高光轮值表**`) inherit the keep
    # decision of their nearest ranged ancestor, so a past volume's tables don't
    # survive as empty skeletons.
    ctx: list[tuple[int, bool]] = []
    for i, (header, body) in enumerate(blocks):
        head = str(header)
        level = len(head) - len(head.lstrip("#"))
        rng = ranges[i]
        while ctx and ctx[-1][0] >= level:
            ctx.pop()
        if rng is not None:
            keep_full = (rng[0] <= chapter_num <= rng[1]) or i == nearest
            ctx.append((level, keep_full))
        else:
            keep_full = ctx[-1][1] if ctx else True
        if not keep_full:
            parts.append(head.rstrip() + "  （非本章区间，正文略）")
            continue
        raw = "\n".join(body)
        kept = (raw if i == nearest else _filter_schedule_rows(raw, lo, hi)).strip("\n")
        parts.append(head.rstrip() + ("\n" + kept if kept.strip() else ""))

    out = "\n\n".join(p for p in parts if p.strip())
    if cap > 0 and len(out) > cap:
        out = out[:cap] + "\n...[truncated]"
    return out


def _read_volume_plan(paths: Paths, config: dict[str, Any], chapter_num: int, cap: int) -> str:
    text = read_text(paths.volume_plan).strip()
    if not bool(config["novel"].get("volume_plan_window_enabled", True)):
        return _read_memory_file(paths.volume_plan, cap)
    return volume_plan_window(
        text, chapter_num, cap,
        lookahead=int(config["novel"].get("volume_plan_window_lookahead", 2)),
    )


def _current_chapter_hint(conn: Any) -> int:
    """Chapter the pipeline is working on = last recorded chapter + 1. Used by the
    context builders, which have no chapter_num parameter. Any failure → 0, which
    makes volume_plan_window fall back to plain head truncation."""
    try:
        rows = recent_metrics(conn, 1)
        return int(rows[0].get("chapter", 0)) + 1 if rows else 0
    except Exception:
        return 0


def _recency_aware_state(raw: str, config: dict[str, Any], max_chars: int = 12000) -> str:
    """Structured truncation of state.md for writing_memory_context.

    Keeps the header (summary/progress/threads/direction) + the most recent N
    chapter sections (``## ChN``), dropping middle chapter sections that have
    already been consolidated by compress_all_memory.
    """
    recent_n = int(config["novel"].get("state_recent_chapters", 5))
    parts = re.split(r"(?=^## Ch\d)", raw, flags=re.MULTILINE)
    if len(parts) <= 1:
        if max_chars > 0 and len(raw) > max_chars:
            return raw[:max_chars] + "\n...[truncated]"
        return raw
    header = parts[0]
    ch_sections = parts[1:]
    def _ch_num(s: str) -> int:
        m = re.match(r"## Ch(\d+)", s)
        return int(m.group(1)) if m else 0
    ch_sections.sort(key=_ch_num)
    kept = ch_sections[-recent_n:] if len(ch_sections) > recent_n else ch_sections
    result = header + "".join(kept)
    if max_chars > 0 and len(result) > max_chars:
        kept_text = "".join(kept)
        avail = max_chars - len(kept_text)
        if avail > 400:
            result = header[:avail] + "\n...[truncated]\n" + kept_text
        else:
            result = result[:max_chars] + "\n...[truncated]"
    return result


def opening_route_text(paths: Paths, cap: int = 6000) -> str:
    path = paths.volume_plan.parent / "opening_route.md"
    return _read_memory_file(path, cap) if path.exists() else ""

# ---------------------------------------------------------------------------
# Context Profile: explicit mapping of which context builder each consumer uses.
# ---------------------------------------------------------------------------

def _pack_sections(sections: list[tuple[str, str, int]], budget: int,
                   header: str | None = None) -> str:
    """Render `(title, body, cap)` triples into `## title` blocks under a char budget.

    Extracted from three byte-identical copies (`cacheable_prefix`,
    `writing_memory_context`, `lite_memory_context`). The exact semantics are
    load-bearing and must not be "improved":

    - each body is capped INDEPENDENTLY at its own `cap`, then the budget is
      applied to the assembled result, so a fat early section cannot silently
      starve a later one of its own allowance;
    - on overflow the section is retried at whatever budget is left, but only if
      more than 400 chars remain — a 200-char fragment of a state dump is noise
      the model has to read anyway;
    - overflow BREAKS rather than continues: the `sections` order is a priority
      order, so skipping ahead to a smaller later section would silently reorder
      priorities.

    `cacheable_prefix` passes `header`, and its bytes are a prompt-cache key —
    any change to the assembly here invalidates the provider cache for every
    chapter of every book. That is also why a known off-by-14 is left alone: the
    overflow branch subtracts the `## title\n` header from `remaining` but not the
    trailing `\n...[truncated]` it then appends, so a truncated tail can overshoot
    `budget` by ~14 chars (measured: 26528 out of a 26513 budget). Harmless at
    these scales, and correcting it would rewrite the cache key for every book.
    """
    parts: list[str] = [header] if header else []
    used = len(header) if header else 0
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > budget:
            remaining = budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def memory_context(paths: Paths, conn: Any, config: dict[str, Any],
                   max_chars: int | None = None) -> str:
    budget = estimate_chars_budget(config)
    # 调用方预算上限（P2 降本）：规划候选/仲裁等消费点可以传一个远小于
    # context_window 推算值的预算，tier1/2 总能装下，tier3/4 自动截断。
    if max_chars is not None and int(max_chars) > 0:
        budget = min(budget, int(max_chars))
    fatigue_window = int(config["novel"]["fatigue_window"])

    creative_brief = read_text(PROMPT_FILE).strip()
    current_state = _read_memory_file(paths.state, int(config["novel"].get("memory_state_chars", 12000)))
    voice_anchor = _read_memory_file(paths.voice, int(config["novel"].get("memory_voice_chars", 8000)))
    voices_table = _read_memory_file(paths.voices, int(config["novel"].get("memory_voices_chars", 12000)))
    opening_route = opening_route_text(paths, int(config["novel"].get("memory_opening_route_chars", 6000)))
    style_block = ""
    if voice_anchor:
        style_block += "\n\n## 叙事声音锚（必须遵循）\n" + voice_anchor
    if voices_table:
        style_block += "\n\n## 人物声音（必须遵循）\n" + voices_table
    if opening_route:
        style_block += "\n\n## 已采纳开篇路线（优先级高于临场发散）\n" + opening_route
    tier1 = "## 创作纲要\n" + creative_brief + "\n\n## 当前状态\n" + current_state + style_block

    volume_plan = _read_volume_plan(
        paths, config, _current_chapter_hint(conn),
        int(config["novel"].get("memory_volume_plan_chars", 16000)),
    )
    # tier4 carries the OLDER TAIL of these two series, not a second full copy.
    # `recent_metrics`/`recent_events` are newest-first, so `recent_metrics(5)`
    # is a prefix of `recent_metrics(fatigue_window)` and `recent_events(20)` is
    # a prefix of `recent_events(40)`. Emitting both in full shipped the same
    # rows twice in every plan/extract prompt -- measured on a live Ch49 novel
    # that is 4,341 + 6,757 = 11.1k of the 121.6k context, ~9%, for zero
    # information. Slicing the tail is safe because the tiers assemble in order
    # and any short budget returns early: tier4 can only appear when tier2 and
    # tier3 already went in whole.
    metrics_recent = recent_metrics(conn, max(fatigue_window, 5))
    events_recent = recent_events(conn, 40, event_types=PLOT_EVENT_TYPES)

    metrics_5 = json.dumps(metrics_recent[:5], ensure_ascii=False, indent=2)
    threads_text = _read_memory_file(paths.threads, int(config["novel"].get("memory_threads_chars", 12000)))
    tier2 = "## 卷纲\n" + volume_plan + "\n\n## 关键指标JSON\n" + metrics_5 + "\n\n## 伏线\n" + threads_text

    characters = _read_memory_file(paths.characters, int(config["novel"].get("memory_characters_chars", 16000)))
    bible = _read_memory_file(paths.bible, int(config["novel"].get("memory_bible_chars", 16000)))
    events_20 = json.dumps(events_recent[:20], ensure_ascii=False, indent=2)
    tier3 = "## 人物\n" + characters + "\n\n## 世界设定\n" + bible + "\n\n## 近期事件JSON\n" + events_20

    timeline = _read_memory_file(paths.timeline, int(config["novel"].get("memory_timeline_chars", 10000)))
    metrics_older = metrics_recent[5:]
    events_older = events_recent[20:]
    tier4 = "## 时间线\n" + timeline
    if metrics_older:
        tier4 += ("\n\n## 更早的指标JSON（承接上面的关键指标，更旧）\n"
                  + json.dumps(metrics_older, ensure_ascii=False, indent=2))
    if events_older:
        tier4 += ("\n\n## 更早的事件JSON（承接上面的近期事件，更旧）\n"
                  + json.dumps(events_older, ensure_ascii=False, indent=2))

    assembled = tier1
    remaining = budget - len(assembled)

    if remaining > len(tier2):
        assembled += "\n\n" + tier2
        remaining = budget - len(assembled)
    else:
        assembled += "\n\n" + truncate_section(tier2, max(remaining - 100, 0))
        return assembled

    if remaining > len(tier3):
        assembled += "\n\n" + tier3
        remaining = budget - len(assembled)
    else:
        assembled += "\n\n" + truncate_section(tier3, max(remaining - 100, 0))
        return assembled

    if remaining > len(tier4):
        assembled += "\n\n" + tier4
    elif remaining > 2000:
        assembled += "\n\n" + truncate_section(tier4, max(remaining - 100, 0))

    return assembled

# Module-level cache for the cacheable prefix so that subsequent calls in the
# same process re-use the EXACT same string (byte-for-byte) when the underlying
# files are unchanged. The cache key is a sha1 of the source file contents +
# budget; when any source changes, the cache is rebuilt and a new prefix string
# is returned (so prefix cache invalidation matches content change).
#
# This also implements task #9 (memory hash skip): the hash is computed over
# bible/characters/voice/voices/prompt content; if all are unchanged since
# last call, the cached string is returned in O(1) (no re-read, no re-format,
# no truncation). Provider prefix caches see identical bytes -> ~free prefill.
_CACHEABLE_PREFIX_CACHE: dict[str, tuple[str, str]] = {}
_CACHEABLE_PREFIX_STATS = {"hits": 0, "misses": 0}


def _files_hash(paths_list: list[Path]) -> str:
    hasher = hashlib.sha1()
    for p in paths_list:
        try:
            data = p.read_bytes() if p.exists() else b""
        except OSError:
            data = b""
        hasher.update(str(p).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(hashlib.sha1(data).digest())
    return hasher.hexdigest()


def cacheable_prefix(
    paths: Paths,
    config: dict[str, Any],
    log_fn: Any = None,
) -> str:
    """Build the EXACT-same-bytes prompt prefix shared across calls.

    This prefix is included verbatim at the top of each LLM call's user message
    (via call_llm's cacheable_prefix arg). Provider-side prefix caches will hit
    as long as the bytes are identical, so we return the same cached string
    when the source files have not changed. On change, the cache key changes
    and downstream invocations naturally invalidate.
    """
    budget = int(config["novel"].get("cacheable_prefix_chars", 30000))
    sources = [PROMPT_FILE, paths.volume_plan.parent / "opening_route.md", paths.voice, paths.voices, paths.bible, paths.characters]
    key = f"{_files_hash(sources)}:{budget}"

    cached = _CACHEABLE_PREFIX_CACHE.get("active")
    if cached and cached[0] == key:
        _CACHEABLE_PREFIX_STATS["hits"] += 1
        return cached[1]
    _CACHEABLE_PREFIX_STATS["misses"] += 1

    creative_brief = _read_memory_file(PROMPT_FILE, 6000)
    voice_anchor = _read_memory_file(paths.voice, 8000)
    voices_table = _read_memory_file(paths.voices, 12000)
    opening_route = opening_route_text(paths, 6000)
    bible = _read_memory_file(paths.bible, 16000)
    characters = _read_memory_file(paths.characters, 16000)

    sections: list[tuple[str, str, int]] = [
        ("创作纲要", creative_brief, 4000),
        ("已采纳开篇路线", opening_route, 5000),
        ("叙事声音锚", voice_anchor, 5000),
        ("人物声音", voices_table, 7000),
        ("世界设定", bible, 7000),
        ("人物", characters, 7000),
    ]
    text = _pack_sections(sections, budget, header="# 稳定参照（可缓存）")
    _CACHEABLE_PREFIX_CACHE["active"] = (key, text)
    if log_fn is not None:
        try:
            stats = _CACHEABLE_PREFIX_STATS
            total = stats["hits"] + stats["misses"]
            hit_rate = (stats["hits"] / total * 100.0) if total else 0.0
            log_fn(
                f"cacheable_prefix rebuilt chars={len(text)} key={key[:12]} "
                f"hits={stats['hits']} misses={stats['misses']} hit_rate={hit_rate:.1f}%"
            )
        except Exception:
            pass
    return text


def writing_memory_context(paths: Paths, conn: Any, config: dict[str, Any],
                           pov_character: str | None = None) -> str:
    """Compact memory context for chapter writing.

    Excludes the content that is already shipped via cacheable_prefix() (creative
    brief, voice anchors, bible, characters). This keeps the variable portion
    small so prefix cache hits more, and avoids duplication.

    Sections (capped):
    - Current State (full state.md)
    - Threads (open)
    - Recent Metrics
    - Volume Plan (small)

    When pov_character is set, threads are filtered to sections mentioning that
    character (plus the most recent 3 sections), simulating limited POV knowledge.
    """
    char_budget = int(config["novel"].get("writing_memory_chars", 50000))

    state_chars = int(config["novel"].get("memory_state_chars", 12000))
    raw_state = _read_memory_file(paths.state, 0)
    current_state = _recency_aware_state(raw_state, config, max_chars=state_chars)
    threads_cap = int(config["novel"].get("memory_threads_chars", 12000))
    threads_text = _read_memory_file(paths.threads, threads_cap)
    if pov_character and threads_text and bool(config["novel"].get("pov_filter_enabled", True)):
        import re as _re
        sections = _re.split(r'(?=^## )', threads_text, flags=_re.MULTILINE)
        kept = []
        for s in sections:
            if not s.strip():
                continue
            if pov_character in s:
                kept.append(s)
        tail_sections = [s for s in sections if s.strip() and s not in kept]
        for s in tail_sections[-3:]:
            if s not in kept:
                kept.append(s)
        threads_text = "\n".join(kept)[:threads_cap]
    # 写作路径的卷纲窗口直接按最终额度裁（而不是先按 16000 建窗再头部截断，
    # 否则本章排期行仍会被切掉——这正是中段选角/爽点排期从未被执行的根因）。
    # 9000 是行过滤后一整套「本章相关」排期表的实测尺寸（约 8.9k），而 writing
    # 总预算 50000、其余分节合计约 26k，抬这个额度不挤压任何其他分节。
    vp_cap = int(config["novel"].get("memory_writing_volume_plan_chars", 9000))
    volume_plan = _read_volume_plan(paths, config, _current_chapter_hint(conn), vp_cap)
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)

    sections: list[tuple[str, str, int]] = [
        ("当前状态", current_state, 10000),
        ("伏线", threads_text, 8000),
        ("近期指标JSON", metrics_5, 2500),
        ("卷纲（本章相关节选）", volume_plan, vp_cap),
    ]
    return _pack_sections(sections, char_budget)
