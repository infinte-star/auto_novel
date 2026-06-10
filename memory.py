from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import PROMPT_FILE, Paths, log, normalize_text, read_text, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from store import db_event, recent_events, recent_metrics

if TYPE_CHECKING:
    from openai import OpenAI

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
全书层面要求：保持卷与卷之间的因果递进（上一卷遗留危机 = 下一卷核心矛盾的来源），
主角能力边界逐卷扩张但始终有约束，不得出现「主角一开始就全知全能」。
为控制 token，先详写前 2 卷，其余各卷给 1 段概要即可。

创作原创素材，不要模仿现有作品。以长期因果与读者期待为优化目标。

## ⚠ 章数上限（最高优先级，覆盖上面的卷数/章数默认值）
如果下方创作简报或附加约束中给出了明确的总章数上限（例如"全书 N 章封闭收束""限 N 章完结""max_chapters=N"等），
则 volume_plan 必须严格按这个上限来规划，**不得**沿用"3 卷 / 每卷 60-80 章"的长篇模板：
- 全书就是 N 章，章节区间不得超过 N（如"第1-6章"，禁止出现"第8-10章""第50章"等超出 N 的锚点）。
- 大事件锚点必须压缩到这 N 章之内，每个锚点标注它落在第几章（且 ≤ N），且最后一个高潮/真相锚点必须落在第 N 章或之前。
- 短篇不必强行分 3 卷；可只写 1 卷（或不分卷），把开局→升级→高潮→收束安排进 N 章。
- 这是硬约束：anchor 完成门会拿这些锚点逐章审计，锚点的章号若超出 N 会导致永远"未兑现"而错误地拖长全书。"""

CREATIVE_BOOST_SYSTEM = """你是一位顶尖网文创意策划，擅长跨题材、跨领域联想，把平庸的创作简报升级成有记忆点、有差异化的爆款雏形。
读取下方创作简报，结合多领域知识（历史、科技、神话、社会学、游戏机制、商业、悬疑结构等）做一次创意增强。
要求：具体、可执行、避免烂大街套路；不偏离简报的题材与核心设定，只在其骨架上注入新意。

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
  "ability_whitelist": [
    {"name": "能力名", "modality": "text|audio|visual|physical|cognitive|supernatural|other", "scope": "这个能力具体能做什么（边界）", "cost": "使用代价/限制（没有则写 none）"}
  ],
  "ability_blacklist": ["主角/关键人物明确做不到、不许做的事，每条一句话（如：不能凭空知道未亲自记录过的内容；记忆不能当法律证据；不能打斗）"],
  "banned_tropes": ["简报里明令禁止的套路，每条一句话（如：反派降智；主角全知全能；靠巧合/天降救兵解决主线；重大胜利零代价；用恐怖等贴标签词代替细节）；若简报中的核心能力/金手指可被反复使用，追加一条“同一能力的使用流程不得逐章原样复用，每次须在机制/代价/约束上有新变化”"],
  "must_hold": ["其它必须全程维持的硬设定，每条一句话（如：限制视角，只写视角人物当下能感知/推断的；关键揭示必须前文公平出现；终章必须收束不留新危机）"]
}

抽取纪律：
- 宁缺毋滥：只收作者真正钉死的红线；模糊的、探索留白的、风格偏好类内容不要收进来。
- 能力白名单只列主角及对剧情有关键作用的人物的**核心**能力，不要把普通技能（会开车、会做饭）也列上。
- 每条都要短、具体、可判定（一个审校者读完能直接判断某一章有没有违反）。
- 若简报几乎没有可抽取的硬约束，相应数组留空即可，不要硬凑。"""

MEMORY_COMPRESS_SYSTEM = """你负责压缩长篇小说引擎的记忆条目。
输入：一个含逐章条目（## ChN 小节）的记忆文件。
输出：一份整合后的 markdown，须保留：
- 所有实体名称及其当前状态（而非历史中间状态）
- 所有未解决的约束与已开启的伏线
- 所有对后续章节仍然相关的因果依赖
- 关键转折点与不可逆的变化
删除：已被取代的状态、例行确认、已解决项、冗余更新。
输出控制在 {max_chars} 个中文字符以内。
不要输出任何 `## ChN`（逐章）标题——只输出整合后的状态描述。
只输出整合后的内容，不要任何解释。"""

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
        lines = ["## 创意增强（请将以下新意自然融入设定，避免平庸化）"]
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
        if len(lines) <= 1:
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


def extract_contract(client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Extract the machine-checkable creative contract from prompt.md.

    Writes memory/contract.md and persists a `contract` event so the per-chapter
    write/review path can enforce author-declared hard rules (ability whitelist/
    blacklist, banned tropes, must-hold settings). Fail-degrades to {} so it can
    never block bootstrap.
    """
    if not bool(config["novel"].get("contract_enabled", True)):
        return {}
    try:
        raw = call_llm(
            client, paths, config, CONTRACT_SYSTEM,
            json_prompt(read_text(PROMPT_FILE)), temperature=0.3, tag="contract",
        )
        contract = load_json_with_repair(client, paths, config, raw, fallback={})
        if not isinstance(contract, dict) or not contract:
            return {}
        md = _contract_to_markdown(contract)
        if md:
            write_text(paths.contract, md + "\n")
        db_event(conn, 0, "contract", contract)
        return contract
    except Exception as e:  # never block bootstrap
        log(paths, f"extract_contract skipped: {e}")
        return {}


def contract_block(paths: Paths, config: dict[str, Any]) -> str:
    """Read memory/contract.md for injection into write/review prompts."""
    if not bool(config["novel"].get("contract_enabled", True)):
        return ""
    try:
        return _read_memory_file(paths.contract, int(config["novel"].get("memory_contract_chars", 6000)))
    except Exception:
        return ""


def contract_capsule(paths: Paths, config: dict[str, Any], cap: int = 800) -> str:
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
        wanted = ("能力白名单", "能力黑名单", "禁止套路")
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



def bootstrap(client: OpenAI, paths: Paths, conn: Any, config: dict[str, Any]) -> None:
    log(paths, "Bootstrapping layered memory")
    boost_block = creative_boost(client, paths, conn, config)
    brief = read_text(PROMPT_FILE)
    if boost_block:
        brief = brief + "\n\n" + boost_block
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
        )
    raw = call_llm(client, paths, config, BOOTSTRAP_SYSTEM, json_prompt(brief), temperature=0.7, tag="bootstrap")
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
    def _section(key: str, heading: str) -> str:
        val = _as_markdown(data.get(key))
        return val if val else f"# {heading}\n\n（bootstrap 未生成，待连载补全）"
    write_text(paths.state, _section("state", "当前状态") + "\n")
    write_text(paths.bible, _section("bible", "世界观圣经") + "\n")
    write_text(paths.characters, _section("characters", "人物") + "\n")
    write_text(paths.timeline, _section("timeline", "时间线") + "\n")
    write_text(paths.threads, _section("threads", "伏笔与线索") + "\n")
    write_text(paths.volume_plan, _section("volume_plan", "卷纲") + "\n")
    # Narrative-voice charter: this is the strongest anti-style-collapse anchor and
    # must exist from chapter 1. Only write it when the model produced one; an empty
    # value falls back to the placeholder created by ensure_project().
    voice_charter = _as_markdown(data.get("voice"))
    if voice_charter:
        write_text(paths.voice, voice_charter + "\n")
    db_event(conn, 0, "bootstrap", data)
    # Extract the machine-checkable creative contract (ability whitelist/blacklist,
    # banned tropes, must-hold settings) so the per-chapter write/review path can
    # enforce author-declared hard rules. Fail-degrades to {} (never blocks).
    contract = extract_contract(client, paths, conn, config)
    if contract:
        log(paths, "Extracted creative contract -> memory/contract.md")
    elif bool(config["novel"].get("contract_enabled", True)):
        # extract_contract fail-degrades to {} on any error (incl. transient 429).
        # A missing contract.md silently disables the ability-whitelist / modality
        # enforcement for the ENTIRE book — exactly the guard that caught 5/6 of
        # v4's breaches. Make the loss loud so it isn't mistaken for a clean run.
        log(
            paths,
            "WARNING: creative contract extraction returned empty — ability-boundary "
            "enforcement (whitelist/modality/blacklist) will be INACTIVE this run. "
            "This usually means the contract LLM call failed (quota/auth). Re-run "
            "after keys recover to restore contract enforcement.",
        )

def estimate_chars_budget(config: dict[str, Any]) -> int:
    context_window = int(config["api"].get("context_window", 1000000))
    reserve = int(config["novel"].get("context_budget_reserve_chars", 40000))
    return max(context_window - reserve, 50000)

def truncate_section(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _read_memory_file(path: Path, cap: int) -> str:
    text = read_text(path).strip()
    if cap > 0 and len(text) > cap:
        return text[:cap] + "\n...[truncated]"
    return text


def opening_route_text(paths: Paths, cap: int = 6000) -> str:
    path = paths.volume_plan.parent / "opening_route.md"
    return _read_memory_file(path, cap) if path.exists() else ""

def memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    budget = estimate_chars_budget(config)
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

    volume_plan = _read_memory_file(paths.volume_plan, int(config["novel"].get("memory_volume_plan_chars", 16000)))
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)
    threads_text = _read_memory_file(paths.threads, int(config["novel"].get("memory_threads_chars", 12000)))
    tier2 = "## 卷纲\n" + volume_plan + "\n\n## 关键指标JSON\n" + metrics_5 + "\n\n## 伏线\n" + threads_text

    characters = _read_memory_file(paths.characters, int(config["novel"].get("memory_characters_chars", 16000)))
    bible = _read_memory_file(paths.bible, int(config["novel"].get("memory_bible_chars", 16000)))
    events_20 = json.dumps(recent_events(conn, 20), ensure_ascii=False, indent=2)
    tier3 = "## 人物\n" + characters + "\n\n## 世界设定\n" + bible + "\n\n## 近期事件JSON\n" + events_20

    timeline = _read_memory_file(paths.timeline, int(config["novel"].get("memory_timeline_chars", 10000)))
    metrics_full = json.dumps(recent_metrics(conn, fatigue_window), ensure_ascii=False, indent=2)
    events_full = json.dumps(recent_events(conn, 40), ensure_ascii=False, indent=2)
    tier4 = "## 时间线\n" + timeline + "\n\n## 完整指标JSON\n" + metrics_full + "\n\n## 完整事件JSON\n" + events_full

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


def file_hash_short(path: Path) -> str:
    """Short sha1 (12 hex chars) of file content; '' if missing."""
    try:
        if not path.exists():
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    return hashlib.sha1(data).hexdigest()[:12]


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
    parts: list[str] = ["# 稳定参照（可缓存）"]
    used = len(parts[0])
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
    text = "\n\n".join(parts)
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


def cacheable_prefix_hit_rate() -> tuple[int, int]:
    """Return (hits, misses) for diagnostics."""
    return _CACHEABLE_PREFIX_STATS["hits"], _CACHEABLE_PREFIX_STATS["misses"]


def writing_memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    """Compact memory context for chapter writing.

    Excludes the content that is already shipped via cacheable_prefix() (creative
    brief, voice anchors, bible, characters). This keeps the variable portion
    small so prefix cache hits more, and avoids duplication.

    Sections (capped):
    - Current State (full state.md)
    - Threads (open)
    - Recent Metrics
    - Volume Plan (small)
    """
    char_budget = int(config["novel"].get("writing_memory_chars", 50000))

    current_state = _read_memory_file(paths.state, int(config["novel"].get("memory_state_chars", 12000)))
    threads_text = _read_memory_file(paths.threads, int(config["novel"].get("memory_threads_chars", 12000)))
    volume_plan = _read_memory_file(paths.volume_plan, int(config["novel"].get("memory_volume_plan_chars", 16000)))
    opening_route = opening_route_text(paths, int(config["novel"].get("memory_opening_route_chars", 6000)))
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)

    sections: list[tuple[str, str, int]] = [
        ("当前状态", current_state, 10000),
        ("已采纳开篇路线", opening_route, 5000),
        ("伏线", threads_text, 8000),
        ("近期指标JSON", metrics_5, 2500),
        ("卷纲（节选）", volume_plan, 6000),
    ]
    parts: list[str] = []
    used = 0
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > char_budget:
            remaining = char_budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _legacy_writing_memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    # Retained for reference only; not used after cacheable_prefix split.
    return ""


def lite_memory_context(paths: Paths, conn: Any, config: dict[str, Any]) -> str:
    """Slim memory context for plan-review and screening calls.

    Drops timeline, full events list, voices table, and recent_events from the
    full memory_context. Keeps the creative brief, current state, voice anchor,
    bible (capped), characters (capped), threads (capped), recent metrics 5 rows.
    """
    char_budget = int(config["novel"].get("plan_review_memory_chars", 10000))
    creative_brief = _read_memory_file(PROMPT_FILE, 3000)
    current_state = _read_memory_file(paths.state, 3500)
    voice_anchor = _read_memory_file(paths.voice, 2000)
    opening_route = opening_route_text(paths, 2500)
    bible = _read_memory_file(paths.bible, 2500)
    characters = _read_memory_file(paths.characters, 2500)
    threads_text = _read_memory_file(paths.threads, 2500)
    metrics_5 = json.dumps(recent_metrics(conn, 5), ensure_ascii=False, indent=2)

    sections: list[tuple[str, str, int]] = [
        ("创作纲要", creative_brief, 1500),
        ("当前状态", current_state, 2500),
        ("已采纳开篇路线", opening_route, 2000),
        ("叙事声音锚", voice_anchor, 1200),
        ("近期指标JSON", metrics_5, 1200),
        ("伏线", threads_text, 1500),
        ("人物", characters, 1500),
        ("世界设定", bible, 1200),
    ]
    parts: list[str] = []
    used = 0
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > char_budget:
            remaining = char_budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)

def should_compress_memory(paths: Paths, config: dict[str, Any], chapter_num: int) -> bool:
    compress_every = int(config["novel"].get("memory_compress_every", 30))
    max_kb = int(config["novel"].get("memory_max_kb", 15))
    if chapter_num > 0 and chapter_num % compress_every == 0:
        return True
    for p in [paths.bible, paths.characters, paths.timeline, paths.threads]:
        if p.exists() and p.stat().st_size > max_kb * 1024:
            return True
    return False

def compress_memory_file(
    client: OpenAI, paths: Paths, config: dict[str, Any], file_path: Path, keep_recent: int = 30
) -> None:
    content = read_text(file_path)
    if not content.strip():
        return
    sections = re.split(r"(?=^## Ch\d+)", content, flags=re.MULTILINE)
    if len(sections) <= 2:
        return
    header = sections[0]
    chapter_sections = sections[1:]
    if len(chapter_sections) <= keep_recent:
        return
    old_sections = chapter_sections[:-keep_recent]
    recent_sections = chapter_sections[-keep_recent:]
    archive_dir = paths.logs_dir / "memory_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    write_text(archive_path, "".join(old_sections))
    old_text = "".join(old_sections)
    # Derive the compression budget from memory_max_kb so a long book can keep
    # more live entity state instead of being clamped to a flat 3000 chars.
    mem_max_kb = int(config["novel"].get("memory_max_kb", 15))
    max_chars = max(3000, mem_max_kb * 1024 // 3)
    system = MEMORY_COMPRESS_SYSTEM.format(max_chars=max_chars)
    compressed = call_llm(client, paths, config, system, old_text, max_tokens=12000, temperature=0.2, tag="memory_compress")
    compressed = normalize_text(compressed)
    new_content = header.rstrip() + "\n\n## Consolidated\n" + compressed + "\n\n" + "".join(recent_sections)
    write_text(file_path, new_content)

def compress_all_memory(client: OpenAI, paths: Paths, config: dict[str, Any]) -> None:
    targets = [
        fp for fp in (paths.bible, paths.characters, paths.timeline, paths.threads)
        if fp.exists() and read_text(fp).strip()
    ]
    if not targets:
        return
    max_workers = int(config["novel"].get("max_parallel_workers", 8))

    def run_one(file_path: Path) -> tuple[Path, Exception | None]:
        try:
            compress_memory_file(client, paths, config, file_path)
            return file_path, None
        except Exception as exc:
            return file_path, exc

    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as executor:
        futures = {executor.submit(run_one, fp): fp for fp in targets}
        for future in as_completed(futures):
            fp, err = future.result()
            if err is not None:
                log(paths, f"compress_memory_file failed for {fp.name}: {err}")

def rhythm_diagnostics(conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    window = int(config["novel"]["repeat_window"])
    rows = recent_metrics(conn, window)
    if not rows:
        return {
            "warnings": [],
            "payoff_counts": {},
            "conflict_counts": {},
            "avg_tension": None,
            "avg_novelty": None,
            "avg_hook": None,
            "chapters_since_payoff": None,
        }

    payoff_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    tensions = []
    novelties = []
    hooks = []
    for row in rows:
        payoff_counts[row.get("payoff_type") or "unknown"] = payoff_counts.get(row.get("payoff_type") or "unknown", 0) + 1
        conflict_counts[row.get("conflict_type") or "unknown"] = conflict_counts.get(row.get("conflict_type") or "unknown", 0) + 1
        if row.get("tension") is not None:
            tensions.append(int(row["tension"]))
        if row.get("novelty") is not None:
            novelties.append(int(row["novelty"]))
        if row.get("hook_strength") is not None:
            hooks.append(int(row["hook_strength"]))

    # Payoff gap: distance (in chapters) from the most recent chapter back to the
    # last chapter whose payoff_type is a concrete reader payoff (not setup/emotional).
    # rows are ordered most-recent-first. 0 means the latest chapter itself paid off.
    payoff_realized = {
        "court_breakthrough", "policy_payoff", "military_victory", "reveal",
        "reversal", "personnel_payoff", "institutional_fix",
    }
    chapters_since_payoff: int | None = None
    for offset, row in enumerate(rows):
        if (row.get("payoff_type") or "") in payoff_realized:
            chapters_since_payoff = offset
            break
    if chapters_since_payoff is None:
        # No realized payoff anywhere in the window — treat the whole window as the gap.
        chapters_since_payoff = len(rows)

    warnings = []
    dominant_payoff = max(payoff_counts.items(), key=lambda x: x[1])
    dominant_conflict = max(conflict_counts.items(), key=lambda x: x[1])
    if dominant_payoff[1] >= max(4, window // 3):
        warnings.append(f"Payoff repetition risk: {dominant_payoff[0]} used {dominant_payoff[1]} times recently.")
    if dominant_conflict[1] >= max(4, window // 3):
        warnings.append(f"Conflict repetition risk: {dominant_conflict[0]} used {dominant_conflict[1]} times recently.")
    avg_novelty = sum(novelties) / len(novelties) if novelties else None
    avg_hook = sum(hooks) / len(hooks) if hooks else None
    if avg_novelty is not None and avg_novelty < 6:
        warnings.append("Novelty is low across recent chapters.")
    if avg_hook is not None and avg_hook < 6:
        warnings.append("Hook strength is low across recent chapters.")
    payoff_max_gap = int(config["novel"].get("payoff_max_gap", 99))
    if chapters_since_payoff >= payoff_max_gap:
        warnings.append(
            f"爽点拖欠：已 {chapters_since_payoff} 章无明确兑现（阈值 {payoff_max_gap}）；下一章 payoff_type 应为兑现类。"
        )

    return {
        "warnings": warnings,
        "payoff_counts": payoff_counts,
        "conflict_counts": conflict_counts,
        "avg_tension": sum(tensions) / len(tensions) if tensions else None,
        "avg_novelty": avg_novelty,
        "avg_hook": avg_hook,
        "chapters_since_payoff": chapters_since_payoff,
        "payoff_max_gap": payoff_max_gap,
    }

def beat_directive(
    volume_plan_text: str,
    chapter_num: int,
    est_total: int,
    chapters_since_payoff: int | None,
    payoff_max_gap: int,
    config: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Whole-book beat scheduler: locate this chapter against the volume_plan
    anchors/阶段高潮 and emit a directive telling the per-chapter planner which
    milestone should land around now.

    Pure parse+inject — no LLM call. Returns (directive_string, effective_gap).
    On any parse failure returns ("", payoff_max_gap) so behaviour degrades to
    the prior static-gap-only steering. effective_gap tightens payoff cadence
    when the located phase is a climax/兑现/真相 phase.
    """
    cfg = (config or {}).get("novel", {}) if config else {}
    if not bool(cfg.get("beat_scheduler_enabled", True)):
        return "", payoff_max_gap
    text = (volume_plan_text or "").strip()
    if not text or chapter_num < 1:
        return "", payoff_max_gap

    try:
        # Collect candidate beat lines: any line that names a 第N章 (or 第A-B章)
        # under 阶段高潮 / 大事件锚点 sections, plus volume headers for context.
        lines = text.splitlines()
        # A "beat" = (nearest_chapter_int, raw_line_cleaned).
        beats: list[tuple[int, str]] = []
        climax_kw = ("高潮", "兑现", "真相", "反转", "对峙", "揭破", "决战", "收网", "锁定", "代价")
        # Track which beat chapters fall in a climax-flavored context line.
        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue
            # Find all chapter numbers referenced on this line.
            nums = [int(m) for m in re.findall(r"第\s*(\d{1,4})\s*[章\-－—]", ln)]
            # Also catch a trailing "第N章" without separator.
            nums += [int(m) for m in re.findall(r"第\s*(\d{1,4})\s*章", ln)]
            if not nums:
                continue
            # Use the chapter number on this line closest to chapter_num.
            nearest = min(nums, key=lambda x: abs(x - chapter_num))
            # Clean markdown emphasis / list markers for a compact directive.
            clean = re.sub(r"[\*#`>]+", "", ln).strip(" -—·").strip()
            if len(clean) > 120:
                clean = clean[:120] + "…"
            beats.append((nearest, clean))

        if not beats:
            return "", payoff_max_gap

        # Pick the 1-3 beats whose chapter is closest to (and ideally >=) the
        # current chapter — the milestones "due around now / just ahead".
        beats_sorted = sorted(beats, key=lambda b: (abs(b[0] - chapter_num), b[0]))
        # De-dup by cleaned text while preserving order.
        seen: set[str] = set()
        picked: list[tuple[int, str]] = []
        for ch, clean in beats_sorted:
            if clean in seen:
                continue
            seen.add(clean)
            picked.append((ch, clean))
            if len(picked) >= 3:
                break

        # Climax detection: is the nearest beat a climax/payoff-flavored one?
        nearest_beat = picked[0] if picked else None
        is_climax_phase = bool(
            nearest_beat
            and any(kw in nearest_beat[1] for kw in climax_kw)
            and abs(nearest_beat[0] - chapter_num) <= 1
        )

        effective_gap = payoff_max_gap
        if is_climax_phase:
            tighten = int(cfg.get("beat_climax_tighten", 2))
            effective_gap = max(2, payoff_max_gap - tighten)

        beat_lines = "\n".join(f"- （第{ch}章）{clean}" for ch, clean in picked)
        since = (
            f"距上次明确兑现已 {chapters_since_payoff} 章。"
            if isinstance(chapters_since_payoff, int)
            else ""
        )
        climax_note = (
            "本章处于卷纲高潮/兑现区段，节奏须更紧：本章应给出兑现类 payoff，不得只做铺垫。"
            if is_climax_phase
            else ""
        )
        directive = (
            f"## 全书节拍（卷纲定位）\n"
            f"你正处于第 {chapter_num} 章（全书约 {est_total} 章）。按卷纲，本阶段临近的里程碑/阶段高潮如下，"
            f"本章须朝其中最近的一个推进或兑现，不得原地空转：\n"
            f"{beat_lines}\n"
            f"{since}{climax_note}"
        ).strip()
        return directive, effective_gap
    except Exception:
        return "", payoff_max_gap


def structural_repetition_analysis(conn: Any, config: dict[str, Any]) -> dict[str, Any]:
    window = int(config["novel"]["repeat_window"])
    rows = recent_metrics(conn, window)
    result: dict[str, Any] = {"warnings": [], "repeated_patterns": [], "tension_shape": "unknown"}
    if len(rows) < 6:
        return result

    sequence = [
        (r.get("conflict_type", ""), r.get("payoff_type", ""), r.get("emotional_tone", ""))
        for r in reversed(rows)
    ]

    # Sliding window pattern detection (window size 3)
    seen_patterns: dict[str, int] = {}
    for i in range(len(sequence) - 2):
        pattern_key = "|".join(f"{s[0]},{s[1]}" for s in sequence[i : i + 3])
        seen_patterns[pattern_key] = seen_patterns.get(pattern_key, 0) + 1
    repeated = [(k, v) for k, v in seen_patterns.items() if v >= 2]
    if repeated:
        result["repeated_patterns"] = [k for k, _ in repeated]
        result["warnings"].append(f"Repeated arc patterns detected: {len(repeated)} patterns appear 2+ times")

    # Tension curve shape analysis
    tensions = [int(r.get("tension", 5)) for r in reversed(rows) if r.get("tension") is not None]
    if len(tensions) >= 6:
        diffs = [tensions[i + 1] - tensions[i] for i in range(len(tensions) - 1)]
        flat_count = sum(1 for d in diffs if abs(d) <= 1)
        if flat_count > len(diffs) * 0.7:
            result["tension_shape"] = "flat"
            result["warnings"].append("Tension curve is flat — lacking dramatic variation")
        else:
            rises = sum(1 for d in diffs if d > 0)
            falls = sum(1 for d in diffs if d < 0)
            if rises > len(diffs) * 0.7:
                result["tension_shape"] = "monotone_rise"
            elif falls > len(diffs) * 0.7:
                result["tension_shape"] = "monotone_fall"
                result["warnings"].append("Tension is monotonically falling — reader engagement at risk")
            else:
                result["tension_shape"] = "varied"

    # Resolution monotony: check if emotional_tone repeats
    tones = [r.get("emotional_tone", "") for r in reversed(rows) if r.get("emotional_tone")]
    if len(tones) >= 5:
        tone_counts: dict[str, int] = {}
        for t in tones:
            tone_counts[t] = tone_counts.get(t, 0) + 1
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])
        if dominant_tone[1] >= len(tones) * 0.6:
            result["warnings"].append(f"Emotional monotony: '{dominant_tone[0]}' dominates {dominant_tone[1]}/{len(tones)} chapters")

    return result
