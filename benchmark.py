from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import ROOT, Paths, read_text


PLATFORM_GUIDANCE = {
    "qidian_male": (
        "平台画像：起点男频。读者更接受长线设定、体系成长、伏笔回收与阶段性大高潮；"
        "前三章仍需清晰卖点，但可以保留中长线谜团。每章要有明确推进，避免纯解释设定。"
    ),
    "fanqie_free": (
        "平台画像：番茄免费阅读·强留存（字节系算法分发，读者多为下沉/碎片化场景，决策极快）。\n"
        "- 你的第一读者是算法不是人：算法只认『吸量×完读率×追读率』的乘积（总留存）来决定流量池升降。\n"
        "- 追读率是生命线：每章末必须留钩子，连续『平路』不得超过 2-3 章；完读率决定收益、追读率决定生死。\n"
        "- 三日留存=口碑评分核心：开篇 1-3 章必须建立『明天还想看』的钩子链，让读者第一时间产生强情绪。\n"
        "- 下沉语体：大白话、低阅读门槛、短句驱动、对话优先；金手指越简单越好，复杂绕口的设定劝退。\n"
        "- 高频爽点：每章至少 1 个明确情绪兑现（爽/愤/好奇/揪心之一），短周期爽点密度高，少慢热铺设定。\n"
        "- 可短剧化：强人设、强情感、强反转的结构既利追读也利 IP 改编；可截图的金句能驱动段评传播。"
    ),
    "jinjiang_female": (
        "平台画像：晋江/女频。关系张力、人设化学反应、情绪递进与角色能动性优先；"
        "冲突要落到人物选择和关系变化，避免只有外部事件推动。"
    ),
    "qimao_free": (
        "平台画像：免费阅读泛用户。节奏要直给，冲突和收益要低门槛可理解；"
        "每章都应有明显情绪收益或悬念推进，减少复杂专名堆叠。"
    ),
    "general": (
        "平台画像：通用网文。优先保证开篇卖点清晰、章节推进稳定、承诺及时兑现、重复模式不过度。"
    ),
}


def platform_guidance(config: dict[str, Any]) -> str:
    preset = str(config.get("novel", {}).get("platform_preset", "general")).strip() or "general"
    return PLATFORM_GUIDANCE.get(preset, PLATFORM_GUIDANCE["general"])


# Platform-specific GOLDEN-FIRST-CHAPTERS rules. The generic OPENING_RULES_BLOCK
# (writing.py) covers craft fundamentals; this layer encodes how each platform's
# audience decides whether to keep reading, which differs sharply (免费流要前几行
# 见冲突；起点男频容忍中线铺垫但要亮体系卖点). Injected into both the opening
# plan prompt and the opening writer prompt for chapter_num <= opening_chapters.
PLATFORM_OPENING = {
    "qidian_male": (
        "## 平台开篇专项（起点男频·黄金三章）\n"
        "- 卖点亮相：前 1/3 必须让读者看清本书的体系/金手指/主角核心反差，可保留中长线谜团但不能让首章只铺设定。\n"
        "- 主角立住：首章用一次可见的选择或冲突展示主角性格与处境，给出明确的长期目标方向。\n"
        "- 钩子：章末留一个能支撑追读的具体悬念或升级预期，而非泛泛的'命运改变'。\n"
        "- 信息克制：世界观边演边给，禁止整段名词/设定倾倒。"
    ),
    "fanqie_free": (
        "## 平台开篇专项（番茄免费·强留存·黄金三句）\n"
        "读者平均 3 秒决定去留、第 3 段就可能划走，黄金三章已进化到『黄金三句』，必须严格执行：\n"
        "- 句1·危机前置：正文第一句就是【正在发生】的危机/冲突/羞辱（动作或对白，具体可视），"
        "禁止任何回忆/铺垫/设定/天气/履历开场；写『正在出事』，不要写『将要出事』。\n"
        "- 句2·人设反差：开篇极短篇幅内立住主角的核心反差或反常行为（弱外表+强承诺、极端处境+反常举动），"
        "给读者一个记住他的理由，别把身份/底牌藏到几十章后。\n"
        "- 句3·金句钩子：章节/段落收在一句可截图、能传播的强情绪金句上（复仇宣言/逆袭宣言/认知颠覆/后果预告），"
        "独立成段，让读者产生强烈情绪或非看下一章不可的期待。\n"
        "- 低门槛卖点：本书爽点方向要一句话能懂，主角处境与诉求直给不绕弯；金手指简单明了，禁止复杂设定倾倒。\n"
        "- 高密度钩子：首章至少 2 个让人想往下翻的悬念/反转点，章末问题直观强烈。\n"
        "- 情绪优先：让读者第一时间产生爽/愤/好奇/揪心中的一种强情绪。\n"
        "- 人名≤5个，先出主角；前 1/4 内必须兑现一次小爽点，证明『这书会爽』。"
    ),
    "qimao_free": (
        "## 平台开篇专项（七猫免费·泛用户）\n"
        "- 直给冲突：开篇即抛出低门槛、可秒懂的冲突与利害关系，少用复杂专名。\n"
        "- 即时收益：首章就给读者一个明显的情绪收益或悬念推进，不慢热。\n"
        "- 主角动机清晰：让读者立刻明白主角要什么、被什么逼着走。\n"
        "- 章末强钩：用一个直观的危机或反转收尾。"
    ),
    "jinjiang_female": (
        "## 平台开篇专项（晋江女频·关系驱动）\n"
        "- 关系张力前置：首章即建立男女主（或核心关系）的化学反应或张力锚点，哪怕只是一次有潜台词的交锋。\n"
        "- 人设化学反应：通过具体互动展现人物魅力与反差，而非旁白介绍。\n"
        "- 情绪锚：给读者一个明确的情绪投射点（心动/意难平/好奇/护短欲）。\n"
        "- 能动性：女主在首章要有自己的选择与诉求，不做纯被动工具。"
    ),
    "general": (
        "## 平台开篇专项（通用·黄金三章）\n"
        "- 卖点清晰：首章让读者明白本书核心吸引力与主角处境。\n"
        "- 冲突前置：尽早抛出核心冲突或悬念，避免长铺垫。\n"
        "- 章末留钩：用具体悬念/反转/危机收尾，制造追读冲动。"
    ),
}


def platform_opening(config: dict[str, Any]) -> str:
    """Return the platform-specific golden-first-chapters opening rules."""
    preset = str(config.get("novel", {}).get("platform_preset", "general")).strip() or "general"
    return PLATFORM_OPENING.get(preset, PLATFORM_OPENING["general"])


# Golden-finger (金手指) design hints, injected into the PLANNER. 免费流读者偏好
# 简单直给、即时反馈的金手指；复杂绕口的设定劝退。付费/起点男频容忍复杂体系，
# 故仅对免费流预设输出该约束（其余返回 ""，不污染长线体系文）。
PLATFORM_GOLDEN_FINGER = {
    "fanqie_free": (
        "## 金手指设计约束（番茄免费流·本章大纲须遵守）\n"
        "- 简单直给：金手指/能力的运作要一句话能懂、即时反馈，禁止绕口的多层设定与一次性大段规则倾倒。\n"
        "- 有代价/稀缺：能力须有清晰代价、冷却或资源稀缺，不能无限白给——否则爽感快速通胀、读者麻木。\n"
        "- 兑现具体：本章用到金手指时要落到一次具体、可见的爽点场面，而非抽象描述其强大。"
    ),
    "qimao_free": (
        "## 金手指设计约束（七猫免费流·本章大纲须遵守）\n"
        "- 低门槛可懂：能力运作直给、少用复杂专名；每次使用都要有可感的情绪收益。\n"
        "- 有代价/稀缺：金手指须有代价或限制，避免无限白给导致爽感贬值。"
    ),
}


def platform_golden_finger(config: dict[str, Any]) -> str:
    """Return planner-side golden-finger constraints for free-flow presets; "" otherwise."""
    preset = str(config.get("novel", {}).get("platform_preset", "general")).strip() or "general"
    return PLATFORM_GOLDEN_FINGER.get(preset, "")


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^一-鿿A-Za-z0-9]", "", text)
    if len(cleaned) < 2:
        return set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _score(query_tokens: set[str], text: str) -> float:
    tokens = _tokenize(text[:6000])
    if not query_tokens or not tokens:
        return 0.0
    return len(query_tokens & tokens) / max(1, len(query_tokens | tokens))


def _candidate_dirs(config: dict[str, Any]) -> list[Path]:
    novel = config.get("novel", {})
    base = ROOT / str(novel.get("benchmark_dir", "benchmarks"))
    platform = str(novel.get("platform_preset", "general"))
    style = str(novel.get("style_preset", "history"))
    dirs = [base / platform / style, base / platform, base / style, base / "common", base]
    out: list[Path] = []
    for d in dirs:
        if d.exists() and d.is_dir() and d not in out:
            out.append(d)
    return out


def _read_benchmark(path: Path) -> tuple[str, str]:
    title = path.stem
    text = read_text(path)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            title = str(data.get("title") or title)
            chunks = []
            for key in ("summary", "opening", "chapter_1", "chapter_3", "payoff_pattern", "notes"):
                if data.get(key):
                    chunks.append(f"{key}: {data[key]}")
            text = "\n".join(chunks) or text
        except Exception:
            pass
    return title, text.strip()


def benchmark_context(
    paths: Paths,
    config: dict[str, Any],
    query: str,
    max_chars: int | None = None,
) -> str:
    """Return a compact local benchmark block for the current platform/genre.

    The project can drop markdown/txt/json samples under benchmarks/<platform>/<style>/.
    This helper is intentionally dependency-free; if no sample exists it returns
    an empty string and the pipeline behaves as before.
    """
    if not bool(config.get("novel", {}).get("benchmark_enabled", True)):
        return ""
    max_chars = int(max_chars or config.get("novel", {}).get("benchmark_context_chars", 5000))
    query_tokens = _tokenize(query)
    candidates: list[tuple[float, str, str, Path]] = []
    for d in _candidate_dirs(config):
        for path in d.glob("*"):
            if path.suffix.lower() not in {".md", ".txt", ".json"} or not path.is_file():
                continue
            if path.name.lower().startswith("readme."):
                continue
            title, text = _read_benchmark(path)
            if not text:
                continue
            candidates.append((_score(query_tokens, text), title, text, path))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    parts = ["## 爆款样本参照（只学习结构/节奏/读者承诺，不模仿具体表达）"]
    used = len(parts[0])
    for score, title, text, path in candidates[: int(config.get("novel", {}).get("benchmark_top_k", 3))]:
        snippet = text[:1800]
        block = f"### {title} score={score:.3f} source={path.relative_to(ROOT)}\n{snippet}"
        if used + len(block) + 2 > max_chars:
            remaining = max_chars - used - 80
            if remaining > 300:
                parts.append(block[:remaining] + "\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)
