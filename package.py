"""Book packaging & chapter-title refinement (书名/简介/概要 + 章节起名).

Two related post-/inline-generation utilities, both decoupled from the core
plan→write→review loop and both strictly optional:

1. build_package() — after a book finishes, produce A/B-testable cover copy
   (titles / intros / tags / first-paragraph candidates) PLUS a spoiler and a
   clean (spoiler-free) synopsis. Writes package.md + logs/package.json. Does
   NOT touch chapters/ or book.md, and never modifies cacheable_prefix sources.
   Wired into pipeline.main (gated by novel.package_after_complete) and exposed
   as `novel.py package <name>`.

2. refine_chapter_title() — turn a plan's working title into a hook-y,
   non-spoilery short chapter title. Used (optionally) just before save_chapter
   to rewrite ONLY the chapter's first title line. Falls back to the plan title
   on any failure. Gated by novel.chapter_title_refine_enabled (default False).

Both reuse trial.PACKAGE_SYSTEM where possible and add a SYNOPSIS_SYSTEM here.
All LLM calls go through call_llm + load_json_with_repair(fallback=...), so a
provider hiccup degrades gracefully instead of crashing the pipeline.
"""
from __future__ import annotations

import json
import re
from typing import Any

from config import Paths, log, read_text, write_text
from llm import call_llm, json_prompt, load_json_with_repair
from memory import cacheable_prefix, memory_context


SYNOPSIS_SYSTEM = """你是网文平台的内容运营，负责为已完结作品撰写【作品概要】。
只返回恰好一个合法 JSON 对象，不要输出其它内容：
{
  "synopsis_spoiler": "<面向编辑/平台的完整剧情概要，可含结局与关键反转，400-700字>",
  "synopsis_clean": "<面向读者的无剧透简介，只抛卖点/钩子/核心冲突，不剧透结局，150-300字>",
  "one_line": "<一句话卖点(<=40字)>",
  "themes": ["核心主题/标签词，3-6个"]
}
要求：
- spoiler 版按时间线梳理主线与关键转折，逻辑连贯，便于编辑快速了解全书。
- clean 版只制造期待、不泄底，结尾收束在悬念或情绪钩子上。
- 不要营销空话，紧扣这本书实际写出来的内容。"""


CHAPTER_TITLE_SYSTEM = """你是中文网文的章节起名编辑。
给你一章的剧情要点，请起一个【钩子化、不剧透】的短章节标题。
只返回恰好一个合法 JSON 对象，不要输出其它内容：
{"title": "<不带'第N章'前缀的纯标题，6-16字，制造悬念或情绪张力，不得剧透本章结局或关键反转>"}
要求：标题要勾人想点开，但不能把本章的核心反转/结局写进标题；不要书名号、不要标点堆砌。"""


HOOK_PACKAGE_SYSTEM = """你是免费阅读平台（番茄为主）的爆款选品/运营，负责在作品【开写之前】先定下"吸量包"。
吸量是流量漏斗第一层：书名/简介决定点击率，烂书名能直接干掉九成机会。请据下方设定产出可 A/B 的吸量素材。
只返回恰好一个合法 JSON 对象，不要输出其它内容：
{
  "titles": ["5 个候选书名，按吸量从高到低"],
  "intros": ["2-3 个候选简介，每个 80-120 字，三段式"],
  "one_line": "一句话最大卖点/爽点(<=30字)",
  "tags": ["题材/卖点标签词，4-8个"],
  "hook_directives": ["3-5 条给开篇作者的吸量落地指令：开篇必须兑现书名/简介承诺的哪个爽点"]
}
书名公式：用大白话一句话剧透最大爽点或冲突，强画面、可短剧化；男频突出系统/无敌/重生/战神，女频突出甜宠/闪婚/马甲/重生/团宠；禁文艺腔、禁抽象、禁看不懂的双关。
简介三段式：①主角身份+开局困境（一句话给标签，不铺垫）②核心反差/独家能力（最值钱的一句，卖设定，越具体越好）③情绪承诺+钩子收尾（留未解谜团，禁剧透结局）；禁形容词大杂烩、禁全程"TA"指代。
要求：紧扣下方实际设定，不要营销空话；若平台画像非番茄，按其调性微调，但仍以"点击率优先"为准。"""


HOOK_SCORE_SYSTEM = """你是免费阅读平台（番茄为主）的爆款选品总监，专门给【尚未开写】作品的吸量素材打分。
吸量是流量漏斗第一层：书名/简介决定封面点击率，烂书名直接干掉九成机会。你要像算法+下沉读者那样冷酷判断"会不会点进去"。
给你一组候选书名、候选简介、标签和题材设定，请独立打分排序。只返回恰好一个合法 JSON 对象，不要输出其它内容：
{
  "ranked_titles": [{"title": "<原候选书名>", "ctr_score": 1-10, "reason": "<=40字打分理由"}],
  "best_title": "<ctr_score 最高的那个书名原文>",
  "ranked_intros": [{"intro": "<原候选简介前20字…>", "score": 1-10, "reason": "<=40字理由"}],
  "track_eval": {
    "verdict": "蓝海|偏蓝海|偏红海|红海",
    "differentiation": "<这本在赛道里的差异化空间，一句话>",
    "suggestion": "<提升吸量/差异化的一条最关键建议>"
  }
}
书名打分维度（点击率优先）：①大白话、零阅读门槛 ②强画面/强冲突、一句话剧透最大爽点 ③可短剧化（强人设+强反转）④差异化（避免与红海同质化的歪嘴龙王/烂大街标题）⑤男频突出系统/无敌/重生/战神，女频突出甜宠/闪婚/马甲/团宠。
扣分项：文艺腔、抽象、看不懂的双关、形容词大杂烩、与海量同类雷同。
ranked_titles 必须覆盖每一个候选书名并按 ctr_score 从高到低排序；best_title 必须是其中分最高的一个。"""


def _build_client(config: dict[str, Any], paths: Paths) -> Any:
    """Reuse trial._build_client so `novel.py package` can run standalone."""
    from trial import _build_client as _bc
    return _bc(config, paths)


def score_hook_package(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    pkg: dict[str, Any],
) -> dict[str, Any]:
    """Independently score/rank a hook package's titles & intros + evaluate the赛道.

    Runs an independent "吸量评判" LLM call (deliberately WITHOUT cacheable_prefix,
    like cold_reader/reader_panel — it must judge click-through cold, not be
    steeped in the book's own framing). Writes logs/hook_package_scored.json and
    appends a section to hook_package.md. Optionally adopts the best-scored title
    into paths.title (title.txt is NOT a cacheable_prefix source, so zero cache
    impact). Returns {} and logs on any failure; never raises.
    """
    if not bool(config["novel"].get("hook_package_scoring_enabled", True)):
        return {}
    titles = [str(t).strip() for t in (pkg.get("titles") or []) if str(t).strip()]
    intros = [str(i).strip() for i in (pkg.get("intros") or []) if str(i).strip()]
    if not titles:
        return {}
    try:
        from benchmark import platform_guidance
        platform = platform_guidance(config)
    except Exception:
        platform = ""
    try:
        user = f"""## 平台/读者画像
{platform}

## 候选书名
{json.dumps(titles, ensure_ascii=False, indent=2)}

## 候选简介
{json.dumps(intros, ensure_ascii=False, indent=2)}

## 一句话卖点
{pkg.get("one_line", "")}

## 标签
{json.dumps(pkg.get("tags") or [], ensure_ascii=False)}

请按点击率优先独立给书名/简介打分排序，并评估赛道（红海/蓝海+差异化空间）。"""
        raw = call_llm(
            client, paths, config, HOOK_SCORE_SYSTEM, json_prompt(user),
            max_tokens=int(config["novel"].get("hook_package_score_max_tokens", 3000) or 3000),
            temperature=0.3, tag="hook_package_score",
        )
        scored = load_json_with_repair(client, paths, config, raw, fallback={})
        if not isinstance(scored, dict) or not scored.get("ranked_titles"):
            return {}
    except Exception as exc:
        log(paths, f"Hook package scoring failed (non-fatal): {exc}")
        return {}

    # Persist machine-readable scores.
    try:
        write_text(paths.logs_dir / "hook_package_scored.json", json.dumps(scored, ensure_ascii=False, indent=2))
    except Exception as exc:
        log(paths, f"Failed to write hook_package_scored.json (non-fatal): {exc}")

    # Append a human-readable section to hook_package.md (if it exists).
    try:
        md_path = paths.book.with_name("hook_package.md")
        existing = read_text(md_path) if md_path.exists() else "# 吸量包\n"
        lines = ["\n## 吸量评分（独立评判·点击率优先）\n"]
        for rt in scored.get("ranked_titles", [])[:10]:
            if isinstance(rt, dict):
                lines.append(f"- [{rt.get('ctr_score', '?')}] {rt.get('title', '')} — {rt.get('reason', '')}")
        if scored.get("best_title"):
            lines.append(f"\n**采纳书名**：{scored['best_title']}\n")
        te = scored.get("track_eval") or {}
        if te:
            lines.append(f"\n### 赛道评估\n- 判定：{te.get('verdict', '')}\n"
                         f"- 差异化空间：{te.get('differentiation', '')}\n"
                         f"- 建议：{te.get('suggestion', '')}\n")
        write_text(md_path, existing.rstrip() + "\n" + "\n".join(lines) + "\n")
    except Exception as exc:
        log(paths, f"Failed to append hook_package.md scores (non-fatal): {exc}")

    # Adopt the best-scored title into title.txt (safe: not a cacheable_prefix source).
    best = str(scored.get("best_title") or "").strip()
    if best and bool(config["novel"].get("hook_package_adopt_title", True)):
        try:
            write_text(paths.title, best)
            log(paths, f"Hook package: adopted best title -> {best!r}")
        except Exception as exc:
            log(paths, f"Failed to adopt best title (non-fatal): {exc}")
    log(paths, f"Hook package scored ({len(scored.get('ranked_titles', []))} titles ranked); best={best!r}")
    return scored


def build_hook_package(
    client: Any,
    paths: Paths,
    conn: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Pre-generation 吸量包: 番茄式书名/三段式简介候选 + 开篇吸量指令.

    Called from bootstrap so naming is decided BEFORE writing (吸量是漏斗第一层).
    Writes hook_package.md + logs/hook_package.json. Advisory artifact — does NOT
    feed cacheable_prefix (zero cache impact) and never modifies bible/characters.
    Returns {} and logs on any failure; never raises (mirrors build_package).
    """
    if not bool(config["novel"].get("hook_package_enabled", True)):
        return {}
    try:
        mem = memory_context(paths, conn, config)
    except Exception:
        mem = ""
    try:
        from benchmark import platform_guidance
        platform = platform_guidance(config)
    except Exception:
        platform = ""
    try:
        user = f"""## 平台/读者画像
{platform}

## 作品设定 / 全局记忆
{mem}

为这本【尚未开写】的作品产出吸量包（书名候选/三段式简介候选/一句话卖点/标签/开篇吸量指令）。"""
        raw = call_llm(
            client, paths, config, HOOK_PACKAGE_SYSTEM, json_prompt(user),
            max_tokens=int(config["novel"].get("hook_package_max_tokens", 4000) or 4000),
            temperature=0.8, tag="hook_package",
        )
        pkg = load_json_with_repair(client, paths, config, raw, fallback={})
        if not isinstance(pkg, dict) or not pkg:
            return {}
        try:
            write_text(paths.logs_dir / "hook_package.json", json.dumps(pkg, ensure_ascii=False, indent=2))
        except Exception as exc:
            log(paths, f"Failed to write hook_package.json (non-fatal): {exc}")
        try:
            md = ["# 吸量包（开写前·书名/简介候选）\n"]
            if pkg.get("one_line"):
                md.append(f"> {pkg['one_line']}\n\n")
            md.append(_render_package_md({
                "titles": pkg.get("titles"),
                "intros": pkg.get("intros"),
                "tags": pkg.get("tags"),
            }))
            if pkg.get("hook_directives"):
                md.append(_section_lines("开篇吸量指令", pkg.get("hook_directives")))
            write_text(paths.book.with_name("hook_package.md"), "".join(p for p in md if p))
        except Exception as exc:
            log(paths, f"Failed to write hook_package.md (non-fatal): {exc}")
        log(paths, f"Hook package generated (titles/intros/one_line) — {len(pkg.get('titles') or [])} title candidates")
        return pkg
    except Exception as exc:
        log(paths, f"Hook package generation failed (non-fatal): {exc}")
        return {}


def _section_lines(title: str, items: Any) -> str:
    if not items:
        return ""
    if isinstance(items, list):
        body = "\n".join(f"- {it}" for it in items)
    else:
        body = str(items)
    return f"## {title}\n{body}\n\n"


def _gather_book_digest(paths: Paths, config: dict[str, Any], max_chars: int = 24000) -> str:
    """Build a compact digest of the finished book for synopsis generation.

    Prefers the layered memory_context (already char-budgeted) + a head/tail
    sample of book.md so the synopsis call sees both the structured state and
    the actual opening/closing prose without shipping the whole multi-MB book.
    """
    parts: list[str] = []
    try:
        from store import init_db
        conn = init_db(paths)
        mem = memory_context(paths, conn, config)
        if mem:
            parts.append("## 全局记忆\n" + mem)
    except Exception:
        pass
    book = read_text(paths.book)
    if book.strip():
        head = book[:8000]
        tail = book[-6000:] if len(book) > 14000 else ""
        parts.append("## 开篇片段\n" + head)
        if tail:
            parts.append("## 结尾片段\n" + tail)
    digest = "\n\n".join(parts)
    return digest[:max_chars]


def build_package(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate cover-copy + synopsis for a finished book.

    Returns the merged package dict (also persisted to logs/package.json and
    rendered to package.md). Returns {} on total failure. Never raises.
    """
    try:
        from store import init_db
        conn = init_db(paths)
        mem = memory_context(paths, conn, config)
    except Exception:
        mem = ""
    prefix = ""
    try:
        prefix = cacheable_prefix(paths, config)
    except Exception:
        prefix = ""

    package: dict[str, Any] = {}

    # 1) A/B cover copy via the shared trial PACKAGE_SYSTEM.
    try:
        from trial import PACKAGE_SYSTEM
        pkg_user = f"""## 全局记忆 / 设定
{mem}

为这本【已完结】作品生成可 A/B 测试的开篇包装(标题/简介/标签/正文第一段候选)。"""
        raw = call_llm(
            client, paths, config, PACKAGE_SYSTEM, json_prompt(pkg_user),
            max_tokens=int(config["novel"].get("package_max_tokens", 12000) or 12000),
            temperature=0.75, cacheable_prefix=prefix or None, tag="package",
        )
        cover = load_json_with_repair(client, paths, config, raw, fallback={})
        if isinstance(cover, dict):
            package.update(cover)
    except Exception as exc:
        log(paths, f"Package cover-copy generation failed (non-fatal): {exc}")

    # 2) Synopsis (spoiler + clean) from a book digest.
    try:
        digest = _gather_book_digest(paths, config)
        syn_user = f"""## 作品摘要素材(记忆 + 开篇/结尾片段)
{digest}

为这本已完结作品撰写概要(spoiler 版 + 无剧透 clean 版)。"""
        raw = call_llm(
            client, paths, config, SYNOPSIS_SYSTEM, json_prompt(syn_user),
            max_tokens=int(config["novel"].get("synopsis_max_tokens", 4000) or 4000),
            temperature=0.5, cacheable_prefix=prefix or None, tag="synopsis",
        )
        syn = load_json_with_repair(client, paths, config, raw, fallback={})
        if isinstance(syn, dict):
            for k in ("synopsis_spoiler", "synopsis_clean", "one_line", "themes"):
                if k in syn:
                    package[k] = syn[k]
    except Exception as exc:
        log(paths, f"Synopsis generation failed (non-fatal): {exc}")

    if not package:
        return {}

    # Persist: machine-readable json + human-readable markdown.
    try:
        write_text(paths.logs_dir / "package.json", json.dumps(package, ensure_ascii=False, indent=2))
    except Exception as exc:
        log(paths, f"Failed to write package.json (non-fatal): {exc}")
    try:
        write_text(paths.book.with_name("package.md"), _render_package_md(package))
    except Exception as exc:
        log(paths, f"Failed to write package.md (non-fatal): {exc}")
    log(paths, "Package generated (titles/intros/tags/synopsis)")
    return package


def _render_package_md(package: dict[str, Any]) -> str:
    def _section(title: str, items: Any) -> str:
        if not items:
            return ""
        if isinstance(items, list):
            body_lines = []
            for it in items:
                if isinstance(it, list):
                    body_lines.append("- " + "、".join(str(x) for x in it))
                else:
                    body_lines.append(f"- {it}")
            body = "\n".join(body_lines)
        else:
            body = str(items)
        return f"## {title}\n{body}\n\n"

    out = ["# 作品包装素材\n"]
    if package.get("one_line"):
        out.append(f"> {package['one_line']}\n\n")
    out.append(_section("书名候选", package.get("titles")))
    out.append(_section("简介候选", package.get("intros")))
    out.append(_section("标签", package.get("tags")))
    out.append(_section("正文第一段候选", package.get("first_paragraphs")))
    out.append(_section("主题词", package.get("themes")))
    if package.get("synopsis_clean"):
        out.append(f"## 无剧透简介\n{package['synopsis_clean']}\n\n")
    if package.get("synopsis_spoiler"):
        out.append(f"## 完整概要(含剧透)\n{package['synopsis_spoiler']}\n\n")
    out.append(_section("包装建议", package.get("package_notes")))
    return "".join(p for p in out if p)


# 第N章 title-line matcher: captures the "第N章" prefix and the trailing title.
_TITLE_LINE_RE = re.compile(
    r"^(\s*第\s*[0-9零一二三四五六七八九十百千两]+\s*章)(.*)$"
)


def refine_chapter_title(
    client: Any,
    paths: Paths,
    config: dict[str, Any],
    chapter_num: int,
    plan: dict[str, Any],
    chapter_text: str,
) -> str:
    """Return a hook-y, non-spoilery short chapter title (no 第N章 prefix).

    Falls back to the plan's title on any failure, so the caller can use the
    result unconditionally. Pure LLM helper; does not touch files.
    """
    fallback = str(plan.get("title") or "").strip()
    if not bool(config["novel"].get("chapter_title_refine_enabled", False)):
        return fallback
    try:
        beats = [str(b).strip() for b in (plan.get("beats") or []) if str(b).strip()][:6]
        user = f"""## 本章剧情要点
- 工作标题：{fallback or "(无)"}
- 冲突：{plan.get("conflict_type", "")}
- 兑现：{plan.get("payoff_type", "")}
- beats：{json.dumps(beats, ensure_ascii=False)}

为第 {chapter_num} 章起一个钩子化、不剧透的短标题。"""
        raw = call_llm(
            client, paths, config, CHAPTER_TITLE_SYSTEM, json_prompt(user),
            max_tokens=400, temperature=0.7, tag="chapter_title",
        )
        data = load_json_with_repair(client, paths, config, raw, fallback={})
        title = str((data or {}).get("title", "")).strip() if isinstance(data, dict) else ""
        # Strip any accidental 第N章 prefix / book brackets the model added.
        title = re.sub(r"^\s*第\s*[0-9零一二三四五六七八九十百千两]+\s*章\s*", "", title)
        title = title.strip(" 　《》「」“”\"'：:—-")
        if title and len(title) <= 30:
            log(paths, f"Refined chapter title Ch{chapter_num}: {fallback!r} -> {title!r}")
            return title
        return fallback
    except Exception as exc:
        log(paths, f"Chapter title refine failed (non-fatal) Ch{chapter_num}: {exc}")
        return fallback


def apply_chapter_title(chapter_text: str, chapter_num: int, new_title: str) -> str:
    """Replace ONLY the title portion of the chapter's first 第N章 line.

    Keeps the "第N章" prefix and all body prose verbatim; swaps the trailing
    title text for `new_title`. If the first line has no 第N章 marker, or
    new_title is empty, the text is returned unchanged.
    """
    if not new_title or not new_title.strip():
        return chapter_text
    new_title = new_title.strip()
    lines = chapter_text.split("\n", 1)
    first = lines[0]
    rest = lines[1] if len(lines) > 1 else ""
    m = _TITLE_LINE_RE.match(first)
    if not m:
        return chapter_text
    prefix = m.group(1).rstrip()
    new_first = f"{prefix} {new_title}"
    return new_first + ("\n" + rest if rest or len(lines) > 1 else "")
