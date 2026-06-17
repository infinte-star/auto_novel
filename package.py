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


def _build_client(config: dict[str, Any], paths: Paths) -> Any:
    """Reuse trial._build_client so `novel.py package` can run standalone."""
    from trial import _build_client as _bc
    return _bc(config, paths)


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
