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
    return paths.chapters_dir.parent / REFINED_DIR_NAME

def refined_chapter_path(paths: Paths, chapter_num: int) -> Path:
    return refined_dir(paths) / f"{chapter_num:04d}.md"

def refined_book_path(paths: Paths) -> Path:
    return paths.book.parent / REFINED_BOOK

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

DIAGNOSE_SYSTEM = """你是一位资深小说编辑。你的任务是诊断一组连续章节存在的问题，并决定每一章的精调强度。

精调强度等级（强度逐级提高）：
- "polish"：仅修润：错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节/对话/结构。
- "restructure"：允许重写段落、合并/拆分场景、调整节奏；不改变章节标题、关键情节点、人物决策。
- "rewrite"：允许重新设计场景顺序、改写大段叙事、补充心理描写；保持每章核心目标和章末状态。

你需要：
1. 阅读这一组章节，识别问题（情节漏洞/重复/节奏/语言/逻辑矛盾）。
2. 为每一章选择一个最适合的强度。
3. 决定是否需要参考小说其他章节作为锚点（如某章引用了 Ch12 的事件，可拉 Ch12 进上下文）；指明章节号即可，最多 4 个。

只输出 JSON，schema：
{
  "group_summary": "本组核心剧情概括（50字内）",
  "issues": ["问题1", "问题2", ...],
  "per_chapter": [
    {"chapter": <int>, "intensity": "polish|restructure|rewrite", "focus": "本章需要重点修改什么（30字内）"}
  ],
  "extra_anchors": [<int>, ...]  // 可空数组，最多 4 个章节号
}
"""

REFINE_SYSTEM_BASE = """你是一位精调小说稿件的编辑兼作家。
- 不要添加任何元注释、标题、解释。
- 直接输出修改后的完整章节正文（中文），保持 markdown 风格。
- 保留章节首行的标题（如有）。
- 保留章末状态与下一章的连贯。
"""

INTENSITY_INSTRUCTIONS = {
    "polish": "本轮精调强度=polish：仅修润错别字、重复词、口癖、不通顺句、连贯性瑕疵。保留原情节、对话顺序、段落结构。不得删减或扩写。",
    "restructure": "本轮精调强度=restructure：允许重写段落、合并/拆分场景、调整节奏与描写比例，但不得改变本章标题、关键情节点、人物决策、章末状态。",
    "rewrite": "本轮精调强度=rewrite：允许重新设计场景顺序、改写大段叙事、补充心理与环境描写。保持每章的核心目标（plan goal）和章末状态不变。",
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

    user = f"""## 任务
诊断小说第 Ch{start}-Ch{end} 这一组章节的问题，决定每章精调强度，并指出还需要哪些其他章节作为锚点。
本小说共 {last_chapter} 章，可选锚点章节号范围 1..{last_chapter}（不在本组之内的章节）。

## 世界观 Bible
{bible[:8000]}

## 主要人物
{characters[:8000]}

## 主线 Threads
{threads[:6000]}

## 待诊断章节
{chapters_text}
"""
    raw = call_llm(
        client,
        paths,
        config,
        DIAGNOSE_SYSTEM,
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

    user = f"""## 精调任务
请精调小说第 Ch{chapter_num} 章。

{intensity_instr}

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

{chr(10).join(neighbour_blocks)}

{chr(10).join(anchor_blocks)}

## 待精调的原章节 Ch{chapter_num}
{original.strip()}

## 输出要求
- 输出修改后的完整章节正文（中文）。
- 第一行保留章节标题（如原文有）。
- 不要解释，不要 JSON，不要 markdown 围栏。
"""
    refined = call_llm(
        client,
        paths,
        config,
        REFINE_SYSTEM_BASE,
        user,
        max_tokens=int(config["novel"].get("refine_chapter_max_tokens", DEFAULT_REFINE_MAX_TOKENS)),
        temperature=float(config["novel"].get("refine_temperature", 0.5)),
    )
    refined = normalize_chapter(refined)
    return refined


def _refined_text_acceptable(original: str, refined: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Sanity-check the refined output. Returns (ok, reason_if_not)."""
    if len(refined.strip()) < 500:
        return False, f"too short: {len(refined.strip())} chars"
    min_keep = float(config["novel"].get("refine_min_keep_ratio", DEFAULT_MIN_KEEP_RATIO))
    if len(refined) < len(original) * min_keep:
        return False, f"shrank below {int(min_keep * 100)}% of original ({len(refined)}/{len(original)})"
    if len(refined) > len(original) * 3:
        return False, f"grew beyond 3x of original ({len(refined)}/{len(original)})"
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
                )
            except Exception as exc:
                log(paths, f"Refine Ch{ch} failed: {exc}; keeping original")
                continue
            ok, reason = _refined_text_acceptable(original, refined, config)
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
