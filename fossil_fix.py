"""Deterministic fossil replacement for finished novels.

Scans chapter files for CJK n-gram phrases that recur across too many chapters
(the "fossil" problem — e.g. "虎口旧疤" appearing 420 times in 106/200 chapters),
then rewrites chapters with rotated synonym variants.  Zero LLM calls.

Entry point: cmd_fix_fossils (wired from novel.py) or fix_book for programmatic use.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default replacement bank
# ---------------------------------------------------------------------------

FOSSIL_REPLACEMENTS: dict[str, list[str]] = {
    "虎口旧疤": [
        "虎口那道白印", "虎口愈合的旧伤", "虎口上结痂的伤痕",
        "手背蜿蜒的疤", "掌根那条淡粉色的线", "虎口磨出的茧旁那道浅沟",
    ],
    "声音压得很低": [
        "压着嗓子", "用气声说", "几乎贴着耳朵说",
        "声量放到只有两人能听到的程度", "咬着字根说", "把声线收到嗓子底部",
    ],
    "每个字都像": [
        "字字", "一字一顿", "语气像", "话音像", "声调仿佛",
    ],
    "声音沙哑": [
        "嗓子发涩", "喉咙像含了砂", "声线带着毛边", "嗓音粗粝", "喉头发紧",
    ],
    "喉结滚动": [
        "喉头动了一下", "吞咽了一下", "脖颈肌肉绷了绷", "下颌收紧",
    ],
    "手指收紧": [
        "五指攥紧", "指节发白", "手掌握成拳", "指尖陷进掌心",
    ],
    "深吸一口气": [
        "胸腔起伏了一下", "鼻腔灌进冷气", "肺里灌满空气", "缓缓吐出一口浊气",
    ],
}


# ---------------------------------------------------------------------------
# CJK helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"
    r"\U00020000-\U0002a6df\U0002a700-\U0002ebef]"
)


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _cjk_runs(text: str) -> list[str]:
    """Extract contiguous CJK runs from text (ignoring punctuation boundaries)."""
    runs: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            buf.append(ch)
        else:
            if buf:
                runs.append("".join(buf))
                buf = []
    if buf:
        runs.append("".join(buf))
    return runs


def _ngrams_from_run(run: str, n: int) -> list[str]:
    return [run[i:i + n] for i in range(len(run) - n + 1)]


# ---------------------------------------------------------------------------
# scan_fossils
# ---------------------------------------------------------------------------

def _read_chapters(chapters_dir: Path) -> list[tuple[int, str]]:
    """Return sorted (chapter_num, text) pairs from a chapters directory."""
    results: list[tuple[int, str]] = []
    if not chapters_dir.exists():
        return results
    for p in sorted(chapters_dir.glob("*.md")):
        stem = p.stem
        if stem.isdigit():
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            results.append((int(stem), text))
    return results


def scan_fossils(
    chapters_dir: Path,
    ngram: int = 6,
    min_frac: float = 0.15,
) -> list[dict[str, Any]]:
    """Detect CJK n-gram phrases that appear across >= min_frac of chapters."""
    chapters = _read_chapters(chapters_dir)
    if not chapters:
        return []
    total_chapters = len(chapters)
    min_chapters = max(2, int(total_chapters * min_frac))

    # phrase -> set of chapter numbers where it appears
    phrase_chapters: dict[str, set[int]] = defaultdict(set)
    # phrase -> total occurrence count across all chapters
    phrase_total: dict[str, int] = Counter()

    for ch_num, text in chapters:
        seen_in_chapter: set[str] = set()
        for run in _cjk_runs(text):
            for ng in _ngrams_from_run(run, ngram):
                phrase_total[ng] += 1
                if ng not in seen_in_chapter:
                    seen_in_chapter.add(ng)
                    phrase_chapters[ng].add(ch_num)

    # Filter to those meeting the chapter-spread threshold
    candidates: list[dict[str, Any]] = []
    for phrase, ch_set in phrase_chapters.items():
        if len(ch_set) >= min_chapters:
            candidates.append({
                "phrase": phrase,
                "chapter_count": len(ch_set),
                "frac": len(ch_set) / total_chapters,
                "total_occurrences": phrase_total[phrase],
                "chapters": sorted(ch_set),
            })

    # Sort by total_occurrences desc for stable ordering
    candidates.sort(key=lambda d: (-d["total_occurrences"], d["phrase"]))

    # Filter overlapping n-grams: if phrase A is a substring of phrase B
    # and B has higher total_occurrences, drop A.  Also deduplicate near-
    # identical n-grams that share n-1 chars (keep highest count).
    kept: list[dict[str, Any]] = []
    dropped_phrases: set[str] = set()
    for c in candidates:
        p = c["phrase"]
        if p in dropped_phrases:
            continue
        # Mark any lower-count phrase that is a substring of p or vice versa
        for other in candidates:
            op = other["phrase"]
            if op == p or op in dropped_phrases:
                continue
            if op in p or p in op:
                # Keep the one with higher occurrence count
                if other["total_occurrences"] <= c["total_occurrences"]:
                    dropped_phrases.add(op)
                else:
                    dropped_phrases.add(p)
                    break
        if p not in dropped_phrases:
            kept.append(c)

    return kept


# ---------------------------------------------------------------------------
# fix_chapter
# ---------------------------------------------------------------------------

def fix_chapter(
    text: str,
    fossils: list[dict[str, Any]],
    replacements: dict[str, list[str]],
    max_per_chapter: int = 1,
    chapter_num: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Replace fossil occurrences in a single chapter's text.

    Returns (fixed_text, per-fossil stats).
    """
    stats: dict[str, Any] = {}
    result = text

    for fossil_info in fossils:
        phrase = fossil_info["phrase"]
        alts = replacements.get(phrase)
        if not alts:
            continue

        count = result.count(phrase)
        if count == 0:
            continue

        kept = min(count, max_per_chapter)
        to_replace = count - kept

        if to_replace <= 0:
            stats[phrase] = {"original_count": count, "kept": count, "replaced": 0}
            continue

        # Build replacement list by rotating through alts seeded on chapter_num
        # so different chapters get different variants
        replace_list: list[str] = []
        base_idx = chapter_num * 7  # spread offset across chapters
        for i in range(to_replace):
            replace_list.append(alts[(base_idx + i) % len(alts)])

        # Replace occurrences: keep the first `kept`, replace the rest
        pieces: list[str] = []
        remaining = result
        occurrence = 0
        replace_idx = 0

        while phrase in remaining:
            pos = remaining.index(phrase)
            pieces.append(remaining[:pos])
            occurrence += 1
            if occurrence <= kept:
                pieces.append(phrase)
            else:
                pieces.append(replace_list[replace_idx])
                replace_idx += 1
            remaining = remaining[pos + len(phrase):]
        pieces.append(remaining)
        result = "".join(pieces)

        stats[phrase] = {
            "original_count": count,
            "kept": kept,
            "replaced": to_replace,
        }

    return result, stats


# ---------------------------------------------------------------------------
# fix_book
# ---------------------------------------------------------------------------

def fix_book(
    chapters_dir: Path,
    output_dir: Path,
    replacements: dict[str, list[str]] | None = None,
    custom_fossils: list[dict[str, Any]] | None = None,
    max_per_chapter: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan for fossils and rewrite chapters with rotated replacements.

    Returns a summary dict with per-chapter and per-fossil statistics.
    """
    if replacements is None:
        replacements = dict(FOSSIL_REPLACEMENTS)

    chapters = _read_chapters(chapters_dir)
    if not chapters:
        print("[fossil_fix] no chapter files found in", chapters_dir)
        return {"chapters": 0, "fossils": [], "per_chapter": {}, "per_fossil": {}}

    print(f"[fossil_fix] loaded {len(chapters)} chapters from {chapters_dir}")

    # Detect or use provided fossils
    if custom_fossils is not None:
        fossils = custom_fossils
    else:
        print("[fossil_fix] scanning for fossils ...")
        fossils = scan_fossils(chapters_dir)

    # Only process fossils that have replacements
    actionable = [f for f in fossils if f["phrase"] in replacements]
    # Also include any replacements-only entries not in scan results
    scanned_phrases = {f["phrase"] for f in fossils}
    for phrase in replacements:
        if phrase not in scanned_phrases:
            # Check if this phrase actually appears anywhere
            total = sum(1 for _, text in chapters if phrase in text)
            if total > 0:
                ch_list = [n for n, text in chapters if phrase in text]
                actionable.append({
                    "phrase": phrase,
                    "chapter_count": len(ch_list),
                    "frac": len(ch_list) / len(chapters),
                    "total_occurrences": sum(
                        text.count(phrase) for _, text in chapters
                    ),
                    "chapters": ch_list,
                })

    if not actionable:
        print("[fossil_fix] no actionable fossils found")
        return {"chapters": len(chapters), "fossils": [], "per_chapter": {}, "per_fossil": {}}

    print(f"[fossil_fix] {len(actionable)} actionable fossils:")
    for f in actionable[:20]:
        print(f"  {f['phrase']}  chapters={f['chapter_count']}  total={f['total_occurrences']}")
    if len(actionable) > 20:
        print(f"  ... and {len(actionable) - 20} more")

    # Process chapters
    per_chapter: dict[int, dict[str, Any]] = {}
    per_fossil: dict[str, dict[str, int]] = defaultdict(lambda: {"replaced": 0, "kept": 0})
    book_parts: list[str] = []

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for ch_num, text in chapters:
        fixed, ch_stats = fix_chapter(
            text, actionable, replacements,
            max_per_chapter=max_per_chapter,
            chapter_num=ch_num,
        )

        if ch_stats:
            per_chapter[ch_num] = ch_stats
            for phrase, s in ch_stats.items():
                per_fossil[phrase]["replaced"] += s["replaced"]
                per_fossil[phrase]["kept"] += s["kept"]

        if not dry_run:
            out_path = output_dir / f"{ch_num:04d}.md"
            out_path.write_text(fixed, encoding="utf-8")
            book_parts.append(fixed)
        else:
            book_parts.append(fixed)

    # Write concatenated book
    if not dry_run:
        book_path = output_dir.parent / "book_fixed.md"
        book_path.write_text("\n\n".join(book_parts), encoding="utf-8")
        print(f"[fossil_fix] wrote {len(chapters)} chapters to {output_dir}")
        print(f"[fossil_fix] concatenated book: {book_path}")

    # Summary
    total_replaced = sum(v["replaced"] for v in per_fossil.values())
    total_kept = sum(v["kept"] for v in per_fossil.values())

    summary = {
        "chapters": len(chapters),
        "fossils": [f["phrase"] for f in actionable],
        "per_chapter": {k: v for k, v in per_chapter.items()},
        "per_fossil": dict(per_fossil),
        "total_replaced": total_replaced,
        "total_kept": total_kept,
        "dry_run": dry_run,
    }

    print(f"[fossil_fix] {'(dry run) ' if dry_run else ''}total replaced: {total_replaced}  kept: {total_kept}")
    for phrase in sorted(per_fossil, key=lambda p: -per_fossil[p]["replaced"]):
        pf = per_fossil[phrase]
        print(f"  {phrase}  replaced={pf['replaced']}  kept={pf['kept']}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cmd_fix_fossils(
    name: str,
    dry_run: bool = False,
    max_keep: int = 1,
    custom_replacements_path: str | None = None,
) -> int:
    """CLI entry point called from novel.py."""
    from pathlib import Path as _P

    # Determine novel directory — import kept local to avoid config.py import-time env dependency
    project_dir = Path(__file__).resolve().parent
    novel_dir = project_dir / "novels" / name

    if not novel_dir.exists():
        print(f"[fossil_fix] ERROR: novel directory not found: {novel_dir}")
        return 2

    # Prefer chapters_refined/ if it exists; fall back to chapters/
    chapters_refined = novel_dir / "chapters_refined"
    chapters_dir = novel_dir / "chapters"
    if chapters_refined.exists() and any(chapters_refined.glob("*.md")):
        source_dir = chapters_refined
        print(f"[fossil_fix] using refined chapters from {chapters_refined}")
    elif chapters_dir.exists():
        source_dir = chapters_dir
        print(f"[fossil_fix] using chapters from {chapters_dir}")
    else:
        print(f"[fossil_fix] ERROR: no chapters directory found in {novel_dir}")
        return 2

    output_dir = novel_dir / "chapters_fixed"

    # Load custom replacements if provided
    replacements: dict[str, list[str]] = dict(FOSSIL_REPLACEMENTS)
    if custom_replacements_path:
        rp = Path(custom_replacements_path)
        if not rp.exists():
            print(f"[fossil_fix] ERROR: custom replacements file not found: {rp}")
            return 2
        try:
            with rp.open(encoding="utf-8") as f:
                custom = json.load(f)
            if not isinstance(custom, dict):
                print("[fossil_fix] ERROR: custom replacements must be a JSON object {phrase: [alternatives]}")
                return 2
            replacements.update(custom)
            print(f"[fossil_fix] loaded {len(custom)} custom replacement entries from {rp}")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[fossil_fix] ERROR: failed to read custom replacements: {exc}")
            return 2

    summary = fix_book(
        chapters_dir=source_dir,
        output_dir=output_dir,
        replacements=replacements,
        max_per_chapter=max_keep,
        dry_run=dry_run,
    )

    if not dry_run and summary.get("total_replaced", 0) > 0:
        print(f"[fossil_fix] done. Fixed chapters: novels/{name}/chapters_fixed/")
        print(f"[fossil_fix] concatenated: novels/{name}/book_fixed.md")
    elif dry_run:
        print(f"[fossil_fix] dry run complete. No files written.")
    else:
        print(f"[fossil_fix] no replacements needed.")

    return 0
