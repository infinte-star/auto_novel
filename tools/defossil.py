"""Fossil phrase replacement + descriptor thinning tool.

Scans chapter files for overused phrases (fossil clauses, short descriptors)
and replaces/deletes them via LLM rewrite, preserving the sentence's meaning
and tone while eliminating mechanical repetition.

Usage:
    python tools/defossil.py --novel tangshuting --chapters 1-76 --dry-run
    python tools/defossil.py --novel tangshuting --chapters 1-76 --apply
    python tools/defossil.py --novel tangshuting --chapters 1-76 --apply --target 虎口旧疤 --keep 12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _setup_env(novel_name: str):
    """Set env vars BEFORE importing config/llm (they read at import time)."""
    novel_dir = ROOT / "novels" / novel_name
    os.environ["NOVEL_CONFIG"] = str(novel_dir / "config.yaml")
    os.environ["NOVEL_PROMPT"] = str(novel_dir / "prompt.md")


DEFAULT_FOSSILS: list[dict[str, Any]] = [
    {
        "pattern": "牙缝里往外",
        "keep": 2,
        "context_radius": 300,
        "action": "rewrite",
    },
    {
        "pattern": "手术刀划开纱布",
        "keep": 2,
        "context_radius": 300,
        "action": "rewrite",
    },
    {
        "pattern": "虎口旧疤",
        "keep": 12,
        "context_radius": 200,
        "action": "rewrite",
    },
]

REWRITE_SYSTEM = (
    "你是一位资深中文小说编辑。你的任务是消除文中反复出现的机械化描写标签，"
    "同时保持原文的语气、信息量和叙事节奏。"
)


def _rewrite_prompt(phrase: str, context: str, sentence: str) -> str:
    return (
        f"下面是一段小说上下文，其中包含一个需要消除的重复描写标签：「{phrase}」。\n\n"
        f"## 上下文（±200-300字）\n{context}\n\n"
        f"## 需要改写的句子\n{sentence}\n\n"
        "## 要求\n"
        f"1. 改写上面这个句子，去除「{phrase}」这个表达\n"
        "2. 保留原句的语气、情感和信息量\n"
        "3. 字数与原句相近（±20%）\n"
        "4. 不要使用与原标签近义的机械替换（如把一个套路换成另一个套路）\n"
        "5. 根据上下文的具体场景，用贴合当下情境的细节来替代\n"
        "6. 只输出改写后的句子，不要输出任何解释\n"
    )


def _delete_prompt(phrase: str, context: str, sentence: str) -> str:
    return (
        f"下面是一段小说上下文，其中包含一个需要删除的重复描写：「{phrase}」。\n\n"
        f"## 上下文（±200字）\n{context}\n\n"
        f"## 包含目标描写的句子\n{sentence}\n\n"
        "## 要求\n"
        f"1. 从这个句子中删除包含「{phrase}」的从句或分句\n"
        "2. 保持句子其余部分的语法完整和通顺\n"
        "3. 如果整句只围绕这个描写，可以返回空字符串\n"
        "4. 只输出修改后的句子（或空字符串），不要输出任何解释\n"
    )


def find_sentence_boundary(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Find the sentence containing the match, bounded by CJK punctuation."""
    terminators = set("。！？\n")
    sent_start = match_start
    while sent_start > 0 and text[sent_start - 1] not in terminators:
        sent_start -= 1
    sent_end = match_end
    while sent_end < len(text) and text[sent_end] not in terminators:
        sent_end += 1
    if sent_end < len(text):
        sent_end += 1
    return sent_start, sent_end


def scan_chapters(
    chapters_dir: Path,
    chapter_range: tuple[int, int],
    fossils: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan chapter files and return all occurrences of target phrases."""
    results: list[dict[str, Any]] = []

    for ch_num in range(chapter_range[0], chapter_range[1] + 1):
        ch_file = chapters_dir / f"{ch_num:04d}.md"
        if not ch_file.exists():
            continue
        text = ch_file.read_text(encoding="utf-8")

        for fossil in fossils:
            pattern = fossil["pattern"]
            for m in re.finditer(re.escape(pattern), text):
                sent_start, sent_end = find_sentence_boundary(text, m.start(), m.end())
                sentence = text[sent_start:sent_end].strip()
                ctx_start = max(0, sent_start - fossil["context_radius"])
                ctx_end = min(len(text), sent_end + fossil["context_radius"])
                context = text[ctx_start:ctx_end]

                results.append({
                    "chapter": ch_num,
                    "phrase": pattern,
                    "match_start": m.start(),
                    "match_end": m.end(),
                    "sentence": sentence,
                    "context": context,
                    "sent_start": sent_start,
                    "sent_end": sent_end,
                    "action": fossil["action"],
                    "keep": fossil["keep"],
                })

    return results


def classify_occurrences(
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mark each occurrence as 'keep', 'rewrite', or 'delete'."""
    by_phrase: dict[str, list[dict[str, Any]]] = {}
    for occ in occurrences:
        by_phrase.setdefault(occ["phrase"], []).append(occ)

    for phrase, occs in by_phrase.items():
        occs.sort(key=lambda x: x["chapter"])
        keep_count = occs[0]["keep"]

        if len(occs) <= keep_count:
            for o in occs:
                o["decision"] = "keep"
            continue

        for o in occs[:keep_count]:
            o["decision"] = "keep"

        remaining = occs[keep_count:]
        for i, o in enumerate(remaining):
            if i % 3 == 2:
                o["decision"] = "delete"
            else:
                o["decision"] = "rewrite"

    return occurrences


def apply_fixes(
    occurrences: list[dict[str, Any]],
    chapters_dir: Path,
    backup_dir: Path,
    client: Any,
    paths: Any,
    config: dict[str, Any],
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """Apply rewrites/deletes to chapter files. Returns the log of changes."""
    if not dry_run:
        from llm import call_llm
    by_chapter: dict[int, list[dict[str, Any]]] = {}
    for occ in occurrences:
        if occ.get("decision") == "keep":
            continue
        by_chapter.setdefault(occ["chapter"], []).append(occ)

    change_log: list[dict[str, Any]] = []

    for ch_num in sorted(by_chapter):
        ch_file = chapters_dir / f"{ch_num:04d}.md"
        text = ch_file.read_text(encoding="utf-8")
        original_text = text

        fixes = sorted(by_chapter[ch_num], key=lambda x: -x["sent_start"])

        for occ in fixes:
            sent_start = occ["sent_start"]
            sent_end = occ["sent_end"]
            old_sentence = text[sent_start:sent_end].strip()

            if old_sentence != occ["sentence"]:
                print(f"  [WARN] Ch{ch_num}: sentence shifted, re-locating...")
                idx = text.find(occ["sentence"])
                if idx == -1:
                    print(f"  [SKIP] Ch{ch_num}: cannot find original sentence")
                    occ["result"] = "skipped_not_found"
                    continue
                sent_start = idx
                sent_end = idx + len(occ["sentence"])
                old_sentence = occ["sentence"]

            if dry_run:
                action_label = occ["decision"].upper()
                print(f"  [{action_label}] Ch{ch_num} '{occ['phrase']}': {old_sentence[:60]}...")
                occ["result"] = f"dry_run_{occ['decision']}"
                change_log.append({
                    "chapter": ch_num,
                    "phrase": occ["phrase"],
                    "decision": occ["decision"],
                    "old": old_sentence,
                    "new": None,
                })
                continue

            if occ["decision"] == "rewrite":
                prompt = _rewrite_prompt(occ["phrase"], occ["context"], old_sentence)
            else:
                prompt = _delete_prompt(occ["phrase"], occ["context"], old_sentence)

            try:
                new_sentence = call_llm(
                    client, paths, config,
                    system=REWRITE_SYSTEM,
                    user=prompt,
                    max_tokens=1000,
                    temperature=0.4,
                    tag="defossil",
                )
                new_sentence = new_sentence.strip().strip('"').strip("「」")
            except Exception as e:
                print(f"  [ERROR] Ch{ch_num} LLM call failed: {e}")
                occ["result"] = "error"
                continue

            if occ["phrase"] in new_sentence:
                print(f"  [WARN] Ch{ch_num}: LLM output still contains '{occ['phrase']}', retrying...")
                try:
                    new_sentence = call_llm(
                        client, paths, config,
                        system=REWRITE_SYSTEM,
                        user=prompt + f"\n\n注意：输出中绝对不能包含「{occ['phrase']}」这几个字。",
                        max_tokens=1000,
                        temperature=0.6,
                        tag="defossil",
                    )
                    new_sentence = new_sentence.strip().strip('"').strip("「」")
                except Exception:
                    pass

            if occ["phrase"] in new_sentence:
                print(f"  [SKIP] Ch{ch_num}: still contains fossil after retry")
                occ["result"] = "skipped_fossil_remains"
                continue

            ws_before = text[sent_start - 1] if sent_start > 0 else ""
            text = text[:sent_start] + new_sentence + text[sent_end:]

            print(f"  [DONE] Ch{ch_num} '{occ['phrase']}' {occ['decision']}")
            occ["result"] = "applied"
            change_log.append({
                "chapter": ch_num,
                "phrase": occ["phrase"],
                "decision": occ["decision"],
                "old": old_sentence,
                "new": new_sentence,
            })

        if not dry_run and text != original_text:
            if not backup_dir.exists():
                backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / f"{ch_num:04d}.md"
            if not backup_file.exists():
                shutil.copy2(ch_file, backup_file)
            ch_file.write_text(text, encoding="utf-8")
            print(f"  [SAVED] Ch{ch_num}")

    return change_log


def rebuild_book(chapters_dir: Path, book_file: Path, chapter_range: tuple[int, int] | None = None):
    """Rebuild book.md from individual chapter files."""
    parts: list[str] = []
    for ch_file in sorted(chapters_dir.glob("*.md")):
        match = re.match(r"(\d+)\.md$", ch_file.name)
        if not match:
            continue
        ch_num = int(match.group(1))
        if chapter_range and (ch_num < chapter_range[0] or ch_num > chapter_range[1]):
            continue
        text = ch_file.read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)

    book_file.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"[BOOK] Rebuilt {book_file} from {len(parts)} chapters")


def main():
    parser = argparse.ArgumentParser(description="Fossil phrase replacement tool")
    parser.add_argument("--novel", required=True, help="Novel name under novels/")
    parser.add_argument("--chapters", default="1-200", help="Chapter range, e.g. 1-76")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying files")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument("--target", help="Only process a specific phrase")
    parser.add_argument("--keep", type=int, help="Override keep count for --target")
    parser.add_argument("--rebuild-book", action="store_true", help="Rebuild book.md after fixes")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and report, no changes")
    args = parser.parse_args()

    if not args.dry_run and not args.apply and not args.scan_only:
        print("Error: specify --dry-run, --apply, or --scan-only")
        sys.exit(1)

    ch_start, ch_end = map(int, args.chapters.split("-"))

    novel_dir = ROOT / "novels" / args.novel
    if not novel_dir.exists():
        print(f"Error: novel directory not found: {novel_dir}")
        sys.exit(1)

    chapters_dir = novel_dir / "chapters"
    backup_dir = novel_dir / "chapters_backup"
    config_path = novel_dir / "config.yaml"

    if args.target:
        fossils = [{
            "pattern": args.target,
            "keep": args.keep or 2,
            "context_radius": 250,
            "action": "rewrite",
        }]
    else:
        fossils = DEFAULT_FOSSILS

    print(f"=== Defossil: {args.novel} Ch{ch_start}-{ch_end} ===")
    print(f"Targets: {', '.join(f['pattern'] for f in fossils)}")

    occurrences = scan_chapters(chapters_dir, (ch_start, ch_end), fossils)
    print(f"\nFound {len(occurrences)} total occurrences:")
    by_phrase: dict[str, int] = {}
    for occ in occurrences:
        by_phrase[occ["phrase"]] = by_phrase.get(occ["phrase"], 0) + 1
    for phrase, count in sorted(by_phrase.items()):
        fossil = next(f for f in fossils if f["pattern"] == phrase)
        print(f"  {phrase}: {count} occurrences (keep {fossil['keep']})")

    if args.scan_only:
        print("\n--- Occurrence details ---")
        for occ in occurrences:
            print(f"  Ch{occ['chapter']:03d} | {occ['phrase']} | {occ['sentence'][:80]}...")
        return

    classify_occurrences(occurrences)

    keep_count = sum(1 for o in occurrences if o.get("decision") == "keep")
    rewrite_count = sum(1 for o in occurrences if o.get("decision") == "rewrite")
    delete_count = sum(1 for o in occurrences if o.get("decision") == "delete")
    print(f"\nDecisions: keep={keep_count}, rewrite={rewrite_count}, delete={delete_count}")

    if args.dry_run:
        print("\n--- Dry run preview ---")
        apply_fixes(occurrences, chapters_dir, backup_dir,
                    None, None, None, dry_run=True)
        return

    print("\nLoading config and LLM client...")
    _setup_env(args.novel)
    sys.path.insert(0, str(ROOT))
    from config import load_config, get_paths, configured_api_endpoints
    from llm import call_llm as _call_llm, LLMClientPool
    from openai import OpenAI

    config = load_config()
    paths = get_paths(config)

    endpoints, primary_count = configured_api_endpoints(config)
    default_headers = {}
    user_agent = str(config["api"].get("user_agent", "")).strip()
    if user_agent:
        default_headers["User-Agent"] = user_agent
    clients = [
        OpenAI(base_url=base_url, api_key=api_key,
               default_headers=default_headers or None)
        for base_url, api_key in endpoints
    ]
    client = (
        LLMClientPool(clients, primary_count, endpoints=endpoints)
        if len(clients) > 1 else clients[0]
    )

    print(f"API endpoints: {len(endpoints)} ({primary_count} primary)")
    print(f"\n--- Applying fixes ---")
    change_log = apply_fixes(
        occurrences, chapters_dir, backup_dir,
        client, paths, config, dry_run=False,
    )

    log_file = novel_dir / "logs" / "defossil_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        json.dumps(change_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nChange log saved to {log_file}")
    print(f"Applied: {sum(1 for c in change_log if c.get('new'))}, "
          f"Total entries: {len(change_log)}")

    if args.rebuild_book:
        book_file = novel_dir / "book.md"
        rebuild_book(chapters_dir, book_file)


if __name__ == "__main__":
    main()
