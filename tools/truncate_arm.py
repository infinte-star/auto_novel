"""Roll a FORKED arm back to Ch N so a second engine can rewrite the same chapters.

Why this exists: `compare.cmd_fork` forks at HEAD only, and says why — "the memory
markdown files and the entity/thread tables describe the book as of its last
written chapter, and there is no faithful way to roll them back to Ch N." That is
true in general. It is not true of **what the v2 engine reads**, and that
difference is what makes a matched-position A/B possible:

* `v2/canon.py:load` reads bible/characters/voice/voices/contract through
  `_read(path, cap)` (a HEAD slice) and then clips again inside `project_facts`
  to ~1k each. tangshuting's bible.md is 84k of which the post-Ch170 material is
  all in trailing `## Consolidated` blocks and `## ChN` sections — none of it
  survives the clip, so none of it reaches a prompt. The markdown is therefore
  left ALONE here: rewriting it would change the arm's context for reasons
  unrelated to the rollback.
* Everything v2 actually reads per chapter is DB-derived — `recent_events`,
  `get_open_threads`, `recent_metrics`, `get_overdue_reader_promises`,
  `get_active_constraints` — and every one of those is keyed on a chapter
  number, so it rolls back by DELETE.
* The RAG index is rebuilt from the truncated chapter set (zero LLM).

What it does NOT claim to roll back is printed on every run, because a rollback
that hides its residue is worse than no rollback: the next person reads a
"matched" A/B and cannot tell which way it was tilted.

    python tools/truncate_arm.py ts_v2match --to 170            # dry run
    python tools/truncate_arm.py ts_v2match --to 170 --apply

SAFETY: refuses any directory without `experiments/fork_<name>.json`. This tool
deletes chapters and DB rows; pointing it at a source book would destroy work
that took hours of API time. Being a fork is the proof that the data is
reproducible, so it is the precondition.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tables keyed on a chapter the row BELONGS to: an exact rollback is a DELETE.
# (column, table) pairs, in dependency-free order.
CHAPTER_KEYED = (
    ("events", "chapter"),
    ("chapter_metrics", "chapter"),
    ("chapter_fingerprints", "chapter"),
    ("agent_reports", "chapter"),
    ("causal_links", "source_chapter"),
    ("stage_constraints", "source_chapter"),
)

# Tables whose rows are LIVE STATE with a "when last touched" column. A row
# introduced after N never existed at N and is deleted. A row introduced at or
# before N whose last update came after N had that status set by a chapter this
# rollback is removing, so the status is reverted to open and the clock clamped.
#
# The revert is right in the case that matters: a thread `recovered` at Ch180 was
# genuinely open at Ch170, and the engine being measured is supposed to be the
# one that pays it off. It is wrong only for a thread closed at or before N and
# then merely touched again later, which the extraction schema does not normally
# produce. Counted and reported either way.
LIVE_STATE = (
    ("open_threads", "introduced_chapter", "advanced"),
    ("reader_promises", "opened_chapter", "open"),
)

# Rows this tool deliberately leaves alone, with the reason. Printed every run.
UNROLLED = {
    "entities": "a state snapshot with no introduced-column, so a row cannot be "
                "un-updated; nothing v2 reads consults it (v2 uses recent_events / "
                "open_threads / recent_metrics / overdue_promises / "
                "active_constraints) and update_structured_state overwrites it "
                "forward as the arm writes",
    "character_relationships": "same shape as entities — a snapshot keyed on "
                "(a, b) with only updated_chapter, and no v2 read touches it",
    "info_revelations": "planted_chapter would let rows be dropped, but the "
                "revealed/due columns are a snapshot and no v2 read touches it, "
                "so deleting rows would lose Ch<=N canon for no gain",
}


def _fork_meta(name: str) -> dict:
    f = ROOT / "experiments" / f"fork_{name}.json"
    if not f.exists():
        raise SystemExit(
            f"[truncate] refusing: {f.name} not found.\n"
            f"            This tool deletes chapters and DB rows. It only runs on a\n"
            f"            fork, because a fork is reproducible and a source book is not."
        )
    return json.loads(f.read_text(encoding="utf-8"))


def _chapter_files(d: Path) -> list[Path]:
    return sorted((d / "chapters").glob("*.md"))


def _truncate_book(nd: Path, keep: int) -> str:
    """book.md cut to Ch1..keep, by locating the Ch(keep+1) boundary.

    Do NOT rebuild by joining the chapter files. `writing.save_chapter` does
    `append_text(paths.book, "\\n\\n" + chapter)` with the SAME string it wrote to
    `chapters/NNNN.md`, and that string keeps its trailing newline — so the live
    book has three newlines between chapters, not two, and a `"\\n\\n".join` of
    stripped chapter files diverges at Ch2 (measured: byte 2244 of a 557k book).
    Reconstructing a text the engine never wrote would change the prose the arm
    continues from, which is the one thing this tool must not do.

    So: find where Ch(keep+1) starts, drop the `"\\n\\n"` that was prepended to it,
    and keep the bytes before that verbatim. Then verify — every kept chapter must
    still be present, in order, and the first dropped one must be gone.
    """
    live = (nd / "book.md").read_text(encoding="utf-8")
    nxt = nd / "chapters" / f"{keep + 1:04d}.md"
    if not nxt.exists():
        raise SystemExit(f"[truncate] missing {nxt}; cannot locate the cut point")
    probe = nxt.read_text(encoding="utf-8").strip()[:200]
    idx = live.find(probe)
    if idx < 0:
        raise SystemExit(
            f"[truncate] refusing: Ch{keep + 1}'s opening text does not appear in "
            f"book.md.\n            book.md was edited outside save_chapter (a "
            f"refined/fixed copy written over it?); the cut point is unknowable.")
    if live.find(probe, idx + 1) >= 0:
        raise SystemExit(f"[truncate] refusing: Ch{keep + 1}'s opening appears twice "
                         f"in book.md; the cut point is ambiguous.")
    if live[idx - 2:idx] != "\n\n":
        raise SystemExit(f"[truncate] refusing: Ch{keep + 1} is not preceded by the "
                         f"blank-line separator save_chapter writes "
                         f"({live[idx - 4:idx]!r}).")
    text = live[:idx - 2]

    cursor = 0
    for n in range(1, keep + 1):
        f = nd / "chapters" / f"{n:04d}.md"
        if not f.exists():
            raise SystemExit(f"[truncate] missing {f}; refusing to touch book.md")
        head = f.read_text(encoding="utf-8").strip()[:120]
        at = text.find(head, cursor)
        if at < 0:
            raise SystemExit(f"[truncate] refusing: Ch{n} is missing from the "
                             f"truncated book.md, or is out of order.")
        cursor = at + len(head)
    if probe in text:
        raise SystemExit(f"[truncate] refusing: Ch{keep + 1} still present after cut.")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="the FORKED novel to roll back")
    ap.add_argument("--to", type=int, required=True, metavar="N",
                    help="keep Ch1..N; the arm will next write Ch N+1")
    ap.add_argument("--max-chapters", type=int, default=0,
                    help="set novel.max_chapters (default: leave as the fork set it)")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    args = ap.parse_args()

    nd = ROOT / "novels" / args.name
    if not nd.is_dir():
        raise SystemExit(f"[truncate] no such novel: {nd}")
    meta = _fork_meta(args.name)
    keep = args.to
    chapters = _chapter_files(nd)
    head = len(chapters)
    if keep >= head:
        raise SystemExit(f"[truncate] {args.name} has {head} chapters; --to {keep} "
                         f"would remove nothing")
    if keep < 1:
        raise SystemExit("[truncate] --to must be >= 1")

    print(f"[truncate] {args.name}  (fork of {meta.get('source')} @ Ch"
          f"{meta.get('fork_at_chapter')})")
    print(f"[truncate] Ch1..{head} -> Ch1..{keep}   (drops {head - keep} chapters; "
          f"next write is Ch{keep + 1})")

    # ---- book.md ---------------------------------------------------------
    book_text = _truncate_book(nd, keep)
    live_len = len((nd / "book.md").read_text(encoding="utf-8"))
    print(f"[truncate] book.md {live_len} -> {len(book_text)} chars")

    # ---- database --------------------------------------------------------
    db = nd / "story_state.db"
    plan: list[tuple[str, str, int]] = []
    live_plan: list[tuple[str, int, int]] = []
    if db.exists():
        conn = sqlite3.connect(db)
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table, col in CHAPTER_KEYED:
            if table not in have:
                continue
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} > ?",
                             (keep,)).fetchone()[0]
            plan.append((table, col, n))
        for table, intro, reopen in LIVE_STATE:
            if table not in have:
                continue
            gone = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {intro} > ?",
                                (keep,)).fetchone()[0]
            revert = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {intro} <= ? AND updated_chapter > ?",
                (keep, keep)).fetchone()[0]
            live_plan.append((table, gone, revert))
        conn.close()
    for table, col, n in plan:
        print(f"[truncate]   DELETE {table:<22} {col} > {keep}: {n} rows")
    for table, gone, revert in live_plan:
        print(f"[truncate]   {table:<22} delete {gone} introduced-after, "
              f"reopen {revert} closed-after")

    # ---- what is NOT rolled back ----------------------------------------
    print("[truncate] NOT rolled back (stated, not hidden):")
    for table, why in UNROLLED.items():
        print(f"[truncate]   - {table}: {why}")
    print("[truncate]   - memory/*.md: left verbatim. v2 reads only a ~1k HEAD clip "
          "of bible/characters (canon._read + project_facts), and every post-Ch"
          f"{keep} block in these files sits in the trailing "
          "`## Consolidated` / `## ChN` region, which the clip never reaches.")
    print("[truncate]   - state.md: v2 only tests that it exists and is non-empty "
          "(v2/run.py bootstrap check); its content reaches no prompt and "
          "update_state_file overwrites it on the first chapter written.")
    print("[truncate]   - voice.md / voices.md: derived from prose, not chapter-keyed. "
          "Any drift they carry is stylistic and affects the arm being measured "
          "conservatively (it anchors v2 to v1's late voice, not away from it).")

    if not args.apply:
        print("\n[truncate] DRY RUN. Re-run with --apply to write.")
        return 0

    # ---- apply -----------------------------------------------------------
    removed = 0
    for f in chapters[keep:]:
        f.unlink()
        removed += 1
    (nd / "book.md").write_text(book_text, encoding="utf-8")
    print(f"[truncate] removed {removed} chapter files, rewrote book.md")

    if db.exists():
        conn = sqlite3.connect(db)
        for table, col, _ in plan:
            conn.execute(f"DELETE FROM {table} WHERE {col} > ?", (keep,))
        # A link/constraint that points forward past the cut no longer has a
        # target: null it rather than deleting the row, whose source is still real.
        if any(t == "causal_links" for t, _, _ in plan):
            conn.execute("UPDATE causal_links SET target_chapter = NULL "
                         "WHERE target_chapter > ?", (keep,))
        if any(t == "stage_constraints" for t, _, _ in plan):
            conn.execute("UPDATE stage_constraints SET resolved_chapter = NULL "
                         "WHERE resolved_chapter > ?", (keep,))
        for (table, intro, reopen), (_, _, _) in zip(LIVE_STATE, live_plan):
            conn.execute(f"DELETE FROM {table} WHERE {intro} > ?", (keep,))
            conn.execute(
                f"UPDATE {table} SET status = ?, updated_chapter = ? "
                f"WHERE {intro} <= ? AND updated_chapter > ?",
                (reopen, keep, keep, keep))
        conn.commit()
        conn.close()
        print("[truncate] database rolled back")

    # ---- RAG index: rebuild, do not patch -------------------------------
    # The index is sharded (retrieval_index/ch*.json + _df.json) and the document
    # frequencies are global, so dropping shards leaves _df.json describing
    # chapters that are gone. Rebuilding is deterministic and zero-LLM.
    for p in (nd / "logs" / "retrieval_index", nd / "logs" / "retrieval_index.json"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    os.environ["NOVEL_CONFIG"] = str((nd / "config.yaml").relative_to(ROOT))
    os.environ.setdefault("NOVEL_PROMPT", str((nd / "prompt.md").relative_to(ROOT)))
    import config as _config
    import retrieval
    cfg = _config.load_config()
    paths = _config.get_paths(cfg)
    n_idx = retrieval.backfill_index(paths, cfg)
    print(f"[truncate] RAG index rebuilt from scratch: {n_idx} chapters")

    if args.max_chapters:
        from compare import _get_config_key, _set_config_key
        cfg_path = nd / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        old = _get_config_key(text, "max_chapters")
        cfg_path.write_text(_set_config_key(text, "max_chapters", args.max_chapters),
                            encoding="utf-8")
        print(f"[truncate] max_chapters: {old} -> {args.max_chapters}")

    from config import find_last_chapter
    print(f"[truncate] find_last_chapter now reports Ch{find_last_chapter(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
