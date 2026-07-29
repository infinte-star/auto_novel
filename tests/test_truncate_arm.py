"""The rollback that makes a matched-position A/B possible, and its refusals.

`tools/truncate_arm.py` deletes chapters and DB rows. Every test here is about a
case where it must REFUSE rather than proceed, plus the one arithmetic it has to
get exactly right: where Ch(keep+1) begins in book.md.

That arithmetic already failed once. The first version rebuilt the book by
joining stripped chapter files on a blank line, which is what `save_chapter`
looks like it does — but it appends `"\\n\\n" + chapter` with the chapter's own
trailing newline intact, so the live book has THREE newlines between chapters and
the rebuild diverged at byte 2244 of a 557k book. The guard caught it; these
tests keep the guard honest.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_truncate_arm", ROOT / "tools" / "truncate_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _chapter(n: int) -> str:
    """A chapter as `save_chapter` writes it: body plus a trailing newline."""
    return f"第{n}章 标题{n}\n\n正文内容{n}。" + ("填充。" * 40) + "\n"


def _make_arm(root: Path, name: str, chapters: int, *, fork_meta: bool = True) -> Path:
    """A novel directory built the way the engine builds one."""
    nd = root / "novels" / name
    (nd / "chapters").mkdir(parents=True)
    book = nd / "book.md"
    for n in range(1, chapters + 1):
        text = _chapter(n)
        (nd / "chapters" / f"{n:04d}.md").write_text(text, encoding="utf-8")
        # Exactly writing.save_chapter: same string to both, "\n\n" prepended.
        with book.open("a", encoding="utf-8") as f:
            f.write(("\n\n" + text) if n > 1 else text)
    if fork_meta:
        (root / "experiments").mkdir(exist_ok=True)
        (root / "experiments" / f"fork_{name}.json").write_text(
            json.dumps({"source": "src", "fork_at_chapter": chapters}),
            encoding="utf-8")
    return nd


class CutPointTest(unittest.TestCase):
    """`_truncate_book` must land on the byte the engine would have left."""

    def test_the_cut_is_the_bytes_before_the_appended_separator(self):
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 5)
            live = (nd / "book.md").read_text(encoding="utf-8")
            got = MOD._truncate_book(nd, 3)
            # What the file held right after Ch3 was appended: Ch1..Ch3 with the
            # engine's own separators and Ch3's trailing newline, nothing added.
            want = _chapter(1) + "\n\n" + _chapter(2) + "\n\n" + _chapter(3)
            self.assertEqual(got, want)
            self.assertTrue(live.startswith(got), "the cut must be a byte-prefix")
            self.assertEqual(live[len(got):len(got) + 2], "\n\n")

    def test_a_joined_rebuild_would_have_been_wrong(self):
        # The actual bug, pinned: the plausible reconstruction is not the truth.
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 4)
            joined = "\n\n".join(
                (nd / "chapters" / f"{n:04d}.md").read_text(encoding="utf-8").strip()
                for n in range(1, 4))
            self.assertNotEqual(joined, MOD._truncate_book(nd, 3))

    def test_keeping_one_chapter_works(self):
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 3)
            self.assertEqual(MOD._truncate_book(nd, 1), _chapter(1))


class RefusalTest(unittest.TestCase):

    def test_refuses_a_directory_with_no_fork_metadata(self):
        # A source book is hours of API time and is not reproducible; being a
        # fork is the proof that the data can be regenerated.
        with TemporaryDirectory() as td:
            MOD.ROOT = Path(td)
            _make_arm(Path(td), "notafork", 3, fork_meta=False)
            with self.assertRaises(SystemExit) as cm:
                MOD._fork_meta("notafork")
            self.assertIn("fork", str(cm.exception))

    def test_refuses_when_the_next_chapter_is_absent_from_the_book(self):
        # book.md overwritten by a refined/fixed copy: the cut point is unknowable.
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 4)
            (nd / "book.md").write_text("完全不同的正文。" * 50, encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                MOD._truncate_book(nd, 2)
            self.assertIn("does not appear in book.md", str(cm.exception))

    def test_refuses_an_ambiguous_cut_point(self):
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 4)
            book = nd / "book.md"
            book.write_text(book.read_text(encoding="utf-8")
                            + "\n\n" + _chapter(3), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                MOD._truncate_book(nd, 2)
            self.assertIn("twice", str(cm.exception))

    def test_refuses_when_the_separator_is_not_what_save_chapter_writes(self):
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 3)
            book = nd / "book.md"
            book.write_text(book.read_text(encoding="utf-8").replace(
                "\n\n" + _chapter(3), "\n===\n" + _chapter(3)), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                MOD._truncate_book(nd, 2)
            self.assertIn("separator", str(cm.exception))

    def test_refuses_when_a_kept_chapter_is_missing_from_the_book(self):
        # Ch2 gone from book.md but still on disk: the arm would continue from a
        # history its own chapter files do not describe.
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 4)
            book = nd / "book.md"
            book.write_text(book.read_text(encoding="utf-8").replace(
                _chapter(2), "被删掉了。\n"), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                MOD._truncate_book(nd, 3)
            self.assertIn("Ch2", str(cm.exception))

    def test_refuses_when_a_kept_chapter_file_is_gone(self):
        with TemporaryDirectory() as td:
            nd = _make_arm(Path(td), "arm", 4)
            (nd / "chapters" / "0002.md").unlink()
            with self.assertRaises(SystemExit):
                MOD._truncate_book(nd, 3)


class DatabaseRollbackTest(unittest.TestCase):
    """The DELETE/revert rules, on the real schema."""

    def _db(self, path: Path) -> sqlite3.Connection:
        import sys
        sys.path.insert(0, str(ROOT))
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE events (id INTEGER PRIMARY KEY, chapter INTEGER NOT NULL);
            CREATE TABLE open_threads (
                id TEXT PRIMARY KEY, description TEXT, status TEXT,
                introduced_chapter INTEGER, due_chapter INTEGER,
                updated_chapter INTEGER, payload_json TEXT);
        """)
        return conn

    def test_a_thread_closed_after_the_cut_is_reopened_and_the_clock_clamped(self):
        # A thread `recovered` at Ch180 was genuinely open at Ch170, and the
        # engine under test is the one that should pay it off. Leaving it closed
        # would hand the arm a payoff it never wrote.
        with TemporaryDirectory() as td:
            conn = self._db(Path(td) / "s.db")
            try:
                conn.execute("INSERT INTO open_threads VALUES "
                             "('t1','x','recovered',150,NULL,180,'{}')")
                conn.execute("INSERT INTO open_threads VALUES "
                             "('t2','y','advanced',175,NULL,190,'{}')")
                conn.execute("INSERT INTO open_threads VALUES "
                             "('t3','z','advanced',160,NULL,165,'{}')")
                conn.commit()
                keep = 170
                conn.execute("DELETE FROM open_threads WHERE introduced_chapter > ?",
                             (keep,))
                conn.execute("UPDATE open_threads SET status='advanced', "
                             "updated_chapter=? WHERE introduced_chapter <= ? "
                             "AND updated_chapter > ?", (keep, keep, keep))
                conn.commit()
                rows = {r[0]: (r[1], r[2]) for r in conn.execute(
                    "SELECT id, status, updated_chapter FROM open_threads")}
            finally:
                conn.close()
            self.assertNotIn("t2", rows, "introduced after the cut: never existed")
            self.assertEqual(rows["t1"], ("advanced", keep))
            self.assertEqual(rows["t3"], ("advanced", 165), "untouched before the cut")

    def test_the_live_state_tables_are_the_ones_v2_actually_reads(self):
        # If this list drifts from what v2/canon.py consults, the rollback stops
        # being faithful for the arm it exists to prepare.
        src = (ROOT / "engine" / "loop.py").read_text(encoding="utf-8")
        for fn in ("recent_events", "get_open_threads", "recent_metrics",
                   "get_overdue_reader_promises", "get_active_constraints"):
            self.assertIn(fn, src)
        tables = {t for t, _ in MOD.CHAPTER_KEYED} | {t for t, _, _ in MOD.LIVE_STATE}
        for backing in ("events", "chapter_metrics", "open_threads",
                        "reader_promises", "stage_constraints"):
            self.assertIn(backing, tables,
                          f"{backing} backs a v2 read and must be rolled back")

    def test_what_it_declines_to_roll_back_is_named(self):
        # A rollback that hides its residue reads as "matched" when it is not.
        self.assertTrue(MOD.UNROLLED)
        for table, why in MOD.UNROLLED.items():
            self.assertGreater(len(why), 20, f"{table} needs a stated reason")


if __name__ == "__main__":
    unittest.main()
