"""What the cross-book sink can SEE of a v2 book.

`telemetry.py` is a strict observer with no consumers today, so nothing here
protects a live code path. What it protects is a number a future consumer would
read: the event filter was v1-only, so a v2-written book backfilled ZERO events
and reported success. A sink that silently imports nothing looks exactly like a
book that had nothing to report.

The seam is the real one — a real SQLite `events` table written by the same
`store.db_event` the engine uses, read by the same `_iter_source_rows` backfill
uses. A test that asserted against `IMPORTED_EVENT_TYPES` directly would pass
even if the SQL had drifted away from the tuple.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import engine.store as store
import commands.telemetry as telemetry
from engine.config import Paths


def _paths(root: Path) -> Paths:
    return Paths(
        book=root / "book.md", state=root / "state.md", title=root / "title.txt",
        bible=root / "b.md", characters=root / "c.md", timeline=root / "t.md",
        threads=root / "th.md", volume_plan=root / "vp.md", compass=root / "cp.md",
        voices=root / "vs.md", voice=root / "v.md", contract=root / "ct.md",
        glossary=root / "g.md", chapters_dir=root / "chapters",
        logs_dir=root / "logs", database=root / "story_state.db",
    )


class ImportedEventsTest(unittest.TestCase):
    """Read-only: `_iter_source_rows` never touches the global DB."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # LIFO: the SQLite handle must be closed before Windows will let the
        # directory go.
        self.addCleanup(self._tmp.cleanup)
        self.conn = store.init_db(_paths(self.root))
        self.addCleanup(self.conn.close_current)

    def _events(self, written: list[tuple[str, dict]]) -> list[str]:
        for i, (etype, payload) in enumerate(written, start=1):
            store.db_event(self.conn, i, etype, payload)
        # No config.yaml in the dir -> `_iter_source_rows` falls back to
        # story_state.db, which is the shipped default path anyway.
        return [str(row.get("event_type"))
                for kind, row in telemetry._iter_source_rows(self.root, {})
                if kind == "event"]

    def test_a_v2_books_rework_events_are_imported(self):
        """The bug: every one of these was invisible, so the count read 0."""
        seen = self._events([
            ("v2_rescue", {"reasons": ["length_band"]}),
            ("v2_repair_l0", {"actions": ["em_dash"]}),
            ("v2_repair_l1", {"actions": ["expand"]}),
            ("card_repair", {"problems": ["payoff 三连"]}),
            ("card_unresolved", {"problems": ["opening_type 与上一章相同"]}),
            ("card_degraded", {"reason": "arc call failed"}),
            ("v2_chapter_trace", {"blocks": []}),
            ("gate_reject", {"gate": "book_wide_fossils"}),
        ])
        self.assertEqual(len(seen), 8, f"a v2 book still imports too little: {seen}")

    def test_the_v1_archive_still_imports(self):
        """Most of the library is v1-written; widening must not narrow."""
        seen = self._events([
            ("plan_arbitration", {"plans": [], "decision": {}}),
            ("cold_reader", {"verdict": "ok"}),
            ("quality_debt", {"reasons": ["style"]}),
            ("panel_report", {"panel": []}),
        ])
        self.assertEqual(sorted(seen), sorted(
            ["plan_arbitration", "cold_reader", "quality_debt", "panel_report"]))

    def test_canon_bulk_and_bookkeeping_stay_out(self):
        """`story_event` is ~1.2k rows a book and is canon, not telemetry.

        Importing it would make the sink a second copy of every book's state —
        the drift risk `canon.apply_delta`'s one-writer rule exists to avoid.
        """
        seen = self._events([
            ("story_event", {"summary": "甲见了乙"}),
            ("chapter_completed", {"chars": 4000}),
            ("chapter_extraction", {"entities": []}),
            ("bootstrap", {"files": 5}),
            ("v2_rescue", {"reasons": []}),
        ])
        self.assertEqual(seen, ["v2_rescue"])

    def test_no_telemetry_enabled_knob_is_promised(self):
        """The key was deleted 2026-07-28; nothing may resurrect it silently.

        Backfill is user-typed and reconstructs from primary data, so a config
        veto would be wrong, not missing. If a live per-chapter writer is ever
        added back, THAT is when a knob becomes meaningful — and this test is
        where the promise has to be re-made deliberately.
        """
        src = Path(telemetry.__file__).read_text(encoding="utf-8")
        self.assertNotIn('get("telemetry_enabled"', src)
        for tmpl in ("config_template.example.yaml",):
            text = (Path(__file__).resolve().parent.parent / tmpl).read_text(
                encoding="utf-8")
            live = [ln for ln in text.splitlines()
                    if ln.strip().startswith("telemetry_enabled")]
            self.assertEqual(live, [], f"{tmpl} re-declares a dead key: {live}")


if __name__ == "__main__":
    unittest.main()
