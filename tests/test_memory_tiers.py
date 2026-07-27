"""`memory_context` must not ship the same rows twice.

tier2 carries the newest 5 chapter_metrics and tier3 the newest 20 story events;
tier4 used to carry `recent_metrics(fatigue_window)` and `recent_events(40)` in
full, which are supersets of those two. Measured on a live Ch49 novel that was
11,064 chars of a 121,617-char context -- 9%, in the single largest prompt the
engine sends (`plan_candidate`, median 131,872 chars over 2,501 library calls),
for zero information.

The invariant these tests hold: whatever appears in tier4 is strictly OLDER than
what tier2/tier3 already showed. It is a real invariant rather than a formatting
preference because both store helpers return newest-first, so the short lists are
prefixes of the long ones -- if that ever changes, the slicing silently drops
rows instead of duplicating them, and these tests are what catches it.
"""
import json
import unittest
from pathlib import Path

import config as _config
import memory


class _FakeConn:
    """Minimal stand-in for the sqlite3 connection `memory_context` is handed.

    Only the two queries `recent_metrics` / `recent_events` issue are answered;
    anything else returns empty so the surrounding helpers degrade quietly.
    """

    def __init__(self, n_metrics: int, n_events: int):
        # `marker` is metrics-only so a count over the whole context can tell a
        # metrics row apart from an event row (both carry a `chapter` field).
        self._metrics = [{"chapter": c, "score": 8.0, "marker": f"M{c}"}
                         for c in range(n_metrics, 0, -1)]
        self._events = [
            {"id": i, "chapter": i, "event_type": "story_event",
             "payload": json.dumps({"what": f"事件{i}"}, ensure_ascii=False),
             "created_at": "2026-07-28"}
            for i in range(n_events, 0, -1)
        ]

    def execute(self, sql: str, params=()):
        rows: list = []
        if "FROM chapter_metrics" in sql:
            rows = self._metrics[: int(params[-1])]
        elif "FROM events" in sql:
            rows = self._events[: int(params[-1])]
        return _FakeCursor(rows)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


def _paths(tmp: Path) -> _config.Paths:
    # Deliberately NOT created: every memory file is absent, `_read_memory_file`
    # returns "" for each, and the test leaves nothing on disk. What is under
    # test is the tier arithmetic, not the file readers.
    return _config.Paths(**{
        f: tmp / f"{f}.md" for f in (
            "book", "state", "title", "bible", "characters", "timeline",
            "threads", "volume_plan", "compass", "voices", "voice",
            "contract", "glossary")
    }, chapters_dir=tmp / "chapters", logs_dir=tmp / "logs", database=tmp / "s.db")


def _config_dict(fatigue_window: int) -> dict:
    return {"novel": {"fatigue_window": fatigue_window}, "api": {"context_window": 200000}}


class TestMemoryContextTierDedupe(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_tmp_memtiers"
        self.paths = _paths(self.tmp)

    def _build(self, n_metrics=30, n_events=60, fatigue_window=12) -> str:
        return memory.memory_context(
            self.paths, _FakeConn(n_metrics, n_events), _config_dict(fatigue_window))

    def test_old_full_headers_are_gone(self):
        ctx = self._build()
        self.assertNotIn("## 完整指标JSON", ctx)
        self.assertNotIn("## 完整事件JSON", ctx)

    def test_tier4_metrics_start_after_tier2(self):
        ctx = self._build(n_metrics=30, fatigue_window=12)
        tier2 = ctx.split("## 关键指标JSON", 1)[1].split("## 伏线", 1)[0]
        tier4 = ctx.split("## 更早的指标JSON", 1)[1].split("\n\n## ", 1)[0]
        # Newest-first: tier2 holds chapters 30..26, tier4 must resume at 25.
        self.assertIn('"M30"', tier2)
        self.assertIn('"M26"', tier2)
        self.assertNotIn('"M25"', tier2)
        self.assertIn('"M25"', tier4)
        self.assertNotIn('"M26"', tier4)

    def test_tier4_events_start_after_tier3(self):
        ctx = self._build(n_events=60)
        tier3 = ctx.split("## 近期事件JSON", 1)[1].split("\n\n## ", 1)[0]
        tier4 = ctx.split("## 更早的事件JSON", 1)[1].split("\n\n## ", 1)[0]
        self.assertIn("事件60", tier3)
        self.assertIn("事件41", tier3)
        self.assertNotIn("事件40", tier3)
        self.assertIn("事件40", tier4)
        self.assertNotIn("事件41", tier4)

    def test_no_metrics_tail_omits_the_section(self):
        # fatigue_window <= 5 means tier2 already showed everything tier4 would.
        ctx = self._build(n_metrics=30, fatigue_window=4)
        self.assertNotIn("## 更早的指标JSON", ctx)
        self.assertIn("## 关键指标JSON", ctx)

    def test_each_row_appears_exactly_once(self):
        """The whole point: no row is rendered into the prompt twice."""
        ctx = self._build(n_metrics=30, n_events=60, fatigue_window=12)
        self.assertEqual(ctx.count('"M30"'), 1)            # newest metric row
        self.assertEqual(ctx.count("事件60"), 1)           # newest event
        self.assertEqual(ctx.count("事件41"), 1)           # last row tier3 shows
        self.assertEqual(ctx.count("事件40"), 1)           # first row tier4 shows

    def test_nothing_is_dropped_between_the_tiers(self):
        """Dedupe must not open a gap: every event 21..60 is still somewhere."""
        ctx = self._build(n_metrics=30, n_events=60, fatigue_window=12)
        missing = [i for i in range(21, 61) if f"事件{i}" not in ctx]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
