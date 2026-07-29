"""`fingerprint_avoidance_context` must summarize, not enumerate.

The block it feeds was measured at 22,813 of 116,592 chars (19.6%) of a real
`plan_candidate` prompt at Ch201 -- the largest single block in the engine's
largest prompt -- and it grew by one line per chapter forever. It was also
unable to deliver its own stated signal ("高频出现的流程组合"): 200 chapters held
194 distinct whole flows.

These tests hold both halves of the fix: the output is bounded and roughly
constant in book length, and the recurring-pattern signal is actually present.
"""
import json
import sqlite3
import unittest

import engine.quality as quality


def _db(flows: list[list[str]], payoff: str = "reveal", conflict: str = "personnel"):
    """In-memory chapter_fingerprints table holding one row per flow."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE chapter_fingerprints (chapter INTEGER PRIMARY KEY,"
        " skeleton_tokens TEXT, narrative_moves TEXT, payoff_type TEXT,"
        " conflict_type TEXT, created_at TEXT)"
    )
    for i, moves in enumerate(flows, start=1):
        conn.execute(
            "INSERT INTO chapter_fingerprints VALUES (?,?,?,?,?,?)",
            (i, "[]", json.dumps(moves), payoff, conflict, "2026-07-28"),
        )
    conn.commit()
    return conn


CFG: dict = {"novel": {}}


class TestFingerprintContext(unittest.TestCase):
    def test_recurring_bigram_is_reported_with_its_count(self):
        # 5 chapters walking the same two moves: that pair is the whole signal.
        ctx = quality.fingerprint_avoidance_context(
            _db([["enter_space", "collect_evidence"]] * 5), CFG)
        self.assertIn("enter_space→collect_evidence ×5", ctx)
        self.assertIn("高频相邻推进对", ctx)

    def test_rare_patterns_are_not_reported(self):
        # Below _FP_MIN_REPEAT there is no evidence of overuse; saying it anyway
        # is what made the old block noise.
        ctx = quality.fingerprint_avoidance_context(
            _db([["a1", "b1"], ["a2", "b2"]]), CFG)
        self.assertNotIn("a1→b1", ctx)
        self.assertNotIn("高频相邻推进对", ctx)

    def test_no_per_chapter_lines(self):
        ctx = quality.fingerprint_avoidance_context(
            _db([["enter_space", "collect_evidence"]] * 5), CFG)
        # The old format was "Ch7: enter_space→…"; nothing may reintroduce it.
        self.assertNotIn("Ch1:", ctx)
        self.assertNotIn("Ch5:", ctx)

    def test_size_is_bounded_in_book_length(self):
        """A 20-chapter book and a 400-chapter book cost about the same."""
        moves = ["enter_space", "collect_evidence", "compare_data",
                 "deduce_conclusion", "new_threat"]
        small = quality.fingerprint_avoidance_context(
            _db([moves[i % 5:] + moves[: i % 5] for i in range(20)]), CFG)
        big = quality.fingerprint_avoidance_context(
            _db([moves[i % 5:] + moves[: i % 5] for i in range(400)]), CFG)
        # Counts get more digits, nothing else grows: 20x the chapters must not
        # cost even 1.5x the characters.
        self.assertLess(len(big), len(small) * 1.5)
        self.assertLess(len(big), 2500)

    def test_type_frequencies_are_present(self):
        ctx = quality.fingerprint_avoidance_context(
            _db([["enter_space", "collect_evidence"]] * 4,
                payoff="reveal", conflict="court"), CFG)
        self.assertIn("已用兑现类型频次：reveal×4", ctx)
        self.assertIn("已用冲突类型频次：court×4", ctx)
        self.assertIn("单步使用频次：", ctx)

    def test_empty_and_missing_inputs_degrade_to_none(self):
        self.assertEqual(quality.fingerprint_avoidance_context(None, CFG), "None")
        self.assertEqual(quality.fingerprint_avoidance_context(_db([]), CFG), "None")
        # Rows present but every flow empty: still nothing to say.
        self.assertEqual(
            quality.fingerprint_avoidance_context(_db([[], []]), CFG), "None")

    def test_a_broken_table_does_not_raise(self):
        """Plan generation must never die on this block (it catches, but so does the callee)."""
        conn = sqlite3.connect(":memory:")
        self.assertEqual(quality.fingerprint_avoidance_context(conn, CFG), "None")


if __name__ == "__main__":
    unittest.main()
