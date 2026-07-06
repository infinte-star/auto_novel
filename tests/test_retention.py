"""Unit tests for the retention metric (retention.py) — the P0 deterministic
objective. Pure functions only; run with:

    python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retention import summarize_retention, retention_by_block, weighted_aggregate  # noqa: E402


class TestSummarizeRetention(unittest.TestCase):
    def test_empty(self):
        r = summarize_retention([], [])
        self.assertEqual(r["panels"], 0)
        self.assertIsNone(r["retention_index"])

    def test_strong_arc_beats_weak_arc(self):
        # A strong opening (high excitement, low drop) must score far above a
        # sagging back-half (low excitement, high drop) — the core P0 property.
        strong = summarize_retention([9.0, 8.5, 9.0, 8.0], [0.1, 0.2, 0.1, 0.2])
        weak = summarize_retention([3.0, 1.6, 3.0, 3.6], [0.8, 1.0, 0.8, 0.8])
        self.assertGreater(strong["retention_index"], weak["retention_index"])
        self.assertGreater(strong["retention_index"], 6.0)
        self.assertLess(weak["retention_index"], 2.0)

    def test_troughs_counted(self):
        r = summarize_retention([8.0, 1.6, 3.0, 9.0], low_excitement=4.0)
        self.assertEqual(r["trough_count"], 2)  # 1.6 and 3.0

    def test_drop_discounts_index(self):
        # Same excitement, higher drop => lower index (excitement×stay-rate).
        low_drop = summarize_retention([7.0, 7.0], [0.1, 0.1])
        high_drop = summarize_retention([7.0, 7.0], [0.8, 0.8])
        self.assertGreater(low_drop["retention_index"], high_drop["retention_index"])

    def test_no_drops_defaults_to_stayrate_one(self):
        r = summarize_retention([6.0, 6.0])
        self.assertEqual(r["mean_drop"], 0.0)
        self.assertEqual(r["retention_index"], 6.0)


class TestRetentionByBlock(unittest.TestCase):
    def test_exposes_midbook_sag(self):
        # Opening block strong, later block sagging — the curve must reveal it
        # even when a single overall average would blur the two together.
        series = [(c, 9.0, 0.1) for c in range(1, 11)] + [(c, 2.5, 0.8) for c in range(31, 41)]
        blocks = retention_by_block(series, block=10)
        by = {b["ch_from"]: b for b in blocks}
        self.assertIn(1, by)
        self.assertIn(31, by)
        self.assertGreater(by[1]["retention_index"], by[31]["retention_index"] + 3.0)


class TestWeightedAggregate(unittest.TestCase):
    def _rows(self):
        return [
            {"persona": "爽点党", "continue_reading": True, "would_pay": True, "excitement": 8},
            {"persona": "女频视角", "continue_reading": False, "would_pay": False, "excitement": 2},
        ]

    def test_uniform_equals_plain_mean(self):
        agg = weighted_aggregate(self._rows(), None)
        self.assertEqual(agg["drop_rate"], 0.5)
        self.assertEqual(agg["avg_excitement"], 5.0)

    def test_deweighting_mismatched_persona_lowers_drop(self):
        # De-weighting 女频视角 for a male-channel novel should reduce the drop
        # rate and raise excitement vs the uniform aggregate.
        weights = {"爽点党": 1.4, "女频视角": 0.4}
        agg = weighted_aggregate(self._rows(), weights)
        self.assertLess(agg["drop_rate"], 0.5)
        self.assertGreater(agg["avg_excitement"], 5.0)


if __name__ == "__main__":
    unittest.main()
