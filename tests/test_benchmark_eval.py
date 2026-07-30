"""Tests for tools/benchmark_eval.py — dimensional parsing, tallying, gap report."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.anchor import DIMENSIONS, DimensionalVerdict, UNMEASURED
from tools.benchmark_eval import (
    BenchmarkReport,
    DimTally,
    _parse_dimensional,
    dimensional_judge_pair,
    gap_report,
)

SAMPLE_RESPONSE = json.dumps({
    "dimensions": {
        "pull": {"winner": "甲", "note": "甲的悬念更具体"},
        "dialogue": {"winner": "乙", "note": "乙对话驱动更强"},
        "payoff_visible": {"winner": "tie", "note": "差不多"},
        "hook": {"winner": "甲", "note": "甲结尾钩子更好"},
        "pacing": {"winner": "乙", "note": "乙节奏更好"},
        "prose_health": {"winner": "甲", "note": "甲更健康"},
        "concreteness": {"winner": "tie", "note": "都不错"},
        "immersion": {"winner": "甲", "note": "甲代入感强"},
    },
    "overall": {"winner": "甲", "reason": "甲整体牵引力更强"}
})


class TestParseDimensional(unittest.TestCase):

    def test_normal_parse(self):
        dims, overall, reasons, overall_reason = _parse_dimensional(
            SAMPLE_RESPONSE, ("a", "b")
        )
        self.assertEqual(dims["pull"], "a")
        self.assertEqual(dims["dialogue"], "b")
        self.assertEqual(dims["payoff_visible"], "tie")
        self.assertEqual(overall, "a")
        self.assertIn("牵引力", overall_reason)

    def test_swapped_sides(self):
        dims, overall, _, _ = _parse_dimensional(SAMPLE_RESPONSE, ("b", "a"))
        self.assertEqual(dims["pull"], "b")
        self.assertEqual(dims["dialogue"], "a")
        self.assertEqual(overall, "b")

    def test_garbage_input(self):
        dims, overall, _, _ = _parse_dimensional("not json at all", ("a", "b"))
        self.assertEqual(dims, {})
        self.assertEqual(overall, UNMEASURED)

    def test_empty_input(self):
        dims, overall, _, _ = _parse_dimensional("", ("a", "b"))
        self.assertEqual(overall, UNMEASURED)

    def test_partial_dimensions(self):
        partial = json.dumps({
            "dimensions": {"pull": {"winner": "甲", "note": "ok"}},
            "overall": {"winner": "tie", "reason": "meh"},
        })
        dims, overall, _, _ = _parse_dimensional(partial, ("a", "b"))
        self.assertEqual(dims["pull"], "a")
        self.assertEqual(dims.get("dialogue", UNMEASURED), UNMEASURED)
        self.assertEqual(overall, "tie")


SAMPLE_RESPONSE_SWAPPED = json.dumps({
    "dimensions": {
        "pull": {"winner": "乙", "note": "乙的悬念更具体"},
        "dialogue": {"winner": "甲", "note": "甲对话驱动更强"},
        "payoff_visible": {"winner": "tie", "note": "差不多"},
        "hook": {"winner": "乙", "note": "乙结尾钩子更好"},
        "pacing": {"winner": "甲", "note": "甲节奏更好"},
        "prose_health": {"winner": "乙", "note": "乙更健康"},
        "concreteness": {"winner": "tie", "note": "都不错"},
        "immersion": {"winner": "乙", "note": "乙代入感强"},
    },
    "overall": {"winner": "乙", "reason": "乙整体牵引力更强"}
})


class TestDimensionalJudgePair(unittest.TestCase):

    def test_concordant_pair(self):
        call_count = [0]

        def mock_call(system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                return SAMPLE_RESPONSE
            return SAMPLE_RESPONSE_SWAPPED

        v = dimensional_judge_pair("text a", "text b", call=mock_call, key="test")
        self.assertIsInstance(v, DimensionalVerdict)
        self.assertEqual(v.overall, "a")
        self.assertTrue(v.concordant)
        self.assertEqual(v.dims["pull"], "a")
        self.assertEqual(v.dims["dialogue"], "b")

    def test_discordant_overall(self):
        def mock_call(system, user):
            return SAMPLE_RESPONSE

        v = dimensional_judge_pair("text a", "text b", call=mock_call, key="disc")
        self.assertEqual(v.overall, "tie")

    def test_empty_text(self):
        v = dimensional_judge_pair("", "text b", call=lambda s, u: "", key="empty")
        self.assertEqual(v.overall, UNMEASURED)
        self.assertFalse(v.concordant)

    def test_call_failure(self):
        def fail_call(system, user):
            raise ConnectionError("timeout")

        v = dimensional_judge_pair("a", "b", call=fail_call, key="fail")
        self.assertEqual(v.overall, UNMEASURED)
        self.assertFalse(v.concordant)


class TestDimTally(unittest.TestCase):

    def test_rates(self):
        t = DimTally(wins_chapter=3, wins_anchor=1, ties=2)
        self.assertEqual(t.n, 6)
        self.assertAlmostEqual(t.chapter_rate, (3 + 1.0) / 6 * 100)
        self.assertAlmostEqual(t.anchor_rate, (1 + 1.0) / 6 * 100)

    def test_empty(self):
        t = DimTally()
        self.assertEqual(t.n, 0)
        self.assertEqual(t.chapter_rate, 0.0)


class TestBenchmarkReport(unittest.TestCase):

    def _make_report(self) -> BenchmarkReport:
        tallies = {}
        for i, (key, _) in enumerate(DIMENSIONS):
            t = DimTally()
            t.wins_chapter = 5 - i if i < 5 else 0
            t.wins_anchor = i
            t.ties = 2
            tallies[key] = t

        return BenchmarkReport(
            dim_tallies=tallies,
            overall_wins_chapter=10,
            overall_wins_anchor=5,
            overall_ties=3,
            verdicts=[],
            anchor_fingerprint="abc123",
            n_chapters=5,
            n_anchors=4,
        )

    def test_overall_rate(self):
        r = self._make_report()
        expected = (10 + 3 * 0.5) / 18 * 100
        self.assertAlmostEqual(r.overall_chapter_rate, expected)

    def test_weakest_dims(self):
        r = self._make_report()
        weakest = r.weakest_dims(3)
        self.assertEqual(len(weakest), 3)
        rates = [rate for _, rate in weakest]
        self.assertEqual(rates, sorted(rates))

    def test_gap_report_output(self):
        r = self._make_report()
        text = gap_report(r)
        self.assertIn("维度化爆款对标报告", text)
        self.assertIn("TOP-3 短板维度", text)
        self.assertIn("pull", text)
        self.assertIn("Overall WR", text)


class TestDimensionConstants(unittest.TestCase):

    def test_eight_dimensions(self):
        self.assertEqual(len(DIMENSIONS), 8)

    def test_unique_keys(self):
        keys = [k for k, _ in DIMENSIONS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_dimensional_verdict_fields(self):
        v = DimensionalVerdict(
            key="test",
            dims={"pull": "a"},
            overall="a",
            reasons={"pull": "ok"},
            overall_reason="fine",
            concordant=True,
        )
        self.assertTrue(v.concordant)
        self.assertEqual(v.dims["pull"], "a")


if __name__ == "__main__":
    unittest.main()
