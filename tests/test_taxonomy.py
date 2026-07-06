"""Unit tests for the fiction failure taxonomy (taxonomy.py) — the ① feature.
Pure functions only; run with:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taxonomy  # noqa: E402


class TestClassify(unittest.TestCase):
    def test_every_prefix_maps(self):
        # Every legacy problems prefix must resolve to a known taxonomy code.
        for prefix, code in taxonomy.PREFIX_TO_CODE.items():
            self.assertIn(code, taxonomy.FAILURE_TAXONOMY, f"{prefix}->{code} not in taxonomy")
            self.assertEqual(taxonomy.classify_problem(f"{prefix}: 出问题了"), code)

    def test_chinese_colon_and_ascii_colon(self):
        self.assertEqual(taxonomy.classify_problem("RETENTION：读者流失"), "retention_sag")
        self.assertEqual(taxonomy.classify_problem("MARKET: weak"), "market_weak")

    def test_unknown_prefix_returns_none(self):
        self.assertIsNone(taxonomy.classify_problem("这是评审的自由文本，没有前缀"))
        self.assertIsNone(taxonomy.classify_problem(""))

    def test_gate_mapping(self):
        self.assertEqual(taxonomy.classify_gate("cross_chapter_repetition"), "fossil_repetition")
        self.assertEqual(taxonomy.classify_gate("book_wide_fossils"), "fossil_repetition")
        self.assertEqual(taxonomy.classify_gate("adjacent_repetition"), "adjacent_repeat")
        self.assertIsNone(taxonomy.classify_gate("nonexistent_gate"))


class TestRouting(unittest.TestCase):
    def test_fix_route_lookup(self):
        self.assertEqual(taxonomy.fix_route("retention_sag"), "arc_replan")
        self.assertEqual(taxonomy.fix_route("style_collapse"), "local")
        self.assertEqual(taxonomy.fix_route("unknown_code"), "local")

    def test_dominant_route_priority(self):
        # arc_replan beats structural beats local.
        self.assertEqual(taxonomy.dominant_route(["style_collapse", "retention_sag"]), "arc_replan")
        self.assertEqual(taxonomy.dominant_route(["style_collapse", "market_weak"]), "structural")
        self.assertEqual(taxonomy.dominant_route(["style_collapse", "adjacent_repeat"]), "local")
        self.assertIsNone(taxonomy.dominant_route([]))

    def test_replan_kind_binary(self):
        self.assertEqual(taxonomy.replan_kind(["fossil_repetition"]), "structural")
        self.assertEqual(taxonomy.replan_kind(["retention_sag"]), "structural")  # arc_replan -> structural
        self.assertEqual(taxonomy.replan_kind(["style_collapse"]), "local")
        self.assertEqual(taxonomy.replan_kind(["adjacent_repeat", "intra_recap"]), "local")
        self.assertIsNone(taxonomy.replan_kind([]))


class TestCodesFromReview(unittest.TestCase):
    def test_derives_from_problems_and_gates(self):
        report = {
            "problems": ["MARKET: 追读弱", "RETENTION: 兴奋度低", "这是没有前缀的评审文本"],
            "gate_rejects": [{"gate": "cross_chapter_repetition"}, {"gate": "adjacent_repetition"}],
        }
        codes = taxonomy.codes_from_review(report)
        self.assertEqual(
            set(codes), {"market_weak", "retention_sag", "fossil_repetition", "adjacent_repeat"}
        )

    def test_empty_report(self):
        self.assertEqual(taxonomy.codes_from_review({}), [])
        self.assertEqual(taxonomy.codes_from_review({"problems": ["纯自由文本"]}), [])


if __name__ == "__main__":
    unittest.main()
