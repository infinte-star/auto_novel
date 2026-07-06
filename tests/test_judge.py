"""Unit tests for the pairwise judge (judge.py) — the ② feature.

The judge's ONLY network dependency is llm.call_llm; we monkeypatch it so these
tests are pure/offline. They verify the two load-bearing bits of judge logic:
frame normalization (A/B → a/b, swapped ordering flips) and the bias-controlled
resolution (a winner only when both orderings agree, else tie).

Run with:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import judge  # noqa: E402


class _Paths:
    logs_dir = None


class TestResolve(unittest.TestCase):
    def test_agree_decisive(self):
        self.assertEqual(judge._resolve("a", "a"), "a")

    def test_disagree_is_tie(self):
        self.assertEqual(judge._resolve("a", "b"), "tie")

    def test_one_tie_is_tie(self):
        self.assertEqual(judge._resolve("a", "tie"), "tie")


class TestJudgePairBiasControl(unittest.TestCase):
    """judge_pair calls _one_judge twice: first with (text_a as A, text_b as B),
    then swapped (text_b as A, text_a as B). We stub _one_judge to return fixed
    verdicts keyed by which text is in the 'A' slot."""

    def _run(self, first_winner: str, second_winner: str):
        calls = {"n": 0}

        def fake_one_judge(client, paths, config, left, right, context, cap, mt, temp):
            calls["n"] += 1
            w = first_winner if calls["n"] == 1 else second_winner
            return {"winner": w, "hook": "tie", "retention": "tie", "prose": "tie", "payoff": "tie", "reason": "x"}

        orig = judge._one_judge
        judge._one_judge = fake_one_judge
        try:
            return judge.judge_pair(None, _Paths(), {"novel": {}}, "TEXT_A", "TEXT_B")
        finally:
            judge._one_judge = orig

    def test_both_pick_text_a_wins_a(self):
        # ordering1: winner "A" == text_a. ordering2 (swapped): winner "B" == text_a.
        out = self._run("A", "B")
        self.assertEqual(out["winner"], "a")
        self.assertTrue(out["agreed"])

    def test_position_bias_always_A_becomes_tie(self):
        # A judge that always says "A" (pure first-position bias): ordering1 "A"→a,
        # ordering2 "A"→b → disagree → tie. This is exactly what swapping catches.
        out = self._run("A", "A")
        self.assertEqual(out["winner"], "tie")
        self.assertFalse(out["agreed"])

    def test_both_pick_text_b_wins_b(self):
        out = self._run("B", "A")
        self.assertEqual(out["winner"], "b")


class TestSampleEvenly(unittest.TestCase):
    def test_returns_all_when_small(self):
        self.assertEqual(judge._sample_evenly([1, 2, 3], 8), [1, 2, 3])

    def test_even_spread_covers_range(self):
        picks = judge._sample_evenly(list(range(1, 101)), 5)
        self.assertEqual(len(picks), 5)
        self.assertEqual(picks[0], 1)
        self.assertGreater(picks[-1], 70)  # includes the back half (sag zone)


if __name__ == "__main__":
    unittest.main()
