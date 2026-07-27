"""Unit tests for the P4 rework trigger and the L0 repair ladder (REDESIGN L5/L6).

Two contracts are load-bearing here:

1. `pipeline._rework_needed` in ``rework_trigger: score`` mode must be
   point-for-point identical to the expression it replaced
   (``score < quality_threshold or not accepted``) across a
   (score × accepted × gate_rejects) grid. The default config must not change
   engine behaviour at all — otherwise the eventual A/B has two variables.
2. Every L0 fixer must be safe to run unconditionally: no dialogue damage, no
   canon corruption (fossil rotation is bank-only), and no change at all when
   nothing fired.

Zero LLM calls: only pure functions are exercised. `apply_l1` needs a client and
is covered by the offline replay + the live A/B instead.
"""
import unittest

import fix
import pipeline
from quality import REGISTRY


def _config(**novel) -> dict:
    cfg = {"quality_threshold": 8.0}
    cfg.update(novel)
    return {"novel": cfg}


class RewordTriggerScoreModeTest(unittest.TestCase):
    """Score mode must reproduce the historical predicate exactly."""

    def test_equivalent_to_legacy_expression_on_full_grid(self):
        threshold = 8.0
        config = _config()
        for score in (0.0, 4.6, 6.4, 6.5, 7.6, 7.99, 8.0, 8.5, 10.0):
            for accepted in (True, False):
                for rejects in ([], [{"gate": "cross_chapter_repetition"}]):
                    review = {"score": score, "accepted": accepted, "gate_rejects": rejects}
                    legacy = score < threshold or not accepted
                    got, reason = pipeline._rework_needed(review, config, 1)
                    self.assertEqual(
                        got, legacy,
                        f"score={score} accepted={accepted} rejects={bool(rejects)}",
                    )
                    self.assertEqual(bool(reason), got)

    def test_unknown_trigger_value_falls_back_to_score_mode(self):
        review = {"score": 7.6, "accepted": False}
        for value in ("", "SCORE", "nonsense", None):
            got, _ = pipeline._rework_needed(review, _config(rework_trigger=value), 1)
            self.assertTrue(got)

    def test_missing_score_key_is_treated_as_below_threshold(self):
        got, _ = pipeline._rework_needed({"accepted": True}, _config(), 1)
        self.assertTrue(got)


class RewordTriggerDeterministicModeTest(unittest.TestCase):
    def setUp(self):
        self.config = _config(rework_trigger="deterministic", rework_score_floor=6.5)

    def _needed(self, **review):
        review.setdefault("score", 7.6)
        review.setdefault("accepted", False)
        return pipeline._rework_needed(review, self.config, 1)

    def test_noise_band_chapter_is_accepted(self):
        """The whole point: 7.6 with accepted=False from the threshold alone."""
        needed, reason = self._needed()
        self.assertFalse(needed)
        self.assertEqual(reason, "")

    def test_score_under_floor_reworks(self):
        needed, reason = self._needed(score=6.4)
        self.assertTrue(needed)
        self.assertIn("floor", reason)

    def test_floor_is_inclusive(self):
        self.assertFalse(self._needed(score=6.5)[0])

    def test_gate_reject_reworks(self):
        needed, reason = self._needed(gate_rejects=[{"gate": "cross_chapter_repetition"}])
        self.assertTrue(needed)
        self.assertIn("cross_chapter_repetition", reason)

    def test_style_collapse_reworks(self):
        needed, reason = self._needed(style_health={"penalty": 2.5})
        self.assertTrue(needed)
        self.assertIn("style_collapse", reason)

    def test_style_penalty_below_block_does_not_rework(self):
        self.assertFalse(self._needed(style_health={"penalty": 1.0})[0])

    def test_hard_contradiction_reworks(self):
        needed, reason = self._needed(contradictions=[{"severity": "hard", "issue": "x"}])
        self.assertTrue(needed)
        self.assertIn("hard_contradictions", reason)

    def test_soft_contradiction_does_not_rework(self):
        self.assertFalse(self._needed(contradictions=[{"severity": "soft"}])[0])

    def test_hard_contract_violation_reworks(self):
        self.assertTrue(self._needed(contract_violations=[{"severity": "hard"}])[0])

    def test_length_band_block_reworks(self):
        needed, reason = self._needed(length_band={"block": True})
        self.assertTrue(needed)
        self.assertIn("length_band", reason)

    def test_adjacent_repetition_block_reworks(self):
        self.assertTrue(self._needed(adjacent_repetition={"level": "block"})[0])

    def test_constraint_pileup_reworks(self):
        needed, reason = self._needed(constraint_violations_structured=[1, 2, 3])
        self.assertTrue(needed)
        self.assertIn("constraints_unmet", reason)

    def test_two_unmet_constraints_do_not_rework(self):
        self.assertFalse(self._needed(constraint_violations_structured=[1, 2])[0])

    def test_unexplained_not_accepted_reworks(self):
        """accepted=False above threshold means an unenumerated block fired."""
        needed, reason = self._needed(score=8.5, accepted=False)
        self.assertTrue(needed)
        self.assertIn("accepted=False", reason)

    def test_clean_high_score_does_not_rework(self):
        self.assertFalse(self._needed(score=8.5, accepted=True)[0])


class PlanRepairsTest(unittest.TestCase):
    def test_l0_before_l1(self):
        review = {
            "length_band": {"flags": ["short"]},
            "cross_chapter_repetition": {"level": "advise"},
        }
        steps = fix.plan_repairs(review, _config())
        self.assertEqual([s["layer"] for s in steps], ["L0", "L1"])
        self.assertEqual(steps[0]["action"], "fossil_rotate")
        self.assertEqual(steps[1]["action"], "expand_to_band")

    def test_three_fossil_gates_collapse_to_one_action(self):
        review = {
            "cross_chapter_repetition": {"level": "advise"},
            "book_fossils": {"fossils": [{"phrase": "声音压得很低"}]},
            "descriptor_frequency": {"flagged": ["虎口旧疤"]},
        }
        steps = fix.plan_repairs(review, _config())
        self.assertEqual([s["action"] for s in steps], ["fossil_rotate"])

    def test_l1_step_count_is_capped(self):
        review = {
            "length_band": {"flags": ["short"]},
            "dialogue_health": {"penalty": 0.4},
            "style_health": {"penalty": 0.5, "flags": ["em_dash_high"]},
        }
        self.assertEqual(len(fix.plan_repairs(review, _config(fix_max_l1_calls=1))), 2)  # 1×L0 + 1×L1
        self.assertEqual(len(fix.plan_repairs(review, _config(fix_max_l1_calls=0))), 1)  # L0 only

    def test_em_dash_escalation_is_last_and_only_with_the_flag(self):
        flagged = {"style_health": {"penalty": 0.5, "flags": ["em_dash_high"]}}
        steps = fix.plan_repairs(flagged, _config())
        self.assertEqual(steps[-1]["action"], fix._EM_DASH_L1_ACTION)
        plain = {"style_health": {"penalty": 0.5, "flags": ["short_sentences"]}}
        self.assertNotIn(
            fix._EM_DASH_L1_ACTION,
            [s["action"] for s in fix.plan_repairs(plain, _config())],
        )

    def test_advisory_and_l2_gates_never_produce_steps(self):
        review = {
            "prose_texture": {"penalty": 0.4},
            "long_span_fatigue": {"penalty": 0.5},
            "information_density": {"penalty": 0.3},
            "genre_adherence": {"penalty": 1.0},
            "adjacent_repetition": {"level": "block"},
        }
        self.assertEqual(fix.plan_repairs(review, _config()), [])

    def test_gates_that_did_not_run_produce_no_steps(self):
        self.assertEqual(fix.plan_repairs({}, _config()), [])
        self.assertEqual(fix.plan_repairs({"style_health": {"penalty": 0.0}}, _config()), [])

    def test_report_key_indirection(self):
        """Reading `length_band_check`/`book_wide_fossils` by gate name finds nothing."""
        self.assertEqual(fix.plan_repairs({"length_band_check": {"flags": ["short"]}}, _config()), [])
        self.assertTrue(fix.plan_repairs({"length_band": {"flags": ["short"]}}, _config()))

    def test_every_action_gate_declares_a_matching_layer(self):
        for gate, action in fix.ACTION_BY_GATE.items():
            layer = REGISTRY.repair(gate)
            self.assertIn(layer, ("L0", "L1"), f"{gate} declares {layer} but has action {action}")


class MergeFragmentLinesTest(unittest.TestCase):
    def test_merges_dangling_clause_forward(self):
        text = "第1章 test\n\n他推开门\n铁锈味涌出来。"
        out = fix.merge_fragment_lines(text, _config())
        self.assertIn("他推开门，铁锈味涌出来。", out)

    def test_does_not_absorb_dialogue(self):
        text = "他推开门\n“别过来。”"
        self.assertEqual(fix.merge_fragment_lines(text, _config()), text)

    def test_leaves_dialogue_fragment_alone(self):
        text = "“别过来\n他往后退了一步。"
        self.assertEqual(fix.merge_fragment_lines(text, _config()), text)

    def test_never_merges_the_title_line(self):
        text = "第12章 旧档案室\n应急灯亮着。"
        self.assertEqual(fix.merge_fragment_lines(text, _config()), text)

    def test_respects_max_line_length(self):
        text = "他推开门\n" + "长" * 200 + "。"
        self.assertEqual(fix.merge_fragment_lines(text, _config(fix_merge_max_line_chars=60)), text)

    def test_complete_sentences_are_untouched(self):
        text = "他推开门。\n铁锈味涌出来。\n灯灭了。"
        self.assertEqual(fix.merge_fragment_lines(text, _config()), text)

    def test_only_merged_lines_get_closing_punctuation(self):
        """A fragment with nothing to merge into must NOT be punctuated in place:
        that moves the metric without improving the prose."""
        text = "他推开门"
        self.assertEqual(fix.merge_fragment_lines(text, _config()), text)

    def test_empty_input(self):
        self.assertEqual(fix.merge_fragment_lines("", _config()), "")


class RotateFossilsTest(unittest.TestCase):
    def test_rotates_a_bank_phrase_beyond_the_keep_limit(self):
        text = "他声音压得很低。\n她也声音压得很低。\n他又一次声音压得很低。"
        review = {"book_fossils": {"fossils": [{"phrase": "声音压得很低"}]}}
        out, replaced = fix.rotate_fossils(text, review, _config(), 3)
        self.assertEqual(replaced, ["声音压得很低"])
        self.assertEqual(out.count("声音压得很低"), 1)

    def test_book_specific_proper_noun_is_never_rotated(self):
        """Canon safety: rotation is bank-only, so a street name survives."""
        text = "老市场街七号的门开了。" * 4
        review = {"book_fossils": {"fossils": [{"phrase": "老市场街七号"}]}}
        out, replaced = fix.rotate_fossils(text, review, _config(), 3)
        self.assertEqual((out, replaced), (text, []))

    def test_bank_phrase_inside_a_repeated_clause_is_found(self):
        text = "他声音压得很低地说。\n她声音压得很低地说。"
        review = {"cross_chapter_repetition": {"repeats": [{"clause": "他声音压得很低地说"}]}}
        out, replaced = fix.rotate_fossils(text, review, _config(), 2)
        self.assertEqual(replaced, ["声音压得很低"])

    def test_no_named_phrases_is_a_no_op(self):
        text = "他声音压得很低。" * 5
        out, replaced = fix.rotate_fossils(text, {}, _config(), 1)
        self.assertEqual((out, replaced), (text, []))


class ApplyL0Test(unittest.TestCase):
    def test_no_fired_gate_means_no_change(self):
        text = "他推开门，铁锈味涌出来。\n\n“别过来。”她说。"
        out, applied = fix.apply_l0(text, {}, _config(), 1)
        self.assertEqual((out, applied), (text, []))

    def test_disabled_by_config(self):
        text = "他推开门\n铁锈味涌出来。"
        review = {"style_health": {"penalty": 0.5, "flags": ["fragment_lines"]}}
        out, applied = fix.apply_l0(text, review, _config(fix_l0_enabled=False), 1)
        self.assertEqual((out, applied), (text, []))

    def test_fragment_repair_is_applied_and_reported(self):
        text = "第1章 test\n\n" + "他推开门\n铁锈味涌出来。\n" * 6
        review = {"style_health": {"penalty": 0.5, "flags": ["fragment_lines_high"]}}
        out, applied = fix.apply_l0(text, review, _config(), 1)
        self.assertIn("merge_fragment_lines", applied)
        self.assertIn("他推开门，铁锈味涌出来。", out)

    def test_empty_text(self):
        self.assertEqual(fix.apply_l0("", {"style_health": {"penalty": 1}}, _config(), 1), ("", []))


class ReduceEmDashIfNeededTest(unittest.TestCase):
    def test_under_target_is_untouched(self):
        text = "他推开门，铁锈味涌出来。" * 20
        self.assertEqual(fix.reduce_em_dash_if_needed(text, _config()), text)

    def test_over_target_is_reduced(self):
        # >= 200 chars: style_health returns no metrics below that, and an
        # unmeasured chapter must be left alone (see test below).
        text = "他推开门——铁锈味——灯灭了。" * 20
        out = fix.reduce_em_dash_if_needed(text, _config())
        self.assertLess(out.count("——"), text.count("——"))

    def test_text_too_short_to_measure_is_untouched(self):
        text = "他推开门——铁锈味——灯灭了。" * 5
        self.assertEqual(fix.reduce_em_dash_if_needed(text, _config()), text)

    def test_disabled_by_config(self):
        text = "他推开门——铁锈味——灯灭了。" * 20
        self.assertEqual(fix.reduce_em_dash_if_needed(text, _config(em_dash_reduce_enabled=False)), text)


if __name__ == "__main__":
    unittest.main()
