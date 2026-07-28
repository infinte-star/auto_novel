"""Unit tests for the L0 repair ladder (REDESIGN L5/L6).

The load-bearing contract: every L0 fixer must be safe to run unconditionally —
no dialogue damage, no canon corruption (fossil rotation is bank-only), and no
change at all when nothing fired.

The rework-trigger tests that used to open this file went with v1: `rework_trigger`
was a rule keyed on the self-score, and v2 has no score in any gate — release is
`quality.hard_block_reasons` alone (`v2/accept.py`). What replaced the
`_repair_fossil_rejects` escape hatch is ordering, not a hatch: v2 repairs before
the accept decision reads the report, so there is nothing to undo (`v2/repair.py`
docstring). Those behaviours are covered by `tests/test_v2_accept.py` and
`tests/test_v2_run.py`.

Zero LLM calls: only pure functions are exercised. `apply_l1` needs a client and
is covered by the offline replay + the live A/B instead.
"""
import unittest

import fix
import fossil_fix
from quality import REGISTRY


def _config(**novel) -> dict:
    cfg = {"quality_threshold": 8.0}
    cfg.update(novel)
    return {"novel": cfg}






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

    def test_hard_fossil_targets_zero_not_keep_one(self):
        """A book-cumulative hard reject can only be cleared by ZERO occurrences.

        Measured: 10 of the 12 archived chapters rejected on an entrenched bank
        phrase contain it exactly ONCE, so under the shared keep-1 target this
        fixer replaced nothing at all and the repair declared for the gate could
        never turn it green.
        """
        text = "他声音压得很低，说了一句。"
        review = {"book_fossils": {"hard_fossils": [{"phrase": "声音压得很低"}]}}
        out, replaced = fix.rotate_fossils(text, review, _config(), 3)
        self.assertEqual(replaced, ["声音压得很低"])
        self.assertNotIn("声音压得很低", out)

    def test_soft_and_hard_phrases_keep_their_own_targets(self):
        """One call, two targets: hard → 0 left, soft → keep 1."""
        text = ("他声音压得很低。" * 2) + ("她深吸一口气。" * 3)
        review = {"book_fossils": {
            "hard_fossils": [{"phrase": "声音压得很低"}],
            "fossils": [{"phrase": "深吸一口气"}],
        }}
        out, replaced = fix.rotate_fossils(text, review, _config(), 1)
        self.assertEqual(set(replaced), {"深吸一口气", "声音压得很低"})
        self.assertNotIn("声音压得很低", out)
        self.assertEqual(out.count("深吸一口气"), 1)


class SafeAltTest(unittest.TestCase):
    """A metric-improvement check cannot catch broken Chinese, so the grammar
    guard has to be structural. See fossil_fix._safe_alt."""

    ALTS = fossil_fix.FOSSIL_REPLACEMENTS["声音压得很低"]

    def test_after_an_attributive_de_only_a_same_head_variant_is_offered(self):
        alt = fossil_fix._safe_alt("声音压得很低", self.ALTS, 0, "的")
        self.assertIsNotNone(alt)
        self.assertTrue(alt.startswith("声"), alt)

    def test_mid_sentence_any_variant_is_fine(self):
        alt = fossil_fix._safe_alt("声音压得很低", self.ALTS, 0, "他")
        self.assertEqual(alt, self.ALTS[0])

    def test_no_same_head_variant_means_keep_the_fossil(self):
        self.assertIsNone(fossil_fix._safe_alt("声音压得很低", ["压着嗓子"], 0, "的"))

    def test_empty_bank_entry(self):
        self.assertIsNone(fossil_fix._safe_alt("声音压得很低", [], 0, ""))

    def test_attributive_position_is_never_left_ungrammatical(self):
        """The real Ch195 shape: 「顾峥的声音压得很低」 must not become
        「顾峥的压着嗓子」."""
        text = "顾峥的声音压得很低，只有她听得见。"
        review = {"book_fossils": {"hard_fossils": [{"phrase": "声音压得很低"}]}}
        out, replaced = fix.rotate_fossils(text, review, _config(), 195)
        self.assertEqual(replaced, ["声音压得很低"])
        self.assertNotIn("的压着嗓子", out)
        self.assertNotIn("的用气声说", out)
        self.assertTrue(out.startswith("顾峥的声"), out)


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
