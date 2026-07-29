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

Zero LLM calls. The L1 fixers appear here anyway: their pre-call exits need no
client at all, and the post-call ones are reached with `_numbered_rewrite`
stubbed. What is asserted about them is not prose quality but whether a spent
call leaves a record — see `L1ExitsAreAudibleTest`.
"""
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import engine.quality as fix
import commands.fossil_fix as fossil_fix
import engine.quality as _engine_quality
from engine.quality import REGISTRY


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

    # --- the band has two sides and only one of them has a fixer here ---

    def test_the_long_side_plans_no_expand(self):
        """`expand_to_band` cannot fix an over-length chapter, so planning it
        burns one of two L1 slots on a guaranteed no-op. 109 of the archive's
        195 planned expands were this case."""
        review = {"length_band": {"flags": ["chapter_too_long(6336)"], "penalty": 1.17}}
        self.assertEqual(fix.plan_repairs(review, _config()), [])

    def test_the_short_side_still_plans_one(self):
        review = {"length_band": {"flags": ["chapter_too_short(2416)"]}}
        self.assertEqual(
            [s["action"] for s in fix.plan_repairs(review, _config())], ["expand_to_band"])

    def test_the_long_side_no_longer_displaces_a_real_fixer(self):
        """`tangshuting_e2e` Ch46 / `yeban_guize` Ch8 both lost `em_dash_targeted`
        to the cap because an over-length chapter held a slot ahead of it."""
        review = {
            "length_band": {"flags": ["chapter_too_long(4021)"], "penalty": 0.01},
            "dialogue_health": {"penalty": 0.4},
            "style_health": {"penalty": 0.5, "flags": ["em_dash_high"]},
        }
        steps = fix.plan_repairs(review, _config(fix_max_l1_calls=2))
        actions = [s["action"] for s in steps if s["layer"] == "L1"]
        self.assertEqual(actions, ["inject_dialogue", fix._EM_DASH_L1_ACTION])

    def test_an_unrecognized_length_flag_still_plans_the_expand(self):
        """Fail-safe direction: the rule is "not the long side", not "is the
        short side", so a drift in the gate's vocabulary cannot silently drop
        the short side's only fixer. The fixer's own floor check declines it,
        out loud."""
        review = {"length_band": {"flags": ["some_future_flag"]}}
        self.assertEqual(
            [s["action"] for s in fix.plan_repairs(review, _config())], ["expand_to_band"])

    def test_every_action_gate_declares_a_matching_layer(self):
        for gate, action in fix.ACTION_BY_GATE.items():
            layer = REGISTRY.repair(gate)
            self.assertIn(layer, ("L0", "L1"), f"{gate} declares {layer} but has action {action}")


class L1ExitsAreAudibleTest(unittest.TestCase):
    """L1 is the only repair layer that spends money; a silent exit hides that.

    `v2/repair.py` prints `blocks N->N` whether the layer ran or not, so a fixer
    that returned the text unchanged left no record of having been called at all.
    Across the settlement A/B's two v2 arms, 13 `fix_expand` calls produced 7
    `v2_repair_l1` events; the other 6 are invisible in the logs, and one of them
    (`ts_v2arm` Ch225, 13:45:02) prints the exact same line as a chapter whose L1
    row never fired. These tests pin the reason to the log.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.paths = types.SimpleNamespace(logs_dir=Path(self._tmp.name))

    def _log_text(self) -> str:
        p = self.paths.logs_dir / "run.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # --- exits that spend nothing, but still answer for a planned step ---

    def test_expand_says_so_when_the_chapter_is_already_at_the_floor(self):
        text = "第1章 门\n\n" + "他推开门，铁锈味涌了出来。" * 40
        out = fix.expand_to_band(None, self.paths, _config(chapter_min_chars=100),
                                 7, text, {})
        self.assertEqual(out, text)
        self.assertIn("L1 expand skipped Ch7", self._log_text())
        self.assertIn("already at the 100 floor", self._log_text())

    def test_expand_says_so_when_no_paragraph_can_be_expanded(self):
        text = "第1章 门\n\n他推开门。\n灯灭了。"
        out = fix.expand_to_band(None, self.paths, _config(chapter_min_chars=3000),
                                 8, text, {})
        self.assertEqual(out, text)
        self.assertIn("no paragraph long enough", self._log_text())

    # --- exits that spend a call and throw the result away ---

    def test_expand_says_so_when_the_call_returns_nothing_usable(self):
        text = "第1章 门\n\n" + "他推开门，铁锈味涌了出来，灯在头顶晃。" * 3
        with mock.patch.object(_engine_quality, "_numbered_rewrite", return_value={}):
            out = fix.expand_to_band(None, self.paths, _config(chapter_min_chars=3000),
                                     9, text, {})
        self.assertEqual(out, text)
        self.assertIn("L1 expand discarded Ch9", self._log_text())
        self.assertIn("no usable paragraph", self._log_text())

    def test_expand_says_so_when_the_splice_did_not_grow_the_chapter(self):
        para = "他推开门，铁锈味涌了出来，灯在头顶晃。" * 3
        text = f"第1章 门\n\n{para}"
        with mock.patch.object(_engine_quality, "_numbered_rewrite", return_value={0: "他推开门。"}):
            out = fix.expand_to_band(None, self.paths, _config(chapter_min_chars=3000),
                                     10, text, {})
        self.assertEqual(out, text)
        self.assertIn("did not grow", self._log_text())

    def test_dialogue_says_so_when_the_call_returns_nothing_usable(self):
        text = "第1章 门\n\n" + "他推开门，铁锈味涌了出来，灯在头顶晃了一下。" * 4
        with mock.patch.object(_engine_quality, "_numbered_rewrite", return_value={}):
            out = fix.inject_dialogue(None, self.paths, _config(), 11, text, {})
        self.assertEqual(out, text)
        self.assertIn("L1 dialogue discarded Ch11", self._log_text())

    def test_dialogue_says_so_when_the_ratio_is_already_at_target(self):
        # `dialogue_health` reports 0.0 below 200 chars, so the sample has to
        # clear that floor for the ratio branch to be the one under test.
        text = "第1章 门\n\n" + "“别过来。”他往后退了一步。" * 30
        out = fix.inject_dialogue(None, self.paths, _config(), 12, text, {})
        self.assertEqual(out, text)
        self.assertIn("already at the", self._log_text())

    # --- and the rule itself, so a new exit cannot be added silently ---

    def test_no_exit_from_a_paid_fixer_is_silent(self):
        """Structural: every `return text` in the two fixers that own their call
        must be answered for inside its OWN branch.

        The window is the enclosing suite, not the N preceding lines: a fixed
        lookback lets a newly-silenced exit hide behind the log line of the
        branch above it. That false negative was real — deleting
        `expand_to_band`'s "no usable paragraph" line left this test green
        because the `except` clause four lines up still logged.

        `em_dash_targeted` is deliberately out of scope: its call lives in
        `writing.reduce_em_dashes_targeted`, which logs each of its own outcomes,
        so its one bare exit is explained by the line above it rather than by a
        duplicate.
        """
        import inspect

        for fixer in (fix.expand_to_band, fix.inject_dialogue):
            lines = inspect.getsource(fixer).splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "return text":
                    continue
                indent = len(line) - len(line.lstrip())
                branch = []
                for prev in reversed(lines[:i]):
                    if not prev.strip():
                        continue
                    if len(prev) - len(prev.lstrip()) < indent:
                        break  # the `if`/`except` that opened this suite
                    branch.append(prev)
                self.assertIn(
                    "log(paths", "\n".join(branch),
                    f"{fixer.__name__} line {i + 1} returns the text without saying why")


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
