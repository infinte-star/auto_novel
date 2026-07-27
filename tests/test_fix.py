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
import fossil_fix
import pipeline
import planning
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


class RiskUpshiftFloorTest(unittest.TestCase):
    """The rework trigger and the risk upshift must not fight over the same score.

    Measured in the P4 A/B: the deterministic arm released Ch47-49 at 6.9/6.9/6.8,
    which the 7.0 risk floor read as collapse, so it bought 3 candidate drafts on
    3 of 4 chapters and ended up MORE expensive than the arm that reworked
    (16.25 vs 14.75 calls/chapter). See `planning._risk_score_floor`.
    """

    def test_score_mode_uses_the_plain_floor(self):
        self.assertEqual(planning._risk_score_floor(_config()), 7.0)
        self.assertEqual(
            planning._risk_score_floor(_config(risk_upshift_score_floor=7.5)), 7.5)

    def test_deterministic_mode_drops_to_the_rework_floor(self):
        cfg = _config(rework_trigger="deterministic")
        self.assertEqual(planning._risk_score_floor(cfg), 6.5)
        # A released 6.8 is normal in this mode, so it must not read as distress.
        self.assertLess(6.8, planning._risk_score_floor(_config()))
        self.assertGreater(6.8, planning._risk_score_floor(cfg))

    def test_deterministic_mode_never_raises_the_floor(self):
        """min(), not assignment: a config that already set a low floor keeps it."""
        cfg = _config(rework_trigger="deterministic",
                      risk_upshift_score_floor=6.0, rework_score_floor=6.5)
        self.assertEqual(planning._risk_score_floor(cfg), 6.0)

    def test_unknown_trigger_value_is_treated_as_score_mode(self):
        self.assertEqual(
            planning._risk_score_floor(_config(rework_trigger="  DETERMINISTIC ")), 6.5)
        self.assertEqual(planning._risk_score_floor(_config(rework_trigger="typo")), 7.0)


class RepairFossilRejectsTest(unittest.TestCase):
    """`pipeline._repair_fossil_rejects` clears a fossil reject only after the
    phrase is provably gone. It changes the text, never the ruler.

    Why it lives in the review loop rather than in `_stage_fix`: a fossil
    gate_reject routes `_classify_replan_failure` straight to a STRUCTURAL replan,
    and `_stage_fix` runs after that decision — the free repair could never
    prevent the expensive re-roll it exists to replace.
    """

    def _state(self, root, text, review):
        from config import Paths

        paths = Paths(
            book=root / "book.md", state=root / "state.md", title=root / "title.txt",
            bible=root / "b.md", characters=root / "c.md", timeline=root / "t.md",
            threads=root / "th.md", volume_plan=root / "vp.md", compass=root / "cp.md",
            voices=root / "vs.md", voice=root / "v.md", contract=root / "ct.md",
            glossary=root / "g.md", chapters_dir=root / "chapters",
            logs_dir=root / "logs", database=root / "story_state.db",
        )
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        st = pipeline.ChapterState(
            client=None, paths=paths, conn=None, config=_config(),
            chapter_num=7, background=None, resume=False,
        )
        st.chapter = text
        st.review = review
        return st

    def _run(self, text, review):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            st = self._state(Path(td), text, review)
            return st, pipeline._repair_fossil_rejects(st, 0)

    def _reject_review(self, **extra):
        review = {
            "score": 6.0, "accepted": True,
            "gate_rejects": [{"gate": "book_wide_fossils_ratio",
                              "phrases": ["声音压得很低"], "fracs": [0.31]}],
            "book_fossils": {"hard_fossils": [{"phrase": "声音压得很低", "frac": 0.31}]},
            "failure_codes": ["fossil_repetition"],
        }
        review.update(extra)
        return review

    def test_repaired_reject_is_cleared_and_the_phrase_is_gone(self):
        st, did = self._run("他声音压得很低，只说了一句。", self._reject_review())
        self.assertTrue(did)
        self.assertEqual(st.review["gate_rejects"], [])
        self.assertNotIn("声音压得很低", st.chapter)
        self.assertEqual(st.review["fossil_rotate"]["replaced"], ["声音压得很低"])
        self.assertEqual(st.review["fossil_rotate"]["stale_gates"], [])

    def test_failure_codes_are_rederived_not_left_stale(self):
        """`_classify_replan_failure` reads `failure_codes` BEFORE gate_rejects, so
        a stale `fossil_repetition` code would keep forcing the structural replan."""
        import taxonomy

        review = self._reject_review()
        self.assertEqual(taxonomy.replan_kind(review["failure_codes"]), "structural")
        st, _ = self._run("他声音压得很低。", review)
        self.assertNotIn("failure_codes", st.review)
        self.assertIsNone(taxonomy.replan_kind(st.review.get("failure_codes") or []))

    def test_an_unrelated_code_source_is_preserved(self):
        """Re-derivation must not amount to wiping the field: a code that comes from
        `problems` rather than from the repaired gate has to survive."""
        review = self._reject_review(problems=["REPEAT: 大段复述上一章"])
        review["failure_codes"] = ["adjacent_repeat", "fossil_repetition"]
        st, _ = self._run("他声音压得很低。", review)
        self.assertEqual(st.review["failure_codes"], ["adjacent_repeat"])

    def test_unrelated_rejects_survive(self):
        review = self._reject_review()
        review["gate_rejects"].append({"gate": "contract_hard", "count": 2})
        st, did = self._run("他声音压得很低。", review)
        self.assertTrue(did)
        self.assertEqual([g["gate"] for g in st.review["gate_rejects"]], ["contract_hard"])

    def test_unrotatable_phrase_keeps_the_reject(self):
        """Bank-only rotation: a book-specific proper noun must stay, and so must
        the reject that named it."""
        review = self._reject_review()
        review["gate_rejects"][0]["phrases"] = ["老市场街七号"]
        review["book_fossils"] = {"hard_fossils": [{"phrase": "老市场街七号"}]}
        st, did = self._run("老市场街七号的门开了。", review)
        self.assertFalse(did)
        self.assertEqual(len(st.review["gate_rejects"]), 1)
        self.assertIn("老市场街七号", st.chapter)

    def test_stale_reject_on_an_absent_phrase_is_cleared_and_labelled(self):
        """A reject for a phrase this chapter does not contain is the latching-gate
        failure mode — a forced replan with nothing here to act on. It clears, but
        as `stale`, so a gate bug never hides inside a repair."""
        st, did = self._run("门开了，谁也没说话。", self._reject_review())
        self.assertTrue(did)
        self.assertEqual(st.review["gate_rejects"], [])
        self.assertEqual(st.review["fossil_rotate"]["stale_gates"],
                         ["book_wide_fossils_ratio"])
        self.assertEqual(st.review["fossil_rotate"]["resolved_gates"], [])

    def test_idempotent_on_resume(self):
        """Resume replays the cached review against an already-rotated chapter;
        the reject must still clear even though nothing is left to replace."""
        st, _ = self._run("他声音压得很低。", self._reject_review())
        st2, did = self._run(st.chapter, self._reject_review())
        self.assertTrue(did)
        self.assertEqual(st2.review["gate_rejects"], [])

    def test_no_fossil_reject_is_a_no_op(self):
        review = {"score": 6.0, "gate_rejects": [{"gate": "contract_hard"}]}
        st, did = self._run("他声音压得很低。", review)
        self.assertFalse(did)
        self.assertIn("声音压得很低", st.chapter)
        self.assertNotIn("fossil_rotate", st.review)

    def test_disabled_with_l0(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            st = self._state(Path(td), "他声音压得很低。", self._reject_review())
            st.config = _config(fix_l0_enabled=False)
            self.assertFalse(pipeline._repair_fossil_rejects(st, 0))
            self.assertEqual(len(st.review["gate_rejects"]), 1)

    def test_score_and_penalty_are_never_touched(self):
        """The score was computed from the pre-rotation text. Crediting the fossil
        penalty back would leave score and measurement describing different texts —
        the same trap `_stage_fix` avoids with `style_health_after_fix`."""
        st, _ = self._run("他声音压得很低。", self._reject_review())
        self.assertEqual(st.review["score"], 6.0)
        self.assertEqual(st.review["book_fossils"]["hard_fossils"][0]["frac"], 0.31)


if __name__ == "__main__":
    unittest.main()
