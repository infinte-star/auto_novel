"""The arbitration decision is untrusted INPUT, including its keys.

`_coerce_index` already establishes the doctrine for arbitration *values*
("selected_index has shown up as '^1', ' 1 ', 1.0, None"). These tests cover the
level above it — the dict's shape — plus the retry rule that reads it.

Why it exists, measured on the archived corpus: 13 of guize_guaitan's 34
arbitrations carry keys mangled by `llm.load_json_with_repair`'s repair call
(`.selected_index`, `''`, `./output.json`, `./merged_plan.json`, `/assistant`,
`./scores`), every other book is clean. Two independent defects followed:

  1. `plan_score` returns 0.0 for an empty `scores` list, and 0.0 is below every
     threshold, so a vacuous decision forced a WHOLE extra plan round (~4 calls).
     11 library-wide, 7 of which no gate would have blocked anyway.
  2. The arbiter's `merged_plan` + `required_constraints` were silently lost
     (salvageable in 0 of guize_guaitan's 10 cases), so the chapter was written
     from an unarbitrated candidate with no hard constraints and nothing said so.

These readers moved to `quality.py` with v1's deletion (D1) — they guard an
untrusted payload with zero LLM calls, exactly like `hard_block_reasons`. Their
live consumers are now `writing.py` (the writer's "当前大纲仲裁分" line and
`chapter_metrics.plan_score`) and `tools/replay_gates.py` (which replays the 934
archived v1 arbitrations the docstring above measures). The re-ask test that used
to live here went with `planning.arbitrate_plan`.

Zero LLM calls: every function under test is pure.
"""
import unittest

from engine.quality import (_coerce_index, _normalize_decision, decision_has_score,
                     plan_score)

GOOD = {
    "selected_index": 1,
    "scores": [{"index": 0, "score": 7.0}, {"index": 1, "score": 8.5}],
    "merged_plan": {"title": "章", "goal": "g"},
    "required_constraints": [{"id": "a", "constraint": "c"}],
}


class NormalizeDecisionTests(unittest.TestCase):

    def test_a_well_formed_decision_is_untouched(self):
        self.assertEqual(_normalize_decision(dict(GOOD)), GOOD)

    def test_leading_dot_key_is_recovered(self):
        d = _normalize_decision({".selected_index": 2, "merged_plan": {"title": "x"}})
        self.assertEqual(d.get("selected_index"), 2)
        self.assertNotIn(".selected_index", d)

    def test_key_repair_does_not_resurrect_lost_content(self):
        # The honest limit of key repair, and the reason the call-site re-ask exists:
        # all 14 archived `.selected_index` payloads ship EMPTY values beside the bad
        # key. Repairing the spelling must not make the decision look usable.
        archived = {".selected_index": 0, "scores": [], "merged_plan": {},
                    "required_constraints": [], "reader_expectation_delta": ""}
        fixed = _normalize_decision(archived)
        self.assertEqual(fixed.get("selected_index"), 0)
        # Asserted through `decision_has_score` since `_decision_usable` was
        # deleted with v1's plan committee: the property is that repair recovers
        # the KEY and invents no CONTENT, and the live predicate still shows it.
        self.assertFalse(decision_has_score(fixed))
        self.assertFalse(fixed.get("merged_plan"))

    def test_bare_score_row_is_rebuilt_into_an_envelope(self):
        # tangshuting Ch175 verbatim shape: the arbiter emitted ONE element of
        # `scores` as the whole object. That is a real measurement, so rebuilding the
        # envelope saves a plan round instead of a re-ask.
        d = _normalize_decision({"index": 0, "score": 5.5,
                                 "pros": ["紧张的枪口对峙场景"], "cons": ["与设定矛盾"]})
        self.assertTrue(decision_has_score(d))
        self.assertEqual(plan_score(d), 5.5)
        self.assertEqual(d["selected_index"], 0)

    def test_score_row_rebuild_never_shadows_a_real_envelope(self):
        # A well-formed decision that happens to carry a stray top-level `score`
        # must keep its own `scores`/`merged_plan`.
        d = _normalize_decision({"score": 1.0, **GOOD})
        self.assertEqual(plan_score(d), 8.5)
        self.assertEqual(d["merged_plan"], GOOD["merged_plan"])

    def test_score_row_rebuild_ignores_non_numeric_scores(self):
        payload = {"score": "high", "note": "x"}
        self.assertEqual(_normalize_decision(payload), payload)

    def test_quoted_and_slashed_key_forms_are_recovered(self):
        for bad in ('"scores"', "./scores", " scores ", "'scores'"):
            with self.subTest(key=bad):
                d = _normalize_decision({bad: [{"index": 0, "score": 9.0}]})
                self.assertEqual(plan_score(d), 9.0, f"{bad!r} not repaired")

    def test_a_correctly_spelled_key_is_never_overwritten(self):
        # If both forms are present the good one wins — repair must not clobber
        # real data with debris that happens to normalize onto the same name.
        d = _normalize_decision({"selected_index": 0, ".selected_index": 9})
        self.assertEqual(d["selected_index"], 0)

    def test_single_key_wrapper_is_unwrapped(self):
        d = _normalize_decision({"./output.json": dict(GOOD)})
        self.assertEqual(d, GOOD)

    def test_wrapper_unwrap_requires_an_expected_key_inside(self):
        # A single-key dict of something else is left alone rather than guessed at.
        payload = {"unrelated": {"foo": 1}}
        self.assertEqual(_normalize_decision(payload), payload)

    def test_unrecognisable_keys_survive_verbatim(self):
        # Nothing is invented and nothing is dropped: the caller decides whether
        # what survived is usable, and the db_event logs the real key names.
        d = _normalize_decision({"./output.json": "{", "/assistant": ""})
        self.assertEqual(sorted(d), ["./output.json", "/assistant"])

    def test_non_dict_input_degrades_to_empty(self):
        for junk in (None, "", [], "text", 3):
            self.assertEqual(_normalize_decision(junk), {})


class DecisionHasScoreTests(unittest.TestCase):
    """A MISSING measurement must be distinguishable from a LOW one."""

    def test_scored_decision(self):
        self.assertTrue(decision_has_score(GOOD))

    def test_empty_scores_is_no_measurement(self):
        self.assertFalse(decision_has_score({"scores": [], "merged_plan": {"t": 1}}))
        self.assertEqual(plan_score({"scores": []}), 0.0,
                         "plan_score must keep its float contract for chapter_metrics")

    def test_arc_style_decision_has_no_score(self):
        # `arc.py` leaves `scores` empty ON PURPOSE — a fabricated score would
        # poison chapter_metrics.plan_score. So 0.0 can never be read as a verdict.
        self.assertFalse(decision_has_score({"scores": [], "source": "arc_card"}))

    def test_malformed_score_rows_do_not_count_as_a_measurement(self):
        self.assertFalse(decision_has_score({"scores": ["8.0"]}))
        self.assertFalse(decision_has_score({"scores": [{"index": 0}]}))
        self.assertFalse(decision_has_score(None))

    def test_a_genuinely_low_score_still_counts_as_measured(self):
        # The whole point: 3.0 is a verdict and must keep reaching the retry gate.
        low = {"selected_index": 0, "scores": [{"index": 0, "score": 3.0}]}
        self.assertTrue(decision_has_score(low))
        self.assertEqual(plan_score(low), 3.0)

    def test_zero_is_a_measurement_when_the_arbiter_really_said_zero(self):
        self.assertTrue(decision_has_score({"scores": [{"index": 0, "score": 0}]}))


class CoerceIndexStillHoldsTests(unittest.TestCase):
    """Regression guard: key repair must not have disturbed value coercion."""

    def test_documented_malformed_values(self):
        self.assertEqual(_coerce_index("^1"), 1)
        self.assertEqual(_coerce_index(" 1 "), 1)
        self.assertEqual(_coerce_index(1.0), 1)
        self.assertEqual(_coerce_index(None), 0)
        self.assertEqual(_coerce_index(True), 0)


if __name__ == "__main__":
    unittest.main()
