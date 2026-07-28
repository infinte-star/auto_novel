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

Zero LLM calls: every function under test is pure.
"""
import json
import unittest
from unittest import mock

import planning
from planning import (_coerce_index, _decision_usable, _normalize_decision,
                      decision_has_score, plan_score)

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
        self.assertFalse(_decision_usable(fixed))

    def test_bare_score_row_is_rebuilt_into_an_envelope(self):
        # tangshuting Ch175 verbatim shape: the arbiter emitted ONE element of
        # `scores` as the whole object. That is a real measurement, so rebuilding the
        # envelope saves a plan round instead of a re-ask.
        d = _normalize_decision({"index": 0, "score": 5.5,
                                 "pros": ["紧张的枪口对峙场景"], "cons": ["与设定矛盾"]})
        self.assertTrue(_decision_usable(d))
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
        # Nothing is invented and nothing is dropped: the caller's `_decision_usable`
        # check is what decides, and the db_event logs the real key names.
        d = _normalize_decision({"./output.json": "{", "/assistant": ""})
        self.assertEqual(sorted(d), ["./output.json", "/assistant"])

    def test_non_dict_input_degrades_to_empty(self):
        for junk in (None, "", [], "text", 3):
            self.assertEqual(_normalize_decision(junk), {})


class DecisionUsableTests(unittest.TestCase):

    def test_scores_alone_is_usable(self):
        self.assertTrue(_decision_usable({"scores": [{"index": 0, "score": 8.0}]}))

    def test_merged_plan_alone_is_usable(self):
        self.assertTrue(_decision_usable({"merged_plan": {"title": "x"}}))

    def test_salvage_debris_is_not_usable(self):
        self.assertFalse(_decision_usable({"./output.json": "{"}))
        self.assertFalse(_decision_usable({}))
        self.assertFalse(_decision_usable(None))

    def test_empty_containers_are_not_usable(self):
        # The exact archived shape: keys present, both values vacuous.
        self.assertFalse(_decision_usable({"scores": [], "merged_plan": {}}))

    def test_selected_index_alone_is_not_usable(self):
        # It carries neither the measurement nor the merge, so it cannot stand in
        # for an arbitration — but it must not crash the predicate either.
        self.assertFalse(_decision_usable({"selected_index": 0}))


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


CANDIDATE = {"title": "候选0", "goal": "g", "beats": ["a"]}


class ArbiterReaskTests(unittest.TestCase):
    """The re-ask branch, stubbed. It fires on 1.7% of arbitrations (16/934), which
    is exactly the frequency at which a typo would live in production unnoticed."""

    def setUp(self):
        self.calls: list[str] = []
        patches = [
            mock.patch.object(planning, "lite_memory_context", return_value="mem"),
            mock.patch.object(planning, "plan_calibration_hint", return_value=""),
            mock.patch.object(planning, "rhythm_diagnostics", return_value={}),
            mock.patch.object(planning, "recent_quality_feedback", return_value=[]),
            mock.patch.object(planning, "used_element_ledger", return_value={}),
            mock.patch.object(planning, "chapter_schedule_directive", return_value=""),
            mock.patch.object(planning, "cacheable_prefix", return_value="pfx"),
            mock.patch.object(planning, "db_event", return_value=None),
            mock.patch.object(planning, "log", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, replies, config_novel=None):
        """Drive arbitrate_plan with a canned sequence of raw LLM replies."""
        def fake_call_llm(client, paths, config, system, user, **kw):
            self.calls.append(kw.get("tag", "?"))
            return replies[min(len(self.calls) - 1, len(replies) - 1)]

        cfg = {"novel": {"telemetry_enabled": False,
                         "used_element_ledger_enabled": False,
                         **(config_novel or {})}, "api": {}}
        with mock.patch.object(planning, "call_llm", side_effect=fake_call_llm), \
             mock.patch.object(planning, "load_json_with_repair",
                               side_effect=lambda c, p, cf, raw, fallback=None:
                                   json.loads(raw) if raw.strip() else (fallback or {})):
            return planning.arbitrate_plan(None, mock.MagicMock(), None, cfg, 7,
                                           [CANDIDATE], [[]])

    def test_a_good_arbitration_costs_exactly_one_call(self):
        good = json.dumps({"selected_index": 0, "scores": [{"index": 0, "score": 8.0}],
                           "merged_plan": {"title": "改写后"}})
        plan, decision = self._run([good])
        self.assertEqual(self.calls, ["plan_arbitrate"])
        self.assertEqual(plan["title"], "改写后")
        self.assertNotIn("arbitration_failed", decision)

    def test_salvage_debris_triggers_exactly_one_reask(self):
        recovered = json.dumps({"selected_index": 0,
                                "scores": [{"index": 0, "score": 7.5}],
                                "merged_plan": {"title": "重问后"}})
        plan, decision = self._run([json.dumps({"./output.json": "{"}), recovered])
        self.assertEqual(self.calls, ["plan_arbitrate", "plan_arbitrate_reask"])
        self.assertEqual(plan["title"], "重问后")
        self.assertEqual(plan_score(decision), 7.5)
        self.assertNotIn("arbitration_failed", decision)

    def test_a_failed_reask_is_recorded_not_retried_further(self):
        # The recovery is bounded: one re-ask, then the plan proceeds UNARBITRATED
        # and says so. Re-rolling candidates cannot fix a parse failure.
        debris = json.dumps({"./output.json": "{"})
        plan, decision = self._run([debris, debris])
        self.assertEqual(self.calls, ["plan_arbitrate", "plan_arbitrate_reask"])
        self.assertTrue(decision["arbitration_failed"])
        self.assertEqual(plan, CANDIDATE, "must fall back to a real candidate plan")

    def test_reask_can_be_disabled_and_then_costs_nothing(self):
        debris = json.dumps({"./output.json": "{"})
        _, decision = self._run([debris], {"arbiter_reask_enabled": False})
        self.assertEqual(self.calls, ["plan_arbitrate"])
        self.assertTrue(decision["arbitration_failed"])

    def test_a_reask_exception_never_wedges_planning(self):
        def boom(client, paths, config, system, user, **kw):
            self.calls.append(kw.get("tag", "?"))
            if kw.get("tag") == "plan_arbitrate_reask":
                raise RuntimeError("gateway 500")
            return json.dumps({"./output.json": "{"})

        cfg = {"novel": {"telemetry_enabled": False,
                         "used_element_ledger_enabled": False}, "api": {}}
        with mock.patch.object(planning, "call_llm", side_effect=boom), \
             mock.patch.object(planning, "load_json_with_repair",
                               side_effect=lambda c, p, cf, raw, fallback=None:
                                   json.loads(raw) if raw.strip() else (fallback or {})):
            plan, decision = planning.arbitrate_plan(None, mock.MagicMock(), None,
                                                     cfg, 7, [CANDIDATE], [[]])
        self.assertTrue(decision["arbitration_failed"])
        self.assertEqual(plan, CANDIDATE)


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
