"""v2 repair: the outer guard that judges a fix by the release rule.

The fixers are `fix.py`'s and are tested there. What is tested here is the
thing v1 has no place to put: a repair kept or reverted according to
`accept.block_reasons` — the same ruler that decides whether the chapter ships.
The failure this guards is not a crash but a trade: a fossil rotation that
dodges its own gate and lands on an adjacent-repeat passes every inner check,
because every inner check is scored on the metric the fixer was aiming at.
"""
import unittest
from unittest import mock

import fix
from v2 import repair


def _config(**over) -> dict:
    novel = {"style_penalty_block": 2.0, "constraint_violation_block_count": 3,
             "fix_l0_enabled": True, "fix_ladder_enabled": True,
             "fix_max_l1_calls": 2}
    novel.update(over)
    return {"novel": novel, "api": {}}


def _l0_report(**over) -> dict:
    """A report whose fired gates route to L0 (style prose + fossil rotation)."""
    report = {
        "chapter": 7,
        "style_health": {"penalty": 2.4, "flags": ["em_dash_bad"]},
        "cross_chapter_repetition": {"level": "reject", "phrases": ["声音压得很低"]},
        "gate_rejects": [{"gate": "cross_chapter_repetition", "level": "reject"}],
    }
    report.update(over)
    return report


def _fossil_report(**over) -> dict:
    """Exactly ONE blocking reason, repairable by L0 rotation."""
    report = {
        "chapter": 7,
        "cross_chapter_repetition": {"level": "reject", "phrases": ["声音压得很低"]},
        "gate_rejects": [{"gate": "cross_chapter_repetition", "level": "reject"}],
    }
    report.update(over)
    return report


def _l1_report(**over) -> dict:
    """A report whose only fired gate routes to L1 (expand to band)."""
    report = {"chapter": 7, "length_band": {"block": True, "flag": "short"}}
    report.update(over)
    return report


def _clean() -> dict:
    return {"chapter": 7, "gate_rejects": []}


class ReasonKindTest(unittest.TestCase):

    def test_evidence_is_stripped_so_a_shrinking_penalty_is_not_a_new_problem(self):
        self.assertEqual(repair.reason_kind("style_collapse(penalty=2.3)"),
                         "style_collapse")
        self.assertEqual(repair.reason_kind("gate_rejects=a,b"), "gate_rejects")
        self.assertEqual(repair.reason_kind("hard_contract=3"), "hard_contract")
        self.assertEqual(repair.reason_kind("adjacent_repeat_block"),
                         "adjacent_repeat_block")

    def test_junk_degrades_to_itself_rather_than_to_empty(self):
        # An empty kind would collide with every other empty kind and make two
        # unrelated reasons compare equal.
        self.assertEqual(repair.reason_kind(""), "")
        self.assertEqual(repair.reason_kind("42"), "42")


class PendingTest(unittest.TestCase):

    def test_the_layers_are_read_off_the_registry_not_hardcoded(self):
        self.assertIn("style_prose", repair.pending(_l0_report(), _config(), "L0"))
        self.assertIn("fossil_rotate", repair.pending(_l0_report(), _config(), "L0"))
        self.assertEqual(repair.pending(_l1_report(), _config(), "L0"), ())
        self.assertIn("expand_to_band", repair.pending(_l1_report(), _config(), "L1"))

    def test_a_clean_report_has_nothing_pending(self):
        for layer in repair.LAYERS:
            self.assertEqual(repair.pending(_clean(), _config(), layer), ())

    def test_junk_does_not_crash_the_predicate(self):
        self.assertEqual(repair.pending(None, _config(), "L0"), ())
        self.assertEqual(repair.pending({}, _config(), "L0"), ())


class RunLayerTest(unittest.TestCase):

    def _run(self, layer="L0", *, report=None, config=None, recheck=None, **kw):
        return repair.run_layer(
            layer, text="原文", report=report if report is not None else _l0_report(),
            config=config or _config(), chapter_num=7,
            recheck=recheck or (lambda t: _clean()), **kw)

    def test_nothing_pending_means_nothing_happens(self):
        out = self._run(report=_clean())
        self.assertFalse(out.changed)
        self.assertEqual(out.text, "原文")
        self.assertTrue(out.cleared)

    def test_a_kept_repair_carries_the_new_text_and_the_new_report(self):
        with mock.patch.object(fix, "apply_l0", return_value=("修好的文", ["em_dash_reduce"])):
            out = self._run()
        self.assertTrue(out.changed)
        self.assertEqual(out.text, "修好的文")
        self.assertEqual(out.applied, ("em_dash_reduce",))
        self.assertEqual(out.layers, ("L0",))
        self.assertTrue(out.cleared)
        self.assertTrue(out.improved)

    def test_a_fixer_that_changed_nothing_does_not_buy_a_recheck(self):
        calls = []
        with mock.patch.object(fix, "apply_l0", return_value=("原文", [])):
            out = self._run(recheck=lambda t: calls.append(t) or _clean())
        self.assertEqual(calls, [])
        self.assertFalse(out.changed)

    def test_a_one_for_one_swap_is_reverted_whole(self):
        # The case the inner guards cannot see: rotation clears its own gate and
        # lands on an adjacent-repeat, which `style_health` never looks at. One
        # block before, one block after — the fixer's own metric improved and
        # the draft is no closer to shipping.
        after = {"chapter": 7, "adjacent_repetition": {"level": "block"}}
        with mock.patch.object(fix, "apply_l0", return_value=("换过词的文", ["fossil_rotate(3)"])):
            out = self._run(report=_fossil_report(), recheck=lambda t: after)
        self.assertEqual(out.text, "原文")
        self.assertEqual(out.applied, ())
        self.assertEqual(out.reverted, ("fossil_rotate(3)",))

    def test_strictly_fewer_blocks_is_kept_even_when_one_is_new(self):
        # Two down to one is progress by the release rule's own count, and the
        # release rule is the only ruler v2 has here. Refusing it because the
        # survivor has a new name would be scoring the repair on something else.
        after = {"chapter": 7, "adjacent_repetition": {"level": "block"}}
        with mock.patch.object(fix, "apply_l0", return_value=("换过词的文", ["fossil_rotate(3)"])):
            out = self._run(recheck=lambda t: after)  # _l0_report has 2 blocks
        self.assertEqual(out.text, "换过词的文")
        self.assertEqual(out.reverted, ())
        self.assertTrue(out.improved)

    def test_the_original_report_object_survives_a_revert(self):
        # `run_layer` must hand back the ORIGINAL report object on revert, not a
        # copy: the caller keeps using it, and a silently-substituted
        # equal-but-different dict is how a stale gate result survives a revert.
        self._last_report = _fossil_report()
        after = {"chapter": 7, "adjacent_repetition": {"level": "block"}}
        with mock.patch.object(fix, "apply_l0", return_value=("换过词的文", ["fossil_rotate(3)"])):
            out = repair.run_layer("L0", text="原文", report=self._last_report,
                                   config=_config(), chapter_num=7,
                                   recheck=lambda t: after)
        self.assertIs(out.report, self._last_report)

    def test_more_reasons_of_the_same_kinds_is_also_a_revert(self):
        before = _l0_report(style_health={"penalty": 2.4, "flags": ["em_dash_bad"]})
        after = {"chapter": 7,
                 "gate_rejects": [{"gate": "cross_chapter_repetition"}],
                 "style_health": {"penalty": 2.4},
                 "length_band": {"block": True}}
        with mock.patch.object(fix, "apply_l0", return_value=("文", ["merge_fragment_lines"])):
            out = repair.run_layer("L0", text="原文", report=before,
                                   config=_config(), chapter_num=7,
                                   recheck=lambda t: after)
        self.assertEqual(out.text, "原文")
        self.assertTrue(out.reverted)

    def test_the_same_problem_less_severe_is_kept_not_reverted(self):
        # `style_collapse(penalty=2.4)` -> `style_collapse(penalty=2.1)` is one
        # reason before and one after. Comparing raw strings would call that a
        # different problem and throw away a real improvement.
        before = {"chapter": 7, "style_health": {"penalty": 2.4, "flags": ["em_dash_bad"]}}
        after = {"chapter": 7, "style_health": {"penalty": 2.1}}
        with mock.patch.object(fix, "apply_l0", return_value=("文", ["em_dash_reduce"])):
            out = repair.run_layer("L0", text="原文", report=before,
                                   config=_config(), chapter_num=7,
                                   recheck=lambda t: after)
        self.assertEqual(out.text, "文")
        self.assertEqual(out.reverted, ())
        self.assertFalse(out.cleared)
        self.assertFalse(out.improved)

    def test_a_crashing_fixer_costs_nothing_and_keeps_the_chapter(self):
        with mock.patch.object(fix, "apply_l0", side_effect=RuntimeError("boom")):
            out = self._run()
        self.assertEqual(out.text, "原文")
        self.assertEqual(out.applied, ())

    def test_a_crashing_recheck_reverts_rather_than_shipping_unjudged_text(self):
        def boom(_):
            raise RuntimeError("gate exploded")
        with mock.patch.object(fix, "apply_l0", return_value=("文", ["em_dash_reduce"])):
            out = self._run(recheck=boom)
        self.assertEqual(out.text, "原文")
        self.assertEqual(out.reverted, ("em_dash_reduce",))

    def test_l1_without_a_client_is_reported_not_silently_skipped(self):
        # Otherwise "L1 was never wired through" and "L1 had nothing to do" look
        # identical, and the A/B's call count is quietly wrong.
        out = self._run("L1", report=_l1_report())
        self.assertEqual(out.skipped, ("expand_to_band",))
        self.assertEqual(out.applied, ())

    def test_a_layer_that_ran_and_kept_nothing_says_so(self):
        """A layer that RAN and kept nothing is a different event from a layer
        with nothing to run — and for L1 the difference is a paid call.

        Both used to print `blocks N->N` and nothing else, which is how the waste
        stayed invisible: across the settlement A/B's two v2 arms, 13 `fix_expand`
        calls produced only 7 `v2_repair_l1` events, so 6 paid calls left no
        record of what they bought.
        """
        lines: list[str] = []
        with mock.patch.object(repair, "_log", lambda p, m: lines.append(m)):
            with mock.patch.object(fix, "apply_l1", return_value=("原文", [])):
                ran = self._run("L1", report=_l1_report(), client=object())
            idle = self._run("L1", report=_clean(), client=object())

        self.assertFalse(ran.changed)
        self.assertIn("L1 kept nothing from [expand_to_band]", "\n".join(lines))
        # The line is only worth having if the OTHER case stays quiet.
        self.assertFalse(idle.changed)
        self.assertEqual(len([l for l in lines if "kept nothing" in l]), 1)

    def test_l1_runs_with_a_client_and_is_judged_the_same_way(self):
        with mock.patch.object(fix, "apply_l1", return_value=("扩写后的文", ["expand_to_band"])) as f:
            out = self._run("L1", report=_l1_report(), client=object())
        self.assertEqual(out.text, "扩写后的文")
        self.assertEqual(out.layers, ("L1",))
        self.assertEqual(f.call_count, 1)


if __name__ == "__main__":
    unittest.main()
