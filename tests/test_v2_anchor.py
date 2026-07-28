"""The blinded pairwise judge: position debiasing, isolation, anchor honesty.

Every test here injects a fake `call`, so the arithmetic that decides a WR is
covered without an API key. That is the point of `judge_pair(call=...)`: the part
most likely to be silently wrong is the position-bias bookkeeping, and a
component you can only test by spending money does not get tested.
"""
import dataclasses
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from v2 import anchor


def _reply(winner, reason="r"):
    return '{"winner": "%s", "reason": "%s"}' % (winner, reason)


class _Judge:
    """A scripted judge. `script` is consumed one reply per call."""

    def __init__(self, *replies):
        self.script = list(replies)
        self.seen = []

    def __call__(self, system, user):
        self.seen.append((system, user))
        return self.script.pop(0) if self.script else _reply("tie")


class ConsistentJudge:
    """Always prefers whichever side contains `marker`, wherever it sits."""

    def __init__(self, marker):
        self.marker = marker
        self.calls = 0

    def __call__(self, system, user):
        self.calls += 1
        first = user.split("────────────────────")[0]
        return _reply(anchor.SIDE_FIRST if self.marker in first else anchor.SIDE_SECOND)


class PositionJudge:
    """Always picks whatever is shown first. Pure position bias, no reading."""

    def __call__(self, system, user):
        return _reply(anchor.SIDE_FIRST)


class ParseVerdictTest(unittest.TestCase):

    def test_maps_through_the_side_assignment(self):
        self.assertEqual(anchor.parse_verdict(_reply("甲"), ("a", "b"))[0], "a")
        self.assertEqual(anchor.parse_verdict(_reply("甲"), ("b", "a"))[0], "b")
        self.assertEqual(anchor.parse_verdict(_reply("乙"), ("b", "a"))[0], "a")

    def test_an_unreadable_reply_is_unmeasured_not_a_tie(self):
        # A tie is evidence of similarity and `tally` pays it half a win. A reply
        # nobody could read is no evidence at all, and calling it a tie is how six
        # dead gateway calls once reported WR=50.0% interpretable=True.
        for raw in ("", "I prefer the first one, obviously.", "{broken", None):
            with self.subTest(raw=raw):
                self.assertEqual(anchor.parse_verdict(raw, ("a", "b"))[0],
                                 anchor.UNMEASURED)

    def test_a_dict_without_a_winner_field_is_unmeasured(self):
        self.assertEqual(
            anchor.parse_verdict('{"reason": "两章都不错"}', ("a", "b"))[0],
            anchor.UNMEASURED)

    def test_prose_around_the_json_is_tolerated(self):
        raw = '好的，我的判断是：\n{"winner": "乙", "reason": "钩子更具体"}\n以上。'
        arm, reason = anchor.parse_verdict(raw, ("a", "b"))
        self.assertEqual(arm, "b")
        self.assertEqual(reason, "钩子更具体")

    def test_an_answered_but_even_call_is_a_real_tie(self):
        # The judge read both chapters and declined to separate them. That IS
        # evidence, and it must stay in the denominator.
        self.assertEqual(anchor.parse_verdict(_reply("平"), ("a", "b"))[0], "tie")

    def test_unknown_winner_token_is_a_tie(self):
        self.assertEqual(anchor.parse_verdict(_reply("丙"), ("a", "b"))[0], "tie")


class JudgePairTest(unittest.TestCase):

    def test_always_judges_both_orders(self):
        j = _Judge(_reply("甲"), _reply("乙"))
        anchor.judge_pair("A文本", "B文本", call=j)
        self.assertEqual(len(j.seen), 2, "a one-order verdict is a different measurement")

    def test_a_consistent_preference_survives_the_swap(self):
        j = ConsistentJudge("独有标记")
        v = anchor.judge_pair("这是甲，含独有标记。", "这是乙。", call=j)
        self.assertEqual(v.winner, "a")
        self.assertFalse(v.flipped)
        self.assertTrue(v.decisive)

    def test_pure_position_bias_is_recorded_as_a_tie(self):
        v = anchor.judge_pair("A", "B", call=PositionJudge())
        self.assertEqual(v.winner, "tie")
        self.assertTrue(v.flipped, "and it must be visible as a flip, not a quiet tie")

    def test_an_honest_double_tie_is_not_a_flip(self):
        v = anchor.judge_pair("A", "B", call=_Judge(_reply("tie"), _reply("tie")))
        self.assertEqual(v.winner, "tie")
        self.assertFalse(v.flipped)

    def test_a_dead_judge_never_awards_a_win(self):
        def boom(system, user):
            raise RuntimeError("gateway 500")
        v = anchor.judge_pair("A", "B", call=boom)
        self.assertEqual(v.winner, anchor.UNMEASURED)
        self.assertFalse(v.measured)
        self.assertIn("judge call failed", v.reasons[0])

    def test_one_dead_order_unmeasures_the_whole_pair(self):
        # Half a measurement is not a measurement: with one order missing the
        # `picks[0] == picks[1]` test can only ever say "tie", so keeping the pair
        # would bank a manufactured tie at half a win.
        calls = {"n": 0}

        def half_dead(system, user):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("gateway 500")
            return _reply("甲")

        v = anchor.judge_pair("A文本", "B文本", call=half_dead)
        self.assertFalse(v.measured)
        self.assertEqual(v.winner, anchor.UNMEASURED)
        self.assertFalse(v.flipped, "a missing order is not position bias")

    def test_missing_chapter_text_is_unmeasured(self):
        v = anchor.judge_pair("", "B文本", call=_Judge(_reply("甲")))
        self.assertFalse(v.measured)

    def test_missing_text_costs_no_calls(self):
        j = _Judge()
        v = anchor.judge_pair("", "B", call=j)
        self.assertEqual(v.winner, anchor.UNMEASURED)
        self.assertEqual(j.seen, [])

    def test_the_arms_are_never_named_to_the_judge(self):
        j = _Judge(_reply("甲"), _reply("甲"))
        anchor.judge_pair("A文本", "B文本", call=j, key="ts_v2arm~ch7")
        blob = "\n".join(s + u for s, u in j.seen)
        for leak in ("ts_v2arm", "ch7", "engine", "v1", "v2", '"a"', '"b"'):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, blob)

    def test_overlong_text_is_truncated_not_sent_whole(self):
        j = _Judge(_reply("tie"), _reply("tie"))
        anchor.judge_pair("甲" * 50000, "乙" * 50000, call=j)
        self.assertLess(len(j.seen[0][1]), anchor.MAX_CHAPTER_CHARS * 2 + 2000)


class TallyTest(unittest.TestCase):

    def _v(self, winner, orders, key="1"):
        return anchor.Verdict(key, winner, orders, ("", ""))

    def test_ties_count_half_for_the_stated_arm(self):
        vs = [self._v("b", ("b", "b")), self._v("tie", ("tie", "tie"))]
        self.assertEqual(anchor.tally(vs, arm="b")["win_rate"], 75.0)
        self.assertEqual(anchor.tally(vs, arm="a")["win_rate"], 25.0)

    def test_a_mostly_flipped_run_is_marked_uninterpretable(self):
        vs = [self._v("tie", ("a", "b")), self._v("tie", ("b", "a")),
              self._v("b", ("b", "b"))]
        t = anchor.tally(vs)
        self.assertEqual(t["flips"], 2)
        self.assertFalse(t["interpretable"],
                         "a run that mostly measured position preference must say so")
        # And the win rate it would print looks perfectly respectable:
        self.assertGreaterEqual(t["win_rate"], 50.0)

    def test_a_clean_run_is_interpretable(self):
        vs = [self._v("b", ("b", "b")), self._v("a", ("a", "a")),
              self._v("tie", ("a", "b"))]
        self.assertTrue(anchor.tally(vs)["interpretable"])

    def test_raw_votes_are_reported_beside_the_deduped_verdict(self):
        vs = [self._v("tie", ("a", "b")), self._v("a", ("a", "a"))]
        t = anchor.tally(vs)
        self.assertEqual(t["raw_votes"], {"a": 3, "b": 1, "total": 4})

    def test_empty_is_a_clean_zero_not_a_crash(self):
        t = anchor.tally([])
        self.assertEqual(t["n"], 0)
        self.assertEqual(t["win_rate"], 0.0)
        self.assertFalse(t["interpretable"])

    def test_an_all_dead_run_reports_no_measurement_not_a_dead_heat(self):
        # The regression this exists for: judging ts_v1arm vs ts_v2arm with a dead
        # gateway printed `WR=50.0% interpretable=True` off zero answered calls.
        # Read literally, that says the redesign changed nothing.
        vs = [self._v(anchor.UNMEASURED, (anchor.UNMEASURED,) * 2, key=str(i))
              for i in range(2)]
        t = anchor.tally(vs)
        self.assertEqual(t["n"], 0)
        self.assertEqual(t["unmeasured"], 2)
        self.assertEqual(t["ties"], 0, "an unanswered pair is not a tie")
        self.assertFalse(t["interpretable"])

    def test_unmeasured_pairs_do_not_dilute_the_win_rate(self):
        # Two clean wins for b plus two failures is a 100% win rate over n=2,
        # not 75% over n=4.
        vs = [self._v("b", ("b", "b")), self._v("b", ("b", "b")),
              self._v(anchor.UNMEASURED, (anchor.UNMEASURED,) * 2),
              self._v(anchor.UNMEASURED, ("a", anchor.UNMEASURED))]
        t = anchor.tally(vs, arm="b")
        self.assertEqual(t["n"], 2)
        self.assertEqual(t["unmeasured"], 2)
        self.assertEqual(t["win_rate"], 100.0)
        self.assertEqual(t["raw_votes"], {"a": 0, "b": 4, "total": 4},
                         "a half-answered pair contributes no votes either")

    def test_a_partial_run_is_uninterpretable_even_when_the_rest_is_clean(self):
        # The surviving half of a run is a different measurement from the one
        # requested, and the difference is invisible in the number itself.
        vs = [self._v("b", ("b", "b")), self._v("a", ("a", "a")),
              self._v(anchor.UNMEASURED, (anchor.UNMEASURED,) * 2)]
        t = anchor.tally(vs)
        self.assertEqual(t["flips"], 0)
        self.assertEqual(t["unmeasured"], 1)
        self.assertFalse(t["interpretable"])


class FlipDirectionTest(unittest.TestCase):
    """A flip's DIRECTION separates the two causes the win rate cannot.

    `flipped` says the judge contradicted itself on side-swap; it does not say
    why. `flip_side` does: all-one-side means the judge answered by position (no
    resolving power on those pairs), split means the prose really was close. The
    handling is the same — count as tie — but the interpretation is not, and the
    v1/v2 matched settlement is the case that needed the distinction: 25 of 30
    pairs flipped and ALL 25 went to the first position, while `null_pair_probe`
    on identical text had reported a 0% first-position rate. Identical text gives
    the judge nothing to rationalize, so a passing probe cannot clear a run.
    """

    def _v(self, orders, winner="tie"):
        return anchor.Verdict("1", winner, orders, ("", ""))

    def test_flip_side_names_the_position_the_judge_picked(self):
        self.assertEqual(self._v(("a", "b")).flip_side, "first")
        self.assertEqual(self._v(("b", "a")).flip_side, "second")

    def test_a_non_flip_has_no_side(self):
        self.assertEqual(self._v(("b", "b"), winner="b").flip_side, "")
        self.assertEqual(self._v(("tie", "tie")).flip_side, "")
        self.assertEqual(
            self._v((anchor.UNMEASURED,) * 2, winner=anchor.UNMEASURED).flip_side, "")

    def test_one_sided_flips_are_reported_as_a_position_preference(self):
        vs = [self._v(("a", "b")) for _ in range(9)] + [self._v(("b", "b"), "b")]
        t = anchor.tally(vs, arm="b")
        self.assertEqual((t["flips_first_position"], t["flips_second_position"]), (9, 0))
        self.assertEqual(t["flip_bias"], 1.0)

    def test_split_flips_are_not_a_position_preference(self):
        vs = [self._v(("a", "b")) for _ in range(5)] + [self._v(("b", "a")) for _ in range(5)]
        t = anchor.tally(vs, arm="b")
        self.assertEqual((t["flips_first_position"], t["flips_second_position"]), (5, 5))
        self.assertEqual(t["flip_bias"], 0.5)

    def test_n_decisive_is_the_sample_size_that_carries_the_verdict(self):
        # The shape of the real settlement: n=30 looks like plenty, and 25 of those
        # pairs contributed nothing but a folded-in tie.
        vs = ([self._v(("a", "b")) for _ in range(25)]
              + [self._v(("b", "b"), "b") for _ in range(5)])
        t = anchor.tally(vs, arm="b")
        self.assertEqual(t["n"], 30)
        self.assertEqual(t["n_decisive"], 5)
        self.assertEqual(t["win_rate"], (5 + 25 * 0.5) / 30 * 100.0)
        self.assertFalse(t["interpretable"])

    def test_flip_bias_is_zero_when_nothing_flipped(self):
        t = anchor.tally([self._v(("b", "b"), "b")], arm="b")
        self.assertEqual(t["flip_bias"], 0.0)
        self.assertEqual(t["n_decisive"], 1)


class RubricPremiseTest(unittest.TestCase):
    """Two premises, one scoring rubric."""


    def _criteria(self, system):
        return system.split("只回答一个问题")[1]

    def test_both_premises_share_the_scoring_criteria_verbatim(self):
        # The premise may vary with what the texts actually are; the ruler may
        # not. Two arms judged by two rubrics are not comparable.
        self.assertEqual(self._criteria(anchor.JUDGE_SYSTEM),
                         self._criteria(anchor.JUDGE_SYSTEM_UNMATCHED))

    def test_the_default_system_still_claims_a_matched_pair(self):
        self.assertIn(anchor.PREMISE_MATCHED, anchor.JUDGE_SYSTEM)
        self.assertIn("同一章号", anchor.JUDGE_SYSTEM)
        self.assertNotIn("剧情位置不同", anchor.JUDGE_SYSTEM)

    def test_the_unmatched_premise_forbids_scoring_plot_position(self):
        # Different chapters mean one side may sit nearer a climax. Unsaid, the
        # judge scores the outline instead of the writing.
        self.assertIn("剧情位置不同", anchor.JUDGE_SYSTEM_UNMATCHED)
        self.assertNotIn("同一章号", anchor.JUDGE_SYSTEM_UNMATCHED)

    def test_judge_pair_sends_the_matched_premise_unless_told_otherwise(self):
        j = _Judge(_reply("tie"), _reply("tie"))
        anchor.judge_pair("A文本", "B文本", call=j)
        self.assertEqual(j.seen[0][0], anchor.JUDGE_SYSTEM)

    def test_the_premise_reaches_the_judge_when_overridden(self):
        j = _Judge(_reply("tie"), _reply("tie"))
        anchor.judge_series([("1", "A文本", "B文本")], call=j,
                            system=anchor.JUDGE_SYSTEM_UNMATCHED)
        self.assertEqual({s for s, _ in j.seen}, {anchor.JUDGE_SYSTEM_UNMATCHED},
                         "both orders must be judged under the same premise")


class NullPairProbeTest(unittest.TestCase):
    """The calibration that tells a biased judge from indistinguishable arms.

    Measured 2026-07-28 on the v1/v2 settlement: 23 of 30 real pairs flipped on
    side-swap, which `tally` reports as `interpretable=False` because a judge
    measuring position looks exactly like that. The null probe separates the two
    causes — deepseek-v4-pro tied identical text 10/10 under both premises, so it
    CAN answer tie, and the flips are the arms being hard to separate rather than
    the instrument being broken.
    """

    def test_an_unbiased_judge_ties_every_null_pair(self):
        r = anchor.null_pair_probe([("1", "text one"), ("2", "text two")],
                                   call=_Judge())      # scripted default is tie
        self.assertEqual((r["pairs"], r["calls"], r["ties"]), (2, 4, 4))
        self.assertEqual(r["first_position"], 0)
        self.assertEqual(r["first_position_rate"], 0.0)
        self.assertTrue(r["usable"])

    def test_a_position_judge_is_caught_even_though_judge_pair_says_tie(self):
        # The bias hides in `Verdict.winner`: picking 甲 in both orders folds into
        # `tie`. Counting winners would score a pure position judge as perfectly
        # calibrated, which is the whole reason this counts CALLS.
        verdict = anchor.judge_pair("same", "same", call=PositionJudge())
        self.assertEqual(verdict.winner, "tie")

        r = anchor.null_pair_probe([("1", "same")], call=PositionJudge())
        self.assertEqual(r["first_position"], 2)
        self.assertEqual(r["ties"], 0)
        self.assertEqual(r["first_position_rate"], 1.0)
        self.assertFalse(r["usable"])

    def test_a_dead_judge_is_unmeasured_and_not_usable(self):
        class Dead:
            def __call__(self, system, user):
                raise RuntimeError("gateway down")

        r = anchor.null_pair_probe([("1", "t")], call=Dead())
        self.assertEqual(r["unmeasured"], 2)
        self.assertEqual(r["ties"], 0)
        self.assertFalse(r["usable"], "no answers is not a calibrated judge")

    def test_the_probe_uses_the_rubric_it_is_handed(self):
        j = _Judge()
        anchor.null_pair_probe([("1", "t")], call=j,
                               system=anchor.JUDGE_SYSTEM_UNMATCHED)
        self.assertTrue(all(s == anchor.JUDGE_SYSTEM_UNMATCHED for s, _ in j.seen),
                        "probing a premise the experiment did not use measures "
                        "the bias of a prompt nobody ran")

    def test_both_texts_sent_are_identical(self):
        # If the probe ever sent two different texts it would stop being a null
        # pair and its known-correct answer would be gone.
        j = _Judge()
        anchor.null_pair_probe([("1", "the chapter body")], call=j)
        for _, user in j.seen:
            halves = [h.strip() for h in user.split("────────────────────") if h.strip()]
            self.assertGreaterEqual(len(halves), 2)
            self.assertIn("the chapter body", halves[0])
            self.assertIn("the chapter body", halves[1])


class IsolationTest(unittest.TestCase):
    """The judge must never bill the novel it is measuring."""

    @dataclasses.dataclass
    class FakePaths:
        logs_dir: Path
        book: Path = Path("book.md")

    def test_judge_paths_redirects_logs_dir(self):
        with TemporaryDirectory() as td:
            live = Path(td) / "novels" / "x" / "logs"
            live.mkdir(parents=True)
            p = anchor.judge_paths(self.FakePaths(live), Path(td))
            self.assertNotEqual(p.logs_dir, live)
            self.assertEqual(p.logs_dir.name, "pairwise_logs")
            self.assertTrue(p.logs_dir.is_dir())
            self.assertEqual(p.book, Path("book.md"), "other paths must survive")

    def test_llm_caller_refuses_a_live_novels_paths(self):
        with self.assertRaises(ValueError) as cm:
            anchor.llm_caller(None, self.FakePaths(Path("novels/tangshuting/logs")), {})
        self.assertIn("judge_paths", str(cm.exception))


class AnchorSetTest(unittest.TestCase):

    def test_missing_anchor_dir_is_reported_not_faked(self):
        with TemporaryDirectory() as td:
            got, why = anchor.anchor_chapters({}, Path(td))
            self.assertEqual(got, [])
            self.assertIn("UNMEASURED", why)

    def test_note_length_files_are_refused_as_anchors(self):
        # `benchmarks/` today holds notes ABOUT 爆款 structure, not prose. Feeding
        # one to a prose judge asks it to compare a chapter against an essay.
        with TemporaryDirectory() as td:
            d = Path(td) / anchor.DEFAULT_ANCHOR_DIR
            d.mkdir(parents=True)
            (d / "retention_patterns.md").write_text("开篇 300 字内必须给出反常。",
                                                     encoding="utf-8")
            got, why = anchor.anchor_chapters({}, Path(td))
            self.assertEqual(got, [])
            self.assertIn("retention_patterns.md", why)

    def test_a_real_chapter_is_accepted(self):
        with TemporaryDirectory() as td:
            d = Path(td) / anchor.DEFAULT_ANCHOR_DIR
            d.mkdir(parents=True)
            (d / "ref01.md").write_text("正文。" * 900, encoding="utf-8")
            got, why = anchor.anchor_chapters({}, Path(td))
            self.assertEqual([a.name for a in got], ["ref01"])
            self.assertEqual(why, "")

    def test_fingerprint_tracks_content_not_order(self):
        a = anchor.AnchorText("x", "内容一")
        b = anchor.AnchorText("y", "内容二")
        self.assertEqual(anchor.anchor_fingerprint([a, b]),
                         anchor.anchor_fingerprint([b, a]))
        self.assertNotEqual(anchor.anchor_fingerprint([a, b]),
                            anchor.anchor_fingerprint([a, anchor.AnchorText("y", "改了")]))

    def test_wr_without_an_anchor_set_is_unavailable_not_zero(self):
        # The failure mode this forbids: printing "WR 0%" (or worse, silently
        # substituting arm-vs-arm) when nobody has supplied a human reference.
        with TemporaryDirectory() as td:
            r = anchor.wr_against_anchor([("ch1", "文本")], call=_Judge(), root=Path(td))
            self.assertFalse(r["available"])
            self.assertNotIn("win_rate", r)
            self.assertIn("reason", r)

    def test_wr_states_the_headline_on_the_engine_and_pins_the_anchor(self):
        with TemporaryDirectory() as td:
            d = Path(td) / anchor.DEFAULT_ANCHOR_DIR
            d.mkdir(parents=True)
            (d / "ref01.md").write_text("人类参考章。" * 400, encoding="utf-8")
            r = anchor.wr_against_anchor(
                [("ch1", "引擎写的独有标记章节。" * 100)],
                call=ConsistentJudge("独有标记"), root=Path(td))
            self.assertTrue(r["available"])
            self.assertEqual(r["arm"], "a", "WR is stated on the engine, not the anchor")
            self.assertEqual(r["win_rate"], 100.0)
            self.assertEqual(r["anchors"], ["ref01"])
            self.assertTrue(r["anchor_fingerprint"])


class OneRubricTest(unittest.TestCase):

    def test_pairwise_ab_uses_the_resident_rubric(self):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "_pab", root / "tools" / "pairwise_ab.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIs(mod.anchor.JUDGE_SYSTEM, anchor.JUDGE_SYSTEM)
        src = (root / "tools" / "pairwise_ab.py").read_text(encoding="utf-8")
        self.assertNotIn("你是一位挑剔的网文读者", src,
                         "a second copy of the rubric makes two experiments incomparable")


if __name__ == "__main__":
    unittest.main()
