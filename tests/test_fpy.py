"""Unit tests for the two First-Pass Yield definitions.

`novel._fpy_stats` (FPY) counts a chapter dirty when any rework artifact exists.
It is derived from checkpoint artifacts rather than llm_calls.jsonl because the
call log has no chapter field — an earlier `write == 1` proxy misreported
huangliang as 0% when its real FPY is 58% (candidate_chapters: 2 makes two write
calls structural, not rework). These tests pin the checkpoint-artifact contract
so that regression can't come back.

`tools/fpy_prime.chapter_verdict` (FPY') asks the narrower question FPY cannot
answer — did the first draft carry a *measured* defect — because every rework
artifact FPY looks for is produced by a rule keyed on `quality_threshold`, so an
experiment that moves that rule moves FPY mechanically in both arms. The tests
below pin the one judgment FPY' makes on its own: which replan labels count.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from novel import _fpy_stats, _reasoning_coverage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import fpy_prime  # noqa: E402


def _chapter(root: Path, num: int, *, completed: bool = True, artifacts: tuple[str, ...] = ()) -> None:
    """Materialize one checkpoint dir with the baseline clean-run artifacts."""
    d = root / "logs" / "checkpoints" / f"ch{num:04d}"
    d.mkdir(parents=True, exist_ok=True)
    baseline = [
        "plan_initial_attempt0_candidates.json",
        "plan_initial_attempt0_reports.json",
        "plan_initial_attempt0_arbitration.json",
        "plan_initial_selected.json",
        "validated_plan.json",
        "chapter_current_v2.md",
        "review_round0.json",
        "final_review.json",
        "chapter_saved.json",
        "extraction.json",
    ]
    if completed:
        baseline.append("chapter_completed.json")
    for name in [*baseline, *artifacts]:
        (d / name).write_text("{}", encoding="utf-8")


class TestFpyStats(unittest.TestCase):
    def test_returns_none_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(_fpy_stats(Path(td)))

    def test_clean_chapters_are_first_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in (1, 2, 3):
                _chapter(root, i)
            got = _fpy_stats(root)
            self.assertEqual(got["total"], 3)
            self.assertEqual(got["clean"], 3)
            self.assertEqual(got["fpy"], 1.0)
            self.assertEqual(got["dirty_chapters"], [])

    def test_incomplete_chapters_are_excluded(self):
        """A half-written chapter is not a failed chapter — it's not a chapter yet."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _chapter(root, 1)
            _chapter(root, 2, completed=False)
            got = _fpy_stats(root)
            self.assertEqual(got["total"], 1)
            self.assertEqual(got["fpy"], 1.0)

    def test_each_rework_marker_disqualifies(self):
        cases = {
            "replan": "plan_quality_replan_attempt0_candidates.json",
            "plan_retry": "plan_initial_attempt1_candidates.json",
            "revise": "chapter_revised_round0.md",
            "re_review": "review_round1.json",
            "debt": "quality_debt.json",
            "hook_redo": "hook_revised.json",
        }
        for reason, artifact in cases.items():
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _chapter(root, 1, artifacts=(artifact,))
                got = _fpy_stats(root)
                self.assertEqual(got["clean"], 0, f"{artifact} should disqualify")
                self.assertEqual(got["reasons"][reason], 1)
                self.assertEqual(got["dirty_chapters"], [1])

    def test_review_round0_alone_is_clean(self):
        """round0 is the mandatory first review — only round1+ counts as rework."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _chapter(root, 1)
            self.assertEqual(_fpy_stats(root)["reasons"]["re_review"], 0)

    def test_overlapping_causes_counted_once_per_chapter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _chapter(root, 1, artifacts=("quality_debt.json", "chapter_revised_round0.md"))
            _chapter(root, 2)
            got = _fpy_stats(root)
            self.assertEqual(got["total"], 2)
            self.assertEqual(got["clean"], 1)
            self.assertEqual(got["fpy"], 0.5)
            self.assertEqual(got["reasons"]["debt"], 1)
            self.assertEqual(got["reasons"]["revise"], 1)
            self.assertEqual(got["dirty_chapters"], [1])


class TestFpyPrime(unittest.TestCase):
    """FPY' must judge only on evidence the release rule cannot manufacture."""

    def _dir(self, td: str, review: dict | None, *artifacts: str) -> Path:
        d = Path(td) / "ch0007"
        d.mkdir(parents=True, exist_ok=True)
        if review is not None:
            (d / "review_round0.json").write_text(
                json.dumps({"payload": review}), encoding="utf-8")
        for a in artifacts:
            (d / a).write_text("{}", encoding="utf-8")
        return d

    def test_clean_first_draft_passes_regardless_of_score(self):
        """A 6.8 with no measured defect is a pass — that is the entire point."""
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, {"score": 6.8, "accepted": True}))
            self.assertTrue(v["ok"])
            self.assertEqual(v["reasons"], [])
            self.assertEqual((v["ch"], v["score"]), (7, 6.8))

    def test_high_score_with_a_gate_reject_fails(self):
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(
                td, {"score": 9.0, "accepted": True,
                     "gate_rejects": [{"gate": "cross_chapter_repetition"}]}))
            self.assertFalse(v["ok"])
            self.assertIn("cross_chapter_repetition", v["reasons"][0])

    def test_pre_write_replans_count_as_failures(self):
        for label in ("plan_initial_attempt1_candidates.json",
                      "plan_critical_attempt0_candidates.json",
                      "plan_fossil_catastrophe_attempt0_candidates.json"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                v = fpy_prime.chapter_verdict(
                    self._dir(td, {"score": 9.0, "accepted": True}, label))
                self.assertFalse(v["ok"], label)
                self.assertEqual(len(v["replans"]), 1)

    def test_score_driven_replans_are_excluded(self):
        """`quality_replan`/`hard_floor` are consequences of the release rule.

        Counting them would put `quality_threshold` back into the criterion, which
        is the circularity FPY' exists to remove.
        """
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(
                td, {"score": 9.0, "accepted": True},
                "plan_quality_replan_attempt0_candidates.json",
                "plan_hard_floor_attempt0_candidates.json"))
            self.assertTrue(v["ok"])
            self.assertEqual(v["replans"], [])

    def test_attempt0_alone_is_not_a_retry(self):
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(
                td, {"score": 9.0, "accepted": True},
                "plan_initial_attempt0_candidates.json"))
            self.assertTrue(v["ok"])

    def test_missing_round0_review_is_na_not_a_pass(self):
        """Silence must not be counted as success in either direction."""
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, None))
            self.assertIsNone(v["ok"])
            self.assertTrue(v["missing"])
            self.assertEqual(v["reasons"], ["no_round0_review"])

    def test_bare_review_payload_is_accepted(self):
        """Older checkpoints stored the review dict without a `payload` wrapper."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ch0007"
            d.mkdir(parents=True)
            (d / "review_round0.json").write_text(
                json.dumps({"score": 6.1, "accepted": False}), encoding="utf-8")
            self.assertEqual(fpy_prime.chapter_verdict(d)["score"], 6.1)

    def test_superseded_backstop_severity_is_normalized(self):
        """A retro-replay must not report an already-fixed engine bug as live.

        review.py's contract backstop stamped keyword-matched `problems` text as a
        HARD violation until b54bfd0. 22 archived chapters fail on that alone.
        """
        review = {"score": 9.0, "accepted": True, "contract_violations": [
            {"severity": "hard", "rule": "能力白名单/模态（由 problems 文本回填）"}]}
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(fpy_prime.chapter_verdict(self._dir(td, review))["ok"])
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, review), raw=True)
            self.assertFalse(v["ok"])
            self.assertIn("hard_contract", v["reasons"][0])

    def test_a_registered_hard_contract_violation_still_fails(self):
        """Normalization is narrow: only the backstop's own synthesized rule."""
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, {
                "score": 9.0, "accepted": True, "contract_violations": [
                    {"severity": "hard", "rule": "能力白名单：只允许错字食谱暗码解密"}]}))
            self.assertFalse(v["ok"])

    def test_normalize_does_not_mutate_the_input(self):
        review = {"contract_violations": [
            {"severity": "hard", "rule": "x（由 problems 文本回填）"}]}
        fpy_prime._normalize(review)
        self.assertEqual(review["contract_violations"][0]["severity"], "hard")

    def test_thresholds_are_pinned_not_read_from_the_novel_config(self):
        """Two arms with different configs must still be judged by one ruler."""
        self.assertEqual(fpy_prime.PINNED["novel"]["style_penalty_block"], 2.0)
        with tempfile.TemporaryDirectory() as td:
            d = self._dir(td, {"score": 9.0, "accepted": True,
                               "style_health": {"penalty": 2.5}})
            self.assertFalse(fpy_prime.chapter_verdict(d)["ok"])

    # --- the second superseded semantic: the flat em-dash trend charge ---------
    #
    # tangshuting Ch6 verbatim. The trend term used to charge a flat +1.0, which
    # stacked onto the static tier's +1.0 to hit `style_penalty_block` (2.0)
    # exactly. It is now graduated by ratio, and 6.36/3.37 = 1.9x charges 0.3, so
    # the same chapter scores 1.3 and is not a collapse. Archived flags written
    # before the graduation carry no `pen=` field, which is how the old charge is
    # recognized.
    FLAT_TREND = {
        "score": 7.5, "accepted": False,
        "style_health": {
            "penalty": 2.0,
            "flags": ["em_dash_high(6.4/k≥6.0)",
                      "em_dash_trend_rise(6.4/k vs mean 3.4/k)"],
            "metrics": {"em_dash_per_kchar": 6.36, "em_dash_recent_mean": 3.37},
        },
    }

    def test_flat_trend_charge_is_restamped(self):
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, self.FLAT_TREND))
            self.assertTrue(v["ok"], v["reasons"])
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, self.FLAT_TREND), raw=True)
            self.assertFalse(v["ok"])
            self.assertIn("style_collapse", v["reasons"][0])

    def test_a_genuine_collapse_survives_the_restamp(self):
        """yeban_guize Ch9: 7.18/k vs a 2.20/k mean is a real 3.3x spike."""
        review = {"score": 5.2, "accepted": False, "style_health": {
            "penalty": 2.0,
            "flags": ["em_dash_high(7.2/k≥6.0)",
                      "em_dash_trend_rise(7.2/k vs mean 2.2/k, ratio=3.3x, pen=1.0)"],
            "metrics": {"em_dash_per_kchar": 7.18, "em_dash_recent_mean": 2.20}}}
        with tempfile.TemporaryDirectory() as td:
            v = fpy_prime.chapter_verdict(self._dir(td, review))
            self.assertFalse(v["ok"])
            self.assertIn("style_collapse", v["reasons"][0])

    def test_non_em_dash_penalty_components_are_preserved(self):
        """Only the em-dash terms are recomputed; the rest of the total stands.

        Their inputs are the chapter TEXT, which round0 no longer describes once
        the draft was revised — so subtracting the em-dash terms from the archived
        total is the only faithful adjustment available.
        """
        review = {"score": 7.0, "accepted": False, "style_health": {
            "penalty": 3.0,   # 1.0 static + 1.0 flat trend + 1.0 dialogue_starved
            "flags": ["em_dash_high(6.8/k≥6.0)",
                      "em_dash_trend_rise(6.8/k vs mean 2.9/k)",
                      "dialogue_starved(5.5%<8%)"],
            "metrics": {"em_dash_per_kchar": 6.79, "em_dash_recent_mean": 2.93}}}
        sh = fpy_prime._normalize(review)["style_health"]
        self.assertAlmostEqual(sh["penalty"], 2.5)   # trend 1.0 -> 0.5
        self.assertIn("dialogue_starved(5.5%<8%)", sh["flags"])
        with tempfile.TemporaryDirectory() as td:
            # tangshuting Ch37: still above the block line after the re-stamp.
            self.assertFalse(fpy_prime.chapter_verdict(self._dir(td, review))["ok"])

    def test_a_capped_penalty_is_left_alone(self):
        """At the cap the archived total is not the sum of its terms."""
        review = {"style_health": {
            "penalty": 4.0,
            "flags": ["em_dash_overload(13.0/k≥12.0)", "sentences_too_short(avg=9.0<12.0)",
                      "fragment_lines(0.5)", "almost_no_dialogue"],
            "metrics": {"em_dash_per_kchar": 13.0, "em_dash_recent_mean": 3.0}}}
        self.assertEqual(fpy_prime._normalize(review)["style_health"]["penalty"], 4.0)

    def test_missing_metrics_or_no_em_dash_flag_is_left_alone(self):
        for sh in ({"penalty": 2.5},                                  # no metrics
                   {"penalty": 2.5, "metrics": {}},                   # no density
                   {"penalty": 2.5, "flags": ["almost_no_dialogue"],  # no em-dash term
                    "metrics": {"em_dash_per_kchar": 1.0}}):
            with self.subTest(sh=sh):
                self.assertEqual(
                    fpy_prime._normalize({"style_health": sh})["style_health"], sh)

    def test_style_restamp_does_not_mutate_the_input(self):
        review = {**self.FLAT_TREND}
        before = json.dumps(review, sort_keys=True)
        fpy_prime._normalize(review)
        self.assertEqual(json.dumps(review, sort_keys=True), before)

    def test_both_normalizations_can_apply_to_one_chapter(self):
        review = {**self.FLAT_TREND, "contract_violations": [
            {"severity": "hard", "rule": "x（由 problems 文本回填）"}]}
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(fpy_prime.chapter_verdict(self._dir(td, review))["ok"])


class TestReasoningCoverage(unittest.TestCase):
    """Reasoning presence is a confounder, not a setting — see _reasoning_coverage."""

    @staticmethod
    def _log(root: Path, rows: list[dict]) -> None:
        p = root / "logs"
        p.mkdir(parents=True, exist_ok=True)
        (p / "llm_calls.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )

    def test_returns_none_without_log(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(_reasoning_coverage(Path(td)))

    def test_counts_only_successful_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._log(root, [
                {"tag": "write", "ok": True, "reasoning_chars": 6000},
                {"tag": "write", "ok": True},                              # absent field == none
                {"tag": "write", "ok": True, "reasoning_chars": 0},
                {"tag": "write", "ok": False, "reasoning_chars": 9000},    # failures excluded
            ])
            self.assertEqual(_reasoning_coverage(root), [("write", 3, 1)])

    def test_untracked_tags_are_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._log(root, [
                {"tag": "bootstrap_bible", "ok": True, "reasoning_chars": 500},
                {"tag": "write", "ok": True, "reasoning_chars": 500},
            ])
            self.assertEqual([t for t, _, _ in _reasoning_coverage(root)], ["write"])

    def test_none_when_no_tracked_tag_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._log(root, [{"tag": "bootstrap_voice", "ok": True}])
            self.assertIsNone(_reasoning_coverage(root))


if __name__ == "__main__":
    unittest.main()
