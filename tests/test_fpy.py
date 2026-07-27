"""Unit tests for novel._fpy_stats (First-Pass Yield, the north-star metric).

FPY is derived from checkpoint artifacts rather than llm_calls.jsonl because
the call log has no chapter field — an earlier `write == 1` proxy misreported
huangliang as 0% when its real FPY is 58% (candidate_chapters: 2 makes two
write calls structural, not rework). These tests pin the checkpoint-artifact
contract so that regression can't come back.
"""
import json
import tempfile
import unittest
from pathlib import Path

from novel import _fpy_stats, _reasoning_coverage


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
