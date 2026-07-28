"""Tests for `tools.fpy_prime.discover_novels` — the corpus filter behind every
library-wide FPY' claim.

Why this is worth a test file: the aggregate it feeds is what settles engine A/Bs.
Measured on the real corpus, including derivative dirs reported the fossil +
chapter_mode gate fixes as +4.0pt when the clean corpus says +6.0pt, and it ranked
`style_collapse` as the #2 remaining killer (24) when 17 of those 24 sat in a
single excluded copy. A silent regression here does not crash anything; it just
makes the next experiment answer the wrong question.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.fpy_prime import discover_novels


def _novel(root: Path, name: str, chapters: dict[int, str]) -> None:
    """Create a novel dir with a checkpoints dir (the discovery precondition)."""
    d = root / "novels" / name
    (d / "logs" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (d / "chapters").mkdir(parents=True, exist_ok=True)
    for num, text in chapters.items():
        (d / "chapters" / f"{num:04d}.md").write_text(text, encoding="utf-8")


class DiscoverNovelsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "experiments").mkdir()
        self.addCleanup(self._tmp.cleanup)

    def names(self, explicit=(), **kw):
        kept, dropped = discover_novels(list(explicit), root=self.root, **kw)
        return kept, dict(dropped)

    def test_independent_books_are_all_kept(self):
        _novel(self.root, "a", {1: "A one"})
        _novel(self.root, "b", {1: "B one"})
        kept, dropped = self.names()
        self.assertEqual(kept, ["a", "b"])
        self.assertEqual(dropped, {})

    def test_ablate_name_convention_is_dropped(self):
        _novel(self.root, "a", {1: "A one"})
        _novel(self.root, "a__ablate_some_key", {1: "different text"})
        kept, dropped = self.names()
        self.assertEqual(kept, ["a"])
        self.assertIn("ablation", dropped["a__ablate_some_key"])

    def test_ablate_metadata_is_dropped_even_without_the_name(self):
        _novel(self.root, "a", {1: "A one"})
        _novel(self.root, "renamed", {1: "different text"})
        (self.root / "experiments" / "ablate_renamed.json").write_text("{}")
        self.assertEqual(self.names()[0], ["a"])

    def test_fork_metadata_is_dropped(self):
        _novel(self.root, "a", {1: "A one"})
        _novel(self.root, "arm", {1: "different text"})
        (self.root / "experiments" / "fork_arm.json").write_text("{}")
        kept, dropped = self.names()
        self.assertEqual(kept, ["a"])
        self.assertIn("fork", dropped["arm"])

    def test_identical_ch1_is_a_copy_and_the_longer_book_is_canonical(self):
        _novel(self.root, "zzz_long", {1: "shared", 2: "x", 3: "y"})
        _novel(self.root, "aaa_short", {1: "shared", 2: "x"})
        kept, dropped = self.names()
        # Alphabetically first would be aaa_short; chapter count must win instead.
        self.assertEqual(kept, ["zzz_long"])
        self.assertIn("Ch1 identical to zzz_long", dropped["aaa_short"])

    def test_equal_length_copies_break_the_tie_alphabetically(self):
        _novel(self.root, "book", {1: "shared", 2: "x"})
        _novel(self.root, "book_v1_backup", {1: "shared", 2: "different"})
        kept, dropped = self.names()
        self.assertEqual(kept, ["book"])
        self.assertIn("1/2 chapters byte-identical", dropped["book_v1_backup"])

    def test_divergent_copy_reports_partial_overlap_not_full(self):
        """The real case: same Ch1, most later chapters differ. Still a copy."""
        _novel(self.root, "book", {1: "s", 2: "a", 3: "b", 4: "c"})
        _novel(self.root, "book_fork", {1: "s", 2: "a", 3: "X", 4: "Y"})
        _, dropped = self.names()
        self.assertIn("2/4 chapters byte-identical", dropped["book_fork"])

    def test_a_fork_cannot_claim_the_canonical_slot_from_its_source(self):
        """The real case, measured 2026-07-28: `p4b_det` was a 25-chapter fork of
        `yeban_guize`, out-ranked it on chapter count, took the canonical slot,
        and was then dropped itself as a fork -- so BOTH disappeared and the
        library aggregate silently lost a whole book (26 chapters, FPY' 77%)."""
        _novel(self.root, "source", {1: "shared", 2: "a"})
        _novel(self.root, "armfork", {1: "shared", 2: "a", 3: "b"})  # longer!
        (self.root / "experiments" / "fork_armfork.json").write_text("{}")
        kept, dropped = self.names()
        self.assertEqual(kept, ["source"])
        self.assertIn("fork", dropped["armfork"])

    def test_an_ablation_cannot_claim_the_canonical_slot_either(self):
        _novel(self.root, "source", {1: "shared"})
        _novel(self.root, "source__ablate_k", {1: "shared", 2: "extra"})
        self.assertEqual(self.names()[0], ["source"])

    def test_same_brief_but_different_ch1_is_not_a_copy(self):
        """`tangshuting_e2e`'s role: an independent run of the same story concept
        shares no byte-identical prose, so it must survive the filter."""
        _novel(self.root, "book", {1: "one telling"})
        _novel(self.root, "book_e2e", {1: "another telling"})
        self.assertEqual(self.names()[0], ["book", "book_e2e"])

    def test_explicit_names_are_never_filtered(self):
        _novel(self.root, "book", {1: "shared"})
        _novel(self.root, "book_copy", {1: "shared"})
        kept, dropped = self.names(explicit=["book_copy", "book"])
        self.assertEqual(kept, ["book_copy", "book"])  # order preserved as given
        self.assertEqual(dropped, {})

    def test_include_all_keeps_derivatives_and_reports_nothing_dropped(self):
        _novel(self.root, "book", {1: "shared"})
        _novel(self.root, "book__ablate_k", {1: "x"})
        kept, dropped = self.names(include_all=True)
        self.assertEqual(kept, ["book", "book__ablate_k"])
        self.assertEqual(dropped, {})

    def test_dirs_without_checkpoints_are_invisible(self):
        _novel(self.root, "book", {1: "x"})
        (self.root / "novels" / "scaffolded_only").mkdir(parents=True)
        self.assertEqual(self.names()[0], ["book"])

    def test_chapterless_novel_is_kept_not_treated_as_a_copy(self):
        """Two books with no chapters yet must not collapse into one via a
        None hash matching."""
        _novel(self.root, "empty_a", {})
        _novel(self.root, "empty_b", {})
        self.assertEqual(self.names()[0], ["empty_a", "empty_b"])


if __name__ == "__main__":
    unittest.main()
