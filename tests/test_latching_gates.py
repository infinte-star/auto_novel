"""Unit tests for the two BLOCKING gates fixed in the latching-gate sweep.

Both gates shared one defect class: **a blocking verdict conditioned on state the
current attempt cannot change.** The block is then unactionable — the forced retry
provably fails — and FPY' counts every one of them as a first-draft failure. The
repo has precedent (`fingerprint_warn_threshold`, deleted as unreachable: max
0.448 measured against a 0.65 line, LESSONS §8).

  A. `book_wide_fossils` hard-fossil rejects were a CUMULATIVE book property
     charged to the current chapter. Once a phrase sat in 82/199 chapters the
     ratio could not be lowered by writing anything, so the gate latched ON and
     rejected chapters that did not contain the phrase at all. Fix: a fossil may
     only turn hard when the chapter under review actually uses it (`in_current`),
     and `book_fossil_hard_ratio` is floored at the candidacy fraction so the key
     stops being dead.

  B. `chapter_mode_monotony` measured its monotony FRACTION with the genre-BIASED
     classifier, which by design returns the baseline label unless a chapter
     clearly breaks form — so under a baseline the frac has a floor near 1.0 and
     is not comparable to `chapter_mode_block_frac` at all. Fix: count with the
     unbiased classifier, keep reporting the biased label.

Zero LLM calls. Measured library effect of the pair: FPY' 81.8% -> 85.8%
(`python tools/replay_gates.py`).
"""
import unittest

from quality import book_wide_fossils, chapter_mode_monotony


def _cfg(**novel) -> dict:
    return {"novel": novel}


# A phrase long enough to survive the 6-char n-gram window, plus filler that
# differs per chapter so nothing else crosses the fossil threshold.
FOSSIL = "声音压得很低"


def _book(fossil_chapters: range | list, total: int) -> dict[int, str]:
    """`total` chapters, `fossil_chapters` of which contain FOSSIL.

    Filler is one repeated per-chapter-unique CJK char, so every filler n-gram
    belongs to exactly one chapter and FOSSIL is the only cross-chapter phrase.
    (A sliding arithmetic generator does NOT work here: chapter texts then differ
    by a constant offset and share most of their 6-char windows.)
    """
    fs = set(fossil_chapters)
    return {
        ch: (f"第{ch}章 标题\n\n" + (FOSSIL if ch in fs else "")
             + chr(0x4E00 + ch) * 200)
        for ch in range(1, total + 1)
    }


class BookFossilNotLatchingTest(unittest.TestCase):
    """A hard fossil must indict only a chapter that actually contains it."""

    def test_entrenched_phrase_is_detected_at_all(self):
        res = book_wide_fossils(_book(range(1, 9), 10), _cfg(), current_chapter=1)
        self.assertIn(FOSSIL, res["phrases"],
                      "the gate must still find a phrase in 8/10 chapters")

    def test_hard_reject_fires_for_a_chapter_that_uses_the_fossil(self):
        res = book_wide_fossils(_book(range(1, 9), 10), _cfg(), current_chapter=3)
        self.assertTrue(any(f["phrase"] == FOSSIL for f in res["hard_fossils"]))

    def test_hard_reject_does_NOT_fire_for_a_chapter_that_avoided_it(self):
        # Ch10 is clean. Under the old rule it was rejected anyway, and no rewrite
        # of Ch10 could ever clear it — the numerator is frozen in Ch1-8.
        res = book_wide_fossils(_book(range(1, 9), 10), _cfg(), current_chapter=10)
        self.assertEqual(res["hard_fossils"], [])
        self.assertIn(FOSSIL, res["phrases"],
                      "avoidance pressure must survive as advisory directives")
        self.assertTrue(res["directives"])

    def test_in_current_flag_tracks_membership_both_ways(self):
        book = _book(range(1, 9), 10)
        by_ch = {ch: {f["phrase"]: f["in_current"]
                      for f in book_wide_fossils(book, _cfg(), current_chapter=ch)["fossils"]}
                 for ch in (5, 10)}
        self.assertIs(by_ch[5][FOSSIL], True)
        self.assertIs(by_ch[10][FOSSIL], False)

    def test_no_current_chapter_means_no_hard_verdict(self):
        # `fix.py` / offline scans call without a chapter; they must not synthesize
        # a blocking verdict out of a book-level scan.
        res = book_wide_fossils(_book(range(1, 9), 10), _cfg())
        self.assertEqual(res["hard_fossils"], [])
        self.assertTrue(res["fossils"])

    def test_hard_ratio_is_floored_at_the_candidacy_fraction(self):
        # A hard_ratio BELOW the candidacy floor cannot select a subset — every
        # candidate already clears it. The floor makes the key describe reality
        # instead of silently doing nothing (the dead-threshold defect).
        book = _book(range(1, 9), 10)
        lo = book_wide_fossils(book, _cfg(book_fossil_hard_ratio=0.05),
                              current_chapter=3)["hard_fossils"]
        mid = book_wide_fossils(book, _cfg(book_fossil_hard_ratio=0.20),
                                current_chapter=3)["hard_fossils"]
        self.assertEqual([f["phrase"] for f in lo], [f["phrase"] for f in mid])

    def test_hard_ratio_above_the_floor_still_discriminates(self):
        # 8/10 chapters clears 0.30 candidacy but not a 0.90 hard line.
        res = book_wide_fossils(_book(range(1, 9), 10),
                                _cfg(book_fossil_hard_ratio=0.90), current_chapter=3)
        self.assertTrue(res["fossils"])
        self.assertEqual(res["hard_fossils"], [])

    def test_whitelist_and_disable_switch_still_honoured(self):
        book = _book(range(1, 9), 10)
        self.assertEqual(
            book_wide_fossils(book, _cfg(), whitelist={FOSSIL}, current_chapter=3)["fossils"], [])
        self.assertEqual(
            book_wide_fossils(book, _cfg(book_fossil_enabled=False), current_chapter=3)["fossils"], [])


REASONING = "推理"      # baseline form for suspense / rule-horror
RELATIONAL = "结盟"
ACTION = "追击"


def _plan(mode_text: str, n: int = 6) -> dict:
    """A plan that BOTH classifiers agree on (no baseline keyword present)."""
    return {"title": "章", "goal": mode_text * n, "beats": [mode_text]}


def _incidental(real_mode: str) -> dict:
    """A plan that is really `real_mode`, but carries one incidental 推理 keyword.

    This is the shape the bias exists for and the shape it breaks on: 3 hits of the
    real form vs 1 baseline hit does not clear `margin` (3), so the BIASED
    classifier returns "reasoning" — the same label for a relational chapter and an
    action one. Counting that label is how the frac reached a floor near 1.0.
    """
    return {"title": "章", "goal": real_mode * 3 + REASONING, "beats": []}


class ChapterModeFractionUnbiasedTest(unittest.TestCase):
    """The block decision must be measurable, i.e. escapable by writing differently."""

    BLOCK = dict(chapter_mode_block_frac=0.8, chapter_mode_warn_frac=0.6,
                 chapter_mode_min_window=4, chapter_mode_window=6)

    def _varied_window(self) -> list[dict]:
        """Really varied (3 relational + 3 action), all labelled "reasoning"."""
        return [_incidental(RELATIONAL)] * 2 + [_incidental(ACTION)] * 4

    def test_biased_baseline_does_not_manufacture_a_full_window(self):
        # A genuinely varied window under baseline="reasoning": the biased
        # classifier stamps every plan "reasoning", so the OLD frac was 7/7 = 1.0
        # and the gate blocked. No re-roll escapes a genre label — that is what made
        # the block unactionable (16 of 18 tangshuting_e2e retries still blocked).
        cfg = _cfg(chapter_mode_baseline="reasoning", **self.BLOCK)
        res = chapter_mode_monotony(_incidental(RELATIONAL), self._varied_window(), cfg)
        self.assertLess(res["mode_frac"], 0.8)
        self.assertNotEqual(res["level"], "block")

    def test_real_monotony_still_blocks_under_a_baseline(self):
        # Same baseline, but the plans really are all the same form. The gate must
        # keep firing — this is the monotony it was built for.
        cfg = _cfg(chapter_mode_baseline="reasoning", **self.BLOCK)
        res = chapter_mode_monotony(_plan(REASONING), [_plan(REASONING)] * 6, cfg)
        self.assertGreaterEqual(res["mode_frac"], 0.8)
        self.assertEqual(res["level"], "block")
        self.assertTrue(res["directives"])

    def test_real_monotony_still_blocks_without_a_baseline(self):
        res = chapter_mode_monotony(_plan(REASONING), [_plan(REASONING)] * 6,
                                    _cfg(**self.BLOCK))
        self.assertEqual(res["level"], "block")

    def test_reported_label_stays_the_biased_one(self):
        # The biased label is the better DESCRIPTION of the chapter; it just cannot
        # be counted. Keeping it is what makes the directive read correctly.
        cfg = _cfg(chapter_mode_baseline="reasoning", **self.BLOCK)
        res = chapter_mode_monotony(_incidental(RELATIONAL), self._varied_window(), cfg)
        self.assertEqual(res["mode"], "reasoning")

    def test_mixed_window_lands_between_the_thresholds(self):
        cfg = _cfg(**self.BLOCK)
        recent = [_plan(REASONING)] * 3 + [_plan(RELATIONAL)] * 3
        res = chapter_mode_monotony(_plan(REASONING), recent, cfg)
        self.assertEqual(res["same_count"], 4)   # 3 + itself
        self.assertEqual(res["window"], 7)
        self.assertEqual(res["level"], "ok")     # 4/7 = 0.571 < warn 0.6

    def test_short_window_never_blocks(self):
        res = chapter_mode_monotony(_plan(REASONING), [_plan(REASONING)] * 2,
                                    _cfg(**self.BLOCK))
        self.assertEqual(res["level"], "ok")

    def test_disable_switch_still_honoured(self):
        res = chapter_mode_monotony(_plan(REASONING), [_plan(REASONING)] * 6,
                                    _cfg(chapter_mode_enabled=False, **self.BLOCK))
        self.assertEqual(res["level"], "ok")
        self.assertIsNone(res["mode"])


class EmptyAbilityWhitelistTest(unittest.TestCase):
    """An ability-free brief must be able to produce NO whitelist at all.

    `CONTRACT_SYSTEM` used to require the core 金手指 in `ability_whitelist`
    unconditionally, "even if the brief only strongly implies it". For a realistic
    brief with no ability system that forces a fabrication — and the whitelist is a
    per-chapter HARD acceptance rule, so a fabricated entry is a violation the
    writer can never clear. Measured cost in the archived corpus: 13 first-draft
    failures across tangshuting + tangshuting_v1_backup, whose 200-chapter 都市甜宠
    brief mentions neither of the two abilities bootstrap invented for it (one in
    bible.md, a different one in contract.md, mutually contradictory).

    The prompt now tells the extractor to leave the array empty for such briefs.
    This test pins the renderer half of the contract: an empty whitelist must omit
    the section entirely, so the reviewer has no whitelist clause it can cite.
    """

    def test_empty_whitelist_omits_the_section(self):
        from memory import _contract_to_markdown
        md = _contract_to_markdown({
            "protagonist": "汤舒婷",
            "iron_rules": ["每章必须有一次具体的味觉描写"],
            "ability_whitelist": [],
            "must_hold": ["全书总章数为200章"],
        })
        self.assertNotIn("能力白名单", md)
        self.assertIn("开写铁律", md)
        self.assertIn("必须全程维持的硬设定", md)

    def test_populated_whitelist_still_renders(self):
        from memory import _contract_to_markdown
        md = _contract_to_markdown({
            "ability_whitelist": [{"name": "味觉共情", "modality": "cognitive"}],
        })
        self.assertIn("能力白名单", md)
        self.assertIn("味觉共情", md)


if __name__ == "__main__":
    unittest.main()
