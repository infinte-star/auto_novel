"""Framework fixes from test_payoff audit: dialogue desert + short chapter."""
import unittest

from engine.quality import hard_block_reasons, length_band_check


def _cfg(**over):
    novel = {
        "style_penalty_block": 2.0,
        "ai_flavor_penalty_block": 2.5,
        "factcheck_hard_blocks_accept": True,
        "contract_blocks_accept": True,
        "constraint_violation_block_count": 3,
        "chapter_min_chars": 1800,
    }
    novel.update(over)
    return {"novel": novel}


class DialogueDesertBlockTest(unittest.TestCase):

    def _review(self, dlg_ratio=0.02, chars=3000):
        return {
            "style_health": {
                "penalty": 0.5,
                "metrics": {"dialogue_char_ratio": dlg_ratio, "chars": chars},
            },
        }

    def test_extreme_low_dialogue_blocks(self):
        reasons = hard_block_reasons(self._review(0.02, 3000), _cfg())
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertTrue(matching, f"expected dialogue_desert in {reasons}")

    def test_normal_dialogue_passes(self):
        reasons = hard_block_reasons(self._review(0.10, 3000), _cfg())
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertFalse(matching, f"unexpected dialogue_desert in {reasons}")

    def test_disabled_when_zero(self):
        reasons = hard_block_reasons(
            self._review(0.01, 3000), _cfg(dialogue_hard_block_ratio=0)
        )
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertFalse(matching)

    def test_short_chapter_exempt(self):
        """Chapters under 2000 chars may legitimately have sparse dialogue."""
        reasons = hard_block_reasons(self._review(0.01, 1500), _cfg())
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertFalse(matching)


class ShortBlockRatioTest(unittest.TestCase):

    def test_075_blocks_below_threshold(self):
        """1349 chars < 1800 * 0.75 = 1350 → block."""
        r = length_band_check("x" * 1349, _cfg())
        self.assertTrue(r["block"])

    def test_075_passes_above_threshold(self):
        """1351 chars > 1800 * 0.75 = 1350 → no block."""
        r = length_band_check("x" * 1351, _cfg())
        self.assertFalse(r["block"])

    def test_old_050_would_not_block(self):
        """1349 chars > 1800 * 0.5 = 900, so old default would pass this."""
        r = length_band_check("x" * 1349, _cfg(length_band_short_block_ratio=0.5))
        self.assertFalse(r["block"])


if __name__ == "__main__":
    unittest.main()
