"""Framework fixes from test_payoff audit: dialogue desert + short chapter."""
import unittest

from engine.quality import hard_block_reasons, length_band_check, style_health


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


class ShuangDialogueHardBlockTest(unittest.TestCase):
    """Part 2: shuang genre raises dialogue_hard_block_ratio to 0.08."""

    def _review(self, dlg_ratio, chars=3000):
        return {
            "style_health": {
                "penalty": 0.5,
                "metrics": {"dialogue_char_ratio": dlg_ratio, "chars": chars},
            },
        }

    def test_shuang_blocks_at_7pct(self):
        reasons = hard_block_reasons(
            self._review(0.07), _cfg(dialogue_hard_block_ratio=0.08))
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertTrue(matching, f"expected block at 7%, got {reasons}")

    def test_shuang_passes_at_9pct(self):
        reasons = hard_block_reasons(
            self._review(0.09), _cfg(dialogue_hard_block_ratio=0.08))
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertFalse(matching, f"unexpected block at 9%")

    def test_wanzu_blocks_at_3pct(self):
        reasons = hard_block_reasons(
            self._review(0.03), _cfg(dialogue_hard_block_ratio=0.04))
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertTrue(matching, f"expected block at 3%, got {reasons}")

    def test_wanzu_passes_at_5pct(self):
        reasons = hard_block_reasons(
            self._review(0.05), _cfg(dialogue_hard_block_ratio=0.04))
        matching = [r for r in reasons if "dialogue_desert" in r]
        self.assertFalse(matching)


class ConfigurableDialogueStarvedPenaltyTest(unittest.TestCase):
    """Part 3: style_dialogue_starved_penalty is configurable."""

    def _low_dialogue_text(self, ratio=0.05):
        dlg = int(3000 * ratio)
        prose = 3000 - dlg
        return "“" + "哈" * (dlg - 2) + "”" + "叙" * prose

    def test_penalty_increases_with_config(self):
        """Same text, higher config penalty → higher total penalty."""
        text = self._low_dialogue_text(0.05)
        base_cfg = {"novel": {"style_dialogue_ratio_min": 0.12}}
        high_cfg = {"novel": {"style_dialogue_ratio_min": 0.12,
                              "style_dialogue_starved_penalty": 1.5}}
        p_default = style_health(text, base_cfg)["penalty"]
        p_high = style_health(text, high_cfg)["penalty"]
        self.assertGreater(p_high, p_default,
                           "configurable penalty should raise total")
        self.assertAlmostEqual(p_high - p_default, 0.5, places=1)

    def test_starved_flag_present(self):
        text = self._low_dialogue_text(0.05)
        result = style_health(text, {"novel": {
            "style_dialogue_ratio_min": 0.12,
            "style_dialogue_starved_penalty": 1.5,
        }})
        self.assertTrue(
            any("dialogue_starved" in f for f in result.get("flags", [])),
            f"expected dialogue_starved flag in {result.get('flags')}")


if __name__ == "__main__":
    unittest.main()
