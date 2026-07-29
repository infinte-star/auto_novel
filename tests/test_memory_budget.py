"""Unit tests for the P2/P3 pure functions.

- memory._pack_sections (the one section-packer behind all four context builders)
- novel._parse_price_table (model-aware cost reporting)
- writing._hook_directives_block (吸量包 opening injection)

`memory.memory_context` needs Paths+conn, so its max_chars behavior is exercised
indirectly via the budget helper here plus the live replay/ablation harnesses.
The compress-ratchet tests went with `_compressible_sections`: v2 has no memory
compression path at all (its context is `v2/canon.py`'s projection), so the whole
chain was orphaned by v1's deletion.
"""
import random
import unittest

from engine.bootstrap import _pack_sections
from novel import _parse_price_table
from engine.write import _hook_directives_block


def _pack_reference(sections, budget, header=None):
    """The inline loop `_pack_sections` replaced, kept verbatim as the oracle.

    `cacheable_prefix`'s output bytes are a provider prompt-cache key, so the
    extraction had to be byte-exact, not merely equivalent-looking. Verified
    against real books at 12 budgets (identical at every one); this oracle keeps
    that guarantee enforceable after the live novels are gone.
    """
    parts = [header] if header else []
    used = len(header) if header else 0
    for title, body, cap in sections:
        body = body.strip()
        if not body:
            continue
        snippet = body if len(body) <= cap else body[:cap] + "\n...[truncated]"
        block = f"## {title}\n{snippet}"
        if used + len(block) + 2 > budget:
            remaining = budget - used - len(f"## {title}\n") - 2
            if remaining > 400:
                parts.append(f"## {title}\n{body[:remaining]}\n...[truncated]")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


class TestPackSections(unittest.TestCase):
    def test_matches_the_inline_loop_it_replaced(self):
        rng = random.Random(20260728)
        for trial in range(400):
            n = rng.randrange(0, 6)
            sections = [
                (f"节{i}", "字" * rng.choice([0, 1, 50, 400, 401, 900, 5000]),
                 rng.choice([1, 100, 400, 1000, 9999]))
                for i in range(n)
            ]
            budget = rng.choice([0, 1, 100, 401, 405, 500, 1500, 6000, 100000])
            header = rng.choice([None, "", "# 稳定参照（可缓存）"])
            with self.subTest(trial=trial, budget=budget, header=header):
                self.assertEqual(_pack_sections(sections, budget, header),
                                 _pack_reference(sections, budget, header))

    def test_empty_bodies_are_dropped_not_emitted_as_bare_headers(self):
        got = _pack_sections([("空", "   ", 100), ("有", "内容", 100)], 10000)
        self.assertEqual(got, "## 有\n内容")

    def test_each_body_is_capped_by_its_own_cap_before_the_budget(self):
        got = _pack_sections([("甲", "x" * 500, 10)], 10000)
        self.assertEqual(got, "## 甲\n" + "x" * 10 + "\n...[truncated]")

    def test_overflow_breaks_instead_of_skipping_to_a_smaller_section(self):
        """`sections` order is a priority order — never reorder by fit."""
        got = _pack_sections([("大", "x" * 5000, 5000), ("小", "y", 10)], 300)
        self.assertNotIn("小", got)

    def test_overflow_tail_needs_more_than_400_chars_to_be_worth_emitting(self):
        self.assertEqual(_pack_sections([("甲", "x" * 5000, 5000)], 405), "")
        self.assertTrue(_pack_sections([("甲", "x" * 5000, 5000)], 500))

    def test_header_is_counted_against_the_budget(self):
        head = "# H"
        self.assertEqual(_pack_sections([], 10, head), head)
        self.assertEqual(_pack_sections([("甲", "x", 10)], 10, head), head)


class TestPriceTable(unittest.TestCase):
    def test_basic_parse(self):
        t = _parse_price_table("deepseek:3.0:15.0, minimax:0.5:2.0")
        self.assertEqual(t, [("deepseek", 3.0, 15.0), ("minimax", 0.5, 2.0)])

    def test_semicolon_and_case(self):
        t = _parse_price_table("DeepSeek-V4:1.1:2.2; GLM:0.3:0.9")
        self.assertEqual(t[0][0], "deepseek-v4")
        self.assertEqual(len(t), 2)

    def test_malformed_entries_skipped(self):
        t = _parse_price_table("bad, deepseek:3.0:15.0, x:y:z, only:2")
        self.assertEqual(t, [("deepseek", 3.0, 15.0)])

    def test_empty(self):
        self.assertEqual(_parse_price_table(""), [])
        self.assertEqual(_parse_price_table(None), [])


class TestHookDirectivesBlock(unittest.TestCase):
    def test_basic_render(self):
        blk = _hook_directives_block({"hook_directives": ["开篇必须当章完成A", "首次B要具体"]})
        self.assertIn("开篇吸量指令", blk)
        self.assertIn("- 开篇必须当章完成A", blk)

    def test_caps_five_items(self):
        blk = _hook_directives_block({"hook_directives": [f"指令{i}" for i in range(9)]})
        self.assertEqual(blk.count("- 指令"), 5)

    def test_char_budget(self):
        long = "很长的指令" * 60  # 300 chars each
        blk = _hook_directives_block({"hook_directives": [long, long, long]})
        # 600-char budget → at most 2 items land
        self.assertLessEqual(blk.count("- 很长"), 2)

    def test_absent_or_malformed(self):
        self.assertEqual(_hook_directives_block({}), "")
        self.assertEqual(_hook_directives_block({"hook_directives": "not-a-list"}), "")
        self.assertEqual(_hook_directives_block(None), "")
        self.assertEqual(_hook_directives_block({"hook_directives": ["", "  "]}), "")


if __name__ == "__main__":
    unittest.main()
