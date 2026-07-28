"""The CCR baseline tool's proxy mapping and its refusal to mix contracts.

The danger in this tool is not a crash, it is a plausible number. A merged_plan
has no `turn` and no `forbid`, so a proxy CCR is computed over five contract
items where a real card has seven. If those two populations were ever averaged
together — or if a v2 arm silently fell back to a proxy reconstruction of its own
plan — the report would still print a tidy percentage, and it would not be a rate
of anything.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import ccr_baseline as ccr


PLAN = {
    "title": "第42章",
    "goal": "找出母亲留下的东西",
    "conflict": "后厨夜里上锁",
    "location": "顾家老宅的后厨",
    "character_focus": ["汤舒婷", "顾峥"],
    "payoff": "铜钥匙上的刻字与母亲遗物对上",
    "hook": "铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死",
    "beats": ["汤舒婷推开后厨的门", "灶台夹层里摸到铜钥匙"],
    "risk": "避免与近期医院场景重复",
}


def _write(root: Path, novel: str, ch: int, *, plan=None, card=None, text=""):
    d = root / "novels" / novel
    cp = d / "logs" / "checkpoints" / f"ch{ch:04d}"
    cp.mkdir(parents=True, exist_ok=True)
    (d / "chapters").mkdir(parents=True, exist_ok=True)
    if plan is not None:
        (cp / "plan_initial_attempt0_arbitration.json").write_text(
            json.dumps({"payload": {"decision": {"merged_plan": plan}}},
                       ensure_ascii=False), encoding="utf-8")
    if card is not None:
        (cp / ccr.CARD_CHECKPOINT).write_text(
            json.dumps({"payload": {"card": card}}, ensure_ascii=False),
            encoding="utf-8")
    if text:
        (d / "chapters" / f"{ch:04d}.md").write_text(text, encoding="utf-8")


class ProxyCardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _dir(self, novel="book", ch=1, **kw):
        _write(self.root, novel, ch, **kw)
        return self.root / "novels" / novel / "logs" / "checkpoints" / f"ch{ch:04d}"

    def test_maps_the_five_fields_a_plan_actually_carries(self):
        card = ccr.proxy_card(self._dir(plan=PLAN))
        self.assertEqual(card["where"], "顾家老宅的后厨")
        self.assertEqual(card["who"], ["汤舒婷", "顾峥"])
        self.assertEqual(card["payoff"], PLAN["payoff"])
        self.assertEqual(card["exit_hook"], PLAN["hook"])
        self.assertEqual(len(card["beats"]), 2)

    def test_never_invents_a_turn_or_a_forbid(self):
        # `conflict` is a situation, not a turning point, and `risk` is planning
        # advice, not a ban. Scoring either under the name `turn`/`forbid` would
        # report a made-up quantity under a real one.
        card = ccr.proxy_card(self._dir(plan=PLAN))
        self.assertNotIn("turn", card)
        self.assertNotIn("forbid", card)
        self.assertEqual(set(card) - set(ccr.PROXY_FIELDS), set())

    def test_a_real_card_is_preferred_over_the_proxy(self):
        d = self._dir(plan=PLAN, card={"where": "别处", "turn": "真转折"})
        self.assertEqual(ccr.real_card(d)["turn"], "真转折")

    def test_no_plan_and_no_card_is_none_not_an_empty_card(self):
        # An empty card would make `contract_fulfilment` return a vacuous 1.0.
        self.assertIsNone(ccr.proxy_card(self._dir()))
        self.assertIsNone(ccr.real_card(self._dir()))

    def test_a_plan_of_only_unmapped_fields_is_none(self):
        self.assertIsNone(ccr.proxy_card(self._dir(plan={"goal": "x", "risk": "y"})))


class MeasureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "experiments").mkdir()
        self.addCleanup(self._tmp.cleanup)
        self._saved = ccr.ROOT
        ccr.ROOT = self.root
        self.addCleanup(lambda: setattr(ccr, "ROOT", self._saved))

    def test_a_faithful_chapter_scores_1_and_a_wandering_one_does_not(self):
        good = ("汤舒婷推开顾家老宅的后厨门。" * 20
                + "顾峥站在门口。" * 20
                + "铜钥匙上的刻字与母亲遗物对上。" * 20
                + "灶台夹层里摸到铜钥匙。" * 20
                + "她把铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死。")
        bad = "她坐在客厅里想事情，什么也没发生。" * 60
        _write(self.root, "book", 1, plan=PLAN, text=good)
        _write(self.root, "book", 2, plan=PLAN, text=bad)
        rows = {r["ch"]: r for r in ccr.measure("book", 1, 99)["rows"]}
        self.assertEqual(rows[1]["ccr"], 1.0)
        self.assertTrue(rows[1]["passed"])
        self.assertLess(rows[2]["ccr"], 0.3)
        self.assertFalse(rows[2]["passed"])

    def test_kind_is_reported_per_novel(self):
        text = "汤舒婷推开顾家老宅的后厨门。" * 60
        _write(self.root, "proxybook", 1, plan=PLAN, text=text)
        _write(self.root, "realbook", 1, card={"where": "顾家老宅的后厨"}, text=text)
        self.assertEqual(ccr.measure("proxybook", 1, 9)["kind"], "PROXY")
        self.assertEqual(ccr.measure("realbook", 1, 9)["kind"], "REAL")

    def test_a_v2_arm_missing_its_card_archive_reads_PROXY_not_REAL(self):
        # The forward contract with `v2/run.py`: if it forgets to archive the card
        # it wrote against, the report must say PROXY rather than quietly measure
        # the arm against a reconstruction of its own plan.
        _write(self.root, "arm", 1, plan=PLAN, text="汤舒婷推开顾家老宅的后厨门。" * 60)
        self.assertEqual(ccr.measure("arm", 1, 9)["kind"], "PROXY")

    def test_refusal_length_chapters_are_skipped_not_scored_zero(self):
        _write(self.root, "book", 1, plan=PLAN, text="太短。")
        self.assertEqual(ccr.measure("book", 1, 9)["rows"], [])

    def test_chapters_with_no_text_on_disk_are_skipped(self):
        _write(self.root, "book", 1, plan=PLAN)
        self.assertEqual(ccr.measure("book", 1, 9)["rows"], [])

    def test_range_filter_applies(self):
        text = "汤舒婷推开顾家老宅的后厨门。" * 60
        for ch in (1, 5, 9):
            _write(self.root, "book", ch, plan=PLAN, text=text)
        got = [r["ch"] for r in ccr.measure("book", 4, 6)["rows"]]
        self.assertEqual(got, [5])


class FieldStatsTest(unittest.TestCase):
    def test_unjudgeable_items_leave_the_denominator(self):
        rows = [{"items": [
            {"field": "where", "verdict": "hit"},
            {"field": "where", "verdict": "miss"},
            {"field": "turn", "verdict": "unjudgeable"},
        ]}]
        stats = ccr.field_stats(rows)
        self.assertEqual(stats["where"], (1, 2))
        self.assertNotIn("turn", stats)


class ReportTest(unittest.TestCase):
    """The per-field breakdown is the part of the report that names WHICH item the
    prose keeps dropping. It went missing for one run because the LIBRARY line
    consumed the `kinds` set with `.pop()` -- a bug no assertion about numbers
    would have caught, since every number printed was correct.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "experiments").mkdir()
        self.addCleanup(self._tmp.cleanup)
        self._saved = ccr.ROOT
        ccr.ROOT = self.root
        self.addCleanup(lambda: setattr(ccr, "ROOT", self._saved))

    def _run(self, *argv):
        import contextlib, io, sys
        buf = io.StringIO()
        old = sys.argv
        sys.argv = ["ccr_baseline.py", *argv]
        try:
            with contextlib.redirect_stdout(buf):
                ccr.main()
        finally:
            sys.argv = old
        return buf.getvalue()

    def test_per_field_table_survives_a_single_kind_corpus(self):
        text = "汤舒婷推开顾家老宅的后厨门。" * 60
        _write(self.root, "book", 1, plan=PLAN, text=text)
        out = self._run("book")
        self.assertIn("LIBRARY", out)
        self.assertIn("per-field hit rate (PROXY", out)
        for f in ccr.PROXY_FIELDS:
            self.assertIn(f, out)

    def test_mixed_kinds_refuse_an_average_but_still_break_down_both(self):
        text = "汤舒婷推开顾家老宅的后厨门。" * 60
        _write(self.root, "proxybook", 1, plan=PLAN, text=text)
        _write(self.root, "realbook", 1, card={"where": "顾家老宅的后厨"}, text=text)
        out = self._run("proxybook", "realbook")
        self.assertNotIn("LIBRARY", out)
        self.assertIn("No library average", out)
        self.assertIn("per-field hit rate (PROXY", out)
        self.assertIn("per-field hit rate (REAL", out)


class ContractIntegrityTest(unittest.TestCase):
    def test_proxy_fields_are_a_strict_subset_of_the_real_contract(self):
        self.assertTrue(set(ccr.PROXY_FIELDS) < set(ccr.REAL_FIELDS))

    def test_the_unmeasured_proxy_fields_are_exactly_turn_and_forbid(self):
        self.assertEqual(set(ccr.REAL_FIELDS) - set(ccr.PROXY_FIELDS),
                         {"turn", "forbid"})

    def test_the_proxy_loses_a_hard_field_and_the_tool_knows_which(self):
        from v2.accept import HARD_FIELDS
        lost = set(HARD_FIELDS) - set(ccr.PROXY_FIELDS)
        self.assertEqual(lost, {"turn"},
                         "if the hard set changes, the header text that names the "
                         "shrunken proxy hard set must change with it")


if __name__ == "__main__":
    unittest.main()
