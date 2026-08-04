"""v2 acceptance set, CCC, and cite-or-drop. Zero LLM calls.

The load-bearing test in here is `AcceptanceSetTest.test_set_is_exactly_what_the
_ruler_reads`: the acceptance set must be derived from
`quality.hard_block_reasons`, not chosen. An engine A/B where one arm quietly
stops running a gate the other arm runs is not an A/B, and that failure mode is
invisible in the result table — the arm that measures less just looks better.
"""
import unittest

import engine.quality as quality
from engine.quality import REGISTRY
import engine.loop as accept


def _cfg(**novel):
    novel.setdefault("style_penalty_block", 2.0)
    novel.setdefault("chapter_min_chars", 500)
    novel.setdefault("style_em_dash_per_kchar_warn", 6.0)
    novel.setdefault("style_em_dash_per_kchar_bad", 12.0)
    return {"novel": novel}


# A card whose fields are concrete in the way `arc.ARC_SYSTEM` rule 2 demands.
CARD = {
    "where": "顾家老宅的后厨",
    "who": ["汤舒婷", "顾峥"],
    "wants": "找出母亲留下的东西",
    "blocked_by": "后厨夜里上锁",
    "turn": "汤舒婷在灶台夹层里摸到一枚刻字的铜钥匙",
    "payoff": "铜钥匙上的刻字与母亲遗物对上",
    "exit_hook": "铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死",
    "beats": ["汤舒婷推开后厨的门", "灶台夹层里摸到铜钥匙", "铜钥匙插进阁楼锁"],
    "forbid": ["声音压得很低"],
    "opening_type": "physical_action",
}

FULFILLED = (
    "第10章 铜钥匙\n\n"
    + "汤舒婷推开顾家老宅后厨的门。" * 20
    + "她在灶台夹层里摸到一枚刻字的铜钥匙。" * 20
    + "顾峥站在门口看着她。" * 20
    + "刻字与母亲遗物对上了。" * 20
    + "她把铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死。"
)


class AcceptanceSetTest(unittest.TestCase):

    def test_set_is_exactly_what_the_ruler_reads(self):
        """Every blocking gate is either in the set or has a written reason."""
        blocking = {n for n in REGISTRY.list_gates() if REGISTRY.may_block(n)}
        classified = set(accept.ACCEPTANCE_GATES) | set(accept.NOT_IN_ACCEPTANCE)
        unclassified = blocking - classified
        self.assertEqual(
            unclassified, set(),
            "a new blocking gate must be added to ACCEPTANCE_GATES or given a "
            "reason in NOT_IN_ACCEPTANCE -- silently omitting it makes the v1/v2 "
            f"A/B measure different things: {sorted(unclassified)}")

    def test_every_member_is_a_registered_blocking_gate(self):
        for name in accept.ACCEPTANCE_GATES:
            with self.subTest(gate=name):
                self.assertIsNotNone(REGISTRY.get(name))
                self.assertTrue(REGISTRY.may_block(name))

    def test_no_book_scope_gate_in_the_set(self):
        for name in accept.ACCEPTANCE_GATES:
            with self.subTest(gate=name):
                self.assertNotEqual(REGISTRY.scope(name), "book")

    def test_every_exclusion_states_a_reason(self):
        for name, why in accept.NOT_IN_ACCEPTANCE.items():
            with self.subTest(gate=name):
                self.assertGreater(len(why.strip()), 20)

    def test_card_gates_are_card_scope_and_disjoint_from_the_set(self):
        self.assertEqual(set(accept.CARD_GATES) & set(accept.ACCEPTANCE_GATES), set())
        for name in accept.CARD_GATES:
            with self.subTest(gate=name):
                self.assertEqual(REGISTRY.scope(name), "card")

    def test_set_size_matches_the_design(self):
        # The original ten plus independent original-brief adherence. Not a magic
        # number -- a reminder that growth needs a reason, since each member is a
        # way for a first draft to fail.
        self.assertEqual(len(accept.ACCEPTANCE_GATES) + len(accept.NATIVE_CHECKS), 11)


class ContractFulfilmentTest(unittest.TestCase):

    def test_a_chapter_that_writes_the_card_passes_at_ccr_1(self):
        r = accept.contract_fulfilment(CARD, FULFILLED, _cfg())
        self.assertTrue(r["passed"])
        self.assertEqual(r["ccr"], 1.0)
        self.assertEqual(r["hard_misses"], [])
        self.assertGreaterEqual(r["judgeable"], 6)

    def test_a_chapter_that_ignores_the_card_fails_on_the_hard_fields(self):
        text = "第10章\n\n" + "她坐在客厅里想事情，什么也没发生。" * 60
        r = accept.contract_fulfilment(CARD, text, _cfg())
        self.assertFalse(r["passed"])
        self.assertEqual({i["field"] for i in r["hard_misses"]},
                         {"where", "turn", "exit_hook"})
        self.assertLess(r["ccr"], 0.3)
        self.assertTrue(r["directives"])

    def test_hook_must_land_in_the_tail_not_just_somewhere(self):
        # The hook is staged in paragraph one and then abandoned -- the exact
        # failure v1 spent a `revise_hook_only` call on, and which v2 has no
        # repair for at all, so this check is the last line. Body-anywhere
        # matching would call it a pass.
        text = ("第10章\n\n铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死。\n\n"
                + "汤舒婷推开顾家老宅后厨的门。" * 20
                + "她在灶台夹层里摸到一枚刻字的铜钥匙。" * 20
                + "接下来的日子平静无事，谁也没有再提起这件事情。" * 30)
        r = accept.contract_fulfilment(CARD, text, _cfg())
        self.assertIn("exit_hook", {i["field"] for i in r["hard_misses"]})

    def test_forbid_breach_is_a_violation_and_blocks(self):
        text = FULFILLED + "顾峥的声音压得很低。"
        r = accept.contract_fulfilment(CARD, text, _cfg())
        self.assertFalse(r["passed"])
        self.assertEqual([v["phrase"] for v in r["violations"]], ["声音压得很低"])

    def test_forbid_does_not_fire_on_a_short_incidental_fragment(self):
        # 「用月光渲染悲伤」 splits into 月光 / 渲染 / 悲伤, all under
        # FORBID_MIN_ANCHOR. A chapter that merely mentions 月光 must NOT be
        # rejected: a false violation costs a rework, a missed one costs a line
        # of advice.
        card = dict(CARD, forbid=["用月光渲染悲伤"])
        text = FULFILLED + "月光落在窗台上。"
        r = accept.contract_fulfilment(card, text, _cfg())
        self.assertEqual(r["violations"], [])

    def test_a_card_cannot_ban_what_it_also_requires(self):
        # Measured on ts_v2arm Ch219. The arc planner filed a whole requirement
        # into `forbid` -- 「…终章必须…陆时砚说'那我负责每晚关灯'…」 -- and
        # `_anchors` lifted 「负责每晚关灯」 out of it while the same card's
        # `payoff` demanded that line verbatim. CCR was 1.0 and the chapter was
        # rejected anyway. Nothing the chapter can write turns that green:
        # including the line breaks `forbid`, omitting it breaks `payoff`. Two
        # such chapters in a row halted the run.
        #
        # This entry carries 「必须」, so `_is_misfiled_requirement` now waives the
        # whole thing before the phrase-level guard is reached -- the Ch220 fix
        # subsumes this case. Both waivers are kept: the phrase-level one is the
        # only guard for a BARE banned phrase that the card also requires (no
        # markers to read), which is `test_a_waiver_is_recorded_rather_than_dropped`.
        card = dict(
            CARD,
            payoff="陆时砚把空碗放在脚边，说‘那我负责每晚关灯’",
            forbid=["第219-220章终章必须落在后门初雪夜，陆时砚说“那我负责每晚关灯”，"
                    "与第1章形成首尾呼应"],
        )
        text = FULFILLED + "陆时砚把空碗放在脚边。“那我负责每晚关灯。”"
        r = accept.contract_fulfilment(card, text, _cfg())
        self.assertEqual(r["violations"], [])
        self.assertEqual([c["why"] for c in r["forbid_conflicts"]],
                         ["requirement_misfiled_as_ban"])

    def test_a_waiver_is_recorded_rather_than_dropped(self):
        # A planner that keeps filing requirements into `forbid` is worth seeing;
        # a silently-waived ban would make that invisible. It must NOT become a
        # writer directive either -- no chapter can repair its own card.
        card = dict(CARD, payoff="她把铜钥匙插进阁楼那把锁",
                    forbid=["铜钥匙插进阁楼那把锁"])
        r = accept.contract_fulfilment(card, FULFILLED, _cfg())
        self.assertTrue(r["passed"])
        self.assertTrue(r["forbid_conflicts"])
        self.assertNotIn("负责", " ".join(r["directives"]))
        for d in r["directives"]:
            self.assertNotIn("铜钥匙插进阁楼那把锁", d)

    def test_a_genuine_ban_still_fires_when_the_card_does_not_require_it(self):
        # The guard must be narrow: it waives only phrases the card itself
        # demands. A ban on something the card never asks for is untouched.
        text = FULFILLED + "顾峥的声音压得很低。"
        r = accept.contract_fulfilment(CARD, text, _cfg())
        self.assertFalse(r["passed"])
        self.assertEqual([v["phrase"] for v in r["violations"]], ["声音压得很低"])
        self.assertEqual(r["forbid_conflicts"], [])

    def test_required_text_reads_every_positive_card_field(self):
        # Named positively so a new card field is opt-in: a field added to the
        # card but not to REQUIRED_FIELDS cannot silently start waiving bans.
        self.assertNotIn("forbid", accept.REQUIRED_FIELDS)
        blob = accept._required_text(CARD)
        for field in ("where", "turn", "payoff", "exit_hook"):
            self.assertIn(str(CARD[field]), blob)
        for name in CARD["who"]:
            self.assertIn(name, blob)      # list-valued
        for beat in CARD["beats"]:
            self.assertIn(beat, blob)
        self.assertNotIn("声音压得很低", blob)   # the forbid entry itself

    def test_required_text_survives_junk_card_values(self):
        self.assertEqual(accept._required_text(None), "")
        self.assertEqual(accept._required_text({}), "")
        # A card whose `who` came back as a dict must not raise -- the card is
        # untrusted LLM output, and a crash here would take down the chapter.
        blob = accept._required_text({"where": "后厨", "who": {"a": 1},
                                      "beats": [None, 3, "摸到铜钥匙"]})
        self.assertIn("后厨", blob)
        self.assertIn("摸到铜钥匙", blob)

    def test_an_obligation_filed_as_a_ban_is_waived_whole(self):
        # Measured on ts_v2arm Ch220 -- the pure form of the Ch219 defect. The
        # requirement lives ONLY inside the misfiled `forbid` string, so there is
        # no second field to compare against and `_required_text` cannot see it.
        # 「后门台阶上」 was lifted out as a ban; the chapter obeyed the finale
        # obligation the same sentence states and was charged for it.
        card = dict(CARD, forbid=[
            "第219-220章终章必须落在汤记后门初雪夜，汤舒婷给陆时砚下一碗雪里蕻面，"
            "两人并肩坐在后门台阶上，与第1章形成首尾呼应"])
        text = FULFILLED + "汤舒婷坐在后门台阶上，手里那只空碗还搁在膝盖上。"
        r = accept.contract_fulfilment(card, text, _cfg())
        self.assertEqual(r["violations"], [])
        self.assertTrue(r["passed"])
        self.assertEqual([c["why"] for c in r["forbid_conflicts"]],
                         ["requirement_misfiled_as_ban"])

    def test_a_must_avoid_ban_is_still_a_ban(self):
        # The guard must not be fooled by an obligation marker inside a genuine
        # prohibition: 「必须避免…」 is a ban, and waiving it would silently stop
        # enforcing the ledger's real avoid list. Asserted on the predicate rather
        # than on a violation, because whether a LONG ban sentence can be charged
        # at all is a separate question -- `_anchors` yields only 3-char fragments
        # from these, below FORBID_MIN_ANCHOR, so none of them is chargeable
        # regardless of the guard. Bare-phrase bans are the chargeable shape and
        # are covered by `test_a_genuine_ban_still_fires_...`.
        for entry in ("必须避免使用「声音压得很低」这种写法",
                      "禁止再写声音压得很低",
                      "本章不要出现声音压得很低",
                      "结尾务必不要落在雪地上"):
            with self.subTest(entry=entry):
                self.assertFalse(accept._is_misfiled_requirement(entry))
                r = accept.contract_fulfilment(
                    dict(CARD, forbid=[entry]), FULFILLED + "顾峥的声音压得很低。",
                    _cfg())
                self.assertEqual(r["forbid_conflicts"], [],
                                 "a prohibition must never be waived as a requirement")

    def test_a_bare_phrase_ban_is_untouched_by_the_obligation_guard(self):
        # The overwhelmingly common shape: `forbid` entries come from the
        # used-element ledger as bare phrases with no markers at all.
        self.assertFalse(accept._is_misfiled_requirement("声音压得很低"))
        self.assertFalse(accept._is_misfiled_requirement(""))
        self.assertFalse(accept._is_misfiled_requirement(None))

    def test_every_obligation_marker_is_recognised(self):
        for m in accept.OBLIGATION_MARKERS:
            with self.subTest(marker=m):
                self.assertTrue(
                    accept._is_misfiled_requirement(f"本章{m}落在后门台阶上"))

    def test_a_waived_obligation_never_becomes_a_writer_directive(self):
        # Same reason as the Ch219 waiver: no chapter can repair its own card.
        card = dict(CARD, forbid=["第220章必须落在后门台阶上"])
        r = accept.acceptance_report(220, FULFILLED, card, _cfg())
        self.assertTrue(r["card_defects"])
        self.assertIn("硬性要求", r["card_defects"][0])
        for d in r["writer_directives_for_next_chapter"]:
            self.assertNotIn("后门台阶上", d)

    def test_abstract_field_is_unjudgeable_not_a_miss(self):        # An abstract card field is a CARD defect (arc.validate_card's job).
        # Charging the prose for it would make CCC unactionable -- no rewrite
        # can make an intent appear as an anchor.
        card = {"where": "的了在", "turn": "他意识到", "exit_hook": "他决定",
                "who": [], "beats": [], "forbid": []}
        r = accept.contract_fulfilment(card, FULFILLED, _cfg())
        self.assertTrue(r["passed"])
        self.assertEqual(r["judgeable"], 0)
        self.assertEqual(r["ccr"], 1.0)
        self.assertGreater(r["unjudgeable"], 0)

    def test_missing_person_is_advisory_not_hard(self):
        card = dict(CARD, who=["汤舒婷", "从未登场的人"])
        r = accept.contract_fulfilment(card, FULFILLED, _cfg())
        self.assertTrue(r["passed"], "a missing walk-on must not reject a chapter")
        self.assertLess(r["ccr"], 1.0, "but it must still cost CCR")

    def test_no_card_means_disabled_and_passing(self):
        for card in (None, {}, "not a dict"):
            with self.subTest(card=card):
                r = accept.contract_fulfilment(card, FULFILLED, _cfg())
                self.assertTrue(r["passed"])
                self.assertEqual(r["ccr"], 1.0)
                self.assertEqual(r["items"], [])

    def test_disable_switch_honoured(self):
        r = accept.contract_fulfilment(CARD, "短", _cfg(ccc_enabled=False))
        self.assertFalse(r["enabled"])
        self.assertTrue(r["passed"])

    def test_refusal_length_text_is_not_judged(self):
        r = accept.contract_fulfilment(CARD, "第1章\n\n很短。", _cfg())
        self.assertTrue(r["passed"])
        self.assertEqual(r["judgeable"], 0)

    def test_ccr_denominator_excludes_unjudgeable_items(self):
        card = {"where": "顾家老宅的后厨", "turn": "他意识到", "exit_hook": "他决定",
                "who": [], "beats": [], "forbid": []}
        r = accept.contract_fulfilment(card, FULFILLED, _cfg())
        self.assertEqual(r["judgeable"], 1)
        self.assertEqual(r["fulfilled"], 1)
        self.assertEqual(r["ccr"], 1.0)


    def test_arabic_numeral_matches_cjk_in_where(self):
        card = {"where": "凌晨1点的急诊室走廊", "who": [], "turn": "",
                "exit_hook": "", "beats": [], "forbid": []}
        text = "第3章\n\n" + "凌晨一点零三分，她冲进急诊室走廊。" * 20
        r = accept.contract_fulfilment(card, text, _cfg())
        self.assertTrue(r["passed"], f"Arabic→CJK numeral should match; misses={r['hard_misses']}")

    def test_compound_location_matches_when_components_are_separate(self):
        card = {"where": "鼎成科技数据中心外围废弃水塔顶", "who": [], "turn": "",
                "exit_hook": "", "beats": [], "forbid": []}
        text = "第7章\n\n" + "他爬上鼎成科技的数据中心旁边那座废弃水塔。" * 20
        r = accept.contract_fulfilment(card, text, _cfg())
        self.assertTrue(r["passed"], f"compound location with separate components should match; misses={r['hard_misses']}")


    def test_payoff_reaction_soft_check_lowers_ccr_on_miss(self):
        card = dict(CARD, payoff_reaction="马叔烟灰散开，愣了三秒")
        text = ("第10章\n\n"
                + "汤舒婷推开顾家老宅后厨的门。" * 20
                + "她在灶台夹层里摸到一枚刻字的铜钥匙。" * 20
                + "顾峥站在门口看着她。" * 20
                + "刻字与母亲遗物对上了。" * 20
                + "铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死。")
        r = accept.contract_fulfilment(card, text, _cfg())
        self.assertTrue(r["passed"], "payoff_reaction miss must not hard-block")
        miss_fields = {i["field"] for i in r["missing"]}
        self.assertIn("payoff_reaction", miss_fields)
        self.assertLess(r["ccr"], 1.0)

    def test_payoff_reaction_hit_when_present(self):
        card = dict(CARD, payoff_reaction="顾峥愣了三秒")
        text = FULFILLED + "顾峥愣了三秒，手里的烟差点掉在地上。"
        r = accept.contract_fulfilment(card, text, _cfg())
        hit_fields = {i["field"] for i in r["items"] if i["verdict"] == "hit"}
        self.assertIn("payoff_reaction", hit_fields)


class CitationCheckTest(unittest.TestCase):

    TEXT = "第3章\n\n她在灶台夹层里摸到一枚刻字的铜钥匙，指腹蹭过那道浅痕。"

    def test_a_real_quote_survives(self):
        r = accept.citation_check([{"issue": "x", "quote": "摸到一枚刻字的铜钥匙"}], self.TEXT)
        self.assertEqual(len(r["kept"]), 1)
        self.assertEqual(r["drop_rate"], 0.0)

    def test_a_fabricated_quote_is_dropped(self):
        r = accept.citation_check([{"issue": "x", "quote": "主角骑着龙飞走了"}], self.TEXT)
        self.assertEqual(r["kept"], [])
        self.assertEqual(r["dropped"][0]["_drop_reason"], "quote_not_in_chapter")

    def test_an_uncited_claim_is_dropped(self):
        # The whole point: an assertion the reviewer will not back with the text
        # is exactly the assertion that used to force a replan on nothing.
        r = accept.citation_check([{"issue": "prose feels flat"}], self.TEXT)
        self.assertEqual(r["kept"], [])
        self.assertEqual(r["dropped"][0]["_drop_reason"], "uncited")

    def test_punctuation_and_quote_marks_are_tolerated(self):
        r = accept.citation_check(
            [{"issue": "x", "quote": "“摸到一枚刻字的铜钥匙，指腹蹭过那道浅痕”"}], self.TEXT)
        self.assertEqual(len(r["kept"]), 1)

    def test_paraphrase_is_not_tolerated(self):
        # No fuzzy fallback here on purpose: a near-miss quote is a fabrication
        # with good luck, whereas a near-miss BEAT was still staged.
        r = accept.citation_check([{"issue": "x", "quote": "摸到了一把刻着字的铜钥匙"}], self.TEXT)
        self.assertEqual(r["kept"], [])

    def test_alternate_quote_keys_are_read(self):
        for key in ("evidence", "excerpt", "原文"):
            with self.subTest(key=key):
                r = accept.citation_check([{"issue": "x", key: "刻字的铜钥匙"}], self.TEXT)
                self.assertEqual(len(r["kept"]), 1)

    def test_empty_input_is_a_clean_zero(self):
        for claims in (None, [], ["not a dict"]):
            with self.subTest(claims=claims):
                r = accept.citation_check(claims, self.TEXT)
                self.assertEqual(r["total"], 0)
                self.assertEqual(r["drop_rate"], 0.0)


class AcceptanceReportTest(unittest.TestCase):
    """The report must be readable by the v1 ruler with no translation layer."""

    def _report(self, text=FULFILLED, card=CARD, **kw):
        return accept.acceptance_report(10, text, card, _cfg(**kw.pop("novel", {})), **kw)

    def test_clean_chapter_is_accepted_with_no_reasons(self):
        r = self._report()
        self.assertTrue(r["accepted"])
        self.assertEqual(r["block_reasons"], [])

    def test_the_ruler_reads_the_payload_directly(self):
        r = self._report()
        self.assertEqual(quality.hard_block_reasons(r, _cfg()), r["block_reasons"])

    def test_no_self_score_key_is_emitted(self):
        # An absent key is honest; a fabricated 8.0 would be laundered into
        # every downstream average and into `novel.py stats`.
        self.assertNotIn("score", self._report())

    def test_ccc_failure_becomes_a_gate_reject_the_ruler_can_see(self):
        text = "第10章\n\n" + "她坐在客厅里想事情，什么也没发生。" * 60
        r = self._report(text=text)
        self.assertIn("contract_fulfilment",
                      [g["gate"] for g in r["gate_rejects"]])
        self.assertFalse(r["accepted"])
        self.assertTrue(any(x.startswith("gate_rejects") for x in r["block_reasons"]))

    def test_length_band_is_stored_under_the_key_the_ruler_reads(self):
        # `length_band_check` -> `length_band`. Getting this wrong makes the gate
        # silently invisible to the ruler.
        self.assertIn("length_band", self._report())

    def test_directives_are_collected_for_the_next_chapter(self):
        text = "第10章\n\n" + "她坐在客厅里想事情，什么也没发生。" * 60
        r = self._report(text=text)
        self.assertTrue(r["writer_directives_for_next_chapter"])

    def test_engine_tag_is_present_so_mixed_corpora_stay_separable(self):
        self.assertEqual(self._report()["engine"], "v2")


class EmDashTrendWiringTest(unittest.TestCase):
    """`style_health`'s em-dash TREND term, wired into v2 2026-07-28.

    **Every case here goes through the real seam** — real `chapter_metrics` rows
    in a real SQLite file, read back by `accept._em_history` inside
    `acceptance_report`. A test that handed `em_history=` to `quality.style_health`
    directly would pass just as happily on the engine as it stood yesterday, which
    never opened the table at all. That is the same self-deception the
    fingerprint-library and opening-route wirings each had to be re-tested for.

    The term is silent without a baseline, so for two months v2 ran a strictly
    smaller gate than `quality.py` documents and nothing said so: an absent
    history reads exactly like a healthy trend.
    """

    # A rise off a low mean: mean(1.0,1.2,1.1) = 1.1, so a draft at ~4/k is a
    # >2x rise while still well under `style_em_dash_per_kchar_warn` (6.0). That
    # gap is the entire case the static tier cannot see.
    PRIOR = [(7, 1.0), (8, 1.2), (9, 1.1)]

    def setUp(self):
        import engine.store as store
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from engine.config import Paths
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        p = Paths(
            book=root / "book.md", state=root / "state.md", title=root / "t.txt",
            bible=root / "b.md", characters=root / "c.md", timeline=root / "tl.md",
            threads=root / "th.md", volume_plan=root / "vp.md",
            compass=root / "cp.md", voices=root / "vs.md", voice=root / "v.md",
            contract=root / "ct.md", glossary=root / "g.md",
            chapters_dir=root / "chapters", logs_dir=root / "logs",
            database=root / "story_state.db",
        )
        p.logs_dir.mkdir(parents=True, exist_ok=True)
        self.conn = store.init_db(p)
        self.addCleanup(self.conn.close_current)

    def _seed(self, rows=None):
        """Put real rows in the real table.

        Raw SQL rather than `writing.update_structured_state`: the seam under test
        is `chapter_metrics` -> `store.recent_metrics` -> `accept._em_history`, and
        the writer needs a plan/extraction/review triple that has nothing to do
        with it. The two columns touched here are the two that side reads.
        """
        for ch, em in (rows if rows is not None else self.PRIOR):
            self.conn.execute(
                "INSERT INTO chapter_metrics (chapter, em_dash_per_kchar, "
                "created_at) VALUES (?, ?, datetime('now'))", (ch, em))
        self.conn.commit()

    def _text(self, em_per_kchar):
        """A ~2000-char clean chapter carrying a chosen em-dash density."""
        base = "汤舒婷推开顾家老宅后厨的门，她在灶台夹层里摸到一枚刻字的铜钥匙。" * 60
        n = round(em_per_kchar * len(base) / 1000.0)
        return "第10章 铜钥匙\n\n" + base + "——" * n

    def _report(self, text, conn=None):
        return accept.acceptance_report(10, text, None, _cfg(), conn=conn)

    def test_the_trend_fires_on_a_rise_the_static_tier_cannot_see(self):
        self._seed()
        r = self._report(self._text(4.0), conn=self.conn)
        flags = r["style_health"]["flags"]
        self.assertTrue(any(f.startswith("em_dash_trend_rise") for f in flags), flags)
        # Proof it is the TREND and not the static tier: 4.0/k is under warn 6.0.
        self.assertFalse(any(f.startswith(("em_dash_high", "em_dash_overload"))
                             for f in flags), flags)
        self.assertEqual(r["style_health"]["metrics"]["em_dash_recent_mean"], 1.1)

    def test_the_directive_reaches_the_next_chapters_writer(self):
        # The whole payoff. `v2/run.py` injects
        # `writer_directives_for_next_chapter`; a flag nobody is told about buys
        # nothing.
        self._seed()
        r = self._report(self._text(4.0), conn=self.conn)
        self.assertTrue(any("破折号" in d
                            for d in r["writer_directives_for_next_chapter"]))

    def test_without_a_conn_the_term_stays_silent(self):
        # Offline callers (tools, tests) pass no conn. Silence is the honest
        # answer there -- an unmeasured trend must not be reported as a rising one.
        r = self._report(self._text(4.0))
        self.assertFalse(any(f.startswith("em_dash_trend")
                             for f in r["style_health"]["flags"]))
        self.assertNotIn("em_dash_recent_mean", r["style_health"]["metrics"])

    def test_an_unreadable_conn_degrades_to_silence_rather_than_aborting(self):
        # The term is advisory and has never blocked, so a metrics read that
        # fails must not take the chapter down with it. Asserting the report is
        # still produced, not merely that no exception escaped.
        class _Broken:
            def execute(self, *a, **k):
                raise RuntimeError("no such table")

        r = self._report(self._text(4.0), conn=_Broken())
        self.assertTrue(r["accepted"])
        self.assertFalse(any(f.startswith("em_dash_trend")
                             for f in r["style_health"]["flags"]))

    def test_a_single_prior_chapter_is_not_a_baseline(self):
        # `em_dash_penalty` needs >=2 points; one chapter's density is noise, and
        # treating it as a mean would fire the trend on chapter 2 of every book.
        self._seed([(9, 1.0)])
        r = self._report(self._text(4.0), conn=self.conn)
        self.assertFalse(any(f.startswith("em_dash_trend")
                             for f in r["style_health"]["flags"]))

    def test_this_chapters_own_row_is_never_its_own_baseline(self):
        # A rescued or resumed chapter already has a `chapter_metrics` row from
        # the previous attempt. Left in, the chapter is compared against itself,
        # the ratio collapses to ~1.0, and the term goes quiet exactly when the
        # engine is retrying a chapter it already found suspect.
        self._seed(self.PRIOR + [(10, 4.0), (11, 4.0)])
        r = self._report(self._text(4.0), conn=self.conn)
        self.assertEqual(r["style_health"]["metrics"]["em_dash_recent_mean"], 1.1)
        self.assertTrue(any(f.startswith("em_dash_trend_rise")
                            for f in r["style_health"]["flags"]))

    def test_wiring_it_does_not_move_the_ruler_on_the_measured_corpus(self):
        # Measured on all 63 archived v2 round-0 drafts: the term fires 4 times
        # and crosses `style_penalty_block` zero times. This pins the arithmetic
        # that makes that true -- the trend charge for a <2.5x rise is 0.5, which
        # cannot reach the 2.0 block line on its own.
        self._seed()
        r = self._report(self._text(4.0), conn=self.conn)
        self.assertLess(r["style_health"]["penalty"], 2.0)
        self.assertEqual(quality.hard_block_reasons(r, _cfg()), [])


class AdvisoryGateDirectivesTest(unittest.TestCase):
    """Advisory gates run inside acceptance_report and contribute directives."""

    def _report(self, text, **kw):
        return accept.acceptance_report(5, text, CARD, _cfg(), **kw)

    def test_ai_flavor_health_appears_in_report(self):
        r = self._report(FULFILLED)
        self.assertIn("ai_flavor_health", r)
        self.assertIsInstance(r["ai_flavor_health"], dict)

    def test_advisory_directives_aggregated(self):
        ai_heavy = (
            "第5章 铜钥匙\n\n"
            + "汤舒婷推开顾家老宅后厨的门。" * 10
            + "她在灶台夹层里摸到一枚刻字的铜钥匙。" * 10
            + "顾峥站在门口看着她。" * 10
            + "他的心中涌起一阵复杂的情绪。" * 40
            + "她的眼眶微微泛红。" * 20
            + "刻字与母亲遗物对上了。" * 10
            + "她把铜钥匙插进阁楼那把锁，锁芯只转了半圈就卡死。"
        )
        r = self._report(ai_heavy)
        afh = r.get("ai_flavor_health") or {}
        if afh.get("directives"):
            wd_joined = "\n".join(r["writer_directives_for_next_chapter"])
            self.assertTrue(
                any(d in wd_joined for d in afh["directives"]),
                "advisory directives must be aggregated into "
                "writer_directives_for_next_chapter (possibly with a priority prefix)")

    def test_advisory_gates_never_block(self):
        r = self._report(FULFILLED)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["block_reasons"], [])

    def test_dialogue_health_appears_in_report(self):
        r = self._report(FULFILLED)
        self.assertIn("dialogue_health", r)

    def test_intra_chapter_repetition_appears(self):
        r = self._report(FULFILLED)
        self.assertIn("intra_chapter_repetition", r)

    def test_prose_texture_appears(self):
        r = self._report(FULFILLED)
        self.assertIn("prose_texture", r)

    def test_paragraph_shape_health_appears(self):
        r = self._report(FULFILLED)
        self.assertIn("paragraph_shape_health", r)


if __name__ == "__main__":
    unittest.main()
