"""StoryState projection: the cache prefix, the budget, and honest truncation.

Everything here is a pure-function test, because everything in `canon.py` that
can be wrong is a pure function. The failure mode this file guards is not a
crash — it is a projection that reads as complete and is not, which is how
`volume_plan.md` starved the mid-book for 40 chapters (LESSONS §6).
"""
import dataclasses
import unittest

import engine.loop as canon


def _threads(n, prefix="伏线"):
    return [{"id": f"t{i}", "description": f"{prefix}{i}" + "内容" * 10}
            for i in range(n)]


class ClipTest(unittest.TestCase):

    def test_a_clipped_string_says_how_much_it_lost(self):
        out = canon._clip("字" * 500, 200)
        self.assertLessEqual(len(out), 200)
        self.assertIn("截断", out)

    def test_a_fitting_string_is_untouched(self):
        self.assertEqual(canon._clip("短文本", 200), "短文本")

    def test_no_room_means_nothing_fits_not_no_limit(self):
        # Measured on this module's first run: reading `cap <= 0` as "unlimited"
        # let `facts` ship 105k against a 2k cap, because a long contract drove
        # the remaining room negative. Both readings are the same integer; only
        # one of them is a budget.
        self.assertEqual(canon._clip("字" * 500, 0), "")
        self.assertEqual(canon._clip("字" * 500, -80), "")

    def test_a_cap_too_small_for_the_marker_yields_nothing_not_a_bare_marker(self):
        self.assertEqual(canon._clip("字" * 500, 5), "")

    def test_items_are_dropped_whole_not_halved(self):
        items = [f"- 第{i}条" + "内容" * 20 for i in range(10)]
        out = canon._clip_items(items, 200)
        self.assertLessEqual(len(out), 200)
        for line in out.split("\n"):
            if line.startswith("- "):
                # Every surviving line must be one of the originals, entire.
                self.assertIn(line, items)
        self.assertIn("未列出", out)

    def test_the_drop_marker_counts_what_is_missing(self):
        items = [f"- {i}" + "内" * 40 for i in range(10)]
        out = canon._clip_items(items, 250)
        kept = sum(1 for line in out.split("\n") if line.startswith("- "))
        self.assertIn(canon.DROP_MARK.format(n=10 - kept), out)

    def test_everything_fitting_means_no_marker(self):
        out = canon._clip_items(["- a", "- b"], 500)
        self.assertEqual(out, "- a\n- b")


class SectionTest(unittest.TestCase):

    def test_an_empty_section_emits_no_header(self):
        # "## 伏线" with nothing under it asserts there are no open threads.
        self.assertEqual(canon.Section("threads", "", False).render(), "")
        self.assertEqual(canon.Section("threads", "   ", False).render(), "")

    def test_a_present_section_is_headed(self):
        self.assertIn("## 未结伏线", canon.Section("threads", "- x", False).render())


class LayeringTest(unittest.TestCase):
    """The cache strategy is an ordering property, so it gets an ordering test."""

    def _state(self, **kw):
        base = dict(brief="纲要", bible="世界", characters="人物",
                    voice="声音", card={"where": "后厨"}, rag="原文")
        base.update(kw)
        return canon.build_story_state(7, **base)

    def test_stable_comes_first_and_volatile_never_precedes_it(self):
        text = self._state().render()
        self.assertLess(text.index(canon.STABLE_HEADER),
                        text.index(canon.VOLATILE_HEADER))

    def test_the_stable_prefix_is_byte_identical_across_chapters(self):
        a = canon.build_story_state(7, brief="纲要", bible="世界", voice="声音",
                        card={"where": "甲地"}, rag="A")
        b = canon.build_story_state(88, brief="纲要", bible="世界", voice="声音",
                        card={"where": "乙地"}, rag="B")
        self.assertEqual(a.stable_prefix(), b.stable_prefix())
        self.assertNotEqual(a.volatile_block(), b.volatile_block())

    def test_a_changed_fact_moves_the_prefix_and_nothing_else_does(self):
        a = self._state()
        self.assertEqual(a.stable_prefix(), self._state(card={"where": "别处"}).stable_prefix())
        self.assertNotEqual(a.stable_prefix(), self._state(bible="改了").stable_prefix())

    def test_every_section_is_declared_stable_or_volatile_exactly_once(self):
        keys = [s.key for s in self._state().sections]
        self.assertEqual(sorted(keys),
                         sorted(canon.STABLE_SECTIONS + canon.VOLATILE_SECTIONS))
        for s in self._state().sections:
            self.assertEqual(s.stable, s.key in canon.STABLE_SECTIONS)

    def test_sizes_reports_the_split(self):
        sizes = self._state().sizes()
        self.assertGreater(sizes["_stable"], 0)
        self.assertGreater(sizes["_volatile"], 0)
        self.assertGreaterEqual(sizes["_total"], sizes["_stable"] + sizes["_volatile"])


class BudgetTest(unittest.TestCase):

    def test_the_declared_budget_is_about_16k(self):
        # The number v2 claims over v1's 80k across four builders. If a section is
        # added or a cap raised, this is the assertion that makes it a decision.
        # 15500 -> 16300: `focus` (800) was added once B3 found that nothing in
        # the projection carried the protagonist's standing state. v1 carries it
        # in `state.md`, which canon never reads, so without this section every
        # chapter opened with a protagonist whose situation had been forgotten.
        # 16300 -> 17400: `route` (600) + `opening` (500) wired
        # `memory/opening_route.md`, which `adopt-trial` writes and v2 had never
        # read. Both are empty on a book with no adopted route — the declared cap
        # rises, the shipped bytes only rise where the user asked for them.
        # 17400 -> 18400: `threads` 1000 -> 2000 for mid-to-late-book thread recall
        self.assertEqual(sum(canon.BUDGET.values()), 18400)

    def test_a_flood_of_input_still_lands_near_budget(self):
        st = canon.build_story_state(
            7,
            brief="纲" * 50000, bible="界" * 50000, characters="人" * 50000,
            contract="约" * 5000, voice="声" * 50000, voices="话" * 50000,
            card={"where": "地" * 5000, "beats": ["拍" * 2000] * 20},
            open_threads=_threads(200), events=[
                {"chapter": i, "payload": {"description": "事" * 300}}
                for i in range(200)],
            used_elements=["元素" * 100] * 200,
            constraints=[{"constraint": "约束" * 200}] * 50,
            rag="原" * 50000,
        )
        total = len(st.render())
        self.assertLess(total, sum(canon.BUDGET.values()) * 1.15,
                        f"projection overran its budget: {st.sizes()}")

    def test_every_section_respects_its_own_cap(self):
        st = canon.build_story_state(
            7, brief="纲" * 9000, bible="界" * 9000, characters="人" * 9000,
            voice="声" * 9000, card={"where": "地" * 9000},
            open_threads=_threads(200), rag="原" * 9000,
            events=[{"chapter": 1, "payload": {"description": "事" * 500}}] * 50,
            used_elements=["元" * 500] * 50)
        for s in st.sections:
            if s.body:
                self.assertLessEqual(len(s.body), canon.BUDGET[s.key] + 40,
                                     f"section {s.key} overran")


class FactsTest(unittest.TestCase):

    def test_the_hard_contract_survives_a_flood_of_worldbuilding(self):
        # The contract is the only part of `facts` that can fail acceptance.
        # Clipping it to fit more scenery trades a blocking fact for a decorative one.
        body = canon.project_facts("界" * 40000, "人" * 40000, "禁止出现任何异能")
        self.assertIn("禁止出现任何异能", body)

    def test_no_contract_means_no_empty_heading(self):
        self.assertNotIn("硬约束", canon.project_facts("世界", "人物", ""))


class CardTest(unittest.TestCase):

    def test_fields_are_labelled_not_dumped_as_json(self):
        body = canon.project_card({"where": "后厨", "exit_hook": "锁芯卡死"})
        self.assertIn("地点：后厨", body)
        self.assertNotIn("{", body)

    def test_forbid_is_emitted_last_and_itemised(self):
        body = canon.project_card(
            {"where": "后厨", "exit_hook": "钩子", "forbid": ["声音压得很低", "月光"]})
        self.assertIn("声音压得很低", body)
        self.assertGreater(body.index("本章禁止"), body.index("地点"))

    def test_forbid_is_not_clipped_away_by_a_huge_beat_list(self):
        body = canon.project_card({"beats": ["拍" * 900] * 8, "forbid": ["唯一禁令"]})
        self.assertIn("唯一禁令", body)

    def test_no_card_falls_back_to_the_arc_note_rather_than_an_empty_section(self):
        self.assertEqual(canon.project_card(None, "第三弧：收束"), "第三弧：收束")
        self.assertEqual(canon.project_card(None, ""), "")

    def test_list_fields_are_flattened_readably(self):
        self.assertIn("在场：汤舒婷；顾峥",
                      canon.project_card({"who": ["汤舒婷", "顾峥"]}))


class FocusTest(unittest.TestCase):
    """The one section that carries what the last chapter left the protagonist in.

    v1 renders it into `state.md`; v2 reads the `chapter_extraction` payload
    instead, so this projection is the whole memory of "where we stand".
    """

    def test_a_dict_is_labelled_per_key_not_dumped_as_json(self):
        body = canon.project_focus({"目标": "拿到病历", "恐惧": "被顾家发现"})
        self.assertIn("- 目标：拿到病历", body)
        self.assertNotIn("{", body)

    def test_list_values_are_flattened(self):
        self.assertIn("甲；乙", canon.project_focus({"资源": ["甲", "乙"]}))

    def test_v1s_markdown_shape_still_projects(self):
        # v1 asked for <=600 chars of markdown. A fork resuming from a v1 book
        # meets exactly that on its first v2 chapter.
        self.assertIn("目标：拿到病历", canon.project_focus("- 目标：拿到病历"))

    def test_nothing_known_is_empty_not_a_header(self):
        self.assertEqual(canon.project_focus(None), "")
        self.assertEqual(canon.project_focus({}), "")
        self.assertEqual(canon.project_focus({"目标": ""}), "")

    def test_it_is_volatile_so_the_prefix_does_not_move_with_it(self):
        a = canon.build_story_state(7, bible="世界", protagonist={"目标": "甲"})
        b = canon.build_story_state(7, bible="世界", protagonist={"目标": "乙"})
        self.assertEqual(a.stable_prefix(), b.stable_prefix())
        self.assertNotEqual(a.volatile_block(), b.volatile_block())


class ThreadsTest(unittest.TestCase):

    def test_overdue_promises_come_first(self):
        body = canon.project_threads(
            [{"id": "a", "description": "普通伏线"}],
            [{"id": "b", "description": "逾期承诺", "overdue_by": 3}])
        self.assertLess(body.index("逾期承诺"), body.index("普通伏线"))
        self.assertIn("[逾期3章]", body)

    def test_a_thread_listed_as_both_appears_once(self):
        t = {"id": "x", "description": "同一条", "overdue_by": 2}
        body = canon.project_threads([t], [t])
        self.assertEqual(body.count("同一条"), 1)

    def test_a_deadline_is_shown_when_present(self):
        self.assertIn("[第40章前]",
                      canon.project_threads([{"id": "a", "description": "d",
                                              "due_chapter": 40}]))

    def test_junk_rows_do_not_crash_the_projection(self):
        self.assertEqual(canon.project_threads([None, {}, {"description": ""}, 7]), "")


class RecentTest(unittest.TestCase):

    def test_events_read_forward_though_the_store_returns_them_backwards(self):
        events = [{"chapter": 3, "payload": {"description": "最近的"}},
                  {"chapter": 1, "payload": {"description": "最早的"}}]
        body = canon.project_recent(events)
        self.assertLess(body.index("最早的"), body.index("最近的"))

    def test_metrics_are_trimmed_to_the_columns_that_matter(self):
        body = canon.project_recent([], [{"chapter": 1, "chars": 3000,
                                          "score": 9.9, "plan_score": 8}])
        self.assertIn("chars", body)
        self.assertNotIn("score", body, "the self-score has no place in v2 context")


class LedgerTest(unittest.TestCase):

    def test_obligations_outrank_the_avoidance_list_when_the_budget_bites(self):
        body = canon.project_ledger(["元素" * 200] * 20,
                                    [{"constraint": "必须兑现铜钥匙"}])
        self.assertIn("必须兑现铜钥匙", body)

    def test_constraints_are_marked_as_mandatory(self):
        self.assertIn("[必须]", canon.project_ledger([], [{"constraint": "x"}]))


class DeltaTest(unittest.TestCase):

    def test_a_missing_field_is_empty_not_an_exception(self):
        d = canon.ChapterDelta.from_payload({"events": [{"description": "e"}]})
        self.assertEqual(len(d.events), 1)
        self.assertEqual(d.entities, ())
        self.assertTrue(canon.ChapterDelta.from_payload(None).empty)
        self.assertTrue(canon.ChapterDelta.from_payload("garbage").empty)

    def test_non_dict_rows_are_dropped_not_persisted(self):
        d = canon.ChapterDelta.from_payload(
            {"entities": [{"name": "a"}, "junk", None, 5]})
        self.assertEqual(len(d.entities), 1)

    def test_both_spellings_of_the_directions_field_are_read(self):
        self.assertEqual(
            canon.ChapterDelta.from_payload({"next_12_directions": ["a"]}).next_directions,
            ("a",))
        self.assertEqual(
            canon.ChapterDelta.from_payload({"next_directions": ["b"]}).next_directions,
            ("b",))

    def test_round_trips_into_the_v1_extraction_schema(self):
        # `apply_delta` hands this straight to `writing.update_structured_state`,
        # so the key spellings are a contract, not a detail.
        ex = canon.ChapterDelta.from_payload({
            "events": [{"description": "e"}], "entities": [{"name": "n"}],
            "threads": [{"id": "t"}], "protagonist_state": {"hp": 1},
            "next_12_directions": ["d"]}).as_extraction()
        self.assertEqual(sorted(ex), sorted(
            ["events", "entities", "threads", "protagonist_state",
             "next_12_directions"]))
        self.assertEqual(ex["protagonist_state"], {"hp": 1})


ROUTE_MD = """# 开篇试写最佳路线：变体B·冷开场

- trial_score: 8.7
- variant_path: logs/opening_trials/t1/b

## 核心卖点
主角能听见死者最后一句话，但每听一次就忘掉一个活人。

## 差异化
同类作品把代价写成寿命，本作把代价写成关系。

## 读者承诺
每卷必有一次「他为了救人主动忘掉某人」的选择。

## 推荐书名
- 遗忘代价
- 最后一句话
- 我听见你死时说的话

## 推荐简介
- 他能听见死者的遗言，代价是忘记生者。

## 正式连载前修改指令
- 第一章开场删掉天气描写，直接从尸体旁的第一句遗言进。
- 第三章之前不要解释规则来源。
"""


class OpeningRouteTest(unittest.TestCase):
    """`adopt-trial`'s output, projected instead of pasted.

    The gap this closes: `novel.py adopt-trial` wrote `memory/opening_route.md`
    and v2 read nothing, so the command was inert on a v2 book. The reason it is
    projected rather than pasted is that the file is a MIXTURE — v1's
    `cacheable_prefix` shipped ten candidate titles and five blurbs in every
    chapter prompt of the whole book.
    """

    def _state(self, chapter=1, route=ROUTE_MD):
        return canon.build_story_state(chapter, brief="纲要", bible="世界", opening_route=route)

    def test_positioning_reaches_the_cacheable_prefix(self):
        head = self._state().stable_prefix()
        for kept in ("核心卖点", "差异化", "读者承诺"):
            self.assertIn(kept, head)
        self.assertIn("每听一次就忘掉一个活人", head)

    def test_packaging_and_bookkeeping_never_reach_the_prompt(self):
        # Titles and blurbs are `package.py`'s job, and trial_score is a number
        # about the trial, not about the story. Anything before the first `## `
        # has no heading and is dropped with them.
        whole = self._state().render()
        for dropped in ("推荐书名", "推荐简介", "遗忘代价", "trial_score",
                        "variant_path", "8.7"):
            self.assertNotIn(dropped, whole)

    def test_revision_directives_are_volatile_and_expire(self):
        early = self._state(chapter=1)
        self.assertIn("删掉天气描写", early.volatile_block())
        self.assertNotIn("删掉天气描写", early.stable_prefix())
        # Past the span they are an instruction about a chapter that already
        # shipped. Empty, and by rule 2 that means no header either.
        late = self._state(chapter=canon.OPENING_ROUTE_SPAN + 1)
        self.assertEqual(late.section("opening"), "")
        self.assertNotIn("开篇执行指令", late.render())

    def test_the_expiry_does_not_touch_the_cached_head(self):
        # The load-bearing one. If the directives had gone into `stable`, the head
        # would differ between chapter 3 and chapter 4 while `stable_key` — a hash
        # of FILES — still claimed it had not moved. That is the one failure mode
        # the key exists to prevent, so the split is what keeps the key honest.
        a = self._state(chapter=1).stable_prefix()
        b = self._state(chapter=canon.OPENING_ROUTE_SPAN + 1).stable_prefix()
        self.assertEqual(a, b)
        self.assertNotEqual(self._state(chapter=1).volatile_block(),
                            self._state(chapter=99).volatile_block())

    def test_no_adopted_route_costs_exactly_zero_bytes(self):
        # Most books never run `trial`. Wiring a capability must not tax them.
        without = canon.build_story_state(1, brief="纲要", bible="世界")
        self.assertEqual(without.section("route"), "")
        self.assertEqual(without.section("opening"), "")
        self.assertNotIn("作品定位", without.render())

    def test_headings_match_by_containment_so_relabelling_cannot_empty_it(self):
        # `trial.py` writes these as prose labels. Equality matching would let a
        # one-word edit there silently empty this section, and an empty section
        # emits no header — so the loss would be invisible.
        route = ROUTE_MD.replace("## 核心卖点", "## 核心卖点（本作）")
        self.assertIn("每听一次就忘掉一个活人",
                      canon.build_story_state(1, opening_route=route).stable_prefix())

    def test_a_long_route_drops_whole_blocks_and_says_so(self):
        route = ROUTE_MD.replace("同类作品把代价写成寿命，本作把代价写成关系。",
                                 "长" * 4000)
        body = canon.build_story_state(1, opening_route=route).section("route")
        self.assertLessEqual(len(body), canon.BUDGET["route"] + 40)
        self.assertIn("未列出", body)

    def test_load_picks_the_file_up_from_disk(self):
        # The seam every other test in this class skips: `build` takes the route
        # as an argument, so all of them would pass on an engine that never READS
        # the file — which is exactly the bug being fixed. This one goes through
        # `canon.load`, i.e. the path `adopt-trial` actually depends on.
        # `conn=None` is fine: `load` wraps every store call in `_safe`.
        import engine.config as config
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            d = Path(td)
            (d / "memory").mkdir()
            p = config.Paths(**{f.name: d / "memory" / f"{f.name}.md"
                                for f in dataclasses.fields(config.Paths)})
            canon.opening_route_path(p).write_text(ROUTE_MD, encoding="utf-8")
            state = canon.load_story_state(p, None, {}, 1)
            self.assertIn("每听一次就忘掉一个活人", state.stable_prefix())
            self.assertIn("删掉天气描写", state.volatile_block())
            self.assertNotIn("推荐书名", state.render())


class StableKeyTest(unittest.TestCase):

    class FakePaths:
        def __init__(self, d):
            self.bible = d / "bible.md"
            self.characters = d / "characters.md"
            self.voice = d / "voice.md"
            self.voices = d / "voices.md"
            self.contract = d / "contract.md"
            self.volume_plan = d / "volume_plan.md"

    def test_key_changes_when_a_source_file_changes(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            d = Path(td)
            p = self.FakePaths(d)
            p.bible.write_text("世界", encoding="utf-8")
            first = canon.stable_key(p)
            self.assertEqual(first, canon.stable_key(p))
            p.bible.write_text("改了的世界", encoding="utf-8")
            self.assertNotEqual(first, canon.stable_key(p))

    def test_missing_files_are_stable_not_random(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            p = self.FakePaths(Path(td))
            self.assertEqual(canon.stable_key(p), canon.stable_key(p))

    def test_adopting_a_route_moves_the_key(self):
        # `adopt-trial` rewrites the cacheable head, so the key MUST move: `run.py`
        # logs hit/miss off it, and a prefix that changed under an unchanged key is
        # a silent cache miss reported as a hit.
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            p = self.FakePaths(Path(td))
            p.bible.write_text("世界", encoding="utf-8")
            before = canon.stable_key(p)
            canon.opening_route_path(p).write_text(ROUTE_MD, encoding="utf-8")
            after = canon.stable_key(p)
            self.assertNotEqual(before, after)
            self.assertEqual(after, canon.stable_key(p))

    def test_the_route_path_is_where_memory_py_also_looks(self):
        # Two readers, one location, asserted by BEHAVIOUR rather than by
        # re-deriving the path here (a third derivation would be the bug). Write
        # the file where canon says it lives; `memory.opening_route_text` — what
        # `trial.py` / `package.py` still use — must find it there.
        import engine.bootstrap as memory
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            p = self.FakePaths(Path(td))
            self.assertEqual(memory.opening_route_text(p), "")
            canon.opening_route_path(p).write_text(ROUTE_MD, encoding="utf-8")
            self.assertIn("核心卖点", memory.opening_route_text(p))

    def test_a_paths_without_volume_plan_yields_no_route_rather_than_crashing(self):
        self.assertIsNone(canon.opening_route_path(object()))


if __name__ == "__main__":
    unittest.main()
