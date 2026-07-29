"""v2 planning: the card contract, the two-layer roll, and the cost of failure.

The thing that can go quietly wrong here is not a crash. It is a card that came
from somewhere other than an arc plan — a repair, a solo re-plan, or worst, a
fabrication — being reported as if the arc had produced it. Every CCR number in
the A/B is measured against these cards, so provenance is a tested property, and
so is the archive path the measurement tool reads.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import engine.store as store
from engine.checkpoint import load_checkpoint
from engine.config import Paths
from engine.plan import volume_transition_directive
import engine.plan as beat
import engine.loop as canon
from tests.conftest import make_paths


def _paths(root: Path) -> Paths:
    return make_paths(root, seed_files=True)


def _config(**over) -> dict:
    novel = {"max_chapters": 0, "arc_span": 10, "scene_dedupe_enabled": False,
             "plan_validate_deep": False}
    novel.update(over)
    return {"novel": novel, "api": {"metrics_enabled": False}}


def _card(ch: int, **over) -> dict:
    """A card an arc planner could plausibly have emitted.

    `opening_type` / `payoff_type` / `where` rotate with the chapter number,
    because a run of identical ones is precisely what `validate_card` rejects —
    a fixture without rotation makes every test a repair-path test by accident.
    """
    from engine.plan import OPENING_TYPES

    card = {
        "ch": ch, "title": f"第{ch}章", "where": f"县医院三楼旧档案室{ch}号房",
        "who": ["汤舒婷", "顾峥"], "pov_character": "汤舒婷",
        "wants": "拿到母亲的病历原件", "blocked_by": "档案室的铁柜被换了新锁",
        "turn": "顾峥把钥匙拍在柜面上", "payoff": "病历第三页少了一张化验单",
        "payoff_type": ("reveal", "reversal", "emotional")[ch % 3],
        "conflict_type": "institution",
        "beats": ["汤舒婷推开档案室的门", "顾峥把钥匙拍在柜面上", "两人翻到第三页"],
        "exit_hook": "走廊尽头的灯忽然全灭了",
        "opening_type": OPENING_TYPES[ch % len(OPENING_TYPES)],
        "forbid": ["声音压得很低"],
    }
    card.update(over)
    return card


class FakeCall:
    """Stands in for `llm.call_llm`. Records every call; answers by tag."""

    def __init__(self, **by_tag):
        self.by_tag = by_tag
        self.calls: list[dict] = []

    def __call__(self, client, paths, config, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        payload = self.by_tag.get(kw.get("tag"), {})
        if callable(payload):
            payload = payload(len([c for c in self.calls if c["tag"] == kw["tag"]]))
        return json.dumps(payload, ensure_ascii=False)

    def tags(self) -> list[str]:
        return [c["tag"] for c in self.calls]


# ---------------------------------------------------------------------------


class SystemPromptTest(unittest.TestCase):

    def test_the_arc_rules_have_one_home(self):
        # If this ever becomes a copy, the two engines drift and neither is wrong.
        from engine.plan import ARC_SYSTEM
        self.assertIn(ARC_SYSTEM, beat.ARC_SYSTEM_V2)
        self.assertIn("next_arc", beat.ARC_SYSTEM_V2)


class CheckpointContractTest(unittest.TestCase):

    def test_the_card_is_archived_where_the_ccr_tool_reads_it(self):
        from tools.ccr_baseline import CARD_CHECKPOINT
        self.assertEqual(beat.CARD_CHECKPOINT, CARD_CHECKPOINT)


class SkeletonTest(unittest.TestCase):

    def test_lines_outside_the_next_window_are_dropped_not_renumbered(self):
        # A line the model pinned to the wrong chapter is a guess about the wrong
        # chapter; sliding it into an empty slot would launder that into a plan.
        skel = beat.normalize_skeleton(
            {"intent": "收束", "chapters": [{"ch": 11, "line": "甲"},
                                            {"ch": 99, "line": "乙"}]},
            [11, 12])
        self.assertEqual(skel["chapters"], {"11": "甲"})

    def test_a_dict_shaped_skeleton_is_accepted_too(self):
        skel = beat.normalize_skeleton({"chapters": {"11": "甲"}}, [11])
        self.assertEqual(skel["chapters"], {"11": "甲"})

    def test_junk_is_none_not_an_empty_promise(self):
        self.assertIsNone(beat.normalize_skeleton(None, [1]))
        self.assertIsNone(beat.normalize_skeleton("nope", [1]))
        self.assertIsNone(beat.normalize_skeleton({"chapters": []}, [1]))

    def test_an_intent_with_no_lines_still_counts(self):
        self.assertEqual(beat.normalize_skeleton({"intent": "收束"}, [1])["intent"], "收束")

    def test_the_block_renders_only_the_chapters_being_planned(self):
        skel = {"intent": "推进", "chapters": {"11": "甲", "12": "乙"}}
        block = beat.skeleton_block(skel, [11])
        self.assertIn("Ch11: 甲", block)
        self.assertNotIn("乙", block)

    def test_no_skeleton_renders_to_nothing(self):
        self.assertEqual(beat.skeleton_block(None, [1]), "")
        self.assertEqual(beat.skeleton_block({"chapters": {}}, [1]), "")

    def test_the_previous_arcs_skeleton_is_found_one_span_back(self):
        store_data = {"arcs": {"1": {"next_skeleton": {"intent": "甲", "chapters": {}}}}}
        self.assertEqual(beat._previous_skeleton(store_data, 11, 10)["intent"], "甲")
        self.assertIsNone(beat._previous_skeleton(store_data, 21, 10))


class PromptTest(unittest.TestCase):

    def _state(self):
        return canon.build_story_state(11, brief="纲要", bible="世界", voice="声音",
                           open_threads=[{"id": "t", "description": "铜钥匙未收"}])

    def test_the_request_pins_the_exact_chapter_numbers(self):
        user = beat.arc_user_prompt(self._state(), [11, 12, 13])
        self.assertIn("[11, 12, 13]", user)

    def test_the_stable_half_is_not_in_the_user_message(self):
        # It rides as `cacheable_prefix`; duplicating it would double the prompt
        # and move the bytes the cache keys on.
        user = beat.arc_user_prompt(self._state(), [11])
        self.assertNotIn("世界", user)
        self.assertIn("铜钥匙未收", user)

    def test_absent_sections_emit_no_empty_headers(self):
        user = beat.arc_user_prompt(self._state(), [11])
        self.assertNotIn("卷纲", user)
        self.assertNotIn("上一弧留下的骨架", user)

    def test_a_prior_skeleton_is_carried_in(self):
        user = beat.arc_user_prompt(self._state(), [11], prev_skeleton="- Ch11: 甲")
        self.assertIn("上一弧留下的骨架", user)
        self.assertIn("- Ch11: 甲", user)


class GenerateArcTest(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self.root)
        self.conn = store.init_db(self.paths)
        self.addCleanup(self.conn.close_current)

    def _run(self, payload, *, start=1, end=3, config=None, state=None):
        call = FakeCall(arc_plan=payload)
        arc = beat.generate_arc(None, self.paths, self.conn, config or _config(),
                                start, end, state=state, call=call)
        return arc, call

    def test_cards_are_keyed_by_their_declared_chapter(self):
        arc, _ = self._run({"arc_intent": "推进",
                            "cards": [_card(1), _card(2), _card(3)]})
        self.assertEqual(sorted(arc["cards"]), [1, 2, 3])
        self.assertEqual(arc["intent"], "推进")
        self.assertEqual(arc["missing"], [])

    def test_a_skipped_chapter_is_reported_not_backfilled(self):
        arc, _ = self._run({"cards": [_card(1), _card(3)]})
        self.assertEqual(arc["missing"], [2])
        self.assertNotIn(2, arc["cards"])

    def test_the_next_skeleton_covers_the_following_window(self):
        arc, _ = self._run({"cards": [_card(1), _card(2), _card(3)],
                            "next_arc": {"intent": "下一弧",
                                         "chapters": [{"ch": 4, "line": "甲"},
                                                      {"ch": 6, "line": "丙"},
                                                      {"ch": 9, "line": "越界"}]}})
        self.assertEqual(sorted(arc["next_skeleton"]["chapters"]), ["4", "6"])

    def test_the_final_arc_leaves_no_skeleton(self):
        arc, call = self._run({"cards": [_card(1), _card(2), _card(3)],
                               "next_arc": {"chapters": [{"ch": 4, "line": "甲"}]}},
                              config=_config(max_chapters=3))
        self.assertIsNone(arc["next_skeleton"])
        self.assertIn("终章", call.calls[0]["user"])

    def test_the_stable_prefix_is_passed_as_the_cache_key_not_inlined(self):
        state = canon.build_story_state(1, brief="纲要", bible="世界")
        _, call = self._run({"cards": [_card(1)]}, start=1, end=1, state=state)
        self.assertEqual(call.calls[0]["cacheable_prefix"], state.stable_prefix())
        self.assertTrue(call.calls[0]["cacheable_prefix"])

    def test_no_cards_raises_rather_than_returning_an_empty_plan(self):
        with self.assertRaises(beat.BeatError):
            self._run({"arc_intent": "推进", "cards": []})

    def test_cards_that_all_fail_normalisation_raise(self):
        with self.assertRaises(beat.BeatError):
            self._run({"cards": [{"ch": 1, "title": "只有标题"}]})


class EnsureCardTest(unittest.TestCase):

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self.root)
        self.conn = store.init_db(self.paths)
        self.addCleanup(self.conn.close_current)

    def _ensure(self, ch, call, config=None):
        return beat.ensure_card(None, self.paths, self.conn, config or _config(),
                                ch, call=call)

    def _seed(self, *cards):
        from engine.plan import save_cards
        save_cards(self.paths, {"cards": {str(c["ch"]): c for c in cards}, "arcs": {}})

    def test_a_stored_card_costs_nothing(self):
        self._seed(_card(7))
        call = FakeCall()
        res = self._ensure(7, call)
        self.assertEqual(res.source, "stored")
        self.assertEqual(call.calls, [])
        self.assertFalse(res.degraded)

    def test_a_missing_card_plans_the_whole_arc_in_one_call(self):
        call = FakeCall(arc_plan={"arc_intent": "开局",
                                  "cards": [_card(c) for c in range(1, 11)]})
        res = self._ensure(1, call)
        self.assertEqual(res.source, "arc")
        self.assertEqual(call.tags(), ["arc_plan"])
        # The other nine chapters are now free.
        self.assertEqual(self._ensure(5, FakeCall()).source, "stored")

    def test_the_arc_never_replans_chapters_already_written(self):
        # Resuming mid-block (a fork): the window is Ch1-10 but Ch1-6 exist.
        call = FakeCall(arc_plan={"cards": [_card(c) for c in range(7, 11)]})
        self._ensure(7, call)
        self.assertIn("[7, 8, 9, 10]", call.calls[0]["user"])

    def test_the_card_is_archived_for_the_ccr_tool(self):
        self._seed(_card(7))
        self._ensure(7, FakeCall())
        archived = load_checkpoint(self.paths, 7, beat.CARD_CHECKPOINT)
        self.assertEqual(archived["turn"], _card(7)["turn"])

    def test_the_contract_reaches_the_writer_as_required_constraints(self):
        self._seed(_card(7))
        res = self._ensure(7, FakeCall())
        targets = [c["target"] for c in res.decision["required_constraints"]]
        self.assertIn(_card(7)["turn"], targets)
        self.assertIn(_card(7)["where"], targets)
        self.assertEqual(res.decision["planner"], "v2_beat")
        self.assertEqual(res.decision["card_source"], "stored")

    # --- the failure ladder ------------------------------------------------

    def _duplicate_of(self, prev, ch):
        """A card that repeats the previous chapter's opening AND its location —
        two of `validate_card`'s findings, neither of which the card can see."""
        return _card(ch, where=prev["where"], opening_type=prev["opening_type"])

    def test_a_repairable_card_costs_one_extra_call_and_says_so(self):
        prev = _card(6)
        self._seed(prev, self._duplicate_of(prev, 7))
        call = FakeCall(arc_card_repair=_card(7, opening_type="dialogue",
                                              where="停车场入口"))
        res = self._ensure(7, call)
        self.assertEqual(res.source, "repaired")
        self.assertEqual(call.tags(), ["arc_card_repair"])
        self.assertEqual(res.unresolved, ())
        self.assertTrue(res.degraded)
        self.assertEqual(res.card["where"], "停车场入口")

    def test_a_failed_repair_replans_the_chapter_alone_rather_than_faking_one(self):
        prev = _card(6)
        self._seed(prev, self._duplicate_of(prev, 7))
        call = FakeCall(arc_card_repair={"nonsense": True},
                        arc_plan={"cards": [_card(7, opening_type="aftermath",
                                                  where="值班室")]})
        res = self._ensure(7, call)
        self.assertEqual(res.source, "single")
        self.assertEqual(call.tags(), ["arc_card_repair", "arc_plan"])
        self.assertEqual(res.card["where"], "值班室")
        self.assertEqual(res.unresolved, ())

    def test_the_third_attempt_is_accepted_with_its_problems_carried_forward(self):
        # Every remaining problem is measured against chapters this attempt
        # cannot rewrite. A fourth try would be a latch, so the writer is told
        # instead -- and the result admits it never cleared them.
        prev = _card(6)
        dup = self._duplicate_of(prev, 7)
        self._seed(prev, dup)
        call = FakeCall(arc_card_repair=dup, arc_plan={"cards": [dup]})
        res = self._ensure(7, call)
        self.assertEqual(res.source, "single")
        self.assertTrue(res.unresolved)
        self.assertEqual(len(call.calls), 2, "no fourth attempt")
        joined = " ".join(str(c) for c in res.decision["required_constraints"])
        self.assertIn("规划未消除", joined)

    def test_a_chapter_the_arc_skipped_gets_its_own_call(self):
        call = FakeCall(arc_plan=lambda n: (
            {"cards": [_card(2)]} if n == 1 else {"cards": [_card(1)]}))
        res = self._ensure(1, call)
        self.assertEqual(res.source, "single")
        self.assertEqual(call.tags(), ["arc_plan", "arc_plan"])

    def test_a_chapter_that_cannot_be_planned_raises_instead_of_writing_blind(self):
        # The arc skips Ch1, and the solo re-plan comes back empty. There is no
        # committee under v2 to catch this, and a fabricated card would be scored
        # by CCR as though someone had planned it -- so the chapter stops.
        call = FakeCall(arc_plan=lambda n: (
            {"cards": [_card(2)]} if n == 1 else {"cards": []}))
        with self.assertRaises(beat.BeatError):
            self._ensure(1, call)

    def test_a_solo_replan_may_relabel_the_chapter_it_was_asked_for(self):
        # Inherited from `arc.generate_arc`: for a one-chapter request, a card
        # whose `ch` disagrees is still that chapter's card -- there is nowhere
        # else it could belong. Documented rather than assumed.
        call = FakeCall(arc_plan=lambda n: (
            {"cards": [_card(2)]} if n == 1 else {"cards": [_card(99)]}))
        res = self._ensure(1, call)
        self.assertEqual(res.card["ch"], 1)
        self.assertEqual(res.source, "single")

    # --- the repair is re-judged by the ruler that rejected it --------------

    def _same_skeleton(self, prev, ch, **over):
        """A card sharing every field `scene_similarity` reads, and nothing else.

        The skeleton is `card_to_plan`'s conflict/payoff/pressure/goal + beats,
        i.e. the card's `blocked_by`/`payoff`/`pressure`/`wants`/`beats`. `where`
        and `opening_type` are deliberately changed so the neighbour checks stay
        quiet and the ONLY finding is the similarity — otherwise the test would
        pass on the strength of a check the old recheck could already see.
        """
        return _card(ch, where=f"另一处完全不同的场地{ch}",
                     opening_type=prev["opening_type"], payoff_type="reversal",
                     wants=prev["wants"], blocked_by=prev["blocked_by"],
                     payoff=prev["payoff"], beats=list(prev["beats"]), **over)

    def _dedupe_config(self):
        return _config(scene_dedupe_enabled=True, scene_dedupe_sim_block=0.82)

    def test_a_scene_dedupe_block_is_what_the_first_pass_reports(self):
        # Fixture guard. If the similarity does not actually fire, every
        # assertion below passes for the wrong reason.
        prev = _card(6)
        dup = self._same_skeleton(prev, 7)
        dup["opening_type"] = _card(7)["opening_type"]  # keep the neighbour quiet
        self._seed(prev, dup)
        call = FakeCall(arc_card_repair=dup, arc_plan={"cards": [dup]})
        res = self._ensure(7, call, self._dedupe_config())
        self.assertTrue(any("相似度" in p for p in res.unresolved), res.unresolved)

    def test_a_repair_that_kept_the_skeleton_is_not_filed_as_fixed(self):
        # The omitted-argument defect: the old recheck called `validate_card`
        # without `scene_sim`, so a repair that changed the label and kept the
        # scene came back clean and shipped as "repaired".
        prev = _card(6)
        dup = self._same_skeleton(prev, 7)
        dup["opening_type"] = _card(7)["opening_type"]
        self._seed(prev, dup)
        # The "repair" changes the title and nothing the gate reads.
        still_dup = dict(dup, title="换个标题")
        good = _card(7, wants="改问护士长要交接班记录",
                     blocked_by="交接班记录被锁进院办",
                     payoff="记录里少了两小时",
                     beats=["汤舒婷截住护士长", "顾峥调出门禁记录", "两人比对时间"])
        call = FakeCall(arc_card_repair=still_dup, arc_plan={"cards": [good]})
        res = self._ensure(7, call, self._dedupe_config())
        self.assertEqual(res.source, "single", "a kept skeleton was filed as fixed")
        self.assertEqual(call.tags(), ["arc_card_repair", "arc_plan"])
        self.assertEqual(res.unresolved, ())

    def test_a_repair_that_changed_the_scene_is_accepted(self):
        # The other direction: the fix must not make every repair fall through
        # to a solo re-plan, which would double the cost of the repair path.
        prev = _card(6)
        dup = self._same_skeleton(prev, 7)
        dup["opening_type"] = _card(7)["opening_type"]
        self._seed(prev, dup)
        fixed = _card(7, wants="改问护士长要交接班记录",
                      blocked_by="交接班记录被锁进院办",
                      payoff="记录里少了两小时",
                      beats=["汤舒婷截住护士长", "顾峥调出门禁记录", "两人比对时间"])
        call = FakeCall(arc_card_repair=fixed)
        res = self._ensure(7, call, self._dedupe_config())
        self.assertEqual(res.source, "repaired")
        self.assertEqual(call.tags(), ["arc_card_repair"])

    def test_the_writer_gets_the_repaired_cards_advisories_not_the_broken_ones(self):
        # `advisories` become `required_constraints`, so carrying the pre-repair
        # card's advisories describes a card the writer never sees.
        prev = _card(6)
        self._seed(prev, self._duplicate_of(prev, 7))
        fixed = _card(7, opening_type="dialogue", where="停车场入口")
        call = FakeCall(arc_card_repair=fixed)
        seen: list[dict] = []

        def fake_continuity(conn, plan, chapter_num, config=None):
            seen.append(plan)
            return [] if plan.get("location") == "停车场入口" else ["旧卡片的告警"]

        import engine.plan as beat_mod
        orig = beat_mod.validate_plan_continuity
        beat_mod.validate_plan_continuity = fake_continuity
        self.addCleanup(setattr, beat_mod, "validate_plan_continuity", orig)
        res = self._ensure(7, call)
        self.assertEqual(res.source, "repaired")
        self.assertEqual(res.advisories, (), f"stale advisories carried: {res.advisories}")
        self.assertEqual(len(seen), 2, "the repaired card was never re-checked")

    def test_the_skeleton_survives_to_the_next_arc_call(self):
        from engine.plan import load_cards

        first = FakeCall(arc_plan={"cards": [_card(c) for c in range(1, 11)],
                                   "next_arc": {"intent": "第二弧",
                                                "chapters": [{"ch": 11, "line": "开庭"}]}})
        self._ensure(1, first)
        self.assertEqual(
            load_cards(self.paths)["arcs"]["1"]["next_skeleton"]["chapters"]["11"],
            "开庭")
        second = FakeCall(arc_plan={"cards": [_card(c) for c in range(11, 21)]})
        self._ensure(11, second)
        self.assertIn("开庭", second.calls[0]["user"])


class ArcVolumeTransitionWiringTests(unittest.TestCase):
    """`beat._volume_transition`: which boundaries reach the arc prompt.

    v1 asked this per chapter, at plan time. v2 asks it once per arc, which means
    the arc call must be handed EVERY boundary inside its span — an `arc_span` of
    10 does not align with volume ranges, so the usual case is a boundary landing
    mid-arc. A per-arc caller that only looked at `start_ch` would silently stop
    steering exactly the transitions v1 caught.
    """

    VP = (
        "## 第1卷：第八条（第1-20章）\n### 卷目标(O)\n林越活过三处讳地。\n\n"
        "## 第2卷：守夜人（第21-47章）\n### 卷目标(O)\n林越进入守夜人协会换取7号讳地准入。\n"
    )

    def _vt(self, chapters, vp=None, **cfg):
        with TemporaryDirectory() as td:
            paths = _paths(Path(td))
            if vp is not None:
                paths.volume_plan.write_text(vp, encoding="utf-8")
            return beat._volume_transition(paths, _config(**cfg), chapters)

    def test_a_boundary_inside_the_span_is_injected(self):
        # Ch21 is the 卷二 opening; an arc of 16..25 straddles it.
        out = self._vt(list(range(16, 26)), self.VP)
        self.assertIn("卷务转场", out)
        self.assertIn("守夜人协会", out)

    def test_an_arc_with_no_boundary_injects_nothing(self):
        self.assertEqual(self._vt(list(range(30, 40)), self.VP), "")

    def test_only_the_hard_level_is_injected_not_the_context_note(self):
        # Ch25 is mid-volume: `volume_transition_directive` still returns a
        # `context` block, and the arc prompt must NOT carry it (the volume-plan
        # window is already in the prompt verbatim).
        self.assertEqual(
            volume_transition_directive(25, self.VP, _config())["level"], "context")
        self.assertEqual(self._vt([25], self.VP), "")

    def test_a_missing_volume_plan_is_not_an_error(self):
        self.assertEqual(self._vt([21], None), "")
        self.assertEqual(self._vt([21], ""), "")

    def test_the_block_reaches_the_prompt(self):
        state = canon.build_story_state(21, brief="纲要", bible="世界", voice="声音")
        user = beat.arc_user_prompt(state, [21], volume_transition="## ⚠ 卷务转场\n收束上一卷")
        self.assertIn("卷务转场", user)


class ArcFingerprintWiringTests(unittest.TestCase):
    """The 全书结构指纹 aggregate: does the READ side actually reach the prompt?

    `v2/run.py:707` has written a fingerprint row every chapter since v2 shipped,
    and until this wiring nothing read the table back — the one reader lost its
    call site with `review.py`. So the property under test is not "does
    `arc_user_prompt` render a string it was handed", which would pass just as
    happily on an engine that never opens the table. **Every case here goes
    through the real seam**: rows are written with the same
    `quality.store_chapter_fingerprint(conn, ch, plan)` call the commit action
    makes, from the same `card_to_plan` plans, and the assertion is on the user
    prompt `generate_arc` actually built.

    The `"None"` case is a separate test rather than a detail: the reader returns
    the literal string when it has nothing to say (a v1 template convention), so
    an unfiltered pass-through would print the word "None" under a header
    promising overused patterns — a header that lies is worse than no header.
    """

    # Three moves, so both a bigram and a trigram form; the same beats in three
    # chapters is what pushes them over `quality._FP_MIN_REPEAT` (3).
    BEATS = ["汤舒婷推开门走进旧档案室", "顾峥翻找柜子里的病历", "两人核对页码"]
    HEADER = "全书结构指纹"

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = _paths(self.root)
        self.conn = store.init_db(self.paths)
        self.addCleanup(self.conn.close_current)

    def _seed(self, chapters):
        """Write real fingerprint rows exactly the way the commit action does."""
        from engine.plan import card_to_plan
        from engine.quality import store_chapter_fingerprint

        for ch in chapters:
            plan, _ = card_to_plan(_card(ch, beats=list(self.BEATS)))
            store_chapter_fingerprint(self.conn, ch, plan)

    def _user(self, **cfg):
        call = FakeCall(arc_plan={"cards": [_card(11)]})
        beat.generate_arc(None, self.paths, self.conn, _config(**cfg), 11, 11,
                          call=call)
        return call.calls[0]["user"]

    def test_the_aggregate_reaches_the_arc_prompt(self):
        self._seed([1, 2, 3])
        user = self._user()
        self.assertIn(self.HEADER, user)
        self.assertIn("enter_space→collect_evidence ×3", user)
        self.assertIn("全书 3 章累积统计", user)

    def test_an_empty_library_emits_no_header_and_never_the_word_None(self):
        user = self._user()
        self.assertNotIn(self.HEADER, user)
        # The reader's own sentinel must not survive into the prompt anywhere.
        self.assertNotIn("None", user)

    def test_the_flag_turns_the_read_off_without_touching_the_write(self):
        self._seed([1, 2, 3])
        self.assertNotIn(self.HEADER, self._user(fingerprint_enabled=False))
        # The rows are still there: `fingerprint_enabled` gates one prompt block,
        # not the offline replayability the library exists for (LESSONS §8).
        n = self.conn.execute("SELECT COUNT(*) FROM chapter_fingerprints").fetchone()
        self.assertEqual(n[0], 3)

    def test_the_block_sits_immediately_before_the_request(self):
        # An avoid-list is actionable only next to the ask it constrains; if a
        # later section is appended after it, this catches the drift.
        self._seed([1, 2, 3])
        user = self._user()
        self.assertLess(user.index(self.HEADER), user.index("## 请求"))
        between = user[user.index(self.HEADER):user.index("## 请求")]
        self.assertNotIn("\n## ", between)


class VolumeTransitionTests(unittest.TestCase):
    """Layer 二 治本: deterministic volume/arc boundary transition steer.

    Guards against arc overstay (yeban_guize ground the 城中村 arc to Ch28 because
    nothing enforced the planned Ch21 → 卷二 transition).

    Moved to `memory.py` when v1 was deleted — it parses `volume_plan.md`, which is
    that module's file. Its consumer moved from v1's per-chapter plan call to
    `v2/beat._volume_transition`, which injects only the HARD `transition` level
    into the arc call; the mid-volume `context` level is still produced and still
    tested here, because the level boundary is what `volume_transition_grace`
    means and a caller that wanted the note back must find it working.
    """

    VP = (
        "## 第1卷：第八条（第1-20章）\n### 卷目标(O)\n林越活过三处讳地。\n\n"
        "## 第2卷：守夜人（第21-47章）\n### 卷目标(O)\n林越进入守夜人协会换取7号讳地准入。\n"
    )

    def _vt(self, ch, **cfg):
        from engine.plan import volume_transition_directive
        return volume_transition_directive(ch, self.VP, {"novel": cfg})

    def test_parse_ranges(self):
        from engine.plan import parse_volume_ranges
        r = parse_volume_ranges(self.VP)
        self.assertEqual([(x["start"], x["end"]) for x in r], [(1, 20), (21, 47)])
        self.assertEqual(r[1]["name"], "守夜人")

    def test_transition_fires_at_volume_boundary(self):
        # Ch21/22 = 卷二开篇 grace window => hard transition (the missed pivot).
        for ch in (21, 22):
            v = self._vt(ch)
            self.assertEqual(v["level"], "transition")
            self.assertTrue(v["is_transition"])
            self.assertIn("卷务转场", v["block"])

    def test_no_transition_first_volume(self):
        # Ch1 is the first volume's opening — no previous volume to close.
        v = self._vt(1)
        self.assertFalse(v["is_transition"])

    def test_mid_volume_is_context_not_transition(self):
        v = self._vt(25)  # 25-21=4 >= grace(2)
        self.assertEqual(v["level"], "context")
        self.assertFalse(v["is_transition"])

    def test_goal_extracted_into_block(self):
        v = self._vt(21)
        self.assertIn("守夜人协会", v["block"])

    def test_grace_configurable(self):
        # grace=1 => only Ch21 transitions, Ch22 is context.
        self.assertTrue(self._vt(21, volume_transition_grace=1)["is_transition"])
        self.assertFalse(self._vt(22, volume_transition_grace=1)["is_transition"])

    def test_disabled_returns_ok(self):
        v = self._vt(21, volume_transition_enabled=False)
        self.assertEqual(v["level"], "ok")
        self.assertEqual(v["block"], "")

    def test_no_ranges_degrades_gracefully(self):
        from engine.plan import volume_transition_directive
        v = volume_transition_directive(21, "no volume headers here", {"novel": {}})
        self.assertEqual(v["level"], "ok")


if __name__ == "__main__":
    unittest.main()
