"""Unit tests for arc.py — the arc-level ChapterCard planner (REDESIGN.md L2).

Only the pure functions are covered here (no LLM, no DB): window arithmetic,
card normalization, the card -> plan/decision projection that keeps the rest of
the pipeline unchanged, and the zero-cost pre-write validation.

The projection tests are the load-bearing ones: every downstream module
(writing/review/quality/store) reads the plan schema, so a card that projects
wrong is a silent quality regression rather than a crash.
"""
import unittest

from arc import (
    OPENING_TYPES,
    arc_window,
    card_to_plan,
    normalize_card,
    validate_card,
)


def _good_card(**over) -> dict:
    card = {
        "ch": 27,
        "title": "旧档案室",
        "where": "县医院三楼旧档案室，凌晨两点，只有应急灯",
        "who": ["陈默", "值班护士林"],
        "pov_character": "陈默",
        "wants": "拿到 1998 年那卷失踪病历",
        "blocked_by": "登记簿上他自己的名字已经被人签过了",
        "pressure": "保安每二十分钟巡一次三楼",
        "turn": "签名是他的笔迹，但日期是三天后",
        "payoff": "陈默把两页登记簿并排压在灯下，指着同一处笔锋比给护士林看",
        "payoff_type": "reveal",
        "conflict_type": "intelligence",
        "beats": ["陈默撬开档案柜的锁扣", "他翻到 1998 年那一格", "护士林在门口叫他的名字"],
        "info_source": "登记簿",
        "thread_actions": ["回收 Ch19 埋的笔迹伏线"],
        "world_state_changes": ["陈默确认医院里有人冒用他的身份"],
        "exit_hook": "走廊尽头，护士林正在给某个人打电话，说的是他的名字",
        "forbid": ["再用镜子意象", "再用'他忽然明白了'式顿悟"],
        "opening_type": "physical_action",
    }
    card.update(over)
    return card


class TestArcWindow(unittest.TestCase):
    def test_blocks_are_anchored_at_chapter_one(self):
        self.assertEqual(arc_window(1, 10), (1, 10))
        self.assertEqual(arc_window(10, 10), (1, 10))
        self.assertEqual(arc_window(11, 10), (11, 20))
        self.assertEqual(arc_window(27, 10), (21, 30))

    def test_window_is_a_pure_function_of_the_chapter(self):
        """A resumed/forked run must recompute identical boundaries without
        remembering when arc planning was switched on."""
        for ch in range(1, 60):
            self.assertEqual(arc_window(ch, 10), arc_window(ch, 10))
        self.assertEqual(arc_window(26, 10), arc_window(29, 10))

    def test_max_chapters_clips_the_tail(self):
        self.assertEqual(arc_window(27, 10, max_chapters=25), (21, 25))
        self.assertEqual(arc_window(27, 10, max_chapters=0), (21, 30))

    def test_never_returns_an_inverted_range(self):
        start, end = arc_window(27, 10, max_chapters=5)
        self.assertLessEqual(start, end)


class TestNormalizeCard(unittest.TestCase):
    def test_rejects_non_dict_and_incomplete_cards(self):
        self.assertIsNone(normalize_card("not a card", 27))
        self.assertIsNone(normalize_card(_good_card(payoff=""), 27))
        self.assertIsNone(normalize_card(_good_card(beats=[]), 27))

    def test_scalar_where_a_list_belongs_is_coerced(self):
        card = normalize_card(_good_card(who="陈默", thread_actions=""), 27)
        self.assertEqual(card["who"], ["陈默"])
        self.assertEqual(card["thread_actions"], [])

    def test_chapter_number_is_forced_by_the_caller(self):
        card = normalize_card(_good_card(ch=999), 27)
        self.assertEqual(card["ch"], 27)

    def test_turn_is_folded_into_the_beats(self):
        """The turn is the chapter's spine; if it lives only in `turn` the
        writer never sees it as an obligation."""
        card = normalize_card(_good_card(), 27)
        self.assertTrue(any(card["turn"] in b for b in card["beats"]))
        self.assertEqual(len(card["beats"]), 4)

    def test_turn_already_present_is_not_duplicated(self):
        turn = "签名是他的笔迹，但日期是三天后"
        card = normalize_card(_good_card(beats=["他撬开锁扣", turn, "护士林出现"]), 27)
        self.assertEqual(len(card["beats"]), 3)

    def test_turn_split_across_two_beats_is_not_folded_in_again(self):
        """Observed on the first live arc card: the model told the turn as
        beat N's action + beat N+1's result, worded just differently enough that
        a prefix check missed it, so the writer got the moment three times."""
        card = normalize_card(_good_card(
            turn="陈默用碎瓷片刮开档案柜门锁的锈壳，从缝隙里勾出折叠的登记簿和铜钥匙0177",
            beats=["陈默用碎瓷片刮开柜门底的腐蚀缝，把手电塞进去",
                   "他从缝隙里勾出折叠的登记簿和一把刻着 0177 的铜钥匙",
                   "护士林在门口叫他的名字"],
        ), 27)
        self.assertEqual(len(card["beats"]), 3)

    def test_unknown_opening_type_is_dropped_not_kept(self):
        card = normalize_card(_good_card(opening_type="montage"), 27)
        self.assertEqual(card["opening_type"], "")
        for t in OPENING_TYPES:
            self.assertEqual(normalize_card(_good_card(opening_type=t), 27)["opening_type"], t)

    def test_missing_pressure_falls_back_to_the_obstacle(self):
        card = normalize_card(_good_card(pressure=""), 27)
        self.assertEqual(card["pressure"], card["blocked_by"])


class TestCardToPlan(unittest.TestCase):
    """The projection is what lets writing/review/quality/store stay untouched."""

    def setUp(self):
        self.card = normalize_card(_good_card(), 27)
        self.plan, self.decision = card_to_plan(self.card)

    def test_plan_carries_every_key_the_committee_emits(self):
        for key in ("title", "goal", "conflict", "conflict_type", "payoff", "payoff_type",
                    "pressure", "beats", "character_focus", "pov_character", "location",
                    "info_source", "world_state_changes", "thread_actions", "hook", "risk"):
            self.assertIn(key, self.plan, key)
            self.assertTrue(self.plan[key] != "" or key == "risk", key)

    def test_concrete_card_fields_map_onto_plan_fields(self):
        self.assertEqual(self.plan["goal"], self.card["wants"])
        self.assertEqual(self.plan["conflict"], self.card["blocked_by"])
        self.assertEqual(self.plan["location"], self.card["where"])
        self.assertEqual(self.plan["hook"], self.card["exit_hook"])
        self.assertEqual(self.plan["character_focus"], self.card["who"])

    def test_card_only_fields_ride_along_for_the_writer(self):
        self.assertEqual(self.plan["opening_type"], "physical_action")
        self.assertEqual(self.plan["forbid"], self.card["forbid"])
        self.assertEqual(self.plan["source"], "arc_card")

    def test_no_fake_arbiter_score(self):
        """plan_score() must read 0.0, not an invented number — it lands in
        chapter_metrics.plan_score and in the writer's quality contract."""
        from planning import plan_score
        self.assertEqual(self.decision["scores"], [])
        self.assertEqual(plan_score(self.decision), 0.0)

    def test_required_constraints_are_structured_dicts(self):
        constraints = self.decision["required_constraints"]
        self.assertTrue(constraints)
        for c in constraints:
            self.assertEqual(set(c) >= {"id", "type", "constraint", "check_method", "target"}, True)
        ids = {c["id"] for c in constraints}
        self.assertTrue({"card_turn", "card_payoff", "card_location", "card_hook"} <= ids)

    def test_every_forbid_becomes_a_constraint(self):
        ids = [c["id"] for c in self.decision["required_constraints"]]
        self.assertEqual(sum(1 for i in ids if i.startswith("card_forbid_")), 2)

    def test_plan_is_not_aliased_to_the_card(self):
        self.plan["beats"].append("mutation")
        self.assertNotIn("mutation", self.card["beats"])


class TestValidateCard(unittest.TestCase):
    def test_clean_card_with_no_history_passes(self):
        self.assertEqual(validate_card(normalize_card(_good_card(), 27), recent_cards=[]), [])

    def test_empty_required_field_is_reported(self):
        card = dict(normalize_card(_good_card(), 27))
        card["payoff"] = ""
        problems = validate_card(card, recent_cards=[])
        self.assertTrue(any("payoff" in p for p in problems))

    def test_repeated_opening_type_is_caught(self):
        prev = normalize_card(_good_card(opening_type="dialogue"), 26)
        card = normalize_card(_good_card(opening_type="dialogue"), 27)
        self.assertTrue(any("opening_type" in p for p in validate_card(card, recent_cards=[prev])))

    def test_repeated_location_is_caught(self):
        prev = normalize_card(_good_card(), 26)
        card = normalize_card(_good_card(), 27)
        self.assertTrue(any("场地" in p for p in validate_card(card, recent_cards=[prev])))

    def test_third_consecutive_payoff_type_is_caught(self):
        prevs = [normalize_card(_good_card(where=f"地点{i}", payoff_type="reveal"), 25 + i)
                 for i in range(2)]
        card = normalize_card(_good_card(where="新地点", payoff_type="reveal"), 27)
        self.assertTrue(any("payoff_type" in p for p in validate_card(card, recent_cards=prevs)))

    def test_two_consecutive_payoff_types_are_allowed(self):
        prev = normalize_card(_good_card(where="别处", payoff_type="reveal"), 26)
        card = normalize_card(_good_card(where="新地点", payoff_type="reveal"), 27)
        self.assertFalse(any("payoff_type" in p for p in validate_card(card, recent_cards=[prev])))

    def test_scene_similarity_blocks_only_at_or_above_the_cut(self):
        card = normalize_card(_good_card(), 27)
        self.assertEqual(validate_card(card, recent_cards=[], scene_sim=0.84, scene_sim_block=0.85), [])
        self.assertTrue(validate_card(card, recent_cards=[], scene_sim=0.85, scene_sim_block=0.85))

    def test_continuity_violations_are_passed_through(self):
        card = normalize_card(_good_card(), 27)
        problems = validate_card(card, recent_cards=[], continuity_violations=["陈默已在 Ch20 死亡"])
        self.assertTrue(any("陈默已在 Ch20 死亡" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
