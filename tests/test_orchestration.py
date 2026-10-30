"""Ranked picks, repair feedback, plan targets, the land formula, council consensus, and the
native tool loops (drafting, follow-ups and the polish pass)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtg_agentic import prompts
from mtg_agentic.agent import Finish, Tool, ToolError, run_tools
from mtg_agentic.assembly import assemble_nonlands
from mtg_agentic.council import council_draft
from mtg_agentic.edit import Toolbox, edit_deck, polish_deck
from mtg_agentic.lands import land_plan
from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import draft_deck
from mtg_agentic.review import DeckReviewer
from mtg_agentic.schema import object_schema
from tests.pipeline_fixtures import (
    JODAH,
    FakeCards,
    FakeModel,
    FakeToolModel,
    draft,
    make_pool,
    make_spec,
    playable_records,
)
from tests.test_council import PLAN, CouncilModel, picks

TARGETS = {"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "lands": 36}


def spells(*numbers):
    return [f"Spell {n}" for n in numbers]


def tiers(core=(), strong=(), flex=()):
    return {"core": list(core), "strong": list(strong), "flex": list(flex)}


def swap(remove, add, reason="fits the plan"):
    return {"remove": remove, "add": add, "reason": reason}


class Fixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = playable_records()
        self.store.import_cards(self.records)
        self.client = FakeCards(self.records)
        self.pool = make_pool(self.store)
        http = patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("Unexpected HTTP"))
        http.start()
        self.addCleanup(http.stop)

    def draft(self, model, **kwargs):
        return draft_deck(kwargs.pop("pool", self.pool), model, self.client, self.store, **kwargs)


class AssemblyTests(Fixture):
    # Spells 0-17 are creatures that draw, ramp and remove; 18-20 instant protection;
    # 21-23 board wipes; 24-69 do nothing a role counts.

    def test_strong_picks_stay_even_when_pool_cards_fill_more_roles(self):
        chosen = spells(*range(24, 70), *range(17))
        result = assemble_nonlands(self.pool, tiers(strong=chosen))
        self.assertEqual(set(result["cards"]), set(chosen))
        self.assertEqual(result["code_swaps"], [])
        self.assertEqual(result["code_added"], [])

    def test_code_swaps_only_the_lowest_flex_picks_to_reach_role_minimums(self):
        chosen = spells(*range(17), *range(24, 70))
        result = assemble_nonlands(self.pool, tiers(flex=chosen))
        self.assertTrue(set(spells(21, 22, 23)) <= set(result["cards"]))  # The three wipes.
        self.assertGreaterEqual(len(set(spells(18, 19, 20)) & set(result["cards"])), 2)
        swapped_out = [s["out"] for s in result["code_swaps"]]
        self.assertEqual(swapped_out, spells(*range(69, 69 - len(swapped_out), -1)))
        self.assertTrue(all(s["from"] == "the pool" for s in result["code_swaps"]))
        reasons = {d["card"]: d["reason"] for d in result["dropped"]}
        self.assertIn("swapped by code", reasons["Spell 69"])

    def test_the_drafters_own_extra_picks_are_swapped_in_before_the_pool(self):
        chosen = spells(*range(17), *range(24, 70))
        result = assemble_nonlands(self.pool, tiers(flex=[*chosen, "Spell 21", "Spell 18"]))
        froms = {s["in"]: s["from"] for s in result["code_swaps"]}
        self.assertEqual(froms["Spell 21"], "your extra picks")
        self.assertEqual(froms["Spell 18"], "your extra picks")

    def test_every_pick_left_out_says_why(self):
        result = assemble_nonlands(self.pool, tiers(strong=spells(*range(70))))
        reasons = {d["card"]: d["reason"] for d in result["dropped"]}
        self.assertEqual(sorted(reasons), spells(*range(63, 70)))
        self.assertTrue(all(r.startswith("no room") for r in reasons.values()))


class RepairTests(Fixture):
    def test_repairs_see_what_code_did_with_the_picks(self):
        chosen = draft()
        chosen["picks"]["flex"][-1] = "Invented Card"
        model = FakeModel(chosen, draft())
        self.draft(model)
        repair = json.loads(model.calls[1][0][-1]["content"])
        self.assertEqual(set(repair["code_changes"]), {"added_by_code", "left_out", "swapped"})
        self.assertIn("Invented Card", repair["errors"][0]["detail"])
        self.assertIn("role_minimums", repair)

    def test_the_best_attempt_returns_not_the_last(self):
        reviewer = Mock(game_changer_names=[])
        problem = {"error": "interaction_review", "detail": "contradicted"}
        reviewer.check.side_effect = [
            {"status": "needs_revision", "errors": [problem]},
            {"status": "needs_revision", "errors": [problem, {**problem, "card": "x"}, {**problem, "card": "y"}]},
        ]
        result = self.draft(FakeModel(draft(), draft()), reviewer=reviewer, max_repairs=1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["best_attempt"], 1)
        self.assertEqual(len(result["validation"]["errors"]), 1)

    def test_an_auto_land_count_follows_the_curve(self):
        targets = {**TARGETS, "lands": 0}
        pool = make_pool(self.store, make_spec(role_targets=targets))
        result = self.draft(FakeModel(draft()), pool=pool)
        self.assertTrue(result["valid"], result["validation"]["errors"])
        # Average mana value 2, 18 cheap ramp-and-draw cards: 31.42 + 6.26 - 5.04 = 32.6.
        self.assertEqual(result["land_plan"]["count"], 33)
        self.assertEqual(result["land_base"]["land_count"], 33)
        self.assertEqual(result["spec"]["role_targets"]["lands"], 33)
        self.assertEqual(result["assembly"]["nonland_slots"], 66)
        self.assertEqual(sum(result["cards"].values()), 99)

    def test_the_land_formula_is_bounded(self):
        plan = land_plan(spells(*range(63)), self.store)
        self.assertEqual((plan["average_mana_value"], plan["cheap_ramp_and_draw"]), (2.0, 18))
        self.assertEqual((plan["estimate"], plan["count"]), (32.6, 33))
        self.assertEqual(land_plan([], self.store)["count"], 36)


class ReviewMappingTests(Fixture):
    def reviewer(self, report):
        bracket = Mock(game_changer_list={"names": []})
        bracket.check.return_value = {"reasons": [], "warnings": [], "spellbook": {"combos": []}, **report}
        rules = Mock()
        rules.ingest_deck.return_value = {"count": 1}
        return DeckReviewer(bracket, rules, Mock(concurrency=1), self.store, self.client)

    def test_an_uncertain_bracket_that_names_no_card_is_a_warning(self):
        report = {"verdict": "uncertain", "warnings": ["Likely win speed is unknown"]}
        result = self.reviewer(report).check({"Spell 30": 1}, JODAH["name"], 3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["warnings"][0]["warning"], "bracket_uncertain")
        self.assertEqual(result["warnings"][0]["reasons"], ["Likely win speed is unknown"])

    def test_an_uncertain_bracket_about_named_cards_still_fails(self):
        flag = {"cards": ["Spell 0", "Spell 1"], "reason": "Two-card combo whose speed is unknown"}
        result = self.reviewer({"verdict": "uncertain", "flags": [flag]}).check(
            {"Spell 30": 1}, JODAH["name"], 3
        )
        self.assertEqual(result["status"], "needs_revision")
        self.assertEqual(result["errors"][0]["cards"], ["Spell 0", "Spell 1"])

    def test_a_mismatch_fails(self):
        result = self.reviewer({"verdict": "mismatch"}).check({"Spell 30": 1}, JODAH["name"], 3)
        self.assertEqual(result["errors"][0]["error"], "bracket_review")


class CouncilTests(Fixture):
    def test_consensus_counts_the_members_own_picks_not_code_padding(self):
        model = CouncilModel({f: picks(*range(60)) for f in ("engine", "resilience", "consistency")})
        deck = council_draft(self.pool, model, model, self.client, self.store)
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        self.assertEqual(deck["council"]["locked"], sorted(spells(*range(60))))
        self.assertEqual(deck["council"]["contested"], [])
        self.assertEqual(len(deck["assembly"]["code_added"]), 3)  # Padding, outside the vote.

    def test_the_plans_role_targets_reach_the_deck(self):
        plan = {**PLAN, "role_targets": {**PLAN["role_targets"], "creatures": 12, "wipes": 0}}
        model = CouncilModel({f: picks(*range(63)) for f in ("engine", "resilience", "consistency")},
                             plan=plan)
        deck = council_draft(self.pool, model, model, self.client, self.store)
        self.assertEqual(deck["spec"]["role_targets"]["creatures"], 12)
        self.assertEqual(deck["spec"]["role_targets"]["wipes"], 0)
        self.assertEqual(deck["council"]["role_targets"]["creatures"], 12)


class ToolLoopTests(unittest.TestCase):
    def test_tools_run_errors_go_back_and_finish_ends_the_loop(self):
        tries = []

        def finish(args):
            tries.append(args["answer"])
            if len(tries) == 1:
                raise ToolError({"problems": ["not yet"]})
            return Finish(args["answer"])

        schema = object_schema({"answer": {"type": "string", "minLength": 1, "maxLength": 20}})
        tools = [
            Tool("echo", "Echo.", schema, lambda a: {"echo": a["answer"]}),
            Tool("finish", "Finish.", schema, finish),
        ]
        model = FakeToolModel(
            [("echo", {"answer": "hi"}), ("nope", {})],
            [("finish", {"answer": "first"})],
            [("echo", {"answer": ""}), ("finish", {"answer": "second"})],
        )
        done = run_tools(model, [{"role": "user", "content": "go"}], tools, max_steps=4,
                         agent="drafter", task="test")
        self.assertEqual(done, {"value": "second", "tools": ["echo"], "steps": 3})
        first, second = model.results(1), model.results(2)
        self.assertEqual(json.loads(first[0]["content"]), {"echo": "hi"})
        self.assertTrue(first[1]["is_error"])
        self.assertEqual(json.loads(second[0]["content"]), {"problems": ["not yet"]})
        self.assertEqual([t["name"] for t in model.calls[0][1]], ["echo", "finish"])

    def test_the_step_before_the_last_says_to_finish(self):
        model = FakeToolModel([], [])
        done = run_tools(model, [{"role": "user", "content": "go"}], [], max_steps=2,
                         agent="drafter", task="test")
        self.assertIsNone(done["value"])
        texts = [b["text"] for b in model.calls[1][0][-1]["content"]]
        self.assertTrue(any("call finish now" in text for text in texts))


class AgenticDraftTests(Fixture):
    def answer(self, numbers=range(63)):
        return {"picks": tiers(strong=spells(*numbers)), "outside": []}

    def test_the_drafter_checks_with_code_then_finishes(self):
        model = FakeToolModel(
            [("check_draft", self.answer()), ("card", {"cards": ["Spell 66"]})],
            [("finish", self.answer())],
        )
        deck = self.draft(model, toolbox=Toolbox(self.store, self.client))
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        self.assertEqual(len(deck["attempts"]), 1)  # A check is not an attempt.
        check, card = (json.loads(r["content"]) for r in model.results(1))
        self.assertTrue(check["ready_for_review"])
        self.assertEqual(check["role_counts"]["wipes"], 3)
        self.assertTrue(card["cards"][0]["addable"])
        system = [m["content"] for m in model.calls[0][0] if m["role"] == "system"]
        self.assertTrue(system[0].startswith("CANDIDATE POOL"))
        self.assertIn(prompts.DRAFT_AGENT, system[1])

    def test_a_failed_finish_comes_back_to_fix_in_the_same_conversation(self):
        reviewer = Mock(game_changer_names=[])
        reviewer.check.side_effect = [
            {"status": "needs_revision", "errors": [{"error": "bracket_review", "detail": "mismatch"}]},
            {"status": "passed", "errors": []},
        ]
        model = FakeToolModel([("finish", self.answer())], [("finish", self.answer())])
        deck = self.draft(model, toolbox=Toolbox(self.store, self.client), reviewer=reviewer)
        self.assertTrue(deck["valid"])
        self.assertEqual(len(deck["attempts"]), 2)
        result = model.results(1)[0]
        self.assertTrue(result["is_error"])
        feedback = json.loads(result["content"])
        self.assertEqual(feedback["errors"][0]["error"], "bracket_review")
        self.assertIn("code_changes", feedback)

    def test_a_card_a_search_offers_can_be_picked(self):
        from mtg_agentic.cards import name_key
        from tests.fixtures import card

        extra = card("Searched Engine", colors="U", mana_cost="{1}{U}", cmc=2, text="Draw two cards.")
        self.store.import_cards([*self.records, extra])
        client = FakeCards([*self.records, extra], search_results=[[extra]])
        chosen = self.answer(range(62))
        chosen["picks"]["core"] = ["Searched Engine"]
        model = FakeToolModel([("search_cards", {"query": "o:draw"})], [("finish", chosen)])
        deck = draft_deck(self.pool, model, client, self.store, toolbox=Toolbox(self.store, client))
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        self.assertIn("Searched Engine", deck["cards"])
        self.assertIn(name_key("Searched Engine"), {name_key(n) for n in deck["cards"]})

    def test_an_unknown_name_is_a_problem_not_an_attempt(self):
        bad = self.answer(range(62))
        bad["picks"]["flex"] = ["Invented Card"]
        model = FakeToolModel([("finish", bad)], [("finish", self.answer())])
        deck = self.draft(model, toolbox=Toolbox(self.store, self.client))
        self.assertTrue(deck["valid"])
        self.assertEqual(len(deck["attempts"]), 1)
        self.assertIn("Invented Card", model.results(1)[0]["content"])


class NativeFollowUpTests(Fixture):
    def deck(self, count=63):
        deck = self.draft(FakeModel(draft(count)))
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        return deck

    def test_tools_then_finish_apply_the_swaps(self):
        deck = self.deck()
        model = FakeToolModel(
            [("card", {"cards": ["Spell 66"]}), ("check_deck", {"swaps": [swap("Spell 40", "Spell 66")]})],
            [("finish", {"reply": "Swapped.", "swaps": [swap("Spell 40", "Spell 1")]})],
            [("finish", {"reply": "Swapped.", "swaps": [swap("Spell 40", "Spell 66")]})],
        )
        result = edit_deck(deck, self.pool, model, "Swap a weak card", tools=Toolbox(self.store))
        self.assertIn("Spell 66", result["cards"])
        self.assertNotIn("Spell 40", result["cards"])
        self.assertTrue(result["valid"], result["validation"]["errors"])
        self.assertEqual(result["edit"]["tools"], ["card", "check_deck"])
        self.assertTrue(model.results(2)[0]["is_error"])  # Spell 1 is already in the deck.
        self.assertEqual(model.calls[0][0][0]["content"], prompts.EDIT_NATIVE)
        self.assertEqual(len(model.calls[0][1]), 7)

    def test_the_polish_pass_keeps_a_valid_improvement_and_flags_code_cards(self):
        deck = self.deck(60)
        model = FakeToolModel([("finish", {"reply": "Tightened.", "swaps": [swap("Spell 40", "Spell 66")]})])
        polished = polish_deck(deck, self.pool, model, tools=Toolbox(self.store))
        self.assertIn("Spell 66", polished["cards"])
        self.assertTrue(polished["polish"]["kept"])
        self.assertEqual(polished["conversation"], [])
        self.assertNotIn("edit", polished)
        payload = json.loads(model.calls[0][0][1]["content"])
        flagged = {row["name"] for row in payload["deck"] if row.get("picked_by") == "code"}
        self.assertEqual(flagged, set(deck["assembly"]["code_added"]))

    def test_a_polish_that_breaks_the_deck_is_dropped(self):
        deck = self.deck()
        # Spell 21 is one of exactly three board wipes, so cutting it misses the minimum.
        bad = ("finish", {"reply": "Cut a wipe.", "swaps": [swap("Spell 21", "Spell 66")]})
        model = FakeToolModel([bad], [bad])
        polished = polish_deck(deck, self.pool, model, tools=Toolbox(self.store))
        self.assertEqual(polished["cards"], deck["cards"])
        self.assertFalse(polished["polish"]["kept"])
        self.assertTrue(polished["valid"])

    def test_an_invalid_deck_is_not_polished(self):
        deck = {**self.deck(), "valid": False}
        model = FakeToolModel()
        self.assertIs(polish_deck(deck, self.pool, model, tools=Toolbox(self.store)), deck)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
