import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtg_agentic.edit import Toolbox, edit_deck
from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import draft_deck
from tests.fixtures import card
from tests.pipeline_fixtures import (
    FakeCards,
    FakeModel,
    draft,
    make_pool,
    make_spec,
    playable_records,
)


def act(tool="finish", *, query="", cards=(), swaps=(), reply=""):
    return {"tool": tool, "query": query, "cards": list(cards), "swaps": list(swaps), "reply": reply}


def swap(remove, add, reason="fits the plan"):
    return {"remove": remove, "add": add, "reason": reason}


class EditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = playable_records()
        self.store.import_cards(self.records)
        self.client = FakeCards(self.records)

    def deck(self, spec=None):
        pool = make_pool(self.store, spec=spec)
        with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("Unexpected HTTP")):
            deck = draft_deck(pool, FakeModel(draft()), self.client, self.store)
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        return deck, pool

    def edit(self, deck, pool, model, request, **kwargs):
        tools = Toolbox(self.store, self.client)
        return edit_deck(deck, pool, model, request, tools=tools, **kwargs)

    def test_a_question_is_answered_without_rebuilding_the_deck(self):
        deck, pool = self.deck()
        model = FakeModel(act(reply="Spell 3 is your best removal."))
        result = self.edit(deck, pool, model, "What is my best removal?")
        self.assertEqual(result["cards"], deck["cards"])
        self.assertFalse(result["edit"]["changed"])
        self.assertEqual(result["conversation"], [
            {"player": "What is my best removal?", "orchestrator": "Spell 3 is your best removal.",
             "swaps": []},
        ])

    def test_tools_then_finish_apply_exact_swaps_and_regenerate_lands(self):
        deck, pool = self.deck()
        model = FakeModel(
            act("card", cards=["Spell 66"]),
            act("check_deck", swaps=[swap("Spell 40", "Spell 66")]),
            act(swaps=[swap("Spell 40", "Spell 66"), swap("Spell 41", "Spell 66")],
                reply="Swapped Spell 40 for Spell 66."),
        )
        with self.assertLogs("mtg_agentic.activity", "INFO") as logs:
            result = self.edit(deck, pool, model, "Swap a weak card for Spell 66")
        self.assertIn("Spell 66", result["cards"])
        self.assertNotIn("Spell 40", result["cards"])
        self.assertIn("Spell 41", result["cards"])  # Spell 66 can only be added once.
        self.assertEqual(sum(result["cards"].values()), 99)
        self.assertTrue(result["valid"], result["validation"]["errors"])
        self.assertEqual(result["edit"]["tools"], ["card", "check_deck"])
        self.assertEqual(result["edit"]["swaps"], [swap("Spell 40", "Spell 66")])
        dry_run = json.loads(model.calls[2][0][-1]["content"])
        self.assertEqual(dry_run["tool_result"], "check_deck")
        self.assertTrue(dry_run["result"]["valid"])
        self.assertTrue(any("agent: orchestrator | start" in line for line in logs.output))
        self.assertTrue(any("tool: card | Spell 66" in line for line in logs.output))

    def test_a_card_the_player_names_from_outside_the_pool_can_be_added(self):
        outside = card("Outside Engine", colors="U", mana_cost="{1}{U}", cmc=2,
                       text="Draw a card. Create a Treasure token.")
        self.store.import_cards([*self.records, outside])
        deck, pool = self.deck()
        model = FakeModel(act(swaps=[swap("Spell 40", "Outside Engine")], reply="Added it."))
        result = self.edit(deck, pool, model, "Please add [[Outside Engine]]")
        self.assertIn("Outside Engine", result["cards"])
        payload = json.loads(model.calls[0][0][1]["content"])
        self.assertEqual(payload["named_cards"], [{"name": "Outside Engine", "addable": True, "note": ""}])
        self.assertIn("Outside Engine", model.calls[0][1]["properties"]["swaps"]["items"]["properties"]["add"]["enum"])

    def test_review_problems_go_back_for_a_repair(self):
        deck, pool = self.deck()
        reviewer = Mock(game_changer_names=[])
        reviewer.check.side_effect = [
            {"status": "needs_revision", "errors": [{"error": "interaction_review", "detail": "unsupported"}]},
            {"status": "passed", "errors": []},
        ]
        model = FakeModel(
            act(swaps=[swap("Spell 40", "Spell 66")], reply="First try."),
            act(swaps=[swap("Spell 40", "Spell 67")], reply="Fixed it."),
        )
        result = self.edit(deck, pool, model, "Improve the deck", reviewer=reviewer)
        self.assertIn("problems", json.loads(model.calls[1][0][-1]["content"]))
        self.assertIn("Spell 67", result["cards"])
        self.assertNotIn("Spell 66", result["cards"])
        self.assertTrue(result["review_passed"])
        self.assertEqual(result["edit"]["reply"], "Fixed it.")

    def test_the_last_step_must_finish(self):
        deck, pool = self.deck()
        model = FakeModel(act("card", cards=["Spell 66"]), act(reply="Out of steps; no change."))
        result = self.edit(deck, pool, model, "Look around", max_steps=2)
        self.assertEqual(model.calls[1][1]["properties"]["tool"]["enum"], ["finish"])
        self.assertFalse(result["edit"]["changed"])

    def test_removing_a_must_include_card_on_request_drops_the_requirement(self):
        deck, pool = self.deck(spec=make_spec(must_include=["Spell 1"]))
        # Spell 15 is another creature, so the creature count stays on target.
        model = FakeModel(act(swaps=[swap("Spell 1", "Spell 15")], reply="Cut Spell 1 as asked."))
        result = self.edit(deck, pool, model, "Cut Spell 1")
        self.assertNotIn("Spell 1", result["cards"])
        self.assertEqual(result["spec"]["must_include"], [])
        self.assertTrue(result["valid"], result["validation"]["errors"])


class EditCliTests(unittest.TestCase):
    def test_edit_deck_command_applies_the_orchestrators_swaps(self):
        from mtg_agentic.__main__ import main
        from mtg_agentic.rules import RulesStore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            records = playable_records()
            with OracleStore(path / "oracle.sqlite3") as store:
                store.import_cards(records)
                pool = make_pool(store)
                with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("HTTP")):
                    deck = draft_deck(pool, FakeModel(draft()), FakeCards(records), store)
            (path / "deck.json").write_text(json.dumps(deck), encoding="utf-8")
            (path / "pool.json").write_text(json.dumps(pool.to_dict()), encoding="utf-8")
            model = FakeModel(act(swaps=[swap("Spell 40", "Spell 66")], reply="Swapped."))
            output = path / "edited.json"
            with (
                patch("mtg_agentic.__main__._make_model", return_value=model),
                patch.object(RulesStore, "download_rules", return_value={}),
            ):
                status = main([
                    "--quiet", "--data-dir", str(path), "edit-deck", str(path / "deck.json"),
                    "--pool", str(path / "pool.json"), "--request", "Swap Spell 40",
                    "--skip-review", "--embedding-backend", "hash", "--output", str(output),
                ])
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 0, result.get("validation"))
        self.assertIn("Spell 66", result["cards"])
        self.assertEqual(result["edit"]["reply"], "Swapped.")
        self.assertEqual(len(result["conversation"]), 1)


if __name__ == "__main__":
    unittest.main()
