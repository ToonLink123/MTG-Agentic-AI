import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.cards import trim_card
from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import build_deck, draft_deck
from tests.fixtures import card
from tests.pipeline_fixtures import (
    JODAH,
    FakeCards,
    FakeModel,
    draft,
    make_pool,
    make_spec,
    records,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OracleStore(Path(self.directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = records()
        self.store.import_cards(self.records)
        self.client = FakeCards(self.records)

    def run_draft(self, model, *, pool=None, **kwargs):
        with patch(
            "mtg_agentic.scryfall.urlopen", side_effect=AssertionError("Unexpected real HTTP")
        ):
            return draft_deck(
                pool or make_pool(self.store), model, self.client, self.store, **kwargs
            )

    def test_valid_draft_gets_code_generated_lands_and_no_verification_calls(self):
        model = FakeModel(draft())
        result = self.run_draft(model)
        self.assertTrue(result["valid"], result)
        self.assertEqual(sum(result["cards"].values()), 99)
        self.assertEqual(result["land_base"]["land_count"], 36)
        self.assertEqual(result["validation"]["card_count"], 100)
        self.assertEqual(self.client.get_calls, [])
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("prices", model.calls[0][0][1]["content"])
        self.assertNotIn('"name": "Plains"', model.calls[0][0][1]["content"])

    def test_unknown_names_fail_immediately_then_json_errors_drive_repair(self):
        bad = draft()
        bad["cards"][-1]["name"] = "Invented Card"
        model = FakeModel(bad, draft())
        result = self.run_draft(model)
        self.assertTrue(result["valid"], result)
        repair = json.loads(model.calls[1][0][-1]["content"])
        self.assertIn("unknown_card", {error["error"] for error in repair["errors"]})
        self.assertEqual(self.client.get_calls, [("Invented Card", False)])
        self.assertEqual(len(result["attempts"]), 2)

    def test_schema_errors_also_get_repair(self):
        result = self.run_draft(FakeModel({"wrong": "shape"}, draft()))
        self.assertTrue(result["valid"], result)
        self.assertIn("schema", {error["error"] for error in result["attempts"][0]["errors"]})

    def test_known_out_of_pool_cards_require_exact_get_card(self):
        extra = card("Outside Pool", colors="U", mana_cost="{U}")
        self.store.import_cards([*self.records, extra])
        self.client = FakeCards([*self.records, extra])
        chosen = draft()
        chosen["cards"][-1]["name"] = extra["name"]
        result = self.run_draft(FakeModel(chosen))
        self.assertTrue(result["valid"], result)
        self.assertEqual(self.client.get_calls, [("Outside Pool", False)])
        self.assertEqual(result["verified_additions"], ["Outside Pool"])

    def test_new_exact_api_card_is_persisted_before_local_validation(self):
        extra = card("New Real Card", colors="U", mana_cost="{U}")
        self.client = FakeCards([*self.records, extra])
        chosen = draft()
        chosen["cards"][-1]["name"] = extra["name"]
        result = self.run_draft(FakeModel(chosen))
        self.assertTrue(result["valid"], result)
        self.assertIsNotNone(self.store.get_card(extra["name"]))
        self.assertIn(extra["name"], self.store.metadata["verified_cards"])

    def test_fuzzy_or_different_named_api_response_never_silently_replaces_invention(self):
        bad = draft()
        bad["cards"][-1]["name"] = "Invented Card"
        with patch.object(self.client, "get_card", return_value=trim_card(self.records[7])):
            result = self.run_draft(FakeModel(bad), max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIn("unknown_card", {error["error"] for error in result["validation"]["errors"]})
        self.assertIsNone(self.store.get_card("Invented Card"))

    def test_exclusions_and_required_cards_are_enforced(self):
        for spec, error in (
            (make_spec(must_include=["Spell 69"]), "must_include"),
            (make_spec(exclude=["Spell 69"]), "excluded_card"),
        ):
            pool = make_pool(self.store, spec)
            if error == "excluded_card":
                pool.cards = [
                    candidate for candidate in pool.cards if candidate.card["name"] != "Spell 69"
                ]
                chosen = draft()
                chosen["cards"][-1]["name"] = "Spell 69"
            else:
                chosen = draft()
            result = self.run_draft(FakeModel(chosen), pool=pool, max_repairs=0)
            self.assertIn(error, {e["error"] for e in result["validation"]["errors"]})

    def test_role_minimums_are_checked_in_code(self):
        pool = make_pool(self.store)
        for candidate in pool.cards:
            candidate.roles &= {"theme", "lands"}
        result = self.run_draft(FakeModel(draft()), pool=pool, max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIn("role_target", {e["error"] for e in result["validation"]["errors"]})

    def test_repair_limit_is_initial_draft_plus_at_most_three_repairs(self):
        bad = draft()
        bad["cards"][-1]["name"] = "Spell 0"
        model = FakeModel(bad, bad, bad, bad)
        result = self.run_draft(model)
        self.assertFalse(result["valid"])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(result["attempts"]), 4)
        self.assertIn("singleton", {e["error"] for e in result["validation"]["errors"]})

    def test_negative_verifications_are_reused_and_budget_is_bounded(self):
        bad = draft()
        bad["cards"][-1]["name"] = "Unknown"
        result = self.run_draft(FakeModel(bad, bad, bad, bad))
        self.assertFalse(result["valid"])
        self.assertEqual(len(self.client.get_calls), 1)
        self.client.get_calls.clear()
        result = self.run_draft(FakeModel(bad), max_repairs=0, max_verifications=0)
        self.assertIn("verification_budget", {e["error"] for e in result["validation"]["errors"]})
        self.assertEqual(self.client.get_calls, [])

    def test_land_and_commander_choices_cannot_override_code_policy(self):
        bad = draft(commander="Other Commander")
        bad["cards"][-1]["name"] = "Plains"
        result = self.run_draft(FakeModel(bad, draft()))
        self.assertTrue(result["valid"], result)
        errors = {e["error"] for e in result["attempts"][0]["errors"]}
        self.assertTrue({"land_selection", "commander_changed"} <= errors)

    def test_game_changer_errors_feed_back_then_repaired_deck_passes(self):
        changed = [
            dict(raw, game_changer=True) if raw["name"] in {f"Spell {n}" for n in range(4)} else raw
            for raw in self.records
        ]
        self.store.import_cards(changed)
        corrected = draft()
        corrected["cards"][3]["name"] = "Spell 69"
        result = self.run_draft(FakeModel(draft(), corrected))
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["validation"]["game_changer_count"], 3)
        self.assertIn("game_changers", {e["error"] for e in result["attempts"][0]["errors"]})

    def test_full_intake_discovery_draft_flow_uses_three_model_calls(self):
        model = FakeModel(
            make_spec().to_dict(),
            {"queries": [{"role": "theme", "query": "t:myojin"}]},
            draft(),
        )
        result = build_deck(
            "Myojin tribal with proliferate", JODAH["name"], model, self.client, self.store
        )
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(self.client.get_calls, [(JODAH["name"], False)])

    def test_invalid_loop_bounds_are_rejected(self):
        for bound in (-1, 4, True):
            with self.assertRaises(ValueError):
                self.run_draft(FakeModel(), max_repairs=bound)
