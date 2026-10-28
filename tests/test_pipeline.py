import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.cards import trim_card
from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import draft_deck, draft_from_request
from tests.fixtures import card
from tests.pipeline_fixtures import (
    JODAH,
    FakeCards,
    FakeModel,
    draft,
    make_pool,
    make_spec,
    playable_records,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OracleStore(Path(self.directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = playable_records()
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
        bad["buckets"]["support"][-1] = "Invented Card"
        model = FakeModel(bad, draft())
        result = self.run_draft(model)
        self.assertTrue(result["valid"], result)
        repair = json.loads(model.calls[1][0][-1]["content"])
        self.assertIn("schema", {error["error"] for error in repair["errors"]})
        self.assertEqual(self.client.get_calls, [])
        self.assertEqual(len(result["attempts"]), 2)

    def test_schema_errors_also_get_repair(self):
        result = self.run_draft(FakeModel({"wrong": "shape"}, draft()))
        self.assertTrue(result["valid"], result)
        self.assertIn("schema", {error["error"] for error in result["attempts"][0]["errors"]})

    def test_known_out_of_pool_cards_are_rejected_without_http(self):
        extra = card("Outside Pool", colors="U", mana_cost="{U}")
        self.store.import_cards([*self.records, extra])
        self.client = FakeCards([*self.records, extra])
        chosen = draft()
        chosen["buckets"]["support"][-1] = extra["name"]
        result = self.run_draft(FakeModel(chosen), max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIn("schema", {error["error"] for error in result["validation"]["errors"]})
        self.assertEqual(self.client.get_calls, [])
        self.assertEqual(result["verified_additions"], [])

    def test_new_api_card_cannot_bypass_pool_or_modify_local_snapshot(self):
        extra = card("New Real Card", colors="U", mana_cost="{U}")
        self.client = FakeCards([*self.records, extra])
        chosen = draft()
        chosen["buckets"]["support"][-1] = extra["name"]
        result = self.run_draft(FakeModel(chosen), max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIsNone(self.store.get_card(extra["name"]))
        self.assertEqual(self.client.get_calls, [])

    def test_fuzzy_or_different_named_api_response_never_silently_replaces_invention(self):
        bad = draft()
        bad["buckets"]["support"][-1] = "Invented Card"
        with patch.object(self.client, "get_card", return_value=trim_card(self.records[7])) as get:
            result = self.run_draft(FakeModel(bad), max_repairs=0)
        get.assert_not_called()
        self.assertFalse(result["valid"])
        self.assertIn("schema", {error["error"] for error in result["validation"]["errors"]})
        self.assertIsNone(self.store.get_card("Invented Card"))

    def test_exclusions_and_required_cards_are_enforced(self):
        pool = make_pool(self.store, make_spec(must_include=["Spell 69"]))
        result = self.run_draft(FakeModel(draft()), pool=pool, max_repairs=0)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["cards"]["Spell 69"], 1)
        pool = make_pool(self.store, make_spec(exclude=["Spell 69"]))
        pool.cards = [c for c in pool.cards if c.card["name"] != "Spell 69"]
        chosen = draft()
        chosen["buckets"]["support"][-1] = "Spell 69"
        result = self.run_draft(FakeModel(chosen), pool=pool, max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIn("schema", {e["error"] for e in result["validation"]["errors"]})

    def test_role_minimums_are_checked_in_code(self):
        sparse = [
            dict(raw, oracle_text="", type_line="Artifact")
            if raw["name"].startswith("Spell ") else raw
            for raw in self.records
        ]
        self.store.import_cards(sparse)
        pool = make_pool(self.store)
        result = self.run_draft(FakeModel(draft()), pool=pool, max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIn("role_target", {e["error"] for e in result["validation"]["errors"]})

    def test_repair_limit_is_initial_draft_plus_at_most_three_repairs(self):
        bad = draft()
        bad["buckets"]["support"][-1] = "Unknown"
        model = FakeModel(bad, bad, bad, bad)
        result = self.run_draft(model)
        self.assertFalse(result["valid"])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(result["attempts"]), 4)
        self.assertIn("schema", {e["error"] for e in result["validation"]["errors"]})

    def test_pool_only_drafting_never_spends_the_legacy_verification_budget(self):
        bad = draft()
        bad["buckets"]["support"][-1] = "Unknown"
        result = self.run_draft(FakeModel(bad, bad, bad, bad))
        self.assertFalse(result["valid"])
        self.assertEqual(self.client.get_calls, [])
        result = self.run_draft(FakeModel(bad), max_repairs=0, max_verifications=0)
        self.assertIn("schema", {e["error"] for e in result["validation"]["errors"]})
        self.assertEqual(self.client.get_calls, [])

    def test_land_and_commander_choices_cannot_override_code_policy(self):
        bad = draft(commander="Other Commander")
        bad["buckets"]["support"][-1] = "Plains"
        result = self.run_draft(FakeModel(bad, draft()))
        self.assertTrue(result["valid"], result)
        errors = {e["error"] for e in result["attempts"][0]["errors"]}
        self.assertIn("schema", errors)
        schema_errors = result["attempts"][0]["errors"]
        self.assertTrue(any(e.get("path") == "$.commander" for e in schema_errors))
        self.assertTrue(any(e.get("path") == "$.buckets.support[62]" for e in schema_errors))

    def test_assembly_respects_game_changer_allowance_without_model_repair(self):
        changed = [
            dict(raw, game_changer=True) if raw["name"] in {f"Spell {n}" for n in range(4)} else raw
            for raw in self.records
        ]
        self.store.import_cards(changed)
        model = FakeModel(draft())
        result = self.run_draft(model)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["validation"]["game_changer_count"], 3)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            len({f"Spell {n}" for n in range(4)} & result["cards"].keys()), 3
        )

    def test_required_game_changers_cannot_bypass_bracket_limit(self):
        names = [f"Spell {n}" for n in range(4)]
        self.store.import_cards([
            dict(raw, game_changer=True) if raw["name"] in names else raw
            for raw in self.records
        ])
        pool = make_pool(self.store, make_spec(must_include=names))
        result = self.run_draft(FakeModel(draft()), pool=pool, max_repairs=0)
        self.assertFalse(result["valid"])
        self.assertIn("game_changers", {e["error"] for e in result["validation"]["errors"]})

    def test_short_overlapping_buckets_are_padded_and_deduplicated(self):
        chosen = draft(40)
        chosen["buckets"]["creatures"] = ["Spell 0", "Spell 1"]
        result = self.run_draft(FakeModel(chosen))
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["assembly"]["model_selected_unique"], 40)
        self.assertEqual(len(result["assembly"]["cards"]), 63)
        self.assertEqual(result["cards"]["Spell 0"], 1)
        self.assertTrue(result["assembly"]["code_added"])

    def test_full_intake_discovery_draft_flow_uses_three_model_calls(self):
        model = FakeModel(
            make_spec().to_dict(),
            {"queries": [{"role": "theme", "query": "t:myojin"}]},
            draft(),
        )
        result = draft_from_request(
            "Myojin tribal with proliferate", JODAH["name"], model, self.client, self.store
        )
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(self.client.get_calls, [(JODAH["name"], False)])

    def test_invalid_loop_bounds_are_rejected(self):
        for bound in (-1, 4, True):
            with self.assertRaises(ValueError):
                self.run_draft(FakeModel(), max_repairs=bound)
