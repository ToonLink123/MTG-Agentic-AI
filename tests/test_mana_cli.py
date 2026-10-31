import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.__main__ import main
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.land_fixtures import fixing_lands
from tests.pipeline_fixtures import make_spec, records


class ManaCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.leader = card(
            "Green Leader", colors="G", types="Legendary Creature — Elf", mana_cost="{G}"
        )
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards([
                *records(), *fixing_lands(), self.leader,
                card("Green Spell", colors="G", mana_cost="{G}{G}"),
            ])
        self.deck = self.path / "spells.json"
        self.deck.write_text(json.dumps({
            "commander": self.leader["name"], "cards": {"Green Spell": 1},
        }), encoding="utf-8")

    def run_cli(self, *args):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("mtg_agentic.__main__._make_model", side_effect=AssertionError("No model")),
            patch("mtg_agentic.__main__.ScryfallClient", side_effect=AssertionError("No HTTP")),
        ):
            status = main(["--quiet", "--data-dir", str(self.path), "mana-base", *args])
        return status, json.loads(output.getvalue())

    def test_standalone_spells_default_to_36_lands_without_a_model_or_pool(self):
        before = self.deck.read_bytes()
        status, report = self.run_cli(str(self.deck))
        self.assertEqual(status, 0)
        self.assertTrue(report["valid"])
        self.assertEqual(report["strategy"], "monocolor")
        self.assertEqual(report["cards"], {"Forest": 36})
        self.assertEqual(report["mana_pips"], {"G": 3.0})
        self.assertEqual(self.deck.read_bytes(), before)

    def test_saved_spec_land_count_and_explicit_override(self):
        spec = make_spec(
            commander=self.leader["name"], color_identity="G",
            role_targets={"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "lands": 40},
        )
        self.deck.write_text(json.dumps({
            "commander": self.leader["name"], "cards": {"Green Spell": 1, "Forest": 20},
            "spec": spec.to_dict(),
        }), encoding="utf-8")
        self.assertEqual(self.run_cli(str(self.deck))[1]["land_count"], 40)
        status, report = self.run_cli(str(self.deck), "--lands", "24")
        self.assertEqual(status, 0)
        self.assertEqual(report["land_count"], 24)

    def test_output_is_new_utf8_report_and_existing_files_are_not_overwritten(self):
        output = self.path / "mana.json"
        status, summary = self.run_cli(str(self.deck), "--output", str(output))
        self.assertEqual(status, 0)
        self.assertEqual(summary["output"], str(output))
        self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["valid"])
        before = output.read_bytes()
        status, report = self.run_cli(str(self.deck), "--output", str(output))
        self.assertEqual(status, 2)
        self.assertEqual(output.read_bytes(), before)

    def test_unknown_cards_and_bad_quantities_are_rejected(self):
        for cards in ({"Unknown": 1}, {"Green Spell": True}, {"Forest": 0}):
            self.deck.write_text(json.dumps({
                "commander": self.leader["name"], "cards": cards,
            }), encoding="utf-8")
            status, report = self.run_cli(str(self.deck))
            self.assertEqual(status, 2)
            self.assertEqual(report["errors"][0]["error"], "input")

    def test_impossible_mana_base_returns_errors_and_exit_one(self):
        status, report = self.run_cli(str(self.deck), "--lands", "1")
        self.assertEqual(status, 1)
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])
