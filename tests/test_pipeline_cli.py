import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.__main__ import main
from mtg_agentic.oracle import OracleStore
from tests.pipeline_fixtures import (
    JODAH,
    FakeCards,
    FakeModel,
    draft,
    make_pool,
    make_spec,
    records,
)


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.cards = records()
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards(self.cards)
            self.pool_data = make_pool(store).to_dict()

    def run_cli(self, *args, model=None):
        if args and args[0] in {"draft", "build"}:
            args = (*args, "--skip-review")
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("mtg_agentic.__main__._make_model", return_value=model or FakeModel()),
            patch(
                "mtg_agentic.scryfall.ScryfallClient.get_card",
                side_effect=FakeCards(self.cards).get_card,
            ),
            patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("No real HTTP")),
        ):
            status = main(["--data-dir", str(self.path), *args])
        return status, json.loads(output.getvalue())

    def test_schema_command_needs_no_model_or_network(self):
        status, schema = self.run_cli("spec-schema")
        self.assertEqual(status, 0)
        self.assertIn("role_targets", schema["properties"])

    def test_intake_writes_json_with_no_bom_and_never_overwrites(self):
        output = self.path / "spec.json"
        args = (
            "intake",
            "--commander",
            JODAH["name"],
            "--request",
            "Myojin",
            "--output",
            str(output),
        )
        status, summary = self.run_cli(*args, model=FakeModel(make_spec().to_dict()))
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), make_spec().to_dict())
        self.assertEqual(summary["output"], str(output))
        before = output.read_bytes()
        status, result = self.run_cli(*args)
        self.assertEqual(status, 2)
        self.assertEqual(output.read_bytes(), before)

    def test_draft_artifact_can_be_passed_to_existing_validator(self):
        pool_file = self.path / "pool.json"
        deck_file = self.path / "deck.json"
        pool_file.write_text(json.dumps(self.pool_data), encoding="utf-8")
        status, result = self.run_cli(
            "draft", "--pool", str(pool_file), "--output", str(deck_file), model=FakeModel(draft())
        )
        self.assertEqual(status, 0)
        self.assertTrue(result["valid"])
        status, report = self.run_cli("validate", str(deck_file))
        self.assertEqual(status, 0)
        self.assertTrue(report["valid"])

    def test_exhausted_draft_returns_exit_one_and_preserves_errors(self):
        pool_file = self.path / "pool.json"
        pool_file.write_text(json.dumps(self.pool_data), encoding="utf-8")
        status, result = self.run_cli(
            "draft", "--pool", str(pool_file), "--repairs", "0", model=FakeModel(draft(62))
        )
        self.assertEqual(status, 1)
        self.assertFalse(result["valid"])
        self.assertTrue(result["validation"]["errors"])

    def test_malformed_spec_returns_machine_readable_schema_errors(self):
        spec_file = self.path / "bad-spec.json"
        spec_file.write_text("{}", encoding="utf-8")
        status, report = self.run_cli("discover", "--spec", str(spec_file))
        self.assertEqual(status, 2)
        self.assertEqual(report["errors"][0]["error"], "schema")

    def test_bracket_four_survives_draft_export_and_standalone_revalidation(self):
        changed = [
            dict(raw, game_changer=True) if raw["name"] in {f"Spell {n}" for n in range(4)} else raw
            for raw in self.cards
        ]
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards(changed)
            data = make_pool(store, make_spec(bracket=4)).to_dict()
        pool_file = self.path / "bracket-four-pool.json"
        deck_file = self.path / "bracket-four-deck.json"
        pool_file.write_text(json.dumps(data), encoding="utf-8")
        status, _ = self.run_cli(
            "draft",
            "--pool",
            str(pool_file),
            "--output",
            str(deck_file),
            model=FakeModel(draft()),
        )
        self.assertEqual(status, 0)
        status, report = self.run_cli("validate", str(deck_file))
        self.assertEqual(status, 0)
        self.assertEqual(report["bracket"], 4)
        self.assertEqual(report["game_changer_count"], 4)
