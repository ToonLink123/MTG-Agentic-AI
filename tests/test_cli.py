import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.__main__ import main
from mtg_agentic.oracle import OracleStore
from tests.fixtures import BOLT, COMMANDER, PLAINS


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.data_dir = self.path / "data"
        with OracleStore(self.data_dir / "oracle.sqlite3") as store:
            store.import_cards([COMMANDER, PLAINS, BOLT])

    def run_cli(self, *args):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("No HTTP")),
        ):
            status = main(["--data-dir", str(self.data_dir), *args])
        return status, json.loads(output.getvalue())

    def test_valid_and_invalid_json_reports(self):
        deck = self.path / "deck.json"
        for cards, expected in (({"Plains": 99}, 0), ({"Plains": 98, "Lightning Bolt": 1}, 1)):
            deck.write_text(
                json.dumps({"commander": "Test Commander", "cards": cards}), encoding="utf-8"
            )
            status, report = self.run_cli("validate", str(deck))
            self.assertEqual(status, expected)
            self.assertEqual(report["valid"], expected == 0)

    def test_text_and_bracket_override(self):
        deck = self.path / "deck.txt"
        deck.write_text("99 Plains", encoding="utf-8-sig")
        status, report = self.run_cli("validate", str(deck), "--commander", "Test Commander")
        self.assertEqual(status, 0)
        status, report = self.run_cli(
            "validate", str(deck), "--commander", "Test Commander", "--bracket", "2"
        )
        self.assertEqual(status, 1)
        self.assertEqual(report["errors"][0]["error"], "game_changers")

    def test_input_errors_are_json(self):
        deck = self.path / "deck.txt"
        deck.write_text("99 Plains", encoding="utf-8")
        self.assertEqual(self.run_cli("validate", str(deck))[0], 2)
        self.assertEqual(self.run_cli("validate", str(self.path / "missing.json"))[0], 2)
        deck = self.path / "deck.json"
        for contents in ("{", "[]"):
            deck.write_text(contents, encoding="utf-8")
            status, report = self.run_cli("validate", str(deck))
            self.assertEqual(status, 2)
            self.assertEqual(report["errors"][0]["error"], "input")
