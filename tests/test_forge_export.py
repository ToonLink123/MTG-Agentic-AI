import tempfile
import unittest
from pathlib import Path

from mtg_agentic.forge import ForgeError, export_deck
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.pipeline_fixtures import JODAH, records


class ExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(records())
        self.deck = {"commander": JODAH["name"], "bracket": 3, "cards": {"Plains": 99}}

    def test_commander_is_separate_and_exports_do_not_overwrite(self):
        path = self.path / "deck.dck"
        result = export_deck(self.deck, self.store, path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("[Commander]\n1 " + JODAH["name"] + "\n[Main]", text)
        self.assertIn("99 Plains", text)
        self.assertFalse(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(result["safety_review"], "not_run")
        with self.assertRaises(FileExistsError):
            export_deck(self.deck, self.store, path)

    def test_invalid_or_partner_decks_are_rejected_before_writing(self):
        for changes in (
            {"cards": {"Plains": 98}},
            {"cards": {"Invented": 99}},
            {"commander": [JODAH["name"], "Partner"]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ForgeError):
                export_deck({**self.deck, **changes}, self.store, self.path / "bad.dck")
        self.assertFalse((self.path / "bad.dck").exists())

    def test_multiface_cards_use_exact_front_names(self):
        dfc = card(
            "Front // Back",
            card_faces=[
                {"name": "Front", "type_line": "Creature", "oracle_text": ""},
                {"name": "Back", "type_line": "Creature", "oracle_text": ""},
            ],
        )
        self.store.add_verified_card(dfc)
        self.deck["cards"] = {"Plains": 98, "Front // Back": 1}
        path = self.path / "dfc.dck"
        export_deck(self.deck, self.store, path)
        self.assertIn("1 Front\n", path.read_text())
        self.assertNotIn("Front // Back", path.read_text())

    def test_metadata_and_card_line_injection_is_rejected(self):
        with self.assertRaises(ValueError):
            export_deck(self.deck, self.store, self.path / "deck.dck", name="X\n[Main]")
