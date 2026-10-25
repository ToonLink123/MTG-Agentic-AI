import tempfile
import unittest
import zipfile
from pathlib import Path

from mtg_agentic.forge_support import ForgeSupport
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.pipeline_fixtures import JODAH, records


class SupportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.cards = self.path / "cards"
        self.cards.mkdir()
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(records())
        with zipfile.ZipFile(self.cards / "cardsfolder.zip", "w") as z:
            z.writestr("j/jodah.txt", "Name:Jodah, Archmage Eternal\nAI:RemoveDeck:Random\n")
            z.writestr("p/plains.txt", "Name:Plains\nTypes:Basic Land Plains\n")
            z.writestr("f/front.txt", "Name:Front\nALTERNATE\nName:Back\n")
        self.deck = {"commander": JODAH["name"], "cards": {"Plains": 99}}

    def test_script_presence_is_distinct_from_ai_warnings(self):
        report = ForgeSupport(self.cards).check(self.deck, self.store)
        self.assertTrue(report["supported"])
        self.assertEqual(len(report["ai_warnings"]), 1)
        self.assertEqual(report["missing"], [])
        self.assertTrue(report["scripts_fingerprint"])

    def test_missing_and_locally_known_misbehavior_are_reported(self):
        self.deck["cards"] = {"Spell 0": 1, "Plains": 98}
        support = ForgeSupport(self.cards, known_issues={"Plains": "Observed bug, issue reference"})
        report = support.check(self.deck, self.store)
        self.assertFalse(report["supported"])
        self.assertEqual(report["missing"][0]["card"], "Spell 0")
        self.assertEqual(report["known_issues"][0]["card"], "Plains")

    def test_missing_back_face_is_not_treated_as_full_support(self):
        self.store.add_verified_card(
            card(
                "Front // Missing",
                card_faces=[
                    {"name": "Front"},
                    {"name": "Missing"},
                ],
            )
        )
        self.deck["cards"] = {"Front // Missing": 1, "Plains": 98}
        report = ForgeSupport(self.cards).check(self.deck, self.store)
        self.assertFalse(report["supported"])
        self.assertIn("face", report["missing"][0]["reason"])

    def test_loose_scripts_override_archives_and_change_fingerprint(self):
        before = ForgeSupport(self.cards)
        (self.cards / "plains.txt").write_text("Name:Plains\nAI:RemoveDeck:All\n")
        after = ForgeSupport(self.cards)
        self.assertNotEqual(before.fingerprint, after.fingerprint)
        self.assertEqual(after.scripts["plains"]["ai_hints"], ["AI:RemoveDeck:All"])
