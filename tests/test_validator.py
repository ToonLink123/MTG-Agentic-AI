import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.oracle import OracleStore
from mtg_agentic.validator import DeckEntry, DeckValidator, validate_deck
from tests.fixtures import BOLT, COMMANDER, DFC, ISLAND, PLAINS, SOL_RING, WASTES, card


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OracleStore(Path(self.directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = [
            COMMANDER,
            PLAINS,
            ISLAND,
            WASTES,
            SOL_RING,
            BOLT,
            DFC,
            card("Snow-Covered Plains", colors="W", types="Basic Snow Land — Plains"),
            card("Banned Card", legal="banned"),
            card("Digital Card", legal="not_legal"),
            card("Missing Legality", legalities={}),
            card(
                "Relentless Rats",
                colors="B",
                text=("A deck can have any number of cards named Relentless Rats."),
            ),
            card(
                "Hare Apparent",
                colors="W",
                text=("A deck can have any number of cards named Hare Apparent."),
            ),
            card(
                "Seven Dwarves",
                colors="R",
                text=("A deck can have up to seven cards named Seven Dwarves."),
            ),
            card("Nazgûl", colors="B", text="A deck can have up to nine cards named Nazgûl."),
            card("Any Commander", colors="WUBRG", types="Legendary Creature — Shapeshifter"),
            *[card(f"Changer {number}", changer=True) for number in range(1, 5)],
        ]
        self.store.import_cards(self.records)

    def validate(self, deck, commander="Test Commander", **kwargs):
        # A network call during validation must fail the test.
        with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("No HTTP")):
            report = validate_deck(deck, commander, self.store, **kwargs)
        json.loads(json.dumps(report))
        return report

    def error_codes(self, report):
        return {error["error"] for error in report["errors"]}

    def test_valid_100_cards_including_commander_and_basic_exemptions(self):
        report = self.validate({"Plains": 40, "Island": 30, "Snow-Covered Plains": 20, "Wastes": 9})
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["card_count"], 100)
        self.assertEqual(report["color_identity"], ["W", "U"])
        self.assertEqual(report["game_changer_count"], 1)
        self.assertEqual(report["game_changers"], ["Test Commander"])

    def test_text_mapping_and_entry_input(self):
        forms = [
            "# maindeck\n98x Plains\n1 Sol Ring\n",
            {"Plains": 98, "Sol Ring": 1},
            ["98 Plains", "Sol Ring"],
            [DeckEntry("Plains", 98), DeckEntry("Sol Ring")],
        ]
        for deck in forms:
            with self.subTest(deck=deck):
                report = self.validate(deck)
                self.assertTrue(report["valid"], report)
        self.assertTrue(
            DeckValidator(self.store).validate({"Plains": 99}, "Test Commander")["valid"]
        )

    def test_card_count_too_small_and_too_large(self):
        for quantity in (98, 100):
            report = self.validate({"Plains": quantity})
            self.assertEqual(self.error_codes(report), {"card_count"})
            self.assertEqual(report["card_count"], quantity + 1)

    def test_singleton_case_and_face_aliases_cannot_bypass_rule(self):
        for deck, name in (
            (["97 Plains", "Sol Ring", "sol ring"], "Sol Ring"),
            (["97 Plains", "Front Face", DFC["name"]], DFC["name"]),
            (["98 Plains", "Test Commander"], "Test Commander"),
        ):
            report = self.validate(deck)
            self.assertIn("singleton", self.error_codes(report))
            self.assertEqual(
                next(e["card"] for e in report["errors"] if e["error"] == "singleton"), name
            )

    def test_color_identity_error_shape(self):
        report = self.validate({"Plains": 98, "Lightning Bolt": 1})
        self.assertIn(
            {"error": "color_identity", "card": "Lightning Bolt", "detail": "R outside WU"},
            report["errors"],
        )

    def test_all_legalities_other_than_legal_fail(self):
        report = self.validate(
            {"Plains": 96, "Banned Card": 1, "Digital Card": 1, "Missing Legality": 1}
        )
        self.assertEqual(self.error_codes(report), {"commander_legality"})
        self.assertEqual(len(report["errors"]), 3)

    def test_unknown_names_are_not_fuzzy_matched_and_quantities_still_count(self):
        report = self.validate(["97 Plains", "Sol Rng", "Invented Card"])
        self.assertEqual(report["card_count"], 100)
        self.assertEqual(self.error_codes(report), {"unknown_card"})
        self.assertEqual(len(report["errors"]), 2)

    def test_unknown_commander_reports_problem_without_inventing_color_errors(self):
        report = self.validate({"Plains": 98, "Lightning Bolt": 1}, commander="Invented Commander")
        self.assertEqual(self.error_codes(report), {"unknown_card"})

    def test_unlimited_duplicates_remain_subject_to_other_rules(self):
        report = self.validate({"Hare Apparent": 99})
        self.assertTrue(report["valid"], report)
        report = self.validate({"Relentless Rats": 99})
        self.assertEqual(self.error_codes(report), {"color_identity"})

    def test_limited_duplicate_exceptions(self):
        for name, maximum in (("Seven Dwarves", 7), ("Nazgûl", 9)):
            for quantity in (maximum, maximum + 1):
                report = self.validate(
                    {name: quantity, "Wastes": 99 - quantity}, commander="Any Commander"
                )
                self.assertEqual(report["valid"], quantity == maximum, report)
                if quantity > maximum:
                    self.assertIn("singleton", self.error_codes(report))

    def test_game_changer_boundary_includes_commander(self):
        for count in (2, 3):
            report = self.validate(
                {"Plains": 99 - count, **{f"Changer {n}": 1 for n in range(1, count + 1)}}
            )
            self.assertEqual(report["game_changer_count"], count + 1)
            self.assertEqual(report["valid"], count == 2)
            if count == 3:
                self.assertIn("game_changers", self.error_codes(report))

    def test_other_brackets_and_duplicate_game_changer_count(self):
        for bracket in range(1, 6):
            report = self.validate({"Plains": 99}, bracket=bracket)
            self.assertEqual(report["valid"], bracket >= 3, report)
        report = self.validate({"Plains": 95, "Changer 1": 4})
        self.assertEqual(report["game_changer_count"], 5)
        self.assertEqual(self.error_codes(report), {"singleton", "game_changers"})
        self.assertTrue(
            self.validate({"Plains": 95, **{f"Changer {n}": 1 for n in range(1, 5)}}, bracket=4)[
                "valid"
            ]
        )

    def test_invalid_quantities_entries_and_brackets(self):
        for deck in (
            {"Plains": 0},
            {"Plains": -1},
            {"Plains": True},
            {"Plains": 99.5},
            ["0 Plains"],
            [None],
            [{}, "99 Plains"],
            {None: 99},
        ):
            report = self.validate(deck)
            self.assertFalse(report["valid"])
            self.assertIn("invalid_entry", self.error_codes(report))
        for deck in (None, 10):
            self.assertIn("invalid_decklist", self.error_codes(self.validate(deck)))
        for bracket in (0, 6, True, "3", 3.0):
            self.assertIn(
                "invalid_bracket", self.error_codes(self.validate({"Plains": 99}, bracket=bracket))
            )

    def test_missing_or_multiple_commanders_are_not_silently_accepted(self):
        for commander in ("", None, ["Test Commander", "Any Commander"]):
            report = self.validate({"Plains": 99}, commander=commander)
            self.assertIn("invalid_commander", self.error_codes(report))

    def test_invalid_commander_type_and_banned_commander(self):
        report = self.validate({"Plains": 99}, commander="Sol Ring")
        self.assertIn("commander_type", self.error_codes(report))
        banned = dict(COMMANDER, legalities={"commander": "banned"})
        self.store.import_cards([banned, PLAINS])
        report = self.validate({"Plains": 99})
        self.assertEqual(self.error_codes(report), {"commander_legality"})

    def test_special_permission_and_front_face_commander_eligibility(self):
        vehicle = card(
            "Test Vehicle",
            types="Legendary Artifact — Vehicle",
            text="Test Vehicle can be your commander.",
            colors="W",
        )
        grist = card("Grist, the Hunger Tide", types="Legendary Planeswalker — Grist")
        back_only = dict(DFC)
        back_only["name"] = "Not a Commander // Commander Back"
        back_only["oracle_id"] = back_only["id"] = back_only["name"]
        back_only["card_faces"] = [
            {"name": "Not a Commander", "type_line": "Enchantment", "oracle_text": ""},
            {
                "name": "Commander Back",
                "type_line": "Legendary Creature — Human",
                "oracle_text": "",
            },
        ]
        self.store.import_cards([vehicle, grist, PLAINS, WASTES, DFC, back_only])
        for name in ("Test Vehicle", "Front Face"):
            self.assertTrue(self.validate({"Plains": 99}, commander=name)["valid"])
        self.assertTrue(self.validate({"Wastes": 99}, commander="Grist, the Hunger Tide")["valid"])
        report = self.validate({"Plains": 99}, commander=back_only["name"])
        self.assertIn("commander_type", self.error_codes(report))

    def test_colorless_commander_does_not_allow_colored_cards(self):
        leader = card("Colorless Commander", types="Legendary Creature — Golem")
        self.store.import_cards([leader, PLAINS, WASTES])
        self.assertTrue(self.validate({"Wastes": 99}, commander=leader["name"])["valid"])
        report = self.validate({"Plains": 99}, commander=leader["name"])
        self.assertIn("color_identity", self.error_codes(report))

    def test_uninitialized_store_fails_closed(self):
        with OracleStore(Path(self.directory.name) / "empty.sqlite3") as empty:
            report = validate_deck({"Plains": 99}, "Test Commander", empty)
        self.assertFalse(report["valid"])
        self.assertEqual(self.error_codes(report), {"oracle_unavailable"})
