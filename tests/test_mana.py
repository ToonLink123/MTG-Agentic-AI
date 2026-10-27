import tempfile
import unittest
from pathlib import Path

from mtg_agentic.mana import source_graph
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.land_fixtures import fixing_lands, land_tutors
from tests.pipeline_fixtures import records


class LandSearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards([*records(), *fixing_lands(), *land_tutors()])

    def graph(self, *names):
        return source_graph(dict.fromkeys(names, 1), "UG", self.store)

    def test_natures_lore_fetches_forest_typed_duals_not_rainbow_lands(self):
        graph = self.graph("Nature's Lore", "Tropical Island", "Breeding Pool", "Command Tower")
        self.assertEqual(
            graph["search_targets"]["Nature's Lore"], ["Breeding Pool", "Tropical Island"]
        )
        self.assertEqual(graph["errors"], [])
        graph = self.graph("Nature's Lore", "Command Tower", "Island")
        self.assertEqual(graph["errors"][0]["error"], "land_search_without_targets")

    def test_farseek_can_find_a_shock_but_not_a_basic_forest(self):
        self.assertEqual(
            self.graph("Farseek", "Breeding Pool", "Forest")["search_targets"]["Farseek"],
            ["Breeding Pool"],
        )
        self.assertTrue(self.graph("Farseek", "Forest")["errors"])

    def test_basic_land_search_cannot_find_a_nonbasic_with_basic_types(self):
        self.assertTrue(self.graph("Cultivate", "Breeding Pool")["errors"])
        graph = self.graph("Cultivate", "Breeding Pool", "Forest")
        self.assertEqual(graph["search_targets"]["Cultivate"], ["Forest"])
        self.assertEqual(graph["errors"], [])

    def test_fetch_credit_uses_only_real_targets_and_their_actual_colors(self):
        graph = self.graph("Misty Rainforest", "Forest")
        self.assertEqual(graph["land_sources"]["Misty Rainforest"], ["G"])
        graph = self.graph("Misty Rainforest", "Forest", "Breeding Pool")
        self.assertEqual(graph["land_sources"]["Misty Rainforest"], ["G", "U"])
        self.assertEqual(graph["source_counts"], {"G": 3, "U": 2})

    def test_fetch_can_fix_a_color_not_named_in_its_search(self):
        graph = self.graph("Flooded Strand", "Breeding Pool")
        self.assertEqual(graph["land_sources"]["Flooded Strand"], ["G", "U"])
        self.assertEqual(graph["errors"], [])

    def test_fetches_without_targets_do_not_create_sources(self):
        graph = self.graph("Misty Rainforest", "Flooded Strand")
        self.assertEqual(graph["source_counts"], {})
        self.assertEqual(len(graph["errors"]), 2)

    def test_each_search_clause_needs_its_own_target(self):
        self.store.add_verified_card(card(
            "Two Searches", types="Sorcery",
            text="Search your library for a Forest card. "
                 "Search your library for an Island card.",
        ))
        graph = self.graph("Two Searches", "Forest")
        self.assertEqual(graph["errors"][0]["error"], "land_search_without_targets")
        self.assertEqual(graph["errors"][0]["missing_selectors"][0]["types"], ["Island"])

    def test_unknown_search_restrictions_are_not_silently_certified(self):
        self.store.add_verified_card(card(
            "Restricted Search", types="Sorcery",
            text="Search your library for a land card with mana value 1 or less.",
        ))
        graph = self.graph("Restricted Search", "Forest")
        self.assertEqual(graph["errors"][0]["error"], "unsupported_land_search")
