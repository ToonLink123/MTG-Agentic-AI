import tempfile
import unittest
from pathlib import Path

from mtg_agentic.discovery import Candidate
from mtg_agentic.lands import (
    allocate_sources,
    calculate_land_base,
    is_land,
    land_sources,
    mana_demand,
)
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.pipeline_fixtures import make_pool, make_spec, records


class LandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OracleStore(Path(self.directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = records()
        self.store.import_cards(self.records)

    def test_early_pips_and_repeated_symbols_get_more_weight(self):
        early = mana_demand(card("Early", mana_cost="{W}"))
        late = mana_demand(card("Late", mana_cost="{6}{U}"))
        double = mana_demand(card("Double", mana_cost="{W}{W}"))
        self.assertGreater(early["W"], late["U"])
        self.assertEqual(double["W"], 2 * early["W"])
        self.assertEqual(mana_demand(card("Hybrid", mana_cost="{W/U}")), {"W": 1.5, "U": 1.5})
        self.assertEqual(mana_demand(card("Phyrexian", mana_cost="{W/P}")), {"W": 1.5})

    def test_largest_remainders_always_sum_to_requested_count(self):
        for total in (0, 2, 5, 36, 99):
            allocation = allocate_sources({"W": 5, "U": 2, "B": 1}, total)
            self.assertEqual(sum(allocation.values()), total)
            if total >= 3:
                self.assertTrue(all(value >= 1 for value in allocation.values()))
        self.assertGreater(allocate_sources({"W": 5, "U": 1}, 36)["W"], 18)

    def test_sources_use_produced_mana_and_legacy_text_without_activation_costs(self):
        self.assertEqual(land_sources(card("Dual", types="Land — Plains Island"), "WU"), {"W", "U"})
        self.assertEqual(
            land_sources(card("Tower", types="Land", produced_mana=list("WUBRG")), "WU"), {"W", "U"}
        )
        filter_land = card("Filter", types="Land", text="{R}, {T}: Add {W}{U}.")
        self.assertEqual(land_sources(filter_land, "WUR"), {"W", "U"})
        fetch = card("Fetch", types="Land", text="Search your library for a Forest or Island card.")
        self.assertEqual(land_sources(fetch, "WUBRG"), {"G", "U"})

    def test_front_face_defines_land_slot_not_a_land_back_face(self):
        modal = card(
            "Spell // Land",
            types="Sorcery // Land",
            card_faces=[
                {"type_line": "Sorcery", "name": "Spell", "oracle_text": ""},
                {"type_line": "Land", "name": "Land", "oracle_text": ""},
            ],
        )
        self.assertFalse(is_land(modal))

    def test_generated_base_is_deterministic_and_exact_size(self):
        spec = make_spec()
        pool = make_pool(self.store, spec)
        spells = {f"Spell {n}": 1 for n in range(63)}
        first = calculate_land_base(spells, spec, pool, self.store)
        self.assertEqual(first, calculate_land_base(spells, spec, pool, self.store))
        self.assertEqual(first["land_count"], 36)
        self.assertEqual(first["errors"], [])
        self.assertEqual(sum(first["cards"].values()), 36)
        self.assertTrue(all(first["source_counts"].get(c, 0) > 0 for c in "WUBRG"))
        self.assertNotIn("Wastes", first["cards"])

    def test_required_nonbasic_land_pinned_and_singleton(self):
        dual = card("Test Dual", types="Land", produced_mana=["W", "U"])
        self.store.import_cards([*self.records, dual])
        spec = make_spec(must_include=["Test Dual"])
        pool = make_pool(self.store, spec)
        pool.cards.append(Candidate(self.store.get_card("Test Dual"), {"lands"}))
        result = calculate_land_base({}, spec, pool, self.store)
        self.assertEqual(result["cards"]["Test Dual"], 1)
        self.assertEqual(result["land_count"], 36)

    def test_land_selection_does_not_spend_used_game_changer_budget(self):
        changed = [
            dict(raw, game_changer=True)
            if raw["name"] in {"Spell 0", "Spell 1", "Spell 2"}
            else raw
            for raw in self.records
        ]
        land = card("Changer Land", types="Land", produced_mana=list("WUBRG"), changer=True)
        self.store.import_cards([*changed, land])
        spec = make_spec()
        pool = make_pool(self.store, spec)
        pool.cards.append(Candidate(self.store.get_card("Changer Land"), {"lands"}))
        result = calculate_land_base(
            {"Spell 0": 1, "Spell 1": 1, "Spell 2": 1}, spec, pool, self.store
        )
        self.assertNotIn("Changer Land", result["cards"])

    def test_excluded_basics_fail_explicitly_instead_of_inventing_lands(self):
        spec = make_spec(exclude=["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"])
        result = calculate_land_base({}, spec, make_pool(self.store, spec), self.store)
        self.assertTrue(result["errors"])
        self.assertEqual(result["land_count"], 0)

    def test_colorless_costs_get_wastes_even_in_colored_decks(self):
        spell = card("Needs Colorless", mana_cost="{2}{C}")
        self.store.import_cards([*self.records, spell])
        result = calculate_land_base(
            {"Needs Colorless": 4}, make_spec(), make_pool(self.store), self.store
        )
        self.assertGreater(result["cards"].get("Wastes", 0), 0)

    def test_negative_or_boolean_source_counts_are_rejected(self):
        for total in (-1, True):
            with self.assertRaises(ValueError):
                allocate_sources({"W": 1}, total)
