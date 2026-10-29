import tempfile
import unittest
from pathlib import Path

from mtg_agentic.discovery import Candidate, eligible
from mtg_agentic.lands import (
    allocate_sources,
    calculate_land_base,
    is_land,
    land_sources,
    mana_demand,
)
from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import draft_deck
from mtg_agentic.schema import DRAFT_ROLES
from tests.fixtures import card
from tests.land_fixtures import fixing_lands, land_tutors
from tests.pipeline_fixtures import FakeCards, FakeModel, make_pool, make_spec, records


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

    def test_sources_without_a_deck_do_not_assume_filters_or_fetch_targets(self):
        self.assertEqual(land_sources(card("Dual", types="Land — Plains Island"), "WU"), {"W", "U"})
        self.assertEqual(
            land_sources(card("Tower", types="Land", produced_mana=list("WUBRG")), "WU"), {"W", "U"}
        )
        filter_land = card("Filter", types="Land", text="{R}, {T}: Add {W}{U}.")
        self.assertEqual(land_sources(filter_land, "WUR"), set())
        fetch = card("Fetch", types="Land", text="Search your library for a Forest or Island card.")
        self.assertEqual(land_sources(fetch, "WUBRG"), set())

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
        # Thirty-six basics cannot supply ten sources of each of five colors.
        self.assertEqual({error["error"] for error in first["errors"]}, {"mana_sources"})
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

    def test_limited_land_destruction_mode_can_generate_required_scorched_ruins(self):
        raw = card(
            "Scorched Ruins", types="Land", produced_mana=["C"],
            text="If Scorched Ruins would enter, sacrifice two untapped lands instead. "
                 "{T}: Add {C}{C}{C}{C}.",
        )
        self.store.add_verified_card(raw)
        spec = make_spec(bracket3_mode="limited-land-destruction", must_include=["Scorched Ruins"])
        result = calculate_land_base({}, spec, make_pool(self.store, spec), self.store)
        self.assertEqual(result["cards"]["Scorched Ruins"], 1)
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

    def test_saved_pool_is_supplemented_with_real_duals_shocks_and_fetches(self):
        self.store.import_cards([*self.records, *fixing_lands()])
        spec = make_spec()
        result = calculate_land_base({}, spec, make_pool(self.store, spec), self.store)
        self.assertTrue(
            {"Tropical Island", "Breeding Pool", "Misty Rainforest", "Flooded Strand"}
            <= result["cards"].keys()
        )
        self.assertEqual(result["land_count"], 36)
        for name in ("Tropical Island", "Breeding Pool", "Misty Rainforest"):
            self.assertEqual(result["cards"][name], 1)
        self.assertNotIn("Simic Guildgate", result["cards"])

    def test_land_tutor_targets_are_pinned_even_when_basic_forest_is_excluded(self):
        self.store.import_cards([*self.records, *fixing_lands(), *land_tutors()])
        spec = make_spec(exclude=["Forest"])
        result = calculate_land_base(
            {"Nature's Lore": 1}, spec, make_pool(self.store, spec), self.store
        )
        targets = result["mana_validation"]["search_targets"]["Nature's Lore"]
        self.assertTrue(targets)
        self.assertTrue(set(targets) <= {"Tropical Island", "Breeding Pool"})
        self.assertNotIn("Forest", result["cards"])
        self.assertEqual(result["land_count"], 36)

    def test_missing_tutor_targets_remain_an_explicit_error(self):
        self.store.import_cards([*self.records, *land_tutors()])
        spec = make_spec(exclude=["Forest"])
        result = calculate_land_base(
            {"Nature's Lore": 1}, spec, make_pool(self.store, spec), self.store
        )
        self.assertIn(
            "land_search_without_targets", {error["error"] for error in result["errors"]}
        )

    def test_optional_premium_lands_still_respect_exclusions_and_brackets(self):
        lands = [
            dict(c, game_changer=True) if c["name"] == "Breeding Pool" else c
            for c in fixing_lands()
        ]
        self.store.import_cards([*self.records, *lands])
        spec = make_spec(bracket=2, exclude=["Tropical Island", "Misty Rainforest"])
        result = calculate_land_base({}, spec, make_pool(self.store, spec), self.store)
        self.assertFalse(
            {"Tropical Island", "Breeding Pool", "Misty Rainforest"} & result["cards"].keys()
        )

    def test_shock_land_is_preferred_over_an_always_tapped_dual(self):
        leader = card("Simic Leader", colors="UG", types="Legendary Creature — Elf")
        self.store.import_cards([*self.records, *fixing_lands(), leader])
        spec = make_spec(
            commander=leader["name"], color_identity="UG",
            exclude=["Tropical Island", "Misty Rainforest", "Flooded Strand", "Command Tower"],
            role_targets={"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "lands": 3},
        )
        pool = make_pool(self.store, spec)
        pool.cards.append(Candidate(self.store.get_card("Simic Guildgate"), {"lands"}))
        result = calculate_land_base({}, spec, pool, self.store)
        self.assertEqual(result["land_count"], 3)
        self.assertIn("Breeding Pool", result["cards"])
        self.assertNotIn("Simic Guildgate", result["cards"])

    def test_draft_repairs_land_dependencies_without_an_llm_land_selection(self):
        leader = card(
            "Simic Leader", colors="UG", types="Legendary Creature — Elf", mana_cost="{G}{U}"
        )
        spells = []
        for n in range(70):
            types, text = "Artifact", ""
            if n < 15:
                types = "Creature — Elf"
                text = ("Draw a card. Create a Treasure token. Destroy target artifact. "
                        "Your commander has hexproof.")
            elif n < 17:
                types, text = "Instant", "Target creature gains indestructible until end of turn."
            elif n < 20:
                types, text = "Sorcery", "Destroy all creatures."
            spells.append(card(
                f"Spell {n}", colors="G", types=types, text=text, mana_cost="{1}{G}", cmc=2
            ))
        data = [*self.records[:7], leader, *spells, *fixing_lands(), *land_tutors()]
        self.store.import_cards(data)
        spec = make_spec(
            commander=leader["name"], color_identity="UG",
            must_include=["Nature's Lore"],
        )
        pool = make_pool(self.store, spec)
        pool.cards = [c for c in pool.cards if eligible(c.card, spec)]
        pool.cards.append(Candidate(self.store.get_card("Nature's Lore"), {"ramp"}))
        # A saved pool deliberately has no premium lands, and the model picks none.
        buckets = {role: [] for role in DRAFT_ROLES}
        model = FakeModel({"commander": leader["name"], "buckets": buckets, "outside": []})
        client = FakeCards(data)
        result = draft_deck(pool, model, client, self.store, max_repairs=0)
        self.assertTrue(result["valid"], result["validation"]["errors"])
        self.assertEqual(sum(result["cards"].values()), 99)
        self.assertEqual(result["land_base"]["land_count"], 36)
        self.assertTrue(
            result["land_base"]["mana_validation"]["search_targets"]["Nature's Lore"]
        )
        self.assertIn("Breeding Pool", result["cards"])
        self.assertIn("Misty Rainforest", result["cards"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(client.get_calls, [])
