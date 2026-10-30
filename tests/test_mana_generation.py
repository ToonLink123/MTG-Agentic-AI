import tempfile
import unittest
from pathlib import Path

from mtg_agentic import generate_mana_base
from mtg_agentic.mana import mana_pips, mana_profile, validate_mana_base
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.land_fixtures import fixing_lands, land_tutors
from tests.pipeline_fixtures import make_spec, records


class ManaGenerationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.cards = [
            *records(), *fixing_lands(), *land_tutors(),
            card("Green Leader", colors="G", types="Legendary Creature — Elf",
                 mana_cost="{G}"),
            card("Simic Leader", colors="UG", types="Legendary Creature — Elf",
                 mana_cost="{G}{U}"),
            card("Sparse Leader", colors="UG", types="Legendary Creature — Elf",
                 mana_cost="{G}"),
            card("Colorless Leader", types="Legendary Creature — Golem", mana_cost="{6}"),
            card("Green Pips", colors="G", mana_cost="{G}{G}{G}"),
            card("Blue Pips", colors="U", mana_cost="{U}"),
            card("Generic Only", mana_cost="{6}"),
            card("Colorless Pip", mana_cost="{2}{C}"),
            card("Boseiju, Who Endures", colors="G", types="Legendary Land",
                 text="{T}: Add {G}.\nChannel — {1}{G}, Discard Boseiju, Who Endures: "
                      "Destroy target artifact."),
            card("War Room", types="Land",
                 text="{T}: Add {C}.\n{3}, {T}, Pay life: Draw a card."),
        ]
        self.store.import_cards(self.cards)

    def spec(self, commander="Green Leader", identity="G", **changes):
        return make_spec(commander=commander, color_identity=identity, **changes)

    def test_mono_deck_is_mostly_basics_with_on_color_utility(self):
        result = generate_mana_base({"Green Pips": 1}, self.spec(), self.store)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["strategy"], "monocolor")
        self.assertGreaterEqual(result["cards"]["Forest"], 32)
        self.assertIn("Boseiju, Who Endures", result["cards"])
        self.assertIn("War Room", result["cards"])
        self.assertEqual(result["cards"]["Boseiju, Who Endures"], 1)
        self.assertFalse(
            {"Command Tower", "City of Brass", "Mana Confluence",
             "Misty Rainforest", "Tropical Island", "Breeding Pool"} & result["cards"].keys()
        )
        self.assertEqual(result["land_count"], 36)
        self.assertEqual(result["selection_reasons"]["Boseiju, Who Endures"], "utility")

    def test_multicolor_uses_rainbow_duals_shocks_and_grounded_fetches(self):
        result = generate_mana_base(
            {"Green Pips": 1, "Blue Pips": 1}, self.spec("Simic Leader", "UG"), self.store
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["strategy"], "multicolor")
        self.assertTrue(
            {"Command Tower", "Tropical Island", "Breeding Pool", "Misty Rainforest"}
            <= result["cards"].keys()
        )
        self.assertTrue(result["mana_validation"]["search_targets"]["Misty Rainforest"])

    def test_five_color_fixing_does_not_stop_at_minimal_color_floors(self):
        from mtg_agentic.land_options import PREFERRED_LANDS
        from mtg_agentic.mana import BASIC_LANDS

        pairs = ["WU", "UB", "BR", "RG", "GW", "WB", "UR", "BG", "RW", "GU"]
        premium = []
        for offset in (0, 10):
            for name, pair in zip(PREFERRED_LANDS[offset:offset + 10], pairs):
                text = "({T}: Add {" + pair[0] + "} or {" + pair[1] + "}.)"
                if offset:
                    text += f" As {name} enters, you may pay 2 life. If you don't, it enters tapped."
                premium.append(card(
                    name, colors=pair, types="Land — " + " ".join(BASIC_LANDS[c] for c in pair),
                    text=text,
                ))
        for name, pair in zip(PREFERRED_LANDS[20:30], pairs):
            premium.append(card(
                name, types="Land", text="{T}, Pay 1 life, Sacrifice this land: "
                f"Search your library for a {BASIC_LANDS[pair[0]]} or {BASIC_LANDS[pair[1]]} "
                "card, put it onto the battlefield, then shuffle.",
            ))
        self.store.import_cards(list({c["name"]: c for c in [*self.cards, *premium]}.values()))
        result = generate_mana_base({"Green Pips": 1, "Blue Pips": 1}, make_spec(), self.store)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(all(result["source_targets"][c] >= 18 for c in "WUBRG"))
        self.assertTrue(all(result["source_counts"][c] >= 18 for c in "WUBRG"))
        self.assertGreaterEqual(len(set(PREFERRED_LANDS[20:30]) & result["cards"].keys()), 6)
        self.assertLess(result["basic_count"], 18)

    def test_pip_proportions_shift_basics_toward_the_heavier_color(self):
        spec = self.spec("Simic Leader", "UG")
        green = generate_mana_base(
            {"Green Pips": 8, "Blue Pips": 1}, spec, self.store
        )
        blue = generate_mana_base(
            {"Green Pips": 1, "Blue Pips": 30}, spec, self.store
        )
        self.assertGreater(green["cards"]["Forest"], green["cards"]["Island"])
        self.assertGreater(blue["cards"]["Island"], blue["cards"]["Forest"])
        self.assertGreater(green["demand_share"]["G"], blue["demand_share"]["G"])
        self.assertEqual(green["mana_pips"], {"G": 25.0, "U": 2.0})

    def test_unused_commander_identity_does_not_invent_pip_demand(self):
        spec = self.spec("Sparse Leader", "UG")
        result = generate_mana_base({"Green Pips": 1}, spec, self.store)
        self.assertEqual(result["strategy"], "monocolor")
        self.assertNotIn("U", result["weighted_pips"])
        self.assertNotIn("Island", result["cards"])
        self.assertNotIn("Command Tower", result["cards"])
        self.assertEqual(result["mana_validation"]["minimums"], {"G": 10})

    def test_optional_utilities_use_demanded_colors_not_unused_identity_colors(self):
        self.store.add_verified_card(card(
            "Otawara, Soaring City", colors="U", types="Legendary Land",
            text="{T}: Add {U}.\nChannel — {3}{U}, Discard Otawara, Soaring City: "
                 "Return target artifact to its owner's hand.",
        ))
        result = generate_mana_base(
            {"Green Pips": 1}, self.spec("Sparse Leader", "UG"), self.store
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertIn("Boseiju, Who Endures", result["cards"])
        self.assertIn("War Room", result["cards"])
        self.assertNotIn("Otawara, Soaring City", result["cards"])
        self.assertNotIn("U", result["source_counts"])
        self.assertGreaterEqual(result["basic_count"], 32)

    def test_required_utility_can_override_the_demanded_color_preference(self):
        self.store.add_verified_card(card(
            "Otawara, Soaring City", colors="U", types="Legendary Land",
            text="{T}: Add {U}.\nChannel — {3}{U}, Discard Otawara, Soaring City: "
                 "Return target artifact to its owner's hand.",
        ))
        result = generate_mana_base(
            {"Green Pips": 1},
            self.spec("Sparse Leader", "UG", must_include=["Otawara, Soaring City"]),
            self.store,
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["cards"]["Otawara, Soaring City"], 1)
        self.assertEqual(result["selection_reasons"]["Otawara, Soaring City"], "required")

    def test_hybrid_phyrexian_and_generic_symbols_have_fractional_pip_demand(self):
        spell = card("Mixed Symbols", mana_cost="{X}{3}{G/U}{2/G}{U/P}{C}")
        self.assertEqual(mana_pips(spell), {"G": 1.0, "U": 1.0, "C": 1.0})

    def test_commander_activated_costs_are_not_confused_with_mana_outputs(self):
        leader = card(
            "Ability Leader", colors="UG", types="Legendary Creature — Elf", mana_cost="{G}",
            text="{U}, {T}: Draw a card.\n{T}: Add {G}.",
        )
        self.store.add_verified_card(leader)
        profile = mana_profile({}, leader["name"], self.store)
        self.assertEqual(profile["mana_pips"], {"G": 1.0})
        self.assertEqual(profile["ability_pips"], {"U": 1.0})
        self.assertEqual(profile["strategy"], "multicolor")

    def test_commander_alternate_cost_retains_all_required_colors(self):
        profile = mana_profile({}, make_spec().commander, self.store)
        self.assertEqual(set(profile["colors"]), set("WUBRG"))
        self.assertEqual(profile["ability_pips"], dict.fromkeys("WUBRG", 1.0))

    def test_generic_costs_do_not_create_colorless_pip_demand(self):
        self.assertEqual(mana_pips(self.store.get_card("Generic Only")), {})
        result = generate_mana_base({"Generic Only": 1}, self.spec(), self.store)
        self.assertNotIn("C", result["weighted_pips"])
        self.assertNotIn("Wastes", result["cards"])

    def test_explicit_colorless_cost_gets_real_colorless_sources(self):
        result = generate_mana_base({"Colorless Pip": 1}, self.spec(), self.store)
        self.assertGreater(result["source_counts"].get("C", 0), 0)
        self.assertEqual(result["mana_validation"]["minimums"]["C"], 1)
        self.assertEqual(result["land_count"], 36)

    def test_colorless_deck_uses_wastes_and_colorless_utility(self):
        result = generate_mana_base(
            {"Generic Only": 1}, self.spec("Colorless Leader", "C"), self.store
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["strategy"], "colorless")
        self.assertGreaterEqual(result["cards"]["Wastes"], 32)
        self.assertIn("War Room", result["cards"])
        self.assertNotIn("Command Tower", result["cards"])

    def test_utility_allowance_is_bounded_and_off_color_utilities_are_rejected(self):
        utilities = [
            card(f"Green Utility {n}", colors="G", types="Land",
                 text="{T}: Add {G}.\n{2}, {T}: Draw a card.")
            for n in range(10)
        ]
        off_color = card("Blue Utility", colors="U", types="Land",
                         text="{T}: Add {U}.\n{2}, {T}: Draw a card.")
        self.store.import_cards([*self.cards, *utilities, off_color])
        result = generate_mana_base(
            {"Green Pips": 1}, self.spec(), self.store,
            candidate_names=[c["name"] for c in [*utilities, off_color]],
        )
        self.assertEqual(result["utility_count"], result["utility_limit"])
        self.assertEqual(result["utility_limit"], 4)
        self.assertGreaterEqual(result["basic_count"], 32)
        self.assertNotIn("Blue Utility", result["cards"])

    def test_required_lands_override_optional_mono_policy_but_exclusions_apply(self):
        result = generate_mana_base(
            {}, self.spec(must_include=["Command Tower"], exclude=["Boseiju, Who Endures"]),
            self.store,
        )
        self.assertEqual(result["cards"]["Command Tower"], 1)
        self.assertNotIn("Boseiju, Who Endures", result["cards"])
        self.assertEqual(result["selection_reasons"]["Command Tower"], "required")

    def test_generator_rejects_unknown_land_or_invalid_spell_quantities(self):
        for spells in ({"Unknown": 1}, {"Forest": 1}, {"Green Pips": True}, {"Green Pips": 0}):
            with self.subTest(spells=spells), self.assertRaises(ValueError):
                generate_mana_base(spells, self.spec(), self.store)

    def test_utility_colorless_mana_does_not_count_as_colored_fixing(self):
        report = validate_mana_base({"Forest": 8, "War Room": 1}, "Green Leader", self.store)
        self.assertFalse(report["valid"])
        self.assertEqual(report["source_counts"], {"G": 8, "C": 1})

    def test_multicolor_takes_fast_popular_nonbasics_first_and_basics_last(self):
        duals = [
            card(f"Simic Dual {n}", colors="UG", types="Land", text="{T}: Add {G} or {U}.",
                 edhrec_rank=100 + n)
            for n in range(20)
        ]
        unpopular = card("Obscure Dual", colors="UG", types="Land",
                         text="{T}: Add {G} or {U}.", edhrec_rank=5000)
        self.store.import_cards([*self.cards, *duals, unpopular])
        result = generate_mana_base(
            {"Green Pips": 1, "Blue Pips": 1}, self.spec("Simic Leader", "UG"), self.store
        )
        self.assertTrue(result["valid"], result["errors"])
        cards = result["cards"]
        self.assertTrue({d["name"] for d in duals} <= cards.keys())
        self.assertTrue({"Tropical Island", "Breeding Pool", "Misty Rainforest",
                         "Command Tower"} <= cards.keys())
        # Always tapped, outside EDHREC's most-played lands, or a fetch for a blue-white type.
        self.assertFalse({"Simic Guildgate", "Obscure Dual", "Flooded Strand"} & cards.keys())
        # Basics fill only what good nonbasics and two utility lands leave.
        self.assertEqual(result["basic_reserve"], 2)
        self.assertEqual(result["basic_count"], 36 - 24 - result["utility_count"])

    def test_with_little_room_the_more_played_land_wins(self):
        popular = card("Popular Dual", colors="UG", types="Land", text="{T}: Add {G} or {U}.",
                       edhrec_rank=50)
        obscure = card("Middling Dual", colors="UG", types="Land", text="{T}: Add {G} or {U}.",
                       edhrec_rank=1500)
        self.store.import_cards([*self.cards, popular, obscure])
        targets = {"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "lands": 3}
        result = generate_mana_base(
            {"Green Pips": 1, "Blue Pips": 1},
            self.spec("Simic Leader", "UG", role_targets=targets), self.store,
        )
        self.assertEqual(result["cards"], {"Forest": 1, "Island": 1, "Popular Dual": 1})

    def test_each_basic_land_search_keeps_another_basic(self):
        result = generate_mana_base(
            {"Green Pips": 1, "Blue Pips": 1, "Cultivate": 1},
            self.spec("Simic Leader", "UG"), self.store,
        )
        self.assertEqual(result["basic_reserve"], 3)
        self.assertGreaterEqual(result["basic_count"], 3)

    def test_plain_basics_come_before_snow_covered_ones(self):
        # "Snow-Covered Swamp" sorts before "Swamp", which once decided the tie.
        leader = card("Black Leader", colors="B", types="Legendary Creature — Human",
                      mana_cost="{B}")
        snow = card("Snow-Covered Swamp", colors="B", types="Basic Snow Land — Swamp",
                    text="({T}: Add {B}.)")
        self.store.import_cards([*self.cards, leader, snow])
        result = generate_mana_base({}, self.spec("Black Leader", "B"), self.store)
        self.assertNotIn("Snow-Covered Swamp", result["cards"])
        self.assertGreaterEqual(result["cards"]["Swamp"], 32)
