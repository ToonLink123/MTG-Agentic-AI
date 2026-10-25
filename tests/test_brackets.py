import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.brackets import (
    POLICY,
    BracketChecker,
    SpellbookClient,
    deterministic_signals,
    game_changers,
)
from mtg_agentic.cards import validation_card
from mtg_agentic.oracle import OracleStore
from mtg_agentic.sources import SourceCache, SourceError
from tests.fixtures import card
from tests.pipeline_fixtures import JODAH, FakeModel, records


def assessment(**changes):
    result = {
        "fit": "fits",
        "estimated_bracket": 3,
        "curve": "Midrange curve",
        "interaction": "Limited removal",
        "likely_win_turn_min": 7,
        "likely_win_turn_max": 10,
        "reasons": ["Slow, visible game plan"],
        "card_references": [],
        "extra_turn_chaining": "unlikely",
        "combo_reviews": [],
    }
    result.update(changes)
    return result


def variant(id_="combo", names=("Spell 0", "Spell 1"), **changes):
    result = {
        "id": id_,
        "status": "OK",
        "uses": [
            {
                "card": {"name": n},
                "quantity": 1,
                "mustBeCommander": False,
                "zoneLocations": ["B"],
                "battlefieldCardState": "Untapped",
            }
            for n in names
        ],
        "requires": [],
        "produces": [{"feature": {"name": "Infinite damage"}}],
        "manaNeeded": "{2}",
        "easyPrerequisites": "Both on battlefield",
        "notablePrerequisites": "No disruption",
        "description": "Activate, untap, repeat.",
    }
    result.update(changes)
    return result


def spellbook_page(variants, *, next_=None, almost=None):
    return (
        json.dumps(
            {"results": {"included": variants, "almostIncluded": almost or []}, "next": next_}
        ).encode(),
        "2026-10-25",
    )


class BracketTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.cache = SourceCache(self.path / "sources.sqlite3")
        self.addCleanup(self.store.close)
        self.addCleanup(self.cache.close)
        self.store.import_cards(records())
        self.client = Mock()
        self.client.search_cards.return_value = [{"name": "Spell 69"}]

    def checker(self, answer=None, page=None):
        cache = Mock()
        cache.fetch.return_value = page or spellbook_page([])
        return BracketChecker(
            self.store,
            self.client,
            self.cache,
            FakeModel(answer or assessment()),
            spellbook=SpellbookClient(cache),
        )

    def test_live_game_changer_list_is_dated_and_cached(self):
        first = game_changers(self.client, self.cache)
        second = game_changers(self.client, self.cache)
        self.assertEqual(first, second)
        self.assertTrue(first["observed_at"])
        self.client.search_cards.assert_called_once_with("is:gamechanger")

    def test_empty_live_list_fails_closed(self):
        self.client.search_cards.return_value = []
        with self.assertRaises(SourceError):
            game_changers(self.client, self.cache)

    def test_signal_counts_are_quantity_aware_and_include_cheap_and_expensive_tutors(self):
        data = [
            (validation_card(card("Cheap", text="Search your library for a card.", cmc=2)), 2),
            (validation_card(card("Costly", text="Search your library for a card.", cmc=4)), 1),
            (
                validation_card(
                    card("Land", text="Search your library for a basic land card.", cmc=1)
                ),
                1,
            ),
            (validation_card(card("Sol Ring", text="{T}: Add {C}{C}.", cmc=1)), 1),
            (validation_card(card("Armageddon", text="Destroy all lands.", cmc=4)), 1),
            (validation_card(card("Turns", text="Take two extra turns after this one.", cmc=5)), 2),
        ]
        result = deterministic_signals(data, ["Cheap"])
        self.assertEqual(result["counts"]["tutors_cheap"], 2)
        self.assertEqual(result["counts"]["tutors_expensive"], 1)
        self.assertEqual(result["counts"]["land_tutors"], 1)
        self.assertEqual(result["counts"]["game_changers"], 2)
        self.assertEqual(result["counts"]["fast_mana"], 1)
        self.assertEqual(result["counts"]["mass_land_denial"], 1)
        self.assertEqual(result["counts"]["extra_turns"], 2)
        self.assertIn("Removed", POLICY["tutor_limits"])

    def test_replacing_destroyed_lands_is_not_hard_mass_land_denial(self):
        c = validation_card(
            card(
                "Replacement",
                text="Destroy all nonbasic lands. For each land "
                "destroyed, its controller may search for a basic land card.",
            )
        )
        result = deterministic_signals([(c, 1)], [])
        self.assertEqual(result["mass_land_denial"], [])
        self.assertEqual(len(result["land_denial_candidates"]), 1)

    def test_fits_runs_exactly_one_llm_assessment(self):
        checker = self.checker()
        result = checker.check({"Spell 0": 1}, JODAH["name"], 3)
        self.assertEqual(result["verdict"], "fits")
        self.assertEqual(len(checker.model.calls), 1)

    def test_fresh_list_overrides_snapshot_and_counts_commander(self):
        self.client.search_cards.return_value = [{"name": JODAH["name"]}]
        result = self.checker(assessment(estimated_bracket=2)).check(
            {"Spell 0": 1}, JODAH["name"], 2
        )
        self.assertEqual(result["verdict"], "mismatch")
        self.assertEqual(result["signals"]["counts"]["game_changers"], 1)

    def test_mass_land_denial_cannot_be_overridden_by_llm(self):
        self.store.add_verified_card(card("Armageddon", text="Destroy all lands."))
        result = self.checker().check({"Armageddon": 1}, JODAH["name"], 3)
        self.assertEqual(result["verdict"], "mismatch")

    def test_bracket_one_extra_turns_fail_but_bracket_two_single_turn_is_not_a_hard_cap(self):
        self.store.add_verified_card(card("Extra", text="Take an extra turn after this one."))
        checker = self.checker()
        self.assertEqual(checker.check({"Extra": 1}, JODAH["name"], 1)["verdict"], "mismatch")
        self.assertEqual(
            self.checker(assessment(likely_win_turn_min=9, likely_win_turn_max=12)).check(
                {"Extra": 1}, JODAH["name"], 2
            )["verdict"],
            "fits",
        )

    def test_many_tutors_do_not_create_an_invented_cap(self):
        self.store.add_verified_card(card("Tutor", text="Search your library for a card.", cmc=4))
        result = self.checker().check({"Tutor": 20}, JODAH["name"], 3)
        self.assertEqual(result["verdict"], "fits")

    def test_unknown_or_overlapping_speed_is_advisory_uncertainty_not_a_hard_violation(self):
        for lo, hi in ((0, 0), (5, 10)):
            result = self.checker(assessment(likely_win_turn_min=lo, likely_win_turn_max=hi)).check(
                {"Spell 0": 1}, JODAH["name"], 3
            )
            self.assertEqual(result["verdict"], "uncertain")
            self.assertEqual(result["errors"], [])

    def test_chain_and_early_combo_risk_trigger_revision(self):
        response = assessment(
            extra_turn_chaining="likely",
            combo_reviews=[{"id": "combo", "early_likelihood": "likely", "reason": "Fast setup"}],
        )
        result = self.checker(response, spellbook_page([variant()])).check(
            {"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3
        )
        self.assertEqual(result["verdict"], "mismatch")
        self.assertTrue(any("early" in s for s in result["errors"]))

    def test_known_extra_turn_loop_is_flagged_even_if_llm_says_unlikely(self):
        response = assessment(
            combo_reviews=[{"id": "combo", "early_likelihood": "unlikely", "reason": "Late setup"}]
        )
        loop = variant(produces=[{"feature": {"name": "Infinite turns"}}])
        result = self.checker(response, spellbook_page([loop])).check(
            {"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3
        )
        self.assertEqual(result["verdict"], "mismatch")
        self.assertEqual(result["signals"]["known_extra_turn_chain_ids"], ["combo"])

    def test_unknown_combo_speed_is_not_certified(self):
        response = assessment(
            combo_reviews=[
                {"id": "combo", "early_likelihood": "unknown", "reason": "No simulation"}
            ]
        )
        result = self.checker(response, spellbook_page([variant()])).check(
            {"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3
        )
        self.assertEqual(result["verdict"], "uncertain")

    def test_low_bracket_combo_presence_requires_intent_review(self):
        response = assessment(
            combo_reviews=[{"id": "combo", "early_likelihood": "unlikely", "reason": "Slow"}]
        )
        result = self.checker(response, spellbook_page([variant()])).check(
            {"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 2
        )
        self.assertEqual(result["verdict"], "uncertain")

    def test_hallucinated_references_missing_combo_review_and_invalid_turn_bounds_rejected(self):
        for changes in [
            {"card_references": ["Imaginary"]},
            {"likely_win_turn_min": 0, "likely_win_turn_max": 9},
            {"likely_win_turn_min": 9, "likely_win_turn_max": 7},
            {"combo_reviews": [{"id": "invented", "early_likelihood": "likely", "reason": "No"}]},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.checker(assessment(**changes)).check({"Spell 0": 1}, JODAH["name"], 3)

    def test_spellbook_excludes_almost_included_wrong_commander_and_nonworking_variants(self):
        wrong = variant("wrong")
        wrong["uses"][0]["mustBeCommander"] = True
        cache = Mock()
        cache.fetch.return_value = spellbook_page(
            [wrong, variant("notworking", status="NW"), variant("missing", names=("X", "Y"))],
            almost=[variant("almost")],
        )
        result = SpellbookClient(cache).find_combos({"Spell 0": 1, "Spell 1": 1}, JODAH["name"])
        self.assertEqual(result["combos"], [])

    def test_spellbook_checks_card_quantities_and_retains_prerequisites(self):
        two_copies = variant("copies")
        two_copies["uses"][0]["quantity"] = 2
        cache = Mock()
        cache.fetch.return_value = spellbook_page([variant(), two_copies])
        result = SpellbookClient(cache).find_combos({"Spell 0": 1, "Spell 1": 1}, JODAH["name"])
        self.assertEqual(len(result["combos"]), 1)
        self.assertEqual(result["combos"][0]["prerequisites"][0], "Both on battlefield")

    def test_spellbook_template_or_truncation_does_not_claim_completeness(self):
        cache = Mock()
        cache.fetch.return_value = spellbook_page([], next_="https://malicious.invalid")
        result = SpellbookClient(cache, max_pages=1).find_combos({}, JODAH["name"])
        self.assertFalse(result["complete"])
        cache.fetch.return_value = spellbook_page([variant(requires=[{"template": {}}])])
        self.assertFalse(SpellbookClient(cache).find_combos({}, JODAH["name"])["complete"])

    def test_spellbook_reconstructs_pagination_without_following_untrusted_url(self):
        cache = Mock()
        cache.fetch.side_effect = [
            spellbook_page([], next_="https://bad.invalid"),
            spellbook_page([variant()]),
        ]
        result = SpellbookClient(cache).find_combos({"Spell 0": 1, "Spell 1": 1}, JODAH["name"])
        self.assertTrue(result["complete"])
        self.assertIn("offset=100", cache.fetch.call_args.args[0])
        self.assertNotIn("bad.invalid", cache.fetch.call_args.args[0])

    def test_bad_spellbook_response_is_an_error_not_zero_combos(self):
        cache = Mock()
        cache.fetch.return_value = (b"{}", "today")
        with self.assertRaises(SourceError):
            SpellbookClient(cache).find_combos({}, JODAH["name"])

    def test_live_validator_list_can_remove_stale_flags(self):
        from mtg_agentic.validator import validate_deck

        stale = copy.deepcopy(JODAH)
        stale["game_changer"] = True
        self.store.add_verified_card(stale)
        result = validate_deck(
            {"Plains": 99}, JODAH["name"], self.store, bracket=2, game_changers=[]
        )
        self.assertTrue(result["valid"])
