import copy
import tempfile
import unittest
from pathlib import Path

from mtg_agentic.discovery import (
    CandidatePool,
    CardQuery,
    build_candidate_pool,
    constrain_query,
    eligible,
    generate_queries,
)
from mtg_agentic.mana import is_land
from mtg_agentic.oracle import OracleStore
from mtg_agentic.schema import SchemaError
from tests.fixtures import card
from tests.land_fixtures import fixing_lands
from tests.pipeline_fixtures import FakeCards, FakeModel, make_spec, records


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = OracleStore(Path(self.directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = records()
        self.store.import_cards(self.records)

    def test_jodah_plan_always_has_theme_anchors_and_role_coverage(self):
        model = FakeModel({"queries": [{"role": "theme", "query": "o:counter"}]})
        queries = generate_queries(make_spec(), model)
        combined = "\n".join(query.query for query in queries)
        for anchor in ("t:myojin", 'o:"divinity counter"', "otag:proliferate"):
            self.assertIn(anchor, combined)
        self.assertTrue({"ramp", "draw", "removal", "wipes"} <= {q.role for q in queries})
        self.assertNotIn("lands", {q.role for q in queries})  # The mana base reads every land.
        self.assertTrue(all(q.query.endswith("f:commander id<=WUBRG") for q in queries))
        self.assertEqual(len(model.calls), 1)

    def test_queries_group_or_and_reject_unbalanced_input(self):
        query = constrain_query("t:artifact OR t:creature", "WU")
        self.assertEqual(query, "(t:artifact OR t:creature) f:commander id<=WU")
        for bad in ("t:creature) OR (id:rg", 'o:"unclosed', "t:land\nOR f:vintage"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                constrain_query(bad, "WU")
        self.assertTrue(constrain_query("t:artifact", "C").endswith("id<=C"))

    def test_malformed_model_queries_are_rejected(self):
        for output in ({}, {"queries": []}, {"queries": [{"role": "invented", "query": "x"}]}):
            with self.subTest(output=output), self.assertRaises(SchemaError):
                generate_queries(make_spec(), FakeModel(output))

    def test_pool_deduplicates_and_merges_roles_with_local_legality_checks(self):
        leader = card("WU Leader", colors="WU", types="Legendary Creature — Human")
        banned = card("Banned", legal="banned")
        outside = card("Outside", colors="R")
        missing = card("Not In Snapshot")
        self.store.import_cards([*self.records, leader, banned, outside])
        spec = make_spec(commander=leader["name"], color_identity="WU")
        a = self.records[7]  # Spell 0, white.
        b = self.records[8]  # Spell 1, blue.
        client = FakeCards([], [[a, banned, outside, missing, leader], [a, b]])
        pool = build_candidate_pool(
            spec, [CardQuery("ramp", "t:artifact"), CardQuery("draw", "o:draw")], client, self.store
        )
        by_name = {c.card["name"]: c for c in pool.cards}
        self.assertEqual(by_name[a["name"]].roles, {"ramp", "draw"})
        self.assertNotIn("Banned", by_name)
        self.assertNotIn("Outside", by_name)
        self.assertNotIn("Not In Snapshot", by_name)
        self.assertNotIn(leader["name"], by_name)
        self.assertEqual(pool.diagnostics[0]["missing_from_snapshot"], 1)
        self.assertTrue(all("f:commander id<=WU" in query for query, _ in client.search_calls))

    def test_cap_preserves_required_cards_and_round_robin_role_diversity(self):
        client = FakeCards([], [self.records[7:17], [self.records[-1]]])
        spec = make_spec(must_include=["Spell 68"])
        pool = build_candidate_pool(
            spec, [CardQuery("ramp", "a"), CardQuery("theme", "b")], client, self.store, limit=9
        )
        names = {candidate.card["name"] for candidate in pool.cards}
        self.assertEqual(len(pool.cards), 9)
        self.assertTrue({"Spell 68", "Spell 69"} <= names)
        self.assertFalse({"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"} & names)
        self.assertEqual(len(client.search_calls), 2)

    def test_after_each_querys_top_cards_the_pool_fills_by_fit(self):
        spell = {n: self.records[7 + n] for n in range(70)}
        narrow_a = [spell[n] for n in (0, 1, 2, 3, 4, 55)]
        narrow_b = [spell[n] for n in (60, 61, 62, 63, 64, 55)]
        broad = [spell[n] for n in range(10, 50)]
        client = FakeCards([], [narrow_a, narrow_b, broad])
        queries = [CardQuery("theme", "a"), CardQuery("mechanics", "b"), CardQuery("creatures", "c")]
        pool = build_candidate_pool(make_spec(), queries, client, self.store, limit=7)
        names = [c.card["name"] for c in pool.cards]
        # Each query's most-played cards taking turns (the creature role query goes deeper,
        # toward three times its target), then the best fit: Spell 55 sits deep in both narrow
        # searches, so it beats the rest of every single search.
        self.assertEqual(names, ["Spell 0", "Spell 60", "Spell 10", "Spell 11", "Spell 12",
                                 "Spell 13", "Spell 55"])
        self.assertFalse(any(d["result_limit_reached"] for d in pool.diagnostics))

    def test_pool_reload_does_not_trust_saved_oracle_or_legality_fields(self):
        pool = build_candidate_pool(
            make_spec(), [CardQuery("theme", "a")], FakeCards(self.records), self.store
        )
        data = pool.to_dict()
        forged = copy.deepcopy(data)
        forged["cards"][0]["oracle_text"] = "Invented ability."
        reloaded = CandidatePool.from_dict(forged, self.store)
        self.assertNotEqual(reloaded.cards[0].card["oracle_text"], "Invented ability.")
        forged["cards"][0]["name"] = "Invented Card"
        with self.assertRaises(ValueError):
            CandidatePool.from_dict(forged, self.store)

    def test_pool_rejects_missing_required_cards_and_invalid_limits(self):
        spec = make_spec(must_include=["Spell 69"])
        data = {"version": 1, "spec": spec.to_dict(), "cards": []}
        with self.assertRaises(ValueError):
            CandidatePool.from_dict(data, self.store)
        for limit in (0, 1001, True):
            with self.assertRaises(ValueError):
                build_candidate_pool(make_spec(), [], FakeCards([]), self.store, limit=limit)

    def test_every_pool_slot_goes_to_nonlands_unless_a_land_is_required(self):
        self.store.import_cards([*self.records, *fixing_lands()])
        spec = make_spec(must_include=["Breeding Pool"])
        for mode in ("popular", "relevant", "mixed"):
            with self.subTest(mode=mode):
                # The search finds every card, lands and basics included.
                pool = build_candidate_pool(
                    spec, [CardQuery("theme", "a")], FakeCards([*self.records, *fixing_lands()]),
                    self.store, limit=71, mode=mode,
                )
                lands = [c.card["name"] for c in pool.cards if is_land(c.card)]
                self.assertEqual(lands, ["Breeding Pool"])
                self.assertEqual(len(pool.cards), 71)
                saved = CandidatePool.from_dict(pool.to_dict(), self.store)
                self.assertEqual(saved.to_dict(), pool.to_dict())

    def test_lower_brackets_filter_game_changers(self):
        changer = card("Changer", changer=True)
        self.store.import_cards([*self.records, changer])
        pool = build_candidate_pool(
            make_spec(bracket=2), [CardQuery("theme", "a")], FakeCards([changer]), self.store
        )
        self.assertNotIn("Changer", {candidate.card["name"] for candidate in pool.cards})

    def test_house_rule_modes_filter_and_reload_the_correct_cards(self):
        data = [
            card("Armageddon", text="Destroy all lands."),
            card("Multi", text="Destroy two target lands."),
            card("Single", text="Destroy target land."),
            card("Scorched Ruins", types="Land", text="Sacrifice two untapped lands. {T}: Add {C}{C}{C}{C}."),
        ]
        for raw in data:
            self.store.add_verified_card(raw)
        for mode in ("unrestricted", "limited-land-destruction"):
            with self.subTest(mode=mode):
                pool = build_candidate_pool(
                    make_spec(bracket3_mode=mode), [CardQuery("theme", "a")],
                    FakeCards(data), self.store,
                )
                names = {c.card["name"] for c in pool.cards}
                self.assertIn("Single", names)
                # A land never takes a pool slot, but the mana base may still use this one.
                ruins = self.store.get_card("Scorched Ruins")
                self.assertTrue(eligible(ruins, make_spec(bracket3_mode=mode)))
                self.assertEqual("Armageddon" in names, mode == "unrestricted")
                self.assertEqual("Multi" in names, mode == "unrestricted")
                self.assertEqual(CandidatePool.from_dict(pool.to_dict(), self.store).spec.bracket3_mode, mode)

    def test_limited_mode_cannot_require_a_forbidden_card(self):
        self.store.add_verified_card(card("Multi", text="Destroy two target lands."))
        with self.assertRaises(SchemaError):
            build_candidate_pool(
                make_spec(bracket3_mode="limited-land-destruction", must_include=["Multi"]),
                [], FakeCards([]), self.store,
            )


    def test_brackets_one_to_three_never_pool_mass_land_denial(self):
        rage = card("Land Reset Myojin", types="Legendary Creature — Spirit", text="Destroy all lands.")
        self.store.import_cards([*self.records, rage])
        for bracket, kept in ((3, False), (4, True)):
            with self.subTest(bracket=bracket):
                pool = build_candidate_pool(
                    make_spec(bracket=bracket), [CardQuery("theme", "a")], FakeCards([rage]), self.store
                )
                self.assertEqual("Land Reset Myojin" in {c.card["name"] for c in pool.cards}, kept)

    def test_empty_subtype_and_mechanic_tags_get_constrained_logged_fallbacks(self):
        myojin = card("Myojin Test", types="Legendary Creature — Spirit")
        proliferate = card("Proliferate Test", text="Proliferate.")
        self.store.import_cards([*self.records, myojin, proliferate])
        client = FakeCards([], [[], [myojin], [], [proliferate]])
        pool = build_candidate_pool(
            make_spec(),
            [CardQuery("theme", "t:myojin"), CardQuery("mechanics", "otag:proliferate")],
            client,
            self.store,
        )
        self.assertEqual(len(client.search_calls), 4)
        self.assertEqual([row["initial_returned"] for row in pool.diagnostics], [0, 0])
        self.assertEqual([row["eligible"] for row in pool.diagnostics], [1, 1])
        for row in pool.diagnostics:
            self.assertTrue(row["fallback_query"].endswith("f:commander id<=WUBRG"))
        saved = CandidatePool.from_dict(pool.to_dict(), self.store)
        self.assertEqual(saved.diagnostics, pool.diagnostics)
        self.assertEqual(saved.queries, pool.queries)

    def test_popular_and_relevant_pools_each_rank_by_one_signal(self):
        # Spell 9 is the most played and Spell 0 the least; a theme card no search found.
        spells = [dict(self.records[7 + n], edhrec_rank=100 - n) for n in range(10)]
        unsearched = dict(self.records[47], edhrec_rank=99999)  # Spell 40
        land = card("Popular Land", types="Land", text="{T}: Add {C}.", edhrec_rank=1)
        self.store.import_cards([*self.records[:7], *spells, *self.records[17:47], unsearched,
                                 *self.records[48:], land])
        relevance = {"Spell 40": 50.0, "Spell 2": 25.0}
        hits = {"terms": [], "hits": [{"name": "Spell 40", "score": 50, "matched": []}]}

        def pool(mode):
            client = FakeCards([], [[*spells, land]])
            return build_candidate_pool(
                make_spec(), [CardQuery("theme", "a")], client, self.store, limit=5,
                local_search=hits, mode=mode, relevance=relevance,
            )

        popular = pool("popular")
        self.assertEqual([c.card["name"] for c in popular.cards],
                         ["Spell 9", "Spell 8", "Spell 7", "Spell 6", "Spell 5"])
        # The theme search's hit joins first, though no query found it; floors take the
        # query's three most relevant (popularity only breaks the tie between cards the search
        # found equally), then the rest by relevance. A land never comes in.
        relevant = pool("relevant")
        names = [c.card["name"] for c in relevant.cards]
        self.assertEqual(names, ["Spell 40", "Spell 2", "Spell 9", "Spell 8", "Spell 7"])
        self.assertEqual(relevant.cards[0].roles, {"theme"})
        saved = CandidatePool.from_dict(relevant.to_dict(), self.store)
        self.assertEqual((saved.mode, popular.mode), ("relevant", "popular"))
        # Mixed still lets the theme search's best hits join outright.
        self.assertIn("Spell 40", [c.card["name"] for c in pool("mixed").cards])
        with self.assertRaises(ValueError):
            pool("random")
