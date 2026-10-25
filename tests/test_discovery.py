import copy
import tempfile
import unittest
from pathlib import Path

from mtg_agentic.discovery import (
    CandidatePool,
    CardQuery,
    build_candidate_pool,
    constrain_query,
    generate_queries,
)
from mtg_agentic.oracle import OracleStore
from mtg_agentic.schema import SchemaError
from tests.fixtures import card
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
        self.assertTrue({"ramp", "draw", "removal", "wipes", "lands"} <= {q.role for q in queries})
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
        self.assertTrue({"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"} <= names)
        self.assertEqual(len(client.search_calls), 2)

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
        for limit in (0, 401, True):
            with self.assertRaises(ValueError):
                build_candidate_pool(make_spec(), [], FakeCards([]), self.store, limit=limit)

    def test_lower_brackets_filter_game_changers(self):
        changer = card("Changer", changer=True)
        self.store.import_cards([*self.records, changer])
        pool = build_candidate_pool(
            make_spec(bracket=2), [CardQuery("theme", "a")], FakeCards([changer]), self.store
        )
        self.assertNotIn("Changer", {candidate.card["name"] for candidate in pool.cards})

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
