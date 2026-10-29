"""Scryfall search syntax over the local snapshot and Oracle tags."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.bulk_search import BulkSearch, QueryError
from mtg_agentic.discovery import CardQuery, build_candidate_pool
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.pipeline_fixtures import make_spec

CARDS = [
    card("Goblin Guide", colors="R", mana_cost="{R}", cmc=1, types="Creature — Goblin Scout",
         text="Haste\nWhenever Goblin Guide attacks, defending player reveals a card.",
         power="2", toughness="2", keywords=["Haste"], edhrec_rank=300),
    card("Mogg Fanatic", colors="R", mana_cost="{R}", cmc=1, types="Creature — Goblin",
         text="Sacrifice Mogg Fanatic: It deals 1 damage to any target.", power="1",
         toughness="1", edhrec_rank=900),
    card("Wall of Omens", colors="W", mana_cost="{1}{W}", cmc=2, types="Creature — Wall",
         text="Defender (This creature can't attack.)\nWhen Wall of Omens enters, draw a card.",
         power="0", toughness="4", keywords=["Defender"], edhrec_rank=100),
    card("Azorius Signet", mana_cost="{2}", cmc=2, types="Artifact",
         text="{1}, {T}: Add {W}{U}.", produced_mana=["W", "U"], edhrec_rank=50),
    card("Detention Sphere", colors="WU", mana_cost="{1}{W}{U}", cmc=3,
         types="Enchantment", text="When Detention Sphere enters, exile another target nonland "
         "permanent.", edhrec_rank=200),
    card("Wrath of God", colors="W", mana_cost="{2}{W}{W}", cmc=4, types="Sorcery",
         text="Destroy all creatures. They can't be regenerated."),
    card("Krenko, Mob Boss", colors="R", mana_cost="{2}{R}{R}", cmc=4,
         types="Legendary Creature — Goblin Warrior", power="3", toughness="3",
         text="{T}: Create X 1/1 red Goblin creature tokens, where X is the number of Goblins "
         "you control.", edhrec_rank=400),
    card("Tarmogoyf", colors="G", mana_cost="{1}{G}", cmc=2, types="Creature — Lhurgoyf",
         power="*", toughness="1+*", text="Tarmogoyf's power is equal to the number of card "
         "types among cards in all graveyards.", legal="banned"),
    card("Grizzly Bears", colors="G", mana_cost="{1}{G}", cmc=2, types="Creature — Bear",
         power="2", toughness="2"),
]
for _card in CARDS:  # Each fixture's card colors equal its color identity.
    _card["colors"] = list(_card["color_identity"])
# Tagger data: "removal" has no direct taggings, only a child tag, as in the real file.
TAGS = [
    {"object": "tag", "id": "ramp", "slug": "ramp", "label": "ramp", "type": "oracle",
     "aliases": ["acceleration"], "child_ids": ["rock"],
     "taggings": [{"oracle_id": "Krenko, Mob Boss", "weight": "weak"}]},
    {"object": "tag", "id": "rock", "slug": "mana-rock", "label": "mana rock", "type": "oracle",
     "parent_ids": ["ramp"], "taggings": [{"oracle_id": "Azorius Signet", "weight": "strong"}]},
    {"object": "tag", "id": "removal", "slug": "removal", "label": "removal", "type": "oracle",
     "child_ids": ["exile"], "taggings": []},
    {"object": "tag", "id": "exile", "slug": "removal-exile", "label": "removal exile",
     "type": "oracle", "taggings": [{"oracle_id": "Detention Sphere", "weight": "median"}]},
    {"object": "tag", "id": "sweeper", "slug": "sweeper", "label": "sweeper", "type": "oracle",
     "aliases": ["wipe", "boardwipe"], "taggings": [{"oracle_id": "Wrath of God"}]},
]


class BulkSearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(CARDS)
        self.store.import_tags(TAGS)
        self.search = BulkSearch(self.store)

    def names(self, query, **options):
        return [c["name"] for c in self.search.search_cards(query, **options)]

    def test_types_text_names_and_boolean_logic(self):
        self.assertEqual(self.names("t:goblin"), ["Goblin Guide", "Krenko, Mob Boss", "Mogg Fanatic"])
        self.assertEqual(self.names('o:"draw a card"'), ["Wall of Omens"])
        self.assertEqual(self.names("goblin"), ["Goblin Guide"])  # A bare word is the name.
        self.assertEqual(self.names('!"krenko, mob boss"'), ["Krenko, Mob Boss"])
        self.assertEqual(self.names("t:creature -t:goblin c:g"), ["Grizzly Bears", "Tarmogoyf"])
        self.assertEqual(self.names("(t:wall OR t:bear) and mv=2"), ["Grizzly Bears", "Wall of Omens"])
        self.assertEqual(self.names("o:/^destroy all/"), ["Wrath of God"])
        self.assertEqual(self.names('o:"sacrifice ~"'), ["Mogg Fanatic"])  # ~ is the card's name.
        self.assertEqual(self.names("kw:haste"), ["Goblin Guide"])

    def test_reminder_text_is_only_in_full_oracle_searches(self):
        self.assertEqual(self.names('o:"can\'t attack"'), [])
        self.assertEqual(self.names('fo:"can\'t attack"'), ["Wall of Omens"])

    def test_colors_identity_stats_mana_and_legality(self):
        self.assertEqual(self.names("c:wu"), ["Detention Sphere"])
        self.assertEqual(self.names("id<=w t:creature"), ["Wall of Omens"])
        self.assertEqual(self.names("id:azorius -t:creature"),
                         ["Azorius Signet", "Detention Sphere", "Wrath of God"])
        self.assertEqual(self.names("c:c"), ["Azorius Signet"])
        self.assertEqual(self.names("c:m"), ["Detention Sphere"])
        self.assertEqual(self.names("pow>=3"), ["Krenko, Mob Boss"])
        self.assertEqual(self.names("tou>pow"), ["Tarmogoyf", "Wall of Omens"])
        self.assertEqual(self.names("m:{R}{R}"), ["Krenko, Mob Boss"])
        self.assertEqual(self.names("produces:u"), ["Azorius Signet"])
        self.assertEqual(self.names("is:commander"), ["Krenko, Mob Boss"])
        self.assertNotIn("Tarmogoyf", self.names("t:creature f:commander"))

    def test_oracle_tags_include_child_tags_and_aliases(self):
        self.assertEqual(self.names("otag:ramp"), ["Azorius Signet", "Krenko, Mob Boss"])
        self.assertEqual(self.names("otag:removal"), ["Detention Sphere"])
        self.assertEqual(self.names("otag:board-wipe"), ["Wrath of God"])  # Alias "boardwipe".
        self.assertEqual(self.names("otag:acceleration -t:creature"), ["Azorius Signet"])
        self.assertEqual(self.names("otag:proliferate"), [])  # Unknown, as on Scryfall.

    def test_edhrec_order_puts_unranked_cards_last_and_limits_results(self):
        self.assertEqual(self.names("mv<=2", order="edhrec"), [
            "Azorius Signet", "Wall of Omens", "Goblin Guide", "Mogg Fanatic",
            "Grizzly Bears", "Tarmogoyf",
        ])
        self.assertEqual(self.names("mv<=2", order="edhrec", max_results=2),
                         ["Azorius Signet", "Wall of Omens"])

    def test_what_the_snapshot_cannot_answer_is_a_repairable_400(self):
        for query in ("r:rare", "set:neo t:goblin", "f:modern", "is:foil", "(t:goblin",
                      't:goblin o:"unclosed', "t:goblin)", "o:/[/", "mv>=many", "t:goblin or"):
            with self.subTest(query=query), self.assertRaises(QueryError) as caught:
                self.search.search_cards(query)
            self.assertEqual(caught.exception.status, 400)

    def test_missing_data_is_fetched_once_through_the_client(self):
        with tempfile.TemporaryDirectory() as directory, \
                OracleStore(Path(directory) / "oracle.sqlite3") as store:
            store.import_cards(CARDS)
            store._db.execute("DELETE FROM metadata WHERE key = 'index_version'")
            client = Mock()
            client.download_oracle_tags.side_effect = lambda s: s.import_tags(TAGS)
            search = BulkSearch(store, client)
            self.assertEqual([c["name"] for c in search.search_cards("otag:ramp t:artifact")],
                             ["Azorius Signet"])
            search.search_cards("otag:sweeper")
            client.download_oracle_cards.assert_called_once_with(store)
            client.download_oracle_tags.assert_called_once_with(store)
            with OracleStore(Path(directory) / "empty.sqlite3") as empty,                     self.assertRaises(QueryError):
                BulkSearch(empty).search_cards("otag:ramp")  # No tags and no client.

    def test_discovery_runs_its_queries_locally(self):
        spec = make_spec(commander="Krenko, Mob Boss", color_identity="R", themes=["goblin tribal"],
                         mechanics=[], win_conditions=[])
        pool = build_candidate_pool(
            spec, [CardQuery("theme", "t:goblin"), CardQuery("ramp", "otag:ramp")],
            self.search, self.store, limit=20,
        )
        names = {c.card["name"] for c in pool.cards}
        self.assertTrue({"Goblin Guide", "Mogg Fanatic", "Azorius Signet"} <= names)
        self.assertNotIn("Krenko, Mob Boss", names)  # The commander is never a candidate.
        self.assertNotIn("Wall of Omens", names)  # Outside the commander's colors.
        ramp = next(d for d in pool.diagnostics if d["role"] == "ramp")
        self.assertEqual(ramp["attempts"][0]["query"], "(otag:ramp) f:commander id<=R")


if __name__ == "__main__":
    unittest.main()
