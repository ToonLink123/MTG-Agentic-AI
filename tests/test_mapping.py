"""Cards and rules mapping: roles from Scryfall tags, compact and two-tier card rows, search
terms, three-card combos, the card graph, engine-map link checks, the rules ledger and the
judge with native tools."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.card_graph import CardGraph, card_search, universe
from mtg_agentic.cards import compact_row, prompt_card, validation_card
from mtg_agentic.discovery import CandidatePool, candidate_rows
from mtg_agentic.engine_map import build_engine_map
from mtg_agentic.oracle import OracleStore
from mtg_agentic.review import Judge, check_links, link_jobs, link_question
from mtg_agentic.roles import oracle_roles, protection_features
from mtg_agentic.rules import HashEmbedder, RulesStore
from mtg_agentic.schema import SPEC_SCHEMA
from mtg_agentic.spec import ground_spec
from mtg_agentic.toolbox import CardTools, Toolbox
from tests.fixtures import card
from tests.pipeline_fixtures import (
    JODAH,
    FakeModel,
    FakeToolModel,
    make_pool,
    make_spec,
    map_reply,
    playable_records,
    records,
)
from tests.test_rules import QUOTE, RULES, cited


def tag(slug, *oracle_ids, children=()):
    return {"object": "tag", "type": "oracle", "id": slug, "slug": slug, "label": slug,
            "child_ids": list(children), "taggings": [{"oracle_id": i} for i in oracle_ids]}


class Store(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)


class TagRoleTests(Store):
    CARDS = [
        card("Burn", colors="R", types="Instant", text="Burn deals 3 damage to any target."),
        card("Witness", colors="G", types="Creature — Human",
             text="When Witness enters, return target creature card from your graveyard to your hand."),
        card("Quake", colors="R", types="Sorcery", text="Quake deals 2 damage to each creature."),
        card("Library", colors="G", types="Enchantment", text="Look at the top three cards."),
        card("Team Shield", colors="W", types="Instant",
             text="Creatures you control gain indestructible until end of turn."),
    ]

    def setUp(self):
        super().setUp()
        self.store.import_cards(self.CARDS)
        self.store.import_tags([
            tag("removal", "Burn", children=["board-wipe"]), tag("board-wipe", "Quake"),
            tag("card-advantage", "Witness", "Library", children=["recursion"]),
            tag("recursion", "Witness"),
        ])

    def roles(self, name):
        return oracle_roles(self.store.get_card(name))

    def test_tagged_cards_take_their_roles_from_the_tags(self):
        self.assertEqual(self.store.get_card("Burn")["functions"], ["removal"])
        self.assertEqual(self.roles("Burn"), {"removal"})  # Damage to "any target" is removal.
        self.assertEqual(self.roles("Quake"), {"removal", "wipes"})  # A tag below removal.
        self.assertEqual(self.roles("Library"), {"draw"})  # Card advantage that is not recursion.
        # Recursion is neither draw nor removal, though the text says "return target".
        self.assertEqual(self.roles("Witness"), {"creatures"})

    def test_untagged_cards_fall_back_to_patterns_that_read_plurals(self):
        shield = self.store.get_card("Team Shield")
        self.assertNotIn("functions", shield)
        self.assertEqual(protection_features(shield)["categories"], ["indestructible"])
        self.assertTrue(protection_features(shield)["instant_independent"])
        self.assertIn("protection", oracle_roles(shield))
        recursion = validation_card(card("Regrowth Elf", types="Creature — Elf",
                                         text="Return target creature card from your graveyard."))
        self.assertNotIn("removal", oracle_roles(recursion))
        sweeper = validation_card(card("Act", types="Sorcery", text="Act deals 13 damage to each creature."))
        self.assertEqual(oracle_roles(sweeper), {"removal", "wipes"})

    def test_a_new_tag_import_rebuilds_the_functions(self):
        self.store.import_tags([tag("removal", "Library")])
        self.assertNotIn("functions", self.store.get_card("Burn"))
        self.assertEqual(self.store.get_card("Library")["functions"], ["removal"])


class CardRowTests(Store):
    def test_prompt_rows_keep_what_a_drafter_reads_and_drop_what_only_code_uses(self):
        raw = card("Bear Engine", colors="G", types="Creature — Bear", mana_cost="{1}{G}", cmc=2,
                   power="2", toughness="2", edhrec_rank=150,
                   text="Trample (This creature can deal excess combat damage.)\nDraw a card.")
        self.store.import_cards([raw])
        self.store.import_tags([tag("draw", "Bear Engine")])
        row = prompt_card(self.store.get_card("Bear Engine"), {"theme"})
        self.assertEqual(row, {
            "name": "Bear Engine", "mana_cost": "{1}{G}", "mv": 2, "type_line": "Creature — Bear",
            "pt": "2/2", "oracle_text": "Trample\nDraw a card.", "tags": ["draw"],
            "played": "staple", "roles": ["theme"],
        })
        self.assertEqual(compact_row(self.store.get_card("Bear Engine"), {"theme"}),
                         "Bear Engine | {1}{G} | Creature — Bear | draw, theme")

    def test_a_large_pool_sends_its_shortlist_in_full_and_one_line_for_the_rest(self):
        self.store.import_cards(playable_records())
        pool = make_pool(self.store)
        rows = candidate_rows(pool, full=10, first=["Spell 50", "Not A Card"])
        self.assertEqual([r["name"] for r in rows["candidates"]][:2], ["Spell 50", "Spell 0"])
        self.assertEqual(len(rows["candidates"]), 10)
        self.assertEqual(len(rows["more_candidates"]), 60)
        self.assertTrue(rows["more_candidates"][0].startswith("Spell 9 | "))
        self.assertEqual(set(candidate_rows(pool, full=100)), {"candidates"})


class SearchTermTests(Store):
    def test_intake_writes_search_terms_and_code_keeps_only_printed_ones(self):
        self.assertIn("search_terms", SPEC_SCHEMA["required"])
        self.store.import_cards([*records(), card("Roller", colors="R", text="Whenever you roll a die, scry 1.")])
        spec = make_spec(search_terms=["whenever you roll", "certain chances", "Roller"])
        # "Roller" is only a name: names are never searched.
        self.assertEqual(ground_spec(spec, self.store).search_terms, ["whenever you roll"])


class ComboAndMapTests(Store):
    def setUp(self):
        super().setUp()
        self.store.import_cards(playable_records())
        self.pool = make_pool(self.store)

    def test_three_card_combos_reach_the_map_and_the_saved_pool(self):
        class Spellbook:
            def find_combos(self, cards, commander, max_cards=2):
                self.max_cards = max_cards
                return {"combos": [
                    {"id": "1-2-3", "cards": ["Spell 1", "Spell 2", "Spell 3"], "results": ["Win"]},
                    {"id": "4-5", "cards": ["Spell 4", "Spell 5"], "results": ["Infinite mana"]},
                ]}

        spellbook, model = Spellbook(), FakeModel(map_reply())
        pool = build_engine_map(self.pool, model, self.store, spellbook=spellbook)
        self.assertEqual(spellbook.max_cards, 3)
        self.assertEqual([c["id"] for c in pool.combos], ["4-5", "1-2-3"])  # Two-card first.
        payload = json.loads(model.calls[0][0][1]["content"])
        self.assertEqual(len(payload["combos"]), 2)
        again = CandidatePool.from_dict(pool.to_dict(), self.store)
        self.assertEqual(again.combos, pool.combos)
        smaller = pool.to_dict()
        smaller["cards"] = [c for c in smaller["cards"] if c["name"] != "Spell 5"]
        self.assertEqual([c["id"] for c in CandidatePool.from_dict(smaller, self.store).combos], ["1-2-3"])

    def test_a_map_link_the_rules_contradict_leaves_the_map_before_drafting(self):
        reply = map_reply(
            {"stage": "makes", "what": "x", "cards": [
                {"name": "Spell 1", "tier": "primary", "role": "x"},
                {"name": "Spell 2", "tier": "adjacent", "role": "x"},
            ]},
            links=[
                {"from": "Spell 1", "to": "Spell 2", "how": "untaps it", "source": "oracle"},
                {"from": "Spell 2", "to": "Spell 3", "how": "copies it", "source": "oracle"},
                {"from": "Spell 3", "to": "Spell 4", "how": "loops", "source": "oracle"},
            ],
        )
        judge = Mock()
        judge.answer.side_effect = lambda jobs: [
            {"status": {"Spell 1": "contradicted", "Spell 2": "supported"}.get(cards[0], "unsupported")}
            for cards, _ in jobs
        ]
        pool = build_engine_map(self.pool, FakeModel(reply), self.store, judge=judge)
        links = {(link["from"], link["to"]): link["verified"] for link in pool.engine_map["links"]}
        self.assertEqual(links, {("Spell 2", "Spell 3"): "supported", ("Spell 3", "Spell 4"): "unverified"})
        self.assertTrue(pool.engine_map["gaps"][0].startswith("The rules contradict Spell 1 -> Spell 2"))
        asked = [question for _, question in judge.answer.call_args.args[0]]
        self.assertTrue(asked[0].startswith("Spell 1 and Spell 2: untaps it."))  # Primary first.
        again = CandidatePool.from_dict(pool.to_dict(), self.store)
        self.assertEqual(again.engine_map["links"], pool.engine_map["links"])

    def test_link_questions_skip_combo_links_and_cards_not_in_the_deck(self):
        engine = {"stages": [], "links": [
            {"from": "A", "to": "B", "how": "x", "source": "oracle"},
            {"from": "A", "to": "C", "how": "x", "source": "combo"},
            {"from": "A", "to": "D", "how": "x", "source": "oracle"},
        ]}
        jobs = link_jobs(engine, {"a", "b", "c"})
        self.assertEqual([cards for cards, _, _ in jobs], [["A", "B"]])
        self.assertEqual(jobs[0][1], link_question(engine["links"][0]))


class GraphTests(Store):
    def setUp(self):
        super().setUp()
        cards = [
            card("Dice Lord", colors="R", types="Legendary Creature — Human", text="Roll a die."),
            *(card(f"Roller {n}", colors="R", types="Creature — Goblin", text="Roll a die.")
              for n in range(3)),
            card("Treasure Maker", colors="R", text="Create a Treasure token.",
                 keywords=["Treasure"]),
            card("Treasure Twin", colors="R", text="Create two Treasure tokens.",
                 keywords=["Treasure"]),
            *(card(f"Filler {n}", colors="R") for n in range(60)),
            card("Blue Roller", colors="U", text="Roll a die."),
        ]
        self.store.import_cards(cards)
        self.store.import_tags([tag("die-roll", "Dice Lord", "Roller 0", "Roller 1", "Roller 2", "Blue Roller")])
        self.spec = make_spec(commander="Dice Lord", color_identity="R", themes=["dice"],
                              mechanics=[], win_conditions=[], search_terms=[])

    def test_a_walk_from_the_commander_finds_what_shares_its_features(self):
        cards = universe(self.spec, self.store)
        names = {c["name"] for c in cards}
        self.assertIn("Dice Lord", names)  # The seed itself.
        self.assertNotIn("Blue Roller", names)  # Outside the colors.
        graph = CardGraph(self.store, cards)
        walked = graph.walk({"Dice Lord": 1.0})
        self.assertEqual(sorted(walked), ["Roller 0", "Roller 1", "Roller 2"])
        self.assertEqual(graph.shared("Treasure Maker", "Treasure Twin"), ["keyword treasure"])

    def test_card_search_fuses_words_and_the_walk(self):
        found = card_search(self.spec, self.store)
        hits = {hit["name"]: hit["matched"] for hit in found["hits"]}
        self.assertEqual(hits["Roller 0"][-1], "synergy")
        self.assertNotIn("Dice Lord", hits)  # The commander seeds the walk; it is no hit.

    def test_a_card_index_adds_meaning_and_ignores_another_models_vectors(self):
        from mtg_agentic.card_graph import index_cards

        self.assertEqual(index_cards(self.store, HashEmbedder()), 67)  # Every legal nonland.
        spec = make_spec(commander="Dice Lord", color_identity="R", themes=["roll a die"],
                         mechanics=[], win_conditions=[], search_terms=[])
        hits = {h["name"]: h["matched"] for h in card_search(spec, self.store, embedder=HashEmbedder())["hits"]}
        self.assertIn("meaning", hits["Roller 1"])
        other = Mock(identity="another-model")
        hits = {h["name"]: h["matched"] for h in card_search(spec, self.store, embedder=other)["hits"]}
        self.assertNotIn("meaning", hits["Roller 1"])
        other.embed.assert_not_called()

    def test_similar_cards_are_offered_as_addable(self):
        tools = CardTools(Toolbox(self.store), self.spec)
        found = tools.similar_cards("Treasure Maker")
        self.assertEqual([row["name"] for row in found["similar"]], ["Treasure Twin"])
        self.assertEqual(found["similar"][0]["shares"], ["keyword treasure"])
        self.assertTrue(tools.pickable("Treasure Twin"))


class LedgerAndToolJudgeTests(Store):
    def setUp(self):
        super().setUp()
        self.store.import_cards(records())
        self.rules = RulesStore(self.path / "rules.sqlite3", HashEmbedder())
        self.addCleanup(self.rules.close)
        self.rules.import_rules(RULES, source="https://example.invalid/rules.txt", fetched_at="today")
        self.client = Mock()
        self.client.get_rulings.return_value = []
        self.rules.ingest_deck(["Spell 0"], self.store, self.client)

    def test_settled_answers_are_kept_by_cards_question_and_rules_version(self):
        key = self.rules.ledger_key([("Spell 0", "")], "Proliferate?")
        self.assertIsNone(self.rules.recall(key))
        self.rules.remember(key, {"status": "unsupported", "errors": []})
        self.assertIsNone(self.rules.recall(key))  # Unsettled answers may still be settled later.
        self.rules.remember(key, {"status": "supported", "errors": [], "answer": "Yes."})
        self.assertEqual(self.rules.recall(key)["answer"], "Yes.")
        self.assertTrue(self.rules.recall(key)["from_ledger"])
        self.assertNotEqual(key, self.rules.ledger_key([("Spell 0", "errata")], "Proliferate?"))
        self.assertNotEqual(key, self.rules.ledger_key([("Spell 0", "")], "Proliferate twice?"))

    def test_rules_by_number_come_with_their_parent_and_subrules(self):
        found = self.rules.rules_by_id(["701.34", "CR 701.34a", "999.9"])
        self.assertEqual([r["id"] for r in found["rules"]], ["CR 701.34", "CR 701.34a"])
        self.assertEqual(found["missing"], ["999.9"])
        self.assertEqual(found["related"]["CR 701.34"]["subrules"], ["CR 701.34a"])
        self.assertEqual(found["related"]["CR 701.34a"]["parent"], "CR 701.34")

    def test_the_judge_looks_rules_up_with_tools_and_must_cite_what_it_saw(self):
        model = FakeToolModel(
            [("get_rule", {"ids": ["701.34a"]}), ("rulings", {})],
            [("finish", cited(id_="CR 999.1"))],
            [("finish", cited())],
        )
        result = self.rules.judge_with_tools(["Spell 0"], "Can I proliferate?", model)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["tools"], ["get_rule", "rulings"])
        rule = json.loads(model.results(1)[0]["content"])
        self.assertIn(QUOTE, rule["rules"][0]["text"])
        problem = model.results(2)[0]
        self.assertTrue(problem["is_error"])
        self.assertIn("CR 999.1 is not in the evidence", problem["content"])

    def test_the_review_judge_uses_tools_and_the_ledger(self):
        model = FakeToolModel([("finish", cited())])
        judge = Judge(self.rules, model, self.store, self.client)
        first = judge.answer([(["Spell 0"], "Can I proliferate?")])
        self.assertEqual(first[0]["status"], "supported")
        again = judge.answer([(["Spell 0"], "Can I proliferate?")])  # No model turn left: cached.
        self.assertTrue(again[0]["from_ledger"])
        self.assertEqual(len(model.calls), 1)

    def test_check_links_marks_and_removes_links(self):
        engine = {"stages": [{"stage": "makes", "what": "x", "cards": [
            {"name": "Spell 0", "tier": "primary", "role": "x"}]}], "gaps": [], "links": [
            {"from": "Spell 0", "to": JODAH["name"], "how": "proliferates", "source": "oracle",
             "verified": "unchecked"}]}
        judge = Mock()
        judge.answer.return_value = [{"status": "supported"}]
        checked = check_links(engine, judge, JODAH["name"])
        self.assertEqual(checked["links"][0]["verified"], "supported")


if __name__ == "__main__":
    unittest.main()
