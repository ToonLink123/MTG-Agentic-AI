"""The player's brief end to end: interview, quoted asks, local search, engine map, coverage."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.__main__ import main
from mtg_agentic.bulk_search import BulkSearch
from mtg_agentic.council import plan_deck
from mtg_agentic.discovery import CandidatePool, CardQuery, build_candidate_pool
from mtg_agentic.engine_map import build_engine_map, clean_map
from mtg_agentic.interview import AGENTS_DECIDE, interview_round
from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import draft_deck, draft_request
from mtg_agentic.retrieval import local_search, search_terms
from mtg_agentic.schema import SchemaError
from mtg_agentic.sources import SourceError
from mtg_agentic.spec import intake
from tests.fixtures import card
from tests.pipeline_fixtures import (
    JODAH,
    FakeCards,
    FakeModel,
    draft,
    intake_reply,
    make_pool,
    make_spec,
    map_reply,
    playable_records,
    records,
)

DICE = [
    card("Treasure Roller", colors="R", mana_cost="{2}{R}", cmc=3,
         text="Whenever you roll a die, create a Treasure token."),
    card("Dice Scholar", colors="U", mana_cost="{1}{U}", cmc=2, types="Creature — Human",
         text="Whenever you roll a die, draw a card."),
    card("Treasure Vault Land", types="Land", text="{T}: Create a Treasure token."),
    card("Plain Bear", colors="G", mana_cost="{1}{G}", cmc=2, types="Creature — Bear"),
]


def brief(request="Myojin", asks=(), deferred=(), interview=()):
    return {"request": request, "interview": list(interview), "asks": list(asks),
            "deferred": list(deferred)}


def ask(number, quote, kind="cards"):
    return {"id": f"A{number}", "quote": quote, "wants": quote, "kind": kind}


def question(text, *options):
    return {"question": text, "why": "It changes the deck.", "options": list(options)}


class InterviewTests(unittest.TestCase):
    def test_a_round_drops_repeats_card_questions_and_the_fixed_choices(self):
        answers = [{"question": "How fast should it win?", "answer": "", "source": "agents"}]
        model = FakeModel({"status": "ask", "understanding": "A Myojin deck.", "questions": [
            question("How fast should it win?", "Fast", "Slow"),
            question("Do you want [[Sol Ring]]?", "Yes", "No"),
            question("How much interaction?", "Lots", "Let the agents decide", "Type my own answer."),
        ]})
        result = interview_round("Myojin", JODAH, answers, model, round_number=2)
        self.assertEqual(result["status"], "ask")
        self.assertEqual([q["question"] for q in result["questions"]], ["How much interaction?"])
        self.assertEqual(result["questions"][0]["options"], ["Lots"])
        self.assertEqual(result["fixed_options"][0], AGENTS_DECIDE)
        payload = json.loads(model.calls[0][0][1]["content"])
        self.assertEqual((payload["round"], payload["answers"]), (2, answers))
        self.assertEqual(payload["commander_card"]["name"], JODAH["name"])
        self.assertEqual(model.calls[0][1]["properties"]["status"]["enum"], ["ask", "ready"])

    def test_the_last_round_can_only_be_ready_and_nothing_new_means_ready(self):
        model = FakeModel({"status": "ready", "understanding": "Done.", "questions": []})
        result = interview_round("Myojin", JODAH, [], model, round_number=3, max_rounds=3)
        schema = model.calls[0][1]["properties"]
        self.assertEqual((schema["status"]["enum"], schema["questions"]["maxItems"]), (["ready"], 0))
        self.assertEqual(result["status"], "ready")
        answers = [{"question": "How fast?", "answer": "Fast", "source": "option"}]
        model = FakeModel({"status": "ask", "understanding": "x", "questions": [question("How fast?", "A", "B")]})
        self.assertEqual(interview_round("Myojin", JODAH, answers, model)["status"], "ready")

    def test_bad_answers_and_rounds_are_rejected_before_any_model_call(self):
        with self.assertRaises(SchemaError):
            interview_round("Myojin", JODAH, [{"question": "x", "answer": "y", "source": "me"}], FakeModel())
        with self.assertRaises(ValueError):
            interview_round("Myojin", JODAH, [], FakeModel(), round_number=7, max_rounds=6)


class IntakeAskTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = records()
        self.store.import_cards(self.records)

    def intake(self, request, model, **options):
        return intake(request, JODAH["name"], model, FakeCards(self.records), self.store, **options)

    def test_asks_must_quote_the_player_and_get_one_repair(self):
        wrong = {"quote": "tribal Myojin", "wants": "Myojin tribal", "kind": "cards"}
        right = {"quote": "Myojin  TRIBAL", "wants": "Myojin tribal", "kind": "cards"}
        model = FakeModel(intake_reply(wrong), intake_reply(right))
        spec = self.intake("Myojin tribal with proliferate", model)
        self.assertEqual(len(model.calls), 2)
        repair = json.loads(model.calls[1][0][-1]["content"])
        self.assertEqual(repair["problems"][0]["quote"], "tribal Myojin")
        # Case and spacing may differ; the words must be the player's.
        self.assertEqual(spec.brief["asks"], [{"id": "A1", **right}])
        self.assertEqual(spec.brief["request"], "Myojin tribal with proliferate")

    def test_unquoted_asks_are_dropped_and_none_left_means_the_whole_request(self):
        made_up = {"quote": "lots of dragons", "wants": "dragons", "kind": "cards"}
        good = {"quote": "proliferate", "wants": "proliferate", "kind": "cards"}
        spec = self.intake("Myojin with proliferate", FakeModel(
            intake_reply(made_up, good), intake_reply(made_up, good)))
        self.assertEqual([a["quote"] for a in spec.brief["asks"]], ["proliferate"])
        spec = self.intake("Myojin with proliferate", FakeModel(
            intake_reply(made_up), intake_reply(made_up)))
        self.assertEqual(spec.brief["asks"], [{
            "id": "A1", "quote": "Myojin with proliferate", "wants": "the request as a whole",
            "kind": "cards",
        }])

    def test_interview_answers_are_quotable_and_agent_questions_are_deferred(self):
        answers = [
            {"question": "How fast?", "answer": "Win by turn eight", "source": "own"},
            {"question": "How much interaction?", "answer": "", "source": "agents"},
        ]
        quote = {"quote": "win by turn eight", "wants": "a turn-8 clock", "kind": "deck"}
        model = FakeModel(intake_reply(quote))
        spec = self.intake("Myojin", model, interview=answers)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(spec.brief["interview"], answers)
        self.assertEqual(spec.brief["deferred"], ["How much interaction?"])
        self.assertEqual(spec.brief["asks"][0]["kind"], "deck")
        self.assertEqual(json.loads(model.calls[0][0][1]["content"])["interview"], answers)


class LocalSearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = [*records(), *DICE]
        self.store.import_cards(self.records)
        self.spec = make_spec(themes=["dice rolling"], brief=brief("Treasure from dice rolls"))

    def test_the_text_index_stems_quotes_safely_and_follows_a_new_snapshot(self):
        self.assertEqual(self.store.text_count("treasure tokens"), 2)
        self.assertEqual(self.store.text_count('roll "a die'), 2)  # Quotes are never syntax.
        names = [name for name, _ in self.store.search_text("rolls")]
        self.assertEqual(sorted(names), ["Dice Scholar", "Treasure Roller"])
        self.store.import_cards([*self.records, card("Dice Tower", text="Roll a die twice.")])
        self.assertEqual(self.store.text_count("roll"), 3)

    def test_terms_come_from_the_player_and_the_commander_minus_common_words(self):
        leader = card("Dice Lord, Gambler", colors="R", types="Legendary Creature — Human",
                      text="Whenever you roll a die, Dice Lord deals 1 damage to target opponent.")
        spec = make_spec(brief=brief("Dice Lord with Treasure from dice rolls"))
        terms = search_terms(spec, leader)
        self.assertEqual(terms["treasure"], ("treasure", 1.5))
        self.assertNotIn("lord", terms)  # Nor is it a theme when the player names it.
        self.assertIn("roll", terms)  # "rolls" and the commander's "roll" are one term.
        self.assertEqual(terms["roll"][1], 1.5)  # The player's words outrank the commander's.
        self.assertIn("damage", terms)
        self.assertNotIn("dice lord", terms)  # The commander's own name is not a theme.
        self.assertNotIn("target", terms)

    def test_cards_matching_more_rare_terms_rank_first_and_lands_stay_out(self):
        found = local_search(self.spec, self.store)
        names = [hit["name"] for hit in found["hits"]]
        # Each matches two of the player's rare terms; a land or a card matching none never shows.
        self.assertEqual(set(names), {"Treasure Roller", "Dice Scholar"})
        rolls = ({"rolls", "rolling"} & {t["term"] for t in found["terms"]}).pop()
        matched = {hit["name"]: sorted(hit["matched"]) for hit in found["hits"]}
        self.assertEqual(matched["Treasure Roller"], sorted([rolls, "treasure"]))
        self.assertEqual(matched["Dice Scholar"], ["dice", rolls])
        # "rolls" and "rolling" stem alike and find the same cards, so they count once.
        self.assertEqual(len({"rolls", "rolling"} & {t["term"] for t in found["terms"]}), 1)
        self.assertTrue(all(isinstance(hit["score"], int) for hit in found["hits"]))

    def test_hits_join_the_pool_as_theme_cards_and_survive_a_reload(self):
        found = local_search(self.spec, self.store)
        client = FakeCards(self.records, search_results=[[], []])
        pool = build_candidate_pool(
            self.spec, [CardQuery("ramp", "otag:ramp")], client, self.store, limit=60,
            local_search=found,
        )
        roles = {c.card["name"]: c.roles for c in pool.cards}
        self.assertEqual(roles["Treasure Roller"], {"theme"})
        self.assertNotIn("Treasure Vault Land", roles)
        again = CandidatePool.from_dict(pool.to_dict(), self.store)
        self.assertEqual(again.local_search, found)


class FakeSpellbook:
    def __init__(self, combos=(), error=None):
        self.combos, self.error, self.calls = list(combos), error, []

    def find_combos(self, cards, commander):
        self.calls.append((cards, commander))
        if self.error:
            raise self.error
        return {"combos": self.combos}


class EngineMapTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = playable_records()
        self.store.import_cards(self.records)
        self.client = FakeCards(self.records)
        asks = [ask(1, "Myojin"), ask(2, "graveyard recursion"), ask(3, "keep it casual", "deck")]
        self.spec = make_spec(brief=brief("Myojin, graveyard recursion, keep it casual", asks))
        self.pool = make_pool(self.store, spec=self.spec)

    def mapped(self, model, **options):
        self.pool.local_search = {"terms": [{"term": "treasure", "cards": 2}],
                                  "hits": [{"name": "Spell 3", "score": 9, "matched": ["treasure"]}]}
        return build_engine_map(self.pool, model, self.store, **options).engine_map

    def test_only_pool_cards_stay_combos_are_proven_and_asks_are_checked(self):
        reply = map_reply(
            {"stage": "pays_off", "what": "wins", "cards": [
                {"name": "spell 1", "tier": "primary", "role": "finisher"},
                {"name": "Not A Card", "tier": "primary", "role": "made up"},
            ]},
            {"stage": "makes", "what": "counters", "cards": [
                {"name": "Spell 2", "tier": "adjacent", "role": "counters"},
            ]},
            links=[
                {"from": "Spell 1", "to": "Spell 2", "how": "infinite", "source": "combo"},
                {"from": "Spell 3", "to": "Spell 4", "how": "claimed", "source": "combo"},
                {"from": JODAH["name"], "to": "Spell 5", "how": "casts it free", "source": "oracle"},
            ],
            asks=[{"ask": "A1", "cards": ["Spell 1", "Spell 1 "], "note": ""}],
        )
        model = FakeModel(reply)
        spellbook = FakeSpellbook([{"cards": ["Spell 1", "Spell 2"], "results": ["Infinite"]}])
        engine = self.mapped(model, spellbook=spellbook)
        self.assertEqual([s["stage"] for s in engine["stages"]], ["makes", "pays_off"])
        self.assertEqual([c["name"] for c in engine["stages"][1]["cards"]], ["Spell 1"])
        self.assertEqual([link["source"] for link in engine["links"]], ["combo", "oracle", "oracle"])
        self.assertEqual(engine["links"][2]["from"], JODAH["name"])
        self.assertEqual([a["cards"] for a in engine["asks"]], [["Spell 1"], [], []])
        self.assertTrue(engine["gaps"][0].startswith('No pool card serves A2: "graveyard recursion"'))
        self.assertIn("1 name(s) on the map are not pool cards", engine["gaps"][1])
        payload = json.loads(model.calls[0][0][1]["content"])
        searched = {c["name"]: c.get("search") for c in payload["candidates"]}
        self.assertEqual(searched["Spell 3"], {"rank": 1, "matched": ["treasure"]})
        self.assertIsNone(searched["Spell 4"])
        self.assertEqual(payload["combos"], [{"cards": ["Spell 1", "Spell 2"], "results": ["Infinite"]}])
        self.assertEqual(model.calls[0][1]["properties"]["asks"]["items"]["properties"]["ask"]["enum"],
                         ["A1", "A2", "A3"])

    def test_no_spellbook_means_no_combos_and_a_saved_map_is_rechecked(self):
        reply = map_reply({"stage": "makes", "what": "x", "cards": [
            {"name": "Spell 1", "tier": "primary", "role": "x"},
            {"name": "Spell 2", "tier": "primary", "role": "x"},
        ]})
        engine = self.mapped(FakeModel(reply), spellbook=FakeSpellbook(error=SourceError("offline")))
        self.pool.engine_map = engine
        saved = self.pool.to_dict()
        saved["cards"] = [c for c in saved["cards"] if c["name"] != "Spell 2"]
        again = CandidatePool.from_dict(saved, self.store)
        self.assertEqual([c["name"] for c in again.engine_map["stages"][0]["cards"]], ["Spell 1"])
        self.assertEqual(again.engine_map["gaps"], engine["gaps"])  # Reloads add no notes.

    def test_gap_searches_run_locally_and_the_mapper_places_what_they_found(self):
        recursion = card("Recursion Engine", colors="B", mana_cost="{2}{B}", cmc=3,
                         text="{T}: Return target creature card from your graveyard to your hand.")
        banned = card("Banned Recursion", colors="B", legal="banned",
                      text="Return target creature card from your graveyard to the battlefield.")
        self.store.import_cards([*self.records, recursion, banned])
        first = map_reply(searches=[
            {"for": "A2", "query": 'o:"from your graveyard"'},
            {"for": "card draw", "query": "set:neo"},
        ])
        second = map_reply({"stage": "spends", "what": "recursion", "cards": [
            {"name": "Recursion Engine", "tier": "primary", "role": "returns dead creatures"},
        ]}, asks=[{"ask": "A2", "cards": ["Recursion Engine"], "note": ""}])
        model = FakeModel(first, second)
        pool = build_engine_map(self.pool, model, self.store, search=BulkSearch(self.store))
        added = {c.card["name"]: c.roles for c in pool.cards if c.card["name"] not in
                 {c.card["name"] for c in self.pool.cards}}
        self.assertEqual(added, {"Recursion Engine": {"theme"}})  # Found for ask A2.
        searches = pool.engine_map["searches"]
        self.assertEqual(searches[0]["added"], ["Recursion Engine"])
        self.assertIn("not available offline", searches[1]["error"])
        self.assertEqual(pool.engine_map["asks"][1]["cards"], ["Recursion Engine"])
        self.assertEqual(model.calls[0][1]["properties"]["searches"]["maxItems"], 6)
        self.assertEqual(model.calls[1][1]["properties"]["searches"]["maxItems"], 0)
        update = json.loads(model.calls[1][0][-1]["content"])
        self.assertEqual([c["name"] for c in update["added_candidates"]], ["Recursion Engine"])
        again = CandidatePool.from_dict(pool.to_dict(), self.store)
        self.assertEqual(again.engine_map["searches"], searches)

    def test_drafters_may_name_cards_outside_the_pool_and_code_checks_each(self):
        engine = card("Outside Engine", colors="U", mana_cost="{1}{U}", cmc=2,
                      text="Draw two cards.")
        banned = card("Banned Engine", colors="U", legal="banned")
        self.store.import_cards([*self.records, engine, banned])
        pool = make_pool(self.store)
        picks = [{"name": "outside engine", "tier": "core", "reason": "draws and ramps"},
                 {"name": "Banned Engine", "tier": "flex", "reason": "x"},
                 {"name": "Made Up Card", "tier": "flex", "reason": "x"},
                 {"name": "Plains", "tier": "flex", "reason": "x"},
                 {"name": "Spell 3", "tier": "strong", "reason": "already a candidate"}]
        model = FakeModel({**draft(), "outside": picks})
        with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("Unexpected HTTP")):
            deck = draft_deck(pool, model, self.client, self.store)
        self.assertIn("Outside Engine", deck["cards"])
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        outside = deck["outside"][0]
        self.assertEqual(outside["accepted"], ["Outside Engine", "Spell 3"])
        self.assertEqual({r["name"]: r["reason"][:14] for r in outside["rejected"]}, {
            "Banned Engine": "outside the co", "Made Up Card": "no card has th",
            "Plains": "lands are gene",
        })

    def test_drafting_sees_the_map_and_must_serve_every_mapped_ask(self):
        self.pool.engine_map = clean_map(
            map_reply(asks=[{"ask": "A1", "cards": ["Spell 66"], "note": ""}]),
            [c.card["name"] for c in self.pool.cards], JODAH["name"], self.spec.brief["asks"],
        )
        pool_message = draft_request(self.pool)[0]["content"]
        self.assertEqual(json.loads(pool_message.split("\n", 1)[1])["engine_map"],
                         self.pool.engine_map)
        fixed = draft()
        fixed["picks"]["flex"] = [f"Spell {n}" for n in (*range(40), *range(41, 63), 66)]
        model = FakeModel(draft(), fixed)
        with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("Unexpected HTTP")):
            deck = draft_deck(self.pool, model, self.client, self.store)
        first = deck["attempts"][0]["errors"]
        self.assertEqual([e["error"] for e in first], ["ask_uncovered"])
        self.assertIn('"Myojin"', first[0]["detail"])
        self.assertIn("Spell 66", first[0]["detail"])
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        report = {a["id"]: a["cards"] for a in deck["asks"]}
        self.assertEqual(report, {"A1": ["Spell 66"], "A2": [], "A3": []})

    def test_the_council_plan_decides_questions_left_to_the_agents(self):
        spec = make_spec(brief=brief(deferred=["How much interaction?"]))
        pool = make_pool(self.store, spec=spec)
        pool.engine_map = {"summary": "s", "resources": ["r"], "stages": [], "links": [],
                           "asks": [], "gaps": []}
        plan = {"game_plan": "Win.", "non_goals": [], "win_conditions": ["Spell 1"],
                "game_changers": [], "key_cards": [],
                "role_targets": {"ramp": 10, "draw": 10, "removal": 8, "wipes": 3,
                                 "creatures": 15, "protection": 4},
                "targets_reason": "Usual counts.",
                "decisions": [{"question": "How much interaction?", "decision": "A moderate amount"}]}
        model = FakeModel(plan)
        result = plan_deck(pool, model, set())
        self.assertEqual(result["decisions"], plan["decisions"])
        schema = model.calls[0][1]["properties"]["decisions"]
        self.assertEqual((schema["minItems"], schema["items"]["properties"]["question"]["enum"]),
                         (1, ["How much interaction?"]))
        pool_message = model.calls[0][0][0]["content"]
        self.assertEqual(json.loads(pool_message.split("\n", 1)[1])["engine_map"], pool.engine_map)
        model = FakeModel({key: value for key, value in plan.items() if key != "decisions"})
        self.assertEqual(plan_deck(self.pool, model, set())["decisions"], [])
        self.assertNotIn("decisions", model.calls[0][1]["properties"])


class BriefCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.cards = playable_records()
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards(self.cards)
            spec = make_spec(brief=brief(asks=[ask(1, "Myojin")]))
            self.pool = make_pool(store, spec=spec).to_dict()

    def run_cli(self, *args, model):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
            patch("mtg_agentic.__main__._make_model", return_value=model),
            patch("mtg_agentic.scryfall.ScryfallClient.get_card",
                  side_effect=FakeCards(self.cards).get_card),
            patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("No real HTTP")),
        ):
            status = main(["--quiet", "--data-dir", str(self.path), *args])
        return status, json.loads(output.getvalue())

    def test_interview_then_intake_with_the_answers(self):
        answers = [{"question": "How fast?", "answer": "Slow and grindy", "source": "option"}]
        (self.path / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
        model = FakeModel({"status": "ready", "understanding": "A slow Myojin deck.", "questions": []})
        status, result = self.run_cli(
            "interview", "--commander", JODAH["name"], "--request", "Myojin",
            "--answers", str(self.path / "answers.json"), "--round", "2", model=model,
        )
        self.assertEqual((status, result["status"], result["round"]), (0, "ready", 2))
        self.assertEqual(json.loads(model.calls[0][0][1]["content"])["answers"], answers)
        quote = {"quote": "slow and grindy", "wants": "a long game", "kind": "deck"}
        status, spec = self.run_cli(
            "intake", "--commander", JODAH["name"], "--request", "Myojin",
            "--interview", str(self.path / "answers.json"), model=FakeModel(intake_reply(quote)),
        )
        self.assertEqual(status, 0, spec)
        self.assertEqual(spec["brief"]["interview"], answers)
        self.assertEqual(spec["brief"]["asks"][0]["quote"], "slow and grindy")

    def test_engine_map_prints_the_mapped_pool_or_a_warning(self):
        (self.path / "pool.json").write_text(json.dumps(self.pool), encoding="utf-8")
        reply = map_reply({"stage": "makes", "what": "x", "cards": [
            {"name": "Spell 1", "tier": "primary", "role": "x"}]},
            asks=[{"ask": "A1", "cards": ["Spell 1"], "note": ""}])
        with patch("mtg_agentic.brackets.SpellbookClient.find_combos",
                   return_value={"combos": []}) as combos:
            status, mapped = self.run_cli("engine-map", "--pool", str(self.path / "pool.json"),
                                          model=FakeModel(reply))
        self.assertEqual(status, 0, mapped)
        combos.assert_called_once()
        self.assertEqual(mapped["engine_map"]["asks"], [{"ask": "A1", "cards": ["Spell 1"], "note": ""}])
        status, failed = self.run_cli(
            "engine-map", "--pool", str(self.path / "pool.json"), "--no-combos",
            model=FakeModel(SchemaError([{"error": "json", "detail": "Expected strict JSON"}])),
        )
        self.assertEqual(status, 1)
        self.assertEqual((failed["engine_map"], failed["map_errors"][0]["error"]), ({}, "json"))


class FollowUpTests(unittest.TestCase):
    def test_the_orchestrator_sees_the_map_and_the_councils_decisions(self):
        from mtg_agentic.edit import Toolbox, edit_deck

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(store.close)
        cards = playable_records()
        store.import_cards(cards)
        spec = make_spec(brief=brief(asks=[ask(1, "Myojin")]))
        pool = make_pool(store, spec=spec)
        pool.engine_map = clean_map(
            map_reply(asks=[{"ask": "A1", "cards": ["Spell 66"], "note": ""}]),
            [c.card["name"] for c in pool.cards], JODAH["name"], spec.brief["asks"],
        )
        with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("HTTP")):
            deck = draft_deck(pool, FakeModel(draft()), FakeCards(cards), store, max_repairs=0)
        deck["council"] = {"game_plan": "Win.", "non_goals": [], "core": [],
                           "decisions": [{"question": "Speed?", "decision": "Slow"}]}
        finish = {"tool": "finish", "query": "", "cards": [],
                  "swaps": [{"remove": "Spell 40", "add": "Spell 66", "reason": "the ask"}],
                  "reply": "Added Spell 66."}
        model = FakeModel(finish)
        result = edit_deck(deck, pool, model, "Serve my Myojin ask", tools=Toolbox(store))
        payload = json.loads(model.calls[0][0][1]["content"])
        self.assertEqual(payload["engine_map"], pool.engine_map)
        self.assertEqual(payload["plan"]["decisions"], [{"question": "Speed?", "decision": "Slow"}])
        self.assertEqual(payload["spec"]["brief"]["asks"][0]["quote"], "Myojin")
        self.assertEqual(result["asks"][0]["cards"], ["Spell 66"])


if __name__ == "__main__":
    unittest.main()
