import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.oracle import OracleStore
from mtg_agentic.pipeline import draft_deck
from mtg_agentic.review import DeckReviewer
from mtg_agentic.schema import SchemaError
from tests.fixtures import card
from tests.pipeline_fixtures import (
    JODAH,
    FakeCards,
    FakeModel,
    draft,
    make_pool,
    make_spec,
    playable_records,
)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = OracleStore(Path(temp.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(playable_records())
        self.client = FakeCards(playable_records())

    def test_bracket_and_rules_errors_drive_bounded_repair_without_resending_all_evidence(self):
        reviewer = Mock()
        reviewer.game_changer_names = []
        failed = {
            "status": "needs_revision",
            "errors": [{"error": "bracket_review", "detail": "mass land denial"}],
            "bracket": {"reasons": ["Remove land denial"]},
            "interactions": [
                {
                    "status": "contradicted",
                    "cards": ["Spell 0"],
                    "question": "Does it work?",
                    "claims": [],
                    "evidence": ["A very large retrieval payload"],
                },
                {"status": "unsupported", "cards": ["Spell 1"], "question": "And this?",
                 "claims": []},
            ],
        }
        reviewer.check.side_effect = [failed, {"status": "passed", "errors": []}]
        model = FakeModel(draft(), draft())
        result = draft_deck(
            make_pool(self.store), model, self.client, self.store, reviewer=reviewer
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["review_passed"])
        self.assertEqual(len(result["attempts"]), 2)
        feedback = json.loads(model.calls[1][0][-1]["content"])
        self.assertEqual(feedback["review"]["bracket_reasons"], ["Remove land denial"])
        # Only real problems go back to the drafter, never an interaction it could not verify.
        self.assertEqual([i["status"] for i in feedback["review"]["interactions"]],
                         ["contradicted"])
        self.assertNotIn("large retrieval", json.dumps(feedback))
        self.assertIn("large retrieval", json.dumps(result["attempts"][0]["review"]))

    def test_illegal_drafts_do_not_spend_review_calls(self):
        reviewer = Mock()
        reviewer.game_changer_names = []
        result = draft_deck(
            make_pool(self.store),
            FakeModel(draft(commander="Other Commander")),
            self.client,
            self.store,
            reviewer=reviewer,
            max_repairs=0,
        )
        self.assertFalse(result["valid"])
        reviewer.check.assert_not_called()

    def test_invalid_safety_json_enters_repair_and_exhaustion_is_invalid(self):
        reviewer = Mock()
        reviewer.game_changer_names = []
        reviewer.check.side_effect = SchemaError([{"error": "schema", "detail": "Wrong type"}])
        result = draft_deck(
            make_pool(self.store),
            FakeModel(draft()),
            self.client,
            self.store,
            reviewer=reviewer,
            max_repairs=0,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["validation"]["errors"][0]["error"], "review_schema")

    def test_library_call_without_reviewer_is_explicitly_unreviewed(self):
        result = draft_deck(make_pool(self.store), FakeModel(draft()), self.client, self.store)
        self.assertTrue(result["valid"])
        self.assertFalse(result["review_passed"])
        self.assertEqual(result["review"]["status"], "not_run")

    def test_custom_mode_flows_from_pool_to_draft_and_reviewer(self):
        reviewer = Mock()
        reviewer.game_changer_names = []
        reviewer.check.return_value = {"status": "passed", "errors": []}
        result = draft_deck(
            make_pool(self.store, make_spec(bracket3_mode="unrestricted")),
            FakeModel(draft()), self.client, self.store, reviewer=reviewer,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["bracket3_mode"], "unrestricted")
        self.assertEqual(result["spec"]["bracket3_mode"], "unrestricted")
        self.assertEqual(reviewer.check.call_args.kwargs["bracket3_mode"], "unrestricted")

    def test_reviewer_passes_mode_to_bracket_checker(self):
        service = self.service()
        service.check({"Spell 0": 1}, JODAH["name"], 3, bracket3_mode="unrestricted")
        self.assertEqual(service.brackets.check.call_args.kwargs["bracket3_mode"], "unrestricted")

    def service(self):
        bracket = Mock()
        bracket.game_changer_list = {"names": []}
        bracket.check.return_value = {"verdict": "fits", "reasons": [], "spellbook": {"combos": []}}
        rules = Mock()
        rules.ingest_deck.return_value = {"count": 3}
        rules.recall.return_value = None  # Nothing settled in the ledger yet.
        rules.interaction_request.side_effect = lambda names, question, **_: ([], [])
        rules.judge_interaction.return_value = {"status": "supported", "claims": []}
        rules.next_look.return_value = None
        model = Mock(concurrency=4, spec=["complete_json", "concurrency"])
        model.complete_json.return_value = {"status": "supported", "claims": [], "lookup": "", "conditions": [], "answer": "An answer."}
        return DeckReviewer(bracket, rules, model, self.store, self.client)

    def test_the_engine_maps_links_between_deck_cards_are_asked_and_all_rulings_synced(self):
        self.store.add_verified_card(card("Myojin Test", text="Enters with a divinity counter."))
        self.store.add_verified_card(card("Proliferator", text="Proliferate."))
        service = self.service()
        engine = engine_map(
            link("Proliferator", "Myojin Test", "proliferate adds another divinity counter"),
            link("Proliferator", "Not In Deck", "irrelevant"),
            link("Myojin Test", "Proliferator", "a known combo", source="combo"),
        )
        result = service.check({"Myojin Test": 1, "Proliferator": 1}, JODAH["name"], 3,
                               engine_map=engine)
        self.assertEqual(result["status"], "passed")
        service.rules.ingest_deck.assert_called_once_with(
            [JODAH["name"], "Myojin Test", "Proliferator"], self.store, self.client
        )
        # Only the oracle link between two deck cards is asked; combos have their own question.
        names, question = service.rules.interaction_request.call_args.args
        self.assertEqual(names, ["Proliferator", "Myojin Test"])
        self.assertIn("proliferate adds another divinity counter", question)
        self.assertEqual(service.rules.interaction_request.call_count, 1)

    def test_no_question_is_written_for_one_commander(self):
        self.store.add_verified_card(card(
            "Myojin of Blooming Dawn",
            text="Myojin of Blooming Dawn enters with an indestructible counter "
                 "on it if you cast it from your hand.",
        ))
        self.store.add_verified_card(card("Proliferator", text="Proliferate."))
        service = self.service()
        result = service.check({"Myojin of Blooming Dawn": 1, "Proliferator": 1}, JODAH["name"], 3)
        self.assertEqual(result["interactions"], [])
        service.rules.interaction_request.assert_not_called()

    def test_settled_answers_come_from_the_ledger(self):
        service = self.service()
        service.brackets.check.return_value["spellbook"]["combos"] = [combo(0, 1)]
        settled = {"status": "supported", "claims": [], "cards": ["Spell 0", "Spell 1"],
                   "question": "q", "from_ledger": True}
        service.rules.recall.return_value = settled
        result = service.check({"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3)
        self.assertEqual(result["interactions"], [settled])
        service.model.complete_json.assert_not_called()
        service.rules.recall.return_value = None
        service.check({"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3)
        self.assertEqual(service.rules.remember.call_count, 1)

    def test_flagged_combos_fail_the_review_only_when_the_rules_contradict_them(self):
        service = self.service()
        service.brackets.check.return_value["spellbook"]["combos"] = [
            {
                "cards": ["Spell 0", "Spell 1"],
                "prerequisites": ["", "Spell 0 does not have\nsummoning sickness."],
                "uses": [
                    {"card": {"name": "Spell 0"}, "zoneLocations": ["B"], "quantity": 1,
                     "battlefieldCardState": "untapped", "graveyardCardState": ""},
                    {"card": {"name": "Spell 1"}, "zoneLocations": ["H", "G"], "quantity": 1},
                ],
                "steps": "Tap Spell 0.\nCast Spell 1.\nRepeat.",
                "results": ["Infinite mana", "Infinite storm count"],
            }
        ]
        verdict = {"status": "unsupported", "cards": ["Spell 0", "Spell 1"],
                   "question": "Loop?", "claims": []}
        service.rules.judge_interaction.return_value = verdict
        result = service.check({"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3)
        # Unverified is reported, not failed: the judge could not prove it either way.
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["errors"], [])
        self.assertEqual([w["cards"] for w in result["warnings"]], [["Spell 0", "Spell 1"]])
        question = service.rules.interaction_request.call_args.args[1]
        self.assertEqual(question.splitlines()[:4], [
            "Steps: Tap Spell 0. Cast Spell 1. Repeat.",
            "Prerequisites: Spell 0 does not have summoning sickness.",
            "Starting state: Spell 0 on the battlefield (untapped); Spell 1 in your hand or in "
            "your graveyard.",
            "Claimed results: Infinite mana; Infinite storm count.",
        ])
        # Raw Spellbook fields and words the rules lookup misreads stay out of the question.
        for noise in ("zoneLocations", "CardState", "reached", "Commander Spellbook"):
            self.assertNotIn(noise, question)
        service.rules.judge_interaction.return_value = {**verdict, "status": "contradicted"}
        result = service.check({"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3)
        self.assertEqual(result["status"], "needs_revision")
        self.assertEqual(result["errors"][0]["error"], "interaction_review")

    def test_an_unverified_question_gets_another_look_with_more_evidence(self):
        service = self.service()
        service.judge.evidence = 36
        service.brackets.check.return_value["spellbook"]["combos"] = [combo(0, 1)]
        service.model.complete_json.side_effect = [
            {"status": "unsupported", "claims": [], "lookup": "divinity counters", "conditions": [], "answer": "An answer."},
            {"status": "supported", "claims": [], "lookup": "", "conditions": [], "answer": "An answer."},
        ]
        service.rules.next_look.side_effect = lambda names, question, evidence, answer, looks, **kw: (
            ([{"role": "user", "content": "again"}], ["first", "more"],
             {"lookup": answer["lookup"], "added": ["more"], "limit": kw["limit"],
              "final": kw["final"]})
            if answer["lookup"] else None
        )
        service.rules.judge_interaction.side_effect = lambda names, question, evidence, answer: {
            "status": answer["status"], "claims": [], "cards": names, "question": question,
            "evidence": evidence, "looks": answer.get("looks"),
        }
        result = service.check({"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["warnings"], [])
        interaction = result["interactions"][0]
        self.assertEqual((interaction["status"], interaction["evidence"]), ("supported", ["first", "more"]))
        # A review gets two passes, so its one extra look is also its last.
        self.assertEqual(interaction["looks"], [
            {"lookup": "divinity counters", "added": ["more"], "limit": 36, "final": True},
        ])
        self.assertEqual(service.model.complete_json.call_count, 2)
        self.assertEqual(service.rules.interaction_request.call_args.kwargs["limit"], 36)

    def test_every_rules_question_is_asked_in_parallel_without_a_cap(self):
        service = self.service()
        service.brackets.check.return_value["spellbook"]["combos"] = [combo(n, n + 25) for n in range(25)]
        lock, running, peak = threading.Lock(), [0], [0]

        def answer(messages, schema):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1
            return {"status": "supported", "claims": [], "lookup": "", "conditions": [], "answer": "An answer."}

        service.model.complete_json.side_effect = answer
        result = service.check({f"Spell {n}": 1 for n in range(50)}, JODAH["name"], 3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["interactions"]), 25)
        self.assertEqual(service.model.complete_json.call_count, 25)
        self.assertTrue(1 < peak[0] <= 4, peak[0])

    def test_a_malformed_answer_fails_only_its_own_question(self):
        service = self.service()
        service.brackets.check.return_value["spellbook"]["combos"] = [combo(0, 1), combo(2, 3)]
        service.model.complete_json.side_effect = lambda messages, schema: (
            {"wrong": True} if "Spell 0" in json.dumps(messages)
            else {"status": "supported", "claims": [], "lookup": "", "conditions": [], "answer": "An answer."}
        )
        service.rules.interaction_request.side_effect = lambda names, question, **_: (
            [{"role": "user", "content": question}], []
        )
        service.rules.judge_interaction.side_effect = lambda names, question, evidence, answer: {
            "status": answer["status"], "claims": [], "cards": names, "question": question,
        }
        result = service.check({f"Spell {n}": 1 for n in range(4)}, JODAH["name"], 3)
        statuses = {r["cards"][0]: r["status"] for r in result["interactions"]}
        self.assertEqual(statuses, {"Spell 0": "unsupported", "Spell 2": "supported"})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["warnings"]), 1)


def link(start, end, how, source="oracle"):
    return {"from": start, "to": end, "how": how, "source": source}


def engine_map(*links):
    return {"stages": [], "links": list(links)}


def combo(first, second):
    return {
        "cards": [f"Spell {first}", f"Spell {second}"], "prerequisites": ["", ""],
        "uses": [{"card": {"name": f"Spell {n}"}, "zoneLocations": ["B"], "quantity": 1}
                 for n in (first, second)],
        "steps": f"Tap Spell {first}. Untap it with Spell {second}.", "results": ["Infinite mana"],
    }