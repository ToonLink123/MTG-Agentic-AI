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
                    "status": "unsupported",
                    "cards": ["Spell 0"],
                    "question": "Does it work?",
                    "claims": [],
                    "evidence": ["A very large retrieval payload"],
                }
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
        self.assertEqual(feedback["review"]["interactions"][0]["status"], "unsupported")
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
        rules.interaction_request.side_effect = lambda names, question: ([], [])
        rules.judge_interaction.return_value = {"status": "supported", "claims": []}
        model = Mock(concurrency=4)
        model.complete_json.return_value = {"status": "supported", "claims": []}
        return DeckReviewer(bracket, rules, model, self.store, self.client)

    def test_jodah_myojin_proliferate_engine_and_all_deck_rulings_are_checked(self):
        self.store.add_verified_card(card("Myojin Test", text="Enters with a divinity counter."))
        self.store.add_verified_card(card("Proliferator", text="Proliferate."))
        service = self.service()
        result = service.check({"Myojin Test": 1, "Proliferator": 1}, JODAH["name"], 3)
        self.assertEqual(result["status"], "passed")
        service.rules.ingest_deck.assert_called_once_with(
            [JODAH["name"], "Myojin Test", "Proliferator"], self.store, self.client
        )
        names, question = service.rules.interaction_request.call_args.args
        self.assertEqual(names, [JODAH["name"], "Myojin Test", "Proliferator"])
        self.assertIn("casting from hand", question)
        self.assertIn("none is already present", question)

    def test_flagged_combos_are_checked_and_unsupported_claims_require_revision(self):
        service = self.service()
        service.brackets.check.return_value["spellbook"]["combos"] = [
            {
                "cards": ["Spell 0", "Spell 1"],
                "prerequisites": ["both untapped"],
                "uses": [],
                "steps": "Loop",
                "results": ["Infinite mana"],
            }
        ]
        service.rules.judge_interaction.return_value = {
            "status": "unsupported",
            "cards": ["Spell 0", "Spell 1"],
            "question": "Loop?",
            "claims": [],
        }
        result = service.check({"Spell 0": 1, "Spell 1": 1}, JODAH["name"], 3)
        self.assertEqual(result["status"], "needs_revision")
        self.assertEqual(result["errors"][0]["error"], "interaction_review")
        self.assertIn("both untapped", service.rules.interaction_request.call_args.args[1])

    def test_new_myojin_review_uses_indestructible_not_divinity_counters(self):
        self.store.add_verified_card(card(
            "Myojin of Blooming Dawn",
            text="Myojin of Blooming Dawn enters with an indestructible counter "
                 "on it if you cast it from your hand.",
        ))
        self.store.add_verified_card(card("Proliferator", text="Proliferate."))
        service = self.service()
        service.check({"Myojin of Blooming Dawn": 1, "Proliferator": 1}, JODAH["name"], 3)
        question = service.rules.interaction_request.call_args.args[1]
        self.assertIn("indestructible counter", question)
        self.assertNotIn("divinity counter", question)

    def test_every_rules_question_is_asked_in_parallel_without_a_cap(self):
        names = [f"Myojin {n}" for n in range(25)]
        for name in names:
            self.store.add_verified_card(card(name, text="Enters with a divinity counter."))
        service = self.service()
        lock, running, peak = threading.Lock(), [0], [0]

        def answer(messages, schema):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1
            return {"status": "supported", "claims": []}

        service.model.complete_json.side_effect = answer
        result = service.check(dict.fromkeys(names, 1), JODAH["name"], 3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["interactions"]), 25)
        self.assertEqual(service.model.complete_json.call_count, 25)
        self.assertTrue(1 < peak[0] <= 4, peak[0])

    def test_a_malformed_answer_fails_only_its_own_question(self):
        for name in ("Myojin A", "Myojin B"):
            self.store.add_verified_card(card(name, text="Enters with a divinity counter."))
        service = self.service()
        service.model.complete_json.side_effect = lambda messages, schema: (
            {"wrong": True} if "Myojin A" in json.dumps(messages) else {"status": "supported", "claims": []}
        )
        service.rules.interaction_request.side_effect = lambda names, question: (
            [{"role": "user", "content": question}], []
        )
        service.rules.judge_interaction.side_effect = lambda names, question, evidence, answer: {
            "status": answer["status"], "claims": [], "cards": names, "question": question,
        }
        result = service.check({"Myojin A": 1, "Myojin B": 1}, JODAH["name"], 3)
        statuses = {r["question"].split(",")[0]: r["status"] for r in result["interactions"]}
        self.assertEqual(statuses, {"For Myojin A": "unsupported", "For Myojin B": "supported"})
        self.assertEqual(result["status"], "needs_revision")
