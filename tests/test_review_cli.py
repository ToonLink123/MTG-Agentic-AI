import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtg_agentic.__main__ import main
from mtg_agentic.oracle import OracleStore
from tests.pipeline_fixtures import FakeModel, draft, make_pool, playable_deck, playable_records


class ReviewCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards(playable_records())
            self.pool = make_pool(store).to_dict()
        self.pool_path = self.path / "pool.json"
        self.pool_path.write_text(json.dumps(self.pool))
        self.deck = self.path / "deck.json"
        self.deck.write_text(json.dumps(playable_deck()))

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--quiet", "--data-dir", str(self.path), *args])
        return code, json.loads(output.getvalue())

    @patch("mtg_agentic.__main__._reviewer")
    @patch("mtg_agentic.__main__._make_model")
    def test_draft_reviews_by_default(self, model, reviewer):
        model.return_value = FakeModel(draft())
        service = reviewer.return_value
        service.game_changer_names = []
        service.check.return_value = {"status": "passed", "errors": []}
        code, result = self.run_cli("draft", "--pool", str(self.pool_path))
        self.assertEqual(code, 0)
        self.assertTrue(result["review_passed"])
        reviewer.assert_called_once()
        service.check.assert_called_once()

    @patch("mtg_agentic.__main__._reviewer")
    @patch("mtg_agentic.__main__._make_model")
    def test_explicit_skip_is_marked_unreviewed(self, model, reviewer):
        model.return_value = FakeModel(draft())
        code, result = self.run_cli("draft", "--pool", str(self.pool_path), "--skip-review")
        self.assertEqual(code, 0)
        self.assertFalse(result["review_passed"])
        reviewer.assert_not_called()

    @patch("mtg_agentic.__main__.BracketChecker")
    @patch("mtg_agentic.__main__._make_model")
    def test_bracket_check_uncertainty_returns_exit_one(self, model, checker):
        checker.return_value.check.return_value = {"verdict": "uncertain", "reasons": ["Unknown"]}
        code, result = self.run_cli("bracket-check", str(self.deck), "--bracket", "2")
        self.assertEqual(code, 1)
        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(checker.return_value.check.call_args.args[-1], 2)

    @patch("mtg_agentic.__main__.BracketChecker")
    @patch("mtg_agentic.__main__._make_model")
    def test_bracket_check_uses_saved_mode_and_explicit_override(self, model, checker):
        checker.return_value.check.return_value = {"verdict": "fits"}
        self.deck.write_text(json.dumps(playable_deck(
            spec={"bracket3_mode": "unrestricted"}
        )))
        code, _ = self.run_cli("bracket-check", str(self.deck))
        self.assertEqual(code, 0)
        self.assertEqual(checker.return_value.check.call_args.kwargs["bracket3_mode"], "unrestricted")
        code, _ = self.run_cli(
            "bracket-check", str(self.deck), "--bracket3-mode", "limited-land-destruction"
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            checker.return_value.check.call_args.kwargs["bracket3_mode"],
            "limited-land-destruction",
        )
        code, result = self.run_cli("bracket-check", str(self.deck), "--bracket", "2")
        self.assertEqual(code, 2)
        self.assertIn("require bracket 3", result["errors"][0]["detail"])

    @patch("mtg_agentic.__main__._reviewer")
    @patch("mtg_agentic.__main__._make_model")
    def test_review_deck_uses_selected_mode(self, model, reviewer):
        reviewer.return_value.game_changer_names = []
        reviewer.return_value.check.return_value = {"status": "passed", "errors": []}
        code, _ = self.run_cli(
            "review-deck", str(self.deck), "--bracket3-mode", "limited-land-destruction"
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            reviewer.return_value.check.call_args.kwargs["bracket3_mode"],
            "limited-land-destruction",
        )

    @patch("mtg_agentic.__main__._reviewer")
    @patch("mtg_agentic.__main__._make_model")
    def test_review_deck_includes_independent_validation(self, model, reviewer):
        reviewer.return_value.game_changer_names = []
        reviewer.return_value.check.return_value = {"status": "passed", "errors": []}
        code, result = self.run_cli("review-deck", str(self.deck))
        self.assertEqual(code, 0)
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(result["playability"]["valid"])
        reviewer.return_value.check.assert_called_once()

    @patch("mtg_agentic.__main__._reviewer")
    @patch("mtg_agentic.__main__._make_model")
    def test_unplayable_deck_cannot_reach_safety_review(self, model, reviewer):
        reviewer.return_value.game_changer_names = []
        self.deck.write_text(json.dumps(playable_deck(cards={"Plains": 99})))
        code, result = self.run_cli("review-deck", str(self.deck))
        self.assertEqual(code, 1)
        self.assertTrue(result["validation"]["valid"])
        self.assertFalse(result["playability"]["valid"])
        self.assertFalse(result["valid"])
        reviewer.return_value.check.assert_not_called()

    @patch("mtg_agentic.__main__.RulesStore")
    @patch("mtg_agentic.__main__._make_model")
    def test_unsupported_interaction_returns_no_success(self, model, rules):
        service = rules.return_value.__enter__.return_value
        service.check_interaction.return_value = {"status": "unsupported", "claims": []}
        code, result = self.run_cli(
            "check-interaction",
            "Spell 0",
            "--question",
            "Does this work?",
            "--embedding-backend",
            "hash",
        )
        self.assertEqual(code, 1)
        self.assertEqual(result["claims"], [])
        service.ingest_deck.assert_called_once()

    @patch("mtg_agentic.__main__.RulesStore")
    @patch("mtg_agentic.__main__._make_model")
    def test_a_question_names_its_cards_in_plain_words(self, model, rules):
        service = rules.return_value.__enter__.return_value
        service.check_interaction.return_value = {"status": "supported", "claims": []}
        code, result = self.run_cli(
            "check-interaction", "--question", "Can jodah cast spell 3 for free?",
            "--embedding-backend", "hash", "--provider", "ollama",
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["answered_by"]["provider"], "ollama")
        cards = ["Jodah, Archmage Eternal", "Spell 3"]
        service.ingest_deck.assert_called_once()
        self.assertEqual(service.ingest_deck.call_args.args[0], cards)
        self.assertEqual(service.check_interaction.call_args.args[0], cards)

    @patch("mtg_agentic.__main__.ScryfallClient")
    @patch("mtg_agentic.__main__.RulesStore")
    @patch("mtg_agentic.__main__._make_model")
    def test_a_name_no_card_matches_is_refused_before_any_model_call(self, model, rules, scryfall):
        from mtg_agentic.scryfall import CardNotFound

        client = scryfall.return_value.__enter__.return_value
        client.get_card.side_effect = CardNotFound("No card", status=404)
        code, result = self.run_cli(
            "check-interaction", "No Such Card Anywhere", "--question", "Does it work?",
            "--embedding-backend", "hash",
        )
        self.assertEqual(code, 2)
        self.assertIn("[[No Such Card Anywhere]]", result["errors"][0]["detail"])
        # Like a commander: Scryfall's exact-then-fuzzy lookup is the last resort.
        client.get_card.assert_called_once_with("No Such Card Anywhere")
        rules.return_value.__enter__.return_value.check_interaction.assert_not_called()

    def test_empty_history_is_read_only_and_does_not_claim_stability(self):
        code, result = self.run_cli("evaluation-history", "--runs-dir", str(self.path / "runs"))
        self.assertEqual(code, 0)
        self.assertFalse(result["stability_gate"])
        self.assertFalse((self.path / "runs").exists())

    @patch("mtg_agentic.__main__._reviewer")
    @patch("mtg_agentic.__main__._make_model")
    def test_evaluate_keeps_all_four_model_failures_in_metrics(self, model, reviewer):
        model.return_value = Mock()
        model.return_value.model = "test"
        model.return_value.context_size = None
        model.return_value.complete_json.side_effect = RuntimeError("unavailable")
        reviewer.return_value.rules.metadata = {"embedding": "test"}
        code, result = self.run_cli("evaluate", "--runs-dir", str(self.path / "runs"))
        self.assertEqual(code, 1)
        self.assertEqual(len(result["cases"]), 4)
        self.assertEqual(result["metrics"]["validator_pass_rate"], 0)
