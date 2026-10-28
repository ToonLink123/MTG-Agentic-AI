import copy
import unittest
from unittest.mock import Mock, patch

from mtg_agentic.forge import ForgeError
from mtg_agentic.pipeline import build_deck


class BuildRunTests(unittest.TestCase):
    def setUp(self):
        self.deck = {
            "valid": True, "review_passed": True,
            "review": {"status": "passed", "bracket": {"verdict": "fits"}},
            "validation": {"valid": True, "errors": []},
        }
        self.simulator = Mock()
        self.simulator.simulate.return_value = {"artifact_dir": "verified-test-logs"}
        self.pods = [Mock(name="pod", fingerprint="pod-one")]
        self.metrics = {"requested_games": 10, "completed_games": 10}

    def run_build(self, **changes):
        options = {"reviewer": Mock(), "simulator": self.simulator, "pods": self.pods}
        options.update(changes)
        with (
            patch("mtg_agentic.pipeline.draft_from_request", return_value=copy.deepcopy(self.deck)),
            patch("mtg_agentic.pipeline.summarize", return_value=self.metrics),
        ):
            return build_deck("Myojin", "Jodah", Mock(), Mock(), Mock(), **options)

    def test_build_requires_ten_completed_games_in_every_pod(self):
        self.pods.append(Mock(name="second", fingerprint="pod-two"))
        result = self.run_build()
        self.assertTrue(result["valid"])
        self.assertTrue(result["build_passed"])
        self.assertEqual(result["simulation"]["status"], "completed")
        self.assertEqual(self.simulator.simulate.call_count, 2)
        self.assertEqual(len(result["simulation"]["pods"]), 2)
        self.assertTrue(all(call.args[2] == 10 for call in self.simulator.simulate.call_args_list))

    def test_failed_or_skipped_safety_review_never_reaches_forge(self):
        for key, value in (
            ("valid", False), ("review_passed", False),
            ("review", {"status": "needs_revision", "bracket": {"verdict": "uncertain"}}),
        ):
            with self.subTest(key=key):
                original = copy.deepcopy(self.deck)
                self.deck[key] = value
                result = self.run_build()
                self.assertFalse(result["valid"])
                self.assertEqual(result["simulation"]["status"], "blocked_by_gates")
                self.deck = original
        self.simulator.simulate.assert_not_called()

    def test_missing_configuration_and_lower_game_count_fail_before_drafting(self):
        for options in ({"reviewer": None}, {"simulator": None}, {"pods": []},
                        {"games": 9}, {"games": True}, {"pods": self.pods * 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_build(**options)
        self.simulator.simulate.assert_not_called()

    def test_incomplete_or_wrong_requested_batch_is_not_a_success(self):
        for metrics in (
            {"requested_games": 10, "completed_games": 9},
            {"requested_games": 1, "completed_games": 1},
        ):
            self.metrics = metrics
            result = self.run_build()
            self.assertFalse(result["valid"])
            self.assertEqual(result["simulation"]["status"], "incomplete")
            self.assertFalse(result["validation"]["valid"])

    def test_engine_failure_is_saved_as_a_failed_build(self):
        self.simulator.simulate.side_effect = ForgeError("Unsupported card")
        result = self.run_build()
        self.assertFalse(result["valid"])
        self.assertEqual(result["simulation"]["status"], "failed")
        self.assertIn("Unsupported card", result["simulation"]["error"])
