import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.evaluation import CASES, evaluation_history, run_evaluation, score_case
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card
from tests.pipeline_fixtures import playable_deck, playable_records


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        data, self.decks = playable_records(), {}
        for case in CASES:
            commander = card(case["commander"], types="Legendary Creature — Human", colors="WUBRG")
            data.append(commander)
            names = []
            for i in range(8):
                if case["id"] == "jodah-myojin":
                    n, text = f"Myojin Test {i}", "This enters with a divinity counter."
                elif case["id"] == "poison":
                    n, text = f"Poison Test {i}", "Toxic 1"
                elif case["id"] == "politics":
                    n, text = f"Politics Test {i}", "You become the monarch."
                else:
                    n, text = f"Vanilla Test {i}", ""
                names.append(n)
                data.append(card(n, types="Creature — Spirit", text=text))
            cards = playable_deck()["cards"]
            for number in range(55, 63):
                del cards[f"Spell {number}"]
            cards.update(dict.fromkeys(names, 1))
            self.decks[case["id"]] = {
                "valid": True,
                "bracket": case["bracket"],
                "commander": case["commander"],
                "cards": cards,
                "review": {
                    "status": "passed",
                    "bracket": {
                        "target_bracket": case["bracket"],
                        "verdict": "fits",
                        "assessment": {"estimated_bracket": case["bracket"]},
                        "game_changer_list": {"names": []},
                    },
                },
            }
        self.store.import_cards(data)

    def runner(self, case):
        return copy.deepcopy(self.decks[case["id"]])

    def test_four_cases_enforce_eight_myojin_and_deliberate_bracket_two(self):
        self.assertEqual(len(CASES), 4)
        self.assertEqual(CASES[0]["minimum"], 8)
        self.assertEqual(CASES[-1]["bracket"], 2)
        for case in CASES:
            self.assertTrue(score_case(case, self.runner(case), self.store)["passed"])

    def test_claimed_validity_is_independently_rechecked(self):
        deck = self.runner(CASES[0])
        deck["cards"]["Plains"] -= 1
        result = score_case(CASES[0], deck, self.store)
        self.assertFalse(result["validator_pass"])
        self.assertFalse(result["passed"])

    def test_theme_shortfall_and_missing_poison_seed_sources_fail(self):
        deck = self.runner(CASES[0])
        del deck["cards"]["Myojin Test 7"]
        deck["cards"]["Spell 62"] = 1
        result = score_case(CASES[0], deck, self.store)
        self.assertFalse(result["theme_pass"])
        self.assertEqual(result["theme_count"], 7)
        for n in self.decks["poison"]["cards"]:
            if n.startswith("Poison Test "):
                self.store.add_verified_card(card(n, text="Proliferate."))
        result = score_case(CASES[1], self.runner(CASES[1]), self.store)
        self.assertEqual(result["theme_count"], 8)
        self.assertFalse(result["theme_pass"])

    def test_unreviewed_uncertain_or_wrong_bracket_are_not_counted_as_accurate(self):
        for modify in (
            lambda d: d.pop("review"),
            lambda d: d["review"]["bracket"].update(verdict="uncertain"),
            lambda d: d["review"]["bracket"]["assessment"].update(estimated_bracket=4),
            lambda d: d.update(bracket=4),
        ):
            deck = self.runner(CASES[0])
            modify(deck)
            self.assertFalse(score_case(CASES[0], deck, self.store)["bracket_correct"])

    def test_artifacts_and_metrics_persist_across_runs_without_overwrite(self):
        first = run_evaluation(self.runner, self.store, self.path / "runs", label="scripted")
        second = run_evaluation(self.runner, self.store, self.path / "runs", label="scripted")
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue(first["passed"])
        self.assertEqual(first["metrics"]["theme_coverage"], 1)
        saved = json.loads((Path(first["artifact_dir"]) / "report.json").read_text())
        self.assertEqual(saved["id"], first["id"])
        self.assertTrue((Path(first["artifact_dir"]) / "jodah-myojin.json").exists())
        self.assertEqual(len(evaluation_history(self.path / "runs")["runs"]), 2)

    def test_failures_stay_in_denominator_and_never_store_exception_body(self):
        def runner(case):
            if case["id"] == "politics":
                raise RuntimeError("secret must not be logged")
            return self.runner(case)

        report = run_evaluation(runner, self.store, self.path / "runs")
        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["validator_pass_rate"], 0.75)
        self.assertEqual(report["metrics"]["bracket_accuracy"], 0.75)
        self.assertNotIn("secret", json.dumps(report))

    def test_stability_gate_requires_three_consecutive_fully_passing_runs(self):
        root = self.path / "runs"
        self.assertFalse(evaluation_history(root)["stability_gate"])
        for _ in range(2):
            run_evaluation(self.runner, self.store, root)
        self.assertFalse(evaluation_history(root)["stability_gate"])
        run_evaluation(self.runner, self.store, root)
        self.assertTrue(evaluation_history(root)["stability_gate"])
        run_evaluation(lambda case: {}, self.store, root)
        self.assertFalse(evaluation_history(root)["stability_gate"])
        self.assertEqual(evaluation_history(root)["forge"], "not_requested")

    def test_stability_gate_does_not_mix_model_configurations(self):
        root = self.path / "runs"
        for model in ("old", "new", "new"):
            run_evaluation(self.runner, self.store, root, configuration={"model": model})
        self.assertFalse(evaluation_history(root)["stability_gate"])

    def test_simulation_requires_fresh_review_and_blocked_cases_never_run(self):
        with self.assertRaises(ValueError):
            run_evaluation(self.runner, self.store, self.path / "runs", simulation_runner=Mock())
        reviewer, simulation = Mock(), Mock()
        reviewer.check.return_value = {
            "status": "needs_revision",
            "bracket": {"verdict": "mismatch"},
        }
        report = run_evaluation(
            self.runner,
            self.store,
            self.path / "runs",
            simulation_runner=simulation,
            reviewer=reviewer,
        )
        simulation.assert_not_called()
        self.assertEqual(reviewer.check.call_count, 4)
        self.assertEqual(report["metrics"]["simulation_case_completion_rate"], 0)
        self.assertFalse(report["passed"])

    def test_simulation_metrics_and_commit_identity_persist_for_each_case(self):
        reviewer = Mock()
        reviewer.check.side_effect = [self.decks[c["id"]]["review"] for c in CASES]
        simulation = Mock(
            return_value=[
                {
                    "completed_games": 10,
                    "requested_games": 10,
                    "candidate_wins": 3,
                    "win_rate": 0.3,
                    "completion_rate": 1,
                    "timeout_rate": 0,
                    "games": [],
                }
            ]
        )
        report = run_evaluation(
            self.runner,
            self.store,
            self.path / "runs",
            simulation_runner=simulation,
            reviewer=reviewer,
        )
        self.assertEqual(simulation.call_count, 4)
        self.assertEqual(report["metrics"]["simulation_case_completion_rate"], 1)
        self.assertEqual(report["cases"][0]["simulation"]["pods"][0]["win_rate"], 0.3)
        self.assertIn("commit", report["repository"])
        history = evaluation_history(self.path / "runs")
        self.assertEqual(history["runs"][0]["simulations"][0]["pods"][0]["win_rate"], 0.3)

    def test_revision_gate_violation_is_a_case_failure(self):
        deck = self.runner(CASES[0])
        deck["revision"] = {
            "iterations": [
                {"candidate_simulated": True, "gates_passed": False, "accepted": True},
            ]
        }
        report = run_evaluation(
            lambda c: deck if c["id"] == CASES[0]["id"] else self.runner(c),
            self.store,
            self.path / "runs",
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["revision_gate_violations"], 1)
