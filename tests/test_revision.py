import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtg_agentic.discovery import Candidate
from mtg_agentic.forge_config import OpponentPod
from mtg_agentic.oracle import OracleStore
from mtg_agentic.revision import apply_swaps, compare, gate_invariant, revise_deck
from tests.fixtures import card
from tests.pipeline_fixtures import FakeModel, make_pool, make_spec, playable_deck, playable_records


def metrics(wins=10, count=50):
    return {
        "context_fingerprint": "same",
        "candidate_wins": wins,
        "completed_games": count,
        "requested_games": 50,
        "completion_rate": count / 50,
        "timeout_rate": (50 - count) / 50,
        "games": [
            {
                "index": i + 1,
                "seed": i + 20,
                "candidate_seat": 1,
                "candidate_won": i < wins,
                "forge_turn": 30,
            }
            for i in range(50)
        ],
    }


class RevisionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(playable_records())
        self.pool = make_pool(self.store)
        self.deck = playable_deck()
        self.proposal = {
            "issues": [
                {
                    "kind": "deck",
                    "metric": "win_rate",
                    "reason": "Low observed wins; hypothesis only.",
                }
            ],
            "swaps": [
                {"remove": "Spell 62", "add": "Spell 69", "reason": "Test another spell."}
            ],
        }
        self.reviewer = Mock()
        self.reviewer.game_changer_names = []
        self.passed = {"status": "passed", "bracket": {"verdict": "fits"}}
        self.reviewer.check.return_value = self.passed
        self.simulator = Mock()
        self.pods = [OpponentPod("test", (), (), "2.0.15")]

    def run_revision(self, *measured, proposal=None, **kwargs):
        model = FakeModel(proposal or self.proposal)
        with patch("mtg_agentic.revision.summarize", side_effect=measured):
            report = revise_deck(
                self.deck,
                self.pool,
                model,
                self.store,
                self.reviewer,
                self.simulator,
                self.pods,
                self.path / "runs",
                max_iterations=1,
                plateau_limit=1,
                **kwargs,
            )
        return report, model

    def test_significant_gain_accepted_and_every_artifact_persisted(self):
        report, _ = self.run_revision(metrics(5), metrics(40))
        self.assertEqual(report["accepted_revisions"], 1)
        self.assertTrue(report["gate_invariant_passed"])
        self.assertEqual(report["deck"]["cards"]["Spell 69"], 1)
        self.assertNotIn("Spell 62", report["deck"]["cards"])
        self.assertEqual(self.reviewer.check.call_count, 2)
        self.assertEqual(self.simulator.simulate.call_count, 2)
        self.assertEqual(
            self.simulator.simulate.call_args_list[0].kwargs,
            self.simulator.simulate.call_args_list[1].kwargs,
        )
        root = Path(report["artifact_dir"])
        for stage in ("baseline", "proposal", "gates", "deck", "candidate", "decision"):
            self.assertTrue((root / f"0001-{stage}.json").exists())
        self.assertEqual(json.loads((root / "report.json").read_text())["accepted_revisions"], 1)

    def test_revision_retains_selected_house_rules(self):
        self.pool.spec = make_spec(bracket3_mode="unrestricted")
        self.deck["bracket3_mode"] = "unrestricted"
        report, _ = self.run_revision(metrics(5), metrics(40))
        self.assertTrue(report["gate_invariant_passed"])
        self.assertEqual(report["deck"]["bracket3_mode"], "unrestricted")
        self.assertTrue(all(
            call.kwargs["bracket3_mode"] == "unrestricted"
            for call in self.reviewer.check.call_args_list
        ))

    def test_revision_cannot_switch_house_rules(self):
        self.pool.spec = make_spec(bracket3_mode="limited-land-destruction")
        self.deck["bracket3_mode"] = "unrestricted"
        report, _ = self.run_revision()
        self.assertEqual(report["stop_reason"], "initial_gates_failed")
        self.reviewer.check.assert_not_called()
        self.simulator.simulate.assert_not_called()

    def test_small_noisy_gain_reverted_and_plateau_stops(self):
        report, _ = self.run_revision(metrics(10), metrics(12))
        self.assertEqual(report["stop_reason"], "plateau")
        self.assertEqual(report["deck"]["cards"], self.deck["cards"])
        self.assertEqual(report["accepted_revisions"], 0)

    def test_saved_review_cannot_bypass_initial_gates(self):
        self.deck.update(valid=True, review=copy.deepcopy(self.passed))
        self.reviewer.check.return_value = {
            "status": "needs_revision",
            "bracket": {"verdict": "mismatch"},
        }
        report, model = self.run_revision()
        self.assertEqual(report["stop_reason"], "initial_gates_failed")
        self.assertFalse(report["deck"]["valid"])
        self.simulator.simulate.assert_not_called()
        self.assertEqual(len(model.calls), 0)

    def test_illegal_revision_never_reaches_reviewer_or_candidate_simulation(self):
        self.proposal["swaps"][0]["add"] = "Spell 62"  # no-op is prohibited
        report, _ = self.run_revision(metrics())
        self.assertEqual(report["iterations"][0]["reason"], "revision_gates_failed")
        self.assertEqual(self.simulator.simulate.call_count, 1)
        self.assertEqual(self.reviewer.check.call_count, 1)

    def test_revision_that_fails_fresh_rules_review_is_never_simulated(self):
        self.reviewer.check.side_effect = [self.passed, {"status": "needs_revision"}]
        report, _ = self.run_revision(metrics())
        self.assertEqual(report["iterations"][0]["reason"], "revision_gates_failed")
        self.assertEqual(self.simulator.simulate.call_count, 1)

    def test_theme_package_cannot_be_traded_away_for_simulator_wins(self):
        self.store.add_verified_card(card("Off-theme Spell", mana_cost="{2}"))
        self.pool.cards.append(Candidate(self.store.get_card("Off-theme Spell"), {"support"}))
        self.proposal["swaps"][0]["add"] = "Off-theme Spell"
        report, _ = self.run_revision(metrics())
        self.assertEqual(report["iterations"][0]["reason"], "revision_gates_failed")
        self.assertEqual(self.simulator.simulate.call_count, 1)
        self.assertEqual(self.reviewer.check.call_count, 1)
        gates = json.loads(
            (Path(report["artifact_dir"]) / "0001-gates.json").read_text(encoding="utf-8")
        )
        self.assertTrue(gates["validation"]["valid"])
        self.assertTrue(gates["playability"]["valid"])
        self.assertFalse(gates["theme_package_preserved"])

    def test_unplayable_baseline_never_reaches_reviewer_or_simulator(self):
        self.deck["cards"] = {"Plains": 99}
        report, model = self.run_revision()
        self.assertEqual(report["stop_reason"], "initial_gates_failed")
        self.assertFalse(report["deck"]["valid"])
        self.reviewer.check.assert_not_called()
        self.simulator.simulate.assert_not_called()
        self.assertEqual(model.calls, [])

    def test_unreliable_simulator_stops_before_model_diagnosis(self):
        report, model = self.run_revision(metrics(count=40))
        self.assertEqual(report["stop_reason"], "simulation_unreliable")
        self.assertEqual(len(model.calls), 0)

    def test_simulation_only_diagnosis_cannot_trigger_deck_swaps(self):
        self.proposal["issues"][0]["kind"] = "simulation"
        report, _ = self.run_revision(metrics())
        self.assertEqual(report["stop_reason"], "no_evidenced_deck_revision")
        self.assertEqual(self.simulator.simulate.call_count, 1)

    def test_unavailable_observations_cannot_be_used_as_diagnosis(self):
        self.proposal["issues"][0]["metric"] = "commander_first_cast"
        report, _ = self.run_revision(metrics())
        self.assertEqual(report["stop_reason"], "error")
        self.assertEqual(self.simulator.simulate.call_count, 1)

    def test_out_of_pool_name_and_duplicate_swaps_fail(self):
        for swaps in (
            [{"remove": "Plains", "add": "Unknown"}],
            self.proposal["swaps"] * 2,
        ):
            with self.assertRaises(ValueError):
                apply_swaps(self.deck, {"swaps": swaps}, self.pool)
        with self.assertRaises(ValueError):
            apply_swaps(self.deck, {"swaps": self.proposal["swaps"] * 6}, self.pool)

    def test_gate_invariant_rejects_any_bypassed_revision(self):
        self.assertFalse(
            gate_invariant(
                {
                    "iterations": [
                        {"candidate_simulated": True, "gates_passed": False},
                    ]
                }
            )
        )
        self.assertFalse(
            gate_invariant(
                {
                    "iterations": [
                        {"accepted": True, "gates_passed": True, "candidate_simulated": False},
                    ]
                }
            )
        )

    def test_context_changes_reliability_regression_and_other_pod_damage_veto(self):
        old, new = metrics(5), metrics(40)
        self.assertTrue(compare([old], [new], confidence=0.99)["accepted"])
        for mutate in (
            lambda m: m.update(context_fingerprint="different"),
            lambda m: m.update(completion_rate=0.98),
            lambda m: m["games"][0].update(seed=999),
            lambda m: [g.update(forge_turn=40) for g in m["games"]],
        ):
            revised = copy.deepcopy(new)
            mutate(revised)
            self.assertFalse(compare([old], [revised], confidence=0.99)["accepted"])
        self.assertFalse(
            compare([old, metrics(30)], [new, metrics(10)], confidence=0.99)["accepted"]
        )
