import tempfile
import unittest
from pathlib import Path

from mtg_agentic.forge_config import file_hash
from mtg_agentic.forge_metrics import commander_cast, parse_game, summarize, wilson

FIXTURES = Path(__file__).parent / "fixtures_forge"


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.raw = {"returncode": 0, "candidate_seat": 1}
        self.clean = (FIXTURES / "clean-result.log").read_text()
        self.clock = (FIXTURES / "clock-false-win.log").read_text()

    def test_real_clean_result_and_turn(self):
        g = parse_game(self.clean, self.raw)
        self.assertEqual(g["status"], "completed")
        self.assertEqual(g["winner_seat"], 2)
        self.assertFalse(g["candidate_won"])
        self.assertEqual(g["forge_turn"], 33)
        self.assertTrue(parse_game(self.clean, {**self.raw, "candidate_seat": 2})["candidate_won"])

    def test_real_clock_result_never_counts_false_candidate_win(self):
        g = parse_game(self.clock, self.raw)
        self.assertEqual(g["status"], "timeout")
        self.assertFalse(g["candidate_won"])
        self.assertIsNone(g["winner"])
        self.assertEqual(g["reported_winner"], "Ai(1)-Agentic Jodah Smoke")

    def test_hard_timeout_exit_errors_and_missing_results_fail_closed(self):
        for update, expected in (
            ({"timed_out": True}, "timeout"),
            ({"returncode": 1}, "engine_error"),
            ({"cancelled": True}, "cancelled"),
            ({"log_limit_exceeded": True}, "log_limit"),
        ):
            self.assertEqual(parse_game(self.clean, {**self.raw, **update})["status"], expected)
        self.assertEqual(parse_game("Simulation mode", self.raw)["status"], "unparsed")
        self.assertEqual(parse_game(self.clean * 2, self.raw)["status"], "unparsed")
        self.assertEqual(
            parse_game(self.clean, self.raw, release="new")["status"], "unsupported_release"
        )

    def test_ai_decision_timeout_is_a_warning_but_other_exceptions_still_fail(self):
        # As printed by a 2.0.15 game under CPU load; the game then finished normally.
        trace = (
            "java.util.concurrent.TimeoutException\n"
            "\tat java.base/java.util.concurrent.FutureTask.get(FutureTask.java:204)\n"
            "\tat forge.ai.AiController.chooseSpellAbilityToPlayFromList(AiController.java:1686)\n"
        )
        g = parse_game(trace + trace + self.clean, self.raw)
        self.assertEqual(g["status"], "completed")
        self.assertEqual(g["runtime_warnings"]["ai_decision_timeout"], 2)
        crash = "java.lang.NullPointerException\n\tat forge.game.Game.x(Game.java:1)\n"
        self.assertEqual(parse_game(trace + crash + self.clean, self.raw)["status"], "engine_error")
        engine_timeout = trace.replace("forge.ai.AiController", "forge.game.GameAction")
        self.assertEqual(parse_game(engine_timeout + self.clean, self.raw)["status"], "engine_error")

    def test_synthetic_inconsistent_winners_are_not_success(self):
        text = self.clean.replace(
            "Ai(1)-Agentic Jodah Smoke has lost because life total reached 0",
            "Ai(1)-Agentic Jodah Smoke has won because all opponents have lost",
        )
        self.assertEqual(parse_game(text, self.raw)["status"], "inconsistent_result")

    def test_win_turn_counts_the_winners_own_turns_not_forge_outcome_counter(self):
        # Ten global turns over four seats; Forge reports ceil(10 / 2) = 5.
        seats = [2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
        trace = "".join(f"Turn: Turn {n} (Ai({s})-Deck {s})\n" for n, s in enumerate(seats, 1))
        text = trace + self.clean.replace("Turn 33", "Turn 5")
        g = parse_game(text, {**self.raw, "candidate_seat": 2})
        self.assertTrue(g["candidate_won"])
        self.assertEqual(g["forge_turn"], 5)
        self.assertEqual(g["candidate_turns"], 3)
        self.assertEqual(g["table_rounds"], 3)
        self.assertEqual(parse_game(text, self.raw)["candidate_turns"], 2)
        # A trace that disagrees with Forge's own counter, or is incomplete, is not used.
        self.assertIsNone(parse_game(trace + self.clean, self.raw)["candidate_turns"])
        gap = text.replace("Turn: Turn 4 ", "missing ")
        self.assertIsNone(parse_game(gap, self.raw)["candidate_turns"])
        # Quiet logs have no trace.
        self.assertIsNone(parse_game(self.clean, self.raw)["candidate_turns"])

    def test_confidence_intervals_include_sampling_uncertainty(self):
        self.assertIsNone(wilson(0, 0))
        self.assertGreater(wilson(1, 10)[1], 0.4)
        self.assertLess(wilson(25, 50)[0], 0.4)
        self.assertGreater(wilson(25, 50)[1], 0.6)

    def test_observed_commander_cast_uses_own_turns_not_global_counter(self):
        text = (FIXTURES / "first-cast.log").read_text()
        result = commander_cast(text, 1, "Jodah, Archmage Eternal")
        self.assertEqual(result["candidate_turns_started"], 5)
        self.assertEqual(result["forge_global_turn"], 20)
        self.assertFalse(commander_cast("", 1, "Jodah, Archmage Eternal")["available"])
        self.assertFalse(
            commander_cast(text.replace("Turn: Turn 2 ", "missing "), 1, "Jodah, Archmage Eternal")[
                "available"
            ]
        )

    def test_summary_keeps_failures_in_completion_denominator_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = []
            for i, text in enumerate((self.clean, self.clock)):
                path = Path(temp) / f"{i}.log"
                path.write_text(text, encoding="utf-8")
                logs.append(
                    {
                        **self.raw,
                        "index": i + 1,
                        "seed": i,
                        "path": str(path),
                        "log_sha256": file_hash(path),
                    }
                )
            manifest = {
                "raw_logs": logs,
                "requested_games": 2,
                "pod_fingerprint": "pod",
                "installation": {"release": "2.0.15", "jar_sha256": "jar"},
                "support": [],
                "timeout_seconds": 45,
                "startup_grace_seconds": 30,
                "quiet": True,
                "seat_policy": "rotate",
                "seed": 0,
            }
            result = summarize(manifest)
            self.assertEqual(result["completed_games"], 1)
            self.assertEqual(result["completion_rate"], 0.5)
            self.assertEqual(result["timeout_rate"], 0.5)
            self.assertEqual(result["win_rate"], 0)
            self.assertFalse(result["win_turn"]["available"])
            self.assertIsNone(result["win_turn_distribution"])
            self.assertFalse(result["tier2"]["opening_lands"]["available"])
            Path(logs[0]["path"]).write_text("modified")
            self.assertEqual(summarize(manifest)["status_counts"]["invalid_log"], 1)
