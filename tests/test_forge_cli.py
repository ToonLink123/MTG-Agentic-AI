import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtg_agentic.__main__ import main
from mtg_agentic.forge import ForgeError
from mtg_agentic.forge_config import file_hash
from mtg_agentic.oracle import OracleStore
from tests.pipeline_fixtures import JODAH, make_pool, records


class ForgeCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)

    @patch("mtg_agentic.__main__.run_forge_command")
    def test_forge_command_needs_no_chat_model_and_preserves_failure_code(self, run):
        run.return_value = ({"status": "preflight_failed", "support": []}, 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "forge-sim",
                    str(self.path / "deck.json"),
                    "--forge-home",
                    str(self.path),
                    "--pod",
                    str(self.path / "pod.json"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "preflight_failed")

    @patch("mtg_agentic.__main__.run_forge_command")
    def test_raw_manifest_output_never_overwrites(self, run):
        path = self.path / "existing.json"
        path.write_text("keep")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "forge-support",
                    str(self.path / "deck.json"),
                    "--forge-home",
                    str(self.path),
                    "--output",
                    str(path),
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(path.read_text(), "keep")
        run.assert_not_called()

    @patch("mtg_agentic.__main__.run_forge_command", side_effect=ForgeError("Unsupported card"))
    def test_forge_failures_are_machine_readable(self, run):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                ["forge-export", str(self.path / "deck.json"), "--forge-home", str(self.path)]
            )
        self.assertEqual(code, 2)
        self.assertIn("Unsupported card", output.getvalue())

    def test_exit_zero_from_java_is_not_enough_when_clock_reported_false_win(self):
        log = Path(__file__).parent / "fixtures_forge" / "clock-false-win.log"
        manifest = {
            "status": "finished",
            "artifact_dir": str(self.path),
            "requested_games": 1,
            "pod_fingerprint": "pod",
            "seed": 0,
            "installation": {"release": "2.0.15", "jar_sha256": "jar"},
            "support": [],
            "timeout_seconds": 45,
            "startup_grace_seconds": 30,
            "quiet": True,
            "seat_policy": "rotate",
            "raw_logs": [
                {
                    "path": str(log),
                    "log_sha256": file_hash(log),
                    "index": 1,
                    "seed": 0,
                    "candidate_seat": 1,
                    "returncode": 0,
                }
            ],
        }
        deck = self.path / "deck.json"
        deck.write_text("{}")
        install = Mock()
        output = io.StringIO()
        with (
            patch("mtg_agentic.forge_cli.load_forge", return_value=(install, Mock())),
            patch("mtg_agentic.forge_cli.OracleStore"),
            patch("mtg_agentic.forge_cli.OpponentPod.load"),
            patch("mtg_agentic.forge_cli.ForgeSimulator") as runner,
            contextlib.redirect_stdout(output),
        ):
            runner.return_value.simulate.return_value = manifest
            code = main(
                [
                    "forge-sim",
                    str(deck),
                    "--forge-home",
                    str(self.path),
                    "--pod",
                    str(self.path / "pod.json"),
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["metrics"]["completed_games"], 0)
        self.assertEqual(json.loads((self.path / "metrics.json").read_text())["timeout_rate"], 1)

    def test_revision_cli_passes_mandatory_reviewer_and_pinned_pods(self):
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards(records())
            pool = make_pool(store).to_dict()
        pool_path, deck_path = self.path / "pool.json", self.path / "deck.json"
        pool_path.write_text(json.dumps(pool))
        deck_path.write_text(json.dumps({"commander": JODAH["name"], "cards": {"Plains": 99}}))
        reviewer, simulator, pod = Mock(), Mock(), Mock()
        with (
            patch("mtg_agentic.__main__._make_model"),
            patch("mtg_agentic.__main__._reviewer", return_value=reviewer),
            patch("mtg_agentic.__main__._simulation", return_value=(simulator, [pod])),
            patch("mtg_agentic.__main__.revise_deck") as revise,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            revise.return_value = {"stop_reason": "plateau", "gate_invariant_passed": True}
            code = main(
                [
                    "--data-dir",
                    str(self.path),
                    "revise",
                    str(deck_path),
                    "--pool",
                    str(pool_path),
                    "--forge-home",
                    str(self.path),
                    "--pod",
                    str(self.path / "pod.json"),
                    "--iterations",
                    "1",
                    "--plateau",
                    "1",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIs(revise.call_args.args[4], reviewer)
        self.assertIs(revise.call_args.args[5], simulator)
        self.assertEqual(revise.call_args.args[6], [pod])
        self.assertEqual(revise.call_args.kwargs["games"], 50)
