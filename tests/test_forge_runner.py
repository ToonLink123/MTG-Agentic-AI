import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mtg_agentic.forge_config import ForgeInstallation
from mtg_agentic.forge_runner import ForgeSimulator
from mtg_agentic.oracle import OracleStore
from tests.pipeline_fixtures import JODAH, records


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(records())
        self.deck = {"commander": JODAH["name"], "cards": {"Plains": 99}, "bracket": 3}
        self.opponents = []
        for i in range(3):
            path = self.path / f"precon{i}.dck"
            path.write_text(
                f"[metadata]\nName=Precon{i}\n[Commander]\n1 {JODAH['name']}\n[Main]\n99 Plains\n"
            )
            self.opponents.append(path)
        self.install = ForgeInstallation(
            self.path, self.path / "test.jar", sys.executable, self.path / "decks", "test", "2.0.15"
        )
        self.support = Mock()
        self.support.check.return_value = {
            "supported": True,
            "missing": [],
            "known_issues": [],
            "ai_warnings": [],
            "scripts_fingerprint": "test",
        }
        self.runner = ForgeSimulator(self.install, self.store, self.support, self.path / "logs")

    def test_raw_run_captures_utf8_and_never_uses_a_shell(self):
        path = self.path / "raw.log"
        report = self.runner._run(
            [sys.executable, "-u", "-c", "print('raw subprocess smoke')"], path
        )
        self.assertEqual(report["returncode"], 0)
        self.assertEqual(path.read_text().strip(), "raw subprocess smoke")
        self.assertFalse(report["timed_out"])

    def test_hung_process_is_hard_killed(self):
        self.runner.timeout = 0.2
        self.runner.startup_grace = 0
        report = self.runner._run(
            [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"],
            self.path / "timeout.log",
        )
        self.assertTrue(report["timed_out"])
        self.assertNotEqual(report["returncode"], 0)
        self.assertLess(report["elapsed_seconds"], 10)

    @unittest.skipUnless(os.name == "nt", "Windows process-tree behavior")
    def test_hard_timeout_also_kills_spawned_child(self):
        import ctypes

        self.runner.timeout = 0.5
        self.runner.startup_grace = 0
        child_program = "import time; time.sleep(30)"
        parent_program = (
            "import subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{child_program!r}]); "
            "print(p.pid, flush=True); time.sleep(30)"
        )
        path = self.path / "tree.log"
        report = self.runner._run([sys.executable, "-u", "-c", parent_program], path)
        self.assertTrue(report["timed_out"])
        child = int(path.read_text().strip())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, child)
        if handle:
            code = ctypes.c_ulong()
            try:
                self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                if code.value == 259:  # STILL_ACTIVE; clean up only the child created above.
                    subprocess.run(
                        ["taskkill.exe", "/PID", str(child), "/T", "/F"],
                        capture_output=True,
                        timeout=15,
                        check=False,
                    )
                self.assertNotEqual(code.value, 259)
            finally:
                kernel.CloseHandle(handle)

    def test_nonfinite_timeouts_are_rejected(self):
        for kwargs in ({"timeout": float("nan")}, {"startup_grace": float("inf")}):
            with self.assertRaises(ValueError):
                ForgeSimulator(self.install, self.store, self.support, self.path / "logs", **kwargs)

    def test_explicit_cancel_stops_current_process(self):
        timer = threading.Timer(0.2, self.runner.cancel)
        timer.start()
        self.addCleanup(timer.cancel)
        report = self.runner._run(
            [sys.executable, "-c", "import time; time.sleep(30)"], self.path / "cancel.log"
        )
        self.assertTrue(report["cancelled"])

    def test_log_budget_kills_noisy_process(self):
        self.runner.max_log_bytes = 1024
        report = self.runner._run(
            [sys.executable, "-u", "-c", "import time; print('x'*4096); time.sleep(30)"],
            self.path / "noisy.log",
        )
        self.assertTrue(report["log_limit_exceeded"])

    def test_preflight_reports_unsupported_cards_without_starting_forge(self):
        self.support.check.return_value = {"supported": False, "missing": ["Spell 0"]}
        with patch.object(self.runner, "_run") as process:
            report = self.runner.simulate(self.deck, self.opponents, 2)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertEqual(len(report["support"]), 4)
        process.assert_not_called()
        self.assertTrue((Path(report["artifact_dir"]) / "raw-manifest.json").exists())

    def test_seeds_seats_and_four_decks_are_recorded_for_each_game(self):
        with patch.object(
            self.runner,
            "_run",
            return_value={
                "cancelled": False,
                "returncode": 0,
                "timed_out": False,
            },
        ) as process:
            report = self.runner.simulate(self.deck, self.opponents, 4, seed=20)
        self.assertEqual([g["seed"] for g in report["raw_logs"]], [20, 21, 22, 23])
        self.assertEqual([g["candidate_seat"] for g in report["raw_logs"]], [1, 4, 3, 2])
        self.assertEqual(process.call_count, 4)
        command = process.call_args.args[0]
        # Full logs by default: win turns and first casts are read from turn events.
        self.assertNotIn("-q", command)
        deck_start = command.index("-d") + 1
        self.assertEqual(len(command[deck_start : command.index("-n")]), 4)
        # Decks are exported beside the run's logs, never into the user's Forge folders.
        decks = Path(command[command.index("-D") + 1])
        self.assertEqual(decks, Path(report["artifact_dir"]) / "decks")
        self.assertFalse(self.install.decks_dir.exists())
        saved = json.loads((Path(report["artifact_dir"]) / "raw-manifest.json").read_text())
        self.assertEqual(saved["requested_games"], 4)

    def test_parallel_workers_keep_game_order_and_quiet_is_opt_in(self):
        runner = ForgeSimulator(
            self.install, self.store, self.support, self.path / "logs", workers=3
        )
        seen = []

        def run(command, path):
            seen.append(command[command.index("-s") + 1])
            return {"cancelled": False, "returncode": 0, "timed_out": False}

        with patch.object(runner, "_run", side_effect=run):
            report = runner.simulate(self.deck, self.opponents, 7, seed=5, quiet=True)
        self.assertEqual(report["status"], "finished")
        self.assertEqual([g["index"] for g in report["raw_logs"]], list(range(1, 8)))
        self.assertEqual(sorted(seen, key=int), [str(s) for s in range(5, 12)])
        self.assertEqual(report["workers"], 3)
        with self.assertRaises(ValueError):
            ForgeSimulator(self.install, self.store, self.support, self.path / "logs", workers=0)

    def test_cancel_during_a_parallel_run_skips_unstarted_games(self):
        runner = ForgeSimulator(
            self.install, self.store, self.support, self.path / "logs", workers=2
        )

        def run(command, path):
            runner.cancel()
            return {"cancelled": True, "returncode": None, "timed_out": False}

        with patch.object(runner, "_run", side_effect=run) as process:
            report = runner.simulate(self.deck, self.opponents, 10)
        self.assertEqual(report["status"], "cancelled")
        self.assertLessEqual(process.call_count, 2)

    def test_invalid_game_count_and_wrong_opponent_count_fail(self):
        for n in (0, True, 1001):
            with self.assertRaises(ValueError):
                self.runner.simulate(self.deck, self.opponents, n)
        with self.assertRaises(ValueError):
            self.runner.simulate(self.deck, [], 1)
