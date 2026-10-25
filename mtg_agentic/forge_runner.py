"""Raw-log-only Forge subprocess plumbing with hard per-game process-tree limits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .forge import ForgeError, export_deck
from .forge_config import ForgeInstallation, OpponentPod, file_hash, read_dck
from .forge_support import ForgeSupport
from .oracle import OracleStore
from .sources import utc_now
from .validator import validate_deck


def kill_process_tree(process: subprocess.Popen) -> None:
    """Only kill the process we created and its descendants."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


class ForgeSimulator:
    """One JVM per game: a hung game cannot poison the rest of a batch.

    Startup costs more than batch mode, but seeds, timeouts, failures and log
    attribution are isolated. No game-log interpretation happens in this layer.
    """

    def __init__(
        self,
        installation: ForgeInstallation,
        store: OracleStore,
        support: ForgeSupport,
        log_dir: str | Path,
        *,
        timeout: float = 120,
        startup_grace: float = 30,
        max_log_bytes: int = 16 * 1024 * 1024,
    ):
        if (
            not math.isfinite(timeout)
            or not math.isfinite(startup_grace)
            or timeout <= 0
            or startup_grace < 0
            or max_log_bytes < 1024
        ):
            raise ValueError(
                "Positive simulation timeout/log budget and nonnegative grace required"
            )
        self.installation, self.store, self.support = installation, store, support
        self.log_dir = Path(log_dir).resolve()
        self.timeout, self.startup_grace, self.max_log_bytes = timeout, startup_grace, max_log_bytes
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Stop the active JVM/process tree and do not start further games."""
        self._cancel.set()

    def simulate(
        self,
        deck: dict[str, Any],
        opponents: OpponentPod | list[str | Path],
        n_games: int,
        *,
        seed: int = 0,
        quiet: bool = True,
    ) -> dict[str, Any]:
        if type(n_games) is not int or not 1 <= n_games <= 1000:
            raise ValueError("n_games must be from 1 to 1000")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1 - n_games:
            raise ValueError("Seed must fit a nonnegative Java long, including all game seeds")
        if not self._lock.acquire(blocking=False):
            raise ForgeError("This simulator already has an active run")
        self._cancel.clear()
        try:
            return self._simulate(deck, opponents, n_games, seed=seed, quiet=quiet)
        finally:
            self._lock.release()

    def _simulate(self, deck, opponents, n_games, *, seed, quiet):
        if isinstance(opponents, OpponentPod):
            pod = opponents
        else:
            if len(opponents) != 3:
                raise ValueError("Commander simulations require exactly three opponents")
            paths = tuple(Path(p).resolve() for p in opponents)
            pod = OpponentPod(
                "caller-supplied",
                paths,
                tuple(file_hash(p) for p in paths),
                self.installation.release,
            )
        if pod.expected_release != self.installation.release:
            raise ForgeError("Pod release does not match this Forge installation")
        if tuple(file_hash(p) for p in pod.opponents) != pod.hashes:
            raise ForgeError("Pinned opponent files changed before the simulation")
        decks = [deck, *[read_dck(p) for p in pod.opponents]]
        support = [self.support.check(d, self.store) for d in decks]
        validations = [
            validate_deck(d["cards"], d["commander"], self.store, bracket=d.get("bracket", 3))
            for d in decks
        ]
        run_id = uuid.uuid4().hex
        directory = self.log_dir / run_id
        directory.mkdir(parents=True)
        report: dict[str, Any] = {
            "version": 1,
            "id": run_id,
            "created_at": utc_now(),
            "installation": self.installation.provenance(),
            "candidate": {
                "commander": deck["commander"],
                "deck_fingerprint": hashlib.sha256(
                    json.dumps(
                        {"commander": deck["commander"], "cards": deck["cards"]}, sort_keys=True
                    ).encode()
                ).hexdigest(),
            },
            "oracle_metadata": self.store.metadata,
            "pod": pod.provenance(),
            "pod_fingerprint": pod.fingerprint,
            "requested_games": n_games,
            "seed": seed,
            "seat_policy": "rotate all four seats; candidate starts in seat 1",
            "quiet": quiet,
            "timeout_seconds": self.timeout,
            "startup_grace_seconds": self.startup_grace,
            "raw_logs": [],
            "support": support,
            "validation": validations,
            "status": "preflight_failed",
        }
        if all(v["valid"] for v in validations) and all(s["supported"] for s in support):
            exports = []
            for i, d in enumerate(decks):
                path = self.installation.decks_dir / f"agentic-{run_id}-{i}.dck"
                exports.append(
                    export_deck(
                        d, self.store, path, name="Agentic Candidate" if i == 0 else f"Opponent {i}"
                    )
                )
            report["exports"] = exports
            report["status"] = "finished"
            for i in range(n_games):
                if self._cancel.is_set():
                    report["status"] = "cancelled"
                    break
                rotation = i % 4
                ordered = exports[rotation:] + exports[:rotation]
                seat = ((4 - rotation) % 4) + 1
                path = directory / f"game-{i + 1:04d}.log"
                command = self.installation.command() + [
                    "-f",
                    "Commander",
                    "-D",
                    str(self.installation.decks_dir),
                    "-d",
                    *[Path(e["path"]).name for e in ordered],
                    "-n",
                    "1",
                    "-c",
                    str(math.ceil(self.timeout)),
                    "-s",
                    str(seed + i),
                ]
                if quiet:
                    command.append("-q")
                raw = dict(self._run(command, path))
                raw.update(index=i + 1, seed=seed + i, candidate_seat=seat, path=str(path))
                report["raw_logs"].append(raw)
                if raw["cancelled"]:
                    report["status"] = "cancelled"
                    break
        with (directory / "raw-manifest.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=True, indent=2)
        report["artifact_dir"] = str(directory)
        return report

    def _run(self, command: list[str], path: Path) -> dict[str, Any]:
        started = time.monotonic()
        timed_out = cancelled = too_large = False
        process = None
        returncode = None
        launch_error = None
        with path.open("xb") as output:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.installation.home,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                deadline = started + self.startup_grace + self.timeout
                while process.poll() is None:
                    cancelled = self._cancel.is_set()
                    timed_out = time.monotonic() >= deadline
                    too_large = path.stat().st_size > self.max_log_bytes
                    if cancelled or timed_out or too_large:
                        kill_process_tree(process)
                        break
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                returncode = process.returncode
            except OSError as exc:
                launch_error = type(exc).__name__
            finally:
                if process is not None and process.poll() is None:
                    kill_process_tree(process)
        return {
            "returncode": returncode,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "log_limit_exceeded": too_large,
            "launch_error": launch_error,
            "elapsed_seconds": time.monotonic() - started,
            "command": command,
            "log_sha256": file_hash(path),
        }


def simulate(
    deck: dict[str, Any],
    opponents: OpponentPod | list[str | Path],
    n_games: int,
    *,
    simulator: ForgeSimulator,
    seed: int = 0,
    quiet: bool = True,
) -> dict[str, Any]:
    return simulator.simulate(deck, opponents, n_games, seed=seed, quiet=quiet)
