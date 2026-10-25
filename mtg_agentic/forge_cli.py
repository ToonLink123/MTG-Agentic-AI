"""Forge CLI plumbing; raw simulations do not imply strategic quality."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from .forge import export_deck
from .forge_config import ForgeInstallation, OpponentPod
from .forge_metrics import summarize
from .forge_runner import ForgeSimulator
from .forge_support import ForgeSupport
from .oracle import OracleStore
from .schema import load_json


def register_commands(commands) -> None:
    for name in ("forge-export", "forge-support", "forge-sim"):
        parser = commands.add_parser(name)
        parser.add_argument("deck", type=Path)
        parser.add_argument("--forge-home", type=Path, required=True)
        parser.add_argument("--decks-dir", type=Path)
        parser.add_argument("--java")
        parser.add_argument("--known-issues", type=Path)
        parser.add_argument("--output", type=Path)
        if name == "forge-export":
            parser.add_argument("--deck-output", type=Path)
        if name == "forge-sim":
            parser.add_argument("--pod", type=Path, required=True)
            parser.add_argument("--games", type=int, default=10)
            parser.add_argument("--timeout", type=float, default=120)
            parser.add_argument("--startup-grace", type=float, default=30)
            parser.add_argument("--seed", type=int, default=20261025)
            parser.add_argument("--verbose-logs", action="store_true")
            parser.add_argument("--logs-dir", type=Path)


def load_forge(args: argparse.Namespace) -> tuple[ForgeInstallation, ForgeSupport]:
    installation = ForgeInstallation.inspect(
        args.forge_home, decks_dir=args.decks_dir, java=args.java
    )
    issues = {}
    if args.known_issues:
        issues = load_json(args.known_issues.read_text(encoding="utf-8-sig"))
        if not isinstance(issues, dict) or any(
            not isinstance(n, str) or not isinstance(reason, str) or not reason.strip()
            for n, reason in issues.items()
        ):
            raise ValueError(
                "Known issues must map exact card names to observed issue descriptions"
            )
    support = ForgeSupport(installation.home / "res/cardsfolder", known_issues=issues)
    return installation, support


def run_command(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    installation, support = load_forge(args)
    deck = load_json(args.deck.read_text(encoding="utf-8-sig"))
    with OracleStore(args.data_dir / "oracle.sqlite3") as store:
        if not store.ready:
            raise ValueError("Run download-oracle before exporting or simulating")
        if args.command == "forge-support":
            report = support.check(deck, store)
            return report, 0 if report["supported"] else 1
        if args.command == "forge-export":
            report = support.check(deck, store)
            if not report["supported"]:
                return {"valid": False, "support": report}, 1
            path = args.deck_output or installation.decks_dir / f"agentic-{uuid.uuid4().hex}.dck"
            exported = export_deck(deck, store, path)
            return {"valid": True, "export": exported, "support": report}, 0
        pod = OpponentPod.load(args.pod, installation)
        simulator = ForgeSimulator(
            installation,
            store,
            support,
            args.logs_dir or args.data_dir / "simulations",
            timeout=args.timeout,
            startup_grace=args.startup_grace,
        )
        try:
            report = simulator.simulate(
                deck, pod, args.games, seed=args.seed, quiet=not args.verbose_logs
            )
        except KeyboardInterrupt:
            simulator.cancel()
            raise
        report["metrics"] = summarize(report)
        with (Path(report["artifact_dir"]) / "metrics.json").open("x", encoding="utf-8") as output:
            json.dump(report["metrics"], output, indent=2)
        ok = (
            report["status"] == "finished"
            and report["metrics"]["completed_games"] == report["requested_games"]
        )
        return report, 0 if ok else 1
