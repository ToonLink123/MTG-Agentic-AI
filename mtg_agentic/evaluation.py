"""Four fixed end-to-end cases, persisted artifacts and honest repeated-run metrics."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from .oracle import OracleStore
from .provenance import repository_state
from .review import DeckReviewer
from .revision import gate_invariant
from .sources import utc_now
from .validator import validate_deck

CASES = (
    {
        "id": "jodah-myojin",
        "commander": "Jodah, Archmage Eternal",
        "bracket": 3,
        "request": "Build Bracket 3 Jodah, Archmage Eternal Myojin tribal with at least eight "
        "distinct Myojin, divinity counters and proliferate. Cast Myojin from hand with Jodah, "
        "preserve counters to proliferate, use fair midrange ramp, and avoid early combos.",
        "theme": r"^Myojin ",
        "field": "name",
        "minimum": 8,
    },
    {
        "id": "poison",
        "commander": "Atraxa, Praetors' Voice",
        "bracket": 3,
        "request": "Build a Bracket 3 Atraxa poison deck with at least eight poison, toxic, "
        "infect or proliferate cards, including at least three ways to give an opponent their "
        "first poison counter. No mass land denial, extra-turn loops or early two-card combos.",
        "theme": r"\b(?:poison|toxic|infect|proliferate)\b",
        "field": "oracle_text",
        "minimum": 8,
    },
    {
        "id": "politics",
        "commander": "Queen Marchesa",
        "bracket": 3,
        "request": "Build Bracket 3 Queen Marchesa table politics with at least eight cards "
        "supporting monarch, goad, voting or multiplayer offers. Use deals and incremental "
        "combat, not mass land denial, extra-turn loops or early two-card combos.",
        "theme": r"\b(?:monarch|goad|vote|votes|tempting offer|join forces|council's dilemma|"
        r"will of the council)\b",
        "field": "oracle_text",
        "minimum": 8,
    },
    {
        "id": "bracket-two",
        "commander": "Ruxa, Patient Professor",
        "bracket": 2,
        "request": "Deliberately build Bracket 2 Ruxa, Patient Professor: at least eight "
        "creatures with no rules text, incremental telegraphed combat, no Game Changers, "
        "no intentional two-card infinite combos, no mass land denial or extra-turn chains. "
        "Prioritize straightforward social games with at least eight turns to develop.",
        "theme": "vanilla",
        "field": "oracle_text",
        "minimum": 8,
    },
)


def score_case(case: dict[str, Any], deck: dict[str, Any], store: OracleStore) -> dict[str, Any]:
    cards, commander = deck["cards"], deck["commander"]
    review = deck.get("review", {})
    bracket = review.get("bracket", {})
    # Independent legality check; never trust the generated artifact's valid flag.
    gc = bracket.get("game_changer_list", {}).get("names")
    validation = validate_deck(cards, commander, store, bracket=case["bracket"], game_changers=gc)
    from .playability import spec_for_deck, validate_playability

    playability = validate_playability(cards, spec_for_deck(deck, store), store)
    themes = []
    poison_sources = []
    for n, q in cards.items():
        card = store.get_card(n)
        if card is None or type(q) is not int or q <= 0:
            continue
        if case["theme"] == "vanilla":
            matches = "Creature" in card["type_line"] and not card["oracle_text"].strip()
        else:
            matches = re.search(case["theme"], card[case["field"]], re.I) is not None
        if matches:
            themes.append(card["name"])
        if re.search(
            r"\b(?:toxic|infect)\b|(?:gets?|give|gain).*poison counter", card["oracle_text"], re.I
        ):
            poison_sources.append(card["name"])
    unique_themes = sorted(set(themes))
    theme_pass = len(unique_themes) >= case["minimum"]
    if case["id"] == "poison":
        theme_pass = theme_pass and len(set(poison_sources)) >= 3
    bracket_correct = (
        deck.get("bracket") == case["bracket"]
        and bracket.get("target_bracket") == case["bracket"]
        and bracket.get("verdict") == "fits"
        and bracket.get("assessment", {}).get("estimated_bracket") == case["bracket"]
    )
    commander_correct = commander == case["commander"]
    return {
        "id": case["id"],
        "validator_pass": validation["valid"],
        "commander_correct": commander_correct,
        "bracket_correct": bracket_correct,
        "theme_cards": unique_themes,
        "theme_count": len(unique_themes),
        "theme_minimum": case["minimum"],
        "theme_coverage": min(1, len(unique_themes) / case["minimum"]),
        "theme_pass": theme_pass,
        "poison_sources": sorted(set(poison_sources)),
        "review_pass": review.get("status") == "passed",
        "validation": validation,
        "playability": playability,
        "passed": bool(
            validation["valid"]
            and playability["valid"]
            and commander_correct
            and bracket_correct
            and theme_pass
            and review.get("status") == "passed"
        ),
    }


def run_evaluation(
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    store: OracleStore,
    output_dir: str | Path,
    *,
    label: str = "",
    configuration: dict[str, Any] | None = None,
    simulation_runner: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    reviewer: DeckReviewer | None = None,
) -> dict[str, Any]:
    if simulation_runner is not None and reviewer is None:
        raise ValueError("Simulation evaluation requires a fresh safety reviewer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = root / run_id
    run_dir.mkdir()
    results = []
    for case in CASES:
        try:
            deck = runner(dict(case))
            if simulation_runner is not None:
                deck["review"] = reviewer.check(deck["cards"], deck["commander"], case["bracket"])
            result = score_case(case, deck, store)
            revision = deck.get("revision")
            result["revision_gate_invariant"] = gate_invariant(revision) if revision else None
            if result["revision_gate_invariant"] is False:
                result["passed"] = False
            with (run_dir / (case["id"] + ".json")).open("x", encoding="utf-8") as f:
                json.dump(deck, f, ensure_ascii=True, indent=2)
            result["simulation"] = {"status": "not_requested"}
            if simulation_runner is not None:
                result["simulation"] = {"status": "blocked_by_gates", "pods": []}
                if result["passed"]:
                    try:
                        pods = simulation_runner(deck)
                        completed = bool(pods) and all(
                            p["completed_games"] == p["requested_games"] for p in pods
                        )
                        result["simulation"] = {
                            "status": "completed" if completed else "incomplete",
                            "pods": [{k: v for k, v in p.items() if k != "games"} for p in pods],
                        }
                    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                        result["simulation"] = {"status": "error", "error": type(exc).__name__}
                result["passed"] = (
                    result["passed"] and result["simulation"]["status"] == "completed"
                )
        except (ValueError, RuntimeError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
            # Keep failed cases in denominators. Avoid persisting exception bodies/secrets.
            result = {
                "id": case["id"],
                "passed": False,
                "validator_pass": False,
                "bracket_correct": False,
                "theme_coverage": 0,
                "theme_pass": False,
                "review_pass": False,
                "error": type(exc).__name__,
            }
        results.append(result)
    metrics = {
        "validator_pass_rate": sum(r["validator_pass"] for r in results) / len(CASES),
        "bracket_accuracy": sum(r["bracket_correct"] for r in results) / len(CASES),
        "theme_coverage": sum(r["theme_coverage"] for r in results) / len(CASES),
        "theme_pass_rate": sum(r["theme_pass"] for r in results) / len(CASES),
        "review_pass_rate": sum(r["review_pass"] for r in results) / len(CASES),
        "case_pass_rate": sum(r["passed"] for r in results) / len(CASES),
        "simulation_case_completion_rate": (
            sum(r.get("simulation", {}).get("status") == "completed" for r in results) / len(CASES)
            if simulation_runner is not None
            else None
        ),
        "revision_gate_violations": sum(r.get("revision_gate_invariant") is False for r in results),
    }
    report = {
        "version": 1,
        "suite": "commander-four-v1",
        "id": run_id,
        "label": label,
        "configuration": configuration or {},
        "created_at": utc_now(),
        "oracle_metadata": store.metadata,
        "repository": repository_state(),
        "cases": results,
        "metrics": metrics,
        "passed": all(r["passed"] for r in results),
        "artifact_dir": str(run_dir),
        "forge": "requested" if simulation_runner is not None else "not_requested",
    }
    with (run_dir / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
    with closing(sqlite3.connect(root / "evaluations.sqlite3")) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, report TEXT)"
        )
        db.execute(
            "INSERT INTO runs VALUES (?,?,?)", (run_id, report["created_at"], json.dumps(report))
        )
    return report


def evaluation_history(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "evaluations.sqlite3"
    if not path.exists():
        return {"runs": [], "stability_gate": False, "forge": "not_requested"}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        reports = [
            json.loads(row[0])
            for row in db.execute("SELECT report FROM runs ORDER BY created_at, id")
        ]
    # Explicit local engineering stability gate, not Wizards policy.
    recent = reports[-3:]
    stable = (
        len(recent) == 3
        and all(r["passed"] for r in recent)
        and all(r.get("configuration", {}) == recent[0].get("configuration", {}) for r in recent)
        and all(
            r["oracle_metadata"].get("updated_at") == recent[0]["oracle_metadata"].get("updated_at")
            for r in recent
        )
    )
    return {
        "runs": [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "label": r["label"],
                "configuration": r.get("configuration", {}),
                "metrics": r["metrics"],
                "passed": r["passed"],
                "repository": r.get("repository"),
                "forge": r.get("forge", "not_requested"),
                "simulations": [
                    {"id": c["id"], **c.get("simulation", {"status": "not_requested"})}
                    for c in r["cases"]
                ],
            }
            for r in reports
        ],
        "stability_gate": stable,
        "required_consecutive_runs": 3,
        "forge": reports[-1].get("forge", "not_requested") if reports else "not_requested",
    }
