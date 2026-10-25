"""Conservative metrics for inspected Forge 2.0.15 output, not game telemetry."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .forge_config import file_hash

RESULT = re.compile(
    r"^Game Result: Game (\d+) ended in (?:(\d+) ms\. (Ai\(([1-4])\)-.+) has won!"
    r"|a Draw! Took (\d+) ms\.)$",
    re.M,
)
OUTCOME = re.compile(r"^Game Outcome: (Ai\([1-4]\)-.+) has won because .+$", re.M)
TURNS = re.compile(r"^Game Outcome: Turn (\d+)$", re.M)
TURN_START = re.compile(r"^Turn: Turn (\d+) \(Ai\(([1-4])\)-.+\)$")
EXCEPTION = re.compile(r"^(?:java\.[\w.]+(?:Exception|Error)|Exception in thread)", re.M)
WARNING_PATTERNS = {
    "target_selection_failure": "Couldn't add to stack, failed to target",
    "fight_ai_warning": "FightAi.canFightAi",
    "default_choice_logic": "Default player choice logic.",
    "default_confirmation_logic": "implementation of confirmAction is used",
}
TIER2 = {
    name: {"available": False, "reason": "Not established by inspected Forge logs; not inferred."}
    for name in (
        "commander_first_cast",
        "opening_lands",
        "lands_by_turn4",
        "screw_flood",
        "myojin_divinity_proliferate_by_turn6",
        "draw_by_turn5",
        "ramp_by_turn5",
    )
}


def wilson(wins: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if not 0 < confidence < 1 or not 0 <= wins <= total:
        raise ValueError("Invalid binomial counts or confidence")
    if total == 0:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, z2 = wins / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * ((p * (1 - p) + z2 / (4 * total)) / total) ** 0.5 / (1 + z2 / total)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def parse_game(text: str, raw: dict[str, Any], *, release: str = "2.0.15") -> dict[str, Any]:
    """One isolated JVM/game only. Clock expiry overrides Forge's false winner."""
    result: dict[str, Any] = {
        "status": "unparsed",
        "winner": None,
        "winner_seat": None,
        "candidate_won": False,
        "forge_turn": None,
        "reported_winner": None,
        "runtime_warnings": {n: text.count(p) for n, p in WARNING_PATTERNS.items() if p in text},
    }
    matches = list(RESULT.finditer(text))
    if len(matches) == 1:
        result["reported_winner"] = matches[0][3]
    if raw.get("cancelled"):
        result["status"] = "cancelled"
    elif raw.get("timed_out") or "Stopping slow match as draw" in text:
        result["status"] = "timeout"
    elif raw.get("log_limit_exceeded"):
        result["status"] = "log_limit"
    elif raw.get("launch_error") or raw.get("returncode") != 0 or EXCEPTION.search(text):
        result["status"] = "engine_error"
    elif release != "2.0.15":
        result["status"] = "unsupported_release"
    elif len(matches) == 1 and raw.get("candidate_seat") in (1, 2, 3, 4):
        m = matches[0]
        winners = OUTCOME.findall(text)
        turns = TURNS.findall(text)
        if len(turns) == 1:
            result["forge_turn"] = int(turns[0])
        if m[3] is None and not winners:
            result["status"] = "draw"
        elif winners == [m[3]]:
            result.update(
                status="completed",
                winner=m[3],
                winner_seat=int(m[4]),
                candidate_won=int(m[4]) == raw["candidate_seat"],
            )
        else:
            result["status"] = "inconsistent_result"
    return result


def commander_cast(text: str, seat: int, commander: str) -> dict[str, Any]:
    """Observe an exact cast event, not resolution or a cast from the command zone."""
    turns, own_turns, current = [], 0, None
    cast = re.compile(
        rf"^Add To Stack: Ai\({seat}\)-.+ cast {re.escape(commander)}(?: targeting .+)?$"
    )
    for line in text.splitlines():
        start = TURN_START.match(line)
        if start:
            current = int(start[1])
            turns.append(current)
            own_turns += int(start[2]) == seat
        elif cast.match(line):
            if turns != list(range(1, (current or 0) + 1)):
                return {"available": False, "reason": "Incomplete turn trace"}
            return {
                "available": True,
                "forge_global_turn": current,
                "candidate_turns_started": own_turns,
                "not_cast": False,
            }
    if not turns or turns != list(range(1, turns[-1] + 1)):
        return {"available": False, "reason": "No complete detailed turn trace"}
    return {
        "available": True,
        "forge_global_turn": None,
        "candidate_turns_started": None,
        "not_cast": True,
    }


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    games = []
    release = manifest.get("installation", {}).get("release")
    for raw in manifest["raw_logs"]:
        path = Path(raw["path"])
        # A mutated, missing or oversized log is not evidence of a completed game.
        if (
            not path.is_file()
            or path.stat().st_size > 16 * 1024 * 1024
            or file_hash(path) != raw.get("log_sha256")
        ):
            game = {"status": "invalid_log", "candidate_won": False}
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            game = parse_game(text, raw, release=release)
            commander = manifest.get("candidate", {}).get("commander")
            if commander and game["status"] in {"completed", "draw"}:
                game["commander_first_cast"] = commander_cast(
                    text, raw["candidate_seat"], commander
                )
        games.append(
            {
                **game,
                "index": raw["index"],
                "seed": raw["seed"],
                "candidate_seat": raw["candidate_seat"],
                "path": raw["path"],
            }
        )
    requested = manifest["requested_games"]
    completed = [g for g in games if g["status"] in {"completed", "draw"}]
    wins = [g for g in completed if g["candidate_won"]]
    statuses = dict(Counter(g["status"] for g in games))
    context = {
        "pod": manifest["pod_fingerprint"],
        "jar": manifest["installation"]["jar_sha256"],
        "scripts": [s.get("scripts_fingerprint") for s in manifest["support"]],
        "timeout": manifest["timeout_seconds"],
        "startup_grace": manifest["startup_grace_seconds"],
        "quiet": manifest["quiet"],
        "seat_policy": manifest["seat_policy"],
        "seed": manifest["seed"],
        "requested_games": requested,
        "oracle_metadata": manifest.get("oracle_metadata"),
    }
    histogram = Counter(g["forge_turn"] for g in wins if g.get("forge_turn") is not None)
    tier2 = {name: dict(value) for name, value in TIER2.items()}
    casts = [
        g["commander_first_cast"]
        for g in completed
        if g.get("commander_first_cast", {}).get("available")
    ]
    if casts:
        tier2["commander_first_cast"] = {
            "available": True,
            "observed_games": len(casts),
            "not_cast_games": sum(c["not_cast"] for c in casts),
            "candidate_turn_distribution": dict(
                Counter(str(c["candidate_turns_started"]) for c in casts if not c["not_cast"])
            ),
            "scope": "Exact cast events, not resolution or command-zone provenance; "
            "candidate's own turn starts, including extra turns.",
        }
    return {
        "version": 1,
        "parser": "forge-2.0.15-observed-v1",
        "context": context,
        "context_fingerprint": hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest(),
        "requested_games": requested,
        "observed_games": len(games),
        "completed_games": len(completed),
        "candidate_wins": len(wins),
        "draws": sum(g["status"] == "draw" for g in games),
        "win_rate": len(wins) / len(completed) if completed else None,
        "win_rate_ci95": wilson(len(wins), len(completed)),
        "wins_per_requested_game": len(wins) / requested,
        "completion_rate": len(completed) / requested,
        "timeout_rate": statuses.get("timeout", 0) / requested,
        "status_counts": statuses,
        "support": [
            {k: s.get(k) for k in ("supported", "missing", "known_issues", "ai_warnings")}
            for s in manifest["support"]
        ],
        "win_turn_distribution": {str(t): n for t, n in sorted(histogram.items())} or None,
        "turn_unit": "Forge Game Outcome turn counter, not inferred multiplayer rounds",
        "tier2": tier2,
        "games": games,
        "limitations": [
            "Clock limits and engine failures are not deck losses or valid wins.",
            "Completed-only win rate may be biased when games fail; inspect completion separately.",
            "AI warnings are simulation concerns, not proof a card is broken.",
            "Politics is evaluated primarily by card-package/rules checks; AI cannot negotiate.",
        ],
    }
