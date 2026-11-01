"""Goldfish: solitaire Commander games against three idle opponents, at Monte Carlo scale.

Code reads each card's Oracle text once and compiles the deck into a small JSON description:
costs, the lands a card makes or fetches, mana rocks and dorks, and the few effects the engine
plays (cards drawn, Treasure, damage to opponents, creature tokens, upkeep draws). Everything
else a card does is not simulated; the report lists those cards as cast-only.

The Rust engine in goldfish-rs/ plays that description; setup builds it with cargo, and there
is no other engine. Every game has its own splitmix64 seed, so a seed gives the same counters
on any number of threads. The pilot is greedy, not optimal: it mulligans to two to five lands,
plays the land that lets it spend the most mana, casts mana first, then the commander, then
card draw, then the most expensive spell it can pay for, and attacks the opponent with the
lowest life with everything that can attack. Opponents never block, remove or counter
anything. The numbers measure consistency and speed, not strength.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .cards import Card, rules_text
from .mana import SYMBOL, direct_sources, front, is_land, matches_land_search, search_selectors
from .oracle import OracleStore

COLOR_BITS = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16, "C": 32}
# Basic land types in the same WUBRG order, for lands that check what you control.
TYPE_BITS = {"Plains": 1, "Island": 2, "Swamp": 4, "Mountain": 8, "Forest": 16}
POP = [bin(i).count("1") for i in range(64)]
# How a land enters; goldfish-rs reads the same codes.
UNTAPPED, TAPPED, FAST, SLOW, CHECK = range(5)
X_VALUE = 2  # An X spell is cast once it can pay X = 2.
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7}
RUST_SOURCE = Path(__file__).resolve().parent.parent / "goldfish-rs"
RUST_BINARY = RUST_SOURCE / "target" / "release" / (
    "goldfish-rs.exe" if os.name == "nt" else "goldfish-rs"
)
RUSTUP = "https://rustup.rs"
SCOPE = (
    "Solitaire games against three idle 40-life opponents: no blocks, removal or "
    "counterspells. The pilot is greedy and only some card effects are simulated."
)


# ---------------------------------------------------------------------------------------------
# Compiling Oracle text


def _mask(colors: Iterable[str]) -> int:
    return sum(COLOR_BITS[c] for c in set(colors) if c in COLOR_BITS)


def _cost(cost: str) -> tuple[list[int], int]:
    """Colored pips as color masks, cheapest-to-satisfy first, and the generic amount."""
    pips, generic, x = [], 0, False
    for symbol in SYMBOL.findall(cost):
        parts = symbol.split("/")
        if symbol.isdigit():
            generic += int(symbol)
        elif symbol in {"X", "Y", "Z"}:
            x = True
        elif "P" in parts:
            continue  # Phyrexian mana: 2 life always pays it.
        elif parts[0] == "2":
            generic += 2  # {2/W}: two generic always pays it.
        elif symbol == "S":
            generic += 1
        elif mask := _mask(parts):
            pips.append(mask)
    pips.sort(key=lambda m: (POP[m], m))
    return pips, generic + (X_VALUE if x else 0)


def _count(word: str) -> int:
    return int(word) if word.isdigit() else NUMBERS.get(word.lower(), 0)


def _one_shot(card: Card) -> list[str]:
    """Lines whose effects happen once, when the card resolves: a spell's text, or a
    permanent's enters trigger. Triggers that repeat and activated abilities are left out."""
    types = front(card, "type_line")
    spell = "Instant" in types or "Sorcery" in types
    lines = []
    for line in rules_text(front(card, "oracle_text")).split("\n"):
        low = line.lower()
        if spell:
            if not low.startswith(("whenever", "at the beginning")) and not re.match(
                r"^[^.]*\{[^}]+\}[^.]*:", line
            ):
                lines.append(line)
        elif re.match(r"when(?!ever)\b[^.]*\benters\b", low):
            lines.append(line)
    return lines


def _mana_units(card: Card, identity: int) -> list[int]:
    """What tapping a nonland permanent once adds, as one color mask per mana."""
    outputs = []
    for line in rules_text(front(card, "oracle_text")).split("\n"):
        if re.search(r"spend this mana only|activate only|could produce|\bamong\b|opponent",
                     line, re.I):
            continue
        match = re.match(r"^\{T\}(?:, Pay \d+ life)?: Add ([^.]+)\.", line) or re.match(
            r"^\{(\d+)\}, \{T\}: Add ([^.]+)\.", line
        )
        if not match:
            continue
        paid = int(match[1]) if len(match.groups()) == 2 else 0
        output = match[match.lastindex]
        any_color = re.match(r"(a|an|one|two|three) mana of any (?:one )?(color|type)", output)
        if any_color:
            mask = identity | (COLOR_BITS["C"] if any_color[2] == "type" else 0)
            units = [mask] * _count(any_color[1]) if mask else []
        else:
            symbols = SYMBOL.findall(output)
            if not symbols or any(s not in COLOR_BITS for s in symbols):
                continue
            units = [_mask(symbols)] if " or " in output else [COLOR_BITS[s] for s in symbols]
        if paid:
            # A Signet turns {1} into two colored mana: one mana more, of either color.
            units = [_mask_union(units)] * max(0, len(units) - paid)
        if units:
            outputs.append(units)
    if not outputs:
        return []
    longest = max(len(u) for u in outputs)
    if longest == 1:
        # A Talisman taps for {C} or for one of its colors: one mana of any of them.
        return [_mask_union(u[0] for u in outputs)]
    return next(u for u in outputs if len(u) == longest)


def _mask_union(masks: Iterable[int]) -> int:
    result = 0
    for mask in masks:
        result |= mask
    return result


def _enter_rule(text: str) -> tuple[int, int]:
    """How a land enters: untapped, tapped, fast, slow, or tapped unless you control a type."""
    if not re.search(r"enters (?:the battlefield )?tapped|it enters tapped", text, re.I):
        return UNTAPPED, 0
    if re.search(r"you may pay \d+ life", text, re.I):
        return UNTAPPED, 0
    if re.search(r"unless you control two or fewer other lands", text, re.I):
        return FAST, 0
    if re.search(r"unless you control two or more other lands", text, re.I):
        return SLOW, 0
    check = re.search(r"unless you control an? ((?:\w+)(?: or an? \w+)*)", text, re.I)
    if check:
        types = _mask_union(TYPE_BITS.get(t, 0) for t in re.findall(r"\w+", check[1]))
        if types:
            return CHECK, types
    return TAPPED, 0


def _land_face(card: Card) -> Card | None:
    """The land half of a card: the card itself, or a modal double-faced card's back."""
    if is_land(card):
        return card
    faces = card.get("card_faces") or []
    if card.get("layout") == "modal_dfc" and len(faces) > 1 and "Land" in faces[1].get(
        "type_line", ""
    ):
        return {"type_line": faces[1]["type_line"], "oracle_text": faces[1].get("oracle_text", "")}
    return None


def _ramp(lines: list[str]) -> tuple[int, int, bool]:
    """Lands a search puts onto the battlefield and into your hand, and whether tapped."""
    text = " ".join(lines)
    if not re.search(r"search your library", text, re.I):
        return 0, 0, False
    tapped = bool(re.search(r"onto the battlefield tapped", text, re.I))
    if re.search(r"put one onto the battlefield[^.]*and the other into your hand", text, re.I):
        return 1, 1, tapped
    two = bool(re.search(r"up to two|two basic land cards", text, re.I))
    if re.search(r"onto the battlefield", text, re.I):
        return (2 if two and re.search(r"put them onto", text, re.I) else 1), 0, tapped
    if re.search(r"into your hand", text, re.I):
        return 0, 2 if two else 1, False
    return 0, 0, False


def compile_deck(deck: dict[str, Any], store: OracleStore) -> dict[str, Any]:
    """The engine input: one piece per distinct card, the 99 as piece indices, and notes."""
    commander = store.get_card(deck["commander"])
    if commander is None:
        raise ValueError(f"Unknown commander: {deck['commander']}")
    cards: dict[str, tuple[Card, int]] = {}
    for name, quantity in deck["cards"].items():
        card = store.get_card(name)
        if card is None:
            raise ValueError(f"Unknown card: {name}")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("Deck quantities must be positive integers")
        previous = cards.get(card["name"], (card, 0))[1]
        cards[card["name"]] = (card, previous + quantity)
    identity_letters = "".join(c for c in "WUBRG" if c in commander["color_identity"]) or "C"
    identity = _mask(identity_letters) if identity_letters != "C" else 0
    ordered = [commander, *(card for card, _ in cards.values())]
    land_faces = [_land_face(card) for card in ordered]
    pieces = []
    for index, card in enumerate(ordered):
        face = land_faces[index]
        land = is_land(card)
        lines = _one_shot(card) if not land else []
        pips, generic = _cost(front(card, "mana_cost"))
        piece: dict[str, Any] = {
            "name": card["name"],
            "mv": int(card.get("cmc") or 0),
            "pips": pips,
            "generic": generic,
            "castable": not land and bool(front(card, "mana_cost")),
            "land": land,
            "back_land": face is not None and not land,
            "colors": 0, "land_types": 0, "enter": UNTAPPED, "check_types": 0,
            "fetch": [], "fetch_tapped": False,
            "creature": "Creature" in front(card, "type_line").split("—")[0],
            "power": 0, "haste": "Haste" in (card.get("keywords") or []),
            "mana": [], "ramp": [], "ramp_battlefield": 0, "ramp_hand": 0, "ramp_tapped": False,
            "treasures": 0, "draw": 0, "upkeep_draw": 0, "burn_each": 0, "burn_one": 0,
            "token_power": 0, "token_haste": False,
            "effects": [],
        }
        if face is not None:
            piece["colors"] = _mask(direct_sources(face, identity_letters))
            piece["land_types"] = _mask_union(
                TYPE_BITS.get(word, 0) for word in front(face, "type_line").split()
            )
            piece["enter"], piece["check_types"] = _enter_rule(front(face, "oracle_text"))
            piece["effects"].append("land")
        if land and (selectors := search_selectors(card)) and re.search(
            r"^([^:\n]*):[^\n]*search your library", front(card, "oracle_text"), re.I | re.M
        ):
            cost = re.search(r"^([^:\n]*):[^\n]*search your library", front(card, "oracle_text"),
                             re.I | re.M)[1]
            if all(s in {"T", "Q", "E"} for s in SYMBOL.findall(cost)) and all(
                s["supported"] for s in selectors
            ):
                piece["fetch"] = _targets(selectors, ordered, land_faces)
                piece["fetch_tapped"] = bool(
                    re.search(r"onto the battlefield tapped", front(card, "oracle_text"), re.I)
                )
                piece["effects"].append("fetch")
        if land:
            pieces.append(piece)
            continue
        power = front(card, "power") if card.get("card_faces") else card.get("power")
        if piece["creature"] and isinstance(power, str) and power.isdigit() and int(power):
            piece["power"] = int(power)
            piece["effects"].append("attacks")
        if units := _mana_units(card, identity):
            piece["mana"] = units
            piece["effects"].append("mana")
        battlefield, hand, tapped = _ramp(lines)
        selectors = search_selectors({"type_line": "", "oracle_text": "\n".join(lines)})
        if (battlefield or hand) and selectors and all(s["supported"] for s in selectors):
            piece.update(
                ramp=_targets(selectors, ordered, land_faces),
                ramp_battlefield=battlefield, ramp_hand=hand, ramp_tapped=tapped,
            )
            piece["effects"].append("ramp")
        text = " ".join(lines)
        if match := re.search(r"\bcreate (a|one|two|three|four|five) Treasure tokens?", text, re.I):
            piece["treasures"] = _count(match[1])
            piece["effects"].append("treasure")
        if match := re.search(
            r"\bdraws? (a|an|one|two|three|four|five|six|seven) (?:additional )?cards?\b", text,
            re.I,
        ):
            piece["draw"] = _count(match[1])
            piece["effects"].append("draw")
        upkeep = rules_text(front(card, "oracle_text"))
        if re.search(r"^At the beginning of your (?:upkeep|draw step)\b[^\n]*\bdraws? "
                     r"(?:a|an additional|one) cards?", upkeep, re.I | re.M):
            piece["upkeep_draw"] = 1
            piece["effects"].append("upkeep draw")
        if match := re.search(r"deals (\d+) damage to each opponent|each opponent loses (\d+) life",
                              text, re.I):
            piece["burn_each"] = int(match[1] or match[2])
            piece["effects"].append("burn")
        elif match := re.search(
            r"deals (\d+) damage to (?:any target|target player|target opponent)", text, re.I
        ):
            piece["burn_one"] = int(match[1])
            piece["effects"].append("burn")
        if match := re.search(
            r"\bcreate (a|one|two|three|four|five) (\d+)/(\d+)[^.]*?creature tokens?([^.]*)",
            text, re.I,
        ):
            piece["token_power"] = _count(match[1]) * int(match[2])
            piece["token_haste"] = "haste" in match[0].lower()
            piece["effects"].append("tokens")
        if piece["mana"] and not piece["creature"] and len(piece["mana"]) >= len(pips) + generic:
            piece["class"] = 0  # Pays for itself at once, like Sol Ring.
        elif piece["mana"] or piece["ramp"] or piece["treasures"]:
            piece["class"] = 1
        elif piece["draw"] or piece["upkeep_draw"]:
            piece["class"] = 3
        else:
            piece["class"] = 4
        if index == 0:
            piece["class"] = min(piece["class"], 2)
        pieces.append(piece)
    for piece in pieces:
        piece.setdefault("class", 4)
        piece["total"] = len(piece["pips"]) + piece["generic"]
    library = [
        index for index, (_, quantity) in enumerate(cards.values(), start=1)
        for _ in range(quantity)
    ]
    return {
        "version": 1,
        "commander": 0,
        "identity": identity,
        "library": library,
        "pieces": pieces,
    }


def _targets(selectors: list[dict[str, Any]], ordered: list[Card],
             land_faces: list[Card | None]) -> list[int]:
    """Deck lands a land search can find (front-face lands only: a search can't find a
    modal card's back face)."""
    return [
        index for index, card in enumerate(ordered)
        if index and land_faces[index] is card
        and any(matches_land_search(card, selector) for selector in selectors)
    ]


# ---------------------------------------------------------------------------------------------
# The Rust engine


def cargo_path() -> str | None:
    """Cargo on PATH, else where rustup installs it, which some shells leave off PATH."""
    found = shutil.which("cargo")
    if found:
        return found
    home = Path(os.getenv("CARGO_HOME") or Path.home() / ".cargo")
    cargo = home / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    return str(cargo) if cargo.is_file() else None


def build_problem() -> str | None:
    """Why the engine can't be built here; None when it can."""
    if not (RUST_SOURCE / "Cargo.toml").is_file():
        return f"the engine's source is missing from {RUST_SOURCE}"
    if cargo_path() is None:
        return f"Rust isn't installed; install it from {RUSTUP}, then run setup again"
    return None


def engine_problem() -> str | None:
    """Why the built engine can't be used: missing, or older than its source. None if ready."""
    if not RUST_BINARY.is_file():
        return "the goldfish engine isn't built"
    sources = [RUST_SOURCE / "Cargo.toml", RUST_SOURCE / "Cargo.lock",
               *(RUST_SOURCE / "src").rglob("*.rs")]
    newest = max(path.stat().st_mtime for path in sources if path.is_file())
    if RUST_BINARY.stat().st_mtime < newest:
        return "the goldfish engine is older than its source"
    return None


def build_engine() -> str:
    """Compile goldfish-rs in release mode. The first build also downloads its two crates."""
    if problem := build_problem():
        raise FileNotFoundError(problem)
    started = time.perf_counter()
    result = subprocess.run(
        [cargo_path(), "build", "--release", "--manifest-path", str(RUST_SOURCE / "Cargo.toml")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        timeout=1800,
    )
    if result.returncode != 0:
        raise OSError("cargo build failed: " + result.stderr.strip()[-800:])
    return f"built in {time.perf_counter() - started:.0f}s"


def play_games_rust(deck: dict[str, Any], games: int, turns: int, seed: int, *,
                    threads: int = 0) -> dict[str, Any]:
    """Counters over `games` solitaire games of `turns` turns each; `threads` 0 uses every
    core."""
    if problem := engine_problem():
        raise FileNotFoundError(
            f"{problem[0].upper()}{problem[1:]}. Run /setup in the terminal app, or "
            f"python -m mtg_agentic setup; it needs Rust ({RUSTUP})."
        )
    result = subprocess.run(
        [str(RUST_BINARY), "--games", str(games), "--turns", str(turns), "--seed", str(seed),
         "--threads", str(threads)],
        input=json.dumps(deck), capture_output=True, text=True, check=False, timeout=3600,
    )
    if result.returncode != 0:
        raise OSError("Rust goldfish engine failed: " + result.stderr.strip()[-500:])
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------------------------
# Reports


def _cumulative(histogram: list[int], games: int) -> list[float]:
    total, result = 0, []
    for count in histogram[:-1]:
        total += count
        result.append(total / games)
    return result


def _median_turn(histogram: list[int], games: int) -> int | None:
    """The first turn by which half the games got there; None if half never did."""
    for turn, share in enumerate(_cumulative(histogram, games), start=1):
        if share >= 0.5:
            return turn
    return None


def _milestone(histogram: list[int], games: int) -> dict[str, Any]:
    reached = _cumulative(histogram, games)
    observed = sum(histogram[:-1])
    mean = (sum(t * n for t, n in enumerate(histogram[:-1], start=1)) / observed
            if observed else None)
    return {
        "median_turn": _median_turn(histogram, games),
        "mean_turn_when_reached": mean,
        "by_turn": reached,
        "not_reached": histogram[-1] / games,
    }


def summarize(deck: dict[str, Any], counters: dict[str, Any]) -> dict[str, Any]:
    games, turns, pieces = counters["games"], counters["turns"], deck["pieces"]
    if not games:
        raise ValueError("No games were played")
    commander = pieces[deck["commander"]]
    cards = []
    for i, piece in enumerate(pieces):
        drawn = counters["drawn"][i]
        if i == deck["commander"] or piece["land"] or not drawn:
            continue
        cast = counters["cast"][i]
        cards.append({
            "name": piece["name"],
            "drawn_per_game": drawn / games,
            "cast_rate": cast / drawn,
            "mean_cast_turn": counters["cast_turn_sum"][i] / cast if cast else None,
            "stranded_rate": counters["stranded"][i] / drawn,
        })
    cards.sort(key=lambda c: (-c["stranded_rate"], c["name"]))
    coverage: dict[str, list[str]] = {}
    for piece in pieces:
        for effect in piece["effects"] or (["cast only"] if piece["castable"] else ["not cast"]):
            coverage.setdefault(effect, []).append(piece["name"])
    return {
        "games": games,
        "turns": turns,
        "scope": SCOPE,
        "on_play_rate": counters["on_play"] / games,
        "mulligans": {
            "distribution": {str(n): k / games for n, k in enumerate(counters["mulligans"]) if k},
            "rate": 1 - counters["mulligans"][0] / games,
            "note": "The first mulligan is free in multiplayer Commander.",
        },
        "opening_lands": {str(n): k / games for n, k in enumerate(counters["opening_lands"]) if k},
        "by_turn": [
            {
                "turn": t + 1,
                "land_drop": counters["land_drops"][t] / games,
                "lands_on_curve": counters["lands_on_curve"][t] / games,
                "mana_available": counters["mana_available"][t] / games,
                "mana_spent": counters["mana_spent"][t] / games,
                "color_screw": counters["color_screw"][t] / games,
            }
            for t in range(turns)
        ],
        "commander": {"name": commander["name"], "mana_value": commander["mv"],
                      **_milestone(counters["commander_turn"], games)},
        "first_kill": _milestone(counters["first_kill"], games),
        "table_kill": _milestone(counters["kill"], games),
        "cards": cards,
        "coverage": {effect: sorted(names) for effect, names in sorted(coverage.items())},
    }


def run_goldfish(deck: dict[str, Any], store: OracleStore, *, games: int = 10_000,
                 turns: int = 10, seed: int = 1, threads: int = 0) -> dict[str, Any]:
    """Compile, play in the Rust engine and summarize."""
    if type(games) is not int or not 1 <= games <= 10_000_000:
        raise ValueError("Games must be from 1 to 10,000,000")
    if type(turns) is not int or not 1 <= turns <= 30:
        raise ValueError("Turns must be from 1 to 30")
    compiled = compile_deck(deck, store)
    if len(compiled["library"]) < 7:
        raise ValueError("A goldfish needs at least seven cards besides the commander")
    started = time.perf_counter()
    counters = play_games_rust(compiled, games, turns, seed, threads=threads)
    elapsed = time.perf_counter() - started
    report = summarize(compiled, counters)
    report.update(seed=seed, seconds=elapsed, deck_size=len(compiled["library"]) + 1)
    if report["deck_size"] != 100:
        report["warning"] = f"The deck has {report['deck_size']} cards, not 100."
    return report
