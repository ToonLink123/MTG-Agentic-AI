"""Pure-code Commander checks. No HTTP, fuzzy matching, or LLM calls."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .cards import Card, name_key
from .oracle import OracleStore


@dataclass(frozen=True)
class DeckEntry:
    name: str
    quantity: int = 1


Decklist = str | Mapping[str, int] | Iterable[str | DeckEntry]


def _parse_decklist(decklist: Decklist) -> tuple[list[DeckEntry], list[dict[str, Any]]]:
    entries: list[DeckEntry] = []
    errors: list[dict[str, Any]] = []
    if isinstance(decklist, Mapping):
        rows = (DeckEntry(name, quantity) for name, quantity in decklist.items())
    elif isinstance(decklist, str):
        rows = iter(decklist.splitlines())
    else:
        try:
            rows = iter(decklist)
        except TypeError:
            return [], [
                {"error": "invalid_decklist", "detail": "Expected text, a mapping, or entries"}
            ]
    for index, row in enumerate(rows, start=1):
        if isinstance(row, str):
            row = row.strip()
            if not row or row.startswith(("#", "//")):
                continue
            match = re.fullmatch(r"(?:(-?\d+)[xX]?\s+)?(.+)", row)
            assert match is not None
            quantity, name = match.groups()
            row = DeckEntry(name, int(quantity) if quantity is not None else 1)
        if (
            not isinstance(row, DeckEntry)
            or not isinstance(row.name, str)
            or not row.name.strip()
            or type(row.quantity) is not int
            or row.quantity <= 0
        ):
            errors.append(
                {
                    "error": "invalid_entry",
                    "line": index,
                    "detail": "Expected a nonempty card name and a positive integer quantity",
                }
            )
            continue
        entries.append(DeckEntry(row.name.strip(), row.quantity))
    return entries, errors


_NUMBER_WORDS = {
    word: number
    for number, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        )
    )
}


def copy_limit(card: Card) -> int | None:
    if "Basic" in card["type_line"].split("—")[0].split():
        return None
    # Read exceptions from Oracle text, not a hardcoded list of card names.
    names = [card["name"]] + [face["name"] for face in card["card_faces"]]
    for name in names:
        escaped = re.escape(name)
        if re.search(
            rf"A deck can have any number of cards named {escaped}\.", card["oracle_text"]
        ):
            return None
        match = re.search(
            rf"A deck can have up to ([\w-]+) cards named {escaped}\.", card["oracle_text"]
        )
        if match:
            number = match.group(1).lower()
            if number.isdigit():
                return int(number)
            if number in _NUMBER_WORDS:
                return _NUMBER_WORDS[number]
    return 1


def can_be_commander(card: Card) -> bool:
    faces = card["card_faces"]
    front = faces[0] if faces else card
    types = front["type_line"].split("—")[0].split()
    subtypes = front["type_line"].split("—", 1)[-1].split()
    if "Legendary" in types and (
        "Creature" in types
        or "Vehicle" in subtypes
        or (
            "Spacecraft" in subtypes
            and front.get("power") not in {None, ""}
            and front.get("toughness") not in {None, ""}
        )
    ):
        return True
    # Grist is a creature in the command zone by its characteristic-defining ability.
    if card["name"] == "Grist, the Hunger Tide":
        return True
    return bool(
        re.search(
            rf"(?:{re.escape(front['name'])}|This card) can be your commander\b",
            front["oracle_text"],
            re.IGNORECASE,
        )
    )


def _colors(colors: set[str]) -> str:
    return "".join(color for color in "WUBRG" if color in colors) or "colorless"


def card_errors(card: Card, commander_colors: set[str] | None) -> list[dict[str, Any]]:
    """Shared legality/color checks for validation, discovery, and verified additions."""
    errors: list[dict[str, Any]] = []
    legality = card.get("legalities", {}).get("commander", "not_legal")
    if legality != "legal":
        errors.append({"error": "commander_legality", "card": card["name"], "detail": legality})
    if commander_colors is not None:
        outside = set(card["color_identity"]) - commander_colors
        if outside:
            errors.append(
                {
                    "error": "color_identity",
                    "card": card["name"],
                    "detail": f"{_colors(outside)} outside {_colors(commander_colors)}",
                }
            )
    return errors


def validate_deck(
    decklist: Decklist,
    commander: str,
    store: OracleStore,
    *,
    bracket: int = 3,
    game_changers: Collection[str] | None = None,
) -> dict[str, Any]:
    """Validate 99 maindeck cards plus one separately supplied commander.

    Names resolve only against the local Oracle snapshot. The commander must not
    also be included in decklist. See README for intentionally unsupported formats.
    """
    entries, errors = _parse_decklist(decklist)
    commander_supplied = isinstance(commander, str) and bool(commander.strip())
    card_count = sum(entry.quantity for entry in entries) + int(commander_supplied)
    report: dict[str, Any] = {
        "valid": False,
        "commander": commander if commander_supplied else None,
        "bracket": bracket,
        "card_count": card_count,
        "color_identity": [],
        "game_changer_count": 0,
        "game_changers": [],
        "oracle_updated_at": store.metadata.get("updated_at"),
        "errors": errors,
    }
    bracket_valid = type(bracket) is int and bracket in range(1, 6)
    if not bracket_valid:
        errors.append(
            {"error": "invalid_bracket", "detail": "Bracket must be an integer from 1 to 5"}
        )
    if card_count != 100:
        errors.append(
            {
                "error": "card_count",
                "detail": f"Expected 100 cards including commander; got {card_count}",
            }
        )
    if not commander_supplied:
        errors.append({"error": "invalid_commander", "detail": "Supply exactly one commander name"})
    if not store.ready:
        errors.append(
            {"error": "oracle_unavailable", "detail": "Download/import an Oracle snapshot first"}
        )
        return report

    quantities: Counter[str] = Counter()
    cards: dict[str, Card] = {}
    unknown: set[str] = set()

    def resolve(name: str, quantity: int) -> Card | None:
        card = store.get_card(name)
        if card is None:
            key = name_key(name)
            if key not in unknown:
                errors.append(
                    {"error": "unknown_card", "card": name, "detail": "No exact local Oracle match"}
                )
                unknown.add(key)
            return None
        key = name_key(card["name"])
        cards[key] = card
        quantities[key] += quantity
        return card

    leader = resolve(commander, 1) if commander_supplied else None
    if leader:
        report["commander"] = leader["name"]
        report["color_identity"] = [color for color in "WUBRG" if color in leader["color_identity"]]
        if not can_be_commander(leader):
            errors.append(
                {
                    "error": "commander_type",
                    "card": leader["name"],
                    "detail": "Not a legendary creature or a card allowed to be your commander",
                }
            )
        if leader["name"] in {"The Prismatic Piper", "Faceless One"}:
            errors.append(
                {
                    "error": "unsupported_commander",
                    "card": leader["name"],
                    "detail": "A chosen commander color is required and is not yet supported",
                }
            )
    for entry in entries:
        resolve(entry.name, entry.quantity)

    commander_colors = set(leader["color_identity"]) if leader else None
    live_game_changers = {name_key(n) for n in game_changers} if game_changers is not None else None
    found_game_changers: list[str] = []
    game_changer_count = 0
    for key, card in cards.items():
        quantity = quantities[key]
        limit = copy_limit(card)
        if limit is not None and quantity > limit:
            errors.append(
                {
                    "error": "singleton",
                    "card": card["name"],
                    "detail": f"{quantity} copies; maximum {limit}",
                }
            )
        errors.extend(card_errors(card, commander_colors))
        if (
            name_key(card["name"]) in live_game_changers
            if live_game_changers is not None
            else card["game_changer"]
        ):
            found_game_changers.append(card["name"])
            game_changer_count += quantity
    report["game_changers"] = found_game_changers
    report["game_changer_count"] = game_changer_count
    if bracket_valid:
        limit = {1: 0, 2: 0, 3: 3, 4: None, 5: None}[bracket]
        if limit is not None and game_changer_count > limit:
            errors.append(
                {
                    "error": "game_changers",
                    "detail": f"{game_changer_count} Game Changers; Bracket {bracket} allows {limit}",
                }
            )
    report["valid"] = not errors
    return report


class DeckValidator:
    def __init__(self, store: OracleStore):
        self.store = store

    def validate(self, decklist: Decklist, commander: str, *, bracket: int = 3) -> dict[str, Any]:
        return validate_deck(decklist, commander, self.store, bracket=bracket)
