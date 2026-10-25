"""Validated Forge deck export. Simulation support is added only after a live smoke."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .oracle import OracleStore
from .validator import validate_deck


class ForgeError(RuntimeError):
    pass


def _line(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n|[]"):
        raise ValueError(
            "Forge names must be nonempty single lines without section/print delimiters"
        )
    return value


def forge_name(card: dict[str, Any]) -> str:
    """Forge card databases index multifaced cards by their front-face name."""
    faces = card.get("card_faces", [])
    return _line(faces[0]["name"] if faces else card["name"])


def export_deck(
    deck: dict[str, Any], store: OracleStore, destination: str | Path, *, name: str | None = None
) -> dict[str, Any]:
    """Write a NEW UTF-8 .dck. This checks legality, not LLM safety-review provenance."""
    if not isinstance(deck, dict) or not isinstance(deck.get("cards"), dict):
        raise ValueError("Expected deck JSON with a cards mapping")
    commander = deck.get("commander")
    validation = validate_deck(deck["cards"], commander, store, bracket=deck.get("bracket", 3))
    if not validation["valid"]:
        raise ForgeError("Deck failed pre-export validation: " + str(validation["errors"]))
    destination = Path(destination)
    if destination.suffix.lower() != ".dck":
        raise ValueError("Forge exports need a .dck extension")
    title = _line(name or destination.stem)
    cards: Counter[str] = Counter()
    for n, q in deck["cards"].items():
        cards[forge_name(store.get_card(n))] += q
    leader = forge_name(store.get_card(commander))
    body = "\n".join(
        [
            "[metadata]",
            "Name=" + title,
            "[Commander]",
            "1 " + leader,
            "[Main]",
            *[f"{q} {n}" for n, q in sorted(cards.items())],
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(body)
    return {
        "path": str(destination.resolve()),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "validation": validation,
        "safety_review": deck.get("review", {}).get("status", "not_run"),
    }
