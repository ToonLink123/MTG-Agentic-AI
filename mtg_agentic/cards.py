"""Small, JSON-serializable card projections, not entire print records."""

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

Card = dict[str, Any]


def name_key(name: str) -> str:
    """Normalize case/spacing, without guessing spelling or removing punctuation."""
    return " ".join(unicodedata.normalize("NFC", name).casefold().split())


def trim_card(card: Card) -> Card:
    """Keep the six gameplay fields and the ID needed to request rulings."""
    faces = card.get("card_faces", [])
    return {
        "id": card["id"],
        "name": card["name"],
        "mana_cost": card.get("mana_cost")
        or " // ".join(face.get("mana_cost", "") for face in faces),
        "type_line": card.get("type_line")
        or " // ".join(face.get("type_line", "") for face in faces),
        "oracle_text": card.get("oracle_text")
        or "\n//\n".join(face.get("oracle_text", "") for face in faces),
        "color_identity": card["color_identity"],
        "game_changer": bool(card.get("game_changer")),
    }


REMINDER = re.compile(r"\s*\([^()]*\)")
# EDHREC rank bands: how widely Commander decks play a card, as a hint beside its text.
POPULARITY = ((300, "staple"), (2000, "popular"), (8000, "played"))


def rules_text(text: str) -> str:
    """Oracle text without reminder text, which restates rules every model already knows."""
    lines = (" ".join(REMINDER.sub("", line).split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def prompt_card(card: Card, roles: Iterable[str] | None = None) -> Card:
    """A card as a model reads it in a candidate list: name, cost, mana value, type, power and
    toughness, rules text without reminders, Scryfall's function tags, how widely it is played,
    and (with `roles`) the deck roles found for it. Every card shares the commander's colors, so
    color identity is left out; so are IDs, which only code uses."""
    base = trim_card(card)
    faces = card.get("card_faces") or []
    front = faces[0] if faces else card
    value = card.get("cmc")
    row: Card = {"name": base["name"], "mana_cost": base["mana_cost"]}
    if isinstance(value, int | float):
        row["mv"] = int(value) if float(value).is_integer() else value
    row["type_line"] = base["type_line"]
    if front.get("power") not in (None, "") or front.get("toughness") not in (None, ""):
        row["pt"] = f"{front.get('power', '')}/{front.get('toughness', '')}"
    row["oracle_text"] = rules_text(base["oracle_text"])
    if card.get("functions"):
        row["tags"] = list(card["functions"])
    rank = card.get("edhrec_rank")
    if isinstance(rank, int):
        row["played"] = next((label for limit, label in POPULARITY if rank <= limit), "rarely")
    if base["game_changer"]:
        row["game_changer"] = True
    if roles is not None:
        row["roles"] = sorted(roles)
    return row


def compact_row(card: Card, roles: Iterable[str] = ()) -> str:
    """One line for a card a model may pick but need not read: name | cost | type | roles and
    tags. Its full text is a tool call away."""
    base = trim_card(card)
    labels = sorted({*roles, *(card.get("functions") or ())})
    return " | ".join(filter(None, [base["name"], base["mana_cost"], base["type_line"], ", ".join(labels)]))


def validation_card(card: Card) -> Card:
    """Retain only additional fields needed for deterministic validation."""
    result = trim_card(card)
    faces = card.get("card_faces", [])
    colors = card.get("colors")
    if colors is None:
        colors = sorted({c for face in faces for c in face.get("colors", [])}, key="WUBRG".find)
    result.update(
        # Local search: colors and keywords for c: and kw:, the EDHREC rank for its ordering.
        colors=colors, keywords=card.get("keywords", []), edhrec_rank=card.get("edhrec_rank"),
        oracle_id=card.get("oracle_id", card["id"]),
        layout=card.get("layout", "normal"),
        cmc=card.get("cmc"),
        produced_mana=card.get("produced_mana"),
        power=card.get("power"),
        toughness=card.get("toughness"),
        legalities={"commander": card.get("legalities", {}).get("commander", "not_legal")},
        card_faces=[
            {
                key: face.get(key, "")
                for key in ("name", "type_line", "oracle_text", "mana_cost", "power", "toughness")
            }
            for face in card.get("card_faces", [])
        ],
    )
    return result
