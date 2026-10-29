"""Small, JSON-serializable card projections, not entire print records."""

import unicodedata
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
