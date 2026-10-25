"""Small synthetic Scryfall records. Unit tests never use the network."""

from typing import Any


def card(
    name: str,
    *,
    colors: str = "",
    types: str = "Artifact",
    text: str = "",
    legal: str = "legal",
    changer: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "object": "card",
        "id": name,
        "oracle_id": name,
        "name": name,
        "mana_cost": "",
        "type_line": types,
        "oracle_text": text,
        "color_identity": list(colors),
        "game_changer": changer,
        "legalities": {"commander": legal},
        "prices": {"usd": "999.99"},
        "image_uris": {"normal": "https://example.invalid/image.png"},
    }
    result.update(extra)
    return result


COMMANDER = card("Test Commander", colors="WU", types="Legendary Creature — Human", changer=True)
PLAINS = card("Plains", colors="W", types="Basic Land — Plains")
ISLAND = card("Island", colors="U", types="Basic Land — Island")
WASTES = card("Wastes", types="Basic Land")
BOLT = card("Lightning Bolt", colors="R", types="Instant")
SOL_RING = card("Sol Ring")
DFC = card(
    "Front Face // Back Face",
    colors="WU",
    types="Legendary Creature — Human // Legendary Creature — Spirit",
    layout="transform",
    card_faces=[
        {
            "name": "Front Face",
            "mana_cost": "{W}",
            "type_line": "Legendary Creature — Human",
            "oracle_text": "Front rules.",
        },
        {
            "name": "Back Face",
            "mana_cost": "",
            "type_line": "Legendary Creature — Spirit",
            "oracle_text": "Back rules.",
        },
    ],
)
