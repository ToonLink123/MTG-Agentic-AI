"""Locally grounded mana-base staples, independent of model query coverage."""

import re
from collections.abc import Iterable

from .cards import Card, name_key
from .mana import BASIC_LANDS, front, is_land, search_selectors
from .oracle import OracleStore
from .spec import DeckSpec
from .validator import card_errors

# Names are discovery anchors only. Oracle supplies types, abilities and legality.
PREFERRED_LANDS = (
    "Tundra", "Underground Sea", "Badlands", "Taiga", "Savannah",
    "Scrubland", "Volcanic Island", "Bayou", "Plateau", "Tropical Island",
    "Hallowed Fountain", "Watery Grave", "Blood Crypt", "Stomping Ground",
    "Temple Garden", "Godless Shrine", "Steam Vents", "Overgrown Tomb",
    "Sacred Foundry", "Breeding Pool",
    "Flooded Strand", "Polluted Delta", "Bloodstained Mire", "Wooded Foothills",
    "Windswept Heath", "Marsh Flats", "Scalding Tarn", "Verdant Catacombs",
    "Arid Mesa", "Misty Rainforest",
    "Command Tower", "City of Brass", "Mana Confluence",
)

UTILITY_LANDS = (
    "Eiganjo, Seat of the Empire", "Otawara, Soaring City", "Takenuma, Abandoned Mire",
    "Sokenzan, Crucible of Defiance", "Boseiju, Who Endures",
    "Castle Ardenvale", "Castle Vantress", "Castle Locthwain",
    "Castle Embereth", "Castle Garenbrig",
    "War Room", "Scavenger Grounds",
)


def preferred_lands(store: OracleStore) -> list[Card]:
    """Never invent a missing card or use hard-coded rules text."""
    return [card for name in PREFERRED_LANDS if (card := store.get_card(name))]


def is_utility_land(card: Card) -> bool:
    """Recognize non-fixing utility, not life payments or a fetch ability."""
    return (
        is_land(card)
        and card["name"] not in PREFERRED_LANDS
        and not search_selectors(card)
        and (card["name"] in UTILITY_LANDS or bool(re.search(
            r"\b(?:channel|draw|scry|surveil|regenerate)\b|"
            r"\b(?:destroy|exile|return|untap) target\b|"
            r"\bcreate\b.*\btoken\b|\bbecomes? a\b.*\bcreature\b|"
            r"\bcreatures you control get\b",
            front(card, "oracle_text"), re.I,
        )))
    )


def mana_land_options(
    spec: DeckSpec, store: OracleStore, candidate_names: Iterable[str] = ()
) -> dict[str, Card]:
    """Discover lands independently of spell discovery; rehydrate optional names locally."""
    names = [*candidate_names, *PREFERRED_LANDS, *UTILITY_LANDS, *spec.must_include]
    for color in dict.fromkeys(spec.color_identity + "C"):
        basic = BASIC_LANDS[color]
        names.extend((basic, "Snow-Covered " + basic))
    excluded = {name_key(n) for n in spec.exclude}
    result = {}
    for name in dict.fromkeys(names):
        card = store.get_card(name)
        if (
            card and is_land(card)
            and name_key(card["name"]) != name_key(spec.commander)
            and name_key(card["name"]) not in excluded
            and not card_errors(card, set(spec.color_identity) - {"C"})
            and not (spec.bracket < 3 and card["game_changer"])
        ):
            result[card["name"]] = card
    return result
