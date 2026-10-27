"""Locally grounded mana-base staples, independent of model query coverage."""

from .cards import Card
from .oracle import OracleStore

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


def preferred_lands(store: OracleStore) -> list[Card]:
    """Never invent a missing card or use hard-coded rules text."""
    return [card for name in PREFERRED_LANDS if (card := store.get_card(name))]
