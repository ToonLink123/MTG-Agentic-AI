"""Small Oracle-style fixtures for typed fixing and land-search dependencies."""

from tests.fixtures import card


def fixing_lands():
    return [
        card("Tropical Island", colors="UG", types="Land — Forest Island",
             text="({T}: Add {G} or {U}.)"),
        card("Breeding Pool", colors="UG", types="Land — Forest Island",
             text="({T}: Add {G} or {U}.)\nAs Breeding Pool enters, you may pay 2 life. "
                  "If you don't, it enters tapped."),
        card("Simic Guildgate", colors="UG", types="Land — Gate",
             text="Simic Guildgate enters tapped.\n{T}: Add {G} or {U}."),
        card("Misty Rainforest", types="Land",
             text="{T}, Pay 1 life, Sacrifice Misty Rainforest: Search your library for "
                  "a Forest or Island card, put it onto the battlefield, then shuffle."),
        card("Flooded Strand", types="Land",
             text="{T}, Pay 1 life, Sacrifice Flooded Strand: Search your library for "
                  "a Plains or Island card, put it onto the battlefield, then shuffle."),
        card("Command Tower", types="Land",
             text="{T}: Add one mana of any color in your commander's color identity."),
    ]


def land_tutors():
    return [
        card("Nature's Lore", colors="G", types="Sorcery", mana_cost="{1}{G}", cmc=2,
             text="Search your library for a Forest card, put that card onto the "
                  "battlefield, then shuffle."),
        card("Farseek", colors="G", types="Sorcery",
             text="Search your library for a Plains, Island, Swamp, or Mountain card, "
                  "put it onto the battlefield tapped, then shuffle."),
        card("Cultivate", colors="G", types="Sorcery",
             text="Search your library for up to two basic land cards, reveal those "
                  "cards, put one onto the battlefield tapped and the other into "
                  "your hand, then shuffle."),
    ]
