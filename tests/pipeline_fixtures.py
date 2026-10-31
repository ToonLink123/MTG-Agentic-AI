import copy
from collections import deque

from mtg_agentic.cards import name_key, trim_card
from mtg_agentic.discovery import Candidate, CandidatePool
from mtg_agentic.schema import DRAFT_TIERS
from mtg_agentic.scryfall import CardNotFound
from mtg_agentic.spec import DeckSpec
from tests.fixtures import card
from tests.land_fixtures import fixing_lands

JODAH = card(
    "Jodah, Archmage Eternal",
    colors="WUBRG",
    types="Legendary Creature — Human Wizard",
    mana_cost="{1}{U}{R}{W}",
    cmc=4,
    text="Flying\nYou may pay {W}{U}{B}{R}{G} rather than pay the mana cost for spells that you cast.",
)
BASICS = [
    card(
        name,
        colors=color if color != "C" else "",
        types="Basic Land" + (" — " + name if color != "C" else ""),
        text="{T}: Add {" + color + "}.",
        produced_mana=[color],
    )
    for color, name in zip("WUBRGC", ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"))
]


def make_spec(**changes):
    data = {
        "commander": JODAH["name"],
        "color_identity": "WUBRG",
        "bracket": 3,
        "themes": ["myojin tribal", "divinity counters"],
        "mechanics": ["proliferate", "counters"],
        "win_conditions": ["myojin combat damage", "alternate wincons"],
        "playstyle": "midrange value",
        "must_include": [],
        "exclude": [],
        "role_targets": {"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "lands": 36},
    }
    data.update(changes)
    return DeckSpec.from_dict(data)


def spec_reply(**changes):
    """A spec as a model writes it: the SPEC_SCHEMA fields, without the brief code adds."""
    data = make_spec(**changes).to_dict()
    data.pop("brief")
    return data


def intake_reply(*asks, **changes):
    """Intake's answer: the spec fields plus asks, by default one quoting "Myojin"."""
    asks = asks or ({"quote": "Myojin", "wants": "Myojin tribal", "kind": "cards"},)
    return {**spec_reply(**changes), "asks": list(asks)}


def map_reply(*stages, asks=(), **changes):
    """An engine map as a model writes it."""
    data = {
        "summary": "Cast Myojin, keep their counters, win in combat.",
        "resources": ["divinity counters"],
        "stages": list(stages), "links": [], "asks": list(asks), "gaps": [], "searches": [],
    }
    data.update(changes)
    return data


def records(count=70):
    return [
        JODAH,
        *BASICS,
        *[
            card(
                f"Spell {n}", colors="WUBRG"[n % 5], mana_cost="{1}{" + "WUBRG"[n % 5] + "}", cmc=2
            )
            for n in range(count)
        ],
    ]


def playable_records():
    """Synthetic Oracle abilities that satisfy roles without trusting pool tags."""
    data = records()
    for raw in data:
        if not raw["name"].startswith("Spell "):
            continue
        number = int(raw["name"].split()[1])
        if number < 18:
            raw.update(
                type_line="Creature — Spirit",
                oracle_text="Draw a card. Create a Treasure token. Destroy target artifact. "
                            "Your commander has hexproof.",
            )
        elif number < 21:
            raw.update(
                type_line="Instant",
                oracle_text="Target creature gains indestructible until end of turn.",
            )
        elif number < 24:
            raw.update(type_line="Sorcery", oracle_text="Destroy all creatures.")
    return [
        *data, *fixing_lands(),
        card("City of Brass", types="Land",
             text="{T}: Add one mana of any color.\n"
                  "Whenever City of Brass becomes tapped, it deals 1 damage to you."),
        card("Mana Confluence", types="Land",
             text="{T}, Pay 1 life: Add one mana of any color."),
    ]


def playable_deck(**changes):
    """An independent 63+36 fixture with ten real land sources of every color."""
    data = {
        "commander": JODAH["name"],
        "bracket": 3,
        "cards": {
            **dict.fromkeys((f"Spell {n}" for n in range(63)), 1),
            "Plains": 7, "Island": 5, "Swamp": 7, "Mountain": 7, "Forest": 5,
            "Command Tower": 1, "City of Brass": 1, "Mana Confluence": 1,
            "Tropical Island": 1, "Breeding Pool": 1,
        },
    }
    data.update(changes)
    return data


def make_pool(store, spec=None, count=70):
    spec = spec or make_spec()
    candidates = [
        Candidate(store.get_card(f"Spell {n}"), {"ramp", "draw", "removal", "wipes", "theme"})
        for n in range(count)
    ]
    candidates.extend(Candidate(store.get_card(card["name"]), {"lands"}) for card in BASICS)
    return CandidatePool(spec, candidates)


def draft(count=63, **changes):
    """A drafter's answer: Spell 0 to Spell count-1 ranked in the flex tier."""
    data = {
        "commander": JODAH["name"],
        "picks": {
            **{tier: [] for tier in DRAFT_TIERS},
            "flex": [f"Spell {n}" for n in range(count)],
        },
        "outside": [],
    }
    data.update(changes)
    return data


class FakeModel:
    def __init__(self, *outputs):
        self.outputs = deque(outputs)
        self.calls = []

    def complete_json(self, messages, schema):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(schema)))
        output = self.outputs.popleft()
        if isinstance(output, Exception):
            raise output
        return copy.deepcopy(output)


class FakeToolModel:
    """Scripted native tool use: each turn is a list of (tool name, input) calls."""

    def __init__(self, *turns):
        self.turns = deque(turns)
        self.calls = []

    def tool_turn(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        number = len(self.calls)
        calls = [
            {"id": f"call-{number}-{index}", "name": name, "input": copy.deepcopy(args)}
            for index, (name, args) in enumerate(self.turns.popleft())
        ]
        return {"calls": calls, "text": "",
                "content": [{"type": "tool_use", **call} for call in calls]}

    def complete_json(self, messages, schema):
        raise AssertionError("A tool-capable model answers through tools here")

    def results(self, turn):
        """The tool results sent back after `turn` (1-based), by tool_use id."""
        message = self.calls[turn][0][-1]
        return [block for block in message["content"] if block["type"] == "tool_result"]


class FakeCards:
    def __init__(self, cards, search_results=None):
        self.records = {name_key(card["name"]): card for card in cards}
        self.search_results = deque(search_results) if search_results is not None else None
        self.search_calls = []
        self.get_calls = []

    def search_cards(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        selected = (
            list(self.records.values())
            if self.search_results is None
            else self.search_results.popleft()
        )
        return [trim_card(card) for card in selected]

    def get_card(self, name, *, fuzzy=True):
        self.get_calls.append((name, fuzzy))
        record = self.records.get(name_key(name))
        if record is None:
            raise CardNotFound("No such card", status=404)
        return trim_card(record)

    def get_card_record(self, name):
        return self.records[name_key(name)]
