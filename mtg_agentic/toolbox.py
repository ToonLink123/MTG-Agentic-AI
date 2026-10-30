"""What an agent's tools reach, and the card tools the drafter and the orchestrator share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import activity
from .brackets import SpellbookClient
from .cards import Card, trim_card
from .discovery import constrain_query, eligible
from .mana import is_land
from .oracle import OracleStore
from .roles import oracle_roles
from .rules import RulesStore
from .schema import NAME_SCHEMA, object_schema
from .scryfall import ScryfallClient, ScryfallError, card_rulings
from .spec import DeckSpec

QUERY = {"type": "string", "minLength": 1, "maxLength": 900}
CARD_NAMES = {"type": "array", "items": NAME_SCHEMA, "minItems": 1, "maxItems": 10}
SEARCH_INPUT = object_schema({"query": QUERY})
CARDS_INPUT = object_schema({"cards": CARD_NAMES})
RULES_INPUT = object_schema({"query": QUERY, "cards": {**CARD_NAMES, "minItems": 0, "maxItems": 6}})


@dataclass
class Toolbox:
    """What the agents' tools reach; a missing one makes its tool report unavailable."""

    store: OracleStore
    client: ScryfallClient | None = None
    rules: RulesStore | None = None
    spellbook: SpellbookClient | None = None
    # Searches go here when set (the local BulkSearch), else to the client's live search.
    search: Any = None
    evidence: int = 12  # Evidence items per rules lookup.


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


class CardTools:
    """Card search, card text, rulings and rules for one agent. A card the agent finds becomes
    addable when it is an eligible nonland (the commander's colors, legality, exclusions, the
    bracket) and not already taken."""

    def __init__(self, tools: Toolbox, spec: DeckSpec):
        self.tools, self.store, self.spec = tools, tools.store, spec
        self.addable: dict[str, Card] = {}

    def taken(self, name: str) -> bool:
        """Whether `name` is already in the deck, so it cannot be added again."""
        return False

    def pickable(self, name: str) -> bool:
        return name in self.addable

    def offer(self, card: Card | None) -> bool:
        if card is None or is_land(card) or self.taken(card["name"]):
            return False
        if not eligible(card, self.spec):
            return False
        self.addable.setdefault(card["name"], card)
        return True

    def lookup(self, name: str) -> Card | None:
        card = self.store.get_card(name)
        if card is None and self.tools.client is not None:
            try:
                card = self.store.get_card(self.tools.client.get_card(name)["name"])
            except (ScryfallError, ValueError):
                card = None
        return card

    def describe(self, card: Card) -> dict[str, Any]:
        return {
            **trim_card(card), "roles": sorted(oracle_roles(card)),
            "in_deck": self.taken(card["name"]), "addable": self.pickable(card["name"]),
        }

    def search_cards(self, query: str, limit: int = 20) -> dict[str, Any]:
        searcher = self.tools.search or self.tools.client
        if searcher is None:
            return {"error": "Card search is unavailable here."}
        query = query.strip()
        activity.tool("card search", query)
        scoped = constrain_query(query, self.spec.color_identity)
        found = searcher.search_cards(scoped, max_results=limit, order="edhrec")
        rows = []
        for result in found:
            card = self.store.get_card(result["name"])
            self.offer(card)
            rows.append(
                {**self.describe(card), "oracle_text": clip(card["oracle_text"], 400)}
                if card
                else {"name": result["name"], "addable": False, "note": "not in local card data"}
            )
        return {"query": scoped, "results": rows}

    def card(self, names: list[str]) -> dict[str, Any]:
        rows = []
        for name in names[:10]:
            activity.tool("card", name)
            card = self.lookup(name)
            self.offer(card)
            rows.append(self.describe(card) if card else {"name": name, "error": "No such card"})
        return {"cards": rows}

    def rulings(self, names: list[str]) -> dict[str, Any]:
        if self.tools.client is None and not self.store.rulings_ready:
            return {"error": "Rulings are unavailable here."}
        rows = []
        for name in names[:5]:
            activity.tool("rulings", name)
            card = self.lookup(name)
            if card is None:
                rows.append({"name": name, "error": "No such card"})
                continue
            rulings = card_rulings(self.store, self.tools.client, card)
            texts = [clip(r["comment"] or "", 500) for r in rulings[:12]]
            rows.append({"name": card["name"], "rulings": texts})
        return {"rulings": rows}

    def rules(self, question: str, names: list[str]) -> dict[str, Any]:
        if self.tools.rules is None:
            return {"error": "The rules index is unavailable here."}
        question = question.strip()
        activity.tool("rules lookup", clip(question, 80))
        found = [card["name"] for card in map(self.lookup, names[:6]) if card]
        evidence = self.tools.rules.retrieve(found, question, limit=self.tools.evidence)
        return {"evidence": [
            {"id": e["id"], "kind": e["kind"], "text": clip(e["text"], 600)} for e in evidence
        ]}

    def combos(self, cards: dict[str, int]) -> dict[str, Any]:
        if self.tools.spellbook is None:
            return {"error": "Combo search is unavailable here."}
        activity.tool("commander spellbook", f"combos among {len(cards) + 1} cards")
        found = self.tools.spellbook.find_combos(cards, self.spec.commander)
        combos = [{"cards": c["cards"], "results": c["results"]} for c in found["combos"][:15]]
        return {"complete": found["complete"], "combos": combos}
