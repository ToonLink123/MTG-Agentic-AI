"""Follow-up edits: the player talks to the orchestrator alone, never the council.

The orchestrator works in a bounded loop. Each step it calls one tool (card search, card text,
rulings, rules lookup, a dry-run deck check, a combo search) or finishes with a reply and
one-for-one nonland swaps. Code applies the swaps exactly, regenerates the lands, validates and
runs the safety review; any problems go back to the orchestrator, which may use more tools and
finish again. A question that needs no swaps changes nothing and rebuilds nothing.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any

from . import activity, prompts
from .brackets import SpellbookClient
from .cards import Card, name_key, trim_card
from .discovery import CandidatePool, constrain_query, eligible
from .lands import is_land
from .llm import JSONModel, request_json
from .oracle import OracleStore
from .pipeline import ask_report, check_cards, game_changer_names, make_lands
from .review import DeckReviewer
from .roles import oracle_roles
from .rules import RulesStore
from .schema import NAME_SCHEMA, SchemaError, object_schema
from .scryfall import ScryfallClient, ScryfallError, card_rulings
from .sources import SourceError
from .spec import DeckSpec

TOOLS = ("search_cards", "card", "rulings", "rules", "check_deck", "combos")
CARD_REF = re.compile(r"\[\[(.+?)\]\]")
MAX_SWAPS = 15


@dataclass
class Toolbox:
    """What the orchestrator's tools reach; a missing one makes its tool report unavailable."""

    store: OracleStore
    client: ScryfallClient | None = None
    rules: RulesStore | None = None
    spellbook: SpellbookClient | None = None
    # Searches go here when set (the local BulkSearch), else to the client's live search.
    search: Any = None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _Session:
    """One follow-up: the deck before it, the cards that may be added, and the tools."""

    def __init__(self, deck: dict[str, Any], pool: CandidatePool, tools: Toolbox, reviewer):
        self.store, self.tools, self.reviewer = tools.store, tools, reviewer
        spec = DeckSpec.from_dict(deck["spec"]) if "spec" in deck else pool.spec
        self.pool = dataclasses.replace(pool, spec=spec)
        self.spec = spec
        records = {name: self.store.get_card(name) for name in deck["cards"]}
        unknown = [name for name, card in records.items() if card is None]
        if unknown:
            raise ValueError(f"Deck has cards missing from the local Oracle data: {unknown[:5]}")
        self.nonlands = {n: q for n, q in deck["cards"].items() if not is_land(records[n])}
        self.lands = sorted(n for n in deck["cards"] if is_land(records[n]))
        self.records = records
        self.addable: dict[str, Card] = {
            c.card["name"]: c.card for c in self.pool.cards
            if not is_land(c.card) and c.card["name"] not in self.nonlands
        }
        self.changers = game_changer_names(self.pool, self.store, reviewer)

    def offer(self, card: Card | None) -> bool:
        """Make a card addable when it is a legal, eligible nonland not already in the deck."""
        if card is None or is_land(card) or card["name"] in self.nonlands:
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
            "in_deck": card["name"] in self.nonlands, "addable": card["name"] in self.addable,
        }

    def apply(self, swaps: list[dict[str, str]]) -> tuple[dict[str, int], DeckSpec]:
        nonlands = dict(self.nonlands)
        for swap in swaps:
            del nonlands[swap["remove"]]
            nonlands[swap["add"]] = 1
        removed = {name_key(swap["remove"]) for swap in swaps}
        # The player asked for a must-include card to go, so it stops being required.
        must = [n for n in self.spec.must_include if name_key(n) not in removed]
        if len(must) == len(self.spec.must_include):
            return nonlands, self.spec
        return nonlands, dataclasses.replace(self.spec, must_include=must)

    def build(self, swaps: list[dict[str, str]]) -> dict[str, Any]:
        """Lands, legality and playability for the deck after `swaps`; no model calls."""
        nonlands, spec = self.apply(swaps)
        pool = dataclasses.replace(self.pool, spec=spec)
        land_base = make_lands(pool, nonlands, self.store, self.changers)
        errors = list(land_base["errors"])
        cards = {**nonlands, **land_base["cards"]}
        validation, playability = check_cards(
            pool, cards, self.store, errors, reviewer=self.reviewer
        )
        return {
            "spec": spec, "cards": cards, "land_base": land_base, "validation": validation,
            "playability": playability, "errors": errors, "asks": ask_report(pool, cards),
        }

    def run(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = action["tool"]
        try:
            return getattr(self, f"_tool_{tool}")(action)
        except (ScryfallError, SourceError, ValueError) as exc:
            return {"error": str(exc)}

    def _tool_search_cards(self, action: dict[str, Any]) -> dict[str, Any]:
        searcher = self.tools.search or self.tools.client
        if searcher is None:
            return {"error": "Card search is unavailable here."}
        query = action["query"].strip()
        activity.tool("card search", query)
        scoped = constrain_query(query, self.spec.color_identity)
        found = searcher.search_cards(scoped, max_results=20, order="edhrec")
        rows = []
        for result in found:
            card = self.store.get_card(result["name"])
            self.offer(card)
            rows.append(
                {**self.describe(card), "oracle_text": _clip(card["oracle_text"], 400)}
                if card
                else {"name": result["name"], "addable": False, "note": "not in local card data"}
            )
        return {"query": scoped, "results": rows}

    def _tool_card(self, action: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for name in action["cards"][:10]:
            activity.tool("card", name)
            card = self.lookup(name)
            self.offer(card)
            rows.append(self.describe(card) if card else {"name": name, "error": "No such card"})
        return {"cards": rows}

    def _tool_rulings(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.tools.client is None and not self.store.rulings_ready:
            return {"error": "Rulings are unavailable here."}
        rows = []
        for name in action["cards"][:5]:
            activity.tool("rulings", name)
            card = self.lookup(name)
            if card is None:
                rows.append({"name": name, "error": "No such card"})
                continue
            rulings = card_rulings(self.store, self.tools.client, card)
            texts = [_clip(r["comment"] or "", 500) for r in rulings[:12]]
            rows.append({"name": card["name"], "rulings": texts})
        return {"rulings": rows}

    def _tool_rules(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.tools.rules is None:
            return {"error": "The rules index is unavailable here."}
        question = action["query"].strip()
        activity.tool("rules lookup", _clip(question, 80))
        names = [card["name"] for card in map(self.lookup, action["cards"][:6]) if card]
        evidence = self.tools.rules.retrieve(names, question, limit=12)
        return {"evidence": [
            {"id": e["id"], "kind": e["kind"], "text": _clip(e["text"], 600)} for e in evidence
        ]}

    def _tool_check_deck(self, action: dict[str, Any]) -> dict[str, Any]:
        swaps = _unique(action["swaps"])
        built = self.build(swaps)
        return {
            "swaps_applied": len(swaps), "valid": not built["errors"],
            "role_counts": built["playability"]["roles"]["role_counts"],
            "role_targets": self.spec.role_targets,
            "problems": built["errors"][:12],
            "lands": built["land_base"]["cards"],
        }

    def _tool_combos(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.tools.spellbook is None:
            return {"error": "Combo search is unavailable here."}
        nonlands, _ = self.apply(_unique(action["swaps"]))
        cards = {**nonlands, **dict.fromkeys(self.lands, 1)}
        activity.tool("commander spellbook", f"combos among {len(cards) + 1} cards")
        found = self.tools.spellbook.find_combos(cards, self.spec.commander)
        combos = [{"cards": c["cards"], "results": c["results"]} for c in found["combos"][:15]]
        return {"complete": found["complete"], "combos": combos}


def _unique(swaps: list[dict[str, str]]) -> list[dict[str, str]]:
    """Each card removed or added once; later swaps reusing a card are dropped."""
    used: set[str] = set()
    kept = []
    for swap in swaps:
        keys = {name_key(swap["remove"]), name_key(swap["add"])}
        if not keys & used and len(keys) == 2:
            used |= keys
            kept.append(swap)
    return kept


def _schema(session: _Session, final: bool) -> dict[str, Any]:
    removable, addable = sorted(session.nonlands), sorted(session.addable)
    swap = object_schema({
        "remove": {"type": "string", "enum": removable},
        "add": {"type": "string", "enum": addable} if addable else NAME_SCHEMA,
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    })
    return object_schema({
        "tool": {"type": "string", "enum": ["finish"] if final else [*TOOLS, "finish"]},
        # Every field is required (strict structured outputs); unused ones are left empty.
        "query": {"type": "string", "minLength": 0, "maxLength": 900},
        "cards": {"type": "array", "items": NAME_SCHEMA, "maxItems": 10},
        "swaps": {"type": "array", "items": swap, "maxItems": MAX_SWAPS if addable else 0},
        "reply": {"type": "string", "minLength": 0, "maxLength": 3000},
    })


def edit_deck(
    deck: dict[str, Any], pool: CandidatePool, model: JSONModel, request: str, *,
    tools: Toolbox, reviewer: DeckReviewer | None = None, max_steps: int = 10,
    max_repairs: int = 2,
) -> dict[str, Any]:
    if not request.strip():
        raise ValueError("A follow-up needs a request")
    if type(max_steps) is not int or not 2 <= max_steps <= 30:
        raise ValueError("max_steps must be from 2 to 30")
    session = _Session(deck, pool, tools, reviewer)
    named = []
    for name in dict.fromkeys(n.strip() for n in CARD_REF.findall(request)):
        card = session.lookup(name)
        named.append({
            "name": card["name"] if card else name,
            "addable": session.offer(card),
            "note": "" if card else "No such card in the local card data",
        })
    pool_rows = [
        {"name": c.card["name"], "mana_cost": c.card["mana_cost"], "roles": sorted(c.roles)}
        for c in session.pool.cards if c.card["name"] in session.addable
    ]
    payload = {
        "spec": session.spec.to_dict(),
        "plan": {k: deck["council"].get(k, []) for k in ("game_plan", "non_goals", "core",
                                                         "decisions")}
        if deck.get("council") else None,
        "engine_map": session.pool.engine_map or None,
        "deck": [
            {**trim_card(session.records[n]), "quantity": q,
             "roles": sorted(oracle_roles(session.records[n]))}
            for n, q in session.nonlands.items()
        ],
        "lands": {n: deck["cards"][n] for n in session.lands},
        "status": {
            "valid": deck.get("valid"), "role_counts": deck.get("role_counts"),
            "role_targets": session.spec.role_targets,
            "problems": deck.get("validation", {}).get("errors", [])[:12],
            "review": deck.get("review", {}).get("status"),
        },
        "pool": pool_rows, "named_cards": named,
        "conversation": deck.get("conversation", [])[-10:], "request": request,
    }
    messages = [
        {"role": "system", "content": prompts.EDIT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    used: list[str] = []
    repairs = 0
    for step in range(1, max_steps + 1):
        final = step == max_steps
        task = "read the request" if step == 1 else f"step {step} of {max_steps}"
        with activity.agent("orchestrator", model, task):
            action = request_json(model, messages, _schema(session, final))
        messages.append({"role": "assistant", "content": json.dumps(action)})
        if action["tool"] != "finish":
            used.append(action["tool"])
            result = session.run(action)
            messages.append({"role": "user", "content": json.dumps({
                "tool_result": action["tool"], "result": result, "steps_left": max_steps - step,
            })})
            continue
        swaps = _unique(action["swaps"])
        reply = action["reply"].strip() or ("Done." if swaps else "No changes.")
        if not swaps:
            return _result(deck, request, reply, [], used, step, model, built=None)
        built = session.build(swaps)
        review = {"status": "not_run", "errors": []}
        if not built["errors"] and reviewer is not None:
            try:
                review = reviewer.check(
                    built["cards"], session.spec.commander, session.spec.bracket,
                    bracket3_mode=session.spec.bracket3_mode,
                )
            except SchemaError as exc:
                review = {"status": "needs_revision", "errors": [{
                    "error": "review_schema", "schema_errors": exc.errors,
                    "detail": "Safety model returned invalid grounded JSON",
                }]}
            built["errors"].extend(review["errors"])
        built["review"] = review
        if not built["errors"] or final or repairs >= max_repairs:
            return _result(deck, request, reply, swaps, used, step, model, built=built)
        repairs += 1
        messages.append({"role": "user", "content": json.dumps({
            "problems": built["errors"][:12],
            "role_counts": built["playability"]["roles"]["role_counts"],
            "steps_left": max_steps - step,
            "instruction": "Your swaps left these problems. Fix them, using tools if useful, then "
            "finish again with the complete swaps from the original deck and an updated reply.",
        })})
    raise AssertionError("the last step always finishes")


def _result(
    deck: dict[str, Any], request: str, reply: str, swaps: list[dict[str, str]],
    used: list[str], steps: int, model: JSONModel, *, built: dict[str, Any] | None,
) -> dict[str, Any]:
    entry = {"player": request, "orchestrator": reply, "swaps": swaps}
    result = {
        **deck,
        "conversation": [*deck.get("conversation", []), entry],
        "edit": {
            "request": request, "reply": reply, "swaps": swaps, "tools": used, "steps": steps,
            "model": getattr(model, "model", None), "changed": built is not None,
        },
    }
    if built is not None:
        validation = built["validation"]
        validation.update(errors=built["errors"], valid=not built["errors"])
        result.update(
            valid=not built["errors"], cards=built["cards"], spec=built["spec"].to_dict(),
            role_counts=built["playability"]["roles"]["role_counts"],
            protection=built["playability"]["roles"]["protection"],
            playability=built["playability"], land_base=built["land_base"],
            validation=validation, review=built["review"],
            review_passed=built["review"]["status"] == "passed", asks=built["asks"],
        )
    return result
