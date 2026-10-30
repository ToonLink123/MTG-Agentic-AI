"""Follow-ups: the player talks to the orchestrator alone, never the council.

The orchestrator works in a bounded loop. Each step it calls tools (card search, card text,
rulings, rules lookup, a dry-run deck check, a combo search) or finishes with a reply and
one-for-one nonland swaps. Code applies the swaps exactly, regenerates the lands, validates and
runs the safety review; any problems go back to the orchestrator, which may use more tools and
finish again. A question that needs no swaps changes nothing and rebuilds nothing. A model with
native tool use (Claude) calls the tools directly, several in one turn; any other model names
one tool per step in its JSON answer.

polish_deck runs the same loop once, unasked, before the player sees a new deck.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from . import activity, prompts
from .agent import Finish, Tool, ToolError, run_tools, supports_tools
from .cards import name_key, trim_card
from .discovery import CandidatePool
from .lands import is_land
from .llm import JSONModel, request_json
from .pipeline import ask_report, check_cards, game_changer_names, make_lands
from .review import DeckReviewer
from .roles import oracle_roles
from .schema import NAME_SCHEMA, SchemaError, object_schema
from .scryfall import ScryfallError
from .sources import SourceError
from .spec import DeckSpec
from .toolbox import CARDS_INPUT, RULES_INPUT, SEARCH_INPUT, SIMILAR_INPUT, CardTools, Toolbox

__all__ = ["Toolbox", "edit_deck", "polish_deck"]

TOOLS = ("search_cards", "card", "similar_cards", "rulings", "rules", "check_deck", "combos")
CARD_REF = re.compile(r"\[\[(.+?)\]\]")
MAX_SWAPS = 15
POLISH_SWAPS = 8
REASON = {"type": "string", "minLength": 1, "maxLength": 300}
SWAP_SCHEMA = object_schema({"remove": NAME_SCHEMA, "add": NAME_SCHEMA, "reason": REASON})
SWAPS_INPUT = object_schema({"swaps": {"type": "array", "items": SWAP_SCHEMA, "maxItems": MAX_SWAPS}})
FINISH_INPUT = object_schema({
    "reply": {"type": "string", "minLength": 1, "maxLength": 3000},
    "swaps": {"type": "array", "items": SWAP_SCHEMA, "maxItems": MAX_SWAPS},
})


class _Session(CardTools):
    """One follow-up: the deck before it, the cards that may be added, and the tools."""

    def __init__(self, deck: dict[str, Any], pool: CandidatePool, tools: Toolbox, reviewer):
        spec = DeckSpec.from_dict(deck["spec"]) if "spec" in deck else pool.spec
        super().__init__(tools, spec)
        self.reviewer = reviewer
        self.pool = dataclasses.replace(pool, spec=spec)
        records = {name: self.store.get_card(name) for name in deck["cards"]}
        unknown = [name for name, card in records.items() if card is None]
        if unknown:
            raise ValueError(f"Deck has cards missing from the local Oracle data: {unknown[:5]}")
        self.nonlands = {n: q for n, q in deck["cards"].items() if not is_land(records[n])}
        self.lands = sorted(n for n in deck["cards"] if is_land(records[n]))
        self.records = records
        self.addable = {
            c.card["name"]: c.card for c in self.pool.cards
            if not is_land(c.card) and c.card["name"] not in self.nonlands
        }
        self.changers = game_changer_names(self.pool, self.store, reviewer)

    def taken(self, name: str) -> bool:
        return name in self.nonlands

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

    def review(self, built: dict[str, Any]) -> dict[str, Any]:
        """The safety review of a clean build; its problems join the build's."""
        review = {"status": "not_run", "errors": []}
        if not built["errors"] and self.reviewer is not None:
            try:
                review = self.reviewer.check(
                    built["cards"], self.spec.commander, self.spec.bracket,
                    bracket3_mode=self.spec.bracket3_mode, engine_map=self.pool.engine_map or None,
                )
            except SchemaError as exc:
                review = {"status": "needs_revision", "errors": [{
                    "error": "review_schema", "schema_errors": exc.errors,
                    "detail": "Safety model returned invalid grounded JSON",
                }]}
            built["errors"].extend(review["errors"])
        built["review"] = review
        return built

    def unknown_swaps(self, swaps: list[dict[str, str]]) -> list[str]:
        """Swaps that remove a card not in the deck or add one that is not addable."""
        return [
            f"{s['remove']} -> {s['add']}: "
            + ("remove a nonland in the deck" if s["remove"] not in self.nonlands
               else "add a card a search, card lookup or the pool made addable")
            for s in swaps if s["remove"] not in self.nonlands or s["add"] not in self.addable
        ]

    def run(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = action["tool"]
        try:
            return getattr(self, f"_tool_{tool}")(action)
        except (ScryfallError, SourceError, ValueError) as exc:
            return {"error": str(exc)}

    def _tool_search_cards(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.search_cards(action["query"])

    def _tool_card(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.card(action["cards"])

    def _tool_similar_cards(self, action: dict[str, Any]) -> dict[str, Any]:
        if not action["cards"]:
            return {"error": "Name the card to find similar cards for in cards."}
        return self.similar_cards(action["cards"][0])

    def _tool_rulings(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.rulings(action["cards"])

    def _tool_rules(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.rules(action["query"], action["cards"])

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
        nonlands, _ = self.apply(_unique(action["swaps"]))
        return self.combos({**nonlands, **dict.fromkeys(self.lands, 1)})


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
        "reason": REASON,
    })
    return object_schema({
        "tool": {"type": "string", "enum": ["finish"] if final else [*TOOLS, "finish"]},
        # Every field is required (strict structured outputs); unused ones are left empty.
        "query": {"type": "string", "minLength": 0, "maxLength": 900},
        "cards": {"type": "array", "items": NAME_SCHEMA, "maxItems": 10},
        "swaps": {"type": "array", "items": swap, "maxItems": MAX_SWAPS if addable else 0},
        "reply": {"type": "string", "minLength": 0, "maxLength": 3000},
    })


def _payload(session: _Session, deck: dict[str, Any], request: str) -> dict[str, Any]:
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
    # Cards code filled in to reach role counts, rather than a drafter's picks.
    padded = set(deck.get("assembly", {}).get("code_added", []))
    return {
        "spec": session.spec.to_dict(),
        "plan": {k: deck["council"].get(k, []) for k in ("game_plan", "non_goals", "core",
                                                         "decisions")}
        if deck.get("council") else None,
        "engine_map": session.pool.engine_map or None,
        "deck": [
            {**trim_card(session.records[n]), "quantity": q,
             "roles": sorted(oracle_roles(session.records[n])),
             **({"picked_by": "code"} if n in padded else {})}
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


def edit_deck(
    deck: dict[str, Any], pool: CandidatePool, model: JSONModel, request: str, *,
    tools: Toolbox, reviewer: DeckReviewer | None = None, max_steps: int = 10,
    max_repairs: int = 2, max_swaps: int = MAX_SWAPS,
) -> dict[str, Any]:
    if not request.strip():
        raise ValueError("A follow-up needs a request")
    if type(max_steps) is not int or not 2 <= max_steps <= 30:
        raise ValueError("max_steps must be from 2 to 30")
    session = _Session(deck, pool, tools, reviewer)
    payload = _payload(session, deck, request)
    if supports_tools(model):
        return _edit_with_tools(session, deck, payload, model, request, max_steps, max_repairs, max_swaps)
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
        swaps = _unique(action["swaps"])[:max_swaps]
        reply = action["reply"].strip() or ("Done." if swaps else "No changes.")
        if not swaps:
            return _result(deck, request, reply, [], used, step, model, built=None)
        built = session.review(session.build(swaps))
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


def _edit_with_tools(
    session: _Session, deck: dict[str, Any], payload: dict[str, Any], model: Any, request: str,
    max_steps: int, max_repairs: int, max_swaps: int,
) -> dict[str, Any]:
    """The follow-up on native tool use: the same tools, called directly, several per turn."""
    repairs = 0

    def guarded(run):
        def call(args: dict[str, Any]) -> Any:
            try:
                return run(args)
            except (ScryfallError, SourceError, ValueError) as exc:
                raise ToolError({"error": str(exc)}) from None
        return call

    def swaps_of(args: dict[str, Any]) -> list[dict[str, str]]:
        swaps = _unique(args["swaps"])
        unknown = session.unknown_swaps(swaps)
        if unknown:
            raise ToolError({"problems": unknown})
        return swaps

    def check_deck(args: dict[str, Any]) -> dict[str, Any]:
        return session._tool_check_deck({"swaps": swaps_of(args)})

    def combos(args: dict[str, Any]) -> dict[str, Any]:
        nonlands, _ = session.apply(swaps_of(args))
        return session.combos({**nonlands, **dict.fromkeys(session.lands, 1)})

    def finish(args: dict[str, Any]) -> Finish:
        nonlocal repairs
        swaps = swaps_of(args)
        if len(swaps) > max_swaps:
            raise ToolError({"problems": [f"At most {max_swaps} swaps; you sent {len(swaps)}."]})
        reply = args["reply"].strip()
        if not swaps:
            return Finish((reply, [], None))
        built = session.review(session.build(swaps))
        if built["errors"] and repairs < max_repairs:
            repairs += 1
            raise ToolError({
                "problems": built["errors"][:12],
                "role_counts": built["playability"]["roles"]["role_counts"],
                "instruction": "Your swaps left these problems. Fix them, using tools if useful, "
                "then finish again with the complete swaps from the original deck and an "
                "updated reply.",
            })
        return Finish((reply, swaps, built))

    tools = [
        Tool("search_cards", "A search in Scryfall syntax over the local card data, already "
             "limited to the deck's colors and Commander legality. Addable results can be swapped "
             "in.", SEARCH_INPUT, guarded(lambda a: session.search_cards(a["query"]))),
        Tool("card", "Oracle text, roles and whether each named card can be added, up to 10.",
             CARDS_INPUT, guarded(lambda a: session.card(a["cards"]))),
        Tool("similar_cards", "Cards that do what the named card does (shared function tags, "
             "keywords and combos), not in the deck; each can be swapped in.", SIMILAR_INPUT,
             guarded(lambda a: session.similar_cards(a["card"]))),
        Tool("rulings", "Publisher rulings for up to 5 named cards.", CARDS_INPUT,
             guarded(lambda a: session.rulings(a["cards"]))),
        Tool("rules", "The Comprehensive Rules and rulings most relevant to a rules question "
             "about the named cards.", RULES_INPUT,
             guarded(lambda a: session.rules(a["query"], a["cards"]))),
        Tool("check_deck", "Dry run of swaps: regenerated lands, role counts and problems.",
             SWAPS_INPUT, guarded(check_deck)),
        Tool("combos", "Commander Spellbook combos in the deck after the swaps.", SWAPS_INPUT,
             guarded(combos)),
        Tool("finish", "Your answer to the player and the final swaps (none for a question). "
             "Code applies them and checks the deck; problems come back to fix.",
             FINISH_INPUT, finish),
    ]
    messages = [
        {"role": "system", "content": prompts.EDIT_NATIVE},
        {"role": "user", "content": json.dumps(payload)},
    ]
    done = run_tools(model, messages, tools, max_steps=max_steps, agent="orchestrator",
                     task="read the request")
    if done["value"] is None:
        reply = "I ran out of steps before finishing, so nothing changed."
        return _result(deck, request, reply, [], done["tools"], done["steps"], model, built=None)
    reply, swaps, built = done["value"]
    reply = reply or ("Done." if swaps else "No changes.")
    return _result(deck, request, reply, swaps, done["tools"], done["steps"], model, built=built)


def polish_deck(
    deck: dict[str, Any], pool: CandidatePool, model: JSONModel, *, tools: Toolbox,
    reviewer: DeckReviewer | None = None, max_steps: int = 8,
) -> dict[str, Any]:
    """One follow-up nobody asked for, before the player sees a new deck: the orchestrator
    tightens it toward the plan, replacing cards code filled in that do not serve it. The
    polished deck is kept only when it still passes every check; otherwise the original stays."""
    if not deck.get("valid"):
        return deck
    result = edit_deck(deck, pool, model, prompts.POLISH, tools=tools, reviewer=reviewer,
                       max_steps=max_steps, max_repairs=1, max_swaps=POLISH_SWAPS)
    edit = result.pop("edit")
    note = {key: edit[key] for key in ("reply", "swaps", "tools", "steps", "model")}
    if edit["changed"] and not result.get("valid"):
        return {**deck, "polish": {**note, "kept": False,
                                   "reason": "The polished deck failed its checks; the draft stays."}}
    result["conversation"] = deck.get("conversation", [])
    if edit["changed"] and result.get("assembly"):
        added = [n for n in result["assembly"].get("code_added", []) if n in result["cards"]]
        result["assembly"] = {**result["assembly"], "code_added": added}
    return {**result, "polish": {**note, "kept": edit["changed"]}}


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
