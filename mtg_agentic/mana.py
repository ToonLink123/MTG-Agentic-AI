"""Conservative land-source graphs. Fetches need actual, reachable deck targets."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .cards import Card
from .oracle import OracleStore

BASIC_LANDS = dict(zip("WUBRGC", ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes")))
SYMBOL = re.compile(r"\{([^}]+)\}")
MIN_COLOR_SOURCES = 10


def front(card: Card, field: str) -> str:
    faces = card.get("card_faces", [])
    return faces[0].get(field, "") if faces else card.get(field, "").split(" // ")[0]


def is_land(card: Card) -> bool:
    return "Land" in front(card, "type_line").split("—")[0].split()


def _symbol_demand(cost: str) -> dict[str, float]:
    demand: Counter[str] = Counter()
    for symbol in SYMBOL.findall(cost):
        colors = [s for s in symbol.split("/") if s in set("WUBRGC")]
        for color in colors:
            demand[color] += (
                (0.5 if "P" in symbol or symbol.startswith("2/") else 1) / len(colors)
            )
    return dict(demand)


def mana_pips(card: Card) -> dict[str, float]:
    """Unweighted front-face casting pips; generic costs do not require {C}."""
    return _symbol_demand(front(card, "mana_cost"))


def mana_demand(card: Card) -> dict[str, float]:
    symbols = SYMBOL.findall(front(card, "mana_cost"))
    derived = sum(
        int(s) if s.isdigit() else 0 if s in {"X", "Y", "Z"} else 2 if s.startswith("2/") else 1
        for s in symbols
    )
    mv = derived if card.get("card_faces") else card.get("cmc")
    mv = mv if isinstance(mv, (float, int)) else derived
    weight = 3.0 if mv <= 2 else 2.0 if mv <= 4 else 1.0
    return {color: pips * weight for color, pips in mana_pips(card).items()}


def mana_profile(
    spells: Mapping[str, int], commander: str, store: OracleStore
) -> dict[str, Any]:
    """Count actual casting and recognized ability costs, including the commander once."""
    pips: Counter[str] = Counter()
    abilities: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    for name, quantity in {**spells, commander: 1}.items():
        if type(quantity) is not int or quantity <= 0:
            continue
        card = store.get_card(name)
        if card is None or is_land(card):
            continue
        for color, amount in mana_pips(card).items():
            pips[color] += amount * quantity
        for color, amount in mana_demand(card).items():
            weights[color] += amount * quantity
        text = front(card, "oracle_text")
        costs = re.findall(r"(?:^|[.\n])\s*([^:.\n]+):", text)
        costs += re.findall(r"\bpay ((?:\{[^}]+\})+) rather than pay\b", text, re.I)
        for cost in costs:
            for color, amount in _symbol_demand(cost).items():
                abilities[color] += amount * quantity
                weights[color] += amount * quantity
    leader = store.get_card(commander)
    identity = "".join(c for c in "WUBRG" if leader and c in leader["color_identity"]) or "C"
    fallback = not weights
    if fallback:
        weights.update({color: 1.0 for color in identity})
    colors = [color for color in "WUBRG" if weights[color] > 0]
    total = sum(weights.values())
    return {
        "mana_pips": dict(pips),
        "ability_pips": dict(abilities),
        "weighted_pips": dict(weights),
        "demand_share": {c: weights[c] / total for c in "WUBRGC" if weights[c] > 0},
        "colors": colors,
        "strategy": "multicolor" if len(colors) > 1 else "monocolor" if colors else "colorless",
        "identity_fallback": fallback,
    }


def direct_sources(card: Card, identity: str, *, free: bool = False) -> set[str]:
    """Do not credit opponent-dependent, conditional, spend-restricted or colored-input filter
    abilities. `free` also drops mana that costs mana or another permanent to make, such as
    {1}, {T}: Add one mana of any color."""
    if not is_land(card):
        return set()
    allowed = set(identity) | {"C"}
    types = front(card, "type_line").split()
    sources = {c for c, basic in BASIC_LANDS.items() if c != "C" and basic in types}
    text = front(card, "oracle_text")
    for ability in text.split("\n"):
        # A typed dual's whole line is reminder text, "({T}: Add {G} or {U}.)"; inside a line
        # it explains something else, like a Treasure token's mana.
        if not ability.startswith("("):
            ability = re.sub(r"\([^)]*\)", "", ability)
        # The restriction is its own sentence after the mana: "Add one mana of any color.
        # Spend this mana only to cast a creature spell" or "Activate only if you control an
        # artifact."
        if re.search(r"spend this mana only|activate only if", ability, re.I):
            continue
        for clause in ability.split("."):
            # "If it has a luck counter on it, instead add one mana of any color."
            if not re.search(r"\badd\b", clause, re.I) or re.match(r"\s*if\b", clause, re.I):
                continue
            cost, _, output = clause.partition(":")
            if not output:
                output, cost = cost, ""
            paid = SYMBOL.findall(cost)
            if any(set(s.split("/")) & set("WUBRG") for s in paid):
                continue
            if free and (
                any(s not in {"T", "Q", "E"} for s in paid)
                or re.search(r"\bsacrifice an?\b", cost, re.I)
            ):
                continue
            # "Any color among legendary permanents you control" depends on the board.
            if re.search(r"(?:could produce|opponent|spend this mana only|\bamong\b)", output, re.I):
                continue
            if re.search(r"mana of any (?:color|type)", output, re.I):
                sources.update(set(identity) - {"C"})
                if "any type" in output.lower():
                    sources.add("C")
            sources.update(s for s in SYMBOL.findall(output) if s in set("WUBRGC"))
    # Legacy records without Oracle production clauses. Never use fetch metadata as production.
    if not text.strip():
        sources.update(card.get("produced_mana") or [])
    return sources & allowed


def search_selectors(card: Card) -> list[dict[str, Any]]:
    """Recognize basic/type/land searches only; unknown restrictions fail closed."""
    selectors = []
    for match in re.finditer(r"search your library for ([^.\n]+?)\bcards?\b", front(card, "oracle_text"), re.I):
        phrase = match[1].lower()
        trailing = front(card, "oracle_text")[match.end():].split(",")[0].split(".")[0]
        types = [t for t in BASIC_LANDS.values() if t != "Wastes" and re.search(rf"\b{t.lower()}\b", phrase)]
        if not types and not re.search(r"\bland\b", phrase):
            continue
        supported = re.sub(r"with a basic land type", "", phrase)
        unknown = bool(re.search(r"\b(?:with|named|that|non-|except)\b", supported) or re.match(r"\s+(?:with|named|that|except)\b", trailing, re.I))
        selectors.append({
            "types": types,
            "basic": bool(re.search(r"\bbasic\b", phrase)) and "with a basic land type" not in phrase,
            "nonbasic": "nonbasic" in phrase,
            "snow": bool(re.search(r"\bsnow\b", phrase)),
            "typed": "with a basic land type" in phrase,
            "supported": not unknown,
        })
    return selectors


def matches_land_search(target: Card, selector: dict[str, Any]) -> bool:
    """Match actual land types/supertypes, not the colors a land can produce."""
    if not is_land(target):
        return False
    types = front(target, "type_line").split()
    return bool(
        selector["supported"]
        and (not selector["basic"] or "Basic" in types)
        and (not selector["nonbasic"] or "Basic" not in types)
        and (not selector["snow"] or "Snow" in types)
        and (not selector["types"] or set(types) & set(selector["types"]))
        and (not selector["typed"] or set(types) & (set(BASIC_LANDS.values()) - {"Wastes"}))
    )


def source_graph(deck: Mapping[str, int], identity: str, store: OracleStore) -> dict[str, Any]:
    lands = {
        card["name"]: card for name, quantity in deck.items()
        if type(quantity) is int and quantity > 0
        and (card := store.get_card(name)) and is_land(card)
    }
    quantities: Counter[str] = Counter()
    for name, quantity in deck.items():
        if type(quantity) is int and quantity > 0 and (card := store.get_card(name)) and card["name"] in lands:
            quantities[card["name"]] += quantity
    edges = {
        name: sorted({
            target for selector in search_selectors(card)
            for target, target_card in lands.items()
            if (target != name or quantities[name] > 1) and matches_land_search(target_card, selector)
        })
        for name, card in lands.items()
    }
    sources = {name: direct_sources(card, identity) for name, card in lands.items()}
    # Fixed point, not recursive optimistic union: a fetch-only cycle has no anchor.
    for _ in lands:
        updated = {n: sources[n] | set().union(*(sources[t] for t in edges[n])) for n in lands}
        if updated == sources:
            break
        sources = updated
    counts: Counter[str] = Counter()
    for name, colors in sources.items():
        for color in colors:
            counts[color] += quantities[name]
    searches = {}
    errors = []
    for name, quantity in deck.items():
        if type(quantity) is not int or quantity <= 0 or not (card := store.get_card(name)):
            continue
        selectors = search_selectors(card)
        if not selectors:
            continue
        matches = [
            sorted(
                n for n, c in lands.items()
                if matches_land_search(c, selector)
                and (n != card["name"] or quantities[n] > 1)
            )
            for selector in selectors
        ]
        targets = sorted({n for group in matches for n in group})
        searches[card["name"]] = targets
        if any(not s["supported"] for s in selectors):
            errors.append({"error": "unsupported_land_search", "card": card["name"], "detail": "Cannot certify this Oracle land-search restriction"})
        elif any(not group for group in matches):
            errors.append({"error": "land_search_without_targets", "card": card["name"], "detail": "Oracle land search has no matching land in the actual deck", "missing_selectors": [s for s, group in zip(selectors, matches) if not group]})
    return {
        "source_counts": dict(counts), "land_sources": {n: sorted(v) for n, v in sources.items()},
        "fetch_edges": edges, "search_targets": searches, "errors": errors,
    }


def validate_mana_base(deck: Mapping[str, int], commander: str, store: OracleStore) -> dict[str, Any]:
    leader = store.get_card(commander)
    identity = "".join(c for c in "WUBRG" if leader and c in leader["color_identity"]) or "C"
    graph = source_graph(deck, identity, store)
    profile = mana_profile(deck, commander, store)
    minimums = {c: MIN_COLOR_SOURCES for c in profile["colors"]}
    if profile["weighted_pips"].get("C", 0):
        minimums["C"] = MIN_COLOR_SOURCES if profile["strategy"] == "colorless" else 1
    errors = list(graph["errors"])
    for color, minimum in minimums.items():
        actual = graph["source_counts"].get(color, 0)
        if actual < minimum:
            errors.append({"error": "mana_sources", "color": color, "detail": f"Need at least {minimum} real {color} land sources; got {actual}"})
    return {"valid": not errors, "minimums": minimums, **graph, "errors": errors}


def land_sources(card: Card, identity: str, deck: Mapping[str, int] | None = None, store: OracleStore | None = None) -> set[str]:
    """Without a deck, return direct production only. Fetch credit needs both arguments."""
    if deck is None or store is None:
        return direct_sources(card, identity)
    return set(source_graph(deck, identity, store)["land_sources"].get(card["name"], []))
