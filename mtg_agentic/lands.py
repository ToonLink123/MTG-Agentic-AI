"""Deterministic, early-pip-weighted land allocation, not gameplay simulation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .cards import Card, name_key
from .discovery import BASIC_LANDS, CandidatePool, eligible
from .oracle import OracleStore
from .spec import DeckSpec
from .validator import copy_limit

SYMBOL = re.compile(r"\{([^}]+)\}")


def is_land(card: Card) -> bool:
    faces = card.get("card_faces", [])
    line = faces[0]["type_line"] if faces else card["type_line"].split(" // ")[0]
    return "Land" in line.split("—")[0].split()


def mana_demand(card: Card) -> dict[str, float]:
    """Front-face costs; early spells get more weight, hybrids split their demand."""
    cost = card["mana_cost"].split(" // ")[0]
    symbols = SYMBOL.findall(cost)
    derived_mv = sum(
        int(symbol)
        if symbol.isdigit()
        else 0
        if symbol in {"X", "Y", "Z"}
        else 2
        if symbol.startswith("2/")
        else 1
        for symbol in symbols
    )
    mv = card.get("cmc")
    mv = mv if isinstance(mv, (float, int)) else derived_mv
    early_weight = 3.0 if mv <= 2 else 2.0 if mv <= 4 else 1.0
    demand: Counter[str] = Counter()
    for symbol in symbols:
        colors = [part for part in symbol.split("/") if part in "WUBRGC" and len(part) == 1]
        if not colors:
            continue
        optional = 0.5 if "P" in symbol or symbol.startswith("2/") else 1.0
        for color in colors:
            demand[color] += early_weight * optional / len(colors)
    return dict(demand)


def land_sources(card: Card, identity: str) -> set[str]:
    """Potential sources, not a guarantee that a conditional land will be active."""
    allowed = set(identity) | {"C"}
    sources: set[str] = set()
    types = card["type_line"].split(" // ")[0]
    for color, basic in BASIC_LANDS.items():
        if color != "C" and basic in types.split():
            sources.add(color)
    produced = card.get("produced_mana")
    if isinstance(produced, list):
        sources.update(produced)
    else:
        # Older snapshots may predate the produced_mana projection.
        text = card["oracle_text"]
        if re.search(r"(?:one mana|mana) of any (?:color|type)", text, re.IGNORECASE):
            sources.update(set(identity) - {"C"})
            if "any type" in text.lower():
                sources.add("C")
        for clause in re.findall(r"\b[Aa]dd ([^.\n]+)", text):
            sources.update(
                part for part in SYMBOL.findall(clause) if part in "WUBRGC" and len(part) == 1
            )
        if "search your library" in text.lower():
            # Fetch targets are only credited for identifiable basic land types.
            for color, basic in BASIC_LANDS.items():
                if color != "C" and basic.lower() in text.lower():
                    sources.add(color)
            if "basic land card" in text.lower():
                sources.update(set(identity) - {"C"})
    return sources & allowed


def allocate_sources(weights: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder allocation with a one-source floor when slots permit."""
    if type(total) is not int or total < 0:
        raise ValueError("Source total must be a nonnegative integer")
    colors = [c for c in "WUBRGC" if weights.get(c, 0) > 0]
    if not colors:
        return {}
    allocations = {c: int(total >= len(colors)) for c in colors}
    remaining = total - sum(allocations.values())
    weight_sum = sum(weights[c] for c in colors)
    quotas = {c: remaining * weights[c] / weight_sum for c in colors}
    for color in colors:
        allocations[color] += math.floor(quotas[color])
    for color in sorted(colors, key=lambda c: (-(quotas[c] % 1), "WUBRGC".index(c)))[
        : total - sum(allocations.values())
    ]:
        allocations[color] += 1
    return allocations


def calculate_land_base(
    spells: dict[str, int],
    spec: DeckSpec,
    pool: CandidatePool,
    store: OracleStore,
) -> dict[str, Any]:
    slots = spec.role_targets["lands"]
    weights: Counter[str] = Counter()
    for name, quantity in {**spells, spec.commander: 1}.items():
        card = store.get_card(name)
        if card and not is_land(card):
            for color, weight in mana_demand(card).items():
                weights[color] += weight * quantity
    for color in spec.color_identity:
        # A small floor keeps all commander colors represented even without colored spells.
        weights[color] += 0.5
    targets = allocate_sources(dict(weights), slots)
    sources: Counter[str] = Counter()
    chosen: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    options = [
        candidate.card
        for candidate in pool.cards
        if is_land(candidate.card) and eligible(candidate.card, spec)
    ]
    by_name = {name_key(card["name"]): card for card in options}
    for name in spec.must_include:
        card = store.get_card(name)
        if card and is_land(card):
            by_name[name_key(card["name"])] = card
    options = list(by_name.values())
    changer_limit = {1: 0, 2: 0, 3: 3, 4: 100, 5: 100}[spec.bracket]
    changers = sum(
        quantity
        for name, quantity in {**spells, spec.commander: 1}.items()
        if (card := store.get_card(name)) and card["game_changer"]
    )

    def add(card: Card) -> None:
        nonlocal changers
        chosen[card["name"]] += 1
        sources.update(land_sources(card, spec.color_identity))
        changers += int(card["game_changer"])

    for name in spec.must_include:
        card = store.get_card(name)
        if card and is_land(card):
            add(card)
    if sum(chosen.values()) > slots:
        errors.append({"error": "land_base", "detail": "Required lands exceed the land target"})
    # Favor fixing where weighted source deficits justify it; never select duplicate nonbasics.
    while sum(chosen.values()) < slots:
        ranked: list[tuple[float, str, Card]] = []
        for card in options:
            if copy_limit(card) is None or chosen[card["name"]]:
                continue
            if card["game_changer"] and changers >= changer_limit:
                continue
            potential = land_sources(card, spec.color_identity)
            score = sum(
                max(0, targets.get(c, 0) - sources[c]) / max(1, targets.get(c, 0))
                for c in potential
            )
            if re.search(r"enters (?:the battlefield )?tapped", card["oracle_text"], re.IGNORECASE):
                score *= 0.75
            if score > 0:
                ranked.append((score, card["name"], card))
        if not ranked:
            break
        best = sorted(ranked, key=lambda entry: (-entry[0], entry[1]))[0][2]
        # A one-color nonbasic is not better than the matching basic in this heuristic.
        if len(land_sources(best, spec.color_identity)) < 2:
            break
        add(best)
    basics = [card for card in options if copy_limit(card) is None]
    while sum(chosen.values()) < slots:
        usable = [card for card in basics if land_sources(card, spec.color_identity) & set(weights)]
        if not usable:
            # If basics are excluded, use remaining singleton lands with known mana production.
            usable = [
                card
                for card in options
                if not chosen[card["name"]]
                and land_sources(card, spec.color_identity) & set(weights)
                and (not card["game_changer"] or changers < changer_limit)
            ]
        if not usable:
            errors.append(
                {
                    "error": "land_base",
                    "detail": "Not enough eligible mana-producing lands in the pool",
                }
            )
            break

        def score(card: Card) -> tuple[float, str]:
            production = land_sources(card, spec.color_identity)
            benefit = sum(weights.get(c, 0) / (sources[c] + 1) for c in production)
            return (-benefit, card["name"])

        add(min(usable, key=score))
    warnings = [
        "Source counts describe potential production, not guaranteed untapped or active sources.",
        "This is a pip-weighted heuristic, not a Karsten probability calculation or simulation.",
    ]
    unmet = [c for c in targets if sources[c] < targets[c]]
    if unmet:
        warnings.append("Source targets not met for: " + "".join(unmet))
    return {
        "cards": dict(chosen),
        "land_count": sum(chosen.values()),
        "weighted_pips": dict(weights),
        "source_targets": targets,
        "source_counts": dict(sources),
        "warnings": warnings,
        "errors": errors,
    }
