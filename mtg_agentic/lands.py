"""Deterministic early-pip weighting, source floors, then exact basic-land fill."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .discovery import CandidatePool, eligible
from .land_options import preferred_lands
from .mana import (
    MIN_COLOR_SOURCES,
    direct_sources,
    front,
    is_land,
    land_sources,
    mana_demand,
    matches_land_search,
    search_selectors,
    source_graph,
    validate_mana_base,
)
from .oracle import OracleStore
from .spec import DeckSpec
from .validator import copy_limit

# Compatibility exports for callers of the original allocation module.
__all__ = ["allocate_sources", "calculate_land_base", "is_land", "land_sources", "mana_demand"]


def allocate_sources(weights: dict[str, float], total: int) -> dict[str, int]:
    """Largest remainder, with a one-source floor when slots permit."""
    if type(total) is not int or total < 0:
        raise ValueError("Source total must be a nonnegative integer")
    colors = [c for c in "WUBRGC" if weights.get(c, 0) > 0]
    if not colors:
        return {}
    allocations = {c: int(total >= len(colors)) for c in colors}
    remaining = total - sum(allocations.values())
    quotas = {c: remaining * weights[c] / sum(weights[c] for c in colors) for c in colors}
    for color in colors:
        allocations[color] += math.floor(quotas[color])
    for color in sorted(colors, key=lambda c: (-(quotas[c] % 1), "WUBRGC".index(c)))[:total - sum(allocations.values())]:
        allocations[color] += 1
    return allocations


def _land_speed(card: dict[str, Any]) -> float:
    text = front(card, "oracle_text")
    if not re.search(r"enters (?:the battlefield )?tapped", text, re.I):
        return 1.0
    # Shock lands can enter untapped. Do not treat them like Guildgates.
    if re.search(r"you may pay 2 life", text, re.I):
        return 0.95
    if re.search(r"unless|if you", text, re.I):
        return 0.85
    return 0.7


def calculate_land_base(spells: dict[str, int], spec: DeckSpec, pool: CandidatePool, store: OracleStore, *, game_changers: set[str] | None = None) -> dict[str, Any]:
    slots = spec.role_targets["lands"]
    weights: Counter[str] = Counter()
    for name, quantity in {**spells, spec.commander: 1}.items():
        if (card := store.get_card(name)) and not is_land(card):
            for color, weight in mana_demand(card).items():
                weights[color] += weight * quantity
    for color in spec.color_identity:
        weights[color] += 0.5
    targets = allocate_sources(dict(weights), slots)
    for color in spec.color_identity:
        targets[color] = max(targets.get(color, 0), MIN_COLOR_SOURCES)
    chosen: Counter[str] = Counter()
    options = {
        c.card["name"]: c.card for c in pool.cards if is_land(c.card) and eligible(c.card, spec)
    }
    # Saved pools may predate premium-land discovery. Code owns lands, so repair
    # their coverage locally instead of asking the model to change nonland buckets.
    for card in preferred_lands(store):
        if is_land(card) and eligible(card, spec):
            options[card["name"]] = card
    for name in spec.must_include:
        if (card := store.get_card(name)) and is_land(card):
            options[card["name"]] = card
    def changer(card):
        return card["name"] in game_changers if game_changers is not None else card["game_changer"]
    changers = sum(q for n, q in {**spells, spec.commander: 1}.items() if (card := store.get_card(n)) and changer(card))
    changer_limit = {1: 0, 2: 0, 3: 3, 4: 100, 5: 100}[spec.bracket]
    def add(card):
        nonlocal changers
        chosen[card["name"]] += 1
        changers += int(changer(card))
    def available(card):
        limit = copy_limit(card)
        return (
            (limit is None or chosen[card["name"]] < limit)
            and (not changer(card) or changers < changer_limit)
        )
    for name in spec.must_include:
        if (card := store.get_card(name)) and is_land(card):
            add(card)
    errors = []
    if sum(chosen.values()) > slots:
        errors.append({"error": "land_base", "detail": "Required lands exceed the land target"})
    basics = sorted((c for c in options.values() if copy_limit(c) is None), key=lambda c: c["name"])
    # Land tutors search types, not colors. Pin real targets before filling slots,
    # including for an off-color type that is available on an on-color dual land.
    searches = [
        (card["name"], selector)
        for name in [*spells, spec.commander, *chosen]
        if (card := store.get_card(name))
        for selector in search_selectors(card)
        if selector["supported"]
    ]
    while sum(chosen.values()) < slots:
        missing = [
            (name, selector) for name, selector in searches
            if not any(
                (n != name or chosen[n] > 1) and matches_land_search(options[n], selector)
                for n in chosen
            )
        ]
        if not missing:
            break
        targets_available = [
            c for c in options.values() if available(c)
            and any(
                (c["name"] != n or chosen[n] > 0) and matches_land_search(c, s)
                for n, s in missing
            )
        ]
        if not targets_available:
            break
        def target_score(card):
            coverage = sum(
                (card["name"] != n or chosen[n] > 0) and matches_land_search(card, s)
                for n, s in missing
            )
            colors = direct_sources(card, spec.color_identity) & set(weights)
            return (-coverage, -len(colors) * _land_speed(card), card["name"])
        add(min(targets_available, key=target_score))
    # Seed real targets before evaluating fetches. A hypothetical Forest is never fixing.
    for color in weights:
        if sum(chosen.values()) >= slots:
            break
        if not any(color in direct_sources(options[n], spec.color_identity) for n in chosen):
            basic = next((c for c in basics if available(c) and color in direct_sources(c, spec.color_identity)), None)
            if basic:
                add(basic)
    while sum(chosen.values()) < slots:
        counts = source_graph(chosen, spec.color_identity, store)["source_counts"]
        ranked = []
        for card in options.values():
            if copy_limit(card) is None or not available(card):
                continue
            trial = dict(chosen)
            trial[card["name"]] = 1
            graph = source_graph(trial, spec.color_identity, store)
            if graph["errors"]:
                continue
            potential = set(graph["land_sources"].get(card["name"], []))
            if len(potential & set(weights)) < 2:
                continue
            # Count improvements to existing fetches too: adding a typed dual can
            # improve several sources, not only the new land's own production.
            score = sum(
                min(
                    max(0, targets[c] - counts.get(c, 0)),
                    max(0, graph["source_counts"].get(c, 0) - counts.get(c, 0)),
                ) / targets[c]
                for c in targets
            ) * _land_speed(card)
            if score > 0:
                ranked.append((-score, card["name"], card))
        if not ranked:
            break
        add(min(ranked, key=lambda item: item[:2])[2])
    while sum(chosen.values()) < slots:
        counts = source_graph(chosen, spec.color_identity, store)["source_counts"]
        usable = [c for c in basics if available(c) and direct_sources(c, spec.color_identity) & set(weights)]
        if not usable:
            usable = [c for c in options.values() if available(c) and direct_sources(c, spec.color_identity) & set(weights)]
        if not usable:
            errors.append({"error": "land_base", "detail": "Not enough eligible mana-producing lands in the pool"})
            break
        def score(card):
            colors = direct_sources(card, spec.color_identity) & set(weights)
            floor_deficit = sum(max(0, MIN_COLOR_SOURCES - counts.get(c, 0)) for c in colors if c in spec.color_identity)
            target_deficit = sum(max(0, targets[c] - counts.get(c, 0)) / targets[c] for c in colors)
            return (-floor_deficit, -target_deficit, -sum(weights[c] / (counts.get(c, 0) + 1) for c in colors), card["name"])
        add(min(usable, key=score))
    mana = validate_mana_base({**spells, **chosen}, spec.commander, store)
    errors.extend(mana["errors"])
    return {
        "cards": dict(chosen), "land_count": sum(chosen.values()), "weighted_pips": dict(weights),
        "source_targets": targets, "source_counts": mana["source_counts"], "mana_validation": mana,
        "warnings": [
            "Target-aware source counts are not untapped-source or statistical consistency guarantees.",
            "Fetches sharing a target compete for it; source counts do not model draws or depletion.",
            "Opponent-dependent and colored-input filter abilities are not credited toward hard floors.",
            "Premium lands use local Oracle data; prices, ownership and budget are not modeled.",
        ] + (["Weighted source targets not all met"] if any(mana["source_counts"].get(c, 0) < n for c, n in targets.items()) else []),
        "errors": errors,
    }
