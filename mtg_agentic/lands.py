"""Independent pip-driven mana-base generation, without an LLM or spell pool."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .land_options import is_utility_land, mana_land_options
from .mana import (
    MIN_COLOR_SOURCES,
    direct_sources,
    front,
    is_land,
    land_sources,
    mana_demand,
    mana_profile,
    matches_land_search,
    search_selectors,
    source_graph,
    validate_mana_base,
)
from .oracle import OracleStore
from .spec import DeckSpec, ground_spec
from .validator import copy_limit

if TYPE_CHECKING:
    from .discovery import CandidatePool

# Compatibility exports for callers of the original allocation module.
__all__ = [
    "allocate_sources", "calculate_land_base", "generate_mana_base",
    "is_land", "land_sources", "mana_demand",
]


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
    if re.search(r"put (?:it|that card) onto the battlefield tapped", text, re.I):
        return 0.85 if re.search(r"untap (?:it|that land)", text, re.I) else 0.7
    if not re.search(r"enters (?:the battlefield )?tapped", text, re.I):
        return 1.0
    # Shock lands can enter untapped. Do not treat them like Guildgates.
    if re.search(r"you may pay 2 life", text, re.I):
        return 0.95
    if re.search(r"unless|if you", text, re.I):
        return 0.85
    return 0.7


def calculate_land_base(spells: dict[str, int], spec: DeckSpec, pool: CandidatePool, store: OracleStore, *, game_changers: set[str] | None = None) -> dict[str, Any]:
    """Compatibility adapter; the generator itself has no candidate-pool dependency."""
    return generate_mana_base(
        spells, spec, store,
        candidate_names=(c.card["name"] for c in pool.cards if is_land(c.card)),
        game_changers=game_changers,
    )


def generate_mana_base(
    spells: Mapping[str, int],
    spec: DeckSpec,
    store: OracleStore,
    *,
    candidate_names: Iterable[str] = (),
    game_changers: set[str] | None = None,
) -> dict[str, Any]:
    """Build lands from the finished nonlands and commander, using local Oracle data."""
    spec = ground_spec(spec, store)
    for name, quantity in spells.items():
        card = store.get_card(name)
        if type(quantity) is not int or quantity <= 0 or card is None or is_land(card):
            raise ValueError("Mana-base generation requires known nonland cards and positive quantities")
    slots = spec.role_targets["lands"]
    profile = mana_profile(spells, spec.commander, store)
    weights = profile["weighted_pips"]
    strategy = profile["strategy"]
    utility_limit = min(2, slots // 12) if strategy == "multicolor" else min(4, slots // 6)
    targets = allocate_sources(dict(weights), slots)
    # Single-source allocations are not consistency targets for a multicolor
    # deck. In particular, Jodah needs every color even when red pips are rare.
    fixing_floor = math.ceil(slots / 2) if len(profile["colors"]) >= 3 else MIN_COLOR_SOURCES
    for color in profile["colors"]:
        targets[color] = max(targets.get(color, 0), fixing_floor)
    if len(weights) == 1:
        color = next(iter(weights))
        targets[color] = max(MIN_COLOR_SOURCES, slots - utility_limit)
    chosen: Counter[str] = Counter()
    options = mana_land_options(spec, store, candidate_names)
    def changer(card):
        return card["name"] in game_changers if game_changers is not None else card["game_changer"]
    changers = sum(q for n, q in {**spells, spec.commander: 1}.items() if (card := store.get_card(n)) and changer(card))
    changer_limit = {1: 0, 2: 0, 3: 3, 4: 100, 5: 100}[spec.bracket]
    reasons = {}
    def add(card, reason):
        nonlocal changers
        chosen[card["name"]] += 1
        reasons.setdefault(card["name"], reason)
        changers += int(changer(card))
    def available(card):
        limit = copy_limit(card)
        return (
            (limit is None or chosen[card["name"]] < limit)
            and (not changer(card) or changers < changer_limit)
        )
    for name in spec.must_include:
        if (card := store.get_card(name)) and is_land(card):
            options[card["name"]] = card
            add(card, "required")
    errors = []
    if sum(chosen.values()) > slots:
        errors.append({"error": "land_base", "detail": "Required lands exceed the land target"})
    if changers > changer_limit:
        errors.append({"error": "land_base", "detail": "Required cards exceed the bracket Game Changer allowance"})
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
        add(min(targets_available, key=target_score), "land_search_target")
    # Seed real targets before evaluating fetches. A hypothetical Forest is never fixing.
    for color in weights:
        if sum(chosen.values()) >= slots:
            break
        if not any(color in direct_sources(options[n], spec.color_identity) for n in chosen):
            basic = next((c for c in basics if available(c) and color in direct_sources(c, spec.color_identity)), None)
            if basic:
                add(basic, "basic")
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
        add(min(ranked, key=lambda item: item[:2])[2], "multicolor_fixing")
    # Utility is a separate, bounded choice. Mono decks gain no fixing benefit
    # from rainbow lands, and should not pay life or sacrifice lands for it.
    while sum(chosen.values()) < slots:
        utility_count = sum(q for n, q in chosen.items() if is_utility_land(options[n]))
        if utility_count >= utility_limit:
            break
        ranked = []
        for card in options.values():
            if not available(card) or not is_utility_land(card):
                continue
            # Identity permits a land; actual pip demand determines whether its
            # colored utility belongs here. Colorless utility remains eligible.
            if set(card["color_identity"]) - set(profile["colors"]):
                continue
            trial = dict(chosen)
            trial[card["name"]] = trial.get(card["name"], 0) + 1
            graph = source_graph(trial, spec.color_identity, store)
            if graph["errors"]:
                continue
            remaining = slots - sum(trial.values())
            floors = {
                c: max(MIN_COLOR_SOURCES, slots - utility_limit)
                if strategy == "monocolor" else MIN_COLOR_SOURCES
                for c in profile["colors"]
            }
            if weights.get("C"):
                floors["C"] = MIN_COLOR_SOURCES if strategy == "colorless" else 1
            if sum(
                max(0, floor - graph["source_counts"].get(c, 0))
                for c, floor in floors.items()
            ) > remaining:
                continue
            sources = direct_sources(card, spec.color_identity) & set(weights)
            ranked.append((-len(sources), -_land_speed(card), card["name"], card))
        if not ranked:
            break
        add(min(ranked, key=lambda item: item[:3])[3], "utility")
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
        best = min(usable, key=score)
        add(best, "basic" if copy_limit(best) is None else "source_fallback")
    mana = validate_mana_base({**spells, **chosen}, spec.commander, store)
    errors.extend(mana["errors"])
    return {
        "valid": not errors, "commander": spec.commander, "strategy": strategy,
        "cards": dict(chosen), "land_count": sum(chosen.values()), "weighted_pips": dict(weights),
        "mana_pips": profile["mana_pips"], "ability_pips": profile["ability_pips"],
        "demand_share": profile["demand_share"], "pip_profile": profile,
        "utility_limit": utility_limit,
        "basic_count": sum(q for n, q in chosen.items() if copy_limit(options[n]) is None),
        "utility_count": sum(q for n, q in chosen.items() if is_utility_land(options[n])),
        "selection_reasons": reasons,
        "source_targets": targets, "source_counts": mana["source_counts"], "mana_validation": mana,
        "warnings": [
            "Target-aware source counts are not untapped-source or statistical consistency guarantees.",
            "Fetches sharing a target compete for it; source counts do not model draws or depletion.",
            "Opponent-dependent and colored-input filter abilities are not credited toward hard floors.",
            "Premium lands use local Oracle data; prices, ownership and budget are not modeled.",
        ] + (["Weighted source targets not all met"] if any(mana["source_counts"].get(c, 0) < n for c, n in targets.items()) else []),
        "errors": errors,
    }
