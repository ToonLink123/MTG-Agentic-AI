"""Independent pip-driven mana-base generation, without an LLM or spell pool."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .land_options import is_utility_land, mana_land_options
from .mana import (
    BASIC_LANDS,
    MIN_COLOR_SOURCES,
    SYMBOL,
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
from .roles import oracle_roles
from .spec import DeckSpec, ground_spec
from .validator import copy_limit

if TYPE_CHECKING:
    from .discovery import CandidatePool

# Compatibility exports for callers of the original allocation module.
__all__ = [
    "allocate_sources", "calculate_land_base", "generate_mana_base",
    "is_land", "land_sources", "mana_demand",
]

# Optional nonbasics must be among about the 2,000 most-played cards on EDHREC; below that sit
# lands like Unknown Shores that a basic beats. Unranked cards (new printings) still qualify.
POPULAR_LAND_RANK = 2000
# A spec's land target of 0 means "auto": code sets the count from the deck's curve and cheap
# ramp and draw. Before any nonlands are known, the estimate is the classic 36.
DEFAULT_LANDS = 36
LAND_RANGE = (32, 40)


def expected_lands(spec: DeckSpec) -> int:
    """The spec's land count, or the default estimate while it is still auto."""
    return spec.role_targets["lands"] or DEFAULT_LANDS


def nonland_slots(spec: DeckSpec) -> int:
    return 99 - expected_lands(spec)


def land_plan(spells: Iterable[str], store: OracleStore) -> dict[str, Any]:
    """Frank Karsten's 99-card estimate, 31.42 + 3.13 x average mana value - 0.28 x cheap
    (mana value 2 or less) ramp and draw, rounded and kept within LAND_RANGE."""
    values: list[float] = []
    cheap = 0
    for name in spells:
        card = store.get_card(name)
        if card is None or is_land(card):
            continue
        value = float(card.get("cmc") or 0)
        values.append(value)
        cheap += value <= 2 and bool(oracle_roles(card) & {"ramp", "draw"})
    if not values:
        return {"count": DEFAULT_LANDS, "average_mana_value": None, "cheap_ramp_and_draw": 0,
                "estimate": float(DEFAULT_LANDS), "basis": "no nonlands yet; the default count"}
    average = sum(values) / len(values)
    estimate = 31.42 + 3.13 * average - 0.28 * cheap
    return {
        "count": min(max(round(estimate), LAND_RANGE[0]), LAND_RANGE[1]),
        "average_mana_value": round(average, 2), "cheap_ramp_and_draw": cheap,
        "estimate": round(estimate, 1),
        "basis": "31.42 + 3.13 x average mana value - 0.28 x cheap ramp and draw",
    }


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


def _popularity(card: dict[str, Any]) -> float:
    """1 for the most-played cards, falling slowly with EDHREC rank: about 0.77 at rank 100,
    0.49 at 1,000, and 0.25 for an unranked card."""
    rank = card.get("edhrec_rank")
    if type(rank) is not int:
        return 0.25
    return 1 / (1 + math.log10(1 + rank / 100))


def _free_fetch(card: dict[str, Any]) -> bool:
    """A land search that costs no mana, like a fetch land's {T}, Pay 1 life, Sacrifice."""
    match = re.search(r"([^\n:]*):[^\n]*search your library", front(card, "oracle_text"), re.I)
    selectors = search_selectors(card)
    return bool(
        match and selectors and all(s["supported"] for s in selectors)
        and all(s in {"T", "Q", "E"} for s in SYMBOL.findall(match[1]))
    )


def optional_fixer(card: dict[str, Any], identity: str, colors: set[str]) -> bool:
    """A nonbasic worth a slot over a basic in a multicolor deck: it enters untapped (or
    usually does), makes two of the deck's colors without spending mana or fetches them, and
    is played in many decks. A fetch must name only the deck's own land types, so Polluted
    Delta (Island or Swamp) stays out of a deck without blue."""
    rank = card.get("edhrec_rank")
    own_types = {BASIC_LANDS[c] for c in colors}
    return (
        copy_limit(card) is not None
        and _land_speed(card) >= 0.85
        and not (type(rank) is int and rank > POPULAR_LAND_RANK)
        and (
            len(direct_sources(card, identity, free=True) & colors) >= 2
            or (_free_fetch(card) and all(
                set(s["types"]) <= own_types for s in search_selectors(card)
            ))
        )
    )


class _Remembered:
    """The store's card lookups, remembered: the fixing passes rebuild source graphs hundreds
    of times per mana base."""

    def __init__(self, store: OracleStore):
        self.store, self.cards = store, {}

    def get_card(self, name: str) -> dict[str, Any] | None:
        if name not in self.cards:
            self.cards[name] = self.store.get_card(name)
        return self.cards[name]


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
    # An auto land count (0) is set from these nonlands' curve.
    slots = spec.role_targets["lands"] or land_plan(spells, store)["count"]
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
    limits = {}  # copy_limit compiles a pattern per name, and the passes ask thousands of times.
    def limit_of(card):
        if card["name"] not in limits:
            limits[card["name"]] = copy_limit(card)
        return limits[card["name"]]
    def available(card):
        limit = limit_of(card)
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
    def plain_first(card):
        # Snow-Covered basics only when something asks for snow.
        return "Snow" in front(card, "type_line").split(), card["name"]
    basics = sorted((c for c in options.values() if limit_of(c) is None), key=plain_first)
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
            return (-coverage, -len(colors) * _land_speed(card), *plain_first(card))
        add(min(targets_available, key=target_score), "land_search_target")
    # Seed real targets before evaluating fetches. A hypothetical Forest is never fixing.
    for color in weights:
        if sum(chosen.values()) >= slots:
            break
        if not any(color in direct_sources(options[n], spec.color_identity) for n in chosen):
            basic = next((c for c in basics if available(c) and color in direct_sources(c, spec.color_identity)), None)
            if basic:
                add(basic, "basic")
    # Nonbasics first, basics last. A multicolor deck keeps one basic per color it casts, plus
    # one for each card that searches for a basic, so fetches and Cultivate still find them;
    # every other slot goes to fast fixing lands while good ones remain.
    colors = set(profile["colors"])
    cards = _Remembered(store)
    basic_searches = sum(
        any(selector["basic"] for selector in search_selectors(card))
        for name in dict.fromkeys([*spells, spec.commander, *chosen])
        if (card := cards.get_card(name))
    )
    reserve = min(slots // 2, len(colors) + basic_searches) if strategy == "multicolor" else 0
    fixers = (
        [c for c in options.values() if optional_fixer(c, spec.color_identity, colors)]
        if strategy == "multicolor" else []
    )
    def room():
        basic_count = sum(q for n, q in chosen.items() if limit_of(options[n]) is None)
        return sum(chosen.values()) + max(0, reserve - basic_count) < slots
    # Only a fetch or a land with a basic land type can change another land's sources; any
    # other land just adds its own, which saves rebuilding the source graph for each one.
    plain = {
        c["name"]: direct_sources(c, spec.color_identity) for c in fixers
        if not search_selectors(c)
        and not set(front(c, "type_line").split()) & set(BASIC_LANDS.values())
    }
    def potential(card, counts):
        """The colors `card` would make in the deck and the deck's source counts with it, or
        (None, None) when adding it leaves a land search without a target."""
        if card["name"] in plain:
            made = plain[card["name"]]
            return made & colors, Counter(counts) + Counter(dict.fromkeys(made, 1))
        trial = dict(chosen)
        trial[card["name"]] = 1
        graph = source_graph(trial, spec.color_identity, cards)
        if graph["errors"]:
            return None, None
        return set(graph["land_sources"].get(card["name"], [])) & colors, graph["source_counts"]
    # First toward each color's source target: the land that closes the most of what is
    # missing, scaled by how often it enters untapped and how many decks play it.
    while sum(chosen.values()) < slots and room():
        counts = source_graph(chosen, spec.color_identity, cards)["source_counts"]
        ranked = []
        for card in fixers:
            if not available(card):
                continue
            made, trial_counts = potential(card, counts)
            if made is None or len(made) < 2:
                continue
            # Count improvements to existing fetches too: adding a typed dual can
            # improve several sources, not only the new land's own production.
            score = sum(
                min(
                    max(0, targets[c] - counts.get(c, 0)),
                    max(0, trial_counts.get(c, 0) - counts.get(c, 0)),
                ) / targets[c]
                for c in targets
            ) * _land_speed(card) * _popularity(card)
            if score > 0:
                ranked.append((-score, card["name"], card))
        if not ranked:
            break
        add(min(ranked, key=lambda item: item[:2])[2], "multicolor_fixing")
    # Utility is a separate, bounded choice. Mono decks gain no fixing benefit
    # from rainbow lands, and should not pay life or sacrifice lands for it.
    while sum(chosen.values()) < slots and room():
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
            graph = source_graph(trial, spec.color_identity, cards)
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
            # Mana that costs a creature or {1} does not make a utility land a better basic.
            sources = direct_sources(card, spec.color_identity, free=True) & set(weights)
            ranked.append((-len(sources), -_land_speed(card) * _popularity(card), card["name"], card))
        if not ranked:
            break
        add(min(ranked, key=lambda item: item[:3])[3], "utility")
    # Past the targets, every fast fixing land left still beats a basic. The ones making more
    # of what the deck casts go first, until only the basics the deck keeps remain.
    share = profile["demand_share"]
    while sum(chosen.values()) < slots and room():
        counts = source_graph(chosen, spec.color_identity, cards)["source_counts"]
        ranked = []
        for card in fixers:
            if not available(card):
                continue
            made, _ = potential(card, counts)
            if made is not None and len(made) >= 2:
                score = sum(share[c] for c in made) * _land_speed(card) * _popularity(card)
                ranked.append((-score, card["name"], card))
        if not ranked:
            break
        add(min(ranked, key=lambda item: item[:2])[2], "multicolor_fixing")
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
            return (-floor_deficit, -target_deficit, -sum(weights[c] / (counts.get(c, 0) + 1) for c in colors), *plain_first(card))
        best = min(usable, key=score)
        add(best, "basic" if copy_limit(best) is None else "source_fallback")
    mana = validate_mana_base({**spells, **chosen}, spec.commander, store)
    errors.extend(mana["errors"])
    return {
        "valid": not errors, "commander": spec.commander, "strategy": strategy,
        "cards": dict(chosen), "land_count": sum(chosen.values()), "weighted_pips": dict(weights),
        "mana_pips": profile["mana_pips"], "ability_pips": profile["ability_pips"],
        "demand_share": profile["demand_share"], "pip_profile": profile,
        "utility_limit": utility_limit, "basic_reserve": reserve,
        "basic_count": sum(q for n, q in chosen.items() if copy_limit(options[n]) is None),
        "utility_count": sum(q for n, q in chosen.items() if is_utility_land(options[n])),
        "selection_reasons": reasons,
        "source_targets": targets, "source_counts": mana["source_counts"], "mana_validation": mana,
        "warnings": [
            "Target-aware source counts are not untapped-source or statistical consistency guarantees.",
            "Fetches sharing a target compete for it; source counts do not model draws or depletion.",
            "Opponent-dependent, conditional, spend-restricted and colored-input filter abilities are not credited toward hard floors.",
            "Premium lands use local Oracle data; prices, ownership and budget are not modeled.",
        ] + (["Weighted source targets not all met"] if any(mana["source_counts"].get(c, 0) < n for c, n in targets.items()) else []),
        "errors": errors,
    }
