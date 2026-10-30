"""The drafter ranks its picks; code fills the nonland slots from them, in its order.

Picks come in tiers, most important first (see DRAFT_TIERS): code never removes a core pick,
keeps strong picks, and may swap a flex pick. Code takes the must-include cards, then every pick
in rank order within the sweeper cap and the Game Changer limit. Open slots are padded from the
pool. When a role minimum or the protection policy is still short, code swaps the lowest flex
picks first for the drafter's own extra picks, then for other pool cards. Every pick code leaves
out is reported with why, so a repair can see what code did with its picks.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .discovery import CandidatePool
from .lands import nonland_slots
from .mana import is_land
from .roles import oracle_roles, protection_features, role_policy
from .schema import DRAFT_SCHEMA, DRAFT_TIERS

GAME_CHANGER_LIMITS = {1: 0, 2: 0, 3: 3, 4: 100, 5: 100}
SWAP_TRIES = 25  # Candidates tried per swap before code gives up on a role shortfall.


def theme_priority(card: dict[str, Any], spec) -> int:
    """Prioritize the requested Myojin engine, not generic counter/token payoffs."""
    requested = " ".join(spec.themes + spec.mechanics + spec.win_conditions).lower()
    if "myojin" not in requested:
        return 0
    text = card["oracle_text"].lower()
    if card["name"].startswith("Myojin "):
        return 3
    if re.search(r"proliferate|divinity counter", text):
        return 2
    if re.search(r"put an? indestructible counter on (?:each|target|another)\b", text):
        return 2
    if "+1/+1" not in text and re.search(r"counters?.*(?:twice|instead)", text):
        return 2
    if (
        ("token" in text and "token" not in requested)
        or ("+1/+1 counter" in text and "+1/+1" not in requested)
    ):
        return -1
    return 0


def selection_schema(pool: CandidatePool) -> dict[str, Any]:
    schema = copy.deepcopy(DRAFT_SCHEMA)
    schema["properties"]["commander"] = {"type": "string", "enum": [pool.spec.commander]}
    names = sorted(c.card["name"] for c in pool.cards if not is_land(c.card))
    for tier in schema["properties"]["picks"]["properties"].values():
        tier["items"] = {"type": "string", "enum": names}
        tier["maxItems"] = min(tier["maxItems"], len(names))
    return schema


def _groups(selections: Mapping[str, Iterable[str]]) -> list[str]:
    """The groups in rank order: the tiers, then any other group in its given order."""
    return [*(t for t in DRAFT_TIERS if t in selections), *(g for g in selections if g not in DRAFT_TIERS)]


def ranked_picks(selections: Mapping[str, Iterable[str]]) -> list[str]:
    """Every pick once, in rank order."""
    return list(dict.fromkeys(name for group in _groups(selections) for name in selections[group]))


def code_report(assembly: dict[str, Any]) -> dict[str, Any]:
    """What code did with a drafter's picks, for its repair: cards code added, every pick it left
    out with why, and the flex picks it swapped."""
    return {
        "added_by_code": assembly.get("code_added", []),
        "left_out": assembly.get("dropped", []),
        "swapped": assembly.get("code_swaps", []),
    }


def assemble_nonlands(
    pool: CandidatePool, selections: Mapping[str, Iterable[str]], *, game_changers=None,
    slots: int | None = None,
) -> dict[str, Any]:
    spec = pool.spec
    slots = nonland_slots(spec) if slots is None else slots
    candidates = {c.card["name"]: c for c in pool.cards if not is_land(c.card)}
    roles = {n: oracle_roles(c.card, c.roles) for n, c in candidates.items()}
    features = {n: protection_features(c.card) for n, c in candidates.items()}
    theme_ranks = {n: theme_priority(c.card, spec) for n, c in candidates.items()}
    order = [n for n in ranked_picks(selections) if n in candidates]
    rank = {n: i for i, n in enumerate(order)}
    first_group: dict[str, str] = {}
    for group in _groups(selections):
        for name in selections[group]:
            first_group.setdefault(name, group)
    must = [n for n in dict.fromkeys(spec.must_include) if n in candidates]
    # Code may only swap flex picks, and any group a caller did not tier.
    replaceable = {
        n for n in order if first_group[n] not in {"core", "strong"} and n not in must
    }
    policy = role_policy(spec)
    minimums = {r: v for r, v in policy["minimums"].items() if r != "lands"}
    changer_names = set(game_changers) if game_changers is not None else {
        c.card["name"] for c in pool.cards if c.card["game_changer"]
    }
    leader = next((c.card for c in pool.cards if c.card["name"] == spec.commander), None)
    # The commander is deliberately not in the candidate pool.
    changers = int(spec.commander in changer_names or bool(leader and leader["game_changer"]))
    changer_limit = GAME_CHANGER_LIMITS[spec.bracket]
    counts: Counter[str] = Counter()
    mechanisms: Counter[str] = Counter()
    instants = 0
    chosen: dict[str, int] = {}
    dropped: dict[str, str] = {}
    swaps: list[dict[str, str]] = []

    def add(name: str) -> None:
        nonlocal instants, changers
        chosen[name] = 1
        counts.update(roles[name])
        mechanisms.update(features[name]["mechanisms"])
        instants += features[name]["instant_independent"]
        changers += name in changer_names

    def remove(name: str) -> None:
        nonlocal instants, changers
        del chosen[name]
        counts.subtract(roles[name])
        mechanisms.subtract(features[name]["mechanisms"])
        instants -= features[name]["instant_independent"]
        changers -= name in changer_names

    def blocked(name: str, leaving: str | None = None) -> str:
        """Why `name` cannot join (with `leaving` out), or "" when it can."""
        wipes = counts["wipes"] - ("wipes" in roles[leaving] if leaving else 0)
        if "wipes" in roles[name] and wipes >= policy["maximum_wipes"]:
            return f"over the cap of {policy['maximum_wipes']} board wipes"
        held = changers - (leaving in changer_names if leaving else 0)
        if name in changer_names and held >= changer_limit:
            return f"over the Bracket {spec.bracket} Game Changer limit of {changer_limit}"
        return ""

    def shortfall(joining: str | None = None, leaving: str | None = None) -> int:
        """How far the role minimums and protection policy are from met, after a change."""
        have, kinds, fast = counts.copy(), mechanisms.copy(), instants
        for name, sign in ((joining, 1), (leaving, -1)):
            if name:
                have.update({r: sign for r in roles[name]})
                kinds.update({m: sign for m in features[name]["mechanisms"]})
                fast += sign * features[name]["instant_independent"]
        short = sum(max(0, need - have[role]) for role, need in minimums.items())
        short += max(0, policy["independent_instants"] - fast)
        return short + max(0, policy["protection_mechanisms"] - sum(1 for v in kinds.values() if v > 0))

    def pad_score(name: str) -> tuple:
        deficit = sum(1 for r in roles[name] if counts[r] < minimums.get(r, 0))
        instant = features[name]["instant_independent"] and instants < policy["independent_instants"]
        mechanism = (
            sum(1 for v in mechanisms.values() if v > 0) < policy["protection_mechanisms"]
            and bool(set(features[name]["mechanisms"]) - {m for m, v in mechanisms.items() if v > 0})
        )
        equipment = equipment_theme and equipment_count() < 15 and "Equipment" in candidates[name].card["type_line"].split()
        popularity = candidates[name].card.get("edhrec_rank")
        return (
            -int(instant), -int(mechanism), -deficit, -int(equipment), -theme_ranks[name],
            -int("theme" in roles[name]), popularity is None, popularity or 0, name,
        )

    def equipment_count() -> int:
        return sum("Equipment" in candidates[n].card["type_line"].split() for n in chosen)

    def swap_in(sources: list[str], label: str) -> None:
        """Swap the lowest flex picks for `sources` cards while that lowers the shortfall."""
        while (current := shortfall()) > 0:
            done = False
            tried = 0
            for card in sources:
                if card in chosen or tried >= SWAP_TRIES:
                    continue
                if shortfall(joining=card) >= current:
                    continue
                tried += 1
                for out in sorted((n for n in chosen if n in replaceable), key=lambda n: -rank[n]):
                    if not blocked(card, out) and shortfall(joining=card, leaving=out) < current:
                        remove(out)
                        add(card)
                        dropped[out] = f"swapped by code for {card} ({label}) to reach the role minimums"
                        dropped.pop(card, None)
                        swaps.append({"out": out, "in": card, "from": label})
                        done = True
                        break
                if done:
                    break
            if not done:
                return

    equipment_theme = bool(re.search(r"equipment|voltron|equipped", " ".join(spec.themes + spec.mechanics), re.I))
    for name in must:
        add(name)
    errors = []
    if len(chosen) > slots:
        errors.append({"error": "nonland_capacity", "detail": "Required nonlands exceed reserved slots"})
    for name in order:
        if name in chosen:
            continue
        if len(chosen) >= slots:
            dropped[name] = "no room: ranked below the last nonland slot"
        elif reason := blocked(name):
            dropped[name] = reason
        else:
            add(name)
    while len(chosen) < slots:
        available = [n for n in candidates if n not in chosen and n not in dropped and not blocked(n)]
        if not available:
            errors.append({"error": "nonland_capacity", "detail": f"Need {slots} distinct eligible nonlands within the sweeper/Game Changer limits; assembled {len(chosen)}"})
            break
        add(min(available, key=pad_score))
    if shortfall() > 0 and replaceable:
        swap_in([n for n in order if n not in chosen], "your extra picks")
        if shortfall() > 0:
            rest = sorted((n for n in candidates if n not in chosen and n not in rank), key=pad_score)
            swap_in(rest, "the pool")
    return {
        "cards": chosen, "errors": errors, "nonland_slots": slots,
        "model_selected_unique": len(order),
        "code_added": sorted(set(chosen) - set(order) - set(must)),
        "model_choices_not_used": sorted(set(order) - set(chosen)),
        "dropped": [{"card": n, "reason": dropped[n]} for n in order if n in dropped and n not in chosen],
        "code_swaps": swaps,
    }
