"""The model selects role preferences. Code owns quantity, padding and mana."""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

from .discovery import CandidatePool
from .mana import is_land
from .roles import oracle_roles, protection_features, role_policy
from .schema import DRAFT_SCHEMA


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
    for bucket in schema["properties"]["buckets"]["properties"].values():
        bucket["items"] = {"type": "string", "enum": names}
        bucket["maxItems"] = min(63, len(names))
    return schema


def assemble_nonlands(pool: CandidatePool, selections: dict[str, list[str]], *, game_changers=None) -> dict[str, Any]:
    spec = pool.spec
    slots = 99 - spec.role_targets["lands"]
    candidates = {c.card["name"]: c for c in pool.cards if not is_land(c.card)}
    roles = {n: oracle_roles(c.card, c.roles) for n, c in candidates.items()}
    features = {n: protection_features(c.card) for n, c in candidates.items()}
    theme_ranks = {n: theme_priority(c.card, spec) for n, c in candidates.items()}
    preferences = list(dict.fromkeys(n for names in selections.values() for n in names))
    ranks = {n: len(preferences) - i for i, n in enumerate(preferences)}
    policy = role_policy(spec)
    counts: Counter[str] = Counter()
    chosen = {}
    mechanisms = set()
    instant_count = 0
    changer_names = set(game_changers) if game_changers is not None else {
        c.card["name"] for c in pool.cards if c.card["game_changer"]
    }
    leader = next((c.card for c in pool.cards if c.card["name"] == spec.commander), None)
    # The commander is deliberately not in the candidate pool.
    changers = int(spec.commander in changer_names)
    if leader and leader["game_changer"]:
        changers = max(changers, 1)
    changer_limit = {1: 0, 2: 0, 3: 3, 4: 100, 5: 100}[spec.bracket]
    def add(name):
        nonlocal instant_count, changers
        chosen[name] = 1
        counts.update(roles[name])
        instant_count += features[name]["instant_independent"]
        mechanisms.update(features[name]["mechanisms"])
        changers += name in changer_names
    for name in spec.must_include:
        if name in candidates:
            add(name)
    errors = []
    if len(chosen) > slots:
        errors.append({"error": "nonland_capacity", "detail": "Required nonlands exceed reserved slots"})
    equipment_theme = bool(re.search(r"equipment|voltron|equipped", " ".join(spec.themes + spec.mechanics), re.I))
    equipment_count = sum("Equipment" in candidates[n].card["type_line"].split() for n in chosen)
    while len(chosen) < slots:
        available = [
            n for n in candidates if n not in chosen
            and not ("wipes" in roles[n] and counts["wipes"] >= policy["maximum_wipes"])
            and not (n in changer_names and changers >= changer_limit)
        ]
        if not available:
            errors.append({"error": "nonland_capacity", "detail": f"Need {slots} distinct eligible nonlands within the sweeper/Game Changer limits; assembled {len(chosen)}"})
            break
        def score(name):
            deficit = sum(1 for r in roles[name] if counts[r] < policy["minimums"].get(r, 0))
            instant = features[name]["instant_independent"] and instant_count < policy["independent_instants"]
            mechanism = len(mechanisms) < policy["protection_mechanisms"] and bool(set(features[name]["mechanisms"]) - mechanisms)
            equipment = equipment_theme and equipment_count < 15 and "Equipment" in candidates[name].card["type_line"].split()
            return (-int(instant), -int(mechanism), -deficit, -int(equipment), -theme_ranks[name], -ranks.get(name, 0), -int("theme" in roles[name]), name)
        best = min(available, key=score)
        add(best)
        equipment_count += "Equipment" in candidates[best].card["type_line"].split()
    return {
        "cards": chosen, "errors": errors, "nonland_slots": slots,
        "model_selected_unique": len(preferences),
        "code_added": sorted(set(chosen) - set(preferences)),
        "model_choices_not_used": sorted(set(preferences) - set(chosen)),
    }
