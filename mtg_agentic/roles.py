"""Deck roles from Scryfall's function tags and Oracle text, and protection heuristics.

A card's role comes first from the Tagger tags the store attaches as "functions" (people tagged
burn as removal, damage sweepers as board wipes, Fact or Fiction as card advantage), then from
Oracle-text patterns, which also cover cards printed after the last tag download. Neither is an
interaction or timing proof."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .cards import Card
from .mana import front, is_land, search_selectors
from .oracle import OracleStore
from .schema import TARGET_ROLES
from .spec import DeckSpec

# The function tags behind each role. Card advantage counts as draw unless it is only recursion.
TAG_ROLES = {
    "ramp": {"ramp"},
    "draw": {"draw", "impulse-draw"},
    "removal": {"removal"},
    "wipes": {"board-wipe"},
    "protection": {"protection", "counterspell"},
}
# A clause about other permanents, so a keyword in it protects more than the card itself.
OTHERS = r"(?:creatures?|permanents?|commanders?|equipped|enchanted|target)\b"
PROTECTION_PATTERNS = {
    "hexproof": r"\bhexproof\b",
    "shroud": r"\bshroud\b",
    "indestructible": r"\bindestructible\b",
    "protection": r"\bprotection from\b",
    "phasing": r"\b(?:phasing|phases? out)\b",
    "flicker": r"\bexile\b.+\breturn\b.+\bbattlefield\b",
    "counterspell": r"\bcounter target (?:\w+ )*spell\b",
    "recursion": r"\breturn\b.+\b(?:creature|permanent)\b.+\bgraveyard\b.+\b(?:hand|battlefield)\b",
}


def protection_features(card: Card) -> dict[str, Any]:
    text = front(card, "oracle_text").lower()
    types = front(card, "type_line")
    categories = set()
    for category, pattern in PROTECTION_PATTERNS.items():
        clauses = [text] if category == "flicker" else re.split(r"\.\s*|\n", text)
        for clause in clauses:
            if not re.search(pattern, clause, re.S):
                continue
            if re.search(r"\b(?:loses?|without|can't have|ignore)\b.*(?:hexproof|shroud|indestructible|protection)", clause):
                continue
            if category == "flicker" and not re.search(r"\b(?:creatures?|permanents?)\b", clause):
                continue
            # A self-only keyword or player-only shield is not a commander protection piece.
            # Plurals count: "creatures you control gain indestructible" protects the board.
            if category in {"hexproof", "shroud", "indestructible", "protection", "phasing"} and not re.search(OTHERS, clause):
                continue
            if re.search(r"\b(?:target player|you) (?:gain|have)", clause) and not re.search(r"\b(?:creatures?|permanents?|commanders?)\b", clause):
                continue
            categories.add(category)
    equipment = "Equipment" in types.split()
    activated = any(":" in clause and any(re.search(PROTECTION_PATTERNS[c], clause) for c in categories) for clause in text.split("\n"))
    instant = bool(categories) and not equipment and (
        "Instant" in types.split() or bool(re.search(r"(?:^|\n)flash\b", text))
        or (activated and not re.search(r"activate only (?:as a sorcery|during your turn)", text))
    )
    mechanisms = set()
    if categories and not instant and any(t in types.split() for t in ("Artifact", "Enchantment", "Creature", "Land")) and categories - {"recursion"}:
        mechanisms.add("static")
    if instant:
        mechanisms.add("instant_speed")
    if "recursion" in categories:
        mechanisms.add("recursion")
    return {"categories": sorted(categories), "mechanisms": sorted(mechanisms), "instant_independent": instant, "equipment": equipment}


def tag_roles(card: Card) -> set[str]:
    """The roles a card's function tags give it."""
    functions = set(card.get("functions") or ())
    roles = {role for role, tags in TAG_ROLES.items() if functions & tags}
    if "card-advantage" in functions and "recursion" not in functions:
        roles.add("draw")
    if "wipes" in roles:
        roles.add("removal")
    return roles


def oracle_roles(card: Card, tags=()) -> set[str]:
    if is_land(card):
        return {"lands"}
    text = front(card, "oracle_text").lower()
    roles = set(tags) & {"theme", "mechanics", "win_condition", "support"}
    if "Creature" in front(card, "type_line").split("—")[0].split():
        roles.add("creatures")
    if card.get("functions"):
        # People tagged this card: trust them over the patterns, which miss burn, damage
        # sweepers and card selection, and read graveyard recursion as removal.
        return roles | tag_roles(card)
    if re.search(r"\bdraw (?:a|an|one|two|three|four|five|six|seven|x|that many|cards equal)\b", text):
        roles.add("draw")
    if re.search(r"\badd (?:\{|[^.\n]*mana)|\bcreate\b[^.\n]*\btreasure\b|\bspells you cast cost\b[^.\n]*less", text) or (search_selectors(card) and "battlefield" in text):
        roles.add("ramp")
    # "Return target creature card from your graveyard" is recursion, not removal.
    if re.search(r"\b(?:destroy|exile|return|counter) (?:up to \w+ )?(?:any number of )?target\b(?! (?:\w+ )?card\b)", text):
        roles.add("removal")
    if re.search(r"\b(?:destroy|exile|return) all (?:nonland )?(?:creatures|artifacts|enchantments|permanents)(?! you control)|\b(?:all|each) creatures? get[s]? -|\bdeals? \w+ damage to each creature\b", text):
        roles.update(("wipes", "removal"))
    if protection_features(card)["categories"]:
        roles.add("protection")
    return roles


# The range each role's minimum may take. A plan sets its own targets (a spellslinger deck needs
# few creatures, a go-wide deck few board wipes); code clamps every target into these bounds.
ROLE_BOUNDS = {
    "ramp": (6, 16), "draw": (5, 15), "removal": (4, 14), "wipes": (0, 4),
    "creatures": (0, 40), "protection": (4, 12),
}
MAXIMUM_WIPES = 4


def bounded_targets(targets: dict[str, int], dependency: str = "medium") -> dict[str, int]:
    """Role targets clamped into ROLE_BOUNDS; a commander the plan depends on keeps 6 protection."""
    result = dict(targets)
    for role, (low, high) in ROLE_BOUNDS.items():
        if role in result:
            result[role] = min(max(result[role], low), high)
    if dependency == "high" and "protection" in result:
        result["protection"] = max(6, result["protection"])
    return result


def role_policy(spec: DeckSpec) -> dict[str, Any]:
    high = spec.commander_dependency == "high"
    minimums = bounded_targets(spec.role_targets, spec.commander_dependency)
    return {"minimums": minimums, "maximum_wipes": MAXIMUM_WIPES, "independent_instants": 3 if high else 2, "protection_mechanisms": 2}


def protection_report(cards: dict[str, int], store: OracleStore) -> dict[str, Any]:
    pieces = {}
    categories: Counter[str] = Counter()
    mechanisms: Counter[str] = Counter()
    independent = 0
    total = 0
    for name, q in cards.items():
        if type(q) is not int or q <= 0 or not (card := store.get_card(name)):
            continue
        features = protection_features(card)
        if not features["categories"]:
            continue
        pieces[card["name"]] = features
        total += q
        independent += q * features["instant_independent"]
        categories.update({c: q for c in features["categories"]})
        mechanisms.update({m: q for m in features["mechanisms"]})
    return {"total": total, "instant_independent": independent, "categories": dict(categories), "mechanisms": dict(mechanisms), "pieces": pieces}


def validate_roles(cards: dict[str, int], spec: DeckSpec, store: OracleStore, tags: dict[str, set[str]] | None = None) -> dict[str, Any]:
    counts = dict.fromkeys(TARGET_ROLES, 0)
    for name, q in cards.items():
        if type(q) is int and q > 0 and (card := store.get_card(name)):
            for role in oracle_roles(card, (tags or {}).get(card["name"], ())):
                if role in counts:
                    counts[role] += q
    policy = role_policy(spec)
    protection = protection_report(cards, store)
    errors = []
    for role, target in policy["minimums"].items():
        if counts[role] < target:
            errors.append({"error": "role_target", "role": role, "detail": f"Need {target}; got {counts[role]} Oracle-grounded {role} cards"})
    if counts["wipes"] > policy["maximum_wipes"]:
        errors.append({"error": "role_balance", "role": "wipes", "detail": f"At most {policy['maximum_wipes']} sweepers; got {counts['wipes']}"})
    if protection["instant_independent"] < policy["independent_instants"]:
        errors.append({"error": "protection_speed", "detail": f"{protection['total']} protection pieces, {protection['instant_independent']} independent instant-speed; need {policy['independent_instants']}"})
    if len(protection["mechanisms"]) < policy["protection_mechanisms"]:
        errors.append({"error": "protection_redundancy", "detail": "Protection needs at least two mechanisms: static, instant_speed, recursion"})
    return {"valid": not errors, "role_counts": counts, "policy": policy, "protection": protection, "errors": errors}
