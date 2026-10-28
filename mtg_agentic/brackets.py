"""Commander bracket evidence, not a social or competitive certification."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlencode

from . import prompts
from .cards import Card, name_key, trim_card
from .llm import JSONModel, request_json
from .mana import is_land
from .oracle import OracleStore
from .schema import BRACKET3_MODES, NAME_SCHEMA, STRING_LIST, SchemaError, object_schema
from .scryfall import ScryfallClient
from .sources import SourceCache, SourceError

POLICY = {
    "version": "2026-02-09",
    "reviewed_on": "2026-10-25",
    "sources": [
        "https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta",
        "https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-october-21-2025",
        "https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-february-9-2026",
    ],
    "game_changer_limits": {1: 0, 2: 0, 3: 3, 4: None, 5: None},
    "turns_expected_before_winning": {1: 9, 2: 8, 3: 6, 4: 4, 5: 0},
    "tutor_limits": "Removed October 21, 2025. Counts are descriptive only.",
    "scope": "Standard rules only; Bracket 1 Rule Zero exceptions require pod agreement.",
}
_MLD_NAMES = {
    name_key(n)
    for n in (
        "Armageddon",
        "Ravages of War",
        "Ruination",
        "Sunder",
        "Winter Orb",
        "Static Orb",
        "Blood Moon",
        "Magus of the Moon",
        "Back to Basics",
        "Contamination",
        "Stasis",
        "Jokulhaups",
        "Obliterate",
        "Decree of Annihilation",
        "Worldfire",
        "Cataclysm",
        "Catastrophe",
        "Global Ruin",
        "Tectonic Break",
        "Wildfire",
        "Burning of Xinye",
        "Destructive Force",
        "Death Cloud",
    )
}
_FAST_NAMES = {
    name_key(n)
    for n in (
        "Sol Ring",
        "Mana Vault",
        "Grim Monolith",
        "Basalt Monolith",
        "Chrome Mox",
        "Mox Diamond",
        "Mox Opal",
        "Mox Amber",
        "Lotus Petal",
        "Lion's Eye Diamond",
        "Ancient Tomb",
        "City of Traitors",
        "Mishra's Workshop",
        "Dark Ritual",
        "Cabal Ritual",
        "Jeska's Will",
        "Simian Spirit Guide",
        "Elvish Spirit Guide",
    )
}
LIKELIHOOD = {"type": "string", "enum": ["likely", "unlikely", "unknown"]}
ASSESSMENT_SCHEMA = object_schema(
    {
        "fit": {"type": "string", "enum": ["fits", "mismatch", "uncertain"]},
        "estimated_bracket": {"type": "integer", "minimum": 1, "maximum": 5},
        "curve": NAME_SCHEMA,
        "interaction": NAME_SCHEMA,
        "likely_win_turn_min": {"type": "integer", "minimum": 0, "maximum": 99},
        "likely_win_turn_max": {"type": "integer", "minimum": 0, "maximum": 99},
        "reasons": {**STRING_LIST, "minItems": 1},
        "card_references": STRING_LIST,
        "extra_turn_chaining": LIKELIHOOD,
        "combo_reviews": {
            "type": "array",
            "maxItems": 300,
            "items": object_schema(
                {"id": NAME_SCHEMA, "early_likelihood": LIKELIHOOD, "reason": NAME_SCHEMA}
            ),
        },
    }
)


def assessment_schema(names: list[str], combo_ids: list[str]) -> dict[str, Any]:
    """Per-deck schema: constrained decoding can only cite deck cards and every combo ID once."""
    schema = copy.deepcopy(ASSESSMENT_SCHEMA)
    props = schema["properties"]
    props["card_references"]["items"] = {"type": "string", "enum": names}
    reviews = props["combo_reviews"]
    reviews["minItems"] = reviews["maxItems"] = len(combo_ids)
    if combo_ids:
        reviews["items"]["properties"]["id"] = {"type": "string", "enum": combo_ids}
    return schema


def game_changers(client: ScryfallClient, cache: SourceCache) -> dict[str, Any]:
    key = "scryfall:game-changers:v1"
    cached = cache.get(key)
    if cached:
        body, stamp = cached
        names = json.loads(body)
    else:
        names = sorted({c["name"] for c in client.search_cards("is:gamechanger")})
        if not names:
            raise SourceError("Scryfall returned an empty Game Changer list")
        stamp = cache.put(key, json.dumps(names).encode())
    return {
        "names": names,
        "observed_at": stamp,
        "source": "https://scryfall.com/search?q=is%3Agamechanger",
        "note": "Scryfall mirror; observation may use its dated 24-hour HTTP response cache.",
    }


class SpellbookClient:
    """Sparse, cached read-only POST queries. Never accept almost-included combos."""

    def __init__(self, cache: SourceCache, *, max_pages: int = 3):
        if not 1 <= max_pages <= 3:
            raise ValueError("Spellbook max_pages must be from 1 to 3")
        self.cache, self.max_pages = cache, max_pages

    def find_combos(self, cards: dict[str, int], commander: str) -> dict[str, Any]:
        payload = {
            "main": [{"card": n, "quantity": q} for n, q in sorted(cards.items())],
            "commanders": [{"card": commander, "quantity": 1}],
        }
        variants: dict[str, Any] = {}
        stamps = []
        url = "https://backend.commanderspellbook.com/find-my-combos?limit=100"
        visited: set[str] = set()
        complete = False
        for _ in range(self.max_pages):
            if url in visited:
                raise SourceError("Spellbook pagination loop")
            visited.add(url)
            body, stamp = self.cache.fetch(url, payload=payload)
            stamps.append(stamp)
            try:
                page = json.loads(body)
                included = page["results"]["included"]
                if not isinstance(included, list):
                    raise ValueError()
                for variant in included:
                    variants[variant["id"]] = variant
                next_url = page["next"]
            except (ValueError, TypeError, KeyError) as exc:
                raise SourceError("Invalid Spellbook combo response") from exc
            if not next_url:
                complete = True
                break
            # Reconstruct pagination locally: never send deck data to supplied URLs.
            url = "https://backend.commanderspellbook.com/find-my-combos?" + urlencode(
                {"limit": 100, "offset": len(visited) * 100}
            )
        available = Counter({name_key(n): q for n, q in cards.items()})
        available[name_key(commander)] += 1
        matched = []
        unresolved_templates = False
        try:
            for v in variants.values():
                uses = v["uses"]
                needed = Counter()
                for use in uses:
                    needed[name_key(use["card"]["name"])] += use["quantity"]
                if v["requires"] and sum(needed.values()) <= 2:
                    unresolved_templates = True
                if (
                    not uses
                    or v["status"] != "OK"
                    or v["requires"]
                    or sum(needed.values()) != 2
                    or len(needed) != 2
                    or any(available[n] < q for n, q in needed.items())
                    or any(
                        u["mustBeCommander"] and name_key(u["card"]["name"]) != name_key(commander)
                        for u in uses
                    )
                ):
                    continue
                results = [p["feature"]["name"] for p in v["produces"]]
                matched.append(
                    {
                        "id": v["id"],
                        "cards": [u["card"]["name"] for u in uses],
                        "results": results,
                        "infinite": any("infinite" in r.casefold() for r in results),
                        "game_winning": any(
                            re.search(r"win the game|opponents? lose", r, re.I) for r in results
                        ),
                        "mana_needed": v["manaNeeded"],
                        "uses": [
                            {
                                **{k: value for k, value in u.items() if k != "card"},
                                "card": {"name": u["card"]["name"]},
                            }
                            for u in uses
                        ],
                        "prerequisites": [v["easyPrerequisites"], v["notablePrerequisites"]],
                        "steps": v["description"],
                        "url": "https://commanderspellbook.com/combo/" + v["id"] + "/",
                    }
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceError("Invalid Spellbook variant fields") from exc
        return {
            "combos": matched,
            "complete": complete and not unresolved_templates,
            "observed_at": stamps,
            "attribution": "Commander Spellbook",
            "source": "https://commanderspellbook.com",
            "scope": "Known exact two-card, template-free variants only; prerequisites not proven.",
        }


_LAND_RESET = re.compile(r"(?:destroy|exile|return) all (?:nonbasic |basic )?lands\b")


def is_mass_land_denial(card: Card) -> bool:
    """Named mass land denial, or resetting every land without replacing it (Brackets 1-3 ban it)."""
    text = card["oracle_text"].casefold()
    replaces = "search" in text and "basic land" in text
    return name_key(card["name"]) in _MLD_NAMES or bool(_LAND_RESET.search(text) and not replaces)


def bracket_rules(bracket: int, bracket3_mode: str = "standard") -> dict[str, Any]:
    """Explicit house rules override only the selected parts of the published policy."""
    if type(bracket) is not int or bracket not in range(1, 6):
        raise ValueError("Bracket must be from 1 to 5")
    if bracket3_mode not in BRACKET3_MODES:
        raise ValueError("Unknown Bracket 3 mode")
    if bracket != 3 and bracket3_mode != "standard":
        raise ValueError("Custom Bracket 3 modes require bracket 3")
    return {
        "bracket3_mode": bracket3_mode,
        "house_rules": bracket3_mode != "standard",
        "mass_land_denial_restricted": bracket <= 3 and bracket3_mode != "unrestricted",
        "two_card_combos_restricted": bracket <= 3 and bracket3_mode != "unrestricted",
        "opponent_land_destruction_limit": 1 if bracket3_mode == "limited-land-destruction" else None,
        "land_exception": "Scorched Ruins" if bracket3_mode == "limited-land-destruction" else None,
        "extra_turn_chaining_restricted": bracket <= 3,
    }


_LAND_OBJECT = r"(?:lands?|plains|islands?|swamps?|mountains?|forests?|permanents?)\b"
_MULTIPLE = r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|[2-9]\d*|1\d+|x|any number of)"


def destroys_multiple_opponent_lands(card: Card) -> bool:
    """Oracle heuristic for one effect hitting multiple lands belonging to one opponent."""
    if name_key(card["name"]) == name_key("Scorched Ruins"):
        return False
    text = card["oracle_text"].casefold()
    for clause in re.split(r"[.\n•]", text):
        for effect in re.finditer(r"\bdestroy\b[^.\n]*", clause):
            words = effect.group()
            # Nonland permanents and effects restricted to our own lands cannot qualify.
            if "nonland" in words and not re.search(
                r"\b(?:lands?|plains|islands?|swamps?|mountains?|forests?)\b", words
            ):
                continue
            if (re.search(r"\b(?:lands?|permanents?) (?:that )?you control\b", words)
                    and not re.search(r"\bopponents?\b", words)):
                continue
            if re.search(r"\b(?:controlled by different players|each controlled by a different player)\b", words):
                continue
            if re.search(
                rf"\bdestroy (?:all|each)\b[^.\n]*\b{_LAND_OBJECT}|"
                rf"\bdestroy (?:up to )?{_MULTIPLE}\b[^.\n]*\b{_LAND_OBJECT}|"
                rf"\bdestroy\b[^.\n]*\ball (?:other )?{_LAND_OBJECT}|"
                r"\bdestroy (?:those|these|the chosen) lands\b", words
            ):
                return True
        if re.search(
            rf"\b(?:each|target) (?:player|opponent)\b[^.\n]*"
            rf"sacrifices? (?:{_MULTIPLE}|all)\b[^.\n]*\b{_LAND_OBJECT}", clause
        ):
            return True
    return False


def land_denial_forbidden(card: Card, bracket: int, bracket3_mode: str = "standard") -> bool:
    rules = bracket_rules(bracket, bracket3_mode)
    return bool(
        (rules["mass_land_denial_restricted"] and is_mass_land_denial(card))
        or (rules["opponent_land_destruction_limit"] is not None
            and destroys_multiple_opponent_lands(card))
    )


def deterministic_signals(cards: list[tuple[Card, int]], gc_names: list[str]) -> dict[str, Any]:
    gc = {name_key(n) for n in gc_names}
    signals: dict[str, Any] = {
        key: []
        for key in (
            "game_changers",
            "mass_land_denial",
            "land_denial_candidates",
            "multiple_opponent_land_destruction",
            "extra_turns",
            "tutors_cheap",
            "tutors_expensive",
            "land_tutors",
            "fast_mana",
            "recursion",
        )
    }
    curve: Counter[str] = Counter()
    for c, quantity in cards:
        name, text = c["name"], c["oracle_text"].casefold()
        item = {"name": name, "quantity": quantity}
        mv = c.get("cmc")
        if not is_land(c):
            curve[str(int(mv)) if isinstance(mv, (int, float)) else "unknown"] += quantity
        if name_key(name) in gc:
            signals["game_changers"].append(item)
        if destroys_multiple_opponent_lands(c):
            signals["multiple_opponent_land_destruction"].append(item)
        land_reset = _LAND_RESET.search(text)
        if is_mass_land_denial(c):
            signals["mass_land_denial"].append(item)
        elif land_reset or re.search(
            r"\b(?:lands|plains|islands|swamps|mountains|forests)\b[^.\n]*"
            r"(?:can't|cannot|don't|do not)[^.\n]*(?:untap|produce)|"
            r"\b(?:each|target) (?:player|opponent)\b[^.\n]*sacrifices?[^.\n]*\blands?\b|"
            r"\bdestroy (?:all |each |(?:up to )?(?:two|three|four|five|\d+) target )"
            r"[^.\n]*\blands?\b", text
        ):
            signals["land_denial_candidates"].append(item)
        if re.search(r"(?:extra|additional) turns?\b", text):
            signals["extra_turns"].append(item)
        if "search your library" in text:
            if re.search(
                r"search your library for (?:up to \w+ |an? )?(?:basic )?land cards?\b", text
            ):
                signals["land_tutors"].append(item)
            else:
                key = (
                    "tutors_cheap"
                    if isinstance(mv, (int, float)) and mv <= 2
                    else "tutors_expensive"
                )
                signals[key].append(item)
        if name_key(name) in _FAST_NAMES or (
            mv is not None and mv <= 1 and "{t}: add {c}{c}" in text
        ):
            signals["fast_mana"].append(item)
        if re.search(r"(?:return|cast|play).*(?:graveyard)", text):
            signals["recursion"].append(item)
    signals["curve"] = dict(sorted(curve.items()))
    signals["counts"] = {
        k: sum(c["quantity"] for c in v) for k, v in signals.items() if isinstance(v, list)
    }
    signals["definitions"] = {
        "tutors": "Oracle search heuristic. Cheap = printed mana value <=2, not activation cost. No official cap.",
        "fast_mana": "Named examples plus simple Oracle patterns, not exhaustive or a numeric restriction.",
        "land_denial": "Named examples and Oracle patterns; candidates require human/context review.",
        "extra_turns": "Card inclusion is not proof of chaining. Recursion and combos inform the LLM.",
    }
    return signals


class BracketChecker:
    def __init__(
        self,
        store: OracleStore,
        client: ScryfallClient,
        cache: SourceCache,
        model: JSONModel,
        *,
        spellbook: SpellbookClient | None = None,
    ):
        self.store, self.client, self.cache, self.model = store, client, cache, model
        self.spellbook = spellbook or SpellbookClient(cache)
        self.game_changer_list = game_changers(client, cache)

    def check(
        self, cards: dict[str, int], commander: str, bracket: int,
        *, bracket3_mode: str = "standard",
    ) -> dict[str, Any]:
        rules = bracket_rules(bracket, bracket3_mode)
        resolved = []
        quantities: Counter[str] = Counter(cards)
        quantities[commander] += 1
        for n, q in quantities.items():
            c = self.store.get_card(n)
            if c is None or type(q) is not int or q < 1:
                raise ValueError("Bracket check requires exact cards and positive quantities")
            resolved.append((c, q))
        signals = deterministic_signals(resolved, self.game_changer_list["names"])
        combos = self.spellbook.find_combos(cards, commander)
        signals["possible_extra_turn_chaining"] = bool(
            signals["counts"]["extra_turns"] > 1
            or (signals["extra_turns"] and signals["recursion"])
        )
        signals["known_extra_turn_chain_ids"] = [
            combo["id"]
            for combo in combos["combos"]
            if any(
                re.search(r"(?:infinite|arbitrarily many).*\bturns\b", effect, re.I)
                for effect in combo["results"]
            )
        ]
        errors, warnings = [], []
        limit = POLICY["game_changer_limits"][bracket]
        if limit is not None and signals["counts"]["game_changers"] > limit:
            errors.append(f"Game Changers exceed Bracket {bracket}'s limit of {limit}.")
        if rules["mass_land_denial_restricted"] and signals["mass_land_denial"]:
            errors.append("Remove mass land denial for Brackets 1–3.")
        if (rules["opponent_land_destruction_limit"] is not None
                and signals["multiple_opponent_land_destruction"]):
            errors.append(
                "Remove cards that can destroy more than one land controlled by one opponent; "
                "Scorched Ruins is allowed."
            )
        if bracket == 1 and signals["extra_turns"]:
            errors.append("Bracket 1 does not include extra-turn cards.")
        if bracket <= 3 and signals["known_extra_turn_chain_ids"]:
            errors.append("Known Spellbook extra-turn loop conflicts with Brackets 1–3.")
        if not combos["complete"] and bracket3_mode != "unrestricted":
            warnings.append("Spellbook search truncated; absence of a combo is not established.")
        if bracket3_mode != "unrestricted" and signals["land_denial_candidates"]:
            warnings.append("Review possible land-denial effects in context.")
        assessment = request_json(
            self.model,
            [
                {
                    "role": "system",
                    "content": prompts.BRACKET,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_bracket": bracket,
                            "policy": POLICY,
                            "selected_rules": rules,
                            "signals": signals,
                            "commander": commander,
                            "cards": [{"quantity": q, **trim_card(c)} for c, q in resolved],
                            "spellbook": combos,
                        }
                    ),
                },
            ],
            assessment_schema(sorted(c["name"] for c, _ in resolved), [c["id"] for c in combos["combos"]]),
        )
        known = {c["name"] for c, _ in resolved}
        reviews = assessment["combo_reviews"]
        expected = {c["id"] for c in combos["combos"]}
        lo, hi = assessment["likely_win_turn_min"], assessment["likely_win_turn_max"]
        if (
            set(assessment["card_references"]) - known
            or {r["id"] for r in reviews} != expected
            or len(reviews) != len(expected)
            or (lo == 0) != (hi == 0)
            or lo > hi
        ):
            raise SchemaError(
                [
                    {
                        "error": "assessment_grounding",
                        "detail": "Bracket assessment contains unsupported references or invalid turn bounds",
                    }
                ]
            )
        dangerous = {c["id"] for c in combos["combos"] if c["infinite"] or c["game_winning"]}
        if bracket <= 2 and dangerous:
            warnings.append(
                "Known two-card infinite/winning combo present; remove or review intent."
            )
        if bracket == 3 and rules["two_card_combos_restricted"] and any(
            r["id"] in dangerous and r["early_likelihood"] == "likely" for r in reviews
        ):
            errors.append("LLM flags a known two-card combo as frequently early; remove or revise.")
        if bracket in {2, 3} and assessment["extra_turn_chaining"] == "likely":
            errors.append("LLM flags extra-turn chaining; remove the chain or revise the plan.")
        if assessment["fit"] == "mismatch":
            errors.extend("LLM assessment: " + r for r in assessment["reasons"])
        expected_turns = POLICY["turns_expected_before_winning"][bracket]
        if bracket <= 4 and (lo == 0 or lo <= expected_turns):
            warnings.append(
                "Likely win speed is unknown or overlaps the bracket's development "
                "window; review consistency rather than treating one game as a violation."
            )
        if bracket == 5:
            warnings.append("A decklist alone does not establish cEDH metagame intent.")
        uncertain = bool(warnings) or assessment["fit"] == "uncertain"
        if rules["two_card_combos_restricted"] and any(
            r["id"] in dangerous and r["early_likelihood"] == "unknown" for r in reviews
        ):
            uncertain = True
        return {
            "verdict": "mismatch" if errors else "uncertain" if uncertain else "fits",
            "target_bracket": bracket,
            "bracket3_mode": bracket3_mode,
            "selected_rules": rules,
            "policy": POLICY,
            "game_changer_list": self.game_changer_list,
            "signals": signals,
            "spellbook": combos,
            "assessment": assessment,
            "reasons": errors + warnings + assessment["reasons"],
            "errors": errors,
            "warnings": warnings,
            "disclaimer": "Advisory pregame aid. Heuristics and LLM estimates, not observed gameplay.",
        }
