"""LLM query planning, live Scryfall discovery, and bounded local-grounded pools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

from .cards import Card, name_key, trim_card
from .llm import JSONModel, request_json
from .oracle import OracleStore
from .schema import QUERY_SCHEMA, ROLES, SchemaError
from .scryfall import ScryfallClient
from .spec import DeckSpec, ground_spec
from .validator import card_errors

BASIC_LANDS = {
    "W": "Plains",
    "U": "Island",
    "B": "Swamp",
    "R": "Mountain",
    "G": "Forest",
    "C": "Wastes",
}
DEFAULT_QUERIES = {
    "ramp": "otag:ramp",
    "draw": "otag:draw",
    "removal": "otag:removal",
    "wipes": "otag:board-wipe",
    "lands": "t:land",
}


@dataclass(frozen=True)
class CardQuery:
    role: str
    query: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "query": self.query}


def constrain_query(query: str, identity: str) -> str:
    """Group all model text so an OR cannot escape the mandatory outer filters."""
    if not isinstance(query, str) or not query.strip() or len(query) > 950:
        raise ValueError("Scryfall query must contain 1 to 950 characters")
    depth = 0
    quoted = False
    escaped = False
    for char in query:
        if ord(char) < 32:
            raise ValueError("Scryfall queries cannot contain control characters")
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced Scryfall query parentheses")
    if depth or quoted or escaped:
        raise ValueError("Unbalanced Scryfall query")
    if re.fullmatch(r"(?:W?U?B?R?G?|C)", identity) is None or not identity:
        raise ValueError("Invalid color identity")
    return f"({query.strip()}) f:commander id<={identity}"


def generate_queries(spec: DeckSpec, model: JSONModel) -> list[CardQuery]:
    spec = DeckSpec.from_dict(spec.to_dict())
    messages = [
        {
            "role": "system",
            "content": (
                "Produce role-tagged Scryfall queries for this validated Commander spec. "
                "Use live Scryfall syntax, including otags. Cover all five role_targets and "
                "the requested themes, mechanics, and win conditions. Do not name guessed cards. "
                'Examples: myojin tribal -> t:myojin; divinity counters -> o:"divinity counter"; '
                "proliferate -> otag:proliferate. Code appends legality and color identity filters. "
                "Return JSON only. Schema: " + json.dumps(QUERY_SCHEMA)
            ),
        },
        {"role": "user", "content": json.dumps(spec.to_dict())},
    ]
    raw = request_json(model, messages, QUERY_SCHEMA)["queries"]
    # Small deterministic anchors make explicit themes survive an incomplete model plan.
    themes = " ".join(spec.themes + spec.mechanics).lower()
    for tribe in re.findall(r"\b([a-z][a-z'-]*)\s+tribal\b", themes):
        raw.append({"role": "theme", "query": f"t:{tribe}"})
    if "divinity counter" in themes:
        raw.append({"role": "theme", "query": 'o:"divinity counter"'})
    if any("proliferat" in mechanic.lower() for mechanic in spec.mechanics):
        raw.append({"role": "mechanics", "query": "otag:proliferate"})
    present = {item["role"] for item in raw}
    raw.extend(
        {"role": role, "query": query}
        for role, query in DEFAULT_QUERIES.items()
        if role not in present
    )
    queries: list[CardQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        constrained = constrain_query(item["query"], spec.color_identity)
        key = (item["role"], constrained.casefold())
        if key not in seen:
            queries.append(CardQuery(item["role"], constrained))
            seen.add(key)
    return queries


@dataclass
class Candidate:
    card: Card
    roles: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {**trim_card(self.card), "roles": sorted(self.roles)}


@dataclass
class CandidatePool:
    spec: DeckSpec
    cards: list[Candidate]
    queries: list[CardQuery] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "spec": self.spec.to_dict(),
            "cards": [candidate.to_dict() for candidate in self.cards],
            "queries": [query.to_dict() for query in self.queries],
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Any, store: OracleStore) -> CandidatePool:
        """Rehydrate names against the local store; saved card JSON is not trusted."""
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Expected a version-1 candidate pool")
        spec = ground_spec(DeckSpec.from_dict(data.get("spec")), store)
        raw_cards = data.get("cards")
        if not isinstance(raw_cards, list) or len(raw_cards) > 400:
            raise ValueError("A candidate pool must contain at most 400 cards")
        cards: list[Candidate] = []
        seen: set[str] = set()
        for raw in raw_cards:
            if (
                not isinstance(raw, dict)
                or not isinstance(raw.get("name"), str)
                or not isinstance(raw.get("roles"), list)
                or any(role not in ROLES for role in raw["roles"])
            ):
                raise ValueError("Invalid candidate pool entry")
            card = store.get_card(raw["name"])
            if card is None or not eligible(card, spec):
                raise ValueError("Candidate pool is stale or contains an ineligible card")
            key = name_key(card["name"])
            if key in seen:
                raise ValueError("Duplicate candidate pool entry")
            seen.add(key)
            cards.append(Candidate(card, set(raw["roles"])))
        if not {name_key(n) for n in spec.must_include} <= seen:
            raise ValueError("Candidate pool omits a must_include card")
        raw_queries = data.get("queries", [])
        if not isinstance(raw_queries, list) or len(raw_queries) > 40:
            raise ValueError("Invalid saved discovery queries")
        queries: list[CardQuery] = []
        for raw in raw_queries:
            if not isinstance(raw, dict) or raw.get("role") not in ROLES:
                raise ValueError("Invalid saved discovery query")
            constrain_query(raw.get("query"), spec.color_identity)
            queries.append(CardQuery(raw["role"], raw["query"]))
        diagnostics = data.get("diagnostics", [])
        if not isinstance(diagnostics, list) or len(diagnostics) > 40:
            raise ValueError("Invalid discovery diagnostics")
        return cls(spec, cards, queries, diagnostics)


def eligible(card: Card, spec: DeckSpec) -> bool:
    return (
        name_key(card["name"]) != name_key(spec.commander)
        and name_key(card["name"]) not in {name_key(n) for n in spec.exclude}
        and not card_errors(card, set(spec.color_identity) - {"C"})
        and not (spec.bracket < 3 and card["game_changer"])
    )


def _empty_query_fallback(query: CardQuery, spec: DeckSpec) -> str | None:
    if query.role == "theme":
        themes = " ".join(spec.themes).lower()
        for tribe in re.findall(r"\b([a-z][a-z'-]*)\s+tribal\b", themes):
            if re.search(rf"\bt:{re.escape(tribe)}\b", query.query, re.IGNORECASE):
                # Myojin, for example, is a name family; its actual subtype is Spirit.
                return tribe
    if query.role == "mechanics":
        match = re.search(r"\botag:([a-z][a-z-]*)\b", query.query, re.IGNORECASE)
        if match:
            return 'o:"' + match.group(1).replace("-", " ") + '"'
    return None


def build_candidate_pool(
    spec: DeckSpec,
    queries: list[CardQuery],
    client: ScryfallClient,
    store: OracleStore,
    *,
    limit: int = 350,
    results_per_query: int = 500,
) -> CandidatePool:
    if type(limit) is not int or not 1 <= limit <= 400:
        raise ValueError("Candidate limit must be from 1 to 400")
    if type(results_per_query) is not int or not 1 <= results_per_query <= 1000:
        raise ValueError("Results per query must be from 1 to 1000")
    spec = ground_spec(spec, store)
    if len(queries) > 40:
        raise ValueError("At most 40 discovery queries are allowed")
    candidates: dict[str, Candidate] = {}
    buckets: list[list[str]] = []
    diagnostics: list[dict[str, Any]] = []
    normalized_queries: list[CardQuery] = []
    for query in queries:
        if query.role not in ROLES:
            raise ValueError("Unknown discovery role")
        # Callers may supply their own plans. Always enforce the filters here too.
        constrained = constrain_query(query.query, spec.color_identity)
        normalized_queries.append(CardQuery(query.role, constrained))
        found = client.search_cards(constrained, max_results=results_per_query, order="edhrec")
        initial_returned = len(found)
        fallback = _empty_query_fallback(query, spec) if not found else None
        fallback_query = constrain_query(fallback, spec.color_identity) if fallback else None
        if fallback_query:
            found = client.search_cards(
                fallback_query, max_results=results_per_query, order="edhrec"
            )
        bucket: list[str] = []
        dropped = 0
        missing = 0
        for result in found:
            card = store.get_card(result["name"])
            if card is None:
                missing += 1
                continue
            if not eligible(card, spec):
                dropped += 1
                continue
            key = name_key(card["name"])
            candidate = candidates.setdefault(key, Candidate(card))
            candidate.roles.add(query.role)
            bucket.append(key)
        buckets.append(list(dict.fromkeys(bucket)))
        diagnostics.append(
            {
                "role": query.role,
                "query": constrained,
                "returned": len(found),
                "initial_returned": initial_returned,
                "fallback_query": fallback_query,
                "eligible": len(bucket),
                "missing_from_snapshot": missing,
                "ineligible": dropped,
                "result_limit_reached": len(found) == results_per_query,
            }
        )

    required: list[str] = []
    for name in spec.must_include:
        card = store.get_card(name)
        if card is None or not eligible(card, spec):
            raise SchemaError(
                [{"error": "must_include", "card": name, "detail": "Required card is ineligible"}]
            )
        key = name_key(card["name"])
        candidates.setdefault(key, Candidate(card, {"support"}))
        required.append(key)
    # Basic lands must be available without depending on popularity-ranked land results.
    for color in dict.fromkeys(spec.color_identity + "C"):
        name = BASIC_LANDS[color]
        for option in (name, "Snow-Covered " + name):
            card = store.get_card(option)
            if card and eligible(card, spec):
                key = name_key(card["name"])
                candidates.setdefault(key, Candidate(card)).roles.add("lands")
                required.append(key)
    required = list(dict.fromkeys(required))
    if len(required) > limit:
        raise ValueError("Candidate cap is smaller than the required cards and basic land options")
    selected = list(required)
    selected_set = set(selected)
    # Round-robin across queries prevents a broad generic search from crowding out themes.
    for row in zip_longest(*buckets):
        for key in row:
            if key is not None and key not in selected_set and len(selected) < limit:
                selected.append(key)
                selected_set.add(key)
    return CandidatePool(
        spec, [candidates[key] for key in selected], normalized_queries, diagnostics
    )
