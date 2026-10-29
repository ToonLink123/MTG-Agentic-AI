"""LLM query planning, live Scryfall discovery, and bounded local-grounded pools."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

from . import activity, prompts
from .brackets import land_denial_forbidden
from .cards import Card, name_key, trim_card
from .engine_map import clean_map
from .land_options import preferred_lands
from .llm import JSONModel, request_json
from .mana import BASIC_LANDS, is_land
from .oracle import OracleStore
from .schema import (
    LOCAL_SEARCH_SCHEMA,
    QUERY_SCHEMA,
    ROLES,
    SchemaError,
    map_schema,
    object_schema,
    validate_json,
)
from .scryfall import ScryfallClient, ScryfallError
from .spec import DeckSpec, ground_spec
from .validator import card_errors

log = logging.getLogger(__name__)

# A saved pool may outgrow the discovery limit: gap searches and drafters' outside picks add
# cards after discovery.
MAX_POOL = 1200
POOL_LIMIT = 1000  # The most candidates discovery selects.
THEME_FLOOR = 6  # Most-played cards each non-role query keeps before the fit ranking.
FIT_FADE = 30  # A card this far down a query's most-played order earns half its fit there.
LOCAL_FIRST = 15  # Theme-search hits that join outright; the rest compete on fit.
LOCAL_FIT = 4.0  # Fit the best theme-search hit adds; the others in proportion to their score.
ROLE_FLOOR = 10  # At least this many most-played cards per role query (ramp, draw, removal...).
DEFAULT_QUERIES = {
    "ramp": "otag:ramp",
    "draw": "otag:draw",
    "removal": "otag:removal",
    "wipes": "otag:board-wipe",
    "lands": "t:land",
    "creatures": "t:creature",
    "protection": 't:instant (o:hexproof or o:indestructible or o:"protection from" or o:"phase out" or o:"counter target spell" or (o:exile o:return o:battlefield))',
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
    outer_end = None
    for index, char in enumerate(query):
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
            if depth == 0 and outer_end is None:
                outer_end = index
            if depth < 0:
                raise ValueError("Unbalanced Scryfall query parentheses")
    if depth or quoted or escaped:
        raise ValueError("Unbalanced Scryfall query")
    if re.fullmatch(r"(?:W?U?B?R?G?|C)", identity) is None or not identity:
        raise ValueError("Invalid color identity")
    query = query.strip()
    suffix = f" f:commander id<={identity}"
    if query.startswith("(") and query.endswith(suffix) and outer_end == len(query) - len(suffix) - 1:
        return query
    return f"({query.strip()}) f:commander id<={identity}"


def generate_queries(spec: DeckSpec, model: JSONModel) -> list[CardQuery]:
    spec = DeckSpec.from_dict(spec.to_dict())
    messages = [
        {
            "role": "system",
            "content": prompts.QUERIES + json.dumps(QUERY_SCHEMA),
        },
        {"role": "user", "content": json.dumps(spec.to_dict())},
    ]
    log.info("discovery: planning card searches")
    with activity.agent("scout", model, "plan Scryfall searches"):
        raw = request_json(model, messages, QUERY_SCHEMA)["queries"]
    raw.extend(q.to_dict() for q in template_queries(spec))
    queries: list[CardQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        try:
            constrained = constrain_query(item["query"], spec.color_identity)
        except ValueError:
            # Preserve malformed syntax for execution feedback rather than aborting the plan.
            constrained = item["query"]
        key = (item["role"], constrained.casefold())
        if key not in seen:
            queries.append(CardQuery(item["role"], constrained))
            seen.add(key)
    # Reserve deterministic fallback coverage even when the model fills its plan.
    anchors = template_queries(spec)
    return queries[:40 - len(anchors)] + [q for q in anchors if q not in queries[:40 - len(anchors)]]


def template_queries(spec: DeckSpec) -> list[CardQuery]:
    """Always retain known role queries; a broken model label cannot suppress one."""
    raw = []
    themes = " ".join(spec.themes + spec.mechanics).lower()
    for tribe in re.findall(r"\b([a-z][a-z'-]*)\s+tribal\b", themes):
        raw.append({"role": "theme", "query": f"t:{tribe}"})
    if "divinity counter" in themes:
        raw.append({"role": "theme", "query": 'o:"divinity counter"'})
    if any("proliferat" in mechanic.lower() for mechanic in spec.mechanics):
        raw.append({"role": "mechanics", "query": "otag:proliferate"})
    if re.search(r"equipment|voltron|equipped", themes):
        raw.extend([
            {"role": "theme", "query": "t:equipment"},
            {"role": "mechanics", "query": 'o:"equipped creature"'},
            {"role": "protection", "query": 't:equipment (o:hexproof or o:shroud or o:indestructible or o:"protection from")'},
        ])
    if re.search(r"poison|infect|toxic", themes):
        raw.append({"role": "theme", "query": '(o:infect or o:toxic or o:"poison counter")'})
    if re.search(r"politic|monarch|goad", themes):
        raw.append({"role": "theme", "query": '(o:monarch or o:goad or o:vote or o:"tempting offer")'})
    if re.search(r"vanilla|no (?:rules text|abilities)", themes):
        raw.append({"role": "theme", "query": "t:creature -o:/./"})
    raw = raw[:13] + [{"role": role, "query": query} for role, query in DEFAULT_QUERIES.items()]
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
    # The local search's terms and ranked hits, and the engine map drawn from this pool ({}
    # until one is built).
    local_search: dict[str, Any] = field(default_factory=lambda: {"terms": [], "hits": []})
    engine_map: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "spec": self.spec.to_dict(),
            "cards": [candidate.to_dict() for candidate in self.cards],
            "queries": [query.to_dict() for query in self.queries],
            "diagnostics": self.diagnostics,
            "local_search": self.local_search,
            "engine_map": self.engine_map,
        }

    @classmethod
    def from_dict(cls, data: Any, store: OracleStore) -> CandidatePool:
        """Rehydrate names against the local store; saved card JSON is not trusted."""
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Expected a version-1 candidate pool")
        spec = ground_spec(DeckSpec.from_dict(data.get("spec")), store)
        raw_cards = data.get("cards")
        if not isinstance(raw_cards, list) or len(raw_cards) > MAX_POOL:
            raise ValueError(f"A candidate pool must contain at most {MAX_POOL} cards")
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
        local = data.get("local_search", {"terms": [], "hits": []})
        validate_json(local, LOCAL_SEARCH_SCHEMA)
        engine = data.get("engine_map") or {}
        if engine:
            # A saved map is re-checked like the cards: only names still in this pool remain.
            engine = {"searches": [], **engine}
            validate_json(engine, map_schema(stored=True))
            nonlands = [c.card["name"] for c in cards if not is_land(c.card)]
            engine = clean_map(engine, nonlands, spec.commander, spec.brief["asks"], notes=False)
        return cls(spec, cards, queries, diagnostics, local, engine)


def eligible(card: Card, spec: DeckSpec) -> bool:
    return (
        name_key(card["name"]) != name_key(spec.commander)
        and name_key(card["name"]) not in {name_key(n) for n in spec.exclude}
        and not card_errors(card, set(spec.color_identity) - {"C"})
        and not (spec.bracket < 3 and card["game_changer"])
        and not land_denial_forbidden(card, spec.bracket, spec.bracket3_mode)
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
    if query.role in {"theme", "mechanics", "win_condition"}:
        return next((q.query for q in template_queries(spec) if q.role == query.role), "t:creature")
    return DEFAULT_QUERIES.get(query.role)


def build_candidate_pool(
    spec: DeckSpec,
    queries: list[CardQuery],
    client: ScryfallClient,
    store: OracleStore,
    *,
    limit: int = 350,
    results_per_query: int | None = None,
    model: JSONModel | None = None,
    max_query_repairs: int = 4,
    local_search: dict[str, Any] | None = None,
) -> CandidatePool:
    """The candidate pool, filled in order: required cards and basics, preferred lands, the
    theme search's hits, each query's most-played cards (so every role stays covered), then
    every other result by fit.

    Fit counts the queries that found a card, each weighted by how narrow it is: a card that
    both the Treasure search and the dice search found outranks the next most-played card of
    "t:creature". `results_per_query=None` keeps every result, which the local BulkSearch
    returns at no cost; the live API pages 175 cards per rate-limited request."""
    if type(limit) is not int or not 1 <= limit <= POOL_LIMIT:
        raise ValueError(f"Candidate limit must be from 1 to {POOL_LIMIT}")
    if results_per_query is not None and (
        type(results_per_query) is not int or not 1 <= results_per_query <= 100_000
    ):
        raise ValueError("Results per query must be from 1 to 100000, or None for all")
    if type(max_query_repairs) is not int or not 0 <= max_query_repairs <= 8:
        raise ValueError("Query repair budget must be from 0 to 8")
    spec = ground_spec(spec, store)
    if len(queries) > 40:
        raise ValueError("At most 40 discovery queries are allowed")
    candidates: dict[str, Candidate] = {}
    buckets: list[list[str]] = []
    diagnostics: list[dict[str, Any]] = []
    normalized_queries: list[CardQuery] = []
    repairs_used = 0
    repair_schema = object_schema({"query": {"type": "string", "minLength": 1, "maxLength": 850}})
    for number, query in enumerate(queries, 1):
        log.info("discovery: search %d/%d (%s)", number, len(queries), query.role)
        if query.role not in ROLES:
            raise ValueError("Unknown discovery role")
        # Callers may supply their own plans. Always enforce the filters here too.
        history = []
        def execute(text):
            try:
                scoped = constrain_query(text, spec.color_identity)
                activity.tool("card search", f"{query.role}: {text}")
                found = client.search_cards(scoped, max_results=results_per_query, order="edhrec")
                history.append({"query": scoped, "returned": len(found), "error": None if found else "zero_results"})
                return scoped, found
            except ScryfallError as exc:
                # Transport, throttling and server failures are not planner mistakes.
                if exc.status not in {400, 422}:
                    raise
                history.append({"query": text, "returned": 0, "error": "scryfall_syntax", "detail": str(exc)})
            except ValueError as exc:
                history.append({"query": text, "returned": 0, "error": "syntax", "detail": str(exc)})
            return None, []
        constrained, found = execute(query.query)
        initial_returned = len(found)
        if not found and model is not None and repairs_used < max_query_repairs:
            repairs_used += 1
            try:
                with activity.agent("scout", model, f"repair the {query.role} search"):
                    repair = request_json(model, [
                        {"role": "system", "content": prompts.QUERY_REPAIR},
                        {"role": "user", "content": json.dumps({"spec": spec.to_dict(), "role": query.role, "feedback": history[-1]})},
                    ], repair_schema)
                repaired, found = execute(repair["query"])
                constrained = repaired or constrained
            except SchemaError as exc:
                history.append({"error": "repair_schema", "details": exc.errors})
        fallback = _empty_query_fallback(query, spec) if not found else None
        fallback_query = constrain_query(fallback, spec.color_identity) if fallback else None
        if fallback_query:
            constrained, found = execute(fallback_query)
        # Saved plans contain valid scoped syntax, including when every attempt failed.
        saved_query = constrained or constrain_query(DEFAULT_QUERIES.get(query.role, "t:creature"), spec.color_identity)
        normalized_queries.append(CardQuery(query.role, saved_query))
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
                "query": query.query,
                "attempts": history,
                "returned": len(found),
                "initial_returned": initial_returned,
                "fallback_query": fallback_query,
                "eligible": len(bucket),
                "missing_from_snapshot": missing,
                "ineligible": dropped,
                "result_limit_reached": results_per_query is not None and len(found) == results_per_query,
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
    # Retain premium fixing even when broad EDHREC searches or the model omit it.
    # Leave room for a complete nonland selection in smaller pools.
    land_budget = max(0, limit - len(selected) - (99 - spec.role_targets["lands"]))
    for card in preferred_lands(store):
        key = name_key(card["name"])
        if key in selected_set or not eligible(card, spec):
            continue
        candidates.setdefault(key, Candidate(card)).roles.add("lands")
        if land_budget:
            selected.append(key)
            selected_set.add(key)
            land_budget -= 1
    def take(key: str) -> None:
        if key not in selected_set and len(selected) < limit:
            selected.append(key)
            selected_set.add(key)

    # The theme search's best hits join outright; the rest add their relevance to fit below.
    local_search = local_search or {"terms": [], "hits": []}
    local_fit: dict[str, float] = {}
    top_score = max((hit["score"] for hit in local_search["hits"]), default=0) or 1
    for hit in local_search["hits"]:
        card = store.get_card(hit["name"])
        if card is None or is_land(card) or not eligible(card, spec):
            continue
        key = name_key(card["name"])
        candidates.setdefault(key, Candidate(card)).roles.add("theme")
        local_fit[key] = LOCAL_FIT * hit["score"] / top_score
        if len(local_fit) <= LOCAL_FIRST:
            take(key)

    # Each query's most-played cards first, taking turns, so a broad generic search cannot
    # crowd out a theme and every role keeps its staples: three times a role's target across
    # its queries, a few for every other query, within three quarters of the remaining room.
    found = [(query.role, bucket) for query, bucket in zip(queries, buckets) if bucket]
    per_role = Counter(role for role, _ in found)
    targets = spec.role_targets

    def wanted(role: str, bucket: list[str]) -> int:
        target = targets.get(role, 0) if role != "lands" else 0
        share = max(ROLE_FLOOR, math.ceil(3 * target / per_role[role])) if target else THEME_FLOOR
        return min(len(bucket), share)

    floors = [wanted(role, bucket) for role, bucket in found]
    room = (limit - len(selected)) * 3 / 4
    if sum(floors) > room:
        floors = [max(1, int(f * room / sum(floors))) for f in floors]
    before = len(selected)
    for row in zip_longest(*(bucket[:f] for (_, bucket), f in zip(found, floors))):
        for key in row:
            if key is not None:
                take(key)
    floored = len(selected) - before
    # The rest by fit. Every query that found a card adds its weight (more for a narrower
    # query), faded by how far down that query's most-played order the card sits, and
    # theme-search relevance adds to it: a card deep in two theme searches outranks the next
    # most-played card of one broad search.
    total = len({key for _, bucket in found for key in bucket})
    fit: dict[str, float] = dict(local_fit)
    first: dict[str, int] = {key: index for index, key in enumerate(local_fit)}
    for _, bucket in found:
        weight = math.log(1 + total / len(bucket))
        for position, key in enumerate(bucket):
            fit[key] = fit.get(key, 0.0) + weight / (1 + position / FIT_FADE)
            first.setdefault(key, len(first))

    for key in sorted(fit, key=lambda key: (-fit[key], first[key])):
        take(key)
    log.info("discovery: %d candidates (%d from query floors, %d by fit)",
             len(selected), floored, len(selected) - before - floored)
    return CandidatePool(
        spec, [candidates[key] for key in selected], normalized_queries, diagnostics,
        local_search,
    )
