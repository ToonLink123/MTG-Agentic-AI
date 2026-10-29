"""The engine map: one shared picture of how the deck works, drawn before anyone drafts.

A model reads the player's brief, the commander and the candidate pool, with the local search's
ranking as a hint and known Commander Spellbook combos as evidence, and traces the deck's
resource loop: what makes the resource, what multiplies it, what spends it, what pays off and
what protects the key pieces. Code keeps only pool cards on the map and reports how many names it
dropped, checks every card ask against the map, and records what the pool could not provide.

The map can also fill its own gaps: the model writes a few searches for what is missing,
code runs them over the whole local card database (free, no rate limit), adds the eligible
finds to the pool, and the model places them on the map in one more call. Drafters and
every council member then build from the same map.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any

from . import activity, prompts
from .cards import name_key, trim_card
from .llm import JSONModel, request_json
from .mana import is_land
from .schema import MAP_STAGES, SchemaError, map_schema
from .scryfall import ScryfallError
from .sources import SourceError

if TYPE_CHECKING:
    from .brackets import SpellbookClient
    from .discovery import CandidatePool
    from .oracle import OracleStore

log = logging.getLogger(__name__)
GAP_SEARCHES = 6  # Searches the mapper may run for its gaps, each adding at most:
PER_SEARCH = 8


def clean_map(
    raw: dict[str, Any], pool_names: list[str], commander: str, asks: list[dict[str, Any]],
    *, combos: list[dict[str, Any]] = (), notes: bool = True,
) -> dict[str, Any]:
    """Only pool cards stay on the map, under their exact names; `notes` adds code's findings
    (dropped names, card asks the map leaves uncovered) to the gaps."""
    canonical = {name_key(n): n for n in pool_names}
    ends = {**canonical, name_key(commander): commander}
    dropped: set[str] = set()

    def known(name: str, table: dict[str, str]) -> str | None:
        found = table.get(name_key(name))
        if found is None:
            dropped.add(name)
        return found

    stages: dict[str, dict[str, Any]] = {}
    for stage in raw["stages"]:
        if stage["stage"] in stages:
            continue
        cards: dict[str, dict[str, Any]] = {}
        for card in stage["cards"]:
            name = known(card["name"], canonical)
            if name is not None:
                cards.setdefault(name, {**card, "name": name})
        stages[stage["stage"]] = {**stage, "cards": list(cards.values())}
    pairs = {frozenset(name_key(n) for n in combo["cards"]) for combo in combos}
    links = []
    for link in raw["links"]:
        start, end = known(link["from"], ends), known(link["to"], ends)
        if start is None or end is None or start == end:
            continue
        # A "combo" link must be a pair Commander Spellbook actually listed.
        proven = frozenset((name_key(start), name_key(end))) in pairs
        links.append({**link, "from": start, "to": end,
                      "source": "combo" if link["source"] == "combo" and proven else "oracle"})
    entries: dict[str, dict[str, Any]] = {}
    for entry in raw["asks"]:
        if entry["ask"] not in entries:
            cards = [known(n, canonical) for n in entry["cards"]]
            entries[entry["ask"]] = {**entry, "cards": list(dict.fromkeys(n for n in cards if n))}
    found: list[str] = []
    covered = []
    for ask in asks:
        entry = entries.get(ask["id"], {"ask": ask["id"], "cards": [], "note": ""})
        covered.append(entry)
        if ask["kind"] == "cards" and not entry["cards"]:
            found.append(f"No pool card serves {ask['id']}: \"{ask['quote'][:120]}\"")
    if dropped:
        found.append(f"{len(dropped)} name(s) on the map are not pool cards and were dropped: "
                     + ", ".join(sorted(dropped)[:5]))
    return {
        "summary": raw["summary"], "resources": raw["resources"],
        "stages": [stages[s] for s in MAP_STAGES if s in stages],
        "links": links, "asks": covered,
        "gaps": ([*found, *raw["gaps"]] if notes else list(raw["gaps"]))[:8],
        "searches": [s for s in raw.get("searches", []) if "added" in s],
    }


def _fill_gaps(
    pool: CandidatePool, searches: list[dict[str, str]], store: OracleStore, search: Any,
) -> tuple[CandidatePool, list[dict[str, Any]]]:
    """Run the mapper's gap searches locally; eligible nonland finds join the pool."""
    from .discovery import MAX_POOL, Candidate, constrain_query, eligible  # It imports this module.

    spec = pool.spec
    asks = {a["id"] for a in spec.brief["asks"]}
    present = {name_key(c.card["name"]) for c in pool.cards}
    cards = list(pool.cards)
    done = []
    for item in searches[:GAP_SEARCHES]:
        entry = {"for": item["for"], "query": item["query"], "added": [], "error": ""}
        try:
            scoped = constrain_query(item["query"], spec.color_identity)
            activity.tool("card search", f"gap {item['for']}: {item['query']}")
            found = search.search_cards(scoped, max_results=200, order="edhrec")
        except (ScryfallError, ValueError) as exc:
            entry["error"] = str(exc)[:300]
            done.append(entry)
            continue
        for result in found:
            if len(entry["added"]) >= PER_SEARCH or len(cards) >= MAX_POOL:
                break
            card = store.get_card(result["name"])
            if card is None or is_land(card) or name_key(card["name"]) in present:
                continue
            if not eligible(card, spec):
                continue
            present.add(name_key(card["name"]))
            # A find for one of the player's asks is a theme card; any other is support.
            cards.append(Candidate(card, {"theme" if item["for"] in asks else "support"}))
            entry["added"].append(card["name"])
        done.append(entry)
    return dataclasses.replace(pool, cards=cards), done


def build_engine_map(
    pool: CandidatePool, model: JSONModel, store: OracleStore, *,
    spellbook: SpellbookClient | None = None, search: Any = None,
) -> CandidatePool:
    """The pool with its engine map, plus any cards the map's gap searches found (only with
    `search`, the local BulkSearch). Raises SchemaError on a malformed first answer."""
    spec = pool.spec
    leader = store.get_card(spec.commander)
    nonlands = [c for c in pool.cards if not is_land(c.card)]
    ranks = {
        name_key(hit["name"]): {"rank": rank, "matched": hit["matched"]}
        for rank, hit in enumerate(pool.local_search.get("hits", []), 1)
    }
    candidates = []
    for candidate in nonlands:
        row = candidate.to_dict()
        hit = ranks.get(name_key(candidate.card["name"]))
        candidates.append({**row, "search": hit} if hit else row)
    combos: list[dict[str, Any]] = []
    if spellbook is not None and nonlands:
        activity.tool("combos", f"Commander Spellbook across {len(nonlands)} candidates")
        try:
            found = spellbook.find_combos({c.card["name"]: 1 for c in nonlands}, spec.commander)
            combos = [{"cards": c["cards"], "results": c["results"][:4]} for c in found["combos"]]
        except (SourceError, OSError, ValueError) as exc:
            log.warning("engine map: no Commander Spellbook combos (%s)", exc)
    asks = spec.brief["asks"]
    messages = [
        {"role": "system", "content": prompts.ENGINE_MAP},
        {"role": "user", "content": json.dumps({
            "spec": spec.to_dict(), "commander": trim_card(leader) if leader else None,
            "candidates": candidates, "combos": combos[:40],
        })},
    ]
    ask_ids = [a["id"] for a in asks]
    log.info("engine map: tracing the resource loop across %d candidates", len(candidates))
    with activity.agent("mapper", model, f"map the engine across {len(candidates)} candidates"):
        raw = request_json(model, messages, map_schema(
            ask_ids, searches=GAP_SEARCHES if search is not None else 0,
        ))
    searched: list[dict[str, Any]] = []
    if search is not None and raw["searches"]:
        pool, searched = _fill_gaps(pool, raw["searches"], store, search)
        added = [name for entry in searched for name in entry["added"]]
        if added:
            # One more call: the model places what its searches found.
            found = [c.to_dict() for c in pool.cards if c.card["name"] in set(added)]
            messages += [
                {"role": "assistant", "content": json.dumps(raw)},
                {"role": "user", "content": json.dumps({
                    "searches": searched, "added_candidates": found,
                    "instruction": "Your searches added these candidates. Return the complete "
                    "map again: put each added card where it serves the loop, update asks and "
                    "gaps, and keep everything else that still holds.",
                })},
            ]
            try:
                with activity.agent("mapper", model, f"place {len(added)} cards its searches found"):
                    raw = request_json(model, messages, map_schema(ask_ids))
            except SchemaError:
                log.warning("engine map: the updated map was invalid; keeping the first one")
    nonlands = [c.card["name"] for c in pool.cards if not is_land(c.card)]
    result = clean_map({**raw, "searches": searched}, nonlands, spec.commander, asks,
                       combos=combos)
    on_map = sum(len(stage["cards"]) for stage in result["stages"])
    added = sum(len(entry["added"]) for entry in searched)
    log.info("engine map: %d cards in %d stages, %d gaps, %d cards found by %d gap searches",
             on_map, len(result["stages"]), len(result["gaps"]), added, len(searched))
    return dataclasses.replace(pool, engine_map=result)
