"""Local card search: the fast, model-free half of discovery.

Terms come from the player's words (request, interview answers, asks, themes) and from the
commander's Oracle text. Code drops words too common to mean anything, then ranks cards by BM25
over their name, type line and text, so rare terms such as die rolls or divinity counters weigh
more than common ones, and a card matching several terms beats one matching a single word. The
engine map's model then judges the ranked cards; the rank is a hint, never a verdict.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from . import activity
from .cards import Card
from .discovery import eligible
from .mana import is_land
from .oracle import OracleStore
from .spec import DeckSpec

# Words that say nothing about which cards fit: English glue, deckbuilding talk and rules
# vocabulary that nearly every card shares.
STOP = frozenset("""
a an and any are as at be been but by can could do does each for from get gets go had has have
he her his how i if in into is it its just like me more most my no not of on one only or our
out over own so some such than that the their them then there these they this those to too up
us very was we what when where which while who will with would you your also about after again
all another around because before being both card cards deck decks build building play plays
playing played player players commander commanders game games really lot lots lean focus theme
themed themes strategy style good great want wants something things thing kind using use used
way ways much many few little big small fast slow casual bracket win wins winning
target targets control controls turn turns spell spells ability abilities battlefield graveyard
library hand mana cost costs until instead additional whenever creature creatures permanent
permanents enters enter leaves under onto spent activate higher lower less among gains gain
equal number times plus tapped untap untapped other may put puts return returns
""".split())
WORD = re.compile(r"[a-z][a-z'-]*[a-z]")
QUOTED = re.compile(r"[\"“]([^\"”\n]{3,60})[\"”]")
PLAYER, COMMANDER = 1.5, 1.0  # The player's words outrank the commander's text.

log = logging.getLogger(__name__)


def _key(term: str) -> str:
    """A crude stem, so "treasure" and "treasures" are searched once."""
    return " ".join(word.rstrip("s") for word in term.split())


def _terms_from(text: str, weight: float, terms: dict[str, tuple[str, float]]) -> None:
    lowered = text.lower()
    found = [m[1] for m in QUOTED.finditer(text)]
    words = WORD.findall(re.sub(r"\([^)]*\)", " ", lowered))  # Reminder text says little.
    for left, right in zip(words, words[1:]):
        if left not in STOP and right not in STOP and len(left) >= 3 and len(right) >= 3:
            found.append(f"{left} {right}")
    found += [w for w in words if w not in STOP and len(w) >= 4]
    for term in found:
        term = " ".join(term.lower().split())
        key = _key(term)
        if key and terms.get(key, ("", 0.0))[1] < weight:
            terms[key] = (term, weight)


def search_terms(spec: DeckSpec, commander: Card | None) -> dict[str, tuple[str, float]]:
    """Candidate terms keyed by stem: (term, weight) from the player's words and the commander."""
    terms: dict[str, tuple[str, float]] = {}
    # The commander's name, in its own text or the player's words, would only find itself.
    names: set[str] = set()
    if commander is not None:
        names = {commander["name"], commander["name"].split(",")[0]}
        names.update(face.get("name", "") for face in commander.get("card_faces", []))
    names = {name for name in names if name.strip()}

    def unnamed(text: str) -> str:
        for name in sorted(names, key=len, reverse=True):
            text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
        return text

    if commander is not None:
        _terms_from(unnamed(commander.get("oracle_text", "")), COMMANDER, terms)
    brief = spec.brief
    player = [
        brief["request"],
        *(a["answer"] for a in brief["interview"] if a["source"] != "agents"),
        *(a["quote"] for a in brief["asks"]),
        *spec.themes, *spec.mechanics, *spec.win_conditions,
    ]
    for text in player:
        _terms_from(unnamed(text), PLAYER, terms)
    return terms


def local_search(
    spec: DeckSpec, store: OracleStore, *, limit: int = 40, max_terms: int = 16
) -> dict[str, Any]:
    """Eligible nonland cards ranked by how well they match the deck's terms."""
    total = store.metadata.get("card_count", 0)
    if not total:
        return {"terms": [], "hits": []}
    common = max(25, int(total * 0.04))
    commander = store.get_card(spec.commander)
    useful = []
    for term, weight in search_terms(spec, commander).values():
        count = store.text_count(term)
        if 0 < count <= common:
            useful.append({"term": term, "weight": weight, "cards": count,
                           "value": weight * math.log(1 + total / count)})
    kept = []
    seen: set[tuple[str, ...]] = set()
    for term in sorted(useful, key=lambda t: (-t["value"], t["term"])):
        rows = store.search_text(term["term"])
        # The index stems words, so "roll", "rolled" and "rolling" find the same cards: one
        # term, counted once.
        found = tuple(name for name, _ in rows)
        if found in seen:
            continue
        seen.add(found)
        kept.append({**term, "rows": rows})
        if len(kept) == max_terms:
            break
    activity.tool("local search", ", ".join(t["term"] for t in kept) or "no useful terms")
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for term in kept:
        for name, relevance in term["rows"]:
            scores[name] = scores.get(name, 0.0) + term["weight"] * relevance
            matched.setdefault(name, []).append(term["term"])
    hits = []
    top = None
    for name in sorted(scores, key=lambda n: (-scores[n], n)):
        card = store.get_card(name)
        if card is None or is_land(card) or not eligible(card, spec):
            continue
        top = top or scores[name]
        if scores[name] < top * 0.2 or len(hits) >= limit:
            break
        hits.append({"name": card["name"], "score": round(scores[name]),
                     "matched": matched[name][:6]})
    log.info("discovery: local search found %d cards from %d terms", len(hits), len(kept))
    return {
        "terms": [{"term": t["term"], "cards": t["cards"]} for t in kept],
        "hits": hits,
    }
