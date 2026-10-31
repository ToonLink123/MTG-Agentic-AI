"""Card discovery built like rules retrieval: a graph walk, meaning and words, fused by rank.

Cards link through what they share: Scryfall Tagger tags, keywords and Commander Spellbook
combos. A feature on too many cards (more than 4%) explains nothing and is left out, as the rules
graph leaves out hub terms. A short personalized walk (card, shared feature, card) from the
commander and the must-include cards finds the cards most tied to them; from one card it finds
cards that do the same job, for "similar card" swaps. When the card index holds EmbeddingGemma
vectors of every card's rules text (setup builds it), the deck's themes also rank cards by
meaning. Reciprocal rank fusion joins the walk, the meaning ranking and the local word search
(retrieval.py), and the fused hits feed discovery like the word search alone used to.
"""

from __future__ import annotations

import logging
import math
from array import array
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from . import activity
from .cards import Card, name_key, rules_text
from .mana import is_land
from .oracle import OracleStore
from .retrieval import local_search
from .spec import DeckSpec

log = logging.getLogger(__name__)
HUB_FRACTION, HUB_MIN = 0.04, 50
WALK_STEPS, RESTART = 3, 0.3
RRF_K = 60
CARD_DIMENSION = 256  # EmbeddingGemma 2's Matryoshka size for the card index: small and fast.
FUSED_HITS = 40


def universe(spec: DeckSpec, store: OracleStore) -> list[Card]:
    """The cards a deck may play (eligible nonlands), plus its commander and must-include
    cards, which seed the walk."""
    from .discovery import eligible  # discovery imports retrieval, which this module uses.

    colors = set(spec.color_identity) - {"C"}
    seeds = {name_key(n) for n in [spec.commander, *spec.must_include]}
    cards = []
    for card in store.cards():
        if name_key(card["name"]) in seeds:
            cards.append(card)
        elif (
            card.get("legalities", {}).get("commander") == "legal"
            and set(card["color_identity"]) <= colors and not is_land(card) and eligible(card, spec)
        ):
            cards.append(card)
    return cards


class CardGraph:
    """Cards and the features they share, for personalized walks."""

    def __init__(self, store: OracleStore, cards: list[Card]):
        self.names = [card["name"] for card in cards]
        self.position = {name_key(card["name"]): i for i, card in enumerate(cards)}
        tags = store.card_tags() if store.tags_ready else {}
        combos = store.combo_index() if store.combos_ready else {}
        members: dict[str, list[int]] = defaultdict(list)
        for i, card in enumerate(cards):
            for slug in tags.get(card.get("oracle_id"), ()):
                members["tag " + slug].append(i)
            for keyword in card.get("keywords") or ():
                members["keyword " + keyword.lower()].append(i)
            for combo in combos.get(name_key(card["name"]), ()):
                members["combo " + combo].append(i)
        hub = max(HUB_MIN, int(len(cards) * HUB_FRACTION))
        self.members = {f: cards_ for f, cards_ in members.items() if 2 <= len(cards_) <= hub}
        self.features: list[list[str]] = [[] for _ in cards]
        for feature, indexes in self.members.items():
            for i in indexes:
                self.features[i].append(feature)

    def __contains__(self, name: str) -> bool:
        return name_key(name) in self.position

    def walk(
        self, seeds: dict[str, float], *, steps: int = WALK_STEPS, exclude: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Each card's share of a short walk from `seeds` (name -> weight): card to a feature it
        has, feature to a card that has it, with a restart. `exclude` (by default every seed)
        is left out of the result."""
        start = {self.position[name_key(n)]: w for n, w in seeds.items() if name_key(n) in self}
        left_out = set(start) if exclude is None else {
            self.position[name_key(n)] for n in exclude if name_key(n) in self
        }
        total = sum(start.values())
        if not total:
            return {}
        current = {i: w / total for i, w in start.items()}
        score: dict[int, float] = defaultdict(float)
        for _ in range(steps):
            mass: dict[str, float] = defaultdict(float)
            for i, weight in current.items():
                if self.features[i]:
                    share = weight / len(self.features[i])
                    for feature in self.features[i]:
                        mass[feature] += share
            current = defaultdict(float)
            for feature, weight in mass.items():
                share = (1 - RESTART) * weight / len(self.members[feature])
                for i in self.members[feature]:
                    current[i] += share
            for i, weight in current.items():
                score[i] += weight
        return {self.names[i]: s for i, s in score.items() if i not in left_out}

    def shared(self, first: str, second: str, limit: int = 4) -> list[str]:
        """The features two cards share, narrowest first."""
        a, b = self.position.get(name_key(first)), self.position.get(name_key(second))
        if a is None or b is None:
            return []
        common = set(self.features[a]) & set(self.features[b])
        return sorted(common, key=lambda f: (len(self.members[f]), f))[:limit]


def card_text(card: Card) -> str:
    faces = card.get("card_faces") or [card]
    parts = [
        ". ".join(filter(None, [face.get("type_line", ""), rules_text(face.get("oracle_text", ""))]))
        for face in faces
    ]
    return " // ".join(parts)


def card_identity(model: str, revision: str | None, format_version: str) -> str:
    """The identity an EmbeddingGemma embedder of CARD_DIMENSION reports, without loading it."""
    return f"hf:{model}@{revision or 'main'}:{CARD_DIMENSION}:{format_version}"


def index_ready(store: OracleStore, identity: str) -> bool:
    info = store.vector_metadata
    return (
        info.get("embedding") == identity and info.get("vector_count", 0) > 0
        and info.get("card_count") == store.metadata.get("card_count")
    )


def index_cards(store: OracleStore, embedder: Any, *, batch: int = 64) -> int:
    """Embed every Commander-legal nonland card's rules text into the card index."""
    cards = [
        card for card in store.cards()
        if card.get("legalities", {}).get("commander") == "legal" and not is_land(card)
    ]
    rows: list[tuple[str, bytes]] = []
    for start in range(0, len(cards), batch):
        chunk = cards[start:start + batch]
        vectors = embedder.embed([card_text(c) for c in chunk], titles=[c["name"] for c in chunk])
        rows += [(c["oracle_id"], array("f", v).tobytes()) for c, v in zip(chunk, vectors)]
        if (start // batch) % 50 == 0:
            log.info("card index: %d of %d cards embedded", len(rows), len(cards))
    return store.import_card_vectors(rows, metadata={
        "embedding": embedder.identity, "card_count": store.metadata.get("card_count"),
    })


def meaning(store: OracleStore, embedder: Any, query: str, oracle_ids: set[str]) -> dict[str, float]:
    """Cosine of each card's vector to `query`, by Oracle ID, for the cards in `oracle_ids`.
    Empty when the index was embedded by another model than `embedder`."""
    if store.vector_metadata.get("embedding") != embedder.identity:
        log.warning("card search: the card index does not match %s, so no meaning ranking",
                    embedder.identity)
        return {}
    try:
        import numpy
    except ImportError:
        log.warning("card search: numpy is missing, so no meaning ranking")
        return {}
    ids, rows = [], []
    for oracle_id, blob in store.card_vectors():
        if oracle_id in oracle_ids:
            ids.append(oracle_id)
            rows.append(numpy.frombuffer(blob, dtype=numpy.float32))
    if not rows:
        return {}
    matrix = numpy.vstack(rows)
    vector = numpy.asarray(embedder.embed([query], query=True)[0], dtype=numpy.float32)
    norms = numpy.linalg.norm(matrix, axis=1) * (numpy.linalg.norm(vector) or 1.0)
    scores = matrix @ vector / numpy.where(norms == 0, 1.0, norms)
    return dict(zip(ids, scores.tolist()))


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    order = sorted((k for k, v in scores.items() if v > 0), key=lambda k: (-scores[k], k))
    return {key: rank for rank, key in enumerate(order)}


def theme_query(spec: DeckSpec) -> str:
    """The deck's themes in words, for the meaning ranking."""
    return "; ".join(dict.fromkeys([*spec.themes, *spec.mechanics, *spec.win_conditions, *spec.search_terms]))


def card_search(
    spec: DeckSpec, store: OracleStore, *, embedder: Any = None, limit: int = FUSED_HITS,
) -> dict[str, Any]:
    """The deck's best-fitting eligible nonlands, shaped like local_search's result: the word
    search, a graph walk from the commander and must-include cards (with the word search's best
    hits as weaker seeds), and with a matching card index the meaning ranking, fused by
    reciprocal rank. `matched` lists the terms a card matched, "synergy" for a walk hit and
    "meaning" for a meaning hit."""
    words = local_search(spec, store, limit=200)
    cards = universe(spec, store)
    if not cards:
        return {"terms": words["terms"], "hits": words["hits"][:limit]}
    graph = CardGraph(store, cards)
    seeds = {spec.commander: 1.0, **{name: 1.0 for name in spec.must_include}}
    seeds.update({hit["name"]: 0.3 for hit in words["hits"][:10] if hit["name"] not in seeds})
    activity.tool("card graph", f"walk from {spec.commander} across {len(cards):,} cards")
    # The word search's hits are weak seeds that may still be hits themselves.
    walked = graph.walk(seeds, exclude=[spec.commander, *spec.must_include])
    excluded = {name_key(n) for n in [spec.commander, *spec.must_include]}
    playable = {c["name"]: c for c in cards if name_key(c["name"]) not in excluded}
    rankings = [
        {hit["name"]: len(words["hits"]) - i for i, hit in enumerate(words["hits"])},
        {name: s for name, s in walked.items() if name in playable},
    ]
    meant: dict[str, float] = {}
    query = theme_query(spec)
    if embedder is not None and query:
        activity.tool("card meaning", query[:80])
        by_id = {c["oracle_id"]: c["name"] for c in playable.values() if c.get("oracle_id")}
        meant = {by_id[i]: s for i, s in meaning(store, embedder, query, set(by_id)).items()}
        rankings.append(meant)
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for name, rank in _ranks(ranking).items():
            fused[name] += 1 / (RRF_K + rank + 1)
    matched = {hit["name"]: list(hit["matched"]) for hit in words["hits"]}
    synergy = set(sorted(rankings[1], key=lambda n: -rankings[1][n])[:200])
    meaningful = set(sorted(meant, key=lambda n: -meant[n])[:200])
    hits = []
    for name in sorted(fused, key=lambda n: (-fused[n], n))[:limit]:
        labels = matched.get(name, [])[:4]
        labels += ["synergy"] * (name in synergy) + ["meaning"] * (name in meaningful)
        hits.append({"name": name, "score": round(fused[name] * 10000), "matched": labels[:6]})
    log.info("discovery: card search fused %d rankings into %d hits", len(rankings), len(hits))
    return {"terms": words["terms"], "hits": hits}


def similar(graph: CardGraph, name: str, allowed: Iterable[str], limit: int = 15) -> list[dict[str, Any]]:
    """Cards that do what `name` does, by a walk from it alone, among `allowed` names."""
    allowed = {name_key(n): n for n in allowed}
    walked = graph.walk({name: 1.0})
    ranked = sorted((n for n in walked if name_key(n) in allowed), key=lambda n: -walked[n])
    return [
        {"name": n, "shares": graph.shared(name, n), "score": round(math.log1p(walked[n] * 1e4), 2)}
        for n in ranked[:limit]
    ]
