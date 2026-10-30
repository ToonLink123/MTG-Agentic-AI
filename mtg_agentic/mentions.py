"""Find the cards a player names in a question, in plain words.

A rules question can name its cards the way players say them: any capitalization, hyphens and
apostrophes ("fable of the mirror breaker"), either face of a double-faced card, or a legendary
card's short name before its comma ("Kiki-Jiki", "Atraxa"). Code scans the words for the
longest names it knows from the local card data. A name in [[double brackets]] is looked up
like a commander instead: exact first, then Scryfall's fuzzy match, so it may be misspelled.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .cards import Card, name_key
from .oracle import OracleStore
from .scryfall import ScryfallError

WORD = re.compile(r"[a-z0-9]+")
# A one-word name that more cards than this use in their rules text is a rules word, such as
# Flash, Exile or Sacrifice; it counts as a card only in brackets.
COMMON = 40
FULL, FACE, SHORT = 0, 1, 2  # How a name matched; a fuller match wins a tie.


def words(text: str) -> tuple[str, ...]:
    """Lowercase words without accents or apostrophes: "Praetors' Voice" -> praetors, voice."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return tuple(WORD.findall(plain.replace("'", "")))


def name_index(store: OracleStore) -> dict[tuple[str, ...], dict[str, int]]:
    """Every Commander-legal or banned card's full name, face names and, for a legendary card,
    the short name before its comma, as words, mapped to {card name: how it matched}."""
    index: dict[tuple[str, ...], dict[str, int]] = {}

    def add(text: str, card: Card, kind: int) -> None:
        key = words(text)
        if key:
            names = index.setdefault(key, {})
            names[card["name"]] = min(kind, names.get(card["name"], kind))

    for card in store.cards():
        # Alchemy rebalances and Un-cards are not_legal; asking about a banned card is fine.
        if card.get("legalities", {}).get("commander", "not_legal") == "not_legal":
            continue
        add(card["name"], card, FULL)
        faces = card.get("card_faces") or []
        for face in faces:
            add(face.get("name", ""), card, FACE)
        front = faces[0] if faces else card
        if "," in front.get("name", "") and "Legendary" in front.get("type_line", ""):
            add(front["name"].split(",")[0], card, SHORT)
    return index


def _pick(names: dict[str, int], prefer: set[str]) -> list[str]:
    """The best candidates: the player's own cards first, then the fullest kind of match."""
    rank = {n: (name_key(n) not in prefer, kind) for n, kind in names.items()}
    best = min(rank.values())
    return sorted(n for n, r in rank.items() if r == best)


def find_cards(
    text: str, store: OracleStore, *, prefer: Iterable[str] = (),
    index: dict[tuple[str, ...], dict[str, int]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The cards `text` names, in order, and each name it could not settle to one card, with
    the cards it could be. `prefer` (a commander, a deck) settles short names such as Jodah."""
    index = index if index is not None else name_index(store)
    preferred = {name_key(n) for n in prefer}
    longest = max(map(len, index), default=0)
    tokens = words(text)
    found: list[str] = []
    unsettled: list[dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        for size in range(min(longest, len(tokens) - i), 0, -1):
            key = tokens[i:i + size]
            names = index.get(key)
            if not names:
                continue
            if size == 1 and (len(key[0]) < 3 or store.text_count(key[0], column="oracle_text") > COMMON):
                continue
            options = _pick(names, preferred)
            if len(options) == 1:
                found.append(options[0])
            else:
                unsettled.append({"name": " ".join(key), "options": options[:6]})
            i += size
            break
        else:
            i += 1
    return list(dict.fromkeys(found)), unsettled


def lookup(
    name: str, store: OracleStore, client: Any = None, *, prefer: Iterable[str] = (),
    index: dict[tuple[str, ...], dict[str, int]] | None = None,
) -> Card | None:
    """One card a player named, the way a commander is found: the exact name, then the same
    words in any capitalization or punctuation, then Scryfall's fuzzy match."""
    card = store.get_card(name)
    if card is not None:
        return card
    index = index if index is not None else name_index(store)
    names = index.get(words(name))
    if names:
        options = _pick(names, {name_key(n) for n in prefer})
        if len(options) == 1:
            return store.get_card(options[0])
    if client is None:
        return None
    try:
        return store.get_card(client.get_card(name)["name"])
    except (ScryfallError, ValueError, OSError):  # Offline, an unknown name stays unknown.
        return None
