"""Scryfall search syntax over the local bulk data, instead of the rate-limited search API.

Discovery and the orchestrator write ordinary Scryfall queries (t:, o:, otag:, mv, c:, id<=,
f:commander, -negation, or, parentheses) and this evaluates them against the Oracle Cards
snapshot, with otag: answered from Scryfall's Oracle Tags bulk file. Results match
ScryfallClient.search_cards: trimmed cards ordered by name or EDHREC rank. A key the snapshot
cannot answer (set, rarity, price, artist...) raises the same kind of error as a Scryfall 400,
which the query-repair step already handles. Scryfall caps /cards/search at 2 requests a
second; a local query has no limit and takes milliseconds.
"""

from __future__ import annotations

import operator
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cards import Card, trim_card
from .oracle import OracleStore
from .scryfall import ScryfallError
from .validator import can_be_commander

COLOR_NAMES = {
    "white": "W", "blue": "U", "black": "B", "red": "R", "green": "G",
    "azorius": "WU", "dimir": "UB", "rakdos": "BR", "gruul": "RG", "selesnya": "GW",
    "orzhov": "WB", "izzet": "UR", "golgari": "BG", "boros": "RW", "simic": "GU",
    "bant": "GWU", "esper": "WUB", "grixis": "UBR", "jund": "BRG", "naya": "RGW",
    "abzan": "WBG", "jeskai": "URW", "sultai": "BGU", "mardu": "RWB", "temur": "GUR",
}
KEYS = {
    "t": "type", "type": "type", "o": "oracle", "oracle": "oracle", "fo": "full",
    "fulloracle": "full", "name": "name", "n": "name", "kw": "keyword", "keyword": "keyword",
    "otag": "otag", "oracletag": "otag", "function": "otag", "mv": "mv", "cmc": "mv",
    "manavalue": "mv", "pow": "power", "power": "power", "tou": "toughness",
    "toughness": "toughness", "c": "color", "color": "color", "id": "identity",
    "identity": "identity", "ci": "identity", "f": "format", "format": "format",
    "legal": "format", "is": "is", "not": "not", "m": "mana", "mana": "mana",
    "produces": "produces",
}
COMPARE = {
    ":": operator.eq, "=": operator.eq, "!=": operator.ne, "<": operator.lt,
    "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}
PERMANENTS = ("artifact", "battle", "creature", "enchantment", "land", "planeswalker")
REMINDER = re.compile(r"\([^)]*\)")
TERM = re.compile(r"([A-Za-z]+)(!=|<=|>=|:|=|<|>)")

Predicate = Callable[["_Row"], bool]


class QueryError(ScryfallError):
    """A query the local search cannot run: discovery repairs it like a Scryfall 400."""

    def __init__(self, detail: str):
        super().__init__(detail, status=400, code="bad_request")


@dataclass(frozen=True)
class _Value:
    kind: str  # "word", "text" (quoted) or "regex"
    text: str


@dataclass
class _Row:
    """One card with everything a predicate reads, lowered once."""

    card: Card
    name: str
    names: frozenset[str]
    types: str
    full: str
    plain: str
    keywords: frozenset[str]
    colors: frozenset[str]
    identity: frozenset[str]
    power: tuple[float, ...]
    toughness: tuple[float, ...]

    @classmethod
    def of(cls, card: Card) -> _Row:
        faces = card.get("card_faces") or []
        full = card.get("oracle_text", "").lower()

        def stats(key: str) -> tuple[float, ...]:
            values = [card.get(key)] + [face.get(key) for face in faces]
            return tuple(_number(v) for v in values if isinstance(v, str) and v)

        return cls(
            card=card, name=card["name"].lower(),
            names=frozenset([card["name"].lower(), *(f["name"].lower() for f in faces)]),
            types=card.get("type_line", "").lower(), full=full, plain=REMINDER.sub("", full),
            keywords=frozenset(k.lower() for k in card.get("keywords") or []),
            colors=frozenset(card.get("colors") or []),
            identity=frozenset(card.get("color_identity") or []),
            power=stats("power"), toughness=stats("toughness"),
        )


def _number(text: str) -> float:
    """Power and toughness such as "3", "*" or "1+*": a star counts as 0, as Scryfall does."""
    match = re.match(r"[-+]?\d+(?:\.\d+)?", text.replace("*", "0"))
    return float(match[0]) if match else 0.0


def _value(query: str, start: int) -> tuple[_Value, int]:
    if start >= len(query) or query[start].isspace():
        raise QueryError("A search term is missing its value")
    mark = query[start]
    if mark in "\"/":
        end = start + 1
        while end < len(query) and query[end] != mark:
            end += 2 if query[end] == "\\" else 1
        if end >= len(query):
            raise QueryError("Unbalanced quotes" if mark == '"' else "Unbalanced regular expression")
        body = query[start + 1:end]
        if mark == '"':
            body = body.replace('\\"', '"')
        return _Value("text" if mark == '"' else "regex", body), end + 1
    end = start
    while end < len(query) and not query[end].isspace() and query[end] != ")":
        end += 1
    return _Value("word", query[start:end]), end


def _tokens(query: str) -> list[Any]:
    tokens: list[Any] = []
    index = 0
    while index < len(query):
        char = query[index]
        if char.isspace():
            index += 1
        elif char in "()":
            tokens.append(char)
            index += 1
        elif char == "-" and index + 1 < len(query) and not query[index + 1].isspace():
            tokens.append("-")
            index += 1
        elif match := TERM.match(query, index):
            value, index = _value(query, match.end())
            tokens.append(("term", match[1].lower(), match[2], value))
        else:
            exact = char == "!"
            value, index = _value(query, index + exact)
            if not exact and value.kind == "word" and value.text.lower() in {"or", "and"}:
                tokens.append(value.text.lower())
            else:
                tokens.append(("term", "name", "!" if exact else ":", value))
    return tokens


def _colors(value: str) -> frozenset[str] | int | str:
    word = value.lower()
    if word.isdigit():
        return int(word)
    if word in {"m", "multicolor"}:
        return "multicolor"
    if word in {"c", "colorless"}:
        return frozenset()
    letters = COLOR_NAMES.get(word, word.upper())
    if not letters or set(letters) - set("WUBRG"):
        raise QueryError(f"Unknown color {value!r}")
    return frozenset(letters)


def _symbols(cost: str) -> Counter[str]:
    return Counter(symbol.upper() for symbol in re.findall(r"\{([^}]+)\}", cost))


class BulkSearch:
    """`search_cards` like ScryfallClient's, answered from the local snapshot. With a client it
    rebuilds an older card index from the saved bulk file, and downloads the Oracle tags the
    first time a query needs them; both come from Scryfall's rate-limit-free file host."""

    def __init__(self, store: OracleStore, client: Any = None):
        self.store, self.client = store, client
        self._rows: list[_Row] | None = None
        self._tags: dict[str, frozenset[str]] = {}

    def rows(self) -> list[_Row]:
        if self._rows is None:
            if self.client is not None and not self.store.current:
                self.client.download_oracle_cards(self.store)
            self._rows = [_Row.of(card) for card in self.store.cards()]
        return self._rows

    def search_cards(
        self, query: str, *, max_results: int | None = None, order: str | None = None
    ) -> list[Card]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be nonempty")
        if max_results is not None and (type(max_results) is not int or max_results < 1):
            raise ValueError("max_results must be a positive integer")
        if order is not None and order not in {"name", "edhrec"}:
            raise ValueError("Unsupported search ordering")
        test = self.compile(query)
        found = [row.card for row in self.rows() if test(row)]
        if order == "edhrec":
            # Scryfall lists cards without an EDHREC rank last.
            found.sort(key=lambda c: (c.get("edhrec_rank") is None, c.get("edhrec_rank") or 0, c["name"]))
        else:
            found.sort(key=lambda c: c["name"])
        return [trim_card(card) for card in found[:max_results]]

    def compile(self, query: str) -> Predicate:
        tokens = _tokens(query)
        position = 0

        def peek() -> Any:
            return tokens[position] if position < len(tokens) else None

        def take() -> Any:
            nonlocal position
            position += 1
            return tokens[position - 1]

        def either() -> Predicate:
            parts = [both()]
            while peek() == "or":
                take()
                parts.append(both())
            return parts[0] if len(parts) == 1 else (lambda row: any(p(row) for p in parts))

        def both() -> Predicate:
            parts = []
            while peek() not in {None, ")", "or"}:
                if peek() == "and":
                    take()
                    continue
                parts.append(single())
            if not parts:
                raise QueryError("An empty group or a dangling \"or\"")
            return parts[0] if len(parts) == 1 else (lambda row: all(p(row) for p in parts))

        def single() -> Predicate:
            token = take()
            if token == "-":
                if peek() is None:
                    raise QueryError("A trailing \"-\"")
                inner = single()
                return lambda row: not inner(row)
            if token == "(":
                group = either()
                if peek() != ")":
                    raise QueryError("Unbalanced parentheses")
                take()
                return group
            if token == ")":
                raise QueryError("Unbalanced parentheses")
            return self._term(*token[1:])

        predicate = either()
        if position != len(tokens):
            raise QueryError("Unbalanced parentheses")
        return predicate

    def _term(self, key: str, op: str, value: _Value) -> Predicate:
        field = KEYS.get(key)
        if field is None:
            raise QueryError(
                f"{key}: is not available offline. Use t:, o:, fo:, otag:, kw:, name, mv, pow, "
                "tou, c:, id:, m:, produces:, is:, not: and f:commander"
            )
        text = value.text.lower()
        if field in {"name", "type", "oracle", "full"}:
            return self._text(field, op, value)
        if op not in {":", "="} and field in {"keyword", "otag", "format", "is", "not", "produces"}:
            raise QueryError(f"{key} only takes \":\"")
        if field == "keyword":
            return lambda row: text in row.keywords
        if field == "otag":
            tagged = self._tagged(value.text)
            return lambda row: row.card.get("oracle_id") in tagged
        if field == "format":
            if text not in {"commander", "edh"}:
                raise QueryError("Only f:commander is available offline")
            return lambda row: row.card.get("legalities", {}).get("commander") == "legal"
        if field in {"is", "not"}:
            test = self._is(text)
            return test if field == "is" else (lambda row: not test(row))
        if field == "produces":
            wanted = set(text.upper())
            return lambda row: wanted <= set(row.card.get("produced_mana") or [])
        if field == "mana":
            wanted = _symbols(value.text) or Counter(value.text.upper())
            return lambda row: not wanted - _symbols(row.card.get("mana_cost", ""))
        if field in {"color", "identity"}:
            return self._color(field, op, value.text)
        return self._stat(field, op, text)

    def _text(self, field: str, op: str, value: _Value) -> Predicate:
        if field == "name" and op == "!":
            exact = value.text.lower()
            return lambda row: exact in row.names
        if op not in {":", "="}:
            raise QueryError("Text searches only take \":\"")
        source = {"name": "name", "type": "types", "oracle": "plain", "full": "full"}[field]
        if value.kind == "regex":
            try:
                pattern = re.compile(value.text.replace("~", "CARDNAME"), re.IGNORECASE)
            except re.error as exc:
                raise QueryError(f"Invalid regular expression: {exc}") from None
            if "CARDNAME" in pattern.pattern:
                # "~" stands for the card's own name, as on Scryfall.
                return lambda row: bool(pattern.search(getattr(row, source).replace(row.name, "cardname")))
            return lambda row: bool(pattern.search(getattr(row, source)))
        needle = value.text.lower()
        if "~" in needle and source in {"plain", "full"}:
            return lambda row: needle.replace("~", row.name) in getattr(row, source)
        return lambda row: needle in getattr(row, source)

    def _tagged(self, tag: str) -> frozenset[str]:
        if not self.store.tags_ready and self.client is not None:
            self.client.download_oracle_tags(self.store)
        if not self.store.tags_ready:
            raise QueryError("otag: needs Scryfall's Oracle tags; run download-oracle once")
        if tag not in self._tags:
            # An unknown tag matches nothing, as it does on Scryfall.
            self._tags[tag] = frozenset(self.store.tagged(tag) or ())
        return self._tags[tag]

    def _is(self, value: str) -> Predicate:
        tests: dict[str, Predicate] = {
            "permanent": lambda row: any(t in row.types for t in PERMANENTS),
            "spell": lambda row: "land" not in row.types.split(" // ")[0],
            "commander": lambda row: can_be_commander(row.card),
            "historic": lambda row: any(t in row.types for t in ("legendary", "artifact", "saga")),
            "vanilla": lambda row: "creature" in row.types and not row.full.strip(),
            "gamechanger": lambda row: bool(row.card.get("game_changer")),
            "dfc": lambda row: row.card.get("layout") in {"transform", "modal_dfc", "meld"},
            "mdfc": lambda row: row.card.get("layout") == "modal_dfc",
            "split": lambda row: row.card.get("layout") == "split",
        }
        tests["gc"] = tests["gamechanger"]
        if value not in tests:
            raise QueryError(f"is:{value} is not available offline; try {', '.join(sorted(tests))}")
        return tests[value]

    def _color(self, field: str, op: str, text: str) -> Predicate:
        attribute = "colors" if field == "color" else "identity"
        wanted = _colors(text)
        if isinstance(wanted, int):
            compare = COMPARE[op]
            return lambda row: compare(len(getattr(row, attribute)), wanted)
        if wanted == "multicolor":
            return lambda row: len(getattr(row, attribute)) >= 2
        if op == ":":
            # c: means "at least these colors"; id: means "fits this identity"; both mean
            # exactly colorless for c.
            op = "=" if not wanted else (">=" if field == "color" else "<=")
        compare = COMPARE[op]
        return lambda row: compare(getattr(row, attribute), wanted)

    def _stat(self, field: str, op: str, text: str) -> Predicate:
        compare = COMPARE[op]
        if field == "mv":
            if text in {"even", "odd"} and op in {":", "="}:
                remainder = 0 if text == "even" else 1
                return lambda row: int(row.card.get("cmc") or 0) % 2 == remainder
            number = self._parse_number(text)
            return lambda row: compare(float(row.card.get("cmc") or 0), number)
        other = {"pow": "power", "power": "power", "tou": "toughness", "toughness": "toughness"}
        if text in other:
            # pow>tou compares each face's own stats.
            return lambda row: any(
                compare(a, b) for a, b in zip(getattr(row, field), getattr(row, other[text]))
            )
        number = self._parse_number(text)
        return lambda row: any(compare(stat, number) for stat in getattr(row, field))

    @staticmethod
    def _parse_number(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            raise QueryError(f"Expected a number, not {text!r}") from None
