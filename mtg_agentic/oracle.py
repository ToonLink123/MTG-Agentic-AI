"""A persistent, exact-name Oracle index for offline validation."""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, TextIO

from .cards import Card, name_key, validation_card

DEFAULT_DATA_DIR = Path(".cache") / "mtg-agentic"


def _array_cards(stream: TextIO) -> Iterator[Card]:
    """Incrementally decode legacy JSON arrays, including single-line exports."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    need_value = True
    after_comma = False
    eof = False
    while True:
        buffer = buffer.lstrip()
        if not started and buffer:
            if buffer[0] != "[":
                raise ValueError("Oracle JSON must be an array")
            buffer = buffer[1:]
            started = True
            continue
        if started and buffer:
            if buffer[0] == "]" and need_value and not after_comma:
                if buffer[1:].strip() or stream.read().strip():
                    raise ValueError("Unexpected data after Oracle array")
                return
            if not need_value:
                if buffer[0] == "]":
                    if buffer[1:].strip() or stream.read().strip():
                        raise ValueError("Unexpected data after Oracle array")
                    return
                if buffer[0] != ",":
                    raise ValueError("Expected a comma in Oracle array")
                buffer = buffer[1:]
                need_value = True
                after_comma = True
                continue
            try:
                card, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Invalid or incomplete Oracle JSON") from None
            else:
                if not isinstance(card, dict):
                    raise ValueError("Oracle records must be card objects")
                yield card
                buffer = buffer[end:]
                need_value = False
                after_comma = False
                continue
        if eof:
            raise ValueError("Incomplete Oracle array")
        chunk = stream.read(64 * 1024)
        buffer += chunk
        eof = not chunk


def read_bulk_cards(path: str | Path) -> Iterator[Card]:
    """Read current JSONL/gzip and legacy JSON-array/gzip Oracle exports."""
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        first = ""
        while not first:
            char = stream.read(1)
            if not char:
                raise ValueError("Oracle bulk file is empty")
            first = char.strip()
        stream.seek(0)
        if first == "[":
            yield from _array_cards(stream)
        else:
            for line in stream:
                if line.strip():
                    card = json.loads(line)
                    if not isinstance(card, dict):
                        raise ValueError("Oracle records must be card objects")
                    yield card


class OracleStore:
    def __init__(self, path: str | Path = DEFAULT_DATA_DIR / "oracle.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS cards (
                card_key TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS names (
                name_key TEXT PRIMARY KEY,
                card_key TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def __enter__(self) -> OracleStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    @property
    def metadata(self) -> dict[str, Any]:
        return {key: json.loads(value) for key, value in self._db.execute("SELECT * FROM metadata")}

    @property
    def ready(self) -> bool:
        return self.metadata.get("card_count", 0) > 0

    def get_card(self, name: str) -> Card | None:
        row = self._db.execute(
            "SELECT cards.data FROM names JOIN cards USING (card_key) WHERE names.name_key = ?",
            (name_key(name),),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def import_cards(self, cards: Iterable[Card], *, metadata: dict[str, Any] | None = None) -> int:
        """Replace a snapshot transactionally; failed imports preserve the old data."""
        count = 0
        with self._db:
            self._db.execute("DELETE FROM names")
            self._db.execute("DELETE FROM cards")
            self._db.execute("DELETE FROM metadata")
            for raw in cards:
                if not isinstance(raw, dict) or raw.get("object", "card") != "card":
                    raise ValueError("Oracle records must be card objects")
                # Tokens/emblems can share real card names. Never let them shadow playable cards.
                if raw.get("layout") in {"token", "double_faced_token", "emblem", "art_series"}:
                    continue
                if (
                    not isinstance(raw.get("id"), str)
                    or not isinstance(raw.get("name"), str)
                    or not raw["name"].strip()
                    or not isinstance(raw.get("color_identity"), list)
                    or any(
                        color not in {"W", "U", "B", "R", "G"} for color in raw["color_identity"]
                    )
                ):
                    raise ValueError("Oracle record is missing a valid ID, name, or color identity")
                card = validation_card(raw)
                key = card["oracle_id"] or card["id"]
                self._db.execute(
                    "INSERT OR REPLACE INTO cards VALUES (?, ?)", (key, json.dumps(card))
                )
                canonical_name = name_key(card["name"])
                aliases = {canonical_name}
                faces = card["card_faces"]
                if faces:
                    aliases.add(name_key(faces[0]["name"]))
                    if card["layout"] in {"split", "adventure", "flip"}:
                        aliases.update(name_key(face["name"]) for face in faces)
                for alias in aliases:
                    existing = self.get_card(alias)
                    if (
                        existing is None
                        or (
                            card["legalities"]["commander"] == "legal"
                            and existing["legalities"]["commander"] != "legal"
                        )
                        or (
                            alias == canonical_name and name_key(existing["name"]) != canonical_name
                        )
                    ):
                        self._db.execute("INSERT OR REPLACE INTO names VALUES (?, ?)", (alias, key))
                count += 1
            if count == 0:
                raise ValueError("Oracle snapshot contains no cards")
            info = dict(metadata or {})
            info["card_count"] = count
            self._db.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in info.items()],
            )
        return count

    def import_file(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> int:
        with closing(read_bulk_cards(path)) as cards:
            return self.import_cards(cards, metadata=metadata)
