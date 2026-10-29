"""A persistent, exact-name Oracle index for offline validation."""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .cards import Card, name_key, validation_card

DEFAULT_DATA_DIR = Path(".cache") / "mtg-agentic"
# Version 2 keeps colors, keywords and the EDHREC rank, which local search needs.
INDEX_VERSION = 2


def tag_key(name: str) -> str:
    """Tag names compare without case, spaces or hyphens, so board-wipe finds boardwipe."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _phrase(text: str) -> str:
    """An FTS5 phrase query; inside double quotes every word is plain text, never syntax."""
    return '"' + " ".join(text.replace('"', " ").split()) + '"'


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
    """Read current JSONL/gzip and legacy JSON-array/gzip exports (cards, or tags)."""
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
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                slug TEXT NOT NULL,
                children TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tag_names (
                name TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                PRIMARY KEY (name, tag_id)
            );
            CREATE TABLE IF NOT EXISTS taggings (
                tag_id TEXT NOT NULL,
                oracle_id TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS taggings_by_tag ON taggings (tag_id);
            CREATE TABLE IF NOT EXISTS tag_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rulings (
                oracle_id TEXT NOT NULL,
                source TEXT NOT NULL,
                published_at TEXT,
                comment TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS rulings_by_card ON rulings (oracle_id);
            CREATE TABLE IF NOT EXISTS ruling_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS combos (
                id TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS combo_cards (
                combo_id TEXT NOT NULL,
                card TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS combo_cards_by_card ON combo_cards (card);
            CREATE TABLE IF NOT EXISTS combo_metadata (
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

    @property
    def current(self) -> bool:
        """Whether the snapshot was indexed with every field local search reads."""
        return self.metadata.get("index_version") == INDEX_VERSION

    def cards(self) -> Iterator[Card]:
        """Every indexed card, one record per Oracle ID."""
        for (data,) in self._db.execute("SELECT data FROM cards"):
            yield json.loads(data)

    def _side_metadata(self, table: str) -> dict[str, Any]:
        return {key: json.loads(value) for key, value in self._db.execute(f"SELECT * FROM {table}")}

    @property
    def tag_metadata(self) -> dict[str, Any]:
        return self._side_metadata("tag_metadata")

    @property
    def ruling_metadata(self) -> dict[str, Any]:
        return self._side_metadata("ruling_metadata")

    @property
    def rulings_ready(self) -> bool:
        return self.ruling_metadata.get("ruling_count", 0) > 0

    def import_rulings(
        self, rulings: Iterable[dict[str, Any]], *, metadata: dict[str, Any] | None = None
    ) -> int:
        """Replace every card's rulings (Scryfall's Rulings bulk file) transactionally."""
        count = 0
        with self._db:
            self._db.execute("DELETE FROM rulings")
            self._db.execute("DELETE FROM ruling_metadata")
            for ruling in rulings:
                if not isinstance(ruling, dict) or ruling.get("object", "ruling") != "ruling":
                    continue
                if not isinstance(ruling.get("oracle_id"), str) or not isinstance(
                    ruling.get("comment"), str
                ):
                    raise ValueError("Rulings need an oracle_id and a comment")
                self._db.execute("INSERT INTO rulings VALUES (?, ?, ?, ?)", (
                    ruling["oracle_id"], str(ruling.get("source", "")),
                    ruling.get("published_at"), ruling["comment"],
                ))
                count += 1
            if count == 0:
                raise ValueError("The rulings file contains no rulings")
            info = {**(metadata or {}), "ruling_count": count}
            self._db.executemany(
                "INSERT INTO ruling_metadata VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in info.items()],
            )
        return count

    def import_ruling_file(
        self, path: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> int:
        with closing(read_bulk_cards(path)) as rulings:
            return self.import_rulings(rulings, metadata=metadata)

    @property
    def combo_metadata(self) -> dict[str, Any]:
        return self._side_metadata("combo_metadata")

    @property
    def combos_ready(self) -> bool:
        return self.combo_metadata.get("combo_count", 0) > 0

    def import_combos(
        self, variants: Iterable[dict[str, Any]], *, metadata: dict[str, Any] | None = None
    ) -> int:
        """Replace the Commander Spellbook variants transactionally, indexed by card name."""
        count = 0
        with self._db:
            for table in ("combos", "combo_cards", "combo_metadata"):
                self._db.execute(f"DELETE FROM {table}")
            for variant in variants:
                cards = sorted({name_key(use["card"]["name"]) for use in variant["uses"]})
                self._db.execute(
                    "INSERT OR REPLACE INTO combos VALUES (?, ?, ?)",
                    (variant["id"], len(cards), json.dumps(variant)),
                )
                self._db.executemany(
                    "INSERT INTO combo_cards VALUES (?, ?)", ((variant["id"], c) for c in cards)
                )
                count += 1
            if count == 0:
                raise ValueError("The Spellbook export contains no usable variants")
            info = {**(metadata or {}), "combo_count": count}
            self._db.executemany(
                "INSERT INTO combo_metadata VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in info.items()],
            )
        return count

    def combo_candidates(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Every stored variant whose cards are all among `names`."""
        keys = sorted({name_key(name) for name in names})
        matched: dict[str, int] = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for (combo_id,) in self._db.execute(
                f"SELECT combo_id FROM combo_cards WHERE card IN ({marks})", chunk
            ):
                matched[combo_id] = matched.get(combo_id, 0) + 1
        found = []
        ids = sorted(matched)
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for combo_id, size, data in self._db.execute(
                f"SELECT id, size, data FROM combos WHERE id IN ({marks})", chunk
            ):
                if matched[combo_id] == size:
                    found.append(json.loads(data))
        return found

    def rulings(self, oracle_id: str) -> list[dict[str, Any]]:
        """A card's rulings, oldest first, shaped like ScryfallClient.get_rulings results."""
        rows = self._db.execute(
            "SELECT source, published_at, comment FROM rulings WHERE oracle_id = ? "
            "ORDER BY published_at, rowid",
            (oracle_id,),
        )
        return [{"source": s, "published_at": p, "comment": c} for s, p, c in rows]

    @property
    def tags_ready(self) -> bool:
        return self.tag_metadata.get("tag_count", 0) > 0

    def import_tags(self, tags: Iterable[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> int:
        """Replace the Oracle tags (Scryfall's Tagger data) transactionally; art tags are skipped."""
        count = 0
        with self._db:
            for table in ("tags", "tag_names", "taggings", "tag_metadata"):
                self._db.execute(f"DELETE FROM {table}")
            for tag in tags:
                if not isinstance(tag, dict) or tag.get("type") != "oracle":
                    continue
                if not isinstance(tag.get("id"), str) or not isinstance(tag.get("slug"), str):
                    raise ValueError("Tag records need an id and a slug")
                children = [c for c in tag.get("child_ids") or [] if isinstance(c, str)]
                self._db.execute(
                    "INSERT OR REPLACE INTO tags VALUES (?, ?, ?)",
                    (tag["id"], tag["slug"], json.dumps(children)),
                )
                # A slug outranks a label, which outranks an alias another tag might share.
                names = [tag["slug"], tag.get("label"), *(tag.get("aliases") or [])]
                for rank, name in enumerate(names):
                    if isinstance(name, str) and tag_key(name):
                        self._db.execute(
                            "INSERT OR IGNORE INTO tag_names VALUES (?, ?, ?)",
                            (tag_key(name), tag["id"], min(rank, 2)),
                        )
                self._db.executemany("INSERT INTO taggings VALUES (?, ?)", (
                    (tag["id"], tagging["oracle_id"]) for tagging in tag.get("taggings") or []
                    if isinstance(tagging, dict) and isinstance(tagging.get("oracle_id"), str)
                ))
                count += 1
            if count == 0:
                raise ValueError("The tag file contains no Oracle tags")
            info = {**(metadata or {}), "tag_count": count}
            self._db.executemany(
                "INSERT INTO tag_metadata VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in info.items()],
            )
        return count

    def import_tag_file(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> int:
        with closing(read_bulk_cards(path)) as tags:
            return self.import_tags(tags, metadata=metadata)

    def tagged(self, tag: str) -> set[str] | None:
        """Oracle IDs with `tag` (slug, label or alias, as Scryfall's otag: accepts) or any tag
        below it: the bulk data lists only direct taggings. None for an unknown tag."""
        row = self._db.execute(
            "SELECT tag_id FROM tag_names WHERE name = ? ORDER BY rank, tag_id LIMIT 1",
            (tag_key(tag),),
        ).fetchone()
        if row is None:
            return None
        found: set[str] = set()
        stack = [row[0]]
        while stack:
            tag_id = stack.pop()
            if tag_id in found:
                continue
            found.add(tag_id)
            children = self._db.execute("SELECT children FROM tags WHERE id = ?", (tag_id,)).fetchone()
            stack.extend(json.loads(children[0]) if children else [])
        oracle_ids: set[str] = set()
        ids = sorted(found)
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT oracle_id FROM taggings WHERE tag_id IN ({marks})", chunk
            )
            oracle_ids.update(oracle_id for (oracle_id,) in rows)
        return oracle_ids

    def get_card(self, name: str) -> Card | None:
        row = self._db.execute(
            "SELECT cards.data FROM names JOIN cards USING (card_key) WHERE names.name_key = ?",
            (name_key(name),),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _text_index(self) -> None:
        """Index names, type lines and Oracle text for full-text search, once per snapshot."""
        count = self.metadata.get("card_count", 0)
        if self.metadata.get("text_index") == count:
            return
        rows = [
            (card["name"], card["type_line"], card["oracle_text"])
            for card in (json.loads(data) for (data,) in self._db.execute("SELECT data FROM cards"))
        ]
        with self._db:
            # Porter stemming lets "rolls", "rolled" and "rolling" all match "roll".
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS card_text USING fts5("
                "name, type_line, oracle_text, tokenize='porter unicode61 remove_diacritics 2')"
            )
            self._db.execute("DELETE FROM card_text")
            self._db.executemany("INSERT INTO card_text VALUES (?, ?, ?)", rows)
            self._db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('text_index', ?)", (json.dumps(count),)
            )

    def text_count(self, phrase: str) -> int:
        """How many cards contain `phrase` in their name, type line or Oracle text."""
        self._text_index()
        return self._db.execute(
            "SELECT count(*) FROM card_text WHERE card_text MATCH ?", (_phrase(phrase),)
        ).fetchone()[0]

    def search_text(self, phrase: str, limit: int = 300) -> list[tuple[str, float]]:
        """Card names containing `phrase`, best first, with BM25 relevance: rarer phrases and
        shorter texts score higher, and a name match counts most, then the type line."""
        self._text_index()
        rows = self._db.execute(
            "SELECT name, -bm25(card_text, 4.0, 2.0, 1.0) FROM card_text WHERE card_text MATCH ? "
            "ORDER BY bm25(card_text, 4.0, 2.0, 1.0) LIMIT ?",
            (_phrase(phrase), limit),
        )
        return [(name, score) for name, score in rows]

    def _index_card(self, raw: Card) -> bool:
        if not isinstance(raw, dict) or raw.get("object", "card") != "card":
            raise ValueError("Oracle records must be card objects")
        if raw.get("layout") in {"token", "double_faced_token", "emblem", "art_series"}:
            return False
        if (
            not isinstance(raw.get("id"), str)
            or not isinstance(raw.get("name"), str)
            or not raw["name"].strip()
            or not isinstance(raw.get("color_identity"), list)
            or any(
                not isinstance(color, str) or color not in {"W", "U", "B", "R", "G"}
                for color in raw["color_identity"]
            )
        ):
            raise ValueError("Oracle record is missing a valid ID, name, or color identity")
        card = validation_card(raw)
        key = card["oracle_id"] or card["id"]
        self._db.execute("INSERT OR REPLACE INTO cards VALUES (?, ?)", (key, json.dumps(card)))
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
                or (alias == canonical_name and name_key(existing["name"]) != canonical_name)
            ):
                self._db.execute("INSERT OR REPLACE INTO names VALUES (?, ?)", (alias, key))
        return True

    def add_verified_card(self, card: Card) -> None:
        """Persist an exact API-verified record without replacing the bulk snapshot."""
        if not self.ready:
            raise ValueError("Import an Oracle snapshot before adding verified cards")
        with self._db:
            if not self._index_card(card):
                raise ValueError("Cannot index a token as a verified card")
            metadata = self.metadata
            verified = metadata.get("verified_cards", {})
            verified[card["name"]] = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('verified_cards', ?)",
                (json.dumps(verified),),
            )
            count = self._db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            self._db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('card_count', ?)", (json.dumps(count),)
            )

    def import_cards(self, cards: Iterable[Card], *, metadata: dict[str, Any] | None = None) -> int:
        """Replace a snapshot transactionally; failed imports preserve the old data."""
        count = 0
        with self._db:
            self._db.execute("DELETE FROM names")
            self._db.execute("DELETE FROM cards")
            self._db.execute("DELETE FROM metadata")
            for raw in cards:
                count += int(self._index_card(raw))
            if count == 0:
                raise ValueError("Oracle snapshot contains no cards")
            info = dict(metadata or {})
            info["card_count"] = count
            info["index_version"] = INDEX_VERSION
            self._db.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in info.items()],
            )
        return count

    def import_file(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> int:
        with closing(read_bulk_cards(path)) as cards:
            return self.import_cards(cards, metadata=metadata)
