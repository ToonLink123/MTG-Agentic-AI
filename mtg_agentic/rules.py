"""Local rule-number chunks, deck rulings, and citation-checked interaction RAG."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from . import activity, prompts
from .cards import name_key
from .llm import HTTPJSONModel, JSONModel, request_json
from .oracle import OracleStore
from .rules_graph import (
    GRAPH_VERSION,
    GraphRows,
    RulesGraph,
    build_graph,
    headline,
    in_family,
    stems,
)
from .rules_index import PrebuiltIndex, inputs_sha256, write_index
from .schema import NAME_SCHEMA, object_schema
from .scryfall import ScryfallClient, card_rulings
from .sources import SourceCache, SourceError, utc_now

RULES_PAGE = "https://magic.wizards.com/en/rules"


class Embedder(Protocol):
    identity: str

    def embed(
        self, texts: list[str], *, titles: list[str] | None = None, query: bool = False
    ) -> list[list[float]]:
        """Unit-comparable vectors; documents may carry titles, queries may use their own prompt."""
        ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class HashEmbedder:
    """Explicit offline lexical baseline, not a semantic language model."""

    identity = "hash-token-sha256-v1:384"

    def embed(
        self, texts: list[str], *, titles: list[str] | None = None, query: bool = False
    ) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            for token in _tokens(text):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % 384] += 1 if digest[4] & 1 else -1
            vectors.append(vector)
        return vectors


class OllamaEmbedder(HTTPJSONModel):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
    ):
        super().__init__(model, base_url=base_url, timeout=timeout)
        self.identity = f"ollama:{base_url.rstrip('/')}:{model}"

    def embed(
        self, texts: list[str], *, titles: list[str] | None = None, query: bool = False
    ) -> list[list[float]]:
        body = self._post("/api/embed", {"model": self.model, "input": texts, "truncate": False})
        return body.get("embeddings", [])


def _vectors(values: Any, expected: int, dimension: int | None = None) -> list[list[float]]:
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError("Embedding response has the wrong number of vectors")
    for vector in values:
        if (
            not isinstance(vector, list)
            or not vector
            or (dimension is not None and len(vector) != dimension)
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in vector)
            or not any(vector)
        ):
            raise ValueError("Invalid or incompatible embedding vector")
        dimension = len(vector)
    return values


def chunk_rules(text: str) -> tuple[str, dict[str, str]]:
    effective = re.search(r"effective as of ([^\r\n]+)", text, re.I)
    if not effective:
        raise ValueError("Rules text has no effective date")
    chunks: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    started = False

    def flush() -> None:
        if current:
            chunks[current] = "\n".join(lines).strip()

    for line in text.splitlines():
        if line.strip() == "Glossary" and started:
            break
        match = re.match(r"^(\d{3}\.\d+[a-z]?)(?:\.\s+|\s+)(\S.*)$", line)
        if match:
            flush()
            current = match[1]
            lines = [line]
            started = True
        elif re.match(r"^\d{3}\.\s", line):
            flush()
            current, lines = None, []
        elif current and line.strip():
            lines.append(line)
    flush()
    if "100.1" not in chunks or len(chunks) < 2:
        raise ValueError("Rules text has no numbered rules body")
    return effective[1].strip(), chunks


CITATION_SCHEMA = object_schema(
    {
        "id": NAME_SCHEMA,
        "quote": {
            "type": "string",
            "minLength": 20,
            "maxLength": 3000,
        },
    }
)
INTERACTION_SCHEMA = object_schema(
    {
        "status": {"type": "string", "enum": ["supported", "contradicted", "unsupported"]},
        "claims": {
            "type": "array",
            "maxItems": 12,
            "items": object_schema(
                {
                    "claim": {"type": "string", "minLength": 1, "maxLength": 1500},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "items": CITATION_SCHEMA,
                    },
                }
            ),
        },
    }
)


RRF_K = 60
SEED_RULES = 8


@dataclass(frozen=True, eq=False)
class _Doc:
    id: str
    kind: str
    text: str
    source: str
    published: str
    card: str
    vector: list[float]
    stems: frozenset[str]
    heading: bool

    @property
    def node(self) -> str:
        return self.id.removeprefix("CR ")

    def evidence(self) -> dict[str, str]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "source": self.source,
            "published": self.published,
        }


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    order = sorted((k for k, v in scores.items() if v > 0), key=lambda k: (-scores[k], k))
    return {key: rank for rank, key in enumerate(order)}


def _rrf(*rankings: dict[str, int]) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for key, rank in ranking.items():
            fused[key] += 1 / (RRF_K + rank + 1)
    return fused


def _subquestions(question: str, *, limit: int = 6) -> list[str]:
    """The sentences of a multi-part question; none for a single sentence."""
    parts = [p for p in re.split(r"(?<=[.?!])\s+", question.strip()) if len(stems(p)) >= 3]
    return parts[:limit] if len(parts) > 1 else []


def _interleave(rankings: list[list[_Doc]]) -> Iterator[_Doc]:
    """Round-robin over rankings, skipping repeats, so each sub-question takes turns."""
    seen: set[str] = set()
    for row in zip(*rankings):
        for doc in row:
            if doc.id not in seen:
                seen.add(doc.id)
                yield doc


def _seeds(groups: list[tuple[dict[str, float], float]]) -> dict[str, float]:
    """Mix seed groups by share, renormalizing over the groups that found anything."""
    groups = [(weights, share) for weights, share in groups if sum(weights.values()) > 0]
    mass = sum(share for _, share in groups)
    seeds: dict[str, float] = defaultdict(float)
    for weights, share in groups:
        total = sum(weights.values())
        for node, weight in weights.items():
            seeds[node] += share / mass * weight / total
    return seeds


class RulesStore:
    """SQLite vectors fused with lexical scores and a walk over the Comprehensive Rules graph."""

    def __init__(
        self, path: str | Path, embedder: Embedder, *, prebuilt: PrebuiltIndex | None = None
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.embedder = embedder
        self.prebuilt = prebuilt
        self._docs: list[_Doc] | None = None
        self._graph_cache: RulesGraph | None = None
        self._df: Counter = Counter()
        self._rule_docs = 0
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, kind TEXT, text TEXT, source TEXT,
                published TEXT, card TEXT, vector TEXT
            );
            CREATE TABLE IF NOT EXISTS rule_metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS ruling_sync (card TEXT PRIMARY KEY, fetched TEXT);
            CREATE TABLE IF NOT EXISTS rule_edges (
                source TEXT, target TEXT, kind TEXT, weight REAL,
                PRIMARY KEY (source, target, kind)
            );
            CREATE TABLE IF NOT EXISTS rule_terms (
                term TEXT, node TEXT, df INTEGER, origin TEXT, PRIMARY KEY (term, node)
            );
        """)
        identity = self._meta("embedding")
        if identity and identity != embedder.identity:
            self.db.close()
            raise ValueError("Embedding model changed; use a new rules database")

    def __enter__(self) -> RulesStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM rule_metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT key,value FROM rule_metadata"))

    def _embed_documents(
        self, docs: list[dict[str, str]], vectors: list[list[float]] | None = None
    ) -> list[tuple[Any, ...]]:
        """Rows for `docs`, embedding them unless verified prebuilt `vectors` are given."""
        dimension = int(self._meta("dimension") or 0) or None
        if vectors is not None:
            vectors = _vectors(vectors, len(docs), dimension)
        else:
            vectors = []
            for start in range(0, len(docs), 32):
                batch = docs[start : start + 32]
                vectors += _vectors(
                    self.embedder.embed(
                        [d["text"] for d in batch], titles=[d.get("title") or "none" for d in batch]
                    ),
                    len(batch),
                    dimension,
                )
                dimension = len(vectors[0])
        return [
            (d["id"], d["kind"], d["text"], d["source"], d["published"], d["card"], json.dumps(v))
            for d, v in zip(docs, vectors)
        ]

    def _write(self, rows: list[tuple[Any, ...]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)", rows)
        self._docs = None
        if rows:
            self.db.executemany(
                "INSERT OR REPLACE INTO rule_metadata VALUES (?,?)",
                [
                    ("embedding", self.embedder.identity),
                    ("dimension", str(len(json.loads(rows[0][-1])))),
                ],
            )

    def _write_graph(self, graph: GraphRows) -> None:
        self.db.execute("DELETE FROM rule_edges")
        self.db.execute("DELETE FROM rule_terms")
        self.db.executemany("INSERT INTO rule_edges VALUES (?,?,?,?)", graph.edges)
        self.db.executemany("INSERT INTO rule_terms VALUES (?,?,?,?)", graph.terms)
        self.db.executemany(
            "INSERT OR REPLACE INTO rule_metadata VALUES (?,?)",
            [
                ("graph_version", GRAPH_VERSION),
                ("graph_edges", str(len(graph.edges))),
                ("graph_terms", str(len({row[0] for row in graph.terms}))),
            ],
        )
        self._graph_cache = None

    def _graph(self) -> RulesGraph:
        if self._graph_cache is None:
            self._graph_cache = RulesGraph(
                self.db.execute("SELECT source,target,weight FROM rule_edges").fetchall(),
                self.db.execute("SELECT term,node,df FROM rule_terms").fetchall(),
                int(self._meta("rule_count") or 0),
            )
        return self._graph_cache

    def _corpus(self) -> list[_Doc]:
        """Parsed, unit-length vectors and stems, kept until the next write."""
        if self._docs is None:
            docs = []
            for id_, kind, text, source, published, card, raw in self.db.execute(
                "SELECT id,kind,text,source,published,card,vector FROM documents"
            ):
                vector = json.loads(raw)
                norm = math.sqrt(sum(x * x for x in vector)) or 1.0
                docs.append(_Doc(
                    id_, kind, text, source, published, card, [x / norm for x in vector],
                    frozenset(stems(text)), kind == "rule" and headline(text) is not None,
                ))
            rules = [d for d in docs if d.kind == "rule"]
            self._df = Counter(t for d in rules for t in d.stems)
            self._rule_docs = len(rules)
            self._docs = docs
        return self._docs

    def _idf(self, stem: str) -> float:
        return math.log((self._rule_docs + 1) / (self._df.get(stem, 0) + 1)) + 1

    @staticmethod
    def _parse_rules(text: str, source: str) -> tuple[str, GraphRows, list[dict[str, str]]]:
        date, chunks = chunk_rules(text)
        graph = build_graph(text, chunks)
        docs = [
            {
                "id": f"CR {n}",
                "kind": "rule",
                "text": body,
                "source": source,
                "published": date,
                "card": "",
                "title": graph.titles[n],
            }
            for n, body in chunks.items()
        ]
        return date, graph, docs

    def import_rules(self, text: str, *, source: str, fetched_at: str) -> dict[str, str]:
        date, graph, docs = self._parse_rules(text, source)
        rules_sha256 = hashlib.sha256(text.encode()).hexdigest()
        inputs = inputs_sha256(docs)
        vectors = None
        if self.prebuilt is not None:
            ids = [d["id"] for d in docs]
            vectors = self.prebuilt.vectors(self.embedder.identity, rules_sha256, inputs, ids)
        rows = self._embed_documents(docs, vectors)
        with self.db:
            self.db.execute("DELETE FROM documents WHERE kind='rule'")
            self._write(rows)
            self._write_graph(graph)
            self.db.executemany(
                "INSERT OR REPLACE INTO rule_metadata VALUES (?,?)",
                [
                    ("effective_date", date),
                    ("source", source),
                    ("fetched_at", fetched_at),
                    ("rule_count", str(len(docs))),
                    ("rules_sha256", rules_sha256),
                    ("inputs_sha256", inputs),
                    ("vectors", f"hf:{self.prebuilt.repo}" if vectors else "local"),
                ],
            )
        return self.metadata

    def _rules_text(self, cache: SourceCache, *, refresh: bool = False) -> tuple[str, str, str]:
        """The current Comprehensive Rules text, its official URL and when it was fetched."""
        landing, _ = cache.fetch(RULES_PAGE, refresh=refresh)
        links = re.findall(
            r'https://media\.wizards\.com/[^"\'<>]+\.txt', html.unescape(landing.decode("utf-8"))
        )
        links = [u for u in links if "MagicCompRules" in u]
        if not links:
            raise SourceError("Official rules page did not link Comprehensive Rules text")
        source = quote(links[0], safe=":/%")
        raw, stamp = cache.fetch(source, refresh=refresh)
        return raw.decode("utf-8-sig"), source, stamp

    def export_index(self, cache: SourceCache, out_dir: Path) -> dict[str, Any]:
        """Write this store's rule vectors for publishing, embedding the current rules first."""
        text, source, stamp = self._rules_text(cache)
        _, _, docs = self._parse_rules(text, source)
        inputs = inputs_sha256(docs)
        # Publish only vectors embedded here from exactly these inputs, never re-published ones.
        if self._meta("inputs_sha256") != inputs or self._meta("vectors") != "local":
            self.import_rules(text, source=source, fetched_at=stamp)
        if self._meta("vectors") != "local":
            raise ValueError("Export needs locally embedded vectors; disable the prebuilt index")
        stored = dict(self.db.execute("SELECT id, vector FROM documents WHERE kind='rule'"))
        return write_index(
            out_dir,
            identity=self.embedder.identity,
            rules_sha256=hashlib.sha256(text.encode()).hexdigest(),
            inputs=inputs,
            effective_date=self._meta("effective_date") or "",
            ids=[d["id"] for d in docs],
            vectors=[json.loads(stored[d["id"]]) for d in docs],
        )

    def download_rules(self, cache: SourceCache, *, refresh: bool = False) -> dict[str, str]:
        text, source, stamp = self._rules_text(cache, refresh=refresh)
        if self._meta("rules_sha256") != hashlib.sha256(text.encode()).hexdigest():
            return self.import_rules(text, source=source, fetched_at=stamp)
        with self.db:
            # Older indexes gain the graph from the cached text without re-embedding anything.
            if self._meta("graph_version") != GRAPH_VERSION:
                self._write_graph(build_graph(text, chunk_rules(text)[1]))
            self.db.executemany(
                "INSERT OR REPLACE INTO rule_metadata VALUES (?,?)",
                [("source", source), ("fetched_at", stamp)],
            )
        return self.metadata

    def ingest_deck(
        self, cards: list[str], oracle: OracleStore, client: ScryfallClient
    ) -> dict[str, Any]:
        canonical: dict[str, Any] = {}
        for name in cards:
            card = oracle.get_card(name)
            if card is None:
                raise ValueError(f"No exact Oracle card for rules review: {name}")
            canonical[name_key(card["name"])] = card
        synced = []
        for key, card in canonical.items():
            activity.tool("rulings", card["name"])
            rulings = card_rulings(oracle, client, card)
            source = f"https://api.scryfall.com/cards/{card['id']}/rulings"
            docs = [
                {
                    "id": f"Oracle {card['name']}",
                    "kind": "oracle",
                    "text": card["oracle_text"] or card["type_line"],
                    "source": "https://scryfall.com/card/" + card["id"],
                    "published": oracle.metadata.get("updated_at", "local snapshot")
                    or "local snapshot",
                    "card": key,
                    "title": card["name"],
                }
            ]
            for ruling in rulings:
                if not isinstance(ruling.get("comment"), str) or not ruling["comment"].strip():
                    raise SourceError("Invalid Scryfall ruling")
                # Only publisher rulings, never user-authored claims.
                if ruling["source"] != "wotc":
                    continue
                digest = hashlib.sha256(
                    (str(ruling["published_at"]) + ruling["comment"]).encode()
                ).hexdigest()[:16]
                docs.append(
                    {
                        "id": f"Ruling {card['name']} {digest}",
                        "kind": "ruling",
                        "text": ruling["comment"],
                        "source": source,
                        "published": ruling["published_at"],
                        "card": key,
                        "title": card["name"] + " ruling",
                    }
                )
            old_docs = {
                row[0]: row[1:]
                for row in self.db.execute(
                    "SELECT id,kind,text,source,published,card FROM documents WHERE card=?", (key,)
                )
            }
            changed = old_docs != {
                d["id"]: (d["kind"], d["text"], d["source"], d["published"], d["card"])
                for d in docs
            }
            rows = self._embed_documents(docs) if changed else []
            with self.db:
                if changed:
                    self.db.execute("DELETE FROM documents WHERE card=?", (key,))
                    self._write(rows)
                self.db.execute("INSERT OR REPLACE INTO ruling_sync VALUES (?,?)", (key, utc_now()))
            synced.append(card["name"])
        return {"cards": synced, "count": len(synced), "synced_at": utc_now()}

    def retrieve(self, cards: list[str], question: str, *, limit: int = 18) -> list[dict[str, str]]:
        """Card evidence first, then rule anchors for specific terms, then fused rule ranking.

        Rules are ranked by reciprocal rank fusion of three signals: embedding cosine, IDF-weighted
        stem overlap, and a personalized PageRank walk over the CR graph that starts from the best
        first-stage rules, the rules defining the question's terms, and the cards' keywords. A
        multi-part question is also ranked per sentence, and the rankings take turns.
        """
        if not question.strip() or not 1 <= limit <= 40:
            raise ValueError("Question and retrieval limit (1–40) are required")
        if not self._meta("effective_date"):
            raise ValueError("Download the Comprehensive Rules before checking interactions")
        if self._meta("graph_version") != GRAPH_VERSION:
            raise ValueError("Rebuild the rules graph: run download-rules")
        keys = {name_key(n) for n in cards}
        missing = [
            key
            for key in keys
            if not self.db.execute("SELECT 1 FROM ruling_sync WHERE card=?", (key,)).fetchone()
        ]
        if missing:
            raise ValueError("Ingest rulings for every interaction card before checking")
        # Multi-part questions also rank rules per sentence, so every part gets evidence.
        texts = [question, *_subquestions(question)]
        vectors = _vectors(
            self.embedder.embed([text + " " + " ".join(cards) for text in texts], query=True),
            len(texts),
            int(self._meta("dimension") or 0),
        )
        docs = self._corpus()
        rules = [d for d in docs if d.kind == "rule"]
        card_docs = [d for d in docs if d.kind != "rule" and d.card in keys]
        signals = [self._signals(rules, text, vector) for text, vector in zip(texts, vectors)]
        _, dense_ranks, lexical_ranks = signals[0]
        first = _rrf(dense_ranks, lexical_ranks)
        graph = self._graph()
        top = sorted(first, key=lambda n: (-first[n], n))[:SEED_RULES]
        oracle_text = " ".join(d.text for d in card_docs if d.kind == "oracle")
        walk = graph.pagerank(_seeds([
            ({n: 1 / (i + 1) for i, n in enumerate(top)}, 0.4),
            (graph.term_targets(question), 0.4),
            (graph.term_targets(oracle_text), 0.2),
        ]))
        graph_ranks = _ranks({d.node: walk.get(d.node, 0.0) for d in rules})
        rankings = []
        for scores, ranks, lexical in signals:
            fused = _rrf(ranks, lexical, graph_ranks)
            rankings.append(
                sorted(rules, key=lambda d, f=fused, s=scores: (-f.get(d.node, 0.0), -s[d.node], d.id))
            )
        position = {d.id: i for i, d in enumerate(rankings[0])}

        def order(doc: _Doc) -> int:
            return position[doc.id]

        words = set(stems(question))
        # Reserve half the context for rules so many rulings cannot crowd them out.
        selected = sorted(
            card_docs, key=lambda d: (d.kind == "oracle", self._overlap(words, d), d.id), reverse=True
        )[: limit // 2]
        # Long multi-part questions otherwise bury short operative definitions under rules that
        # repeat their wording, so each specific term the question names keeps its defining rule.
        specific = sorted(
            (t for t in graph.terms_in(question) if not graph.is_hub(t)),
            key=lambda t: (-graph.idf(t), t),
        )
        anchors: list[_Doc] = []
        for term in specific:
            chosen = {d.id for d in anchors}
            family = [
                d
                for d in rules
                if not d.heading
                and d.id not in chosen
                and any(in_family(d.node, target) for target in graph.defines[term])
            ]
            if family and len(anchors) < max(1, limit // 4):
                anchors.append(min(family, key=order))
        selected.extend(anchors[: limit - len(selected)])
        chosen = {d.id for d in selected}
        selected.extend(d for d in _interleave(rankings) if d.id not in chosen)
        return [d.evidence() for d in selected[:limit]]

    def _signals(
        self, rules: list[_Doc], text: str, vector: list[float]
    ) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
        """Cosine scores plus cosine and IDF-weighted stem-overlap rankings for one query."""
        norm = math.sqrt(sum(x * x for x in vector))
        unit = [x / norm for x in vector]
        words = set(stems(text))
        dense = {d.node: sum(a * b for a, b in zip(unit, d.vector)) for d in rules}
        lexical = {d.node: self._overlap(words, d) for d in rules}
        return dense, _ranks(dense), _ranks(lexical)

    def _overlap(self, words: set[str], doc: _Doc) -> float:
        total = sum(self._idf(t) for t in words) or 1.0
        return sum(self._idf(t) for t in words & doc.stems) / total

    def check_interaction(
        self, cards: list[str], question: str, model: JSONModel
    ) -> dict[str, Any]:
        messages, evidence = self.interaction_request(cards, question)
        answer = request_json(model, messages, INTERACTION_SCHEMA)
        return self.judge_interaction(cards, question, evidence, answer)

    def interaction_request(
        self, cards: list[str], question: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Retrieve evidence and build the judge's messages. SQLite and the embedder stay on
        the calling thread, so only the model call itself can run in parallel."""
        activity.tool("rules lookup", " + ".join(n.split(",")[0] for n in cards))
        evidence = self.retrieve(cards, question)
        payload = {"cards": cards, "question": question, "evidence": evidence}
        messages = [
            {"role": "system", "content": prompts.INTERACTION},
            {"role": "user", "content": json.dumps(payload)},
        ]
        return messages, evidence

    def judge_interaction(
        self,
        cards: list[str],
        question: str,
        evidence: list[dict[str, str]],
        answer: dict[str, Any],
    ) -> dict[str, Any]:
        """Accept an answer only if every claim cites retrieved rules or rulings verbatim."""
        by_id = {e["id"]: e for e in evidence}
        problems = []
        if answer["status"] != "unsupported" and not answer["claims"]:
            problems.append("A rules conclusion requires cited claims.")
        for claim in answer["claims"]:
            has_rule = False
            for citation in claim["citations"]:
                doc = by_id.get(citation["id"])
                if doc is None or citation["quote"] not in doc["text"]:
                    problems.append("Unknown citation or non-verbatim evidence quote.")
                elif doc["kind"] in {"rule", "ruling"}:
                    has_rule = True
            if not has_rule:
                problems.append("Each claim requires a CR rule or publisher ruling.")
        if answer["status"] == "unsupported" or problems:
            # Never echo unsupported model claims back as facts or repair instructions.
            answer = {"status": "unsupported", "claims": []}
        return {
            **answer,
            "cards": cards,
            "question": question,
            "evidence": evidence,
            "errors": problems,
            "rules_metadata": self.metadata,
            "disclaimer": "Citations are mechanically checked; logical entailment remains LLM judgment.",
        }


def check_interaction(
    cards: list[str], question: str, *, rules: RulesStore, model: JSONModel
) -> dict[str, Any]:
    return rules.check_interaction(cards, question, model)
