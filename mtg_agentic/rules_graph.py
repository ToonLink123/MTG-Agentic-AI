"""Comprehensive Rules graph: cross-references, glossary definitions and the rule hierarchy.

Embeddings find rules that read like the question. The graph adds the rules those depend on:
"see rule" references, the rule that defines each glossary term a rule uses, and parent/subrule
links. It is built offline from the same CR text, so it needs no model and never drifts from it.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

GRAPH_VERSION = "cr-graph-v2"
RULE_ID = re.compile(r"\d{3}(?:\.\d+[a-z]?)?")
REFERENCE = re.compile(
    r"\b(?:rules?|sections?|see)\s+(\d{3}(?:\.\d+[a-z]?)?"
    r"(?:(?:,\s*|,?\s*(?:and|or|through)\s+|[–-])\d{3}(?:\.\d+[a-z]?)?)*)",
    re.I,
)
STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in into is it its of on or that the their "
    "them then this to was were which while with".split()
)
# Edge weights: a cited rule matters most, a defined term less, and the hierarchy only links
# a rule to its own family so related keywords in one section do not crowd each other out.
WEIGHTS = {"reference": 1.0, "term": 0.5, "parent": 0.3, "child": 0.3}
# Terms used by this share of rules ("spell", "player") explain nothing on their own.
HUB_FRACTION, HUB_MIN = 0.04, 20
# Player wording that differs from the CR's own term ("alternate cost" means "alternative cost").
ALIASES = {"alternat": "alternativ"}


def stems(text: str) -> list[str]:
    """Lowercase word stems without stopwords, so "proliferating" matches "Proliferate"."""
    out = []
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        if token in STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s", "e"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
        out.append(ALIASES.get(token, token))
    return out


def headline(text: str) -> str | None:
    """The title of a heading-only rule such as "701.34. Proliferate", else None."""
    if "\n" in text.strip():
        return None
    match = re.match(r"^\d{3}\.\d+[a-z]?\.?\s+(\S.*)$", text.strip())
    if not match or match[1].endswith(".") or len(match[1].split()) > 6:
        return None
    return match[1]


def parent(node: str, nodes: Iterable[str]) -> str | None:
    if re.fullmatch(r"\d{3}\.\d+[a-z]", node) and node[:-1] in nodes:
        return node[:-1]
    return node[:3] if "." in node else None


def in_family(node: str, target: str) -> bool:
    """A node belongs to a target section or rule when it is the target or one of its subrules."""
    if node == target:
        return True
    if len(target) == 3:
        return node.startswith(target + ".")
    return bool(re.fullmatch(re.escape(target) + "[a-z]", node))


def references(text: str) -> list[str]:
    return [target for match in REFERENCE.finditer(text) for target in RULE_ID.findall(match[1])]


def _sections(lines: list[str]) -> dict[str, str]:
    sections = {}
    for line in lines:
        match = re.match(r"^(\d{3})\.\s+(\S.*)$", line.strip())
        if match:
            sections[match[1]] = match[2]
    return sections


def _glossary(lines: list[str]) -> list[tuple[str, str]]:
    start, seen_rules = None, False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^100\.1\.?\s", stripped):
            seen_rules = True
        elif stripped == "Glossary" and seen_rules:
            start = i + 1
            break
    entries: list[list[str]] = []
    current: list[str] = []
    for line in lines[start:] if start is not None else []:
        stripped = line.strip()
        if stripped == "Credits":
            break
        if stripped:
            current.append(stripped)
        elif current:
            entries.append(current)
            current = []
    if current:
        entries.append(current)
    return [
        (re.sub(r"\s*\(.*?\)", "", entry[0]).strip(), " ".join(entry[1:]))
        for entry in entries
        if len(entry) > 1
    ]


def _contains(tokens: list[str], index: dict[str, list[tuple[str, ...]]]) -> Counter:
    found: Counter = Counter()
    for i, token in enumerate(tokens):
        for term in index.get(token, ()):
            if tuple(tokens[i : i + len(term)]) == term:
                found[term] += 1
    return found


def _index(terms: Iterable[tuple[str, ...]]) -> dict[str, list[tuple[str, ...]]]:
    index: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for term in sorted(set(terms), key=len, reverse=True):
        index[term[0]].append(term)
    return index


@dataclass(frozen=True)
class GraphRows:
    edges: list[tuple[str, str, str, float]]
    terms: list[tuple[str, str, int, str]]
    titles: dict[str, str]


def build_graph(text: str, chunks: dict[str, str]) -> GraphRows:
    """Rows for the rule_edges/rule_terms tables plus a context title for every chunk."""
    lines = text.splitlines()
    sections = _sections(lines)
    nodes = set(chunks) | {rule[:3] for rule in chunks}
    defines: dict[tuple[str, ...], dict[str, str]] = defaultdict(dict)
    for term, definition in _glossary(lines):
        key = tuple(stems(term))
        for target in references(definition):
            if key and target in nodes:
                defines[key].setdefault(target, "glossary")
    for section, title in sections.items():
        if section in nodes and stems(title):
            defines[tuple(stems(title))].setdefault(section, "section")
    for rule, body in chunks.items():
        title = headline(body)
        if title and stems(title):
            defines[tuple(stems(title))].setdefault(rule, "headline")

    index = _index(defines)
    mentions = {rule: _contains(stems(body), index) for rule, body in chunks.items()}
    df = Counter(term for found in mentions.values() for term in found)
    hub = max(HUB_MIN, HUB_FRACTION * len(chunks))

    edges: dict[tuple[str, str, str], float] = {}
    for rule, body in chunks.items():
        up = parent(rule, chunks)
        if up in nodes:
            edges[(rule, up, "parent")] = WEIGHTS["parent"]
            edges[(up, rule, "child")] = WEIGHTS["child"]
        for target in references(body):
            if target != rule and target in nodes:
                edges[(rule, target, "reference")] = WEIGHTS["reference"]
        for term in mentions[rule]:
            if df[term] > hub:
                continue
            for target in defines[term]:
                if not in_family(rule, target) and not in_family(target, rule):
                    edges.setdefault((rule, target, "term"), WEIGHTS["term"])

    titles = {}
    for rule in chunks:
        up = parent(rule, chunks)
        parts = [sections.get(rule[:3])]
        if up and up in chunks:
            parts.append(headline(chunks[up]))
        parts.append(headline(chunks[rule]))
        titles[rule] = " > ".join(p for p in parts if p) or "none"
    return GraphRows(
        edges=[(s, t, kind, w) for (s, t, kind), w in sorted(edges.items())],
        terms=[
            (" ".join(term), target, df[term], origin)
            for term, targets in sorted(defines.items())
            for target, origin in sorted(targets.items())
        ],
        titles=titles,
    )


class RulesGraph:
    """In-memory adjacency for personalized PageRank over CR rules and sections."""

    def __init__(
        self,
        edges: Iterable[tuple[str, str, float]],
        terms: Iterable[tuple[str, str, int]],
        documents: int,
    ):
        self.out: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.out_weight: Counter = Counter()
        for source, target, weight in edges:
            self.out[source].append((target, weight))
            self.out_weight[source] += weight
        self.defines: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self.df: dict[tuple[str, ...], int] = {}
        for key, target, df in terms:
            term = tuple(key.split())
            self.defines[term].append(target)
            self.df[term] = df
        self.documents = documents
        self._index = _index(self.defines)

    def idf(self, term: tuple[str, ...]) -> float:
        return max(0.1, math.log((self.documents + 1) / (self.df.get(term, 0) + 1)))

    def is_hub(self, term: tuple[str, ...]) -> bool:
        return self.df.get(term, 0) > max(HUB_MIN, HUB_FRACTION * self.documents)

    def terms_in(self, text: str) -> Counter:
        return _contains(stems(text), self._index)

    def term_targets(self, text: str) -> dict[str, float]:
        """Defining rules of every term in text, weighted by how specific the term is."""
        weights: dict[str, float] = defaultdict(float)
        for term in self.terms_in(text):
            targets = self.defines[term]
            for target in targets:
                weights[target] += self.idf(term) / len(targets)
        return weights

    def pagerank(
        self,
        seeds: dict[str, float],
        *,
        restart: float = 0.3,
        iterations: int = 40,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Personalized PageRank: random walks that restart at the seeds."""
        total = sum(seeds.values())
        if total <= 0:
            return {}
        seeds = {node: weight / total for node, weight in seeds.items() if weight > 0}
        rank = dict(seeds)
        for _ in range(iterations):
            nxt: dict[str, float] = defaultdict(float)
            dangling = 0.0
            for node, mass in rank.items():
                links = self.out.get(node)
                if not links:
                    dangling += mass
                    continue
                share = (1 - restart) * mass / self.out_weight[node]
                for target, weight in links:
                    nxt[target] += share * weight
            for node, weight in seeds.items():
                nxt[node] += (restart + (1 - restart) * dangling) * weight
            delta = sum(abs(nxt.get(n, 0.0) - rank.get(n, 0.0)) for n in set(nxt) | set(rank))
            rank = nxt
            if delta < tolerance:
                break
        return dict(rank)
