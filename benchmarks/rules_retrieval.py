"""Rules-retrieval benchmark: 18 interaction questions with hand-picked gold CR rules.

Compares the previous retrieval (hashed-token vectors plus hand-written definition patterns)
with the current fusion of embeddings, IDF-weighted stem overlap and the Comprehensive Rules
graph. Needs `download-rules` for the hash and EmbeddingGemma 2 backends. Both indexes are
copied to a temporary directory first, so the originals are never modified. See BENCHMARK.md.

    python -m benchmarks.rules_retrieval --data-dir .cache/mtg-agentic
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import tempfile
import time
from contextlib import ExitStack, closing
from pathlib import Path

from mtg_agentic.embeddings import EmbeddingGemmaEmbedder
from mtg_agentic.oracle import DEFAULT_DATA_DIR
from mtg_agentic.rules import HashEmbedder, RulesStore, _tokens
from mtg_agentic.rules_graph import RulesGraph
from mtg_agentic.sources import SourceCache

# Gold rules were chosen from rules knowledge before looking at any method's output.
CASES = [
    ("Does casting a Myojin from hand by paying Jodah's alternative cost still make it enter "
     "with a divinity counter?", {"118.9", "118.9a", "118.9c", "614.1c", "614.12"}),
    ("Can proliferate add another divinity counter to a Myojin after its last divinity counter "
     "was removed?", {"701.34a"}),
    ("If a creature becomes a copy of another creature, does it also copy that creature's "
     "+1/+1 counters?", {"707.2"}),
    ("With deathtouch and trample, is assigning 1 damage to each blocker enough before the rest "
     "tramples over?", {"702.2c", "702.19b", "702.19c"}),
    ("When my commander would be put into the graveyard, can I put it into the command zone "
     "instead?", {"903.9a", "903.9"}),
    ("How much extra does it cost to cast my commander from the command zone for the third "
     "time?", {"903.8"}),
    ("Can I target my own creature that has hexproof with my own spells?", {"702.11b", "702.11a"}),
    ("When does a triggered ability that triggered while I was casting a spell go on the stack?",
     {"603.3", "603.3b", "603.2"}),
    ("Does a creature with 0 toughness die even if it has indestructible?", {"704.5f"}),
    ("In which layer do effects that set power and toughness apply compared to +1/+1 counters?",
     {"613.4b", "613.4c", "613.4"}),
    ("Is the mana value of a spell with X in its cost equal to the chosen X while it is on the "
     "stack?", {"202.3e"}),
    ("If I create a token copy of my legendary creature, do I have to put one into the "
     "graveyard?", {"704.5j"}),
    ("Damage dealt to a planeswalker: what happens to its loyalty counters?", {"120.3c"}),
    ("Two replacement effects would apply to the same event. Who chooses the order they apply "
     "in?", {"616.1"}),
    ("Can a creature with haste attack the turn it comes under my control?", {"302.6", "702.10b"}),
    ("While a split second spell is on the stack, can players still activate mana abilities?",
     {"702.61a", "702.61b"}),
    ("Is a divinity counter a keyword counter that grants an ability?", {"122.1b"}),
    ("Can my two partner commanders have different colors, and what is my deck's color "
     "identity?", {"702.124c", "903.4"}),
]
CUTOFFS = (5, 10, 18)


def previous_retrieve(path: Path, question: str, limit: int) -> list[str]:
    """The pre-graph algorithm: cosine + token overlap, plus hand-written definition patterns."""
    with closing(sqlite3.connect(path)) as db:
        rows = db.execute("SELECT id,text,vector FROM documents WHERE kind='rule'").fetchall()
    query = HashEmbedder().embed([question])[0]
    norm = math.sqrt(sum(x * x for x in query))
    terms = set(_tokens(question))
    lower = question.lower()
    patterns = []
    if "proliferat" in lower:
        patterns.append(r"\bto proliferate means\b")
    if "counter" in lower:
        patterns += [r"\ba counter is a marker\b", r"\ba keyword counter\b"]
    if re.search(r"alternat(?:e|ive) cost", lower):
        patterns += [r"\ban alternative cost is\b", r"\ban alternative cost doesn['’]t change\b"]
    if "cast" in lower:
        patterns.append(r"\bto cast a spell is to\b")
    if "enters" in lower:
        patterns.append(r"\bsome replacement effects modify how a permanent enters\b")
    scored, pinned = [], []
    for id_, text, raw in rows:
        vector = json.loads(raw)
        cosine = sum(a * b for a, b in zip(query, vector)) / (
            norm * math.sqrt(sum(x * x for x in vector)) or 1
        )
        overlap = len(terms & set(_tokens(text))) / max(1, len(terms))
        scored.append((cosine + overlap, id_))
        if any(re.search(p, text, re.I) for p in patterns):
            pinned.append(id_)
    picked = sorted(pinned)[:limit]
    picked += [id_ for _, id_ in sorted(scored, reverse=True) if id_ not in picked]
    return [id_.removeprefix("CR ") for id_ in picked[:limit]]


def current_retrieve(store: RulesStore, question: str, limit: int) -> list[str]:
    return [d["id"].removeprefix("CR ") for d in store.retrieve([], question, limit=limit)]


def score(results: list[tuple[list[str], set[str]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in CUTOFFS:
        out[f"hit@{k}"] = sum(bool(gold & set(ids[:k])) for ids, gold in results) / len(results)
        out[f"recall@{k}"] = sum(len(gold & set(ids[:k])) / len(gold) for ids, gold in results) / len(
            results
        )
    ranks = [first_gold(ids, gold) for ids, gold in results]
    out["mrr"] = sum(1 / r for r in ranks if r) / len(ranks)
    return out


def first_gold(ids: list[str], gold: set[str]) -> int | None:
    return next((i + 1 for i, id_ in enumerate(ids) if id_ in gold), None)


def copy_index(source: Path, target: Path) -> Path:
    if not source.is_file():
        raise SystemExit(f"Missing {source}; run download-rules for that embedding backend")
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=18, help="Evidence items per question")
    args = parser.parse_args(argv)
    with ExitStack() as stack:
        temp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        cache = stack.enter_context(SourceCache(args.data_dir / "sources.sqlite3"))
        hash_db = copy_index(args.data_dir / "rules-hash.sqlite3", temp / "hash.sqlite3")
        gemma_db = copy_index(
            args.data_dir / "rules-embeddinggemma-2.sqlite3", temp / "gemma.sqlite3"
        )
        gemma = EmbeddingGemmaEmbedder()
        hashed = stack.enter_context(RulesStore(hash_db, HashEmbedder()))
        fused = stack.enter_context(RulesStore(gemma_db, gemma))
        for store in (hashed, fused):
            store.download_rules(cache)  # Builds a missing graph; unchanged text is not re-embedded.
        plain = stack.enter_context(RulesStore(gemma_db, gemma))
        # No edges and no glossary terms: no term anchors and nothing for the walk to follow.
        plain._graph = lambda: RulesGraph([], [], int(plain._meta("rule_count") or 0))
        configs = {
            "Previous: hashed tokens + definition patterns": lambda q: previous_retrieve(
                hash_db, q, args.limit
            ),
            "Hashed tokens + graph fusion": lambda q: current_retrieve(hashed, q, args.limit),
            "EmbeddingGemma 2 + stem overlap, no graph": lambda q: current_retrieve(
                plain, q, args.limit
            ),
            "EmbeddingGemma 2 + graph fusion (default)": lambda q: current_retrieve(
                fused, q, args.limit
            ),
        }
        rows, per_case = [], {}
        for name, run in configs.items():
            start = time.monotonic()
            results = [(run(question), gold) for question, gold in CASES]
            elapsed = (time.monotonic() - start) / len(CASES)
            rows.append((name, score(results), elapsed))
            per_case[name] = results
    columns = [f"{m}@{k}" for k in CUTOFFS for m in ("hit", "recall")]
    print("| Retrieval | " + " | ".join(columns) + " | MRR | s/query |")
    print("|---" * (len(columns) + 3) + "|")
    for name, metrics, elapsed in rows:
        values = " | ".join(f"{metrics[c]:.2f}" for c in columns)
        print(f"| {name} | {values} | {metrics['mrr']:.2f} | {elapsed:.2f} |")
    print("\nFirst gold rank / gold recall@limit per question (- = no gold rule retrieved):")
    for i, (question, gold) in enumerate(CASES):
        cells = []
        for results in per_case.values():
            ids = results[i][0]
            rank = first_gold(ids, gold)
            cells.append(f"{rank or '-'} / {len(gold & set(ids)) / len(gold):.2f}")
        print(f"| {i + 1} | {question[:60]} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
