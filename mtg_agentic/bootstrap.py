"""First-run setup: everything the pipeline reads locally, and a cheap check of what is missing.

`python -m mtg_agentic setup` downloads each missing piece once: Scryfall's Oracle Cards,
Oracle tags and rulings, Commander Spellbook's combo export, the Comprehensive Rules with their
prebuilt vectors, and the EmbeddingGemma 2 weights that embed rulings and rules questions
(about 2.3 GB in all, most of it the model and the combo export). The TUI runs it on a first
start; `status()` tells it what is missing without any network call or heavy import.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import re
import sqlite3
import sys
from pathlib import Path

from .embeddings import EMBEDDINGGEMMA_MODEL, EMBEDDINGGEMMA_REVISION
from .oracle import OracleStore

STEPS = (
    "Card data", "Oracle tags", "Rulings", "Commander Spellbook", "Rules index", "Embedding model",
)
RULES_INDEX = "rules-embeddinggemma-2.sqlite3"  # The default embedding backend's rules index.
ALREADY = "already here"

log = logging.getLogger(__name__)


def stage(name: str, detail: str) -> None:
    # The TUI checks off its setup steps on these lines, so keep their shape.
    log.info("setup: %d/%d %s: %s", STEPS.index(name) + 1, len(STEPS), name, detail)


def embedding_problem() -> str | None:
    """Why this Python cannot embed rules questions, read from package metadata so it is
    instant; None when transformers is new enough for EmbeddingGemma 2."""
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return f"{sys.executable} has no transformers."
    if tuple(int(part) for part in re.findall(r"\d+", version)[:2]) < (5, 21):
        return f"{sys.executable} has transformers {version}; rules reviews need 5.21 or newer."
    return None


def model_cached() -> bool:
    """Whether the pinned EmbeddingGemma 2 weights are in the Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    found = try_to_load_from_cache(
        EMBEDDINGGEMMA_MODEL, "model.safetensors", revision=EMBEDDINGGEMMA_REVISION
    )
    return isinstance(found, str)


def fetch_model() -> str:
    """Download the pinned EmbeddingGemma 2 files into the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    return snapshot_download(EMBEDDINGGEMMA_MODEL, revision=EMBEDDINGGEMMA_REVISION)


def status(data_dir: Path, rules_index: str = RULES_INDEX) -> dict[str, bool]:
    """Which setup steps are done. The model counts as done when this Python could not use it
    anyway, so a start does not run setup for it; setup itself says why it skips it."""
    done = dict.fromkeys(STEPS, False)
    oracle = data_dir / "oracle.sqlite3"
    if oracle.is_file():
        with OracleStore(oracle) as store:
            done["Card data"] = store.ready and store.current
            done["Oracle tags"] = store.tags_ready
            done["Rulings"] = store.rulings_ready
            done["Commander Spellbook"] = store.combos_ready
    rules = data_dir / rules_index
    if rules.is_file():
        with contextlib.suppress(sqlite3.Error), contextlib.closing(
            sqlite3.connect(rules.resolve().as_uri() + "?mode=ro", uri=True)
        ) as db:
            row = db.execute("SELECT value FROM rule_metadata WHERE key='rule_count'").fetchone()
            done["Rules index"] = bool(row and int(row[0]) > 0)
    done["Embedding model"] = embedding_problem() is not None or model_cached()
    return done
