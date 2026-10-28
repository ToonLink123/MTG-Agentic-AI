"""Prebuilt Comprehensive Rules vectors, shared through a public Hugging Face dataset.

Embedding all 3,000+ rules takes minutes on a CPU. A published index holds only rule numbers and
float32 vectors, never rule text; the text still comes from Wizards. Its manifest names the exact
rules text, embedding model and embedded inputs, so vectors are used only when they were made
from exactly what this machine would embed. Anything else falls back to embedding locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from array import array
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .scryfall import USER_AGENT

FORMAT = "mtg-agentic-cr-vectors-v1"
DEFAULT_REPO = "jungter/mtg-agentic-cr-embeddings"
MAX_MANIFEST_BYTES = 1 << 20

log = logging.getLogger(__name__)


def inputs_sha256(docs: list[dict[str, str]]) -> str:
    """Hash of exactly what gets embedded: each rule's number, title and text, in order."""
    rows = [[d["id"], d.get("title") or "none", d["text"]] for d in docs]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def index_dir(identity: str, rules_sha256: str) -> str:
    """Where one index lives in the dataset: its rules text, then its embedding model."""
    model = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"indexes/{rules_sha256[:16]}-{model}"


def write_index(
    out_dir: Path,
    *,
    identity: str,
    rules_sha256: str,
    inputs: str,
    effective_date: str,
    ids: list[str],
    vectors: list[list[float]],
) -> dict[str, Any]:
    """Write `manifest.json` and little-endian float32 `vectors.f32` for publishing."""
    dimension = len(vectors[0])
    if len(vectors) != len(ids) or any(len(v) != dimension for v in vectors):
        raise ValueError("Every rule needs one vector of the same dimension")
    packed = array("f", (x for vector in vectors for x in vector))
    if sys.byteorder == "big":
        packed.byteswap()
    data = packed.tobytes()
    manifest = {
        "format": FORMAT,
        "embedding": identity,
        "dimension": dimension,
        "dtype": "float32-le",
        "count": len(ids),
        "effective_date": effective_date,
        "rules_sha256": rules_sha256,
        "inputs_sha256": inputs,
        "vectors_sha256": hashlib.sha256(data).hexdigest(),
        "ids": ids,
    }
    target = out_dir / index_dir(identity, rules_sha256)
    target.mkdir(parents=True, exist_ok=True)
    (target / "vectors.f32").write_bytes(data)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {"path": str(target), **{k: v for k, v in manifest.items() if k != "ids"}}


class PrebuiltIndex:
    """Reads published indexes from a public Hugging Face dataset; no token is needed."""

    def __init__(
        self, repo: str = DEFAULT_REPO, *, timeout: float = 60, opener: Callable[..., Any] = urlopen
    ):
        self.repo, self.timeout, self.opener = repo, timeout, opener

    def _get(self, path: str, limit: int) -> bytes | None:
        url = f"https://huggingface.co/datasets/{self.repo}/resolve/main/{path}"
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with self.opener(request, timeout=self.timeout) as response:
                data = response.read(limit + 1)
        except HTTPError as exc:
            if exc.code != 404:
                log.warning("rules: prebuilt index request failed (HTTP %s)", exc.code)
            return None
        except (URLError, OSError, ValueError) as exc:
            log.warning("rules: prebuilt index unreachable (%s)", exc)
            return None
        return data if len(data) <= limit else None

    def vectors(
        self, identity: str, rules_sha256: str, inputs: str, ids: list[str]
    ) -> list[list[float]] | None:
        """Vectors for exactly these inputs, or None to embed locally instead."""
        base = index_dir(identity, rules_sha256)
        raw = self._get(f"{base}/manifest.json", MAX_MANIFEST_BYTES)
        if raw is None:
            log.info("rules: no prebuilt index in %s for these rules and this model", self.repo)
            return None
        expected = {
            "format": FORMAT, "embedding": identity, "dtype": "float32-le",
            "rules_sha256": rules_sha256, "inputs_sha256": inputs, "ids": ids,
        }
        try:
            manifest = json.loads(raw)
            dimension = manifest["dimension"]
            mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
        except (ValueError, KeyError, TypeError, AttributeError):
            mismatched, dimension = ["manifest"], 0
        if mismatched or type(dimension) is not int or dimension < 1:
            reasons = ", ".join(mismatched or ["dimension"])
            log.warning("rules: prebuilt index does not match (%s)", reasons)
            return None
        size = len(ids) * dimension * 4
        data = self._get(f"{base}/vectors.f32", size)
        digest = hashlib.sha256(data).hexdigest() if data is not None else None
        if data is None or len(data) != size or digest != manifest.get("vectors_sha256"):
            log.warning("rules: prebuilt vectors are missing or fail their checksum")
            return None
        values = array("f")
        values.frombytes(data)
        if sys.byteorder == "big":
            values.byteswap()
        log.info("rules: using %d prebuilt vectors from %s", len(ids), self.repo)
        return [values[i * dimension : (i + 1) * dimension].tolist() for i in range(len(ids))]
