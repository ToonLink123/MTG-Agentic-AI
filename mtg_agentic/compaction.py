"""Deterministic request compaction with no model pass (see compactionLogic.md).

Requests here are a system prompt plus a few large user turns, not a long transcript, so
compaction bounds text instead of summarizing a working state. The newest turn is the
protected tail; older turns shrink first, largest first, by middle elision. Every cut is
remembered and re-applied, so a compacted prefix stays byte-identical across requests and
keeps its prompt cache.
"""

from __future__ import annotations

import hashlib

TRIGGER = 0.95  # Compact once a prompt reaches 95% of its budget.
MIN_YIELD = 0.15  # Older turns must hold this share of the turns for cutting them to help.
MIN_PART = 1000  # Never bound a part below this many characters.
MARK = "[… {} characters elided to fit the context window …]"


class Compactor:
    def __init__(self) -> None:
        self.cuts: dict[str, str] = {}  # sha256 of an original turn -> its bounded text

    def reapply(self, originals: list[str]) -> list[str]:
        return [self.cuts.get(_key(text), text) for text in originals]

    def shrink(self, originals: list[str], excess: int) -> tuple[list[str], int]:
        """Cut about `excess` characters; return the turns and how many are still over."""
        texts = self.reapply(originals)
        if not texts:
            return texts, excess
        older = sorted(range(len(texts) - 1), key=lambda i: len(texts[i]), reverse=True)
        if sum(len(texts[i]) for i in older) < MIN_YIELD * sum(map(len, texts)):
            older = []  # Low yield: cutting the older turns would not free enough.
        for i in [*older, len(texts) - 1]:  # The newest turn is bounded last.
            text = texts[i]
            marker = len(MARK.format(f"{len(text):,}"))  # The longest marker this cut can need.
            keep = max(MIN_PART, len(text) - excess - marker)
            if excess <= 0 or keep + marker >= len(text):
                continue
            head = keep * 2 // 3
            bounded = (
                text[:head] + MARK.format(f"{len(text) - keep:,}") + text[len(text) - keep + head :]
            )
            excess -= len(text) - len(bounded)
            texts[i] = self.cuts[_key(originals[i])] = bounded
        return texts, max(excess, 0)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
