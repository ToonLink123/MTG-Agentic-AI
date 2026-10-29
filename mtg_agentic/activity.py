"""Agent and tool progress events, one stderr line each.

`agent: orchestrator | start | claude-sonnet-5-5 | game plan and core` marks an LLM agent at work,
`tool: card search | otag:ramp` a tool it uses. The terminal UI draws these as a live agent
panel; on the plain CLI they are ordinary progress lines. Logging is thread-safe, so parallel
agents may report at once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)


@contextmanager
def agent(name: str, model: Any, task: str) -> Iterator[None]:
    """Report one agent's model call: start, then done with its time, or failed."""
    label = getattr(model, "model", None) or type(model).__name__
    started = time.monotonic()
    log.info("agent: %s | start | %s | %s", name, label, task)
    try:
        yield
    except BaseException as exc:
        log.info("agent: %s | failed | %s", name, type(exc).__name__)
        raise
    log.info("agent: %s | done | %.1fs", name, time.monotonic() - started)


def progress(name: str, detail: str) -> None:
    log.info("agent: %s | progress | %s", name, detail)


def tool(name: str, detail: str) -> None:
    log.info("tool: %s | %s", name, detail)
