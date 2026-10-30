"""Agent and tool progress events, one stderr line each.

`agent: orchestrator | start | claude-sonnet-5-5 | game plan and core` marks an LLM agent at work,
`tool: card search | otag:ramp` a tool it uses. Each line carries a structured event (see
events.py): the terminal UI reads those to draw a live agent panel; on the plain CLI they are
ordinary progress lines. Logging is thread-safe, so parallel agents may report at once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .events import event

log = logging.getLogger(__name__)


@contextmanager
def agent(name: str, model: Any, task: str) -> Iterator[None]:
    """Report one agent's model call: start, then done with its time, or failed."""
    label = getattr(model, "model", None) or type(model).__name__
    started = time.monotonic()
    log.info("agent: %s | start | %s | %s", name, label, task,
             extra=event("agent", name=name, state="start", model=label, task=task))
    try:
        yield
    except BaseException as exc:
        log.info("agent: %s | failed | %s", name, type(exc).__name__,
                 extra=event("agent", name=name, state="failed", note=type(exc).__name__))
        raise
    seconds = time.monotonic() - started
    log.info("agent: %s | done | %.1fs", name, seconds,
             extra=event("agent", name=name, state="done", seconds=round(seconds, 1)))


def progress(name: str, detail: str) -> None:
    log.info("agent: %s | progress | %s", name, detail,
             extra=event("agent", name=name, state="progress", note=detail))


def tool(name: str, detail: str) -> None:
    log.info("tool: %s | %s", name, detail, extra=event("tool", name=name, detail=detail))
