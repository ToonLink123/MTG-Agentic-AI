"""Progress as structured events: text for people, JSON lines for the terminal app.

Code reports progress with `log.info(text, extra=event(kind, ...))`. The plain CLI prints the
text. `python -m mtg_agentic --events ...` (what the TUI runs) prints one JSON object per line on
stderr instead: the event with its text as "message", and {"type": "log", "message": ...} for any
other progress line. The TUI reads those fields rather than matching text with patterns, so
wording can change without breaking it.

Kinds: agent (name, state: start | progress | done | failed, model, task, note, seconds), tool
(name, detail), stage (scope: council | setup, index, total, name, detail), usage (one model
call: model, input, cached, written, output, cost), and log.
"""

from __future__ import annotations

import json
import logging
from typing import Any


def event(kind: str, **fields: Any) -> dict[str, Any]:
    """The logging `extra` that attaches an event to a progress line."""
    return {"event": {"type": kind, **fields}}


class JsonLines(logging.Formatter):
    """Each record as one JSON line: its event, or a plain log line, with the text as message."""

    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "event", None) or {"type": "log"}
        return json.dumps({**data, "message": record.getMessage()}, ensure_ascii=True)


def parse(line: str) -> dict[str, Any] | None:
    """An event from one stderr line, or None for anything else (a warning, a traceback)."""
    if not line.startswith("{"):
        return None
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("type"), str) else None
