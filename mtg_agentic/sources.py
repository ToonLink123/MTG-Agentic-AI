"""Dated, bounded HTTP cache for public rules and read-only combo searches."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .scryfall import USER_AGENT


class SourceError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceCache:
    """No stale-success fallback. A failed refresh leaves the old cache intact."""

    def __init__(self, path: str | Path, *, timeout: float = 90, ttl: float = 86400):
        if timeout <= 0 or ttl <= 0:
            raise ValueError("Source timeout and TTL must be positive")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sources "
            "(key TEXT PRIMARY KEY, fetched REAL, stamp TEXT, body BLOB)"
        )
        self.timeout, self.ttl = timeout, ttl
        self._next_spellbook = 0.0

    def __enter__(self) -> SourceCache:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def get(self, key: str) -> tuple[bytes, str] | None:
        row = self.db.execute(
            "SELECT body, stamp FROM sources WHERE key=? AND fetched>?",
            (key, time.time() - self.ttl),
        ).fetchone()
        return (bytes(row[0]), row[1]) if row else None

    def put(self, key: str, body: bytes) -> str:
        stamp = utc_now()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?)",
                (key, time.time(), stamp, body),
            )
        return stamp

    def fetch(
        self, url: str, *, payload: dict[str, Any] | None = None, refresh: bool = False
    ) -> tuple[bytes, str]:
        parts = urlsplit(url)
        allowed = {"magic.wizards.com", "media.wizards.com", "backend.commanderspellbook.com"}
        if parts.scheme != "https" or parts.netloc not in allowed or parts.fragment:
            raise SourceError("Unexpected evidence source URL")
        if payload is not None and (
            parts.netloc != "backend.commanderspellbook.com" or parts.path != "/find-my-combos"
        ):
            raise SourceError("Only read-only Spellbook deck searches support POST")
        data = json.dumps(payload, sort_keys=True).encode() if payload is not None else None
        key = url + ":" + hashlib.sha256(data or b"").hexdigest()
        cached = None if refresh else self.get(key)
        if cached:
            return cached
        if parts.netloc == "backend.commanderspellbook.com":
            delay = self._next_spellbook - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_spellbook = time.monotonic() + 0.8
        request = Request(
            url,
            data=data,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                final = urlsplit(response.geturl())
                if final.scheme != "https" or final.netloc != parts.netloc:
                    raise SourceError("Evidence source redirected to an unexpected host")
                body = response.read(16 * 1024 * 1024 + 1)
            if len(body) > 16 * 1024 * 1024:
                raise SourceError("Evidence response exceeded 16 MiB")
        except HTTPError as exc:
            status = exc.code
            if status == 429 and parts.netloc == "backend.commanderspellbook.com":
                delay = 60.0
                retry = exc.headers.get("Retry-After", "")
                try:
                    delay = max(delay, float(retry))
                except ValueError:
                    try:
                        delay = max(delay, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                self._next_spellbook = time.monotonic() + delay
            exc.close()
            raise SourceError(f"Evidence source returned HTTP {status}") from None
        except (URLError, OSError) as exc:
            raise SourceError("Cannot reach the evidence source; retry later") from exc
        stamp = self.put(key, body)
        return body, stamp
