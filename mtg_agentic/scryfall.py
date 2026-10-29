"""Rate-limited Scryfall access with a persistent SQLite response cache."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from .cards import Card, trim_card, validation_card
from .oracle import DEFAULT_DATA_DIR, OracleStore

API_URL = "https://api.scryfall.com"
USER_AGENT = "MTG-Agentic-AI/0.1.0"


class ScryfallError(RuntimeError):
    def __init__(self, detail: str, *, status: int | None = None, code: str | None = None):
        super().__init__(detail)
        self.status = status
        self.code = code


class CardNotFound(ScryfallError):
    pass


class ScryfallClient:
    """Use one client per process/thread to share cache and rate-limit state."""

    def __init__(
        self,
        cache_path: str | Path = DEFAULT_DATA_DIR / "responses.sqlite3",
        *,
        user_agent: str = USER_AGENT,
        request_delay: float = 0.1,
        cache_ttl: float = 24 * 60 * 60,
        timeout: float = 60,
    ):
        if not user_agent.strip():
            raise ValueError("A nonempty User-Agent is required")
        if request_delay < 0.1:
            raise ValueError("The minimum request delay is 100ms")
        if cache_ttl <= 0 or timeout <= 0:
            raise ValueError("Cache TTL and timeout must be positive")
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = sqlite3.connect(cache_path)
        self._cache.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(url TEXT PRIMARY KEY, expires_at REAL NOT NULL, status INTEGER NOT NULL, "
            "body TEXT NOT NULL)"
        )
        self._lock = threading.Lock()
        self._last_request: float | None = None
        self._last_endpoint: dict[str, float] = {}
        self._blocked_until = 0.0

    def __enter__(self) -> ScryfallClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._cache.close()

    @staticmethod
    def _api_url(url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "api.scryfall.com":
            raise ScryfallError("Refusing an unexpected Scryfall API URL")
        return url

    def _open(self, url: str) -> Any:
        endpoint = urlsplit(url).path
        with self._lock:
            now = time.monotonic()
            earliest = self._blocked_until
            if self._last_request is not None:
                earliest = max(earliest, self._last_request + self.request_delay)
            if endpoint in {"/cards/search", "/cards/named"} and endpoint in self._last_endpoint:
                earliest = max(earliest, self._last_endpoint[endpoint] + 0.5)
            if earliest > now:
                time.sleep(earliest - now)
            self._last_request = time.monotonic()
            self._last_endpoint[endpoint] = self._last_request
            request = Request(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json, */*;q=0.8"},
            )
            try:
                return urlopen(request, timeout=self.timeout)
            except HTTPError as exc:
                if exc.code == 429:
                    delay = 30.0
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                date = parsedate_to_datetime(retry_after)
                                delay = max(delay, date.timestamp() - time.time())
                            except (ValueError, TypeError, OverflowError):
                                pass
                    self._blocked_until = time.monotonic() + delay
                raise
            except (URLError, OSError) as exc:
                raise ScryfallError(f"Scryfall connection failed: {exc}") from exc

    def _json(self, url: str, *, use_cache: bool = True) -> dict[str, Any]:
        self._api_url(url)
        cached = None
        if use_cache:
            cached = self._cache.execute(
                "SELECT status, body FROM responses WHERE url = ? AND expires_at > ?",
                (url, time.time()),
            ).fetchone()
        if cached:
            status, body = cached
            payload = json.loads(body)
        else:
            try:
                with self._open(url) as response:
                    status = 200
                    payload = json.load(response)
            except HTTPError as exc:
                status = exc.code
                try:
                    payload = json.loads(exc.read())
                except (ValueError, OSError):
                    payload = {"details": f"Scryfall returned HTTP {status}"}
                finally:
                    exc.close()
            except (ValueError, OSError) as exc:
                raise ScryfallError(f"Invalid Scryfall response: {exc}") from exc
            if not isinstance(payload, dict):
                raise ScryfallError("Expected a Scryfall JSON object")
            if use_cache and status in {200, 404}:
                with self._cache:
                    self._cache.execute(
                        "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                        (url, time.time() + self.cache_ttl, status, json.dumps(payload)),
                    )
        if status != 200 or payload.get("object") == "error":
            error_type = CardNotFound if status == 404 else ScryfallError
            raise error_type(
                payload.get("details", f"Scryfall returned HTTP {status}"),
                status=status,
                code=payload.get("code"),
            )
        return payload

    def _pages(
        self, url: str, first_page: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        visited: set[str] = set()
        while url:
            if url in visited:
                raise ScryfallError("Scryfall returned a pagination loop")
            visited.add(url)
            page = first_page if first_page is not None else self._json(url)
            first_page = None
            if not isinstance(page.get("data"), list):
                raise ScryfallError("Scryfall returned a list without data")
            yield from page["data"]
            if not page.get("has_more"):
                return
            url = page.get("next_page")
            if not isinstance(url, str) or not url:
                raise ScryfallError("Scryfall omitted the next page URL")
            self._api_url(url)

    def search_cards(
        self, query: str, *, max_results: int | None = None, order: str | None = None
    ) -> list[Card]:
        """Return all search pages, with no images, prices, print data, or other noise."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be nonempty")
        if max_results is not None and (type(max_results) is not int or max_results < 1):
            raise ValueError("max_results must be a positive integer")
        if order is not None and order not in {"name", "edhrec"}:
            raise ValueError("Unsupported search ordering")
        params = {"q": query, "unique": "cards"}
        if order is not None:
            params["order"] = order
        url = f"{API_URL}/cards/search?{urlencode(params)}"
        try:
            first = self._json(url)
        except CardNotFound:
            return []
        pages = self._pages(url, first)
        selected = pages if max_results is None else islice(pages, max_results)
        return [trim_card(card) for card in selected]

    def _named_card(self, name: str, *, fuzzy: bool) -> Card:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Card name must be nonempty")
        try:
            card = self._json(f"{API_URL}/cards/named?{urlencode({'exact': name})}")
        except CardNotFound:
            if not fuzzy:
                raise
            card = self._json(f"{API_URL}/cards/named?{urlencode({'fuzzy': name})}")
        return card

    def get_card(self, name: str, *, fuzzy: bool = True) -> Card:
        """Exact first; disable fuzzy fallback when verifying generated names."""
        return trim_card(self._named_card(name, fuzzy=fuzzy))

    def get_card_record(self, name: str) -> Card:
        """Exact-only rule fields for indexing newly verified cards, never LLM context."""
        return validation_card(self._named_card(name, fuzzy=False))

    def get_rulings(self, card_id: str) -> list[dict[str, Any]]:
        """Live rulings for one card; the pipeline reads them locally with card_rulings."""
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError("Card ID must be nonempty")
        return [
            {key: ruling.get(key) for key in ("source", "published_at", "comment")}
            for ruling in self._pages(f"{API_URL}/cards/{quote(card_id, safe='')}/rulings")
        ]

    def _bulk(self, kind: str, store: OracleStore, load) -> None:
        """Download one bulk file from Scryfall's file host, which has no rate limit, hand it to
        `load(path, metadata)`, then keep it beside the store as `<kind>.<ext>`."""
        metadata = self._json(f"{API_URL}/bulk-data/{kind}", use_cache=False)
        url = metadata.get("jsonl_download_uri") or metadata.get("download_uri")
        if not isinstance(url, str):
            raise ScryfallError(f"The {kind} bulk metadata has no download URL")
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc not in {"data.scryfall.io", "data.scryfall.com"}:
            raise ScryfallError("Refusing an unexpected Scryfall bulk URL")
        suffix = ".jsonl.gz" if metadata.get("jsonl_download_uri") else ".json"
        destination = store.path.parent / f"{kind}{suffix}"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=store.path.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with self._open(url) as response:
                        shutil.copyfileobj(response, output)
                except HTTPError as exc:
                    status = exc.code
                    exc.close()
                    raise ScryfallError(f"The {kind} bulk download failed", status=status) from exc
            load(temporary, {
                "updated_at": metadata.get("updated_at"),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "source": url,
            })
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def download_oracle_cards(self, store: OracleStore, *, force: bool = False) -> dict[str, Any]:
        """Download/index once. Refresh explicitly when legality or sets change. An index built
        by an older version is rebuilt from the bulk file already on disk, with no download."""
        if store.ready and store.current and not force:
            return store.metadata
        saved = next((p for p in (store.path.parent / "oracle-cards.jsonl.gz",
                                  store.path.parent / "oracle-cards.json") if p.is_file()), None)
        if store.ready and not force and saved is not None:
            old = store.metadata
            store.import_file(saved, metadata={
                k: old[k] for k in ("updated_at", "downloaded_at", "source") if k in old
            })
            return store.metadata
        self._bulk("oracle-cards", store, lambda path, info: store.import_file(path, metadata=info))
        return store.metadata

    def download_rulings(self, store: OracleStore, *, force: bool = False) -> dict[str, Any]:
        """Every card's rulings (Scryfall's Rulings bulk file), downloaded once like the cards."""
        if store.rulings_ready and not force:
            return store.ruling_metadata
        self._bulk("rulings", store, lambda path, info: store.import_ruling_file(path, metadata=info))
        return store.ruling_metadata

    def download_oracle_tags(self, store: OracleStore, *, force: bool = False) -> dict[str, Any]:
        """Scryfall's Oracle tags (the Tagger data behind otag:), downloaded once like the cards."""
        if store.tags_ready and not force:
            return store.tag_metadata
        self._bulk("oracle-tags", store, lambda path, info: store.import_tag_file(path, metadata=info))
        return store.tag_metadata


def card_rulings(store: OracleStore, client: Any, card: Card) -> list[dict[str, Any]]:
    """A card's rulings from the local Rulings bulk file, downloaded the first time, instead of
    one rate-limited API call per card. A client without bulk downloads answers itself."""
    if not store.rulings_ready and hasattr(client, "download_rulings"):
        client.download_rulings(store)
    if store.rulings_ready:
        return store.rulings(card["oracle_id"])
    return client.get_rulings(card["id"])
