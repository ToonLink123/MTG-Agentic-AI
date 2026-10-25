import gzip
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from mtg_agentic.cards import trim_card
from mtg_agentic.oracle import OracleStore
from mtg_agentic.scryfall import API_URL, CardNotFound, ScryfallClient, ScryfallError
from tests.fixtures import BOLT, DFC, SOL_RING


def response(payload):
    return io.BytesIO(json.dumps(payload).encode())


def http_error(status, details="Failure", retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(
        API_URL,
        status,
        "Failure",
        headers,
        response({"object": "error", "code": "not_found", "details": details}),
    )


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class ScryfallTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.client = ScryfallClient(self.path / "cache.sqlite3")
        self.addCleanup(self.client.close)
        self.clock = Clock()
        self.monotonic = patch("mtg_agentic.scryfall.time.monotonic", self.clock.monotonic)
        self.sleep = patch("mtg_agentic.scryfall.time.sleep", self.clock.sleep)
        self.monotonic.start()
        self.sleep.start()
        self.addCleanup(self.monotonic.stop)
        self.addCleanup(self.sleep.stop)
        self.http_patch = patch("mtg_agentic.scryfall.urlopen")
        self.http = self.http_patch.start()
        self.addCleanup(self.http_patch.stop)

    def test_exact_lookup_headers_encoding_and_trim(self):
        self.http.return_value = response(BOLT)
        result = self.client.get_card("Lightning Bolt")
        request = self.http.call_args.args[0]
        self.assertEqual(parse_qs(urlsplit(request.full_url).query), {"exact": ["Lightning Bolt"]})
        self.assertEqual(request.get_header("User-agent"), "MTG-Agentic-AI/0.1.0")
        self.assertIn("application/json", request.get_header("Accept"))
        self.assertEqual(
            set(result),
            {
                "id",
                "name",
                "mana_cost",
                "type_line",
                "oracle_text",
                "color_identity",
                "game_changer",
            },
        )

    def test_fuzzy_only_after_404_and_both_responses_cached(self):
        self.http.side_effect = [http_error(404), response(SOL_RING)]
        self.assertEqual(self.client.get_card("Sol Rng")["name"], "Sol Ring")
        self.assertEqual(self.client.get_card("Sol Rng")["name"], "Sol Ring")
        queries = [
            parse_qs(urlsplit(call.args[0].full_url).query) for call in self.http.call_args_list
        ]
        self.assertEqual(queries, [{"exact": ["Sol Rng"]}, {"fuzzy": ["Sol Rng"]}])
        self.assertEqual(self.clock.sleeps, [0.5])

    def test_fuzzy_failure_is_not_found(self):
        self.http.side_effect = [http_error(404), http_error(404, "No unique match")]
        with self.assertRaisesRegex(CardNotFound, "No unique match"):
            self.client.get_card("Not a real card")

    def test_no_fuzzy_on_server_error(self):
        self.http.side_effect = http_error(503)
        with self.assertRaises(ScryfallError) as error:
            self.client.get_card("Sol Ring")
        self.assertEqual(error.exception.status, 503)
        self.assertEqual(self.http.call_count, 1)

    def test_persistent_cache_and_expiry(self):
        self.http.return_value = response(SOL_RING)
        self.client.get_card("Sol Ring")
        with ScryfallClient(self.path / "cache.sqlite3") as other:
            self.assertEqual(other.get_card("Sol Ring")["name"], "Sol Ring")
            self.assertEqual(self.http.call_count, 1)
        with self.client._cache:
            self.client._cache.execute("UPDATE responses SET expires_at = 0")
        self.http.return_value = response(SOL_RING)
        self.client.get_card("Sol Ring")
        self.assertEqual(self.http.call_count, 2)

    def test_search_pagination_and_cache(self):
        next_page = f"{API_URL}/cards/search?page=2&q=artifact"
        self.http.side_effect = [
            response({"data": [SOL_RING], "has_more": True, "next_page": next_page}),
            response({"data": [BOLT], "has_more": False}),
        ]
        result = self.client.search_cards("artifact")
        self.assertEqual([entry["name"] for entry in result], ["Sol Ring", "Lightning Bolt"])
        self.assertEqual(self.client.search_cards("artifact"), result)
        self.assertEqual(self.http.call_count, 2)
        self.assertEqual(self.clock.sleeps, [0.5])

    def test_empty_search_and_late_page_failure(self):
        self.http.side_effect = http_error(404)
        self.assertEqual(self.client.search_cards("none"), [])
        self.http.side_effect = [
            response(
                {
                    "data": [SOL_RING],
                    "has_more": True,
                    "next_page": f"{API_URL}/cards/search?page=2",
                }
            ),
            http_error(404),
        ]
        with self.assertRaises(CardNotFound):
            self.client.search_cards("artifact")

    def test_unsafe_or_broken_pagination_is_rejected(self):
        for next_page in ("http://api.scryfall.com/x", "https://example.com/x", None):
            with self.subTest(next_page=next_page):
                self.http.return_value = response(
                    {"data": [SOL_RING], "has_more": True, "next_page": next_page}
                )
                with self.assertRaises(ScryfallError):
                    self.client.search_cards(str(next_page))

    def test_pagination_loop_is_rejected(self):
        url = f"{API_URL}/cards/search?q=loop&unique=cards"
        self.http.return_value = response({"data": [], "has_more": True, "next_page": url})
        with self.assertRaisesRegex(ScryfallError, "loop"):
            self.client.search_cards("loop")

    def test_rulings_paginate_trim_and_observe_100ms_delay(self):
        self.http.side_effect = [
            response(
                {
                    "data": [
                        {"source": "wotc", "published_at": "2026-01-01", "comment": "A", "id": 1}
                    ],
                    "has_more": True,
                    "next_page": f"{API_URL}/cards/test/rulings?page=2",
                }
            ),
            response({"data": [{"source": "wotc", "comment": "B"}], "has_more": False}),
        ]
        rulings = self.client.get_rulings("test")
        self.assertEqual(len(rulings), 2)
        self.assertEqual(set(rulings[0]), {"source", "published_at", "comment"})
        self.assertEqual(self.clock.sleeps, [0.1])
        self.assertEqual(self.client.get_rulings("test"), rulings)
        self.assertEqual(self.http.call_count, 2)

    def test_rate_limit_error_honors_retry_after_and_is_not_cached(self):
        self.http.side_effect = [http_error(429, retry_after="45"), response(SOL_RING)]
        with self.assertRaises(ScryfallError) as error:
            self.client.get_card("Sol Ring")
        self.assertEqual(error.exception.status, 429)
        self.client.get_card("Sol Ring")
        self.assertEqual(self.clock.sleeps, [45.0])

    def test_rate_limit_without_header_waits_30_seconds(self):
        self.http.side_effect = [http_error(429), response(SOL_RING)]
        with self.assertRaises(ScryfallError):
            self.client.get_card("Sol Ring")
        self.client.get_card("Sol Ring")
        self.assertEqual(self.clock.sleeps, [30.0])

    def test_connection_and_invalid_json_errors(self):
        for error in (URLError("Offline"), OSError("Socket failed")):
            self.http.side_effect = error
            with self.assertRaises(ScryfallError):
                self.client.get_card("Sol Ring")
        self.http.side_effect = None
        self.http.return_value = io.BytesIO(b"not json")
        with self.assertRaises(ScryfallError):
            self.client.get_card("Sol Ring")
        self.http.return_value = response([])
        with self.assertRaises(ScryfallError):
            self.client.get_card("Sol Ring")

    def test_input_validation(self):
        for method in (self.client.search_cards, self.client.get_card, self.client.get_rulings):
            with self.assertRaises(ValueError):
                method(" ")
        with self.assertRaises(ValueError):
            ScryfallClient(self.path / "other.sqlite3", request_delay=0.01)
        self.http.assert_not_called()

    def test_multiface_projection(self):
        raw = dict(DFC)
        raw.pop("mana_cost")
        raw.pop("oracle_text")
        projected = trim_card(raw)
        self.assertEqual(projected["mana_cost"], "{W} // ")
        self.assertEqual(projected["oracle_text"], "Front rules.\n//\nBack rules.")
        self.assertFalse(projected["game_changer"])

    def test_bulk_download_once_and_explicit_refresh(self):
        with OracleStore(self.path / "oracle.sqlite3") as store:
            for field, contents in (
                ("jsonl_download_uri", gzip.compress((json.dumps(SOL_RING) + "\n").encode())),
                ("download_uri", json.dumps([BOLT]).encode()),
            ):
                self.http.side_effect = [
                    response(
                        {field: "https://data.scryfall.io/oracle.json", "updated_at": "today"}
                    ),
                    io.BytesIO(contents),
                ]
                info = self.client.download_oracle_cards(store, force=store.ready)
                self.assertEqual(info["card_count"], 1)
                self.assertEqual(info["updated_at"], "today")
                calls = self.http.call_count
                self.assertEqual(self.client.download_oracle_cards(store), info)
                self.assertEqual(self.http.call_count, calls)
            self.assertIsNone(store.get_card("Sol Ring"))
            self.assertIsNotNone(store.get_card("Lightning Bolt"))

    def test_bad_bulk_does_not_replace_snapshot_or_leave_temporary_file(self):
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards([SOL_RING])
            before = set(self.path.iterdir())
            self.http.side_effect = [
                response({"download_uri": "https://data.scryfall.io/oracle.json"}),
                io.BytesIO(b'[{"bad": true}]'),
            ]
            with self.assertRaises(ValueError):
                self.client.download_oracle_cards(store, force=True)
            self.assertIsNotNone(store.get_card("Sol Ring"))
            self.assertEqual(set(self.path.iterdir()), before)

    def test_bulk_download_refuses_unknown_host(self):
        self.http.return_value = response({"download_uri": "https://example.com/bulk.json"})
        with OracleStore(self.path / "oracle.sqlite3") as store:
            with self.assertRaises(ScryfallError):
                self.client.download_oracle_cards(store)
        self.assertEqual(self.http.call_count, 1)
