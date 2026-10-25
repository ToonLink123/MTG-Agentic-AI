import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from mtg_agentic.sources import SourceCache, SourceError

URL = "https://backend.commanderspellbook.com/find-my-combos"


def response(body=b"{}"):
    result = io.BytesIO(body)
    result.geturl = lambda: URL
    return result


class SourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.cache = SourceCache(Path(temp.name) / "source.sqlite3")
        self.addCleanup(self.cache.close)

    @patch("mtg_agentic.sources.urlopen")
    def test_read_only_post_is_cached_by_payload_with_dates_and_user_agent(self, http):
        http.side_effect = [response(), response()]
        first = self.cache.fetch(URL, payload={"main": []})
        self.assertEqual(first, self.cache.fetch(URL, payload={"main": []}))
        self.cache.fetch(URL, payload={"main": [{"card": "Sol Ring"}]})
        self.assertEqual(http.call_count, 2)
        request = http.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertIn("MTG-Agentic", request.get_header("User-agent"))
        self.assertTrue(first[1])

    @patch("mtg_agentic.sources.urlopen")
    def test_refresh_failure_preserves_old_cache_but_does_not_report_success(self, http):
        http.return_value = response()
        before = self.cache.fetch(URL)
        http.side_effect = URLError("offline")
        with self.assertRaises(SourceError):
            self.cache.fetch(URL, refresh=True)
        self.assertEqual(self.cache.fetch(URL), before)

    @patch("mtg_agentic.sources.urlopen")
    def test_429_sets_a_cooldown_without_retry_storm(self, http):
        http.side_effect = HTTPError(URL, 429, "Slow", {"Retry-After": "120"}, io.BytesIO())
        with self.assertRaises(SourceError):
            self.cache.fetch(URL)
        self.assertGreater(self.cache._next_spellbook, 0)
        self.assertEqual(http.call_count, 1)

    def test_unexpected_hosts_or_write_endpoints_are_rejected(self):
        for url, payload in [
            ("http://localhost:11434", None),
            ("https://evil.invalid/rules", None),
            ("https://backend.commanderspellbook.com/suggestions", {"anything": 1}),
        ]:
            with self.subTest(url=url), self.assertRaises(SourceError):
                self.cache.fetch(url, payload=payload)

    @patch("mtg_agentic.sources.time.time")
    def test_expired_entries_are_not_used(self, clock):
        clock.return_value = 0
        self.cache.put("key", b"old")
        clock.return_value = 86401
        self.assertIsNone(self.cache.get("key"))
