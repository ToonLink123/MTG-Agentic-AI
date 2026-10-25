import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from mtg_agentic.oracle import OracleStore
from mtg_agentic.scryfall import CardNotFound, ScryfallClient
from tests.fixtures import BOLT, SOL_RING, card


class ClientExtensionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    @patch("mtg_agentic.scryfall.urlopen")
    def test_result_cap_stops_before_next_page_and_preserves_sort_order(self, http):
        http.return_value = io.BytesIO(
            json.dumps(
                {
                    "data": [SOL_RING, BOLT],
                    "has_more": True,
                    "next_page": "https://api.scryfall.com/cards/search?page=2",
                }
            ).encode()
        )
        with ScryfallClient(self.path / "cache.sqlite3") as client:
            result = client.search_cards("t:artifact", max_results=1, order="edhrec")
        self.assertEqual(len(result), 1)
        self.assertIn("order=edhrec", http.call_args.args[0].full_url)
        self.assertEqual(http.call_count, 1)

    @patch("mtg_agentic.scryfall.urlopen")
    def test_exact_only_lookup_never_attempts_fuzzy(self, http):
        http.side_effect = HTTPError(
            "https://api.scryfall.com", 404, "Missing", {}, io.BytesIO(b"{}")
        )
        with ScryfallClient(self.path / "cache.sqlite3") as client:
            with self.assertRaises(CardNotFound):
                client.get_card("Imaginary", fuzzy=False)
        self.assertEqual(http.call_count, 1)

    @patch("mtg_agentic.scryfall.urlopen")
    def test_card_record_retains_rules_and_reuses_named_cache(self, http):
        land = card("Test Land", types="Land", produced_mana=["W"], cmc=0)
        http.return_value = io.BytesIO(json.dumps(land).encode())
        with ScryfallClient(self.path / "cache.sqlite3") as client:
            self.assertEqual(len(client.get_card("Test Land", fuzzy=False)), 7)
            record = client.get_card_record("Test Land")
        self.assertEqual(record["legalities"]["commander"], "legal")
        self.assertEqual(record["produced_mana"], ["W"])
        self.assertEqual(http.call_count, 1)

    def test_verified_additions_preserve_existing_cards_and_snapshot_metadata(self):
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards([SOL_RING], metadata={"updated_at": "old"})
            store.add_verified_card(BOLT)
            self.assertEqual(store.metadata["updated_at"], "old")
            self.assertEqual(store.metadata["card_count"], 2)
            self.assertIn("Lightning Bolt", store.metadata["verified_cards"])
            self.assertIsNotNone(store.get_card("Sol Ring"))
        with OracleStore(self.path / "oracle.sqlite3") as reopened:
            self.assertIsNotNone(reopened.get_card("Lightning Bolt"))
