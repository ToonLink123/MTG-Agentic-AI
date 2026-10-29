import gzip
import json
import tempfile
import unittest
from pathlib import Path

from mtg_agentic.oracle import OracleStore, read_bulk_cards
from tests.fixtures import BOLT, DFC, PLAINS, SOL_RING, card


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)

    def test_exact_case_insensitive_local_lookup_and_persistence(self):
        self.assertFalse(self.store.ready)
        self.store.import_cards([SOL_RING, BOLT], metadata={"updated_at": "today"})
        self.assertTrue(self.store.ready)
        self.assertEqual(self.store.get_card("  SOL   RING ")["name"], "Sol Ring")
        self.assertIsNone(self.store.get_card("Sol Rng"))
        self.assertNotIn("prices", self.store.get_card("Sol Ring"))
        with OracleStore(self.path / "oracle.sqlite3") as other:
            self.assertEqual(other.metadata, {"card_count": 2, "updated_at": "today", "index_version": 2})
            self.assertEqual(other.get_card("Sol Ring")["name"], "Sol Ring")

    def test_array_jsonl_and_gzip_imports(self):
        records = [SOL_RING, PLAINS, BOLT]
        for compressed in (False, True):
            for jsonl in (False, True):
                with self.subTest(compressed=compressed, jsonl=jsonl):
                    contents = (
                        "\n".join(json.dumps(entry) for entry in records)
                        if jsonl
                        else json.dumps(records)
                    )
                    path = self.path / "bulk"
                    path.write_bytes(
                        gzip.compress(contents.encode()) if compressed else contents.encode()
                    )
                    self.assertEqual(list(read_bulk_cards(path)), records)
                    self.assertEqual(self.store.import_file(path), 3)

    def test_array_streams_across_chunks_and_handles_bom(self):
        large = card("Large", text="A" * 100000)
        path = self.path / "bulk.json"
        path.write_text(json.dumps([SOL_RING, large, BOLT]), encoding="utf-8-sig")
        self.assertEqual(self.store.import_file(path), 3)
        self.assertEqual(len(self.store.get_card("Large")["oracle_text"]), 100000)

    def test_failed_import_is_atomic(self):
        self.store.import_cards([SOL_RING], metadata={"updated_at": "old"})

        def broken():
            yield BOLT
            raise ValueError("Interrupted")

        with self.assertRaises(ValueError):
            self.store.import_cards(broken(), metadata={"updated_at": "new"})
        self.assertEqual(self.store.metadata,
                         {"card_count": 1, "updated_at": "old", "index_version": 2})
        self.assertIsNotNone(self.store.get_card("Sol Ring"))
        self.assertIsNone(self.store.get_card("Lightning Bolt"))

    def test_empty_malformed_and_truncated_exports_preserve_snapshot(self):
        self.store.import_cards([SOL_RING])
        for contents in (
            "",
            "[]",
            "[",
            "[{}]",
            "[123]",
            '[{"name": "bad"}]',
            json.dumps([BOLT])[:-1],
            json.dumps([BOLT]) + " extra",
            json.dumps([BOLT])[:-1] + ",]",
            json.dumps([BOLT, BOLT]).replace("}, {", "} {"),
            "123\n",
        ):
            with self.subTest(contents=contents[:40]):
                path = self.path / "bad.json"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.store.import_file(path)
                self.assertIsNotNone(self.store.get_card("Sol Ring"))

    def test_face_aliases_share_one_card_and_back_only_names_do_not_resolve(self):
        self.store.import_cards([DFC])
        self.assertEqual(self.store.get_card("Front Face")["name"], DFC["name"])
        self.assertEqual(self.store.get_card(DFC["name"])["id"], DFC["id"])
        self.assertIsNone(self.store.get_card("Back Face"))

    def test_split_card_face_names_are_indexed(self):
        split = dict(DFC, layout="split")
        self.store.import_cards([split])
        self.assertEqual(self.store.get_card("Back Face")["id"], DFC["id"])

    def test_tokens_do_not_shadow_real_cards_and_name_variants_do_not_break_import(self):
        token = card("Sol Ring", layout="token", oracle_id="token")
        variant = card("Sol Ring", legal="not_legal", oracle_id="variant")
        self.store.import_cards([token, variant, SOL_RING])
        self.assertEqual(self.store.get_card("Sol Ring")["legalities"]["commander"], "legal")
        self.store.import_cards([SOL_RING, variant, token])
        self.assertEqual(self.store.get_card("Sol Ring")["legalities"]["commander"], "legal")

    def test_non_card_records_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.import_cards([{"object": "error"}])
