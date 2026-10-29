"""Commander Spellbook's export, streamed into the local store and searched there."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.brackets import (
    SPELLBOOK_EXPORT,
    SpellbookClient,
    compact_variant,
    stream_variants,
)
from mtg_agentic.oracle import OracleStore
from mtg_agentic.sources import SourceError


def use(name, quantity=1, commander=False):
    return {"card": {"name": name, "oracleId": "x", "imageUriFrontPng": "https://img.invalid"},
            "zoneLocations": ["B"], "mustBeCommander": commander, "quantity": quantity}


def variant(id_, *uses, status="OK", requires=(), produces=("Infinite mana",)):
    return {
        "id": id_, "status": status, "uses": list(uses), "requires": list(requires),
        "produces": [{"feature": {"id": 1, "name": p}, "quantity": 1} for p in produces],
        "manaNeeded": "{2}", "easyPrerequisites": "", "notablePrerequisites": "Untapped.",
        "description": "Tap, untap, repeat.", "legalities": {"commander": True},
    }


VARIANTS = [
    variant("1-2", use("Basalt Monolith"), use("Rings of Brighthearth")),
    variant("3-4", use("Dramatic Reversal"), use("Isochron Scepter"),
            produces=("Infinite mana", "Win the game")),
    variant("5", use("Basalt Monolith"), requires=[{"template": {"name": "Untapper"}}]),
    variant("6-7-8", use("Kiki-Jiki, Mirror Breaker"), use("Zealous Conscripts"), use("Sol Ring")),
    variant("9-10", use("Kiki-Jiki, Mirror Breaker", commander=True), use("Pestermite")),
    variant("11-12", use("Basalt Monolith"), use("Power Artifact"), status="NR"),
]
EXPORT = json.dumps({"timestamp": "today", "version": "7", "variants": VARIANTS})


class SpellbookLocalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = OracleStore(Path(directory.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)

    def test_the_export_streams_across_chunk_boundaries(self):
        for chunk in (7, 64, 1 << 20):
            with self.subTest(chunk=chunk):
                found = list(stream_variants(io.StringIO(EXPORT), chunk=chunk))
                self.assertEqual([v["id"] for v in found], [v["id"] for v in VARIANTS])
        with self.assertRaises(SourceError):
            list(stream_variants(io.StringIO(EXPORT[:-200]), chunk=64))
        with self.assertRaises(SourceError):
            list(stream_variants(io.StringIO('{"timestamp": "today"}')))

    def test_only_valid_variants_of_two_cards_or_fewer_are_kept_without_images(self):
        kept = [compact_variant(v) for v in VARIANTS]
        self.assertEqual([k["id"] if k else None for k in kept], ["1-2", "3-4", "5", None, "9-10", None])
        self.assertEqual(kept[0]["uses"][0]["card"], {"name": "Basalt Monolith"})
        self.assertEqual(kept[0]["uses"][0]["zoneLocations"], ["B"])
        self.assertEqual(kept[2]["requires"], [{"template": True}])

    def test_local_combos_match_the_api_rules_with_nothing_truncated(self):
        self.store.import_combos(c for c in map(compact_variant, VARIANTS) if c)
        spellbook = SpellbookClient(store=self.store)
        found = spellbook.find_combos(
            {"Dramatic Reversal": 1, "Isochron Scepter": 1, "Pestermite": 1, "Sol Ring": 1},
            "Kiki-Jiki, Mirror Breaker",
        )
        self.assertEqual([c["id"] for c in found["combos"]], ["3-4", "9-10"])
        self.assertTrue(found["combos"][0]["game_winning"])
        self.assertTrue(found["complete"])
        self.assertEqual(found["source"], SPELLBOOK_EXPORT)
        # Kiki-Jiki in the 99 is not the commander the variant needs.
        found = spellbook.find_combos({"Kiki-Jiki, Mirror Breaker": 1, "Pestermite": 1}, "Sol Ring")
        self.assertEqual(found["combos"], [])
        # A template variant whose named card is present leaves the search incomplete.
        found = spellbook.find_combos({"Basalt Monolith": 1, "Rings of Brighthearth": 1}, "Sol Ring")
        self.assertEqual([c["id"] for c in found["combos"]], ["1-2"])
        self.assertFalse(found["complete"])

    def test_the_export_downloads_once_and_streams_into_the_store(self):
        response = io.BytesIO(EXPORT.encode())
        response.geturl = lambda: SPELLBOOK_EXPORT
        with patch("mtg_agentic.brackets.urlopen", return_value=response) as fetch:
            spellbook = SpellbookClient(store=self.store)
            info = spellbook.download_variants(self.store)
            self.assertEqual(spellbook.download_variants(self.store), info)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual((info["combo_count"], info["source"]), (4, SPELLBOOK_EXPORT))
        self.assertEqual(len(self.store.combo_candidates(["Basalt Monolith"])), 1)

    def test_without_the_export_or_a_cache_there_is_no_silent_empty_answer(self):
        with self.assertRaises(SourceError):
            SpellbookClient(store=self.store).find_combos({"Sol Ring": 1}, "Kiki-Jiki, Mirror Breaker")


if __name__ == "__main__":
    unittest.main()
