"""First-run setup: what is missing, and the command that downloads only that."""

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic import bootstrap
from mtg_agentic.__main__ import main
from mtg_agentic.oracle import OracleStore
from tests.pipeline_fixtures import records

TAGS = [{"object": "tag", "id": "t", "slug": "ramp", "label": "ramp", "type": "oracle",
         "taggings": [{"oracle_id": "Spell 1"}]}]
RULINGS = [{"oracle_id": "Spell 1", "source": "wotc", "published_at": "2024-01-01",
            "comment": "A ruling."}]
NO_RUST = "Rust isn't installed; install it from https://rustup.rs, then run setup again"
GOLDFISH_READY = bootstrap.goldfish_ready  # setUp replaces it; one test checks the real one.
COMBOS = [{"id": "1-2", "status": "OK", "requires": [], "produces": [],
           "uses": [{"card": {"name": "Spell 1"}, "quantity": 1, "mustBeCommander": False},
                    {"card": {"name": "Spell 2"}, "quantity": 1, "mustBeCommander": False}]}]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        for name, value in (("model_cached", True), ("embedding_problem", None),
                            ("goldfish_ready", True)):
            patcher = patch.object(bootstrap, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Setup never runs cargo in unit tests.
        patcher = patch("mtg_agentic.__main__.build_problem", return_value=NO_RUST)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_status_reads_what_is_already_here_without_downloading(self):
        self.assertEqual(bootstrap.status(self.path), {
            "Card data": False, "Oracle tags": False, "Rulings": False,
            "Commander Spellbook": False, "Rules index": False, "Embedding model": True,
            "Card index": False, "Goldfish engine": True,
        })
        with OracleStore(self.path / "oracle.sqlite3") as store:
            store.import_cards(records())
            store.import_tags(TAGS)
            store.import_rulings(RULINGS)
            store.import_combos(COMBOS)
            store.import_card_vectors([("Spell 1", b"\0" * 8)], metadata={
                "embedding": bootstrap.card_index_identity(),
                "card_count": store.metadata["card_count"],
            })
        with contextlib.closing(sqlite3.connect(self.path / bootstrap.RULES_INDEX)) as db, db:
            db.execute("CREATE TABLE rule_metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO rule_metadata VALUES ('rule_count', '4')")
        self.assertTrue(all(bootstrap.status(self.path).values()))

    def run_setup(self, *args):
        def cards(client, store, force=False):
            store.import_cards(records(), metadata={"updated_at": "today"})
            return store.metadata

        def rules(index, sources, refresh=False):
            with index.db:
                index.db.execute("INSERT OR REPLACE INTO rule_metadata VALUES ('rule_count', '4')")
            return {"rule_count": "4"}

        output, progress = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(output), contextlib.redirect_stderr(progress),
            patch("mtg_agentic.scryfall.ScryfallClient.download_oracle_cards", cards),
            patch("mtg_agentic.scryfall.ScryfallClient.download_oracle_tags",
                  lambda client, store, force=False: {"tag_count": store.import_tags(TAGS)}),
            patch("mtg_agentic.scryfall.ScryfallClient.download_rulings",
                  lambda client, store, force=False: {"ruling_count": store.import_rulings(RULINGS)}),
            patch("mtg_agentic.rules.RulesStore.download_rules", rules),
            patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("No real HTTP")),
        ):
            status = main(["--data-dir", str(self.path), "setup", "--embedding-backend", "hash",
                           *args])
        return status, json.loads(output.getvalue()), progress.getvalue().splitlines()

    def test_setup_downloads_only_what_is_missing_and_logs_every_stage(self):
        status, result, lines = self.run_setup("--skip-combos")
        self.assertEqual(status, 0)
        self.assertEqual(result["steps"], {
            "Card data": "77 cards", "Oracle tags": "1 tags", "Rulings": "1 rulings",
            "Commander Spellbook": "skipped: --skip-combos", "Rules index": "4 rules",
            "Embedding model": "skipped: the hash backend needs no download",
            "Card index": "skipped: the hash backend embeds no cards",
            "Goldfish engine": f"skipped: {NO_RUST}",
        })
        stages = [line for line in lines if line.startswith("setup: ")]
        self.assertEqual(stages[:2], ["setup: 1/8 Card data: downloading",
                                      "setup: 1/8 Card data: 77 cards"])
        self.assertEqual(len(stages), 16)  # Each step starts, then reports.
        # A second run finds the downloads here; the skips say so again.
        status, again, _ = self.run_setup("--skip-combos")
        self.assertEqual([k for k, v in again["steps"].items() if v != bootstrap.ALREADY],
                         ["Commander Spellbook", "Embedding model", "Card index",
                          "Goldfish engine"])

    def test_setup_builds_the_goldfish_engine_when_missing_or_refreshed(self):
        with (
            patch("mtg_agentic.__main__.build_problem", return_value=None),
            patch("mtg_agentic.__main__.engine_problem",
                  return_value="the goldfish engine isn't built"),
            patch("mtg_agentic.__main__.build_engine", return_value="built in 21s") as build,
        ):
            steps = self.run_setup("--skip-combos")[1]["steps"]
        self.assertEqual(steps["Goldfish engine"], "built in 21s")
        build.assert_called_once()
        with (
            patch("mtg_agentic.__main__.build_problem", return_value=None),
            patch("mtg_agentic.__main__.engine_problem", return_value=None),
            patch("mtg_agentic.__main__.build_engine", return_value="built in 1s") as build,
        ):
            self.assertEqual(self.run_setup("--skip-combos")[1]["steps"]["Goldfish engine"],
                             bootstrap.ALREADY)
            build.assert_not_called()
            refreshed = self.run_setup("--skip-combos", "--refresh")[1]["steps"]
        self.assertEqual(refreshed["Goldfish engine"], "built in 1s")

    def test_status_counts_the_engine_done_when_built_or_when_rust_is_missing(self):
        for built, buildable, ready in ((None, None, True), ("isn't built", None, False),
                                        ("isn't built", NO_RUST, True)):
            with (
                patch("mtg_agentic.goldfish.engine_problem", return_value=built),
                patch("mtg_agentic.goldfish.build_problem", return_value=buildable),
            ):
                self.assertIs(GOLDFISH_READY(), ready)

    def test_the_model_is_skipped_when_this_python_cannot_use_it(self):
        # The EmbeddingGemma rules index is already here, so no torch import is needed.
        with contextlib.closing(sqlite3.connect(self.path / bootstrap.RULES_INDEX)) as db, db:
            db.execute("CREATE TABLE rule_metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO rule_metadata VALUES ('rule_count', '4')")
        with (
            patch.object(bootstrap, "model_cached", return_value=False),
            patch.object(bootstrap, "embedding_problem", return_value="transformers 4.57 is too old."),
            patch.object(bootstrap, "fetch_model") as fetch,
        ):
            result = self.run_setup("--skip-combos", "--embedding-backend", "embeddinggemma")[1]
        fetch.assert_not_called()
        self.assertEqual(result["steps"]["Embedding model"],
                         "skipped: transformers 4.57 is too old.")


if __name__ == "__main__":
    unittest.main()
