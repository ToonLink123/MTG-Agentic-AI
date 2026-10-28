import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.cards import validation_card
from mtg_agentic.oracle import OracleStore
from mtg_agentic.rules import HashEmbedder, OllamaEmbedder, RulesStore, chunk_rules
from mtg_agentic.validator import can_be_commander
from tests.fixtures import card
from tests.pipeline_fixtures import JODAH, FakeModel, records

RULES = """Magic: The Gathering Comprehensive Rules
These rules are effective as of September 25, 2026.
Contents
100. General
701. Keyword Actions

100. General
100.1. These Magic rules apply to any Magic game with two or more players.
Example: An example that belongs to the first rule.
100.1a A two-player game is a game that begins with only two players.
701. Keyword Actions
701.34. Proliferate
701.34a To proliferate means to choose any number of permanents and/or players that have a counter, then give each one additional counter of each kind that permanent or player already has.
Glossary
This glossary must not get appended to the final rule.
"""
QUOTE = "To proliferate means to choose any number of permanents and/or players that have a counter"


def cited(status="supported", id_="CR 701.34a", quote=QUOTE):
    return {
        "status": status,
        "claims": [
            {
                "claim": "Proliferate requires an existing counter.",
                "citations": [{"id": id_, "quote": quote}],
            }
        ],
    }


class RulesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.rules = RulesStore(self.path / "rules.sqlite3", HashEmbedder())
        self.addCleanup(self.store.close)
        self.addCleanup(self.rules.close)
        self.store.import_cards(records())
        self.client = Mock()
        self.client.get_rulings.return_value = []
        self.rules.import_rules(
            RULES, source="https://example.invalid/rules.txt", fetched_at="today"
        )

    def test_chunks_skip_toc_keep_subrules_and_stop_at_glossary(self):
        date, chunks = chunk_rules(RULES)
        self.assertEqual(date, "September 25, 2026.")
        self.assertEqual(len(chunks), 4)
        self.assertIn("Example:", chunks["100.1"])
        self.assertNotIn("glossary", chunks["701.34a"].lower())

    def test_bad_rule_import_preserves_previous_index(self):
        before = self.rules.metadata
        with self.assertRaises(ValueError):
            self.rules.import_rules("not rules", source="bad", fetched_at="tomorrow")
        self.assertEqual(self.rules.metadata, before)

    def test_download_resolves_space_in_official_text_link_and_reuses_index(self):
        cache = Mock()
        cache.fetch.side_effect = [
            (
                b'<a href="https://media.wizards.com/2026/downloads/MagicCompRules 20260925.txt">TXT</a>',
                "today",
            ),
            (RULES.encode(), "today"),
        ]
        self.rules.download_rules(cache)
        self.assertIn("%20", cache.fetch.call_args.args[0])
        self.assertEqual(self.rules.metadata["rule_count"], "4")

    def test_ingest_every_card_including_empty_rulings_and_exclude_nonpublisher_comments(self):
        self.client.get_rulings.side_effect = [
            [],
            [
                {"source": "wotc", "published_at": "2024-01-01", "comment": QUOTE},
                {"source": "user", "published_at": "2024-01-01", "comment": "untrusted"},
            ],
        ]
        result = self.rules.ingest_deck(
            [JODAH["name"], "Spell 0", "Spell 0"], self.store, self.client
        )
        self.assertEqual(result["count"], 2)
        self.assertEqual(self.client.get_rulings.call_count, 2)
        docs = self.rules.retrieve(["Spell 0"], "proliferate counter")
        self.assertTrue(any(d["kind"] == "ruling" for d in docs))
        self.assertFalse(any(d["text"] == "untrusted" for d in docs))

    def test_unchanged_rule_text_does_not_reembed_after_http_refresh(self):
        cache = Mock()
        cache.fetch.side_effect = [
            (b'<a href="https://media.wizards.com/current/MagicCompRules.txt">TXT</a>', "new"),
            (RULES.encode(), "new"),
        ]
        with unittest.mock.patch.object(self.rules, "_embed_documents") as embed:
            self.rules.download_rules(cache)
            embed.assert_not_called()
        self.assertEqual(self.rules.metadata["fetched_at"], "new")

    def test_reingestion_of_same_rulings_does_not_reembed(self):
        embedder = Mock(wraps=self.rules.embedder)
        self.rules.embedder = embedder
        embedder.identity = HashEmbedder.identity
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        calls = embedder.embed.call_count
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        self.assertEqual(embedder.embed.call_count, calls)

    def test_retrieval_requires_rules_and_synchronized_cards(self):
        with self.assertRaises(ValueError):
            self.rules.retrieve(["Spell 0"], "Counters?")
        with RulesStore(self.path / "empty.sqlite3", HashEmbedder()) as empty:
            with self.assertRaises(ValueError):
                empty.retrieve([], "Counters?")

    def test_valid_citations_are_returned_with_dates_sources_and_actual_evidence(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        result = self.rules.check_interaction(
            [JODAH["name"]], "Proliferate counters?", FakeModel(cited())
        )
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["errors"], [])
        self.assertTrue(all("published" in d and "source" in d for d in result["evidence"]))

    def test_proliferate_definition_cannot_be_crowded_out_by_long_cost_questions(self):
        distractions = "\n".join(
            f"702.{n}a Casting from hand uses alternate costs, replacement effects, "
            "counters, battlefield conditions, and zones, exactly as stated."
            for n in range(30)
        )
        self.rules.import_rules(
            RULES.replace("Glossary", distractions + "\nGlossary"),
            source="https://example.invalid/rules.txt", fetched_at="today",
        )
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        evidence = self.rules.retrieve(
            [JODAH["name"]],
            "Casting from hand uses alternate costs, replacement effects, counters, "
            "battlefield conditions, and zones. Can proliferate add an existing counter?",
            limit=4,
        )
        self.assertEqual(len(evidence), 4)
        self.assertIn("CR 701.34a", {e["id"] for e in evidence})

    def test_unknown_rule_nonverbatim_quote_and_oracle_only_claims_are_refused(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        for response in [
            cited(id_="CR 999.9"),
            cited(quote="An invented quote that is not present in these rules."),
            cited(
                id_="Oracle " + JODAH["name"],
                quote="You may pay {W}{U}{B}{R}{G} rather than pay the mana cost",
            ),
        ]:
            with self.subTest(response=response):
                result = self.rules.check_interaction(
                    [JODAH["name"]], "Counters?", FakeModel(response)
                )
                self.assertEqual(result["status"], "unsupported")
                self.assertEqual(result["claims"], [])
                self.assertTrue(result["errors"])

    def test_no_cited_claim_is_not_a_supported_conclusion(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        result = self.rules.check_interaction(
            [JODAH["name"]], "Counters?", FakeModel({"status": "supported", "claims": []})
        )
        self.assertEqual(result["status"], "unsupported")

    def test_unsupported_model_claims_are_not_echoed_as_facts(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        result = self.rules.check_interaction(
            [JODAH["name"]], "Counters?", FakeModel(cited(status="unsupported"))
        )
        self.assertEqual(result["claims"], [])

    def test_contradicted_cited_interactions_remain_contradicted(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        result = self.rules.check_interaction(
            [JODAH["name"]], "Counters?", FakeModel(cited(status="contradicted"))
        )
        self.assertEqual(result["status"], "contradicted")

    def test_embedding_identity_and_invalid_vectors_fail_before_overwriting(self):
        bad = Mock()
        bad.identity = "different"
        with self.assertRaises(ValueError):
            RulesStore(self.path / "rules.sqlite3", bad)
        self.rules.embedder = Mock()
        self.rules.embedder.embed.return_value = [[float("nan")]]
        before = self.rules.metadata
        with self.assertRaises(ValueError):
            self.rules.import_rules(RULES, source="new", fetched_at="new")
        self.assertEqual(before, self.rules.metadata)

    def test_ollama_embeddings_use_local_embed_endpoint_without_truncation(self):
        embedder = OllamaEmbedder()
        embedder._post = Mock(return_value={"embeddings": [[1, 2]]})
        self.assertEqual(embedder.embed(["rule"]), [[1, 2]])
        embedder._post.assert_called_once_with(
            "/api/embed", {"model": "nomic-embed-text", "input": ["rule"], "truncate": False}
        )

    def test_current_legendary_vehicle_and_spacecraft_eligibility(self):
        for types, extra, expected in [
            ("Legendary Artifact — Vehicle", {}, True),
            ("Artifact — Vehicle", {}, False),
            ("Legendary Artifact — Spacecraft", {"power": "5", "toughness": "5"}, True),
            ("Legendary Artifact — Spacecraft", {}, False),
        ]:
            with self.subTest(types=types, extra=extra):
                self.assertEqual(
                    can_be_commander(validation_card(card("Ship", types=types, **extra))), expected
                )
