import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.cards import validation_card
from mtg_agentic.oracle import OracleStore
from mtg_agentic.rules import (
    HashEmbedder,
    OllamaEmbedder,
    RulesStore,
    _interleave,
    _subquestions,
    chunk_rules,
)
from mtg_agentic.rules_graph import GRAPH_VERSION
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
GRAPH_RULES = (
    RULES.split("Glossary")[0]
    + "\n".join(f"200.{n}. Distraction {n} says nothing useful at all." for n in range(1, 21))
    + """
118. Costs
118.9. Alternative costs are paid instead of the mana cost. See rule 601.2f.
601. Casting Spells
601.2f The player determines the total amount to pay.
Glossary
Alternative Cost
A cost paid rather than the mana cost. See rule 118.9.
"""
)


class TopicEmbedder:
    """Deterministic vectors: one axis for alternative costs, one for everything else."""

    identity = "topic-test:2"

    def embed(self, texts, *, titles=None, query=False):
        return [[1.0, 0.0] if "alternative" in t.lower() else [0.0, 1.0] for t in texts]


def unsettled(lookup):
    return {"status": "unsupported", "claims": [], "lookup": lookup, "conditions": [],
            "answer": "Not established yet."}


def cited(status="supported", id_="CR 701.34a", quote=QUOTE):
    return {
        "status": status,
        "lookup": "",
        "conditions": [],
        "answer": "Yes: proliferate needs a counter already there.",
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

    def test_ingest_reads_rulings_from_the_local_bulk_file(self):
        self.store.import_rulings([
            {"object": "ruling", "oracle_id": "Spell 0", "source": "wotc",
             "published_at": "2024-01-01", "comment": QUOTE},
            {"object": "ruling", "oracle_id": "Spell 0", "source": "scryfall",
             "published_at": "2024-02-01", "comment": "A Scryfall note, not a ruling."},
        ])
        result = self.rules.ingest_deck([JODAH["name"], "Spell 0"], self.store, self.client)
        self.assertEqual(result["count"], 2)
        self.client.get_rulings.assert_not_called()
        self.client.download_rulings.assert_not_called()
        rulings = [d for d in self.rules.retrieve(["Spell 0"], "proliferate counter")
                   if d["kind"] == "ruling"]
        self.assertEqual([d["text"] for d in rulings], [QUOTE])

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
            [JODAH["name"]], "Counters?", FakeModel({"status": "supported", "claims": [], "lookup": "", "conditions": [],
                                         "answer": "Yes."})
        )
        self.assertEqual(result["status"], "unsupported")

    def test_unsupported_model_claims_are_not_echoed_as_facts(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        result = self.rules.check_interaction(
            [JODAH["name"]], "Counters?", FakeModel(cited(status="unsupported"))
        )
        self.assertEqual(result["claims"], [])

    def test_quotes_match_whatever_the_quote_marks_and_failed_steps_say_why(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        evidence = [{"id": "CR 1.1", "kind": "rule",
                     "text": "1.1. A player can\u2019t win \u2014 unless it\u2019s their turn."}]
        typed = {"claim": "Typed with straight quotes.", "citations": [
            {"id": "CR 1.1", "quote": "A player can't win - unless it's their turn."}]}
        invented = {"claim": "Invented.", "citations": [
            {"id": "CR 1.1", "quote": "A player can always win on any turn at all."}]}
        answer = {**cited(), "claims": [typed, invented]}
        result = self.rules.judge_interaction([JODAH["name"]], "Q?", evidence, answer)
        self.assertEqual(result["established"], [typed])
        self.assertEqual(result["rejected"],
                         [{**invented, "problems": ["The quote from CR 1.1 is not in its text"]}])
        # The judge's direct answer and open conditions are kept even when unsupported.
        self.assertEqual(result["answer"], cited()["answer"])
        self.assertEqual(result["conditions"], [])

    def test_card_evidence_shows_every_face_with_its_type_line_and_stats(self):
        from mtg_agentic.rules import card_document

        fable = {
            "name": "Fable // Reflection", "type_line": "Enchantment // Enchantment Creature",
            "card_faces": [
                {"name": "Fable", "mana_cost": "{2}{R}", "type_line": "Enchantment \u2014 Saga",
                 "oracle_text": "III \u2014 Transform it.", "power": "", "toughness": ""},
                {"name": "Reflection", "mana_cost": "", "type_line":
                 "Enchantment Creature \u2014 Goblin Shaman", "oracle_text": "{1}, {T}: Copy.",
                 "power": "2", "toughness": "2"},
            ],
        }
        self.assertEqual(card_document(fable), (
            "Fable {2}{R}\nEnchantment \u2014 Saga\nIII \u2014 Transform it.\n//\n"
            "Reflection\nEnchantment Creature \u2014 Goblin Shaman 2/2\n{1}, {T}: Copy."
        ))

    def test_an_unsupported_answer_shows_its_checked_steps_and_what_is_missing(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        evidence = self.rules.retrieve([JODAH["name"]], "Counters?")
        good = cited()["claims"][0]
        invented = {"claim": "Something else.",
                    "citations": [{"id": "CR 701.34a", "quote": "An invented quote, not in the rules."}]}
        answer = {"status": "unsupported", "claims": [good, invented],
                  "lookup": "mana abilities of creatures"}
        result = self.rules.judge_interaction([JODAH["name"]], "Counters?", evidence, answer)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["claims"], [])  # Never the answer itself.
        self.assertEqual(result["established"], [good])
        self.assertEqual(result["missing"], "mana abilities of creatures")
        # A "supported" answer with one bad citation is unsupported, keeping its good steps.
        answer = {"status": "supported", "claims": [good, invented], "lookup": ""}
        result = self.rules.judge_interaction([JODAH["name"]], "Counters?", evidence, answer)
        self.assertEqual((result["status"], result["established"]), ("unsupported", [good]))
        self.assertNotIn("missing", result)

    def test_contradicted_cited_interactions_remain_contradicted(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        result = self.rules.check_interaction(
            [JODAH["name"]], "Counters?", FakeModel(cited(status="contradicted"))
        )
        self.assertEqual(result["status"], "contradicted")

    def test_an_unsettled_question_gets_more_looks_until_it_settles(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        question = "Does a two-player game begin with only two players?"
        # Two items: Jodah's text and one rule, so the proliferate rule is missing at first.
        first = self.rules.retrieve([JODAH["name"]], question, limit=2)
        self.assertNotIn("CR 701.34a", [e["id"] for e in first])
        model = FakeModel(unsettled("proliferate"), unsettled("magic rules games"), cited())
        result = self.rules.check_interaction([JODAH["name"]], question, model, limit=2)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["looks"], [{"lookup": "proliferate", "added": ["CR 701.34a"]},
                                           {"lookup": "magic rules games", "added": ["CR 100.1"]}])
        self.assertIn("CR 701.34a", [e["id"] for e in result["evidence"]])
        self.assertEqual(len(model.calls), 3)

    def test_looks_stop_at_a_repeat_nothing_new_or_the_last_pass(self):
        self.rules.ingest_deck([JODAH["name"]], self.store, self.client)
        question = "Does a two-player game begin with only two players?"

        def ask(*answers, **options):
            model = FakeModel(*answers)
            return self.rules.check_interaction([JODAH["name"]], question, model, **options), model

        # Asking for the same thing again ends the looks.
        result, model = ask(unsettled("proliferate"), unsettled("proliferate"), limit=2)
        self.assertEqual((len(model.calls), len(result["looks"])), (2, 1))
        self.assertEqual(result["missing"], "proliferate")
        # Nothing new to find (18 items already hold every rule), or no lookup named.
        for answer in (unsettled("two-player game"), unsettled("")):
            with self.subTest(lookup=answer["lookup"]):
                result, model = ask(answer)
                self.assertEqual((result["status"], len(model.calls)), ("unsupported", 1))
                self.assertNotIn("looks", result)
        # The last pass is marked final; its lookup is what is still missing, never looked up.
        result, model = ask(unsettled("proliferate"), unsettled("magic rules games"),
                            limit=2, passes=2)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(json.loads(model.calls[-1][0][-1]["content"])["final"])
        self.assertEqual(result["missing"], "magic rules games")

    def test_graph_walk_reaches_a_referenced_rule_neither_vectors_nor_words_find(self):
        with RulesStore(self.path / "graph.sqlite3", TopicEmbedder()) as rules:
            rules.import_rules(GRAPH_RULES, source="https://example.invalid", fetched_at="today")
            ids = [d["id"] for d in rules.retrieve([], "How do alternative costs work?", limit=3)]
        # The glossary anchors the defining rule; its "see rule" reference pulls in 601.2f
        # ahead of distractors that tie with it on cosine and stem overlap.
        self.assertEqual(ids, ["CR 118.9", "CR 601.2f", "CR 100.1"])

    def test_card_text_lets_a_short_question_reach_the_rules_its_cards_use(self):
        self.store.add_verified_card(card(
            "Alt Spell", types="Instant",
            text="You may cast this spell for its alternative cost of {0}.",
        ))
        with RulesStore(self.path / "graph.sqlite3", TopicEmbedder()) as rules:
            rules.import_rules(GRAPH_RULES, source="https://example.invalid", fetched_at="today")
            rules.ingest_deck(["Alt Spell"], self.store, self.client)
            ids = {
                card_text: [d["id"] for d in rules.retrieve(
                    ["Alt Spell"], "Does this work?", limit=4, card_text=card_text
                )]
                for card_text in (False, True)
            }
        # The question names no rules term; the card's text names "alternative cost".
        self.assertNotIn("CR 118.9", ids[False])
        self.assertIn("CR 118.9", ids[True])
        self.assertEqual(ids[True][0], "Oracle Alt Spell")

    def test_old_index_gains_the_graph_on_download_without_reembedding(self):
        with self.rules.db:
            self.rules.db.execute("DELETE FROM rule_edges")
            self.rules.db.execute("DELETE FROM rule_terms")
            self.rules.db.execute("DELETE FROM rule_metadata WHERE key='graph_version'")
        self.rules._graph_cache = None
        with self.assertRaisesRegex(ValueError, "download-rules"):
            self.rules.retrieve([], "Proliferate?")
        cache = Mock()
        cache.fetch.side_effect = [
            (b'<a href="https://media.wizards.com/current/MagicCompRules.txt">TXT</a>', "new"),
            (RULES.encode(), "new"),
        ]
        with unittest.mock.patch.object(self.rules, "_embed_documents") as embed:
            metadata = self.rules.download_rules(cache)
            embed.assert_not_called()
        self.assertEqual(metadata["graph_version"], GRAPH_VERSION)
        ids = [d["id"] for d in self.rules.retrieve([], "Can I proliferate?", limit=2)]
        self.assertIn("CR 701.34a", ids)

    def test_multi_part_questions_rank_each_sentence_and_take_turns(self):
        self.assertEqual(_subquestions("Can proliferate add a counter?"), [])
        self.assertEqual(
            _subquestions("When does it enter with a counter? Can proliferate add another one?"),
            ["When does it enter with a counter?", "Can proliferate add another one?"],
        )
        a, b, c = (Mock(id=x) for x in "abc")
        self.assertEqual([d.id for d in _interleave([[a, b, c], [c, a, b]])], ["a", "c", "b"])

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
