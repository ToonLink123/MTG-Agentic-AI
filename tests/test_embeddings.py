import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.__main__ import _embedder, _embedding_options, _rules_path
from mtg_agentic.embeddings import (
    EMBEDDINGGEMMA_MODEL,
    EMBEDDINGGEMMA_REVISION,
    EmbeddingGemmaEmbedder,
    format_inputs,
)
from mtg_agentic.rules import HashEmbedder


class EmbeddingTests(unittest.TestCase):
    def test_model_card_prefixes_for_documents_and_questions(self):
        self.assertEqual(
            format_inputs(["Rule text."], titles=["Keyword Actions > Proliferate"]),
            ["title: Keyword Actions > Proliferate | text: Rule text."],
        )
        self.assertEqual(format_inputs(["Rule text."]), ["title: none | text: Rule text."])
        self.assertEqual(
            format_inputs(["Can I proliferate?"], query=True),
            ["task: question answering | query: Can I proliferate?"],
        )
        with self.assertRaises(ValueError):
            format_inputs(["a", "b"], titles=["only one"])

    def test_invalid_dimensions_and_batches_fail_before_loading_anything(self):
        with self.assertRaises(ValueError):
            EmbeddingGemmaEmbedder(dimension=300)
        with self.assertRaises(ValueError):
            EmbeddingGemmaEmbedder(batch_size=0)

    def test_cli_defaults_to_pinned_embeddinggemma_with_its_own_index(self):
        parser = argparse.ArgumentParser()
        _embedding_options(parser)
        args = parser.parse_args([])
        args.data_dir = Path("data")
        self.assertEqual(args.embedding_backend, "embeddinggemma")
        self.assertEqual(_rules_path(args), Path("data") / "rules-embeddinggemma-2.sqlite3")
        with patch("mtg_agentic.embeddings.EmbeddingGemmaEmbedder") as model:
            _embedder(args)
        model.assert_called_once_with(
            EMBEDDINGGEMMA_MODEL, revision=EMBEDDINGGEMMA_REVISION, device=None
        )
        args = parser.parse_args(["--embedding-backend", "hash"])
        args.data_dir = Path("data")
        self.assertIsInstance(_embedder(args), HashEmbedder)
        self.assertEqual(_rules_path(args), Path("data") / "rules-hash.sqlite3")


if __name__ == "__main__":
    unittest.main()
