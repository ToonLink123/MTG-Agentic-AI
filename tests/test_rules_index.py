import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from mtg_agentic.rules import HashEmbedder, RulesStore
from mtg_agentic.rules_index import PrebuiltIndex, index_dir
from tests.test_rules import RULES

REPO = "someone/cr-vectors"
LINK = b'<a href="https://media.wizards.com/2026/downloads/MagicCompRules 20260925.txt">TXT</a>'


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        self.documents = 0

    def embed(self, texts, *, titles=None, query=False):
        if not query:
            self.documents += len(texts)
        return super().embed(texts, titles=titles, query=query)


class FakeHub:
    """Serves an exported directory the way huggingface.co resolves dataset files."""

    def __init__(self, root):
        self.root, self.paths = Path(root), []

    def __call__(self, request, timeout):
        path = request.full_url.removeprefix(f"https://huggingface.co/datasets/{REPO}/resolve/main/")
        self.paths.append(path)
        if not (self.root / path).is_file():
            raise HTTPError(request.full_url, 404, "Not Found", {}, None)
        return io.BytesIO((self.root / path).read_bytes())


def rules_cache():
    cache = Mock()
    cache.fetch.side_effect = lambda url, refresh=False: (LINK if "rules" in url else RULES.encode(), "today")
    return cache


class PrebuiltIndexTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        with RulesStore(self.path / "built.sqlite3", HashEmbedder()) as built:
            self.export = built.export_index(rules_cache(), self.path / "hub")
            self.built = dict(built.db.execute("SELECT id, vector FROM documents"))
        self.folder = self.path / "hub" / index_dir(HashEmbedder.identity, self.export["rules_sha256"])

    def fresh(self, name, opener):
        embedder = CountingEmbedder()
        store = RulesStore(self.path / name, embedder, prebuilt=PrebuiltIndex(REPO, opener=opener))
        self.addCleanup(store.close)
        return store, embedder

    def test_export_writes_rule_numbers_and_vectors_but_no_rule_text(self):
        manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        data = (self.folder / "vectors.f32").read_bytes()
        self.assertEqual(manifest["ids"], ["CR 100.1", "CR 100.1a", "CR 701.34", "CR 701.34a"])
        self.assertEqual(len(data), manifest["count"] * manifest["dimension"] * 4)
        self.assertEqual(manifest["vectors_sha256"], hashlib.sha256(data).hexdigest())
        self.assertNotIn("roliferate", (self.folder / "manifest.json").read_text(encoding="utf-8"))

    def test_a_matching_published_index_replaces_the_embedding_pass(self):
        hub = FakeHub(self.path / "hub")
        store, embedder = self.fresh("fresh.sqlite3", hub)
        metadata = store.download_rules(rules_cache())
        self.assertEqual(embedder.documents, 0)
        self.assertEqual(metadata["vectors"], f"hf:{REPO}")
        self.assertEqual(dict(store.db.execute("SELECT id, vector FROM documents")), self.built)
        self.assertEqual(store.retrieve([], "How does proliferate work?")[0]["id"], "CR 701.34a")
        # The index is kept locally, so the next run reads SQLite and asks the Hub nothing.
        store.download_rules(rules_cache())
        self.assertEqual(len(hub.paths), 2)

    def test_any_mismatch_or_outage_falls_back_to_local_embeddings(self):
        manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        cases = {
            "different inputs": lambda: manifest | {"inputs_sha256": "0" * 64},
            "different model": lambda: manifest | {"embedding": "other-model"},
        }
        for name, changed in cases.items():
            with self.subTest(name):
                (self.folder / "manifest.json").write_text(json.dumps(changed()), encoding="utf-8")
                store, embedder = self.fresh(f"{name}.sqlite3", FakeHub(self.path / "hub"))
                self.assertEqual(store.download_rules(rules_cache())["vectors"], "local")
                self.assertEqual(embedder.documents, 4)
        (self.folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        vectors = (self.folder / "vectors.f32").read_bytes()
        (self.folder / "vectors.f32").write_bytes(vectors[:-1] + bytes([vectors[-1] ^ 1]))
        store, embedder = self.fresh("corrupt.sqlite3", FakeHub(self.path / "hub"))
        self.assertEqual(store.download_rules(rules_cache())["vectors"], "local")
        self.assertEqual(embedder.documents, 4)

        def offline(request, timeout):
            raise URLError("no network")

        store, embedder = self.fresh("offline.sqlite3", offline)
        self.assertEqual(store.download_rules(rules_cache())["vectors"], "local")
        self.assertEqual(embedder.documents, 4)

    def test_export_publishes_only_vectors_embedded_locally(self):
        store, embedder = self.fresh("fresh.sqlite3", FakeHub(self.path / "hub"))
        store.download_rules(rules_cache())
        with self.assertRaises(ValueError):
            store.export_index(rules_cache(), self.path / "again")
        with RulesStore(self.path / "fresh.sqlite3", embedder) as local:
            local.export_index(rules_cache(), self.path / "again")
            self.assertEqual(local.metadata["vectors"], "local")
        self.assertEqual(embedder.documents, 4)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "needs the embeddings extra")
class LazyModelTests(unittest.TestCase):
    def test_embeddinggemma_loads_weights_only_when_it_first_embeds(self):
        import transformers

        from mtg_agentic.embeddings import EmbeddingGemmaEmbedder

        with patch.object(transformers.AutoModel, "from_pretrained") as load:
            embedder = EmbeddingGemmaEmbedder(device="cpu")
        load.assert_not_called()
        self.assertTrue(embedder.identity.startswith("hf:google/embeddinggemma-2@"))


if __name__ == "__main__":
    unittest.main()
