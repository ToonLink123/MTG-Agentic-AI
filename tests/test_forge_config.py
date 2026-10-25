import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic.forge import ForgeError
from mtg_agentic.forge_config import ForgeInstallation, OpponentPod, file_hash, read_dck


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.jar = self.path / "forge-gui-desktop-2.0.15-jar-with-dependencies.jar"
        self.jar.write_bytes(b"test-only jar")
        (self.path / "forge.profile.properties").write_text("userDir=./isolated-user\n")
        with patch(
            "mtg_agentic.forge_config.shutil.which", return_value=str(self.path / "java.exe")
        ):
            self.install = ForgeInstallation.inspect(self.path)
        self.deck = self.path / "precon.dck"
        self.deck.write_text(
            "[metadata]\nName=Test\n[Commander]\n1 Test Commander+|C15\n[Main]\n99 Plains|C15\n"
        )

    def test_profile_and_jvm_arguments_are_explicit(self):
        self.assertEqual(self.install.decks_dir, self.path / "isolated-user/decks/commander")
        self.assertEqual(self.install.command()[-1], "sim")
        self.assertIn("-Djava.awt.headless=true", self.install.command())
        self.assertEqual(self.install.release, "2.0.15")

    def test_precon_import_keeps_commander_separate_and_aggregates_prints(self):
        deck = read_dck(self.deck)
        self.assertEqual(deck["commander"], "Test Commander")
        self.assertEqual(deck["cards"], {"Plains": 99})

    def test_multiple_commanders_are_rejected(self):
        self.deck.write_text("[Commander]\n1 First\n1 Background\n[Main]\n98 Plains\n")
        with self.assertRaises(ForgeError):
            read_dck(self.deck)

    def test_pod_requires_three_pinned_opponents_and_expected_release(self):
        data = {
            "version": 1,
            "name": "fixed",
            "forge_release": "2.0.15",
            "opponents": [{"path": "precon.dck", "sha256": file_hash(self.deck)}] * 3,
        }
        path = self.path / "pod.json"
        path.write_text(json.dumps(data))
        pod = OpponentPod.load(path, self.install)
        self.assertEqual(len(pod.opponents), 3)
        self.assertTrue(pod.fingerprint)
        self.deck.write_text(self.deck.read_text() + "\n# edited\n")
        with self.assertRaises(ForgeError):
            OpponentPod.load(path, self.install)

    def test_java_properties_escaped_windows_paths(self):
        from mtg_agentic.forge_config import _properties

        path = self.path / "props"
        path.write_text(r"userDir=D:\\Forge\\User" + "\n")
        self.assertEqual(_properties(path)["userDir"], "D:\\Forge\\User")
