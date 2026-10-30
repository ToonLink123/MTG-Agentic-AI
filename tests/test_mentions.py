import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mtg_agentic.mentions import find_cards, lookup, name_index, words
from mtg_agentic.oracle import OracleStore
from mtg_agentic.scryfall import CardNotFound
from tests.fixtures import card

FABLE = card(
    "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki", colors="R",
    types="Enchantment — Saga // Enchantment Creature — Goblin Shaman",
    card_faces=[
        {"name": "Fable of the Mirror-Breaker", "type_line": "Enchantment — Saga",
         "oracle_text": "", "mana_cost": "{2}{R}"},
        {"name": "Reflection of Kiki-Jiki", "type_line": "Enchantment Creature — Goblin Shaman",
         "oracle_text": "", "mana_cost": ""},
    ],
    layout="transform",
)
CARDS = [
    FABLE,
    card("Village Bell-Ringer", colors="W", types="Creature — Human Scout",
         text="Flash. When this creature enters, untap all creatures you control."),
    card("Jodah, Archmage Eternal", colors="WUBRG", types="Legendary Creature — Human Wizard"),
    card("Jodah, the Unifier", colors="WUBRG", types="Legendary Creature — Human Wizard"),
    card("Atraxa, Praetors' Voice", colors="WUBG", types="Legendary Creature — Phyrexian Angel"),
    card("Flash", colors="U", types="Instant", text="You may put a creature card from your hand "
                                                     "onto the battlefield."),
    # An Alchemy rebalance is not legal in Commander, so it never answers for the real card.
    card("A-Village Bell-Ringer", colors="W", types="Creature — Human Scout", legal="not_legal"),
    # Flash is a keyword on many cards, so the word alone is not the card Flash.
    *(card(f"Flash Filler {n}", text="Flash") for n in range(45)),
]


class MentionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = OracleStore(Path(temp.name) / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(CARDS)

    def test_words_ignore_case_hyphens_apostrophes_and_accents(self):
        self.assertEqual(words("Atraxa, Praetors' Voice"), ("atraxa", "praetors", "voice"))
        self.assertEqual(words("Village Bell-Ringer"), ("village", "bell", "ringer"))
        self.assertEqual(words("Lim-Dûl's Vault"), ("lim", "duls", "vault"))

    def test_questions_name_cards_in_plain_words(self):
        found, unsettled = find_cards(
            "can fable of the mirror breaker loop with a village bell ringer and atraxa?",
            self.store,
        )
        self.assertEqual(found, [FABLE["name"], "Village Bell-Ringer", "Atraxa, Praetors' Voice"])
        self.assertEqual(unsettled, [])
        # Either face of a double-faced card names it.
        found, _ = find_cards("does reflection of kiki-jiki untap?", self.store)
        self.assertEqual(found, [FABLE["name"]])

    def test_a_rules_word_is_not_the_card_with_that_name(self):
        found, unsettled = find_cards("does flash let me respond?", self.store)
        self.assertEqual((found, unsettled), ([], []))

    def test_a_short_name_for_several_cards_is_reported_unless_the_deck_settles_it(self):
        found, unsettled = find_cards("can jodah cast it?", self.store)
        self.assertEqual(found, [])
        self.assertEqual(unsettled, [{"name": "jodah",
                                      "options": ["Jodah, Archmage Eternal", "Jodah, the Unifier"]}])
        found, unsettled = find_cards(
            "can jodah cast it?", self.store, prefer=["Jodah, the Unifier"]
        )
        self.assertEqual((found, unsettled), (["Jodah, the Unifier"], []))

    def test_bracketed_names_are_found_like_a_commander(self):
        index = name_index(self.store)
        self.assertEqual(lookup("Village Bell-Ringer", self.store, index=index)["name"],
                         "Village Bell-Ringer")
        self.assertEqual(lookup("village bell ringer", self.store, index=index)["name"],
                         "Village Bell-Ringer")
        client = Mock()
        client.get_card.return_value = {"name": "Village Bell-Ringer"}
        self.assertEqual(lookup("vilage bel ringer", self.store, client, index=index)["name"],
                         "Village Bell-Ringer")
        client.get_card.assert_called_once_with("vilage bel ringer")
        client.get_card.side_effect = CardNotFound("No card", status=404)
        self.assertIsNone(lookup("nothing like it", self.store, client, index=index))
        self.assertIsNone(lookup("jodah", self.store, index=index))
