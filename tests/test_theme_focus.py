import unittest
from types import SimpleNamespace

from mtg_agentic.assembly import assemble_nonlands, theme_priority
from mtg_agentic.cards import validation_card
from mtg_agentic.discovery import Candidate
from tests.fixtures import card
from tests.pipeline_fixtures import make_spec


class ThemeFocusTests(unittest.TestCase):
    def test_myojin_support_is_not_generic_tokens_or_plus_one_counters(self):
        spec = make_spec(win_conditions=["Myojin combat"])
        self.assertGreater(theme_priority(card("Engine", text="Proliferate."), spec), 0)
        self.assertGreater(theme_priority(card("Boon", text="Put a divinity counter on target creature."), spec), 0)
        self.assertLess(theme_priority(card("Tokens", text="If tokens would be created, create twice that many tokens instead."), spec), 0)
        self.assertLess(theme_priority(card("Scales", text="Put a +1/+1 counter on each creature."), spec), 0)
        self.assertLess(theme_priority(card(
            "Mondrak", text="If tokens would be created, create twice that many tokens instead. "
            "Sacrifice two other creatures: Put an indestructible counter on Mondrak.",
        ), spec), 0)

    def test_explicitly_requested_secondary_themes_are_not_penalized(self):
        spec = make_spec(themes=["Myojin tribal", "tokens", "+1/+1 counters"])
        self.assertEqual(theme_priority(card("Tokens", text="Create two tokens."), spec), 0)
        self.assertEqual(theme_priority(card("Scales", text="Put a +1/+1 counter on each creature."), spec), 0)

    def test_non_myojin_decks_are_unchanged(self):
        spec = make_spec(themes=["tokens"], mechanics=[], win_conditions=["combat"])
        self.assertEqual(theme_priority(card("Tokens", text="Create two tokens."), spec), 0)

    def test_padding_prefers_proliferate_even_when_model_ranks_tokens_first(self):
        spec = make_spec(win_conditions=["Myojin combat"])
        pool = SimpleNamespace(spec=spec, cards=[
            Candidate(validation_card(card(f"Generic {i}")), {"creatures"}) for i in range(63)
        ] + [
            Candidate(validation_card(card("Tokens", text="Create two tokens.")), {"theme"}),
            Candidate(validation_card(card("Engine", text="Proliferate.")), {"theme"}),
        ])
        result = assemble_nonlands(pool, {"theme": ["Tokens", "Engine"]})
        self.assertIn("Engine", result["cards"])
        self.assertNotIn("Tokens", result["cards"])
