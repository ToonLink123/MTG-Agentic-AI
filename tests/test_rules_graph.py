import unittest

from mtg_agentic.rules import chunk_rules
from mtg_agentic.rules_graph import RulesGraph, build_graph, headline, in_family, stems

RULES = """Magic: The Gathering Comprehensive Rules
These rules are effective as of September 25, 2026.
Contents
100. General
Glossary
Credits

100. General
100.1. These Magic rules apply to any Magic game with two or more players.
118. Costs
118.9. Some spells have alternative costs. See rule 601.2f.
118.9a Only one alternative cost can be applied to any one spell as it's being cast.
601. Casting Spells
601.2. To cast a spell is to take it from where it is, put it on the stack, and pay its costs.
601.2f The player determines the total amount to pay.
701. Keyword Actions
701.34. Proliferate
701.34a To proliferate means to give each chosen permanent or player another counter.
702. Keyword Abilities
702.1. Some effects let a creature proliferate after an alternative cost is paid.
Glossary
Alternative Cost
A cost a spell may have that its controller may pay rather than its mana cost. See rule 118.9.
\u00a0
Proliferate (Keyword Action)
To give each chosen permanent or player another counter. See rule 701.34, "Proliferate."
\u00a0
Credits
Thanks to everyone.
"""


class RulesGraphTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_rules(RULES)[1]
        self.rows = build_graph(RULES, self.chunks)
        self.edges = {(s, t, kind) for s, t, kind, _ in self.rows.edges}

    def test_stems_headlines_and_families(self):
        self.assertEqual(stems("Proliferating the counters"), ["proliferat", "counter"])
        self.assertEqual(stems("Proliferate"), stems("proliferated"))
        self.assertEqual(stems("alternate cost"), stems("alternative costs"))
        self.assertEqual(headline(self.chunks["701.34"]), "Proliferate")
        self.assertIsNone(headline(self.chunks["100.1"]))
        self.assertTrue(in_family("701.34a", "701.34"))
        self.assertTrue(in_family("701.34a", "701"))
        self.assertFalse(in_family("701.34", "701.3"))

    def test_references_glossary_terms_and_hierarchy_become_edges(self):
        self.assertIn(("118.9", "601.2f", "reference"), self.edges)
        self.assertIn(("118.9a", "118.9", "parent"), self.edges)
        self.assertIn(("118.9", "118.9a", "child"), self.edges)
        self.assertIn(("118.9", "118", "parent"), self.edges)
        # Using a glossary term links a rule to the rule that defines it.
        self.assertIn(("702.1", "701.34", "term"), self.edges)
        self.assertIn(("702.1", "118.9", "term"), self.edges)
        # A rule is never linked to its own family by the terms it defines.
        self.assertNotIn(("118.9a", "118.9", "term"), self.edges)
        terms = {(term, node): origin for term, node, _, origin in self.rows.terms}
        self.assertEqual(terms[("proliferat", "701.34")], "glossary")
        self.assertEqual(terms[("alternativ cost", "118.9")], "glossary")
        self.assertEqual(terms[("cast spell", "601")], "section")
        self.assertNotIn("glossary", " ".join(self.chunks.values()).lower())

    def test_titles_give_subrules_their_section_and_heading(self):
        self.assertEqual(self.rows.titles["701.34a"], "Keyword Actions > Proliferate")
        self.assertEqual(self.rows.titles["601.2f"], "Casting Spells")

    def test_personalized_pagerank_follows_references_but_not_unrelated_rules(self):
        graph = RulesGraph(
            [(s, t, w) for s, t, _, w in self.rows.edges],
            [(term, node, df) for term, node, df, _ in self.rows.terms],
            len(self.chunks),
        )
        walk = graph.pagerank({"118.9": 1.0})
        self.assertGreater(walk.get("601.2f", 0), 0)
        self.assertEqual(walk.get("100.1", 0), 0)
        self.assertAlmostEqual(sum(walk.values()), 1.0, places=6)
        self.assertEqual(graph.term_targets("Can I proliferate?"), {"701.34": graph.idf(("proliferat",))})
        self.assertEqual(graph.pagerank({}), {})


if __name__ == "__main__":
    unittest.main()
