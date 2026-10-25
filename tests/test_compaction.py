import unittest

from mtg_agentic.compaction import MIN_PART, Compactor


class CompactionTests(unittest.TestCase):
    def test_older_turns_shrink_first_and_the_newest_turn_stays_verbatim(self):
        older, newest = "a" * 4000 + "b" * 4000, "keep me"
        texts, short = Compactor().shrink([older, newest], 3000)
        self.assertEqual(short, 0)
        self.assertEqual(texts[1], newest)
        self.assertLessEqual(len(texts[0]), len(older) - 3000)
        # Middle elision keeps the first two thirds and the last third of what remains.
        head, tail = texts[0].split("[…")[0], texts[0].split("…]")[1]
        self.assertTrue(head.startswith("a") and tail.endswith("b"))
        self.assertAlmostEqual(len(head), 2 * len(tail), delta=2)

    def test_largest_older_turn_is_cut_first(self):
        texts, _ = Compactor().shrink(["s" * 3000, "L" * 9000, "new"], 2000)
        self.assertEqual(texts[0], "s" * 3000)
        self.assertIn("characters elided", texts[1])

    def test_low_yield_skips_older_turns_and_bounds_the_newest(self):
        texts, short = Compactor().shrink(["tiny", "n" * 9000], 2000)
        self.assertEqual(texts[0], "tiny")
        self.assertIn("characters elided", texts[1])
        self.assertEqual(short, 0)

    def test_parts_never_go_below_the_minimum_and_report_the_shortfall(self):
        texts, short = Compactor().shrink(["x" * 1500], 10_000)
        self.assertGreaterEqual(len(texts[0]), MIN_PART)
        self.assertGreater(short, 0)

    def test_cuts_are_reapplied_so_prefixes_stay_identical(self):
        compactor = Compactor()
        pool = "p" * 9000
        first, _ = compactor.shrink([pool, "repair 1"], 3000)
        self.assertEqual(compactor.reapply([pool, "repair 2"]), [first[0], "repair 2"])
