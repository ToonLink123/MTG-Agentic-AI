"""Goldfish: compiling Oracle text, building the Rust engine, and the games it plays."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mtg_agentic.__main__ import main
from mtg_agentic.goldfish import (
    CHECK,
    FAST,
    SLOW,
    TAPPED,
    UNTAPPED,
    build_engine,
    build_problem,
    compile_deck,
    engine_problem,
    play_games_rust,
    run_goldfish,
    summarize,
)
from mtg_agentic.oracle import OracleStore
from tests.fixtures import card

W, U, B, R, G, C = 1, 2, 4, 8, 16, 32
# Tests that play games need the engine setup builds; goldfish-rs has its own `cargo test`.
ENGINE_MISSING = engine_problem()


def stub_engine(deck, games, turns, seed, *, threads=0):
    """Counters shaped like the engine's, for tests about everything around the games: every
    hand keeps and nothing is ever reached."""
    never = [0] * turns + [games]
    return {
        "games": games, "turns": turns, "on_play": 0,
        "mulligans": [games] + [0] * 7, "opening_lands": [0, 0, 0, games, 0, 0, 0, 0],
        **{name: [0] * turns for name in (
            "land_drops", "lands_on_curve", "mana_available", "mana_spent", "color_screw")},
        "commander_turn": never, "first_kill": list(never), "kill": list(never),
        **{name: [0] * len(deck["pieces"]) for name in (
            "drawn", "cast", "cast_turn_sum", "stranded")},
    }


def creature(name, cost, mv, power, colors="G", text="", **extra):
    return card(name, colors=colors, types="Creature — Elf", mana_cost=cost, cmc=mv,
                power=str(power), toughness="1", text=text, **extra)


CARDS = [
    card("Green Leader", colors="GU", types="Legendary Creature — Elf", mana_cost="{G}", cmc=1,
         power="2", toughness="2"),
    card("Forest", colors="G", types="Basic Land — Forest", text="({T}: Add {G}.)"),
    card("Island", colors="U", types="Basic Land — Island", text="({T}: Add {U}.)"),
    card("Sol Ring", mana_cost="{1}", cmc=1, text="{T}: Add {C}{C}."),
    card("Simic Signet", colors="GU", mana_cost="{2}", cmc=2, text="{1}, {T}: Add {G}{U}."),
    card("Talisman of Curiosity", colors="GU", mana_cost="{2}", cmc=2,
         text="{T}: Add {C}.\n{T}: Add {G} or {U}. This artifact deals 1 damage to you."),
    creature("Llanowar Elves", "{G}", 1, 1, text="{T}: Add {G}."),
    card("Cultivate", colors="G", types="Sorcery", mana_cost="{2}{G}", cmc=3,
         text="Search your library for up to two basic land cards, reveal those cards, put one "
              "onto the battlefield tapped and the other into your hand, then shuffle."),
    card("Misty Rainforest", types="Land",
         text="{T}, Pay 1 life, Sacrifice this land: Search your library for a Forest or Island "
              "card, put it onto the battlefield, then shuffle."),
    card("Evolving Wilds", types="Land",
         text="{T}, Sacrifice this land: Search your library for a basic land card, put it onto "
              "the battlefield tapped, then shuffle."),
    card("Breeding Pool", colors="GR", types="Land — Forest Mountain",
         text="({T}: Add {R} or {G}.)\nAs this land enters, you may pay 2 life. If you don't, it "
              "enters tapped."),
    card("Hinterland Harbor", colors="GU", types="Land",
         text="This land enters tapped unless you control a Forest or an Island.\n"
              "{T}: Add {G} or {U}."),
    card("Botanical Sanctum", colors="GU", types="Land",
         text="This land enters tapped unless you control two or fewer other lands.\n"
              "{T}: Add {G} or {U}."),
    card("Dreamroot Cascade", colors="GU", types="Land",
         text="This land enters tapped unless you control two or more other lands.\n"
              "{T}: Add {G} or {U}."),
    card("Thornwood Falls", colors="GU", types="Land",
         text="This land enters tapped.\nWhen this land enters, you gain 1 life.\n"
              "{T}: Add {G} or {U}."),
    card("Turntimber Symbiosis // Turntimber, Serpentine Wood", colors="G",
         types="Sorcery // Land", mana_cost="{4}{G}{G}{G} // ", cmc=7, layout="modal_dfc",
         card_faces=[
             {"name": "Turntimber Symbiosis", "type_line": "Sorcery", "mana_cost": "{4}{G}{G}{G}",
              "oracle_text": "Look at the top seven cards of your library."},
             {"name": "Turntimber, Serpentine Wood", "type_line": "Land", "mana_cost": "",
              "oracle_text": "As this land enters, you may pay 3 life. If you don't, it enters "
                             "tapped.\n{T}: Add {G}."},
         ]),
    card("Genesis Wave", colors="G", types="Sorcery", mana_cost="{X}{G}{G}{G}", cmc=3),
    card("Gitaxian Probe", colors="U", types="Sorcery", mana_cost="{U/P}", cmc=1,
         text="Look at target player's hand.\nDraw a card."),
    card("Divination", colors="U", types="Sorcery", mana_cost="{2}{U}", cmc=3,
         text="Draw two cards."),
    card("Phyrexian Arena", colors="B", types="Enchantment", mana_cost="{1}{B}{B}", cmc=3,
         text="At the beginning of your upkeep, you draw a card and lose 1 life."),
    card("Purphoros's Intervention", colors="R", types="Sorcery", mana_cost="{X}{R}", cmc=1),
    card("Fiery Confluence", colors="R", types="Sorcery", mana_cost="{2}{R}{R}", cmc=4,
         text="Fiery Confluence deals 2 damage to each opponent."),
    card("Lightning Bolt", colors="R", types="Instant", mana_cost="{R}", cmc=1,
         text="Lightning Bolt deals 3 damage to any target."),
    card("Raise the Alarm", colors="W", types="Instant", mana_cost="{1}{W}", cmc=2,
         text="Create two 1/1 white Soldier creature tokens."),
    card("Smothering Tithe", colors="W", types="Enchantment", mana_cost="{3}{W}", cmc=4,
         text="Whenever an opponent draws a card, that player may pay {2}. If the player "
              "doesn't, you create a Treasure token."),
    creature("Wood Elves", "{2}{G}", 3, 1,
             text="When this creature enters, search your library for a Forest card, put that "
                  "card onto the battlefield, then shuffle."),
    creature("Goblin Guide", "{R}", 1, 2, colors="R", keywords=["Haste"]),
    card("Ancestral Vision", colors="U", types="Sorcery", text="Suspend 4—{U}"),
    card("Hybrid Elf", colors="GU", types="Creature — Elf", mana_cost="{G/U}{G/U}", cmc=2,
         power="2", toughness="2"),
]


class GoldfishCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.store.import_cards(CARDS)
        others = [c["name"] for c in CARDS[3:]]
        self.deck = {"commander": "Green Leader",
                     "cards": {"Forest": 99 - len(others), **dict.fromkeys(others, 1)}}
        self.compiled = compile_deck(self.deck, self.store)
        self.pieces = {p["name"]: p for p in self.compiled["pieces"]}


class CompileTests(GoldfishCase):
    def test_mana_rocks_signets_talismans_and_dorks(self):
        sol, signet = self.pieces["Sol Ring"], self.pieces["Simic Signet"]
        self.assertEqual(sol["mana"], [C, C])
        self.assertEqual(sol["class"], 0)  # Pays for itself at once.
        self.assertEqual(signet["mana"], [G | U])  # {1} in, {G}{U} out: one more mana.
        self.assertEqual(signet["class"], 1)
        self.assertEqual(self.pieces["Talisman of Curiosity"]["mana"], [C | G | U])
        elves = self.pieces["Llanowar Elves"]
        self.assertEqual((elves["mana"], elves["creature"], elves["power"]), ([G], True, 1))

    def test_land_entry_rules_and_colors(self):
        expected = {
            # Mana outside the commander's colors is never counted: in a green-blue deck,
            # Breeding Pool makes only green.
            "Forest": (UNTAPPED, G), "Breeding Pool": (UNTAPPED, G),
            "Hinterland Harbor": (CHECK, G | U), "Botanical Sanctum": (FAST, G | U),
            "Dreamroot Cascade": (SLOW, G | U), "Thornwood Falls": (TAPPED, G | U),
        }
        for name, (enter, colors) in expected.items():
            self.assertEqual((self.pieces[name]["enter"], self.pieces[name]["colors"]),
                             (enter, colors), name)
        self.assertEqual(self.pieces["Hinterland Harbor"]["check_types"], G | U)
        self.assertEqual(self.pieces["Breeding Pool"]["land_types"], R | G)

    def test_fetches_land_ramp_and_modal_lands(self):
        index = {p["name"]: i for i, p in enumerate(self.compiled["pieces"])}
        misty, wilds = self.pieces["Misty Rainforest"], self.pieces["Evolving Wilds"]
        self.assertEqual(misty["fetch"], sorted([index["Forest"], index["Breeding Pool"]]))
        self.assertFalse(misty["fetch_tapped"])
        self.assertEqual(wilds["fetch"], [index["Forest"]])
        self.assertTrue(wilds["fetch_tapped"])
        cultivate = self.pieces["Cultivate"]
        self.assertEqual((cultivate["ramp_battlefield"], cultivate["ramp_hand"],
                          cultivate["ramp_tapped"]), (1, 1, True))
        self.assertEqual(cultivate["ramp"], [index["Forest"]])
        self.assertEqual(self.pieces["Wood Elves"]["ramp_battlefield"], 1)
        modal = self.pieces["Turntimber Symbiosis // Turntimber, Serpentine Wood"]
        self.assertTrue(modal["back_land"] and modal["castable"])
        self.assertEqual((modal["colors"], modal["enter"]), (G, UNTAPPED))

    def test_costs_and_one_shot_effects(self):
        self.assertEqual(self.pieces["Genesis Wave"]["pips"], [G, G, G])
        self.assertEqual(self.pieces["Genesis Wave"]["generic"], 2)  # X is cast as 2.
        self.assertEqual(self.pieces["Gitaxian Probe"]["total"], 0)  # Phyrexian: pay life.
        self.assertEqual(self.pieces["Hybrid Elf"]["pips"], [G | U, G | U])
        self.assertEqual(self.pieces["Gitaxian Probe"]["draw"], 1)
        self.assertEqual(self.pieces["Divination"]["draw"], 2)
        self.assertEqual(self.pieces["Phyrexian Arena"]["upkeep_draw"], 1)
        self.assertEqual(self.pieces["Phyrexian Arena"]["draw"], 0)
        self.assertEqual(self.pieces["Fiery Confluence"]["burn_each"], 2)
        self.assertEqual(self.pieces["Lightning Bolt"]["burn_one"], 3)
        self.assertEqual(self.pieces["Raise the Alarm"]["token_power"], 2)
        self.assertEqual(self.pieces["Smothering Tithe"]["treasures"], 0)  # A repeating trigger.
        self.assertTrue(self.pieces["Goblin Guide"]["haste"])
        self.assertFalse(self.pieces["Ancestral Vision"]["castable"])  # No mana cost.


class EngineSetupTests(GoldfishCase):
    def test_a_missing_engine_asks_for_setup_and_bad_settings_fail_first(self):
        for kwargs in ({"games": 0}, {"turns": 0}):
            with self.assertRaises(ValueError):
                run_goldfish(self.deck, self.store, **kwargs)
        with patch("mtg_agentic.goldfish.RUST_BINARY", self.path / "missing"):
            with self.assertRaisesRegex(FileNotFoundError, "isn't built. Run /setup"):
                run_goldfish(self.deck, self.store, games=5)

    def test_reports_pass_the_thread_count_and_flag_odd_deck_sizes(self):
        with patch("mtg_agentic.goldfish.play_games_rust", side_effect=stub_engine) as engine:
            report = run_goldfish(self.deck, self.store, games=50, turns=6, threads=2)
            self.assertEqual(engine.call_args.kwargs, {"threads": 2})
            self.assertEqual((report["games"], report["deck_size"]), (50, 100))
            self.assertNotIn("warning", report)
            small = {**self.deck, "cards": {**self.deck["cards"], "Forest": 1}}
            report = run_goldfish(small, self.store, games=5)
        self.assertIn("not 100", report["warning"])

    def test_engine_is_ready_only_when_built_after_its_source_changed(self):
        source = self.path / "goldfish-rs"
        (source / "src").mkdir(parents=True)
        for name in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
            (source / name).write_text("", encoding="utf-8")
        binary = source / "target" / "release" / "goldfish-rs"
        with (
            patch("mtg_agentic.goldfish.RUST_SOURCE", source),
            patch("mtg_agentic.goldfish.RUST_BINARY", binary),
        ):
            self.assertEqual(engine_problem(), "the goldfish engine isn't built")
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"")
            os.utime(binary, (1, 1))
            self.assertEqual(engine_problem(), "the goldfish engine is older than its source")
            os.utime(binary, None)
            os.utime(source / "src" / "main.rs", (1, 1))
            self.assertIsNone(engine_problem())

    def test_build_needs_the_source_and_cargo_and_reports_failures(self):
        source = self.path / "goldfish-rs"
        with patch("mtg_agentic.goldfish.RUST_SOURCE", source):
            self.assertIn("source is missing", build_problem())
            source.mkdir()
            (source / "Cargo.toml").write_text("", encoding="utf-8")
            with patch("mtg_agentic.goldfish.cargo_path", return_value=None):
                self.assertIn("Rust isn't installed; install it from https://rustup.rs",
                              build_problem())
                with self.assertRaises(FileNotFoundError):
                    build_engine()
            with (
                patch("mtg_agentic.goldfish.cargo_path", return_value="cargo"),
                patch("mtg_agentic.goldfish.subprocess.run") as run,
            ):
                self.assertIsNone(build_problem())
                run.return_value = SimpleNamespace(returncode=0, stderr="")
                self.assertTrue(build_engine().startswith("built in"))
                self.assertEqual(run.call_args.args[0][:3], ["cargo", "build", "--release"])
                run.return_value = SimpleNamespace(returncode=101, stderr="error: linker failed")
                with self.assertRaisesRegex(OSError, "linker failed"):
                    build_engine()

    def test_cli_writes_a_report_without_a_model(self):
        path = self.path / "deck.json"
        path.write_text(json.dumps(self.deck), encoding="utf-8")
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("mtg_agentic.__main__._make_model", side_effect=AssertionError("No model")),
            patch("mtg_agentic.goldfish.play_games_rust", side_effect=stub_engine),
        ):
            status = main(["--quiet", "--data-dir", str(self.path), "goldfish", str(path),
                           "--games", "40", "--turns", "5"])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual((report["games"], len(report["by_turn"])), (40, 5))
        self.assertEqual(report["commander"]["name"], "Green Leader")


@unittest.skipIf(ENGINE_MISSING, f"{ENGINE_MISSING}; run python -m mtg_agentic setup")
class RustEngineTests(GoldfishCase):
    def test_mono_forest_commander_is_cast_turn_one_and_kills_by_commander_damage(self):
        deck = compile_deck({"commander": "Green Leader", "cards": {"Forest": 99}}, self.store)
        counters = play_games_rust(deck, 50, 12, seed=3)
        self.assertEqual(counters["commander_turn"][0], 50)
        self.assertEqual(counters["lands_on_curve"], [50] * 12)
        # Two damage a turn from turn 2: 21 commander damage after eleven attacks.
        self.assertEqual(counters["first_kill"][11], 50)
        self.assertEqual(counters["kill"][12], 50)
        report = summarize(deck, counters)
        self.assertEqual(report["commander"]["median_turn"], 1)
        self.assertEqual(report["first_kill"]["median_turn"], 12)
        self.assertIsNone(report["table_kill"]["median_turn"])

    def test_a_seed_gives_the_same_games_on_any_number_of_threads(self):
        first = play_games_rust(self.compiled, 300, 8, seed=2**40, threads=1)
        for threads in (3, 0):
            self.assertEqual(play_games_rust(self.compiled, 300, 8, 2**40, threads=threads),
                             first)
        self.assertNotEqual(play_games_rust(self.compiled, 300, 8, seed=12), first)
        self.assertEqual(first["games"], 300)
        self.assertEqual(sum(first["mulligans"]), 300)
        # Ancestral Vision has no mana cost: drawn, never cast, always stranded.
        vision = list(self.pieces).index("Ancestral Vision")
        self.assertEqual(first["cast"][vision], 0)
        self.assertEqual(first["stranded"][vision], first["drawn"][vision])

    def test_reports_list_cards_and_what_is_simulated(self):
        report = run_goldfish(self.deck, self.store, games=500, turns=8, seed=11)
        self.assertEqual((report["games"], report["deck_size"]), (500, 100))
        names = {c["name"] for c in report["cards"]}
        self.assertIn("Sol Ring", names)
        self.assertNotIn("Forest", names)
        self.assertIn("Ancestral Vision", report["coverage"]["not cast"])
        self.assertIn("Turntimber Symbiosis // Turntimber, Serpentine Wood",
                      report["coverage"]["land"])
        vision = next(c for c in report["cards"] if c["name"] == "Ancestral Vision")
        self.assertEqual((vision["cast_rate"], vision["stranded_rate"]), (0, 1))
        self.assertGreater(report["by_turn"][3]["mana_available"], 3)
