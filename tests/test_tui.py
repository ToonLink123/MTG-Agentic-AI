import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from mtg_agentic.__main__ import main
from mtg_agentic.cards import trim_card
from tests.pipeline_fixtures import JODAH, make_spec

HAS_TEXTUAL = importlib.util.find_spec("textual") is not None
if HAS_TEXTUAL:
    from textual.widgets import Input, Static

    from mtg_agentic.tui import (
        COUNCIL_STEPS,
        DRAFT_STEPS,
        PLAN_STEP,
        SETUP_STEPS,
        ChatApp,
        QuestionPicker,
        SessionPicker,
        Settings,
        card_type,
        identity,
        mana,
        parse_question,
        render_deck,
        render_interaction,
        render_map,
        render_review,
        render_spec,
        run_cli,
    )

SPEC = make_spec().to_dict()
POOL = {
    "version": 1, "spec": SPEC, "diagnostics": [],
    "cards": [{**trim_card(JODAH), "name": "Sol Ring", "roles": ["ramp"]}],
    "queries": [{"role": "ramp", "query": "otag:ramp"}],
}
ENGINE = {
    "summary": "Ramp into Myojin and keep their counters.", "resources": ["divinity counters"],
    "stages": [{"stage": "makes", "what": "mana for Myojin", "cards": [
        {"name": "Sol Ring", "tier": "primary", "role": "fast mana"},
    ]}],
    "links": [], "asks": [{"ask": "A1", "cards": ["Sol Ring"], "note": ""}],
    "gaps": ["No card returns a dead Myojin"],
}
MAPPED = {**POOL, "engine_map": ENGINE}
READY = {"status": "ready", "round": 1, "max_rounds": 6, "understanding": "A Myojin deck.",
         "questions": []}
QUESTIONS = {**READY, "status": "ask", "understanding": "Myojin, cast from hand.", "questions": [
    {"question": "How fast should it win?", "why": "Sets the curve and the ramp.",
     "options": ["Fast, by turn 7", "Slow and grindy"]},
    {"question": "How much interaction?", "why": "Sets removal and protection.",
     "options": ["Lots", "Just enough"]},
]}
DECK = {
    "valid": False, "commander": JODAH["name"], "bracket": 3, "spec": SPEC,
    "cards": {"Sol Ring": 1, "Island": 10}, "role_counts": {"ramp": 1, "lands": 10},
    "validation": {"errors": [{"error": "deck_size", "detail": "Expected 99 cards"}]},
    "review": {"status": "not_run", "errors": []},
}


def plain(renderable, width=120):
    console = Console(width=width, file=io.StringIO(), color_system=None)
    console.print(renderable)
    return console.file.getvalue()


@unittest.skipUnless(HAS_TEXTUAL, "the TUI needs the optional textual package")
class TuiHelperTests(unittest.TestCase):
    def test_double_brackets_mark_cards_and_stay_readable_in_the_question(self):
        cards, question = parse_question("Does [[Doubling Season]] double [[ Myojin of Night's Reach ]] counters?")
        self.assertEqual(cards, ["Doubling Season", "Myojin of Night's Reach"])
        self.assertEqual(question, "Does Doubling Season double Myojin of Night's Reach counters?")
        self.assertEqual(parse_question("no cards here"), ([], "no cards here"))

    def test_follow_ups_use_the_orchestrator_model_unless_one_is_chosen(self):
        lead = ["--provider", "claude", "--model", "claude-sonnet-5-5", "--effort", "xhigh"]
        self.assertEqual(Settings(provider="claude").orchestrator_args(), lead)
        self.assertEqual(Settings(provider="claude", orchestrate=True).orchestrator_args(), lead)
        chosen = Settings(provider="claude", model="claude-opus-5-5")
        self.assertEqual(chosen.orchestrator_args(), ["--provider", "claude", "--model", "claude-opus-5-5"])
        self.assertEqual(Settings(provider="ollama").orchestrator_args(), ["--provider", "ollama"])

    def test_model_takes_an_optional_provider_that_resets_the_model(self):
        settings = Settings(provider="ollama", model="qwen2.5:7b")
        self.assertEqual(settings.model_args(), ["--provider", "ollama", "--model", "qwen2.5:7b"])
        settings.choose_model(["Claude"])
        self.assertEqual(settings.model_args(), ["--provider", "claude"])
        settings.choose_model(["claude-haiku-5-5"])
        self.assertEqual((settings.provider, settings.model), ("claude", "claude-haiku-5-5"))
        settings.choose_model(["ollama", "qwen3:8b"])
        self.assertEqual((settings.provider, settings.model), ("ollama", "qwen3:8b"))
        settings.choose_model(["default"])
        self.assertIsNone(settings.model)
        with self.assertRaises(ValueError):
            settings.choose_model(["claude", "a", "b"])
        self.assertEqual(settings.provider, "ollama")  # A rejected command changes nothing.

    def test_mode_and_review_accept_only_their_values(self):
        settings = Settings()
        settings.choose_mode("unrestricted")
        settings.choose_review("off")
        self.assertEqual((settings.mode, settings.review), ("unrestricted", False))
        settings.choose_mode("auto")
        self.assertIsNone(settings.mode)
        with self.assertRaises(ValueError):
            settings.choose_mode("loose")
        with self.assertRaises(ValueError):
            settings.choose_review("maybe")

    def test_pool_accepts_only_its_modes(self):
        settings = Settings()
        self.assertEqual(settings.pool, "mixed")
        settings.choose_pool("relevant")
        self.assertEqual(settings.pool, "relevant")
        with self.assertRaises(ValueError):
            settings.choose_pool("random")
        self.assertEqual(settings.pool, "relevant")

    def test_startup_warns_when_this_python_cannot_embed_rules_questions(self):
        from mtg_agentic.tui import embedding_problem

        versions = {"5.21.0": None, "5.30.1": None, "4.57.6": "4.57.6", "5.2.0": "5.2.0"}
        for version, flagged in versions.items():
            with patch("importlib.metadata.version", return_value=version):
                problem = embedding_problem()
            self.assertEqual(problem is None, flagged is None, version)
            if flagged:
                self.assertIn(f"transformers {flagged}", problem)
                self.assertIn(sys.executable, problem)

    def test_an_uncertain_bracket_and_an_auto_land_count_read_plainly(self):
        from mtg_agentic.tui import render_spec, unverified

        warning = {"warning": "bracket_uncertain", "detail": "x",
                   "reasons": ["Likely win speed is unknown"]}
        shown = plain(unverified([warning, {"cards": ["A", "B"], "missing": "copy rules"}]))
        self.assertIn("Bracket fit uncertain, not a failure", shown)
        self.assertIn("Likely win speed is unknown", shown)
        self.assertIn("Unverified: A + B", shown)
        spec = {"commander": "Jodah", "color_identity": "WUBRG", "bracket": 3,
                "playstyle": "midrange", "commander_dependency": "medium",
                "role_targets": {"ramp": 10, "lands": 0}}
        self.assertIn("lands auto", plain(render_spec(spec)))

    def test_claude_and_the_council_are_the_defaults(self):
        settings = Settings()
        self.assertEqual((settings.provider, settings.orchestrate), ("claude", True))
        self.assertIn("council", settings.describe())

    def test_the_engine_map_runs_on_the_member_model_in_orchestrate_mode(self):
        member = ["--provider", "claude", "--model", "claude-haiku-5-5", "--effort", "high"]
        self.assertEqual(Settings(provider="claude", orchestrate=True).member_args(), member)
        self.assertEqual(Settings(provider="claude", orchestrate=False).member_args(), ["--provider", "claude"])
        self.assertEqual(Settings(provider="ollama", orchestrate=True).member_args(),
                         ["--provider", "ollama"])

    def test_spec_map_and_deck_show_the_players_asks(self):
        brief = {
            "request": "Myojin with lots of removal", "interview": [],
            "asks": [{"id": "A1", "quote": "Myojin", "wants": "Myojin tribal", "kind": "cards"}],
            "deferred": ["How much interaction?"],
        }
        text = plain(render_spec(make_spec(brief=brief).to_dict()))
        self.assertIn('A1 "Myojin"  Myojin tribal', text)
        self.assertIn("How much interaction?", text)
        text = plain(render_map(ENGINE))
        self.assertIn("Makes", text)
        self.assertIn("Sol Ring", text)
        self.assertIn("A1 1 card", text)
        self.assertIn("No card returns a dead Myojin", text)
        asks = [
            {"id": "A1", "quote": "Myojin", "wants": "Myojin", "kind": "cards",
             "mapped": ["Sol Ring"], "cards": ["Sol Ring"]},
            {"id": "A2", "quote": "removal", "wants": "removal", "kind": "cards",
             "mapped": ["Spell 1"], "cards": []},
            {"id": "A3", "quote": "keep it casual", "wants": "low power", "kind": "deck",
             "mapped": [], "cards": []},
        ]
        text = plain(render_deck({**DECK, "asks": asks}, lambda name: None))
        self.assertRegex(text, r'✓ "Myojin" +Sol Ring')
        self.assertRegex(text, r'✗ "removal" +no card in the deck serves it')
        self.assertIn("deck-wide: low power", text)

    def test_an_answer_lists_its_rules_first_then_answers_the_question(self):
        result = {
            "status": "unsupported", "claims": [], "question": "Can these loop?",
            "cards": ["Reflection", "Bell-Ringer"],
            "established": [
                {"claim": "The token copy has the enters ability.", "citations": [
                    {"id": "Ruling Reflection 1", "quote": "Any enters-the-battlefield abilities"}]},
                {"claim": "It untaps.", "citations": [
                    {"id": "Ruling Reflection 1", "quote": "Any enters-the-battlefield abilities"},
                    {"id": "CR 701.26b", "quote": "To untap a permanent, rotate it back"}]},
            ],
            "answer": "Only with mana each time; the mana is not established.",
            "conditions": ["Reflection has been under your control since your turn began"],
            "missing": "mana from the two creatures",
            "rejected": [{"claim": "x", "citations": [],
                          "problems": ["The quote from CR 1.1 is not in its text"]}],
            "answered_by": {"provider": "claude", "model": "claude-haiku-5-5", "effort": "high"},
        }
        text = plain(render_interaction(result))
        self.assertIn("Established so far", text)
        # Each source once, numbered, with what was quoted from it; the steps' prose stays in
        # the saved file, and the direct answer comes after the rules.
        self.assertEqual(text.count("Ruling Reflection 1"), 1)
        self.assertIn("2. CR 701.26b", text)
        self.assertNotIn("The token copy has the enters ability.", text)
        self.assertLess(text.index("CR 701.26b"), text.index("Answer"))
        self.assertIn("Only with mana each time", text)
        self.assertIn("Provided: Reflection has been under your control", text)
        self.assertIn("Not established: mana from the two creatures", text)
        self.assertIn("Left out: The quote from CR 1.1 is not in its text.", text)
        self.assertIn("Answered by claude-haiku-5-5 (high)", text)

    def test_an_unverified_interaction_is_a_warning_not_a_failure(self):
        warning = {"warning": "interaction_unverified", "cards": ["Reflection", "Bell-Ringer"],
                   "question": "Loop?", "detail": "not verified"}
        review = {"status": "passed", "errors": [], "warnings": [warning],
                  "interactions": [{"status": "unsupported"}]}
        text = plain(render_review(review))
        self.assertIn("✓ Review passed", text)
        self.assertIn("Unverified: Reflection + Bell-Ringer", text)
        text = plain(render_deck({**DECK, "review": review}, lambda name: None))
        self.assertIn("Unverified: Reflection + Bell-Ringer", text)

    def test_mana_and_identity_read_in_wubrg_order_with_hybrid_symbols_grouped(self):
        self.assertEqual(mana("{W}{U}{B/P}{R}{G}").plain, "WU(B/P)RG")
        self.assertEqual(mana("{1}{U} // {3}{R}").plain, "1U // 3R")
        self.assertEqual(identity(["B", "G", "R", "U", "W"]).plain, "WUBRG")
        self.assertEqual(identity([]).plain, "C")

    def test_deck_groups_cards_by_front_face_type_with_lands_last(self):
        self.assertEqual(card_type({"type_line": "Artifact Creature — Golem"}), "Creature")
        self.assertEqual(card_type({"type_line": "Artifact Land"}), "Land")
        self.assertEqual(card_type({"type_line": "Sorcery // Land"}), "Sorcery")
        self.assertEqual(card_type(None), "Other")
        types = {"Sol Ring": "Artifact", "Island": "Basic Land — Island", JODAH["name"]: JODAH["type_line"]}

        def lookup(name):
            return {"type_line": types[name], "mana_cost": "{1}", "cmc": 1}

        text = plain(render_deck(DECK, lookup, Path("deck.json")))
        self.assertIn("1 problem", text)
        self.assertIn("Artifacts 1", text)
        self.assertIn("Lands 10", text)
        self.assertIn("10 Island", text)
        self.assertIn("deck_size: Expected 99 cards", text)
        self.assertLess(text.index("Artifacts"), text.index("Lands"))


class FakeCli:
    def __init__(self, results, wait=None, lines=None):
        self.results, self.wait, self.calls = results, wait, []
        self.lines = lines or {}

    async def __call__(self, argv, data_dir, on_line):
        self.calls.append(argv)
        on_line(f"running {argv[0]}")
        for line in self.lines.get(argv[0], []):
            on_line(line)
        if self.wait is not None:
            await self.wait.wait()
        result = self.results[argv[0]]
        # A list answers successive calls, such as interview rounds.
        return result.pop(0) if isinstance(result, list) else result


@unittest.skipUnless(HAS_TEXTUAL, "the TUI needs the optional textual package")
class TuiAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)

    async def until(self, pilot, condition):
        for _ in range(100):
            if condition():
                return
            await pilot.pause(0.05)
        self.fail("the app never reached the expected state")

    async def send(self, app, pilot, text, *, settle=True):
        app.query_one(Input).value = text
        await pilot.press("enter")
        await pilot.pause()
        if settle:
            await app.workers.wait_for_complete()
            await pilot.pause()

    async def test_a_name_sets_the_commander_then_a_request_runs_the_pipeline(self):
        cli = FakeCli({
            "card": (0, trim_card(JODAH)), "interview": (0, READY), "intake": (0, SPEC),
            "discover": (0, POOL), "engine-map": (0, MAPPED), "draft": (1, DECK),
        })
        app = ChatApp(self.data_dir, Settings(provider="ollama", orchestrate=False), runner=cli)
        async with app.run_test() as pilot:
            await self.send(app, pilot, "jodah")
            self.assertEqual(app.commander, JODAH["name"])
            await self.send(app, pilot, "/review off")
            await self.send(app, pilot, "/mode unrestricted")
            await self.send(app, pilot, "/pool popular")
            await self.send(app, pilot, "/model ollama qwen3:8b")
            await self.send(app, pilot, "Bracket 3 Myojin tribal")
        interview, intake, discover, engine, draft = cli.calls[1:]
        model = ["--provider", "ollama", "--model", "qwen3:8b"]
        self.assertEqual(cli.calls[0], ["card", "jodah"])
        self.assertEqual(interview[:5], ["interview", "--commander", JODAH["name"], "--request", "Bracket 3 Myojin tribal"])
        self.assertEqual(intake[:5], ["intake", "--commander", JODAH["name"], "--request", "Bracket 3 Myojin tribal"])
        self.assertEqual(Path(intake[intake.index("--interview") + 1]).name, "01-interview.json")
        self.assertIn("unrestricted", interview)
        self.assertIn("unrestricted", intake)
        self.assertEqual((interview[-4:], intake[-4:], engine[-4:]), (model, model, model))
        self.assertEqual(Path(discover[2]).name, "01-spec.json")
        self.assertEqual(discover[3:5], ["--pool-mode", "popular"])
        self.assertEqual((engine[0], Path(engine[2]).name), ("engine-map", "01-pool.json"))
        self.assertEqual(Path(draft[2]).name, "01-pool.json")
        self.assertEqual(json.loads((app.session / "01-pool.json").read_text(encoding="utf-8")), MAPPED)
        self.assertIn("--skip-review", draft)
        self.assertEqual(json.loads(app.deck_path.read_text(encoding="utf-8")), DECK)
        self.assertEqual([s["name"] for s in app.last.steps], DRAFT_STEPS)
        self.assertEqual([s["state"] for s in app.last.steps], ["done"] * 4 + ["warn"])
        self.assertFalse(app.busy)

    async def test_ask_needs_no_brackets_and_passes_the_commander_for_short_names(self):
        answer = {"status": "supported", "claims": [], "cards": [JODAH["name"]],
                  "question": "Can jodah loop?"}
        cli = FakeCli({"check-interaction": (0, answer)})
        app = ChatApp(self.data_dir, Settings(provider="ollama"), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "/ask Can jodah loop?")
            await self.send(app, pilot, "/ask Does [[Doubling Seasn]] work?")
        plain_words, bracketed = cli.calls
        self.assertEqual(plain_words[:5], ["check-interaction", "--question", "Can jodah loop?",
                                           "--commander", JODAH["name"]])
        self.assertEqual(bracketed[:3], ["check-interaction", "Doubling Seasn", "--question"])
        # Each answer is kept in the session, evidence and all, for a later look.
        for name in ("ask-01.json", "ask-02.json"):
            saved = json.loads((app.session / name).read_text(encoding="utf-8"))
            self.assertEqual(saved, answer)

    async def test_an_error_envelope_stops_the_pipeline_at_that_step(self):
        error = {"valid": False, "errors": [{"error": "llm", "detail": "Ollama is not running"}]}
        cli = FakeCli({"interview": (0, READY), "intake": (2, error)})
        app = ChatApp(self.data_dir, Settings(orchestrate=False), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal")
        self.assertEqual([argv[0] for argv in cli.calls], ["interview", "intake"])
        self.assertEqual([s["state"] for s in app.last.steps],
                         ["done", "failed", "pending", "pending", "pending"])
        self.assertIsNone(app.deck)

    async def test_escape_stops_the_run_and_new_requests_wait_until_then(self):
        cli = FakeCli({"interview": (0, READY)}, wait=asyncio.Event())
        app = ChatApp(self.data_dir, Settings(), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            self.assertTrue(app.busy)
            await self.send(app, pilot, "another deck", settle=False)
            self.assertEqual(len(cli.calls), 1)
            await self.send(app, pilot, "/mode standard", settle=False)  # Settings still apply.
            self.assertEqual(app.settings.mode, "standard")
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertFalse(app.busy)
            self.assertEqual(app.last.steps[0]["state"], "stopped")

    async def test_orchestrate_mode_runs_the_council_and_shows_its_agents_and_tools(self):
        council = {
            "orchestrator": "claude-sonnet-5-5", "member": "claude-haiku-5-5",
            "game_plan": "Cast [[Myojin]] from hand with Jodah.", "non_goals": ["tokens"],
            "core": ["Sol Ring"], "dropped_game_changers": [],
            "drafts": [{"focus": "engine", "role_counts": {}, "problems": 0}],
            "locked": ["Sol Ring"], "contested": ["Spell 1"], "open_slots": 1, "votes": {},
            "decision": {"keep": ["Spell 1"], "notes": "followed the vote"},
        }
        lines = [
            "council: 1/5 plan: claude-sonnet-5-5 sets the game plan and core",
            "agent: orchestrator | start | claude-sonnet-5-5 | game plan and core cards",
            "agent: orchestrator | done | 3.0s",
            "council: 2/5 drafts: 3 members draft around 1 core cards",
            "agent: engine | start | claude-haiku-5-5 | draft around the core",
            "tool: assemble | 63 nonland slots by role",
            "agent: engine | done | 2.0s",
            "council: 3/5 debate: 40 cards agreed; 1 contested for 1 slots",
            "council: 4/5 decision: claude-sonnet-5-5 settles 1 contested cards",
            "council: 5/5 final: lands, validation and safety review",
            "agent: judges | start | claude-haiku-5-5 | 2 rules questions, up to 8 at a time",
            "agent: judges | progress | 2/2 answered",
            "agent: judges | done | 4.0s",
        ]
        cli = FakeCli(
            {"interview": (0, READY), "intake": (0, SPEC), "discover": (0, POOL),
             "engine-map": (0, MAPPED), "council": (1, {**DECK, "council": council})},
            lines={"council": lines},
        )
        app = ChatApp(self.data_dir, Settings(provider="claude"), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "/orchestrate on")
            self.assertTrue(app.settings.orchestrate)
            await self.send(app, pilot, "Myojin tribal")
            turn = app.last
        lead = ["--provider", "claude", "--model", "claude-sonnet-5-5", "--effort", "xhigh"]
        member = ["--provider", "claude", "--model", "claude-haiku-5-5", "--effort", "high"]
        interview, intake, discover, engine, run = cli.calls
        self.assertEqual((interview[-6:], intake[-6:], discover[-6:]), (lead, lead, lead))
        self.assertEqual(engine[-6:], member)
        self.assertEqual((run[0], Path(run[2]).name, run[-2:]), ("council", "01-pool.json", ["--provider", "claude"]))
        self.assertEqual([s["name"] for s in turn.steps], COUNCIL_STEPS)
        self.assertEqual([s["state"] for s in turn.steps], ["done"] * 8 + ["warn"])
        self.assertEqual(turn.steps[PLAN_STEP + 1]["detail"], "3 members draft around 1 core cards")
        self.assertEqual({n: a["state"] for n, a in turn.agents.items()},
                         {"orchestrator": "done", "engine": "done", "judges": "done"})
        self.assertEqual(turn.agents["judges"]["calls"], 1)
        self.assertEqual(turn.tools, [("assemble", "63 nonland slots by role")])
        self.assertFalse(any(line.startswith(("agent:", "tool:")) for line in turn.tail))
        self.assertEqual(turn.lines[-len(lines):], lines)  # /log still has every event

    async def test_the_interview_asks_until_ready_and_its_answers_reach_intake(self):
        cli = FakeCli({
            "interview": [(0, QUESTIONS), (0, READY)], "intake": (0, SPEC), "discover": (0, POOL),
            "engine-map": (1, {**POOL, "valid": False, "map_errors": [
                {"error": "schema", "detail": "$.stages must be array"}]}),
            "draft": (1, DECK),
        })
        app = ChatApp(self.data_dir, Settings(orchestrate=False), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker))
            self.assertEqual(app.screen.question["question"], "How fast should it win?")
            await pilot.press("enter")  # The first pre-written answer.
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker)
                             and app.screen.number == 2)
            await pilot.press("down", "down", "down", "enter")  # Type my own answer.
            await pilot.press(*"Just enough, mostly removal")
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
        answers = [
            {"question": "How fast should it win?", "answer": "Fast, by turn 7", "source": "option"},
            {"question": "How much interaction?", "answer": "Just enough, mostly removal",
             "source": "own"},
        ]
        names = [argv[0] for argv in cli.calls]
        self.assertEqual(names, ["interview", "interview", "intake", "discover", "engine-map", "draft"])
        self.assertEqual(cli.calls[1][cli.calls[1].index("--round") + 1], "2")
        intake = cli.calls[2]
        saved = Path(intake[intake.index("--interview") + 1])
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), answers)
        # A malformed map is a warning: the draft uses the pool discovery saved.
        self.assertEqual(json.loads((app.session / "01-pool.json").read_text(encoding="utf-8")), POOL)
        self.assertEqual([s["state"] for s in app.last.steps], ["done", "done", "done", "warn", "warn"])
        self.assertEqual(app.last.steps[0]["detail"], "2 answers")
        self.assertEqual([t["role"] for t in app.transcript],
                         ["player", "interview", "interview", "interviewer", "assistant"])

    async def test_escape_leaves_the_answer_box_then_stops_the_interview(self):
        cli = FakeCli({"interview": (0, QUESTIONS)})
        app = ChatApp(self.data_dir, Settings(), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker))
            await pilot.press("down", "down", "down", "enter")  # Type my own answer.
            await pilot.pause()
            box = app.screen.query_one(Input)
            self.assertTrue(box.display)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, QuestionPicker)
            self.assertFalse(box.display)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, QuestionPicker)
            self.assertFalse(app.busy)
        self.assertEqual([argv[0] for argv in cli.calls], ["interview"])
        self.assertEqual(app.last.steps[0]["state"], "stopped")
        self.assertIsNone(app.deck)

    async def test_export_writes_a_text_list_beside_the_deck_and_copies_it(self):
        from mtg_agentic.validator import _parse_decklist

        deck = self.data_dir / "deck.json"
        deck.write_text(json.dumps(DECK), encoding="utf-8")
        app = ChatApp(self.data_dir, Settings(), runner=FakeCli({}))
        async with app.run_test() as pilot:
            await self.send(app, pilot, "/export")  # No deck yet: nothing to write.
            self.assertFalse((self.data_dir / "deck.txt").exists())
            await self.send(app, pilot, f"/load {deck}")
            with patch.object(app, "copy_to_clipboard") as copy:
                await self.send(app, pilot, "/export")
                await self.send(app, pilot, "/export")  # Never overwrites: deck-2.txt.
                await self.send(app, pilot, f"/export {self.data_dir / 'deck.txt'}")
        expected = f"1 {JODAH['name']}\n1 Sol Ring\n10 Island\n"
        self.assertEqual((self.data_dir / "deck.txt").read_text(encoding="utf-8"), expected)
        self.assertEqual((self.data_dir / "deck-2.txt").read_text(encoding="utf-8"), expected)
        self.assertFalse((self.data_dir / "deck-3.txt").exists())
        self.assertEqual(copy.call_count, 2)
        copy.assert_called_with(expected)
        entries, errors = _parse_decklist(expected)
        self.assertEqual(errors, [])
        self.assertEqual([(e.quantity, e.name) for e in entries],
                         [(1, JODAH["name"]), (1, "Sol Ring"), (10, "Island")])

    async def test_follow_ups_go_to_the_orchestrator_alone_and_save_new_versions(self):
        swap = {"remove": "Sol Ring", "add": "Arcane Signet", "reason": "cheaper"}
        edited = {**DECK, "valid": True, "edit": {
            "reply": "Swapped Sol Ring.", "swaps": [swap], "tools": ["card"], "changed": True,
        }}
        answered = {**edited, "edit": {"reply": "Island taps for U.", "swaps": [], "tools": [],
                                       "changed": False}}
        cli = FakeCli({
            "interview": (0, READY), "intake": (0, SPEC), "discover": (0, POOL),
            "engine-map": (0, MAPPED), "council": (1, DECK), "edit-deck": (0, edited),
        })
        app = ChatApp(self.data_dir, Settings(provider="claude", orchestrate=True), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal")
            first = app.deck_path
            await self.send(app, pilot, "Swap Sol Ring for something cheaper")
            self.assertEqual(app.deck_path.name, "01-deck-v2.json")
            cli.results["edit-deck"] = (0, answered)
            await self.send(app, pilot, "What does Island do?")
            self.assertEqual(app.deck_path.name, "01-deck-v2.json")  # A question keeps the version.
            self.assertEqual(json.loads(app.deck_path.read_text(encoding="utf-8")), answered)
            session = json.loads((app.session / "session.json").read_text(encoding="utf-8"))
        self.assertEqual([argv[0] for argv in cli.calls], [
            "interview", "intake", "discover", "engine-map", "council", "edit-deck", "edit-deck",
        ])
        edit = cli.calls[5]
        self.assertEqual(edit[:2], ["edit-deck", str(first)])
        self.assertEqual(Path(edit[edit.index("--pool") + 1]), first.parent / "01-pool.json")
        self.assertEqual(edit[edit.index("--request") + 1], "Swap Sol Ring for something cheaper")
        self.assertEqual(edit[-6:], ["--provider", "claude", "--model", "claude-sonnet-5-5",
                                     "--effort", "xhigh"])
        self.assertEqual(cli.calls[6][1], str(first.parent / "01-deck-v2.json"))
        self.assertEqual([t["role"] for t in session["turns"]],
                         ["player", "assistant", "player", "orchestrator", "player", "orchestrator"])
        self.assertEqual(session["turns"][3]["text"], "Swapped Sol Ring.\n- Sol Ring + Arcane Signet")
        self.assertEqual(session["deck"], "01-deck-v2.json")

    async def test_new_starts_over_and_session_resumes_a_past_one(self):
        old = self.data_dir / "tui" / "20261027-101500"
        old.mkdir(parents=True)
        (old / "01-deck.json").write_text(json.dumps(DECK), encoding="utf-8")
        (old / "01-pool.json").write_text(json.dumps(POOL), encoding="utf-8")
        turns = [{"role": "player", "text": "Myojin tribal"},
                 {"role": "assistant", "text": "Drafted 01-deck.json: 1 problem."}]
        (old / "session.json").write_text(json.dumps({
            "version": 1, "commander": JODAH["name"], "valid": False, "deck": "01-deck.json",
            "turns": turns,
        }), encoding="utf-8")
        app = ChatApp(self.data_dir, Settings(), runner=FakeCli({}))
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            first = app.session
            await self.send(app, pilot, "/new")
            self.assertNotEqual(app.session, first)
            self.assertIsNone(app.commander)
            self.assertEqual(len(app.tabs), 2)
            # Open tabs come first in the list (tab 1, this tab), then the saved session, which
            # resumes in this tab because it is still empty.
            await self.send(app, pilot, "/session 3")
            self.assertEqual((app.session, app.commander), (old, JODAH["name"]))
            self.assertEqual(len(app.tabs), 2)
            self.assertEqual((app.deck_path, app.pool_path), (old / "01-deck.json", old / "01-pool.json"))
            self.assertEqual((app.transcript, app.runs), (turns, 1))
            await self.send(app, pilot, "/new")
            await self.send(app, pilot, "/session")
            self.assertIsInstance(app.screen, SessionPicker)
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, SessionPicker)
            self.assertIsNone(app.deck)
            await self.send(app, pilot, "/session")
            await pilot.press("down", "enter")  # The second open tab holds the resumed session.
            await pilot.pause(0.2)
            self.assertEqual(app.session, old)
            self.assertEqual(app.deck, DECK)
            await self.send(app, pilot, "/session 1")
            self.assertEqual((app.session, app.commander), (first, JODAH["name"]))
            await self.send(app, pilot, "/close")
            self.assertEqual(len(app.tabs), 2)
            self.assertNotIn(first, [t.session for t in app.tabs])

    async def test_tabs_run_at_once_and_a_question_waits_for_its_own_tab(self):
        gate = asyncio.Event()
        calls = []
        results = {"intake": SPEC, "discover": POOL, "engine-map": MAPPED, "draft": DECK}

        async def runner(argv, data_dir, on_line):
            calls.append(argv)
            if argv[0] != "interview":
                return 1 if argv[0] == "draft" else 0, results[argv[0]]
            await gate.wait()
            first = argv[argv.index("--request") + 1] == "Myojin tribal"
            return 0, QUESTIONS if first and argv[argv.index("--round") + 1] == "1" else READY

        app = ChatApp(self.data_dir, Settings(orchestrate=False), runner=runner)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            one = app.tab
            await self.send(app, pilot, "/new", settle=False)  # Allowed while tab 1 works.
            two = app.tab
            self.assertIsNot(one, two)
            self.assertTrue(one.busy)
            self.assertIn("tab 2/2 · 1 working", plain(app.query_one("#topbar", Static).content))
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Second deck", settle=False)
            gate.set()
            await self.until(pilot, lambda: two.deck is not None and one.waiting)
            # Tab 1's question waits off screen; tab 2 finished on its own.
            self.assertNotIsInstance(app.screen, QuestionPicker)
            self.assertEqual(one.status(), "? waiting for your answers")
            self.assertEqual([e["note"] for e in app.session_entries()[:2]],
                             ["? waiting for your answers", "open"])
            self.assertEqual(two.deck_path.parent, two.session)
            self.assertIsNone(one.deck)
            await pilot.press("f6")
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker))
            self.assertIs(app.tab, one)
            await pilot.press("enter")
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker)
                             and app.screen.number == 2)
            await pilot.press("f6")  # Away from an open question: it waits again.
            await self.until(pilot, lambda: app.tab is two and one.waiting)
            self.assertNotIsInstance(app.screen, QuestionPicker)
            await pilot.press("shift+f6")
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker))
            self.assertEqual(app.screen.number, 2)
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
        answers = json.loads((one.session / "01-interview.json").read_text(encoding="utf-8"))
        self.assertEqual([a["answer"] for a in answers], ["Fast, by turn 7", "Lots"])
        self.assertEqual((one.deck_path.parent, two.deck_path.parent), (one.session, two.session))
        self.assertNotEqual(one.session, two.session)
        self.assertEqual(sum(argv[0] == "draft" for argv in calls), 2)

    async def test_ctrl_n_opens_a_session_even_while_a_question_is_open(self):
        cli = FakeCli({"interview": (0, QUESTIONS)})
        app = ChatApp(self.data_dir, Settings(), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            first = app.tab
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker))
            await pilot.press("ctrl+n")
            await self.until(pilot, lambda: app.tab is not first and first.waiting)
            self.assertNotIsInstance(app.screen, QuestionPicker)
            self.assertEqual(len(app.tabs), 2)
            await pilot.press("shift+f6")
            await self.until(pilot, lambda: isinstance(app.screen, QuestionPicker))
            await pilot.press("escape")  # With no answer box open, Esc stops the draft.
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertFalse(first.busy)

    async def test_a_first_start_downloads_the_local_data_before_any_deck_request(self):
        lines = [
            "setup: 1/7 Card data: downloading", "setup: 1/7 Card data: 33,000 cards",
            "setup: 2/7 Oracle tags: already here", "setup: 3/7 Rulings: already here",
            "setup: 4/7 Commander Spellbook: downloading",
            "setup: 4/7 Commander Spellbook: skipped: --skip-combos",
            "setup: 5/7 Rules index: already here", "setup: 6/7 Embedding model: already here",
            "setup: 7/7 Card index: downloading", "setup: 7/7 Card index: 28,000 cards embedded",
        ]
        gate = asyncio.Event()
        cli = FakeCli({"setup": (0, {"ready": True, "steps": {}}),
                       "card": (0, trim_card(JODAH))}, wait=gate, lines={"setup": lines})
        missing = dict.fromkeys(SETUP_STEPS, False)
        with patch("mtg_agentic.tui.setup_status", return_value=missing):
            app = ChatApp(self.data_dir, Settings(), runner=cli, setup=True)
            async with app.run_test() as pilot:
                await self.until(pilot, lambda: app.setting_up and cli.calls)
                await self.send(app, pilot, "jodah", settle=False)  # Waits for setup.
                self.assertEqual(cli.calls, [["setup"]])
                gate.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(app.setting_up)
                steps = [(s["name"], s["state"], s["detail"]) for s in app.last.steps]
                await self.send(app, pilot, "jodah")
        self.assertEqual([argv[0] for argv in cli.calls], ["setup", "card"])
        self.assertEqual(steps[0], ("Card data", "done", "33,000 cards"))
        self.assertEqual(steps[3], ("Commander Spellbook", "warn", "skipped: --skip-combos"))
        self.assertEqual({state for _, state, _ in steps}, {"done", "warn"})

    async def test_a_start_with_everything_local_runs_no_setup(self):
        cli = FakeCli({})
        with patch("mtg_agentic.tui.setup_status", return_value=dict.fromkeys(SETUP_STEPS, True)):
            app = ChatApp(self.data_dir, Settings(), runner=cli, setup=True)
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                self.assertFalse(app.setting_up)
        self.assertEqual(cli.calls, [])

    async def test_tab_completes_the_suggested_command(self):
        app = ChatApp(self.data_dir, Settings(), runner=FakeCli({}))
        async with app.run_test() as pilot:
            prompt = app.query_one(Input)
            await pilot.press(*"/sess")
            await pilot.pause(0.1)
            await pilot.press("tab")
            self.assertEqual(prompt.value, "/session")
            self.assertIs(app.focused, prompt)

    async def test_deck_commands_use_the_current_deck_file(self):
        cli = FakeCli({
            "validate": (1, {"valid": False, "errors": []}),
            "bracket-check": (0, {"verdict": "fits"}),
            "review-deck": (0, {"status": "passed", "valid": True, "errors": []}),
        })
        app = ChatApp(self.data_dir, Settings(provider="claude"), runner=cli)
        deck = self.data_dir / "deck.json"
        deck.write_text(json.dumps(DECK), encoding="utf-8")
        async with app.run_test() as pilot:
            await self.send(app, pilot, "/validate")
            self.assertEqual(cli.calls, [])
            await self.send(app, pilot, f"/load {deck}")
            self.assertEqual(app.commander, JODAH["name"])
            await self.send(app, pilot, "/validate")
            await self.send(app, pilot, "/bracket")
            await self.send(app, pilot, "/review on")  # A setting, not a review.
            await self.send(app, pilot, "/review")
        self.assertEqual(cli.calls, [
            ["validate", str(deck)],
            ["bracket-check", str(deck), "--provider", "claude"],
            ["review-deck", str(deck), "--provider", "claude"],
        ])

    async def test_run_cli_streams_a_real_command_and_returns_its_json(self):
        lines = []
        code, schema = await run_cli(["spec-schema"], self.data_dir, lines.append)
        self.assertEqual(code, 0)
        self.assertIn("commander", schema["properties"])
        code, result = await run_cli(["validate", str(self.data_dir / "missing.json")], self.data_dir, lines.append)
        self.assertEqual(code, 2)
        self.assertEqual(result["errors"][0]["error"], "input")


class TuiCommandTests(unittest.TestCase):
    def test_tui_without_textual_explains_how_to_install_it(self):
        output = io.StringIO()
        with patch.dict(sys.modules):
            for name in [n for n in sys.modules if n == "mtg_agentic.tui" or n.split(".")[0] == "textual"]:
                del sys.modules[name]
            sys.modules["textual"] = None
            with contextlib.redirect_stdout(output):
                status = main(["tui"])
        self.assertEqual(status, 2)
        self.assertIn('pip install -e ".[tui]"', json.loads(output.getvalue())["errors"][0]["detail"])


if __name__ == "__main__":
    unittest.main()
