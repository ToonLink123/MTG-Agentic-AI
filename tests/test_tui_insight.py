"""The terminal app's insight into a deck: progress events and the cost meter, /why, versions,
the checkpoint before drafting, and the curve, mana and repeated-problem lines."""

import json
import logging
import tempfile
import unittest
from pathlib import Path

from mtg_agentic.events import JsonLines, event, parse
from mtg_agentic.oracle import OracleStore
from tests.pipeline_fixtures import JODAH, records
from tests.test_tui import DECK, HAS_TEXTUAL, MAPPED, POOL, READY, SPEC, FakeCli, ev, plain

if HAS_TEXTUAL:
    from textual.widgets import Input, Static

    from mtg_agentic.tui import (
        ChatApp,
        CheckpointScreen,
        Settings,
        render_deck,
        render_why,
    )

PLAN = {
    "game_plan": "Ramp into Myojin.", "non_goals": [], "win_conditions": [], "game_changers": [],
    "key_cards": ["Sol Ring"], "core": ["Sol Ring"], "dropped_game_changers": [], "decisions": [],
    "role_targets": {"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "creatures": 15, "protection": 4},
    "targets_reason": "Standard counts.",
}


class EventTests(unittest.TestCase):
    def test_progress_lines_carry_events_as_json(self):
        record = logging.LogRecord("x", logging.INFO, "", 0, "tool: card | %s", ("Sol Ring",), None)
        record.event = event("tool", name="card", detail="Sol Ring")["event"]
        line = JsonLines().format(record)
        self.assertEqual(parse(line), {"type": "tool", "name": "card", "detail": "Sol Ring",
                                       "message": "tool: card | Sol Ring"})
        plain_record = logging.LogRecord("x", logging.INFO, "", 0, "draft: attempt 1", (), None)
        self.assertEqual(parse(JsonLines().format(plain_record))["type"], "log")
        self.assertIsNone(parse("Traceback (most recent call last):"))
        self.assertIsNone(parse('{"no": "type"}'))

    def test_agents_report_structured_events(self):
        from mtg_agentic import activity

        with self.assertLogs("mtg_agentic.activity", "INFO") as logs, activity.agent("mapper", None, "map"):
            activity.tool("card search", "otag:ramp")
        kinds = [(r.event["type"], r.event.get("state")) for r in logs.records]
        self.assertEqual(kinds, [("agent", "start"), ("tool", None), ("agent", "done")])


def lookup(name):
    costs = {"Sol Ring": 1, "Arcane Signet": 2, "Myojin": 7}
    if name in costs:
        return {"name": name, "type_line": "Artifact", "mana_cost": "{1}", "cmc": costs[name]}
    if name == "Island":
        return {"name": name, "type_line": "Basic Land — Island", "mana_cost": "", "cmc": 0}
    return {"name": name, "type_line": "Legendary Creature", "mana_cost": "{4}", "cmc": 4}


@unittest.skipUnless(HAS_TEXTUAL, "the TUI needs the optional textual package")
class DeckInsightTests(unittest.TestCase):
    def test_the_deck_view_shows_the_curve_mana_and_a_minimal_mark_for_code_cards(self):
        deck = {
            **DECK, "cards": {"Sol Ring": 1, "Arcane Signet": 1, "Myojin": 1, "Island": 10},
            "assembly": {"code_added": ["Arcane Signet"]},
            "land_base": {"mana_pips": {"U": 12.0, "W": 3.0}, "source_counts": {"U": 10, "W": 4},
                          "source_targets": {"U": 10, "W": 8}},
        }
        text = plain(render_deck(deck, lookup, Path("deck.json")))
        curve = text.replace("█", "▁")
        self.assertIn("1▁1 2▁1", curve)
        self.assertIn("7+▁1", curve)
        self.assertIn("W 3 pips · 4/8 sources", text)
        self.assertIn("U 12 pips · 10/10 sources", text)
        self.assertIn("Arcane Signet ·", text)
        self.assertNotIn("Sol Ring ·", text)
        self.assertIn("/why CARD", text)

    def test_the_same_problem_on_every_attempt_suggests_a_follow_up(self):
        error = {"error": "role_target", "role": "wipes", "detail": "Need 3; got 1 Oracle-grounded wipes cards"}
        deck = {**DECK, "validation": {"errors": [error]},
                "attempts": [{"attempt": n, "errors": [error]} for n in (1, 2, 3, 4)]}
        text = plain(render_deck(deck, lookup, None))
        self.assertIn("Same problem on 4/4 attempts", text)
        self.assertIn('Try asking: "Add 2 more board wipes"', text)
        once = {**deck, "attempts": [{"attempt": 1, "errors": []}, {"attempt": 2, "errors": [error]}]}
        self.assertNotIn("Same problem", plain(render_deck(once, lookup, None)))

    def test_why_tells_who_picked_a_card_and_where_it_sits_in_the_engine(self):
        swap = {"remove": "Myojin", "add": "Arcane Signet", "reason": "cheaper ramp"}
        deck = {
            **DECK, "cards": {"Sol Ring": 1, "Arcane Signet": 1, "Spell 9": 1, "Island": 10},
            "spec": {**SPEC, "must_include": ["Sol Ring"]},
            "council": {"core": ["Sol Ring"], "key_cards": ["Sol Ring"], "locked": ["Spell 7"],
                        "votes": {"Spell 9": {"keep": 2, "cut": 1, "reasons": ["engine (keep): draws"]}},
                        "decision": {"keep": ["Spell 9"]}},
            "picks": {"core": [], "strong": ["Spell 9"], "flex": []},
            "assembly": {"code_added": [], "code_swaps": [], "dropped": [
                {"card": "Spell 8", "reason": "no room: ranked below the last nonland slot"}]},
            "polish": {"kept": True, "swaps": [swap]},
            "land_base": {"selection_reasons": {"Island": "basic"}},
        }
        pool = {"engine_map": {**MAPPED["engine_map"], "links": [
            {"from": "Sol Ring", "to": "Myojin", "how": "casts it early", "source": "oracle",
             "verified": "supported"}]}, "combos": []}
        def text(name):
            return plain(render_why(name, deck, pool, lookup))
        sol = text("Sol Ring")
        self.assertIn("council's core as a key card", sol)
        self.assertIn("Makes (primary): fast mana", sol)
        self.assertIn("Sol Ring → Myojin: casts it early ✓ rules", sol)
        voted = text("Spell 9")
        self.assertIn("Debated: 2 keep, 1 cut. The orchestrator kept it.", voted)
        self.assertIn("engine (keep): draws", voted)
        self.assertIn("The drafter ranked it strong.", voted)
        self.assertIn("polish pass added it for Myojin: cheaper ramp", text("Arcane Signet"))
        self.assertIn("polish pass cut it for Arcane Signet", text("Myojin"))
        self.assertIn("Code left it out: no room", text("Spell 8"))
        self.assertIn("chosen as basic", text("Island"))


@unittest.skipUnless(HAS_TEXTUAL, "the TUI needs the optional textual package")
class InsightAppTests(unittest.IsolatedAsyncioTestCase):
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
        app.query_one("#prompt", Input).value = text
        await pilot.press("enter")
        await pilot.pause()
        if settle:
            await app.workers.wait_for_complete()
            await pilot.pause()

    def last_text(self, app):
        """Everything the newest reply in the chat shows."""
        turn = list(app.chat.children)[-1]
        return "\n".join(plain(s.content) for s in turn.query(Static))

    async def test_versions_diff_and_undo_walk_the_saved_versions(self):
        folder = self.data_dir / "decks"
        folder.mkdir()
        for number, cards in ((1, {"Sol Ring": 1}), (2, {"Arcane Signet": 1}), (3, {"Arcane Signet": 1, "Myojin": 1})):
            name = "01-deck.json" if number == 1 else f"01-deck-v{number}.json"
            edit = {"reply": f"Change {number}."} if number > 1 else {}
            (folder / name).write_text(json.dumps({**DECK, "cards": cards, "edit": edit}), encoding="utf-8")
        app = ChatApp(self.data_dir, Settings(), runner=FakeCli({}))
        async with app.run_test() as pilot:
            await self.send(app, pilot, "/versions")
            self.assertIn("No deck yet", self.last_text(app))
            await self.send(app, pilot, f"/load {folder / '01-deck-v3.json'}")
            await self.send(app, pilot, "/versions")
            listed = self.last_text(app)
            self.assertIn("v3 (current)", listed)
            self.assertIn("+1 −1", listed)
            self.assertIn("Change 2.", listed)
            await self.send(app, pilot, "/diff v1")
            changed = self.last_text(app)
            self.assertIn("v1 → v3", changed)
            self.assertIn("- Sol Ring", changed)
            self.assertIn("+ Myojin", changed)
            await self.send(app, pilot, "/undo")
            self.assertEqual(app.deck_path.name, "01-deck-v2.json")
            await self.send(app, pilot, "/undo")
            self.assertEqual(app.deck_path.name, "01-deck.json")
            await self.send(app, pilot, "/undo")
            self.assertIn("Nothing to undo", self.last_text(app))
            self.assertEqual(sorted(p.name for p in folder.glob("*.json")),
                             ["01-deck-v2.json", "01-deck-v3.json", "01-deck.json"])

    async def test_the_checkpoint_pins_bans_and_retargets_before_the_council_drafts(self):
        with OracleStore(self.data_dir / "oracle.sqlite3") as store:
            store.import_cards([*records(), {**JODAH, "name": "Sol Ring", "id": "Sol Ring",
                                             "oracle_id": "Sol Ring", "type_line": "Artifact"}])
        usage = ev("usage", "  claude-sonnet-5-5 | 3.0s", model="claude-sonnet-5-5", input=2000,
                   cached=0, written=0, output=100, cost=0.0075)
        cli = FakeCli({
            "interview": (0, READY), "intake": (0, SPEC), "discover": (0, POOL),
            "engine-map": (0, MAPPED), "council-plan": (0, PLAN), "council": (1, DECK),
        }, lines={"council-plan": [usage]})
        settings = Settings(provider="claude", orchestrate=True, checkpoint=True)
        app = ChatApp(self.data_dir, settings, runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            await self.until(pilot, lambda: isinstance(app.screen, CheckpointScreen))
            for command in ("pin spell 5", "pin Not A Card", "ban Sol Ring", "target creatures 50", ""):
                app.screen.query_one(Input).value = command
                await pilot.press("enter")
                await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            session = app.session
            topbar = plain(app.query_one("#topbar", Static).content)
        names = [argv[0] for argv in cli.calls]
        self.assertEqual(names[-3:], ["engine-map", "council-plan", "council"])
        council = cli.calls[-1]
        plan_path = Path(council[council.index("--plan") + 1])
        self.assertEqual(plan_path.name, "01-plan.json")
        pool = json.loads((session / "01-pool.json").read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual((pool["spec"]["must_include"], pool["spec"]["exclude"]), (["Spell 5"], ["Sol Ring"]))
        self.assertEqual([c["name"] for c in pool["cards"]], ["Spell 5"])
        self.assertEqual(plan["core"], [])
        self.assertEqual(plan["role_targets"]["creatures"], 40)  # Kept within its bounds.
        self.assertIn("≈ $0.01 this session", topbar)

    async def test_esc_at_the_checkpoint_stops_the_draft(self):
        cli = FakeCli({"interview": (0, READY), "intake": (0, SPEC), "discover": (0, POOL),
                       "engine-map": (0, MAPPED), "draft": (1, DECK)})
        app = ChatApp(self.data_dir, Settings(orchestrate=False, checkpoint=True), runner=cli)
        async with app.run_test() as pilot:
            app.commander = JODAH["name"]
            await self.send(app, pilot, "Myojin tribal", settle=False)
            await self.until(pilot, lambda: isinstance(app.screen, CheckpointScreen))
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIn("Stopped at the checkpoint.", self.last_text(app))
        self.assertNotIn("draft", [argv[0] for argv in cli.calls])

    async def test_checkpoint_is_a_setting(self):
        app = ChatApp(self.data_dir, Settings(), runner=FakeCli({}))
        async with app.run_test() as pilot:
            self.assertFalse(app.settings.checkpoint)
            await self.send(app, pilot, "/checkpoint on")
            self.assertTrue(app.settings.checkpoint)
            self.assertIn("pauses after the engine map", self.last_text(app))
            await self.send(app, pilot, "/checkpoint maybe")
            self.assertTrue(app.settings.checkpoint)


if __name__ == "__main__":
    unittest.main()
