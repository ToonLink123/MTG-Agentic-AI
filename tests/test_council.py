import argparse
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mtg_agentic import prompts
from mtg_agentic.council import council_draft, plan_deck
from mtg_agentic.oracle import OracleStore
from mtg_agentic.schema import SchemaError
from tests.fixtures import card
from tests.pipeline_fixtures import JODAH, FakeCards, draft, make_pool, make_spec, playable_records

PLAN = {
    "game_plan": "Cast Myojin from hand with Jodah, keep their counters, win in combat.",
    "non_goals": ["token swarm"],
    "win_conditions": ["Spell 1"],
    "game_changers": [],
    "key_cards": ["Spell 2", "Spell 3"],
    "role_targets": {"ramp": 10, "draw": 10, "removal": 8, "wipes": 3, "creatures": 15,
                     "protection": 4},
    "targets_reason": "A midrange Myojin deck with the usual counts.",
}


def picks(*numbers):
    selection = draft(count=0)
    selection["picks"]["flex"] = [f"Spell {n}" for n in numbers]
    return selection


class CouncilModel:
    """Answers each council prompt; member drafts are chosen by focus."""

    def __init__(self, drafts, *, plan=PLAN, keep=None, concurrency=3):
        self.drafts, self.plan, self.keep = drafts, plan, keep
        self.concurrency = concurrency
        self.calls, self.lock, self.running, self.peak = [], threading.Lock(), 0, 0

    def complete_json(self, messages, schema):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        kind = (
            "plan" if prompts.COUNCIL_PLAN in system
            else "draft" if prompts.COUNCIL_MEMBER in system
            else "vote" if prompts.COUNCIL_CRITIQUE in system
            else "decide" if prompts.COUNCIL_DECIDE in system
            else "repair"
        )
        with self.lock:
            self.calls.append(kind)
            self.running += 1
            self.peak = max(self.peak, self.running)
        time.sleep(0.01)
        with self.lock:
            self.running -= 1
        focus = json.loads(messages[-1]["content"]).get("focus")
        if kind == "plan":
            return self.plan
        if kind == "draft":
            result = self.drafts[focus]
            if isinstance(result, Exception):
                raise result
            return result
        payload = json.loads(messages[1]["content"])
        if kind == "vote":
            return {"votes": [
                {"card": c["name"], "vote": "keep" if len(c["chosen_by"]) > 1 else "cut",
                 "reason": f"{focus} view"}
                for c in payload["contested"]
            ]}
        if kind == "decide":
            contested = [c["name"] for c in payload["contested"]]
            return {"keep": self.keep if self.keep is not None else contested[: payload["open_slots"]],
                    "notes": "followed the vote"}
        return draft()


class CouncilTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.store = OracleStore(self.path / "oracle.sqlite3")
        self.addCleanup(self.store.close)
        self.records = playable_records()
        self.store.import_cards(self.records)
        self.client = FakeCards(self.records)
        self.pool = make_pool(self.store)

    def run_council(self, model, **kwargs):
        with patch("mtg_agentic.scryfall.urlopen", side_effect=AssertionError("Unexpected HTTP")):
            return council_draft(self.pool, model, model, self.client, self.store, **kwargs)

    def test_plan_then_parallel_drafts_then_one_debate_round_then_a_decision(self):
        model = CouncilModel({
            "engine": picks(*range(63)),
            "resilience": picks(*range(60), 63, 64, 65),
            "consistency": picks(*range(60), 66, 67, 68),
        })
        with self.assertLogs("mtg_agentic.council", "INFO") as logs:
            deck = self.run_council(model)
        council = deck["council"]
        self.assertEqual(model.calls[0], "plan")
        self.assertEqual(sorted(model.calls[1:4]), ["draft"] * 3)
        self.assertEqual(model.calls[4:7], ["vote"] * 3)
        self.assertEqual(model.calls[7], "decide")
        self.assertGreater(model.peak, 1)
        stages = [line.split("council: ")[1][:3] for line in logs.output]
        self.assertEqual(stages, ["1/5", "2/5", "3/5", "4/5", "5/5"])
        self.assertTrue(deck["valid"], deck["validation"]["errors"])
        self.assertTrue(set(council["core"]) <= set(deck["cards"]))
        self.assertEqual(council["core"], ["Spell 1", "Spell 2", "Spell 3"])
        self.assertTrue(set(council["core"]) <= set(deck["spec"]["must_include"]))
        self.assertTrue(set(council["core"]) <= set(council["locked"]))
        self.assertFalse(set(council["locked"]) & set(council["contested"]))
        self.assertEqual(council["open_slots"], 63 - len(council["locked"]))
        for name in council["contested"]:
            self.assertEqual(sum(council["votes"][name][v] for v in ("keep", "cut")), 3)

    def test_members_outside_picks_join_the_debate_like_candidates(self):
        extra = card("Outside Engine", colors="U", mana_cost="{1}{U}", cmc=2,
                     text="Draw two cards.")
        self.store.import_cards([*self.records, extra])
        self.pool = make_pool(self.store)
        engine = {**picks(*range(62)),
                  "outside": [{"name": "Outside Engine", "tier": "flex", "reason": "fits"}]}
        model = CouncilModel({
            "engine": engine, "resilience": picks(*range(63)), "consistency": picks(*range(63)),
        })
        deck = self.run_council(model)
        council = deck["council"]
        self.assertEqual(council["drafts"][0]["outside"],
                         {"accepted": ["Outside Engine"], "rejected": []})
        self.assertIn("Outside Engine", council["contested"])
        self.assertTrue(deck["valid"], deck["validation"]["errors"])

    def test_unanimous_drafts_skip_the_debate(self):
        same = picks(*range(63))
        model = CouncilModel({"engine": same, "resilience": same, "consistency": same})
        deck = self.run_council(model)
        self.assertEqual(model.calls.count("vote") + model.calls.count("decide"), 0)
        self.assertEqual(deck["council"]["contested"], [])
        self.assertTrue(deck["valid"], deck["validation"]["errors"])

    def test_a_malformed_member_draft_is_left_out(self):
        model = CouncilModel({
            "engine": picks(*range(63)),
            "resilience": SchemaError([{"error": "json", "detail": "bad"}]),
            "consistency": picks(*range(60), 66, 67, 68),
        })
        deck = self.run_council(model)
        self.assertEqual([d["focus"] for d in deck["council"]["drafts"]], ["engine", "consistency"])
        self.assertEqual(model.calls.count("vote"), 2)

    def test_an_approved_plan_is_checked_and_not_planned_again(self):
        from mtg_agentic.council import given_plan

        approved = {**plan_deck(self.pool, CouncilModel({}), set()), "core": ["Spell 1", "Not A Card"],
                    "role_targets": {**PLAN["role_targets"], "creatures": 99, "bogus": 3}}
        checked = given_plan(self.pool, approved, {"Spell 1"})
        self.assertEqual(checked["core"], ["Spell 1"])  # A pool card; the made-up name is gone.
        low = make_pool(self.store, make_spec(bracket=2))
        self.assertEqual(given_plan(low, approved, {"Spell 1"})["core"], [])  # No Game Changers.
        self.assertEqual(checked["role_targets"]["creatures"], 40)
        self.assertNotIn("bogus", checked["role_targets"])
        with self.assertRaises(ValueError):
            given_plan(self.pool, {"core": []}, set())
        model = CouncilModel({f: picks(*range(63)) for f in ("engine", "resilience", "consistency")})
        deck = self.run_council(model, plan={**approved, "role_targets": PLAN["role_targets"]})
        self.assertNotIn("plan", model.calls)
        self.assertTrue(deck["valid"], deck["validation"]["errors"])

    def test_game_changers_past_the_bracket_limit_leave_the_core(self):
        plan = {**PLAN, "win_conditions": ["Spell 1"], "game_changers": ["Spell 2", "Spell 3"],
                "key_cards": ["Spell 4", "Spell 5"]}
        changers = {"Spell 1", "Spell 2", "Spell 3", "Spell 4"}
        result = plan_deck(self.pool, CouncilModel({}, plan=plan), changers)
        self.assertEqual(result["core"], ["Spell 1", "Spell 2", "Spell 3", "Spell 5"])
        self.assertEqual(result["dropped_game_changers"], ["Spell 4"])


class CouncilCliTests(unittest.TestCase):
    def test_claude_council_defaults_to_sonnet_leading_haiku(self):
        from mtg_agentic.__main__ import _council_models

        args = argparse.Namespace(
            provider="claude", model=None, orchestrator_model=None, member_model=None,
            orchestrator_effort="xhigh", member_effort="high",
        )
        with patch("mtg_agentic.__main__._make_model", side_effect=lambda a: (a.model, a.effort)):
            self.assertEqual(
                _council_models(args),
                (("claude-sonnet-5-5", "xhigh"), ("claude-haiku-5-5", "high")),
            )
            args.provider, args.model = "ollama", "qwen2.5:7b"
            self.assertEqual(
                _council_models(args), (("qwen2.5:7b", "xhigh"), ("qwen2.5:7b", "high"))
            )

    def test_council_plan_command_prints_a_plan_the_council_command_accepts(self):
        from mtg_agentic.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with OracleStore(path / "oracle.sqlite3") as store:
                store.import_cards(playable_records())
                (path / "pool.json").write_text(json.dumps(make_pool(store).to_dict()), encoding="utf-8")
            model = CouncilModel({focus: picks(*range(63)) for focus in ("engine", "resilience", "consistency")})
            with patch("mtg_agentic.__main__._make_model", return_value=model):
                status = main(["--quiet", "--data-dir", str(path), "council-plan", "--pool",
                               str(path / "pool.json"), "--output", str(path / "plan.json")])
                plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
                plan["role_targets"]["creatures"] = 12
                (path / "edited.json").write_text(json.dumps(plan), encoding="utf-8")
                status_two = main(["--quiet", "--data-dir", str(path), "council", "--pool",
                                   str(path / "pool.json"), "--plan", str(path / "edited.json"),
                                   "--skip-review", "--output", str(path / "deck.json")])
            deck = json.loads((path / "deck.json").read_text(encoding="utf-8"))
        self.assertEqual((status, status_two), (0, 0))
        self.assertEqual(plan["core"], ["Spell 1", "Spell 2", "Spell 3"])
        self.assertEqual(model.calls.count("plan"), 1)  # The council did not plan again.
        self.assertEqual(deck["spec"]["role_targets"]["creatures"], 12)

    def test_council_command_drafts_a_saved_pool(self):
        from mtg_agentic.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            records = playable_records()
            with OracleStore(path / "oracle.sqlite3") as store:
                store.import_cards(records)
                (path / "pool.json").write_text(json.dumps(make_pool(store).to_dict()), encoding="utf-8")
            model = CouncilModel({focus: picks(*range(63)) for focus in ("engine", "resilience", "consistency")})
            output = path / "deck.json"
            with patch("mtg_agentic.__main__._make_model", return_value=model):
                status = main([
                    "--quiet", "--data-dir", str(path), "council", "--pool", str(path / "pool.json"),
                    "--skip-review", "--output", str(output),
                ])
            deck = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 0, deck.get("validation"))
        self.assertEqual(deck["commander"], JODAH["name"])
        self.assertEqual(deck["council"]["core"], ["Spell 1", "Spell 2", "Spell 3"])


if __name__ == "__main__":
    unittest.main()
