"""The drafter with tools: it checks its picks with code before it commits.

Instead of drafting blind and learning from a failed attempt, a tool-capable drafter (Claude)
can ask code what its picks would build (check_draft: the land count from its curve, role counts,
protection, every problem, and what code would add or leave out), search the whole local card
data for what the pool lacks, read cards and look up combos, all before finish. Each finish is
one attempt of the Drafting session: code builds it, checks it and runs the safety review, and
any problems come back as the tool's error result, so the drafter fixes them in the same
conversation. The session keeps the best attempt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import prompts
from .agent import Finish, Tool, ToolError, run_tools
from .assembly import code_report, ranked_picks
from .lands import is_land, nonland_slots
from .pipeline import Drafting
from .roles import role_policy
from .schema import DRAFT_TIERS, NAME_SCHEMA, OUTSIDE_SCHEMA, TIER_PICKS, object_schema
from .scryfall import ScryfallError
from .sources import SourceError
from .toolbox import CARDS_INPUT, SEARCH_INPUT, CardTools, Toolbox

log = logging.getLogger(__name__)
PICKS_INPUT = object_schema({
    "picks": object_schema({
        tier: {"type": "array", "items": NAME_SCHEMA, "maxItems": TIER_PICKS, "uniqueItems": True}
        for tier in DRAFT_TIERS
    }),
    "outside": OUTSIDE_SCHEMA,
})


class DraftTools(CardTools):
    """The drafter's card tools: a card a search or lookup finds may be picked like a candidate
    when it is an eligible nonland; code admits it to the pool when the drafter uses it."""

    def __init__(self, session: Drafting, toolbox: Toolbox):
        super().__init__(toolbox, session.pool.spec)
        self.session = session

    def candidates(self) -> set[str]:
        return {c.card["name"] for c in self.session.pool.cards if not is_land(c.card)}

    def pickable(self, name: str) -> bool:
        return name in self.addable or name in self.candidates()

    def selection(self, args: dict[str, Any], *, record: bool) -> dict[str, list[str]]:
        """The picks as tiers. Cards a tool offered join the pool first; a name that is neither
        a candidate nor offered is a problem the drafter must fix."""
        candidates = self.candidates()
        names = [name for tier in DRAFT_TIERS for name in args["picks"][tier]]
        unknown = [n for n in dict.fromkeys(names) if n not in candidates and n not in self.addable]
        if unknown:
            raise ToolError({"problems": [
                f"{name}: not a candidate and not a card a search or lookup offered. Copy names "
                "exactly, search for the card, or list it under outside." for name in unknown[:12]
            ]})
        offered = [{"name": n, "tier": "strong", "reason": "offered by a tool"}
                   for n in dict.fromkeys(names) if n not in candidates]
        if offered:
            self.session.selection({"picks": {}, "outside": offered}, record=False)
        return self.session.selection(args, record=record)


def _report(session: Drafting, built: dict[str, Any]) -> dict[str, Any]:
    """What check_draft shows: the build without its full Oracle data."""
    spec = built["pool"].spec
    store = session.store
    curve: dict[str, int] = {}
    for name in built["assembly"].get("cards", {}):
        card = store.get_card(name)
        value = int(card.get("cmc") or 0) if card else 0
        key = str(min(value, 7)) + ("+" if value >= 7 else "")
        curve[key] = curve.get(key, 0) + 1
    roles = built["playability"]["roles"]
    return {
        "ready_for_review": not built["errors"],
        "land_count": spec.role_targets["lands"], "nonland_slots": nonland_slots(spec),
        "land_plan": built["land_plan"],
        "role_counts": {k: v for k, v in roles["role_counts"].items() if k != "lands"},
        "role_minimums": {k: v for k, v in role_policy(spec)["minimums"].items() if k != "lands"},
        "protection": {k: roles["protection"][k] for k in ("total", "instant_independent", "mechanisms")},
        "curve": dict(sorted(curve.items())),
        "problems": built["errors"][:15],
        "code_changes": code_report(built["assembly"]),
    }


def agentic_draft(
    session: Drafting, model: Any, toolbox: Toolbox, base: list[dict[str, str]], *,
    agent: str = "drafter", max_steps: int = 16,
) -> None:
    """Draft with tools until a finish passes, the attempts run out or the steps do. Attempts
    land in `session`, whose best one the caller returns."""
    tools = DraftTools(session, toolbox)

    def guarded(run):
        def call(args: dict[str, Any]) -> Any:
            try:
                return run(args)
            except (ScryfallError, SourceError, ValueError) as exc:
                raise ToolError({"error": str(exc)}) from None
        return call

    def check_draft(args: dict[str, Any]) -> dict[str, Any]:
        return _report(session, session.build(tools.selection(args, record=False)))

    def combos(args: dict[str, Any]) -> dict[str, Any]:
        tiers = tools.selection(args, record=False)
        names = tools.candidates()
        picks = [n for n in ranked_picks(tiers) if n in names][:nonland_slots(session.pool.spec)]
        return tools.combos(dict.fromkeys(picks, 1))

    def finish(args: dict[str, Any]) -> Finish:
        deck = session.attempt(tools.selection(args, record=True), raw=args)
        if deck["valid"] or not session.left:
            return Finish(deck)
        if session.stuck():
            log.info("draft: the same problems three attempts running; stopping")
            return Finish(deck)
        feedback = session.feedback()
        feedback.pop("previous_picks")  # The drafter just sent them.
        feedback["instruction"] = (
            "This finish left the problems above. Fix every one, check_draft again, then "
            f"finish with the complete revised picks; {session.left} finish(es) left."
        )
        raise ToolError(feedback)

    toolset = [
        Tool("check_draft", "Build these picks with code, without the safety review: the land "
             "count from your curve, role counts against the minimums, protection, the curve, "
             "every problem, and what code would add or leave out.", PICKS_INPUT,
             guarded(check_draft)),
        Tool("search_cards", "A search in Scryfall syntax over the whole local card data, "
             "already limited to the commander's colors and Commander legality. Results marked "
             "addable may go in your picks.", SEARCH_INPUT,
             guarded(lambda a: tools.search_cards(a["query"]))),
        Tool("card", "Oracle text, roles and whether each named card may be picked, up to 10.",
             CARDS_INPUT, guarded(lambda a: tools.card(a["cards"]))),
        Tool("combos", "Known Commander Spellbook combos among these picks and the commander.",
             PICKS_INPUT, guarded(combos)),
        Tool("finish", "Your final picks. Code builds them, runs every check and the safety "
             "review; any problems come back for you to fix, then finish again.", PICKS_INPUT,
             guarded(finish)),
    ]
    payload = json.loads(base[2]["content"])
    if session.attempts:
        # A council's ranked picks already failed: start from what code found.
        payload["previous_attempt"] = session.feedback()
    messages = [
        base[0],
        {"role": "system", "content": base[1]["content"] + "\n\n" + prompts.DRAFT_AGENT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    task = "repair the ranked picks with tools" if session.attempts else "draft the nonlands with checks"
    run_tools(model, messages, toolset, max_steps=max_steps, agent=agent, task=task)
