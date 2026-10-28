"""Council drafting: one orchestrator, several members, one bounded debate.

The orchestrator fixes the game plan and the core (win conditions, Game Changers, key engine
pieces) once, so members never disagree about what the deck is. Members then draft
independently, each with a different focus. Cards every draft chose are kept outright; members
vote once on the rest and the orchestrator settles those slots. Code still owns assembly,
lands, legality and the safety review, exactly as in a single-model draft.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from functools import partial
from typing import Any

from . import activity, prompts
from .assembly import selection_schema
from .discovery import CandidatePool
from .lands import is_land
from .llm import JSONModel, concurrency, parallel, request_json
from .oracle import OracleStore
from .pipeline import build_from_selection, draft_deck, draft_request, game_changer_names
from .review import DeckReviewer
from .roles import oracle_roles
from .schema import NAME_SCHEMA, STRING_LIST, SchemaError, object_schema
from .scryfall import ScryfallClient

ORCHESTRATOR = ("claude-sonnet-5-5", "xhigh")
MEMBER = ("claude-haiku-5-5", "high")
FOCI = {
    "engine": "Engine and synergy: make the core and the game plan come together as often "
    "as possible.",
    "resilience": "Interaction and resilience: removal, protection and answers that keep the "
    "plan alive.",
    "consistency": "Consistency: curve, ramp and card flow so the deck casts its plan on time.",
}
GAME_CHANGER_LIMITS = {1: 0, 2: 0, 3: 3, 4: 100, 5: 100}
STAGES = ("plan", "drafts", "debate", "decision", "final")

log = logging.getLogger(__name__)


def stage(name: str, detail: str) -> None:
    # The TUI advances its checklist on these lines, so keep their shape.
    log.info("council: %d/%d %s: %s", STAGES.index(name) + 1, len(STAGES), name, detail)


def _pick(names: list[str], **limits: int) -> dict[str, Any]:
    items = {"type": "string", "enum": names}
    return {"type": "array", "items": items, "uniqueItems": True, **limits}


def _nonlands(pool: CandidatePool) -> dict[str, Any]:
    return {c.card["name"]: c for c in pool.cards if not is_land(c.card)}


def plan_deck(pool: CandidatePool, model: JSONModel, changers: set[str]) -> dict[str, Any]:
    """The orchestrator's game plan and core, with the Game Changer cap enforced in code."""
    spec = pool.spec
    names = sorted(_nonlands(pool))
    available = [n for n in names if n in changers]
    limit = max(0, GAME_CHANGER_LIMITS[spec.bracket] - int(spec.commander in changers))
    allowed = min(limit, len(available), 6)
    schema = object_schema({
        "game_plan": {"type": "string", "minLength": 1, "maxLength": 1500},
        "non_goals": {**STRING_LIST, "maxItems": 8},
        "win_conditions": _pick(names, minItems=1, maxItems=6),
        "game_changers": _pick(available, maxItems=allowed) if allowed
        else {"type": "array", "items": NAME_SCHEMA, "maxItems": 0},
        "key_cards": _pick(names, maxItems=8),
    })
    payload = {
        "spec": spec.to_dict(), "nonland_slots": 99 - spec.role_targets["lands"],
        "game_changer_limit": limit, "game_changers_available": available,
        "candidates": [c.to_dict() for c in _nonlands(pool).values()],
    }
    with activity.agent("orchestrator", model, "game plan and core cards"):
        plan = request_json(model, [
            {"role": "system", "content": prompts.COUNCIL_PLAN},
            {"role": "user", "content": json.dumps(payload)},
        ], schema)
    picked = [*plan["win_conditions"], *plan["game_changers"], *plan["key_cards"]]
    core = list(dict.fromkeys(picked))
    # A Game Changer can also be listed as a win condition or key card; keep the first ones.
    dropped = [n for n in core if n in changers][limit:]
    return {**plan, "core": [n for n in core if n not in dropped], "dropped_game_changers": dropped}


def _with_core(pool: CandidatePool, core: list[str]) -> CandidatePool:
    """The pool with the core forced in, so assembly and validation keep every core card."""
    must = list(dict.fromkeys([*pool.spec.must_include, *core]))
    return dataclasses.replace(pool, spec=dataclasses.replace(pool.spec, must_include=must))


def _members(count: int) -> list[tuple[str, str]]:
    foci = list(FOCI.items())
    return [foci[i % len(foci)] for i in range(count)]


def draft_members(
    pool: CandidatePool, model: JSONModel, council: dict[str, Any], members: int,
    store: OracleStore, reviewer: DeckReviewer | None,
) -> list[dict[str, Any]]:
    """Independent drafts in parallel; only the model calls leave this thread."""
    shared = draft_request(pool, {"council": council})
    base = [
        {"role": "system", "content": shared[0]["content"] + "\n\n" + prompts.COUNCIL_MEMBER},
        shared[1],
    ]
    schema = selection_schema(pool)

    def ask(focus: str, brief: str) -> dict[str, Any] | None:
        message = {"role": "user", "content": json.dumps({"focus": focus, "brief": brief})}
        try:
            with activity.agent(focus, model, "draft around the core"):
                return request_json(model, [*base, message], schema)
        except SchemaError:
            log.warning("council: the %s draft returned invalid JSON and is left out", focus)
            return None

    seats = _members(members)
    raws = parallel([partial(ask, focus, brief) for focus, brief in seats], concurrency(model))
    drafts = []
    for (focus, _), raw in zip(seats, raws):
        if raw is None:
            continue
        built = build_from_selection(pool, raw["buckets"], store, reviewer=reviewer)
        drafts.append({
            "focus": focus,
            "nonlands": sorted(built["assembly"]["cards"]),
            "role_counts": built["playability"]["roles"]["role_counts"],
            "problems": built["errors"],
        })
    return drafts


def debate(
    pool: CandidatePool, model: JSONModel, council: dict[str, Any], drafts: list[dict[str, Any]],
    contested: list[str], open_slots: int,
) -> list[list[dict[str, str]]]:
    """One round: every member votes keep or cut on each contested card, in parallel."""
    candidates = _nonlands(pool)
    payload = {
        "council": council, "open_slots": open_slots,
        "drafts": [
            {
                "focus": d["focus"], "role_counts": d["role_counts"],
                "role_targets": pool.spec.role_targets, "problems": d["problems"][:8],
            }
            for d in drafts
        ],
        "contested": [
            {
                **candidates[name].to_dict(),
                "chosen_by": [d["focus"] for d in drafts if name in d["nonlands"]],
            }
            for name in contested
        ],
    }
    vote = object_schema({
        "card": {"type": "string", "enum": contested},
        "vote": {"type": "string", "enum": ["keep", "cut"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    })
    schema = object_schema({"votes": {
        "type": "array", "items": vote, "minItems": len(contested), "maxItems": len(contested),
    }})
    base = [
        {"role": "system", "content": prompts.COUNCIL_CRITIQUE},
        {"role": "user", "content": json.dumps(payload)},
    ]

    def ask(focus: str, brief: str) -> list[dict[str, str]]:
        message = {"role": "user", "content": json.dumps({"focus": focus, "brief": brief})}
        try:
            with activity.agent(focus, model, f"vote on {len(contested)} contested cards"):
                votes = request_json(model, [*base, message], schema)["votes"]
        except SchemaError:
            log.warning("council: the %s member's votes were invalid and are left out", focus)
            return []
        # One vote per card: a repeated card keeps its first vote.
        unique = {}
        for item in votes:
            unique.setdefault(item["card"], {**item, "focus": focus})
        return list(unique.values())

    seats = _members(len(drafts))
    return parallel([partial(ask, focus, brief) for focus, brief in seats], concurrency(model))


def decide(
    pool: CandidatePool, model: JSONModel, council: dict[str, Any], locked: list[str],
    contested: list[str], tally: dict[str, dict[str, Any]], open_slots: int,
) -> dict[str, Any]:
    """The orchestrator settles the contested slots; a malformed answer falls back to the vote."""
    candidates = _nonlands(pool)
    locked_roles = Counter(
        role for name in locked
        for role in oracle_roles(candidates[name].card, candidates[name].roles)
    )
    payload = {
        "council": council, "locked": locked, "open_slots": open_slots,
        "role_targets": pool.spec.role_targets, "locked_role_counts": dict(locked_roles),
        "contested": [{**candidates[name].to_dict(), **tally[name]} for name in contested],
    }
    schema = object_schema({
        "keep": _pick(contested, maxItems=len(contested)),
        "notes": {"type": "string", "maxLength": 1500},
    })
    by_votes = sorted(contested, key=lambda n: (-tally[n]["keep"], tally[n]["cut"], n))
    try:
        with activity.agent("orchestrator", model, f"settle {len(contested)} contested cards"):
            decision = request_json(model, [
                {"role": "system", "content": prompts.COUNCIL_DECIDE},
                {"role": "user", "content": json.dumps(payload)},
            ], schema)
    except SchemaError:
        log.warning("council: the decision was invalid JSON; following the vote")
        decision = {"keep": by_votes[:open_slots], "notes": "Fell back to the members' vote."}
    # Unchosen contested cards rank below the decision but above the rest of the pool.
    ranked = decision["keep"] + [n for n in by_votes if n not in decision["keep"]]
    return {**decision, "ranked": ranked}


def council_draft(
    pool: CandidatePool, orchestrator: JSONModel, member: JSONModel, client: ScryfallClient,
    store: OracleStore, *, members: int = 3, max_repairs: int = 3,
    reviewer: DeckReviewer | None = None,
) -> dict[str, Any]:
    if type(members) is not int or not 2 <= members <= 6:
        raise ValueError("A council needs 2 to 6 members")
    checked = CandidatePool.from_dict(pool.to_dict(), store)
    changers = game_changer_names(checked, store, reviewer)
    lead = getattr(orchestrator, "model", "orchestrator")
    stage("plan", f"{lead} sets the game plan and core")
    plan = plan_deck(checked, orchestrator, changers)
    council = {key: plan[key] for key in ("game_plan", "non_goals", "core")}
    working = _with_core(checked, plan["core"])
    slots = 99 - working.spec.role_targets["lands"]

    stage("drafts", f"{members} members draft around {len(plan['core'])} core cards")
    drafts = draft_members(working, member, council, members, store, reviewer)
    record: dict[str, Any] = {
        "orchestrator": getattr(orchestrator, "model", None),
        "member": getattr(member, "model", None),
        **plan,
        "drafts": [
            {"focus": d["focus"], "role_counts": d["role_counts"], "problems": len(d["problems"])}
            for d in drafts
        ],
    }
    if not drafts:
        stage("final", "no member draft succeeded; the orchestrator drafts alone")
        deck = draft_deck(working, orchestrator, client, store, max_repairs=max_repairs,
                          reviewer=reviewer, context={"council": council}, agent="orchestrator")
        return {**deck, "council": {**record, "fallback": "orchestrator_alone"}}

    chosen = [set(d["nonlands"]) for d in drafts]
    locked = sorted(set.intersection(*chosen))
    contested = sorted(set.union(*chosen) - set(locked))
    open_slots = slots - len(locked)
    votes: list[list[dict[str, str]]] = []
    decision = {"keep": [], "notes": "", "ranked": []}
    tally: dict[str, dict[str, Any]] = {}
    if contested:
        agreed = f"{len(locked)} cards agreed; {len(contested)} contested for {open_slots} slots"
        stage("debate", agreed)
        votes = debate(working, member, council, drafts, contested, open_slots)
        tally = {name: {"keep": 0, "cut": 0, "reasons": []} for name in contested}
        for item in (v for member_votes in votes for v in member_votes):
            entry = tally[item["card"]]
            entry[item["vote"]] += 1
            entry["reasons"].append(f"{item['focus']} ({item['vote']}): {item['reason']}")
        stage("decision", f"{lead} settles {len(contested)} contested cards")
        decision = decide(working, orchestrator, council, locked, contested, tally, open_slots)
    else:
        stage("debate", "every draft agreed; nothing to debate")
        stage("decision", "nothing contested")

    order = list(dict.fromkeys([*plan["core"], *locked, *decision["ranked"]]))
    stage("final", "lands, validation and safety review")
    deck = draft_deck(working, orchestrator, client, store, max_repairs=max_repairs,
                      reviewer=reviewer, initial=order, context={"council": council},
                      agent="orchestrator")
    record.update(
        locked=locked, contested=contested, open_slots=open_slots, votes=tally,
        decision={"keep": decision["keep"], "notes": decision.get("notes", "")},
    )
    return {**deck, "council": record}
