"""Council drafting: one orchestrator, several members, one bounded debate.

The orchestrator fixes the game plan, the core (win conditions, Game Changers, key engine pieces)
and the role targets the plan needs once, and decides any interview question the player left to
the agents, so members never disagree about what the deck is. Every member also drafts from the
same engine map, when the pool has one. Members then rank their picks independently, each with a
different focus. Cards every member ranks within the nonland slots are kept outright; members vote
once on the rest and the orchestrator settles those slots. Members are compared by their own
picks, never by what code filled in around them. Code still owns assembly, lands, legality and
the safety review, exactly as in a single-model draft.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from functools import partial
from typing import Any

from . import activity, prompts
from .assembly import GAME_CHANGER_LIMITS, ranked_picks, selection_schema
from .cards import name_key
from .discovery import CandidatePool
from .events import event
from .lands import is_land, nonland_slots
from .llm import JSONModel, concurrency, parallel, request_json
from .oracle import OracleStore
from .pipeline import (
    admit_outside,
    build_from_selection,
    draft_deck,
    draft_request,
    game_changer_names,
    pool_context,
)
from .review import DeckReviewer
from .roles import ROLE_BOUNDS, bounded_targets, oracle_roles
from .schema import DRAFT_TIERS, NAME_SCHEMA, STRING_LIST, SchemaError, object_schema
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
STAGES = ("plan", "drafts", "debate", "decision", "final")

log = logging.getLogger(__name__)


def stage(name: str, detail: str) -> None:
    # The TUI advances its checklist on these events.
    index = STAGES.index(name) + 1
    log.info("council: %d/%d %s: %s", index, len(STAGES), name, detail, extra=event(
        "stage", scope="council", index=index, total=len(STAGES), name=name, detail=detail,
    ))


def _pick(names: list[str], **limits: int) -> dict[str, Any]:
    items = {"type": "string", "enum": names}
    return {"type": "array", "items": items, "uniqueItems": True, **limits}


def _nonlands(pool: CandidatePool) -> dict[str, Any]:
    return {c.card["name"]: c for c in pool.cards if not is_land(c.card)}


def role_bounds(spec) -> dict[str, tuple[int, int]]:
    """The role targets a plan may set: ROLE_BOUNDS, with 6 protection for a commander the
    plan depends on."""
    bounds = dict(ROLE_BOUNDS)
    if spec.commander_dependency == "high":
        bounds["protection"] = (6, ROLE_BOUNDS["protection"][1])
    return bounds


def plan_deck(pool: CandidatePool, model: JSONModel, changers: set[str]) -> dict[str, Any]:
    """The orchestrator's game plan, core and role targets, with the Game Changer cap and the
    role bounds enforced in code."""
    spec = pool.spec
    names = sorted(_nonlands(pool))
    available = [n for n in names if n in changers]
    limit = max(0, GAME_CHANGER_LIMITS[spec.bracket] - int(spec.commander in changers))
    allowed = min(limit, len(available), 6)
    bounds = role_bounds(spec)
    fields = {
        "game_plan": {"type": "string", "minLength": 1, "maxLength": 1500},
        "non_goals": {**STRING_LIST, "maxItems": 8},
        "win_conditions": _pick(names, minItems=1, maxItems=6),
        "game_changers": _pick(available, maxItems=allowed) if allowed
        else {"type": "array", "items": NAME_SCHEMA, "maxItems": 0},
        "key_cards": _pick(names, maxItems=8),
        "role_targets": object_schema({
            role: {"type": "integer", "minimum": low, "maximum": high}
            for role, (low, high) in bounds.items()
        }),
        "targets_reason": {"type": "string", "minLength": 1, "maxLength": 600},
    }
    # Interview questions the player left to the agents: the orchestrator decides each once.
    deferred = spec.brief["deferred"]
    if deferred:
        fields["decisions"] = {
            "type": "array", "minItems": len(deferred), "maxItems": len(deferred),
            "items": object_schema({
                "question": {"type": "string", "enum": deferred},
                "decision": {"type": "string", "minLength": 1, "maxLength": 400},
            }),
        }
    payload = {
        "spec": spec.to_dict(), "nonland_slots": nonland_slots(spec),
        "game_changer_limit": limit, "game_changers_available": available,
        "role_bounds": {role: list(span) for role, span in bounds.items()},
    }
    task = "game plan, core cards, role targets" + (f" and {len(deferred)} open questions" if deferred else "")
    with activity.agent("orchestrator", model, task):
        plan = request_json(model, [
            pool_context(pool),
            {"role": "system", "content": prompts.COUNCIL_PLAN},
            {"role": "user", "content": json.dumps(payload)},
        ], object_schema(fields))
    decisions: dict[str, dict[str, str]] = {}
    for item in plan.get("decisions", []):
        decisions.setdefault(item["question"], item)
    plan = {**plan, "decisions": list(decisions.values())}
    picked = [*plan["win_conditions"], *plan["game_changers"], *plan["key_cards"]]
    core = list(dict.fromkeys(picked))
    # A Game Changer can also be listed as a win condition or key card; keep the first ones.
    dropped = [n for n in core if n in changers][limit:]
    targets = bounded_targets(plan["role_targets"], spec.commander_dependency)
    return {
        **plan, "role_targets": targets,
        "core": [n for n in core if n not in dropped], "dropped_game_changers": dropped,
    }


PLAN_KEYS = ("game_plan", "non_goals", "win_conditions", "game_changers", "key_cards",
             "role_targets", "targets_reason", "decisions", "core", "dropped_game_changers")


def given_plan(pool: CandidatePool, plan: Any, changers: set[str]) -> dict[str, Any]:
    """A plan made earlier (council-plan) and perhaps edited by the player at a checkpoint,
    checked like a fresh one: only pool cards stay in its lists, Game Changers past the bracket
    limit leave the core, and role targets are clamped to their bounds."""
    if not isinstance(plan, dict) or any(key not in plan for key in PLAN_KEYS):
        raise ValueError("A council plan needs " + ", ".join(PLAN_KEYS))
    names = set(_nonlands(pool))
    spec = pool.spec
    lists = {
        key: [n for n in dict.fromkeys(plan[key]) if isinstance(n, str) and n in names]
        for key in ("win_conditions", "game_changers", "key_cards", "core")
    }
    limit = max(0, GAME_CHANGER_LIMITS[spec.bracket] - int(spec.commander in changers))
    dropped = [n for n in lists["core"] if n in changers][limit:]
    targets = {r: v for r, v in plan["role_targets"].items() if r in ROLE_BOUNDS and type(v) is int}
    return {
        **plan, **lists, "core": [n for n in lists["core"] if n not in dropped],
        "dropped_game_changers": [*plan["dropped_game_changers"], *dropped],
        "role_targets": bounded_targets(targets, spec.commander_dependency),
    }


def _apply_plan(pool: CandidatePool, plan: dict[str, Any]) -> CandidatePool:
    """The pool with the core forced in, so assembly and validation keep every core card, and
    the plan's role targets in place of intake's (the land count stays as it was)."""
    spec = pool.spec
    must = list(dict.fromkeys([*spec.must_include, *plan["core"]]))
    targets = {**spec.role_targets, **plan.get("role_targets", {})}
    return dataclasses.replace(pool, spec=dataclasses.replace(spec, must_include=must, role_targets=targets))


def _members(count: int) -> list[tuple[str, str]]:
    foci = list(FOCI.items())
    return [foci[i % len(foci)] for i in range(count)]


def _tiers(raw: dict[str, Any], accepted: list[str]) -> dict[str, list[str]]:
    """A member's tiers, with its accepted outside picks at the end of their tiers."""
    tiers = {tier: list(raw["picks"][tier]) for tier in DRAFT_TIERS}
    tier_of = {name_key(p["name"]): p["tier"] for p in raw.get("outside", [])}
    for name in accepted:
        tiers[tier_of.get(name_key(name), "strong")].append(name)
    return tiers


def draft_members(
    pool: CandidatePool, model: JSONModel, council: dict[str, Any], members: int,
    store: OracleStore, reviewer: DeckReviewer | None,
) -> tuple[list[dict[str, Any]], CandidatePool]:
    """Independent drafts in parallel; only the model calls leave this thread. Returns the
    drafts and the pool with every member's accepted outside picks, which the debate and the
    final draft then use like any other candidate. A draft's "picks" are the member's own deck:
    the must-include cards, then its ranked picks, up to the nonland slots."""
    shared = draft_request(pool, {"council": council})
    # The pool comes first, byte for byte the same for every member, so all but the first read
    # it from the prompt cache.
    base = [
        shared[0],
        {"role": "system", "content": shared[1]["content"] + "\n\n" + prompts.COUNCIL_MEMBER},
        shared[2],
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
    admitted: dict[int, tuple[list[str], list[dict[str, str]]]] = {}
    for seat, raw in enumerate(raws):
        if raw is not None and raw.get("outside"):
            pool, accepted, rejected = admit_outside(pool, raw["outside"], store)
            admitted[seat] = (accepted, rejected)
    candidates = _nonlands(pool)
    slots = nonland_slots(pool.spec)
    drafts = []
    for seat, ((focus, _), raw) in enumerate(zip(seats, raws)):
        if raw is None:
            continue
        accepted, rejected = admitted.get(seat, ([], []))
        tiers = _tiers(raw, accepted)
        own = [n for n in dict.fromkeys([*pool.spec.must_include, *ranked_picks(tiers)]) if n in candidates]
        built = build_from_selection(pool, tiers, store, reviewer=reviewer)
        drafts.append({
            "focus": focus,
            "picks": own[:slots],
            "tiers": tiers,
            "role_counts": built["playability"]["roles"]["role_counts"],
            "problems": built["errors"],
            "outside": {"accepted": accepted, "rejected": rejected},
        })
    return drafts, pool


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
                "chosen_by": [d["focus"] for d in drafts if name in d["picks"]],
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
    reviewer: DeckReviewer | None = None, toolbox: Any = None, plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The council's draft. With `plan` (from council-plan, perhaps edited at a checkpoint)
    the orchestrator does not plan again."""
    if type(members) is not int or not 2 <= members <= 6:
        raise ValueError("A council needs 2 to 6 members")
    checked = CandidatePool.from_dict(pool.to_dict(), store)
    changers = game_changer_names(checked, store, reviewer)
    lead = getattr(orchestrator, "model", "orchestrator")
    if plan is not None:
        stage("plan", "the approved game plan, core and role targets")
        plan = given_plan(checked, plan, changers)
    else:
        stage("plan", f"{lead} sets the game plan, core and role targets")
        plan = plan_deck(checked, orchestrator, changers)
    council = {key: plan[key] for key in ("game_plan", "non_goals", "core", "decisions")}
    working = _apply_plan(checked, plan)
    slots = nonland_slots(working.spec)

    stage("drafts", f"{members} members draft around {len(plan['core'])} core cards")
    drafts, working = draft_members(working, member, council, members, store, reviewer)
    record: dict[str, Any] = {
        "orchestrator": getattr(orchestrator, "model", None),
        "member": getattr(member, "model", None),
        **plan,
        "drafts": [
            {"focus": d["focus"], "role_counts": d["role_counts"], "problems": len(d["problems"]),
             "outside": d["outside"]}
            for d in drafts
        ],
    }
    if not drafts:
        stage("final", "no member draft succeeded; the orchestrator drafts alone")
        deck = draft_deck(working, orchestrator, client, store, max_repairs=max_repairs,
                          reviewer=reviewer, context={"council": council}, agent="orchestrator",
                          toolbox=toolbox)
        return {**deck, "council": {**record, "fallback": "orchestrator_alone"}}

    chosen = [set(d["picks"]) for d in drafts]
    locked = sorted(set.intersection(*chosen))
    contested = sorted(set.union(*chosen) - set(locked))
    open_slots = max(0, slots - len(locked))
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

    # Code may swap only the flex tier: the core, the agreed cards and the cards the decision
    # kept stay unless they break a cap.
    kept = set(decision["keep"])
    tiers = {
        "core": list(plan["core"]),
        "strong": [*locked, *decision["keep"]],
        "flex": [n for n in decision["ranked"] if n not in kept],
    }
    stage("final", "lands, validation and safety review")
    deck = draft_deck(working, orchestrator, client, store, max_repairs=max_repairs,
                      reviewer=reviewer, initial=tiers, context={"council": council},
                      agent="orchestrator", toolbox=toolbox)
    record.update(
        locked=locked, contested=contested, open_slots=open_slots, votes=tally,
        decision={"keep": decision["keep"], "notes": decision.get("notes", "")},
    )
    return {**deck, "council": record}
