"""Bucketed selection -> exact assembly -> playability/safety gates -> bounded repair."""

from __future__ import annotations

import json
import logging
from typing import Any

from . import activity, prompts
from .assembly import assemble_nonlands, selection_schema
from .discovery import CandidatePool, build_candidate_pool, generate_queries, template_queries
from .forge import ForgeError
from .forge_metrics import summarize
from .lands import generate_mana_base, is_land
from .llm import JSONModel, request_json
from .oracle import OracleStore
from .playability import validate_playability
from .review import DeckReviewer
from .schema import SchemaError
from .scryfall import ScryfallClient
from .spec import DeckSpec, ground_spec, intake
from .validator import validate_deck

log = logging.getLogger(__name__)


def draft_request(checked: CandidatePool, context: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """The drafting prompt for a pool; `context` adds data such as a council's game plan."""
    spec = checked.spec
    return [
        {"role": "system", "content": prompts.DRAFT},
        {"role": "user", "content": json.dumps({
            "spec": spec.to_dict(), "nonland_slots": 99 - spec.role_targets["lands"],
            "candidates": [c.to_dict() for c in checked.cards if not is_land(c.card)],
            "land_policy": "All lands, fetch targets, color floors and exact quantity are owned by code.",
            **(context or {}),
        })},
    ]


def game_changer_names(checked: CandidatePool, store: OracleStore, reviewer: DeckReviewer | None = None) -> set[str]:
    names = set(reviewer.game_changer_names) if reviewer else {
        c.card["name"] for c in checked.cards if c.card["game_changer"]
    }
    leader = store.get_card(checked.spec.commander)
    if reviewer is None and leader and leader["game_changer"]:
        names.add(checked.spec.commander)
    return names


def make_lands(checked: CandidatePool, nonlands: dict[str, int], store: OracleStore, changer_names: set[str]) -> dict[str, Any]:
    """Code-owned lands for a finished nonland list, from its colored pip demand."""
    spec = checked.spec
    log.info("mana-base: generating lands from the assembled nonland pip demand")
    activity.tool("mana base", f"{spec.role_targets['lands']} lands from pip demand")
    return generate_mana_base(
        nonlands, spec, store,
        candidate_names=(c.card["name"] for c in checked.cards if is_land(c.card)),
        game_changers=changer_names,
    )


def check_cards(checked: CandidatePool, cards: dict[str, int], store: OracleStore, errors: list[Any], *, reviewer: DeckReviewer | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Legality, must-includes and playability; problems are appended to `errors`."""
    spec = checked.spec
    activity.tool("validate", "legality, must-includes and playability")
    validation = validate_deck(cards, spec.commander, store, bracket=spec.bracket, game_changers=reviewer.game_changer_names if reviewer else None)
    errors.extend(validation["errors"])
    for name in spec.must_include:
        if cards.get(name, 0) < 1:
            errors.append({"error": "must_include", "card": name, "detail": "Required card is missing"})
    policy_tags = {c.card["name"]: c.roles for c in checked.cards}
    playability = validate_playability(cards, spec, store, tags=policy_tags)
    # Land assembly and independent playability may identify the same failure.
    for error in playability["errors"]:
        if error not in errors:
            errors.append(error)
    return validation, playability


def build_from_selection(checked: CandidatePool, buckets: dict[str, list[str]] | None, store: OracleStore, *, reviewer: DeckReviewer | None = None) -> dict[str, Any]:
    """Code-owned assembly of the model's preferences: nonlands, lands, legality, playability."""
    spec = checked.spec
    changer_names = game_changer_names(checked, store, reviewer)
    errors: list[Any] = []
    assembly: dict[str, Any] = {}
    land_base: dict[str, Any] = {}
    cards: dict[str, int] = {}
    if buckets is not None:
        slots = 99 - spec.role_targets["lands"]
        activity.tool("assemble", f"{slots} nonland slots by role")
        assembly = assemble_nonlands(checked, buckets, game_changers=changer_names)
        errors.extend(assembly["errors"])
        land_base = make_lands(checked, assembly["cards"], store, changer_names)
        errors.extend(land_base["errors"])
        cards = {**assembly["cards"], **land_base["cards"]}
    validation, playability = check_cards(checked, cards, store, errors, reviewer=reviewer)
    return {
        "cards": cards, "assembly": assembly, "land_base": land_base,
        "validation": validation, "playability": playability, "errors": errors,
    }


def draft_deck(pool: CandidatePool, model: JSONModel, client: ScryfallClient, store: OracleStore, *, max_repairs: int = 3, max_verifications: int = 12, reviewer: DeckReviewer | None = None, initial: list[str] | None = None, context: dict[str, Any] | None = None, agent: str = "drafter") -> dict[str, Any]:
    """Draft, check and repair. With `initial` (a council's ranked picks) the first attempt
    assembles those instead of asking the model, which then only repairs what fails."""
    if type(max_repairs) is not int or not 0 <= max_repairs <= 3:
        raise ValueError("max_repairs must be from 0 to 3")
    if type(max_verifications) is not int or not 0 <= max_verifications <= 99:
        raise ValueError("max_verifications must be from 0 to 99")
    # max_verifications remains accepted for API compatibility; selection is now pool-only.
    checked = CandidatePool.from_dict(pool.to_dict(), store)
    spec = checked.spec
    schema = selection_schema(checked)
    messages = draft_request(checked, context)
    attempts = []
    last = {}
    for attempt in range(max_repairs + 1):
        log.info("draft: attempt %d/%d", attempt + 1, max_repairs + 1)
        errors = []
        raw = None
        if attempt == 0 and initial is not None:
            raw = {"commander": spec.commander, "buckets": {"council": list(initial)}}
        else:
            try:
                task = f"repair attempt {attempt + 1}" if attempt else "draft the nonlands"
                with activity.agent(agent, model, task):
                    raw = request_json(model, messages, schema)
            except SchemaError as exc:
                errors.extend(exc.errors)
        built = build_from_selection(checked, raw["buckets"] if raw else None, store, reviewer=reviewer)
        errors.extend(built["errors"])
        cards, playability, validation = built["cards"], built["playability"], built["validation"]
        review = {"status": "not_run", "errors": []}
        if not errors and reviewer:
            try:
                review = reviewer.check(
                    cards, spec.commander, spec.bracket, bracket3_mode=spec.bracket3_mode
                )
            except SchemaError as exc:
                review = {"status": "needs_revision", "errors": [{"error": "review_schema", "detail": "Safety model returned invalid grounded JSON", "schema_errors": exc.errors}]}
            errors.extend(review["errors"])
        validation.update(errors=errors, valid=not errors)
        attempts.append({"attempt": attempt + 1, "errors": errors, "review": review})
        log.info(
            "draft: attempt %d %s",
            attempt + 1,
            f"has {len(errors)} problem(s)" if errors else "passed",
        )
        last = {
            "version": 1, "valid": not errors, "commander": spec.commander, "bracket": spec.bracket,
            "bracket3_mode": spec.bracket3_mode,
            "cards": cards, "spec": spec.to_dict(), "role_counts": playability["roles"]["role_counts"],
            "protection": playability["roles"]["protection"], "playability": playability,
            "assembly": built["assembly"], "land_base": built["land_base"], "validation": validation,
            "review": review, "review_passed": review["status"] == "passed", "attempts": attempts,
            "pool_size": len(checked.cards), "verified_additions": [],
            "simulation": {"status": "not_run", "required_for_build": True},
            "queries": [q.to_dict() for q in checked.queries], "discovery": checked.diagnostics,
        }
        if not errors:
            return last
        if attempt < max_repairs:
            review_feedback = {
                "status": review["status"],
                "bracket_reasons": review.get("bracket", {}).get("reasons", []),
                "interactions": [
                    {key: result[key] for key in ("status", "cards", "question", "claims", "errors") if key in result}
                    for result in review.get("interactions", [])
                ],
            }
            messages = messages[:2] + [{"role": "user", "content": json.dumps({
                "repair": attempt + 1, "previous_buckets": raw, "errors": errors,
                "protection": playability["roles"]["protection"], "role_counts": last["role_counts"],
                "review": review_feedback, "instruction": "Return complete revised role buckets. Commander and spec are fixed.",
            })}]
    return last


def discover(spec: DeckSpec, model: JSONModel, client: ScryfallClient, store: OracleStore, *, pool_limit: int = 350) -> CandidatePool:
    grounded = ground_spec(spec, store)
    plan_error = None
    try:
        queries = generate_queries(grounded, model)
    except SchemaError as exc:
        queries = template_queries(grounded)
        plan_error = exc.errors
    pool = build_candidate_pool(grounded, queries, client, store, limit=pool_limit, model=model)
    if plan_error and pool.diagnostics:
        pool.diagnostics[0]["planner_errors"] = plan_error
    return pool


def draft_from_request(request: str, commander: str, model: JSONModel, client: ScryfallClient, store: OracleStore, *, pool_limit: int = 350, max_repairs: int = 3, reviewer: DeckReviewer | None = None, bracket3_mode: str | None = None) -> dict[str, Any]:
    spec = intake(request, commander, model, client, store, bracket3_mode=bracket3_mode)
    pool = discover(spec, model, client, store, pool_limit=pool_limit)
    return draft_deck(pool, model, client, store, max_repairs=max_repairs, reviewer=reviewer)


def build_deck(request: str, commander: str, model: JSONModel, client: ScryfallClient, store: OracleStore, *, pool_limit: int = 350, max_repairs: int = 3, reviewer: DeckReviewer | None = None, simulator=None, pods=(), games: int = 10, seed: int = 20261025, bracket3_mode: str | None = None) -> dict[str, Any]:
    """A completed build requires safety review and at least ten games per pod."""
    if reviewer is None or simulator is None or not 1 <= len(pods) <= 3:
        raise ValueError("A build requires a safety reviewer, Forge simulator and one to three opponent pods; use draft for an untested list")
    if type(games) is not int or games < 10:
        raise ValueError("A build requires at least 10 games per opponent pod")
    if len({p.fingerprint for p in pods}) != len(pods):
        raise ValueError("Select distinct opponent pods")
    deck = draft_from_request(
        request, commander, model, client, store,
        pool_limit=pool_limit, max_repairs=max_repairs, reviewer=reviewer,
        bracket3_mode=bracket3_mode,
    )
    report = {
        "status": "blocked_by_gates", "required_games_per_pod": games,
        "seed": seed, "pods": [],
    }
    deck["simulation"] = report
    if not (
        deck["valid"] and deck.get("review_passed")
        and deck.get("review", {}).get("status") == "passed"
        and deck["review"].get("bracket", {}).get("verdict") == "fits"
    ):
        deck.update(valid=False, build_passed=False)
        return deck
    try:
        for pod in pods:
            log.info("build: running %d Forge games for %s", games, pod.name)
            raw = simulator.simulate(deck, pod, games, seed=seed)
            report["pods"].append({**summarize(raw), "artifact_dir": raw["artifact_dir"]})
        complete = all(
            p["requested_games"] == games and p["completed_games"] == games
            for p in report["pods"]
        )
        report["status"] = "completed" if complete else "incomplete"
    except (ForgeError, OSError, ValueError) as exc:
        report.update(status="failed", error=str(exc))
    passed = report["status"] == "completed"
    deck.update(valid=passed, build_passed=passed)
    if not passed:
        deck["validation"]["errors"].append({
            "error": "simulation", "detail": "Every opponent pod must have all requested games complete cleanly",
        })
        deck["validation"]["valid"] = False
    return deck
