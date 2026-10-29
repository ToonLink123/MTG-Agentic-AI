"""Strict deck specifications and intake grounded in fetched commander text."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from . import activity, prompts
from .brackets import land_denial_forbidden
from .cards import Card, name_key
from .llm import JSONModel, request_json
from .oracle import OracleStore
from .schema import (
    ANSWERS_SCHEMA,
    BRIEF_SCHEMA,
    EMPTY_BRIEF,
    INTAKE_SCHEMA,
    SPEC_SCHEMA,
    SchemaError,
    validate_json,
)
from .scryfall import ScryfallClient
from .validator import can_be_commander, card_errors

log = logging.getLogger(__name__)
# Curly quotes and dashes a model may type for the player's straight ones.
_PLAIN = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-"})


@dataclass(frozen=True)
class DeckSpec:
    commander: str
    color_identity: str
    bracket: int
    themes: list[str]
    mechanics: list[str]
    win_conditions: list[str]
    playstyle: str
    must_include: list[str]
    exclude: list[str]
    role_targets: dict[str, int]
    commander_dependency: str = "medium"
    bracket3_mode: str = "standard"
    # The player's words (request, interview, asks, deferred questions); see BRIEF_SCHEMA.
    brief: dict[str, Any] = dataclasses.field(default_factory=lambda: deepcopy(EMPTY_BRIEF))

    @classmethod
    def from_dict(cls, data: Any) -> DeckSpec:
        # Migrate version-1 saved specs, not provider responses (which use SPEC_SCHEMA).
        data = deepcopy(data)
        if isinstance(data, dict):
            data.setdefault("bracket3_mode", "standard")
            values = [data.get("themes"), data.get("win_conditions")]
            themes = " ".join(s for v in values if isinstance(v, list) for s in v if isinstance(s, str))
            data.setdefault("commander_dependency", "high" if re.search(r"equipment|voltron|commander.damage", themes, re.I) else "medium")
            if isinstance(data.get("role_targets"), dict):
                data["role_targets"].setdefault("creatures", 15)
                data["role_targets"].setdefault("protection", 6 if data["commander_dependency"] == "high" else 4)
        brief = data.pop("brief", deepcopy(EMPTY_BRIEF)) if isinstance(data, dict) else None
        validate_json(data, SPEC_SCHEMA)
        validate_json(brief, BRIEF_SCHEMA)
        result = cls(**json.loads(json.dumps({**data, "brief": brief})))
        errors: list[dict[str, Any]] = []
        if result.bracket != 3 and result.bracket3_mode != "standard":
            errors.append({"error": "spec", "detail": "Custom Bracket 3 modes require bracket 3"})
        for field in ("themes", "mechanics", "win_conditions", "must_include", "exclude"):
            values = getattr(result, field)
            if len({name_key(value) for value in values}) != len(values):
                errors.append({"error": "spec", "detail": f"{field} contains duplicate names"})
        conflict = {name_key(n) for n in result.must_include} & {
            name_key(n) for n in result.exclude
        }
        if conflict:
            errors.append({"error": "spec", "detail": "must_include and exclude overlap"})
        if name_key(result.commander) in {
            name_key(n) for n in result.must_include + result.exclude
        }:
            errors.append(
                {"error": "spec", "detail": "The commander cannot be in must_include or exclude"}
            )
        if errors:
            raise SchemaError(errors)
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ground_spec(spec: DeckSpec, store: OracleStore, commander: Card | None = None) -> DeckSpec:
    """Check facts in code, never trust an LLM's commander or color identity."""
    spec = DeckSpec.from_dict(spec.to_dict())
    leader = commander or store.get_card(spec.commander)
    errors: list[dict[str, Any]] = []
    if leader is None or name_key(leader["name"]) != name_key(spec.commander):
        errors.append({"error": "spec_commander", "detail": "Commander must resolve exactly"})
    else:
        identity = "".join(c for c in "WUBRG" if c in leader["color_identity"]) or "C"
        if spec.color_identity != identity:
            errors.append(
                {"error": "spec_identity", "detail": f"Commander identity must be {identity}"}
            )
        errors.extend(card_errors(leader, set(leader["color_identity"])))
        if land_denial_forbidden(leader, spec.bracket, spec.bracket3_mode):
            errors.append({"error": "spec_commander", "detail": "Commander violates the selected land-denial rules"})
        if not can_be_commander(leader) or leader["name"] in {
            "The Prismatic Piper",
            "Faceless One",
        }:
            errors.append(
                {"error": "spec_commander", "detail": "Unsupported or ineligible commander"}
            )
    constraints: dict[str, list[str]] = {}
    for field in ("must_include", "exclude"):
        constraints[field] = []
        for name in getattr(spec, field):
            card = store.get_card(name)
            if card is None:
                errors.append(
                    {
                        "error": "unknown_card",
                        "card": name,
                        "detail": "No exact local Oracle match; refresh the snapshot if needed",
                    }
                )
                continue
            constraints[field].append(card["name"])
            if field == "must_include":
                errors.extend(card_errors(card, set(spec.color_identity) - {"C"}))
                if land_denial_forbidden(card, spec.bracket, spec.bracket3_mode):
                    errors.append({
                        "error": "must_include", "card": name,
                        "detail": "Required card violates the selected land-denial rules",
                    })
    if errors:
        raise SchemaError(errors)
    grounded = spec.to_dict()
    grounded.update(constraints)
    grounded["role_targets"]["protection"] = max(6 if spec.commander_dependency == "high" else 4, grounded["role_targets"]["protection"])
    return DeckSpec.from_dict(grounded)


def _plain(text: str) -> str:
    return " ".join(text.translate(_PLAIN).casefold().split())


def _unquoted(asks: list[dict[str, Any]], words: list[str]) -> list[str]:
    """Quotes that are not the player's words: in the request or an answer they gave."""
    said = [_plain(text) for text in words]
    return [a["quote"] for a in asks if not any(_plain(a["quote"]) in text for text in said)]


def intake(
    request: str, commander: str, model: JSONModel, client: ScryfallClient, store: OracleStore,
    *, bracket3_mode: str | None = None, interview: list[dict[str, str]] | None = None,
) -> DeckSpec:
    """The spec plus a brief: the player's words, split into asks that quote them exactly."""
    if not isinstance(request, str) or not request.strip() or len(request) > 8000:
        raise ValueError("Intake request must contain 1 to 8000 characters")
    answers = deepcopy(interview or [])
    validate_json(answers, ANSWERS_SCHEMA)
    if not store.ready:
        raise ValueError("Download an Oracle snapshot before intake")
    fetched = client.get_card(commander, fuzzy=False)
    local = store.get_card(fetched["name"])
    if local is None:
        raise ValueError("Commander is absent from the local snapshot; refresh Oracle Cards first")
    messages = [
        {
            "role": "system",
            "content": prompts.INTAKE + json.dumps(INTAKE_SCHEMA),
        },
        {"role": "user", "content": json.dumps({
            "request": request, "interview": answers, "commander_card": fetched,
            "bracket3_mode": bracket3_mode,
        })},
    ]
    log.info("intake: turning the request into a deck spec")
    with activity.agent("intake", model, "turn the request into a deck spec"):
        data = request_json(model, messages, INTAKE_SCHEMA)
    words = [request, *(a["answer"] for a in answers if a["source"] != "agents")]
    unquoted = _unquoted(data["asks"], words)
    if unquoted:
        # One chance to copy the player's words exactly, then code drops what still is not.
        messages += [
            {"role": "assistant", "content": json.dumps(data)},
            {"role": "user", "content": json.dumps({
                "problems": [{"quote": q, "detail": "Not found in the request or an answer"}
                             for q in unquoted],
                "instruction": "Return the same JSON object. Copy every ask's quote character "
                "for character from the request or an interview answer; change nothing else.",
            })},
        ]
        with activity.agent("intake", model, "quote the player's words exactly"):
            try:
                data = request_json(model, messages, INTAKE_SCHEMA)
            except SchemaError:
                log.warning("intake: the quote repair was invalid; keeping the first answer")
    unquoted = _unquoted(data["asks"], words)
    asks = [a for a in data.pop("asks") if a["quote"] not in unquoted]
    if unquoted:
        log.warning("intake: dropped %d ask(s) that do not quote the player", len(unquoted))
    if not asks:
        asks = [{"quote": request.strip()[:400], "wants": "the request as a whole", "kind": "cards"}]
    data["brief"] = {
        "request": request, "interview": answers,
        "asks": [{"id": f"A{number}", **ask} for number, ask in enumerate(asks, 1)],
        "deferred": [a["question"] for a in answers if a["source"] == "agents"],
    }
    if bracket3_mode is not None:
        data["bracket3_mode"] = bracket3_mode
    spec = DeckSpec.from_dict(data)
    # Ensure the response cannot silently switch the requested commander.
    if name_key(spec.commander) != name_key(fetched["name"]):
        raise SchemaError(
            [{"error": "spec_commander", "detail": "Intake changed the requested commander"}]
        )
    return ground_spec(spec, store, {**local, **fetched})
