"""The interview before intake: the lead model asks the player how the deck should play.

Each round the model reads the request, the commander and every answer so far, then either asks
one to four playstyle questions with pre-written answers or says it is ready. The interface adds
two choices to every question, "Let the agents decide" and "Type my own answer"; a question left
to the agents travels to the council as a decision for the orchestrator. Rounds stop when the
model is ready, or at `max_rounds`, where the schema only accepts "ready".
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from . import activity, prompts
from .cards import Card, name_key, trim_card
from .llm import JSONModel, request_json
from .schema import ANSWERS_SCHEMA, BRACKET3_MODES, object_schema, validate_json

MAX_ROUNDS = 6
AGENTS_DECIDE = "Let the agents decide"
OWN_ANSWER = "Type my own answer"
_FIXED = {name_key(AGENTS_DECIDE), name_key(OWN_ANSWER), "other"}

log = logging.getLogger(__name__)


def interview_schema(final: bool) -> dict[str, Any]:
    question = object_schema({
        "question": {"type": "string", "minLength": 1, "maxLength": 300},
        "why": {"type": "string", "minLength": 1, "maxLength": 200},
        "options": {
            "type": "array", "minItems": 2, "maxItems": 4, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 160},
        },
    })
    return object_schema({
        "status": {"type": "string", "enum": ["ready"] if final else ["ask", "ready"]},
        "understanding": {"type": "string", "minLength": 1, "maxLength": 800},
        "questions": {"type": "array", "items": question, "maxItems": 0 if final else 4},
    })


def interview_round(
    request: str, commander: Card, answers: list[dict[str, str]], model: JSONModel, *,
    round_number: int = 1, max_rounds: int = MAX_ROUNDS, bracket3_mode: str | None = None,
) -> dict[str, Any]:
    """One round: {"status": "ask" | "ready", "understanding", "questions", ...}."""
    if not isinstance(request, str) or not request.strip() or len(request) > 8000:
        raise ValueError("The request must contain 1 to 8000 characters")
    if type(max_rounds) is not int or not 1 <= max_rounds <= 20:
        raise ValueError("max_rounds must be from 1 to 20")
    if type(round_number) is not int or not 1 <= round_number <= max_rounds:
        raise ValueError("round must be from 1 to max_rounds")
    if bracket3_mode is not None and bracket3_mode not in BRACKET3_MODES:
        raise ValueError("Unknown Bracket 3 mode")
    answers = deepcopy(answers)
    validate_json(answers, ANSWERS_SCHEMA)
    final = round_number >= max_rounds
    messages = [
        {"role": "system", "content": prompts.INTERVIEW},
        {"role": "user", "content": json.dumps({
            "request": request, "commander_card": trim_card(commander),
            "bracket3_mode": bracket3_mode, "answers": answers,
            "round": round_number, "max_rounds": max_rounds,
        })},
    ]
    with activity.agent("interviewer", model, f"round {round_number}: what to ask the player"):
        reply = request_json(model, messages, interview_schema(final))
    asked = {name_key(a["question"]) for a in answers}
    questions = []
    for item in reply["questions"] if reply["status"] == "ask" else []:
        # Playstyle only: a question that names a card [[like this]] is dropped.
        if name_key(item["question"]) in asked or "[[" in item["question"]:
            continue
        options = [
            o for o in dict.fromkeys(item["options"])
            if name_key(o).rstrip(".!") not in _FIXED and "[[" not in o
        ]
        questions.append({**item, "options": options})
        asked.add(name_key(item["question"]))
    if reply["status"] == "ask" and not questions:
        log.info("interview: every question repeated an earlier one; ready")
    return {
        "status": "ask" if questions else "ready",
        "round": round_number, "max_rounds": max_rounds,
        "understanding": reply["understanding"], "questions": questions,
        "fixed_options": [AGENTS_DECIDE, OWN_ANSWER],
    }
