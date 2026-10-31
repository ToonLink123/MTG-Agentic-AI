"""Native tool use: a model works through tools until a tool's result ends the work.

A tool-capable model (ClaudeClient.tool_turn) answers each turn with the tool calls it made,
several at once when they are independent. Code runs every call and sends the results back, so
the model can look up cards, check a draft with code and fix what code reports before it
commits. Each tool checks its input against its full schema locally, since the wire schema
leaves long card-name enums out. A tool returns Finish(value) to end the loop, or raises
ToolError to send the model an error result it can act on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import activity
from .schema import SchemaError, validate_json

NUDGE = "Call a tool, or call finish with your answer."
LAST_STEP = "One step left: call finish now with your best answer."


def supports_tools(model: Any) -> bool:
    return callable(getattr(model, "tool_turn", None))


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Any]
    # The full schema to check input against, built when called (it may grow, such as the
    # cards a search made addable); by default input_schema itself.
    check: Callable[[], dict[str, Any]] | None = None

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class Finish:
    value: Any


class ToolError(Exception):
    """A result the model must act on, such as the problems a finish left."""

    def __init__(self, result: Any):
        super().__init__(json.dumps(result)[:200])
        self.result = result


def _result(call: dict[str, Any], payload: Any, *, error: bool = False) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(payload)}
    return {**block, "is_error": True} if error else block


def run_tools(
    model: Any, messages: list[dict[str, Any]], tools: list[Tool], *, max_steps: int,
    agent: str, task: str,
) -> dict[str, Any]:
    """Run the loop. Returns {"value": the Finish value or None, "tools": every other tool
    used, in order, "steps": model turns taken}."""
    by_name = {tool.name: tool for tool in tools}
    specs = [tool.spec() for tool in tools]
    conversation = [dict(message) for message in messages]
    used: list[str] = []
    for step in range(1, max_steps + 1):
        label = task if step == 1 else f"{task}: step {step} of {max_steps}"
        with activity.agent(agent, model, label):
            turn = model.tool_turn(conversation, specs)
        conversation.append({"role": "assistant", "content": turn["content"]})
        results: list[dict[str, Any]] = []
        for call in turn["calls"]:
            tool = by_name.get(call["name"])
            if tool is None:
                results.append(_result(call, {"error": f"There is no tool named {call['name']}."}, error=True))
                continue
            try:
                validate_json(call["input"], tool.check() if tool.check else tool.input_schema)
                outcome = tool.run(call["input"])
            except SchemaError as exc:
                results.append(_result(call, {"error": "Invalid input", "problems": exc.errors[:12]}, error=True))
                continue
            except ToolError as exc:
                results.append(_result(call, exc.result, error=True))
                continue
            if isinstance(outcome, Finish):
                return {"value": outcome.value, "tools": used, "steps": step}
            used.append(tool.name)
            results.append(_result(call, outcome))
        if not results:
            results.append({"type": "text", "text": NUDGE})
        if step == max_steps - 1:
            results.append({"type": "text", "text": LAST_STEP})
        conversation.append({"role": "user", "content": results})
    return {"value": None, "tools": used, "steps": max_steps}
