"""The fixed JSON contracts and a validator for the schema keywords used here."""

from __future__ import annotations

import json
import re
from typing import Any

ROLES = (
    "theme",
    "mechanics",
    "win_condition",
    "ramp",
    "draw",
    "removal",
    "wipes",
    "creatures",
    "protection",
    "lands",
    "support",
)
TARGET_ROLES = ("ramp", "draw", "removal", "wipes", "lands", "creatures", "protection")
DRAFT_ROLES = ("ramp", "draw", "removal", "wipes", "creatures", "protection", "theme", "support")


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


NAME_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 200}
STRING_LIST = {
    "type": "array",
    "items": NAME_SCHEMA,
    "maxItems": 99,
    "uniqueItems": True,
}
SPEC_SCHEMA = object_schema(
    {
        "commander": NAME_SCHEMA,
        "color_identity": {
            "type": "string",
            "minLength": 1,
            "pattern": r"^(?:W?U?B?R?G?|C)$",
        },
        "bracket": {"type": "integer", "minimum": 1, "maximum": 5},
        "themes": STRING_LIST,
        "mechanics": STRING_LIST,
        "win_conditions": STRING_LIST,
        "playstyle": NAME_SCHEMA,
        "commander_dependency": {"type": "string", "enum": ["low", "medium", "high"]},
        "must_include": STRING_LIST,
        "exclude": STRING_LIST,
        "role_targets": object_schema(
            {role: {"type": "integer", "minimum": 0, "maximum": 99} for role in TARGET_ROLES}
        ),
    }
)
QUERY_SCHEMA = object_schema(
    {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": object_schema(
                {
                    "role": {"type": "string", "enum": list(ROLES)},
                    "query": {"type": "string", "minLength": 1, "maxLength": 850},
                }
            ),
        },
    }
)
DRAFT_SCHEMA = object_schema(
    {
        "commander": NAME_SCHEMA,
        "buckets": object_schema({
            role: {"type": "array", "items": NAME_SCHEMA, "maxItems": 63, "uniqueItems": True}
            for role in DRAFT_ROLES
        }),
    }
)


class SchemaError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__("; ".join(error["detail"] for error in errors))


def load_json(text: str) -> Any:
    """Reject prose, fences, duplicate keys, and non-JSON NaN/Infinity values."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError("Non-finite numbers are not JSON")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise SchemaError(
            [{"error": "json", "path": "$", "detail": "Expected strict JSON"}]
        ) from exc


def validate_json(value: Any, schema: dict[str, Any]) -> None:
    """Validate our published contracts without accepting coerced types."""
    errors: list[dict[str, Any]] = []

    def fail(path: str, detail: str) -> None:
        errors.append({"error": "schema", "path": path, "detail": detail})

    def visit(item: Any, rule: dict[str, Any], path: str) -> None:
        kind = rule.get("type")
        matches = {
            "object": isinstance(item, dict),
            "array": isinstance(item, list),
            "string": isinstance(item, str),
            "integer": type(item) is int,
        }
        if kind not in matches or not matches[kind]:
            fail(path, f"{path} must be {kind}")
            return
        if "enum" in rule and item not in rule["enum"]:
            fail(path, f"{path} is not an allowed value")
        if kind == "object":
            properties = rule.get("properties", {})
            for key in rule.get("required", []):
                if key not in item:
                    fail(path + "." + key, f"{path}.{key} is required")
            for key, child in item.items():
                if key in properties:
                    visit(child, properties[key], path + "." + key)
                elif rule.get("additionalProperties") is False:
                    fail(path, f"{path} contains an unexpected field")
        elif kind == "array":
            if len(item) < rule.get("minItems", 0) or len(item) > rule.get(
                "maxItems", float("inf")
            ):
                fail(path, f"{path} has an invalid number of items")
            if rule.get("uniqueItems") and len(
                {json.dumps(v, sort_keys=True) for v in item}
            ) != len(item):
                fail(path, f"{path} must not contain duplicates")
            for index, child in enumerate(item):
                visit(child, rule["items"], f"{path}[{index}]")
        elif kind == "string":
            if (
                not item.strip()
                or len(item) < rule.get("minLength", 0)
                or len(item) > rule.get("maxLength", float("inf"))
            ):
                fail(path, f"{path} has an invalid string length")
            if "pattern" in rule and re.fullmatch(rule["pattern"], item) is None:
                fail(path, f"{path} has an invalid format")
        elif kind == "integer":
            if item < rule.get("minimum", -float("inf")) or item > rule.get(
                "maximum", float("inf")
            ):
                fail(path, f"{path} is outside the allowed range")

    visit(value, schema, "$")
    if errors:
        raise SchemaError(errors)
