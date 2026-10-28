"""Dark terminal chat for the deck pipeline: name a commander, describe a deck, watch it build.

Every action runs the same `python -m mtg_agentic ...` command a user would type, as a
subprocess, so the chat streams its stderr progress, renders its JSON result and can stop it
at any time. Needs `textual` (`pip install -e ".[tui]"`); nothing loads it unless `tui` runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from .council import MEMBER, ORCHESTRATOR
from .llm import PROVIDERS
from .oracle import OracleStore
from .schema import BRACKET3_MODES

GOLD, BLUE, TEAL = "#d9a441", "#7aa2d6", "#5fb3a1"
GREEN, AMBER, RED, MUTED = "#6cc28a", "#e0b050", "#e06c75", "#8b8f98"
# Mana symbols. Black is drawn lilac so it stays readable on the dark background.
PIPS = {
    "W": "#f0e6c0", "U": "#6fa8dc", "B": "#b39ddb", "R": "#e8735a", "G": "#72c17f",
    "C": "#a8a8a8",
}
THEME = Theme(
    name="mtg-dark",
    primary=GOLD,
    secondary=BLUE,
    accent=TEAL,
    foreground="#e6e1d6",
    background="#0e1014",
    surface="#15181e",
    panel="#1b1f27",
    success=GREEN,
    warning=AMBER,
    error=RED,
    dark=True,
)
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HELP = (
    ("a commander's name", "Look the card up and make it your commander"),
    ("a message", "No deck yet: draft one. With a deck: ask the orchestrator or request changes"),
    ("/draft REQUEST", "Draft a fresh deck in this session (the council when /orchestrate is on)"),
    ("/commander NAME", "Switch commanders"),
    ("/card NAME", "Show a card's Oracle text"),
    ("/search QUERY", "Search Scryfall, e.g. t:equipment id:w"),
    ("/ask [[A]] [[B]] QUESTION", "Rules question, answered with cited rules and rulings"),
    ("/validate", "Offline legality and playability check of the current deck"),
    ("/bracket", "Bracket assessment of the current deck"),
    ("/review", "Bracket plus rules review of the current deck"),
    ("/review on|off", "Whether each draft runs that review itself (on by default)"),
    ("/orchestrate on|off", "Draft with a council: an orchestrator plans, members debate"),
    ("/deck", "Show the current deck again"),
    ("/load PATH", "Make a saved deck JSON the current deck"),
    ("/export [PATH]", "Save the current deck as a text list, commander first, and copy it"),
    ("/new", "Start a new session"),
    ("/session [N]", "Pick a past session to resume, or resume number N"),
    ("/model [PROVIDER] [MODEL]", "Show or switch the LLM; a provider alone uses its default"),
    ("/mode [MODE]", "Bracket 3 rules: auto (read from the request), " + ", ".join(BRACKET3_MODES)),
    ("/log", "Full progress log of the last run"),
    ("/stop  /clear  /quit", "Also Esc, Ctrl+L and Ctrl+Q; Tab completes a command"),
)
SUGGESTIONS = [
    "/commander ", "/card ", "/search ", "/ask [[", "/validate", "/bracket", "/review",
    "/review on", "/review off", "/orchestrate on", "/orchestrate off", "/deck", "/load ",
    "/export", "/draft ", "/new", "/session",
    *(f"/model {p}" for p in PROVIDERS),
    *(f"/mode {m}" for m in ("auto", *BRACKET3_MODES)), "/log", "/stop", "/clear", "/help", "/quit",
]
# Commands that only read or change chat state, so they stay available during a run.
# `/review on|off` is one too; `/review` alone starts a deck review.
WHILE_BUSY = {"/stop", "/help", "/model", "/mode", "/orchestrate", "/log", "/quit"}
COUNCIL_STEPS = [
    "Intake", "Discovery", "Plan & core", "Council drafts", "Debate", "Decision", "Final deck",
]
# The council command logs "council: 2/5 drafts: ..." as it moves through its stages.
# Agent and tool events from mtg_agentic.activity feed the live agent panel.
AGENT_EVENT = re.compile(
    r"^agent: (?P<name>[^|]+?) \| (?P<state>start|progress|done|failed) \| (?P<rest>.*)$"
)
TOOL_EVENT = re.compile(r"^tool: (?P<name>[^|]+?) \| (?P<detail>.*)$")
LILAC, PINK = "#c49bd6", "#e8a0bf"
AGENT_COLORS = {
    "orchestrator": GOLD, "drafter": GOLD, "engine": TEAL, "resilience": BLUE,
    "consistency": LILAC, "intake": BLUE, "scout": TEAL, "reviewer": PINK, "judges": PINK,
}
COUNCIL_STAGE = re.compile(r"council: (\d+)/\d+ \w+: (.*)")
TYPE_ORDER = ("Creature", "Planeswalker", "Battle", "Instant", "Sorcery", "Artifact", "Enchantment")
PLURALS = {"Sorcery": "Sorceries", "Other": "Other"}
VERDICT = {"fits": GREEN, "uncertain": AMBER, "mismatch": RED}
SUPPORT = {"supported": GREEN, "contradicted": RED, "unsupported": AMBER}
CARD_REF = re.compile(r"\[\[(.+?)\]\]")

Runner = Callable[[list[str], Path, Callable[[str], None]], Awaitable[tuple[int, Any]]]


@dataclass
class Settings:
    provider: str = "ollama"
    model: str | None = None
    mode: str | None = None  # Bracket 3 house rules; None lets intake read them from the request.
    review: bool = True
    orchestrate: bool = False

    def model_args(self) -> list[str]:
        return ["--provider", self.provider, *(["--model", self.model] if self.model else [])]

    def lead_args(self) -> list[str]:
        """Intake and discovery: in orchestrate mode on Claude, the orchestrator runs them."""
        if self.orchestrate and self.provider == "claude" and not self.model:
            return ["--provider", "claude", "--model", ORCHESTRATOR[0], "--effort", ORCHESTRATOR[1]]
        return self.model_args()

    def orchestrator_args(self) -> list[str]:
        """Follow-ups go to the orchestrator alone; on Claude that is the council's lead model."""
        if self.provider == "claude" and not self.model:
            return ["--provider", "claude", "--model", ORCHESTRATOR[0], "--effort", ORCHESTRATOR[1]]
        return self.model_args()

    def council_roles(self) -> tuple[str, str]:
        if self.provider == "claude":
            return f"{ORCHESTRATOR[0]} ({ORCHESTRATOR[1]})", f"3 × {MEMBER[0]} ({MEMBER[1]})"
        model = self.model or "provider default"
        return model, f"3 × {model}"

    # Each choose_* applies one command's value; a ValueError carries the message for the chat.

    def choose_model(self, words: list[str]) -> None:
        """`/model [PROVIDER] [MODEL|default]`, e.g. `/model claude` or `/model ollama qwen3:8b`."""
        provider = words[0].lower() if words and words[0].lower() in PROVIDERS else None
        rest = words[1:] if provider else words
        if len(rest) > 1:
            raise ValueError(
                f"Use /model [PROVIDER] [MODEL|default]. Providers: {', '.join(PROVIDERS)}"
            )
        if provider:
            # Model names differ per provider, so a new provider starts on its default model.
            self.provider, self.model = provider, None
        if rest:
            self.model = None if rest[0] == "default" else rest[0]

    def choose_mode(self, value: str) -> None:
        if value not in {"auto", *BRACKET3_MODES}:
            raise ValueError(f"Mode is one of: auto, {', '.join(BRACKET3_MODES)}")
        self.mode = None if value == "auto" else value

    def choose_review(self, value: str) -> None:
        if value not in {"on", "off"}:
            raise ValueError("Use /review on or off; /review alone reviews the current deck")
        self.review = value == "on"

    def choose_orchestrate(self, value: str) -> None:
        if value not in {"on", "off"}:
            raise ValueError("Use /orchestrate on or off")
        self.orchestrate = value == "on"

    def describe(self) -> str:
        return (
            f"{self.provider} · {self.model or 'default model'} · "
            f"review {'on' if self.review else 'off'} · mode {self.mode or 'auto'}"
            + (" · council" if self.orchestrate else "")
        )


def embedding_problem() -> str | None:
    """Why this Python cannot embed rules questions, read from package metadata so the TUI
    starts instantly; None when transformers is new enough for EmbeddingGemma 2."""
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return f"{sys.executable} has no transformers."
    if tuple(int(part) for part in re.findall(r"\d+", version)[:2]) < (5, 21):
        return f"{sys.executable} has transformers {version}; rules reviews need 5.21 or newer."
    return None


def parse_question(text: str) -> tuple[list[str], str]:
    """`[[Card]]` marks each card, Scryfall-bot style; the question keeps the names inline."""
    cards = [name.strip() for name in CARD_REF.findall(text) if name.strip()]
    return cards, " ".join(CARD_REF.sub(lambda match: match[1].strip(), text).split())


async def run_cli(
    argv: list[str], data_dir: Path, on_line: Callable[[str], None]
) -> tuple[int, Any]:
    """Run one CLI command, streaming stderr lines to `on_line`; return (exit code, stdout JSON)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "mtg_agentic", "--data-dir", str(data_dir), *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        limit=1 << 22,
    )

    async def progress() -> None:
        async for raw in process.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.strip():
                on_line(line)

    try:
        out, _ = await asyncio.gather(process.stdout.read(), progress())
        code = await process.wait()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise
    text = out.decode("utf-8", "replace").strip()
    try:
        return code, json.loads(text) if text else None
    except json.JSONDecodeError:
        return code, None


def mana(cost: str) -> Text:
    """{2}{U}{B} drawn as 2UB, each colored symbol in its color."""
    text = Text(no_wrap=True)
    for face, part in enumerate((cost or "").split(" // ")):
        if face:
            text.append(" // ", MUTED)
        for symbol in re.findall(r"\{([^}]+)\}", part):
            # Hybrid and Phyrexian symbols keep parentheses so W(B/P)G stays readable.
            shown_symbol = f"({symbol})" if "/" in symbol else symbol
            text.append(shown_symbol, next((PIPS[c] for c in symbol if c in PIPS), MUTED))
    return text


def identity(colors: Iterable[str]) -> Text:
    ordered = sorted("".join(colors) or "C", key="WUBRGC".find)
    return Text.assemble(*((c, f"bold {PIPS.get(c, MUTED)}") for c in ordered))


def problem(error: Any) -> str:
    if not isinstance(error, dict):
        return str(error)
    head = " · ".join(str(error[key]) for key in ("error", "card") if error.get(key))
    detail = error.get("detail")
    text = f"{head}: {detail}" if head and detail else str(detail or head or error)
    return text if len(text) <= 300 else text[:297] + "..."


def problems(errors: list[Any], limit: int = 8) -> Group:
    lines = [Text(f"• {problem(error)}", RED) for error in errors[:limit]]
    if len(errors) > limit:
        lines.append(Text(f"  + {len(errors) - limit} more in the saved JSON", MUTED))
    return Group(*lines)


def labeled(rows: Iterable[tuple[str, Any]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=MUTED, no_wrap=True)
    grid.add_column()
    for label, value in rows:
        # Model-written text can contain brackets, so plain strings never parse as markup.
        grid.add_row(label, Text(value) if isinstance(value, str) else value)
    return grid


def render_help() -> Table:
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=GOLD, no_wrap=True)
    grid.add_column()
    for command, description in HELP:
        grid.add_row(command, description)
    return grid


def render_card(card: dict[str, Any]) -> Group:
    head = Table.grid(expand=True)
    head.add_column()
    head.add_column(justify="right")
    head.add_row(Text(card["name"], f"bold {GOLD}"), mana(card.get("mana_cost", "")))
    parts: list[Any] = [head, Text(card.get("type_line", ""), MUTED)]
    if card.get("oracle_text"):
        parts.append(Text(card["oracle_text"]))
    foot = Text.assemble(("Identity ", MUTED), identity(card.get("color_identity", [])))
    if card.get("game_changer"):
        foot.append("  · Game Changer", AMBER)
    return Group(*parts, foot)


def render_search(cards: list[dict[str, Any]], limit: int = 25) -> Group:
    if not cards:
        return Group(Text("No cards matched.", MUTED))
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(style=MUTED, overflow="ellipsis", no_wrap=True)
    for card in cards[:limit]:
        table.add_row(card["name"], mana(card.get("mana_cost", "")), card.get("type_line", ""))
    parts: list[Any] = [Text(f"{len(cards)} card{'s' * (len(cards) != 1)}", "bold"), table]
    if len(cards) > limit:
        parts.append(Text(f"+ {len(cards) - limit} more", MUTED))
    return Group(*parts)


def render_spec(spec: dict[str, Any]) -> Table:
    head = Text.assemble(
        (spec["commander"], "bold"), "  ", identity(spec["color_identity"]),
        (f"  Bracket {spec['bracket']}", GOLD),
    )
    if spec.get("bracket3_mode", "standard") != "standard":
        head.append(f" ({spec['bracket3_mode']})", GOLD)
    head.append(f"  · {spec['playstyle']}", MUTED)
    rows = [("Spec", head)]
    for label, key in (
        ("Themes", "themes"), ("Mechanics", "mechanics"), ("Win cons", "win_conditions"),
        ("Must include", "must_include"), ("Exclude", "exclude"),
    ):
        if spec.get(key):
            rows.append((label, ", ".join(spec[key])))
    rows.append(("Targets", " · ".join(f"{k} {v}" for k, v in spec["role_targets"].items())))
    rows.append(("Dependency", f"commander dependency {spec['commander_dependency']}"))
    return labeled(rows)


def render_pool(pool: dict[str, Any]) -> Table:
    roles = Counter(role for card in pool["cards"] for role in card.get("roles", []))
    head = Text.assemble(
        (f"{len(pool['cards'])} candidates", "bold"), f" from {len(pool['queries'])} queries"
    )
    return labeled([
        ("Pool", head),
        ("Roles", Text(" · ".join(f"{role} {n}" for role, n in roles.most_common()), MUTED)),
    ])


def card_type(card: dict[str, Any] | None) -> str:
    if card is None:
        return "Other"
    words = card["type_line"].split(" // ")[0].replace("—", " ").split()
    if "Land" in words:
        return "Land"
    return next((kind for kind in TYPE_ORDER if kind in words), "Other")


class DeckColumns:
    """Card groups flowed into as many balanced columns as the width allows."""

    def __init__(self, groups: list[tuple[str, list[tuple[int, str, str]]]]):
        self.rows: list[tuple[str, Text, Text]] = []
        for title, cards in groups:
            if self.rows:
                self.rows.append(("gap", Text(), Text()))
            count = sum(quantity for quantity, _, _ in cards)
            self.rows.append(("head", Text(f"{title} {count}", f"bold {GOLD}"), Text()))
            for quantity, name, cost in cards:
                label = Text(f"{quantity} ", MUTED) if quantity > 1 else Text()
                self.rows.append(("card", label + Text(name), mana(cost)))

    def __rich_console__(self, console, options):
        # Use only as many columns as fit the longest name beside the longest cost (a column's
        # cost cells share one width), so no card is cut off when the width allows it.
        need = max(r[1].cell_len for r in self.rows) + max(r[2].cell_len for r in self.rows) + 1
        gap = 3
        fits = (options.max_width + gap) // (need + gap)
        count = max(1, min(4, fits, len(self.rows) // 10 or 1))
        size = -(-len(self.rows) // count)
        columns: list[list[tuple[str, Text, Text]]] = [[]]
        for row in self.rows:
            if not columns[-1] and row[0] == "gap":
                continue
            columns[-1].append(row)
            if len(columns[-1]) >= size and len(columns) < count:
                # Never end a column on a heading: carry it over with its cards.
                carry = []
                while columns[-1] and columns[-1][-1][0] in {"head", "gap"}:
                    carry.insert(0, columns[-1].pop())
                columns.append([r for r in carry if r[0] != "gap"])
        grid = Table.grid(padding=(0, gap, 0, 0), expand=True)
        tables = []
        for column in columns:
            grid.add_column(ratio=1)
            table = Table.grid(expand=True)
            table.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
            table.add_column(justify="right", no_wrap=True)
            for _, left, right in column:
                table.add_row(left, right)
            tables.append(table)
        grid.add_row(*tables)
        yield grid


def render_edit(edit: dict[str, Any]) -> Group:
    """The orchestrator's answer, its swaps, and the tools it used."""
    reply = Text.assemble(("orchestrator", f"bold {GOLD}"), "  ", edit.get("reply", ""))
    parts: list[Any] = [reply]
    swaps = edit.get("swaps", [])
    if swaps:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(overflow="fold", ratio=1, style=MUTED)
        for swap in swaps:
            grid.add_row(
                Text.assemble(("- ", RED), swap["remove"]),
                Text.assemble(("+ ", GREEN), (swap["add"], GREEN)),
                swap["reason"],
            )
        parts += [Text(), grid]
    tools = edit.get("tools", [])
    if tools:
        parts.append(Text("orchestrator used: " + ", ".join(dict.fromkeys(tools)), MUTED))
    return Group(*parts)


def render_deck(
    deck: dict[str, Any], lookup: Callable[[str], dict[str, Any] | None], path: Path | None = None,
    *, columns: bool = True,
) -> Group:
    cards: dict[str, int] = deck.get("cards", {})
    errors = deck.get("validation", {}).get("errors", deck.get("errors", []))
    head = Text()
    if "valid" in deck:
        if deck["valid"]:
            head.append("✓ Deck passed", f"bold {GREEN}")
        else:
            head.append(f"✗ {len(errors)} problem{'s' * (len(errors) != 1)}", f"bold {RED}")
        head.append("  ·  ", MUTED)
    head.append(f"{deck['commander']} + {sum(cards.values())} cards", MUTED)
    if "bracket" in deck:
        head.append(f"  ·  Bracket {deck['bracket']}", MUTED)
    review = deck.get("review", {}).get("status")
    if review:
        color = {"passed": GREEN, "not_run": MUTED}.get(review, RED)
        head.append("  ·  ", MUTED)
        head.append(f"review {review.replace('_', ' ')}", color)
    parts: list[Any] = [head]
    if deck.get("role_counts"):
        targets = deck.get("spec", {}).get("role_targets", {})
        roles = Text("Roles  ", MUTED)
        for index, (role, count) in enumerate(deck["role_counts"].items()):
            if index:
                roles.append(" · ", MUTED)
            target = targets.get(role)
            roles.append(f"{role} {count}", AMBER if target is not None and count < target else "")
            if target is not None:
                roles.append(f"/{target}", MUTED)
        parts.append(roles)
    leader = lookup(deck["commander"])
    groups: dict[str, list[tuple[float, int, str, str]]] = {}
    for name, quantity in cards.items():
        card = lookup(name)
        groups.setdefault(card_type(card), []).append(
            ((card or {}).get("cmc") or 0, quantity, name, (card or {}).get("mana_cost", ""))
        )
    ordered = [("Commander", [(1, deck["commander"], (leader or {}).get("mana_cost", ""))])]
    for kind in (*TYPE_ORDER, "Land", "Other"):
        if kind in groups:
            rows = sorted(groups[kind], key=lambda row: (row[0], row[2]))
            ordered.append((PLURALS.get(kind, kind + "s"), [r[1:] for r in rows]))
    parts += [Text(), DeckColumns(ordered)] if columns else []
    if errors:
        parts += [Text(), problems(errors)]
    if path is not None:
        hint = "" if columns else " · /deck shows the list"
        parts.append(Text(f"Deck file {shown(path)}{hint}", MUTED))
    return Group(*parts)


def render_council(council: dict[str, Any]) -> Table:
    drafts = council.get("drafts", [])
    rows: list[tuple[str, Any]] = [
        ("Council", f"{council.get('orchestrator') or 'orchestrator'} led "
                    f"{len(drafts)} × {council.get('member') or 'member'}"),
        ("Plan", council.get("game_plan", "")),
        ("Core", ", ".join(council.get("core", [])) or "none"),
    ]
    if council.get("non_goals"):
        rows.append(("Avoid", ", ".join(council["non_goals"])))
    if council.get("dropped_game_changers"):
        dropped = ", ".join(council["dropped_game_changers"])
        rows.append(("Dropped", Text(f"{dropped} (over the Game Changer limit)", AMBER)))
    summary = " · ".join(
        f"{d['focus']} {d['problems']} problem{'s' * (d['problems'] != 1)}" for d in drafts
    )
    rows.append(("Drafts", summary or "none succeeded"))
    if "locked" in council:
        rows.append(("Debate", f"{len(council['locked'])} agreed · "
                               f"{len(council['contested'])} contested for "
                               f"{council['open_slots']} slots · "
                               f"{len(council['decision']['keep'])} kept"))
    if council.get("decision", {}).get("notes"):
        rows.append(("Notes", council["decision"]["notes"]))
    if council.get("fallback"):
        message = "No member draft succeeded; the orchestrator drafted alone."
        rows.append(("Fallback", Text(message, AMBER)))
    return labeled(rows)


def render_validation(result: dict[str, Any]) -> Group:
    errors = result.get("errors", [])
    head = Text("✓ Passed" if result.get("valid") else f"✗ {len(errors)} problems",
                f"bold {GREEN if result.get('valid') else RED}")
    if result.get("scope"):
        head.append(f"  ·  {result['scope']}", MUTED)
    return Group(head, *([problems(errors)] if errors else []))


def render_bracket(report: dict[str, Any]) -> Group:
    verdict = report.get("verdict", "unknown")
    head = Text.assemble(
        (f"● {verdict}", f"bold {VERDICT.get(verdict, MUTED)}"),
        (f"  target Bracket {report.get('target_bracket')}", ""),
    )
    if report.get("bracket3_mode", "standard") != "standard":
        head.append(f" ({report['bracket3_mode']})", MUTED)
    assessment = report.get("assessment") or {}
    rows: list[tuple[str, Any]] = []
    if assessment:
        head.append(f"  ·  estimated Bracket {assessment['estimated_bracket']}", MUTED)
        low, high = assessment.get("likely_win_turn_min"), assessment.get("likely_win_turn_max")
        rows += [
            ("Curve", assessment.get("curve", "")),
            ("Interaction", assessment.get("interaction", "")),
            ("Win turn", f"turns {low}-{high}" if low and high else "unknown"),
        ]
    errors, warnings = report.get("errors", []), report.get("warnings", [])
    reasons = [Text(f"• {r}", RED) for r in errors] + [Text(f"• {r}", AMBER) for r in warnings]
    reasons += [Text(f"• {r}") for r in assessment.get("reasons", [])]
    parts: list[Any] = [head]
    if rows:
        parts.append(labeled(rows))
    parts += reasons[:10]
    if report.get("disclaimer"):
        parts.append(Text(report["disclaimer"], MUTED))
    return Group(*parts)


def render_review(result: dict[str, Any]) -> Group:
    passed = result.get("valid", result.get("status") == "passed")
    parts: list[Any] = [Text(
        "✓ Review passed" if passed else "✗ Review: needs revision",
        f"bold {GREEN if passed else RED}",
    )]
    if result.get("bracket"):
        parts.append(render_bracket(result["bracket"]))
    interactions = result.get("interactions", [])
    if interactions:
        supported = sum(1 for item in interactions if item.get("status") == "supported")
        parts.append(labeled([("Interactions", f"{supported}/{len(interactions)} supported")]))
    if result.get("errors"):
        parts.append(problems(result["errors"]))
    return Group(*parts)


def render_interaction(result: dict[str, Any]) -> Group:
    status = result.get("status", "unknown")
    parts: list[Any] = [Text.assemble(
        (f"● {status}", f"bold {SUPPORT.get(status, MUTED)}"), "  ", result.get("question", ""),
    )]
    for claim in result.get("claims", []):
        parts.append(Text(f"• {claim['claim']}"))
        for citation in claim.get("citations", []):
            quote = " ".join(citation.get("quote", "").split())
            quote = quote if len(quote) <= 220 else quote[:217] + "..."
            cited = Text.assemble("    ", (citation.get("id", ""), BLUE), ("  " + quote, MUTED))
            parts.append(cited)
    if status == "unsupported" and not result.get("claims"):
        parts.append(Text("The retrieved rules and rulings did not support an answer.", MUTED))
    if result.get("errors"):
        parts.append(problems(result["errors"]))
    if result.get("disclaimer"):
        parts.append(Text(result["disclaimer"], MUTED))
    return Group(*parts)


def decklist_text(deck: dict[str, Any]) -> str:
    """`1 Card Name` per unique card, commander first: the list most deck sites import."""
    rows = [(1, deck["commander"]), *((qty, name) for name, qty in deck["cards"].items())]
    return "".join(f"{quantity} {name}\n" for quantity, name in rows)


def free_path(path: Path) -> Path:
    """`path`, or the first `stem-2`, `stem-3`... beside it that does not exist yet."""
    candidate, number = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
        number += 1
    return candidate


SESSION_FILE = "session.json"


def new_session_dir(data_dir: Path, taken: Iterable[Path] = ()) -> Path:
    """A fresh session folder name. Folders are created on first save, so also avoid `taken`."""
    base = data_dir / "tui" / datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate, number, taken = base, 2, set(taken)
    while candidate.exists() or candidate in taken:
        candidate = base.with_name(f"{base.name}-{number}")
        number += 1
    return candidate


def pool_for(deck_path: Path) -> Path | None:
    """The candidate pool saved beside a session deck: 01-deck-v3.json -> 01-pool.json."""
    match = re.match(r"(\d+)-", deck_path.name)
    pool = deck_path.parent / f"{match[1]}-pool.json" if match else None
    return pool if pool is not None and pool.is_file() else None


def read_session(folder: Path) -> dict[str, Any]:
    """A session folder's transcript and latest deck; folders without a transcript work too."""
    data: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        data = json.loads((folder / SESSION_FILE).read_text(encoding="utf-8"))
    decks = sorted(folder.glob("*-deck*.json"), key=lambda p: p.stat().st_mtime)
    deck = Path(data["deck"]) if data.get("deck") else None
    if deck is not None and not deck.is_absolute():
        deck = folder / deck
    if deck is None or not deck.is_file():
        deck = decks[-1] if decks else None
    commander, valid = data.get("commander"), data.get("valid")
    if deck is not None and commander is None:
        with contextlib.suppress(OSError, ValueError):
            saved = json.loads(deck.read_text(encoding="utf-8"))
            commander, valid = saved.get("commander"), saved.get("valid")
    return {
        "path": folder, "name": folder.name, "commander": commander, "valid": valid,
        "deck": deck, "decks": len(decks), "turns": data.get("turns", []),
    }


def list_sessions(root: Path) -> list[dict[str, Any]]:
    """Past sessions, newest first, skipping folders with neither a deck nor any chat."""
    folders = sorted((p for p in root.glob("*") if p.is_dir()), key=lambda p: p.name, reverse=True)
    sessions = [read_session(folder) for folder in folders]
    return [s for s in sessions if s["deck"] is not None or s["turns"]]


def session_label(info: dict[str, Any], current: bool = False) -> Text:
    try:
        when = datetime.strptime(info["name"][:15], "%Y%m%d-%H%M%S").strftime("%b %d %H:%M")
    except ValueError:
        when = info["name"]
    status = {True: ("✓", GREEN), False: ("✗", RED)}.get(info["valid"], ("·", MUTED))
    label = Text.assemble(
        (when, MUTED), "  ", status, " ", (info["commander"] or "no commander yet", "bold"),
        (f"  · {info['decks']} deck file{'s' * (info['decks'] != 1)}"
         f" · {len(info['turns'])} messages", MUTED),
    )
    if current:
        label.append("  (current)", GOLD)
    return label


def shown(path: Path) -> str:
    """A path relative to the working directory when it is inside it."""
    with contextlib.suppress(ValueError):
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    return str(path)


def elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


class Turn(Vertical):
    """One reply: a checklist of CLI steps with a live tail of their progress, then results."""

    def __init__(self, steps: Iterable[str] = (), body: Iterable[Any] = ()):
        self.steps = [
            {"name": name, "state": "pending", "detail": "", "start": 0.0, "end": 0.0}
            for name in steps
        ]
        super().__init__(classes="turn" if self.steps else "turn plain")
        self.body = list(body)
        self.lines: list[str] = []  # everything, for /log
        self.tail: list[str] = []  # progress that is not an agent or tool event
        self.agents: dict[str, dict[str, Any]] = {}
        self.tools: list[tuple[str, str]] = []
        self.frame = 0
        self.ticker: Timer | None = None

    def compose(self) -> ComposeResult:
        steps = Static(self.render_steps(), classes="steps")
        steps.display = bool(self.steps)
        agents = Static(classes="agents")
        agents.display = False
        activity = Static(classes="activity")
        activity.display = False
        yield steps
        yield agents
        yield activity
        for renderable in self.body:
            yield Static(renderable, classes="body")

    def tick(self) -> None:
        self.frame += 1
        self.query_one(".steps", Static).update(self.render_steps())
        if self.agents or self.tools:
            self.draw_agents()

    def working(self) -> int:
        return sum(1 for agent in self.agents.values() if agent["state"] == "running")

    def record(self, line: str) -> bool:
        """Apply an agent or tool event; False for any other progress line."""
        if match := AGENT_EVENT.match(line):
            agent = self.agents.setdefault(match["name"], {
                "model": "", "task": "", "state": "running", "start": 0.0, "end": 0.0,
                "note": "", "calls": 0,
            })
            state, rest, now = match["state"], match["rest"], time.monotonic()
            if state == "start":
                model, _, task = rest.partition(" | ")
                agent.update(model=model, task=task, state="running", start=now, end=0.0,
                             note="", calls=agent["calls"] + 1)
            elif state == "progress":
                agent["note"] = rest
            else:
                agent.update(state=state, end=now, note="" if state == "done" else rest)
            return True
        if match := TOOL_EVENT.match(line):
            self.tools.append((match["name"], match["detail"]))
            return True
        return False

    def render_agents(self) -> Group:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1)
        grid.add_column(no_wrap=True, min_width=13)
        grid.add_column(no_wrap=True, style=MUTED)
        grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
        grid.add_column(justify="right", no_wrap=True, min_width=6)
        now = time.monotonic()
        for name, agent in self.agents.items():
            color = AGENT_COLORS.get(name, "")
            icon, icon_color = {
                "running": (SPINNER[self.frame % len(SPINNER)], color or GOLD),
                "done": ("✓", GREEN), "failed": ("✗", RED),
            }[agent["state"]]
            calls = f" ×{agent['calls']}" if agent["calls"] > 1 else ""
            task = agent["task"] + (f" · {agent['note']}" if agent["note"] else "")
            grid.add_row(
                Text(icon, f"bold {icon_color}"),
                Text(name + calls, f"bold {color}".strip()),
                agent["model"],
                Text(task, "" if agent["state"] == "running" else MUTED),
                Text(elapsed((agent["end"] or now) - agent["start"]), MUTED),
            )
        parts: list[Any] = [grid] if self.agents else []
        if self.tools:
            if any(step["state"] == "running" for step in self.steps):
                for tool, detail in self.tools[-3:]:
                    parts.append(Text.assemble(
                        ("  ↳ ", TEAL), (tool, "bold"), ("  " + detail, MUTED),
                        no_wrap=True, overflow="ellipsis",
                    ))
            count = len(self.tools)
            parts.append(Text(f"  {count} tool call{'s' * (count != 1)}", MUTED))
        return Group(*parts)

    def draw_agents(self) -> None:
        panel = self.query_one(".agents", Static)
        panel.display = True
        panel.update(self.render_agents())

    def render_steps(self) -> Table:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1)
        grid.add_column(no_wrap=True, min_width=12)
        grid.add_column(justify="right", no_wrap=True, min_width=7)
        grid.add_column(no_wrap=True, overflow="ellipsis")
        now = time.monotonic()
        for step in self.steps:
            icon, color = {
                "pending": ("○", MUTED), "running": (SPINNER[self.frame % len(SPINNER)], GOLD),
                "done": ("✓", GREEN), "warn": ("!", AMBER), "failed": ("✗", RED),
                "stopped": ("■", AMBER),
            }[step["state"]]
            took = (step["end"] or now) - step["start"] if step["start"] else None
            grid.add_row(
                Text(icon, f"bold {color}"),
                Text(step["name"], MUTED if step["state"] == "pending" else "bold"),
                Text(elapsed(took) if took is not None else "", MUTED),
                Text(step["detail"], MUTED),
            )
        return grid

    def start(self, index: int, detail: str = "") -> None:
        step = self.steps[index]
        if step["state"] != "running":
            step["start"] = time.monotonic()
        step.update(state="running", detail=detail or step["detail"])
        if self.ticker is None:
            self.ticker = self.set_interval(0.1, self.tick)
        else:
            self.ticker.resume()
        self.tick()

    def finish(self, index: int, detail: str = "", state: str = "done") -> None:
        step = self.steps[index]
        step.update(state=state, detail=detail, end=time.monotonic())
        if self.ticker is not None and not any(s["state"] == "running" for s in self.steps):
            self.ticker.pause()
            # The progress tail only helps while waiting, or to explain a failure.
            self.show_tail(8 if state == "failed" else 0)
        self.tick()

    def end(self, state: str, message: Text) -> None:
        """Close every unfinished step after a stop or an unexpected error."""
        for index, step in enumerate(self.steps):
            if step["state"] == "running":
                self.finish(index, state=state)
        self.add(message)

    def progress(self, line: str) -> None:
        self.lines.append(line)
        if self.record(line):
            self.draw_agents()
            if line.startswith("agent:"):
                refresh = getattr(self.app, "refresh_chrome", None)
                if refresh is not None:
                    refresh()
        else:
            self.tail.append(line)
            self.show_tail(4)

    def show_tail(self, count: int) -> None:
        activity = self.query_one(".activity", Static)
        activity.display = bool(count and self.tail)
        if activity.display:
            tail = "\n".join(self.tail[-count:])
            activity.update(Text(tail, no_wrap=True, overflow="ellipsis"))

    def add(self, renderable: Any) -> None:
        self.mount(Static(renderable, classes="body"))


class PromptInput(Input):
    """The chat input; Tab accepts the suggested /command instead of moving focus."""

    BINDINGS = [Binding("tab", "complete", "Complete", show=False)]

    def action_complete(self) -> None:
        self.cursor_position = len(self.value)
        self.action_cursor_right()  # Accepts the suggestion when the cursor is at the end.


class SessionPicker(ModalScreen[str | None]):
    """Arrow keys choose a past session, Enter resumes it, Esc cancels."""

    DEFAULT_CSS = """
    SessionPicker { align: center middle; }
    SessionPicker > Vertical {
        width: 100; max-width: 95%; height: auto; max-height: 80%;
        padding: 1 2; background: $surface; border: tall $primary;
    }
    SessionPicker OptionList { height: auto; max-height: 20; border: none; margin: 1 0; }
    SessionPicker OptionList:focus > .option-list--option-highlighted {
        color: $foreground; background: $primary 25%; text-style: bold;
    }
    """
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, sessions: list[dict[str, Any]], current: str):
        super().__init__()
        self.sessions, self.current = sessions, current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text("Resume a session", f"bold {GOLD}"))
            yield OptionList(*(
                Option(session_label(s, s["name"] == self.current), id=s["name"])
                for s in self.sessions
            ))
            yield Static(Text("Up/Down choose · Enter resume · Esc cancel", MUTED))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatApp(App[None]):
    TITLE = "MTG Agentic"
    CSS = """
    Screen { background: $background; }
    #topbar { height: 1; padding: 0 2; background: $panel; }
    #chat { height: 1fr; padding: 1 2 0 2; scrollbar-size-vertical: 1; }
    .you { height: auto; align-horizontal: right; margin-bottom: 1; }
    .you > Static {
        width: auto; max-width: 80%; padding: 0 1;
        background: $primary 12%; border-right: tall $primary;
    }
    Turn { height: auto; margin-bottom: 1; padding: 0 1; border-left: tall $secondary 50%; }
    Turn > .agents { margin: 1 0 0 2; }
    Turn > .activity { color: $text-muted; padding-left: 2; }
    Turn > .body { margin-top: 1; }
    Turn.plain > .body { margin-top: 0; }
    #prompt { margin: 0 2 1 2; border: tall $panel-lighten-2; background: $surface; }
    #prompt:focus { border: tall $primary; }
    """
    BINDINGS = [
        Binding("escape", "stop", "Stop run", priority=True),
        Binding("ctrl+l", "clear", "Clear chat"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, data_dir: Path, settings: Settings, runner: Runner = run_cli):
        super().__init__()
        self.data_dir, self.settings, self.runner = data_dir, settings, runner
        self.session = new_session_dir(data_dir)
        self.commander: str | None = None
        self.deck: dict[str, Any] | None = None
        self.deck_path: Path | None = None
        self.pool_path: Path | None = None
        self.transcript: list[dict[str, str]] = []
        self.last: Turn | None = None
        self.store: OracleStore | None = None
        self.runs = 0
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        yield VerticalScroll(id="chat")
        yield PromptInput(id="prompt", suggester=SuggestFromList(SUGGESTIONS, case_sensitive=False))
        yield Footer()

    async def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = THEME.name
        oracle = self.data_dir / "oracle.sqlite3"
        if oracle.is_file():
            self.store = OracleStore(oracle)
        self.chat = self.query_one("#chat", VerticalScroll)
        self.chat.anchor()
        await self.greet()
        self.refresh_chrome()
        self.query_one("#prompt", Input).focus()

    async def greet(self) -> None:
        greeting = [Text.assemble(
            ("Name a commander to begin", "bold"), ", for example Jodah, Archmage Eternal. Then "
            "describe the deck you want in plain words. ", ("/help", GOLD),
            " lists everything else.",
        )]
        if self.store is None or not self.store.ready:
            greeting.append(Text(
                "Card data is missing. Run once in a terminal: "
                "python -m mtg_agentic download-oracle",
                AMBER,
            ))
        if problem := embedding_problem():
            greeting.append(Text(
                f"{problem} Every step runs on this Python, so drafts with review will fail. "
                "Restart the TUI from the project's .venv, or use /review off.",
                AMBER,
            ))
        await self.reply(*greeting)

    def on_unmount(self) -> None:
        if self.store is not None:
            self.store.close()

    def lookup(self, name: str) -> dict[str, Any] | None:
        return self.store.get_card(name) if self.store is not None else None

    def refresh_chrome(self) -> None:
        left = Text.assemble(("◆ MTG Agentic", f"bold {GOLD}"), "   ")
        left.append(self.commander or "no commander yet", "" if self.commander else MUTED)
        right = Text(self.settings.describe(), MUTED)
        if self.busy:
            working = self.last.working() if self.last is not None else 0
            agents = f" {working} agent{'s' * (working != 1)}" if working else ""
            label = f"●{agents} working  "
            right = Text.assemble((label, GOLD), right)
        bar = Table.grid(expand=True)
        bar.add_column(no_wrap=True, overflow="ellipsis")
        bar.add_column(justify="right", no_wrap=True)
        bar.add_row(left, right)
        self.query_one("#topbar", Static).update(bar)
        prompt = self.query_one("#prompt", Input)
        if self.busy:
            prompt.placeholder = "Working... Esc stops the run"
        elif self.deck is not None:
            prompt.placeholder = "Ask the orchestrator about the deck or for changes, or /help"
        elif self.commander:
            prompt.placeholder = f"Describe a deck for {self.commander}, or /help"
        else:
            prompt.placeholder = "Name a commander to begin, or /help"

    async def reply(self, *renderables: Any, steps: Iterable[str] = ()) -> Turn:
        turn = Turn(steps, renderables)
        await self.chat.mount(turn)
        return turn

    def note(self, role: str, text: str) -> None:
        """Keep the session's chat so /session can bring it back."""
        self.transcript.append({"role": role, "text": text})
        self.save_session()

    def save_session(self) -> None:
        if not self.transcript and self.deck_path is None:
            return

        def relative(path: Path | None) -> str | None:
            if path is None:
                return None
            return path.name if path.parent == self.session else str(path)

        self.session.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1, "commander": self.commander,
            "valid": self.deck.get("valid") if self.deck else None,
            "deck": relative(self.deck_path), "pool": relative(self.pool_path),
            "updated": datetime.now().isoformat(timespec="seconds"), "turns": self.transcript,
        }
        text = json.dumps(data, indent=2) + "\n"
        (self.session / SESSION_FILE).write_text(text, encoding="utf-8")

    def save(self, name: str, data: Any) -> Path:
        self.session.mkdir(parents=True, exist_ok=True)
        path = self.session / name
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        command, _, rest = text.partition(" ") if text.startswith("/") else ("", "", text)
        command, rest = command.lower(), rest.strip()
        setting = command in WHILE_BUSY or (command == "/review" and bool(rest))
        if self.busy and not setting:
            self.notify("A run is in progress. Esc stops it.", severity="warning")
            return
        self.chat.anchor()  # Sending a message jumps back to the newest reply, as in any chat.
        await self.chat.mount(Vertical(Static(Text(text)), classes="you"))
        if not command or (command == "/draft" and rest):
            self.note("player", rest)
        await self.dispatch(command, rest)

    async def dispatch(self, command: str, rest: str) -> None:
        if not command:
            if self.commander is None:
                self.launch(["Card lookup"], self.card_job(rest, commander=True))
            elif self.deck is not None:
                # With a deck, the player talks to the orchestrator alone, never the council.
                self.launch(["Orchestrator"], self.edit_job(rest))
            else:
                self.start_draft(rest)
        elif command == "/draft":
            if self.commander is None:
                await self.reply(Text("Name a commander first.", AMBER))
            elif not rest:
                await self.reply(Text("Describe the deck: /draft Bracket 3 Myojin tribal", AMBER))
            else:
                self.start_draft(rest)
        elif command == "/new":
            await self.new_session()
        elif command == "/session":
            await self.pick_session(rest)
        elif command == "/help":
            await self.reply(render_help())
        elif command in {"/commander", "/card"} and rest:
            self.launch(["Card lookup"], self.card_job(rest, commander=command == "/commander"))
        elif command == "/search" and rest:
            job = self.single(["search", rest], render_search, lambda r: f"{len(r)} cards")
            self.launch(["Search"], job)
        elif command == "/ask":
            cards, question = parse_question(rest)
            if not cards or not question:
                await self.reply(Text(
                    "Mark each card in double brackets: /ask Does [[Doubling Season]] "
                    "double proliferate with [[Atraxa, Praetors' Voice]]?", AMBER,
                ))
                return
            argv = ["check-interaction", *cards, "--question", question]
            job = self.single([*argv, *self.settings.model_args()], render_interaction,
                              lambda r: r["status"])
            self.launch(["Rules check"], job)
        elif command in {"/model", "/mode", "/orchestrate"} or (command == "/review" and rest):
            await self.change_setting(command, rest)
        elif command in {"/validate", "/bracket", "/review", "/deck"}:
            await self.deck_command(command)
        elif command == "/load" and rest:
            await self.load(Path(rest.strip("\"'")).expanduser())
        elif command == "/export":
            await self.export(rest)
        elif command == "/log":
            lines = self.last.lines[-200:] if self.last is not None else []
            await self.reply(Text("\n".join(lines) or "Nothing has run yet.", MUTED))
        elif command == "/stop":
            self.action_stop()
        elif command == "/clear":
            self.action_clear()
        elif command == "/quit":
            self.exit()
        else:
            message = f"Unknown command or missing argument: {command}. Try /help."
            await self.reply(Text(message, AMBER))

    def start_draft(self, request: str) -> None:
        if self.settings.orchestrate:
            self.launch(COUNCIL_STEPS, self.council_job(request))
        else:
            self.launch(["Intake", "Discovery", "Draft"], self.draft_job(request))

    async def new_session(self) -> None:
        self.session = new_session_dir(self.data_dir, taken={self.session})
        self.commander = self.deck = self.deck_path = self.pool_path = self.last = None
        self.runs, self.transcript = 0, []
        await self.chat.remove_children()
        await self.greet()
        self.refresh_chrome()

    async def pick_session(self, choice: str) -> None:
        sessions = list_sessions(self.data_dir / "tui")
        if not sessions:
            await self.reply(Text("No past sessions yet.", MUTED))
            return
        if choice:
            index = int(choice) - 1 if choice.isdigit() else -1
            match = sessions[index] if 0 <= index < len(sessions) else next(
                (s for s in sessions if s["name"] == choice), None
            )
            if match is None:
                await self.reply(Text(f"No session {choice}; /session lists them.", AMBER))
            else:
                await self.resume(match["path"])
            return

        def picked(name: str | None) -> None:
            if name:
                self.call_later(self.resume, self.data_dir / "tui" / name)

        self.push_screen(SessionPicker(sessions, self.session.name), picked)

    async def resume(self, folder: Path) -> None:
        info = read_session(folder)
        self.session, self.transcript, self.last = folder, list(info["turns"]), None
        self.deck = self.deck_path = self.pool_path = None
        if info["deck"] is not None:
            with contextlib.suppress(OSError, ValueError):
                self.deck = json.loads(info["deck"].read_text(encoding="utf-8"))
                self.deck_path, self.pool_path = info["deck"], pool_for(info["deck"])
        self.commander = info["commander"] or (self.deck or {}).get("commander")
        numbers = [int(m[1]) for p in folder.glob("*.json") if (m := re.match(r"(\d+)-", p.name))]
        self.runs = max(numbers, default=0)
        await self.chat.remove_children()
        await self.reply(Text.assemble(("Resumed session ", MUTED), (folder.name, f"bold {GOLD}")))
        for entry in self.transcript:
            if entry["role"] == "player":
                await self.chat.mount(Vertical(Static(Text(entry["text"])), classes="you"))
            else:
                color = GOLD if entry["role"] == "orchestrator" else MUTED
                label = (entry["role"], f"bold {color}")
                await self.reply(Text.assemble(label, "  ", entry["text"]))
        if self.deck is not None:
            await self.reply(render_deck(self.deck, self.lookup, self.deck_path))
        self.refresh_chrome()

    async def deck_command(self, command: str) -> None:
        if self.deck is None or self.deck_path is None:
            message = "No deck yet. Describe one to draft it, or /load a saved deck JSON."
            await self.reply(Text(message, AMBER))
        elif command == "/deck":
            await self.reply(render_deck(self.deck, self.lookup, self.deck_path))
        elif command == "/validate":
            self.launch(["Validate"], self.single(
                ["validate", str(self.deck_path)], render_validation,
                lambda r: "passed" if r.get("valid") else f"{len(r.get('errors', []))} problems",
            ))
        elif command == "/bracket":
            self.launch(["Bracket"], self.single(
                ["bracket-check", str(self.deck_path), *self.settings.model_args()],
                render_bracket, lambda r: r.get("verdict", ""),
            ))
        else:
            self.launch(["Review"], self.single(
                ["review-deck", str(self.deck_path), *self.settings.model_args()],
                render_review, lambda r: r.get("status", "").replace("_", " "),
            ))

    async def load(self, path: Path) -> None:
        try:
            deck = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            await self.reply(Text(f"Could not read {path}: {exc}", RED))
            return
        if not (
            isinstance(deck, dict)
            and isinstance(deck.get("commander"), str)
            and isinstance(deck.get("cards"), dict)
        ):
            await self.reply(Text("Expected deck JSON with a commander and a cards mapping.", RED))
            return
        self.deck, self.deck_path, self.commander = deck, path, deck["commander"]
        self.pool_path = pool_for(path)
        self.refresh_chrome()
        await self.reply(render_deck(deck, self.lookup, path))

    async def export(self, rest: str) -> None:
        """Write the deck as a text list beside its JSON, or to a new PATH, and copy it."""
        if self.deck is None or self.deck_path is None:
            message = "No deck yet. Describe one to draft it, or /load a saved deck JSON."
            await self.reply(Text(message, AMBER))
            return
        target = (
            Path(rest.strip("\"'")).expanduser() if rest
            else free_path(self.deck_path.with_suffix(".txt"))
        )
        text = decklist_text(self.deck)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Like the CLI's --output, an export never overwrites an existing file.
            with target.open("x", encoding="utf-8") as file:
                file.write(text)
        except FileExistsError:
            await self.reply(Text(f"{target} already exists; choose a new filename.", AMBER))
            return
        except OSError as exc:
            await self.reply(Text(f"Could not write {target}: {exc}", RED))
            return
        self.copy_to_clipboard(text)
        unique = len(self.deck["cards"]) + 1
        total = sum(self.deck["cards"].values()) + 1
        await self.reply(Text.assemble(
            "Exported ", (f"{unique} unique cards", "bold"), f" ({total} total) to ",
            (shown(target), GOLD), ("\nAlso copied to the clipboard.", MUTED),
        ))

    async def change_setting(self, command: str, value: str) -> None:
        """`/model`, `/mode`, `/orchestrate` and `/review on|off`; alone they show the choice."""
        s = self.settings
        try:
            if command == "/model" and value:
                s.choose_model(value.split())
            elif command == "/mode" and value:
                s.choose_mode(value.lower())
            elif command == "/orchestrate" and value:
                s.choose_orchestrate(value.lower())
            elif command == "/review":
                s.choose_review(value.lower())
        except ValueError as exc:
            await self.reply(Text(str(exc), AMBER))
            return
        self.refresh_chrome()
        if command == "/model":
            rows = [
                ("provider", s.provider), ("model", s.model or "provider default"),
                ("providers", Text(", ".join(PROVIDERS), MUTED)),
            ]
        elif command == "/mode":
            rows = [
                ("mode", s.mode or "auto: read from the request"),
                ("modes", Text(", ".join(("auto", *BRACKET3_MODES)), MUTED)),
            ]
        elif command == "/orchestrate":
            lead, members = s.council_roles()
            rows = [
                ("orchestrate", "on: deck requests go to the council" if s.orchestrate
                 else "off: deck requests use one model"),
                ("orchestrator", lead), ("members", members),
            ]
        else:
            rows = [("review", "on: drafts run the bracket and rules review" if s.review
                     else "off: drafts skip the bracket and rules review")]
        await self.reply(labeled(rows))

    def launch(self, steps: list[str], job: Callable[[Turn], Awaitable[None]]) -> None:
        """Run `job` as the one active run; Esc cancels it, which kills its CLI process."""

        async def body() -> None:
            turn = self.last = await self.reply(steps=steps)
            self.busy = True
            self.refresh_chrome()
            try:
                await job(turn)
            except asyncio.CancelledError:
                turn.end("stopped", Text("Stopped.", AMBER))
                raise
            except Exception as exc:  # A surprising result must not take the chat down.
                message = f"Unexpected error: {type(exc).__name__}: {exc}. See /log."
                turn.end("failed", Text(message, RED))
            finally:
                self.busy = False
                self.refresh_chrome()

        self.run_worker(body(), group="run", exclusive=True, exit_on_error=False)

    async def step(
        self, turn: Turn, index: int, argv: list[str],
        on_line: Callable[[str], None] | None = None,
    ) -> Any:
        """Run one CLI command as step `index`; return its JSON, or None after showing why not."""
        turn.start(index)

        def progress(line: str) -> None:
            turn.progress(line)
            if on_line is not None:
                on_line(line)

        code, result = await self.runner(argv, self.data_dir, progress)
        # Exit 1 is a negative verdict with a full result; exit 2 is an error envelope.
        if code in (0, 1) and result is not None:
            return result
        for i, step in enumerate(turn.steps):
            if step["state"] == "running":
                turn.finish(i, f"exit {code}", "failed")
        errors = result.get("errors") if isinstance(result, dict) else None
        if errors:
            turn.add(problems(errors))
        else:
            turn.add(Text(f"The command exited with code {code} and no result.", RED))
        return None

    def single(
        self, argv: list[str], render: Callable[[Any], Any], detail: Callable[[Any], str]
    ) -> Callable[[Turn], Awaitable[None]]:
        async def job(turn: Turn) -> None:
            result = await self.step(turn, 0, argv)
            if result is not None:
                turn.finish(0, detail(result))
                turn.add(render(result))

        return job

    def card_job(self, name: str, *, commander: bool) -> Callable[[Turn], Awaitable[None]]:
        async def job(turn: Turn) -> None:
            card = await self.step(turn, 0, ["card", name])
            if card is None:
                if commander:
                    hint = "Name a commander to begin, for example Jodah, Archmage Eternal."
                    turn.add(Text(hint, MUTED))
                return
            turn.finish(0, card["name"])
            turn.add(render_card(card))
            if not commander:
                return
            self.commander = card["name"]
            self.refresh_chrome()
            self.note("assistant", f"Commander: {card['name']}.")
            front = card["type_line"].split(" // ")[0]
            if "can be your commander" not in card["oracle_text"] and not (
                "Legendary" in front and "Creature" in front
            ):
                warning = "This is not a legendary creature, so intake will likely reject it."
                turn.add(Text(warning, AMBER))
            turn.add(Text.assemble(
                (card["name"], "bold"),
                " is your commander. Describe the deck you want, for example: ",
                ("Bracket 3, Myojin tribal with divinity counters and proliferate", GOLD),
            ))

        return job

    async def prepare(self, turn: Turn, request: str, model: list[str]) -> tuple[str, Path] | None:
        """Intake and discovery as steps 0 and 1; returns the run's file stem and pool path."""
        self.runs += 1
        stem = f"{self.runs:02d}"
        mode = ["--bracket3-mode", self.settings.mode] if self.settings.mode else []
        argv = ["intake", "--commander", self.commander, "--request", request, *mode, *model]
        spec = await self.step(turn, 0, argv)
        if spec is None:
            return None
        spec_path = self.save(f"{stem}-spec.json", spec)
        turn.finish(0, f"{''.join(spec['color_identity'])} · Bracket {spec['bracket']}")
        turn.add(render_spec(spec))
        pool = await self.step(turn, 1, ["discover", "--spec", str(spec_path), *model])
        if pool is None:
            return None
        pool_path = self.save(f"{stem}-pool.json", pool)
        turn.finish(1, f"{len(pool['cards'])} candidates")
        turn.add(render_pool(pool))
        return stem, pool_path

    def show_deck(self, turn: Turn, index: int, stem: str, deck: dict[str, Any]) -> None:
        self.deck, self.deck_path = deck, self.save(f"{stem}-deck.json", deck)
        self.pool_path = pool_for(self.deck_path)
        errors = deck.get("validation", {}).get("errors", [])
        problems = f"{len(errors)} problem{'s' * (len(errors) != 1)}"
        status = "passed" if deck.get("valid") else problems
        turn.finish(index, status, "done" if deck.get("valid") else "warn")
        if deck.get("council"):
            turn.add(render_council(deck["council"]))
        turn.add(render_deck(deck, self.lookup, self.deck_path))
        turn.add(Text("Reply to ask the orchestrator about this deck or for changes. "
                      "/draft starts over, /export saves the list.", MUTED))
        self.refresh_chrome()
        self.note("assistant", f"Drafted {self.deck_path.name}: {status}.")

    def draft_job(self, request: str) -> Callable[[Turn], Awaitable[None]]:
        async def job(turn: Turn) -> None:
            model = self.settings.model_args()
            prepared = await self.prepare(turn, request, model)
            if prepared is None:
                return
            stem, pool_path = prepared
            review = [] if self.settings.review else ["--skip-review"]
            deck = await self.step(turn, 2, ["draft", "--pool", str(pool_path), *review, *model])
            if deck is not None:
                self.show_deck(turn, 2, stem, deck)

        return job

    def council_job(self, request: str) -> Callable[[Turn], Awaitable[None]]:
        async def job(turn: Turn) -> None:
            prepared = await self.prepare(turn, request, self.settings.lead_args())
            if prepared is None:
                return
            stem, pool_path = prepared
            last = len(COUNCIL_STEPS) - 1

            def close_until(index: int) -> None:
                for i in range(2, index):
                    if turn.steps[i]["state"] in {"running", "pending"}:
                        turn.finish(i, turn.steps[i]["detail"])

            def advance(line: str) -> None:
                # Council stages 1-5 are steps 2-6; each stage line closes the ones before it.
                match = COUNCIL_STAGE.search(line)
                if match:
                    index = min(1 + int(match[1]), last)
                    close_until(index)
                    turn.start(index, match[2])

            review = [] if self.settings.review else ["--skip-review"]
            argv = ["council", "--pool", str(pool_path), *review, *self.settings.model_args()]
            deck = await self.step(turn, 2, argv, on_line=advance)
            if deck is not None:
                close_until(last)
                self.show_deck(turn, last, stem, deck)

        return job

    def next_version(self) -> Path:
        """`01-deck.json` -> `01-deck-v2.json`, `-v3`... in this session."""
        assert self.deck_path is not None
        base = re.sub(r"-v\d+$", "", self.deck_path.stem)
        number = 2
        while (self.session / f"{base}-v{number}.json").exists():
            number += 1
        return self.session / f"{base}-v{number}.json"

    def edit_job(self, request: str) -> Callable[[Turn], Awaitable[None]]:
        async def job(turn: Turn) -> None:
            assert self.deck_path is not None
            pool = ["--pool", str(self.pool_path)] if self.pool_path else []
            review = [] if self.settings.review else ["--skip-review"]
            argv = ["edit-deck", str(self.deck_path), *pool, "--request", request, *review,
                    *self.settings.orchestrator_args()]
            deck = await self.step(turn, 0, argv)
            if deck is None:
                return
            edit = deck.get("edit", {})
            swaps = edit.get("swaps", [])
            if edit.get("changed"):
                target = self.next_version()
            else:
                # Only the conversation grew; keep it with this session's current version.
                inside = self.deck_path.parent == self.session
                target = self.deck_path if inside else self.next_version()
            self.session.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(deck, indent=2) + "\n", encoding="utf-8")
            self.deck, self.deck_path = deck, target
            errors = deck.get("validation", {}).get("errors", [])
            if not edit.get("changed"):
                turn.finish(0, "answered")
            elif deck.get("valid"):
                turn.finish(0, f"{len(swaps)} swap{'s' * (len(swaps) != 1)} · passed")
            else:
                problems = f"{len(errors)} problem{'s' * (len(errors) != 1)}"
                turn.finish(0, f"{len(swaps)} swap{'s' * (len(swaps) != 1)} · {problems}", "warn")
            turn.add(render_edit(edit))
            if edit.get("changed"):
                turn.add(render_deck(deck, self.lookup, self.deck_path, columns=False))
            changes = "".join(f"\n- {s['remove']} + {s['add']}" for s in swaps)
            self.note("orchestrator", edit.get("reply", "") + changes)

        return job

    def action_stop(self) -> None:
        if isinstance(self.screen, SessionPicker):
            self.screen.dismiss(None)
            return
        if not self.busy:
            self.notify("Nothing is running.")
            return
        self.workers.cancel_group(self, "run")

    def action_clear(self) -> None:
        if self.busy:
            self.notify("A run is in progress. Esc stops it.", severity="warning")
            return
        self.chat.remove_children()


def run_tui(data_dir: Path, *, provider: str | None = None, model: str | None = None) -> int:
    settings = Settings(
        provider=provider or os.getenv("MTG_LLM_PROVIDER") or "ollama",
        model=model or os.getenv("MTG_LLM_MODEL") or None,
    )
    ChatApp(data_dir, settings).run()
    return 0
