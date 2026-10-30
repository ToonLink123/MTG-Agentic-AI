"""Opt-in Claude provider over the official Anthropic SDK (`pip install anthropic`)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from .compaction import TRIGGER, Compactor
from .llm import MALFORMED, LLMError, Message, malformed_kind, malformed_summary
from .schema import SchemaError, load_json, validate_json

try:
    import anthropic
except ImportError:  # Optional dependency; only this provider needs it.
    anthropic = None

# Server-side refusal fallback exists only for these models; Haiku 5.5 has none.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# Haiku 5.5 bills prompts above this size on its higher rate card.
HAIKU_TIER_TOKENS = 100_000
# Enums longer than this (card names from a large pool) are sent as plain strings: compiling
# hundreds of names into the output grammar costs more than the local check that replaces it.
WIRE_ENUM_LIMIT = 64
NAME_HINT = "An exact card name from the supplied data; checked locally."
# A request whose cached prefix another request is writing waits up to this long for that one to
# start answering, so it reads the cache instead of paying to write it again.
WARM_WAIT = 180.0
# A written prefix is read for five minutes after its last use; treat it as warm a bit less.
CACHE_TTL = 270.0


def _wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema Claude decodes with. Pick tiers and any long enum become plain strings;
    request_json still validates the answer against the full schema."""
    result = copy.deepcopy(schema)
    tiers = result.get("properties", {}).get("picks", {}).get("properties", {})
    for tier in tiers.values():
        if "enum" in tier.get("items", {}):
            tier["items"] = {"type": "string", "description": NAME_HINT}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("enum"), list) and len(node["enum"]) > WIRE_ENUM_LIMIT:
                node.pop("enum")
                node["description"] = NAME_HINT
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(result)
    return result


class ClaudeClient:
    """API keys are supplied by the caller/environment, never stored."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        api_key: str,
        timeout: float = 300,
        effort: str = "high",
        max_tokens: int = 64000,
        progress: TextIO | None = None,
        concurrency: int = 8,
        compact_at: int | None = None,
    ):
        if not api_key:
            raise ValueError("Set CLAUDE_API (environment or .env) for the claude provider")
        if not model.strip() or timeout <= 0:
            raise ValueError("LLM model and a positive timeout are required")
        if anthropic is None:
            raise ValueError("The claude provider needs the Anthropic SDK: pip install anthropic")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.progress = progress
        # Requests independent questions may run at once; the SDK retries rate limits.
        self.concurrency = concurrency
        self._lock = threading.Lock()
        # Prompts are never cut by default, so large candidate pools reach the model whole.
        # Pass compact_at=HAIKU_TIER_TOKENS to keep Haiku within its cheaper tier instead.
        self.compact_at = compact_at
        self.usage = dict.fromkeys(
            ("calls", "input", "cached", "written", "output", "largest", "compactions", *MALFORMED), 0
        )
        self._compactor = Compactor()
        # Shared prefixes being written (key -> released when the response starts) and written.
        self._warming: dict[str, threading.Event] = {}
        self._warmed: dict[str, float] = {}
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any:
        text = self._ask(messages, schema)
        try:
            return self._parse(text, schema)
        except SchemaError as exc:
            # Structured outputs cannot enforce lengths or ranges. Retry once, append-only,
            # so the original prompt is served from cache.
            retry = {
                "role": "user",
                "content": json.dumps(
                    {
                        "previous_response": text,
                        "schema_errors": exc.errors,
                        "instruction": "Return the complete corrected JSON.",
                    }
                ),
            }
            return self._parse(self._ask([*messages, retry], schema), schema)

    def _parse(self, text: str, schema: dict[str, Any]) -> Any:
        try:
            result = load_json(text)
            validate_json(result, schema)
            return result
        except SchemaError as exc:
            self._malformed(malformed_kind(exc), str(exc))
            raise

    def _malformed(self, kind: str, detail: str = "") -> None:
        with self._lock:
            self.usage[kind] += 1
            summary = malformed_summary(self.usage)
        if self.progress:
            line = f"MALFORMED {MALFORMED[kind]}" + (f": {detail[:120]}" if detail else "")
            line += f" | {summary} so far"
            print("  " + line, file=self.progress, flush=True)

    def _ask(self, messages: list[Message], schema: dict[str, Any]) -> str:
        split = _system_count(messages)
        system = _system(messages[:split])
        turns = [dict(m) for m in messages[split:]]
        output_config = {
            "effort": self.effort,
            # Unsupported constraints move into descriptions; they are validated locally.
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(_wire_schema(schema))},
        }
        for turn, text in zip(turns, self._fit(turns, system, output_config)):
            turn["content"] = text
        if len(turns) > 1:
            # Earlier turns (the candidate pool before a repair, a prompt before a retry)
            # repeat the previous request; mark them so the read is guaranteed. Top-level
            # caching covers the newest turn, so identical re-asks hit too.
            turns[-2]["content"] = [
                {
                    "type": "text",
                    "text": turns[-2]["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        content = self._stream(system, messages=turns, output_config=output_config)
        return "".join(block.text for block in content if block.type == "text")

    def tool_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """One turn of native tool use (see agent.run_tools): the tool calls Claude made, and
        its content to send back as the assistant turn, thinking included."""
        split = _system_count(messages)
        system = _system(messages[:split])
        specs = [
            {
                "name": tool["name"], "description": tool["description"], "strict": True,
                "input_schema": anthropic.transform_schema(_wire_schema(tool["input_schema"])),
            }
            for tool in tools
        ]
        content = self._stream(
            system, messages=[dict(m) for m in messages[split:]], tools=specs,
            tool_choice={"type": "auto"}, output_config={"effort": self.effort},
            prefix=json.dumps(specs, sort_keys=True),
        )
        calls = [
            {"id": block.id, "name": block.name, "input": block.input}
            for block in content if block.type == "tool_use"
        ]
        return {
            "calls": calls,
            "text": "".join(block.text for block in content if block.type == "text"),
            "content": [param for block in content if (param := _param(block)) is not None],
        }

    def _stream(self, system: dict[str, Any], *, prefix: str = "", **request: Any) -> list[Any]:
        """Stream one request; return its content blocks after any mid-stream fallback."""
        fallback = (
            {"betas": ["server-side-fallback-2026-07-01"], "extra_body": {"fallbacks": "default"}}
            if self.model in FALLBACK_MODELS
            else {}
        )
        # Parallel calls share one terminal, so only the main thread draws a live line.
        live = threading.current_thread() is threading.main_thread()
        status = _Status(self.progress, self.model, live=live) if self.progress else None
        with self._prefix(_prefix_key(self.model, prefix, system)) as started:
            try:
                with self._client.beta.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    # Summaries cost nothing extra and let the progress line show live reasoning.
                    thinking={"type": "adaptive", "display": "summarized"},
                    cache_control={"type": "ephemeral"},
                    **request,
                    **fallback,
                    **system,
                ) as stream:
                    for event in stream:
                        started()
                        if status:
                            status.update(event)
                    response = stream.get_final_message()
            except anthropic.AuthenticationError:
                raise LLMError("Claude rejected the API key; check CLAUDE_API") from None
            except anthropic.APIStatusError as exc:
                raise LLMError(f"Claude API returned HTTP {exc.status_code}: {_detail(exc)}") from None
            except anthropic.APIConnectionError as exc:
                raise LLMError("Cannot reach the Claude API; check your network connection") from exc
            finally:
                if status:
                    status.clear()
        self._record(response.usage, status)
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined the request")
        if response.stop_reason == "max_tokens":
            self._malformed("truncated")
            raise LLMError("Claude truncated its JSON response")
        # A mid-stream fallback keeps the declined partial; only content after the last switch counts.
        content = response.content
        start = max(
            (i + 1 for i, block in enumerate(content) if block.type == "fallback"), default=0
        )
        return content[start:]

    @contextmanager
    def _prefix(self, key: str | None) -> Iterator[Callable[[], None]]:
        """Parallel requests sharing a cached prefix (council members reading one pool) would
        each pay to write it, since a cache entry is readable only once a response has begun.
        The first one goes ahead; the others wait until it starts answering, then read it."""
        if key is None:
            yield lambda: None
            return
        with self._lock:
            warm = time.monotonic() - self._warmed.get(key, -CACHE_TTL) < CACHE_TTL
            gate = None if warm else self._warming.get(key)
            leader = not warm and gate is None
            if leader:
                gate = self._warming[key] = threading.Event()
        if gate is not None and not leader:
            gate.wait(WARM_WAIT)
        if not leader:
            yield lambda: None
            return

        def started() -> None:
            if not gate.is_set():
                with self._lock:
                    self._warmed[key] = time.monotonic()
                    self._warming.pop(key, None)
                gate.set()

        try:
            yield started
        finally:
            if not gate.is_set():
                # Failed before answering: release the others, which then write it themselves.
                with self._lock:
                    self._warming.pop(key, None)
                gate.set()

    def _fit(
        self, turns: list[dict[str, Any]], system: dict[str, str], output_config: dict[str, Any]
    ) -> list[str]:
        """Compact the turns, with no model pass, before the prompt reaches `compact_at`."""
        originals = [turn["content"] for turn in turns]
        texts = self._compactor.reapply(originals)
        if not self.compact_at:
            return texts
        trigger = int(self.compact_at * TRIGGER)
        fixed = len(_system_text(system)) + len(json.dumps(output_config))
        chars = fixed + sum(map(len, texts))
        # Prompts run at least 1.5 characters per token, so below this bound the request is
        # certainly under the trigger and needs no count.
        if chars / 1.5 < trigger:
            return texts
        tokens = self._client.messages.count_tokens(
            model=self.model,
            messages=[{**turn, "content": text} for turn, text in zip(turns, texts)],
            output_config=output_config,
            **system,
        ).input_tokens
        if tokens < trigger:
            return texts
        ratio = chars / tokens
        texts, short = self._compactor.shrink(originals, math.ceil((tokens - trigger) * ratio))
        with self._lock:
            self.usage["compactions"] += 1
        if self.progress:
            after = round((fixed + sum(map(len, texts))) / ratio)
            note = f"compacted prompt ~{tokens:,} -> ~{after:,} tokens (budget {self.compact_at:,})"
            if short:
                note += f"; stopped short, ~{round(short / ratio):,} tokens over"
            print("  " + note, file=self.progress, flush=True)
        return texts

    def _record(self, usage: Any, status: _Status | None) -> None:
        cached = usage.cache_read_input_tokens or 0
        written = usage.cache_creation_input_tokens or 0
        prompt = usage.input_tokens + cached + written
        totals = self.usage
        with self._lock:
            totals["calls"] += 1
            totals["input"] += usage.input_tokens
            totals["cached"] += cached
            totals["written"] += written
            totals["output"] += usage.output_tokens
            totals["largest"] = max(totals["largest"], prompt)
        if status:
            line = (
                f"{self.model} | {status.elapsed():.1f}s | prompt {prompt:,} tok "
                f"({cached:,} cached, {written:,} written) | out {usage.output_tokens:,}"
            )
            if self._haiku and prompt > HAIKU_TIER_TOKENS:
                line += " | over 100K: higher Haiku rate"
            print("  " + line, file=status.out, flush=True)

    @property
    def _haiku(self) -> bool:
        return self.model.startswith("claude-haiku-5-5")

    def report(self) -> None:
        """Print run totals to the progress stream."""
        totals = self.usage
        if not self.progress or not totals["calls"]:
            return
        prompt = totals["input"] + totals["cached"] + totals["written"]
        line = (
            f"claude: {totals['calls']} calls | prompt {prompt:,} tok, "
            f"{totals['cached'] / prompt:.0%} from cache | out {totals['output']:,} | "
            f"largest prompt {totals['largest']:,}"
        )
        if self._haiku:
            over = totals["largest"] > HAIKU_TIER_TOKENS
            line += " (over Haiku's 100K tier)" if over else " (within Haiku's 100K tier)"
        line += f" | {malformed_summary(totals)}"
        if totals["compactions"]:
            line += f" | compacted {totals['compactions']} request(s)"
        print(line, file=self.progress, flush=True)


class _Status:
    """A self-overwriting progress line for one streamed call; drawn only on a terminal."""

    def __init__(self, out: TextIO, model: str, *, live: bool = True):
        self.out, self.model = out, model
        self.live = live and out.isatty()
        self.start = self.drawn = time.monotonic()
        self.phase, self.chars, self.thought, self.width = "waiting", 0, "", 0

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def update(self, event: Any) -> None:
        if event.type == "content_block_start":
            kind = event.content_block.type
            self.phase = {"thinking": "thinking", "text": "writing"}.get(kind, self.phase)
        elif event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                self.thought = (self.thought + event.delta.thinking)[-200:]
            elif event.delta.type == "text_delta":
                self.chars += len(event.delta.text)
        now = time.monotonic()
        if self.live and now - self.drawn >= 0.1:
            self.drawn = now
            line = f"  {self.model} | {self.phase} {self.elapsed():.0f}s"
            if self.phase == "writing":
                line += f" | {self.chars:,} chars"
            elif self.thought:
                # Keep the newest reasoning visible; the line must not wrap or \r breaks.
                room = shutil.get_terminal_size().columns - len(line) - 7
                thought = " ".join(self.thought.split())
                if room > 10:
                    line += " | " + (thought if len(thought) <= room else "..." + thought[-room:])
            self._draw(line)

    def clear(self) -> None:
        if self.width:
            self._draw("")
            self.out.write("\r")

    def _draw(self, line: str) -> None:
        self.out.write("\r" + line.ljust(self.width))
        self.out.flush()
        self.width = len(line)


def _system_count(messages: list[dict[str, Any]]) -> int:
    """How many leading messages are system messages."""
    return next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))


def _system(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Leading system messages become the system prompt. With several, the first is shared
    context (a candidate pool) that later stages and parallel calls send byte for byte, so a
    cache breakpoint right after it lets every one of them read it."""
    texts = [m["content"] for m in messages]
    if not texts:
        return {}
    if len(texts) == 1:
        return {"system": texts[0]}
    return {"system": [
        {"type": "text", "text": texts[0], "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "\n\n".join(texts[1:])},
    ]}


def _system_text(system: dict[str, Any]) -> str:
    value = system.get("system", "")
    return value if isinstance(value, str) else "".join(block["text"] for block in value)


def _prefix_key(model: str, prefix: str, system: dict[str, Any]) -> str | None:
    """The shared prefix a request writes to the cache, when it has one (see _system)."""
    value = system.get("system")
    if not isinstance(value, list):
        return None
    digest = hashlib.sha256()
    for part in (model, prefix, value[0]["text"]):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _param(block: Any) -> dict[str, Any] | None:
    """A response content block as it is sent back in the next request."""
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if block.type == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return None


def _detail(exc: Any) -> str:
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message or f"request {exc.request_id}"
