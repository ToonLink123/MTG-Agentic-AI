"""Opt-in Claude provider over the official Anthropic SDK (`pip install anthropic`)."""

from __future__ import annotations

import json
import math
import shutil
import time
from typing import Any, TextIO

from .compaction import TRIGGER, Compactor
from .llm import LLMError, Message
from .schema import SchemaError, load_json, validate_json

try:
    import anthropic
except ImportError:  # Optional dependency; only this provider needs it.
    anthropic = None

# Server-side refusal fallback exists only for these models; Haiku 5.5 has none.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# Haiku 5.5 bills prompts above this size on its higher rate card.
HAIKU_TIER_TOKENS = 100_000


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
        # Haiku 5.5 is cheaper while a prompt stays within its tier, so compact to stay there.
        self.compact_at = HAIKU_TIER_TOKENS if self._haiku else None
        self.usage = dict.fromkeys(
            ("calls", "input", "cached", "written", "output", "largest", "compactions"), 0
        )
        self._compactor = Compactor()
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any:
        text = self._ask(messages, schema)
        try:
            result = load_json(text)
            validate_json(result, schema)
            return result
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
            return load_json(self._ask([*messages, retry], schema))

    def _ask(self, messages: list[Message], schema: dict[str, Any]) -> str:
        # Leading system messages become the system prompt; the rest are conversation turns.
        split = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
        system = {"system": "\n\n".join(m["content"] for m in messages[:split])} if split else {}
        turns = [dict(m) for m in messages[split:]]
        output_config = {
            "effort": self.effort,
            # Unsupported constraints move into descriptions; they are validated locally.
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        }
        fallback = (
            {"betas": ["server-side-fallback-2026-07-01"], "extra_body": {"fallbacks": "default"}}
            if self.model in FALLBACK_MODELS
            else {}
        )
        status = _Status(self.progress, self.model) if self.progress else None
        try:
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
            with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=turns,
                # Summaries cost nothing extra and let the progress line show live reasoning.
                thinking={"type": "adaptive", "display": "summarized"},
                output_config=output_config,
                cache_control={"type": "ephemeral"},
                **fallback,
                **system,
            ) as stream:
                for event in stream:
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
            raise LLMError("Claude truncated its JSON response")
        # A mid-stream fallback keeps the declined partial; only text after the last switch counts.
        content = response.content
        start = max(
            (i + 1 for i, block in enumerate(content) if block.type == "fallback"), default=0
        )
        return "".join(block.text for block in content[start:] if block.type == "text")

    def _fit(
        self, turns: list[dict[str, Any]], system: dict[str, str], output_config: dict[str, Any]
    ) -> list[str]:
        """Compact the turns, with no model pass, before the prompt reaches `compact_at`."""
        originals = [turn["content"] for turn in turns]
        texts = self._compactor.reapply(originals)
        if not self.compact_at:
            return texts
        trigger = int(self.compact_at * TRIGGER)
        fixed = len(system.get("system", "")) + len(json.dumps(output_config))
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
        if totals["compactions"]:
            line += f" | compacted {totals['compactions']} request(s)"
        print(line, file=self.progress, flush=True)


class _Status:
    """A self-overwriting progress line for one streamed call; drawn only on a terminal."""

    def __init__(self, out: TextIO, model: str):
        self.out, self.model = out, model
        self.live = out.isatty()
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


def _detail(exc: Any) -> str:
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message or f"request {exc.request_id}"
