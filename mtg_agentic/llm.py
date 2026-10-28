"""Small, injectable JSON LLM interface. Local Ollama is the default."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .schema import SchemaError, load_json, validate_json

Message = dict[str, str]
PROVIDERS = ("ollama", "llama-cpp", "openai-compatible", "claude")
# A malformed reply: unparseable JSON, a schema violation, or output cut off at the token limit.
MALFORMED = {"json": "invalid JSON", "schema": "schema", "truncated": "truncated"}

# LiquidAI LFM3 1.7B, Q8_0 GGUF (https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), served by llama.cpp.
LFM3_MODEL = "LiquidAI/LFM3-1.7B-GGUF:Q8_0"

logger = logging.getLogger(__name__)
T = TypeVar("T")


class LLMError(RuntimeError):
    pass


class JSONModel(Protocol):
    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any: ...


def request_json(model: JSONModel, messages: list[Message], schema: dict[str, Any]) -> Any:
    result = model.complete_json(messages, schema)
    # Validate even injected providers. Structured-output hints are not a trust boundary.
    validate_json(result, schema)
    return result


def parallel(calls: list[Callable[[], T]], workers: int) -> list[T]:
    """Run independent calls on up to `workers` threads; results keep the input order.

    The first exception cancels calls that have not started yet and is re-raised.
    """
    if workers <= 1 or len(calls) <= 1:
        return [call() for call in calls]
    with ThreadPoolExecutor(max_workers=min(workers, len(calls))) as pool:
        futures = [pool.submit(call) for call in calls]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def concurrency(model: JSONModel) -> int:
    """Requests this model takes at once; local servers queue them, so they default to one."""
    value = getattr(model, "concurrency", 1)
    return value if type(value) is int and value > 1 else 1


def malformed_kind(exc: SchemaError) -> str:
    return "json" if any(error.get("error") == "json" for error in exc.errors) else "schema"


def malformed_summary(counts: dict[str, int]) -> str:
    total = sum(counts[kind] for kind in MALFORMED)
    if not total:
        return "0 malformed"
    parts = ", ".join(f"{counts[k]} {label}" for k, label in MALFORMED.items() if counts[k])
    return f"{total} malformed ({parts})"


def _chat_completion(body: dict[str, Any]) -> tuple[str, bool, int, int]:
    try:
        choice = body["choices"][0]
        message = choice["message"]
        if message.get("refusal"):
            raise LLMError("LLM refused the request")
        truncated = choice.get("finish_reason") == "length"
        content = message.get("content")
        if not isinstance(content, str) and not truncated:
            raise TypeError
        usage = body.get("usage") or {}
        return content or "", truncated, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError("LLM returned an invalid chat-completions envelope") from exc


class HTTPJSONModel:
    provider = "llm"
    # A local server answers one request per slot and queues the rest, which can trip the
    # read timeout; hosted APIs override this.
    concurrency = 1

    def __init__(
        self, model: str, *, base_url: str, timeout: float = 300, api_key: str | None = None
    ):
        parts = urlsplit(base_url)
        if not model.strip() or timeout <= 0:
            raise ValueError("LLM model and a positive timeout are required")
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Invalid LLM base URL")
        if parts.scheme != "https" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Non-local LLM endpoints require HTTPS")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key
        self.usage = dict.fromkeys(("calls", "prompt", "output", *MALFORMED), 0)
        self._lock = threading.Lock()

    def _chat(self, messages: list[Message], schema: dict[str, Any]) -> tuple[str, bool, int, int]:
        """Return (content, truncated, prompt tokens, output tokens) for one request."""
        raise NotImplementedError

    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any:
        start = time.monotonic()
        content, truncated, prompt, output = self._chat(messages, schema)
        outcome, detail = "ok", ""
        try:
            if truncated:
                outcome = "truncated"
                raise LLMError(f"{self.provider} truncated its JSON response")
            result = load_json(content)
            validate_json(result, schema)
            return result
        except SchemaError as exc:
            outcome, detail = malformed_kind(exc), str(exc)
            raise
        finally:
            self._record(outcome, detail, time.monotonic() - start, prompt, output)

    def _record(self, outcome: str, detail: str, elapsed: float, prompt: int, output: int) -> None:
        usage = self.usage
        status = "ok"
        with self._lock:
            usage["calls"] += 1
            usage["prompt"] += prompt
            usage["output"] += output
            if outcome != "ok":
                usage[outcome] += 1
                status = f"MALFORMED {MALFORMED[outcome]}" + (f": {detail[:120]}" if detail else "")
            summary = malformed_summary(usage)
        logger.info(
            "  %s | %.1fs | prompt %s tok | out %s | %s | %s so far",
            self.model, elapsed, f"{prompt:,}", f"{output:,}", status, summary,
        )

    def report(self) -> None:
        """Log run totals, including malformed replies, to the progress stream."""
        usage = self.usage
        if usage["calls"]:
            logger.info(
                "%s: %d calls | %s | prompt %s tok | out %s",
                self.provider, usage["calls"], malformed_summary(usage),
                f"{usage['prompt']:,}", f"{usage['output']:,}",
            )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise LLMError("LLM response exceeded 4 MiB")
            body = load_json(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise LLMError("LLM returned an invalid response envelope")
            return body
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise LLMError(f"LLM returned HTTP {status}; check the endpoint and model") from None
        except (URLError, OSError) as exc:
            raise LLMError(
                "Cannot reach the LLM; start Ollama or check the configured endpoint"
            ) from exc
        except UnicodeError as exc:
            raise LLMError("LLM response was not UTF-8") from exc


class OllamaClient(HTTPJSONModel):
    provider = "Ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
        context_size: int = 65536,
    ):
        super().__init__(model, base_url=base_url, timeout=timeout)
        if type(context_size) is not int or context_size < 4096:
            raise ValueError("Ollama context size must be at least 4096")
        self.context_size = context_size
        self.output_tokens = min(8192, context_size // 4)

    def _chat(self, messages: list[Message], schema: dict[str, Any]) -> tuple[str, bool, int, int]:
        if len(json.dumps(messages)) // 3 + self.output_tokens > self.context_size:
            raise LLMError(
                "Prompt may exceed the context window; reduce --pool-limit or raise --context-size"
            )
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.context_size,
                    "num_predict": self.output_tokens,
                },
            },
        )
        tokens = body.get("prompt_eval_count", 0), body.get("eval_count", 0)
        if body.get("done_reason") == "length":
            return "", True, *tokens
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError("Ollama returned an invalid chat envelope")
        return message["content"], False, *tokens


class LlamaCppClient(HTTPJSONModel):
    """Local llama.cpp `llama-server`. It loads GGUF architectures Ollama may not yet support."""

    provider = "llama.cpp"

    def __init__(
        self,
        model: str = LFM3_MODEL,
        *,
        base_url: str = "http://localhost:8080/v1",
        timeout: float = 300,
        context_size: int = 65536,
        api_key: str | None = None,
        thinking: bool = True,
    ):
        super().__init__(model, base_url=base_url, timeout=timeout, api_key=api_key)
        if type(context_size) is not int or context_size < 4096:
            raise ValueError("llama.cpp context size must be at least 4096")
        self.context_size = context_size
        self.thinking = thinking
        # The least output room a prompt must leave; a reply may use all of the remaining context.
        self.output_tokens = min(8192, context_size // 4)

    def _chat(self, messages: list[Message], schema: dict[str, Any]) -> tuple[str, bool, int, int]:
        prompt = len(json.dumps(messages)) // 3
        if prompt + self.output_tokens > self.context_size:
            raise LLMError(
                "Prompt may exceed the context window; reduce --pool-limit or raise --context-size"
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            # Reasoning models think before the constrained JSON, so leave them the rest of the context.
            "max_tokens": self.context_size - prompt,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "mtg_response", "strict": True, "schema": schema},
            },
        }
        if not self.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return _chat_completion(self._post("/chat/completions", payload))


class OpenAICompatibleClient(HTTPJSONModel):
    """Opt-in adapter; API keys are supplied by the caller/environment, never stored."""

    provider = "LLM"
    concurrency = 8

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 300,
    ):
        if not api_key:
            raise ValueError("Set MTG_LLM_API_KEY for the API provider")
        super().__init__(model, base_url=base_url, timeout=timeout, api_key=api_key)

    def _chat(self, messages: list[Message], schema: dict[str, Any]) -> tuple[str, bool, int, int]:
        body = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "mtg_response", "strict": True, "schema": schema},
                },
            },
        )
        return _chat_completion(body)
