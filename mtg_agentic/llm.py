"""Small, injectable JSON LLM interface. Local Ollama is the default."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .schema import load_json, validate_json

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class JSONModel(Protocol):
    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any: ...


def request_json(model: JSONModel, messages: list[Message], schema: dict[str, Any]) -> Any:
    result = model.complete_json(messages, schema)
    # Validate even injected providers. Structured-output hints are not a trust boundary.
    validate_json(result, schema)
    return result


class HTTPJSONModel:
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

    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any:
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
        if body.get("done_reason") == "length":
            raise LLMError("Ollama truncated its JSON response")
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError("Ollama returned an invalid chat envelope")
        content = message["content"]
        return load_json(content)


class OpenAICompatibleClient(HTTPJSONModel):
    """Opt-in adapter; API keys are supplied by the caller/environment, never stored."""

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

    def complete_json(self, messages: list[Message], schema: dict[str, Any]) -> Any:
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
        try:
            choice = body["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") == "length" or message.get("refusal"):
                raise LLMError("LLM refused or truncated the response")
            return load_json(message["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM returned an invalid chat-completions envelope") from exc
