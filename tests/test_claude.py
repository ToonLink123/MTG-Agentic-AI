import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtg_agentic import claude
from mtg_agentic.__main__ import _claude_api_key
from mtg_agentic.llm import LLMError, request_json
from mtg_agentic.schema import DRAFT_SCHEMA, SPEC_SCHEMA, SchemaError
from tests.pipeline_fixtures import draft, make_spec


def text(value):
    return SimpleNamespace(type="text", text=value)


def reply(*content, stop_reason="end_turn", prompt=1000, cached=0):
    usage = SimpleNamespace(
        input_tokens=prompt - cached,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
        output_tokens=50,
    )
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage)


def fake_client(*responses, error=None, **options):
    client = claude.ClaudeClient(api_key="test-key", **options)
    stream = client._client = MagicMock()
    manager = stream.beta.messages.stream.return_value
    if error:
        manager.__enter__.side_effect = error
    else:
        manager.__enter__.return_value.get_final_message.side_effect = responses
    return client, stream.beta.messages.stream


def api_error(cls, status, body=None):
    response = MagicMock(status_code=status, headers={"request-id": "req_123"})
    return cls("error", response=response, body=body)


SPEC = make_spec().to_dict()


@unittest.skipIf(claude.anthropic is None, "anthropic SDK not installed")
class ClaudeTests(unittest.TestCase):
    def test_haiku_default_sends_system_prompt_strict_schema_and_no_fallback(self):
        client, stream = fake_client(reply(text(json.dumps(SPEC))))
        messages = [
            {"role": "system", "content": "Return a deck spec."},
            {"role": "user", "content": "Myojin"},
        ]
        self.assertEqual(request_json(client, messages, SPEC_SCHEMA), SPEC)
        kwargs = stream.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-haiku-5-5")
        self.assertEqual(kwargs["system"], "Return a deck spec.")
        self.assertEqual(kwargs["messages"], messages[1:])
        self.assertEqual(kwargs["output_config"]["effort"], "high")
        self.assertEqual(kwargs["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(kwargs["cache_control"], {"type": "ephemeral"})
        sent = kwargs["output_config"]["format"]["schema"]
        self.assertFalse(sent["additionalProperties"])
        self.assertNotIn("minimum", sent["properties"]["bracket"])
        self.assertNotIn("extra_body", kwargs)
        self.assertNotIn("betas", kwargs)
        self.assertNotIn("temperature", kwargs)

    def test_models_with_server_side_fallback_opt_in(self):
        client, stream = fake_client(reply(text(json.dumps(SPEC))), model="claude-opus-5-5")
        client.complete_json([{"role": "user", "content": "x"}], SPEC_SCHEMA)
        kwargs = stream.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"fallbacks": "default"})
        self.assertEqual(kwargs["betas"], ["server-side-fallback-2026-07-01"])

    def test_repeated_prefix_gets_an_explicit_cache_breakpoint(self):
        client, stream = fake_client(reply(text(json.dumps(SPEC))))
        pool, repair = {"role": "user", "content": "pool"}, {"role": "user", "content": "fix"}
        client.complete_json([pool, repair], SPEC_SCHEMA)
        sent = stream.call_args.kwargs["messages"]
        self.assertEqual(
            sent[0]["content"],
            [{"type": "text", "text": "pool", "cache_control": {"type": "ephemeral"}}],
        )
        self.assertEqual(sent[1], repair)
        self.assertEqual(pool["content"], "pool")

    def test_schema_violation_is_retried_once_append_only(self):
        long = {**SPEC, "playstyle": "x" * 300}
        client, stream = fake_client(reply(text(json.dumps(long))), reply(text(json.dumps(SPEC))))
        prompt = [{"role": "user", "content": "Myojin"}]
        self.assertEqual(client.complete_json(prompt, SPEC_SCHEMA), SPEC)
        retry = stream.call_args.kwargs["messages"]
        self.assertEqual(retry[0]["content"][0]["text"], "Myojin")
        self.assertIn("invalid string length", retry[1]["content"])
        self.assertEqual((client.usage["schema"], client.usage["json"]), (1, 0))

    def test_only_text_after_a_mid_stream_fallback_is_parsed(self):
        client, _ = fake_client(
            reply(text('{"par'), SimpleNamespace(type="fallback"), text(json.dumps(SPEC)))
        )
        self.assertEqual(client.complete_json([], SPEC_SCHEMA), SPEC)

    def test_large_repeated_bucket_enums_are_wire_only_and_checked_after_retry(self):
        import copy

        schema = copy.deepcopy(DRAFT_SCHEMA)
        names = [f"Spell {n}" for n in range(350)]
        for bucket in schema["properties"]["buckets"]["properties"].values():
            bucket["items"] = {"type": "string", "enum": names}
        invalid = draft()
        invalid["buckets"]["support"] = ["Invented Card"]
        client, stream = fake_client(
            reply(text(json.dumps(invalid))), reply(text(json.dumps(invalid))),
        )
        with self.assertRaises(SchemaError):
            client.complete_json([], schema)
        sent = stream.call_args.kwargs["output_config"]["format"]["schema"]
        self.assertNotIn("enum", sent["properties"]["buckets"]["properties"]["support"]["items"])
        self.assertEqual(schema["properties"]["buckets"]["properties"]["support"]["items"]["enum"], names)
        self.assertEqual(stream.call_count, 2)

    def test_refusal_and_truncation_are_errors(self):
        for stop_reason in ("refusal", "max_tokens"):
            with self.subTest(stop_reason=stop_reason):
                client, _ = fake_client(reply(text("{}"), stop_reason=stop_reason))
                with self.assertRaises(LLMError):
                    client.complete_json([], SPEC_SCHEMA)

    def test_progress_reports_each_call_and_run_totals_with_cache_share(self):
        out = io.StringIO()
        client, _ = fake_client(
            reply(text(json.dumps(SPEC)), prompt=52000, cached=50000),
            reply(text(json.dumps(SPEC)), prompt=120000),
            progress=out,
        )
        client.complete_json([], SPEC_SCHEMA)
        client.complete_json([], SPEC_SCHEMA)
        client.report()
        lines = out.getvalue().splitlines()
        self.assertIn("prompt 52,000 tok (50,000 cached, 0 written)", lines[0])
        self.assertIn("over 100K", lines[1])
        self.assertIn("2 calls", lines[2])
        self.assertIn("29% from cache", lines[2])
        self.assertIn("largest prompt 120,000 (over Haiku's 100K tier)", lines[2])
        self.assertIn("0 malformed", lines[2])

    def test_prompt_over_the_budget_is_compacted_before_sending(self):
        out = io.StringIO()
        client, stream = fake_client(
            reply(text(json.dumps(SPEC))), reply(text(json.dumps(SPEC))), progress=out
        )
        client.compact_at = 2000
        counts = [SimpleNamespace(input_tokens=6000), SimpleNamespace(input_tokens=1500)]
        client._client.messages.count_tokens.side_effect = counts
        pool = {"role": "user", "content": "p" * 9000}
        client.complete_json([pool, {"role": "user", "content": "repair 1"}], SPEC_SCHEMA)
        first = stream.call_args.kwargs["messages"]
        self.assertLess(len(first[0]["content"][0]["text"]), 9000)
        self.assertIn("characters elided", first[0]["content"][0]["text"])
        self.assertEqual(first[1]["content"], "repair 1")
        self.assertIn("compacted prompt ~6,000", out.getvalue())
        # The next repair re-applies the same cut, so the cached prefix is unchanged.
        client.complete_json([pool, {"role": "user", "content": "repair 2"}], SPEC_SCHEMA)
        second = stream.call_args.kwargs["messages"]
        self.assertEqual(second[0], first[0])
        self.assertEqual(client.usage["compactions"], 1)

    def test_small_prompts_and_non_haiku_models_are_never_counted(self):
        for options in ({}, {"model": "claude-opus-5-5"}):
            with self.subTest(**options):
                client, _ = fake_client(reply(text(json.dumps(SPEC))), **options)
                client.complete_json([{"role": "user", "content": "x" * 5000}], SPEC_SCHEMA)
                client._client.messages.count_tokens.assert_not_called()
        self.assertIsNone(claude.ClaudeClient("claude-opus-5-5", api_key="k").compact_at)

    def test_api_errors_are_clear_and_do_not_echo_the_key(self):
        sdk = claude.anthropic
        body = {"error": {"message": "model: not found"}}
        cases = [
            (api_error(sdk.AuthenticationError, 401), "CLAUDE_API"),
            (api_error(sdk.NotFoundError, 404, body), "404: model: not found"),
            (api_error(sdk.InternalServerError, 500), "500: request req_123"),
            (sdk.APIConnectionError(request=MagicMock()), "Cannot reach"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                client, _ = fake_client(error=error)
                with self.assertRaises(LLMError) as caught:
                    client.complete_json([], SPEC_SCHEMA)
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("test-key", str(caught.exception))

    def test_api_key_and_model_are_required(self):
        with self.assertRaisesRegex(ValueError, "CLAUDE_API"):
            claude.ClaudeClient(api_key="")
        with self.assertRaises(ValueError):
            claude.ClaudeClient(" ", api_key="test-key")


class ClaudeKeyTests(unittest.TestCase):
    def test_key_comes_from_environment_before_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("# local\nOTHER=1\nexport CLAUDE_API='file-key'", encoding="utf-8")
            with patch.dict(os.environ, {"CLAUDE_API": "env-key"}):
                self.assertEqual(_claude_api_key(env_file), "env-key")
            with patch.dict(os.environ, {"CLAUDE_API": ""}):
                self.assertEqual(_claude_api_key(env_file), "file-key")
                self.assertEqual(_claude_api_key(Path(tmp) / "missing.env"), "")
