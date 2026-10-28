import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from mtg_agentic.llm import (
    LFM3_MODEL,
    LlamaCppClient,
    LLMError,
    OllamaClient,
    OpenAICompatibleClient,
    request_json,
)
from mtg_agentic.schema import SPEC_SCHEMA, SchemaError
from tests.pipeline_fixtures import make_spec


class LLMTests(unittest.TestCase):
    @patch("mtg_agentic.llm.urlopen")
    def test_ollama_requests_schema_without_api_key(self, http):
        http.return_value = io.BytesIO(
            json.dumps(
                {"done": True, "message": {"content": json.dumps(make_spec().to_dict())}}
            ).encode()
        )
        result = request_json(OllamaClient(), [{"role": "user", "content": "Myojin"}], SPEC_SCHEMA)
        self.assertEqual(result["commander"], "Jodah, Archmage Eternal")
        request = http.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"], SPEC_SCHEMA)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertIsNone(request.get_header("Authorization"))

    @patch("mtg_agentic.llm.urlopen")
    def test_openai_adapter_sends_schema_and_explicit_key(self, http):
        http.return_value = io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": json.dumps(make_spec().to_dict())}}]}
            ).encode()
        )
        client = OpenAICompatibleClient("test-model", api_key="test-key")
        request_json(client, [], SPEC_SCHEMA)
        request = http.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data)["response_format"]["type"], "json_schema")

    def test_api_key_is_required_and_remote_http_or_credentials_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleClient("test", api_key="")
        for url in ("http://remote.example", "https://key@remote.example", "file:///model"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaClient(base_url=url)

    @patch("mtg_agentic.llm.urlopen")
    def test_http_and_connection_errors_are_clear_and_do_not_echo_response_body(self, http):
        http.side_effect = HTTPError(
            "https://example.com", 401, "Error", {}, io.BytesIO(b"test-key")
        )
        with self.assertRaises(LLMError) as caught:
            OpenAICompatibleClient("test", api_key="test-key").complete_json([], SPEC_SCHEMA)
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-key", str(caught.exception))
        http.side_effect = URLError("Connection refused")
        with self.assertRaisesRegex(LLMError, "start Ollama"):
            OllamaClient().complete_json([], SPEC_SCHEMA)

    @patch("mtg_agentic.llm.urlopen")
    def test_malformed_schema_output_is_rejected(self, http):
        http.return_value = io.BytesIO(json.dumps({"message": {"content": "{}"}}).encode())
        with self.assertRaises(SchemaError):
            request_json(OllamaClient(), [], SPEC_SCHEMA)

    @patch("mtg_agentic.llm.urlopen")
    def test_invalid_and_truncated_envelopes_are_rejected(self, http):
        for envelope in ({}, {"message": []}, {"done_reason": "length"}, {"choices": []}):
            with self.subTest(envelope=envelope):
                http.return_value = io.BytesIO(json.dumps(envelope).encode())
                client = (
                    OpenAICompatibleClient("test", api_key="test")
                    if "choices" in envelope
                    else OllamaClient()
                )
                with self.assertRaises(LLMError):
                    client.complete_json([], SPEC_SCHEMA)

    @patch("mtg_agentic.llm.urlopen")
    def test_context_guard_fails_before_sending_truncated_prompt(self, http):
        with self.assertRaisesRegex(LLMError, "context"):
            OllamaClient(context_size=4096).complete_json(
                [{"role": "user", "content": "x" * 15000}], SPEC_SCHEMA
            )
        http.assert_not_called()

    @patch("mtg_agentic.llm.urlopen")
    def test_llama_cpp_defaults_to_local_lfm3_with_schema_and_no_key(self, http):
        http.return_value = io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(make_spec().to_dict())}}],
                    "usage": {"prompt_tokens": 900, "completion_tokens": 40},
                }
            ).encode()
        )
        client = LlamaCppClient()
        request_json(client, [{"role": "user", "content": "Myojin"}], SPEC_SCHEMA)
        request = http.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(body["model"], LFM3_MODEL)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SPEC_SCHEMA)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertNotIn("chat_template_kwargs", body)
        prompt = len(json.dumps([{"role": "user", "content": "Myojin"}])) // 3
        self.assertEqual(body["max_tokens"], 65536 - prompt)
        self.assertEqual((client.usage["prompt"], client.usage["output"]), (900, 40))
        http.return_value = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": json.dumps(make_spec().to_dict())}}]}).encode()
        )
        request_json(LlamaCppClient(thinking=False), [], SPEC_SCHEMA)
        body = json.loads(http.call_args.args[0].data)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    @patch("mtg_agentic.llm.urlopen")
    def test_malformed_replies_are_counted_and_logged_per_call_and_in_totals(self, http):
        replies = [
            {"message": {"content": "not json"}},
            {"message": {"content": "{}"}},
            {"done_reason": "length"},
            {"message": {"content": json.dumps(make_spec().to_dict())}, "eval_count": 7},
        ]
        http.side_effect = [io.BytesIO(json.dumps(r).encode()) for r in replies]
        client = OllamaClient()
        with self.assertLogs("mtg_agentic.llm", "INFO") as logs:
            for error in (SchemaError, SchemaError, LLMError):
                with self.assertRaises(error):
                    request_json(client, [], SPEC_SCHEMA)
            request_json(client, [], SPEC_SCHEMA)
            client.report()
        self.assertEqual(
            {k: client.usage[k] for k in ("calls", "json", "schema", "truncated", "output")},
            {"calls": 4, "json": 1, "schema": 1, "truncated": 1, "output": 7},
        )
        self.assertIn("MALFORMED invalid JSON", logs.output[0])
        self.assertIn("MALFORMED schema", logs.output[1])
        self.assertIn("MALFORMED truncated", logs.output[2])
        self.assertIn("| ok |", logs.output[3])
        self.assertIn(
            "4 calls | 3 malformed (1 invalid JSON, 1 schema, 1 truncated)", logs.output[4]
        )
