import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from mtg_agentic.llm import LLMError, OllamaClient, OpenAICompatibleClient, request_json
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
