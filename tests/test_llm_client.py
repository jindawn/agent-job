import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from agenthub.llm.base import LLMError
from agenthub.llm.openai_compat import OpenAICompatClient


def serve(responses):
    """responses: list of (status, body_dict); last one repeats. Returns (server, base_url, seen_requests)."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                         "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            status, body = responses[min(len(seen) - 1, len(responses) - 1)]
            raw = json.dumps(body).encode()
            self.send_response(status); self.send_header("Content-Length", str(len(raw))); self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", seen


def stop(srv):
    srv.shutdown()
    srv.server_close()


OK = {"model": "m", "usage": {"prompt_tokens": 11, "completion_tokens": 7}, "choices": [{"message": {
    "content": "hi", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "kb_search", "arguments": "{\"query\": \"退货\"}"}}]}}]}


class ClientTests(unittest.TestCase):
    def test_parses_tool_calls_usage_and_sends_auth(self):
        srv, url, seen = serve([(200, OK)])
        self.addCleanup(stop, srv)
        r = OpenAICompatClient(url, "sk-test", "m").chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
        self.assertEqual((r.tool_calls[0].name, r.tool_calls[0].arguments), ("kb_search", {"query": "退货"}))
        self.assertEqual((r.usage.prompt_tokens, r.usage.completion_tokens), (11, 7))
        self.assertEqual(seen[0]["auth"], "Bearer sk-test"); self.assertEqual(seen[0]["path"], "/v1/chat/completions")
        self.assertIn("tools", seen[0]["body"])

    def test_tool_choice_none_omits_tools(self):
        srv, url, seen = serve([(200, OK)])
        self.addCleanup(stop, srv)
        OpenAICompatClient(url, "k", "m").chat([], tools=[{"type": "function"}], tool_choice="none")
        self.assertNotIn("tools", seen[0]["body"])

    def test_retries_on_503_and_429_then_succeeds(self):
        srv, url, seen = serve([(503, {"e": 1}), (429, {"e": 1}), (200, OK)])
        self.addCleanup(stop, srv)
        r = OpenAICompatClient(url, "k", "m", backoff=0.01).chat([])
        self.assertEqual(len(seen), 3); self.assertEqual(r.content, "hi")

    def test_does_not_retry_on_400_and_gives_up_after_max_retries(self):
        srv, url, seen = serve([(400, {"error": "bad"})])
        self.addCleanup(stop, srv)
        with self.assertRaises(LLMError):
            OpenAICompatClient(url, "k", "m", backoff=0.01).chat([])
        self.assertEqual(len(seen), 1)
        srv2, url2, seen2 = serve([(500, {"e": 1})])
        self.addCleanup(stop, srv2)
        with self.assertRaises(LLMError):
            OpenAICompatClient(url2, "k", "m", max_retries=2, backoff=0.01).chat([])
        self.assertEqual(len(seen2), 3)

    def test_malformed_tool_arguments_surface_as_raw(self):
        bad = json.loads(json.dumps(OK)); bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{oops"
        srv, url, _ = serve([(200, bad)])
        self.addCleanup(stop, srv)
        r = OpenAICompatClient(url, "k", "m").chat([])
        self.assertEqual(r.tool_calls[0].arguments, {"__raw__": "{oops"})

    def test_connection_error_is_wrapped(self):
        with self.assertRaises(LLMError):
            OpenAICompatClient("http://127.0.0.1:1/v1", "k", "m", max_retries=0, timeout=1).chat([])


if __name__ == "__main__":
    unittest.main()
