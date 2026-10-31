import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agenthub.bootstrap import build_agent
from agenthub.config import Settings
from agenthub.evals.runner import load_cases, run_all, run_case
from agenthub.mcp_server import MCPServer
from agenthub.service import AgentService
from tests.helpers import KB, make_agent, ScriptedLLM, say

ROOT = Path(__file__).resolve().parent.parent


def server():
    agent = make_agent(ScriptedLLM([say("x")]))
    return MCPServer(agent.registry, agent.ctx)


class MCPTests(unittest.TestCase):
    def test_handshake_list_and_call(self):
        s = server()
        init = s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-03-26")
        self.assertIsNone(s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        tools = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual({t["name"] for t in tools}, {"kb_search", "sql_query", "calculator"})  # write tool NOT exposed
        self.assertTrue(all("inputSchema" in t for t in tools))
        ok = s.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "calculator", "arguments": {"expression": "6*7"}}})
        self.assertEqual(ok["result"]["content"][0]["text"], "6*7 = 42"); self.assertFalse(ok["result"]["isError"])
        bad = s.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "sql_query", "arguments": {"sql": "DROP TABLE orders"}}})
        self.assertTrue(bad["result"]["isError"])
        self.assertEqual(s.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "create_ticket", "arguments": {"title": "x"}}})["error"]["code"], -32602)
        self.assertEqual(s.handle({"jsonrpc": "2.0", "id": 6, "method": "nope"})["error"]["code"], -32601)

    def test_stdio_loop_handles_garbage_lines(self):
        out = io.StringIO()
        inp = io.StringIO('not json\n\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n[1,2]\n')
        server().serve(inp, out)
        lines = [json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual(lines[0]["error"]["code"], -32700)
        self.assertEqual(lines[1], {"jsonrpc": "2.0", "id": 1, "result": {}})
        self.assertEqual(lines[2]["error"]["code"], -32700)

    def test_real_subprocess_over_stdio(self):
        env = {**os.environ, "DB_PATH": str(Path(tempfile.mkdtemp()) / "m.db"), "PYTHONPATH": str(ROOT)}
        reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "kb_search", "arguments": {"query": "满多少包邮"}}}]
        p = subprocess.run([sys.executable, "-m", "agenthub.mcp_server"], input="\n".join(map(json.dumps, reqs)) + "\n",
                           capture_output=True, text=True, timeout=30, cwd=ROOT, env=env)
        resp = [json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual([r["id"] for r in resp], [1, 2])          # exactly one reply per request, none for the notification
        self.assertIn("99元包邮", resp[1]["result"]["content"][0]["text"])


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = build_agent(Settings(db_path=Path(tempfile.mkdtemp()) / "e.db"))

    def test_full_suite_passes_on_mock(self):
        results, summary = run_all(self.agent)
        self.assertEqual(summary["total"], len(load_cases()))
        self.assertGreaterEqual(summary["pass_rate"], 0.95, [(r.id, r.failures) for r in results if not r.passed])
        self.assertEqual(summary["retrieval_hit_at_k"], 1.0)

    def test_harness_can_actually_fail(self):
        bad = [{"id": "x1", "category": "t", "question": "满多少钱包邮", "expect_tools": ["sql_query"], "keywords": ["不存在的词"], "expect_source": "不存在的文档.md"},
               {"id": "x2", "category": "t", "question": "订单EC20250002的信息", "forbid_keywords": ["138****5678"]}]
        r1, r2 = (run_case(self.agent, c) for c in bad)
        self.assertFalse(r1.passed); self.assertEqual(len(r1.failures), 3)
        self.assertFalse(r2.passed)


class ServiceTests(unittest.TestCase):
    def test_service_roundtrip_with_approval(self):
        svc = AgentService(build_agent(Settings(db_path=Path(tempfile.mkdtemp()) / "s.db")))
        r = svc.chat("紧急！支付系统宕机了，帮我创建一个工单")
        self.assertEqual(r["status"], "awaiting_approval")
        r = svc.approve(r["session_id"], {r["pending_approvals"][0]["id"]: True})
        self.assertIn("T-", r["answer"])
        self.assertEqual(svc.health()["status"], "ok")
        json.dumps(r)  # must be JSON-serialisable for the HTTP layer

    def test_fastapi_layer_if_installed(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")
        from agenthub.server import create_app
        c = TestClient(create_app(AgentService(build_agent(Settings(db_path=Path(tempfile.mkdtemp()) / "f.db")))))
        self.assertEqual(c.get("/health").status_code, 200)
        r = c.post("/v1/chat", json={"message": "满多少钱包邮"}).json()
        self.assertEqual(r["status"], "done")
        self.assertEqual(c.post("/v1/approve", json={"session_id": "zzz", "decisions": {}}).status_code, 409)
        self.assertEqual(c.post("/v1/chat", json={"message": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
