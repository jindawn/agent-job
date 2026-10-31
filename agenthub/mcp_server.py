"""Minimal MCP (Model Context Protocol) server over stdio - newline-delimited JSON-RPC 2.0.

Exposes the read-only tools (kb_search, sql_query, calculator) so any MCP client (Claude Desktop,
Cursor, other agents) can reuse the same knowledge base and guarded SQL access.
Side-effecting tools (create_ticket) are NOT exposed: MCP clients have their own approval UX
and we don't want writes to bypass ours.

Claude Desktop config:
  {"mcpServers": {"agenthub": {"command": "python", "args": ["-m", "agenthub.mcp_server"]}}}
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from . import __version__
from .config import Settings
from .db import init_db
from .rag.retriever import HybridRetriever
from .tools.builtin import build_registry
from .tools.registry import ToolContext, ToolRegistry

PROTOCOL_VERSION = "2025-06-18"
READ_ONLY = ("kb_search", "sql_query", "calculator")


class MCPServer:
    def __init__(self, registry: ToolRegistry, ctx: ToolContext, expose: tuple[str, ...] = READ_ONLY) -> None:
        self.registry, self.ctx, self.expose = registry, ctx, expose

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:  # notification (e.g. notifications/initialized): never respond
            return None
        if method == "initialize":
            return self._ok(mid, {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "agenthub", "version": __version__},
            })
        if method == "ping":
            return self._ok(mid, {})
        if method == "tools/list":
            tools = []
            for n in self.expose:
                t = self.registry.get(n)
                if t:
                    tools.append({"name": t.name, "description": t.description, "inputSchema": t.parameters})
            return self._ok(mid, {"tools": tools})
        if method == "tools/call":
            name = params.get("name")
            if name not in self.expose:
                return self._err(mid, -32602, f"Unknown tool: {name}")
            res = self.registry.execute(name, params.get("arguments") or {}, self.ctx)
            return self._ok(mid, {"content": [{"type": "text", "text": res.content}], "isError": not res.ok})
        return self._err(mid, -32601, f"Method not found: {method}")

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if not isinstance(msg, dict):
                    raise ValueError("request must be an object")
            except ValueError:
                out = self._err(None, -32700, "Parse error")
            else:
                out = self.handle(msg)
            if out is not None:
                stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stdout.flush()

    @staticmethod
    def _ok(mid: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _err(mid: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main() -> None:
    s = Settings.from_env()
    init_db(s.db_path)
    ctx = ToolContext(retriever=HybridRetriever.from_dir(s.kb_dir), db_path=s.db_path, role="mcp")
    print("agenthub MCP server ready (stdio)", file=sys.stderr)
    MCPServer(build_registry(), ctx).serve()


if __name__ == "__main__":
    main()
