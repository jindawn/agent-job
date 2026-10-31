"""Deterministic rule-based stand-in for an LLM.

It speaks the same function-calling protocol as a real model (emits tool_calls, then a final
answer composed from tool messages) so the *whole pipeline* - agent loop, tools, guardrails,
HITL, tracing, evals, API - runs offline and in CI. It is NOT a language model: eval numbers
produced with it validate the plumbing and retrieval, not answer quality.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..textutil import estimate_tokens
from .base import LLMResponse, ToolCall, Usage

_ORDER_ID = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2}\d{8})(?![A-Za-z0-9])")
_DANGER = re.compile(r"(drop|delete|update|insert|alter)\b.*", re.I | re.S)
_POLICY_WORDS = re.compile(r"退|换|政策|能不能|可以|规则|多久|物流|快递")
_TICKET = re.compile(r"创建.{0,4}工单|提交.{0,4}工单|开.{0,3}工单|建.{0,3}工单|报修")
_STATS = re.compile(r"统计|各状态|总金额|订单数量")


class MockLLM:
    model = "mock-llm"

    def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        u = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[u]["content"]
        turn = messages[u + 1:]
        called = {tc["function"]["name"] for m in turn for tc in (m.get("tool_calls") or [])}
        available = {t["function"]["name"] for t in tools} if tools else set()
        prompt_tokens = sum(estimate_tokens(m.get("content") or "") for m in messages)

        if tool_choice != "none":
            for name, args in self._plan(question):
                if name in available and name not in called:
                    call = ToolCall(id=f"mock_{len(called)}_{name}", name=name, arguments=args)
                    return LLMResponse("", [call], Usage(prompt_tokens, 20), self.model)

        answer = self._compose([m for m in turn if m["role"] == "tool"])
        return LLMResponse(answer, [], Usage(prompt_tokens, estimate_tokens(answer)), self.model)

    # ---- planning -------------------------------------------------------
    def _plan(self, q: str) -> list[tuple[str, dict[str, Any]]]:
        plan: list[tuple[str, dict[str, Any]]] = []
        if m := _DANGER.search(q):
            return [("sql_query", {"sql": m.group(0).strip()})]
        if oid := _ORDER_ID.search(q):
            plan.append(("sql_query", {"sql": (
                "SELECT order_id, customer, phone, status, amount, created_at, logistics_no "
                f"FROM orders WHERE order_id = '{oid.group(1)}'")}))
            if _POLICY_WORDS.search(q):
                plan.append(("kb_search", {"query": _ORDER_ID.sub("", q)}))
            return plan
        if _STATS.search(q) and "订单" in q:
            return [("sql_query", {"sql": "SELECT status, COUNT(*) AS cnt, SUM(amount) AS total FROM orders GROUP BY status"})]
        if _TICKET.search(q):
            prio = "P0" if re.search(r"宕机|不可用|全部用户", q) else ("P1" if "紧急" in q else "P2")
            return [("create_ticket", {"title": q[:40], "description": q, "priority": prio})]
        exprs = [e.strip() for e in re.findall(r"[\d\.\+\-\*/×÷\(\)\s%]{3,}", q) if re.search(r"\d", e) and re.search(r"[\+\-\*/×÷]", e)]
        if re.search(r"计算|算一下|等于多少", q) and exprs:
            expr = max(exprs, key=len).replace("×", "*").replace("÷", "/")
            return [("calculator", {"expression": expr})]
        return [("kb_search", {"query": q})]

    # ---- answer composition --------------------------------------------
    def _compose(self, tool_msgs: list[dict[str, Any]]) -> str:
        if not tool_msgs:
            return "我没有找到可用的信息，请换一种问法。"
        parts: list[str] = []
        for m in tool_msgs:
            c = m.get("content", "")
            name = m.get("name", "")
            if c.startswith("ERROR") or c.startswith("REJECTED"):
                parts.append(f"操作未执行：{c.split(':', 1)[-1].strip()}")
            elif name == "kb_search":
                lines = [ln for ln in c.splitlines() if ln.startswith("[")][:2]
                bullets = []
                for ln in lines:
                    mm = re.match(r"\[\d+\] \(来源: (.*?)\) (.*)", ln)
                    if mm:
                        bullets.append(f"- {mm.group(2)} [来源: {mm.group(1)}]")
                parts.append("根据知识库：\n" + "\n".join(bullets) if bullets else c)
            elif name == "sql_query":
                parts.append("查询结果：\n" + c)
            elif name == "calculator":
                parts.append("计算结果：" + c)
            else:
                parts.append(c)
        return "\n\n".join(parts)
