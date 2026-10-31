from __future__ import annotations

import ast
import operator as op
import re
import sqlite3
from typing import Any

from ..guardrails import mask_pii, validate_sql
from .registry import Tool, ToolContext, ToolRegistry

# ------------------------------------------------------------------ kb_search

def _kb_search(args: dict[str, Any], ctx: ToolContext) -> str:
    hits = ctx.retriever.search(args["query"], top_k=int(args.get("top_k", 4)))
    if not hits:
        return "未检索到相关知识库内容。"
    return "\n".join(f"[{i}] (来源: {h.chunk.source}) {h.chunk.text.replace(chr(10), ' ')}" for i, h in enumerate(hits, 1))


# ------------------------------------------------------------------ calculator
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
            raise ValueError("指数过大")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("不支持的表达式")


def safe_eval(expr: str) -> float:
    if len(expr) > 200:
        raise ValueError("表达式过长")
    return _eval(ast.parse(expr.strip(), mode="eval"))


def _calculator(args: dict[str, Any], ctx: ToolContext) -> str:
    try:
        v = safe_eval(args["expression"])
    except ZeroDivisionError:
        return "ERROR: 除数不能为 0"
    except (ValueError, SyntaxError) as e:
        return f"ERROR: 无法计算: {e}"
    if isinstance(v, float):
        s = f"{round(v, 10):.10f}".rstrip("0").rstrip(".")
    else:
        s = str(v)
    return f"{args['expression']} = {s}"


# ------------------------------------------------------------------ sql_query
_ALLOWED_TABLES = {"orders"}
_SQLITE_OK, _SQLITE_DENY = 0, 1
_A_SELECT, _A_READ, _A_FUNCTION, _A_RECURSIVE = 21, 20, 31, 33


def _authorizer(action, arg1, arg2, dbname, source):
    if action == _A_READ:
        return _SQLITE_OK if arg1 in _ALLOWED_TABLES else _SQLITE_DENY
    if action in (_A_SELECT, _A_FUNCTION, _A_RECURSIVE):
        return _SQLITE_OK
    return _SQLITE_DENY


def run_readonly_sql(db_path, sql: str, max_rows: int = 50) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(_authorizer)  # layer 2: engine-level allow-list (tables + actions)
        budget = {"n": 0}

        def stop() -> int:  # layer 3: abort runaway queries
            budget["n"] += 1
            return 1 if budget["n"] > 2000 else 0

        conn.set_progress_handler(stop, 1000)
        inner = sql.strip().rstrip(";")
        cur = conn.execute(f"SELECT * FROM ({inner}) LIMIT {int(max_rows)}")
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()
    finally:
        conn.close()


def _sql_query(args: dict[str, Any], ctx: ToolContext) -> str:
    sql = args["sql"]
    err = validate_sql(sql)  # layer 1: syntax-level policy
    if err:
        return f"ERROR: SQL 被安全策略拒绝: {err}"
    try:
        cols, rows = run_readonly_sql(ctx.db_path, sql)
    except sqlite3.DatabaseError as e:
        msg = "访问被安全策略拒绝(仅允许读取 orders 表)" if "not authorized" in str(e) else str(e)
        return f"ERROR: SQL 被安全策略拒绝或执行失败: {msg}"
    if not rows:
        return "查询结果为空。"
    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v) for v in r) for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ create_ticket

def _create_ticket(args: dict[str, Any], ctx: ToolContext) -> str:
    with sqlite3.connect(ctx.db_path) as conn:
        cur = conn.execute(
            "INSERT INTO tickets(title, description, priority) VALUES (?,?,?)",
            (mask_pii(args["title"]), mask_pii(args.get("description", "")), args.get("priority", "P2")),
        )
        tid = cur.lastrowid
    return f"工单已创建: T-{tid:04d} (优先级 {args.get('priority', 'P2')})"


def build_registry() -> ToolRegistry:
    return ToolRegistry([
        Tool(
            "kb_search",
            "检索公司知识库（退换货、物流、会员积分、IT工单SLA、信息安全等政策）。回答政策类问题前必须先调用。",
            {"type": "object", "required": ["query"], "properties": {
                "query": {"type": "string", "maxLength": 200, "description": "检索关键词或完整问题"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 8}}},
            _kb_search,
        ),
        Tool(
            "sql_query",
            "对订单库执行只读 SELECT。表 orders(order_id, customer, phone, status, amount, created_at, logistics_no)。"
            "status 取值: 待发货/已发货/已签收/已退款。",
            {"type": "object", "required": ["sql"], "properties": {"sql": {"type": "string", "maxLength": 500}}},
            _sql_query,
        ),
        Tool(
            "calculator",
            "精确计算数学表达式，支持 + - * / ** % // 括号及 round/min/max/abs。涉及金额、折扣的计算必须使用本工具。",
            {"type": "object", "required": ["expression"], "properties": {"expression": {"type": "string", "maxLength": 200}}},
            _calculator,
        ),
        Tool(
            "create_ticket",
            "创建内部 IT/客服工单（有副作用，需用户确认后才会执行）。",
            {"type": "object", "required": ["title"], "properties": {
                "title": {"type": "string", "maxLength": 100},
                "description": {"type": "string", "maxLength": 1000},
                "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]}}},
            _create_ticket,
            requires_approval=True,
        ),
    ])
