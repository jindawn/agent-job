from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..guardrails import defuse, mask_pii

MAX_OUTPUT_CHARS = 4000
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any], "ToolContext"], str]
    requires_approval: bool = False  # human-in-the-loop for side-effecting tools

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolContext:
    retriever: Any = None
    db_path: Any = None
    role: str = "support"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str
    latency_ms: float = 0.0


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    if "__raw__" in args:
        return "参数不是合法的 JSON 对象"
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"缺少必填参数: {req}"
    for k, v in args.items():
        if k not in props:
            return f"未知参数: {k}"
        spec = props[k]
        t = _TYPES.get(spec.get("type", ""))
        if t and (not isinstance(v, t) or (spec["type"] in ("integer", "number") and isinstance(v, bool))):
            return f"参数 {k} 类型应为 {spec['type']}"
        if "enum" in spec and v not in spec["enum"]:
            return f"参数 {k} 必须是 {spec['enum']} 之一"
        if isinstance(v, str) and len(v) > spec.get("maxLength", 4000):
            return f"参数 {k} 过长"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "minimum" in spec and v < spec["minimum"]:
                return f"参数 {k} 不能小于 {spec['minimum']}"
            if "maximum" in spec and v > spec["maximum"]:
                return f"参数 {k} 不能大于 {spec['maximum']}"
    return None


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        return [t.schema() for n, t in self._tools.items() if allowed is None or n in allowed]

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, f"ERROR: 未知工具 {name}，可用工具: {', '.join(self._tools)}")
        err = validate_args(tool.parameters, args)
        if err:
            return ToolResult(name, False, f"ERROR: 参数校验失败: {err}")
        try:
            out = tool.fn(args, ctx)
            ok = not out.startswith("ERROR")
        except Exception as e:  # tool failures are data for the model, not crashes
            out, ok = f"ERROR: 工具执行异常: {type(e).__name__}: {e}", False
        out, _ = defuse(out)          # retrieved/DB text must not smuggle instructions
        out = mask_pii(out)           # data minimisation: PII never reaches the LLM
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n...[输出已截断]"
        return ToolResult(name, ok, out, (time.perf_counter() - t0) * 1000)


def args_key(name: str, args: dict[str, Any]) -> str:
    return name + json.dumps(args, sort_keys=True, ensure_ascii=False)
