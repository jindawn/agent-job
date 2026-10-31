"""Lightweight tracing: one Trace per user request, spans for each LLM / tool call.
Maps 1:1 onto OpenTelemetry spans if you want to export later."""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    kind: str  # "llm" | "tool" | "guard"
    start: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def set(self, **kw: Any) -> None:
        self.attrs.update(kw)


class Trace:
    def __init__(self, price_in_per_1k: float = 0.0, price_out_per_1k: float = 0.0) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.spans: list[Span] = []
        self.t0 = time.perf_counter()
        self.price_in, self.price_out = price_in_per_1k, price_out_per_1k

    @contextmanager
    def span(self, name: str, kind: str, **attrs: Any) -> Iterator[Span]:
        sp = Span(name, kind, attrs=dict(attrs))
        try:
            yield sp
        except Exception as e:
            sp.attrs["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - sp.start) * 1000
            self.spans.append(sp)

    def summary(self) -> dict[str, Any]:
        llm = [s for s in self.spans if s.kind == "llm"]
        tools = [s for s in self.spans if s.kind == "tool"]
        p = sum(s.attrs.get("prompt_tokens", 0) for s in llm)
        c = sum(s.attrs.get("completion_tokens", 0) for s in llm)
        return {
            "trace_id": self.id,
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tools": [s.name for s in tools],
            "prompt_tokens": p,
            "completion_tokens": c,
            "cost": round(p / 1000 * self.price_in + c / 1000 * self.price_out, 6),
            "latency_ms": round((time.perf_counter() - self.t0) * 1000, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "spans": [
                {"name": s.name, "kind": s.kind, "duration_ms": round(s.duration_ms, 2), **s.attrs}
                for s in self.spans
            ],
        }
