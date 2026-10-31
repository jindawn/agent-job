"""Token-budgeted conversation memory.

Trims whole *turns* (user msg + the assistant/tool messages that follow) oldest-first, so an
assistant tool_call is never separated from its tool results (providers reject that).
The system message and the latest turn are always kept.
"""
from __future__ import annotations

import json
from typing import Any

from .textutil import estimate_tokens


def message_tokens(m: dict[str, Any]) -> int:
    n = estimate_tokens(m.get("content") or "")
    if m.get("tool_calls"):
        n += estimate_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
    return n + 4


def fit(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    system = [m for m in messages[:1] if m["role"] == "system"]
    rest = messages[len(system):]
    turns: list[list[dict[str, Any]]] = []
    for m in rest:
        if m["role"] == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    used = sum(message_tokens(m) for m in system)
    kept: list[list[dict[str, Any]]] = []
    for t in reversed(turns):
        cost = sum(message_tokens(m) for m in t)
        if kept and used + cost > budget:
            break
        kept.append(t)
        used += cost
    return system + [m for t in reversed(kept) for m in t]
