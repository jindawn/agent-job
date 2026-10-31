"""Framework-neutral service layer (so HTTP handlers stay trivial and logic stays unit-testable)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterator

from .agent import Agent


class AgentService:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def chat(self, message: str, session_id: str | None = None, role: str = "support") -> dict[str, Any]:
        return asdict(self.agent.run(message, session_id, role))

    def approve(self, session_id: str, decisions: dict[str, bool]) -> dict[str, Any]:
        return asdict(self.agent.resume(session_id, decisions))

    def stream(self, message: str, session_id: str | None = None, role: str = "support") -> Iterator[dict[str, Any]]:
        return self.agent.stream(message, session_id, role)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": getattr(self.agent.llm, "model", "?"), "tools": self.agent.registry.names(),
                "kb_chunks": len(self.agent.ctx.retriever.chunks)}
