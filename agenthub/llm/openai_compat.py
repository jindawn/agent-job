"""Zero-dependency client for any OpenAI-compatible /chat/completions endpoint
(OpenAI, DeepSeek, Qwen/DashScope compat mode, vLLM, Ollama, ...).

Features: function calling, retry with exponential backoff on 408/429/5xx and
network errors, tolerant parsing of malformed tool-call JSON.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .base import LLMError, LLMResponse, ToolCall, Usage

_RETRYABLE = {408, 429, 500, 502, 503, 504}


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 0.8,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.temperature = temperature

    def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools and tool_choice != "none":
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        data = self._post("/chat/completions", payload)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from e

        calls: list[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except (ValueError, TypeError):
                args = {"__raw__": str(raw)}  # surfaces as a validation error to the model
            calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""), arguments=args))

        u = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=calls,
            usage=Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)),
            model=data.get("model", self.model),
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in _RETRYABLE and attempt < self.max_retries:
                    last = LLMError(f"HTTP {e.code}: {detail}")
                else:
                    raise LLMError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                if attempt >= self.max_retries:
                    raise LLMError(f"network error: {e}") from e
            time.sleep(self.backoff * (2**attempt))
        raise LLMError(f"exhausted retries: {last}")
