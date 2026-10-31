from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    provider: str = "mock"  # "mock" | "openai" (any OpenAI-compatible endpoint)
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    embed_model: str = ""  # empty -> offline hashing embedder
    max_steps: int = 6
    context_token_budget: int = 6000
    kb_dir: Path = field(default_factory=lambda: ROOT / "data" / "kb")
    db_path: Path = field(default_factory=lambda: ROOT / "data" / "demo.db")
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("LLM_API_KEY", "")
        provider = os.getenv("LLM_PROVIDER") or ("openai" if key else "mock")
        return cls(
            provider=provider,
            base_url=os.getenv("LLM_BASE_URL", cls.base_url),
            api_key=key,
            model=os.getenv("LLM_MODEL", cls.model),
            embed_model=os.getenv("EMBED_MODEL", ""),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
            context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "6000")),
            kb_dir=Path(os.getenv("KB_DIR", str(ROOT / "data" / "kb"))),
            db_path=Path(os.getenv("DB_PATH", str(ROOT / "data" / "demo.db"))),
            price_in_per_1k=float(os.getenv("PRICE_IN_PER_1K", "0")),
            price_out_per_1k=float(os.getenv("PRICE_OUT_PER_1K", "0")),
        )
