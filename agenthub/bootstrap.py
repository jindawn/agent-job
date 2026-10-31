from __future__ import annotations

from .agent import Agent
from .config import Settings
from .db import init_db
from .llm.mock import MockLLM
from .llm.openai_compat import OpenAICompatClient
from .rag.embedder import HashingEmbedder, OpenAIEmbedder
from .rag.retriever import HybridRetriever
from .tools.builtin import build_registry
from .tools.registry import ToolContext


def build_llm(s: Settings):
    if s.provider == "mock":
        return MockLLM()
    return OpenAICompatClient(s.base_url, s.api_key, s.model)


def build_agent(s: Settings | None = None) -> Agent:
    s = s or Settings.from_env()
    init_db(s.db_path)
    embedder = OpenAIEmbedder(s.base_url, s.api_key, s.embed_model) if s.embed_model and s.api_key else HashingEmbedder()
    retriever = HybridRetriever.from_dir(s.kb_dir, embedder)
    ctx = ToolContext(retriever=retriever, db_path=s.db_path)
    return Agent(build_llm(s), build_registry(), ctx, s.max_steps, s.context_token_budget,
                 s.price_in_per_1k, s.price_out_per_1k)
