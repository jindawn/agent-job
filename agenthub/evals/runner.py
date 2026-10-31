"""Offline-capable evaluation harness.

Per case it checks: expected / forbidden tools, answer keywords, forbidden strings (PII / secrets),
and retrieval hit@k (does the expected source doc appear in the retriever's top-k for the raw question).
Aggregates pass-rate by category, latency p50/p95, steps, tokens, cost; writes a JSON report;
exits non-zero below --min-pass so it can gate CI.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent import Agent

DATASET = Path(__file__).with_name("dataset.jsonl")


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    retrieval_hit: bool | None = None
    latency_ms: float = 0.0
    tools: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    answer: str = ""


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_case(agent: Agent, case: dict[str, Any], top_k: int = 4) -> CaseResult:
    t0 = time.perf_counter()
    res = agent.run(case["question"], role=case.get("role", "support"))
    if res.status == "awaiting_approval":
        approve = case.get("approve", True)
        res = agent.resume(res.session_id, {c["id"]: approve for c in res.pending_approvals})
    latency = (time.perf_counter() - t0) * 1000

    tools = res.trace["tools"]
    fails: list[str] = []
    for t in case.get("expect_tools", []):
        if t not in tools:
            fails.append(f"missing tool {t}")
    for t in case.get("forbid_tools", []):
        if t in tools:
            fails.append(f"forbidden tool used: {t}")
    low = res.answer.lower()
    for k in case.get("keywords", []):
        if k.lower() not in low:
            fails.append(f"missing keyword {k!r}")
    for k in case.get("forbid_keywords", []):
        if k.lower() in low:
            fails.append(f"leaked {k!r}")

    hit = None
    if case.get("expect_source"):
        docs = [h.chunk.doc for h in agent.ctx.retriever.search(case["question"], top_k=top_k)]
        hit = case["expect_source"] in docs
        if not hit:
            fails.append(f"retrieval miss: {case['expect_source']} not in top-{top_k}")

    return CaseResult(case["id"], case["category"], not fails, fails, hit, latency, tools,
                      res.trace["llm_calls"], res.trace["prompt_tokens"] + res.trace["completion_tokens"], res.answer)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    cats: dict[str, list[CaseResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    ret = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    lat = [r.latency_ms for r in results]
    return {
        "total": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / max(1, len(results)), 4),
        "retrieval_hit_at_k": round(sum(ret) / len(ret), 4) if ret else None,
        "by_category": {c: f"{sum(r.passed for r in rs)}/{len(rs)}" for c, rs in cats.items()},
        "latency_ms_p50": round(_pct(lat, 0.5), 1),
        "latency_ms_p95": round(_pct(lat, 0.95), 1),
        "avg_llm_calls": round(statistics.mean(r.llm_calls for r in results), 2) if results else 0,
        "total_tokens": sum(r.tokens for r in results),
    }


def run_all(agent: Agent, cases: list[dict[str, Any]] | None = None) -> tuple[list[CaseResult], dict[str, Any]]:
    results = [run_case(agent, c) for c in (cases or load_cases())]
    return results, summarize(results)


def format_report(results: list[CaseResult], summary: dict[str, Any], model: str) -> str:
    lines = [f"Model: {model}", f"Pass: {summary['passed']}/{summary['total']} ({summary['pass_rate']:.0%})",
             f"Retrieval hit@k: {summary['retrieval_hit_at_k']}", f"By category: {summary['by_category']}",
             f"Latency p50/p95: {summary['latency_ms_p50']}/{summary['latency_ms_p95']} ms",
             f"Avg LLM calls: {summary['avg_llm_calls']}  Tokens: {summary['total_tokens']}", ""]
    for r in results:
        lines.append(f"[{'PASS' if r.passed else 'FAIL'}] {r.id:<9} tools={r.tools} {'; '.join(r.failures)}")
    return "\n".join(lines)


def save_report(results: list[CaseResult], summary: dict[str, Any], model: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "summary": summary,
                                "cases": [r.__dict__ for r in results]}, ensure_ascii=False, indent=2), encoding="utf-8")
