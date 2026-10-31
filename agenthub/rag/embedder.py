"""Embedders.

HashingEmbedder: offline, deterministic signed feature hashing over unigrams+bigrams.
It is a *lexical-overlap* vector (not semantic) - good enough to run the whole stack
offline and in CI. Swap in OpenAIEmbedder (or bge-m3 / text-embedding-v3 behind an
OpenAI-compatible endpoint) for real semantic recall: the retriever only depends on
`embed(list[str]) -> list[SparseVec]`.
"""
from __future__ import annotations

import json
import math
import urllib.request
import zlib
from typing import Protocol

from ..textutil import tokenize

SparseVec = dict[int, float]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVec]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[SparseVec]:
        out: list[SparseVec] = []
        for t in texts:
            v: SparseVec = {}
            toks = tokenize(t)
            # also add CJK unigrams so short queries still overlap
            toks += [c for c in t.lower() if "\u4e00" <= c <= "\u9fff"]
            for tok in toks:
                h = zlib.crc32(tok.encode())
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 else -1.0
                v[idx] = v.get(idx, 0.0) + sign
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            out.append({k: x / norm for k, x in v.items()})
        return out


class OpenAIEmbedder:
    """Dense embeddings from any OpenAI-compatible /embeddings endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def embed(self, texts: list[str]) -> list[SparseVec]:
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)["data"]
        vecs = []
        for item in sorted(data, key=lambda d: d["index"]):
            e = item["embedding"]
            norm = math.sqrt(sum(x * x for x in e)) or 1.0
            vecs.append({i: x / norm for i, x in enumerate(e)})
        return vecs


def cosine(a: SparseVec, b: SparseVec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(i, 0.0) for i, x in a.items())
