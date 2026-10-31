"""Hybrid retriever: BM25 + vector search fused with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .chunker import Chunk, load_dir
from .embedder import Embedder, HashingEmbedder, SparseVec, cosine


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vec_rank: int | None = None


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, rrf_k: int = 60) -> None:
        self.chunks = chunks
        self.embedder = embedder or HashingEmbedder()
        self.rrf_k = rrf_k
        self.bm25 = BM25([c.heading + "\n" + c.text for c in chunks])
        self.vecs: list[SparseVec] = self.embedder.embed([c.heading + "\n" + c.text for c in chunks]) if chunks else []

    @classmethod
    def from_dir(cls, directory, embedder: Embedder | None = None) -> "HybridRetriever":
        return cls(load_dir(directory), embedder)

    def search(self, query: str, top_k: int = 4, pool: int = 20, doc: str | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        idx = [i for i, c in enumerate(self.chunks) if doc is None or c.doc == doc]
        bm = self.bm25.scores(query)
        qv = self.embedder.embed([query])[0]
        vs = {i: cosine(qv, self.vecs[i]) for i in idx}

        bm_rank = {i: r for r, i in enumerate(sorted(idx, key=lambda i: -bm[i])[:pool]) if bm[i] > 0}
        vec_rank = {i: r for r, i in enumerate(sorted(idx, key=lambda i: -vs[i])[:pool]) if vs[i] > 0}

        fused: dict[int, float] = {}
        for ranks in (bm_rank, vec_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [Hit(self.chunks[i], fused[i], bm_rank.get(i), vec_rank.get(i)) for i in top]
