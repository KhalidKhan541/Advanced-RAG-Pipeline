"""Hybrid retrieval: reciprocal rank fusion of sparse (BM25) and dense results.

This module contrasts two retrieval strategies used by the benchmark:

- :class:`HybridRetriever`: fuses BM25 and dense rankings with reciprocal
  rank fusion (RRF), the "advanced" approach.
- :class:`NaiveRetriever`: dense-only top-k retrieval with no fusion or
  reranking, the "naive" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .embeddings import BM25Index, Chunk, DenseIndex, SentenceEncoder

RRF_K = 60
"""Constant used in the RRF fusion formula ``1 / (k + rank)``."""


@dataclass
class RetrievedChunk:
    """A retrieval hit carrying its fused and component scores.

    Attributes:
        chunk: The retrieved chunk itself.
        score: Final fused score (hybrid) or similarity (naive).
        sparse_score: Raw BM25 score (0.0 when unavailable).
        dense_score: Cosine similarity from the dense index.
        rank: Final 0-based rank in the merged result list.
    """

    chunk: Chunk
    score: float
    sparse_score: float = 0.0
    dense_score: float = 0.0
    rank: int = 0


class HybridRetriever:
    """Retriever that fuses BM25 and dense rankings via reciprocal rank fusion.

    Args:
        bm25: Sparse BM25 index, populated via :meth:`add_chunks`.
        dense: Dense index over embeddings, populated via :meth:`add_chunks`.
        encoder: Sentence encoder used to embed queries.
        alpha: Relative weight of sparse vs dense contributions when
            computing per-chunk hybrid scores in :meth:`hybrid_score`.
            Defaults to 0.5 (balanced).
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        encoder: SentenceEncoder,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.bm25 = bm25
        self.dense = dense
        self.encoder = encoder
        self.alpha = alpha
        self._chunks: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks in both the sparse and dense indices.

        Args:
            chunks: Chunks to add. Each is embedded with the encoder and
                inserted into both the BM25 and dense indexes.
        """
        docs = list(chunks)
        if not docs:
            return
        embeddings = self.encoder.encode_corpus([c.text for c in docs])
        self.bm25.add_documents(docs)
        self.dense.add_documents(docs, embeddings)
        for chunk in docs:
            self._chunks[chunk.id] = chunk

    def retrieve(self, query: str, top_k: int = 10, alpha: Optional[float] = None) -> List[RetrievedChunk]:
        """Retrieve chunks for a query using reciprocal rank fusion.

        BM25 and dense results are each ranked independently, then merged
        with the RRF score ``sum(1 / (RRF_K + rank))`` per chunk. Duplicates
        are merged, keeping the best component scores.

        Args:
            query: Raw query string.
            top_k: Number of results to return.
            alpha: Override for the sparse/dense weight in the per-chunk
                hybrid score; ``None`` uses the instance default.

        Returns:
            Ranked list of :class:`RetrievedChunk` objects, best first.
        """
        dense_hits = self._dense_search(query, top_k)
        sparse_hits = self.bm25.search(query, top_k)

        fused: Dict[str, RetrievedChunk] = {}
        for rank, hit in enumerate(dense_hits):
            chunk_id = str(hit["id"])
            fused[chunk_id] = RetrievedChunk(
                chunk=self._chunks[chunk_id],
                score=1.0 / (RRF_K + rank + 1),
                dense_score=float(hit["score"]),
            )
        for rank, hit in enumerate(sparse_hits):
            chunk_id = str(hit["id"])
            rrf = 1.0 / (RRF_K + rank + 1)
            if chunk_id in fused:
                fused[chunk_id].score += rrf
                fused[chunk_id].sparse_score = float(hit["score"])
            else:
                fused[chunk_id] = RetrievedChunk(
                    chunk=self._chunks[chunk_id],
                    score=rrf,
                    sparse_score=float(hit["score"]),
                )

        ranked = sorted(fused.values(), key=lambda r: r.score, reverse=True)
        for i, result in enumerate(ranked[:top_k]):
            result.rank = i
            w = self.alpha if alpha is None else alpha
            result.score = self.hybrid_score(query, result.chunk, w)
        return ranked[:top_k]

    def hybrid_score(self, query: str, chunk: Chunk, alpha: Optional[float] = None) -> float:
        """Compute a blended sparse+dense score for a single chunk.

        Sparse (BM25) and dense (cosine) scores are min-max normalized over
        the corpus and combined as ``alpha * sparse_norm + (1 - alpha) * dense_norm``.

        Args:
            query: Raw query string.
            chunk: The chunk to score.
            alpha: Sparse weight; ``None`` uses the instance default.

        Returns:
            Blended score in ``[0, 1]``.
        """
        w = self.alpha if alpha is None else alpha
        sparse_norm = self._normalize_sparse(self.bm25.search(query, len(self._chunks) or 1))
        dense_norm = self._normalize_dense(self._dense_search(query, len(self._chunks) or 1))
        sparse_val = sparse_norm.get(chunk.id, 0.0)
        dense_val = dense_norm.get(chunk.id, 0.0)
        return float(w * sparse_val + (1.0 - w) * dense_val)

    # -- internals ----------------------------------------------------------

    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, object]]:
        q_emb = self.encoder.embed_query(query)
        return self.dense.search(q_emb, top_k)

    @staticmethod
    def _normalize_sparse(hits: List[Dict[str, object]]) -> Dict[str, float]:
        scores = [float(h["score"]) for h in hits]
        if not scores:
            return {}
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi > lo else 1.0
        return {str(h["id"]): (float(h["score"]) - lo) / span for h in hits}

    @staticmethod
    def _normalize_dense(hits: List[Dict[str, object]]) -> Dict[str, float]:
        scores = [float(h["score"]) for h in hits]
        if not scores:
            return {}
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi > lo else 1.0
        return {str(h["id"]): (float(h["score"]) - lo) / span for h in hits}


class NaiveRetriever:
    """Baseline retriever: dense-only top-k search with no fusion or reranking.

    Mirrors the naive RAG setup — embedding similarity alone decides the
    results, ignoring lexical signal.

    Args:
        dense: Dense index populated via :meth:`add_chunks`.
        encoder: Sentence encoder used to embed documents and queries.
    """

    def __init__(self, dense: DenseIndex, encoder: SentenceEncoder) -> None:
        self.dense = dense
        self.encoder = encoder
        self._chunks: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed and index chunks in the dense index only.

        Args:
            chunks: Chunks to index.
        """
        docs = list(chunks)
        if not docs:
            return
        embeddings = self.encoder.encode_corpus([c.text for c in docs])
        self.dense.add_documents(docs, embeddings)
        for chunk in docs:
            self._chunks[chunk.id] = chunk

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """Return the top-k most similar chunks by cosine similarity alone.

        Args:
            query: Raw query string.
            top_k: Number of results to return.

        Returns:
            Ranked list of :class:`RetrievedChunk` objects (sparse scores 0.0).
        """
        hits = self.dense.search(self.encoder.embed_query(query), top_k)
        results = []
        for rank, hit in enumerate(hits):
            chunk = self._chunks[str(hit["id"])]
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(hit["score"]),
                    dense_score=float(hit["score"]),
                    rank=rank,
                )
            )
        return results


def build_retrievers(chunks: Sequence[Chunk], encoder: Optional[SentenceEncoder] = None) -> Dict[str, object]:
    """Convenience factory: build a hybrid and a naive retriever over chunks.

    Args:
        chunks: Chunks to index in both retrievers.
        encoder: Shared encoder; created with defaults when not provided.

    Returns:
        Mapping with keys ``"hybrid"`` (HybridRetriever) and ``"naive"``
        (NaiveRetriever), both already populated with ``chunks``.
    """
    enc = encoder or SentenceEncoder()
    hybrid = HybridRetriever(BM25Index(), DenseIndex(), enc)
    naive = NaiveRetriever(DenseIndex(), enc)
    hybrid.add_chunks(list(chunks))
    naive.add_chunks(list(chunks))
    return {"hybrid": hybrid, "naive": naive}
