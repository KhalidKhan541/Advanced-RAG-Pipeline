import logging
from typing import List, Optional

from .hybrid_retriever import RetrievedChunk


class CrossEncoderReranker:
    """Cross-encoder reranking of retrieved chunks."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self._model = None
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self.logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Cross-encoder unavailable ({e}); using lexical fallback")
            self._model = None

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: Optional[int] = None) -> List[RetrievedChunk]:
        self._load_model()
        if self._model is not None:
            scores = self._model.predict([(query, c.chunk.text) for c in chunks])
            for c, s in zip(chunks, scores):
                c.score = float(s)
        else:
            self._lexical_rerank(query, chunks)
        chunks.sort(key=lambda c: c.score, reverse=True)
        for rank, c in enumerate(chunks, start=1):
            c.rank = rank
        return chunks[:top_k] if top_k else chunks

    def score(self, query: str, text: str) -> float:
        self._load_model()
        if self._model is not None:
            return float(self._model.predict([(query, text)])[0])
        return self._lexical_score(query, text)

    def _lexical_rerank(self, query: str, chunks: List[RetrievedChunk]):
        for c in chunks:
            c.score = self._lexical_score(query, c.chunk.text)

    def _lexical_score(self, query: str, text: str) -> float:
        q_tokens = set(self._tokenize(query))
        d_tokens = set(self._tokenize(text))
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens.intersection(d_tokens))
        return overlap / (len(q_tokens) ** 0.5 + 1e-9)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in text.lower().split() if t.isalnum()]


class EnsembleReranker:
    """Weighted ensemble of multiple rerankers."""

    def __init__(self, rerankers: List, weights: Optional[List[float]] = None):
        self.rerankers = rerankers
        self.weights = weights or [1.0 / len(rerankers)] * len(rerankers)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if not chunks:
            return chunks
        all_scores = []
        for reranker in self.rerankers:
            scored = list(chunks)
            reranker.rerank(query, scored)
            vals = [c.score for c in scored]
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            all_scores.append([(v - lo) / span for v in vals])
        final = [sum(w * s for w, s in zip(self.weights, col)) for col in zip(*all_scores)]
        for c, s in zip(chunks, final):
            c.score = s
        chunks.sort(key=lambda c: c.score, reverse=True)
        for rank, c in enumerate(chunks, start=1):
            c.rank = rank
        return chunks[:top_k] if top_k else chunks


class NoReranker:
    """Pass-through reranker (naive baseline / ablation control)."""

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: Optional[int] = None) -> List[RetrievedChunk]:
        chunks.sort(key=lambda c: c.score, reverse=True)
        for rank, c in enumerate(chunks, start=1):
            c.rank = rank
        return chunks[:top_k] if top_k else chunks
