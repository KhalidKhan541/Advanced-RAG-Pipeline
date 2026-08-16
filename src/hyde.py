import logging
from typing import Callable, List, Optional

import numpy as np

from .hybrid_retriever import RetrievedChunk


class HyDEGenerator:
    """Generates hypothetical documents (HyDE) for query expansion."""

    def __init__(self, llm: Optional[Callable[[str], str]] = None):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def generate(self, query: str) -> str:
        if self.llm is not None:
            return self.llm(query)
        return self._default_hypothesis(query)

    def generate_batch(self, queries: List[str]) -> List[str]:
        return [self.generate(q) for q in queries]

    def _default_hypothesis(self, query: str) -> str:
        tokens = [t for t in query.lower().split() if len(t) > 3]
        topic = " ".join(tokens[:4]) if tokens else query
        return (
            f"Hypothetical document. The answer to the question \"{query}\" "
            f"relates to {topic}. Key entities and concepts include "
            f"{', '.join(tokens[:6]) if tokens else 'the subject matter'}. "
            f"Contextual facts, dates, names, and causal explanations about "
            f"{topic} are presented in this document, along with supporting "
            f"evidence and references."
        )


class HyDERetriever:
    """Retrieval augmented with hypothetical document embeddings."""

    def __init__(self, base_retriever, hyde_generator: HyDEGenerator, encoder):
        self.base_retriever = base_retriever
        self.hyde_generator = hyde_generator
        self.encoder = encoder

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        hypothesis = self.hyde_generator.generate(query)
        chunks = self.base_retriever.retrieve(hypothesis, top_k=top_k)
        for c in chunks:
            c.chunk.metadata = dict(c.chunk.metadata)
            c.chunk.metadata['hyde_hypothesis'] = hypothesis
        return chunks

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        hypothesis = self.hyde_generator.generate(query)
        direct = self.base_retriever.retrieve(query, top_k=top_k * 2)
        hyde = self.base_retriever.retrieve(hypothesis, top_k=top_k * 2)
        return self._rrf_fuse([direct, hyde], top_k)

    def _rrf_fuse(self, result_lists: List[List[RetrievedChunk]], top_k: int, k: int = 60) -> List[RetrievedChunk]:
        scores = {}
        chunk_map = {}
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                key = id(item.chunk)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                chunk_map.setdefault(key, item)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = [chunk_map[key] for key, _ in ordered]
        for rank, item in enumerate(out, start=1):
            item.rank = rank
        return out


def default_hyde_pipeline(retriever, encoder, llm=None):
    """Factory: retriever wrapped with HyDE."""
    return HyDERetriever(retriever, HyDEGenerator(llm=llm), encoder)
