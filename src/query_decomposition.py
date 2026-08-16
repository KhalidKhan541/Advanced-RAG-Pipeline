"""Query decomposition and multi-hop reasoning for advanced RAG.

This module contrasts two strategies used by the benchmark:

- :class:`QueryDecomposer`: breaks a complex question into sub-questions
  using an LLM callable when available, otherwise a rule-based fallback.
- :class:`MultiHopReasoner`: decomposes the query, retrieves evidence per
  sub-question, and issues follow-up sub-queries across hops until the
  retrieved scores clear a sufficiency threshold. Evidence from all hops is
  fused into a single ranked list.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .hyde import HyDEGenerator
from .hybrid_retriever import RetrievedChunk

SUFFICIENCY_THRESHOLD = 0.3
"""Minimum top-1 score for a hop to be considered sufficient evidence."""

_CONNECTIVE_SPLIT = re.compile(
    r"\s+(?:and|but|however|while|whereas)\s+|;|\?",
    re.IGNORECASE,
)
_RELATION_BETWEEN = re.compile(
    r"\b(?:relationship|relation|difference|comparison)\s+between\s+(.+?)\s+and\s+(.+)",
    re.IGNORECASE,
)
_COMPARE_X_AND_Y = re.compile(
    r"\bcompare\s+(.+?)\s+and\s+(.+)",
    re.IGNORECASE,
)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_'\-]{1,}\b")
_SENTENCE_STARTERS = frozenset(
    {
        "The", "This", "That", "These", "Those", "It", "Its", "In", "At",
        "On", "And", "But", "A", "An", "For", "As", "He", "She", "They",
        "We", "There", "However", "While",
    }
)


class QueryDecomposer:
    """Splits complex queries into answerable sub-questions.

    Args:
        llm: Optional callable ``llm(query)`` returning the decomposed
            sub-queries as a string (one per line) or a list of strings.
            When ``None`` the rule-based :meth:`_default_decompose` is used.
    """

    def __init__(self, llm: Optional[Callable[[str], object]] = None) -> None:
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def decompose(self, query: str) -> List[str]:
        """Decompose a query into sub-questions.

        Args:
            query: Raw query string.

        Returns:
            List of sub-question strings, ordered best first.
        """
        if self.llm is not None:
            return self._llm_decompose(query)
        return self._default_decompose(query)

    def _llm_decompose(self, query: str) -> List[str]:
        raw = self.llm(query)
        if isinstance(raw, list):
            parts = [str(p).strip() for p in raw]
        else:
            parts = [line.strip(" \t-0123456789.") for line in str(raw).splitlines()]
        return [p for p in parts if p]

    def _default_decompose(self, query: str) -> List[str]:
        """Rule-based decomposition over connective and relational patterns."""
        cleaned = query.strip()
        if not cleaned:
            return []

        rel = _RELATION_BETWEEN.search(cleaned)
        if rel:
            x, y = rel.group(1).strip(), rel.group(2).strip()
            if x and y:
                return [
                    f"What is {x}?",
                    f"What is {y}?",
                    f"how do {x} and {y} relate?",
                ]

        cmp = _COMPARE_X_AND_Y.search(cleaned)
        if cmp:
            x, y = cmp.group(1).strip(), cmp.group(2).strip()
            if x and y:
                return [
                    f"What is {x}?",
                    f"What is {y}?",
                    f"how do {x} and {y} compare?",
                ]

        parts = [p.strip() for p in _CONNECTIVE_SPLIT.split(cleaned) if p.strip()]
        return parts if len(parts) > 1 else [cleaned]


@dataclass
class MultiHopResult:
    """Outcome of a multi-hop reasoning pass.

    Attributes:
        query: The original query.
        sub_queries: Decomposed sub-questions, in order.
        hops: Retrieved chunks per retrieval round (one entry per round).
        evidence: Deduplicated, score-ranked chunks from every hop.
        reasoning_trace: Human-readable log of hop-by-hop decisions.
    """

    query: str
    sub_queries: List[str]
    hops: List[List[RetrievedChunk]]
    evidence: List[RetrievedChunk]
    reasoning_trace: List[str]


class MultiHopReasoner:
    """Iterative retrieve-and-refine reasoning over a single retriever.

    Args:
        retriever: Any object exposing ``retrieve(query, top_k)`` and
            returning :class:`RetrievedChunk` objects.
        llm: Optional LLM callable forwarded to the internal
            :class:`QueryDecomposer`.
    """

    def __init__(
        self,
        retriever,
        llm: Optional[Callable[[str], object]] = None,
    ) -> None:
        self.retriever = retriever
        self.decomposer = QueryDecomposer(llm=llm)
        self.hyde = HyDEGenerator()
        self.logger = logging.getLogger(__name__)

    def reason(self, query: str, max_hops: int = 3, top_k: int = 5) -> MultiHopResult:
        """Run multi-hop reasoning over a query.

        Each sub-question is retrieved once per hop. When the best score
        falls below :data:`SUFFICIENCY_THRESHOLD` (or no results come
        back), a follow-up sub-query is built from the capitalized entity
        terms of the previous hop and retrieval repeats, up to ``max_hops``
        per sub-question.

        Args:
            query: Raw query string.
            max_hops: Maximum retrieval rounds per sub-question.
            top_k: Chunks retrieved per round.

        Returns:
            :class:`MultiHopResult` with all hop evidence fused.
        """
        sub_queries = self.decomposer.decompose(query) or [query]
        hops: List[List[RetrievedChunk]] = []
        trace: List[str] = []
        collected: Dict[str, RetrievedChunk] = {}

        for sub_query in sub_queries:
            current = sub_query
            for hop in range(1, max_hops + 1):
                results = self.retriever.retrieve(current, top_k=top_k)
                trace.append(f'hop {hop} on "{current}": {len(results)} result(s)')
                for hit in results:
                    collected.setdefault(hit.chunk.id, hit)
                hops.append(list(results))

                if self._is_sufficient(results):
                    break
                follow_up = self._follow_up(results)
                if not follow_up or follow_up == current:
                    trace.append("no follow-up query generated; stopping")
                    break
                trace.append(f'scores below threshold; follow-up: "{follow_up}"')
                current = follow_up

        evidence = self._fuse(collected)
        return MultiHopResult(
            query=query,
            sub_queries=sub_queries,
            hops=hops,
            evidence=evidence,
            reasoning_trace=trace,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retriever-compatible interface: run multi-hop reasoning and return fused evidence.

        Lets :class:`MultiHopReasoner` drop into any pipeline slot that
        expects a ``retrieve(query, top_k)`` object.

        Args:
            query: Raw query string.
            top_k: Chunks retrieved per hop.

        Returns:
            Fused, ranked evidence chunks.
        """
        result = self.reason(query, top_k=top_k)
        for rank, hit in enumerate(result.evidence, start=1):
            hit.rank = rank
        return result.evidence

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _is_sufficient(results: List[RetrievedChunk]) -> bool:
        """A hop is sufficient when it returns hits scoring above threshold."""
        return bool(results) and results[0].score >= SUFFICIENCY_THRESHOLD

    def _follow_up(self, results: List[RetrievedChunk]) -> str:
        """Build a follow-up sub-query from the previous hop's entities.

        Capitalized tokens from the hop results are deduplicated into a
        short entity query; when none are found, a HyDE hypothesis over the
        top hit substitutes as the follow-up.
        """
        if not results:
            return ""
        entities: List[str] = []
        for hit in results:
            for token in _ENTITY_PATTERN.findall(hit.chunk.text):
                if token not in _SENTENCE_STARTERS and token not in entities:
                    entities.append(token)
            if len(entities) >= 5:
                break
        if entities:
            return " ".join(entities[:5])
        return self.hyde.generate(results[0].chunk.text[:400])

    @staticmethod
    def _fuse(collected: Dict[str, RetrievedChunk]) -> List[RetrievedChunk]:
        """Deduplicate hop evidence by chunk id and rank by score."""
        ranked = sorted(collected.values(), key=lambda r: r.score, reverse=True)
        for i, hit in enumerate(ranked):
            hit.rank = i
        return ranked


def build_multi_hop(
    retriever,
    llm: Optional[Callable[[str], object]] = None,
) -> MultiHopReasoner:
    """Convenience factory for a :class:`MultiHopReasoner`.

    Args:
        retriever: Retriever exposing ``retrieve(query, top_k)``.
        llm: Optional LLM callable for query decomposition.

    Returns:
        A configured :class:`MultiHopReasoner`.
    """
    return MultiHopReasoner(retriever, llm=llm)
