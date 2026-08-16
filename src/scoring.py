"""Faithfulness and relevance scoring for RAG answers (RAGAS-inspired).

The naive RAG baseline trusts retrieved output blindly; the advanced
counterpart verifies it. This module provides token-overlap metrics that
approximate RAGAS-style faithfulness and relevance without an LLM judge:

- :class:`AnswerScorer`: core overlap metrics with light stemming.
- :class:`RAGEvaluator`: per-sample scores plus dataset aggregates.
- :class:`FaithfulnessChecker`: per-claim support verification.

Stemming (stripping ``s``/``es``/``ed``/``ing`` suffixes) acts as a cheap
synonym fallback so morphological variants like ``run``/``runs``/``running``
count as matches.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

SUPPORT_OVERLAP_THRESHOLD = 0.4
"""Minimum claim token overlap for :class:`FaithfulnessChecker` support."""

_WORD_RE = re.compile(r"[a-z0-9]+")
_STEM_SUFFIXES = ("ing", "ed", "es", "s")
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "with", "by", "and", "or", "but",
        "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
        "do", "does", "did", "it", "its", "this", "that", "these", "those",
        "not", "no", "if", "as", "from", "about", "into", "than",
    }
)


def _tokenize(text: str) -> List[str]:
    """Lowercase a string and split it into alphanumeric tokens."""
    return _WORD_RE.findall(text.lower())


def _stem(token: str) -> str:
    """Strip a small set of morphological suffixes as a synonym fallback.

    Args:
        token: A single lowercase token.

    Returns:
        The token with at most one ``s``/``es``/``ed``/``ing`` suffix
        removed; stems shorter than three characters are left untouched so
        words like ``is`` are not reduced to ``i``.
    """
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _stemmed(text: str) -> set[str]:
    """Tokenize ``text`` and stem every token."""
    return {_stem(token) for token in _tokenize(text)}


class AnswerScorer:
    """Token-overlap scoring of answers and retrieved contexts."""

    def faithfulness_score(self, answer: str, contexts: List[str]) -> float:
        """Fraction of answer tokens present in the retrieved contexts.

        Args:
            answer: Generated answer text.
            contexts: Retrieved chunks used to produce the answer.

        Returns:
            Score in ``[0, 1]``; 0.0 when the answer is empty.
        """
        answer_tokens = _stemmed(answer)
        if not answer_tokens:
            return 0.0
        context_tokens: set[str] = set()
        for context in contexts:
            context_tokens |= _stemmed(context)
        if not context_tokens:
            return 0.0
        return sum(1 for token in answer_tokens if token in context_tokens) / len(answer_tokens)

    def relevance_score(self, query: str, answer: str) -> float:
        """Fraction of query key tokens covered by the answer.

        Query stopwords are excluded so only content-bearing terms count.

        Args:
            query: The user question.
            answer: The generated answer.

        Returns:
            Score in ``[0, 1]``; 0.0 when the query has no key tokens.
        """
        query_keys = self._key_tokens(query)
        if not query_keys:
            return 0.0
        answer_tokens = _stemmed(answer)
        return sum(1 for token in query_keys if token in answer_tokens) / len(query_keys)

    def context_relevance(self, query: str, contexts: List[str]) -> float:
        """Best query key-token overlap ratio across any single context.

        Args:
            query: The user question.
            contexts: Retrieved chunks.

        Returns:
            Score in ``[0, 1]``; 0.0 when there is nothing to score.
        """
        query_keys = self._key_tokens(query)
        if not query_keys or not contexts:
            return 0.0
        best = 0.0
        for context in contexts:
            context_tokens = _stemmed(context)
            if not context_tokens:
                continue
            ratio = sum(1 for token in query_keys if token in context_tokens) / len(query_keys)
            best = max(best, ratio)
        return best

    def answer_similarity(self, gold: str, predicted: str) -> float:
        """Token Jaccard similarity between a gold and predicted answer.

        Args:
            gold: Reference answer.
            predicted: Generated answer.

        Returns:
            Score in ``[0, 1]``; 1.0 when both texts are empty.
        """
        gold_tokens = _stemmed(gold)
        predicted_tokens = _stemmed(predicted)
        union = gold_tokens | predicted_tokens
        if not union:
            return 1.0
        return len(gold_tokens & predicted_tokens) / len(union)

    @staticmethod
    def _key_tokens(text: str) -> set[str]:
        """Stemmed tokens of ``text`` with stopwords removed."""
        return {_stem(token) for token in _tokenize(text) if token not in _STOPWORDS}


@dataclass
class RAGScore:
    """Aggregate quality metrics for a single RAG sample.

    Attributes:
        faithfulness: Fraction of answer tokens grounded in contexts.
        answer_relevance: Query key-token coverage by the answer.
        context_relevance: Best query key-token coverage in one context.
        answer_similarity: Token Jaccard vs the gold answer; 0.0 when no
            gold answer is available.
        context_precision: Fraction of retrieved contexts actually used
            (sharing at least one token) by the answer.
        latency_ms: Wall-clock time of the scoring pass, in milliseconds.
    """

    faithfulness: float
    answer_relevance: float
    context_relevance: float
    answer_similarity: float
    context_precision: float
    latency_ms: float


_METRIC_FIELDS = (
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "answer_similarity",
    "context_precision",
    "latency_ms",
)


class RAGEvaluator:
    """Scores RAG samples and aggregates them across a dataset.

    Args:
        scorer: :class:`AnswerScorer` used for the overlap metrics; a new
            instance is created when not provided.
    """

    def __init__(self, scorer: Optional[AnswerScorer] = None) -> None:
        self.scorer = scorer or AnswerScorer()

    def evaluate_sample(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        gold: Optional[str] = None,
    ) -> RAGScore:
        """Score one ``(query, answer, contexts)`` sample.

        Args:
            query: User question.
            answer: Generated answer.
            contexts: Retrieved contexts, as a list of strings.
            gold: Optional gold answer for similarity scoring.

        Returns:
            :class:`RAGScore` with all metrics plus scoring latency.
        """
        start = time.perf_counter()
        contexts = list(contexts)
        faithfulness = self.scorer.faithfulness_score(answer, contexts)
        answer_relevance = self.scorer.relevance_score(query, answer)
        context_relevance = self.scorer.context_relevance(query, contexts)
        answer_similarity = self.scorer.answer_similarity(gold, answer) if gold else 0.0
        context_precision = self._context_precision(answer, contexts)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RAGScore(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_relevance=context_relevance,
            answer_similarity=answer_similarity,
            context_precision=context_precision,
            latency_ms=latency_ms,
        )

    def evaluate_dataset(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        """Score every sample and aggregate per-metric means.

        Args:
            rows: Sequence of dicts with keys ``query``, ``answer``,
                ``contexts`` (list of strings) and optional ``gold``.

        Returns:
            Mapping with ``aggregate`` (per-metric mean over all samples)
            and ``per_sample`` (one ``RAGScore`` dict per input row).
        """
        samples = [
            self.evaluate_sample(
                query=str(row.get("query", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", []) or []),
                gold=row.get("gold"),
            )
            for row in rows
        ]
        per_sample = [asdict(sample) for sample in samples]
        if samples:
            aggregate = {
                metric: float(np.mean([getattr(sample, metric) for sample in samples]))
                for metric in _METRIC_FIELDS
            }
        else:
            aggregate = {metric: 0.0 for metric in _METRIC_FIELDS}
        return {"aggregate": aggregate, "per_sample": per_sample}

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _context_precision(answer: str, contexts: List[str]) -> float:
        """Fraction of contexts sharing at least one token with the answer."""
        if not contexts:
            return 0.0
        answer_tokens = _stemmed(answer)
        if not answer_tokens:
            return 0.0
        used = sum(1 for context in contexts if answer_tokens & _stemmed(context))
        return used / len(contexts)


class FaithfulnessChecker:
    """Verifies that individual claims are supported by the contexts.

    Args:
        threshold: Minimum claim token overlap — the fraction of claim
            tokens found in a single context — required for a claim to
            count as supported. Defaults to
            :data:`SUPPORT_OVERLAP_THRESHOLD`.
    """

    def __init__(self, threshold: float = SUPPORT_OVERLAP_THRESHOLD) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def check_claim_support(self, claim: str, contexts: List[str]) -> Dict[str, object]:
        """Check a claim against the retrieved contexts.

        Args:
            claim: Single claim to verify.
            contexts: Retrieved chunks the answer was grounded on.

        Returns:
            Mapping with ``supported`` (bool), ``score`` (best claim-token
            overlap ratio in ``[0, 1]``) and ``supporting_context`` (the
            best matching context when supported, else ``None``).
        """
        claim_tokens = _stemmed(claim)
        if not claim_tokens:
            return {"supported": False, "score": 0.0, "supporting_context": None}

        best_score, best_context = 0.0, None
        for context in contexts:
            context_tokens = _stemmed(context)
            if not context_tokens:
                continue
            overlap = sum(1 for token in claim_tokens if token in context_tokens) / len(claim_tokens)
            if overlap > best_score:
                best_score, best_context = overlap, context

        supported = best_score >= self.threshold
        return {
            "supported": supported,
            "score": best_score,
            "supporting_context": best_context if supported else None,
        }