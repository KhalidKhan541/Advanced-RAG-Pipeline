"""Per-claim citation tracking for advanced RAG answers.

The naive RAG baseline returns a bare answer with no provenance; this module
implements the advanced counterpart: every claim in an answer is matched
against the corpus through the retriever, and the supporting chunks are
surfaced as numbered citations with a references block.

The pipeline is:

1. :class:`ClaimSplitter` splits an answer into atomic claims.
2. :class:`CitationTracker` retrieves evidence per claim, assigns ``[n]``
   citation indices in order of first use, and renders a cited answer with
   inline markers plus a ``References`` block.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from .hybrid_retriever import RetrievedChunk

SUPPORT_THRESHOLD = 0.15
"""Minimum retrieval score for a chunk to support a claim."""

MIN_CLAIM_LENGTH = 24
"""Fragments shorter than this (in characters) merge into the previous claim."""

SNIPPET_LENGTH = 280
"""Maximum characters kept in a reference snippet."""

REFERENCES_SEPARATOR = "\n\nReferences\n"
"""Delimiter between the cited answer body and its references block."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"\s*;\s*|\s+but\s+|\s+whereas\s+", re.IGNORECASE)


@dataclass
class Claim:
    """A single atomic claim extracted from an answer.

    Attributes:
        text: Claim text as it appears in the answer.
        source_chunk_ids: Chunk ids that support the claim; empty when the
            claim could not be supported by any retrieved chunk.
        confidence: Retrieval score of the strongest supporting chunk,
            in ``[0, 1]``; 0.0 for unsupported claims.
    """

    text: str
    source_chunk_ids: List[str]
    confidence: float


@dataclass
class Citation:
    """A numbered citation linking a claim to one supporting chunk.

    Attributes:
        claim_text: The claim this citation supports.
        chunk_id: Identifier of the supporting chunk.
        chunk_text: Truncated snippet of the chunk text.
        score: Retrieval score of the chunk for the claim.
        citation_index: 1-based index rendered as ``[1]``, ``[2]``, ...,
            assigned in order of first use across the whole answer.
    """

    claim_text: str
    chunk_id: str
    chunk_text: str
    score: float
    citation_index: int


class ClaimSplitter:
    """Splits answer text into atomic claims.

    Claims are split on sentence boundaries (``.``, ``!``, ``?``) and on
    clause markers (``;``, `` but ``, `` whereas ``). Fragments shorter than
    :attr:`min_claim_length` are merged into the previous claim so that
    trailing clauses such as ``"using RAG."`` stay attached to their subject.

    Args:
        min_claim_length: Minimum character length a standalone fragment
            must have; shorter fragments merge into the previous claim.
    """

    def __init__(self, min_claim_length: int = MIN_CLAIM_LENGTH) -> None:
        self.min_claim_length = min_claim_length

    def split_answer(self, answer: str) -> List[str]:
        """Split an answer into individual claims.

        Args:
            answer: Full answer text.

        Returns:
            Ordered list of claim strings, each non-empty and trimmed.
        """
        fragments: List[str] = []
        for sentence in _SENTENCE_SPLIT.split(answer.strip()):
            for clause in _CLAUSE_SPLIT.split(sentence):
                clause = clause.strip()
                if not clause:
                    continue
                if fragments and len(clause) < self.min_claim_length:
                    fragments[-1] = f"{fragments[-1]} {clause}"
                else:
                    fragments.append(clause)
        return fragments


@dataclass
class CitationResult:
    """Full outcome of citing an answer.

    Attributes:
        answer: The original, unmodified answer.
        cited_answer: The answer with ``[n]`` markers appended per claim,
            plus a ``References`` block listing the chunk snippets.
        claims: Claims extracted from the answer, in order.
        citations: Mapping of claim text to its list of citations.
        references: Formatted ``[n] snippet`` lines, in citation order.
    """

    answer: str
    cited_answer: str
    claims: List[Claim]
    citations: Dict[str, List[Citation]]
    references: List[str]


class CitationTracker:
    """Assigns per-claim citations by retrieving evidence for each claim.

    Args:
        retriever: Any object exposing ``retrieve(query, top_k)`` and
            returning :class:`RetrievedChunk` objects.
    """

    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self.splitter = ClaimSplitter()
        self.logger = logging.getLogger(__name__)

    def cite(self, answer: str, query: str, top_k: int = 5) -> CitationResult:
        """Cite every claim in an answer against the corpus.

        The answer is split into claims; each claim is used as a retrieval
        query. Chunks scoring at least :data:`SUPPORT_THRESHOLD` become
        citations, with ``[n]`` indices assigned in order of first use.
        Claims with no supporting chunk are marked ``[unsupported]`` in the
        cited answer and carry zero confidence.

        Args:
            answer: The answer text to cite.
            query: The original user query (used for logging context).
            top_k: Chunks retrieved per claim.

        Returns:
            A :class:`CitationResult` with the cited answer, structured
            citations, and the references block.
        """
        claims = self.splitter.split_answer(answer)
        if not claims:
            self.logger.warning("no claims extracted from answer; query=%r", query)
            return CitationResult(answer=answer, cited_answer=answer, claims=[], citations={}, references=[])

        citations: Dict[str, List[Citation]] = {}
        claim_records: List[Claim] = []
        index_by_chunk: Dict[str, int] = {}
        references: List[str] = []
        next_index = 1

        for claim in claims:
            hits = self.retriever.retrieve(claim, top_k=top_k)
            best_score = float(hits[0].score) if hits else 0.0
            claim_citations: List[Citation] = []

            for hit in hits:
                if hit.score < SUPPORT_THRESHOLD:
                    continue
                index = index_by_chunk.get(hit.chunk.id)
                if index is None:
                    index = next_index
                    next_index += 1
                    index_by_chunk[hit.chunk.id] = index
                    references.append(f"[{index}] {self._snippet(hit.chunk.text)}")
                claim_citations.append(
                    Citation(
                        claim_text=claim,
                        chunk_id=hit.chunk.id,
                        chunk_text=self._snippet(hit.chunk.text),
                        score=float(hit.score),
                        citation_index=index,
                    )
                )

            if claim_citations:
                source_ids = [c.chunk_id for c in claim_citations]
                confidence = max(c.score for c in claim_citations)
            else:
                self.logger.info("unsupported claim (best score %.3f): %r", best_score, claim)
                source_ids = []
                confidence = 0.0

            citations[claim] = claim_citations
            claim_records.append(Claim(text=claim, source_chunk_ids=source_ids, confidence=confidence))

        cited_answer = self._render_cited_answer(answer, claims, citations, references)
        return CitationResult(
            answer=answer,
            cited_answer=cited_answer,
            claims=claim_records,
            citations=citations,
            references=references,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
        """Collapse whitespace and truncate chunk text into a snippet."""
        text = " ".join(text.split())
        return text if len(text) <= length else text[: length - 1] + "\u2026"

    @staticmethod
    def _render_cited_answer(
        answer: str,
        claims: List[str],
        citations: Dict[str, List[Citation]],
        references: List[str],
    ) -> str:
        """Insert ``[n]`` markers after each claim and append references.

        Each claim is located in the original answer by forward search from
        the previous match, so markers land exactly where the claim ends
        even when claims repeat elsewhere in the text.
        """
        body: List[str] = []
        cursor = 0
        for claim in claims:
            pos = answer.find(claim, cursor)
            if pos == -1:
                continue
            body.append(answer[cursor:pos])
            body.append(claim)
            markers = "".join(f"[{c.citation_index}]" for c in citations[claim])
            body.append(markers if markers else " [unsupported]")
            cursor = pos + len(claim)
        body.append(answer[cursor:])

        rendered = "".join(body)
        if references:
            rendered += REFERENCES_SEPARATOR + "\n".join(references)
        return rendered


def format_cited_answer(result: CitationResult) -> str:
    """Render the cited answer nicely with markers and a references section.

    The answer body is kept as produced by :meth:`CitationTracker.cite`;
    the references block is rebuilt from the structured result so it always
    matches the current citations, even for answers with no claims.

    Args:
        result: The :class:`CitationResult` to render.

    Returns:
        Multi-line string: answer with inline ``[n]`` markers followed by
        the ``References`` block.
    """
    body = result.cited_answer.split(REFERENCES_SEPARATOR, 1)[0]
    if result.references:
        block = "\n".join(result.references)
    else:
        block = "  (no supporting references found)"
    return body + REFERENCES_SEPARATOR + block