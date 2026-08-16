"""Parent-child chunk retrieval for advanced RAG.

This module contrasts two chunking strategies used by the benchmark:

- :class:`ChunkingStrategy`: sentence-level child chunks built with a
  sliding window, backed by larger parent chunks that supply context at
  retrieval time (the "advanced" approach).
- :class:`FlatChunking`: fixed-size character chunks with no hierarchy
  (the "naive" baseline).

:class:`ParentChildRetriever` searches the child chunks and can return
either child text, parent context, or both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, List

from .embeddings import Chunk
from .hybrid_retriever import RetrievedChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ChunkingStrategy:
    """Sentence-based hierarchical chunking with sliding windows.

    Child chunks group ``child_size`` sentences with a sliding window of
    ``overlap`` sentences; parent chunks group ``parent_size`` sentences
    without overlap. Each child is linked to the parent that covers its
    first sentence.

    Args:
        child_size: Sentences per child chunk.
        parent_size: Sentences per parent chunk.
        overlap: Sentence overlap between consecutive child windows.
    """

    def __init__(self, child_size: int = 2, parent_size: int = 6, overlap: int = 1) -> None:
        if child_size < 1:
            raise ValueError(f"child_size must be >= 1, got {child_size}")
        if parent_size < child_size:
            raise ValueError(
                f"parent_size ({parent_size}) must be >= child_size ({child_size})"
            )
        if not 0 <= overlap < child_size:
            raise ValueError(f"overlap must be in [0, child_size), got {overlap}")
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences on sentence-final punctuation.

        Args:
            text: Raw text.

        Returns:
            List of non-empty stripped sentences.
        """
        parts = _SENTENCE_SPLIT.split(text.strip())
        return [p.strip() for p in parts if p.strip()]

    def create_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        """Create child chunks, each linked to its covering parent.

        Args:
            text: Document text.
            doc_id: Unique document identifier used in chunk ids.

        Returns:
            Child chunks with ``parent_id`` set to the parent covering
            their first sentence.
        """
        sentences = self.split_sentences(text)
        children: List[Chunk] = []
        step = self.child_size - self.overlap
        for i in range(0, len(sentences), step):
            window = sentences[i : i + self.child_size]
            if not window:
                break
            parent_id = f"{doc_id}::parent::{i // self.parent_size}"
            children.append(
                Chunk(
                    id=f"{doc_id}::child::{i // step}",
                    text=" ".join(window),
                    metadata={
                        "doc_id": doc_id,
                        "chunk_type": "child",
                        "sentence_start": i,
                    },
                    parent_id=parent_id,
                )
            )
        return children

    def create_parent_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        """Create non-overlapping parent chunks for a document.

        Args:
            text: Document text.
            doc_id: Unique document identifier used in chunk ids.

        Returns:
            Parent chunks; the final chunk may be shorter than
            ``parent_size`` sentences.
        """
        sentences = self.split_sentences(text)
        parents: List[Chunk] = []
        for j in range(0, len(sentences), self.parent_size):
            window = sentences[j : j + self.parent_size]
            if not window:
                break
            parents.append(
                Chunk(
                    id=f"{doc_id}::parent::{j // self.parent_size}",
                    text=" ".join(window),
                    metadata={
                        "doc_id": doc_id,
                        "chunk_type": "parent",
                        "sentence_start": j,
                    },
                )
            )
        return parents


@dataclass
class ParentChildIndex:
    """In-memory index linking child chunks to their parent chunks.

    Attributes:
        children: All indexed child chunks.
        parents: All indexed parent chunks.
        child_to_parent: Mapping from child chunk id to its parent chunk.
        strategy: Chunking strategy used for newly added documents.
    """

    children: List[Chunk] = field(default_factory=list)
    parents: List[Chunk] = field(default_factory=list)
    child_to_parent: Dict[str, Chunk] = field(default_factory=dict)
    strategy: ChunkingStrategy = field(default_factory=ChunkingStrategy)

    def add_document(self, text: str, doc_id: str) -> None:
        """Chunk and index a document, linking children to parents.

        Args:
            text: Document text.
            doc_id: Unique document identifier.
        """
        children = self.strategy.create_chunks(text, doc_id)
        parents = self.strategy.create_parent_chunks(text, doc_id)
        parent_by_id = {p.id: p for p in parents}
        for child in children:
            parent = parent_by_id.get(child.parent_id or "")
            if parent is not None:
                self.child_to_parent[child.id] = parent
        self.children.extend(children)
        self.parents.extend(parents)


class ParentChildRetriever:
    """Retrieves child chunks, optionally replacing them with parent context.

    The wrapped ``retriever`` operates on child chunks; this class adds the
    parent-child context layer on top.

    Args:
        retriever: Retriever exposing ``retrieve(query, top_k)`` and
            returning :class:`RetrievedChunk` objects over child chunks.
        index: :class:`ParentChildIndex` backing the child-to-parent links.
    """

    def __init__(self, retriever, index: ParentChildIndex) -> None:
        self.retriever = retriever
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        parent_context: bool = True,
    ) -> List[RetrievedChunk]:
        """Retrieve children, optionally returning parent text for context.

        Args:
            query: Raw query string.
            top_k: Number of child hits to fetch.
            parent_context: When ``True`` the returned chunk text is the
                covering parent's text (the child text is a subset of it);
                otherwise the raw child text is returned.

        Returns:
            :class:`RetrievedChunk` objects whose ``metadata`` carries
            ``child_id``, ``parent_id`` and ``strategy``.
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        out: List[RetrievedChunk] = []
        for hit in results:
            child = hit.chunk
            parent = self.index.child_to_parent.get(child.id)
            metadata = dict(child.metadata)
            metadata["child_id"] = child.id
            if parent_context and parent is not None:
                chunk = replace(child, text=parent.text, parent_id=parent.id)
                metadata["parent_id"] = parent.id
                metadata["strategy"] = "parent_context"
            else:
                chunk = child
                metadata["parent_id"] = child.parent_id
                metadata["strategy"] = "child_only"
            chunk = replace(chunk, metadata=metadata)
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=hit.score,
                    sparse_score=hit.sparse_score,
                    dense_score=hit.dense_score,
                    rank=hit.rank,
                )
            )
        return out

    def retrieve_with_both(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Retrieve child chunks plus their parents as separate hits.

        Produces one :class:`RetrievedChunk` per child and one per distinct
        covering parent, tagged ``strategy="child"`` / ``"parent"`` and
        ranked together by score. Used for ablation studies.

        Args:
            query: Raw query string.
            top_k: Number of child hits to fetch.

        Returns:
            Combined, score-ranked list of child and parent hits.
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        out: List[RetrievedChunk] = []
        seen_parents = set()
        for hit in results:
            child = hit.chunk
            child_metadata = dict(child.metadata)
            child_metadata.update(
                {"child_id": child.id, "parent_id": child.parent_id, "strategy": "child"}
            )
            out.append(
                RetrievedChunk(
                    chunk=replace(child, metadata=child_metadata),
                    score=hit.score,
                    sparse_score=hit.sparse_score,
                    dense_score=hit.dense_score,
                )
            )
            parent = self.index.child_to_parent.get(child.id)
            if parent is None or parent.id in seen_parents:
                continue
            seen_parents.add(parent.id)
            parent_metadata = dict(parent.metadata)
            parent_metadata.update(
                {"child_id": child.id, "parent_id": parent.id, "strategy": "parent"}
            )
            out.append(
                RetrievedChunk(
                    chunk=replace(parent, metadata=parent_metadata),
                    score=hit.score,
                    sparse_score=hit.sparse_score,
                    dense_score=hit.dense_score,
                )
            )
        out.sort(key=lambda r: r.score, reverse=True)
        for i, hit in enumerate(out):
            hit.rank = i
        return out


class FlatChunking:
    """Naive fixed-size character chunking, no hierarchy.

    Splits text into consecutive windows of ``chunk_size`` characters with
    ``overlap`` characters shared between neighbors. Serves as the naive
    baseline in chunking comparisons.

    Args:
        chunk_size: Target characters per chunk.
        overlap: Characters shared between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if not 0 <= overlap < chunk_size:
            raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        """Create flat fixed-size chunks for a document.

        Args:
            text: Document text.
            doc_id: Unique document identifier used in chunk ids.

        Returns:
            Flat list of :class:`Chunk` objects with no parent links.
        """
        step = self.chunk_size - self.overlap
        chunks: List[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"{doc_id}::flat::{len(chunks)}",
                        text=piece,
                        metadata={"doc_id": doc_id, "chunk_type": "flat", "offset": start},
                    )
                )
            if end >= len(text):
                break
            start += step
        return chunks
