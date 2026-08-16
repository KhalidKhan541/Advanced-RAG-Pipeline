"""Embeddings and sparse retrieval primitives for the RAG pipeline.

This module provides three building blocks used across the pipeline:

- ``SentenceEncoder``: a thin wrapper around sentence-transformers models with
  a deterministic TF-IDF/hashing fallback when the optional dependency is not
  installed.
- ``BM25Index``: a pure-numpy Okapi BM25 sparse index (k1=1.5, b=0.75) with no
  external search libraries.
- ``DenseIndex``: a numpy-only dense index over precomputed embeddings using
  cosine similarity.

The ``Chunk`` dataclass is the shared unit of text passed between stages.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A unit of retrievable text with optional provenance metadata.

    Attributes:
        id: Unique identifier for the chunk.
        text: The chunk content, used for both sparse and dense retrieval.
        metadata: Arbitrary provenance information (source file, page, etc.).
        parent_id: Identifier of the parent document, when chunking documents.
    """

    id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)
    parent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase a string and split it into alphanumeric tokens.

    Args:
        text: Raw input text.

    Returns:
        List of lowercase tokens; non-alphanumeric characters act as
        separators and are discarded.
    """
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """Normalize a matrix of vectors to unit L2 norm, in place.

    Zero vectors are left untouched to avoid division by zero.

    Args:
        vectors: Shape ``(n, dim)`` float array.

    Returns:
        The same array with each row normalized to unit norm.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors /= norms
    return vectors


# ---------------------------------------------------------------------------
# Sentence encoder
# ---------------------------------------------------------------------------


class SentenceEncoder:
    """Embed text with a sentence-transformers model.

    Uses ``SentenceTransformer`` when available; otherwise falls back to a
    deterministic TF-IDF hashing encoder (fixed hashing vocabulary of 512
    dimensions) so the pipeline still runs without the optional dependency.

    Attributes:
        model_name: Name or path of the sentence-transformers model.
        dim: Dimensionality of produced embeddings.
        using_sentence_transformers: Whether the real model is in use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", hash_dim: int = 512) -> None:
        self.model_name = model_name
        self._hash_dim = hash_dim
        self.using_sentence_transformers = False
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.using_sentence_transformers = True
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except ImportError:
            self.dim = hash_dim
            self._idf = np.zeros(hash_dim, dtype=np.float64)
            self._doc_freq = np.zeros(hash_dim, dtype=np.int64)
            self._n_docs = 0

    # -- hashing helpers ----------------------------------------------------

    def _hash_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        """Fold tokens into a fixed-dimension sparse bag-of-words vector."""
        vec = np.zeros(self._hash_dim, dtype=np.float64)
        for tok in tokens:
            idx = abs(hash(tok)) % self._hash_dim
            vec[idx] += 1.0
        return vec

    def _fit_hash_idf(self, texts: Sequence[str]) -> None:
        """Fit the hashed-vector IDF weights from a corpus of texts."""
        for text in texts:
            seen = set()
            for tok in tokenize(text):
                idx = abs(hash(tok)) % self._hash_dim
                if idx not in seen:
                    seen.add(idx)
                    self._doc_freq[idx] += 1
            self._n_docs += 1
        n = max(self._n_docs, 1)
        self._idf = np.log(1.0 + n / (self._doc_freq.astype(np.float64) + 1.0))

    # -- public API ---------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into an embedding matrix.

        Args:
            texts: List of raw strings.

        Returns:
            Float array of shape ``(len(texts), dim)`` with L2-normalized rows.
        """
        if self.using_sentence_transformers:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32)

        self._fit_hash_idf(texts)
        matrix = np.vstack([self._hash_tokens(tokenize(t)) for t in texts])
        matrix *= self._idf
        return normalize_l2(matrix.astype(np.float32))

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: Raw query text.

        Returns:
            L2-normalized embedding vector of shape ``(dim,)``.
        """
        if self.using_sentence_transformers:
            emb = self._model.encode(query, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        vec = self._hash_tokens(tokenize(query)) * self._idf
        return normalize_l2(vec.astype(np.float32).reshape(1, -1))[0]

    def encode_corpus(self, docs: List[str]) -> np.ndarray:
        """Encode a corpus of documents (alias of :meth:`encode`).

        Provided so callers can distinguish query-time from corpus-time
        encoding, matching sentence-transformers' API surface.
        """
        return self.encode(docs)


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 retrieval over a tokenized corpus, implemented in pure numpy.

    Uses the standard parameters k1=1.5 and b=0.75. The index is built from
    document-term frequency matrices lazily: ``search`` is only meaningful
    after at least one call to :meth:`add_documents`.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._docs: List[Chunk] = []
        self._term_doc_freq: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._n_docs: int = 0
        self._total_terms: int = 0
        self._idf: Dict[str, float] = {}
        self._tf: Dict[str, Dict[str, int]] = {}

    # -- index construction -------------------------------------------------

    def add_documents(self, docs: Sequence[Chunk]) -> None:
        """Add documents to the index and recompute BM25 statistics.

        Args:
            docs: Iterable of :class:`Chunk` objects to index.
        """
        for doc in docs:
            tokens = tokenize(doc.text)
            if not tokens:
                continue
            self._docs.append(doc)
            term_counts: Dict[str, int] = {}
            for tok in tokens:
                term_counts[tok] = term_counts.get(tok, 0) + 1
            self._tf[doc.id] = term_counts
            for term in term_counts:
                self._term_doc_freq[term] = self._term_doc_freq.get(term, 0) + 1
            self._total_terms += len(tokens)
            self._n_docs += 1

        self._avg_doc_len = self._total_terms / max(self._n_docs, 1)
        n_docs = self._n_docs
        self._idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._term_doc_freq.items()
        }

    # -- retrieval ----------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, object]]:
        """Score all documents against a query and return the best matches.

        Args:
            query: Raw query string.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys ``id``, ``text``, ``metadata``,
            ``parent_id`` and ``score`` (BM25 score), sorted descending by
            score.
        """
        query_terms = tokenize(query)
        if not query_terms or not self._docs:
            return []

        scores: List[float] = []
        for doc in self._docs:
            doc_len = sum(self._tf[doc.id].values())
            denom = 1.0 - self.B + self.B * (doc_len / max(self._avg_doc_len, 1e-9))
            total = 0.0
            for term in query_terms:
                tf = self._tf[doc.id].get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                total += idf * (tf * (self.K1 + 1.0)) / (tf + self.K1 * denom)
            scores.append(total)

        ranked = sorted(zip(scores, self._docs), key=lambda pair: pair[0], reverse=True)
        results = []
        for score, doc in ranked[:top_k]:
            if score <= 0.0:
                continue
            results.append(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "parent_id": doc.parent_id,
                    "score": score,
                }
            )
        return results

    # -- statistics ---------------------------------------------------------

    def get_term_frequencies(self) -> Dict[str, int]:
        """Return the global document frequency of every indexed term.

        Returns:
            Mapping of term -> number of documents containing that term.
        """
        return dict(self._term_doc_freq)


# ---------------------------------------------------------------------------
# Dense index
# ---------------------------------------------------------------------------


class DenseIndex:
    """Numpy-only dense retrieval over precomputed embeddings.

    Stores the embedding matrix and returns results ranked by cosine
    similarity (equivalently, dot product on L2-normalized vectors).
    """

    def __init__(self) -> None:
        self._docs: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_row: Dict[str, int] = {}

    def add_documents(self, docs: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Index documents with their precomputed embeddings.

        Args:
            docs: Iterable of :class:`Chunk` objects, aligned with ``embeddings``.
            embeddings: Float matrix of shape ``(len(docs), dim)``. Rows are
                L2-normalized on write.
        """
        if len(docs) != len(embeddings):
            raise ValueError(
                f"docs ({len(docs)}) and embeddings ({len(embeddings)}) must align"
            )
        emb = normalize_l2(np.asarray(embeddings, dtype=np.float32).copy())
        for doc in docs:
            self._id_to_row[doc.id] = len(self._docs)
            self._docs.append(doc)
        self._embeddings = emb

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, object]]:
        """Find the most similar documents to a query embedding.

        Args:
            query_embedding: L2-normalized query vector of shape ``(dim,)``.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys ``id``, ``text``, ``metadata``,
            ``parent_id`` and ``score`` (cosine similarity in [-1, 1]),
            sorted descending by score.
        """
        if self._embeddings is None or len(self._docs) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = normalize_l2(q.copy())
        sims = (self._embeddings @ q.T).ravel()

        top = min(top_k, len(self._docs))
        indices = np.argsort(-sims)[:top]
        results = []
        for idx in indices:
            doc = self._docs[int(idx)]
            results.append(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "parent_id": doc.parent_id,
                    "score": float(sims[idx]),
                }
            )
        return results
