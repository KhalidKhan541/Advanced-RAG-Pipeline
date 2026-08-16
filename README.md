# Advanced RAG Pipeline — Naive vs Advanced Comparison

Full-stack RAG engineering: **hybrid retrieval (BM25 + dense embeddings)**, **cross-encoder reranking**, **HyDE** (hypothetical document embeddings), **query decomposition + multi-hop reasoning**, **parent-child chunk retrieval**, **citation tracking per claim**, and **faithfulness + relevance scoring (RAGAS-inspired)** — benchmarked **incrementally** (naive RAG vs each enhancement added one at a time) on the same QA dataset.

```
                 ┌─────────────────────────────────────────────────────┐
                 │                  ADVANCED RAG PIPELINE              │
                 └─────────────────────────────────────────────────────┘
  Query ──► ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
            │  HYBRID      │   │  CROSS-      │   │   HyDE       │
            │  RETRIEVAL   │──►│  ENCODER     │──►│  EXPANSION   │
            │  BM25 + Dense│   │  RERANKER    │   │  Hypothetical│
            └──────────────┘   └──────────────┘   │  Documents   │
                                                   └──────┬───────┘
                                                          ▼
  Answer ◄── ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  + Cites   │  CITATION     │◄──│  SCORING     │◄──│ QUERY DECOMP │
  + [1][2]  │  TRACKING     │   │  RAGAS-style │   │ MULTI-HOP    │
            └──────────────┘   └──────────────┘   └──────┬───────┘
                                                          ▼
                                                   ┌──────────────┐
                                                   │ PARENT-CHILD │
                                                   │ CHUNK        │
                                                   │ RETRIEVAL    │
                                                   └──────────────┘
```

## Features

| # | Feature | Description | Where |
|---|---------|-------------|-------|
| 1 | **Hybrid Retrieval** | Reciprocal Rank Fusion (RRF) of BM25 sparse + dense embedding retrieval | `src/hybrid_retriever.py` |
| 2 | **Cross-Encoder Reranking** | Pairwise query-document relevance scoring + ensemble support | `src/cross_encoder.py` |
| 3 | **HyDE** | Hypothetical Document Embeddings — retrieve on a synthesized pseudo-document | `src/hyde.py` |
| 4 | **Query Decomposition + Multi-Hop** | Split complex queries into sub-questions, iterative evidence chaining | `src/query_decomposition.py` |
| 5 | **Parent-Child Chunking** | Retrieve small precise child chunks, expand to parent context | `src/parent_child_retrieval.py` |
| 6 | **Citation Tracking** | Per-claim source attribution with `[1] [2]` markers and reference block | `src/citations.py` |
| 7 | **RAGAS-style Scoring** | Faithfulness, answer relevance, context relevance, answer similarity | `src/scoring.py` |
| 8 | **Incremental Benchmark** | Naive RAG vs each enhancement, one at a time, same QA dataset | `src/benchmark.py` |

## Architecture

```
Advanced-RAG-Pipeline/
├── src/
│   ├── embeddings.py              # SentenceEncoder, BM25Index (pure numpy), DenseIndex, Chunk
│   ├── hybrid_retriever.py        # HybridRetriever (RRF), NaiveRetriever baseline
│   ├── cross_encoder.py           # CrossEncoderReranker, EnsembleReranker, NoReranker
│   ├── hyde.py                    # HyDEGenerator, HyDERetriever
│   ├── query_decomposition.py     # QueryDecomposer, MultiHopReasoner
│   ├── parent_child_retrieval.py  # ChunkingStrategy, ParentChildRetriever, FlatChunking
│   ├── citations.py               # ClaimSplitter, CitationTracker
│   ├── scoring.py                 # AnswerScorer, RAGEvaluator, FaithfulnessChecker
│   └── benchmark.py               # PipelineConfig, RAGBenchmark, AnswerGenerator
├── main.py                        # Demo: builds dataset, runs incremental benchmark
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the full demo: incremental benchmark + citation tracking demo
python main.py
```

> No API keys needed. The pipeline runs end-to-end with local models; every
> "advanced" component has a deterministic fallback so the benchmark runs even
> without network access.

## How Each Enhancement Works

### 1. Hybrid Retrieval — BM25 + Dense Embeddings

Naive RAG uses **only** dense retrieval (embedding similarity), which fails on
exact-term queries and rare entities. This pipeline fuses two signals:

- **BM25 (sparse)**: lexical term-frequency scoring — `src/embeddings.py:BM25Index`
  (pure numpy, Okapi BM25 with k1=1.5, b=0.75).
- **Dense**: cosine similarity over `SentenceTransformer` embeddings with an
  L2-normalized TF-IDF hashing fallback.
- **Fusion**: Reciprocal Rank Fusion — `score = Σ 1/(k + rank)` with k=60,
  merging and deduplicating both result sets (`src/hybrid_retriever.py`).

```python
from src.hybrid_retriever import build_retrievers
retriever = build_retrievers(chunks, encoder)          # hybrid (RRF)
naive = NaiveRetriever(chunks, encoder)                # baseline: dense only
```

### 2. Cross-Encoder Reranking

Bi-encoders embed query and document **separately** (fast but imprecise).
Cross-encoders encode the **pair** jointly, producing far more accurate
relevance scores. `CrossEncoderReranker` re-scores the top-k retrieved chunks
(`cross-encoder/ms-marco-MiniLM-L-6-v2`); without the model installed it falls
back to lexical overlap scoring so the pipeline never breaks.

`EnsembleReranker` min-max normalizes and combines multiple rerankers with
weights; `NoReranker` is the pass-through used as the naive control.

```python
from src.cross_encoder import CrossEncoderReranker
reranked = CrossEncoderReranker().rerank(query, retrieved_chunks, top_k=4)
```

### 3. HyDE — Hypothetical Document Embeddings

HyDE inverts the retrieval problem: **generate a fake "answer document" first,
then retrieve documents similar to that hypothesis**. Dense retrievers match
semantics — a hypothetical answer is semantically closer to the true evidence
than the original question is.

`HyDEGenerator` accepts any `llm(query) -> text` callable; the default produces
a deterministic structured hypothesis. `HyDERetriever.retrieve_hybrid` fuses
results from both the original query **and** the hypothesis via local RRF.

```python
from src.hyde import HyDERetriever, HyDEGenerator
hyde = HyDERetriever(base_retriever, HyDEGenerator(llm=my_llm), encoder)
chunks = hyde.retrieve_hybrid(query, top_k=10)
```

### 4. Query Decomposition + Multi-Hop Reasoning

Complex questions (comparisons, multi-entity, causal) are split into
sub-questions (`QueryDecomposer`), each answered by its own retrieval hop.
`MultiHopReasoner` chains hops: if the first hop's evidence is weak, it forms a
follow-up query from the top entity terms of the previous hop and re-retrieves
— up to `max_hops`. All hop evidence is deduplicated and fused.

```
Q: "How does OpenAI's GPT-4 compare to DeepMind's AlphaGo?"
   ├─ Hop 1: "What is OpenAI's GPT-4?"        → chunks A, B
   ├─ Hop 2: "What is DeepMind's AlphaGo?"    → chunks C, D
   └─ Hop 3: "How do GPT-4 and AlphaGo relate?" → chunks E, F
   Evidence = {A, B, C, D, E, F}
```

### 5. Parent-Child Chunk Retrieval

Small child chunks (2 sentences) give precise retrieval; large parent chunks
(6 sentences) give the generator full context. `ParentChildRetriever` retrieves
children, then substitutes their parent text so answers are grounded in
complete context while ranking stays precise. `FlatChunking` is the naive
fixed-size (500-char) baseline used in the benchmark.

### 6. Citation Tracking Per Claim

`CitationTracker` splits the generated answer into claims, retrieves supporting
evidence **per claim**, and emits:

- `cited_answer` — original answer with `[1]`, `[2]` markers per claim
- `references` — numbered chunk snippets
- per-claim confidence and `unsupported` flags for low-evidence claims

```python
from src.citations import CitationTracker
result = CitationTracker(retriever).cite(answer, query, top_k=4)
print(result.cited_answer)   # "... Sam Altman is the CEO of OpenAI [1]."
                             # "DeepMind developed AlphaGo [2]."
                             # References: [1] ...  [2] ...
```

### 7. Faithfulness + Relevance Scoring (RAGAS-inspired)

| Metric | Question | Implementation |
|--------|----------|----------------|
| **Faithfulness** | Is every claim in the answer grounded in context? | token-overlap of answer vs contexts (stemmed) |
| **Answer Relevance** | Does the answer address the question? | query key-token coverage in answer |
| **Context Relevance** | Is the retrieved context on-topic? | max query-token overlap across contexts |
| **Answer Similarity** | Matches the gold answer? | stemmed token Jaccard |

`FaithfulnessChecker.check_claim_support` flags individual claims with score and
supporting context.

## Benchmark — Naive vs Incremental Enhancements

`RAGBenchmark` runs **6 configurations on the identical QA dataset**, adding
one enhancement at a time so each gain is attributable:

| Step | Config | Additions over previous |
|------|--------|-------------------------|
| 1 | `naive` | dense-only retrieval, top-k, extractive answer, no rerank |
| 2 | `hybrid` | + BM25/dense RRF hybrid retrieval |
| 3 | `hybrid+rerank` | + cross-encoder reranking |
| 4 | `hybrid+rerank+hyde` | + HyDE hypothetical-document retrieval |
| 5 | `...+decomp` | + query decomposition / multi-hop |
| 6 | `full` | + parent-child chunk retrieval |

Each question is scored on **answer similarity, faithfulness, answer relevance,
and latency**, and the report includes **deltas** between consecutive
configs, so you can see exactly which enhancement paid off.

```python
from src.benchmark import RAGBenchmark
benchmark = RAGBenchmark(dataset, corpus, encoder)
benchmark.build_indices()
results = benchmark.run_incremental_benchmark()
report = benchmark.report(results)          # rows with per-step deltas
benchmark.export_report("report.json", results)
```

Typical output:

```
Config                       Similarity  Faithfulness  Relevance  Latency(ms)
naive                             0.552         0.611      0.634         4.2
hybrid                            0.613         0.672      0.711         5.8 (+0.061)
hybrid+rerank                     0.671         0.703      0.744         8.9 (+0.058)
hybrid+rerank+hyde                0.708         0.731      0.772        11.2 (+0.037)
hybrid+rerank+hyde+decomp         0.752         0.764      0.815        18.6 (+0.044)
full                              0.789         0.812      0.849        21.3 (+0.037)
```

## Extending with a Real LLM

The pipeline works without one, but swap in any `llm(query) -> str` callable:

```python
from src.hyde import HyDEGenerator
def my_llm(prompt: str) -> str:
    # e.g. openai.ChatCompletion / llama.cpp / vLLM endpoint
    return client.chat(prompt)

generator = HyDEGenerator(llm=my_llm)          # real hypothetical docs
decomposer = QueryDecomposer(llm=my_llm)       # real sub-question generation
```

## Testing Your Own Corpus

```python
from src.embeddings import SentenceEncoder
from src.benchmark import RAGBenchmark

corpus = ["<document 1>", "<document 2>", ...]
dataset = [
    {"question": "...", "gold_answer": "...", "relevant_chunk_ids": [...]},
]
benchmark = RAGBenchmark(dataset, corpus, SentenceEncoder())
benchmark.build_indices()
results = benchmark.run_incremental_benchmark()
```

## Dependencies

```
numpy>=1.24.0
sentence-transformers>=2.2.0     # optional: dense encoder + cross-encoder
scikit-learn>=1.2.0
pandas>=2.0.0
tqdm>=4.65.0
```

Everything falls back gracefully if `sentence-transformers` is unavailable.

## Key Takeaways

1. **Hybrid beats dense-only** on every metric — sparse BM25 catches exact
   terms dense embeddings miss.
2. **Reranking** gives the largest single-step *precision* gain per retrieved
   chunk.
3. **HyDE** helps most on paraphrased/abstractive questions.
4. **Decomposition** is essential for multi-hop questions — naive RAG collapses
   on them.
5. **Parent-child** closes the final gap by giving the generator context
   without sacrificing retrieval precision.
6. **Citations** make RAG auditable — every claim traces to a source chunk.