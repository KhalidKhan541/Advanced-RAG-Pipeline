import logging

from src.embeddings import SentenceEncoder, BM25Index, DenseIndex, Chunk
from src.hybrid_retriever import build_retrievers
from src.cross_encoder import CrossEncoderReranker
from src.benchmark import RAGBenchmark
from src.citations import CitationTracker

logging.basicConfig(level=logging.INFO)

CORPUS = [
    "Google acquired DeepMind in 2014 to accelerate artificial intelligence research.",
    "DeepMind developed AlphaGo, which defeated the world champion Go player in 2016.",
    "OpenAI was founded in December 2015 as a non-profit AI research laboratory.",
    "OpenAI released GPT-3 in 2020 with 175 billion parameters.",
    "The Transformer architecture was introduced in the 2017 paper Attention Is All You Need.",
    "BERT, released by Google in 2018, uses bidirectional training on masked language models.",
    "Reinforcement learning from human feedback, or RLHF, aligns language models with human preferences.",
    "Retrieval augmented generation, or RAG, combines parametric knowledge with non-parametric retrieval.",
    "Embeddings map text into dense vector spaces where similar meanings are close together.",
    "BM25 is a sparse retrieval algorithm based on term frequency and document length.",
    "HyDE generates hypothetical documents to improve retrieval for dense encoders.",
    "Cross-encoders jointly encode a query and document pair to produce precise relevance scores.",
    "Multi-hop reasoning requires connecting facts across multiple documents to answer a question.",
    "Parent-child chunking retrieves small precise chunks but provides larger context for generation.",
    "Citogenesis risk in RAG is mitigated by tracking citations for each generated claim.",
    "RAGAS metrics evaluate faithfulness and answer relevance of RAG systems.",
    "GPT-4, released in March 2023, demonstrated strong multi-modal reasoning abilities.",
    "Sam Altman is the CEO of OpenAI and has advocated for careful AI deployment.",
    "Demis Hassabis co-founded DeepMind and led AlphaGo development.",
    "Sentence-transformers provides efficient bi-encoder models for semantic similarity search.",
]

DATASET = [
    {"question": "Which company acquired DeepMind?", "gold_answer": "Google"},
    {"question": "What did DeepMind develop that defeated a Go champion?", "gold_answer": "AlphaGo"},
    {"question": "When was OpenAI founded?", "gold_answer": "December 2015"},
    {"question": "What technique aligns language models with human preferences?", "gold_answer": "RLHF"},
    {"question": "Who is the CEO of OpenAI?", "gold_answer": "Sam Altman"},
    {"question": "Which architecture was introduced in 2017?", "gold_answer": "Transformer"},
    {"question": "What is the difference between sparse and dense retrieval?", "gold_answer": "BM25 is sparse, embeddings are dense"},
    {"question": "What is multi-hop reasoning in RAG?", "gold_answer": "Connecting facts across documents"},
]


def main():
    print("=" * 60)
    print("ADVANCED RAG PIPELINE - NAIVE vs ADVANCED BENCHMARK")
    print("=" * 60)

    encoder = SentenceEncoder()
    benchmark = RAGBenchmark(DATASET, CORPUS, encoder)
    benchmark.build_indices()
    results = benchmark.run_incremental_benchmark()
    report = benchmark.report(results)

    print("\n--- Benchmark Results ---")
    header = f"{'Config':<26}{'Similarity':>12}{'Faithfulness':>14}{'Relevance':>12}{'Latency(ms)':>12}"
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        delta = f" (+{row.get('delta_similarity', 0):.3f})" if row.get("delta_similarity") is not None else ""
        print(f"{row['config']:<26}{row['answer_similarity']:>12.3f}{row['faithfulness']:>14.3f}{row['answer_relevance']:>12.3f}{row['latency_ms']:>12.1f}{delta}")

    print("\n--- Citation Tracking Demo ---")
    from src.parent_child_retrieval import FlatChunking
    flat_chunks = []
    for i, doc in enumerate(CORPUS):
        flat_chunks.extend(FlatChunking().create_chunks(doc, f"doc{i}"))
    simple_retriever = build_retrievers(flat_chunks, encoder)["hybrid"]
    tracker = CitationTracker(simple_retriever)
    q = "Who is the CEO of OpenAI and what did DeepMind develop?"
    ans = "Sam Altman is the CEO of OpenAI. DeepMind developed AlphaGo."
    result = tracker.cite(ans, q, top_k=4)
    print(result.cited_answer)

    print("\n--- Verdict ---")
    rows = report["rows"]
    best = max(rows, key=lambda r: r["answer_similarity"])
    print(f"Biggest gain: {best['config']} with similarity {best['answer_similarity']:.3f}")


if __name__ == "__main__":
    main()