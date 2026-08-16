import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hybrid_retriever import HybridRetriever, NaiveRetriever, RetrievedChunk, build_retrievers
from .cross_encoder import CrossEncoderReranker, NoReranker
from .hyde import HyDERetriever, HyDEGenerator
from .query_decomposition import MultiHopReasoner
from .parent_child_retrieval import ChunkingStrategy, ParentChildRetriever, FlatChunking
from .scoring import RAGEvaluator, FaithfulnessChecker


@dataclass
class PipelineConfig:
    """Configuration for a RAG pipeline variant."""
    name: str
    use_hybrid: bool = False
    use_rerank: bool = False
    use_hyde: bool = False
    use_decomposition: bool = False
    use_parent_child: bool = False


@dataclass
class PipelineResult:
    """Results for a single pipeline configuration."""
    config: PipelineConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    per_question: List[Dict] = field(default_factory=list)


class AnswerGenerator:
    """Extractive answer generation from retrieved chunks."""

    @staticmethod
    def generate(query: str, chunks: List[RetrievedChunk], max_sentences: int = 3) -> str:
        q_tokens = set(t.lower() for t in query.split() if len(t) > 2)
        scored_sentences = []
        for chunk in chunks:
            for sentence in chunk.chunk.text.replace('\n', ' ').split('.'):
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_tokens = set(t.lower() for t in sentence.split())
                overlap = len(q_tokens.intersection(s_tokens))
                if overlap > 0:
                    scored_sentences.append((overlap, sentence))
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return '. '.join(s for _, s in scored_sentences[:max_sentences]) + ('.' if scored_sentences else "I could not find a definitive answer.")


class RAGBenchmark:
    """Incremental benchmark: naive RAG vs each enhancement."""

    def __init__(self, dataset: List[dict], corpus: List[str], encoder, reranker=None):
        self.dataset = dataset
        self.corpus = corpus
        self.encoder = encoder
        self.reranker = reranker or CrossEncoderReranker()
        self.evaluator = RAGEvaluator()
        self.retriever = None
        self.pc_retriever = None

    def build_indices(self):
        flat = FlatChunking()
        self.flat_chunks = [flat.create_chunks(doc, f"doc{i}") for i, doc in enumerate(self.corpus)]
        self.flat_chunks = [c for cl in self.flat_chunks for c in cl]

        retrievers = build_retrievers(self.flat_chunks, self.encoder)
        self.retrievers = retrievers

        strategy = ChunkingStrategy()
        self.parent_chunks, self.child_chunks = [], []
        for i, doc in enumerate(self.corpus):
            self.parent_chunks.extend(strategy.create_parent_chunks(doc, f"doc{i}"))
            self.child_chunks.extend(strategy.create_chunks(doc, f"doc{i}"))

        child_index = build_retrievers(self.child_chunks, self.encoder)
        self.pc_index = {"parents": self.parent_chunks, "children": self.child_chunks,
                         "retriever": child_index}

    def _make_pc_retriever(self, retriever, index):
        class _Wrapper:
            def __init__(self, r, parents, children):
                self.r = r
                self.parents = parents
                self.children = children
            def retrieve(self, query, top_k=5):
                child_results = self.r.retrieve(query, top_k=top_k)
                out = []
                for rc in child_results:
                    parent = next((p for p in self.parents if p.id == rc.chunk.parent_id), None)
                    if parent is not None:
                        new_chunk = type(rc.chunk)(id=parent.id, text=parent.text,
                                                   metadata={"child_id": rc.chunk.id, "parent_id": parent.id, "strategy": "parent-child"}, parent_id=None)
                        out.append(RetrievedChunk(chunk=new_chunk, score=rc.score,
                                                  sparse_score=rc.sparse_score, dense_score=rc.dense_score, rank=rc.rank))
                    else:
                        out.append(rc)
                return out
        return _Wrapper(retriever, index["parents"], index["children"])

    def _get_retriever(self, config: PipelineConfig, query: str):
        base = self.retrievers["naive"] if not config.use_hybrid else self.retrievers["hybrid"]
        if config.use_parent_child:
            child_retriever = self.pc_index["retriever"]["hybrid"] if config.use_hybrid else self.pc_index["retriever"]["naive"]
            base = self._make_pc_retriever(child_retriever, self.pc_index)
        if config.use_hyde:
            base = HyDERetriever(base, HyDEGenerator(), self.encoder)
        if config.use_decomposition:
            base = MultiHopReasoner(base)
        return base

    def run_pipeline(self, config: PipelineConfig, question: str) -> Dict:
        start = time.time()
        retriever = self._get_retriever(config, question)
        chunks = retriever.retrieve(question, top_k=6)
        if config.use_rerank:
            chunks = self.reranker.rerank(question, chunks, top_k=4)
        contexts = [c.chunk.text for c in chunks]
        answer = AnswerGenerator.generate(question, chunks)
        latency = (time.time() - start) * 1000
        return {"answer": answer, "contexts": contexts, "latency_ms": latency}

    def run_incremental_benchmark(self) -> List[PipelineResult]:
        configs = [
            PipelineConfig("naive", use_hybrid=False, use_rerank=False),
            PipelineConfig("hybrid", use_hybrid=True),
            PipelineConfig("hybrid+rerank", use_hybrid=True, use_rerank=True),
            PipelineConfig("hybrid+rerank+hyde", use_hybrid=True, use_rerank=True, use_hyde=True),
            PipelineConfig("hybrid+rerank+hyde+decomp", use_hybrid=True, use_rerank=True, use_hyde=True, use_decomposition=True),
            PipelineConfig("full", use_hybrid=True, use_rerank=True, use_hyde=True, use_decomposition=True, use_parent_child=True),
        ]
        results = []
        for cfg in configs:
            per_q = []
            for row in self.dataset:
                out = self.run_pipeline(cfg, row["question"])
                score = self.evaluator.evaluate_sample(row["question"], out["answer"], out["contexts"], row.get("gold_answer"))
                per_q.append({
                    "question": row["question"],
                    "gold": row.get("gold_answer", ""),
                    "predicted": out["answer"],
                    "faithfulness": score.faithfulness,
                    "answer_relevance": score.answer_relevance,
                    "answer_similarity": score.answer_similarity,
                    "latency_ms": out["latency_ms"],
                })
            metrics = {
                "answer_similarity": sum(r["answer_similarity"] for r in per_q) / len(per_q),
                "faithfulness": sum(r["faithfulness"] for r in per_q) / len(per_q),
                "answer_relevance": sum(r["answer_relevance"] for r in per_q) / len(per_q),
                "latency_ms": sum(r["latency_ms"] for r in per_q) / len(per_q),
            }
            results.append(PipelineResult(config=cfg, metrics=metrics, per_question=per_q))
        return results

    def report(self, results: List[PipelineResult]) -> dict:
        report_rows = []
        prev = None
        for r in results:
            row = {"config": r.config.name, **r.metrics}
            if prev is not None:
                row["delta_similarity"] = r.metrics["answer_similarity"] - prev.metrics["answer_similarity"]
                row["delta_faithfulness"] = r.metrics["faithfulness"] - prev.metrics["faithfulness"]
            prev = r
            report_rows.append(row)
        return {"rows": report_rows}

    def export_report(self, path: str, results: List[PipelineResult]):
        with open(path, "w") as f:
            json.dump(self.report(results), f, indent=2)