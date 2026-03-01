"""Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion."""

from typing import List

from .bm25 import BM25Retriever
from .dense import DenseRetriever


def reciprocal_rank_fusion(
    ranked_lists: List[List[tuple[int, float]]],
    k: int = 60,
) -> List[tuple[int, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    RRF_score(d) = sum over all lists of 1 / (k + rank(d))
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


class HybridRetriever:
    """Combines BM25 and dense retrieval via Reciprocal Rank Fusion."""

    def __init__(
        self,
        corpus: List[str],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        self.corpus = corpus
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(model_name=embedding_model)
        self.dense.index_documents(corpus)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        """Run BM25 and dense in parallel, merge with RRF, return top-K."""
        bm25_results = self.bm25.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        return fused[:top_k]

    def get_chunks(self, indices: List[int]) -> List[str]:
        """Return chunk texts for given indices."""
        return [self.corpus[i] for i in indices]
