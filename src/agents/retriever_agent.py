"""Retriever Agent — Hybrid BM25 + Dense retrieval with RRF."""

from typing import List

from src.retrieval import HybridRetriever


class RetrieverAgent:
    """Runs hybrid retrieval and returns top-K chunks."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 20):
        self.retriever = retriever
        self.top_k = top_k

    def retrieve(self, query: str) -> List[str]:
        """Return top-K chunks for the query."""
        results = self.retriever.search(query, top_k=self.top_k)
        indices = [r[0] for r in results]
        return self.retriever.get_chunks(indices)
