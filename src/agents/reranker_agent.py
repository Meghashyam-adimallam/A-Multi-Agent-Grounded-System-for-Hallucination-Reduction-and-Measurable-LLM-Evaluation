"""Reranker Agent — Cross-encoder re-ranking of retrieved chunks."""

from typing import List

from sentence_transformers import CrossEncoder


class RerankerAgent:
    """Re-ranks chunks by relevance using a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 5):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, chunks: List[str]) -> List[str]:
        """Return top-N chunks after cross-encoder re-ranking."""
        if not chunks:
            return []
        pairs = [(query, chunk) for chunk in chunks]
        scores = self.model.predict(pairs)
        # Handle numpy/scalar scores
        if hasattr(scores, "flatten"):
            scores = scores.flatten()
        ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        return [chunk for _, chunk in ranked[: self.top_n]]
