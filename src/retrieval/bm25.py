"""BM25 keyword-based retrieval."""

from typing import List

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 retriever for exact-match and keyword queries."""

    def __init__(self, corpus: List[str], tokenizer=None):
        if tokenizer is None:
            tokenizer = lambda s: s.lower().split()
        self.tokenizer = tokenizer
        tokenized = [tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus

    def search(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        """Return (index, score) pairs for top-k documents."""
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
