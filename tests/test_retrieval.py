"""Unit tests for retrieval components."""

import pytest
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from src.retrieval.hybrid import reciprocal_rank_fusion


def test_bm25_search():
    corpus = ["hello world", "foo bar", "hello foo", "world of warcraft"]
    r = BM25Retriever(corpus)
    results = r.search("hello", top_k=2)
    assert len(results) == 2
    assert results[0][0] in [0, 2]  # indices of docs containing "hello"


def test_rrf_merge():
    list1 = [(0, 1.0), (1, 0.9), (2, 0.8)]
    list2 = [(1, 1.0), (2, 0.9), (0, 0.8)]
    merged = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(merged) == 3
    # doc 1 appears in both with high rank
    doc_ids = [m[0] for m in merged]
    assert doc_ids == [1, 0, 2] or doc_ids == [0, 1, 2]


@pytest.mark.slow
def test_hybrid_retriever():
    """Requires sentence-transformers model download."""
    corpus = ["retriever agent runs hybrid search", "BM25 and dense", "reciprocal rank fusion"]
    r = HybridRetriever(corpus, rrf_k=60)
    results = r.search("hybrid search", top_k=2)
    assert len(results) <= 2
    chunks = r.get_chunks([results[0][0]])
    assert "hybrid" in chunks[0].lower() or "retriever" in chunks[0].lower()
