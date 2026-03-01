"""Dense vector retrieval with FAISS."""

from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Dense embedding retriever using FAISS."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str | Path | None = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] | None = None
        self.index = None
        self.index_path = Path(index_path) if index_path else None

    def index_documents(self, corpus: List[str]):
        """Build FAISS index from corpus."""
        self.corpus = corpus
        embeddings = self.model.encode(corpus, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 20) -> List[tuple[int, float]]:
        """Return (index, score) pairs for top-k documents."""
        if self.index is None:
            raise RuntimeError("Call index_documents first")
        q_emb = self.model.encode([query])
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
