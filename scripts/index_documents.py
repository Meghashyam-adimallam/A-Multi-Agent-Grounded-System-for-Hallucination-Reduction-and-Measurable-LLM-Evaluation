"""Index documents into the hybrid retriever. Run once before using the API."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import load_documents, chunk_documents
from src.retrieval import HybridRetriever
import json


def main():
    docs_path = Path("data/documents")
    chunks_path = Path("data/chunks")
    chunks_path.mkdir(parents=True, exist_ok=True)

    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        print("Created data/documents. Add PDF/TXT/MD files and run again.")
        return

    print("Loading documents...")
    docs = load_documents(docs_path)
    if not docs:
        print("No documents found in data/documents.")
        return

    print(f"Chunking {len(docs)} documents...")
    chunks = chunk_documents(docs)
    corpus = [c.page_content for c in chunks]
    print(f"Created {len(corpus)} chunks.")

    print("Building hybrid index (BM25 + dense)...")
    retriever = HybridRetriever(corpus)
    print("Index built.")

    # Save corpus for API to load
    with open(chunks_path / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved corpus to {chunks_path / 'corpus.json'}")


if __name__ == "__main__":
    main()
