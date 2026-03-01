"""Hallucination benchmark — naive LLM vs RAG vs multi-agent pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def load_test_set():
    """Load evaluation questions."""
    path = Path("data/eval/test_questions.json")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_naive_llm(query: str) -> list[str]:
    """Naive LLM — no RAG, no verification."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    resp = llm.invoke([HumanMessage(content=query)])
    return [resp.content] if resp.content else []


def run_rag_only(query: str, retriever, reranker, generator) -> list[str]:
    """Standard RAG — retrieve + generate, no verification."""
    chunks = retriever.search(query, top_k=20)
    indices = [r[0] for r in chunks]
    chunks = retriever.get_chunks(indices)
    if not chunks:
        return []
    pairs = [(query, c) for c in chunks]
    scores = reranker.model.predict(pairs)
    if hasattr(scores, "flatten"):
        scores = scores.flatten()
    ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
    evidence = [c for _, c in ranked[:5]]
    answer = generator.generate(query, evidence)
    return [answer]


def run_multi_agent(query: str, pipeline) -> list[str]:
    """Full multi-agent pipeline."""
    result = pipeline.run(query)
    if result.get("status") == "verified" and result.get("answer"):
        return [result["answer"]]
    if result.get("status") == "refused":
        return []  # Refused = no answer
    return []


def compute_metrics(results: list[dict], test_set: list[dict], condition: str = "multi") -> dict:
    """Compute correctness. For multi: verified for answerable, refused for unanswerable."""
    total = len(results)
    if total == 0:
        return {"correct": 0, "total": 0, "correct_rate": 0}

    if condition == "multi" and test_set:
        correct = 0
        for r, q in zip(results, test_set):
            status = r.get("status", "error")
            expected_answerable = q.get("expected_answerable", True)
            if expected_answerable and status == "verified":
                correct += 1
            elif not expected_answerable and status == "refused":
                correct += 1
        return {"total": total, "correct": correct, "correct_rate": correct / total}
    # For naive/rag: just report verified count
    verified = sum(1 for r in results if r.get("status") == "verified")
    return {"total": total, "verified": verified, "correct_rate": verified / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["naive", "rag", "multi", "all"], default="multi")
    parser.add_argument("--threshold", type=float, default=0.05, help="Max hallucination rate to pass")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test queries (0=all)")
    args = parser.parse_args()

    test_set = load_test_set()
    if not test_set:
        print("No test set at data/eval/test_questions.json")
        sys.exit(1)

    if args.limit:
        test_set = test_set[: args.limit]

    queries = [q["query"] for q in test_set]

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        sys.exit(1)

    results = []

    if args.condition in ("multi", "all"):
        print("Running multi-agent pipeline...")
        from src.retrieval import HybridRetriever
        from src.pipeline import RAGPipeline

        corpus_path = Path("data/chunks/corpus.json")
        if not corpus_path.exists():
            print("Run index_documents.py first.")
            sys.exit(1)
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        retriever = HybridRetriever(corpus)
        pipeline = RAGPipeline(retriever)

        for i, q in enumerate(queries):
            try:
                r = pipeline.run(q)
                results.append(r)
                status = r.get("status", "?")
                print(f"  [{i+1}/{len(queries)}] {q[:50]}... -> {status}")
            except Exception as e:
                print(f"  [{i+1}] {q[:50]}... ERROR: {e}")
                results.append({"status": "error", "error": str(e)})

    elif args.condition == "rag":
        print("Running RAG-only...")
        from src.retrieval import HybridRetriever
        from src.agents import RerankerAgent, AnswerGenerator

        with open("data/chunks/corpus.json", encoding="utf-8") as f:
            corpus = json.load(f)
        retriever = HybridRetriever(corpus)
        reranker = RerankerAgent()
        generator = AnswerGenerator()
        for i, q in enumerate(queries):
            try:
                r = run_rag_only(q, retriever, reranker, generator)
                results.append({"status": "verified" if r else "empty", "answer": r[0] if r else None})
                print(f"  [{i+1}/{len(queries)}] {q[:50]}...")
            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
                results.append({"status": "error", "error": str(e)})

    elif args.condition == "naive":
        print("Running naive LLM...")
        for i, q in enumerate(queries):
            try:
                r = run_naive_llm(q)
                results.append({"status": "verified" if r else "empty", "answer": r[0] if r else None})
                print(f"  [{i+1}/{len(queries)}] {q[:50]}...")
            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
                results.append({"status": "error", "error": str(e)})

    metrics = compute_metrics(results, test_set, args.condition)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # CI gate: pass if correct_rate >= (1 - threshold)
    min_rate = 1.0 - args.threshold
    passed = metrics.get("correct_rate", 0) >= min_rate
    if not passed:
        print(f"\nFAIL: correct_rate {metrics.get('correct_rate', 0):.2f} < {min_rate:.2f}")
        sys.exit(1)
    print("\nPASS: Benchmark passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
