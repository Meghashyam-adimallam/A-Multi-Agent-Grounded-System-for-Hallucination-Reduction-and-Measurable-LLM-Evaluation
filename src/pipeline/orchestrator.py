"""Pipeline orchestrator — coordinates all 6 agents end-to-end."""

import time
from typing import Any

from src.agents import (
    RetrieverAgent,
    RerankerAgent,
    AnswerGenerator,
    NaiveAnswerGenerator,
    ClaimDecomposer,
    VerificationAgent,
    ConfidenceAgent,
)
from src.retrieval import HybridRetriever


def _maybe_trace(name: str, fn, *args, **kwargs):
    """Run fn, optionally wrapped in Langfuse span."""
    try:
        from langfuse import Langfuse
        import os
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            lf = Langfuse()
            with lf.span(name=name) as span:
                result = fn(*args, **kwargs)
                span.end()
                return result
    except Exception:
        pass
    return fn(*args, **kwargs)


class RAGPipeline:
    """End-to-end multi-agent grounded RAG pipeline."""

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 20,
        top_n: int = 5,
        confidence_threshold: float = 0.70,
        max_retries: int = 2,
    ):
        self.retriever_agent = RetrieverAgent(retriever, top_k=top_k)
        self.reranker_agent = RerankerAgent(top_n=top_n)
        self.answer_generator = AnswerGenerator()
        self.naive_answer_generator = NaiveAnswerGenerator()
        self.claim_decomposer = ClaimDecomposer()
        self.verification_agent = VerificationAgent()
        self.confidence_agent = ConfidenceAgent(
            threshold=confidence_threshold,
            max_retries=max_retries,
        )

    def run(self, query: str) -> dict[str, Any]:
        """
        Run full pipeline. Returns verified answer or structured refusal.
        """
        t_total_start = time.perf_counter()
        retry_count = 0
        evidence_chunks: list[str] = []
        agent_latencies: dict[str, float] = {}

        while retry_count <= self.confidence_agent.max_retries:
            # 1. Retrieve
            t0 = time.perf_counter()
            chunks = self.retriever_agent.retrieve(query)
            agent_latencies["retriever"] = time.perf_counter() - t0

            if not chunks:
                total_latency = time.perf_counter() - t_total_start
                return self._refusal(
                    query,
                    0.0,
                    [],
                    [],
                    "insufficient_evidence",
                    agent_latencies,
                    retry_count,
                    total_latency,
                )

            # 2. Rerank
            t0 = time.perf_counter()
            evidence_chunks = self.reranker_agent.rerank(query, chunks)
            agent_latencies["reranker"] = time.perf_counter() - t0

            # 3. Generate
            t0 = time.perf_counter()
            answer = self.answer_generator.generate(query, evidence_chunks)
            agent_latencies["answer_generator"] = time.perf_counter() - t0

            # 4. Decompose
            t0 = time.perf_counter()
            claims = self.claim_decomposer.decompose(answer)
            agent_latencies["claim_decomposer"] = time.perf_counter() - t0

            # Cap claims to bound verification time (Verified RAG speed optimization)
            max_claims = 3
            if len(claims) > max_claims:
                claims = claims[:max_claims]

            if not claims:
                return {
                    "answer": answer,
                    "confidence_score": 1.0,
                    "claims": [],
                    "verdicts": [],
                    "status": "verified",
                    "agent_latencies": agent_latencies,
                    "evidence_found": evidence_chunks[:3] if evidence_chunks else [],
                    "retries": retry_count,
                    "re_retrieval_triggered": retry_count > 0,
                    "total_latency": time.perf_counter() - t_total_start,
                }

            # 5. Verify
            t0 = time.perf_counter()
            verdicts = self.verification_agent.verify_all(claims, evidence_chunks)
            agent_latencies["verification_agent"] = time.perf_counter() - t0

            # 6. Confidence
            verdict_labels = [v[0] for v in verdicts]
            claim_evidence = [v[1] for v in verdicts]
            support_ratio = self.confidence_agent.compute_support_ratio(verdict_labels)

            if support_ratio >= self.confidence_agent.threshold:
                return {
                    "answer": answer,
                    "confidence_score": support_ratio,
                    "claims": claims,
                    "verdicts": verdict_labels,
                    "status": "verified",
                    "agent_latencies": agent_latencies,
                    "evidence_found": evidence_chunks[:3] if evidence_chunks else [],
                    "claim_evidence": claim_evidence,
                    "retries": retry_count,
                    "re_retrieval_triggered": retry_count > 0,
                    "total_latency": time.perf_counter() - t_total_start,
                }

            if self.confidence_agent.should_retry(support_ratio, retry_count):
                retry_count += 1
                continue

            unverified = [
                c for c, v in zip(claims, verdicts) if v[0] != "SUPPORTED"
            ]
            total_latency = time.perf_counter() - t_total_start
            return self._refusal(
                query,
                support_ratio,
                unverified,
                evidence_chunks[:3],
                "max_retries_exceeded",
                agent_latencies,
                retry_count,
                total_latency,
            )

        total_latency = time.perf_counter() - t_total_start
        return self._refusal(
            query,
            0.0,
            [],
            evidence_chunks[:3] if evidence_chunks else [],
            "max_retries_exceeded",
            agent_latencies,
            retry_count,
            total_latency,
        )

    def run_standard_rag(self, query: str) -> dict[str, Any]:
        """
        Standard RAG baseline: single-pass retrieve → rerank → generate → decompose → verify.
        No re-retrieval loop, no structured refusal — always returns an answer plus scores.
        """
        t_total_start = time.perf_counter()
        agent_latencies: dict[str, float] = {}

        # Retrieve
        t0 = time.perf_counter()
        chunks = self.retriever_agent.retrieve(query)
        agent_latencies["retriever"] = time.perf_counter() - t0
        if not chunks:
            total_latency = time.perf_counter() - t_total_start
            return {
                "answer": "No relevant evidence found in the uploaded documents.",
                "confidence_score": 0.0,
                "claims": [],
                "verdicts": [],
                "status": "rag",
                "agent_latencies": agent_latencies,
                "evidence_found": [],
                "claim_evidence": [],
                "retries": 0,
                "re_retrieval_triggered": False,
                "total_latency": total_latency,
            }

        # Rerank
        t0 = time.perf_counter()
        evidence_chunks = self.reranker_agent.rerank(query, chunks)
        agent_latencies["reranker"] = time.perf_counter() - t0

        # Generate
        t0 = time.perf_counter()
        answer = self.answer_generator.generate(query, evidence_chunks)
        agent_latencies["answer_generator"] = time.perf_counter() - t0

        # Decompose
        t0 = time.perf_counter()
        claims = self.claim_decomposer.decompose(answer)
        agent_latencies["claim_decomposer"] = time.perf_counter() - t0

        if not claims:
            total_latency = time.perf_counter() - t_total_start
            return {
                "answer": answer,
                "confidence_score": 1.0,
                "claims": [],
                "verdicts": [],
                "status": "rag",
                "agent_latencies": agent_latencies,
                "evidence_found": evidence_chunks[:3] if evidence_chunks else [],
                "claim_evidence": [],
                "retries": 0,
                "re_retrieval_triggered": False,
                "total_latency": total_latency,
            }

        # Verify
        t0 = time.perf_counter()
        verdicts = self.verification_agent.verify_all(claims, evidence_chunks)
        agent_latencies["verification_agent"] = time.perf_counter() - t0

        verdict_labels = [v[0] for v in verdicts]
        claim_evidence = [v[1] for v in verdicts]
        support_ratio = self.confidence_agent.compute_support_ratio(verdict_labels)

        total_latency = time.perf_counter() - t_total_start
        return {
            "answer": answer,
            "confidence_score": support_ratio,
            "claims": claims,
            "verdicts": verdict_labels,
            "status": "rag",
            "agent_latencies": agent_latencies,
            "evidence_found": evidence_chunks[:3] if evidence_chunks else [],
            "claim_evidence": claim_evidence,
            "retries": 0,
            "re_retrieval_triggered": False,
            "total_latency": total_latency,
        }

    def run_fast_rag(self, query: str) -> dict[str, Any]:
        """
        Fast RAG: retrieve → rerank → generate only.
        No claim decomposition, no verification — optimized for speed / UX.
        """
        t_total_start = time.perf_counter()
        agent_latencies: dict[str, float] = {}

        # Retrieve
        t0 = time.perf_counter()
        chunks = self.retriever_agent.retrieve(query)
        agent_latencies["retriever"] = time.perf_counter() - t0
        if not chunks:
            total_latency = time.perf_counter() - t_total_start
            return {
                "answer": "No relevant evidence found in the uploaded documents.",
                "confidence_score": None,
                "claims": [],
                "verdicts": [],
                "status": "fast_rag",
                "agent_latencies": agent_latencies,
                "evidence_found": [],
                "claim_evidence": [],
                "retries": 0,
                "re_retrieval_triggered": False,
                "total_latency": total_latency,
            }

        # Rerank
        t0 = time.perf_counter()
        evidence_chunks = self.reranker_agent.rerank(query, chunks)
        agent_latencies["reranker"] = time.perf_counter() - t0

        # Generate grounded answer
        t0 = time.perf_counter()
        answer = self.answer_generator.generate(query, evidence_chunks)
        agent_latencies["answer_generator"] = time.perf_counter() - t0

        total_latency = time.perf_counter() - t_total_start
        return {
            "answer": answer,
            "confidence_score": None,
            "claims": [],
            "verdicts": [],
            "status": "fast_rag",
            "agent_latencies": agent_latencies,
            "evidence_found": evidence_chunks[:3] if evidence_chunks else [],
            "claim_evidence": [],
            "retries": 0,
            "re_retrieval_triggered": False,
            "total_latency": total_latency,
        }

    def run_naive(self, query: str) -> dict[str, Any]:
        """
        Naive LLM baseline: generate answer without grounding, then evaluate claims
        against retrieved evidence using the same verification logic.
        """
        t_total_start = time.perf_counter()
        agent_latencies: dict[str, float] = {}

        # Generate naive answer (no retrieved context).
        t0 = time.perf_counter()
        answer = self.naive_answer_generator.generate(query)
        agent_latencies["answer_generator_naive"] = time.perf_counter() - t0

        # Decompose into claims.
        t0 = time.perf_counter()
        claims = self.claim_decomposer.decompose(answer)
        agent_latencies["claim_decomposer"] = time.perf_counter() - t0

        if not claims:
            total_latency = time.perf_counter() - t_total_start
            return {
                "answer": answer,
                "confidence_score": 1.0,
                "claims": [],
                "verdicts": [],
                "status": "naive",
                "agent_latencies": agent_latencies,
                "evidence_found": [],
                "claim_evidence": [],
                "retries": 0,
                "re_retrieval_triggered": False,
                "total_latency": total_latency,
            }

        # Retrieve and rerank evidence only for evaluation (not for generation).
        t0 = time.perf_counter()
        chunks = self.retriever_agent.retrieve(query)
        agent_latencies["retriever"] = time.perf_counter() - t0

        if not chunks:
            total_latency = time.perf_counter() - t_total_start
            return {
                "answer": answer,
                "confidence_score": 0.0,
                "claims": claims,
                "verdicts": ["UNVERIFIED"] * len(claims),
                "status": "naive",
                "agent_latencies": agent_latencies,
                "evidence_found": [],
                "claim_evidence": [None] * len(claims),
                "retries": 0,
                "re_retrieval_triggered": False,
                "total_latency": total_latency,
            }

        t0 = time.perf_counter()
        evidence_chunks = self.reranker_agent.rerank(query, chunks)
        agent_latencies["reranker"] = time.perf_counter() - t0

        # Verify claims against evidence.
        t0 = time.perf_counter()
        verdicts = self.verification_agent.verify_all(claims, evidence_chunks)
        agent_latencies["verification_agent"] = time.perf_counter() - t0

        verdict_labels = [v[0] for v in verdicts]
        claim_evidence = [v[1] for v in verdicts]
        support_ratio = self.confidence_agent.compute_support_ratio(verdict_labels)

        total_latency = time.perf_counter() - t_total_start
        return {
            "answer": answer,
            "confidence_score": support_ratio,
            "claims": claims,
            "verdicts": verdict_labels,
            "status": "naive",
            "agent_latencies": agent_latencies,
            "evidence_found": evidence_chunks[:3] if evidence_chunks else [],
            "claim_evidence": claim_evidence,
            "retries": 0,
            "re_retrieval_triggered": False,
            "total_latency": total_latency,
        }

    def _refusal(
        self,
        query: str,
        support_ratio: float,
        unverified: list,
        evidence: list,
        reason: str,
        latencies: dict,
        retries: int,
        total_latency: float,
    ) -> dict:
        r = self.confidence_agent.build_refusal(
            query=query,
            support_ratio=support_ratio,
            unverified_claims=unverified,
            evidence_found=evidence,
            reason=reason,
        )
        r["agent_latencies"] = latencies
        r["retries"] = retries
        r["re_retrieval_triggered"] = retries > 0
        r["total_latency"] = total_latency
        return r
