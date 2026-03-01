"""Verification Agent — NLI-based claim verification with strict thresholds."""

from typing import List, Literal

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from numpy.linalg import norm

Verdict = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


class VerificationAgent:
    """
    Verifies each claim against evidence using NLI with strict rules:

    - For each claim/evidence pair, run NLI (premise=evidence, hypothesis=claim).
    - IF entailment_prob > 0.80 AND cosine_sim >= 0.50 → SUPPORTED.
    - ELIF contradiction_prob > 0.80 AND cosine_sim >= 0.50 → CONTRADICTED.
    - ELSE → UNVERIFIED.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        entail_threshold: float = 0.80,
        contra_threshold: float = 0.80,
        sim_threshold: float = 0.50,
    ):
        self.model = CrossEncoder(model_name)
        self.embedder = SentenceTransformer(embed_model_name)
        self.entail_threshold = entail_threshold
        self.contra_threshold = contra_threshold
        self.sim_threshold = sim_threshold

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(norm(a) * norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def verify(
        self,
        claim: str,
        evidence_chunks: List[str],
        evidence_embeddings: np.ndarray | None = None,
    ) -> tuple[Verdict, str | None]:
        """
        Verify claim against evidence. Returns (verdict, best_evidence_chunk).
        NLI: premise=evidence, hypothesis=claim.
        If evidence_embeddings is provided (e.g. from verify_all), reuse to avoid re-encoding.
        """
        if not evidence_chunks:
            return "UNVERIFIED", None

        # Compute NLI probabilities for each (evidence, claim) pair.
        pairs = [(chunk, claim) for chunk in evidence_chunks]
        # scores shape: (n, 3) with [contradiction, neutral, entailment]
        scores = self.model.predict(pairs, apply_softmax=True)

        # Reuse precomputed evidence embeddings when provided (verify_all passes them).
        if evidence_embeddings is not None and len(evidence_embeddings) == len(evidence_chunks):
            evid_embs = evidence_embeddings
        else:
            evid_embs = self.embedder.encode(evidence_chunks)
        claim_emb = self.embedder.encode([claim])[0]

        best_contra_idx = None
        best_contra_prob = 0.0

        # First pass: look for strong entailment with sufficient similarity.
        for i, prob_vec in enumerate(scores):
            prob_contra = float(prob_vec[0])
            prob_entail = float(prob_vec[2])
            sim = self._cosine_sim(claim_emb, evid_embs[i])

            if sim < self.sim_threshold:
                continue

            if prob_entail >= self.entail_threshold:
                return "SUPPORTED", evidence_chunks[i]

            if prob_contra >= self.contra_threshold and prob_contra > best_contra_prob:
                best_contra_prob = prob_contra
                best_contra_idx = i

        if best_contra_idx is not None:
            return "CONTRADICTED", evidence_chunks[best_contra_idx]

        return "UNVERIFIED", None

    def verify_all(
        self, claims: List[str], evidence_chunks: List[str]
    ) -> List[tuple[Verdict, str | None]]:
        """Verify each claim. Returns list of (verdict, best_chunk). Evidence embeddings computed once and reused."""
        if not evidence_chunks:
            return [("UNVERIFIED", None)] * len(claims)
        evidence_embeddings = self.embedder.encode(evidence_chunks)
        return [
            self.verify(c, evidence_chunks, evidence_embeddings=evidence_embeddings)
            for c in claims
        ]
