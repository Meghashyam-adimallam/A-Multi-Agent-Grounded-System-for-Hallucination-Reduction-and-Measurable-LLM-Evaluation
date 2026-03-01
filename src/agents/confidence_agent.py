"""Confidence Agent — Support ratio, re-retrieval, structured refusal."""

from typing import List, Literal

from .verification_agent import Verdict

RefusalReason = Literal["insufficient_evidence", "max_retries_exceeded"]


class ConfidenceAgent:
    """Aggregates verdicts, triggers re-retrieval, or produces structured refusal."""

    def __init__(self, threshold: float = 0.70, max_retries: int = 2):
        self.threshold = threshold
        self.max_retries = max_retries

    def compute_support_ratio(self, verdicts: List[Verdict]) -> float:
        """Support ratio = SUPPORTED / total claims."""
        if not verdicts:
            return 0.0
        supported = sum(1 for v in verdicts if v == "SUPPORTED")
        return supported / len(verdicts)

    def should_retry(self, support_ratio: float, retry_count: int) -> bool:
        """Trigger re-retrieval if below threshold and retries remain."""
        return support_ratio < self.threshold and retry_count < self.max_retries

    def build_refusal(
        self,
        query: str,
        support_ratio: float,
        unverified_claims: List[str],
        evidence_found: List[str],
        reason: RefusalReason,
    ) -> dict:
        """Structured refusal JSON."""
        return {
            "status": "refused",
            "query": query,
            "answer": None,
            "confidence_score": support_ratio,
            "claims": None,
            "verdicts": None,
            "unverified_claims": unverified_claims,
            "evidence_found": evidence_found,
            "refusal_reason": reason,
            "explanation": (
                "Insufficient evidence to verify all claims. "
                "Cannot provide a grounded answer."
                if reason == "insufficient_evidence"
                else "Maximum re-retrieval attempts reached. Evidence remains insufficient."
            ),
        }
