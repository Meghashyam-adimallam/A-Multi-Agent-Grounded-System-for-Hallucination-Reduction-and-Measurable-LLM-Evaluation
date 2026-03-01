"""Claim Decomposer — Breaks answers into atomic verifiable claims."""

import json
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

DECOMPOSITION_PROMPT = """Break the following answer into atomic claims. Each claim must be:
1. A single, independently verifiable fact
2. Checkable against evidence without needing other claims
3. Short and unambiguous

Return a JSON array of strings, e.g. ["Claim 1", "Claim 2", "Claim 3"].
No other text — only the JSON array."""


class ClaimDecomposer:
    """Extracts atomic claims from generated answers."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def decompose(self, answer: str) -> List[str]:
        """Return list of atomic claims."""
        try:
            messages = [
                SystemMessage(content=DECOMPOSITION_PROMPT),
                HumanMessage(content=answer),
            ]
            response = self.llm.invoke(messages)
            text = response.content.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
            claims = json.loads(text)
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return claims
            return [answer]  # fallback
        except (json.JSONDecodeError, TypeError):
            return [answer]  # fallback to single claim
