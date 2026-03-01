"""Answer Generator — Grounded response with inline citations."""

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

CITATION_SYSTEM_PROMPT = """You are a factual assistant. You must ONLY use information from the provided evidence chunks to answer the question.

RULES:
1. Every factual claim MUST cite a source chunk using [1], [2], etc. corresponding to the chunk index.
2. Do NOT add information from your training data. If the evidence does not support a claim, do not make it.
3. If the evidence is insufficient to answer, say so clearly.
4. If the user asks multiple sub-questions in one message (for example, numbered lines like "1. ...", "2. ..."), answer EACH sub-question separately.
   - Prefix each answer with the same number and a colon, e.g. "1: ...", "2: ...".
   - Put a blank line between answers so they are easy to read.
5. Be concise and precise.
6. Use temperature 0 — no creativity."""

NAIVE_SYSTEM_PROMPT = """You are a helpful general-purpose assistant.

Answer the user's question using your own knowledge.
If the user asks multiple sub-questions in one message (for example, numbered lines like "1. ...", "2. ..."), answer EACH sub-question separately:
- Prefix each answer with the same number and a colon, e.g. "1: ...", "2: ...".
- Put a blank line between answers so they are easy to read.
You do not need to use citations.
Be clear, precise, and avoid speculation."""


class AnswerGenerator:
    """Generates grounded answers with mandatory citations."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def generate(self, query: str, evidence_chunks: List[str]) -> str:
        """Generate answer using only the provided evidence, with citations."""
        evidence_text = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(evidence_chunks)
        )
        messages = [
            SystemMessage(content=CITATION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Evidence:\n{evidence_text}\n\nQuestion: {query}\n\nAnswer (with citations [1], [2], etc.):"
            ),
        ]
        response = self.llm.invoke(messages)
        return response.content


class NaiveAnswerGenerator:
    """Naive LLM answer generator — no grounding, no citations."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def generate(self, query: str) -> str:
        messages = [
            SystemMessage(content=NAIVE_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nAnswer:"),
        ]
        response = self.llm.invoke(messages)
        return response.content
