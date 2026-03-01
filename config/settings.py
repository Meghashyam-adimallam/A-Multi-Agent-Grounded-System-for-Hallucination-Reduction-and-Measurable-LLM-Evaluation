"""Configuration for the Multi-Agent Grounded RAG System."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Vector store
    vector_store: str = "faiss"  # faiss | chromadb
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "data/faiss_index"

    # Retrieval
    top_k: int = 20  # Hybrid retrieval returns top-K
    top_n: int = 5   # Reranker returns top-N
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # NLI / Verification
    nli_model: str = "cross-encoder/nli-deberta-v3-large"

    # LLM
    llm_provider: str = "openai"  # openai | local
    openai_model: str = "gpt-4"
    local_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: float = 0.0

    # Confidence
    confidence_threshold: float = 0.70
    max_retrieval_retries: int = 2

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Langfuse (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Paths
    documents_path: str = "data/documents"
    chunks_path: str = "data/chunks"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
