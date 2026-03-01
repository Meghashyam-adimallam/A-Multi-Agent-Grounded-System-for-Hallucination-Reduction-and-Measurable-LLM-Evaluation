"""Langfuse tracer — per-agent spans, latency, and cost tracking."""

import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

_tracer = None


def get_tracer():
    """Lazy-init Langfuse tracer. Returns None if keys not configured."""
    global _tracer
    if _tracer is not None:
        return _tracer
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY") or os.getenv("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        return None
    try:
        from langfuse import Langfuse
        _tracer = Langfuse()
        return _tracer
    except Exception:
        return None


@contextmanager
def trace_agent(name: str, metadata: dict | None = None) -> Generator[dict, None, None]:
    """
    Context manager for agent spans. Records latency.
    If Langfuse is configured, creates a span. Otherwise no-op.
    """
    start = time.perf_counter()
    span = None
    tracer = get_tracer()
    if tracer:
        try:
            trace = tracer.trace(name=name, metadata=metadata or {})
            span = trace
        except Exception:
            pass
    try:
        yield {"start": start}
    finally:
        elapsed = time.perf_counter() - start
        if span:
            try:
                span.end(metadata={"latency_seconds": elapsed})
            except Exception:
                pass


def trace_pipeline(query: str) -> Any:
    """Create a root trace for the full pipeline. Returns trace or None."""
    tracer = get_tracer()
    if not tracer:
        return None
    try:
        return tracer.trace(name="rag_pipeline", input=query)
    except Exception:
        return None
