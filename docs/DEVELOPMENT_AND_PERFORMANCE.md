# Development & Performance — Adaptive Verified RAG

## Why a single question can take ~5 minutes

Verified RAG runs a **6-stage pipeline** plus **cold start**. Here’s where time goes.

### 1. Cold start (first request only)

On the **first** `/query` call, the pipeline is built lazily. That loads:

| Component | What loads | Typical time (CPU) |
|-----------|------------|---------------------|
| **HybridRetriever** | BM25 index + DenseRetriever (SentenceTransformer + FAISS) | 10–30 s |
| **RerankerAgent** | CrossEncoder `ms-marco-MiniLM-L-6-v2` | 5–15 s |
| **VerificationAgent** | NLI CrossEncoder `nli-deberta-v3-base` + embedder `all-MiniLM-L6-v2` | 15–40 s |
| **AnswerGenerator / ClaimDecomposer** | LangChain OpenAI client (light) | &lt; 1 s |

So the **first** query can easily add **30–90+ seconds** before any real work.

### 2. Per-query pipeline (Verified RAG)

Each stage runs **sequentially**:

| Stage | What it does | Typical time |
|-------|----------------|--------------|
| **Retriever** | BM25 + dense search, RRF | 0.5–2 s |
| **Reranker** | CrossEncoder on 15 (query, chunk) pairs | 2–8 s |
| **Generator** | 1 OpenAI API call (answer with citations) | 3–15 s |
| **Claim Decomposer** | 1 OpenAI API call (atomic claims) | 2–10 s |
| **Verifier** | NLI + embeddings per claim (see below) | 10–60+ s |
| **Confidence** | Ratio only | &lt; 0.1 s |

So **after** cold start, a single Verified RAG query is often **20–90 seconds**, and the **Verifier** is usually the slowest part.

### 3. Why the Verifier is slow

- **NLI**: For each claim we run the NLI model on **(evidence, claim)** for every evidence chunk. With 3 claims and 4 chunks that’s **12** cross-encoder passes.
- **Embeddings**: Previously we re-encoded the **same** evidence chunks for **every** claim (3 × 4 = 12 evidence encodings). We now encode evidence **once** in `verify_all` and reuse (see code).
- **CPU**: All of this runs on CPU by default; GPU would speed NLI and embeddings a lot.

So **5 minutes** for one question is plausible when:

- It’s the **first** request (cold start 1–2 min) **plus**
- One full Verified RAG run (1–3 min) **plus**
- Network/OpenAI latency or a slow machine.

---

## What’s already in place (Adaptive Verified RAG)

Your project **already** supports the comparison you described:

- **Naive** → LLM only, then verify claims against retrieved evidence (for scoring).
- **Standard RAG** → Retrieve → Rerank → Generate → Decompose → Verify (no re-retrieval loop).
- **Verified RAG** → Same as above with re-retrieval loop and structured refusal when confidence &lt; threshold.

**API**

- `POST /query` with `mode`: `"naive"` | `"rag"` | `"multi"` (multi = Verified RAG).
- `evaluation_mode: true` runs **all three** in parallel and returns the Verified RAG result plus logged metrics.

**UI**

- **`/chat`** — “Adaptive Verified RAG Chatbot”: mode selector (Naive / Standard RAG / Verified RAG), compare answers, trust report.
- **`/app`** — Upload → Index → Query (step-by-step).
- Bulk evaluation: run many questions through all three modes and see aggregates (e.g. average confidence, latency, hallucination).

So the **dataset** (your uploaded documents) is what the retriever searches; the **comparison** (naive vs RAG vs verified) is already implemented in the API and chat UI.

---

## What’s needed in development (priority)

### High impact (reduce latency)

1. **Warmup at startup**  
   Call `get_pipeline()` (or a small dummy query) inside the FastAPI lifespan so all models load **before** the first user request. First query then skips cold start.

2. **Verifier: reuse evidence embeddings**  
   **Done.** Evidence chunks are encoded once in `verify_all` and passed into `verify()` so we don’t re-encode the same chunks per claim.

3. **Batch NLI where possible**  
   The NLI model already gets a list of pairs; keep batching (e.g. all (evidence, claim) pairs for a claim in one `predict` call) and avoid per-pair Python loops that call `predict` one-by-one.

4. **Optional GPU**  
   Use CUDA for SentenceTransformer and CrossEncoders if available (e.g. set `device="cuda"`). Cuts NLI and reranker time a lot.

### Medium impact (UX and cost)

5. **Streaming**  
   Stream the generator output so the user sees the answer as it’s generated; total time is unchanged but perceived latency drops.

6. **Cheaper/faster LLMs for Decomposer**  
   e.g. `gpt-3.5-turbo` or a small local model for claim decomposition only; keep a stronger model for the main answer if needed.

7. **Caching**  
   Cache retrieval (and optionally rerank) results for identical queries so repeated “what is the dataset about?” calls don’t re-run the full pipeline.

### Lower priority (stability and ops)

8. **Timeouts and limits**  
   Set timeouts for OpenAI and a max time per pipeline run so a single request can’t hang for 5+ minutes.

9. **Observability**  
   You already have `agent_latencies` and `total_latency` in the response; add a simple dashboard or logs to see which stage dominates (usually Verifier or Generator).

10. **Health check**  
    Optional: `GET /health/warm` that triggers pipeline load and returns 200 when ready, so load balancers or the UI can wait until the app is “warm”.

---

## Quick reference: pipeline flow

```
User → Hybrid Retriever → Reranker → Generator → Claim Decomposer
    → Verifier → Confidence Engine → Trust Report (or Refusal)
```

- **Naive**: Generate (no context) → then retrieve + rerank + decompose + verify for scoring.
- **RAG / Verified**: Retrieve → Rerank → Generate → Decompose → Verify → Confidence; Verified adds retry/refusal logic.

---

## Summary

- **Why so slow:** Cold start (many models) + 6 sequential stages + 2 LLM calls + Verifier (N×M NLI + embeddings). First request plus one Verified RAG run can reach several minutes.
- **What’s in place:** Naive vs RAG vs Verified comparison in API and Chat UI; your “dataset” is the indexed documents.
- **What to do next:** Warmup at startup, keep verifier optimization (evidence embeddings reused), consider GPU and streaming; then caching and timeouts.
