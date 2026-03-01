# Multi-Agent Grounded RAG — Project Flow & What We Achieve

**Project motive:** Our main goal is to **reduce hallucination**. We get an initial hallucination score from a query, perform operations (add documents, re-build index, ask specific questions), then run again and see the score go down. Measure → act → improve.

---

## What Uvicorn Host Means

| Host        | Who can reach the API              | When to use                    |
|------------|-------------------------------------|--------------------------------|
| `127.0.0.1`| Only this computer (localhost)      | Development, testing on your PC |
| `0.0.0.0`  | This computer + others on your network | Same LAN (e.g. phone, another PC) |

**We use `127.0.0.1`** so the API is only reachable from your machine. No change to logic — only who can call it.

**What we achieve:** A running HTTP server that accepts queries, runs the 6-agent pipeline, and returns a verified answer or a structured refusal.

---

## End-to-End Project Flow (2D)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: OFFLINE — BUILD THE INDEX (once)                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

     YOU ADD DOCUMENTS                    SCRIPT RUNS                    INDEX SAVED
     ─────────────────                    ───────────                   ───────────

     data/documents/                      python scripts/                data/chunks/
     ├── file1.pdf                        index_documents.py             └── corpus.json
     ├── file2.txt       ─────────────►   (load → chunk →                 (chunks + BM25
     └── sample.md                        BM25 + dense index)              + dense index
                                                 │
                                                 ▼
                                          "Created 4 chunks."
                                          "Saved corpus to ..."


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 2: ONLINE — SERVE & QUERY (every time)                │
└─────────────────────────────────────────────────────────────────────────────────────┘

     YOU START SERVER                     YOU WRITE A QUERY              YOU GET RESPONSE
     ─────────────────                    ─────────────────              ───────────────

     uvicorn src.api.main:app             Browser / Swagger /            Verified answer
     --host 127.0.0.1                     POST /query                     OR structured
     --port 8000                              │                           refusal
           │                                 │
           ▼                                 ▼
     ┌──────────┐                    ┌──────────────┐
     │ API live │ ◄──────────────────│ {"query":    │
     │ :8000    │   HTTP POST         │  "What does  │
     └──────────┘                    │  Retriever   │
           │                          │  Agent do?"} │
           │                          └──────────────┘
           │
           ▼
     Pipeline runs (6 agents)
           │
           ▼
     ┌──────────────┐
     │ answer +     │
     │ confidence   │  or  refusal + explanation
     └──────────────┘
```

---

## 2D Flow: Documents vs Query

```
                         YOUR DOCUMENTS (input to index)
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  data/documents/  (PDF, TXT, MD)                             │
    │  → Load → Chunk (RecursiveCharacterTextSplitter)             │
    │  → BM25 index + Dense (FAISS) index                           │
    │  → Saved as data/chunks/corpus.json                          │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    │  (index is ready)
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Uvicorn: API server listening on 127.0.0.1:8000             │
    │  • GET  /         → Landing page (try a query)                │
    │  • GET  /docs     → Swagger UI                                │
    │  • GET  /health   → {"status": "ok"}                         │
    │  • POST /query    → Run pipeline, return answer or refusal    │
    └──────────────────────────────────────────────────────────────┘
                                    │
                         YOUR QUERY (input at runtime)
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Query: "What does the Retriever Agent do?"                  │
    │  → Retriever: search corpus (BM25 + dense, RRF)               │
    │  → Reranker: top-N chunks                                    │
    │  → Generator: answer with citations                           │
    │  → Decomposer: atomic claims                                  │
    │  → Verifier: NLI per claim                                    │
    │  → Confidence: ratio ≥ 0.70? → answer : retry or refuse     │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Response: { "answer": "...", "confidence_score": 0.85,       │
    │             "status": "verified", ... }                       │
    │  OR       { "status": "refused", "explanation": "..." }      │
    └──────────────────────────────────────────────────────────────┘
```

---

## Summary: What We Do and What We Achieve

| Step | What you do | What the system does | What we achieve |
|------|-------------|----------------------|------------------|
| **1. Add documents** | Put PDF/TXT/MD in `data/documents/` | — | Corpus for retrieval |
| **2. Index** | Run `python scripts/index_documents.py` | Loads, chunks, builds BM25 + dense index, saves `corpus.json` | Searchable evidence base |
| **3. Start server** | Run `uvicorn ... --host 127.0.0.1 --port 8000` | Starts HTTP server on port 8000 | API ready for queries |
| **4. Write query** | Open `/` or `/docs`, type a question, or POST to `/query` | Runs 6-agent pipeline on your query | One request in |
| **5. Get result** | Read JSON response | Verified answer with citations, or structured refusal | No hallucinated answer; either grounded reply or clear refusal |

**In short:** We take **your documents** (once), build an **index**. Then we take **your query** (each time), run it through the pipeline against that index, and return a **verified answer** or a **structured refusal**.
