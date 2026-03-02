A Multi-Agent Grounded System for Hallucination Reduction and Measurable LLM Evaluation

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)

Multi-agent grounded RAG system that reduces hallucinations and **measures** the improvement. Runs the same questions through **Naive LLM, Standard RAG, Fast RAG, and Verified RAG**, stores all runs, and generates **comparative graphs**. On our evaluation, **Verified RAG achieved zero hallucinated claims** vs 50% for Naive and 32% for Standard RAG.

## 🔧 Tech Stack

**Python** · **PyTorch** · LangChain · sentence-transformers (DeBERTa-v3) · FAISS · BM25 · FastAPI · minimal chat UI

## ✨ What this project demonstrates

- **Multi‑mode answering**
  - **Fast RAG** — retrieve → rerank → answer (no verification; GPT‑style UX).
  - **Standard RAG** — retrieval + citations + light verification.
  - **Verified RAG** — claim decomposition + NLI verification + structured refusal.
  - **Naive LLM** — no retrieval, used as a worst‑case baseline.
- **Evaluation Mode**
  - For each question, automatically runs **all 3 main modes** (Naive, Standard RAG, Verified RAG).
  - Logs per‑question metrics and updates an **Evaluation Summary** + charts in the UI.
- **Bulk evaluation**
  - Paste 10–50 questions at once.
  - The system runs Naive/Standard/Verified for each question and **auto‑builds a report**.
- **Persistent results** → `storage/evaluation_log.json`, `storage/eval_charts/*.png`

## 💡 Why this matters

Hallucination is a major risk for LLMs in production. Standard RAG helps but doesn't eliminate it. This project adds **claim-level verification** (NLI + confidence threshold) and makes the improvement **measurable and visual**.

## 📊 Metrics we track

Per mode: **hallucination rate**, **refusal rate**, **supported-claim precision**, **average confidence**, **latency**. Rendered as bar charts in the UI.

## 📈 Example results (1 PDF question, 58 claims)

| Mode          | Hallucination Rate | Supported Claim Precision | Avg Confidence | Avg Latency |
|---------------|--------------------|---------------------------|----------------|-------------|
| Naive         | 50%                | 50%                       | 0.50           | 67.5 s      |
| Standard RAG  | 32%                | 68%                       | 0.68           | 48.7 s      |
| Verified RAG  | **0%**             | **100%**                  | **1.00**       | 33.7 s      |

Verified RAG eliminates hallucinations and is faster than both baselines.

## 🏗️ Architecture (high level)

```
Query → [1] Retriever (Hybrid BM25 + Dense) → [2] Reranker → [3] Answer Generator
     → [4] Claim Decomposer → [5] Verification Agent → [6] Confidence Agent
     → Verified Answer OR Structured Refusal
```

## 📁 Project Structure

```
├── config/          # Settings
├── src/
│   ├── agents/      # Retriever, Reranker, Generator, Decomposer, Verifier, Confidence
│   ├── ingestion/   # Document loaders, chunking
│   ├── retrieval/   # BM25, dense, hybrid + RRF
│   ├── pipeline/    # Orchestrator
│   ├── observability/  # Optional tracing
│   └── api/         # FastAPI
├── data/
│   ├── documents/   # Input PDFs/TXT/MD
│   ├── chunks/      # Indexed corpus
│   └── eval/        # Test questions for benchmark
├── scripts/         # index_documents, run_benchmark
├── docs/            # ARCHITECTURE.md, PROJECT_FLOW.md
└── .github/workflows/  # CI evaluation gate
```

## 🚀 Setup (local, Windows friendly)

**Important:** Use a fresh venv to avoid PyTorch conflicts with global packages.

```powershell
# From project root
# Create venv (required — avoids torch/CUDA conflicts)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch CPU first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install rest
pip install langchain langchain-community langchain-openai langchain-text-splitters pypdf rank_bm25 faiss-cpu sentence-transformers fastapi uvicorn python-dotenv

# Copy env template
copy .env.example .env
# Edit .env: add OPENAI_API_KEY
```

## ▶️ Run the web app

```powershell
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Then open the **chat UI**:

- Chatbot: `http://127.0.0.1:8001/chat`
- Step‑by‑step app: `http://127.0.0.1:8001/app`

### 📤 Upload & index documents

- Supported types: **PDF, TXT, MD, CSV**.
- In `/chat` or `/app`:
  1. Drop/upload files in the **Documents** panel.
  2. Click **Upload**.
  3. Click **Rebuild index** once. It should say: `Index ready (… chunks).`

### 💬 Use the chatbot

- **Choose answering mode** (header dropdown):
  - `Fast RAG (no verification)` — best UX speed.
  - `Standard RAG` — retrieval + citations.
  - `Verified RAG` — full verification + structured refusal.
  - `Naive LLM` — baseline.
- Ask a question in the bottom input box → answer appears in a dark bubble.

### 📋 Run evaluation from the UI

- **Per-question:** Sidebar → toggle **Evaluation Mode** ON. Each question runs all 3 modes; metrics update in **Evaluation & Mode Comparison**.
- **Bulk (recommended):** Paste 10–50 questions → **Run bulk evaluation**. Results → `storage/evaluation_log.json`, `storage/eval_charts/*.png`

## 👥 Contributors

- **Achuth Reddy**  
  - Designed and implemented the **multi‑agent RAG pipeline** (retriever, reranker, generator, decomposer, verifier, confidence agent).  
  - Added **evaluation logic** in the backend (FastAPI) including multi‑mode execution, metric aggregation, JSON logging, and chart generation.  
  - Tuned performance (faster NLI model, capped claims, parallel execution, Fast RAG mode).

- **Meghashyam**  
  - Built the **Adaptive Verified RAG Chatbot UI** (mode selection, document upload/indexing, chat view).  
  - Implemented the **Evaluation Mode** and **Bulk evaluation** UX, including auto‑updating metrics and graphs.  
  - Wrote and refined the **project README**, selected evaluation PDFs/questions, and ran experiments to generate the final metrics and visualizations.

## 🧪 Tests

```powershell
pip install pytest
pytest                    # Fast tests only (excludes slow/hybrid)
pytest -m slow            # Include slow tests (requires model download)
```
