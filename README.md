A Multi-Agent Grounded System for Hallucination Reduction and Measurable LLM Evaluation
with 
with e Verified RAG & Evaluation Platform_
with 
Recruiter-ready project that turns a hallucination‑resistant RAG chatbot into a **measurement platform**.  
It runs the same questions through **Naive LLM, Standard RAG, Fast RAG, and Verified RAG**, stores all runs, and generates **comparative graphs** showing how much hallucination Verified RAG removes.

This project builds a **multi-agent grounded RAG system** focused on reducing hallucinations in LLM responses.  
Instead of just answering questions, it **compares multiple modes side‑by‑side** (Naive, Standard RAG, Verified RAG) on the same evaluation set.  
For every question and mode, it quantifies **hallucination rate, supported‑claim precision, confidence, and latency**, and visualizes them as charts.  
On our LLM‑evaluation example, **Verified RAG achieved zero hallucinated claims**, while Naive and Standard RAG still hallucinated a significant fraction of their outputs.  
The result is a chatbot that doubles as a **hallucination evaluation and explanation tool**, not just a demo UI.

## Tech Stack

**Python** · **PyTorch** · LangChain · sentence-transformers (DeBERTa-v3) · FAISS · BM25 · FastAPI · minimal chat UI

## What this project demonstrates

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
- **Persistent results**
  - All runs saved to `storage/evaluation_log.json`.
  - Latest aggregate report in `storage/evaluation_report_latest.json`.
  - Charts saved as PNGs in `storage/eval_charts/` (ready for slides/resume).

## Why this matters

- **Hallucination is a major deployment risk** for LLM applications in production (legal, medical, financial, etc.).  
- **Standard RAG helps, but does not eliminate hallucinations** – many answers are still partially unsupported.  
- **Verified RAG introduces claim‑level verification** (via NLI + confidence threshold) and structured refusal.  
- This project makes hallucination reduction **measurable and visual**, so teams can justify using safer pipelines.

## Metrics we track (per mode)

For each mode (Naive, Standard RAG, Verified RAG, Fast RAG), the platform aggregates:

- **Hallucination rate** = unsupported claims / total claims
- **Refusal rate** = refused questions / total questions
- **Supported claim precision** = supported claims / total claims
- **Average confidence**
- **Average latency**
- **Total claims processed**

The UI renders these as bar charts and an “Evaluation Summary” panel so you can say, e.g.:

> “Verified RAG reduced hallucination rate by 65% vs Naive and 35% vs Standard RAG on my dataset.”

## Example evaluation results (LLM eval PDF, 1 question)

This is an example run on a single “LLM app evaluation” PDF question (58 claims total).  
You can re‑run bulk evaluation with more questions to update these numbers.

| Mode          | Hallucination Rate | Supported Claim Precision | Average Confidence | Average Latency |
|--------------|--------------------|---------------------------|--------------------|-----------------|
| Naive        | 50%                | 50%                       | 0.50               | 67.5 s          |
| Standard RAG | 32%                | 68%                       | 0.68               | 48.7 s          |
| Verified RAG | **0%**             | **100%**                  | **1.00**           | 33.7 s          |

**Conclusion:** On this evaluation example, Verified RAG eliminates hallucinated claims while increasing precision and confidence, and it is actually faster than both Naive and Standard RAG due to the optimized pipeline.

## Architecture (high level)

```
Query → [1] Retriever (Hybrid BM25 + Dense) → [2] Reranker → [3] Answer Generator
     → [4] Claim Decomposer → [5] Verification Agent → [6] Confidence Agent
     → Verified Answer OR Structured Refusal
```

- **Hybrid retrieval**: BM25 + dense vectors + Reciprocal Rank Fusion.
- **Reranking**: Cross‑encoder for top‑N evidence selection.
- **Answer generation**: GPT‑3.5 with enforced citations.
- **Claim decomposition**: atomic, independently verifiable claims.
- **Verification**: NLI model (DeBERTa‑v3) per claim + cosine similarity.
- **Confidence**: support ratio; may re‑retrieve (for Verified RAG) or refuse.

## Project Structure

```
Multi_Hybrid_Rag/
├── config/          # Settings
├── src/
│   ├── agents/      # Retriever, Reranker, Generator, Decomposer, Verifier, Confidence
│   ├── ingestion/   # Document loaders, chunking
│   ├── retrieval/   # BM25, dense, hybrid + RRF
│   ├── pipeline/    # Orchestrator
│   ├── observability/  # Langfuse tracing (optional)
│   └── api/         # FastAPI
├── data/
│   ├── documents/   # Input PDFs/TXT/MD
│   ├── chunks/      # Indexed corpus
│   └── eval/        # Test questions for benchmark
├── scripts/         # index_documents, run_benchmark
├── docs/            # ARCHITECTURE.md, PROJECT_FLOW.md
└── .github/workflows/  # CI evaluation gate
```

## Setup (local, Windows friendly)

**Important:** Use a fresh venv to avoid PyTorch conflicts with global packages.

```powershell
cd Multi_Hybrid_Rag

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

## Run the web app

```powershell
cd Multi_Hybrid_Rag
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Then open the **chat UI**:

- Chatbot: `http://127.0.0.1:8001/chat`
- Step‑by‑step app: `http://127.0.0.1:8001/app`

### Upload & index documents

- Supported types: **PDF, TXT, MD, CSV**.
- In `/chat` or `/app`:
  1. Drop/upload files in the **Documents** panel.
  2. Click **Upload**.
  3. Click **Rebuild index** once. It should say: `Index ready (… chunks).`

### Use the chatbot

- **Choose answering mode** (header dropdown):
  - `Fast RAG (no verification)` — best UX speed.
  - `Standard RAG` — retrieval + citations.
  - `Verified RAG` — full verification + structured refusal.
  - `Naive LLM` — baseline.
- Ask a question in the bottom input box → answer appears in a dark bubble.

### Run evaluation from the UI

There are **two** ways to build metrics:

- **Per‑question Evaluation Mode**
  - Sidebar → toggle **Evaluation Mode (per‑question · all 3 modes)** ON.
  - Each chat question runs **Naive, Standard RAG, Verified RAG** in parallel.
  - The verified answer is shown; metrics accumulate in the **Evaluation & Mode Comparison** panel, which can auto‑refresh.

- **Bulk evaluation (recommended for reports)**
  - Scroll to **Evaluation & Mode Comparison**.
  - In **Bulk evaluation**, paste 10–50 questions (one per line or as a numbered list).
  - (Optional) Click **Use last chat questions** to load recent chat prompts into the bulk box.
  - Click **Run bulk evaluation — auto‑report below**.
  - When it finishes, you get:
    - An **Evaluation Summary** with metrics per mode.
    - Bar charts for hallucination rate, refusal rate, precision, confidence, and latency.
    - Results saved to:
      - `storage/evaluation_log.json`
      - `storage/evaluation_report_latest.json`
      - `storage/eval_charts/*.png`

## Extensibility & Optional Integrations

- **Langfuse observability** — set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env` to trace each agent’s latency.
- **GitHub Actions / CI** — you can wire `storage/evaluation_report_latest.json` into a CI check that blocks merges when hallucination rate worsens.

## UI Preview

![Chat UI](docs/ui-chat.png)  
*Adaptive Verified RAG Chatbot with mode selector (Fast / Standard / Verified / Naive) and a verified answer vs refusal example.*

![Evaluation graphs](storage/eval_charts/hallucination_rate.png)  
*The Evaluation & Mode Comparison panel showing hallucination rate, precision, confidence, and latency charts.*

*Add your own screenshots: place `ui-chat.png` in `docs/` and run bulk evaluation to generate charts in `storage/eval_charts/`.*

## Contributors

- **Achuth Reddy**  
  - Designed and implemented the **multi‑agent RAG pipeline** (retriever, reranker, generator, decomposer, verifier, confidence agent).  
  - Added **evaluation logic** in the backend (FastAPI) including multi‑mode execution, metric aggregation, JSON logging, and chart generation.  
  - Tuned performance (faster NLI model, capped claims, parallel execution, Fast RAG mode).

- **Meghashyam**  
  - Built the **Adaptive Verified RAG Chatbot UI** (mode selection, document upload/indexing, chat view).  
  - Implemented the **Evaluation Mode** and **Bulk evaluation** UX, including auto‑updating metrics and graphs.  
  - Wrote and refined the **project README**, selected evaluation PDFs/questions, and ran experiments to generate the final metrics and visualizations.

## Tests

```powershell
pip install pytest
pytest                    # Fast tests only (excludes slow/hybrid)
pytest -m slow            # Include slow tests (requires model download)
```
