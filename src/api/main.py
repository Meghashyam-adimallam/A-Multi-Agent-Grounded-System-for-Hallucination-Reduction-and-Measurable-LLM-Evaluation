"""FastAPI deployment — POST /query endpoint."""

import os
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock, Timer
from typing import Literal
from uuid import uuid4

import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()


# Paths for persistent storage
STORAGE_ROOT = Path("storage")
UPLOADS_DIR = STORAGE_ROOT / "uploads"
INDEX_DIR = STORAGE_ROOT / "index"
CORPUS_PATH = INDEX_DIR / "corpus.json"
META_PATH = INDEX_DIR / "metadata.json"
EVAL_LOG_PATH = STORAGE_ROOT / "evaluation_log.json"
EVAL_CHARTS_DIR = STORAGE_ROOT / "eval_charts"
EVAL_REPORT_PATH = STORAGE_ROOT / "evaluation_report_latest.json"


# Lazy imports to avoid loading heavy models at startup
_pipeline = None

# In-memory evaluation tracking for current session
evaluation_results: list[dict] = []
_evaluation_session_id = str(uuid4())
_evaluation_session_started_at = datetime.utcnow().isoformat()
_eval_lock = Lock()


def get_pipeline():
    """Load RAG pipeline using persistent index if available."""
    global _pipeline
    if _pipeline is None:
        import json
        from src.retrieval import HybridRetriever
        from src.pipeline import RAGPipeline

        if not CORPUS_PATH.exists():
            raise RuntimeError(
                "No index found. Upload documents and rebuild the index first."
            )
        with open(CORPUS_PATH, encoding="utf-8") as f:
            corpus = json.load(f)
        retriever = HybridRetriever(corpus)
        # Speed: no re-retrieval (max_retries=0), smaller top_n for faster rerank/generate
        _pipeline = RAGPipeline(retriever, top_k=15, top_n=4, max_retries=0)
    return _pipeline


def _open_chat_browser():
    """Open default browser to the chat UI after server is ready."""
    host = os.environ.get("UVICORN_HOST", "127.0.0.1")
    port = os.environ.get("UVICORN_PORT", "8001")
    webbrowser.open(f"http://{host}:{port}/chat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open chat in browser shortly after server starts
    Timer(1.5, _open_chat_browser).start()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Multi-Agent Grounded RAG API",
    description="Hallucination-resistant Q&A with claim verification and evaluation",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


class QueryRequest(BaseModel):
    query: str
    mode: Literal["fast", "naive", "rag", "multi"] = "multi"
    evaluation_mode: bool = False


class BulkEvalRequest(BaseModel):
    """Run all questions through Naive, Standard RAG, and Verified RAG; return report."""
    questions: list[str]


class QueryResponse(BaseModel):
    answer: str | None = None
    confidence_score: float | None = None
    hallucination_score: float | None = None  # 0 = low risk, 1 = high; from verdicts/confidence
    claims: list[str] | None = None
    verdicts: list[str] | None = None
    claim_evidence: list[str | None] | None = None
    status: str | None = None
    query: str | None = None
    unverified_claims: list[str] | None = None
    evidence_found: list[str] | None = None
    refusal_reason: str | None = None
    explanation: str | None = None
    agent_latencies: dict[str, float] | None = None
    re_retrieval_triggered: bool | None = None
    retries: int | None = None
    total_latency: float | None = None

    class Config:
        extra = "ignore"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Custom landing page with links and query form."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Grounded RAG</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 2rem;
        }
        .container { max-width: 640px; margin: 0 auto; }
        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d9ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .card h2 {
            font-size: 0.9rem;
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .links {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .links a {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #3b82f6, #7c3aed);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .links a:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        }
        .links a.secondary {
            background: rgba(148, 163, 184, 0.2);
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        .links a.secondary:hover {
            background: rgba(148, 163, 184, 0.3);
        }
        textarea {
            width: 100%;
            min-height: 80px;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            color: #e8e8e8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            resize: vertical;
            margin-bottom: 0.75rem;
        }
        textarea:focus {
            outline: none;
            border-color: #3b82f6;
        }
        button {
            padding: 0.6rem 1.25rem;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        #result {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.9);
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        .status { font-size: 0.8rem; color: #10b981; margin-top: 0.5rem; }
        .error { color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Agent Grounded RAG</h1>
        <p class="subtitle">Hallucination-resistant Q&A with claim verification</p>

        <div class="card">
            <h2>Quick links</h2>
            <div class="links">
                <a href="/chat">Adaptive Verified RAG Chatbot</a>
                <a href="/app" class="secondary">Upload &amp; run (step-by-step)</a>
                <a href="/docs">Swagger UI</a>
                <a href="/redoc" class="secondary">ReDoc</a>
                <a href="/health" class="secondary">Health</a>
            </div>
        </div>

        <div class="card">
            <h2>Try a query</h2>
            <textarea id="query" placeholder="e.g. What does the Retriever Agent do?">What does the Retriever Agent do?</textarea>
            <button id="btn" onclick="runQuery()">Run query</button>
            <div id="result"></div>
        </div>
    </div>
    <script>
        async function runQuery() {
            const q = document.getElementById('query').value.trim();
            const btn = document.getElementById('btn');
            const result = document.getElementById('result');
            if (!q) return;
            btn.disabled = true;
            result.innerHTML = '<span class="status">Running... (first query may take 1–2 min)</span>';
            try {
                const r = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: q })
                });
                const data = await r.json();
                result.innerHTML = '<span class="status">Status: ' + (data.status || '—') + '</span>\\n' + JSON.stringify(data, null, 2);
            } catch (e) {
                result.innerHTML = '<span class="error">Error: ' + e.message + '</span>';
            }
            btn.disabled = false;
        }
    </script>
</body>
</html>
"""


@app.get("/api/status")
def api_status():
    """Return whether documents and index exist, plus simple metadata."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    files = list(UPLOADS_DIR.glob("*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md", ".csv")]
    chunk_count = 0
    if CORPUS_PATH.exists():
        import json
        with open(CORPUS_PATH, encoding="utf-8") as f:
            corpus = json.load(f)
        chunk_count = len(corpus)
    return {
        "documents_count": len(files),
        "index_exists": CORPUS_PATH.exists(),
        "documents": [f.name for f in files],
        "chunk_count": chunk_count,
    }


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    """Upload one or more documents (PDF, TXT, MD, CSV) to storage/uploads/."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    allowed = {".pdf", ".txt", ".md", ".csv"}
    saved = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            raise HTTPException(400, f"File type not allowed: {f.filename}. Use PDF, TXT, MD, or CSV.")
        path = UPLOADS_DIR / f.filename
        content = await f.read()
        path.write_bytes(content)
        saved.append(f.filename)
    return {"uploaded": saved, "count": len(saved)}


@app.delete("/api/file/{filename}")
def api_delete_file(filename: str):
    """Delete a document from storage/uploads/. Does NOT automatically rebuild index."""
    target = UPLOADS_DIR / filename
    if not target.exists():
        raise HTTPException(404, f"File not found: {filename}")
    try:
        target.unlink()
    except Exception as e:
        raise HTTPException(500, f"Could not delete file: {e}")
    return {"deleted": filename}


def _run_indexing():
    """Run indexing over all stored uploads and save corpus + metadata. Resets pipeline cache."""
    global _pipeline
    from src.ingestion import load_documents, chunk_documents
    import json

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous index files and metadata for a fresh rebuild
    for p in (CORPUS_PATH, META_PATH):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # Load all documents from persistent uploads directory (resolve to absolute path)
    uploads_path = UPLOADS_DIR.resolve()
    docs = load_documents(uploads_path)
    if not docs:
        return {
            "chunks": 0,
            "documents": 0,
            "message": "No documents to index. Upload PDF, TXT, MD, or CSV files first. CSV files must have a header row and be UTF-8.",
        }

    # Chunk and build corpus
    chunks = chunk_documents(docs)
    corpus = [c.page_content for c in chunks]

    # Build simple metadata: per-file chunk ranges
    file_ranges: dict[str, dict[str, int]] = {}
    current_indices: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source") or ""
        name = Path(source).name if source else f"chunk_{idx}"
        if name not in file_ranges:
            file_ranges[name] = {"start": idx, "end": idx}
        else:
            file_ranges[name]["end"] = idx
        current_indices[name] = current_indices.get(name, 0) + 1

    meta = {
        "total_chunks": len(corpus),
        "files": [
            {
                "name": name,
                "start_idx": ranges["start"],
                "end_idx": ranges["end"],
                "num_chunks": current_indices.get(name, 0),
            }
            for name, ranges in file_ranges.items()
        ],
    }

    # Persist corpus and metadata
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Reset pipeline so it reloads on next query
    _pipeline = None

    return {
        "chunks": len(corpus),
        "documents": len(meta["files"]),
        "message": "Index built successfully.",
    }


@app.post("/api/index")
def api_index():
    """Build index from documents in data/documents/. May take 1–2 min."""
    try:
        result = _run_indexing()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def app_ui():
    """Step-by-step UI: Upload → Index → Query."""
    ui_path = Path(__file__).parent / "app_ui.html"
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_ui():
    """Adaptive Verified RAG Chatbot interface."""
    ui_path = Path(__file__).parent / "chat_ui.html"
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


def _compute_hallucination_score(result: dict) -> float:
    """Compute hallucination score 0–1 from confidence and verdicts. Higher = more risk."""
    if result.get("status") == "refused":
        return 1.0
    verdicts = result.get("verdicts") or []
    if verdicts:
        bad = sum(1 for v in verdicts if v in ("CONTRADICTED", "UNVERIFIED"))
        return bad / len(verdicts)
    conf = result.get("confidence_score")
    if conf is not None:
        return round(1.0 - float(conf), 4)
    return 0.0


def _build_evaluation_record(question: str, mode_label: str, result: dict) -> dict:
    """Build a single evaluation record from a pipeline result."""
    timestamp = datetime.utcnow().isoformat()
    status = result.get("status")
    refused = status == "refused"

    confidence = float(result.get("confidence_score") or 0.0)
    hallucination_score = float(
        result.get("hallucination_score") or _compute_hallucination_score(result)
    )
    latency = float(result.get("total_latency") or 0.0)

    claims = result.get("claims") or []
    verdicts = result.get("verdicts") or []
    unverified_claims = result.get("unverified_claims") or []

    num_claims = 0
    supported_claims = 0

    if claims:
        num_claims = len(claims)
        if verdicts:
            supported_claims = sum(1 for v in verdicts if v == "SUPPORTED")
    elif unverified_claims:
        num_claims = len(unverified_claims)
        supported_claims = 0
    else:
        # No explicit claims available — treat the whole answer as one claim.
        num_claims = 1
        supported_claims = 0

    unsupported_claims = max(0, num_claims - supported_claims)

    return {
        "timestamp": timestamp,
        "question": question,
        "mode": mode_label,
        "refused": refused,
        "confidence": confidence,
        "hallucination_score": hallucination_score,
        "num_claims": num_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "latency": latency,
    }


def _append_evaluation_record(record: dict) -> None:
    """Append an evaluation record to in-memory list and persist to JSON."""
    global evaluation_results

    with _eval_lock:
        evaluation_results.append(record)
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

        existing: list = []
        if EVAL_LOG_PATH.exists():
            try:
                with open(EVAL_LOG_PATH, encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        if not isinstance(existing, list):
            existing = []

        # Find or create current session group
        session = None
        for s in existing:
            if s.get("session_id") == _evaluation_session_id:
                session = s
                break

        if session is None:
            session = {
                "session_id": _evaluation_session_id,
                "started_at": _evaluation_session_started_at,
                "results": [],
            }
            existing.append(session)

        session["results"] = evaluation_results.copy()

        with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


def _aggregate_evaluation_metrics(records: list[dict]) -> dict:
    """Aggregate per-mode metrics from flat evaluation records."""
    if not records:
        return {
            "per_mode": {},
            "overall": {"total_questions": 0, "total_claims": 0},
            "relative_reduction_vs_naive": None,
            "relative_reduction_vs_standard": None,
        }

    per_mode: dict[str, dict] = {}
    for rec in records:
        mode = rec.get("mode") or "Unknown"
        bucket = per_mode.setdefault(
            mode,
            {
                "total_questions": 0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "refused_questions": 0,
                "total_confidence": 0.0,
                "total_latency": 0.0,
            },
        )
        bucket["total_questions"] += 1
        bucket["total_claims"] += int(rec.get("num_claims") or 0)
        bucket["supported_claims"] += int(rec.get("supported_claims") or 0)
        bucket["unsupported_claims"] += int(rec.get("unsupported_claims") or 0)
        if rec.get("refused"):
            bucket["refused_questions"] += 1
        bucket["total_confidence"] += float(rec.get("confidence") or 0.0)
        bucket["total_latency"] += float(rec.get("latency") or 0.0)

    metrics: dict[str, dict] = {}
    for mode, bucket in per_mode.items():
        total_q = max(1, bucket["total_questions"])
        total_claims = max(1, bucket["total_claims"])

        hallucination_rate = bucket["unsupported_claims"] / float(total_claims)
        precision = bucket["supported_claims"] / float(total_claims)
        refusal_rate = bucket["refused_questions"] / float(total_q)
        avg_confidence = bucket["total_confidence"] / float(total_q)
        avg_latency = bucket["total_latency"] / float(total_q)

        metrics[mode] = {
            "hallucination_rate": hallucination_rate,
            "refusal_rate": refusal_rate,
            "supported_claim_precision": precision,
            "average_confidence": avg_confidence,
            "average_latency": avg_latency,
            "total_claims": bucket["total_claims"],
            "total_questions": bucket["total_questions"],
        }

    # Overall summary
    unique_questions = {rec.get("question", "") for rec in records}
    total_claims_overall = sum(int(rec.get("num_claims") or 0) for rec in records)

    overall = {
        "total_questions": len(unique_questions),
        "total_claims": total_claims_overall,
    }

    # Relative hallucination reduction for Verified vs baselines
    naive_rate = (
        metrics.get("Naive", {}).get("hallucination_rate")
        if "Naive" in metrics
        else None
    )
    standard_rate = (
        metrics.get("Standard RAG", {}).get("hallucination_rate")
        if "Standard RAG" in metrics
        else None
    )
    verified_rate = (
        metrics.get("Verified RAG", {}).get("hallucination_rate")
        if "Verified RAG" in metrics
        else None
    )

    def _relative_reduction(baseline: float | None, target: float | None) -> float | None:
        if baseline is None or target is None or baseline <= 0:
            return None
        return max(0.0, (baseline - target) / baseline)

    rel_vs_naive = _relative_reduction(naive_rate, verified_rate)
    rel_vs_standard = _relative_reduction(standard_rate, verified_rate)

    return {
        "per_mode": metrics,
        "overall": overall,
        "relative_reduction_vs_naive": rel_vs_naive,
        "relative_reduction_vs_standard": rel_vs_standard,
    }


def _build_evaluation_charts(per_mode: dict[str, dict]) -> dict:
    """Build base64-encoded PNG charts for key metrics using matplotlib.

    Charts are returned as base64 data URIs for the UI, and also saved as PNG files
    under storage/eval_charts/ so results are captured on disk.
    """
    try:
        import io
        import base64

        import matplotlib.pyplot as plt
    except Exception:
        # If matplotlib is not available, skip chart generation.
        return {}

    if not per_mode:
        return {}

    mode_labels = ["Naive", "Standard RAG", "Verified RAG"]
    colors = ["#64748b", "#3b82f6", "#22c55e"]

    def _encode_figure(
        values_key: str, ylabel: str, out_key: str, ylim: tuple | None = None
    ) -> str | None:
        values = [float(per_mode.get(m, {}).get(values_key) or 0.0) for m in mode_labels]
        if all(v == 0.0 for v in values):
            return None

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.bar(mode_labels, values, color=colors)
        ax.set_ylabel(ylabel, color="#e5e7eb")
        ax.set_xticklabels(mode_labels, rotation=0, color="#e5e7eb")
        ax.tick_params(colors="#9ca3af")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_facecolor("#020617")
        fig.patch.set_facecolor("#020617")
        plt.tight_layout()

        # Save to in-memory buffer for data URI
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")

        # Also persist chart PNG to disk for later use in reports/presentations.
        try:
            EVAL_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = EVAL_CHARTS_DIR / f"{_evaluation_session_id}_{out_key}.png"
            fig.savefig(file_path, format="png", dpi=150, bbox_inches="tight")
        except Exception:
            # Chart saving failure should not break the API.
            pass
        finally:
            plt.close(fig)

        return f"data:image/png;base64,{encoded}"

    charts: dict[str, str] = {}
    chart_defs = [
        ("hallucination_rate", "Hallucination rate", (0.0, 1.0), "hallucination_rate"),
        ("refusal_rate", "Refusal rate", (0.0, 1.0), "refusal_rate"),
        (
            "supported_claim_precision",
            "Supported claim precision",
            (0.0, 1.0),
            "supported_claim_precision",
        ),
        ("average_confidence", "Average confidence", (0.0, 1.0), "average_confidence"),
        ("average_latency", "Average latency (seconds)", None, "average_latency"),
    ]

    for key, ylabel, ylim, out_key in chart_defs:
        img = _encode_figure(key, ylabel, out_key, ylim)
        if img:
            charts[out_key] = img

    return charts


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the grounded RAG pipeline on a query."""
    try:
        pipeline = get_pipeline()
        # Evaluation mode: run all three modes IN PARALLEL, log metrics, return only Verified RAG to UI.
        if request.evaluation_mode:
            mode_map = [
                ("naive", "Naive"),
                ("rag", "Standard RAG"),
                ("multi", "Verified RAG"),
            ]
            verified_result: dict | None = None
            results_by_mode: dict[str, dict] = {}

            def run_one(internal_mode: str, label: str) -> tuple[str, str, dict]:
                if internal_mode == "naive":
                    mode_result = pipeline.run_naive(request.query)
                elif internal_mode == "rag":
                    mode_result = pipeline.run_standard_rag(request.query)
                else:
                    mode_result = pipeline.run(request.query)
                return (internal_mode, label, mode_result)

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(run_one, internal_mode, label): (internal_mode, label)
                    for internal_mode, label in mode_map
                }
                for future in as_completed(futures):
                    try:
                        internal_mode, label, mode_result = future.result()
                        mode_result["hallucination_score"] = _compute_hallucination_score(
                            mode_result
                        )
                        record = _build_evaluation_record(
                            request.query, label, mode_result
                        )
                        _append_evaluation_record(record)
                        results_by_mode[internal_mode] = mode_result
                        if internal_mode == "multi":
                            verified_result = mode_result
                    except Exception:
                        pass

            result = verified_result or results_by_mode.get("rag") or results_by_mode.get("naive")
            if result is None:
                result = pipeline.run(request.query)
                result["hallucination_score"] = _compute_hallucination_score(result)
            return QueryResponse.model_validate(result)

        # Normal single-mode execution
        if request.mode == "fast":
            result = pipeline.run_fast_rag(request.query)
        elif request.mode == "naive":
            result = pipeline.run_naive(request.query)
        elif request.mode == "rag":
            result = pipeline.run_standard_rag(request.query)
        else:
            result = pipeline.run(request.query)
        result["hallucination_score"] = _compute_hallucination_score(result)
        return QueryResponse.model_validate(result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_three_modes_for_question(pipeline, question: str) -> None:
    """Run Naive, Standard RAG, Verified RAG for one question in parallel; append records."""
    mode_map = [
        ("naive", "Naive"),
        ("rag", "Standard RAG"),
        ("multi", "Verified RAG"),
    ]

    def run_one(internal_mode: str, label: str) -> tuple[str, str, dict]:
        if internal_mode == "naive":
            mode_result = pipeline.run_naive(question)
        elif internal_mode == "rag":
            mode_result = pipeline.run_standard_rag(question)
        else:
            mode_result = pipeline.run(question)
        return (internal_mode, label, mode_result)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_one, internal_mode, label): (internal_mode, label)
            for internal_mode, label in mode_map
        }
        for future in as_completed(futures):
            try:
                _internal_mode, label, mode_result = future.result()
                mode_result["hallucination_score"] = _compute_hallucination_score(
                    mode_result
                )
                record = _build_evaluation_record(question, label, mode_result)
                _append_evaluation_record(record)
            except Exception:
                pass


@app.post("/api/eval/bulk")
def api_eval_bulk(request: BulkEvalRequest):
    """
    Run each question through Naive, Standard RAG, and Verified RAG (same matrix per question).
    Returns processed count and the evaluation report so the UI can show it immediately.
    """
    questions = [q.strip() for q in request.questions if q and q.strip()]
    if not questions:
        return {
            "processed": 0,
            "failed": 0,
            "message": "No questions provided. Enter one question per line.",
            "report": None,
        }

    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    failed = 0
    for question in questions:
        try:
            _run_three_modes_for_question(pipeline, question)
        except Exception:
            failed += 1

    processed = len(questions) - failed

    with _eval_lock:
        records = list(evaluation_results)

    if not records:
        return {
            "processed": processed,
            "failed": failed,
            "message": "No results to report (all runs failed).",
            "report": None,
        }

    aggregates = _aggregate_evaluation_metrics(records)
    per_mode = aggregates["per_mode"]
    charts = _build_evaluation_charts(per_mode)
    report = {
        "has_data": True,
        "per_mode": per_mode,
        "overall": aggregates["overall"],
        "relative_reduction_vs_naive": aggregates["relative_reduction_vs_naive"],
        "relative_reduction_vs_standard": aggregates["relative_reduction_vs_standard"],
        "charts": charts,
    }

    # Persist latest aggregated report snapshot to disk for later reference.
    try:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {
        "processed": processed,
        "failed": failed,
        "message": f"Evaluated {processed} question(s). Report below.",
        "report": report,
    }


@app.get("/api/eval/report")
def api_evaluation_report():
    """
    Aggregate evaluation metrics for the current session and return
    per-mode metrics plus base64-encoded charts.
    """
    with _eval_lock:
        records = list(evaluation_results)

    if not records:
        return {
            "has_data": False,
            "message": "No evaluation data yet. Enable Evaluation Mode and ask some questions.",
        }

    aggregates = _aggregate_evaluation_metrics(records)
    per_mode = aggregates["per_mode"]
    charts = _build_evaluation_charts(per_mode)

    report = {
        "has_data": True,
        "per_mode": per_mode,
        "overall": aggregates["overall"],
        "relative_reduction_vs_naive": aggregates["relative_reduction_vs_naive"],
        "relative_reduction_vs_standard": aggregates["relative_reduction_vs_standard"],
        "charts": charts,
    }

    # Persist latest report snapshot to disk as well.
    try:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return report
