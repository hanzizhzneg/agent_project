import logging
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.middleware import RequestContextMiddleware
from app.rag.agentic_graph import build_agentic_rag_graph
from app.rag.loader import list_supported_files, load_document_file, split_documents
from app.rag.vectorstore import create_incremental_writer, load_vector_store
from app.schemas import (
    AskRequest,
    AskResponse,
    IngestAsyncResponse,
    IngestRequest,
    IngestResponse,
    IngestTaskStatusResponse,
)
from app.security import require_api_key

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("enterprise-qa-agentic")
task_lock = threading.Lock()
ingest_tasks: dict[str, dict] = {}

app = FastAPI(
    title="Enterprise QA Agentic RAG API",
    version="2.0.0",
    description="LangGraph-based Agentic RAG with Ollama and Chroma/FAISS",
)
web_dir = Path(__file__).resolve().parent.parent / "web"
app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/ready")
def ready():
    return {"status": "ready", "vector_db_path": str(settings.vector_db_path)}


@app.get("/")
def home_page():
    return FileResponse(str(web_dir / "index.html"))


@app.get("/test")
def test_page():
    return FileResponse(str(web_dir / "test.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, _: None = Depends(require_api_key)):
    docs_path = Path(req.docs_dir).resolve() if req.docs_dir else settings.knowledge_base_path
    return _ingest_docs(
        docs_path=docs_path,
        store_type=req.store_type,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )


@app.post("/ingest_async", response_model=IngestAsyncResponse)
def ingest_async(req: IngestRequest, _: None = Depends(require_api_key)):
    docs_path = Path(req.docs_dir).resolve() if req.docs_dir else settings.knowledge_base_path
    task_id = str(uuid.uuid4())
    with task_lock:
        ingest_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "detail": "Task queued.",
            "result": None,
        }

    def run_task():
        try:
            _update_task(task_id, "running", 5, "start", f"Start ingest from {docs_path}")
            result = _ingest_docs(
                docs_path=docs_path,
                store_type=req.store_type,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
                progress_callback=lambda p, s, d: _update_task(task_id, "running", p, s, d),
            )
            _update_task(task_id, "completed", 100, "completed", "Ingest completed.", result)
        except Exception as exc:
            _update_task(task_id, "failed", 100, "failed", str(exc))

    threading.Thread(target=run_task, daemon=True).start()
    return IngestAsyncResponse(task_id=task_id, status="queued", message="Async ingest started.")


@app.get("/ingest_tasks/{task_id}", response_model=IngestTaskStatusResponse)
def ingest_task_status(task_id: str, _: None = Depends(require_api_key)):
    with task_lock:
        task = ingest_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return IngestTaskStatusResponse(**task)


def _update_task(
    task_id: str,
    status: str,
    progress: int,
    stage: str,
    detail: str,
    result: IngestResponse | None = None,
):
    with task_lock:
        if task_id not in ingest_tasks:
            return
        ingest_tasks[task_id].update(
            {
                "status": status,
                "progress": progress,
                "stage": stage,
                "detail": detail,
                "result": result.model_dump() if result else ingest_tasks[task_id].get("result"),
            }
        )


def _ingest_docs(
    docs_path: Path,
    store_type: str,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback=None,
) -> IngestResponse:
    if store_type not in {"chroma", "faiss"}:
        raise HTTPException(status_code=400, detail="store_type must be chroma or faiss")
    if not docs_path.exists():
        raise HTTPException(status_code=400, detail=f"docs_dir not found: {docs_path}")

    if progress_callback:
        progress_callback(10, "discovering_files", "Scanning supported files.")
    logger.info("ingest start: docs_path=%s store_type=%s", docs_path, store_type)
    files = list_supported_files(docs_path)
    logger.info("ingest stage: discovered_files=%s", len(files))
    if not files:
        raise HTTPException(
            status_code=400, detail="No supported documents found (.pdf/.docx/.doc)."
        )

    try:
        add_docs, finalize = create_incremental_writer(
            store_type=store_type, base_dir=settings.vector_db_path
        )
        docs_count = 0
        chunks_count = 0
        total = len(files)
        for idx, file_path in enumerate(files, start=1):
            docs = load_document_file(file_path)
            docs_count += len(docs)
            chunks = split_documents(docs, chunk_size, chunk_overlap)
            chunks_count += len(chunks)
            add_docs(chunks)
            if progress_callback:
                progress = 10 + int(80 * idx / total)
                progress_callback(
                    progress,
                    "processing_file",
                    f"Processed {idx}/{total}: {file_path.name}",
                )
        finalize()
    except Exception as exc:
        logger.exception("ingest failed while building vector store")
        raise HTTPException(
            status_code=504,
            detail=(
                "Vector build timed out or failed while calling Ollama. "
                "Check Ollama service/model status and retry."
            ),
        ) from exc
    if progress_callback:
        progress_callback(95, "finalizing", "Vector store persisted.")
    logger.info(
        "ingest finished: store_type=%s docs=%s chunks=%s",
        store_type,
        docs_count,
        chunks_count,
    )
    return IngestResponse(
        message="Ingestion completed.",
        store_type=store_type,
        docs_count=docs_count,
        chunks_count=chunks_count,
    )


@app.post("/upload_and_ingest", response_model=IngestResponse)
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    store_type: str = Form("chroma"),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(120),
    _: None = Depends(require_api_key),
):
    allowed = {".pdf", ".docx", ".doc"}
    saved = 0
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in allowed:
            continue
        target = settings.knowledge_base_path / Path(upload.filename).name
        with target.open("wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        saved += 1

    if saved == 0:
        raise HTTPException(
            status_code=400,
            detail="No supported files uploaded (.pdf/.docx/.doc).",
        )

    return _ingest_docs(
        docs_path=settings.knowledge_base_path,
        store_type=store_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _: None = Depends(require_api_key)):
    try:
        vector_store = load_vector_store(req.store_type, settings.vector_db_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    retriever = vector_store.as_retriever(search_kwargs={"k": req.top_k})
    graph = build_agentic_rag_graph(retriever)
    try:
        final_state = graph.invoke(
            {
                "question": req.question,
                "query": req.question,
                "top_k": req.top_k,
                "max_iterations": req.max_iterations,
                "iteration": 1,
                "retrieved_docs": [],
                "relevance_decision": "",
                "relevance_reason": "",
                "trace": [],
                "answer": "",
            }
        )
    except Exception as exc:
        logger.exception("ask failed while retrieving/generating answer")
        msg = str(exc)
        if "Error loading hnsw index" in msg or "hnsw segment reader" in msg:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Vector index is corrupted or incomplete (Chroma HNSW load failed). "
                    "Please remove the current vector store directory and re-run /ingest."
                ),
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="Ask failed due to vector retrieval or model generation error.",
        ) from exc

    docs = final_state.get("retrieved_docs", [])
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "n/a"),
        }
        for doc in docs
    ]
    return AskResponse(
        question=req.question,
        final_query=final_state["query"],
        answer=final_state["answer"],
        iterations_used=final_state["iteration"],
        trace=final_state["trace"],
        sources=sources,
    )
