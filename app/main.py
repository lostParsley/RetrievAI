"""FastAPI entrypoint with the API routes and a tiny teaching UI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, get_settings
from app.rag_pipeline import RAGPipeline
from app.schemas import AskRequest, AskResponse, HealthResponse, IndexRequest, IndexResponse


settings = get_settings()
pipeline = RAGPipeline(settings)
app = FastAPI(title="Minimal RAG App")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the single-page teaching UI."""

    return FileResponse(Path(BASE_DIR / "templates" / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    """
    Indexes documents from a local folder.
    """
    try:
        result = pipeline.index_documents(request.docs_path)
        return IndexResponse(**result)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - kept simple for API callers
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    Asks a question and retrieves the most relevant chunks from the indexed documents.
    """
    try:
        result = pipeline.ask(request.question, request.top_k)
        return AskResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - kept simple for API callers
        raise HTTPException(status_code=500, detail=str(exc)) from exc
