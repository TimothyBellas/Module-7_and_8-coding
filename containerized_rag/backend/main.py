"""FastAPI routes, dependency lifecycle, and HTTP error handling."""

from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .ollama_client import DependencyUnavailable
from .rag import RAGService
from .schemas import AskRequest, AskResponse, HealthResponse, IngestResponse, StatsResponse

logger = logging.getLogger(__name__)


def get_service(request: Request) -> RAGService:
    """Resolve the lifespan-owned service; tests can provide an isolated instance."""
    return request.app.state.rag


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    """Construct a configured application without contacting external services."""
    config = settings or Settings()
    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Index on startup in a worker thread and keep useful diagnostics on failure."""
        rag = service or RAGService(config)
        application.state.rag = rag
        try:
            if config.auto_ingest:
                try:
                    result = await run_in_threadpool(rag.ingest)
                    logger.info("Indexed %s files / %s chunks", result.document_count, result.chunk_count)
                except Exception:
                    logger.exception("Automatic ingestion failed; retry with POST /ingest")
            yield
        finally:
            rag.close()

    application = FastAPI(
        title="Containerized RAG Chatbot", version="1.0.0",
        description="Local document retrieval and Ollama generation.", lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware, allow_origins=config.cors_origins,
        allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @application.exception_handler(DependencyUnavailable)
    async def dependency_error(_request: Request, exc: DependencyUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception):
        logger.error("Unhandled API error", exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(status_code=500, content={"detail": "Internal error. Check backend logs."})

    @application.get("/", tags=["status"])
    def root() -> dict[str, str]:
        """Confirm the API is running and point to interactive documentation."""
        return {"message": "Containerized RAG API is running.", "docs": "/docs"}

    @application.get("/health", response_model=HealthResponse, tags=["status"])
    def health(rag: RAGService = Depends(get_service)) -> HealthResponse:
        """Always return liveness; inspect ready/status for dependency health."""
        return rag.health()

    @application.get("/stats", response_model=StatsResponse, tags=["status"])
    def stats(rag: RAGService = Depends(get_service)) -> StatsResponse:
        """Return model information and indexed document/chunk counts."""
        return rag.stats()

    @application.post("/ingest", response_model=IngestResponse, tags=["documents"])
    def ingest(rag: RAGService = Depends(get_service)) -> IngestResponse:
        """Synchronize configured text documents with persistent ChromaDB."""
        try:
            return rag.ingest()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=503, detail="Cannot read the configured document folder.") from exc

    @application.post("/ask", response_model=AskResponse, tags=["chat"])
    def ask(payload: AskRequest, rag: RAGService = Depends(get_service)) -> AskResponse:
        """Answer a validated question with sources and retrieval confidence."""
        return rag.ask(payload.question)

    return application


app = create_app()
