"""FastAPI endpoints consumed by Streamlit, Swagger, and verification scripts."""

from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from config import Settings
from rag import RAGService
from schemas import (
    AskRequest, AskResponse, HealthResponse, IngestResponse, ReadyResponse,
    RootResponse, StatsResponse,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = RAGService(settings)
        yield

    app = FastAPI(title="Containerized RAG API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def service(request: Request) -> RAGService:
        return request.app.state.rag

    @app.get("/", response_model=RootResponse)
    def root():
        return RootResponse(
            message="RAG API is running.", docs="/docs", health="/health",
            model=settings.MODEL_NAME,
        )

    @app.get("/ready", response_model=ReadyResponse)
    def ready(rag: RAGService = Depends(service)):
        # Containers can start before the generation model has been pulled.
        try:
            rag.client.heartbeat()
            rag.collection.count()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="ChromaDB is unavailable.") from exc
        return ReadyResponse(status="ready")

    @app.get("/health", response_model=HealthResponse)
    def health(rag: RAGService = Depends(service)):
        count = 0
        try:
            rag.client.heartbeat()
            count = rag.collection.count()
            chroma_ok = True
        except Exception:
            chroma_ok = False
        ollama_ok, model_available = rag.ollama_status()
        healthy = chroma_ok and ollama_ok and model_available
        if not chroma_ok:
            message = "ChromaDB is unavailable. Check the backend logs."
        elif not ollama_ok:
            message = "Ollama is unavailable. Check the ollama service."
        elif not model_available:
            message = f"Pull the model: docker compose exec ollama ollama pull {settings.MODEL_NAME}"
        else:
            message = "ChromaDB and Ollama are connected; the model is available."
        return HealthResponse(
            status="healthy" if healthy else "degraded",
            chromadb="connected" if chroma_ok else "error",
            ollama="connected" if ollama_ok else "disconnected",
            model=settings.MODEL_NAME, model_available=model_available,
            document_count=count, message=message,
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats(rag: RAGService = Depends(service)):
        try:
            return rag.stats()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Cannot read ChromaDB statistics.") from exc

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(rag: RAGService = Depends(service)):
        return rag.ingest()

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, rag: RAGService = Depends(service)):
        return rag.ask(body.question)

    return app


app = create_app()
