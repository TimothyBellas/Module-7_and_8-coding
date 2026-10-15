"""Public request and response contracts for the API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class AskRequest(BaseModel):
    """A standalone question about the indexed documents."""

    question: Question


class Source(BaseModel):
    """A retrieved chunk with the label used for inline citations."""

    citation: str
    source: str
    chunk_index: int
    distance: float
    text: str


class AskResponse(BaseModel):
    """A grounded answer or an explicit abstention."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    chunks_retrieved: int
    model: str
    grounded: bool


class IngestResponse(BaseModel):
    """Index size after synchronizing the on-disk document corpus."""

    message: str
    document_count: int
    chunk_count: int
    chunks_embedded: int
    chunks_removed: int


class StatsResponse(BaseModel):
    """Count source files separately from their indexed chunks."""

    document_count: int
    chunk_count: int
    model: str
    embedding_model: str
    collection: str
    max_results: int
    confidence_threshold: float


class HealthResponse(BaseModel):
    """HTTP liveness plus dependency readiness, even when a dependency is down."""

    status: Literal["ok", "degraded"]
    ready: bool
    chromadb: Literal["connected", "disconnected"]
    ollama: Literal["connected", "disconnected"]
    model_available: bool
    embedding_model_available: bool
