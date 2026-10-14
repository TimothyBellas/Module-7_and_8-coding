"""Pydantic models for every request and response exposed by the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)


class RetrievedChunk(BaseModel):
    citation: str
    source: str
    paragraph: int
    text: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: Literal["high", "medium", "low"]
    chunks_retrieved: int
    chunks: list[RetrievedChunk]
    model: str


class IngestResponse(BaseModel):
    message: str
    files_indexed: int
    chunks_indexed: int
    document_count: int


class StatsResponse(BaseModel):
    document_count: int
    source_count: int
    model: str
    embedding_model: str
    max_results: int
    confidence_threshold: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    chromadb: Literal["connected", "error"]
    ollama: Literal["connected", "disconnected"]
    model: str
    model_available: bool
    document_count: int
    message: str


class ReadyResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    message: str
    docs: str
    health: str
    model: str
