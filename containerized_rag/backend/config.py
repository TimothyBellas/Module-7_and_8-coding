"""Validated environment configuration shared by the API and RAG pipeline."""

from pathlib import Path

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Read .env for local development; container environment takes precedence."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    ollama_url: AnyHttpUrl
    model_name: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text:v1.5"
    embedding_document_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "
    chroma_path: Path = BACKEND_DIR / "chroma_data"
    collection_name: str = "course_notes"
    docs_path: Path = BACKEND_DIR / "docs"
    max_results: int = Field(default=3, ge=1, le=8)
    chunk_size: int = Field(default=1000, ge=200, le=3000)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_batch_size: int = Field(default=16, ge=1, le=64)
    confidence_threshold: float = Field(default=0.55, gt=0, le=2)
    high_confidence_threshold: float = Field(default=0.30, ge=0, le=2)
    request_timeout: float = Field(default=300, gt=0)
    health_timeout: float = Field(default=3, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=2)
    context_window: int = Field(default=8192, ge=2048)
    max_answer_tokens: int = Field(default=512, ge=32, le=2048)
    auto_ingest: bool = True
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_chunking_and_thresholds(self) -> "Settings":
        """Reject combinations that would make chunking or scoring ambiguous."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.high_confidence_threshold >= self.confidence_threshold:
            raise ValueError("HIGH_CONFIDENCE_THRESHOLD must be below CONFIDENCE_THRESHOLD")
        return self
