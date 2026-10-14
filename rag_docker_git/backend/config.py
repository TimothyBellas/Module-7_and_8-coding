"""Read runtime configuration from environment variables or a local .env file."""

from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OLLAMA_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama3.2:1b"
    CHROMA_PATH: Path = Path("./chroma_data")
    DOCS_PATH: Path = Path(__file__).resolve().parent / "docs"
    COLLECTION_NAME: str = "rag_documents"
    MAX_RESULTS: int = Field(default=3, ge=1, le=20)
    CONFIDENCE_THRESHOLD: float = Field(default=1.0, gt=0, le=2)
    OLLAMA_TIMEOUT: float = Field(default=180, gt=0)
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("OLLAMA_URL")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OLLAMA_URL must be an http or https URL")
        return value.rstrip("/")

    @field_validator("MODEL_NAME", "COLLECTION_NAME")
    @classmethod
    def reject_blank_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Model and collection names cannot be blank")
        return value
