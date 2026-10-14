"""Required CI endpoint checks using temporary ChromaDB storage."""

import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from my_rag_api import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        CHROMA_PATH=tmp_path / "chroma_data",
        OLLAMA_URL="http://ollama.test:11434",
        MODEL_NAME="llama3.2:1b",
    )
    # No ingestion occurs in these checks, so no embedding model is downloaded.
    # Entering TestClient as a context manager starts the application's lifespan.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_root_returns_200(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "RAG API is running."


def test_health_returns_200_with_expected_fields(client, monkeypatch):
    def mock_ollama_tags(url, **kwargs):
        assert url == "http://ollama.test:11434/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3.2:1b"}]},
            request=httpx.Request("GET", url),
        )

    # Check the real health route and database while replacing external HTTP.
    monkeypatch.setattr("rag.httpx.get", mock_ollama_tags)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert {
        "status", "chromadb", "ollama", "model", "model_available",
        "document_count", "message",
    } <= body.keys()
    assert body["status"] == "healthy"
    assert body["chromadb"] == "connected"
    assert body["ollama"] == "connected"
    assert body["model"] == "llama3.2:1b"
    assert body["model_available"] is True
    assert body["document_count"] == 0


def test_stats_returns_200_with_document_count(client):
    response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()
    assert "document_count" in body
    assert isinstance(body["document_count"], int)
    assert body["document_count"] == 0
