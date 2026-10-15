"""Isolated real Chroma storage and a deterministic Ollama HTTP transport."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.ollama_client import OllamaClient
from backend.rag import RAGService


class FakeOllama:
    """Simulate only the HTTP dependency, keeping API and Chroma behavior real."""

    def __init__(self):
        self.models = ["llama3.2:1b", "nomic-embed-text:v1.5"]
        self.answer = "Python functions organize reusable code. [S1]"
        self.offline = False
        self.fail_embeddings = False
        self.invalid_embeddings = False
        self.calls: list[tuple[str, dict]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return known vectors so threshold tests have known cosine distances."""
        if self.offline:
            raise httpx.ConnectError("Simulated outage", request=request)
        payload = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.path, payload))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.models]})
        if payload.get("model") not in self.models:
            return httpx.Response(404, json={"error": "model not found"})
        if request.url.path == "/api/embed":
            if self.fail_embeddings:
                return httpx.Response(500, json={"error": "simulated embedding failure"})
            if self.invalid_embeddings:
                return httpx.Response(200, json={"embeddings": []})
            vectors = []
            for text in payload["input"]:
                text = text.lower()
                if "medium" in text:
                    vectors.append([0.6, 0.8, 0.0])
                elif "python" in text:
                    vectors.append([1.0, 0.0, 0.0])
                elif "streamlit" in text:
                    vectors.append([0.0, 1.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 1.0])
            return httpx.Response(200, json={"embeddings": vectors})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": self.answer}})
        return httpx.Response(404)

    def count(self, path: str) -> int:
        """Count expensive calls when checking abstention and idempotence."""
        return sum(url == path for url, _ in self.calls)


@pytest.fixture
def environment(tmp_path):
    """Build two source files and a unique persistent database per test."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "python.txt").write_text("Python functions organize reusable code.", encoding="utf-8")
    (docs / "streamlit.txt").write_text("Streamlit session state preserves a chat history.", encoding="utf-8")
    settings = Settings(
        _env_file=None, ollama_url="http://ollama.test:11434",
        docs_path=docs, chroma_path=tmp_path / "chroma", auto_ingest=False,
    )
    fake = FakeOllama()
    client = httpx.Client(base_url=str(settings.ollama_url), transport=httpx.MockTransport(fake.handle))
    service = RAGService(settings, OllamaClient(settings, client))
    return settings, fake, service


@pytest.fixture
def api(environment):
    """Exercise actual lifespan and routing using the isolated service."""
    settings, fake, service = environment
    with TestClient(create_app(settings, service)) as client:
        yield client, fake, service
