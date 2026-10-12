"""Real persistent ChromaDB, deterministic test embeddings, and mock Ollama HTTP."""

from pathlib import Path
import re
import shutil
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))
sys.path.insert(0, str(PROJECT / "frontend"))

from config import Settings
from rag import RAGService


class TestEmbeddingFunction:
    """Small word-category vectors avoid model downloads in isolated tests."""

    __test__ = False

    def is_legacy(self):
        return True

    groups = [
        {"rag", "retrieval", "generation", "augmented"},
        {"docker", "compose", "container", "containers", "services"},
        {"fastapi", "api", "http", "endpoint", "endpoints"},
        {"chromadb", "embedding", "embeddings", "vectors", "distance"},
        {"ollama", "model", "models"},
        {"configuration", "environment", "settings", "variables"},
    ]

    def __call__(self, input):
        embeddings = []
        for text in input:
            words = set(re.findall(r"[a-z]+", text.lower()))
            embeddings.append([float(len(words & group)) for group in self.groups] + [0.1])
        return embeddings

    def name(self):
        return "test-word-categories"


@pytest.fixture
def settings(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(PROJECT / "backend" / "docs", docs)
    return Settings(
        _env_file=None,
        CHROMA_PATH=tmp_path / "database",
        DOCS_PATH=docs,
        OLLAMA_URL="http://ollama.test:11434",
    )


@pytest.fixture
def service(settings):
    return RAGService(settings, embedding_function=TestEmbeddingFunction())


@pytest.fixture
def api_client(service, settings, monkeypatch):
    from fastapi.testclient import TestClient
    import my_rag_api

    monkeypatch.setattr(my_rag_api, "RAGService", lambda configured: service)
    with TestClient(my_rag_api.create_app(settings)) as client:
        yield client
