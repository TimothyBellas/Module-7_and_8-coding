import httpx
import pytest
from fastapi import HTTPException

from conftest import TestEmbeddingFunction
from rag import RAGService


def mock_ollama(monkeypatch, answer="RAG combines retrieval and generation. [S1]", status=200):
    def post(url, *, json, timeout):
        assert url.endswith("/api/chat")
        assert json["stream"] is False
        assert "[S1]" in json["messages"][1]["content"]
        assert "ONLY" in json["messages"][0]["content"]
        return httpx.Response(
            status, json={"message": {"content": answer}},
            request=httpx.Request("POST", url),
        )
    monkeypatch.setattr("rag.httpx.post", post)


def test_ingest_updates_and_persists_without_duplicates(service, settings):
    first = service.ingest()
    assert first.files_indexed == 6
    assert first.chunks_indexed == first.document_count == 18
    assert service.ingest().document_count == 18
    assert RAGService(settings, TestEmbeddingFunction()).collection.count() == 18

    (settings.DOCS_PATH / "rag.txt").write_text("A replacement RAG paragraph.", encoding="utf-8")
    (settings.DOCS_PATH / "ollama.txt").unlink()
    changed = service.ingest()
    assert changed.files_indexed == 5
    assert changed.document_count == 13
    stored = service.collection.get(include=["documents", "metadatas"])
    assert all(item["source"] != "ollama.txt" for item in stored["metadatas"])
    assert "A replacement RAG paragraph." in stored["documents"]


def test_empty_reindex_keeps_previous_database(service, settings):
    service.ingest()
    for path in settings.DOCS_PATH.glob("*.txt"):
        path.unlink()
    with pytest.raises(HTTPException) as error:
        service.ingest()
    assert error.value.status_code == 400
    assert service.collection.count() == 18


def test_api_before_ingest_and_question_validation(api_client):
    assert api_client.get("/").json()["model"] == "llama3.2:1b"
    assert api_client.get("/ready").status_code == 200
    assert "Re-index Documents" in api_client.post("/ask", json={"question": "What is RAG?"}).json()["answer"]
    for question in ["", "   ", "x" * 2001]:
        assert api_client.post("/ask", json={"question": question}).status_code == 422


def test_api_ingest_retrieval_citations_and_stats(api_client, monkeypatch):
    mock_ollama(monkeypatch)
    assert api_client.post("/ingest").json()["document_count"] == 18
    response = api_client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert "[S1]" in body["answer"]
    assert body["sources"]
    assert body["chunks_retrieved"] == len(body["chunks"]) == 3
    assert body["confidence"] == "high"
    stats = api_client.get("/stats").json()
    assert stats["document_count"] == 18
    assert stats["source_count"] == 6
    assert stats["model"] == "llama3.2:1b"


def test_no_relevant_chunks_does_not_call_ollama(service, monkeypatch):
    service.ingest()
    service.settings.CONFIDENCE_THRESHOLD = 0.00001
    def unexpected_call(*args, **kwargs):
        raise AssertionError("No generation should occur after filtering all chunks")
    monkeypatch.setattr("rag.httpx.post", unexpected_call)
    answer = service.ask("RAG Docker FastAPI ChromaDB Ollama configuration")
    assert answer.confidence == "low"
    assert answer.chunks_retrieved == 0
    assert answer.sources == []


@pytest.mark.parametrize("answer", ["A claim with no source.", "An invented citation. [S99]"])
def test_missing_or_invented_citations_use_exact_excerpt(service, monkeypatch, answer):
    service.ingest()
    mock_ollama(monkeypatch, answer)
    result = service.ask("What is RAG?")
    assert result.chunks[0].text in result.answer
    assert "[S1]" in result.answer and "[S99]" not in result.answer
    assert result.sources == [result.chunks[0].source]


@pytest.mark.parametrize("mode, expected", [("unreachable", 503), ("missing", 503), ("timeout", 504), ("invalid", 502)])
def test_ollama_failures_are_readable_api_errors(api_client, monkeypatch, mode, expected):
    api_client.post("/ingest")
    def post(url, **kwargs):
        if mode == "unreachable":
            raise httpx.ConnectError("unreachable")
        if mode == "timeout":
            raise httpx.ReadTimeout("slow model")
        if mode == "missing":
            return httpx.Response(404, request=httpx.Request("POST", url))
        return httpx.Response(200, content=b"not-json", request=httpx.Request("POST", url))
    monkeypatch.setattr("rag.httpx.post", post)
    response = api_client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == expected
    assert response.json()["detail"]


@pytest.mark.parametrize("available, status", [(True, "healthy"), (False, "degraded")])
def test_health_reports_model_availability(api_client, monkeypatch, available, status):
    models = [{"name": "llama3.2:1b"}] if available else []
    monkeypatch.setattr("rag.httpx.get", lambda url, **kwargs: httpx.Response(
        200, json={"models": models}, request=httpx.Request("GET", url)
    ))
    body = api_client.get("/health").json()
    assert body["status"] == status
    assert body["ollama"] == "connected"
    assert body["model_available"] is available


def test_health_reports_disconnected_ollama(api_client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr("rag.httpx.get", unavailable)
    body = api_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"] == "disconnected"


def test_cors_accepts_streamlit_origin(api_client):
    response = api_client.options("/ask", headers={
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
