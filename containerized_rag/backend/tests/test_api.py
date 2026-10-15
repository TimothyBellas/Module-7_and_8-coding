"""API contracts, guardrails, failure handling, and real persistent indexing."""

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.ollama_client import OllamaClient
from backend.rag import RAGService


def test_root_returns_200(api):
    client, _, _ = api
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_returns_200_with_expected_fields(api):
    client, _, _ = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "ready": True, "chromadb": "connected", "ollama": "connected",
        "model_available": True, "embedding_model_available": True,
    }


def test_stats_returns_document_count(api):
    client, _, _ = api
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["document_count"] == 0
    assert response.json()["chunk_count"] == 0
    assert response.json()["model"] == "llama3.2:1b"


@pytest.mark.parametrize("question", ["", "  \n\t  ", "x" * 2001])
def test_invalid_question_returns_422_without_model_calls(api, question):
    client, fake, _ = api
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 422
    assert fake.calls == []


def test_missing_question_returns_422(api):
    client, _, _ = api
    assert client.post("/ask", json={}).status_code == 422


def test_ask_empty_index_has_clear_fallback(api):
    client, fake, _ = api
    response = client.post("/ask", json={"question": "What is Python?"})
    assert response.status_code == 200
    data = response.json()
    assert "No documents" in data["answer"]
    assert data["confidence"] == "low" and data["grounded"] is False
    assert data["sources"] == []
    assert fake.calls == []


def test_ingest_stores_source_metadata_and_counts(api):
    client, _, service = api
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["document_count"] == 2
    assert response.json()["chunk_count"] == 2
    metadata = service.collection.get(include=["metadatas"])["metadatas"]
    assert {m["source"] for m in metadata} == {"python.txt", "streamlit.txt"}
    assert all(m["chunk_index"] == 0 for m in metadata)
    assert client.get("/stats").json()["document_count"] == 2


def test_ingest_is_idempotent_and_reuses_embeddings(api):
    client, fake, _ = api
    first = client.post("/ingest").json()
    calls = fake.count("/api/embed")
    second = client.post("/ingest").json()
    assert second["chunk_count"] == first["chunk_count"]
    assert second["chunks_embedded"] == 0
    assert second["chunks_removed"] == 0
    assert fake.count("/api/embed") == calls


def test_reindex_removes_deleted_document(api):
    client, _, service = api
    client.post("/ingest")
    (service.settings.docs_path / "streamlit.txt").unlink()
    data = client.post("/ingest").json()
    assert data["document_count"] == 1
    assert data["chunk_count"] == 1
    assert data["chunks_removed"] == 1


def test_reindex_replaces_changed_content(api):
    client, _, service = api
    client.post("/ingest")
    (service.settings.docs_path / "python.txt").write_text("Python lists are mutable.", encoding="utf-8")
    data = client.post("/ingest").json()
    stored = service.collection.get(include=["documents"])["documents"]
    assert data["chunks_embedded"] == 1 and data["chunks_removed"] == 1
    assert "Python lists are mutable." in stored
    assert "Python functions organize reusable code." not in stored


def test_embedding_failure_keeps_previous_index(api):
    client, fake, service = api
    client.post("/ingest")
    previous = set(service.collection.get(include=[])["ids"])
    (service.settings.docs_path / "python.txt").write_text("Python content changed.", encoding="utf-8")
    fake.fail_embeddings = True
    assert client.post("/ingest").status_code == 503
    assert set(service.collection.get(include=[])["ids"]) == previous


def test_empty_corpus_does_not_destroy_previous_index(api):
    client, _, service = api
    client.post("/ingest")
    for path in service.settings.docs_path.iterdir():
        path.unlink()
    assert client.post("/ingest").status_code == 400
    assert client.get("/stats").json()["document_count"] == 2


def test_answer_has_valid_citations_and_high_confidence(api):
    client, fake, _ = api
    client.post("/ingest")
    response = client.post("/ask", json={"question": "  What are Python functions?  "})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True and data["confidence"] == "high"
    assert data["sources"][0]["source"] == "python.txt"
    assert data["sources"][0]["citation"] == "S1"
    assert data["chunks_retrieved"] == len(data["sources"])
    prompt = next(payload for path, payload in fake.calls if path == "/api/chat")
    assert prompt["stream"] is False
    assert prompt["messages"][0]["role"] == "system"
    assert "untrusted" in prompt["messages"][0]["content"]
    assert "DOCUMENT EXCERPTS" in prompt["messages"][1]["content"]


def test_retained_weaker_match_gets_medium_confidence(api):
    client, _, service = api
    (service.settings.docs_path / "streamlit.txt").unlink()
    client.post("/ingest")
    data = client.post("/ask", json={"question": "medium match"}).json()
    assert data["confidence"] == "medium"
    assert data["sources"][0]["distance"] == pytest.approx(0.4, abs=1e-5)


def test_unrelated_question_abstains_without_chat_call(api):
    client, fake, _ = api
    client.post("/ingest")
    data = client.post("/ask", json={"question": "Who lives on the moon?"}).json()
    assert data["confidence"] == "low" and data["grounded"] is False
    assert data["sources"] == []
    assert fake.count("/api/chat") == 0


@pytest.mark.parametrize("answer", ["Unsupported answer.", "A claim [S99]."])
def test_missing_or_invented_citations_are_rejected(api, answer):
    client, fake, _ = api
    client.post("/ingest")
    fake.answer = answer
    data = client.post("/ask", json={"question": "What is Python?"}).json()
    assert data["grounded"] is False
    assert data["confidence"] == "low"
    assert "valid source citations" in data["answer"]


def test_model_abstention_is_not_marked_grounded(api):
    client, fake, _ = api
    client.post("/ingest")
    fake.answer = "I don't know based on the indexed documents."
    data = client.post("/ask", json={"question": "What is Python?"}).json()
    assert data["confidence"] == "low"
    assert data["grounded"] is False and data["sources"] == []


def test_ollama_outage_is_reported_as_degraded_health_and_503(api):
    client, fake, _ = api
    client.post("/ingest")
    fake.offline = True
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded" and health.json()["ready"] is False
    assert health.json()["ollama"] == "disconnected"
    assert client.post("/ask", json={"question": "What is Python?"}).status_code == 503


def test_missing_model_is_visible_in_health_and_ask(api):
    client, fake, _ = api
    client.post("/ingest")
    fake.models.remove("llama3.2:1b")
    health = client.get("/health").json()
    assert health["ollama"] == "connected" and health["model_available"] is False
    assert health["ready"] is False
    assert client.post("/ask", json={"question": "What is Python?"}).status_code == 503


def test_invalid_embedding_batch_returns_503(api):
    client, fake, _ = api
    fake.invalid_embeddings = True
    assert client.post("/ingest").status_code == 503


def test_index_survives_service_reconstruction(environment):
    settings, fake, original = environment
    original.ingest()
    before = original.stats()
    original.close()
    adapter = httpx.Client(base_url=str(settings.ollama_url), transport=httpx.MockTransport(fake.handle))
    restored = RAGService(settings, OllamaClient(settings, adapter))
    try:
        after = restored.stats()
        assert after.document_count == before.document_count == 2
        assert after.chunk_count == before.chunk_count == 2
    finally:
        restored.close()


def test_startup_automatically_ingests(environment):
    settings, _, service = environment
    settings = settings.model_copy(update={"auto_ingest": True})
    with TestClient(create_app(settings, service)) as client:
        assert client.get("/stats").json()["document_count"] == 2
