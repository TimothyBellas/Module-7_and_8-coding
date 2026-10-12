import httpx
import pytest
from streamlit.testing.v1 import AppTest

from conftest import PROJECT


def test_frontend_buttons_question_and_citation_display(monkeypatch):
    import api_client
    calls = []
    monkeypatch.setattr(api_client, "get_health", lambda: {
        "status": "healthy", "model": "llama3.2:1b", "message": "Connected"
    })
    monkeypatch.setattr(api_client, "get_stats", lambda: {"document_count": 18, "source_count": 6})
    def ingest():
        calls.append("ingest")
        return {"files_indexed": 6, "chunks_indexed": 18}
    monkeypatch.setattr(api_client, "ingest_documents", ingest)
    monkeypatch.setattr(api_client, "ask_question", lambda question: {
        "answer": "RAG uses retrieved context. [S1]",
        "sources": ["rag.txt"], "confidence": "high", "chunks_retrieved": 1,
        "chunks": [{"citation": "S1", "source": "rag.txt", "paragraph": 1,
                    "text": "RAG uses retrieved documents.", "distance": 0.1}],
    })
    app = AppTest.from_file(str(PROJECT / "frontend" / "app.py")).run()
    assert not app.exception
    app.sidebar.button[0].click().run()
    assert calls == ["ingest"]
    app.text_input[0].set_value("What is RAG?").run()
    ask = next(button for button in app.button if button.label == "Ask")
    ask.click().run()
    assert not app.exception
    assert any("[S1]" in item.value for item in app.markdown)
    assert any("rag.txt" in item.value for item in app.markdown)
    assert app.session_state["last_question"] == "What is RAG?"


def test_frontend_http_client_uses_backend_address(monkeypatch):
    import api_client
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs.get("json")))
        return httpx.Response(200, json={"answer": "ok"})
    monkeypatch.setattr(api_client, "API_URL", "http://backend:8000")
    monkeypatch.setattr(api_client.httpx, "request", request)
    assert api_client.ask_question("What is RAG?")["answer"] == "ok"
    assert calls == [("POST", "http://backend:8000/ask", {"question": "What is RAG?"})]


def test_frontend_displays_backend_errors(monkeypatch):
    import api_client
    monkeypatch.setattr(api_client.httpx, "request", lambda *args, **kwargs: httpx.Response(
        503, json={"detail": "Ollama is unavailable."}
    ))
    with pytest.raises(api_client.APIError, match="Ollama is unavailable"):
        api_client.ask_question("What is RAG?")
