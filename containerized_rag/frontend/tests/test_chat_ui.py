"""Interface behavior with controlled API responses and real Streamlit session state."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def ui(monkeypatch):
    """Simulate backend responses without launching servers or models."""
    state = {"documents": 2, "ingests": 0, "offline": False}
    monkeypatch.setenv("BACKEND_URL", "http://backend.test")

    def request(method, url, **kwargs):
        req = httpx.Request(method, url)
        if state["offline"]:
            raise httpx.ConnectError("Simulated outage", request=req)
        if url.endswith("/health"):
            data = {
                "ready": True, "chromadb": "connected", "ollama": "connected",
                "model_available": True, "embedding_model_available": True,
            }
        elif url.endswith("/stats"):
            data = {"document_count": state["documents"], "chunk_count": 4, "model": "llama3.2:1b"}
        elif url.endswith("/ingest"):
            state["documents"] = 3
            state["ingests"] += 1
            data = {"document_count": 3, "chunk_count": 6}
        elif url.endswith("/ask"):
            data = {
                "answer": "Python functions organize code. [S1]", "confidence": "high",
                "sources": [{"citation": "S1", "source": "python.txt", "chunk_index": 0,
                             "distance": 0.1, "text": "Python functions organize code."}],
            }
        else:
            raise AssertionError(f"Unexpected test request: {url}")
        return httpx.Response(200, json=data, request=req)

    monkeypatch.setattr(httpx, "request", request)
    return AppTest.from_file(str(APP)).run(), state


def test_chat_renders_sources_and_preserves_history(ui):
    app, _ = ui
    assert not app.exception
    app.chat_input[0].set_value("What is Python?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "[S1]" in app.chat_message[1].markdown[0].value
    assert len(app.expander) == 1
    assert "Sources" in app.expander[0].label
    app.run()
    assert len(app.chat_message) == 2
    assert len(app.session_state["messages"]) == 2


def test_reindex_updates_sidebar_count(ui):
    app, state = ui
    app.button[0].click().run()
    assert not app.exception
    assert state["ingests"] == 1
    assert app.metric[0].value == "3"


def test_clear_conversation_keeps_index(ui):
    app, state = ui
    app.chat_input[0].set_value("What is Python?").run()
    app.button[2].click().run()
    assert not app.exception
    assert len(app.chat_message) == 0
    assert state["documents"] == 2


def test_backend_outage_shows_error_without_crashing(ui):
    app, state = ui
    state["offline"] = True
    app.run()
    assert not app.exception
    assert any("unavailable" in error.value for error in app.error)
