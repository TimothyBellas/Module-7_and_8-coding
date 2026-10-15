"""Streamlit chat client with session history, source excerpts, and index controls."""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
import streamlit as st

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("FRONTEND_TIMEOUT", "600"))
STATUS_TIMEOUT = float(os.environ.get("STATUS_TIMEOUT", "5"))

st.set_page_config(page_title="Document Chat", page_icon="📚", layout="centered")


def api_request(method: str, path: str, *, timeout: float = REQUEST_TIMEOUT, **kwargs) -> dict:
    """Call the configured API and preserve its helpful error messages."""
    try:
        response = httpx.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise RuntimeError(f"{detail}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("The request took too long. The model may still be warming up; try again.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("The chatbot service is unavailable. Check that the application is running.") from exc


def render_answer(message: dict) -> None:
    """Show the answer, its retrieval confidence, and inspectable evidence."""
    st.markdown(message["content"])
    if "confidence" in message:
        st.caption(f"Retrieval confidence: {message['confidence'].title()}")
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Sources · {len(sources)} retrieved excerpts"):
            for source in sources:
                st.markdown(f"**[{source['citation']}] {source['source']}**")
                st.caption(
                    f"Chunk {source['chunk_index']} · cosine distance {source['distance']:.3f}"
                )
                st.text(source["text"])
                st.divider()


def main() -> None:
    """Render sidebar controls and handle a single submitted chat turn."""
    if not BACKEND_URL:
        st.error("Set BACKEND_URL in your environment or .env file before starting the interface.")
        st.stop()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("📚 Document Chat")
        st.caption("Your notes. Local answers.")
        st.divider()
        try:
            health = api_request("GET", "/health", timeout=STATUS_TIMEOUT)
            if health["ready"]:
                st.success("Ready to chat")
            else:
                st.warning("Some services need attention")
            st.write(f"Document store: {health['chromadb']}")
            st.write(f"Language model: {health['ollama']}")
            if not health["model_available"] or not health["embedding_model_available"]:
                st.caption("Waiting for the configured models to become available.")
        except RuntimeError as exc:
            st.error(str(exc))
        try:
            stats = api_request("GET", "/stats", timeout=STATUS_TIMEOUT)
            left, right = st.columns(2)
            left.metric("Documents", stats["document_count"])
            right.metric("Chunks", stats["chunk_count"])
            st.caption(f"Answer model: {stats['model']}")
        except RuntimeError:
            st.caption("Document count is temporarily unavailable.")
        if notice := st.session_state.pop("index_notice", None):
            st.success(notice)
        if st.button("Re-index Documents", use_container_width=True):
            try:
                with st.spinner("Updating the document index…"):
                    result = api_request("POST", "/ingest")
                st.session_state.index_notice = (
                    f"Indexed {result['document_count']} documents ({result['chunk_count']} chunks)."
                )
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        if st.button("Refresh status", use_container_width=True):
            st.rerun()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Answers use the indexed notes. Open Sources to check the supporting text.")

    st.title("Ask your documents")
    st.caption("Explore Python, APIs, Streamlit, RAG, Docker, and testing.")
    if not st.session_state.messages:
        st.info("Try: How do named volumes preserve data when containers are recreated?")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_answer(message)
    if question := st.chat_input("Ask a question about your notes", max_chars=2000):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding evidence and writing an answer…"):
                    result = api_request("POST", "/ask", json={"question": question})
                message = {
                    "role": "assistant", "content": result["answer"],
                    "confidence": result["confidence"], "sources": result["sources"],
                }
            except RuntimeError as exc:
                message = {"role": "assistant", "content": str(exc), "sources": []}
            st.session_state.messages.append(message)
            render_answer(message)


if __name__ == "__main__":
    main()
