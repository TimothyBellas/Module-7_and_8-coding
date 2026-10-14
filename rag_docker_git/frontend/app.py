"""Ask questions about indexed documents through the FastAPI RAG backend."""

import streamlit as st

from api_client import APIError, ask_question, get_health, get_stats, ingest_documents

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚", layout="wide")
st.title("RAG Study Assistant")
st.write("Index your documents, ask a question, and inspect the sources behind the answer.")

with st.sidebar:
    st.header("Documents")
    if st.button("Re-index Documents", type="primary", use_container_width=True):
        try:
            with st.spinner("Indexing documents… The first run downloads the embedding model."):
                result = ingest_documents()
            st.success(f"Indexed {result['files_indexed']} files and {result['chunks_indexed']} chunks.")
        except APIError as exc:
            st.error(str(exc))
    st.button("Refresh Status", use_container_width=True)
    try:
        health = get_health()
        stats = get_stats()
        st.metric("Indexed chunks", stats["document_count"])
        st.metric("Source files", stats["source_count"])
        st.caption(f"Model: {health['model']}")
        if health["status"] == "healthy":
            st.success("All services connected")
        else:
            st.warning(health["message"])
    except APIError as exc:
        st.warning(str(exc))

st.caption("Try: What is RAG? • How do Docker Compose services communicate?")
with st.form("question_form"):
    question = st.text_input("Your question", placeholder="What is RAG?")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving context and generating an answer…"):
                result = ask_question(question.strip())
            st.session_state["last_answer"] = result
            st.session_state["last_question"] = question.strip()
        except APIError as exc:
            st.error(str(exc))

if "last_answer" in st.session_state:
    result = st.session_state["last_answer"]
    st.subheader("Answer")
    st.caption(st.session_state["last_question"])
    st.markdown(result["answer"])
    st.caption(
        f"Retrieval confidence: {result['confidence']} · "
        f"Chunks retrieved: {result['chunks_retrieved']}"
    )
    if result["sources"]:
        st.write("**Cited source files:** " + ", ".join(result["sources"]))
    if result["chunks"]:
        with st.expander("View retrieved passages and citation labels"):
            for chunk in result["chunks"]:
                st.markdown(
                    f"**[{chunk['citation']}] {chunk['source']}** — "
                    f"paragraph {chunk['paragraph']} · distance {chunk['distance']:.3f}"
                )
                st.write(chunk["text"])
