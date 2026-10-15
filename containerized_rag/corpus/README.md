# Printable corpus

`course_notes.pdf` is a 12-page printable reference containing the educational content of the six source documents in `backend/docs/`.

The six original UTF-8 text files total 6,586 whitespace-delimited words, including titles and source-note labels. They are the actual ingestible corpus. This PDF stays outside `DOCS_PATH` to avoid indexing duplicate text.

Each document occupies two printed pages. Topics are Python, HTTP/FastAPI, Streamlit chat, RAG/ChromaDB, Docker/configuration, and testing/CI.

The PDF is a reference copy of the included starter corpus. Re-indexing the chatbot updates ChromaDB from the text files; it does not regenerate this PDF.
