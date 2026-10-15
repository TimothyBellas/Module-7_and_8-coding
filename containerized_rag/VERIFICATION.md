# Verification record

Prepared on October 15, 2026.

## Completed in the build environment

| Check | Result |
| --- | --- |
| Application dependencies | Installed successfully in a clean virtual environment |
| Dependency consistency | `python -m pip check`: no broken requirements |
| Automated tests | **36 passed**, running Python 3.12.14 |
| Required endpoints | `/`, `/health`, and `/stats` contracts passed |
| Ingestion and ChromaDB | Real temporary persistent storage; metadata, idempotence, changed/deleted files, failure preservation, and reconstruction passed |
| RAG mechanics | Controlled Ollama HTTP responses; distance filtering, prompt structure, confidence, missing/invalid citations, and abstention passed |
| API failure handling | Blank/overlong questions, offline Ollama, missing models, invalid embeddings, and empty corpus passed |
| Streamlit | Chat submission, source expander, history rerun, re-index counts, clear history, and outage handling passed |
| Python 3.11 compatibility preflight | Pinned application dependencies resolved successfully for Linux Python 3.11 binary distributions |
| Source/configuration checks | Python source compiled; Compose and CI YAML parsed and required services, jobs, volumes, settings, and build paths checked |
| Startup shell script | POSIX shell syntax check passed |
| Corpus | Six UTF-8 documents, 6,586 words; printable PDF has 12 pages and was rendered for visual review |

The Python 3.11 preflight resolves packages; it does not execute them in a Python 3.11 container. The Chroma persistence test reconstructs the service using the same directory; Docker volume persistence still requires the separate container test.

## Remaining checks on the target computer

Docker is not installed in the build environment. The following results are **not claimed**:

- Actual Docker image builds or `docker compose config` validation by the Docker CLI.
- A successful three-service Compose startup and model download.
- Semantic retrieval and generation quality with the real Ollama models.
- Browser access through published ports and the complete re-index/question/citation flow.
- Named-volume persistence across container removal and recreation.
- A successful hosted GitHub Actions run.

The project includes commands for each check in `README.md`, an optional live `scripts/verify_stack.py` check, and a CI workflow that builds both images. Complete the container walkthrough and confirm green GitHub Actions jobs before reporting the entire deployment as verified.
