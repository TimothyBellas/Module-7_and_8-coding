# Validation record

Prepared on October 12, 2026.

## Completed in the preparation environment

- Installed the project dependencies under **Python 3.11.17**, matching the
  Python 3.11 base used by both Dockerfiles.
- **25 automated checks passed** with `python -m pytest`.
- Backend checks used **real persistent ChromaDB**, deterministic test
  embeddings, and **mocked Ollama HTTP responses**. They covered indexing six
  files / 18 paragraphs, repeat indexing, changed and deleted documents,
  database reopening, empty input protection, question validation, source
  labels, retrieval filtering, API errors, health reporting, and CORS.
- Streamlit's **AppTest** clicked **Re-index Documents**, submitted a question,
  and verified answer and source-label display. Its backend calls were mocked.
- Configuration checks covered `.env` changes, environment precedence,
  invalid settings, three-service configuration, storage, startup dependencies,
  and `.env` exclusions in the Git and Docker ignore files.
- The Compose YAML passed validation against the **official Compose
  specification JSON schema**.
- A separate API smoke check downloaded and used the **real Chroma ONNX
  all-MiniLM-L6-v2 embedding model**. It indexed the sample documents and
  retrieved the expected top source for both questions below:

  | Question | Top source | Cosine distance |
  | --- | --- | --- |
  | What is RAG? | `rag.txt` | approximately 0.531 |
  | How do Docker Compose services communicate? | `docker_compose.txt` | approximately 0.248 |

  This smoke check verified API retrieval, source-citation output, and repeated
  indexing. **Its Ollama response was mocked; it did not test live generation.**
- Python source files passed syntax compilation.

The dependency check found that ChromaDB 1.0.12 requires FastAPI 0.115.9; the
project uses that compatible version. PostHog is pinned to 3.25.0 because this
Chroma release uses the older client interface. One test-client deprecation
warning appeared; all 25 checks passed.

## Still required on the computer running Docker

Docker was unavailable in the preparation environment. Therefore the following
have **not** been claimed as completed:

1. Docker image builds and actual container startup.
2. Docker Compose network communication between the three containers.
3. Pulling `llama3.2:1b` inside the Ollama container and generating with that
   actual model.
4. Clicking through the running browser interface at `localhost:8501`.
5. Confirming the live `/health` endpoint and named-volume persistence after
   `docker compose down` / `up`.

Follow the README startup steps, then run:

```powershell
docker compose exec -T backend python verify_stack.py
```

That script checks the actual services and installed model. To keep evidence,
run `powershell -ExecutionPolicy Bypass -File .\Verify-Stack.ps1`, and use the
README's persistence check. A successful script run plus the manual browser
flow establishes the remaining end-to-end requirements.

## Optional local test commands

Local tests require Python 3.11 and dependency installation; they are not needed
just to start the project with Docker. In the root project folder on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -r frontend\requirements.txt -r tests\requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

These isolated tests do not download Ollama or the embedding model. Passing
them alone does not prove that the Docker stack or live model is running.
