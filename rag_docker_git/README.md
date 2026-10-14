# Three-service RAG application

A FastAPI backend retrieves paragraph chunks from persistent ChromaDB and asks
Ollama's `llama3.2:1b` model to answer with citations. A Streamlit frontend lets
you re-index documents, ask questions, and inspect the retrieved passages.

## GitHub Actions CI

The workflow in `.github/workflows/ci.yml` runs on pushes to `main` and on
pull requests. It also supports a manual run from the Actions tab.

- **test:** sets up Python 3.11, installs the application and test dependencies,
  and runs `python -m pytest backend/tests tests`. This includes the three
  required checks in `backend/tests/test_api.py`: `/`, `/health`, and `/stats`.
- **docker:** builds `rag-backend:ci` from `backend/Dockerfile` and
  `rag-frontend:ci` from `frontend/Dockerfile`, then uses `docker image inspect`
  to confirm both images exist. A failed build or missing image fails the job.

Tests use temporary ChromaDB directories and mock external Ollama responses.
They do not require a running Ollama server, a pulled model, or a committed
`.env` file. The existing configuration test reads the tracked `.env.example`.
The workflow builds the Docker images; it does not start the three-service
stack or publish images to a registry.

Follow [CI-SETUP.md](CI-SETUP.md) to push the project and verify the two jobs
in GitHub's Actions tab. The assignment's final GitHub verification requires
an actual successful Actions run; see [VALIDATION.md](VALIDATION.md) for the
checks completed while preparing this version.

## Start on Windows

1. Extract the ZIP. Open the extracted `rag-docker` folder in VS Code.
2. Open **Docker Desktop** and wait until its engine is running. Use Linux
   containers. You do not need to install Python or Ollama on Windows for this
   project; both run inside containers.
3. Open a PowerShell terminal in that folder. Make sure `docker-compose.yml`
   is in the current folder, then run:

   ```powershell
   docker info
   docker compose up --build
   ```

   Leave this terminal running. The first build installs dependencies and can
   take several minutes. Compose starts Ollama first, then the backend, then
   the frontend when their health checks pass.

4. Open a **second** terminal in the same folder and download the model:

   ```powershell
   docker compose exec ollama ollama pull llama3.2:1b
   docker compose ps
   ```

5. Visit **http://localhost:8501**. Click **Re-index Documents**. Wait for the
   success message. The first indexing run also downloads Chroma's embedding
   model, so it needs internet access and can take longer than later runs.
6. Ask **What is RAG?**. Check that the answer has citations such as `[S1]`.
   Expand **View retrieved passages and citation labels** to match each label
   to a source filename and paragraph.
7. Visit **http://localhost:8000/health**. After the model is downloaded, expect
   `status: healthy`, `chromadb: connected`, `ollama: connected`, and
   `model_available: true`.

Modern Docker Desktop uses `docker compose`. If your course uses the older
`docker-compose` command, the equivalent startup command is
`docker-compose up --build`; replace `docker compose` consistently in the other
commands if using that older executable.

The ZIP includes a working `.env` with no secrets. If downloading this project
from Git instead, create it first with `Copy-Item .env.example .env`.

## Verify the running stack

After the model is pulled and all services are running, execute this in the
second terminal:

```powershell
docker compose exec -T backend python verify_stack.py
```

The script checks actual HTTP communication between the services, indexes the
documents with real embeddings, generates a cited answer through the installed
Ollama model, checks blank-question validation, and verifies that repeated
indexing does not duplicate chunks. It exits with a nonzero status on failure.

To save the output as assignment evidence, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Verify-Stack.ps1
```

This writes `verification-results.txt`. Also verify the browser flow manually;
the HTTP script does not automate clicking Streamlit controls.

## Project files

| Path | Purpose |
| --- | --- |
| `.github/workflows/ci.yml` | Python tests and both Docker builds on pushes to main and pull requests |
| `backend/tests/test_api.py` | Required root, health, and statistics endpoint tests |
| `CI-SETUP.md` | Windows instructions for pushing and checking GitHub Actions |
| `pytest.ini` | Discover both test directories and resolve application imports |
| `docker-compose.yml` | Three services, ports, volumes, and healthy startup dependencies |
| `.env` / `.env.example` | Runtime values and safe documented defaults |
| `.gitignore` / `.dockerignore` | Exclude local environments, caches, and `.env` |
| `backend/Dockerfile` | Python 3.11, dependency layer, application, Uvicorn, health check |
| `backend/config.py` | Centralized validated Settings class |
| `backend/my_rag_api.py` | FastAPI routes and CORS |
| `backend/rag.py` | Paragraph indexing, retrieval filtering, prompt assembly, Ollama calls |
| `backend/schemas.py` | Request and response schemas |
| `backend/requirements.txt` | Backend package versions |
| `backend/docs/` | Six sample text files; replace or extend with your own documents |
| `backend/verify_stack.py` | Live verification against all three services |
| `backend/.dockerignore` | Exclusions for the actual backend build context |
| `frontend/Dockerfile` | Python 3.11, Streamlit, and frontend health check |
| `frontend/app.py` | Document indexing, question form, answers, citations, and health display |
| `frontend/api_client.py` | Container-to-container HTTP requests and readable errors |
| `frontend/requirements.txt` | Frontend package versions |
| `frontend/.dockerignore` | Exclusions for the actual frontend build context |
| `frontend/.streamlit/config.toml` | Dark interface theme |
| `Verify-Stack.ps1` | Runs the live checks and saves their output |
| `tests/` | API, persistence, configuration, and Streamlit checks with test doubles |
| `VALIDATION.md` | Checks completed during preparation and remaining live checks |

## Services and persistence

| Service | Browser / host address | Address used inside containers | Storage |
| --- | --- | --- | --- |
| Frontend | `http://localhost:8501` | `http://frontend:8501` | Session state in memory |
| Backend | `http://localhost:8000` | `http://backend:8000` | `chroma_data` volume |
| Ollama | `http://localhost:11434` | `http://ollama:11434` | `ollama_data` volume |

Chroma's local `all-MiniLM-L6-v2` ONNX embedding model uses an
`embedding_cache` volume. Ollama generates answers; it does not generate the
embeddings in this setup. ChromaDB runs inside the backend, so exactly three
services are needed. Host port bindings use the loopback address.

`depends_on` uses `service_healthy` so the application services wait for their
dependencies. The backend container checks `/ready`, which does not require a
downloaded Ollama model. `/health` separately reports the model's availability;
it returns a diagnostic JSON body with `status: degraded` until everything is
connected.

To check persistence after indexing:

```powershell
$before = (Invoke-RestMethod http://localhost:8000/stats).document_count
docker compose down
docker compose up -d --wait
$after = (Invoke-RestMethod http://localhost:8000/stats).document_count
"Before: $before; After: $after"
```

The two counts should match, and `docker compose exec ollama ollama list` should
still show the downloaded model. Ordinary `docker compose down` preserves the
named volumes. Adding `-v` removes them and their data.

## Add documents and change settings

Add UTF-8 `.txt` files to `backend/docs/`, then click **Re-index Documents**.
That folder is mounted read-only into the backend, so document edits do not
require an image rebuild. Paragraphs are separated by blank lines. Re-indexing
also removes obsolete chunks once the new chunks have been stored successfully.
If all input files are empty or missing, indexing returns a clear error and
keeps the existing index.

Edit the root `.env` to change model, result count, confidence threshold, debug
logging, or timeouts. Apply changed container environment values with:

```powershell
docker compose up -d --force-recreate backend frontend
Invoke-RestMethod http://localhost:8000/stats
```

Wait for the backend to start before requesting `/stats`. A plain
`docker compose restart` does not reload environment variables. When changing
`MODEL_NAME`, also pull that model inside the Ollama container. For example,
after setting `MODEL_NAME=llama3.2:3b`:

```powershell
docker compose exec ollama ollama pull llama3.2:3b
docker compose up -d --force-recreate --wait backend frontend
Invoke-RestMethod http://localhost:8000/stats
```

The returned `model` should be `llama3.2:3b`. Restore `llama3.2:1b` and recreate
the services when finished. Keep `CHROMA_PATH` and `DOCS_PATH` as absolute
container paths; Compose mounts storage at those configured paths. If changing
host ports, use the corresponding new browser URL.

Retrieval confidence is a similarity heuristic, not an answer correctness
score: best cosine distance below 0.5 is high, below 1.0 is medium, otherwise
low. Chunks must also pass the configured `CONFIDENCE_THRESHOLD`. The system
prompt requests grounded answers and citations. Missing or invented citation
labels trigger a clearly labeled, directly quoted source-excerpt fallback.
Valid citation labels alone cannot prove that every generated claim is correct;
the source passages are available for inspection.

## API checks in Swagger

Open **http://localhost:8000/docs**. The available routes are:

| Method and route | Behavior |
| --- | --- |
| `GET /` | Application message and configured model |
| `GET /ready` | Application/database readiness; unavailable database returns 503 |
| `GET /health` | ChromaDB, Ollama, model availability, and chunk count |
| `GET /stats` | Indexed chunk count, source count, model, and retrieval settings |
| `POST /ingest` | Load or update all `.txt` files from `docs/` |
| `POST /ask` | Accept `{"question": "What is RAG?"}` and return a cited answer |

A blank or whitespace-only question returns HTTP 422. Before indexing, `/ask`
returns a clear message telling you to index documents. A missing or
unreachable Ollama model returns HTTP 503 when generation is needed. Generation
timeouts return 504. An invalid upstream response returns 502.

## Troubleshooting

- **Docker API / dockerDesktopLinuxEngine pipe error:** open Docker Desktop,
  wait for its engine to finish starting, and run `docker info` again.
- **Port 11434 already in use:** a Windows Ollama installation may already be
  listening there. Set `OLLAMA_PORT=11435` in `.env`, then run
  `docker compose up -d`. Keep `OLLAMA_URL=http://ollama:11434`; the internal
  container port stays 11434.
- **Backend or Streamlit port already in use:** change `BACKEND_PORT` or
  `FRONTEND_PORT` in `.env`, recreate the services, and use the new browser port.
- **Model not installed:** run the model pull command inside the container;
  models installed in Windows Ollama are separate from this volume.
- **First indexing is slow or fails:** check internet access for the embedding
  model download and read `docker compose logs backend`.
- **Answer takes too long:** CPU generation can take time. Increase
  `OLLAMA_TIMEOUT` and set `REQUEST_TIMEOUT` to a larger value, then recreate the
  application containers.
- **Inspect errors:** use `docker compose logs backend frontend ollama`.

## Official references

- [GitHub Actions: building and testing Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [GitHub Actions workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [Docker image inspection](https://docs.docker.com/reference/cli/docker/image/inspect/)
- [Docker Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/)
- [Docker Compose environment files](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/)
- [Why restart does not apply environment changes](https://docs.docker.com/reference/cli/docker/compose/restart/)
- [Ollama Docker setup](https://docs.ollama.com/docker)
- [Chroma embedding functions](https://docs.trychroma.com/docs/embeddings/embedding-functions)
- [Streamlit Docker deployment](https://docs.streamlit.io/deploy/tutorials/docker)
