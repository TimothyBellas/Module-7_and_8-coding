"""Run real HTTP checks against all three running Compose services.

Run with: docker compose exec -T backend python verify_stack.py
This script uses real Chroma embeddings and the installed Ollama model.
"""

import json
import os
import re
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BACKEND = "http://backend:8000"
FRONTEND = "http://frontend:8501"
OLLAMA = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
MODEL = os.getenv("MODEL_NAME", "llama3.2:1b")


def http(method, url, body=None, expected_status=200):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            status = response.status
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        status = exc.code
        text = exc.read().decode("utf-8")
    if status != expected_status:
        raise RuntimeError(f"{method} {url}: expected HTTP {expected_status}, got {status}: {text}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    print("Live three-service RAG verification")
    print(f"Configured model: {MODEL}")
    root = http("GET", f"{BACKEND}/")
    if root["model"] != MODEL:
        raise RuntimeError("The root endpoint did not report the configured model.")
    print("PASS: backend root endpoint")

    frontend_health = http("GET", f"{FRONTEND}/_stcore/health")
    if frontend_health != "ok":
        raise RuntimeError(f"Unexpected Streamlit health response: {frontend_health}")
    print("PASS: Streamlit is reachable over the Compose network")

    tags = http("GET", f"{OLLAMA}/api/tags")
    installed = {item["name"] for item in tags["models"]}
    expected_model = MODEL if ":" in MODEL else f"{MODEL}:latest"
    if expected_model not in installed:
        raise RuntimeError(f"First run: docker compose exec ollama ollama pull {MODEL}")
    print("PASS: Ollama is reachable and the configured model is installed")

    health = http("GET", f"{BACKEND}/health")
    if health["status"] != "healthy" or health["ollama"] != "connected":
        raise RuntimeError(f"Health check failed: {health}")
    print("PASS: /health reports connected ChromaDB and Ollama")

    ingest = http("POST", f"{BACKEND}/ingest")
    if ingest["document_count"] < 1:
        raise RuntimeError("Indexing did not store any chunks.")
    print(f"PASS: indexed {ingest['files_indexed']} files and {ingest['chunks_indexed']} chunks")

    answer = http("POST", f"{BACKEND}/ask", {"question": "What is RAG?"})
    if not answer["answer"].strip() or not answer["sources"] or not answer["chunks"]:
        raise RuntimeError(f"The answer did not include source evidence: {answer}")
    labels = set(re.findall(r"\[S(\d+)\]", answer["answer"]))
    available = {chunk["citation"][1:] for chunk in answer["chunks"]}
    if not labels or not labels.issubset(available):
        raise RuntimeError("The answer contains missing or invalid citation labels.")
    print("PASS: end-to-end retrieval and answer with valid source citations")
    print("Question: What is RAG?")
    print("Answer:", answer["answer"])
    print("Sources:", ", ".join(answer["sources"]))
    print("Retrieval confidence:", answer["confidence"])

    http("POST", f"{BACKEND}/ask", {"question": "   "}, expected_status=422)
    print("PASS: blank questions return HTTP 422")
    stats = http("GET", f"{BACKEND}/stats")
    second_ingest = http("POST", f"{BACKEND}/ingest")
    if stats["document_count"] != second_ingest["document_count"]:
        raise RuntimeError("Re-indexing unchanged files changed the chunk count.")
    print("PASS: repeated indexing does not duplicate chunks")
    print("All live checks passed. Open http://localhost:8501 and verify the interface manually.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
