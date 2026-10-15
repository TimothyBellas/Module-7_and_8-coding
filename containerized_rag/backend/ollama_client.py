"""Small Ollama HTTP adapter; no cloud API or automatic embedding downloads."""

import math

import httpx

from .config import Settings


class DependencyUnavailable(RuntimeError):
    """A required local dependency is missing, unresponsive, or invalid."""


class OllamaClient:
    """Use one connection pool for embedding, model checks, and generation."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=str(settings.ollama_url).rstrip("/"),
            timeout=settings.request_timeout,
        )

    def close(self) -> None:
        """Release the HTTP connection pool."""
        self.client.close()

    def _post(self, path: str, payload: dict) -> dict:
        """Turn upstream failures into a useful 503 response at the API boundary."""
        try:
            response = self.client.post(path, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            return data
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise DependencyUnavailable(
                    "An Ollama model is missing. Check MODEL_NAME and EMBEDDING_MODEL, "
                    "then restart the ollama service to download them."
                ) from exc
            raise DependencyUnavailable(
                f"Ollama returned HTTP {exc.response.status_code}. Check docker compose logs ollama."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailable(
                "Ollama is unavailable or timed out. Check docker compose ps and logs ollama."
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a bounded batch, failing rather than silently truncating input."""
        data = self._post("/api/embed", {
            "model": self.settings.embedding_model,
            "input": texts,
            "truncate": False,
            "keep_alive": "5m",
        })
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise DependencyUnavailable("Ollama returned an invalid embedding batch.")
        dimension = len(vectors[0]) if vectors and isinstance(vectors[0], list) else 0
        for vector in vectors:
            if not isinstance(vector, list) or not vector or len(vector) != dimension:
                raise DependencyUnavailable("Ollama returned inconsistent embedding dimensions.")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                raise DependencyUnavailable("Ollama returned an invalid embedding value.")
        return vectors

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a single non-streamed answer with deterministic-leaning settings."""
        data = self._post("/api/chat", {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.context_window,
                "num_predict": self.settings.max_answer_tokens,
            },
        })
        message = data.get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise DependencyUnavailable("Ollama returned an empty or invalid answer.")
        return answer.strip()

    def health(self) -> tuple[bool, bool, bool]:
        """Check server connectivity and both configured model tags quickly."""
        try:
            response = self.client.get("/api/tags", timeout=self.settings.health_timeout)
            response.raise_for_status()
            names = {
                model["name"] for model in response.json()["models"]
                if isinstance(model, dict) and isinstance(model.get("name"), str)
            }
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return False, False, False

        def present(name: str) -> bool:
            return name in names or (":" not in name and f"{name}:latest" in names)

        return True, present(self.settings.model_name), present(self.settings.embedding_model)
