"""HTTP calls from the Streamlit container to the FastAPI container."""

import os

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))


class APIError(Exception):
    pass


def request(method: str, endpoint: str, *, body: dict | None = None, timeout: float = 10):
    try:
        response = httpx.request(
            method, f"{API_URL}{endpoint}", json=body, timeout=timeout
        )
    except httpx.TimeoutException as exc:
        raise APIError("The request timed out. Wait a moment and try again.") from exc
    except httpx.RequestError as exc:
        raise APIError("Cannot connect to the backend. Check that the backend service is running.") from exc
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(str(detail))
    try:
        return response.json()
    except ValueError as exc:
        raise APIError("The backend returned an invalid response.") from exc


def get_health():
    return request("GET", "/health")


def get_stats():
    return request("GET", "/stats")


def ingest_documents():
    return request("POST", "/ingest", timeout=REQUEST_TIMEOUT)


def ask_question(question: str):
    return request("POST", "/ask", body={"question": question}, timeout=REQUEST_TIMEOUT)
