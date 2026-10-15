"""Run real API acceptance checks after Compose starts; standard library only."""

import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    """Check health, ingestion, citations, and invalid input using the live stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Published backend URL, including scheme")
    args = parser.parse_args()

    def call(path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    assert "message" in call("/")
    health = call("/health")
    assert health["ready"], f"Dependencies are not ready: {health}"
    indexed = call("/ingest", {})
    assert indexed["document_count"] >= 5, indexed
    before = call("/stats")
    call("/ingest", {})
    after = call("/stats")
    assert (before["document_count"], before["chunk_count"]) == (
        after["document_count"], after["chunk_count"],
    ), "Repeated ingestion created duplicate records"
    try:
        call("/ask", {"question": "   "})
    except urllib.error.HTTPError as exc:
        assert exc.code == 422, f"Expected 422, got {exc.code}"
    else:
        raise AssertionError("Blank question should return 422")
    answerable = call("/ask", {
        "question": "How do named volumes preserve data when containers are recreated?",
    })
    print("Answerable question:", json.dumps(answerable, indent=2))
    assert answerable["grounded"] and answerable["sources"], (
        "The real model did not return a supported cited answer; inspect output and sources."
    )
    for question in [
        "What exact score did this project receive from its instructor?",
        "What is the secret recipe for the Moon King's birthday cake?",
    ]:
        result = call("/ask", {"question": question})
        print("Unanswered question:", question, json.dumps(result, indent=2))
        assert result["grounded"] is False, (
            "The model answered an unsupported question; inspect evidence and tune retrieval."
        )
    print("Live API acceptance checks passed. Verify the browser interface and persistence separately.")


if __name__ == "__main__":
    main()
