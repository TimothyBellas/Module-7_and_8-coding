"""Boundary cases for text chunking and environment validation."""

import pytest
from pydantic import ValidationError

from backend.config import Settings
from backend.rag import chunk_text


def test_chunking_covers_long_text_with_bounded_overlap():
    text = "".join(chr(0x400 + i) for i in range(1100))
    chunks = chunk_text(text, size=200, overlap=30)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks)
    assert all(left[-30:] == right[:30] for left, right in zip(chunks, chunks[1:]))
    reconstructed = chunks[0] + "".join(chunk[30:] for chunk in chunks[1:])
    assert reconstructed == text


def test_empty_and_short_text():
    assert chunk_text(" \n\t ") == []
    assert chunk_text("A short note.") == ["A short note."]


def test_chunking_prefers_a_nearby_paragraph_boundary():
    paragraph = "a" * 150
    chunks = chunk_text(paragraph + "\n\n" + "b" * 150, 200, 20)
    assert chunks[0] == paragraph
    assert chunks[-1].endswith("b" * 100)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings_fail(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def test_settings_read_environment_and_validate_combinations(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "example-model:1b")
    settings = Settings(_env_file=None, ollama_url="http://ollama.test")
    assert settings.model_name == "example-model:1b"
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, ollama_url="http://ollama.test", chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValidationError, match="HIGH_CONFIDENCE_THRESHOLD"):
        Settings(_env_file=None, ollama_url="http://ollama.test", high_confidence_threshold=0.9)
