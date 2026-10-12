import pytest
import yaml
from pydantic import ValidationError

from config import Settings
from conftest import PROJECT


def test_dotenv_change_takes_effect_for_new_settings(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=llama3.2:1b\nMAX_RESULTS=4\nDEBUG=true\n")
    first = Settings(_env_file=env_file)
    assert first.MODEL_NAME == "llama3.2:1b"
    assert first.MAX_RESULTS == 4 and first.DEBUG is True
    env_file.write_text("MODEL_NAME=llama3.2:3b\n")
    assert Settings(_env_file=env_file).MODEL_NAME == "llama3.2:3b"


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=llama3.2:1b\n")
    monkeypatch.setenv("MODEL_NAME", "llama3.2:3b")
    assert Settings(_env_file=env_file).MODEL_NAME == "llama3.2:3b"


@pytest.mark.parametrize("values", [
    {"MAX_RESULTS": 0}, {"CONFIDENCE_THRESHOLD": 0},
    {"OLLAMA_URL": "ollama:11434"}, {"MODEL_NAME": "   "},
])
def test_invalid_settings_are_rejected(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_compose_service_network_volumes_and_startup():
    compose = yaml.safe_load((PROJECT / "docker-compose.yml").read_text())
    services = compose["services"]
    assert set(services) == {"backend", "frontend", "ollama"}
    assert services["backend"]["depends_on"]["ollama"]["condition"] == "service_healthy"
    assert services["frontend"]["depends_on"]["backend"]["condition"] == "service_healthy"
    assert services["backend"]["env_file"] == services["frontend"]["env_file"] == ".env"
    assert set(compose["volumes"]) == {"chroma_data", "ollama_data", "embedding_cache"}
    env = dict(line.split("=", 1) for line in (PROJECT / ".env").read_text().splitlines() if line and not line.startswith("#"))
    assert env["API_URL"] == "http://backend:8000"
    assert env["OLLAMA_URL"] == "http://ollama:11434"
    for context in [PROJECT, PROJECT / "backend", PROJECT / "frontend"]:
        assert ".env" in (context / ".dockerignore").read_text().splitlines()
    assert ".env" in (PROJECT / ".gitignore").read_text().splitlines()
