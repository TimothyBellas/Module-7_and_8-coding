#!/bin/sh
# Run the server and fetch missing models inside the SAME Ollama service.
set -eu

: "${MODEL_NAME:?Set MODEL_NAME}"
: "${EMBEDDING_MODEL:?Set EMBEDDING_MODEL}"

rm -f /tmp/models-ready
ollama serve &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 0' INT TERM

until ollama list >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Ollama server exited before becoming ready." >&2
        exit 1
    fi
    sleep 1
done

for model in "$MODEL_NAME" "$EMBEDDING_MODEL"; do
    if ! ollama show "$model" >/dev/null 2>&1; then
        echo "Downloading $model (first run only)…"
        ollama pull "$model"
    fi
done

touch /tmp/models-ready
echo "Ollama is ready; both models are available."
wait "$server_pid"
