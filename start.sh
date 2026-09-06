#!/usr/bin/env bash
# Start the transcription service, seeding .env on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "No .env found, so one was created from .env.example."
    echo "Local transcription works with these defaults. Edit .env if you want"
    echo "cloud analysis (GEMINI_API_KEY) or a different Ollama model."
    echo
fi

compose_file=docker-compose.yml
if [ "${1:-}" = "--gpu" ]; then
    compose_file=docker-compose.gpu.yml
    shift
    echo "Using the GPU compose file. This needs an NVIDIA GPU and container runtime."
fi

exec docker compose -f "$compose_file" up --build "$@"
