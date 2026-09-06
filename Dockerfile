# CPU-compatible Dockerfile (works on Windows, Mac, Linux)
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/huggingface

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch CPU version (2.8.0 matches what speechbrain 1.1.x expects)
RUN pip3 install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies from requirements
COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Speaker embeddings. speechbrain 1.1.x is the first line compatible with
# huggingface_hub 1.x, and its model is ungated: no account, no token.
RUN pip3 install --no-cache-dir "speechbrain==1.1.1"

# Pre-download models during build so they're baked into the image
# large-v3-turbo has no Systran conversion; faster-whisper itself resolves the
# 'turbo' name to this repo. Pinned by commit so the baked weights cannot change.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo', \
                  revision='0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf', \
                  local_dir='/opt/whisper-large-v3-turbo')"

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('speechbrain/spkrec-ecapa-voxceleb', local_dir='/opt/huggingface/speechbrain_ecapa')"

# All weights are baked in above. Block any further contact with HuggingFace so
# nothing about what gets transcribed leaves this machine at runtime.
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DISABLE_TELEMETRY=1

# Create directories
RUN mkdir -p /app/uploads /app/outputs /app/models

# Copy application files
COPY app.py .
COPY templates templates/

EXPOSE 5000

CMD ["python3", "app.py"]