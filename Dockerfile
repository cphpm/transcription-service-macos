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

# Install PyTorch CPU version (pinned to 2.8.0 to satisfy pyannote.audio>=4.0.0 floor constraint)
RUN pip3 install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies from requirements
COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Install speaker diarization packages (pyannote.audio 3.x uses soundfile, not torchcodec — ARM64 compatible)
RUN pip3 install --no-cache-dir "pyannote.audio==3.3.2" && \
    pip3 install --no-cache-dir "speechbrain==1.0.3"

# Pre-download models during build so they're baked into the image
ARG HF_TOKEN
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Systran/faster-whisper-base'); \
snapshot_download('Systran/faster-whisper-medium'); \
snapshot_download('Systran/faster-whisper-large-v3')"

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('speechbrain/spkrec-ecapa-voxceleb', local_dir='/opt/huggingface/speechbrain_ecapa')"

RUN if [ -n "$HF_TOKEN" ]; then python3 -c "\
import os, sys, traceback; \
from huggingface_hub import snapshot_download; \
try: snapshot_download('pyannote/speaker-diarization-3.1', token=os.environ['HF_TOKEN']); \
except Exception as e: print('ERROR:', e, flush=True); traceback.print_exc(); sys.exit(1)"; \
else echo 'HF_TOKEN not set, skipping pyannote pre-download'; fi

# Create directories
RUN mkdir -p /app/uploads /app/outputs /app/models

# Copy application files
COPY app.py .
COPY templates templates/

EXPOSE 5000

CMD ["python3", "app.py"]