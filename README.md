# Audio/Video Transcription Service

Flask-based transcription service using OpenAI's Whisper model, with optional speaker identification. Everything runs locally.

## Features

- 🎙️ **Multi-format support**: MP3, MP4, WAV, M4A, FLAC, AVI, MOV, WebM, MKV
- 👥 **Optional speaker identification**: Off by default. When on, detects one to six speakers without assuming there is more than one
- 🚀 **Runs offline**: Model weights ship in the image; no account, token or outbound call is needed to transcribe
- 🎯 **Fixed transcription model**: Whisper large-v3-turbo, matched large-v3 output in testing at about a quarter of the time
- 🌐 **Web interface**: Drag-and-drop file upload with progress tracking
- ⚡ **Real-time progress**: Live updates with cancellation support
- 🎙️ **AI transcript processing**: Local Gemma 4 (`gemma4:e4b`) via Ollama, or the cloud Gemini API

## Requirements

Docker is the only thing you need to install. On a Mac, either download Docker
Desktop from docker.com or install it from the terminal:

```bash
brew install --cask docker
```

Docker Desktop must be running before any of the commands below work. Launch it
from Applications, or with `open -a Docker`, and wait for the whale icon in the
menu bar to stop animating. Check that the daemon is up:

```bash
docker info
```

If that prints an error about not being able to connect, Docker Desktop is not
running yet.

## Quick Start

```bash
./start.sh
```

Then open http://localhost:8080.

On first run this creates `.env` from `.env.example`. The defaults are enough for
local transcription, so there is nothing to fill in before starting. Edit `.env`
later if you want cloud analysis or a different Ollama model, and an existing
`.env` is never overwritten.

Add `--gpu` to use an NVIDIA GPU, which needs the NVIDIA container runtime:

```bash
./start.sh --gpu
```

Any further arguments are passed to Docker Compose, so `./start.sh -d` runs it in
the background. You can also call Compose directly if you prefer; `.env` is
optional there and the service falls back to its built-in defaults without it.

## Working with the container

The first build downloads PyTorch, Whisper and the model weights, so it takes a
while and produces an image of roughly 4 GB. Later builds reuse the cached
layers and only rebuild what changed.

Everything below runs from the project directory. The service is one container
named `transcription-service`.

```bash
# Start in the background instead of holding the terminal
./start.sh -d

# Follow the log output, which is where transcription progress appears
docker compose logs -f

# Check whether it is running and which port it is on
docker compose ps

# Stop it, keeping the built image
docker compose down

# Rebuild after changing app.py or the templates
docker compose up --build -d
```

To look around inside the running container, open a shell in it:

```bash
docker exec -it transcription-service bash
```

That drops you into `/app`, where `app.py` and `templates/` live. Transcripts are
written to `/app/outputs`, which is the `outputs/` folder in the project, so
anything saved there survives the container being removed. The same applies to
`uploads/`.

If a build fails or behaves oddly, rebuild without the cache:

```bash
docker compose build --no-cache
```

To reclaim disk space from old images once you are done:

```bash
docker image prune
```

## Usage

1. Open http://localhost:8080 in your browser
2. Select processing device (CPU), language (auto-detect by default), and whether to identify speakers
3. Upload or drag-and-drop an audio/video file
4. Click "Transcribe" and wait for processing
5. View, copy, or download the transcript

## Output Format

Transcripts are saved in `./outputs/` with timestamps and speaker labels:

With speaker identification on:

```
[00:00:00 - 00:00:04] Speaker 1: Hello, how are you?
[00:00:04 - 00:00:07] Speaker 2: I'm doing well, thanks!
```

With it off:

```
[00:00:00 - 00:00:04] Hello, how are you?
[00:00:04 - 00:00:07] I'm doing well, thanks!
```

## Supported File Formats

- **Audio**: MP3, WAV, M4A, FLAC
- **Video**: MP4, AVI, MOV, WebM, MKV

## API Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload file for transcription
- `POST /ai-analysis` - Summarise or analyse a finished transcript
- `GET /download/<filename>` - Download transcript
- `POST /cancel/<task_id>` - Cancel running transcription
- `GET /gpu-status` - Whether a transcription or analysis is currently running
- `GET /health` - System health check
- `GET /favicon.svg` - Browser tab icon

## Configuration

### Model Selection
Transcription always runs Whisper large-v3-turbo. There is no model choice to make.

### Language
Auto-detect by default. Pick a language in the interface when the audio is short or noisy, since detection is less reliable there. `WHISPER_LANGUAGE` in `.env` sets the starting value.

### Speaker identification
On or off. On labels each line with a speaker using SpeechBrain ECAPA-TDNN embeddings (`spkrec-ecapa-voxceleb`), clustered with a distance threshold calibrated against known single- and two-speaker audio. Off produces a plain timestamped transcript and runs faster.

No account and no token are needed. If speaker identification cannot run, the interface says so and the transcript comes back without labels rather than with invented ones.

### Privacy
Model weights are baked into the image at build time and no login is required to build or run. At runtime the container is set to `HF_HUB_OFFLINE` with telemetry disabled, so transcription never contacts HuggingFace or any other third party. The only outbound call the service can make is to the Gemini API, and only when you explicitly pick the cloud option for transcript analysis.

### Device Selection
- **CPU**: Works everywhere, slower (the only option on macOS)
- **GPU**: Considerably faster, requires an NVIDIA GPU and the GPU compose file

The device selector is hidden when no CUDA device is present.

## Troubleshooting

### Slow transcription on Mac
- This is normal - Macs use CPU processing
- **Tips**:
  - Name the language instead of leaving it on auto
  - Consider using smaller file chunks
  - Expect ~1-2 minutes per minute of audio on CPU

### Port already in use
- Change the port in `docker-compose.yml`:
  ```yaml
  ports:
    - "8081:5000"  # Use 8081 instead of 8080
  ```

## AI transcript processing

Summaries and other analysis of a finished transcript can run either locally or in the cloud. Pick the model in the "AI Model Selection" card in the web interface.

### Local Gemma 4 (default)

Runs on your own machine through [Ollama](https://ollama.com), so the transcript never leaves it.

```bash
ollama pull gemma4:e4b
```

Then set both values in `.env`:

```
OLLAMA_BASE_URL=http://host.docker.internal:11434
GEMMA_MODEL_NAME=gemma4:e4b
```

`host.docker.internal` reaches an Ollama installed on the host from inside the container. Use `http://localhost:11434` when running the app outside Docker. To use a different local model, pull it and change `GEMMA_MODEL_NAME` to its Ollama tag.

### Cloud Gemini

Set `GEMINI_API_KEY` in `.env` with a key from https://aistudio.google.com/app/apikey.

Either option is greyed out in the interface when it cannot run, with a note saying what to configure. If Ollama is running but lacks the configured model, the note gives you the `ollama pull` command.

Analysis waits up to `OLLAMA_TIMEOUT_SECONDS` (600 by default). For scale, a cold model load costs roughly 35 seconds and a 45-minute transcript takes about 90 seconds more. `OLLAMA_KEEP_ALIVE` holds the model in memory so only the first analysis pays the load.

`GET /health` reports whether Ollama is reachable and which models it has available.

## License

This project uses OpenAI's Whisper model, run through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and SpeechBrain's
ECAPA-TDNN speaker embeddings. Check their licensing terms for commercial use.
