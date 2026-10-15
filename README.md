# Unified AI Compute Platform for Sequence Modeling

A locally-run platform for training and fine-tuning sequence models. The working feature today is **Music Composer**: upload a MIDI file, pick a model (RNN, Transformer v1, or Transformer v2), fine-tune it on your song, and generate new MIDI from it.
(Transformer2 currently works best). The MIDI parser currently assumes a single instrument/channel

**Market Predictor** (visible on the home page) is not yet implemented — the frontend button exists but there's no backend behind it yet.

## Requirements
- **Python 3.10+**
- **[Git LFS](https://git-lfs.com/)** — the pretrained model weights (`backend/music/trained_models/`) are stored via Git LFS, not as regular files. **Install it before cloning**:
  ```bash
  sudo apt install git-lfs      # or your platform's equivalent
  git lfs install
  ```
- An **NVIDIA GPU** is strongly recommended for fine-tuning/generation speed, but not required — everything falls back to CPU (just slower).
- Either a local **Python environment**, or **Docker + Docker Compose**, to run it (two options below).

## Getting the code

```bash
git lfs install
git clone git@github.com:jetam/AiPlatform.git
cd AiPlatform
```

## Option A: Run with Docker (recommended)

```bash
docker compose build
docker compose up
```

This starts two containers:
- **backend** — the FastAPI/uvicorn API server, on `http://localhost:5000`
- **frontend** — a static file server for the web UI, on `http://localhost:3000`

Then open **`http://localhost:3000/frontend/composer.html`** in your browser.

Model weights are mounted from `backend/music/trained_models/` on your machine into the container (not baked into the image), so fine-tuned models persist across rebuilds.

**If you have an NVIDIA GPU:** install [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host first, then uncomment the GPU block in `docker-compose.yml`:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
Without this, the container still runs but it uses the CPU, which will be slower for fine-tuning/generation.

## Option B: Run natively (without Docker)

Install dependencies (ideally in a virtualenv):
```bash
pip install -r requirements.txt
```

Start both servers (each in its own terminal):
```bash
./startServer.sh          # API server on :5000
./startStaticServer.sh    # static file server on :3000
```

Then open **`http://localhost:3000/frontend/composer.html`**.


## How it works (Composer flow)

1. Drop a MIDI file onto the page
2. Pick a model from the dropdown — Model will be loaded and fine-tuned on your uploaded song.
3. Click Generate — produces a new MIDI file for download, in the style (hopefully) of the fine-tuned model.

Each browser session gets its own isolated model/song state (up to 3 concurrent sessions at once — older idle sessions are evicted to make room for new ones, since each loaded model can be large).
The reason for this is that each fine-tuned model takes up a lot of space.

## Todo in later versions of Composer:

- No way to cancel an in-progress fine-tune or generation from the UI yet.
- No progress indicator during fine-tuning/generation (just a spinner).
- MIDI parsing assumes a single instrument/channel.
- No train/validation split on any model — no signal on overfitting vs. genuine learning.
- No authentication — fine for a single local user, not yet suitable for exposing beyond localhost.
- Session state (pool of size 3) is in-memory only; nothing persists across a server restart except the model weights themselves.


