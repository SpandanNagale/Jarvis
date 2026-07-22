# ARIA — Local AI Voice Assistant

> A fully offline, voice-activated AI assistant for Windows, powered by **Ollama** (Qwen3:14b), **faster-whisper** (speech-to-text), **Coqui XTTS-v2** (text-to-speech, with Piper as a fallback), and **openWakeWord** (wake word detection) — with a live HUD overlay.

---

## ✨ Features

| Category | Capability |
|---|---|
| 🎙️ **Voice** | "Hey Jarvis" wake word · hold-Space push-to-talk · XTTS-v2 natural voice (voice cloning supported) with Piper fallback |
| 🖥️ **HUD** | Always-on-top overlay — idle / listening / thinking / speaking states, live transcript + reply text |
| 🧠 **AI** | Qwen3:14b via Ollama · adaptive chain-of-thought (`/think` / `/no_think`) |
| 🖥️ **System Control** | Open/close apps · volume · brightness · lock PC · list windows |
| 🌐 **Network** | DuckDuckGo web search · fetch web pages · GitHub search · Wikipedia |
| 👁️ **Vision** | Screenshot analysis, on-screen text reading (qwen2.5vl:7b) |
| 💾 **Memory** | ChromaDB-backed persistent facts · cross-session conversation history |
| 🛠️ **Coding** | Fuzzy file finder · code reader · explain code · refactor suggestions · traceback diagnosis |

> **Note on the wake word:** the activation phrase is currently still **"Hey Jarvis"** — a
> custom "Hey Aria" openWakeWord model is being trained separately (see
> [`wakeword_training/`](wakeword_training/)). Once that model is ready, swap
> `WAKE_WORD_MODEL_NAME` in `ARIA_5.py` to `"hey_aria"` and drop the trained model file in.
> Everything else (persona, replies, HUD) already says ARIA.

---

## 🏗️ Project Structure

```
ARIA/
├── ARIA_1.py                  # Core tools: system control + base tool schemas
├── ARIA_2.py                  # Phase 2: voice pipeline (STT + TTS + wake word)
├── ARIA_3.py                  # Phase 3: push-to-talk + improved audio handling
├── ARIA_4.py                  # Phase 4: coding tools + memory integration
├── ARIA_5.py                  # Phase 5+ (latest): Qwen3 thinking + network + vision
├── ARIA_tools_coding.py       # Coding tools (find file, explain, refactor, debug)
├── ARIA_tools_memory.py       # ChromaDB memory layer + session persistence
├── ARIA_tools_network.py      # Web search, fetch page, GitHub, Wikipedia
├── ARIA_tools_vision.py       # Screenshot analysis via a vision model
├── ARIA_tts.py                # XTTS-v2 voice output, with layered GPU/CPU/Piper fallback
├── ARIA_hud.py                # Always-on-top HUD overlay (PySide6)
├── wake_debug.py              # Utility to debug wake word detection scores
├── requirements.txt           # Python dependencies
└── memory/                    # ChromaDB storage (auto-created, gitignored)
```

> **`ARIA_5.py` is the main entry point** — it builds on all previous phases.

---

## ⚙️ Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support — recommended for faster-whisper `large-v3-turbo` and XTTS-v2. This project is tuned for a 12GB-class card (RTX 5070); XTTS's GPU fallback chain (see `ARIA_tts.py`) exists because VRAM gets tight once Ollama's model is also resident.
- **RAM**: 16 GB+ recommended (Qwen3:14b needs ~10 GB VRAM)
- **Microphone**: Any standard microphone

### Software
- Python 3.12
- [Ollama](https://ollama.com/) installed and running
- [Piper TTS](https://github.com/rhasspy/piper) binary on your `PATH` (fallback voice)
- CUDA Toolkit (for GPU-accelerated STT/TTS)

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/SpandanNagale/ARIA.git
cd ARIA
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **RTX 50-series (Blackwell) GPUs:** `requirements.txt` already pins `torch==2.9.1+cu128` and
> pulls from the PyTorch cu128 wheel index — needed for Blackwell (`sm_120`) support. It also
> pins `transformers<5` (a newer `transformers` breaks XTTS-v2's loader) and adds `torchcodec`
> (required by `torchaudio>=2.9`'s audio backend). If you hit GPU/DLL issues on a different card
> or driver, these are the first pins to double-check.

### 4. Pull the Ollama models

```bash
ollama pull qwen3:14b
ollama pull qwen2.5vl:7b   # optional, for screen/vision tools
```

### 5. XTTS-v2 voice setup

XTTS-v2's weights (~1.8GB) download automatically on first run — no manual step needed. Two options:

- **Default voice** (no setup): just run the project, XTTS uses a built-in speaker.
- **Voice cloning**: record a 6–10 second, single-speaker, quiet-room WAV sample, save it as
  `reference.wav` in the project root. `ARIA_tts.py` will use it automatically if present.

Piper remains as the automatic fallback if XTTS fails to load or errors mid-session — download
the `en_US-sam-medium` voice and place both files in the project root:

```
en_US-sam-medium.onnx
en_US-sam-medium.onnx.json
```

Download from: https://github.com/rhasspy/piper/releases

### 6. (Optional) Set a GitHub token for higher API rate limits

```bash
# Windows PowerShell
$env:GITHUB_TOKEN = "your_token_here"

# Linux / macOS
export GITHUB_TOKEN="your_token_here"
```

---

## ▶️ Running ARIA

```bash
python ARIA_5.py
```

ARIA will load the wake word model, STT model, and XTTS-v2, then open the HUD overlay and start listening.

**Activation modes:**
- Say **"Hey Jarvis"** → auto-detects and starts recording your command
- Hold **Spacebar** → push-to-talk mode

**HUD controls:**
- **Ctrl+Alt+H** → toggle the overlay on/off
- **Left-click + drag** → reposition
- **Mouse wheel** (while hovering it) → resize

Press `Ctrl+C` to quit.

---

## 🗣️ Example Commands

```
"Hey Jarvis, what time is it?"
"Hey Jarvis, open notepad."
"Hey Jarvis, set volume to 50."
"Hey Jarvis, set brightness to 80."
"Hey Jarvis, lock the PC."
"Hey Jarvis, search for Python async tutorials."
"Hey Jarvis, what is RAG in AI?"
"Hey Jarvis, search me on GitHub."
"Hey Jarvis, find repos for LangChain."
"Hey Jarvis, explain my ARIA_5.py file."
"Hey Jarvis, remember that I prefer dark mode."
"Hey Jarvis, what do you know about me?"
"Hey Jarvis, what's on my screen?"
```

---

## 🧩 Architecture

```
User Voice
    │
    ▼
openWakeWord ──► wake detected?
    │                  │
    │           faster-whisper (STT)
    │                  │
    │           User text ──────────────► ARIA_hud (live transcript)
    │                  │
    │           should_think()?
    │            /think or /no_think
    │                  │
    │           Ollama (Qwen3:14b)
    │                  │
    │           Tool calls?──► Tool implementations
    │                  │          (system / coding / memory / network / vision)
    │           Final reply ────────────► ARIA_hud (live reply + state)
    │                  │
    ▼         XTTS-v2 (GPU→CPU→Piper fallback) ──► Audio playback
ChromaDB ◄──── save_messages()
(memory)
```

The voice loop (`ARIA_5.py`'s `chat_loop`) runs on a background thread; the HUD's Qt event loop
owns the main thread. State updates flow one-way through a thread-safe queue (`ARIA_hud.push(...)`).

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Local LLM inference via Ollama |
| `faster-whisper` | GPU-accelerated speech-to-text |
| `openwakeword` | "Hey Jarvis" wake word detection |
| `coqui-tts` | XTTS-v2 text-to-speech (natural voice, cloning support) |
| `piper-tts` | Offline text-to-speech (fallback voice) |
| `torch` / `torchaudio` / `torchcodec` | GPU inference backend for XTTS-v2 |
| `PySide6` | HUD overlay (Qt) |
| `sounddevice` / `soundfile` | Audio I/O |
| `chromadb` | Persistent vector memory |
| `duckduckgo-search` | Web search (no API key needed) |
| `beautifulsoup4` | Web page text extraction |
| `requests` | HTTP (GitHub, Wikipedia, fetch) |
| `Pillow` | Screenshot capture for vision tools |
| `pycaw` | Windows audio volume control |
| `screen_brightness_control` | Monitor brightness control |
| `pygetwindow` | List open windows |
| `pyperclip` | Clipboard access for coding tools |
| `keyboard` | Spacebar push-to-talk + HUD toggle hotkey |

---

## 📁 What's Gitignored

| Item | Reason |
|---|---|
| `venv/` | Virtual environment (recreate with `pip install -r requirements.txt`) |
| `*.onnx` / `*.onnx.json` | Piper voice model (~60 MB) — download separately |
| `*.wav` | Runtime audio files generated by TTS, and any `reference.wav` voice sample |
| `memory/` | ChromaDB data — user-specific, created at runtime |
| `.env` | Secrets (e.g. `GITHUB_TOKEN`) |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.
