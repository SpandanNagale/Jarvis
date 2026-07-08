# JARVIS — Local AI Voice Assistant

> A fully offline, voice-activated AI assistant for Windows, powered by **Ollama** (Qwen3:14b), **faster-whisper** (speech-to-text), **Piper TTS** (text-to-speech), and **openwakeword** (wake word detection).

---

## ✨ Features

| Category | Capability |
|---|---|
| 🎙️ **Voice** | "Hey Jarvis" wake word · hold-Space push-to-talk · Piper TTS voice replies |
| 🧠 **AI** | Qwen3:14b via Ollama · adaptive chain-of-thought (`/think` / `/no_think`) |
| 🖥️ **System Control** | Open/close apps · volume · brightness · lock PC · list windows |
| 🌐 **Network** | DuckDuckGo web search · fetch web pages · GitHub search · Wikipedia |
| 💾 **Memory** | ChromaDB-backed persistent facts · cross-session conversation history |
| 🛠️ **Coding** | Fuzzy file finder · code reader · explain code · refactor suggestions · traceback diagnosis |

---

## 🏗️ Project Structure

```
Jarvis/
├── Jarvis_1.py                # Core tools: system control + base tool schemas
├── Jarvis_2.py                # Phase 2: voice pipeline (STT + TTS + wake word)
├── Jarvis_3.py                # Phase 3: push-to-talk + improved audio handling
├── Jarvis_4.py                # Phase 4: coding tools + memory integration
├── Jarvis_5.py                # Phase 5 (latest): Qwen3 thinking + network access
├── Jarvis_tools_coding.py     # Coding tools (find file, explain, refactor, debug)
├── Jarvis_tools_memory.py     # ChromaDB memory layer + session persistence
├── Jarvis_tools_network.py    # Web search, fetch page, GitHub, Wikipedia
├── wake_debug.py              # Utility to debug wake word detection scores
├── requirements.txt           # Python dependencies
└── memory/                    # ChromaDB storage (auto-created, gitignored)
```

> **`Jarvis_5.py` is the main entry point** — it builds on all previous phases.

---

## ⚙️ Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster-whisper `large-v3-turbo`)
- **RAM**: 16 GB+ recommended (Qwen3:14b needs ~10 GB VRAM)
- **Microphone**: Any standard microphone

### Software
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- [Piper TTS](https://github.com/rhasspy/piper) binary on your `PATH`
- CUDA Toolkit (for GPU-accelerated STT)

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/jarvis.git
cd jarvis
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

### 4. Pull the Ollama model

```bash
ollama pull qwen3:14b
```

### 5. Download the Piper TTS voice model

Download the `en_US-sam-medium` voice and place both files in the project root:

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

## ▶️ Running JARVIS

```bash
python Jarvis_5.py
```

JARVIS will load the wake word model and STT model, then start listening.

**Activation modes:**
- Say **"Hey Jarvis"** → auto-detects and starts recording your command
- Hold **Spacebar** → push-to-talk mode

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
"Hey Jarvis, explain my Jarvis_5.py file."
"Hey Jarvis, remember that I prefer dark mode."
"Hey Jarvis, what do you know about me?"
```

---

## 🧩 Architecture

```
User Voice
    │
    ▼
openwakeword ──► wake detected?
    │                  │
    │           faster-whisper (STT)
    │                  │
    │           User text
    │                  │
    │           should_think()?
    │            /think or /no_think
    │                  │
    │           Ollama (Qwen3:14b)
    │                  │
    │           Tool calls?──► Tool implementations
    │                  │          (system / coding / memory / network)
    │           Final reply
    │                  │
    ▼              Piper TTS ──► Audio playback
ChromaDB ◄──── save_messages()
(memory)
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Local LLM inference via Ollama |
| `faster-whisper` | GPU-accelerated speech-to-text |
| `openwakeword` | "Hey Jarvis" wake word detection |
| `piper-tts` | Offline text-to-speech |
| `sounddevice` / `soundfile` | Audio I/O |
| `chromadb` | Persistent vector memory |
| `duckduckgo-search` | Web search (no API key needed) |
| `beautifulsoup4` | Web page text extraction |
| `requests` | HTTP (GitHub, Wikipedia, fetch) |
| `pycaw` | Windows audio volume control |
| `screen_brightness_control` | Monitor brightness control |
| `pygetwindow` | List open windows |
| `pyperclip` | Clipboard access for coding tools |
| `keyboard` | Spacebar push-to-talk |

---

## 📁 What's Gitignored

| Item | Reason |
|---|---|
| `venv/` | Virtual environment (recreate with `pip install -r requirements.txt`) |
| `*.onnx` / `*.onnx.json` | Piper voice model (~60 MB) — download separately |
| `*.wav` | Runtime audio files generated by TTS |
| `memory/` | ChromaDB data — user-specific, created at runtime |
| `.env` | Secrets (e.g. `GITHUB_TOKEN`) |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.
