"""
Jarvis_5.py  —  Phase 7: Intelligence + Network Access
=======================================================
Extends Jarvis_4 with:
  • Qwen3:14b — native chain-of-thought thinking (replaces Qwen2.5:14b)
  • Adaptive thinking — complex questions get /think, simple commands skip it
  • Network tools: web_search, fetch_page, search_github, search_wikipedia

New voice commands
------------------
  "Hey Jarvis, search me on GitHub"          → search_github(query="SpandanNagale", kind="users")
  "Hey Jarvis, find repos for LangChain"     → search_github(query="LangChain", kind="repositories")
  "Hey Jarvis, what is RAG in AI?"           → search_wikipedia / reasons before answering
  "Hey Jarvis, search for FastAPI tutorials" → web_search
  "Hey Jarvis, fetch the page at github.com/SpandanNagale" → fetch_page

Thinking mode
-------------
  Simple commands (open app, set volume, what time is it) → /no_think, fast
  Complex queries (explain this, what is X, search results analysis) → /think

Run:
    python Jarvis_5.py
"""

import sys
import os
import re
import atexit

if sys.platform == "win32":
    try:
        import nvidia.cublas
        import nvidia.cudnn
        found_dirs = []
        for pkg in (nvidia.cublas, nvidia.cudnn):
            pkg_dir = (
                os.path.dirname(pkg.__file__)
                if getattr(pkg, "__file__", None)
                else list(pkg.__path__)[0]
            )
            bin_dir = os.path.join(pkg_dir, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                found_dirs.append(bin_dir)
        if found_dirs:
            os.environ["PATH"] = os.pathsep.join(found_dirs) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
import keyboard
import ollama
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

from Jarvis_1 import (
    TOOLS as BASE_TOOLS,
    TOOL_IMPLEMENTATIONS as BASE_IMPLS,
    to_assistant_message,
)
from Jarvis_tools_coding import CODING_TOOLS, CODING_TOOL_IMPLEMENTATIONS
from Jarvis_tools_memory import (
    MEMORY_TOOLS,
    MEMORY_TOOL_IMPLEMENTATIONS,
    build_initial_messages,
    save_messages,
)
from Jarvis_tools_network import NETWORK_TOOLS, NETWORK_TOOL_IMPLEMENTATIONS

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL = "qwen3:14b"   # upgrade from qwen2.5:14b — same VRAM, adds native thinking

# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = BASE_TOOLS + CODING_TOOLS + MEMORY_TOOLS + NETWORK_TOOLS
ALL_IMPLS = {
    **BASE_IMPLS,
    **CODING_TOOL_IMPLEMENTATIONS,
    **MEMORY_TOOL_IMPLEMENTATIONS,
    **NETWORK_TOOL_IMPLEMENTATIONS,
}


def run_tool_call(tool_call) -> str:
    name = tool_call["function"]["name"]
    args = tool_call["function"].get("arguments", {}) or {}
    func = ALL_IMPLS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    return func(**args)


# ── Thinking mode ──────────────────────────────────────────────────────────────
# Qwen3 reads /think and /no_think tokens at the START of the user turn.
# We detect "complex" queries and prepend the right token.

_SIMPLE_PATTERNS = re.compile(
    r"^(open|close|set|lock|what time|what's the time|what is the time"
    r"|what's my|volume|brightness|mute|unmute|list windows|hey jarvis)",
    re.IGNORECASE,
)

_COMPLEX_PATTERNS = re.compile(
    r"(explain|why|how does|what is|what are|analyse|analyze|debug|diagnose"
    r"|refactor|suggest|compare|difference|search|find|look up|who is|tell me about"
    r"|summarize|summarise|research|github|wikipedia)",
    re.IGNORECASE,
)


def should_think(text: str) -> bool:
    """Return True if this query warrants chain-of-thought reasoning."""
    if _SIMPLE_PATTERNS.match(text.strip()):
        return False
    if _COMPLEX_PATTERNS.search(text):
        return True
    # Default: think for longer queries
    return len(text.split()) > 8


def inject_thinking_token(messages: list[dict], think: bool) -> list[dict]:
    """Prepend /think or /no_think to the last user message for Qwen3."""
    token = "/think" if think else "/no_think"
    patched = list(messages)
    for i in reversed(range(len(patched))):
        if patched[i]["role"] == "user":
            content = patched[i]["content"]
            if not content.startswith("/think") and not content.startswith("/no_think"):
                patched[i] = {**patched[i], "content": f"{token} {content}"}
            break
    return patched


def strip_think_block(text: str) -> str:
    """Remove the <think>...</think> block Qwen3 generates — it's internal
    reasoning and shouldn't be spoken aloud."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Audio config ──────────────────────────────────────────────────────────────

SAMPLE_RATE              = 16000
CHUNK                    = 1280
WAKE_THRESHOLD           = 0.30
SILENCE_RMS              = 300
SPEECH_START_TIMEOUT_CHUNKS = 40
MAX_SILENCE_CHUNKS       = 25
MAX_COMMAND_CHUNKS       = 150

PIPER_VOICE = "en_US-sam-medium"
PIPER_OUT   = "jarvis_reply.wav"

# ── Model loading ─────────────────────────────────────────────────────────────

print("Loading wake word model...")
wake_model = WakeWordModel(wakeword_models=["hey_jarvis"])

print("Loading speech-to-text model...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def transcribe(audio_int16: np.ndarray) -> str:
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    segments, _ = stt_model.transcribe(audio_float32, language="en", vad_filter=True)
    return " ".join(seg.text for seg in segments).strip()


def speak(text: str):
    try:
        text = re.sub(r"[\ud800-\udfff]", "", text)
        # Strip markdown-style formatting that sounds wrong spoken aloud
        text = re.sub(r"\*+|`+|#+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # [label](url) → label
        safe = text.encode("ascii", errors="ignore").decode("ascii").strip()
        if not safe:
            print("[TTS] No speakable text, skipping.")
            return
        subprocess.run(
            ["piper", "--model", PIPER_VOICE, "--output_file", PIPER_OUT],
            input=safe.encode("utf-8"),
            check=True,
        )
        data, sr = sf.read(PIPER_OUT, dtype="float32")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"[TTS failed: {e}]")


def record_command(stream) -> np.ndarray | None:
    frames, silence_count, speech_started = [], 0, False
    for _ in range(MAX_COMMAND_CHUNKS):
        chunk, _ = stream.read(CHUNK)
        chunk = chunk.flatten()
        frames.append(chunk)
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        if rms >= SILENCE_RMS:
            speech_started = True
            silence_count = 0
        else:
            silence_count += 1
            if not speech_started and silence_count > SPEECH_START_TIMEOUT_CHUNKS:
                return None
            if speech_started and silence_count > MAX_SILENCE_CHUNKS:
                break
    return np.concatenate(frames) if speech_started else None


def record_while_held(stream, key="space") -> np.ndarray | None:
    frames = []
    while keyboard.is_pressed(key):
        chunk, _ = stream.read(CHUNK)
        frames.append(chunk.flatten())
    return np.concatenate(frames) if frames else None


# ── Main loop ─────────────────────────────────────────────────────────────────

def chat_loop():
    messages = build_initial_messages()
    atexit.register(save_messages, messages)

    print(f"\nJARVIS online (Phase 7 — {MODEL} with thinking + network).")
    print("Say 'hey jarvis' or hold SPACE. Ctrl+C to quit.\n")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK)
    stream.start()

    try:
        while True:
            if keyboard.is_pressed("space"):
                print("\nSpacebar held — listening...")
                command_audio = record_while_held(stream)
            else:
                chunk, _ = stream.read(CHUNK)
                score = wake_model.predict(chunk.flatten()).get("hey_jarvis", 0)
                if score <= WAKE_THRESHOLD:
                    continue
                print("\nWake word detected — listening...")
                command_audio = record_command(stream)

            if command_audio is None or len(command_audio) < SAMPLE_RATE * 0.3:
                print("Didn't catch anything, still listening.")
                continue

            user_text = transcribe(command_audio)
            if not user_text:
                continue
            print(f"You said: {user_text}")

            # Decide whether to think
            thinking = should_think(user_text)
            print(f"[{'thinking' if thinking else 'fast mode'}]")

            messages.append({"role": "user", "content": user_text})
            patched = inject_thinking_token(messages, thinking)

            response = ollama.chat(model=MODEL, messages=patched, tools=ALL_TOOLS)
            msg = response["message"]
            # Store clean content (without the think token prefix) in history
            messages.append(to_assistant_message(msg))

            while msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args = tool_call["function"].get("arguments", {})
                    print(f"[tool] {name}({args})")
                    result = run_tool_call(tool_call)
                    print(f"[result] {result[:300]}")
                    messages.append({"role": "tool", "content": result})

                response = ollama.chat(model=MODEL, messages=messages, tools=ALL_TOOLS)
                msg = response["message"]
                messages.append(to_assistant_message(msg))

            # Strip internal reasoning before displaying / speaking
            reply = strip_think_block(msg.get("content", ""))
            print(f"JARVIS: {reply}")

            stream.stop()
            speak(reply)
            stream.start()

            save_messages(messages)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    chat_loop()