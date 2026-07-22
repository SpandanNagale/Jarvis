"""
ARIA_5.py  —  Phase 7 + 8: Intelligence + Network + Screen Awareness
                 + XTTS voice (ARIA_tts) + HUD overlay (ARIA_hud)
=======================================================
Extends ARIA_4 with:
  • Qwen3:14b — native chain-of-thought thinking (replaces Qwen2.5:14b)
  • Adaptive thinking — complex questions get /think, simple commands skip it
  • Network tools: web_search, fetch_page, search_github, search_wikipedia
  • Vision tools: analyze_screen, read_screen_text, analyze_region (qwen2.5vl:7b)
  • XTTS-v2 voice output (Piper fallback) — see ARIA_tts.py
  • Always-on-top HUD overlay showing state + live transcript/reply — see
    ARIA_hud.py. Qt owns the main thread, so the voice loop below runs on
    a background thread; see __main__ at the bottom of this file.

Wake word: still "hey jarvis" (see WAKE_WORD_MODEL_NAME below) — a custom
"hey aria" openWakeWord model is being trained separately (see
wakeword_training/); swap the constant once that model file exists.

New voice commands
------------------
  "Hey Jarvis, search me on GitHub"
  "Hey Jarvis, what is RAG in AI?"
  "Hey Jarvis, what's on my screen?"
  "Hey Jarvis, what does this error say?"
  "Hey Jarvis, read what's in the terminal"

Run:
    python ARIA_5.py
"""

import sys
import os
import re
import atexit
import threading

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
import keyboard
import ollama
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

from ARIA_1 import (
    TOOLS as BASE_TOOLS,
    TOOL_IMPLEMENTATIONS as BASE_IMPLS,
    to_assistant_message,
)
from ARIA_tools_coding import CODING_TOOLS, CODING_TOOL_IMPLEMENTATIONS
from ARIA_tools_memory import (
    MEMORY_TOOLS,
    MEMORY_TOOL_IMPLEMENTATIONS,
    build_initial_messages,
    save_messages,
)
from ARIA_tools_network import NETWORK_TOOLS, NETWORK_TOOL_IMPLEMENTATIONS
from ARIA_tools_vision import VISION_TOOLS, VISION_TOOL_IMPLEMENTATIONS
import ARIA_tts
from ARIA_tts import speak
import ARIA_hud

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL = "qwen3:14b"   # upgrade from qwen2.5:14b — same VRAM, adds native thinking
ARIA_tts.OLLAMA_MODEL = MODEL

# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = BASE_TOOLS + CODING_TOOLS + MEMORY_TOOLS + NETWORK_TOOLS + VISION_TOOLS
ALL_IMPLS = {
    **BASE_IMPLS,
    **CODING_TOOL_IMPLEMENTATIONS,
    **MEMORY_TOOL_IMPLEMENTATIONS,
    **NETWORK_TOOL_IMPLEMENTATIONS,
    **VISION_TOOL_IMPLEMENTATIONS,
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
    r"|what's my|volume|brightness|mute|unmute|list windows|hey jarvis|hey aria)",
    re.IGNORECASE,
)

_COMPLEX_PATTERNS = re.compile(
    r"(explain|why|how does|what is|what are|analyse|analyze|debug|diagnose"
    r"|refactor|suggest|compare|difference|search|find|look up|who is|tell me about"
    r"|summarize|summarise|research|github|wikipedia"
    r"|screen|my screen|what.s on|what does this|read this|what.s happening"
    r"|what.s wrong|error|terminal|window|showing|visible)",
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

# Swap to "hey_aria" (and drop the trained model file in) once the custom
# wake word model finishes training — see wakeword_training/.
WAKE_WORD_MODEL_NAME = "hey_jarvis"
WAKE_WORD_PHRASE     = "hey jarvis"

# ── Model loading ─────────────────────────────────────────────────────────────

print("Loading wake word model...")
wake_model = WakeWordModel(wakeword_models=[WAKE_WORD_MODEL_NAME])

print("Loading speech-to-text model...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def transcribe(audio_int16: np.ndarray) -> str:
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    segments, _ = stt_model.transcribe(audio_float32, language="en", vad_filter=True)
    return " ".join(seg.text for seg in segments).strip()


def record_command(stream, on_chunk=None) -> np.ndarray | None:
    frames, silence_count, speech_started = [], 0, False
    for _ in range(MAX_COMMAND_CHUNKS):
        chunk, _ = stream.read(CHUNK)
        chunk = chunk.flatten()
        frames.append(chunk)
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        if on_chunk:
            on_chunk(rms)
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


def record_while_held(stream, key="space", on_chunk=None) -> np.ndarray | None:
    frames = []
    while keyboard.is_pressed(key):
        chunk, _ = stream.read(CHUNK)
        chunk = chunk.flatten()
        frames.append(chunk)
        if on_chunk:
            on_chunk(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
    return np.concatenate(frames) if frames else None


# ── Main loop ─────────────────────────────────────────────────────────────────

def chat_loop():
    messages = build_initial_messages()
    atexit.register(save_messages, messages)

    print(f"\nARIA online (Phase 7 — {MODEL} with thinking + network).")
    print(f"Say '{WAKE_WORD_PHRASE}' or hold SPACE. Ctrl+C to quit.\n")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK)
    stream.start()

    def on_chunk(rms):
        ARIA_hud.push(state="listening", rms=rms)

    try:
        while True:
            ARIA_hud.push(state="idle")

            if keyboard.is_pressed("space"):
                print("\nSpacebar held — listening...")
                command_audio = record_while_held(stream, on_chunk=on_chunk)
            else:
                chunk, _ = stream.read(CHUNK)
                score = wake_model.predict(chunk.flatten()).get(WAKE_WORD_MODEL_NAME, 0)
                if score <= WAKE_THRESHOLD:
                    continue
                print("\nWake word detected — listening...")
                command_audio = record_command(stream, on_chunk=on_chunk)

            if command_audio is None or len(command_audio) < SAMPLE_RATE * 0.3:
                print("Didn't catch anything, still listening.")
                continue

            ARIA_hud.push(state="thinking")
            user_text = transcribe(command_audio)
            if not user_text:
                continue
            print(f"You said: {user_text}")
            ARIA_hud.push(transcript=user_text, reply="")

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
            print(f"ARIA: {reply}")
            ARIA_hud.push(state="speaking", reply=reply)

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
    app = ARIA_hud.start_hud()
    threading.Thread(target=chat_loop, daemon=True).start()
    sys.exit(app.exec())