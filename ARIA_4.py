"""
ARIA_4.py  —  Phase 5 + 6 integration
========================================
Extends ARIA_3 with:
  • Phase 5: Coding assistant tools (read file, explain, refactor, diagnose)
  • Phase 6: ChromaDB memory (remember facts, recall across sessions, personality)

Run this file exactly like ARIA_3.py:
    python ARIA_4.py

Nothing else to start — ChromaDB is fully embedded (no server needed).

New voice commands you can try
-------------------------------
  "Hey Jarvis, remember that I prefer dark mode."
  "Hey Jarvis, what do you know about my preferences?"
  "Hey Jarvis, explain the code in C:/Users/shree/project/main.py"
  "Hey Jarvis, what's wrong with this?" (with a traceback on the clipboard)
  "Hey Jarvis, suggest a refactor for my clipboard"
  "Hey Jarvis, forget everything — list what you know first."
"""

import sys
import os

# ── cuBLAS / cuDNN DLL fix (must be before any faster_whisper import) ────────
if sys.platform == "win32":
    try:
        import nvidia.cublas
        import nvidia.cudnn
        found_dirs = []
        for pkg in (nvidia.cublas, nvidia.cudnn):
            if getattr(pkg, "__file__", None):
                pkg_dir = os.path.dirname(pkg.__file__)
            else:
                pkg_dir = list(pkg.__path__)[0]
            bin_dir = os.path.join(pkg_dir, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                found_dirs.append(bin_dir)
        if found_dirs:
            os.environ["PATH"] = os.pathsep.join(found_dirs) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

import atexit
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
import keyboard
import ollama
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

# ── Tool layers ──────────────────────────────────────────────────────────────
from ARIA_1 import (
    TOOLS as BASE_TOOLS,
    TOOL_IMPLEMENTATIONS as BASE_IMPLS,
    run_tool_call as _base_run_tool_call,
    to_assistant_message,
    MODEL,
)
from ARIA_tools_coding import CODING_TOOLS, CODING_TOOL_IMPLEMENTATIONS
from ARIA_tools_memory import (
    MEMORY_TOOLS,
    MEMORY_TOOL_IMPLEMENTATIONS,
    build_initial_messages,
    save_messages,
)

# Merge all tools into one flat registry
ALL_TOOLS = BASE_TOOLS + CODING_TOOLS + MEMORY_TOOLS
ALL_IMPLS = {**BASE_IMPLS, **CODING_TOOL_IMPLEMENTATIONS, **MEMORY_TOOL_IMPLEMENTATIONS}


def run_tool_call(tool_call) -> str:
    name = tool_call["function"]["name"]
    args = tool_call["function"].get("arguments", {}) or {}
    func = ALL_IMPLS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    return func(**args)


# ── Audio / STT / TTS config ─────────────────────────────────────────────────
SAMPLE_RATE              = 16000
CHUNK                    = 1280          # 80 ms — openWakeWord requires exactly this
WAKE_THRESHOLD           = 0.30
SILENCE_RMS              = 300           # tune per mic
SPEECH_START_TIMEOUT_CHUNKS = 40         # ~3.2 s grace period
MAX_SILENCE_CHUNKS       = 25            # ~2 s of quiet ends the command
MAX_COMMAND_CHUNKS       = 150           # ~12 s hard cap

PIPER_VOICE = "en_US-sam-medium"
PIPER_OUT   = "aria_reply.wav"

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
        import re
        # 1. Drop lone surrogate code points (\uD800–\uDFFF) — they crash
        #    espeak's UTF-8 encoder with "surrogates not allowed".
        text = re.sub(r"[\ud800-\udfff]", "", text)
        # 2. Keep only ASCII — the English Piper model can't phonemize CJK,
        #    Thai, Arabic, etc., and espeak will either crash or produce garbage.
        safe_text = text.encode("ascii", errors="ignore").decode("ascii").strip()
        if not safe_text:
            print("[TTS] Reply contained no speakable ASCII text, skipping voice.]")
            return
        subprocess.run(
            ["piper", "--model", PIPER_VOICE, "--output_file", PIPER_OUT],
            input=safe_text.encode("utf-8"),
            check=True,
        )
        data, samplerate = sf.read(PIPER_OUT, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"[TTS failed, skipping voice: {e}]")


def record_command(stream) -> np.ndarray | None:
    """Wait for speech to start, then record until silence.
    Returns None if no speech detected within the timeout."""
    frames = []
    silence_count = 0
    speech_started = False

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

    if not speech_started:
        return None
    return np.concatenate(frames)


def record_while_held(stream, key="space") -> np.ndarray | None:
    """Manual PTT fallback — exact control, no silence detection."""
    frames = []
    while keyboard.is_pressed(key):
        chunk, _ = stream.read(CHUNK)
        frames.append(chunk.flatten())
    if not frames:
        return None
    return np.concatenate(frames)


# ── Main loop ─────────────────────────────────────────────────────────────────

def chat_loop():
    # Phase 6: restore previous session history + inject known facts into system
    messages = build_initial_messages()

    # Save history automatically on exit (Ctrl+C or normal termination)
    atexit.register(save_messages, messages)

    print("JARVIS online (Phase 5+6). Say 'hey jarvis' or hold SPACE. Ctrl+C to quit.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK
    )
    stream.start()

    try:
        while True:
            # ── Wake detection ──────────────────────────────────────────────
            if keyboard.is_pressed("space"):
                print("\nSpacebar held — listening...")
                command_audio = record_while_held(stream)
            else:
                chunk, _ = stream.read(CHUNK)
                chunk = chunk.flatten()
                prediction = wake_model.predict(chunk)
                score = prediction.get("hey_jarvis", 0)
                if score <= WAKE_THRESHOLD:
                    continue
                print("\nWake word detected — listening...")
                command_audio = record_command(stream)

            if command_audio is None or len(command_audio) < SAMPLE_RATE * 0.3:
                print("Didn't catch anything, still listening.")
                continue

            # ── Transcribe ─────────────────────────────────────────────────
            user_text = transcribe(command_audio)
            if not user_text:
                continue
            print(f"You said: {user_text}")

            messages.append({"role": "user", "content": user_text})

            # ── LLM turn (with tool loop) ───────────────────────────────────
            response = ollama.chat(model=MODEL, messages=messages, tools=ALL_TOOLS)
            msg = response["message"]
            messages.append(to_assistant_message(msg))

            while msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args = tool_call["function"].get("arguments", {})
                    print(f"[tool call] {name}({args})")
                    result = run_tool_call(tool_call)
                    print(f"[tool result] {result[:200]}")   # truncate long code blocks
                    messages.append({"role": "tool", "content": result})

                response = ollama.chat(model=MODEL, messages=messages, tools=ALL_TOOLS)
                msg = response["message"]
                messages.append(to_assistant_message(msg))

            print(f"JARVIS: {msg['content']}")

            # ── Speak (mic paused so JARVIS doesn't hear itself) ────────────
            stream.stop()
            speak(msg["content"])
            stream.start()

            # ── Persist history snapshot after every turn ───────────────────
            save_messages(messages)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        # atexit handler will call save_messages one final time


if __name__ == "__main__":
    chat_loop()