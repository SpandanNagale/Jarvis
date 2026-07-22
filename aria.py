"""
aria.py  —  ARIA main entry point

A local, offline-first voice assistant: Ollama (Qwen3:14b) for reasoning and
tool-calling, faster-whisper for speech-to-text, XTTS-v2 for voice output
(Piper as automatic fallback — see voice/tts.py), openWakeWord for wake-word
detection, and an always-on-top HUD overlay showing live state (idle /
listening / thinking / speaking) plus transcript/reply text (see ui/hud.py).

Qt must own the main thread on Windows, so the voice loop below runs on a
background thread; see __main__ at the bottom of this file.

Wake word: still "hey jarvis" (see WAKE_WORD_MODEL_NAME below) — a custom
"hey aria" openWakeWord model is being trained separately (see
wakeword_training/); swap the constant once that model file exists.

Tool categories
----------------
  System control  — open/close apps, volume, brightness, lock, windows
  tools/coding.py    — read/explain/refactor files, diagnose tracebacks
  tools/memory.py    — persistent facts + cross-session history (ChromaDB)
  tools/network.py   — web search, page fetch, GitHub, Wikipedia
  tools/vision.py    — screenshot analysis via a vision model

Voice commands
--------------
  "Hey Jarvis, what time is it?"
  "Hey Jarvis, open notepad."
  "Hey Jarvis, remember that I prefer dark mode."
  "Hey Jarvis, search me on GitHub."
  "Hey Jarvis, what's on my screen?"

Run:
    python aria.py
"""

import sys
import os
import re
import subprocess
import webbrowser
import atexit
import threading
from datetime import datetime

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
import psutil
import keyboard
import ollama
from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc
import pygetwindow as gw
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

from config import MODEL
from tools.coding import CODING_TOOLS, CODING_TOOL_IMPLEMENTATIONS
from tools.memory import (
    MEMORY_TOOLS,
    MEMORY_TOOL_IMPLEMENTATIONS,
    build_initial_messages,
    save_messages,
)
from tools.network import NETWORK_TOOLS, NETWORK_TOOL_IMPLEMENTATIONS
from tools.vision import VISION_TOOLS, VISION_TOOL_IMPLEMENTATIONS
from voice import tts
from voice.tts import speak
from ui import hud

tts.OLLAMA_MODEL = MODEL

# ── Core system-control tools ──────────────────────────────────────────────────

def open_app(app_name: str) -> str:
    # Add your own apps here. If something isn't on your system PATH
    # (most third-party apps aren't), use the full .exe path instead,
    # e.g. r"C:\Program Files\Steam\steam.exe".
    apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "explorer": "explorer.exe",
        "chrome": "chrome.exe",
        "task manager": "taskmgr.exe",
    }
    target = apps.get(app_name.lower())
    if not target:
        return f"I don't have '{app_name}' mapped to an application yet."
    try:
        subprocess.Popen(target)
        return f"Opened {app_name}."
    except FileNotFoundError:
        return f"Couldn't find {app_name} on this system."


def close_app(app_name: str) -> str:
    # Maps spoken names to actual process names. Check Task Manager's
    # "Details" tab if an app's real process name isn't obvious.
    process_names = {
        "notepad": "notepad.exe",
        "calculator": "calculatorapp.exe",
        "explorer": "explorer.exe",
        "chrome": "chrome.exe",
        "task manager": "taskmgr.exe",
    }
    target = process_names.get(app_name.lower(), app_name.lower().replace(" ", "") + ".exe")
    closed = False
    for proc in psutil.process_iter(["name"]):
        if proc.info["name"] and proc.info["name"].lower() == target:
            proc.terminate()
            closed = True
    return f"Closed {app_name}." if closed else f"Couldn't find a running process for {app_name}."


def get_system_status() -> str:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    return f"CPU usage is {cpu}%, RAM usage is {ram}%."


def get_datetime() -> str:
    return datetime.now().strftime("It's %I:%M %p on %A, %B %d.")


def set_volume(level: int) -> str:
    level = max(0, min(100, level))
    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume
    volume.SetMasterVolumeLevelScalar(level / 100, None)
    return f"Volume set to {level}%."


def set_brightness(level: int) -> str:
    level = max(0, min(100, level))
    try:
        sbc.set_brightness(level)
        return f"Brightness set to {level}%."
    except Exception as e:
        return f"Couldn't change brightness on this display: {e}"


def lock_pc() -> str:
    subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
    return "Locking the PC."


def open_url(url: str) -> str:
    if not url.startswith("http"):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}."


def list_open_windows() -> str:
    titles = [t for t in gw.getAllTitles() if t.strip()]
    if not titles:
        return "No open windows found."
    return "Open windows: " + ", ".join(titles[:10])


BASE_TOOL_IMPLEMENTATIONS = {
    "open_app": open_app,
    "close_app": close_app,
    "get_system_status": get_system_status,
    "get_datetime": get_datetime,
    "set_volume": set_volume,
    "set_brightness": set_brightness,
    "lock_pc": lock_pc,
    "open_url": open_url,
    "list_open_windows": list_open_windows,
}

BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application on the PC, such as notepad, calculator, or chrome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the application, e.g. 'notepad' or 'calculator'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_app",
            "description": "Close a running application by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the application to close, e.g. 'notepad' or 'chrome'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Get current CPU and RAM usage percentages.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the system master volume to a specific level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Volume level from 0 (mute) to 100 (max).",
                    }
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_brightness",
            "description": "Set the screen brightness to a specific level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Brightness level from 0 to 100.",
                    }
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_pc",
            "description": "Lock the PC, requiring a password to unlock.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a website in the default browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL or domain to open, e.g. 'youtube.com'.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_open_windows",
            "description": "List the titles of currently open windows.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def to_assistant_message(msg) -> dict:
    """Strip the Ollama response down to a clean dict before it goes back
    into history — passing the raw object back as-is can confuse the
    model's chat template on later turns and produce garbled output."""
    clean = {"role": "assistant", "content": msg.get("content", "") or ""}
    if msg.get("tool_calls"):
        clean["tool_calls"] = msg["tool_calls"]
    return clean


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = BASE_TOOLS + CODING_TOOLS + MEMORY_TOOLS + NETWORK_TOOLS + VISION_TOOLS
ALL_IMPLS = {
    **BASE_TOOL_IMPLEMENTATIONS,
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

    print(f"\nARIA online ({MODEL} with thinking + network).")
    print(f"Say '{WAKE_WORD_PHRASE}' or hold SPACE. Ctrl+C to quit.\n")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK)
    stream.start()

    def on_chunk(rms):
        hud.push(state="listening", rms=rms)

    try:
        while True:
            hud.push(state="idle")

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

            hud.push(state="thinking")
            user_text = transcribe(command_audio)
            if not user_text:
                continue
            print(f"You said: {user_text}")
            hud.push(transcript=user_text, reply="")

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
            hud.push(state="speaking", reply=reply)

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
    app = hud.start_hud()
    threading.Thread(target=chat_loop, daemon=True).start()
    sys.exit(app.exec())
