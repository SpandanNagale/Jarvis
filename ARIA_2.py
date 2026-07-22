import sys
import os

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
            print(f"Checking {bin_dir} -> exists: {os.path.isdir(bin_dir)}")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                found_dirs.append(bin_dir)
        if found_dirs:
            os.environ["PATH"] = os.pathsep.join(found_dirs) + os.pathsep + os.environ.get("PATH", "")
    except ImportError as e:
        print(f"CUDA DLL packages not found ({e}) — falling back to whatever's on PATH.")

import sounddevice as sd
import numpy as np
import keyboard
import soundfile as sf
import subprocess
import shutil
import ollama
from faster_whisper import WhisperModel

from ARIA_1 import TOOLS, run_tool_call, SYSTEM_PROMPT, MODEL

SAMPLE_RATE = 16000
PIPER_VOICE = "en_US-sam-medium"   # swap for whichever voice you picked
PIPER_OUT = "aria_reply.wav"

print("Loading speech-to-text model...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def record_while_held(key="space"):
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    print("\nHold SPACE and speak, release when done...")
    keyboard.wait(key)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        while keyboard.is_pressed(key):
            sd.sleep(30)

    if not frames:
        return None
    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    segments, _ = stt_model.transcribe(audio, language="en")
    return " ".join(seg.text for seg in segments).strip()


def speak(text: str):
    # Find the piper executable path, checking system PATH and the current virtual env's Scripts directory
    piper_cmd = shutil.which("piper")
    if not piper_cmd:
        py_dir = os.path.dirname(sys.executable)
        maybe_piper = shutil.which("piper", path=py_dir)
        if maybe_piper:
            piper_cmd = maybe_piper
        else:
            piper_cmd = "piper"  # Fallback

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(
        [piper_cmd, "--model", PIPER_VOICE, "--output_file", PIPER_OUT],
        input=text.encode("utf-8"),
        env=env,
        check=True,
    )
    data, samplerate = sf.read(PIPER_OUT, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("ARIA online. Hold SPACE to talk, Ctrl+C to quit.")

    while True:
        audio = record_while_held()
        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            continue

        user_text = transcribe(audio)
        if not user_text:
            continue
        print(f"You said: {user_text}")

        messages.append({"role": "user", "content": user_text})
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = response["message"]
        messages.append(msg)

        if msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                result = run_tool_call(tool_call)
                messages.append({"role": "tool", "content": result})
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            msg = response["message"]
            messages.append(msg)

        print(f"ARIA: {msg['content']}")
        speak(msg["content"])


if __name__ == "__main__":
    chat_loop()