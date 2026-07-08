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

from Jarvis_1 import TOOLS, run_tool_call, to_assistant_message, SYSTEM_PROMPT, MODEL

SAMPLE_RATE = 16000
CHUNK = 1280              # 80ms — openWakeWord requires exactly this frame size
WAKE_THRESHOLD = 0.30
SILENCE_RMS = 300         # tune this if it cuts you off early or waits too long
SPEECH_START_TIMEOUT_CHUNKS = 40  # ~3.2s grace period to actually start talking
MAX_SILENCE_CHUNKS = 25   # ~2s of quiet ends the command, once you've started
MAX_COMMAND_CHUNKS = 150  # ~12s hard cap so it can't listen forever

PIPER_VOICE = "en_US-sam-medium"   # match whatever you downloaded
PIPER_OUT = "jarvis_reply.wav"

print("Loading wake word model...")
wake_model = WakeWordModel(wakeword_models=["hey_jarvis"])

print("Loading speech-to-text model...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def transcribe(audio_int16: np.ndarray) -> str:
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    segments, _ = stt_model.transcribe(audio_float32, language="en", vad_filter=True)
    return " ".join(seg.text for seg in segments).strip()


def speak(text: str):
    try:
        safe_text = text.encode("utf-8", errors="ignore").decode("utf-8")
        subprocess.run(
            ["piper", "--model", PIPER_VOICE, "--output_file", PIPER_OUT],
            input=safe_text.encode("utf-8"),
            check=True,
        )
        data, samplerate = sf.read(PIPER_OUT, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"[TTS failed, skipping voice for this reply: {e}]")


def record_command(stream) -> np.ndarray | None:
    """Waits (generously) for the user to actually start talking, then
    records until they stop. Returns None if no speech ever started, so
    the caller can skip transcription instead of feeding silence to Whisper."""
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
    """Manual fallback: records for as long as the key is held, no
    silence-detection guesswork involved — exact control by design."""
    frames = []
    while keyboard.is_pressed(key):
        chunk, _ = stream.read(CHUNK)
        frames.append(chunk.flatten())
    if not frames:
        return None
    return np.concatenate(frames)


def chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("JARVIS online. Say 'hey jarvis' or hold SPACE to talk. Ctrl+C to quit.")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK)
    stream.start()

    try:
        while True:
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

            user_text = transcribe(command_audio)
            if not user_text:
                continue
            print(f"You said: {user_text}")

            messages.append({"role": "user", "content": user_text})
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            msg = response["message"]
            messages.append(to_assistant_message(msg))

            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    print(f"[tool call] {tool_call['function']['name']}({tool_call['function'].get('arguments')})")
                    result = run_tool_call(tool_call)
                    print(f"[tool result] {result}")
                    messages.append({"role": "tool", "content": result})
                response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
                msg = response["message"]
                messages.append(to_assistant_message(msg))
            else:
                print("[no tool called this turn]")

            print(f"JARVIS: {msg['content']}")

            # Pause the mic during playback so JARVIS doesn't hear itself
            # and re-trigger the wake word off its own voice.
            stream.stop()
            speak(msg["content"])
            stream.start()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    chat_loop()