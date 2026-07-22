"""
ARIA_tts.py  —  XTTS-v2 voice output with layered fallback

Drop-in replacement for the Piper-based speak(text) in ARIA_5.py. Given the
RTX 5070's 12GB VRAM is nearly full once Ollama (qwen3:14b) and faster-whisper
are both resident, synthesis is attempted in this order on every call:

    1. XTTS-v2 on GPU, models left as-is                 (fastest)
    2. On CUDA OOM: unload the Ollama model, retry on GPU
    3. Still OOM: run this utterance's synthesis on CPU
    4. Any other XTTS failure: fall back to Piper entirely

Set the environment variable ARIA_FORCE_PIPER=1 to skip XTTS and always use
Piper (manual escape hatch if XTTS is misbehaving mid-session).

Voice cloning: drop a 6-10s mono WAV of a single speaker, minimal background
noise, at REFERENCE_VOICE_WAV (defaults to "reference.wav" in the project
root). If that file isn't present, a built-in XTTS speaker is used instead.

Ollama model name for the unload step is configurable — ARIA_5.py sets
ARIA_tts.OLLAMA_MODEL to whatever MODEL it's using, since ARIA_tts can't
import that back (ARIA_5 imports this module, not the other way around).
"""

import os
import re
import subprocess

import numpy as np
import ollama
import sounddevice as sd
import soundfile as sf

OLLAMA_MODEL = "qwen3:14b"

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
REFERENCE_VOICE_WAV = "reference.wav"

PIPER_VOICE = "en_US-sam-medium"
PIPER_OUT = "aria_reply.wav"

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # skip interactive license prompt

# ---------------------------------------------------------------------------
# XTTS engine — loaded once at import time, same pattern as the wake word /
# whisper models in ARIA_5.py. If loading fails, we fall back to Piper for
# the whole session and never touch XTTS again.
# ---------------------------------------------------------------------------

_xtts = None
_xtts_device = "cuda"
_default_speaker = None

try:
    import torch
    from TTS.api import TTS

    print(f"[tts] Loading {XTTS_MODEL_NAME} on GPU...")
    _xtts = TTS(XTTS_MODEL_NAME, progress_bar=False).to(_xtts_device)
    _default_speaker = _xtts.speakers[0] if getattr(_xtts, "speakers", None) else None
    print("[tts] XTTS-v2 ready.")
except Exception as e:
    print(f"[tts] XTTS-v2 failed to load ({e}); using Piper for this session.")
    _xtts = None


def _is_oom(exc: Exception) -> bool:
    if _xtts is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _unload_ollama_model():
    """Ask Ollama to drop the resident LLM immediately, freeing VRAM for XTTS."""
    try:
        ollama.generate(model=OLLAMA_MODEL, prompt="", keep_alive=0)
        print(f"[tts] Unloaded {OLLAMA_MODEL} from Ollama to free VRAM.")
    except Exception as e:
        print(f"[tts] Couldn't unload Ollama model: {e}")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _synth(text: str, device: str) -> tuple[np.ndarray, int]:
    if device != _xtts_device_current():
        _xtts.to(device)
    kwargs = {"text": text, "language": "en"}
    if os.path.isfile(REFERENCE_VOICE_WAV):
        kwargs["speaker_wav"] = REFERENCE_VOICE_WAV
    elif _default_speaker:
        kwargs["speaker"] = _default_speaker
    wav = _xtts.tts(**kwargs)
    sr = _xtts.synthesizer.output_sample_rate
    return np.array(wav, dtype=np.float32), sr


def _xtts_device_current() -> str:
    return next(_xtts.synthesizer.tts_model.parameters()).device.type


def _play(audio: np.ndarray, sr: int):
    sd.play(audio, sr)
    sd.wait()


def _speak_xtts(text: str) -> bool:
    """Try the GPU -> unload-Ollama -> CPU fallback chain. Returns True on success."""
    sentences = _split_sentences(text) or [text]

    for attempt in ("gpu", "gpu_unloaded", "cpu"):
        try:
            device = "cpu" if attempt == "cpu" else "cuda"
            if attempt == "gpu_unloaded":
                torch.cuda.empty_cache()
                _unload_ollama_model()
            for sentence in sentences:
                audio, sr = _synth(sentence, device)
                _play(audio, sr)
            if device == "cpu":
                # Move back to GPU so the next call gets the fast path again.
                _xtts.to("cuda")
            return True
        except Exception as e:
            if attempt != "cpu" and _is_oom(e):
                print(f"[tts] GPU out of memory on '{attempt}' attempt, falling back...")
                continue
            print(f"[tts] XTTS synthesis failed ({e}); falling back to Piper.")
            return False
    return False


def _speak_piper(text: str):
    try:
        text = re.sub(r"[\ud800-\udfff]", "", text)
        text = re.sub(r"\*+|`+|#+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        safe = text.encode("ascii", errors="ignore").decode("ascii").strip()
        if not safe:
            print("[tts] No speakable text, skipping.")
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
        print(f"[tts] Piper fallback also failed: {e}")


# ---------------------------------------------------------------------------
# Public entry point — matches the old Piper speak(text) signature exactly.
# ---------------------------------------------------------------------------

def speak(text: str):
    text = re.sub(r"\*+|`+|#+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.strip()
    if not text:
        print("[tts] No speakable text, skipping.")
        return

    force_piper = os.environ.get("ARIA_FORCE_PIPER") == "1"
    if _xtts is not None and not force_piper:
        print("[tts] Speaking (XTTS)...")
        if _speak_xtts(text):
            return
    print("[tts] Speaking (Piper)...")
    _speak_piper(text)
