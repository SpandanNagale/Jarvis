"""
Jarvis_tools_vision.py  —  Phase 8: Screen Awareness

Tools:
    analyze_screen(question)          — Full screenshot + VLM analysis
    read_screen_text()                — Optimised for reading text/code on screen
    analyze_region(x, y, w, h, q)    — Analyse a specific screen region

Model: qwen2.5vl:7b (pulled separately)
Ollama automatically unloads qwen3:14b and loads qwen2.5vl:7b when vision
tools are called, then swaps back on the next text turn. Expect a 3-5s
swap delay on the first vision call per session; subsequent calls are fast.

Design notes
------------
* PIL.ImageGrab captures the screen to a temp file — no extra screen
  capture libraries needed, works on Windows natively.
* Screenshots are downscaled to 1920x1080 max before sending to the VLM
  to keep token count reasonable and avoid Ollama's resize artifacts.
* The voice reply is a short spoken summary; the full analysis is printed
  to the console so you can read the details.
"""

import base64
import tempfile
import os
from pathlib import Path
from PIL import ImageGrab, Image
import ollama

VISION_MODEL = "qwen2.5vl:7b"
MAX_SCREENSHOT_SIZE = (1920, 1080)


def _capture_screen(region: tuple | None = None) -> str:
    """Capture the screen (or a region) and save to a temp PNG.
    Returns the file path."""
    img = ImageGrab.grab(bbox=region, all_screens=True)

    # Downscale if larger than MAX_SCREENSHOT_SIZE to keep tokens manageable
    if img.size[0] > MAX_SCREENSHOT_SIZE[0] or img.size[1] > MAX_SCREENSHOT_SIZE[1]:
        img.thumbnail(MAX_SCREENSHOT_SIZE, Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, "PNG")
    tmp.close()
    return tmp.name


def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ask_vision(image_path: str, prompt: str) -> str:
    """Send an image to qwen2.5vl:7b and return the response."""
    try:
        image_b64 = _image_to_base64(image_path)
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Vision model error: {e}"
    finally:
        # Clean up temp file
        try:
            os.unlink(image_path)
        except Exception:
            pass


def _spoken_summary(full_text: str, max_sentences: int = 3) -> str:
    """Return a short spoken-friendly version of a longer analysis."""
    sentences = [s.strip() for s in full_text.replace("\n", " ").split(".") if s.strip()]
    short = ". ".join(sentences[:max_sentences])
    if short and not short.endswith("."):
        short += "."
    return short or full_text[:300]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def analyze_screen(question: str = "What is on my screen right now?") -> str:
    """Take a full screenshot and analyse it with a vision model."""
    print("[vision] Capturing screen...")
    path = _capture_screen()
    print(f"[vision] Sending to {VISION_MODEL}...")
    full = _ask_vision(path, question)
    print(f"[vision full analysis]\n{full}\n")
    # Return a short spoken summary — the full text is already printed
    spoken = _spoken_summary(full, max_sentences=3)
    return f"(Full analysis printed to console.) {spoken}"


def read_screen_text() -> str:
    """Screenshot optimised for reading code, error messages, or UI text."""
    print("[vision] Capturing screen for text reading...")
    path = _capture_screen()
    prompt = (
        "Read and transcribe all visible text on this screen. "
        "Focus on: error messages, code, terminal output, dialog boxes, "
        "and any important labels or values. Format clearly."
    )
    print(f"[vision] Sending to {VISION_MODEL}...")
    full = _ask_vision(path, prompt)
    print(f"[vision text read]\n{full}\n")
    spoken = _spoken_summary(full, max_sentences=2)
    return f"(Full text printed to console.) {spoken}"


def analyze_region(x: int, y: int, width: int, height: int,
                   question: str = "What is in this region?") -> str:
    """Capture and analyse a specific screen region (pixel coordinates).

    Useful when you know roughly where something is —
    e.g. the top-left error panel, or a specific browser tab.
    """
    region = (x, y, x + width, y + height)
    print(f"[vision] Capturing region {region}...")
    path = _capture_screen(region=region)
    print(f"[vision] Sending to {VISION_MODEL}...")
    full = _ask_vision(path, question)
    print(f"[vision region analysis]\n{full}\n")
    spoken = _spoken_summary(full, max_sentences=3)
    return f"(Full analysis printed to console.) {spoken}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

VISION_TOOL_IMPLEMENTATIONS = {
    "analyze_screen": analyze_screen,
    "read_screen_text": read_screen_text,
    "analyze_region": analyze_region,
}

VISION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_screen",
            "description": (
                "Take a screenshot of the full screen and analyse it with a vision model. "
                "Use when the user asks 'what's on my screen', 'what does this show', "
                "'what am I looking at', or any question about visible content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The specific question to ask about the screen. "
                            "Defaults to a general description if not provided."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_screen_text",
            "description": (
                "Screenshot the screen and extract all readable text — error messages, "
                "code, terminal output, UI labels. Use when the user says 'read this', "
                "'what does this error say', 'what does the terminal show', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_region",
            "description": (
                "Capture and analyse a specific pixel region of the screen. "
                "Use when the user specifies a location like 'the top-right corner' "
                "or 'the error panel on the left'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Left edge of the region in pixels.",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Top edge of the region in pixels.",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Width of the region in pixels.",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Height of the region in pixels.",
                    },
                    "question": {
                        "type": "string",
                        "description": "What to ask about the region.",
                    },
                },
                "required": ["x", "y", "width", "height"],
            },
        },
    },
]