"""Shared configuration constants used across ARIA's modules.

Exists mainly to break the import cycle between aria.py (the main entry
point) and tools/coding.py, which both need MODEL.
"""

MODEL = "qwen3:14b"
