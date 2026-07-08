"""
Jarvis_tools_coding.py  —  Phase 5: Coding Assistant Tools

Provides tool implementations that work on files by partial name, folder hint,
full path, or clipboard — without any vision / OCR involvement:

  find_file(query)              — fuzzy-search common folders by partial name
  read_code_file(path)          — read a file (or clipboard) and show a preview
  explain_code(path)            — ask Ollama to explain the code in the file
  suggest_refactor(path)        — ask Ollama for refactor suggestions
  diagnose_traceback(text)      — paste / speak a traceback and get a diagnosis

Path resolution priority (used by all file tools)
--------------------------------------------------
1. If path is empty / "clipboard" → read from clipboard
2. If path is an exact existing file → use it directly
3. Otherwise → fuzzy-search SEARCH_ROOTS for a filename that contains the hint
   (folder keywords like "downloads", "documents" narrow the search).
   Returns the best single match or an error listing close candidates.

Design notes
------------
* Clipboard is read on demand with pyperclip (pure-Python, no extra binary).
* Ollama calls here use the SAME model as the main chat but WITHOUT the tool
  schema so the model returns plain prose, not tool calls.
* Each tool returns a concise plain-English result string — same contract as
  the other tools — so run_tool_call() in Jarvis_4 needs no changes.
"""

import os
import difflib
import ollama
import pyperclip
from pathlib import Path

from Jarvis_1 import MODEL

# ---------------------------------------------------------------------------
# Search roots — folders Jarvis will look inside (non-recursive at depth 1
# for speed; sub-folder names in the query narrow to specific roots)
# ---------------------------------------------------------------------------

_HOME = Path.home()
_CWD  = Path.cwd()

# Common locations ordered by likelihood.
# cwd and its parent are listed first so files in the active project folder
# are always found immediately, regardless of nesting depth.
SEARCH_ROOTS: list[Path] = [
    _CWD,                                  # folder JARVIS is run from
    _CWD.parent,                           # one level up (e.g. "Fun project")
    _HOME / "OneDrive" / "Desktop",
    _HOME / "Desktop",
    _HOME / "OneDrive" / "Documents",
    _HOME / "Documents",
    _HOME / "Downloads",
    _HOME / "OneDrive" / "Downloads",
    _HOME / "OneDrive",
    _HOME / "Pictures",
    _HOME / "Videos",
    _HOME / "Music",
    _HOME,
    Path("C:/Users") / os.environ.get("USERNAME", "") / "Downloads",
]

# Remove duplicates and non-existent roots
_seen: set[Path] = set()
SEARCH_ROOTS = [
    r for r in SEARCH_ROOTS
    if r.exists() and r not in _seen and not _seen.add(r)  # type: ignore[func-returns-value]
]

# Folder alias → actual root path (spoken keywords the LLM might pass)
_FOLDER_ALIASES: dict[str, str] = {
    "downloads": "Downloads",
    "download": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents",
    "document": "Documents",
    "docs": "Documents",
    "onedrive": "OneDrive",
    "pictures": "Pictures",
    "videos": "Videos",
    "music": "Music",
    "home": "",          # home root itself
}

# Max chars we'll send to the model for a single analysis — keeps prompts sane
MAX_CODE_CHARS = 8_000

# Max search depth when walking directories — 5 covers deeply nested project
# folders like OneDrive\Desktop\VS code\Fun project\Jarvis\
MAX_DEPTH = 5


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _candidate_roots(folder_hint: str | None) -> list[Path]:
    """Return the subset of SEARCH_ROOTS to search, narrowed by folder hint."""
    if not folder_hint:
        return SEARCH_ROOTS
    key = folder_hint.strip().lower()
    alias = _FOLDER_ALIASES.get(key)
    if alias == "":                         # "home" → search home root
        return [_HOME]
    if alias:
        candidates = [r for r in SEARCH_ROOTS if r.name == alias]
        if candidates:
            return candidates
    # If alias unknown, still fall back to all roots
    return SEARCH_ROOTS


def _walk(root: Path, depth: int = 0) -> list[Path]:
    """Yield all files under root up to MAX_DEPTH."""
    files: list[Path] = []
    if depth > MAX_DEPTH:
        return files
    try:
        for entry in root.iterdir():
            if entry.is_file():
                files.append(entry)
            elif entry.is_dir() and not entry.name.startswith("."):
                files.extend(_walk(entry, depth + 1))
    except PermissionError:
        pass
    return files


def _score(query_lower: str, filename_lower: str) -> float:
    """Simple relevance score: exact substring > word overlap > sequence match."""
    if query_lower in filename_lower:
        return 1.0
    q_words = set(query_lower.replace("_", " ").replace("-", " ").split())
    f_words = set(filename_lower.replace("_", " ").replace("-", " ").replace(".", " ").split())
    overlap = len(q_words & f_words) / max(len(q_words), 1)
    if overlap > 0:
        return 0.5 + overlap * 0.4
    return difflib.SequenceMatcher(None, query_lower, filename_lower).ratio() * 0.5


def _resolve_path(path_hint: str) -> tuple[Path, str]:
    """Resolve a partial path / filename hint to a real file.

    Returns (resolved_path, label) or raises FileNotFoundError with suggestions.
    """
    hint = path_hint.strip()

    # Check if it's already a valid absolute or relative path
    p = Path(hint)
    if p.is_file():
        return p, str(p)

    # Pull out any folder keyword from the hint so we can narrow the search.
    # e.g. "internship report in downloads" → query="internship report", folder="downloads"
    folder_hint: str | None = None
    query = hint.lower()
    for kw in _FOLDER_ALIASES:
        for sep in (f" in {kw}", f" from {kw}", f" on {kw}"):
            if sep in query:
                query = query.replace(sep, "").strip()
                folder_hint = kw
                break
        if folder_hint:
            break

    roots = _candidate_roots(folder_hint)

    # Collect all files from the selected roots and score them
    scored: list[tuple[float, Path]] = []
    for root in roots:
        for f in _walk(root):
            s = _score(query, f.name.lower())
            if s > 0.1:
                scored.append((s, f))

    if not scored:
        searched = ", ".join(str(r) for r in roots[:4])
        raise FileNotFoundError(
            f"No file matching '{hint}' found in: {searched}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_file = scored[0]

    # If the top match is confident enough (substring match), use it directly
    if best_score >= 0.9:
        return best_file, str(best_file)

    # If there are multiple close matches, ask the user to be more specific
    top = scored[:4]
    if len(top) > 1 and top[0][0] - top[1][0] < 0.05:
        options = "; ".join(f.name for _, f in top)
        raise FileNotFoundError(
            f"Found {len(top)} similar files — be more specific. Options: {options}"
        )

    return best_file, str(best_file)


def _read_source(path: str) -> tuple[str, str]:
    """Return (source_text, label).

    If path is empty / 'clipboard' → clipboard.
    Otherwise try exact path first, then fuzzy resolve.
    """
    if not path or path.strip().lower() in ("", "clipboard", "clip"):
        text = pyperclip.paste()
        return text, "clipboard"

    # Exact path first
    p = Path(path.strip())
    if p.is_file():
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(), str(p)

    # Fuzzy resolve
    resolved, label = _resolve_path(path)   # raises FileNotFoundError with suggestions
    with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(), label


def _trim(code: str) -> str:
    if len(code) > MAX_CODE_CHARS:
        return code[:MAX_CODE_CHARS] + "\n\n... [truncated for length]"
    return code


def _ask_model(system: str, user: str) -> str:
    """Fire a one-shot Ollama chat without tools and return the text reply."""
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def find_file(query: str) -> str:
    """Search common folders for a file matching the query and return its path."""
    if not query.strip():
        return "Please provide a filename or partial name to search for."
    try:
        resolved, label = _resolve_path(query)
        return f"Found: {label}"
    except FileNotFoundError as e:
        return str(e)


def read_code_file(path: str = "") -> str:
    """Read a source file (or the clipboard) and return a short preview."""
    try:
        code, label = _read_source(path)
    except FileNotFoundError as e:
        return str(e)
    if not code.strip():
        return f"'{label}' appears to be empty."
    lines = code.splitlines()
    preview = "\n".join(lines[:20])
    more = f"\n... (+{len(lines)-20} more lines)" if len(lines) > 20 else ""
    return f"Read {len(lines)} lines from {label}:\n{preview}{more}"


def explain_code(path: str = "") -> str:
    """Explain what the code in the given file (or clipboard) does."""
    try:
        code, label = _read_source(path)
    except FileNotFoundError as e:
        return str(e)
    if not code.strip():
        return f"'{label}' is empty — nothing to explain."
    system = (
        "You are a senior software engineer. Explain the following code clearly "
        "and concisely in plain English. Focus on what it does and why. "
        "No markdown, no bullet points — write as if speaking aloud."
    )
    user = f"Code from {label}:\n\n{_trim(code)}"
    return _ask_model(system, user)


def suggest_refactor(path: str = "") -> str:
    """Suggest concrete refactoring improvements for the code in the file."""
    try:
        code, label = _read_source(path)
    except FileNotFoundError as e:
        return str(e)
    if not code.strip():
        return f"'{label}' is empty — nothing to refactor."
    system = (
        "You are a senior software engineer doing a code review. "
        "Suggest the top three most impactful refactoring improvements for the "
        "code below. Be specific and practical. No markdown, no bullet points — "
        "write naturally as if speaking your suggestions aloud."
    )
    user = f"Code from {label}:\n\n{_trim(code)}"
    return _ask_model(system, user)


def diagnose_traceback(text: str = "") -> str:
    """Diagnose a Python traceback or error message and suggest a fix."""
    if not text or text.strip().lower() in ("clipboard", "clip", ""):
        text = pyperclip.paste()
    if not text.strip():
        return "No traceback text found in the argument or clipboard."
    system = (
        "You are an expert Python debugger. Given the traceback or error message "
        "below, explain the root cause in one or two sentences and suggest the "
        "most likely fix. No markdown, no bullet points — speak naturally."
    )
    user = f"Error / traceback:\n\n{_trim(text)}"
    return _ask_model(system, user)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

CODING_TOOL_IMPLEMENTATIONS = {
    "find_file": find_file,
    "read_code_file": read_code_file,
    "explain_code": explain_code,
    "suggest_refactor": suggest_refactor,
    "diagnose_traceback": diagnose_traceback,
}

CODING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": (
                "Search the user's common folders (Downloads, Desktop, Documents, "
                "OneDrive, etc.) for a file matching a partial name or description. "
                "Use this when the user mentions a file by name without giving a full path, "
                "e.g. 'internship report in downloads' or 'main.py in my project'. "
                "Returns the full path if found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Partial filename or description, optionally with a folder hint. "
                            "Examples: 'internship report in downloads', 'main.py', "
                            "'resume on desktop'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_code_file",
            "description": (
                "Read a file from disk and show a short preview. "
                "Accepts a full path, a partial filename with optional folder hint "
                "(e.g. 'internship report in downloads'), or 'clipboard'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path, partial filename, or 'clipboard'. "
                            "Include folder hints like 'in downloads' or 'on desktop' "
                            "to narrow the search."
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
            "name": "explain_code",
            "description": (
                "Explain what a file's content does in plain English. "
                "Accepts a full path, a partial filename with optional folder hint "
                "(e.g. 'Jarvis_4.py in my project'), or 'clipboard'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path, partial filename with folder hint, or 'clipboard'."
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
            "name": "suggest_refactor",
            "description": (
                "Suggest refactoring improvements for the code in a file or clipboard. "
                "Accepts a full path, a partial filename with optional folder hint, or 'clipboard'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Full path, partial filename with folder hint, or 'clipboard'."
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
            "name": "diagnose_traceback",
            "description": (
                "Diagnose a Python traceback or error message and suggest a fix. "
                "The text can be passed directly or read from the clipboard."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The traceback or error text. "
                            "Leave empty or pass 'clipboard' to read from clipboard."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
]