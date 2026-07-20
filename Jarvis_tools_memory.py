"""
Jarvis_tools_memory.py  —  Phase 6: Memory + Personality (ChromaDB)

Provides a lightweight persistent memory layer on top of ChromaDB so JARVIS
can remember facts, past corrections, and recurring context across restarts.

Two public surfaces
-------------------
1.  Tool implementations (MEMORY_TOOL_IMPLEMENTATIONS / MEMORY_TOOLS)
    Jarvis_4 merges these into the global tool registry so the LLM can call
    them from any turn:

        remember_fact(fact)           — persist a free-text fact
        recall_facts(query)           — semantic search over stored facts
        forget_fact(fact_id)          — delete by ChromaDB document ID
        list_facts()                  — dump all stored facts (up to 20)

2.  Session-history persistence helpers (called directly by Jarvis_4)
        save_messages(msgs)           — snapshot the current message list
        load_messages()               — restore the last session's history
        build_initial_messages()      — build the messages[] list for a new
                                        session, pre-loaded with relevant facts
                                        injected into the system prompt

Design notes
------------
* ChromaDB runs fully embedded (no server process needed).
* Two collections: "jarvis_facts" for user-stated facts, "jarvis_history" for
  the raw assistant/user message pairs from the last session.
* Embeddings use ChromaDB's default sentence-transformers model — no extra GPU
  needed; it runs fine on CPU for these small payloads.
* History is limited to the last MAX_HISTORY_TURNS pairs to keep token counts
  reasonable on restart.
* The SYSTEM_PROMPT here is the Phase 6 personality-tuned version; Jarvis_4
  imports it instead of the plain one in Jarvis_1.
"""

import json
import uuid
import chromadb
from chromadb.config import Settings
from pathlib import Path

# ---------------------------------------------------------------------------
# ChromaDB setup — persists in a 'memory/' subfolder next to this script
# ---------------------------------------------------------------------------

_DB_PATH = str(Path(__file__).parent / "memory")
_client = chromadb.PersistentClient(path=_DB_PATH)
_facts_col = _client.get_or_create_collection("jarvis_facts")
_history_col = _client.get_or_create_collection("jarvis_history")

MAX_RECALL_RESULTS = 5
MAX_HISTORY_TURNS  = 10   # pairs of user+assistant messages kept across restarts

# ---------------------------------------------------------------------------
# Phase 6 personality-tuned system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2 = (
    "You are JARVIS, a sharp and slightly dry-humoured AI assistant running "
    "entirely on the user's local PC. You have a consistent personality: "
    "efficient, direct, occasionally witty — but never sycophantic. "
    "You remember facts the user has told you and refer back to them naturally "
    "when relevant. Use the available tools whenever the user asks about system "
    "status, the time, opening or closing apps, adjusting volume or brightness, "
    "opening a website, checking open windows, locking the PC, reading or "
    "explaining code, diagnosing errors, managing what you remember, searching "
    "the web or GitHub, looking up Wikipedia, or analysing what's on the screen. "
    "For screen questions ('what's on my screen', 'what does this error say', "
    "'read the terminal') — always call analyze_screen or read_screen_text "
    "rather than guessing. "
    "Never claim you performed an action unless you actually called the matching "
    "tool first. Keep replies short and conversational — no markdown, no bullet "
    "points, no lists. Speak as if through a voice interface."
)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def remember_fact(fact: str) -> str:
    """Persist a free-text fact into ChromaDB."""
    if not fact.strip():
        return "No fact provided — nothing stored."
    doc_id = str(uuid.uuid4())
    _facts_col.add(documents=[fact.strip()], ids=[doc_id])
    return f"Got it, I'll remember that: \"{fact.strip()}\""


def recall_facts(query: str) -> str:
    """Semantic search over stored facts and return the most relevant ones."""
    if not query.strip():
        return "Please provide a query to search your memories."
    count = _facts_col.count()
    if count == 0:
        return "I don't have any stored memories yet."
    n = min(MAX_RECALL_RESULTS, count)
    results = _facts_col.query(query_texts=[query], n_results=n)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "I couldn't find anything relevant in my memory."
    joined = "; ".join(f'"{d}"' for d in docs)
    return f"Here's what I remember that's relevant: {joined}"


def forget_fact(fact_id: str) -> str:
    """Delete a specific fact by its ChromaDB document ID."""
    # The LLM usually won't know raw IDs — this is mostly for 'forget everything
    # you know about X' flows where Jarvis first does a recall then deletes.
    try:
        _facts_col.delete(ids=[fact_id])
        return f"Memory {fact_id} deleted."
    except Exception as e:
        return f"Couldn't delete that memory: {e}"


def list_facts() -> str:
    """Return all stored facts (up to 20)."""
    count = _facts_col.count()
    if count == 0:
        return "No memories stored yet."
    results = _facts_col.get(limit=20)
    docs = results.get("documents", [])
    ids  = results.get("ids", [])
    lines = [f"[{i}] {d}" for i, d in zip(ids, docs)]
    header = f"I have {count} stored fact(s)" + (" (showing first 20):" if count > 20 else ":")
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Session-history helpers  (called by Jarvis_4, not exposed as LLM tools)
# ---------------------------------------------------------------------------

def save_messages(messages: list[dict]):
    """Snapshot the current message list to ChromaDB so the next session can
    resume from where we left off.  Only user + assistant turns are stored
    (not system or tool messages — those are ephemeral).
    Overwrites whatever was stored before (one session at a time)."""
    try:
        # Clear old history
        existing = _history_col.get()
        if existing["ids"]:
            _history_col.delete(ids=existing["ids"])

        # Keep only the last MAX_HISTORY_TURNS pairs
        turns = [m for m in messages if m["role"] in ("user", "assistant")]
        turns = turns[-(MAX_HISTORY_TURNS * 2):]

        if not turns:
            return

        docs, ids, metas = [], [], []
        for idx, msg in enumerate(turns):
            docs.append(msg.get("content") or "[no content]")
            ids.append(f"hist_{idx:04d}")
            metas.append({"role": msg["role"], "index": idx})

        _history_col.add(documents=docs, ids=ids, metadatas=metas)
    except Exception as e:
        print(f"[memory] Failed to save history: {e}")


def load_messages() -> list[dict]:
    """Restore the last session's user/assistant messages in order."""
    try:
        result = _history_col.get()
        if not result["ids"]:
            return []
        pairs = sorted(
            zip(result["metadatas"], result["documents"]),
            key=lambda x: x[0]["index"],
        )
        return [{"role": m["role"], "content": doc} for m, doc in pairs]
    except Exception as e:
        print(f"[memory] Failed to load history: {e}")
        return []


def _get_all_facts_brief() -> str:
    """Return all stored facts as a compact string for injection into the
    system prompt at startup."""
    count = _facts_col.count()
    if count == 0:
        return ""
    results = _facts_col.get(limit=30)
    docs = results.get("documents", [])
    if not docs:
        return ""
    return "Known facts about the user: " + "; ".join(f'"{d}"' for d in docs) + "."


def build_initial_messages() -> list[dict]:
    """Build the messages[] list for a fresh session.

    Structure:
        [system prompt (with injected facts)]
        [last N user/assistant turns from previous session]

    The injected facts give the model immediate recall without needing to call
    recall_facts on every single turn.
    """
    facts_blurb = _get_all_facts_brief()
    system_content = SYSTEM_PROMPT_V2
    if facts_blurb:
        system_content += f"\n\n{facts_blurb}"

    messages = [{"role": "system", "content": system_content}]

    history = load_messages()
    if history:
        print(f"[memory] Restored {len(history)} messages from last session.")
        messages.extend(history)

    return messages


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

MEMORY_TOOL_IMPLEMENTATIONS = {
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
    "forget_fact": forget_fact,
    "list_facts": list_facts,
}

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Persistently store a fact the user wants JARVIS to remember "
                "across sessions — e.g. preferences, names, routines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, in plain English.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": (
                "Search JARVIS's memory for facts relevant to the given query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": (
                "Delete a specific stored fact by its ID "
                "(use list_facts first to find the ID)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": "The ID of the fact to delete.",
                    }
                },
                "required": ["fact_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_facts",
            "description": "List all facts stored in JARVIS's memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]