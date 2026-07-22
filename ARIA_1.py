import ollama
import psutil
import subprocess
import webbrowser
from datetime import datetime
from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc
import pygetwindow as gw

MODEL = "qwen3:14b"

SYSTEM_PROMPT = (
    "You are ARIA, a concise AI assistant running locally on the user's PC. "
    "Use the available tools whenever the user asks about system status, the "
    "time, or wants to open/close an app, adjust volume or brightness, open a "
    "website, check open windows, or lock the PC. Never claim you performed an "
    "action — closing an app, changing volume, locking the PC, etc. — unless "
    "you actually called the matching tool first; if you're not calling a "
    "tool, you have not done the thing. Keep replies short and conversational, "
    "the way you'd speak out loud — no markdown, no bullet points."
)

# ---- Tool implementations ----

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


TOOL_IMPLEMENTATIONS = {
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

# ---- Tool schemas, OpenAI-style function calling format ----

TOOLS = [
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


def run_tool_call(tool_call) -> str:
    name = tool_call["function"]["name"]
    args = tool_call["function"].get("arguments", {}) or {}
    func = TOOL_IMPLEMENTATIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    return func(**args)


def to_assistant_message(msg) -> dict:
    """Strip the Ollama response down to a clean dict before it goes back
    into history — passing the raw object back as-is can confuse the
    model's chat template on later turns and produce garbled output."""
    clean = {"role": "assistant", "content": msg.get("content", "") or ""}
    if msg.get("tool_calls"):
        clean["tool_calls"] = msg["tool_calls"]
    return clean


def chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("ARIA online. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

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

        print(f"ARIA: {msg['content']}\n")


if __name__ == "__main__":
    chat_loop()