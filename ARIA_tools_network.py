"""
ARIA_tools_network.py  —  Phase 7: Intelligence + Network Access

Tools:
    web_search(query)              — DuckDuckGo search (no API key needed)
    fetch_page(url)                — Read a URL and return a text summary
    search_github(query, kind)     — Search GitHub users, repos, or code
    search_wikipedia(query)        — Wikipedia summary for a topic

Design notes
------------
* DuckDuckGo is used via the ddgs library — no account, no rate-limit key.
* GitHub uses the public REST API (unauthenticated, 10 req/min limit).
  Add a GITHUB_TOKEN env var to raise that to 30 req/min if needed.
* Wikipedia uses the /api/rest_v1/page/summary endpoint — returns a single
  clean paragraph, ideal for voice output.
* fetch_page strips all HTML down to readable text via BeautifulSoup and
  truncates to MAX_PAGE_CHARS so it fits in the context window.
* All tools return plain English strings — same contract as every other tool.
"""

import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

MAX_PAGE_CHARS   = 6_000
MAX_SEARCH_CHARS = 3_000

GITHUB_HEADERS = {"Accept": "application/vnd.github+json"}
_token = os.environ.get("GITHUB_TOKEN")
if _token:
    GITHUB_HEADERS["Authorization"] = f"Bearer {_token}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo web search and return a digest of the top results."""
    if not query.strip():
        return "Please provide a search query."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No results found for '{query}'."
        lines = []
        for r in results:
            title = r.get("title", "")
            href  = r.get("href", "")
            body  = r.get("body", "")[:200]
            lines.append(f"• {title}\n  {href}\n  {body}")
        digest = "\n\n".join(lines)
        if len(digest) > MAX_SEARCH_CHARS:
            digest = digest[:MAX_SEARCH_CHARS] + "\n... [truncated]"
        return digest
    except Exception as e:
        return f"Web search failed: {e}"


def fetch_page(url: str) -> str:
    """Fetch a web page and return its readable text content."""
    if not url.strip():
        return "Please provide a URL."
    if not url.startswith("http"):
        url = "https://" + url
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "ARIA/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + "\n... [truncated]"
        return text or "Page fetched but no readable text found."
    except Exception as e:
        return f"Could not fetch page: {e}"


def search_github(query: str, kind: str = "repositories") -> str:
    """Search GitHub for users, repositories, or topics.

    kind: 'repositories' | 'users' | 'topics'
    """
    if not query.strip():
        return "Please provide a GitHub search query."

    kind = kind.lower().strip()
    if kind not in ("repositories", "users", "topics"):
        kind = "repositories"

    url = f"https://api.github.com/search/{kind}"
    try:
        resp = requests.get(
            url,
            params={"q": query, "per_page": 5, "sort": "stars", "order": "desc"},
            headers=GITHUB_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        total = data.get("total_count", 0)

        if not items:
            return f"No GitHub {kind} found for '{query}'."

        lines = [f"GitHub {kind} for '{query}' — {total:,} total results, top {len(items)}:\n"]
        for item in items:
            if kind == "users":
                lines.append(
                    f"  @{item['login']} — {item.get('html_url', '')}"
                )
            elif kind == "topics":
                lines.append(
                    f"  #{item['name']} — {item.get('description', 'no description')}"
                )
            else:
                stars = item.get("stargazers_count", 0)
                desc  = (item.get("description") or "no description")[:100]
                lines.append(
                    f"  {item['full_name']} ⭐{stars:,}\n"
                    f"  {desc}\n"
                    f"  {item.get('html_url', '')}"
                )
        return "\n".join(lines)
    except requests.HTTPError as e:
        if resp.status_code == 403:
            return "GitHub rate limit hit. Set a GITHUB_TOKEN env var to increase the limit."
        return f"GitHub API error: {e}"
    except Exception as e:
        return f"GitHub search failed: {e}"


def search_wikipedia(query: str) -> str:
    """Return a Wikipedia summary paragraph for the given topic."""
    if not query.strip():
        return "Please provide a topic to look up."
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "ARIA/1.0"})
        if resp.status_code == 404:
            # Try a search instead
            search_url = "https://en.wikipedia.org/w/api.php"
            sr = requests.get(search_url, params={
                "action": "opensearch", "search": query,
                "limit": 1, "format": "json"
            }, timeout=10)
            results = sr.json()
            if results[1]:
                # Re-try summary with the first result's title
                title = results[1][0]
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
                resp = requests.get(url, timeout=10, headers={"User-Agent": "ARIA/1.0"})
            else:
                return f"No Wikipedia article found for '{query}'."
        resp.raise_for_status()
        data = resp.json()
        summary = data.get("extract", "")
        title   = data.get("title", query)
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not summary:
            return f"No summary available for '{title}'."
        # Trim to first 3 sentences for voice output
        sentences = summary.split(". ")
        short = ". ".join(sentences[:3]).strip()
        if not short.endswith("."):
            short += "."
        return f"{title}: {short}\n\nFull article: {page_url}"
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

NETWORK_TOOL_IMPLEMENTATIONS = {
    "web_search":       web_search,
    "fetch_page":       fetch_page,
    "search_github":    search_github,
    "search_wikipedia": search_wikipedia,
}

NETWORK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo and return a digest of the top results. "
                "Use for current events, how-to questions, finding links, or anything "
                "that benefits from live web results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch the readable text content of a web page given its URL. "
                "Use after web_search when you need to read the full content of a result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_github",
            "description": (
                "Search GitHub for repositories, users, or topics. "
                "Use when the user asks about a GitHub profile, project, or library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. a username, repo name, or topic.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["repositories", "users", "topics"],
                        "description": (
                            "What to search for. Use 'users' for profile lookups, "
                            "'repositories' for projects, 'topics' for technology tags."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Look up a topic on Wikipedia and return a concise summary. "
                "Use for factual questions, definitions, or background on a subject."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or person to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]