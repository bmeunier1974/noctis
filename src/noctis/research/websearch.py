"""Client-side web-search grounding for local/keyless models.

Server-side web search is an Anthropic-only lever, capability-gated OFF for every ``$0``
local/self-hosted backend (see ``llm.py``: ``server_web_search``). To let *any* tool-capable
model ground its work, the research paths declare a client-side ``web_search`` function tool
and dispatch it here: a thin HTTP GET to the noctis-ollama search sidecar, which is the only
component that touches the network. Both consumers share this module — the agent research
loop (``tools.ResearchToolbox.tool_web_search``) and ideation (``propose_specs``).

Kept deliberately dependency-light (stdlib only) so importing it never drags a heavy module
into another's import graph. The sidecar lives in the noctis-ollama repo (``search-service/``);
start it with ``scripts/search.sh start``. Override its location with ``NOCTIS_SEARCH_URL``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request

logger = logging.getLogger("noctis.websearch")

TOOL_NAME = "web_search"
DEFAULT_URL = "http://127.0.0.1:11435"
MAX_RESULTS = 10  # hard cap — one grounding result stays inside the model's context window


def sidecar_url() -> str:
    """The sidecar base URL: ``NOCTIS_SEARCH_URL`` or the localhost default."""
    return os.getenv("NOCTIS_SEARCH_URL", DEFAULT_URL).rstrip("/")


def client_tool_spec(description: str) -> dict:
    """An Anthropic-style function-tool spec for the client ``web_search``.

    The description is caller-supplied because the two research paths frame grounding
    differently (thesis grounding vs. structure grounding); the name and input schema are
    shared so every provider sees the same tool contract.
    """
    return {
        "name": TOOL_NAME,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "description": f"Hits to return (default 5, max {MAX_RESULTS}).",
                },
            },
            "required": ["query"],
        },
    }


def search(query: str, max_results: int = 5) -> dict:
    """GET the sidecar and return ``{"query", "results"}`` — or ``{"error", "results": []}`` if
    it is unreachable. Never raises: grounding is optional and must not break a research loop.
    """
    base = sidecar_url()
    n = max(1, min(int(max_results or 5), MAX_RESULTS))
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "n": n})
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # localhost sidecar only
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 — degrade to a tool-visible error, never crash research
        logger.info("web_search sidecar unreachable at %s: %s", base, exc)
        return {
            "error": (
                f"web_search sidecar unreachable at {base}: {exc}. "
                "Start it with noctis-ollama's scripts/search.sh start."
            ),
            "results": [],
        }
    return {"query": query, "results": payload.get("results", [])}
