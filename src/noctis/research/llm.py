"""The provider-neutral LLM seam for agent research.

The research loop talks to exactly one boundary here — ``LLMClient.complete(system, tools,
messages) -> Turn`` plus a ``capabilities`` record — and never imports a provider SDK. Provider
and model are a config choice (``research.model`` as a LiteLLM ``provider/model`` string), so the
identical gated loop runs on OpenAI (default), Anthropic, or a local/OpenAI-compatible backend.

The one structural normalization LiteLLM forces lives entirely behind this boundary: our
Anthropic-native tool specs, message envelopes, and response shapes are mapped to the **OpenAI
chat format** (``messages`` / ``tools`` / ``tool_calls``) once, here. Provider-specific levers
(prompt caching, server-side web search, effort, thinking) are each gated on a capability flag,
so a provider that lacks one no-ops cleanly rather than erroring.

LiteLLM is pinned behind the optional ``[llm]`` extra, never core — missing it degrades to no
client (the runtime then falls back to the legacy loop), exactly like ``anthropic`` today.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger("noctis.research.llm")

# The Anthropic server-side web-search tool type (an Anthropic-only lever; capability-gated).
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"

# Transport bounds — the loop's time budget is only checked between rounds, so a completion
# that never returns wedges the whole research phase. LiteLLM's own default request timeout is
# 6000s (100 minutes of silence per call), which in loop infrastructure reads as a hang; every
# completion carries this explicit ceiling instead. On a stream it bounds each chunk read; the
# wall-clock cap then bounds the pathological stream that keeps trickling bytes forever.
_REQUEST_TIMEOUT_S = 600.0
_STREAM_WALL_CLOCK_CAP_S = 1800.0


@dataclass(frozen=True)
class Capabilities:
    """Which provider-specific levers the loop may pull. A ``False`` flag makes its lever a
    clean no-op (the lever is simply not sent), so the same research runs on any provider."""

    prompt_cache: bool = False  # Anthropic-explicit cache_control breakpoints
    server_web_search: bool = False  # Anthropic server-side web_search tool
    effort: bool = False  # native reasoning_effort param (OpenAI GPT-5 reasoning dial)
    thinking: bool = False  # extended/adaptive thinking dial (Anthropic)
    # Token-by-token streaming (P5): forward reasoning/content deltas to a renderer as they
    # arrive, then reassemble the same Turn. On for the well-tested hosted providers; off for
    # the local/OpenAI-compatible class, whose streaming tool-call reassembly varies — a False
    # flag makes the loop run non-streaming (the completed Event still renders a moment later).
    streaming: bool = False


@dataclass(frozen=True)
class ToolCall:
    """One function tool call the model requested, provider-neutral."""

    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    """The provider-neutral result of one completion."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "tool_use" | "end_turn" | "pause_turn" | "length" | "error"
    usage: dict  # input/output/cache_creation/cache_read tokens
    assistant_message: dict = field(default_factory=dict)  # OpenAI-format turn to append
    # The thinking text a reasoning backend surfaced (``reasoning_content``). Small local models
    # sometimes write their tool call as literal markup in here instead of the native tool-call
    # channel — the loop needs to see that to tell a misfire from a deliberate conclusion.
    reasoning: str = ""


class LLMClient(Protocol):
    """The whole surface the research and ideation loops depend on."""

    capabilities: Capabilities

    def complete(
        self,
        *,
        system,
        tools: list[dict],
        messages: list[dict],
        max_tokens: int,
        tool_choice: dict | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> Turn: ...


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic wire shapes → OpenAI chat format (done once, here)
# ─────────────────────────────────────────────────────────────────────────────
def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Map our Anthropic-format tool specs (``{name, description, input_schema}``) to OpenAI
    function tools (``{type: function, function: {name, description, parameters}}``). A server
    tool (an entry with a ``type`` but no ``input_schema`` — e.g. web_search) passes through
    untouched for providers that understand it; capability-gating decides whether it is offered
    at all."""
    out: list[dict] = []
    for spec in tools:
        if "input_schema" in spec:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec["input_schema"],
                    },
                }
            )
        else:
            out.append(spec)  # server tool (e.g. web_search) — pass through
    return out


def _turn_from_openai(resp) -> Turn:
    """Normalize a LiteLLM/OpenAI ``ModelResponse`` to a neutral :class:`Turn`."""
    choice = resp.choices[0]
    msg = choice.message
    text = getattr(msg, "content", None) or ""
    raw_calls = getattr(msg, "tool_calls", None) or []
    tool_calls: list[ToolCall] = []
    assistant_calls: list[dict] = []
    for call in raw_calls:
        raw_args = call.function.arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else (raw_args or {})
        tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))
        assistant_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": json.dumps(args)},
            }
        )
    finish = str(getattr(choice, "finish_reason", None) or "end_turn")
    # "length" passes through unmasked: a truncated completion is a stumble the loop must be
    # able to see (masking it as end_turn silently ended sessions on small-window backends).
    stop_reason = {"tool_calls": "tool_use", "stop": "end_turn"}.get(finish, finish)
    assistant_message: dict = {"role": "assistant"}
    if text:
        assistant_message["content"] = text
    if assistant_calls:
        assistant_message["tool_calls"] = assistant_calls
    return Turn(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=_usage_from_openai(getattr(resp, "usage", None)),
        assistant_message=assistant_message,
        reasoning=getattr(msg, "reasoning_content", None) or "",
    )


def _forward_deltas(chunk, on_delta: Callable[[str, str], None]) -> None:
    """Forward one streaming chunk's incremental deltas to ``on_delta`` as ``(kind, text)`` —
    ``("think", …)`` for a reasoning delta, ``("say", …)`` for a content delta (P5).

    Defensive by design: a chunk without choices, without a ``delta``, or the usage-only final
    chunk (``stream_options={"include_usage": True}`` sends one with no delta) forwards nothing.
    Read every field the same way :func:`_turn_from_openai` reads the assembled message, so the
    live stream and the final block are the same text by construction."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        on_delta("think", reasoning)
    content = getattr(delta, "content", None)
    if content:
        on_delta("say", content)


def _usage_from_openai(usage) -> dict:
    """LiteLLM surfaces cache tokens on ``usage`` and/or ``prompt_tokens_details``; read every
    field defensively so a provider that omits one contributes 0 rather than raising."""
    if usage is None:
        return dict.fromkeys(
            (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ),
            0,
        )
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read = int(
        getattr(usage, "cache_read_input_tokens", None) or getattr(details, "cached_tokens", 0) or 0
    )
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    return {
        "input_tokens": max(prompt - cache_read - cache_write, 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Provider inference, key resolution, capability derivation
# ─────────────────────────────────────────────────────────────────────────────
def provider_of(model: str) -> str:
    """The LiteLLM provider prefix. Explicit ``provider/model`` wins; otherwise infer from a
    bare model id so a legacy ``claude-opus-4-8`` config keeps working."""
    if "/" in model:
        return model.split("/", 1)[0]
    lowered = model.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "openai"  # OpenAI-compatible default (also covers local base_url deployments)


def effective_web_search(requested: bool, capabilities: Capabilities) -> bool:
    """Whether the server-side ``web_search`` tool is actually offered this session (#11).

    Web search is *optional grounding* in the protocol, so on a provider without a server-side
    search capability (OpenAI, and every ``$0`` local/self-hosted backend — ``ollama/…``,
    ``vllm/…``, an OpenAI-compatible ``base_url``) it **auto-disables**: idea-grounding degrades,
    but no gate, holdout, or journal entry depends on it. Pure so the runtime and tests agree."""
    return requested and capabilities.server_web_search


def capabilities_for(provider: str) -> Capabilities:
    """Which provider-specific levers to pull, so the same gated loop runs everywhere.

    ``anthropic`` gets explicit prompt-cache breakpoints, the server-side web_search tool, and the
    ``thinking`` dial — its reasoning depth is governed by ``thinking`` (not a separate effort
    param, which maps to a ``budget_tokens`` that Opus 4.8 rejects), so ``effort`` stays off here.
    ``openai`` has the native ``reasoning_effort`` dial (``effort=True``) and caches automatically
    (no breakpoints). Both hosted providers get ``streaming`` (P5) — LiteLLM's streaming +
    ``stream_chunk_builder`` reassembly is well-exercised on them. A local / OpenAI-compatible
    backend gets nothing provider-specific: streaming stays off there (its tool-call streaming
    varies), so a $0 backend runs the same gated loop non-streaming."""
    if provider == "anthropic":
        return Capabilities(
            prompt_cache=True, server_web_search=True, effort=False, thinking=True, streaming=True
        )
    if provider == "openai":
        return Capabilities(effort=True, streaming=True)
    return Capabilities()  # local / OpenAI-compatible: nothing provider-specific


def _key_for(provider: str, settings) -> str | None:
    """Resolve the API key per provider prefix from ``.env``: ``anthropic/*`` → ANTHROPIC_API_KEY,
    ``openai/*`` → OPENAI_API_KEY. A local/OpenAI-compatible backend needs no key."""
    if provider == "anthropic":
        return getattr(settings, "anthropic_api_key", None) or os.getenv("ANTHROPIC_API_KEY")
    if provider == "openai":
        return getattr(settings, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
    return None


def thinking_for(model: str, thinking: str = "off", *, deliberate: bool = False) -> dict | None:
    """The Anthropic ``thinking`` parameter to pin for a model, or ``None`` to send nothing.

    Three concerns collapse into one return value:

    - **The thinking trap (issue #10):** a *Sonnet* model runs **adaptive** thinking when
      ``thinking`` is omitted — spending output tokens the gated research loop doesn't need — so we
      pin an explicit ``{"type": "disabled"}`` for ``anthropic/claude-sonnet-*``. That pin holds
      under *both* dial settings of the observability watch dial: turning Sonnet's thinking back on
      is a separate, deliberate cost decision, never a side effect of an observability knob.
    - **The watch dial (P2):** Opus 4.8 runs *without* thinking when the parameter is omitted, so an
      operator who wants to watch the Anthropic fallback model reason sets ``thinking="on"``. For a
      non-Sonnet Anthropic model that returns ``{"type": "adaptive", "display": "summarized"}`` —
      adaptive is the only on-mode Opus 4.7+ accepts (``{"type": "enabled", "budget_tokens": N}``
      is a 400), and ``display: "summarized"`` is required or the thinking blocks stream with empty
      text (``display`` defaults to ``"omitted"`` on this family). LiteLLM maps the summarized
      blocks into ``reasoning_content`` → ``Turn.reasoning`` → the ``think`` event feed.
    - **The deliberate coder decision (issue #17):** ``deliberate=True`` marks a caller making the
      separate, budgeted cost decision the Sonnet pin defers to — the dedicated strategy-authoring
      ("coder") client, whose reasoning-heavy job (scenario-window and warmup arithmetic) warrants
      thinking even on Sonnet. Under ``thinking="on"`` it opts Sonnet into the same adaptive
      summarized thinking as a non-Sonnet model; under ``thinking="off"`` the cheap path still
      wins. It is never set by the observability watch dial, so a Sonnet *driver* stays pinned off.

    OpenAI has no ``thinking`` dial (it uses effort) and local backends surface reasoning for free,
    so both return ``None`` under any setting (``None`` → the kwarg is never added)."""
    if provider_of(model) != "anthropic":
        return None
    if "sonnet" in model.lower() and not (deliberate and thinking == "on"):
        return {"type": "disabled"}
    if thinking == "on":
        return {"type": "adaptive", "display": "summarized"}
    return None


def cached_system(system_text: str, *, cache: bool = True) -> str | list[dict]:
    """Wrap a static system prompt in one cached content block (Anthropic explicit caching).

    A single ``cache_control: {"type": "ephemeral"}`` breakpoint on the last (only) system block
    caches tools + system together — tools render earlier in the request prefix, so the breakpoint
    covers both. Build this once before a completion loop and reuse it by identity every round;
    never interpolate a timestamp, round counter, or other per-round variance into it.

    ``cache=False`` (a provider whose caching is automatic, e.g. OpenAI, or unsupported, e.g. a
    local model) returns the plain string, so the breakpoint is a clean no-op there. The one
    shared home for the static-system breakpoint the agent loop and the coder client both use,
    gated on the client's ``Capabilities.prompt_cache``."""
    if not cache:
        return system_text
    return [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]


# ─────────────────────────────────────────────────────────────────────────────
# The LiteLLM-backed client (SDK; the operator's #8 decision)
# ─────────────────────────────────────────────────────────────────────────────
class LiteLLMClient:
    """Realizes :class:`LLMClient` via ``litellm.completion`` (imported lazily, [llm] extra)."""

    def __init__(
        self, *, model, capabilities, api_key=None, base_url=None, thinking=None, effort=None
    ):
        self.model = model
        self.capabilities = capabilities
        self._api_key = api_key
        self._base_url = base_url
        self._thinking = thinking  # e.g. {"type": "disabled"} for Sonnet; None sends nothing
        self._effort = effort  # profile reasoning effort ("high"/"medium"); capability-gated

    def _completion_kwargs(self, *, system, tools, messages, max_tokens, tool_choice=None) -> dict:
        """Assemble the ``litellm.completion`` kwargs. Pure and side-effect-free so the thinking
        pin, effort dial, and tool/key wiring are unit-testable without a live call."""
        oai_messages = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
            # Explicit transport ceiling (see _REQUEST_TIMEOUT_S) — never LiteLLM's 100-minute
            # default. A timeout raises into the loop's existing api_error handling.
            "timeout": _REQUEST_TIMEOUT_S,
        }
        oai_tools = to_openai_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools
            # litellm (>=~1.9x) eagerly imports its MCP/proxy handler chain — which needs the
            # optional `fastapi` — whenever `tools` is present, before it ever checks whether any
            # tool is actually an MCP tool. Ours never are (plain function tools), so we opt out
            # of that gateway path via this internal flag; without it a completion-with-tools on
            # any provider dies with "No module named 'fastapi'", and the agent loop always sends
            # tools. Keeps the [llm] extra from dragging in a web framework we never execute.
            kwargs["_skip_mcp_handler"] = True
        # OpenAI-format forcing ({"type": "function", "function": {"name": …}}); LiteLLM
        # translates per provider. Callers that don't force (the research loop) send nothing.
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._thinking is not None:
            kwargs["thinking"] = self._thinking
        # Reasoning effort (#12) rides the provider's native ``reasoning_effort`` param, sent only
        # where the capability says it is native (OpenAI). It is suppressed when ``thinking`` is
        # pinned (Sonnet: thinking off is the deliberate cheap path — effort would re-enable it).
        if self.capabilities.effort and self._effort and self._thinking is None:
            kwargs["reasoning_effort"] = self._effort
        return kwargs

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        import litellm  # lazy: [llm] extra, never imported at module load

        kwargs = self._completion_kwargs(
            system=system,
            tools=tools,
            messages=messages,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
        # Stream only when a renderer asked for it AND the provider is a known-good streamer;
        # otherwise the loop runs exactly as before. The renderer never learns which path ran —
        # both return the same assembled Turn (parity is the guard against assembly bugs).
        if on_delta is not None and self.capabilities.streaming:
            return self._complete_streaming(litellm, kwargs, on_delta)
        return _turn_from_openai(litellm.completion(**kwargs))

    def _complete_streaming(self, litellm, kwargs, on_delta) -> Turn:
        """Stream the completion, teeing reasoning/content deltas to ``on_delta`` as they arrive,
        then rebuild the byte-identical :class:`Turn` the non-streaming path would return via
        LiteLLM's own ``stream_chunk_builder`` — so the research loop is unchanged and usage
        accounting is preserved (``include_usage`` puts the same totals on the final chunk, which
        the builder folds back onto ``resp.usage`` → ``_turn_from_openai`` → the budget).

        A transport/model exception raised while iterating the stream propagates unchanged, into
        the loop's existing retry / ``api_error`` handling — streaming adds a rendering path, never
        a new failure mode. Only a *rendering* hiccup (a raising ``on_delta``) is swallowed, so a
        console glitch can never abort a completion the model actually produced."""
        stream_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        chunks: list = []
        # The per-read timeout in kwargs bounds silence between chunks; this deadline bounds
        # total stream time, so a backend that trickles bytes forever still can't wedge the
        # loop. Aborting raises into the same api_error handling as any transport failure.
        deadline = time.monotonic() + _STREAM_WALL_CLOCK_CAP_S
        for chunk in litellm.completion(**stream_kwargs):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"streaming completion still running after {_STREAM_WALL_CLOCK_CAP_S:.0f}s; "
                    "aborting the stream"
                )
            chunks.append(chunk)
            try:
                _forward_deltas(chunk, on_delta)
            except Exception:  # noqa: BLE001 — a render hiccup must not abort stream assembly
                logger.debug("stream delta render failed; continuing", exc_info=True)
        resp = litellm.stream_chunk_builder(chunks, messages=kwargs["messages"])
        if resp is None:  # a stream that yielded nothing assemblable — treat as a transport miss
            raise RuntimeError("streaming completion produced no assemblable chunks")
        return _turn_from_openai(resp)


def _client_blocked(provider: str, settings) -> str | None:
    """Why no LLM client can be built for ``provider``, as a human-readable reason, or ``None``
    when one can. The single gate :func:`client_for` and :func:`client_status` share, so the loud
    startup status line can never drift from what the real builder actually does."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        return "the [llm] extra is not installed (run `uv sync --extra llm`)"
    if provider in ("anthropic", "openai") and not _key_for(provider, settings):
        env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        return f"no {env} in .env for provider {provider!r}"
    return None


@dataclass(frozen=True)
class ClientStatus:
    """Whether agent research can run, and if not, why — for a startup line so an operator never
    silently drops to the legacy loop. ``ok`` mirrors ``build_llm_client(settings) is not None``."""

    ok: bool
    model: str
    provider: str
    reason: str | None  # populated iff ``not ok``


def client_status(settings) -> ClientStatus:
    """Would :func:`build_llm_client` succeed for the configured research model, and if not, why —
    computed without side effects (no client built, no network) so a CLI can announce the research
    engine before the loop starts."""
    research = settings.research
    model = getattr(research, "model", None) or research.agent.model
    provider = provider_of(model)
    reason = _client_blocked(provider, settings)
    return ClientStatus(ok=reason is None, model=model, provider=provider, reason=reason)


def client_for(
    settings,
    model: str,
    *,
    thinking: str = "off",
    deliberate: bool = False,
    effort: str | None = None,
):
    """Build a :class:`LiteLLMClient` for ``model`` (``provider/model`` grammar), or ``None``
    when the ``[llm]`` extra (litellm) is missing or the resolved provider needs a key that
    ``.env`` doesn't carry. The one client builder every LLM consumer (agent research,
    ideation, the coder authoring client) shares, so key resolution, capabilities, and the
    Sonnet thinking pin can't drift between them. ``research.base_url`` applies here too — it
    accompanies a local/OpenAI-compatible model by config contract.

    ``deliberate`` marks the coder's deliberate, budgeted thinking decision (issue #17): it opts a
    Sonnet-class coder into adaptive thinking under ``thinking="on"``, overriding the cheap-path
    pin the observability watch dial defers to (see :func:`thinking_for`)."""
    provider = provider_of(model)
    blocked = _client_blocked(provider, settings)
    if blocked is not None:
        logger.info("no LLM client for %r: %s", model, blocked)
        return None
    return LiteLLMClient(
        model=model,
        capabilities=capabilities_for(provider),
        api_key=_key_for(provider, settings),
        base_url=getattr(settings.research, "base_url", None),
        thinking=thinking_for(model, thinking, deliberate=deliberate),
        effort=effort,
    )


def build_llm_client(settings):
    """Build the research :class:`LLMClient` from config, or ``None`` to fall back to the legacy
    loop — see :func:`client_for` for when that happens."""
    research = settings.research
    model = getattr(research, "model", None) or research.agent.model
    # Deferred import: cost.py imports provider_of from this module, so import it lazily here to
    # keep module load acyclic. The active cost_profile supplies the reasoning-effort level.
    from noctis.research.cost import resolve_budgets

    return client_for(
        settings,
        model,
        thinking=research.agent.thinking,
        effort=resolve_budgets(research).effort,
    )
