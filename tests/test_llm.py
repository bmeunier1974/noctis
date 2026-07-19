"""The provider-neutral LLM seam: Anthropic→OpenAI shape normalization, provider inference,
per-prefix key resolution, capability derivation, and the no-litellm graceful fallback. All
pure — no network, no ``litellm`` install required (the core suite runs without the extra)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from noctis.config.settings import Settings
from noctis.research.llm import (
    Capabilities,
    ClientStatus,
    LiteLLMClient,
    ToolCall,
    _key_for,
    _turn_from_openai,
    _usage_from_openai,
    build_llm_client,
    cached_system,
    capabilities_for,
    client_for,
    client_status,
    effective_web_search,
    provider_of,
    thinking_for,
    to_openai_tools,
)

# The four operator-chosen models (#10): swapping between them is a config change, no code change.
FOUR_MODELS = [
    ("openai/gpt-5.4", "openai", "ok"),
    ("openai/gpt-5.5", "openai", "ok"),
    ("anthropic/claude-sonnet-5", "anthropic", "ak"),
    ("anthropic/claude-opus-4-8", "anthropic", "ak"),
]


def test_to_openai_tools_maps_function_specs():
    tools = [
        {
            "name": "run_backtest",
            "description": "bt",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    assert to_openai_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "run_backtest",
                "description": "bt",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_to_openai_tools_passes_server_tools_through():
    ws = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
    assert to_openai_tools([ws]) == [ws]  # a server tool has no input_schema → untouched


def test_provider_inference():
    assert provider_of("anthropic/claude-opus-4-8") == "anthropic"
    assert provider_of("claude-opus-4-8") == "anthropic"  # bare legacy id
    assert provider_of("openai/gpt-5.4") == "openai"
    assert provider_of("gpt-5.4") == "openai"
    assert provider_of("ollama/llama3") == "ollama"


def test_capabilities_by_provider():
    a = capabilities_for("anthropic")
    # Anthropic: cache + server web search + thinking + streaming; effort OFF (depth is the
    # thinking dial — a separate effort param maps to a budget_tokens Opus 4.8 rejects).
    assert a.prompt_cache and a.server_web_search and a.thinking and a.streaming and not a.effort
    # OpenAI: native reasoning_effort dial on + streaming; caches automatically (no breakpoints).
    assert capabilities_for("openai") == Capabilities(effort=True, streaming=True)
    # Local: nothing provider-specific — streaming OFF (its tool-call streaming varies).
    assert capabilities_for("ollama") == Capabilities()
    assert capabilities_for("ollama").streaming is False


def test_key_resolution_per_prefix():
    s = Settings(anthropic_api_key="ak", openai_api_key="ok")
    assert _key_for("anthropic", s) == "ak"
    assert _key_for("openai", s) == "ok"
    assert _key_for("ollama", s) is None  # local needs no key


def test_turn_normalization_from_openai_shape():
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hi",
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="run_backtest", arguments='{"x": 1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=8,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=20,
        ),
    )
    turn = _turn_from_openai(resp)
    assert turn.stop_reason == "tool_use"
    assert turn.text == "hi"
    assert turn.tool_calls == [ToolCall(id="c1", name="run_backtest", arguments={"x": 1})]
    # The assistant turn to append is OpenAI-format with tool_calls.
    assert turn.assistant_message["tool_calls"][0]["function"]["name"] == "run_backtest"
    # usage mapped to our four fields; input = prompt - read - write = 120 - 20 - 100 = 0.
    assert turn.usage == {
        "input_tokens": 0,
        "output_tokens": 8,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 20,
    }


def test_turn_surfaces_reasoning_content():
    """A reasoning backend's thinking text (``reasoning_content``) reaches ``Turn.reasoning`` —
    the loop needs it to tell a tool-call misfire (markup in the thinking channel) from a
    deliberate conclusion. A message without the attribute yields the empty string."""
    msg = SimpleNamespace(content="", tool_calls=None, reasoning_content="pondering <tool_call>…")
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)
    assert _turn_from_openai(resp).reasoning == "pondering <tool_call>…"

    bare = SimpleNamespace(content="hi", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=bare, finish_reason="stop")], usage=None
    )
    assert _turn_from_openai(resp).reasoning == ""


def test_length_finish_is_not_masked():
    """``finish_reason="length"`` reaches the loop unmasked — a truncated completion is a
    stumble the loop must be able to see, not a deliberate end_turn."""
    msg = SimpleNamespace(content="", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="length")], usage=None
    )
    assert _turn_from_openai(resp).stop_reason == "length"


def test_usage_from_openai_reads_cached_tokens_details():
    """OpenAI-style automatic caching reports cached tokens under prompt_tokens_details."""
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=900),
    )
    mapped = _usage_from_openai(usage)
    assert mapped["cache_read_input_tokens"] == 900
    assert mapped["input_tokens"] == 100


def test_build_llm_client_is_none_without_litellm(monkeypatch):
    """Missing the ``[llm]`` extra (litellm) → no client → runtime falls back to the legacy loop,
    even when a provider key is present. Simulated deterministically so the assertion holds whether
    or not litellm is actually installed (setting ``sys.modules['litellm'] = None`` makes the
    lazy ``import litellm`` inside ``build_llm_client`` raise ImportError)."""
    import sys

    monkeypatch.setitem(sys.modules, "litellm", None)
    assert build_llm_client(Settings(anthropic_api_key="ak", openai_api_key="ok")) is None


def test_client_status_agrees_with_build_llm_client(monkeypatch):
    """The startup status line must never claim the agent runs when it can't: ``client_status.ok``
    tracks ``build_llm_client(...) is not None`` exactly, because both consult one gate."""
    import sys

    # litellm present + keyless local backend → buildable → ok, no reason.
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace())
    s = Settings(research={"model": "ollama_chat/noctis-qwen3.5:9b"})
    status = client_status(s)
    assert status.ok and status.reason is None
    assert status.model == "ollama_chat/noctis-qwen3.5:9b"
    assert status.provider == "ollama_chat"
    assert (build_llm_client(s) is not None) == status.ok


def test_client_status_reports_missing_llm_extra(monkeypatch):
    """The [llm]-extra footgun the operator hit: no litellm → not ok, with a reason naming the fix,
    even with a hosted-provider key present. This is the string the CLI prints in yellow."""
    import sys

    monkeypatch.setitem(sys.modules, "litellm", None)
    status = client_status(Settings(anthropic_api_key="ak", openai_api_key="ok"))
    assert not status.ok
    assert "uv sync --extra llm" in status.reason


def test_client_status_reports_missing_provider_key(monkeypatch):
    """litellm present but a hosted provider with no key → not ok, reason names the env var."""
    import sys

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace())
    status = client_status(Settings(research={"model": "anthropic/claude-opus-4-8"}))
    assert not status.ok
    assert isinstance(status, ClientStatus)
    assert "ANTHROPIC_API_KEY" in status.reason


# ─── #10: four-model config + OpenAI default + Sonnet thinking guard ──────────
def test_default_research_model_is_openai_gpt_54():
    """Criterion 2: the shipped default resolves to the OpenAI flagship, no config needed."""
    assert Settings().research.model == "openai/gpt-5.4"


def test_four_models_resolve_provider_and_key():
    """Criterion 1: each of the four model strings picks the right provider + .env key with no
    code change — switching is purely a config edit."""
    s = Settings(anthropic_api_key="ak", openai_api_key="ok")
    for model, provider, expected_key in FOUR_MODELS:
        assert provider_of(model) == provider
        assert _key_for(provider, s) == expected_key


def test_thinking_for_disables_sonnet_only():
    """Criterion 3: the thinking trap is closed for Sonnet only; Opus and GPT are untouched."""
    assert thinking_for("anthropic/claude-sonnet-5") == {"type": "disabled"}
    assert thinking_for("anthropic/claude-opus-4-8") is None  # Opus: no thinking when omitted
    assert thinking_for("openai/gpt-5.4") is None  # GPT: no thinking dial at all
    assert thinking_for("openai/gpt-5.5") is None
    assert thinking_for("ollama/llama3") is None


def test_thinking_dial_opts_in_anthropic_nonsonnet_only():
    """P2 watch dial: ``thinking="on"`` turns on adaptive summarized thinking for the Anthropic
    fallback model only. Sonnet's cheap-path pin and every non-Anthropic no-op are unmoved under
    both settings, and the returned shape never carries ``budget_tokens`` (the ``{type: enabled,
    budget_tokens}`` shape Opus 4.8 rejects with a 400)."""
    # Opus: off by default, adaptive + summarized when the operator opts a watch session in.
    assert thinking_for("anthropic/claude-opus-4-8", "off") is None
    on = thinking_for("anthropic/claude-opus-4-8", "on")
    assert on == {"type": "adaptive", "display": "summarized"}
    assert "budget_tokens" not in on  # regression guard: the 400 shape must never return

    # Sonnet stays the deliberate cheap path (thinking off) under BOTH dial settings.
    assert thinking_for("anthropic/claude-sonnet-5", "off") == {"type": "disabled"}
    assert thinking_for("anthropic/claude-sonnet-5", "on") == {"type": "disabled"}

    # Non-Anthropic has no thinking dial → None under both settings.
    for m in ("openai/gpt-5.4", "openai/gpt-5.5", "ollama/llama3"):
        assert thinking_for(m, "off") is None
        assert thinking_for(m, "on") is None


def test_deliberate_thinking_turns_on_sonnet_without_moving_the_watch_dial():
    """#17: the coder makes a *deliberate*, budgeted thinking decision — ``deliberate=True`` opts
    a Sonnet-class coder into adaptive summarized thinking under ``thinking="on"``, overriding the
    cheap-path pin. The observability watch dial (``deliberate`` unset) is untouched: a Sonnet
    DRIVER still stays pinned off under both settings (issue #10), and the on-shape never carries
    ``budget_tokens`` (the ``{type: enabled, budget_tokens}`` shape Opus 4.8 rejects with a 400)."""
    # Deliberate + on → Sonnet thinks (adaptive summarized), no budget_tokens.
    on = thinking_for("anthropic/claude-sonnet-5", "on", deliberate=True)
    assert on == {"type": "adaptive", "display": "summarized"}
    assert "budget_tokens" not in on
    # Deliberate + off → still the cheap path (thinking off is a real coder setting).
    assert thinking_for("anthropic/claude-sonnet-5", "off", deliberate=True) == {"type": "disabled"}
    # The watch dial (deliberate unset) is UNCHANGED — a Sonnet driver stays pinned off.
    assert thinking_for("anthropic/claude-sonnet-5", "on") == {"type": "disabled"}
    assert thinking_for("anthropic/claude-sonnet-5", "off") == {"type": "disabled"}
    # A deliberate non-Sonnet Anthropic model matches the watch dial (adaptive on / None off);
    # a deliberate non-Anthropic model still has no thinking dial.
    assert thinking_for("anthropic/claude-opus-4-8", "on", deliberate=True) == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert thinking_for("openai/gpt-5.4", "on", deliberate=True) is None


def test_client_for_deliberate_threads_coder_thinking_to_completion_kwargs(monkeypatch):
    """The shared ``client_for`` builder threads ``deliberate`` to ``thinking_for``, so a Sonnet
    coder client actually carries the adaptive-thinking config onto its ``litellm.completion``
    kwargs — the driver path (no ``deliberate``) still pins the same Sonnet client off."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    s = Settings(anthropic_api_key="ak")

    coder = client_for(s, "anthropic/claude-sonnet-5", thinking="on", deliberate=True)
    assert coder._thinking == {"type": "adaptive", "display": "summarized"}
    kwargs = coder._completion_kwargs(
        system="sys", tools=[], messages=[{"role": "user", "content": "go"}], max_tokens=64
    )
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}

    # The same model built the driver way (no deliberate) stays pinned off.
    driver = client_for(s, "anthropic/claude-sonnet-5", thinking="on")
    assert driver._thinking == {"type": "disabled"}


def test_cached_system_wraps_only_when_caching_is_supported():
    """The shared static-system cache helper: a ``prompt_cache`` provider gets one cached content
    block; an auto-caching/no-caching provider gets the plain string (a clean no-op breakpoint)."""
    wrapped = cached_system("SYSTEM TEXT", cache=True)
    assert wrapped == [
        {"type": "text", "text": "SYSTEM TEXT", "cache_control": {"type": "ephemeral"}}
    ]
    assert cached_system("SYSTEM TEXT", cache=False) == "SYSTEM TEXT"


def test_completion_kwargs_pin_thinking_only_for_sonnet():
    """The Sonnet pin rides on ``litellm.completion`` kwargs; other models send no ``thinking``.
    Also checks the once-here system prepend + Anthropic→OpenAI tool mapping."""
    tools = [{"name": "bt", "description": "d", "input_schema": {"type": "object"}}]
    messages = [{"role": "user", "content": "go"}]

    sonnet = LiteLLMClient(
        model="anthropic/claude-sonnet-5",
        capabilities=capabilities_for("anthropic"),
        thinking=thinking_for("anthropic/claude-sonnet-5"),
    )
    kwargs = sonnet._completion_kwargs(system="sys", tools=tools, messages=messages, max_tokens=64)
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["max_tokens"] == 64

    opus = LiteLLMClient(
        model="anthropic/claude-opus-4-8",
        capabilities=capabilities_for("anthropic"),
        thinking=thinking_for("anthropic/claude-opus-4-8"),
    )
    assert "thinking" not in opus._completion_kwargs(
        system="sys", tools=tools, messages=messages, max_tokens=64
    )

    gpt = LiteLLMClient(
        model="openai/gpt-5.4",
        capabilities=capabilities_for("openai"),
        thinking=thinking_for("openai/gpt-5.4"),
    )
    assert "thinking" not in gpt._completion_kwargs(
        system="sys", tools=tools, messages=messages, max_tokens=64
    )


def test_completion_kwargs_opt_out_of_litellm_mcp_gateway_when_tools_present():
    """litellm eagerly imports its MCP/proxy handler (which needs the optional ``fastapi``)
    whenever ``tools`` is present — so a completion-with-tools dies with "No module named
    'fastapi'" on any provider. The client opts out via ``_skip_mcp_handler`` exactly when it
    sends tools (the agent loop always does), and never adds it on a tool-free call."""
    client = LiteLLMClient(model="ollama_chat/x", capabilities=capabilities_for("ollama"))
    msgs = [{"role": "user", "content": "go"}]
    tools = [{"name": "bt", "description": "d", "input_schema": {"type": "object"}}]

    with_tools = client._completion_kwargs(system="s", tools=tools, messages=msgs, max_tokens=8)
    assert with_tools["_skip_mcp_handler"] is True

    no_tools = client._completion_kwargs(system="s", tools=[], messages=msgs, max_tokens=8)
    assert "_skip_mcp_handler" not in no_tools  # tool-free calls stay byte-identical


def test_completion_kwargs_send_effort_only_where_native_and_no_thinking_pin():
    """#12: the profile's reasoning effort rides the native ``reasoning_effort`` param, sent only
    where the capability marks it native (OpenAI) and never when a ``thinking`` pin is active."""
    tools: list = []
    kw = dict(system="s", tools=tools, messages=[{"role": "user", "content": "go"}], max_tokens=8)

    gpt = LiteLLMClient(
        model="openai/gpt-5.4", capabilities=capabilities_for("openai"), effort="medium"
    )
    assert gpt._completion_kwargs(**kw)["reasoning_effort"] == "medium"

    # Anthropic depth is the thinking dial, not a reasoning_effort param → never sent (cap off).
    opus = LiteLLMClient(
        model="anthropic/claude-opus-4-8", capabilities=capabilities_for("anthropic"), effort="high"
    )
    assert "reasoning_effort" not in opus._completion_kwargs(**kw)

    # Even an effort-capable client suppresses effort while a thinking pin is active.
    pinned = LiteLLMClient(
        model="x",
        capabilities=Capabilities(effort=True),
        effort="high",
        thinking={"type": "disabled"},
    )
    assert "reasoning_effort" not in pinned._completion_kwargs(**kw)

    # A local backend (no effort capability) never sends it.
    local = LiteLLMClient(
        model="ollama/llama3", capabilities=capabilities_for("ollama"), effort="high"
    )
    assert "reasoning_effort" not in local._completion_kwargs(**kw)


def test_build_llm_client_wires_model_key_and_thinking(monkeypatch):
    """The build path (with ``[llm]`` present) resolves each model to the right client: correct
    provider capabilities, per-prefix key, and the Sonnet-only thinking pin."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    s = Settings(anthropic_api_key="ak", openai_api_key="ok")

    s.research.model = "anthropic/claude-sonnet-5"
    sonnet = build_llm_client(s)
    assert sonnet.model == "anthropic/claude-sonnet-5"
    assert sonnet.capabilities.thinking  # anthropic caps
    assert sonnet._thinking == {"type": "disabled"}
    assert sonnet._api_key == "ak"

    s.research.model = "anthropic/claude-opus-4-8"
    assert build_llm_client(s)._thinking is None  # Opus: no thinking by default (dial off)
    s.research.agent.thinking = "on"  # opt the watch session into provider-native reasoning
    assert build_llm_client(s)._thinking == {"type": "adaptive", "display": "summarized"}
    s.research.agent.thinking = "off"  # restore for the remaining assertions

    s.research.model = "openai/gpt-5.4"
    gpt = build_llm_client(s)
    assert gpt._thinking is None
    assert gpt._api_key == "ok"
    # OpenAI: native reasoning_effort dial + streaming.
    assert gpt.capabilities == Capabilities(effort=True, streaming=True)


# ─── #11: $0 local / free backend ────────────────────────────────────────────
def test_effective_web_search_auto_disables_without_capability():
    """Web search is offered only when requested AND the provider can serve it; a local/OpenAI
    backend (no ``server_web_search``) auto-disables it — optional grounding, not a gate."""
    assert effective_web_search(True, capabilities_for("anthropic")) is True
    assert effective_web_search(True, capabilities_for("openai")) is False  # auto-disabled
    assert effective_web_search(True, capabilities_for("ollama")) is False  # local: auto-disabled
    assert effective_web_search(False, capabilities_for("anthropic")) is False  # not requested


def test_build_llm_client_local_needs_no_key(monkeypatch):
    """A ``$0`` local backend (``ollama/…`` or an OpenAI-compatible ``base_url``) builds a client
    with **no API key** and every Anthropic-only lever off — so a full session costs nothing but
    hardware, and cache/effort/thinking cleanly no-op."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    s = Settings()  # no keys at all
    s.research.model = "ollama/llama3"
    s.research.base_url = "http://localhost:11434"
    client = build_llm_client(s)
    assert client is not None  # a keyless local backend still builds
    assert client._api_key is None
    assert client._base_url == "http://localhost:11434"
    assert client.capabilities == Capabilities()  # cache/web_search/effort/thinking all no-op
    assert client._thinking is None  # ollama is not Sonnet


# ─── P5: token-by-token streaming (parity, usage, capability + error gating) ──────────────────
def _chunk(reasoning=None, content=None, usage=None):
    """One streaming chunk. A usage-only final chunk (no reasoning/content) carries no choices,
    exactly as ``stream_options={"include_usage": True}`` sends one."""
    if reasoning is None and content is None:
        return SimpleNamespace(choices=[], usage=usage)
    delta = SimpleNamespace(reasoning_content=reasoning, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def _reassemble(chunks, messages=None):
    """A faithful stand-in for ``litellm.stream_chunk_builder``: concatenate the reasoning and
    content deltas and take the last usage — the same shape the non-streaming path returns."""
    reasoning = "".join(
        c.choices[0].delta.reasoning_content
        for c in chunks
        if c.choices and c.choices[0].delta.reasoning_content
    )
    content = "".join(
        c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
    )
    usage = next((c.usage for c in reversed(chunks) if getattr(c, "usage", None)), None)
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_litellm(chunks, *, raise_at=None):
    """A stand-in ``litellm`` module: ``completion(stream=True)`` yields the chunks (optionally
    raising partway to model a mid-stream transport drop); ``completion()`` returns the assembled
    response; ``stream_chunk_builder`` reassembles. Records the kwargs the client sent."""
    import types

    mod = types.ModuleType("litellm")
    recorded: dict = {}

    def completion(**kwargs):
        recorded["kwargs"] = kwargs
        if kwargs.get("stream"):

            def gen():
                for i, ch in enumerate(chunks):
                    if raise_at is not None and i == raise_at:
                        raise RuntimeError("connection dropped mid-stream")
                    yield ch

            return gen()
        return _reassemble(chunks)

    mod.completion = completion
    mod.stream_chunk_builder = _reassemble
    mod._recorded = recorded
    return mod


def test_forward_deltas_extracts_reasoning_then_content_and_ignores_usage_only():
    from noctis.research.llm import _forward_deltas

    seen: list = []
    _forward_deltas(_chunk(reasoning="think-bit"), lambda k, t: seen.append((k, t)))
    _forward_deltas(_chunk(content="say-bit"), lambda k, t: seen.append((k, t)))
    _forward_deltas(_chunk(usage=SimpleNamespace()), lambda k, t: seen.append((k, t)))  # no choices
    _forward_deltas(SimpleNamespace(choices=None), lambda k, t: seen.append((k, t)))  # defensive
    assert seen == [("think", "think-bit"), ("say", "say-bit")]


def test_complete_streams_deltas_in_order_and_assembles_identical_turn(monkeypatch):
    """The headline P5 guard: streaming forwards reasoning/content deltas in arrival order AND
    the assembled Turn (text, reasoning, usage, tool_calls, stop_reason) is identical to the
    non-streaming path over the same content — so the research loop never learns it streamed."""
    import sys

    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=8,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    chunks = [
        _chunk(reasoning="pon"),
        _chunk(reasoning="dering "),
        _chunk(content="Insp"),
        _chunk(content="ecting."),
        _chunk(usage=usage),  # usage-only final chunk (include_usage)
    ]
    fake = _fake_litellm(chunks)
    monkeypatch.setitem(sys.modules, "litellm", fake)

    client = LiteLLMClient(
        model="anthropic/claude-opus-4-8", capabilities=capabilities_for("anthropic")
    )
    seen: list = []
    turn = client.complete(
        system="s",
        tools=[],
        messages=[{"role": "user", "content": "go"}],
        max_tokens=64,
        on_delta=lambda k, t: seen.append((k, t)),
    )

    # Deltas teed in order: reasoning before content; the usage-only chunk teed nothing.
    assert seen == [("think", "pon"), ("think", "dering "), ("say", "Insp"), ("say", "ecting.")]
    # Streaming was actually requested, with usage-on-final so the budget still sees totals.
    assert fake._recorded["kwargs"]["stream"] is True
    assert fake._recorded["kwargs"]["stream_options"] == {"include_usage": True}
    # Parity: the streamed Turn equals the non-streaming Turn over the same content.
    baseline = _turn_from_openai(_reassemble(chunks))
    assert turn.text == baseline.text == "Inspecting."
    assert turn.reasoning == baseline.reasoning == "pondering "
    assert turn.tool_calls == baseline.tool_calls == []
    assert turn.stop_reason == baseline.stop_reason == "end_turn"
    # Usage totals equal the non-streaming baseline (the context-budget calibration guard).
    assert turn.usage == baseline.usage
    assert turn.usage["input_tokens"] == 120 and turn.usage["output_tokens"] == 8


def test_complete_runs_non_streaming_without_on_delta(monkeypatch):
    """No renderer asked to stream ⇒ the plain non-streaming path, byte-for-byte as before."""
    import sys

    fake = _fake_litellm([_chunk(content="hello")])
    monkeypatch.setitem(sys.modules, "litellm", fake)
    client = LiteLLMClient(model="openai/gpt-5.4", capabilities=capabilities_for("openai"))
    turn = client.complete(
        system="s", tools=[], messages=[{"role": "user", "content": "go"}], max_tokens=64
    )
    assert "stream" not in fake._recorded["kwargs"]  # never asked for a stream
    assert turn.text == "hello"


def test_complete_runs_non_streaming_when_capability_off(monkeypatch):
    """A $0 local backend (``streaming`` off) runs non-streaming even with a renderer attached —
    the capability gate wins, so an unproven backend never exercises stream assembly."""
    import sys

    fake = _fake_litellm([_chunk(content="hi")])
    monkeypatch.setitem(sys.modules, "litellm", fake)
    client = LiteLLMClient(model="ollama/llama3", capabilities=capabilities_for("ollama"))
    seen: list = []
    turn = client.complete(
        system="s",
        tools=[],
        messages=[{"role": "user", "content": "go"}],
        max_tokens=64,
        on_delta=lambda k, t: seen.append((k, t)),
    )
    assert "stream" not in fake._recorded["kwargs"]
    assert seen == []  # nothing streamed on a non-streaming provider
    assert turn.text == "hi"


def test_streaming_midstream_error_propagates_to_the_loop(monkeypatch):
    """A transport/model exception raised while iterating the stream propagates unchanged out of
    complete(), into the loop's existing retry / api_error handling — streaming adds no new
    failure mode. Deltas before the failure still rendered."""
    import sys

    chunks = [_chunk(reasoning="a"), _chunk(reasoning="b"), _chunk(content="c")]
    fake = _fake_litellm(chunks, raise_at=1)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    client = LiteLLMClient(
        model="anthropic/claude-opus-4-8", capabilities=capabilities_for("anthropic")
    )
    seen: list = []
    with pytest.raises(RuntimeError, match="mid-stream"):
        client.complete(
            system="s",
            tools=[],
            messages=[{"role": "user", "content": "go"}],
            max_tokens=64,
            on_delta=lambda k, t: seen.append((k, t)),
        )
    assert seen == [("think", "a")]  # the delta before the drop was rendered live


def test_completion_kwargs_carry_an_explicit_transport_timeout():
    """Every completion must carry its own transport ceiling: LiteLLM's default request
    timeout is 6000s (100 minutes of silence per call), which in the research loop — whose
    time budget is only checked between rounds — reads as a hang."""
    client = LiteLLMClient(
        model="anthropic/claude-opus-4-8", capabilities=capabilities_for("anthropic")
    )
    kwargs = client._completion_kwargs(
        system="s", tools=[], messages=[{"role": "user", "content": "go"}], max_tokens=8
    )
    assert 0 < kwargs["timeout"] < 6000
