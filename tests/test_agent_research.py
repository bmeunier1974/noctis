"""The agent research loop — a scripted fake LLMClient plays the full protocol through the
neutral ``complete()`` seam: write_strategy → symbols/data → backtests + sweep → verdict; the
exhaustion gate and the iteration budget are exercised on the way."""

from __future__ import annotations

import contextlib
import json

import pytest

from noctis.champions import ChampionRegistry
from noctis.config.settings import Settings
from noctis.memory import InMemoryMemory
from noctis.research import (
    Capabilities,
    ResearchToolbox,
    ToolCall,
    Turn,
    build_system_prompt,
    run_agent_research,
)
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.library import parse_header, strategy_source
from tests.test_research_tools import LENIENT, PROBE, FakeLake, _make_toolbox, make_bars

# The Anthropic capability set (all provider-specific levers on); the no-lever set for
# auto-caching/local providers (OpenAI, ollama, ...).
ANTHROPIC_CAPS = Capabilities(prompt_cache=True, server_web_search=True, effort=True, thinking=True)
NO_CAPS = Capabilities()


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """This module exercises the agent loop protocol, not subprocess isolation — every
    write gate and promotion-plan validation runs through the seam's in-process runner."""


# ── a scripted fake LLMClient (the neutral complete() boundary) ──────────────────────────────
class FakeLLM:
    """Plays a fixed sequence of :class:`Turn` results; records every ``complete()`` call."""

    def __init__(self, turns, capabilities=ANTHROPIC_CAPS):
        self._turns = list(turns)
        self.capabilities = capabilities
        self.calls: list[dict] = []

    def complete(self, *, system, tools, messages, max_tokens, on_delta=None):
        self.calls.append(
            {
                "system": system,
                "tools": tools,
                "messages": messages,
                "max_tokens": max_tokens,
                "on_delta": on_delta,
            }
        )
        if not self._turns:
            raise AssertionError("script exhausted — loop should have stopped already")
        return self._turns.pop(0)


def tool_turn(*calls, usage=None):
    """A Turn requesting function tool calls. Each ``calls`` item is (name, args, id)."""
    tcs = [ToolCall(id=i, name=n, arguments=a) for (n, a, i) in calls]
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tcs
        ],
    }
    return Turn(
        text="",
        tool_calls=tcs,
        stop_reason="tool_use",
        usage=usage or {},
        assistant_message=assistant,
    )


def text_turn(text="done", usage=None, stop_reason="end_turn"):
    """A Turn that ends the conversation with plain text (no tool calls)."""
    return Turn(
        text=text,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=usage or {},
        assistant_message={"role": "assistant", "content": text},
    )


def misfire_turn(channel="reasoning"):
    """A tool-call misfire: a small local backend wrote its tool call as literal markup in the
    thinking (or text) channel, where no template parses it — so the parsed turn arrives with
    empty content and zero native tool calls, exactly what ``_turn_from_openai`` yields."""
    markup = (
        "<tool_call>\n<function=preview_bars>\n<parameter=symbol>AAA</parameter>\n"
        "</function>\n</tool_call>"
    )
    return Turn(
        text=markup if channel == "text" else "",
        tool_calls=[],
        stop_reason="end_turn",
        usage={},
        assistant_message={"role": "assistant"},
        reasoning=markup if channel == "reasoning" else "",
    )


def _system_text(call) -> str:
    """The system prompt as text, whether it was sent as a plain string or as cached blocks."""
    system = call["system"]
    if isinstance(system, list):
        return "".join(b.get("text", "") for b in system)
    return system


def _count_breakpoints(call) -> int:
    """Every live cache_control breakpoint on one complete() call — system + message blocks."""
    total = 0
    system = call["system"]
    if isinstance(system, list):
        total += sum(isinstance(b, dict) and "cache_control" in b for b in system)
    for msg in call["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(isinstance(b, dict) and "cache_control" in b for b in content)
    return total


def _msg_text(msg) -> str:
    """A message's content as text (string content, or concatenated cached text parts)."""
    content = msg.get("content")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _script():
    """The full protocol: formulate → match → (premature verdict) → optimize → decide."""
    i = iter(range(100))

    def cid():
        return f"tu_{next(i)}"

    syms = ["AAA", "BBB"]
    return [
        tool_turn(("list_strategies", {}, cid()), ("get_champions", {}, cid())),
        tool_turn(("write_strategy", {"name": "probe", "source": PROBE}, cid())),
        tool_turn(
            ("list_symbols", {}, cid()),
            ("ensure_data", {"symbols": syms, "start": "2024-01-01", "end": "2024-06-30"}, cid()),
        ),
        tool_turn(("preview_bars", {"symbol": "AAA", "rows": 5}, cid())),
        # Premature verdict — must be refused by the exhaustion gate.
        tool_turn(
            ("evaluate_vs_champion", {"name": "probe", "symbols": syms, "params": {}}, cid())
        ),
        tool_turn(
            ("run_backtest", {"name": "probe", "symbols": syms, "params": {"lookback": 10}}, cid())
        ),
        tool_turn(
            ("run_backtest", {"name": "probe", "symbols": syms, "params": {"lookback": 25}}, cid())
        ),
        tool_turn(("run_sweep", {"name": "probe", "symbols": syms, "n_trials": 3}, cid())),
        tool_turn(("get_experiment_log", {"name": "probe"}, cid())),
        tool_turn(
            (
                "evaluate_vs_champion",
                {"name": "probe", "symbols": syms, "params": {"lookback": 18}},
                cid(),
            )
        ),
        text_turn("Champion promoted; session complete."),
    ]


def test_agent_loop_plays_full_protocol(tmp_path):
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(_script())

    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=20
    )

    # The session ended because the agent finished, inside every budget.
    assert summary.stopped_reason == "agent_done"
    assert summary.iterations == 11
    assert summary.promotions == 1
    assert summary.candidates == ["probe"]

    # Journal populated: 2 explicit backtests + 3 sweep trials + sweep_complete + verdicts.
    journal = toolbox.journal.records("probe")
    assert sum(r.get("event") == "trial" for r in journal) == 5
    assert any(r.get("event") == "sweep_complete" for r in journal)
    verdicts = [r for r in journal if r.get("event") == "verdict"]
    assert [v["promoted"] for v in verdicts] == [True]  # the premature one never reached it

    # The exhaustion gate refusal came back as the most recent tool message the model saw.
    last_before_call_5 = client.calls[5]["messages"][-1]
    assert last_before_call_5["role"] == "tool"
    assert "exhaustion gate" in _msg_text(last_before_call_5)

    # Champion recorded + file stamped (the write-back the human sees).
    entries = toolbox.registry.list()
    assert [e.family for e in entries] == ["probe"]
    source = strategy_source(toolbox.strategies_dir, "probe")
    assert parse_header(source).status == "champion"
    assert "lookback: int = 18" in source

    # Transport contract: system prompt + curated tools on every call (neutral tool specs).
    first = client.calls[0]
    system_text = _system_text(first)
    assert "FORMULATE" in system_text and "exhaust" in system_text.lower()
    assert {t["name"] for t in first["tools"]} >= {
        "write_strategy",
        "run_sweep",
        "evaluate_vs_champion",
        "reject_strategy",
    }


def test_iteration_budget_halts_mid_protocol(tmp_path):
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(_script())
    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=3
    )
    assert summary.stopped_reason == "max_iterations"
    assert summary.iterations == 3
    assert len(client.calls) == 3  # not one API call past the budget
    assert summary.promotions == 0


@pytest.mark.parametrize("channel", ["reasoning", "text"])
def test_text_form_tool_call_misfire_is_retried_not_fatal(tmp_path, channel):
    """A tool-call misfire (zero native tool calls because the backend wrote the call as literal
    ``<tool_call>`` markup in its thinking or text channel) must not end the session as
    ``agent_done``: the loop answers with a corrective user message and the next completion
    resumes the protocol to its real conclusion."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM([misfire_turn(channel), *_script()], capabilities=NO_CAPS)

    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=25
    )

    # The session recovered and played the whole protocol; the misfire round still burned an
    # iteration against the budget.
    assert summary.stopped_reason == "agent_done"
    assert summary.promotions == 1
    assert summary.iterations == 12

    # The retry is a user-role correction asking for a native re-issue — and the misfired
    # assistant turn was NOT appended (its markup never parsed; nothing to replay).
    retry = client.calls[1]["messages"][-1]
    assert retry["role"] == "user"
    assert "native tool" in _msg_text(retry).lower()
    assert all(m.get("content") or m.get("tool_calls") for m in client.calls[1]["messages"])


def test_persistent_misfires_end_via_iteration_budget(tmp_path):
    """Misfire retries are bounded by the ordinary iteration budget: a model that never recovers
    ends the session as ``max_iterations`` — a legitimate stop — not an infinite retry loop."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM([misfire_turn() for _ in range(5)], capabilities=NO_CAPS)

    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=5
    )

    assert summary.stopped_reason == "max_iterations"
    assert summary.iterations == 5
    assert len(client.calls) == 5  # not one API call past the budget


def test_plain_text_conclusion_is_not_retried(tmp_path):
    """A turn WITH plain text and no tool calls is the agent's deliberate conclusion — it must
    keep ending the session on the first try, not trigger a misfire retry."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM([text_turn("No viable edge this session.")], capabilities=NO_CAPS)

    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=25
    )

    assert summary.stopped_reason == "agent_done"
    assert summary.iterations == 1
    assert len(client.calls) == 1


def test_truncated_turn_is_retried_not_fatal(tmp_path):
    """A completion cut off by the output limit (``finish_reason="length"``) before it produced
    any tool call is a stumble, not a conclusion — the loop asks for a shorter turn and retries
    instead of silently ending the session."""
    toolbox = _make_toolbox(tmp_path)
    truncated = Turn(
        text="",
        tool_calls=[],
        stop_reason="length",
        usage={},
        assistant_message={"role": "assistant"},
        reasoning="I was mid-thought when",
    )
    client = FakeLLM([truncated, *_script()], capabilities=NO_CAPS)

    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=25
    )

    assert summary.stopped_reason == "agent_done"
    assert summary.promotions == 1
    assert summary.iterations == 12  # the truncated round still burned an iteration
    retry = client.calls[1]["messages"][-1]
    assert retry["role"] == "user"
    assert "cut off" in _msg_text(retry).lower()


def test_context_budget_reserves_output_headroom(tmp_path):
    """The prompt-side budget targets ``context_window - max_tokens``, not the whole window: a
    backend that bounds prompt+output together (llama-server ``num_ctx``) must always have
    ``max_tokens`` of reply room left, or the completion can only truncate."""
    toolbox = _make_toolbox(tmp_path)
    _pad_dispatch(toolbox, 2_000)
    system = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=20)
    base = (len(system) + len(json.dumps(list(toolbox.tool_specs()), default=str))) // 4
    max_tokens = 1_000
    window = base + max_tokens + 1_500  # prompt budget: base + 1_500, as in the test above

    client = FakeLLM(_budget_script(), capabilities=NO_CAPS)
    summary = run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=20,
        max_tokens=max_tokens,
        context_window=window,
    )
    assert summary.stopped_reason == "agent_done"
    assert len(client.calls) == 6
    for call in client.calls:
        assert _request_estimate(call) <= window - max_tokens


def test_rejected_tool_call_exception_is_retried_not_fatal(tmp_path):
    """A backend that parses tool calls server-side (llama-server) rejects a call whose JSON
    arguments were cut off by the output limit — the completion raises instead of returning a
    turn. That is the same stumble in exception form: the loop corrects and retries, and only
    a genuinely unrelated failure still ends the session as api_error."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(_script())
    inner = client.complete
    state = {"raised": False}

    def flaky_complete(**kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError(
                'litellm.APIConnectionError: Ollama_chatException - {"error":"llama-server '
                'returned invalid tool call arguments for \\"write_strategy\\": unexpected '
                'end of JSON input"}'
            )
        return inner(**kwargs)

    client.complete = flaky_complete
    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=25
    )

    assert summary.stopped_reason == "agent_done"
    assert summary.promotions == 1
    assert summary.iterations == 12  # the rejected completion still burned an iteration
    retry = client.calls[0]["messages"][-1]
    assert retry["role"] == "user"
    assert "rejected" in _msg_text(retry).lower()


def test_unrelated_api_error_still_ends_session(tmp_path):
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM([])
    client.complete = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("connection refused"))
    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=25
    )
    assert summary.stopped_reason == "api_error"
    assert summary.iterations == 0


def test_context_budget_calibrates_to_observed_usage():
    """The chars/4 estimate under-counts real tokenizers (worst on code-heavy history); the
    budget calibrates itself against the prompt size the backend reports, so eviction fires
    where the *backend* sees pressure, not where the heuristic guesses it."""
    from noctis.research.agent import _ContextBudget, _estimate_tokens

    # max_tokens=0 → no output reserve: the test drives the prompt-side window directly.
    # The tool-semantics sets come from the toolbox in the loop (ResearchToolbox declares
    # them beside the tools); here the budget's own mechanics are what's under test.
    budget = _ContextBudget(
        context_window=1_000,
        max_tokens=0,
        system="",
        tools=[],
        verdict_tools=ResearchToolbox.VERDICT_TOOLS,
        history_tools=ResearchToolbox.STRATEGY_HISTORY_TOOLS,
    )
    budget.record("c1", "preview_bars", {"symbol": "AAA"})
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 3_000},
        {"role": "assistant", "content": "acted on it"},
    ]
    raw = _estimate_tokens(0, messages)
    assert raw <= 1_000 * 0.9  # below the trigger on the raw heuristic…

    budget.evict_to_fit(messages)
    assert "x" * 100 in messages[1]["content"]  # …so nothing evicts before calibration

    # The backend reports the same request was actually 1.5× the estimate → recalibrated,
    # the very same history now crosses the trigger and evicts.
    budget.observe(messages, {"input_tokens": int(raw * 1.5)})
    budget.evict_to_fit(messages)
    assert "evicted to fit the context budget" in messages[1]["content"]


def test_thinking_only_empty_turn_is_retried_not_fatal(tmp_path):
    """A plan-then-stop turn — the model reasons in its thinking channel and ends without any
    tool call, text, or markup — is not a conclusion either: reasoning is invisible to the
    session, so the loop asks for an action or a conclusion and retries."""
    toolbox = _make_toolbox(tmp_path)
    silent = Turn(
        text="",
        tool_calls=[],
        stop_reason="end_turn",
        usage={},
        assistant_message={"role": "assistant"},
        reasoning="I'll explore these alternative symbols next",
    )
    client = FakeLLM([silent, *_script()], capabilities=NO_CAPS)

    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=25
    )

    assert summary.stopped_reason == "agent_done"
    assert summary.promotions == 1
    assert summary.iterations == 12
    retry = client.calls[1]["messages"][-1]
    assert retry["role"] == "user"
    assert "invisible" in _msg_text(retry).lower()


def test_max_tokens_pin_reaches_every_completion(tmp_path):
    """The research.agent.max_tokens compat knob: unset ⇒ the built-in default on every call;
    pinned ⇒ the pin on every call (a small-context local backend can bound prompt+output)."""
    from noctis.research.agent import _MAX_TOKENS

    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM([tool_turn(("list_strategies", {}, "tu_0")), text_turn()])
    run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=5)
    assert all(c["max_tokens"] == _MAX_TOKENS for c in client.calls)

    toolbox = _make_toolbox(tmp_path)
    pinned = FakeLLM([tool_turn(("list_strategies", {}, "tu_0")), text_turn()])
    run_agent_research(
        toolbox=toolbox, client=pinned, budget_minutes=60.0, max_iterations=5, max_tokens=2048
    )
    assert all(c["max_tokens"] == 2048 for c in pinned.calls)


def _request_estimate(call) -> int:
    """The same provider-neutral estimate the loop uses, applied to one recorded request."""
    from noctis.research.agent import _estimate_tokens

    base = len(_system_text(call)) + len(json.dumps(call["tools"], default=str))
    return _estimate_tokens(base, call["messages"])


def _pad_dispatch(toolbox, chars: int) -> None:
    """Inflate every tool result so a small context_window comes under real pressure while
    the journal/gate side effects stay exactly the real toolbox's."""
    orig = toolbox.dispatch
    toolbox.dispatch = lambda name, args: {**orig(name, args), "pad": "x" * chars}


def _budget_script(cid_prefix="tu"):
    """min_trials=3-compatible: three distinct-param backtests, a log read, then a verdict."""
    i = iter(range(100))

    def cid():
        return f"{cid_prefix}_{next(i)}"

    rounds = [
        tool_turn(
            (
                "run_backtest",
                {"name": "probe", "symbols": ["AAA"], "params": {"lookback": lb}},
                cid(),
            )
        )
        for lb in (8, 12, 20)
    ]
    rounds += [
        tool_turn(("get_experiment_log", {"name": "probe"}, cid())),
        tool_turn(("reject_strategy", {"name": "probe", "reason": "class-level: no edge"}, cid())),
        text_turn("session complete"),
    ]
    return rounds


def test_context_window_bounds_every_request(tmp_path):
    """P5: with research.agent.context_window set, a session under result pressure completes
    every round and no request's size estimate ever exceeds the budget — the oldest completed
    rounds' result bodies evict to pointer lines instead."""
    toolbox = _make_toolbox(tmp_path)
    _pad_dispatch(toolbox, 2_000)
    # The budget must exceed the fixed prefix (system + tools) — eviction can only shrink
    # history. Slack is sized so one full round fits but several rounds cannot.
    system = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=20)
    base = (len(system) + len(json.dumps(list(toolbox.tool_specs()), default=str))) // 4
    window = base + 1_500

    client = FakeLLM(_budget_script(), capabilities=NO_CAPS)
    summary = run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=20,
        context_window=window,
    )
    assert summary.stopped_reason == "agent_done"
    assert summary.rejections == 1
    assert len(client.calls) == 6  # every scripted round ran; nothing starved the session
    for call in client.calls:
        assert _request_estimate(call) <= window
    # Old rounds were pointered, oldest first; the pointer names the re-fetch tool.
    final_history = client.calls[-1]["messages"]
    pointers = [m for m in final_history if "evicted to fit the context budget" in _msg_text(m)]
    assert pointers, "expected at least one evicted tool result"
    assert 'get_experiment_log(name="probe")' in _msg_text(pointers[0])


def test_verdict_after_eviction_still_passes_exhaustion_gate(tmp_path):
    """P5 invariant: the exhaustion gate reads state/experiments/*.jsonl, not the context —
    a verdict issued after its trials' results were evicted from the history still counts
    every journaled trial."""
    toolbox = _make_toolbox(tmp_path)  # min_trials=3
    _pad_dispatch(toolbox, 2_000)
    system = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=20)
    base = (len(system) + len(json.dumps(list(toolbox.tool_specs()), default=str))) // 4

    client = FakeLLM(_budget_script(), capabilities=NO_CAPS)
    summary = run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=20,
        context_window=base + 1_500,
    )
    # The verdict round's own request already carried pointered trials — eviction provably
    # happened before the gate was asked.
    verdict_request = client.calls[4]["messages"]
    assert any("evicted to fit the context budget" in _msg_text(m) for m in verdict_request)
    assert summary.rejections == 1  # the gate saw 3 distinct journaled param sets and allowed it
    journal = toolbox.journal.records("probe")
    assert sum(r.get("event") == "trial" for r in journal) == 3  # disk ground truth untouched
    assert any(r.get("event") == "verdict" for r in journal)


def test_verdict_boundary_compaction_collapses_decided_strategy(tmp_path):
    """P5: a successful verdict collapses that strategy's optimization history to pointer
    lines even with no size pressure; the verdict line and unrelated results survive."""
    toolbox = _make_toolbox(tmp_path)  # min_trials=3
    i = iter(range(100))

    def cid():
        return f"tu_{next(i)}"

    script = [
        tool_turn(("preview_bars", {"symbol": "AAA", "rows": 5}, cid())),  # not probe history
        *[
            tool_turn(
                (
                    "run_backtest",
                    {"name": "probe", "symbols": ["AAA"], "params": {"lookback": lb}},
                    cid(),
                )
            )
            for lb in (8, 12, 20)
        ],
        tool_turn(("reject_strategy", {"name": "probe", "reason": "class-level: no edge"}, cid())),
        tool_turn(("list_strategies", {}, cid())),  # one more round to observe the compaction
        text_turn(),
    ]
    client = FakeLLM(script, capabilities=NO_CAPS)
    summary = run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=20,
        context_window=200_000,  # roomy: compaction is verdict-triggered, not size-triggered
    )
    assert summary.rejections == 1

    history = client.calls[-1]["messages"]  # the request after the verdict round
    texts = [_msg_text(m) for m in history if m.get("role") == "tool"]
    compacted = [t for t in texts if 'superseded by the reject_strategy verdict on "probe"' in t]
    assert len(compacted) == 3  # all three probe backtest bodies collapsed
    assert 'get_experiment_log(name="probe")' in compacted[0]  # the pointer names the re-fetch
    # The verdict result itself and the unrelated preview_bars result survive in full.
    assert any('"strategy":"probe"' in t and '"rejected":true' in t for t in texts) or any(
        "rejected" in t and "superseded" not in t for t in texts
    )
    assert any('"symbol":"AAA"' in t for t in texts)  # preview_bars untouched


def test_no_context_window_leaves_history_byte_identical(tmp_path):
    """P5 regression: with the knob unset every mechanism is inert — every tool message is
    exactly today's serialization (flat 20K cap) and no pointer line ever appears."""
    from noctis.research.agent import _tool_result_content

    toolbox = _make_toolbox(tmp_path)
    raw_results: list[dict] = []
    orig = toolbox.dispatch

    def spy(name, args):
        result = orig(name, args)
        raw_results.append(result)
        return result

    toolbox.dispatch = spy
    client = FakeLLM(_budget_script(), capabilities=NO_CAPS)
    summary = run_agent_research(
        toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=20
    )
    assert summary.rejections == 1
    history = client.calls[-1]["messages"]
    tool_contents = [m["content"] for m in history if m.get("role") == "tool"]
    assert tool_contents == [_tool_result_content(r) for r in raw_results]
    flat = json.dumps(history)
    assert "evicted to fit the context budget" not in flat
    assert "superseded by the" not in flat


def test_usage_rollup_emitted_once(tmp_path, caplog):
    """Exactly one per-session rollup with rounds, token totals, and a cache-hit ratio."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(
        [
            tool_turn(
                ("list_strategies", {}, "tu_0"),
                usage={
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 2000,
                    "cache_read_input_tokens": 0,
                },
            ),
            text_turn(
                usage={
                    "input_tokens": 50,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 2000,
                }
            ),
        ]
    )
    with caplog.at_level("INFO", logger="noctis.research.agent"):
        run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=5)

    rollups = [r.getMessage() for r in caplog.records if "agent research usage:" in r.getMessage()]
    assert len(rollups) == 1
    line = rollups[0]
    assert "2 rounds" in line
    assert "input=150" in line and "output=30" in line
    assert "cache_write=2000" in line and "cache_read=2000" in line
    # cache_read / (cache_read + input + cache_write) = 2000 / (2000 + 150 + 2000) = 0.4819…
    assert "cache_hit_ratio=0.482" in line


def test_usage_rollup_zero_cache_on_fresh_session(tmp_path, caplog):
    """A fresh session with no usage on the fake client: no raise, cache-read is 0."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM([text_turn()])
    with caplog.at_level("INFO", logger="noctis.research.agent"):
        summary = run_agent_research(
            toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=5
        )
    assert summary.stopped_reason == "agent_done"
    rollups = [r.getMessage() for r in caplog.records if "agent research usage:" in r.getMessage()]
    assert len(rollups) == 1
    assert "cache_read=0" in rollups[0]
    assert "cache_hit_ratio=0.000" in rollups[0]


def test_system_prompt_carries_cache_breakpoint(tmp_path):
    """With prompt_cache capability, the system prefix is one cached block reused by identity."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(
        [tool_turn(("list_strategies", {}, "tu_0")), text_turn()], capabilities=ANTHROPIC_CAPS
    )
    run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=5)

    system = client.calls[0]["system"]
    assert isinstance(system, list) and len(system) == 1
    block = system[0]
    assert block["type"] == "text" and "FORMULATE" in block["text"]
    assert block["cache_control"] == {"type": "ephemeral"}
    # Built once before the loop and reused by identity every round.
    assert client.calls[1]["system"] is client.calls[0]["system"]
    # ≤2 live breakpoints on every request (static system + one moving).
    assert all(_count_breakpoints(c) <= 2 for c in client.calls)


def test_no_prompt_cache_capability_sends_plain_string(tmp_path):
    """A provider whose caching is automatic (OpenAI) or unsupported (local): no breakpoints."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(
        [tool_turn(("list_strategies", {}, "tu_0")), text_turn()], capabilities=NO_CAPS
    )
    run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=5)
    assert isinstance(client.calls[0]["system"], str)
    assert "FORMULATE" in client.calls[0]["system"]
    assert all(_count_breakpoints(c) == 0 for c in client.calls)


def test_moving_breakpoint_on_tool_result_history(tmp_path):
    """A moving breakpoint annotates the last (tool-result) message each round; ≤2 live; no
    stale breakpoint pollutes older turns."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(
        [
            tool_turn(("list_strategies", {}, "tu_0"), ("get_champions", {}, "tu_1")),
            tool_turn(("list_symbols", {}, "tu_2")),
            text_turn(),
        ],
        capabilities=ANTHROPIC_CAPS,
    )
    run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, max_iterations=10)

    # Round-2 request: the last message is a tool-result message; its content carries the
    # moving breakpoint. Two tool results were produced, so only the last is annotated.
    r1 = client.calls[1]
    last = r1["messages"][-1]
    assert last["role"] == "tool"
    assert isinstance(last["content"], list) and last["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert _count_breakpoints(r1) == 2  # static system + one moving

    # ≤2 live breakpoints on EVERY request.
    assert all(_count_breakpoints(c) <= 2 for c in client.calls)

    # No stale accumulation: in a later request exactly one message is annotated.
    r2 = client.calls[2]
    annotated = sum(
        isinstance(m.get("content"), list) and any("cache_control" in b for b in m["content"])
        for m in r2["messages"]
    )
    assert annotated == 1


def test_capabilities_gate_web_search(tmp_path):
    """`web_search` is one tool name with two implementations, picked by capability.

    Anthropic serves it server-side (a tool entry with a ``type``); a $0 local backend gets the
    client sidecar tool (an entry with an ``input_schema``). Never both, and the operator
    ``web_search`` flag gates availability entirely.
    """
    # Anthropic: server-side tool offered; the client tool of the same name is withdrawn.
    toolbox = _make_toolbox(tmp_path)
    with_caps = FakeLLM([text_turn()], capabilities=ANTHROPIC_CAPS)
    run_agent_research(toolbox=toolbox, client=with_caps, budget_minutes=60.0, web_search=True)
    a_tools = with_caps.calls[0]["tools"]
    assert any(t.get("name") == "web_search" and "type" in t for t in a_tools)
    assert not any(t.get("name") == "web_search" and "input_schema" in t for t in a_tools)

    # $0 local backend: no server-side capability, so the client sidecar tool stands in.
    toolbox2 = _make_toolbox(tmp_path / "b")
    no_caps = FakeLLM([text_turn()], capabilities=NO_CAPS)
    run_agent_research(toolbox=toolbox2, client=no_caps, budget_minutes=60.0, web_search=True)
    l_tools = no_caps.calls[0]["tools"]
    assert any(t.get("name") == "web_search" and "input_schema" in t for t in l_tools)
    assert not any(t.get("name") == "web_search" and "type" in t for t in l_tools)

    # Operator flag off: neither implementation is offered.
    toolbox3 = _make_toolbox(tmp_path / "c")
    off = FakeLLM([text_turn()], capabilities=NO_CAPS)
    run_agent_research(toolbox=toolbox3, client=off, budget_minutes=60.0, web_search=False)
    assert not any(t.get("name") == "web_search" for t in off.calls[0]["tools"])


def test_local_backend_runs_full_protocol_with_gates_intact(tmp_path):
    """#11: the same gated loop runs on a ``$0`` local backend (NO_CAPS — no cache/effort/thinking
    and no *server-side* web_search). With web_search requested, the client sidecar tool stands in
    (announced loudly, not silently); the exhaustion gate, journaling, and promotion fire exactly
    as on the paid Anthropic path."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(_script(), capabilities=NO_CAPS)  # local provider: every lever off
    events: list = []

    summary = run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=20,
        web_search=True,
        on_event=events.append,
    )

    # The substitution was announced (not silent), and the CLIENT web_search tool reached the
    # model — a local model can still ground via the sidecar. The notice is a legacy plain string.
    str_events = [e for e in events if isinstance(e, str)]
    assert any("local web_search sidecar" in e for e in str_events)
    assert any(
        t.get("name") == "web_search" and "input_schema" in t for t in client.calls[0]["tools"]
    )
    # But no server-side web_search leaked in (the client tool carries no `type`).
    assert all(
        not any(t.get("name") == "web_search" and "type" in t for t in call["tools"])
        for call in client.calls
    )
    # No Anthropic-only levers leaked onto the wire: system stays a plain string, no cache blocks.
    assert isinstance(client.calls[0]["system"], str)
    assert _count_breakpoints(client.calls[-1]) == 0

    # Discipline is model-independent: same verdict, journal, and promotion as the paid path.
    assert summary.stopped_reason == "agent_done"
    assert summary.promotions == 1
    journal = toolbox.journal.records("probe")
    assert sum(r.get("event") == "trial" for r in journal) == 5
    assert any(r.get("event") == "sweep_complete" for r in journal)
    assert [r["promoted"] for r in journal if r.get("event") == "verdict"] == [True]
    # The premature verdict was still refused by the exhaustion gate under the local backend.
    assert "exhaustion gate" in _msg_text(client.calls[5]["messages"][-1])


class _HeartbeatSink:
    """A callable event sink that also exposes an ``activity()`` context manager, like the Console
    — so the loop's P6 heartbeat wrapping (model calls + tool dispatch) is asserted without a
    terminal. ``verbose=1`` is -v: no token stream, so model calls get a heartbeat too."""

    def __init__(self):
        self.events: list = []
        self.labels: list[str] = []
        self.verbose = 1

    def __call__(self, ev):
        self.events.append(ev)

    @contextlib.contextmanager
    def activity(self, label):
        self.labels.append(label)
        yield


def test_heartbeat_wraps_model_calls_and_tool_dispatch(tmp_path):
    """P6: at -v the loop brackets every blocking model call and tool sweep in
    ``on_event.activity()``, so the console can render a live spinner instead of going silent for
    minutes. A plain-callable sink has no ``activity()`` and is unaffected (the other tests, which
    pass ``events.append``, assert the byte-identical path implicitly)."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(_script())
    sink = _HeartbeatSink()

    run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=20,
        on_event=sink,
    )

    # Every blocking model call was heartbeated (verbose=1 ⇒ no token stream to signal life);
    # FakeLLM names no model, so the label is the bare "thinking" — one per complete() call.
    assert sink.labels.count("thinking") == len(client.calls)
    # Tool dispatch heartbeats carry a compact "<tool> <salient-arg>" label.
    assert "write_strategy probe" in sink.labels  # name arg
    assert "run_sweep probe" in sink.labels  # the long sweep — the whole reason for the heartbeat
    assert "preview_bars AAA" in sink.labels  # symbol arg
    assert "list_strategies" in sink.labels  # no salient arg ⇒ bare tool name


def test_prefix_trim_caps_advisory_memory_only(tmp_path):
    """#12 economy lever: ``prefix_trim`` caps the advisory memory tail (recent findings) in the
    system prefix to the last 5, shrinking the cache write/reads — but only *advisory* context;
    the protocol, contract, and gate language stay intact."""
    toolbox = _make_toolbox(tmp_path)
    for i in range(8):
        toolbox.memory.append_finding(f"finding-{i}")

    full = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=40, prefix_trim=False)
    trimmed = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=40, prefix_trim=True)

    assert "finding-0" in full  # untrimmed keeps the whole recent window
    assert "finding-2" not in trimmed  # trimmed drops all but the last 5…
    assert "finding-3" in trimmed and "finding-7" in trimmed  # …keeping the freshest
    # Trimming is cost-only: the gate/contract language survives in both.
    assert "FORMULATE" in trimmed and "exhaust" in trimmed.lower()


def test_no_client_is_a_noop(tmp_path):
    toolbox = _make_toolbox(tmp_path)
    summary = run_agent_research(toolbox=toolbox, client=None, budget_minutes=60.0)
    assert summary.stopped_reason == "no_client"
    assert summary.iterations == 0


def test_directive_reaches_prompt_and_kickoff(tmp_path):
    from noctis.config.settings import ResearchConfig
    from noctis.research import Mandate

    # Selector default is off (repo config.yaml may set one — don't assert on Settings()).
    assert ResearchConfig().mandate is None

    toolbox = _make_toolbox(tmp_path)
    body = "Find a strategy on very volatile stocks; high risk appetite"
    mandate = Mandate(
        text=body,
        source="cli",
        summary="high-risk volatile-names mandate",
        references=[],
        config_overrides={},
    )
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10, mandate=mandate)
    assert "OPERATOR MANDATE" in prompt
    assert "very volatile stocks" in prompt
    assert "holdout_symbols" in prompt  # the discovery + nomination guidance
    # Without a mandate the block is absent.
    plain = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert "OPERATOR MANDATE" not in plain

    # Through the loop: the system prompt carries the full body; the kickoff carries only the
    # one-line SUMMARY, never the full body (a multi-KB mandate must not be embedded twice).
    # NO_CAPS keeps the kickoff a plain string (no cache wrapping) so we can read it directly.
    client = FakeLLM([text_turn()], capabilities=NO_CAPS)
    run_agent_research(toolbox=toolbox, client=client, budget_minutes=60.0, mandate=mandate)
    call = client.calls[0]
    system_text = _system_text(call)
    assert "OPERATOR MANDATE" in system_text
    assert "very volatile stocks" in system_text
    kickoff = call["messages"][0]["content"]
    assert "high-risk volatile-names mandate" in kickoff  # the summary
    assert "very volatile stocks" not in kickoff  # NOT the full body


def test_system_prompt_carries_market_reality(tmp_path):
    toolbox = _make_toolbox(tmp_path)
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert "MARKET REALITY" in prompt
    assert "round_trip_cost_bp" in prompt
    assert '"AAA"' in prompt  # per-symbol digest (buy-hold benchmark + bar-move stats)
    assert "cost arithmetic" in prompt  # the FORMULATE discipline references it


def test_market_reality_digest_is_deterministic(tmp_path):
    """The market-reality digest serializes with sorted keys, byte-identical across insertion
    orders — cheap insurance for a future cross-session cache."""
    toolbox = _make_toolbox(tmp_path)
    toolbox.market_context = lambda: {"zeta": 1, "alpha": 2, "middle": 3}
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert '{"alpha": 2, "middle": 3, "zeta": 1}' in prompt


def test_tool_result_serialization_is_compact():
    """Tool results serialize with compact separators — no incidental whitespace re-sent."""
    from noctis.research.agent import _tool_result_content

    text = _tool_result_content({"b": 1, "a": {"c": 2, "d": [1, 2]}})
    assert ", " not in text  # no space after commas
    assert ": " not in text  # no space after colons
    assert text == '{"b":1,"a":{"c":2,"d":[1,2]}}'


def test_system_prompt_survives_market_context_failure(tmp_path):
    toolbox = _make_toolbox(tmp_path)

    def boom():
        raise RuntimeError("lake offline")

    toolbox.market_context = boom
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert "MARKET REALITY" in prompt
    assert "unavailable this session" in prompt


def test_directive_block_carries_pushback_rule(tmp_path):
    from noctis.research import Mandate

    toolbox = _make_toolbox(tmp_path)
    mandate = Mandate(
        text="very volatile stocks",
        source="cli",
        summary="very volatile stocks",
        references=[],
        config_overrides={},
    )
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10, mandate=mandate)
    # The steer-don't-suspend guardrail (the honesty contract) survives the rename verbatim.
    assert "search prior, not a suspension of arithmetic" in prompt
    assert "never overrides the gates, the protocol, or the honesty rules" in prompt


def test_system_prompt_carries_state_and_memory(tmp_path):
    strategies_dir = tmp_path / "strategies"
    settings = Settings(
        strategies_dir=str(strategies_dir),
        state_dir=str(tmp_path / "state"),
        universe=["AAA", "BBB"],
    )
    memory = InMemoryMemory()
    memory.append_finding("PROMOTED something once")
    memory.record_rejected("dead_family", {"x": 1}, reason="no edge")
    toolbox = ResearchToolbox(
        settings=settings,
        lake=FakeLake({"AAA": make_bars(seed=1)}),
        registry=ChampionRegistry(tmp_path / "c.json", capacity=3),
        families=FamilyRegistry(),
        memory=memory,
        rules=LENIENT,
    )
    from noctis.strategies.library import write_strategy

    write_strategy(toolbox.strategies_dir, "probe", PROBE, toolbox.families)
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert "probe" in prompt  # library index
    assert "PROMOTED something once" in prompt  # findings digest
    assert "dead_family" in prompt  # rejected ideas digest
    assert "FORMULATE" in prompt and "DECIDE" in prompt


def test_prompt_embeds_consolidated_memory_views(tmp_path):
    """P3 stage 1: the memory tail the prompt embeds is one line per lesson class — repeated
    events merge with a ×N marker, old class lessons outlive the raw tail, and dead ends
    merge per family without ever dropping a class."""
    toolbox = _make_toolbox(tmp_path)
    toolbox.memory.append_finding("REJECTED strategy old_class - the old lesson")
    for i in range(6):
        toolbox.memory.append_finding(f"REJECTED strategy hot_class - attempt {i}")
    for i in range(3):
        toolbox.memory.record_rejected("hot_class", {"lookback": i}, reason="cost-bound")

    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    trimmed = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10, prefix_trim=True)
    for text in (prompt, trimmed):
        # 7 raw events → 2 class lines; a raw economy tail of 5 would have lost old_class.
        assert "old_class - the old lesson" in text
        # Newest phrasing kept, merged repeats marked ×6 (json-escaped as × in the prompt).
        assert "attempt 5 (\\u00d76)" in text
        assert text.count("hot_class - attempt") == 1
        # Dead ends merged per family: latest params + times, the class itself never dropped.
        assert '"times": 3' in text and '"lookback": 2' in text


def test_prompt_embeds_distilled_block_plus_last_three_raw(tmp_path):
    """P3 stage 2: once a distilled block exists, sessions embed it + the 3 newest raw
    entries instead of the consolidated tail."""
    toolbox = _make_toolbox(tmp_path)
    for i in range(8):
        toolbox.memory.append_finding(f"standalone lesson number {i}")
    toolbox.memory.set_distilled(["- distilled: minute-bar mean reversion is cost-bound"])

    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert "distilled: minute-bar mean reversion is cost-bound" in prompt
    for i in (5, 6, 7):  # the 3 newest raw entries ride along
        assert f"standalone lesson number {i}" in prompt
    for i in (0, 4):  # older raw entries are represented by the distilled block only
        assert f"standalone lesson number {i}" not in prompt


def test_system_prompt_stubs_rejected_strategies(tmp_path):
    """P1 (context plan): rejected library entries collapse to {name, status} stubs in the
    prompt — the class-level lesson already arrives via memory's rejected_ideas and the
    exhausted_classes digest — while the list_strategies TOOL still returns everything in
    full and the file on disk is untouched."""
    from noctis.strategies.library import set_header, write_strategy

    toolbox = _make_toolbox(tmp_path)
    corpse = PROBE.replace('name = "probe"', 'name = "corpse"').replace(
        "Toy probe: long above its own moving average.", "Corpse-only thesis marker."
    )
    write_strategy(toolbox.strategies_dir, "corpse", corpse, toolbox.families)
    set_header(toolbox.strategies_dir, "corpse", families=toolbox.families, status="rejected")

    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert '{"name": "corpse", "status": "rejected"}' in prompt  # the stub, nothing more
    assert "Corpse-only thesis marker." not in prompt
    # The live strategy keeps its full entry (thesis + params + param_space).
    assert "Toy probe" in prompt and '"param_space"' in prompt
    # The tool surface is unchanged: rejected entries stay fully described on demand.
    listed = {e["name"]: e for e in toolbox.dispatch("list_strategies", {})["strategies"]}
    assert listed["corpse"]["thesis"] == "Corpse-only thesis marker."
    assert listed["corpse"]["param_space"]


def test_champion_board_embed_carries_sharpe_and_mandate_source(tmp_path):
    """The champion board shows the neutral Sharpe basis + provenance for the `auto` rule."""
    from tests.test_champions import make_scorecard
    from tests.test_research_tools import LENIENT, _make_toolbox

    toolbox = _make_toolbox(tmp_path)
    toolbox.registry.consider(
        make_scorecard("sma_crossover", test_metric=1.5, train_metric=1.6),
        LENIENT,
        mandate_source="profile:aggressive",
    )
    prompt = build_system_prompt(toolbox, budget_minutes=60.0, max_iterations=10)
    assert '"sharpe"' in prompt  # the common cross-profile yardstick
    assert '"mandate_source": "profile:aggressive"' in prompt  # provenance beside it


# ── P1: the event seam — reasoning / narration / usage teed inline ───────────────────────────
def _reasoning_turn(reasoning, narration, usage):
    """A tool turn carrying reasoning + narration + usage, exactly as a reasoning backend
    surfaces them on a :class:`Turn` today (all three captured, none surfaced before P1)."""
    return Turn(
        text=narration,
        tool_calls=[ToolCall(id="tu_0", name="list_strategies", arguments={})],
        stop_reason="tool_use",
        usage=usage,
        assistant_message={
            "role": "assistant",
            "content": narration,
            "tool_calls": [
                {
                    "id": "tu_0",
                    "type": "function",
                    "function": {"name": "list_strategies", "arguments": "{}"},
                }
            ],
        },
        reasoning=reasoning,
    )


def test_reasoning_narration_and_usage_are_teed_as_events(tmp_path):
    """P1 headline: think (reasoning), say (narration), and per-round usage — all three live on
    the Turn today and were captured-then-dropped — are surfaced as typed Events, in that order,
    at level 2. The invariant: none of them ever reach memory or disk (they must not be able to
    reach a decision)."""
    import pathlib

    from noctis.research.agent import _usage_line

    toolbox = _make_toolbox(tmp_path)
    reasoning = "The mean-reversion thesis needs its cost check before I write on_bar."
    narration = "Inspecting the library and champions first."
    usage = {
        "input_tokens": 120,
        "output_tokens": 34,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    client = FakeLLM(
        [_reasoning_turn(reasoning, narration, usage), text_turn("done")], capabilities=NO_CAPS
    )
    events: list = []
    run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=5,
        on_event=events.append,
    )

    typed = [e for e in events if not isinstance(e, str)]
    # Round-1 order: think → usage (both at the top of the round) → say (narration beside the
    # action) → tool (the dispatched call).
    assert [e.kind for e in typed][:4] == ["think", "usage", "say", "tool"]
    think = next(e for e in typed if e.kind == "think")
    assert think.text == reasoning and think.level == 2
    say = next(e for e in typed if e.kind == "say" and e.text == narration)
    assert say.level == 2  # inter-round narration is the firehose, not the -v feed
    usage_ev = next(e for e in typed if e.kind == "usage")
    assert usage_ev.level == 2 and usage_ev.meta == usage and usage_ev.text == _usage_line(usage)
    # The agent's final conclusion is a level-1 say so it still shows at -v like today.
    conclusion = next(e for e in typed if e.kind == "say" and e.text == "done")
    assert conclusion.level == 1

    # Invariant (overview principle 3): nothing surfaced for the operator enters memory…
    mem_blob = " ".join(toolbox.memory.findings()) + " ".join(
        str(r) for r in toolbox.memory.rejected_ideas()
    )
    assert reasoning not in mem_blob and narration not in mem_blob
    # …nor any file under the state tree (journals, champions, distilled memory).
    for path in pathlib.Path(tmp_path).rglob("*"):
        if not path.is_file():
            continue
        try:
            blob = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        assert reasoning not in blob, f"reasoning leaked into {path}"
        assert narration not in blob, f"narration leaked into {path}"


def test_tool_event_surfaces_gate_facing_numbers(tmp_path):
    """P1: the per-tool-call Event carries the gate-facing numbers a promotion/rejection turns
    on — overfit gap, temporal + symbol holdouts, test activity — in both its text and its
    structured meta, so the -v feed alone tells the story. Neutral numbers only."""
    toolbox = _make_toolbox(tmp_path)
    client = FakeLLM(
        [
            tool_turn(
                (
                    "run_backtest",
                    {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 10}},
                    "tu_0",
                )
            ),
            tool_turn(("run_backtest", {"name": "ghost", "symbols": ["AAA"]}, "tu_1")),
            text_turn("done"),
        ],
        capabilities=NO_CAPS,
    )
    events: list = []
    run_agent_research(
        toolbox=toolbox,
        client=client,
        budget_minutes=60.0,
        max_iterations=5,
        on_event=events.append,
    )
    typed = [e for e in events if not isinstance(e, str)]

    bt = next(
        e for e in typed if e.kind == "tool" and e.meta.get("ok") and "run_backtest" in e.text
    )
    for key in ("gap", "holdout_metric", "test_activity"):
        assert key in bt.meta, f"{key} missing from tool meta"
        assert f"{key}=" in bt.text, f"{key} missing from tool line"

    # A failed dispatch (unknown strategy) is an ok=False event the renderer colors red.
    err = next(e for e in typed if e.kind == "tool" and "ghost" in e.text)
    assert err.meta["ok"] is False and "ERROR" in err.text
