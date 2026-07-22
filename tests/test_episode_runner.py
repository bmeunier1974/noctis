"""The episode runner — one forced structured-emit call, briefing in / typed record out.

Every test drives the runner with a fake :class:`~noctis.research.llm.LLMClient` returning
scripted :class:`~noctis.research.llm.Turn`s (the same fake-client pattern as the strategy-author
and ideation tests), so the suite touches no network and needs no API key. The runner talks to
the neutral ``complete()`` seam only, so a fake that records its calls and replays a script is
the whole harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from noctis.config.settings import AgentResearchConfig
from noctis.research import Capabilities
from noctis.research.episode import (
    API_ERROR,
    MISFIRES_EXHAUSTED,
    OK,
    EmitContract,
    EpisodeResult,
    EpisodeRunner,
)
from noctis.research.ledger import SessionLedger
from noctis.research.llm import ToolCall, Turn

TOOL_NAME = "emit_decision"

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["promote", "reject", "hold"]},
        "confidence": {"type": "number"},
    },
    "required": ["action", "confidence"],
}


@dataclass(frozen=True)
class Decision:
    """The typed record one kind of episode emits."""

    action: str
    confidence: float


def parse_decision(payload: dict) -> Decision:
    """The single typed-parse both transports meet at — raises on a schema-invalid payload."""
    action = payload["action"]  # KeyError on a missing field ⇒ a schema misfire
    confidence = payload["confidence"]
    if action not in {"promote", "reject", "hold"}:
        raise ValueError(f"action {action!r} not one of promote/reject/hold")
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    return Decision(action=action, confidence=float(confidence))


CONTRACT: EmitContract[Decision] = EmitContract(
    name=TOOL_NAME,
    description="Emit the decision as one JSON object.",
    schema=SCHEMA,
    parse=parse_decision,
)


# ── Turn builders and a scripted fake client ─────────────────────────────────────────────
def emit_turn(payload: dict, *, usage: dict | None = None) -> Turn:
    """A clean forced tool call carrying ``payload`` — the compliant-backend transport."""
    call = ToolCall(id="e1", name=TOOL_NAME, arguments=payload)
    return Turn(text="", tool_calls=[call], stop_reason="tool_use", usage=usage or {})


def text_turn(
    text: str, *, stop_reason: str = "end_turn", usage: dict | None = None, reasoning: str = ""
) -> Turn:
    """A plain-text answer — the JSON-in-text fallback transport, or a misfire."""
    return Turn(
        text=text,
        tool_calls=[],
        stop_reason=stop_reason,
        usage=usage or {},
        reasoning=reasoning,
    )


class FakeClient:
    """Replays a script of :class:`Turn`s (or raises a scripted exception) through the neutral
    ``complete()`` seam, recording every call's kwargs — mirrors the agent/ideation fakes."""

    def __init__(self, script, *, model: str = "fake/model", capabilities=None):
        self._script = list(script)
        self.model = model
        self.capabilities = capabilities or Capabilities()
        self.calls: list[dict] = []

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        self.calls.append(
            {
                "system": system,
                "tools": tools,
                "messages": messages,
                "max_tokens": max_tokens,
                "tool_choice": tool_choice,
                "on_delta": on_delta,
            }
        )
        if not self._script:
            raise AssertionError("client script exhausted — the runner should have stopped")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _run(script, *, retries: int = 2, **kwargs) -> tuple[EpisodeResult[Decision], FakeClient]:
    client = FakeClient(script)
    runner = EpisodeRunner(client=client, retries=retries)
    result = runner.run(contract=CONTRACT, system="SYS", briefing="BRIEF", **kwargs)
    return result, client


# ── 1. A clean forced call is a pure briefing-in / typed-dataclass-out function ───────────
def test_clean_forced_call_returns_the_typed_value():
    result, client = _run(
        [
            emit_turn(
                {"action": "promote", "confidence": 0.8},
                usage={"input_tokens": 10, "output_tokens": 5},
            )
        ]
    )

    assert result.ok
    assert result.value == Decision("promote", 0.8)
    assert result.outcome == OK
    assert result.misfires == 0
    assert result.tokens == 15
    assert result.model == "fake/model"
    assert len(client.calls) == 1


def test_clean_call_uses_the_forced_emit_idiom_with_the_strict_schema():
    _, client = _run([emit_turn({"action": "hold", "confidence": 0.5})])

    call = client.calls[0]
    # The proven forced-tool_choice idiom (OpenAI format; LiteLLM translates per provider).
    assert call["tool_choice"] == {"type": "function", "function": {"name": TOOL_NAME}}
    # Exactly one function tool, carrying the caller's strict JSON schema.
    assert len(call["tools"]) == 1
    assert call["tools"][0]["name"] == TOOL_NAME
    assert call["tools"][0]["input_schema"] == SCHEMA
    # The system prompt and briefing are exactly what the caller handed in — nothing accreted.
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "BRIEF"}]


def test_each_episode_rebuilds_the_prompt_fresh_with_no_carried_transcript():
    client = FakeClient(
        [
            emit_turn({"action": "hold", "confidence": 0.1}),
            emit_turn({"action": "reject", "confidence": 0.2}),
        ]
    )
    runner = EpisodeRunner(client=client, retries=2)

    runner.run(contract=CONTRACT, system="SYS", briefing="first")
    runner.run(contract=CONTRACT, system="SYS", briefing="second")

    # The second episode carries ONLY its own briefing — episodes are stateless across calls.
    assert client.calls[1]["messages"] == [{"role": "user", "content": "second"}]


# ── 2. Misfire → corrective retry → success ───────────────────────────────────────────────
def test_markup_misfire_then_corrective_retry_then_success():
    markup = text_turn("<tool_call>{...}</tool_call>")  # a tool call written as literal markup
    good = emit_turn({"action": "reject", "confidence": 0.3})
    result, client = _run([markup, good])

    assert result.ok
    assert result.value == Decision("reject", 0.3)
    assert result.misfires == 1
    assert len(client.calls) == 2
    # The retry rebuilt the prompt as the briefing plus the classifier's corrective user turn.
    retry_messages = client.calls[1]["messages"]
    assert len(retry_messages) == 2
    assert retry_messages[0] == {"role": "user", "content": "BRIEF"}
    assert retry_messages[1]["role"] == "user"
    assert "native tool-call mechanism" in retry_messages[1]["content"]


def test_truncation_misfire_retries_with_the_truncation_corrective():
    cut = text_turn('{"action":"promote","confi', stop_reason="length")  # cut mid-payload
    good = emit_turn({"action": "promote", "confidence": 0.7})
    result, client = _run([cut, good])

    assert result.ok
    assert result.misfires == 1
    retry = client.calls[1]["messages"][1]["content"]
    assert "cut off by the output limit" in retry  # the truncation corrective, not a schema one


def test_completion_error_misfire_is_classified_and_retried():
    # A backend that parses tool calls itself rejects a truncated call as an exception.
    err = ValueError("Invalid tool call arguments: unexpected end of JSON input")
    good = emit_turn({"action": "promote", "confidence": 0.6})
    result, client = _run([err, good])

    assert result.ok
    assert result.misfires == 1
    assert len(client.calls) == 2


# ── 3. Retries exhausted → typed failure, not an exception cascade ────────────────────────
def test_retries_exhausted_yields_a_typed_failure():
    markup = text_turn("<tool_call>garbage</tool_call>")
    result, client = _run([markup, markup, markup], retries=2)

    assert not result.ok
    assert result.value is None
    assert result.outcome == MISFIRES_EXHAUSTED
    assert result.misfires == 3  # initial + 2 retries, all misfired
    assert len(client.calls) == 3


def test_retry_bound_is_honored_exactly():
    markup = text_turn("<tool_call>x</tool_call>")
    # retries=0 ⇒ one attempt only, no retry.
    result, client = _run([markup], retries=0)
    assert not result.ok
    assert len(client.calls) == 1
    assert result.misfires == 1


def test_prose_without_json_exhausts_as_a_typed_failure():
    junk = text_turn("I think we should promote it, quite confidently.")
    result, client = _run([junk, junk, junk], retries=2)

    assert not result.ok
    assert result.outcome == MISFIRES_EXHAUSTED
    assert result.misfires == 3
    assert len(client.calls) == 3


# ── 4. JSON-in-text fallback validates against the SAME schema as the tool transport ──────
def test_json_in_text_fallback_is_validated_by_the_same_parse():
    reply = 'Here is my decision:\n```json\n{"action":"hold","confidence":0.5}\n```'
    result, client = _run([text_turn(reply)])

    assert result.ok
    assert result.value == Decision("hold", 0.5)
    assert result.misfires == 0
    assert len(client.calls) == 1  # the text answer parsed — no retry needed


def test_bare_json_object_in_text_is_extracted_and_validated():
    result, _ = _run([text_turn('{"action":"promote","confidence":0.9}')])
    assert result.ok
    assert result.value == Decision("promote", 0.9)


def test_the_same_parse_rejects_an_invalid_payload_on_both_transports():
    # Invalid via the tool call, then invalid via JSON-in-text, then a clean emit: both invalid
    # payloads meet the SAME parse and misfire identically — one schema, two transports.
    bad_tool = emit_turn({"action": "explode", "confidence": 0.1})
    bad_text = text_turn('{"action":"nope","confidence":0.2}')
    good = emit_turn({"action": "promote", "confidence": 0.9})
    result, client = _run([bad_tool, bad_text, good], retries=3)

    assert result.ok
    assert result.value == Decision("promote", 0.9)
    assert result.misfires == 2
    assert len(client.calls) == 3
    # Each corrective carried the schema-validation reason from the shared parse.
    assert "not one of promote/reject/hold" in client.calls[1]["messages"][1]["content"]


def test_schema_invalid_text_payload_carries_the_reason_into_the_corrective():
    bad = text_turn('{"action":"explode","confidence":0.5}')
    good = emit_turn({"action": "hold", "confidence": 0.5})
    result, client = _run([bad, good])

    assert result.ok
    assert result.misfires == 1
    retry = client.calls[1]["messages"][1]["content"]
    assert "not one of promote/reject/hold" in retry


def test_malformed_json_in_text_is_a_misfire_not_a_crash():
    # Truncated/broken JSON does not parse to an object, so extraction finds none and the turn
    # misfires (here: an output-limit truncation) rather than raising.
    broken = text_turn('{"action":"promote", "confidence"', stop_reason="length")
    good = emit_turn({"action": "promote", "confidence": 0.4})
    result, client = _run([broken, good])

    assert result.ok
    assert result.misfires == 1
    assert len(client.calls) == 2


# ── 5. Episode/retry counting is exposed for budget enforcement, counted in one place ─────
def test_episode_counter_counts_each_episode_once_retries_folded_in():
    # run 1: a clean emit (1 completion). run 2: a misfire then a success (2 completions).
    client = FakeClient(
        [
            emit_turn({"action": "hold", "confidence": 0.1}),
            text_turn("<tool_call>x</tool_call>"),
            emit_turn({"action": "hold", "confidence": 0.2}),
        ]
    )
    runner = EpisodeRunner(client=client, retries=2)
    assert runner.episodes == 0

    runner.run(contract=CONTRACT, system="SYS", briefing="a")
    assert runner.episodes == 1  # one episode, one completion

    runner.run(contract=CONTRACT, system="SYS", briefing="b")
    assert runner.episodes == 2  # a retried episode is still ONE episode, not two


def test_a_failed_episode_still_increments_the_counter():
    markup = text_turn("<tool_call>x</tool_call>")
    client = FakeClient([markup, markup, markup])
    runner = EpisodeRunner(client=client, retries=2)

    result = runner.run(contract=CONTRACT, system="SYS", briefing="a")

    assert not result.ok
    assert runner.episodes == 1


# ── 6. A genuine transport failure is a typed failure, never a cascade ────────────────────
def test_non_misfire_exception_yields_a_typed_api_error():
    result, client = _run([ConnectionError("backend unreachable")])

    assert not result.ok
    assert result.outcome == API_ERROR
    assert result.value is None
    assert len(client.calls) == 1  # a genuine outage is not retried


# ── 7. Every episode output is a typed record suitable for ledger persistence ─────────────
def test_result_fields_persist_to_the_session_ledger(tmp_path):
    result, _ = _run(
        [
            text_turn("<tool_call>x</tool_call>"),  # one misfire
            emit_turn(
                {"action": "promote", "confidence": 0.8},
                usage={"input_tokens": 100, "output_tokens": 20},
            ),
        ]
    )

    # The result carries exactly what a caller writes to the ledger's episode line.
    ledger = SessionLedger(tmp_path, "s1")
    ledger.record_episode(
        stage="decide",
        model=result.model,
        outcome=result.outcome,
        tokens=result.tokens,
        misfires=result.misfires,
    )

    episode = ledger.episodes()[0]
    assert episode.stage == "decide"
    assert episode.model == "fake/model"
    assert episode.outcome == OK
    assert episode.tokens == 120  # summed across the misfire (0) and the emit (120)
    assert episode.misfires == 1


def test_result_is_a_frozen_typed_record():
    result, _ = _run([emit_turn({"action": "hold", "confidence": 0.5})])
    assert isinstance(result, EpisodeResult)
    import dataclasses

    assert dataclasses.is_dataclass(result)


# ── 8. Max-tokens threading and the config knob ───────────────────────────────────────────
def test_max_tokens_default_and_per_call_override_thread_to_the_completion():
    client = FakeClient([emit_turn({"action": "hold", "confidence": 0.5})])
    runner = EpisodeRunner(client=client, retries=2, max_tokens=1234)
    runner.run(contract=CONTRACT, system="SYS", briefing="a")
    assert client.calls[0]["max_tokens"] == 1234

    client2 = FakeClient([emit_turn({"action": "hold", "confidence": 0.5})])
    runner2 = EpisodeRunner(client=client2, retries=2, max_tokens=1234)
    runner2.run(contract=CONTRACT, system="SYS", briefing="a", max_tokens=99)
    assert client2.calls[0]["max_tokens"] == 99


def test_episode_retries_config_knob_default_and_override():
    assert AgentResearchConfig().episode_retries == 2
    assert AgentResearchConfig(episode_retries=5).episode_retries == 5
