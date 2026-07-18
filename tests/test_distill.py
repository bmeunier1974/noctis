"""Stage-2 memory distillation (context plan P3): knob off by default, session counting,
the periodic trigger at close, and graceful degradation to the stage-1 view."""

from __future__ import annotations

from noctis.config.settings import ResearchConfig, Settings
from noctis.memory import InMemoryMemory
from noctis.research.distill import (
    bump_research_session,
    distill_findings,
    maybe_distill,
)
from noctis.research.llm import Turn


class FakeDistillClient:
    """Returns one scripted distillation; records every call."""

    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def complete(self, *, system, tools, messages, max_tokens) -> Turn:
        self.calls.append({"system": system, "tools": tools, "messages": messages})
        return Turn(text=self.text, tool_calls=[], stop_reason="end_turn", usage={})


class ExplodingClient:
    def complete(self, **_kw):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("no LLM call may happen while the knob is off")


def _memory(n_findings: int = 12) -> InMemoryMemory:
    memory = InMemoryMemory()
    for i in range(n_findings):
        memory.append_finding(f"REJECTED strategy s{i} — lesson {i}")
    return memory


def test_knob_defaults_off_and_makes_no_call(tmp_path):
    assert ResearchConfig().memory_distill_every == 0  # shipped default: off
    settings = Settings(state_dir=str(tmp_path))
    for _ in range(50):
        bump_research_session(settings.state_dir)
    # Even with many sessions banked, an off knob never touches the client.
    assert maybe_distill(settings, _memory(), client=ExplodingClient()) is False


def test_periodic_trigger_distills_resets_and_waits_again(tmp_path):
    settings = Settings(state_dir=str(tmp_path), research={"memory_distill_every": 2})
    memory = _memory()
    client = FakeDistillClient("- lesson one\n- lesson two\nprose ignored")

    bump_research_session(settings.state_dir)
    assert maybe_distill(settings, memory, client=client) is False  # 1 < 2: not due yet
    assert client.calls == [] and memory.distilled() == []

    bump_research_session(settings.state_dir)
    assert maybe_distill(settings, memory, client=client) is True
    assert memory.distilled() == ["- lesson one", "- lesson two"]
    # The full findings history (not a tail) went into the one call, findings only.
    prompt = client.calls[0]["messages"][0]["content"]
    assert "lesson 0" in prompt and "lesson 11" in prompt
    # Counter reset on success: the next close is not due until N more sessions.
    assert maybe_distill(settings, memory, client=client) is False
    assert len(client.calls) == 1


def test_no_client_degrades_to_stage1_and_stays_due(tmp_path):
    # conftest pins all provider keys empty, so build_llm_client resolves to None here.
    settings = Settings(state_dir=str(tmp_path), research={"memory_distill_every": 1})
    memory = _memory()
    bump_research_session(settings.state_dir)
    assert maybe_distill(settings, memory) is False  # degrades silently, stage-1 view holds
    assert memory.distilled() == []
    # Still due: a transient no-client close must not silently skip the cycle.
    assert maybe_distill(settings, memory, client=FakeDistillClient("- ok")) is True


def test_distill_needs_history_and_bullets(tmp_path):
    client = FakeDistillClient("- a lesson")
    assert distill_findings(_memory(3), client) is False  # too little history to fold
    assert client.calls == []

    memory = _memory()
    assert distill_findings(memory, FakeDistillClient("no bullets here")) is False
    assert memory.distilled() == []  # a refusal/empty answer never clobbers the block

    # A transport error degrades, never raises out of the close phase.
    class Boom:
        def complete(self, **_kw):
            raise RuntimeError("network down")

    memory2 = _memory()
    assert distill_findings(memory2, Boom()) is False
    assert memory2.distilled() == []
