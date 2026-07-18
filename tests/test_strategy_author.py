"""The StrategyAuthor engine — brief in, validated strategy file out.

Every test drives the engine with a fake coder client (the plain-text ``complete()``
boundary), so the suite touches no network and needs no API key (the autouse ``_clean_env``
fixture pins the provider keys empty). The write gate runs in-process via ``fast_gate`` —
the same checks and error contract as the production subprocess runner, minus the spawn.
"""

from __future__ import annotations

import pytest

from noctis.research import Capabilities, Turn
from noctis.research.author import (
    AuthoringError,
    StrategyAuthor,
    StrategyBrief,
)
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry
from tests.test_research_tools import PROBE

# PROBE authored under a class name that will not match the file name — the write gate
# rejects it deterministically ("class sets name=...").
BROKEN = PROBE.replace('name = "probe"', 'name = "mismatch"')

BRIEF = StrategyBrief(
    thesis="Long above a short moving average; the drift persists intraday.",
    entry_exit="Long when close > SMA(lookback); flat otherwise.",
    param_space="lookback int 5..40",
    scenarios="A rally pulls long; a steady decline stays flat.",
    style="momentum",
    symbols=("AAA", "BBB"),
)


def fenced(source: str) -> str:
    """Wrap strategy source in a python code fence, as a coder reply would."""
    return f"Here is the file:\n```python\n{source}```\n"


def named(source_name: str) -> str:
    """PROBE re-pointed at a fresh file name (its `name` attribute must match the file)."""
    return PROBE.replace('name = "probe"', f'name = "{source_name}"')


class FakeCoder:
    """Plays a fixed list of text replies through the neutral ``complete()`` seam and
    records every call — mirrors the fake-LLM prior art in the agent/ideation tests, but
    for the coder's plain-text (no-tool) completion shape."""

    def __init__(self, replies, capabilities=None):
        self._replies = list(replies)
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
        if not self._replies:
            raise AssertionError("coder script exhausted — the engine should have stopped")
        text = self._replies.pop(0)
        return Turn(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            usage={},
            assistant_message={"role": "assistant", "content": text},
        )


def _author(tmp_path, families, replies) -> tuple[StrategyAuthor, FakeCoder]:
    client = FakeCoder(replies)
    engine = StrategyAuthor(client=client, strategies_dir=tmp_path, families=families)
    return engine, client


@pytest.fixture
def families():
    return FamilyRegistry()


# ── 1. Happy path: a good brief → a validated file in the working tier ────────────────────
def test_happy_path_authors_validated_file_into_working_tier(tmp_path, families, fast_gate):
    engine, client = _author(tmp_path, families, [fenced(PROBE)])

    result = engine.author("probe", BRIEF)

    assert result["name"] == "probe"
    # Authored files land in the gitignored working tier (__tmp/), never the seed root.
    assert library.strategy_path(tmp_path, "probe") == tmp_path / "__tmp" / "probe.py"
    assert "probe" in families
    assert len(client.calls) == 1  # exactly one completion for a first-try success


def test_engine_needs_no_api_key(tmp_path, families, fast_gate, monkeypatch):
    # The autouse _clean_env already pins these empty; assert the engine authors regardless.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    engine, _ = _author(tmp_path, families, [fenced(PROBE)])
    assert engine.author("probe", BRIEF)["name"] == "probe"


# ── 2. Validation error → private retry carrying the error → success lands ────────────────
def test_validation_error_triggers_private_retry_that_lands(tmp_path, families, fast_gate):
    engine, client = _author(tmp_path, families, [fenced(BROKEN), fenced(PROBE)])

    # The caller sees only the final outcome — no exception on the way.
    result = engine.author("probe", BRIEF)

    assert result["name"] == "probe"
    assert library.strategy_path(tmp_path, "probe") == tmp_path / "__tmp" / "probe.py"
    assert len(client.calls) == 2
    # The retry carried the gate's error context to the coder.
    retry_msg = client.calls[1]["messages"][0]["content"]
    assert "class sets name" in retry_msg


# ── 3. Retries exhausted (2) → typed error carrying the final validation error ────────────
def test_retries_exhausted_raises_authoring_error_with_final_validation_error(
    tmp_path, families, fast_gate
):
    engine, client = _author(tmp_path, families, [fenced(BROKEN)] * 3)

    with pytest.raises(AuthoringError) as excinfo:
        engine.author("probe", BRIEF)

    assert len(client.calls) == 3  # initial + 2 retries, then it gives up
    assert library.strategy_path(tmp_path, "probe") is None  # nothing landed
    err = excinfo.value
    assert isinstance(err.validation_error, library.StrategyValidationError)
    assert isinstance(err.__cause__, library.StrategyValidationError)
    assert "class sets name" in str(err.validation_error)


# ── 4. A non-code reply is rejected and counts as an attempt ──────────────────────────────
def test_non_code_reply_is_rejected_and_counts_as_attempt(tmp_path, families, fast_gate):
    engine, client = _author(
        tmp_path, families, ["I think we should go long the dips.", fenced(PROBE)]
    )

    result = engine.author("probe", BRIEF)

    assert result["name"] == "probe"
    assert len(client.calls) == 2  # the non-code reply consumed one attempt


def test_all_non_code_replies_exhaust_the_attempt_budget(tmp_path, families, fast_gate):
    engine, client = _author(tmp_path, families, ["no code here"] * 3)

    with pytest.raises(AuthoringError):
        engine.author("probe", BRIEF)

    assert len(client.calls) == 3  # three non-code replies, capped at the retry budget


# ── 5. Coder calls are stateless single completions (thinking off) ────────────────────────
def test_coder_calls_are_stateless_single_completions(tmp_path, families, fast_gate):
    engine, client = _author(tmp_path, families, [fenced(BROKEN), fenced(PROBE)])
    engine.author("probe", BRIEF)

    for call in client.calls:
        # A bare codegen completion: no tool-use loop, no forced tool, no streaming
        # (thinking is pinned off where the client is built, client_for(thinking="off")).
        assert call["tools"] == []
        assert call["tool_choice"] is None
        assert call["on_delta"] is None
        # Each completion is a fresh, self-contained prompt — one user turn, no carried
        # assistant/tool history the client would have to hold between calls.
        assert len(call["messages"]) == 1
        assert call["messages"][0]["role"] == "user"


def test_a_fresh_authoring_job_carries_no_prior_state(tmp_path, families, fast_gate):
    client = FakeCoder([fenced(PROBE), fenced(PROBE.replace("probe", "probe_two"))])
    engine = StrategyAuthor(client=client, strategies_dir=tmp_path, families=families)

    engine.author("probe", BRIEF)
    engine.author("probe_two", BRIEF)

    # Each job builds a fresh single-message prompt — no accumulated history between jobs.
    second_call = client.calls[1]
    assert len(second_call["messages"]) == 1
    assert "probe_two" in second_call["messages"][0]["content"]


# ── 6. Reference adaptation: a named library strategy's source enters the prompt ──────────
def test_reference_source_is_composed_into_the_prompt(tmp_path, families, fast_gate):
    library.write_strategy(tmp_path, "ref_strat", named("ref_strat"), families)
    brief = StrategyBrief(
        thesis="Adapt the reference's proven structure to a new symbol set.",
        entry_exit="Long above the SMA, mirroring the reference.",
        param_space="lookback int 5..40",
        scenarios="A rally pulls long; a steady decline stays flat.",
        reference="ref_strat",
    )
    engine, client = _author(tmp_path, families, [fenced(named("adapted"))])

    result = engine.author("adapted", brief)

    assert result["name"] == "adapted"
    assert library.strategy_path(tmp_path, "adapted") == tmp_path / "__tmp" / "adapted.py"
    # The reference's full source reached the coder to translate, not just its name.
    assert named("ref_strat") in client.calls[0]["messages"][0]["content"]


def test_unknown_reference_is_rejected_before_any_coder_completion(tmp_path, families, fast_gate):
    brief = StrategyBrief(
        thesis="Adapt a reference that does not exist.",
        entry_exit="Long above the SMA.",
        param_space="lookback int 5..40",
        scenarios="A rally pulls long; a decline stays flat.",
        reference="no_such_strategy",
    )
    engine, client = _author(tmp_path, families, [fenced(PROBE)])

    with pytest.raises(library.StrategyValidationError) as excinfo:
        engine.author("adapted", brief)

    assert "no_such_strategy" in str(excinfo.value)
    assert client.calls == []  # rejected before spending a completion
    assert library.strategy_path(tmp_path, "adapted") is None


# ── 7. Revision: an existing target name composes its current source as the change target ─
def test_existing_name_composes_current_source_as_a_revision(tmp_path, families, fast_gate):
    library.write_strategy(tmp_path, "probe", PROBE, families)
    revised = PROBE.replace(
        "Toy probe: long above its own moving average.",
        "Revised probe: long above its own moving average.",
    )
    engine, client = _author(tmp_path, families, [fenced(revised)])

    result = engine.author("probe", BRIEF)

    assert result["name"] == "probe"
    # The current version was composed into the prompt as the change target.
    assert PROBE in client.calls[0]["messages"][0]["content"]
    # The validated revision replaced the file via the normal write path.
    assert library.strategy_source(tmp_path, "probe") == revised


def test_failed_revision_leaves_the_existing_file_untouched(tmp_path, families, fast_gate):
    library.write_strategy(tmp_path, "probe", PROBE, families)
    engine, client = _author(tmp_path, families, [fenced(BROKEN)] * 3)

    with pytest.raises(AuthoringError):
        engine.author("probe", BRIEF)

    assert len(client.calls) == 3
    # The previous version is intact — the write gate never replaced it (library guarantee).
    assert library.strategy_source(tmp_path, "probe") == PROBE
