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
from noctis.research.contract_sheet import CONTRACT_SHEET, SECTIONS
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


# ── 1b. The coder's system prompt grounds it in the exact graded API surface ──────────────
def test_system_prompt_carries_the_api_contract_sheet(tmp_path, families, fast_gate):
    # External behavior: every completion's system prompt embeds the full contract sheet, so the
    # coder sees the real signatures the write gate executes (not just TEMPLATE.py's elisions).
    engine, client = _author(tmp_path, families, [fenced(PROBE)])
    engine.author("probe", BRIEF)

    system = client.calls[0]["system"]
    assert CONTRACT_SHEET in system
    # Spot-check the coverage the sheet promises — builders, expectations, tape rules, indicator
    # State classes and tail functions with warmup semantics, and the exact ExitRules fields.
    for marker in (
        "flat(n)",
        "vol_spike(n, amplitude=0.05)",
        "gap(pct)",
        "always_flat()",
        "2-8",
        "60-2000",
        "sma(values, period)",
        "SmaState(period)",
        "ZScoreState(",
        ".update(bar)",
        "ExitRules(stop_pct=None, take_profit_pct=None, trail_pct=None)",
    ):
        assert marker in system, f"contract sheet missing {marker!r}"
    assert "warmup" in system.lower() and "fraction" in system.lower()


def test_system_prompt_lists_every_declared_api_signature(tmp_path, families, fast_gate):
    # Stronger form of the coverage promise: the prompt names each row of the shared data table,
    # so no builder/expectation/indicator/exit field is silently omitted from the coder's view.
    engine, client = _author(tmp_path, families, [fenced(PROBE)])
    engine.author("probe", BRIEF)

    system = client.calls[0]["system"]
    for section in SECTIONS:
        for entry in section.entries:
            assert entry.signature() in system, f"{entry.name} missing from the coder prompt"


def test_contract_sheet_survives_a_missing_seed_template(tmp_path, families, fast_gate):
    # Best-effort template behavior is preserved: with no TEMPLATE.py the engine still authors,
    # and the contract sheet — which does not depend on the template — is still in the prompt.
    engine, client = _author(tmp_path, families, [fenced(PROBE)])
    assert engine.author("probe", BRIEF)["name"] == "probe"
    assert CONTRACT_SHEET in client.calls[0]["system"]


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


# ── 8. Per-attempt observability callback: one call per completion, carrying the outcome ───
# The engine reports each coder completion — including each private retry — through an optional
# on_attempt(attempt, error) hook, resolved AFTER that attempt's validation (error=None on
# success, the StrategyValidationError otherwise). The toolbox adapts this into a session event;
# the engine itself stays toolbox-state-free.
def _seen_attempts(engine, name, brief) -> list[tuple[int, Exception | None]]:
    seen: list[tuple[int, Exception | None]] = []
    try:
        engine.author(name, brief, on_attempt=lambda n, err: seen.append((n, err)))
    except AuthoringError:
        pass
    return seen


def test_on_attempt_fires_once_on_first_try_success(tmp_path, families, fast_gate):
    engine, _ = _author(tmp_path, families, [fenced(PROBE)])
    seen = _seen_attempts(engine, "probe", BRIEF)
    assert seen == [(1, None)]  # exactly one completion: attempt 1, no validation error


def test_on_attempt_reports_retry_then_success_with_outcomes(tmp_path, families, fast_gate):
    engine, _ = _author(tmp_path, families, [fenced(BROKEN), fenced(PROBE)])
    seen = _seen_attempts(engine, "probe", BRIEF)
    assert [n for n, _ in seen] == [1, 2]  # one event per completion
    assert isinstance(seen[0][1], library.StrategyValidationError)  # attempt 1 failed the gate
    assert "class sets name" in str(seen[0][1])  # the real gate message is the outcome
    assert seen[1][1] is None  # attempt 2 landed


def test_on_attempt_fires_for_every_exhausted_retry(tmp_path, families, fast_gate):
    engine, _ = _author(tmp_path, families, [fenced(BROKEN)] * 3)
    seen = _seen_attempts(engine, "probe", BRIEF)
    assert [n for n, _ in seen] == [1, 2, 3]  # initial + 2 private retries, each its own event
    assert all(isinstance(err, library.StrategyValidationError) for _, err in seen)


def test_on_attempt_reports_a_non_code_reply_as_a_failed_attempt(tmp_path, families, fast_gate):
    engine, _ = _author(tmp_path, families, ["I think we should go long the dips.", fenced(PROBE)])
    seen = _seen_attempts(engine, "probe", BRIEF)
    assert seen[0][0] == 1 and isinstance(seen[0][1], library.StrategyValidationError)
    assert "code block" in str(seen[0][1])  # the non-code reply is the attempt's outcome
    assert seen[1] == (2, None)


def test_author_without_on_attempt_is_unchanged(tmp_path, families, fast_gate):
    # The callback is optional; omitting it changes nothing about authoring.
    engine, client = _author(tmp_path, families, [fenced(BROKEN), fenced(PROBE)])
    assert engine.author("probe", BRIEF)["name"] == "probe"
    assert len(client.calls) == 2
