"""The StrategyAuthor engine — brief in, validated strategy file out.

Every test drives the engine with a fake coder client (the plain-text ``complete()``
boundary), so the suite touches no network and needs no API key (the autouse ``_clean_env``
fixture pins the provider keys empty). The write gate runs in-process via ``fast_gate`` —
the same checks and error contract as the production subprocess runner, minus the spawn.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from noctis.research import Capabilities, Turn
from noctis.research.author import (
    WORKED_EXAMPLE_NAME,
    AuthoringError,
    StrategyAuthor,
    StrategyBrief,
)
from noctis.research.contract_sheet import CONTRACT_SHEET, SECTIONS
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry
from tests.test_research_tools import PROBE

# The committed seed tier the author engine reads its worked example from (read-only input).
_REPO_SEEDS = Path(__file__).resolve().parents[1] / "strategies"


def _worked_example_source() -> str:
    """The full source of the committed seed the coder prompt folds in as its worked example."""
    return (_REPO_SEEDS / f"{WORKED_EXAMPLE_NAME}.py").read_text(encoding="utf-8")


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


@pytest.fixture
def seeded_dir(tmp_path):
    """An author seeds root carrying the committed worked-example seed (read-only input).

    Mirrors a real install where the coder's system prompt can fold in one complete seed
    strategy; the write target (__tmp/) still lives under this same throwaway path.
    """
    shutil.copyfile(
        _REPO_SEEDS / f"{WORKED_EXAMPLE_NAME}.py", tmp_path / f"{WORKED_EXAMPLE_NAME}.py"
    )
    return tmp_path


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


# ── 1c. The coder owns tape construction and carries the feasibility rules ────────────────
def test_system_prompt_states_coder_owned_tapes_and_feasibility_rules(
    tmp_path, families, fast_gate
):
    # External behavior: every completion's system prompt tells the coder it owns tape
    # construction (the brief's scenario sketch is intent, not a tape to transcribe) and carries
    # the four feasibility rules that killed the unsatisfiable-brief retry loop.
    engine, client = _author(tmp_path, families, [fenced(PROBE)])
    engine.author("probe", BRIEF)

    low = client.calls[0]["system"].lower()
    # Coder owns tape construction; the scenario sketch is intent, not tape dictation.
    assert "own tape construction" in low
    assert "intent" in low
    # Feasibility rule 1: derive warmup from the Params defaults before an expectation window.
    assert "warmup" in low
    assert "params default" in low
    # Feasibility rule 2: higher-timeframe strategies multiply warmup.
    assert "higher-timeframe" in low
    assert "multipl" in low  # multiplies / multiply warmup
    # Feasibility rule 3: a scale-free percentile-rank rule cannot be silenced by chop amplitude.
    assert "percentile" in low
    assert "scale-free" in low
    assert "amplitude" in low
    # Feasibility rule 4: falsify the level condition (a steady selloff under a long-only rule).
    assert "falsif" in low
    assert "selloff" in low


def test_feasibility_rules_survive_a_missing_seed_template(tmp_path, families, fast_gate):
    # The feasibility rules do not depend on TEMPLATE.py: a bare library still ships them.
    engine, client = _author(tmp_path, families, [fenced(PROBE)])
    assert engine.author("probe", BRIEF)["name"] == "probe"
    low = client.calls[0]["system"].lower()
    assert "own tape construction" in low
    assert "scale-free" in low


# ── 1d. The coder system prompt unconditionally folds in one complete worked example ──────
def test_system_prompt_carries_a_complete_worked_example(seeded_dir, families, fast_gate):
    # External behavior: every completion's system prompt embeds one complete seed strategy's
    # full source — a real shipped file using the exact graded APIs — so the coder always sees a
    # working example, not just TEMPLATE.py's skeleton.
    engine, client = _author(seeded_dir, families, [fenced(PROBE)])
    engine.author("probe", BRIEF)

    assert _worked_example_source() in client.calls[0]["system"]


def test_worked_example_is_present_when_the_brief_names_no_reference(
    seeded_dir, families, fast_gate
):
    # The worked example is UNCONDITIONAL: BRIEF carries no reference, yet the seed source is
    # still folded into the system prompt (the mechanism no longer depends on a referenced brief).
    assert BRIEF.reference is None
    engine, client = _author(seeded_dir, families, [fenced(PROBE)])
    engine.author("probe", BRIEF)

    assert _worked_example_source() in client.calls[0]["system"]


def test_worked_example_and_reference_are_both_composed(seeded_dir, families, fast_gate):
    # A referenced brief still gets the reference source in the USER prompt IN ADDITION to the
    # unconditional worked example in the SYSTEM prompt — the two mechanisms stack, not replace.
    library.write_strategy(seeded_dir, "ref_strat", named("ref_strat"), families)
    brief = StrategyBrief(
        thesis="Adapt the reference's proven structure to a new symbol set.",
        entry_exit="Long above the SMA, mirroring the reference.",
        param_space="lookback int 5..40",
        scenarios="A rally pulls long; a steady decline stays flat.",
        reference="ref_strat",
    )
    engine, client = _author(seeded_dir, families, [fenced(named("adapted"))])
    engine.author("adapted", brief)

    system = client.calls[0]["system"]
    user = client.calls[0]["messages"][0]["content"]
    assert _worked_example_source() in system  # the worked example still rides the system prompt
    assert named("ref_strat") in user  # and the reference source is still composed in on top


def test_chosen_worked_example_is_a_stateful_entry_exit_seed(seeded_dir, families, fast_gate):
    # The chosen seed is a real stateful entry/exit example (not a stateless one-liner): it
    # resets incremental state in on_start, latches a position across bars, and ends every bar
    # with a target — the whole-file pattern the coder must reproduce.
    src = _worked_example_source()
    assert "def on_start(" in src and "def on_bar(" in src
    assert "self._pos" in src  # a position latched across bars
    assert "ctx.set_target(" in src  # every bar ends with a directional/flat target
    assert "def scenarios(" in src and "always_flat()" in src  # its own known-outcome oracle


def test_missing_worked_example_seed_still_authors_rules_only(tmp_path, families, fast_gate):
    # Degraded install: with no seed on disk the engine still authors (best-effort, same graceful
    # path as a missing TEMPLATE.py) and the API contract + rules still ground the coder — only
    # the worked example is absent.
    engine, client = _author(tmp_path, families, [fenced(PROBE)])
    assert engine.author("probe", BRIEF)["name"] == "probe"

    system = client.calls[0]["system"]
    assert _worked_example_source() not in system  # no seed on disk → no worked example
    assert CONTRACT_SHEET in system  # the rules-only prompt still carries the full API surface


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


# ── 4b. Prompt caching lands on the coder's enlarged system prompt (#17) ──────────────────
def test_system_prompt_is_one_cached_block_when_client_supports_prompt_cache(
    tmp_path, families, fast_gate
):
    # External behavior: on a caching-capable client, the coder's system prompt reaches
    # complete() as ONE cached content block, so retries within a job re-read (never re-pay) the
    # enlarged contract-sheet + worked-example system prompt (#14-#16).
    client = FakeCoder([fenced(PROBE)], capabilities=Capabilities(prompt_cache=True))
    engine = StrategyAuthor(client=client, strategies_dir=tmp_path, families=families)
    engine.author("probe", BRIEF)

    system = client.calls[0]["system"]
    assert isinstance(system, list) and len(system) == 1
    block = system[0]
    assert block["type"] == "text"
    assert CONTRACT_SHEET in block["text"]  # the enlarged system prompt is the cached content
    assert block["cache_control"] == {"type": "ephemeral"}


def test_cached_system_prompt_is_reused_by_identity_across_attempts(tmp_path, families, fast_gate):
    # The cache breakpoint is built once and reused by identity on every retry within a job — the
    # "cache once, read thereafter" contract, so a retried job never rewrites the system prefix.
    client = FakeCoder(
        [fenced(BROKEN), fenced(PROBE)], capabilities=Capabilities(prompt_cache=True)
    )
    engine = StrategyAuthor(client=client, strategies_dir=tmp_path, families=families)
    engine.author("probe", BRIEF)

    assert len(client.calls) == 2
    assert client.calls[1]["system"] is client.calls[0]["system"]


def test_no_prompt_cache_capability_leaves_the_system_prompt_a_plain_string(
    tmp_path, families, fast_gate
):
    # A provider whose caching is automatic (OpenAI) or unsupported (local): no breakpoint — the
    # system prompt reaches complete() as the plain string, exactly as before.
    client = FakeCoder([fenced(PROBE)], capabilities=Capabilities(prompt_cache=False))
    engine = StrategyAuthor(client=client, strategies_dir=tmp_path, families=families)
    engine.author("probe", BRIEF)

    system = client.calls[0]["system"]
    assert isinstance(system, str)
    assert CONTRACT_SHEET in system


def test_authoring_telemetry_unchanged_when_prompt_caching_is_on(tmp_path, families, fast_gate):
    # Wiring the cache breakpoint does not disturb the per-completion authoring telemetry (#9):
    # on a caching-capable client every attempt still reports its outcome, one event per completion.
    client = FakeCoder(
        [fenced(BROKEN), fenced(PROBE)], capabilities=Capabilities(prompt_cache=True)
    )
    engine = StrategyAuthor(client=client, strategies_dir=tmp_path, families=families)
    seen: list[tuple[int, Exception | None]] = []
    engine.author("probe", BRIEF, on_attempt=lambda n, err, src: seen.append((n, err)))

    assert [n for n, _ in seen] == [1, 2]  # one event per completion, retries included
    assert isinstance(seen[0][1], library.StrategyValidationError)  # attempt 1 failed the gate
    assert seen[1][1] is None  # attempt 2 landed


# ── 5. Coder calls are stateless single completions ───────────────────────────────────────
def test_coder_calls_are_stateless_single_completions(tmp_path, families, fast_gate):
    engine, client = _author(tmp_path, families, [fenced(BROKEN), fenced(PROBE)])
    engine.author("probe", BRIEF)

    for call in client.calls:
        # A bare codegen completion: no tool-use loop, no forced tool, no streaming (the
        # coder's thinking dial rides on the client built at the composition root, not here).
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
# on_attempt(attempt, error, source) hook, resolved AFTER that attempt's validation (error=None
# on success, the StrategyValidationError otherwise). The toolbox adapts this into a session event
# and an on-disk failure record; the engine itself stays toolbox-state-free.
def _seen_attempts(engine, name, brief) -> list[tuple[int, Exception | None]]:
    seen: list[tuple[int, Exception | None]] = []
    try:
        engine.author(name, brief, on_attempt=lambda n, err, src: seen.append((n, err)))
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


# ── 8b. The callback additively carries the attempted source (#18) ─────────────────────────
# The engine passes each attempt's material as a third argument — the extracted code block on a
# gate rejection or success, the raw reply text on a non-code reply — so a toolbox-side sink can
# persist the exact bytes the coder produced. The engine keeps none of it (stateless across jobs).
def _seen_with_source(engine, name, brief) -> list[tuple[int, Exception | None, str]]:
    seen: list[tuple[int, Exception | None, str]] = []
    try:
        engine.author(name, brief, on_attempt=lambda n, err, src: seen.append((n, err, src)))
    except AuthoringError:
        pass
    return seen


def test_on_attempt_carries_the_extracted_source_on_a_gate_rejection(tmp_path, families, fast_gate):
    engine, _ = _author(tmp_path, families, [fenced(BROKEN), fenced(PROBE)])
    seen = _seen_with_source(engine, "probe", BRIEF)

    # Attempt 1 failed the gate; the callback carries the exact rejected source (the code block).
    assert seen[0][0] == 1
    assert isinstance(seen[0][1], library.StrategyValidationError)
    assert seen[0][2] == BROKEN
    # Attempt 2 landed; the source it carries is the file that passed.
    assert seen[1][1] is None
    assert seen[1][2] == PROBE


def test_on_attempt_carries_the_raw_reply_when_no_code_block(tmp_path, families, fast_gate):
    reply = "I think we should go long the dips."
    engine, _ = _author(tmp_path, families, [reply, fenced(PROBE)])
    seen = _seen_with_source(engine, "probe", BRIEF)

    # A non-code reply has no code block: the callback carries the raw reply text to persist.
    assert isinstance(seen[0][1], library.StrategyValidationError)
    assert seen[0][2] == reply
