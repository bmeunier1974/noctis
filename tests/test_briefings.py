"""Build-time fit assertion for the v1 episodic briefing builders (epic #62 / story #67).

The formulate and decide briefings are rebuilt fresh from disk on every call out of the shared
digest builders plus the session-ledger tail, and each asserts the rendered prompt fits the
configured context window — trimming only advisory blocks in a fixed priority order (memory tail
→ library stubs → digest breadth), never the gate-facing numbers, and failing loudly when even a
fully-trimmed briefing still does not fit. These tests lock:

* statelessness — two calls straddling a disk change reflect the change and share no state;
* the fit assertion at an 8k window, with the trim order exercised block-by-block;
* gate-facing numbers surviving every trim level (silent truncation is structurally impossible);
* a loud failure when the un-trimmable core alone overflows the window;
* the ~1.5-3k token target band on realistic fixture state.
"""

from __future__ import annotations

import pytest

from noctis.research import Mandate
from noctis.research.agent import _estimate_tokens
from noctis.research.briefings import (
    BriefingTooLargeError,
    decide_briefing,
    formulate_briefing,
)
from noctis.research.ledger import SessionLedger
from noctis.strategies.library import set_header, write_strategy
from tests.test_champions import make_scorecard
from tests.test_research_tools import LENIENT, PROBE, _make_toolbox

# Sentinels planted in each advisory block so trim decisions are observable in the rendered text.
_MEMORY_SENTINEL = "MEMSENTINEL-finding-do-not-repeat"
_LIBRARY_SENTINEL = "libsentinel"
_BREADTH_KEY = "trend_efficiency"  # a per-symbol character field, only in the digest-breadth block
_LEDGER_SENTINEL = "LEDGERSENTINEL"
_EXHAUSTED_LABEL = "minute rsi mean reversion"

# Gate-facing markers that must survive every trim level of the FORMULATE briefing.
_FORMULATE_GATE_MARKERS = (
    "round_trip_cost_bp",  # market cost arithmetic
    _EXHAUSTED_LABEL,  # exhausted-class hygiene guard
    "test_metric",  # champion board (beat-the-weakest bar)
    _LEDGER_SENTINEL,  # the session narrative — never dropped
)

_HUGE = 10_000_000


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """These tests exercise briefing assembly, not subprocess write-gate isolation."""


def _tokens(text: str) -> int:
    return _estimate_tokens(len(text), [])


def _bloat_memory(box) -> None:
    """Push the advisory memory tail well past an 8k window (distinct dead-end families with
    long reasons — the consolidated rejected view keeps the latest 20, uncapped in chars)."""
    for i in range(20):
        box.memory.record_rejected(f"bloatfam{i}", {"lookback": i}, reason="y" * 1600)


def _named(source_name: str, new_name: str, marker: str) -> str:
    return PROBE.replace('name = "probe"', f'name = "{new_name}"').replace(
        "Toy probe: long above its own moving average.", marker
    )


def _populate(tmp_path):
    """A realistic session: a populated market digest, several champions, a memory file with
    findings and dead ends, a handful of library strategies (one rejected), an exhausted class,
    a journal of ranked trials for ``probe``, and a session ledger with theses and a verdict."""
    box = _make_toolbox(tmp_path)  # universe AAA..DDD with bars; ships a 'probe' strategy

    # Library: two live strategies + one rejected corpse (collapsed to a stub by the index).
    for name in ("alpha_mom", _LIBRARY_SENTINEL):
        write_strategy(
            box.strategies_dir, name, _named("probe", name, f"{name} thesis marker."), box.families
        )
    write_strategy(
        box.strategies_dir, "corpse", _named("probe", "corpse", "corpse thesis."), box.families
    )
    set_header(box.strategies_dir, "corpse", families=box.families, status="rejected")

    # Champions (the beat-the-weakest bar).
    box.registry.consider(
        make_scorecard("alpha_mom", test_metric=1.5, train_metric=1.6),
        LENIENT,
        mandate_source="profile:aggressive",
    )
    box.registry.consider(
        make_scorecard("gamma_break", test_metric=1.2, train_metric=1.3),
        LENIENT,
        mandate_source="profile:balanced",
    )

    # Memory: advisory findings + a rejected dead end.
    box.memory.append_finding(f"PROMOTED alpha_mom once — {_MEMORY_SENTINEL}")
    box.memory.append_finding("DEAD END minute RSI mean reversion nets negative after 4bp round")
    box.memory.record_rejected("rsi_scalp", {"lookback": 3}, reason="gross edge below cost")

    # A cross-session exhausted class (research-hygiene guard).
    box.exhausted.record(
        _EXHAUSTED_LABEL,
        "gross edge/trade below the 4bp round trip on every symbol tried",
        example="corpse",
    )

    # Journal evidence for the decide subject 'probe': thesis + class tag + ranked trials.
    box.journal.record_thesis("probe", "Long above own moving average while the trend is up.")
    box.journal.record_class_tag("probe", "intraday momentum")
    box.journal.record_trial(
        "probe",
        source="backtest",
        symbols=["AAA", "BBB"],
        params={"lookback": 12, "edge": 1.0},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=1.41, train_metric=1.55, lookback=12),
    )
    box.journal.record_trial(
        "probe",
        source="sweep",
        symbols=["AAA"],
        params={"lookback": 20, "edge": 1.1},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=0.92, train_metric=1.03, lookback=20),
    )
    box.journal.record_trial(
        "probe",
        source="sweep",
        symbols=["BBB"],
        params={"lookback": 30, "edge": 0.9},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=0.55, train_metric=0.80, lookback=30),
    )

    ledger = SessionLedger(box.state_dir, session_id="sess-1")
    ledger.record_session_start(mandate="profile:aggressive", budgets={}, models={})
    ledger.record_thesis("probe", f"{_LEDGER_SENTINEL} momentum long above own MA at 1h")
    ledger.record_thesis("corpse", "Minute RSI mean reversion buys oversold dips.")
    ledger.record_verdict(
        "corpse",
        verdict="reject",
        lesson="minute RSI mean reversion nets negative after the 4bp round trip",
        promoted=False,
    )

    mandate = Mandate(
        text="Pursue liquid-name momentum that clears cost at 1h.",
        source="profile:aggressive",
        summary="aggressive: liquid-name momentum, tune on Sharpe",
        references=[],
        config_overrides={},
        symbols=["AAA", "BBB"],
    )
    return box, ledger, mandate


# ── statelessness: rebuilt fresh from disk, no state carried between calls ──────────────────
def test_formulate_briefing_is_deterministic_and_rebuilt_fresh_from_disk(tmp_path):
    box, ledger, mandate = _populate(tmp_path)

    before = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)
    # Same inputs, same bytes — no hidden per-call state.
    assert before == formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)

    # Mutate the disk sources the two shared inputs read from (the ledger JSONL + the library).
    ledger.record_thesis("newidea", "NEWDISKSENTINEL breakout on opening gaps")
    write_strategy(
        box.strategies_dir,
        "newlib",
        _named("probe", "newlib", "newlib thesis marker."),
        box.families,
    )
    after = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)

    assert "NEWDISKSENTINEL" not in before and "NEWDISKSENTINEL" in after
    assert "newlib" not in before and "newlib" in after


def test_decide_briefing_rebuilt_fresh_from_disk(tmp_path):
    box, ledger, mandate = _populate(tmp_path)
    before = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)

    box.journal.record_trial(
        "probe",
        source="sweep",
        symbols=["CCC"],
        params={"lookback": 44, "edge": 1.3},
        window={"train": 200, "test": 100},
        card=make_scorecard("probe", test_metric=1.77, train_metric=1.80, lookback=44),
    )
    after = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)

    assert '"lookback": 44' not in before and '"lookback": 44' in after


# ── the trim order, exercised advisory-block by advisory-block ──────────────────────────────
def test_formulate_trim_order_drops_memory_then_library_then_breadth(tmp_path):
    box, ledger, mandate = _populate(tmp_path)

    full = formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)
    assert _MEMORY_SENTINEL in full and _LIBRARY_SENTINEL in full and _BREADTH_KEY in full

    # A window one token below each successive render forces exactly the next advisory drop.
    b1 = formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(full) - 1)
    assert _MEMORY_SENTINEL not in b1  # memory tail dropped first
    assert _LIBRARY_SENTINEL in b1 and _BREADTH_KEY in b1
    for marker in _FORMULATE_GATE_MARKERS:
        assert marker in b1

    b2 = formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(b1) - 1)
    assert _MEMORY_SENTINEL not in b2 and _LIBRARY_SENTINEL not in b2  # library stubs dropped next
    assert _BREADTH_KEY in b2
    for marker in _FORMULATE_GATE_MARKERS:
        assert marker in b2

    b3 = formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(b2) - 1)
    assert _BREADTH_KEY not in b3  # digest breadth dropped last
    for marker in _FORMULATE_GATE_MARKERS:
        assert marker in b3

    # Every advisory block already trimmed and the un-trimmable core still overflows → loud fail.
    with pytest.raises(BriefingTooLargeError):
        formulate_briefing(box, ledger, mandate=mandate, context_window=_tokens(b3) - 1)


def test_formulate_fit_assertion_at_8k_window_trims_and_keeps_gate_numbers(tmp_path):
    box, ledger, mandate = _populate(tmp_path)
    _bloat_memory(box)  # push the advisory memory tail past an 8k window
    assert _tokens(formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE)) > 8000

    fitted = formulate_briefing(box, ledger, mandate=mandate, context_window=8000)
    assert _tokens(fitted) <= 8000
    assert _MEMORY_SENTINEL not in fitted  # advisory memory trimmed to fit
    for marker in _FORMULATE_GATE_MARKERS:  # gate-facing numbers never trimmed
        assert marker in fitted


def test_formulate_raises_loudly_when_core_exceeds_window(tmp_path):
    box, ledger, mandate = _populate(tmp_path)
    # A window smaller than the un-trimmable core is a loud failure, never a silent truncation.
    with pytest.raises(BriefingTooLargeError):
        formulate_briefing(box, ledger, mandate=mandate, context_window=10)


# ── decide briefing: gate-facing candidate evidence, never trimmed ─────────────────────────
def test_decide_briefing_carries_ranked_journal_evidence(tmp_path):
    box, ledger, mandate = _populate(tmp_path)
    brief = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE)

    assert "min_trials_gate" in brief  # the exhaustion floor the verdict is judged against
    assert '"n_distinct_params": 3' in brief
    assert "top_trials" in brief and '"lookback": 12' in brief  # ranked trials + params
    assert '"verdict": "reject"' in brief  # journaled verdicts surfaced


def test_decide_gate_numbers_survive_trim_at_8k(tmp_path):
    box, ledger, mandate = _populate(tmp_path)
    _bloat_memory(box)

    fitted = decide_briefing(box, ledger, "probe", mandate=mandate, context_window=8000)
    assert _tokens(fitted) <= 8000
    assert _MEMORY_SENTINEL not in fitted  # advisory memory trimmed
    # The candidate's gate-facing evidence survives the trim.
    assert "min_trials_gate" in fitted
    assert '"n_distinct_params": 3' in fitted
    assert '"lookback": 12' in fitted


# ── the ~1.5-3k token target band on realistic (un-bloated) state ───────────────────────────
def test_briefings_land_in_target_token_band_on_realistic_state(tmp_path):
    box, ledger, mandate = _populate(tmp_path)
    formulate = _tokens(formulate_briefing(box, ledger, mandate=mandate, context_window=_HUGE))
    decide = _tokens(decide_briefing(box, ledger, "probe", mandate=mandate, context_window=_HUGE))
    assert 800 < formulate < 3500, formulate
    assert 800 < decide < 3500, decide
