"""Live-holdout plan 4: the TRADING driver choice is explicit.

The driver was chosen off ``data.provider`` invisibly — an operator reading ``databento`` had
no signal that TRADING was replaying the catalog rather than streaming (the single most
surprising fact of the 2026-07-07 diagnosis). Now it resolves through one helper and is logged
loudly at every TRADING entry, and a ``trading.execution`` knob can force the choice.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest

from noctis.config import load_settings
from noctis.engine import build_runtime, resolve_trading_driver
from noctis.memory import MemoryStore

from ._session_helpers import (
    _bars_local,
    _FakeLake,
    _FakeRegistry,
    _run_phase,
    _uptrend,
)


def _settings(tmp_path, *, provider="databento", execution="auto"):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL]\n"
        f"state_dir: {tmp_path}/state/\n"
        f"strategies_dir: {tmp_path}/strategies/\n"
        f"data:\n  provider: {provider}\n"
        f"trading:\n  execution: {execution}\n"
    )
    return load_settings(config_path=cfg)


def _runtime(tmp_path, *, provider="databento", execution="auto", feed_factory=None):
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    return build_runtime(
        _settings(tmp_path, provider=provider, execution=execution),
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=_FakeRegistry(),
        reports_dir=str(tmp_path / "reports"),
        feed_factory=feed_factory,
    )


# ── resolution ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "provider,execution,expected",
    [
        ("databento", "auto", "replay"),  # today's default: no live feed → replay
        ("yfinance", "auto", "live"),  # today's opt-in: yfinance → live
        ("databento", "replay", "replay"),
        ("yfinance", "replay", "replay"),  # forced replay wins even under yfinance
        ("databento", "live", "live"),  # forced intent even without a feed
        ("yfinance", "live", "live"),
    ],
)
def test_resolve_trading_driver_matrix(tmp_path, provider, execution, expected):
    settings = _settings(tmp_path, provider=provider, execution=execution)
    assert resolve_trading_driver(settings) == expected


# ── loud logging at TRADING entry ────────────────────────────────────────────────────────────
def test_replay_logs_a_loud_warning(tmp_path, caplog):
    runtime = _runtime(tmp_path, provider="databento", execution="auto")
    with caplog.at_level(logging.WARNING, logger="noctis.runtime"):
        _run_phase(runtime)
    assert any("will REPLAY the catalog" in r.getMessage() for r in caplog.records)


def test_execution_replay_forces_replay_under_yfinance(tmp_path, caplog):
    # execution=replay must never build a live feed, even with data.provider=yfinance.
    runtime = _runtime(tmp_path, provider="yfinance", execution="replay")
    with caplog.at_level(logging.WARNING, logger="noctis.runtime"):
        outcome = _run_phase(runtime)
    assert any("will REPLAY the catalog" in r.getMessage() for r in caplog.records)
    assert outcome.sessions  # the batch replay driver ran


def test_execution_live_mismatch_warns_then_falls_back(tmp_path, caplog):
    # execution=live with a non-live provider: state the mismatch, attempt, then fall back —
    # an unhonored "live" intent is exactly the silent surprise plan 4 kills.
    def _raising_feed(**kwargs):
        raise RuntimeError("no live feed in test")

    runtime = _runtime(tmp_path, provider="databento", execution="live", feed_factory=_raising_feed)
    with caplog.at_level(logging.WARNING, logger="noctis.runtime"):
        outcome = _run_phase(runtime)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "execution=live but data.provider=databento" in msgs
    assert "falling back to catalog replay" in msgs
    assert any(e.startswith("Live feed unavailable") for e in outcome.events)


def test_fill_rationale_labels_exit_fills_by_reason():
    """The report's trade rows say WHY a fill happened: orphan flattens keep their label,
    protective exits carry their reason, everything else is a champion signal."""
    from noctis.broker.seam import Fill, Side
    from noctis.engine.trading_phase import _fill_rationale

    signal = Fill("AAPL", Side.BUY, 10.0, 100.0, 0.0, 1)
    stop = Fill("AAPL", Side.SELL, 10.0, 90.0, 0.0, 2, reason="stop")
    trail = Fill("MSFT", Side.SELL, 5.0, 80.0, 0.0, 3, reason="trail")

    assert _fill_rationale(signal, orphaned=set()) == "champion signal"
    assert _fill_rationale(stop, orphaned=set()) == "protective exit (stop)"
    assert _fill_rationale(trail, orphaned=set()) == "protective exit (trail)"
    # An orphan symbol has no strategy this session — any fill on it IS the flatten.
    assert _fill_rationale(stop, orphaned={"AAPL"}) == "orphan flatten"
