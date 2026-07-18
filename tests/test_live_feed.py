"""The live streaming feed wired into the TRADING loop (driver-level contract).

A deterministic, scripted **fake** feed (:class:`ScriptedFeed`) stands in for the real
yfinance feed — no network. Covers: paced-vs-batch driver parity, degraded-halt +
recovery, stop-between-polls, live reconciliation flagging drift, provider gating by call
counts, and paper-only routing under a live feed. (yfinance-specific feed behaviour — closed-
bar emission, fetch throttling, staleness — lives in ``test_yfinance_feed.py``.)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from noctis.backtest.scorecard import Scorecard
from noctis.broker import LiveBrokerUnavailableError, PaperBroker
from noctis.broker.live_stub import LiveBroker
from noctis.champions import build_registry
from noctis.champions.registry import ChampionEntry
from noctis.config import load_settings
from noctis.config.gate import resolve_execution_mode
from noctis.data import MarketDataLake
from noctis.data.types import to_ns
from noctis.engine import Phase, SimulatedSleeper, build_runtime
from noctis.live import SessionConfig, run_trading, run_trading_day
from noctis.memory import MemoryStore
from noctis.strategies import Candidate
from noctis.strategies.base import Bar

from ._data_helpers import MockVendor, make_ohlcv

ET = ZoneInfo("America/New_York")


# --- fakes -------------------------------------------------------------------------------


class ScriptedFeed:
    """A fake live feed: hands the driver one pre-built minute group per poll.

    Like the real live adapter it is clock-bounded — never ``exhausted`` — so only the
    session end (or a stop) ends the day; drained groups just poll as ``{}``.
    """

    def __init__(self, symbols: list[str], groups: list[dict[str, Bar]], degraded=None):
        self.symbols = symbols
        self._groups = list(groups)
        self._degraded_schedule = degraded
        self._i = 0
        self.poll_calls = 0
        self.flush_called = False
        self._degraded = False
        self.exhausted = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def poll_once(self) -> dict[str, Bar]:
        group = self._groups[self._i] if self._i < len(self._groups) else {}
        if self._degraded_schedule is not None:
            self._degraded = (
                self._degraded_schedule[self._i]
                if self._i < len(self._degraded_schedule)
                else False
            )
        self._i += 1
        self.poll_calls += 1
        return group

    def flush(self) -> dict[str, Bar]:
        self.flush_called = True
        return {}


class _StopAfter:
    """A stop flag that trips (True) only after ``k`` checks — so ``k`` polls run."""

    def __init__(self, k: int):
        self.k = k
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks > self.k


def _bar(row) -> Bar:
    return Bar(
        ts_event=int(row["ts_event"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def _groups_from_bars(df: pd.DataFrame, symbol: str) -> list[dict[str, Bar]]:
    return [{symbol: _bar(r)} for _, r in df.iterrows()]


def _uptrend(n: int = 120) -> pd.DataFrame:
    return make_ohlcv([100.0 + i * 0.5 for i in range(n)])


# --- 2) paced (clock-bounded) driver == batch (data-bounded) driver over the same data -----


def test_streaming_driver_matches_batch_driver():
    bars = _uptrend(120)
    candidates = [Candidate("sma_crossover", {"fast": 3, "slow": 8})]

    batch = run_trading(candidates=candidates, bars_by_symbol={"AAPL": bars})

    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    feed = ScriptedFeed(["AAPL"], _groups_from_bars(bars, "AAPL"))
    live = run_trading_day(
        SessionConfig(candidates=candidates),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=len(bars) + 5),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=1.0,
        record_bars=True,
    )

    assert live.summary.orders_submitted == batch.orders_submitted
    assert live.summary.fills == batch.fills
    assert live.summary.positions == batch.positions
    assert live.summary.final_equity == pytest.approx(batch.final_equity)
    # The streaming side retained the bars it built, for close reconciliation.
    assert set(live.live_bars) == {"AAPL"}
    assert len(live.live_bars["AAPL"]) == len(bars)


# --- 3) degraded halts emission, recovery resumes, event recorded -------------------------


def _run_stream(bars, *, degraded, poll=1.0):
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    feed = ScriptedFeed(["AAPL"], _groups_from_bars(bars, "AAPL"), degraded=degraded)
    return run_trading_day(
        SessionConfig(candidates=[Candidate("sma_crossover", {"fast": 3, "slow": 8})]),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=len(bars) + 5),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=poll,
    )


def test_degraded_halts_then_recovery_resumes_emission():
    bars = _uptrend(30)
    # A never-degraded baseline emits orders.
    clean = _run_stream(bars, degraded=[False] * 30)
    assert clean.summary.orders_submitted > 0

    # Degrade the first half; observation continues but emission halts, then recovers.
    half = _run_stream(bars, degraded=[True] * 15 + [False] * 15)
    assert half.summary.halted_for_degraded == 15  # every degraded bar observed, none traded
    assert half.summary.bars_processed == 30  # observation never stopped
    assert half.summary.orders_submitted > 0  # emission resumed after recovery
    # The summary carries the transitions for the close report.
    assert any("halted" in e for e in half.summary.events)
    assert any("recovered" in e.lower() for e in half.summary.events)


def test_all_degraded_emits_no_orders():
    bars = _uptrend(30)
    result = _run_stream(bars, degraded=[True] * 30)
    assert result.summary.orders_submitted == 0
    assert result.summary.fills == 0
    assert result.summary.halted_for_degraded == 30


# --- 4) stop request breaks between polls -------------------------------------------------


def test_stop_breaks_between_polls():
    bars = _uptrend(50)
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    feed = ScriptedFeed(["AAPL"], _groups_from_bars(bars, "AAPL"))
    stop = _StopAfter(5)

    result = run_trading_day(
        SessionConfig(candidates=[Candidate("sma_crossover", {"fast": 3, "slow": 8})]),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=1000),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=1.0,
        stop_event=stop,
    )

    # Broke after exactly 5 polls — far short of the 50-bar session (between polls).
    assert result.summary.polls == 5
    assert feed.poll_calls == 5
    # Exited cleanly with finalized state and did NOT act on a partial bar after the stop.
    assert feed.flush_called is False
    assert isinstance(result.summary.positions, dict)
    assert result.summary.final_equity > 0


# --- runtime-level helpers ----------------------------------------------------------------


def _make_settings(tmp_path, *, provider: str, poll_interval_s: float = 5000.0):
    lake_dir = tmp_path / "lake"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL, MSFT]\n"
        "session:\n  calendar: XNYS\n  timezone: America/New_York\n"
        "research_time_budget_minutes: 60\n"
        "champion_count: 3\n"
        "time_limit_hours: 12\n"
        "data:\n"
        f"  provider: {provider}\n"
        f"  lake_dir: {lake_dir}\n"
        "  dataset: EQUS.MINI\n"
        "live_feed:\n"
        f"  poll_interval_s: {poll_interval_s}\n"
        f"state_dir: {tmp_path}/state/\n"
    )
    return load_settings(config_path=cfg), lake_dir


def _seed_catalog(lake_dir):
    lake = MarketDataLake(lake_dir, MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL", "MSFT"], to_ns("2026-01-01"), to_ns("2026-12-31")
    )
    return lake


def _seed_champion(registry, family="sma_crossover", params=None):
    params = params or {"fast": 3, "slow": 8}
    sc = Scorecard(family=family, params=params)
    registry.champions.append(
        ChampionEntry(
            family=family, params=params, scorecard=sc, crowned_at="2026-01-01", rationale="seed"
        )
    )
    registry.save()


class _RecordingFeedFactory:
    """Builds a ScriptedFeed for the runtime and records how often it was called."""

    def __init__(self, groups):
        self.calls = 0
        self.feeds: list[ScriptedFeed] = []
        self._groups = groups

    def __call__(self, *, symbols):
        self.calls += 1
        groups = [{s: g[s] for s in symbols if s in g} for g in self._groups]
        feed = ScriptedFeed(list(symbols), groups)
        self.feeds.append(feed)
        return feed


def _run_one_cycle(settings, lake, *, feed_factory, tmp_path):
    registry = build_registry(settings)
    _seed_champion(registry)
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=registry,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=3,
        feed_factory=feed_factory,
        sleeper_factory=lambda start: SimulatedSleeper(start),
    )
    result = runtime.run(start=datetime(2027, 1, 4, 6, 0, tzinfo=ET))
    return runtime, result


# --- 6) provider gating: default builds no feed; yfinance builds + polls -------------------


def test_default_provider_builds_no_feed(tmp_path):
    settings, lake_dir = _make_settings(tmp_path, provider="databento")
    lake = _seed_catalog(lake_dir)
    factory = _RecordingFeedFactory([])
    _runtime, result = _run_one_cycle(settings, lake, feed_factory=factory, tmp_path=tmp_path)

    assert Phase.TRADING in result.history
    assert factory.calls == 0  # a bare (non-yfinance) run never touches the feed


def test_yfinance_provider_builds_and_polls_feed(tmp_path):
    settings, lake_dir = _make_settings(tmp_path, provider="yfinance")
    lake = _seed_catalog(lake_dir)
    groups = _groups_from_bars(_uptrend(4), "AAPL")
    factory = _RecordingFeedFactory(groups)
    _runtime, result = _run_one_cycle(settings, lake, feed_factory=factory, tmp_path=tmp_path)

    assert Phase.TRADING in result.history
    assert factory.calls >= 1  # opt-in provider builds the live feed
    assert sum(f.poll_calls for f in factory.feeds) > 0  # and actually polled it


# --- 5) live reconciliation flags a seeded divergence (was a self-compare no-op) -----------


def test_reconcile_compares_live_bars_against_catalog(tmp_path):
    settings, lake_dir = _make_settings(tmp_path, provider="yfinance")
    lake = _seed_catalog(lake_dir)
    registry = build_registry(settings)
    _seed_champion(registry)
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=registry,
        reports_dir=str(tmp_path / "reports"),
    )
    catalog = lake.get_bars("EQUS.MINI", "ohlcv-1m", ["AAPL"], 0, 2**63 - 1)["AAPL"]
    assert len(catalog) > 0

    # Live bars identical to the catalog → no drift, not flagged.
    runtime._live_bars = {"AAPL": catalog.copy()}
    clean = runtime._reconcile()
    assert clean.n_compared == len(catalog)
    assert clean.flagged is False

    # Seed a >0.5% divergence on one overlapping bar → flagged (real live-vs-catalog compare;
    # the old self-compare could never flag because both sides came from the catalog).
    diverged = catalog.copy()
    diverged.loc[diverged.index[0], "close"] = float(diverged.iloc[0]["close"]) * 1.10
    runtime._live_bars = {"AAPL": diverged}
    flagged = runtime._reconcile()
    assert flagged.flagged is True
    assert flagged.max_drift > flagged.threshold


# --- 7) paper-only holds even with a live feed --------------------------------------------


def test_paper_only_gate_untouched_with_live_feed():
    settings = load_settings(mode="paper", allow_live=False, data={"provider": "yfinance"})
    # The execution gate is entirely unaffected by opting into the live *data* feed.
    assert settings.data.provider == "yfinance"
    assert resolve_execution_mode(settings) == "paper"
    with pytest.raises(LiveBrokerUnavailableError):
        LiveBroker(settings)


def test_live_orders_route_through_simulated_exchange():
    bars = _uptrend(60)
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    feed = ScriptedFeed(["AAPL"], _groups_from_bars(bars, "AAPL"))
    broker = PaperBroker()  # the SimulatedExchange — the only functional execution path
    result = run_trading_day(
        SessionConfig(
            candidates=[Candidate("sma_crossover", {"fast": 3, "slow": 8})], broker=broker
        ),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=len(bars) + 5),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=1.0,
    )
    assert result.summary.orders_submitted > 0
    assert len(broker.fills) == result.summary.fills > 0  # every fill lives in the paper broker
