"""End-to-end smoke test: one (and two) full night→day→close cycles on cached data.

No network, mocked vendor, injectable time. Proves the whole machine runs unattended and
stops cleanly at the configured time limit / on request, with all state flushed.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from noctis.champions import ChampionRegistry, build_registry
from noctis.config import load_settings
from noctis.data import MarketDataLake
from noctis.data.types import to_ns
from noctis.engine import Phase, SimulatedSleeper, build_runtime
from noctis.memory import MemoryStore

from ._data_helpers import MockVendor

ET = ZoneInfo("America/New_York")


def _make_settings(tmp_path, *, time_limit_hours):
    lake_dir = tmp_path / "lake"
    cfg = tmp_path / "config.yaml"
    tl = "null" if time_limit_hours is None else str(time_limit_hours)
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL, MSFT]\n"
        "session:\n  calendar: XNYS\n  timezone: America/New_York\n"
        "research_time_budget_minutes: 60\n"
        "champion_count: 3\n"
        f"time_limit_hours: {tl}\n"
        "data:\n"
        f"  lake_dir: {lake_dir}\n"
        "  dataset: EQUS.MINI\n"
        f"state_dir: {tmp_path}/state/\n"
    )
    settings = load_settings(config_path=cfg)
    return settings, lake_dir


def _seed_catalog(lake_dir, start="2026-01-01", end="2026-12-31"):
    vendor = MockVendor()
    lake = MarketDataLake(lake_dir, vendor, budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage("EQUS.MINI", "ohlcv-1m", ["AAPL", "MSFT"], to_ns(start), to_ns(end))
    return lake, vendor


def test_full_cycle_stops_cleanly_at_time_limit(tmp_path):
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=12)
    lake, vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")
    registry = build_registry(settings)
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        registry=registry,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=6,
        sleeper_factory=lambda start: SimulatedSleeper(start),
    )
    assert runtime.has_data()

    # Coverage last_ts before the run, per symbol → later prove sync never re-buys it.
    cov_before = {r.symbol: r.last_ts for r in lake.coverage.all()}
    vendor.fetch_ranges.clear()

    # Start pre-open on a 2027 Monday so the catalog (2026) is fully covered history.
    start = datetime(2027, 1, 4, 6, 0, tzinfo=ET)
    result = runtime.run(start=start)

    # 1) The machine visited the full cycle and stopped on the time limit.
    assert Phase.RESEARCH in result.history
    assert Phase.TRADING in result.history
    assert Phase.CLOSE in result.history
    assert result.stopped_reason == "time_limit"
    assert result.cycles_completed >= 1

    # 2) Research actually proposed and evaluated candidates.
    assert result.research_iterations >= 6

    # 3) A report was written and is readable.
    assert result.reports
    report_text = (tmp_path / "reports").glob("*.md")
    assert any("Close-of-day report" in p.read_text() for p in report_text)

    # 4) Fetch-once held: every fetch during the run was a tail (never re-bought coverage).
    for s, _e in vendor.fetch_ranges:
        assert s > min(cov_before.values())

    # 5) State is flushed and reloadable in a fresh process (restart survival).
    reloaded = ChampionRegistry(tmp_path / "state" / "champions.json", capacity=3)
    assert len(reloaded.list()) == len(registry.list())
    assert len(registry.list()) <= settings.champion_count
    assert "## Champions" in memory.read()


def test_two_cycles_are_cumulative(tmp_path):
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=None)
    lake, _vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")
    registry = build_registry(settings)
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        registry=registry,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=5,
        sleeper_factory=lambda start: SimulatedSleeper(start),
    )
    start = datetime(2027, 1, 4, 6, 0, tzinfo=ET)
    result = runtime.run(start=start, max_cycles=2)

    assert result.stopped_reason == "max_cycles"
    assert result.cycles_completed == 2
    # Two close phases → the registry accumulated across both nights (never reset).
    assert result.history.count(Phase.CLOSE) == 2
    assert result.research_iterations >= 10  # 5 per research visit, at least two visits
    # Memory kept its four sections after repeated reorganization.
    text = memory.read()
    for header in ("## Champions", "## Learnings", "## Rejected ideas", "## Index / changelog"):
        assert header in text


def test_runtime_research_panel_and_symbol_holdout_are_fixed(tmp_path):
    """The fit set is the first fit_set_size ready symbols, the holdout the next
    symbol_holdout_size — fixed at startup and identical for every candidate."""
    from dataclasses import replace

    from noctis.strategies import Candidate

    lake_dir = tmp_path / "lake"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL, MSFT, NVDA, JPM]\n"
        "research:\n  fit_set_size: 3\n  symbol_holdout_size: 1\n"
        "data:\n"
        f"  lake_dir: {lake_dir}\n"
        "  dataset: EQUS.MINI\n"
        f"state_dir: {tmp_path}/state/\n"
    )
    settings = load_settings(config_path=cfg)
    vendor = MockVendor()
    lake = MarketDataLake(lake_dir, vendor, budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI",
        "ohlcv-1m",
        ["AAPL", "MSFT", "NVDA", "JPM"],
        to_ns("2026-01-01"),
        to_ns("2026-12-31"),
    )
    memory = MemoryStore(tmp_path / "MEMORY.md")
    runtime = build_runtime(
        settings, market_lake=lake, memory=memory, reports_dir=str(tmp_path / "reports")
    )

    assert list(runtime.research_panel) == ["AAPL", "MSFT", "NVDA"]
    assert list(runtime.symbol_holdout) == ["JPM"]
    assert runtime.has_data()

    # Every candidate is evaluated on the same panel and the same held-out symbols.
    runtime._pipeline_config = replace(runtime._pipeline_config, prefilter_min_score=None)
    for cand in (
        Candidate("sma_crossover", {"fast": 3, "slow": 8}),
        Candidate("donchian_breakout", {"channel": 15}),
    ):
        sc = runtime._evaluate(cand)
        assert sc.stage == "validated"
        assert set(sc.symbols) == {"AAPL", "MSFT", "NVDA"}
        assert sc.symbol_holdout_metric is not None  # JPM was scored, one causal pass


class _RecordingSleeper:
    """A simulated sleeper that records every ``sleep_until`` target, so a test can prove the
    loop does not wait out a closed market when the run should already have stopped."""

    def __init__(self, start):
        self._t = start
        self.waits: list = []

    def now(self):
        return self._t

    def sleep_until(self, t):
        self.waits.append(t)
        if t > self._t:
            self._t = t


def test_expired_time_limit_does_not_wait_out_closed_market(tmp_path):
    """A run started while the market is closed with an already-elapsed time limit stops at
    once instead of pacing to the next open (the RealSleeper weekend hang: the machine's
    time-up check runs only after the wait, so an unbounded wait would sleep out the weekend
    before the limit could take effect)."""
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=0)
    lake, _vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")

    captured: dict = {}

    def _factory(start):
        captured["sleeper"] = _RecordingSleeper(start)
        return captured["sleeper"]

    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=1,
        sleeper_factory=_factory,
    )
    # 2027-01-02 is a Saturday — the market is closed, so an unbounded loop would pace to the
    # next open (Monday) before honoring the elapsed limit.
    start = datetime(2027, 1, 2, 6, 0, tzinfo=ET)
    result = runtime.run(start=start)

    assert result.stopped_reason == "time_limit"
    assert captured["sleeper"].waits == []  # never paced toward the next open


class _WallClockSleeper(SimulatedSleeper):
    """A controllable clock that *reports* as real-time pacing. Lets a test drive the
    real-time branch (fill a closed market with back-to-back research) deterministically,
    without ever calling ``time.sleep``."""

    wall_clock = True


def _stub_phases(runtime, captured, *, session_minutes=20):
    """Replace the three phase bodies with counters. Research consumes wall-clock time (as a
    real session does), so under real-time pacing the loop advances toward the open through
    successive sessions; trading/close just tally."""
    calls = {"research": 0, "trading": 0, "close": 0}

    def _research():
        calls["research"] += 1
        captured["sleeper"].advance(session_minutes * 60)

    def _trading(t, sleeper):
        calls["trading"] += 1

    def _close(t):
        calls["close"] += 1
        runtime.result.cycles_completed += 1

    runtime._run_research = _research
    runtime._run_trading = _trading
    runtime._run_close = _close
    return calls


def test_closed_market_fills_with_back_to_back_research_under_real_time(tmp_path):
    """Under real-time pacing the ~3.5h pre-open gap is filled with many research sessions —
    the loop never idles the closed market away, it just skips trading until the open."""
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=None)
    lake, _vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")

    captured: dict = {}

    def _factory(start):
        captured["sleeper"] = _WallClockSleeper(start)
        return captured["sleeper"]

    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=1,
        sleeper_factory=_factory,
    )
    calls = _stub_phases(runtime, captured)

    start = datetime(2027, 1, 4, 6, 0, tzinfo=ET)  # pre-open Monday
    result = runtime.run(start=start, max_cycles=1)

    assert calls["research"] >= 5  # back-to-back, not a single session + long wait
    assert calls["trading"] == 1  # crossed into the open and traded
    assert calls["close"] == 1
    assert result.stopped_reason == "max_cycles"


def test_closed_market_jumps_to_open_under_simulated_clock(tmp_path):
    """Contrast: under a simulated clock (no real closed time to fill) the same setup runs
    research exactly once, then jumps straight to the open. Proves the wall_clock flag is
    what flips the behavior."""
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=None)
    lake, _vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")

    captured: dict = {}

    def _factory(start):
        captured["sleeper"] = SimulatedSleeper(start)
        return captured["sleeper"]

    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=1,
        sleeper_factory=_factory,
    )
    calls = _stub_phases(runtime, captured)

    start = datetime(2027, 1, 4, 6, 0, tzinfo=ET)  # pre-open Monday
    result = runtime.run(start=start, max_cycles=1)

    assert calls["research"] == 1  # one session, then jumped to the open
    assert calls["trading"] == 1
    assert result.stopped_reason == "max_cycles"


class _RecordingWallClock(SimulatedSleeper):
    """A controllable clock that *reports* as real-time pacing and records every
    ``sleep_until`` target, with a hook fired after each wait so a test can trip a stop
    mid-wait exactly as a SIGINT handler would."""

    wall_clock = True

    def __init__(self, start):
        super().__init__(start)
        self.waits: list = []
        self.on_wait = lambda: None

    def sleep_until(self, t):
        self.waits.append(t)
        super().sleep_until(t)
        self.on_wait()


def test_expired_time_limit_does_not_wait_out_the_session_close(tmp_path):
    """After a completed trading day the loop paces to the session close — but never past
    the run's time limit. The RESEARCH branch already bounds its waits; an unbounded
    pace-to-close would park a short `--time-limit-hours` run against the clock for the
    rest of the session (observed live: a replay day settled in under a minute, then the
    process sat in the wait-for-close for hours until SIGTERM)."""
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=0.05)  # 3 minutes
    lake, _vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")

    captured: dict = {}

    def _factory(start):
        captured["sleeper"] = _RecordingWallClock(start)
        return captured["sleeper"]

    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=1,
        sleeper_factory=_factory,
    )
    _stub_phases(runtime, captured)  # trading completes instantly, like a settled replay day

    # Mid-session Monday: the close (16:00 ET) is ~6h away, the limit only 3 minutes.
    start = datetime(2027, 1, 4, 10, 0, tzinfo=ET)
    result = runtime.run(start=start)

    assert result.stopped_reason == "time_limit"
    from datetime import timedelta

    deadline = start + timedelta(hours=0.05)
    # The run ends at (a poll chunk past) the deadline — never slept out to the close.
    assert captured["sleeper"].now() <= deadline + timedelta(minutes=1)


def test_stop_request_halts_cleanly(tmp_path):
    settings, lake_dir = _make_settings(tmp_path, time_limit_hours=None)
    lake, _vendor = _seed_catalog(lake_dir)
    memory = MemoryStore(tmp_path / "MEMORY.md")
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        reports_dir=str(tmp_path / "reports"),
        sleeper_factory=lambda start: SimulatedSleeper(start),
    )
    runtime.request_stop()  # as a SIGTERM handler would
    start = datetime(2027, 1, 4, 6, 0, tzinfo=ET)
    result = runtime.run(start=start)
    assert result.stopped_reason == "stop_requested"
    assert result.history[-1] is Phase.STOPPED
