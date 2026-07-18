"""P4 — the inline TRADING feed: per-decision ``trade``/``refuse``, feed health, and heartbeat.

The two trading drivers gained an optional console sink (``on_event``) that tees what already
happens — a fill, a refusal, a feed transition, a poll pulse — as typed
:class:`~noctis.observability.events.Event`s, without touching the report or the quiet-replay
collapse. These tests pin that:

* one ``trade`` event per ACTUAL fill (the dead-band skip stays silent);
* the refusal feed collapses to one event per distinct reason, ``orders_refused`` unchanged;
* the heartbeat fires every ``heartbeat_polls`` polls and only at level 2;
* feed transitions reach BOTH the report (strings, byte-identical) and the console (``feed``);
* a bare run (``on_event=None``) constructs nothing and leaves the report untouched.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from noctis.config import load_settings
from noctis.engine import SimulatedSleeper, build_runtime
from noctis.live import RiskLimits, SessionConfig, run_trading, run_trading_day
from noctis.memory import MemoryStore
from noctis.observability import Console, Event
from noctis.strategies.base import Bar

from ._data_helpers import make_ohlcv
from ._session_helpers import _bars_local, _FakeLake, _FakeRegistry, _run_phase, _uptrend

# --- strategy + feed stubs ---------------------------------------------------------------


class _AlwaysLong:
    """Targets long every bar — maximum rebalance dust (and, on a crash, endless refusals)."""

    def on_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx, bar) -> None:
        ctx.set_target(1)


class _Candidate:
    def __init__(self, strat):
        self._strat = strat

    def build(self, families):
        return self._strat


class _ScriptedFeed:
    """A minimal fake live feed: hands the driver one pre-built minute group per poll."""

    def __init__(self, symbols, groups):
        self.symbols = list(symbols)
        self._groups = list(groups)
        self._i = 0
        self.degraded = False
        self.exhausted = False  # clock-bounded, like the live adapter it fakes

    def poll_once(self):
        group = self._groups[self._i] if self._i < len(self._groups) else {}
        self._i += 1
        return group

    def flush(self):
        return {}


def _bar(row) -> Bar:
    return Bar(
        ts_event=int(row["ts_event"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def _groups(df, symbol="AAPL"):
    return [{symbol: _bar(r)} for _, r in df.iterrows()]


def _kinds(events, kind):
    return [e for e in events if isinstance(e, Event) and e.kind == kind]


# --- 1) trade events: one per fill; the dead-band skip stays silent ----------------------


def test_one_trade_event_per_fill():
    # Every fill the batch driver records must surface exactly one `trade` event.
    bars = {"AAPL": make_ohlcv([100.0 + i * 0.5 for i in range(20)])}
    events: list = []
    summary = run_trading(
        candidates=[_Candidate(_AlwaysLong())], bars_by_symbol=bars, on_event=events.append
    )
    trades = _kinds(events, "trade")
    assert len(trades) == summary.fills > 0  # exactly one event per recorded fill
    assert trades[0].meta["side"] == "BUY"  # the entry from flat is a buy
    assert all(t.meta["side"] in ("BUY", "SELL") for t in trades)
    assert all(t.meta["symbol"] == "AAPL" and t.meta["qty"] > 0 for t in trades)
    assert trades[0].level == 2  # per-decision detail is the -vv firehose


def test_dead_band_skip_emits_no_trade_event():
    # With the band on, a held long re-trues only once (the entry); the suppressed sub-share
    # adjustments `continue` before the fill, so they emit nothing.
    bars = {"AAPL": make_ohlcv([100.0 + i * 0.5 for i in range(20)])}
    events: list = []
    summary = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        rebalance_band_pct=50.0,
        on_event=events.append,
    )
    assert summary.fills == 1  # just the entry; the held long then holds
    assert len(_kinds(events, "trade")) == 1  # and exactly one trade event, not per bar


# --- 2) refusal feed collapses to one per reason; the count stays honest ------------------


def _crash_limits():
    return RiskLimits(max_position_pct=50.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=3.0)


def test_refusal_feed_collapses_to_one_per_reason():
    # A long at a 50% cap, then a price crash: equity falls through the 3% floor, the daily-loss
    # latch trips, and the notional cap keeps re-desiring an exposure increase that is refused
    # every low bar. The inline feed must collapse that per-bar flood to one `refuse` event per
    # distinct reason — the event-side of the quiet-replay collapse.
    bars = {"AAPL": make_ohlcv([100.0, 100.0, 60.0, 60.0, 60.0, 60.0])}
    events: list = []
    summary = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        limits=_crash_limits(),
        on_event=events.append,
    )
    refusals = _kinds(events, "refuse")
    assert summary.orders_refused >= 2  # refused every low bar (the honest full count)
    assert refusals  # ...but the feed shows at least one
    reasons = {e.meta["reason"] for e in refusals}
    assert len(refusals) == len(reasons)  # one event per DISTINCT reason, never per bar
    assert summary.orders_refused > len(refusals)  # many refusals collapsed to few events
    assert any("daily loss" in r for r in reasons)


def test_refusal_accounting_is_byte_identical_with_and_without_the_feed():
    # Threading the inline feed must not move `orders_refused` (a quiet-replay regression guard).
    bars = {"AAPL": make_ohlcv([100.0, 100.0, 60.0, 60.0, 60.0, 60.0])}
    quiet = run_trading(
        candidates=[_Candidate(_AlwaysLong())], bars_by_symbol=bars, limits=_crash_limits()
    )
    with_feed = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        limits=_crash_limits(),
        on_event=[].append,
    )
    assert with_feed.orders_refused == quiet.orders_refused
    assert with_feed.fills == quiet.fills
    assert with_feed.halt_latched == quiet.halt_latched


# --- 3) trade/refuse are level-2 (gated out at -v) ---------------------------------------


def test_trade_and_refuse_are_level_two():
    bars = {"AAPL": make_ohlcv([100.0, 100.0, 60.0, 60.0, 60.0, 60.0])}
    out_v: list[str] = []
    run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        limits=_crash_limits(),
        on_event=Console(1, sink=out_v.append, color=False),  # -v
    )
    assert not any("BUY" in ln or "refuse" in ln.lower() or "daily loss" in ln for ln in out_v)

    out_vv: list[str] = []
    run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        limits=_crash_limits(),
        on_event=Console(2, sink=out_vv.append, color=False),  # -vv
    )
    assert any("BUY" in ln for ln in out_vv)
    assert any("daily loss" in ln for ln in out_vv)


# --- 4) heartbeat: every N polls, level-2 only -------------------------------------------


def _run_live(*, on_event=None, heartbeat_polls=0, n_polls=12, poll_s=1.0):
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    feed = _ScriptedFeed(["AAPL"], _groups(make_ohlcv([100.0 + i * 0.5 for i in range(8)])))
    return run_trading_day(
        SessionConfig(
            candidates=[_Candidate(_AlwaysLong())],
            on_event=on_event,
            heartbeat_polls=heartbeat_polls,
        ),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=n_polls),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=poll_s,
    )


def test_heartbeat_fires_every_n_polls():
    events: list = []
    result = _run_live(on_event=events.append, heartbeat_polls=3)
    beats = _kinds(events, "heartbeat")
    assert result.summary.polls > 0
    assert len(beats) == result.summary.polls // 3  # one pulse every 3 polls, no more
    assert beats[0].meta["polls"] == 3 and "equity" in beats[0].text
    assert all(b.level == 2 for b in beats)


def test_heartbeat_disabled_by_zero():
    events: list = []
    _run_live(on_event=events.append, heartbeat_polls=0)
    assert _kinds(events, "heartbeat") == []


def test_heartbeat_hidden_at_level_one():
    out: list[str] = []
    _run_live(on_event=Console(1, sink=out.append, color=False), heartbeat_polls=2)
    assert not any("poll " in ln and "equity" in ln for ln in out)  # level-2, gated at -v
    out2: list[str] = []
    _run_live(on_event=Console(2, sink=out2.append, color=False), heartbeat_polls=2)
    assert any("poll " in ln and "equity" in ln for ln in out2)


# --- 5) feed transitions tee to BOTH report and console; report byte-identical ------------


class _DegradingFeed(_ScriptedFeed):
    """A feed whose degraded flag follows a per-poll schedule (drives the health transitions)."""

    def __init__(self, symbols, groups, schedule):
        super().__init__(symbols, groups)
        self._schedule = list(schedule)

    def poll_once(self):
        idx = self._i
        group = super().poll_once()
        self.degraded = self._schedule[idx] if idx < len(self._schedule) else False
        return group


def _run_degrading(*, on_event=None):
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    df = make_ohlcv([100.0 + i * 0.5 for i in range(6)])
    feed = _DegradingFeed(["AAPL"], _groups(df), schedule=[True] * 3 + [False] * 3)
    return run_trading_day(
        SessionConfig(candidates=[_Candidate(_AlwaysLong())], on_event=on_event),
        feed,
        session_start=start,
        session_end=start + timedelta(seconds=len(df) + 2),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=1.0,
    )


def test_feed_transitions_reach_both_report_and_console():
    console: list = []
    result = _run_degrading(on_event=console.append)
    # The summary keeps the exact transition strings the report kept before...
    assert any("halted" in s for s in result.summary.events)
    assert any("recovered" in s.lower() for s in result.summary.events)
    # ...and the console additionally sees them as level-1 `feed` events with the same text.
    feed_events = _kinds(console, "feed")
    assert any("halted" in e.text for e in feed_events)
    assert any("recovered" in e.text.lower() for e in feed_events)
    assert all(e.level == 1 for e in feed_events)  # feed health shows at -v


def test_report_is_byte_identical_without_a_console():
    with_console = _run_degrading(on_event=[].append)
    bare = _run_degrading(on_event=None)
    # The console never changes what the summary hands the report.
    assert bare.summary.events == with_console.summary.events


# --- 6) Runtime replay narration: a per-session banner; the report stays report-only ------


def _replay_runtime(tmp_path, *, on_event=None):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL]\n"
        f"state_dir: {tmp_path}/state/\n"
        f"strategies_dir: {tmp_path}/strategies/\n"
    )
    settings = load_settings(config_path=cfg)
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    return build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=_FakeRegistry(),
        reports_dir=str(tmp_path / "reports"),
        on_event=on_event,
    )


def test_replay_emits_a_phase_banner_per_session(tmp_path):
    # The catch-up replay narrates each session inline instead of one batch INFO.
    events: list = []
    runtime = _replay_runtime(tmp_path, on_event=events.append)
    _run_phase(runtime)
    banners = _kinds(events, "phase")
    assert len(banners) == 1  # exactly one session in the fake lake
    assert "TRADING replay" in banners[0].text and "2026-03-09" in banners[0].text
    assert banners[0].level == 1  # a session banner shows at -v


def test_replay_keeps_per_decision_events_out_of_the_report(tmp_path):
    # The champion trades the uptrend, so `trade` events reach the console — but the outcome's
    # report lines (what the CLOSE report renders) stay strings-only.
    events: list = []
    runtime = _replay_runtime(tmp_path, on_event=events.append)
    outcome = _run_phase(runtime)
    assert _kinds(events, "trade")  # the feed actually had decisions to show
    assert not any(isinstance(e, Event) for e in outcome.events)  # report is report-only
