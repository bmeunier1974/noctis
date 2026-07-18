"""Shared harness for the TRADING-phase replay tests (session slicing + account
continuity): synthetic exchange-local minute bars, a minimal in-memory lake/registry,
a tmp-path runtime factory, and a driver for one TRADING entry through the phase's
public interface (``runtime.trading.run`` → :class:`~noctis.engine.TradingOutcome`)."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from noctis.config import load_settings
from noctis.data.types import NS_PER_SECOND, empty_bars
from noctis.engine import build_runtime
from noctis.engine.sessions import session_date
from noctis.memory import MemoryStore

ET = ZoneInfo("America/New_York")

TRADE_T = datetime(2026, 3, 9, 10, 0, tzinfo=ET)  # an open Monday; value unused by replay


def _bars_local(day: date, closes, start=(9, 30)) -> pd.DataFrame:
    """Minute bars starting at ``start`` exchange-local time on ``day`` (open == close)."""
    t0 = pd.Timestamp(datetime.combine(day, time(*start), tzinfo=ET)).value
    ts = [t0 + i * 60 * NS_PER_SECOND for i in range(len(closes))]
    c = [float(x) for x in closes]
    return pd.DataFrame(
        {
            "ts_event": ts,
            "open": c,
            "high": [x + 0.5 for x in c],
            "low": [x - 0.5 for x in c],
            "close": c,
            "volume": [1000] * len(c),
        }
    )


def _concat(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _uptrend(n=30, start=100.0, step=2.0):
    return [start + i * step for i in range(n)]


class _FakeLake:
    """In-memory lake: no coverage registry → the trading roster is the config universe."""

    def __init__(self, bars: dict[str, pd.DataFrame]):
        self.bars = bars

    def check_symbol_ready(self, symbol, dataset=None, schema=None) -> bool:
        return symbol in self.bars and len(self.bars[symbol]) > 0

    def get_bars(self, dataset, schema, symbols, start, end):
        return {s: self.bars.get(s, empty_bars()) for s in symbols}


class _FakeEntry:
    family = "sma_crossover"
    params = {"fast": 3, "slow": 8}
    live_symbols = None
    test_metric = 1.0


class _FakeRegistry:
    def list(self):
        return [_FakeEntry()]


def _make_runtime(tmp_path, lake, universe=("AAPL",)):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        f"universe: [{', '.join(universe)}]\n"
        f"state_dir: {tmp_path}/state/\n"
        f"strategies_dir: {tmp_path}/strategies/\n"  # empty → in-package seed families only
    )
    settings = load_settings(config_path=cfg)
    return build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=_FakeRegistry(),
        reports_dir=str(tmp_path / "reports"),
    )


def _run_phase(runtime, bars=None, *, t=TRADE_T, sleeper=None):
    """One TRADING entry through the phase's public interface — the runtime's own
    :class:`~noctis.engine.TradingPhase` over its startup bars. Pass ``bars`` to model the
    entry-time refresh (e.g. a lake that grew since the runtime was built)."""
    bars = runtime.trading_bars if bars is None else bars
    return runtime.trading.run(t, sleeper, bars)


def _ledger_path(runtime) -> Path:
    return Path(runtime.settings.state_dir) / "trading_sessions.json"


def _account_path(runtime) -> Path:
    return Path(runtime.settings.state_dir) / "paper_account.json"


def _traded_dates(record) -> set[date]:
    """The session dates present in one :class:`SessionRecord`'s replay slice."""
    out: set[date] = set()
    for df in record.bars.values():
        out.update(session_date(int(ts), ET) for ts in df["ts_event"])
    return out
