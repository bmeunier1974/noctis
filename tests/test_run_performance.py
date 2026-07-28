"""The run's realised record: trades, the equity curve, performance and the benchmark (#142).

Story #142 publishes the profitability picture — and the whole design problem is keeping it
*separate* from the backtest numbers that decide promotions. So this file pins the boundaries as
hard as it pins the numbers:

* trades gain a timestamp, fees, slippage and **champion attribution**, and every existing per-day
  report stays byte-identical because the new fields default to absent;
* the daily equity mark is appended to the run's own account ledger at each CLOSE, and the curve
  the record publishes is **re-derived from that ledger at every write** — never carried in memory,
  so three nights of segments produce exactly the curve one long night would;
* the ``performance`` block is the *paper account's* realised record and says so in its own
  ``source`` field, while backtest scorecards stay under ``strategies[]``;
* the benchmark is computed from bars **already in the shared lake** and degrades to nulls with a
  note rather than fetching anything;
* a run that never traded reports ``traded: false`` and ``performance: null`` — not zeros.

``tests/test_metrics.py`` holds the hand-computed fixtures for the metrics themselves.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from noctis.broker.persistence import EQUITY_CURVE_NAME, AccountStore, EquityLedger
from noctis.reporting import schema
from noctis.reporting.metrics import Benchmark, DailySession, TradeFill
from noctis.reporting.report import ReportData, Trade, render_report, write_report_json
from noctis.reporting.run_record import (
    EngineIdentity,
    RunArtifacts,
    SegmentArtifact,
    build,
)
from noctis.reporting.run_store import RUN_RECORD_NAME, open_run, read_benchmark, read_sessions

from ._session_helpers import _bars_local, _FakeLake, _make_runtime, _run_phase, _uptrend

START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)
HOUR = 3600.0

ENGINE = EngineIdentity(
    engine_version=1,
    fingerprint={"gates": "f63d47b7b9604ab1", "backtest": "3ba3e0bf1c97134f"},
    comparable_key="1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe",
    noctis_version="0.1.0",
)


class FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


# ── 1. trade enrichment, and the report bytes that must not move ───────────────────────────


def test_a_trade_carries_its_timestamp_fees_slippage_and_champion():
    trade = Trade(
        "NVDA",
        "BUY",
        12,
        118.4012,
        "champion signal",
        ts="2026-07-28T14:31:00.000Z",
        fees=0.14,
        slippage_bps=1.0,
        champion="vol_breakout_v3",
    )

    assert trade.ts == "2026-07-28T14:31:00.000Z"
    assert trade.fees == 0.14
    assert trade.slippage_bps == 1.0
    assert trade.champion == "vol_breakout_v3"


def test_an_unenriched_trade_serialises_to_exactly_the_fields_it_always_had():
    """Report stability, stated as the contract it is: the four new fields are **absent** — not
    null — from a report that carries none, so every per-day JSON already on disk is reproduced
    byte for byte."""
    data = ReportData(as_of="2026-07-03", trades=[Trade("AAPL", "buy", 10, 190.5, "breakout")])

    entry = json.loads(json.dumps(_trade_dicts(data)))[0]

    assert entry == {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 190.5,
        "rationale": "breakout",
    }


def test_an_enriched_trade_serialises_the_fields_it_was_given():
    data = ReportData(
        as_of="2026-07-03",
        trades=[
            Trade(
                "AAPL",
                "buy",
                10,
                190.5,
                "breakout",
                ts="2026-07-03T14:31:00.000Z",
                fees=0.19,
                slippage_bps=1.0,
                champion="momo_1",
            )
        ],
    )

    entry = _trade_dicts(data)[0]

    assert entry["ts"] == "2026-07-03T14:31:00.000Z"
    assert entry["fees"] == 0.19
    assert entry["slippage_bps"] == 1.0
    assert entry["champion"] == "momo_1"


def test_the_rendered_markdown_report_is_unchanged_by_the_enrichment(tmp_path):
    """The Markdown report renders exactly the five columns it always did, for an enriched trade
    as well as a plain one — an operator's per-day report is not this story's to change."""
    plain = ReportData(as_of="2026-07-03", trades=[Trade("AAPL", "BUY", 10, 190.5, "signal")])
    enriched = ReportData(
        as_of="2026-07-03",
        trades=[
            Trade(
                "AAPL",
                "BUY",
                10,
                190.5,
                "signal",
                ts="2026-07-03T14:31:00.000Z",
                fees=0.19,
                champion="momo_1",
            )
        ],
    )

    assert render_report(plain) == render_report(enriched)
    assert "| AAPL | BUY | 10 | 190.5000 | signal |" in render_report(plain)


def test_a_written_report_json_for_unenriched_trades_is_byte_identical(tmp_path):
    data = ReportData(
        as_of="2026-07-03",
        start_equity=100_000.0,
        end_equity=101_000.0,
        trades=[Trade("AAPL", "buy", 10, 190.5, "breakout")],
        positions={"AAPL": 10},
    )

    written = write_report_json(data, tmp_path).read_text(encoding="utf-8")

    assert (
        '  "trades": [\n'
        "    {\n"
        '      "price": 190.5,\n'
        '      "quantity": 10,\n'
        '      "rationale": "breakout",\n'
        '      "side": "buy",\n'
        '      "symbol": "AAPL"\n'
        "    }\n"
        "  ]" in written
    )
    assert '"slippage_bps"' not in written and '"champion"' not in written and '"ts"' not in written


def _trade_dicts(data: ReportData) -> list[dict]:
    from noctis.reporting.report import report_dict

    return report_dict(data)["trades"]


# ── 2. the daily equity mark, and the ledger it is appended to ─────────────────────────────


def test_the_ledger_appends_one_mark_per_session_and_reads_them_back_in_order(tmp_path):
    ledger = EquityLedger(tmp_path / EQUITY_CURVE_NAME)

    ledger.mark(date="2026-07-28", equity=100_412.33)
    ledger.mark(date="2026-07-29", equity=100_980.0, realized_pnl=567.67)

    assert [(m["date"], m["equity"]) for m in ledger.marks()] == [
        ("2026-07-28", 100_412.33),
        ("2026-07-29", 100_980.0),
    ]
    assert ledger.marks()[1]["realized_pnl"] == 567.67


def test_remarking_a_date_supersedes_it_rather_than_doubling_the_curve(tmp_path):
    """CLOSE can run twice for one date (a re-run, a crash, a catch-up). The curve is one mark per
    session date, last write wins — so a rewrite is idempotent, exactly like the record itself."""
    ledger = EquityLedger(tmp_path / EQUITY_CURVE_NAME)

    ledger.mark(date="2026-07-28", equity=100_000.0)
    ledger.mark(date="2026-07-28", equity=100_412.33)

    assert [(m["date"], m["equity"]) for m in ledger.marks()] == [("2026-07-28", 100_412.33)]


def test_an_unreadable_line_costs_that_mark_and_nothing_else(tmp_path):
    """Evidence, never a gate: a truncated append (a kill mid-write) loses one day, not the run."""
    path = tmp_path / EQUITY_CURVE_NAME
    ledger = EquityLedger(path)
    ledger.mark(date="2026-07-28", equity=100.0)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"date": "2026-07-29", "equ\n')
    ledger.mark(date="2026-07-30", equity=102.0)

    assert [m["date"] for m in ledger.marks()] == ["2026-07-28", "2026-07-30"]


def test_a_ledger_that_was_never_written_has_no_curve(tmp_path):
    assert EquityLedger(tmp_path / EQUITY_CURVE_NAME).marks() == []


def test_the_close_phase_appends_the_accounts_mark_with_the_sessions_trades(tmp_path):
    """The mark is taken from the **account store** — the cumulative paper account — not from the
    session's own start/end equity, so the curve is the account's and survives a restart by
    construction."""
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    runtime = _make_runtime(tmp_path, lake)
    runtime._run_trading(datetime(2026, 3, 9, 14, 30, tzinfo=UTC), None)

    runtime._run_close(datetime(2026, 3, 9, 21, 0, tzinfo=UTC))

    marks = EquityLedger(Path(runtime.settings.state_dir) / EQUITY_CURVE_NAME).marks()
    account = AccountStore(Path(runtime.settings.state_dir) / "paper_account.json").summary()
    assert [m["date"] for m in marks] == ["2026-03-09"]
    assert marks[0]["equity"] == pytest.approx(account.equity)
    assert marks[0]["trades"], "the session's fills ride with the mark they belong to"
    assert marks[0]["trades"][0]["symbol"] == "AAPL"


def test_a_close_with_no_account_yet_appends_nothing(tmp_path):
    """A research-only night has no account to mark, and an invented 100k flat line would be a
    claim about trading that never happened."""
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    runtime = _make_runtime(tmp_path, lake)

    runtime._run_close(datetime(2026, 3, 9, 21, 0, tzinfo=UTC))

    assert EquityLedger(Path(runtime.settings.state_dir) / EQUITY_CURVE_NAME).marks() == []


# ── 3. champion attribution, end to end through a real TRADING phase ───────────────────────


def test_trades_from_a_real_session_carry_their_stamp_fees_slippage_and_champion(tmp_path):
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    runtime = _make_runtime(tmp_path, lake)

    outcome = _run_phase(runtime)

    assert outcome.trades, "the fixture is meant to trade"
    trade = outcome.trades[0]
    assert trade.champion == "sma_crossover"  # the fixture registry's one champion
    assert trade.ts is not None and trade.ts.endswith("Z")
    assert trade.fees is not None and trade.fees > 0
    assert trade.slippage_bps == runtime.settings.backtest.slippage_bps


# ── 4. the record: sessions, performance, and the rule that they may be null ───────────────


def _session(
    day: str, equity: float, *, fills: tuple[TradeFill, ...] = (), **overrides
) -> DailySession:
    return DailySession(date=day, equity=equity, fills=fills, **overrides)


def _fill(symbol="NVDA", side="BUY", qty=10.0, price=100.0, champion="momo_1") -> TradeFill:
    return TradeFill(
        ts="2026-07-28T14:31:00.000Z",
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        fees_usd=0.1,
        slippage_bps=1.0,
        champion=champion,
        rationale="champion signal",
    )


SESSIONS = (
    _session("2026-07-28", 100_000.0, fills=(_fill(),), orders_submitted=1),
    _session(
        "2026-07-29",
        101_000.0,
        fills=(_fill(side="SELL", price=110.0),),
        orders_submitted=1,
        positions_end={},
        realized_pnl=1000.0,
    ),
    _session("2026-07-30", 100_500.0),
)


def _segment(index: int, *, trades: int, resumed: bool = False) -> SegmentArtifact:
    return SegmentArtifact(
        index=index,
        started_utc="2026-07-27T14:22:33.418Z",
        stopped_utc="2026-07-27T18:22:33.418Z",
        stopped_reason="time_limit",
        status="stopped",
        resumed=resumed,
        counters={"cycles": 1, "trades": trades},
        phase_seconds={"RESEARCH": 3600.0, "TRADING": 600.0},
        engine=ENGINE,
    )


def _artifacts(**overrides) -> RunArtifacts:
    base = dict(
        run_id="20260727T142233Z-a1b2c3",
        created_utc="2026-07-27T14:22:33.418Z",
        last_active_utc="2026-07-30T02:10:04.002Z",
        engine=ENGINE,
        complete=True,
        segments=(_segment(0, trades=2),),
        sessions=SESSIONS,
        trials=1000,
    )
    base.update(overrides)
    return RunArtifacts(**base)  # type: ignore[arg-type]


def test_the_record_carries_one_entry_per_traded_session_with_its_trade_log():
    record = build(_artifacts())

    assert [s["as_of"] for s in record["sessions"]] == ["2026-07-28", "2026-07-29", "2026-07-30"]
    first = record["sessions"][0]
    assert first["equity"] == 100_000.0
    assert first["orders_submitted"] == 1
    assert first["trades"][0]["symbol"] == "NVDA"
    assert first["trades"][0]["champion"] == "momo_1"
    assert first["trades"][0]["fees_usd"] == 0.1
    assert first["trades"][0]["slippage_bps"] == 1.0
    assert schema.validate(record) == []


def test_the_performance_block_is_the_paper_accounts_own_record_and_says_so():
    """Criterion: backtest/scorecard numbers and realised numbers are separate, distinctly-named
    sections. The realised one names itself, and carries no scorecard field at all."""
    record = build(_artifacts())

    performance = record["performance"]
    assert performance["source"] == "paper_account"
    assert performance["equity_curve"][0] == {"date": "2026-07-28", "equity": 100000.0}
    assert performance["account"]["end_equity"] == 100_500.0
    assert "scorecard" not in json.dumps(performance)
    assert performance["risk_adjusted"]["n_trials_used"] == 1000


def test_the_deflated_sharpe_is_published_beside_the_runs_cumulative_trial_count():
    """The number this project is uniquely able to compute: N is the run's own journaled trial
    count (``run.cumulative_trials``), so the deflation is auditable from the record alone."""
    record = build(_artifacts())

    risk = record["performance"]["risk_adjusted"]
    assert risk["n_trials_used"] == record["run"]["cumulative_trials"] == 1000
    assert risk["deflated_sharpe"] is not None
    assert risk["deflated_sharpe"] < risk["psr"]


def test_a_run_that_never_traded_reports_traded_false_and_null_performance():
    record = build(_artifacts(segments=(_segment(0, trades=0),), sessions=()))

    assert record["run"]["traded"] is False
    assert record["performance"] is None
    assert record["sessions"] == []
    assert schema.validate(record) == []


def test_a_run_whose_sessions_traded_is_traded_even_without_a_segment_counter():
    """``traded`` is derived from every piece of evidence the record carries, not latched by
    whichever one was written first (story #137's rule, one source wider)."""
    record = build(_artifacts(segments=(_segment(0, trades=0),)))

    assert record["run"]["traded"] is True
    assert record["performance"] is not None


def test_the_trade_log_is_capped_honestly():
    many = tuple(
        _session(f"2026-07-{day:02d}", 100_000.0, fills=tuple(_fill() for _ in range(40)))
        for day in range(1, 32)
    )

    record = build(_artifacts(sessions=many))

    kept = sum(len(session["trades"]) for session in record["sessions"])
    assert kept == schema.TRADE_CAP if 31 * 40 > schema.TRADE_CAP else kept == 31 * 40
    assert record["run"]["truncated"] == {} or "trades" in record["run"]["truncated"]


def test_the_performance_block_carries_the_benchmark_it_was_handed():
    bench = Benchmark(
        symbols=("NVDA",),
        points=(("2026-07-28", 100.0), ("2026-07-29", 101.0), ("2026-07-30", 99.0)),
    )

    record = build(_artifacts(benchmark=bench))

    benchmark = record["performance"]["benchmark"]
    assert benchmark["name"] == "equal_weight_universe_bh"
    assert benchmark["symbols"] == ["NVDA"]
    assert benchmark["beta"] is not None
    assert benchmark["note"] is None


def test_no_bar_series_or_holdout_preview_ever_reaches_the_record():
    """AGENTS.md rule 3, at the record's edge: holdout *metrics* may appear, holdout bars and
    anything a bar could be reconstructed from may not. The benchmark contributes statistics
    only — its own price series never lands in the document."""
    bench = Benchmark(
        symbols=("NVDA",),
        points=(("2026-07-28", 100.0), ("2026-07-29", 101.0), ("2026-07-30", 99.0)),
    )

    serialized = json.dumps(build(_artifacts(benchmark=bench)))

    for forbidden in ("points", "bars", "ts_event", "open", "high", "low", "close", "volume"):
        assert f'"{forbidden}"' not in serialized, forbidden


def test_the_record_is_deterministic_over_the_same_sessions():
    assert build(_artifacts()) == build(_artifacts())


# ── 5. the store: the curve is re-derived, and the benchmark comes out of the lake ─────────


def _open(tmp_path, clock, **kwargs):
    return open_run(
        tmp_path / "runs",
        clock=clock,
        argv=("run",),
        election_metric="sharpe",
        **kwargs,
    )


def _state_dir(run_dir: Path) -> Path:
    from noctis.config.settings import run_scoped_paths

    return Path(run_scoped_paths(run_dir)["state_dir"])


def _journal_sessions(run_dir: Path, marks: list[tuple[str, float]], *, trades=True) -> None:
    ledger = EquityLedger(_state_dir(run_dir) / EQUITY_CURVE_NAME)
    for day, equity in marks:
        ledger.mark(
            date=day,
            equity=equity,
            orders_submitted=1 if trades else 0,
            trades=[
                {
                    "ts": f"{day}T14:31:00.000Z",
                    "symbol": "NVDA",
                    "side": "BUY",
                    "quantity": 1,
                    "price": 100.0,
                    "fees": 0.01,
                    "slippage_bps": 1.0,
                    "champion": "momo_1",
                }
            ]
            if trades
            else [],
        )


def test_the_store_reads_the_curve_back_off_the_runs_own_ledger(tmp_path):
    clock = FakeClock()
    store = _open(tmp_path, clock)
    _journal_sessions(store.run_dir, [("2026-07-28", 100_000.0), ("2026-07-29", 101_000.0)])

    sessions = read_sessions(store.run_dir)

    assert [(s.date, s.equity) for s in sessions] == [
        ("2026-07-28", 100_000.0),
        ("2026-07-29", 101_000.0),
    ]
    assert sessions[0].fills[0].champion == "momo_1"


def test_a_run_with_no_ledger_reads_back_no_sessions(tmp_path):
    store = _open(tmp_path, FakeClock())

    assert read_sessions(store.run_dir) == ()


def test_the_written_record_derives_its_curve_from_the_ledger_at_every_write(tmp_path):
    clock = FakeClock()
    store = _open(tmp_path, clock)
    _journal_sessions(store.run_dir, [("2026-07-28", 100_000.0), ("2026-07-29", 101_000.0)])
    clock.advance(HOUR)
    store.close(reason="time_limit", counters={"cycles": 1, "trades": 2})

    record = json.loads((store.run_dir / RUN_RECORD_NAME).read_text())

    assert [p["equity"] for p in record["performance"]["equity_curve"]] == [100_000.0, 101_000.0]
    assert record["run"]["traded"] is True
    assert schema.validate(record) == []


def test_the_curve_survives_a_restart_because_it_is_never_carried_in_memory(tmp_path):
    """The resume rule (epic D4) applied to the curve: a second segment that marks two more days
    publishes all four, without anything having been kept across the process boundary."""
    clock = FakeClock()
    first = _open(tmp_path, clock)
    _journal_sessions(first.run_dir, [("2026-07-28", 100_000.0), ("2026-07-29", 101_000.0)])
    clock.advance(HOUR)
    first.close(reason="time_limit", counters={"trades": 1})

    second = _open(tmp_path, clock, run_id=first.run_id, resume=True)
    _journal_sessions(second.run_dir, [("2026-07-30", 99_000.0), ("2026-07-31", 102_000.0)])
    clock.advance(HOUR)
    second.close(reason="time_limit", counters={"trades": 1})

    record = json.loads((second.run_dir / RUN_RECORD_NAME).read_text())
    assert [p["date"] for p in record["performance"]["equity_curve"]] == [
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
    ]


# ── the benchmark, out of the shared lake ──────────────────────────────────────────────────


def _lake_bars(lake_dir: Path, symbol: str, days: list[tuple[str, float]]) -> None:
    """Write one symbol's daily bars into the shared lake, exactly where the engine reads them."""
    rows = []
    for day, close in days:
        ts = int(pd.Timestamp(f"{day} 20:00:00", tz="UTC").value)
        rows.append(
            {
                "ts_event": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100,
            }
        )
    frame = pd.DataFrame(rows)
    path = lake_dir / "EQUS.MINI" / "ohlcv-1m" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _inputs(lake_dir: Path) -> dict:
    return {
        "settings": {"resolved": {}},
        "data": {"provider": "databento", "dataset": "EQUS.MINI", "lake_dir": str(lake_dir)},
    }


DAYS = ["2026-07-28", "2026-07-29", "2026-07-30"]


def _traded_sessions() -> tuple[DailySession, ...]:
    return tuple(_session(day, 100_000.0 + i * 500, fills=(_fill(),)) for i, day in enumerate(DAYS))


def test_the_benchmark_is_equal_weight_buy_and_hold_over_the_names_the_run_traded(tmp_path):
    """Two names, both held from the first session: NVDA +20%, MSFT −10% ⇒ the equal-weight
    level ends at (1.20 + 0.90)/2 = 1.05, a +5% benchmark. No vendor call, no new data."""
    lake = tmp_path / "data_lake"
    _lake_bars(lake, "NVDA", list(zip(DAYS, [100.0, 110.0, 120.0], strict=True)))
    _lake_bars(lake, "MSFT", list(zip(DAYS, [200.0, 190.0, 180.0], strict=True)))
    sessions = tuple(
        _session(day, 100_000.0, fills=(_fill(symbol="NVDA"), _fill(symbol="MSFT"))) for day in DAYS
    )

    bench = read_benchmark(sessions, _inputs(lake))

    assert bench.symbols == ("MSFT", "NVDA")
    assert bench.points[0][1] == pytest.approx(1.0)
    assert bench.points[-1][1] == pytest.approx(1.05)
    assert bench.note is None


def test_a_symbol_the_lake_does_not_hold_leaves_the_benchmark_null_with_a_note(tmp_path):
    """No new vendor spend, ever. A run whose names are not in the lake is simply not
    benchmarked, and the record says why rather than fetching."""
    lake = tmp_path / "data_lake"
    lake.mkdir()

    bench = read_benchmark(_traded_sessions(), _inputs(lake))

    assert bench.points == ()
    assert bench.note is not None and "lake" in bench.note
    block = build(_artifacts(sessions=_traded_sessions(), benchmark=bench))["performance"]
    assert block["benchmark"]["total_return_pct"] is None
    assert block["benchmark"]["note"] == bench.note


def test_a_run_that_traded_nothing_asks_the_lake_for_nothing(tmp_path):
    bench = read_benchmark((), _inputs(tmp_path / "data_lake"))

    assert bench.symbols == () and bench.points == ()


def test_the_benchmark_never_leaves_the_runs_own_session_window(tmp_path):
    """The window is the run's sessions — bars before the account opened or after its last mark
    are never read, so the comparison is against the same days the strategy actually traded."""
    lake = tmp_path / "data_lake"
    _lake_bars(
        lake,
        "NVDA",
        [("2026-07-01", 50.0), *zip(DAYS, [100.0, 110.0, 120.0], strict=True), ("2026-08-30", 5.0)],
    )

    bench = read_benchmark(_traded_sessions(), _inputs(lake))

    assert [day for day, _level in bench.points] == DAYS


def test_the_record_write_reads_the_benchmark_out_of_the_shared_lake(tmp_path):
    """End to end through the store: the run's own trades name the roster, the shared lake prices
    it, and the record carries the comparison — with no vendor client anywhere in the path."""
    lake = tmp_path / "data_lake"
    _lake_bars(lake, "NVDA", list(zip(DAYS, [100.0, 110.0, 120.0], strict=True)))
    clock = FakeClock()
    store = _open(tmp_path, clock, inputs=_inputs(lake))
    _journal_sessions(
        store.run_dir, list(zip(DAYS, [100_000.0, 101_000.0, 102_000.0], strict=True))
    )
    clock.advance(HOUR)
    store.close(reason="time_limit", counters={"trades": 3})

    record = json.loads((store.run_dir / RUN_RECORD_NAME).read_text())

    benchmark = record["performance"]["benchmark"]
    assert benchmark["symbols"] == ["NVDA"]
    assert benchmark["total_return_pct"] == pytest.approx(20.0)
    assert benchmark["correlation"] is not None
    # The fixture freezes only the ``data`` block a benchmark needs, so the inputs section is
    # deliberately partial; everything this story writes is still schema-valid.
    assert [problem for problem in schema.validate(record) if "inputs" not in problem] == []
