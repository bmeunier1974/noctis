"""Tests for the Typer CLI skeleton."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.research.surface import SessionCounters

runner = CliRunner()

# A fixed "now" so the auto-backfill window is deterministic across runs (fetch-once).
FROZEN_NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


def _install_mock_vendor(monkeypatch):
    """Give ``run`` a fake DataBento vendor — no network. ``bootstrap.build_lake`` imports the
    client at call time, so patching the module attribute takes effect. Returns the shared
    instance so tests can assert its cost/fetch call counts."""
    from ._data_helpers import MockVendor

    vendor = MockVendor()
    monkeypatch.setattr(
        "noctis.data.databento_provider.DataBentoVendorClient",
        lambda *args, **kwargs: vendor,
    )
    return vendor


def _backfill_config(tmp_path, *, auto_backfill=False, budget=None):
    """A paper config with a two-symbol universe pointed at a tmp_path lake.

    ``research_time_budget_minutes: 0`` keeps the runtime's RESEARCH phase a no-op: with
    ``--time-limit-hours 0`` exactly one phase runs before the machine stops, and when the
    market is closed at test time that phase is RESEARCH — an unbounded-iteration loop that
    would otherwise spin for the default 60-minute budget (a time-of-day-dependent hang).
    """
    lake_dir = tmp_path / "lake"
    lines = [
        "mode: paper",
        "universe: [AAPL, MSFT]",
        "research_time_budget_minutes: 0",
        "data:",
        f"  lake_dir: {lake_dir}",
        "  dataset: EQUS.MINI",
    ]
    if auto_backfill:
        lines.append("  auto_backfill: true")
    if budget is not None:
        lines.append(f"  budget_usd: {budget}")
    lines.append(f"state_dir: {tmp_path}/state/")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("\n".join(lines) + "\n")
    return str(cfg)


def _bare_config(tmp_path, mode: str) -> str:
    """A minimal config whose lake/state point at tmp_path.

    Isolating ``lake_dir`` matters: with the default ``data_lake/`` a developer machine
    holding an ingested lake would make ``run`` enter the real (unbounded) trading loop,
    while an empty tmp lake early-returns on the no-data path everywhere, like CI.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: {mode}\ndata:\n  lake_dir: {tmp_path}/lake\nstate_dir: {tmp_path}/state/\n"
    )
    return str(cfg)


def _paper_config(tmp_path):
    return _bare_config(tmp_path, "paper")


def _live_config(tmp_path):
    return _bare_config(tmp_path, "live")


def test_run_paper_exits_zero_and_prints_mode(tmp_path):
    result = runner.invoke(app, ["run", "--config", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "PAPER mode" in result.output


def test_run_live_without_gate_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE", raising=False)
    result = runner.invoke(app, ["run", "--config", _live_config(tmp_path)])
    assert result.exit_code != 0
    assert "SAFETY GATE" in result.output


def test_research_live_without_gate_exits_nonzero(tmp_path, monkeypatch):
    """No verb is a silent downgrade (#247): ``research`` arms the same gate ``run`` does, so
    ``mode: live`` without ``ALLOW_LIVE`` refuses at startup with the same line."""
    monkeypatch.delenv("ALLOW_LIVE", raising=False)
    result = runner.invoke(app, ["research", "--config", _live_config(tmp_path)])
    assert result.exit_code != 0
    assert "SAFETY GATE" in result.output


def test_run_live_with_gate_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_LIVE", "true")
    result = runner.invoke(app, ["run", "--config", _live_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "LIVE mode" in result.output


def test_status_reports_resolved_mode(tmp_path):
    result = runner.invoke(app, ["status", "--config", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "mode (resolved):   paper" in result.output
    assert "account:           none yet" in result.output
    # Plan 4: the resolved driver is stated (default provider databento → replay).
    assert "trading driver:    replay (execution=auto)" in result.output
    # Plan 5: no forward record until a champion trades a live-holdout session.
    assert "forward record:    none yet" in result.output


def test_status_shows_per_champion_forward_record(tmp_path):
    from datetime import date

    from noctis.engine import ForwardLedger

    fl = ForwardLedger(Path(tmp_path) / "state" / "forward_ledger.json")
    fl.record("sma_crossover@x", "sma_crossover", date(2026, 7, 6), {"AAPL": 123.45})
    fl.save()
    result = runner.invoke(app, ["status", "--config", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "forward record:" in result.output
    assert "sma_crossover" in result.output
    assert "+123.45" in result.output


def test_status_forward_record_corrupt_ledger_is_graceful(tmp_path):
    state = Path(tmp_path) / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "forward_ledger.json").write_text("{ corrupt")
    result = runner.invoke(app, ["status", "--config", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "forward record:    unreadable ledger" in result.output


def test_status_trading_driver_reflects_execution(tmp_path):
    # A forced replay under data.provider: yfinance must still read as replay.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n  provider: yfinance\n"
        f"state_dir: {tmp_path}/state/\ntrading:\n  execution: replay\n"
    )
    result = runner.invoke(app, ["status", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "trading driver:    replay (execution=replay)" in result.output


def test_report_sweep_stale_dry_run_lists_without_moving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The reports of the reserved run every unaddressed command reads (story #131).
    reports = tmp_path / "workspace" / "runs" / "legacy" / "reports"
    reports.mkdir(parents=True)
    (reports / "2099-01-01.md").write_text("future")
    (reports / "2020-01-01.md").write_text("past")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")
    result = runner.invoke(app, ["report", "--config", str(cfg), "--sweep-stale"])
    assert result.exit_code == 0, result.output
    assert "2099-01-01.md" in result.output
    assert "2020-01-01.md" not in result.output
    assert (reports / "2099-01-01.md").is_file()  # dry-run: not moved
    assert not (reports / "archive").exists()


def test_report_sweep_stale_apply_moves_future_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # The reports of the reserved run every unaddressed command reads (story #131).
    reports = tmp_path / "workspace" / "runs" / "legacy" / "reports"
    reports.mkdir(parents=True)
    (reports / "2099-01-01.md").write_text("future")
    (reports / "2020-01-01.md").write_text("past")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")
    result = runner.invoke(app, ["report", "--config", str(cfg), "--sweep-stale", "--no-dry-run"])
    assert result.exit_code == 0, result.output
    assert not (reports / "2099-01-01.md").exists()
    assert (reports / "archive" / "2099-01-01.md").is_file()
    assert (reports / "2020-01-01.md").is_file()  # past untouched


def _seed_account(tmp_path, cash=101_000.0):
    """Persist a continuous paper account under tmp_path's state dir."""
    from datetime import date
    from pathlib import Path

    from noctis.broker.persistence import AccountStore

    store = AccountStore(Path(tmp_path) / "state" / "paper_account.json")
    broker = store.load()
    broker.cash = cash
    store.save(broker, date(2026, 7, 6))
    return store.path


def test_account_command_shows_the_continuous_account(tmp_path):
    cfg = _paper_config(tmp_path)
    result = runner.invoke(app, ["account", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "No paper account yet" in result.output

    _seed_account(tmp_path)
    result = runner.invoke(app, ["account", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "opened:           2026-07-06" in result.output
    assert "equity:           101,000.00" in result.output
    assert "cumulative P&L:   +1,000.00" in result.output

    status = runner.invoke(app, ["status", "--config", cfg])
    assert "account:           equity 101,000.00 (+1,000.00 since 2026-07-06" in status.output


def test_account_reset_archives_and_corrupt_file_recovers(tmp_path):
    cfg = _paper_config(tmp_path)
    path = _seed_account(tmp_path)
    path.write_text("{corrupt")  # a torn write

    result = runner.invoke(app, ["account", "--config", cfg])
    assert result.exit_code != 0
    assert "corrupt paper account" in result.output

    result = runner.invoke(app, ["account", "--reset", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "Archived to" in result.output
    assert not path.exists()
    assert list(path.parent.glob("paper_account.*.json"))  # evidence archived, not deleted

    result = runner.invoke(app, ["account", "--reset", "--config", cfg])
    assert "No paper account to reset." in result.output


def test_research_metric_flag_validates_before_anything_else(tmp_path):
    cfg = _paper_config(tmp_path)
    result = runner.invoke(app, ["research", "--metric", "nonsense", "-c", cfg])
    assert result.exit_code != 0
    assert "sharpe" in result.output and "total_return" in result.output
    # A valid metric proceeds to the next requirement (no key/extra in the test env). The message
    # is provider-neutral now (#10): it names the [llm] extra and the default provider's key.
    result = runner.invoke(app, ["research", "--metric", "total_return", "-c", cfg])
    assert result.exit_code != 0
    assert "[llm] extra" in result.output and "OPENAI_API_KEY" in result.output


def test_research_end_of_session_lists_undecided(tmp_path, monkeypatch):
    """#55: the one-shot ``research`` command surfaces the summary's undecided list in its
    end-of-session output (a session that assembled and ran is faked so no LLM is needed)."""
    from noctis.engine.research import ResearchSummary

    summary = ResearchSummary(
        iterations=4,
        promotions=0,
        rejections=1,
        stopped_reason="agent_done",
        candidates=["alpha", "beta"],
        undecided=["alpha", "beta"],
    )

    class _Budgets:
        name = "test-profile"
        max_iterations = 20

    class _Toolbox:
        def session_counters(self):
            return SessionCounters(backtests_run=3)

    class _Session:
        model = "fake/model"
        budgets = _Budgets()
        toolbox = _Toolbox()

        def run(self, *, max_iterations=None, stop_event=None):
            return summary

    monkeypatch.setattr("noctis.bootstrap.build_research_session", lambda **kwargs: _Session())
    result = runner.invoke(app, ["research", "-c", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "undecided" in result.output.lower()
    assert "alpha" in result.output and "beta" in result.output


def test_research_end_of_session_omits_undecided_when_empty(tmp_path, monkeypatch):
    """An empty undecided list adds no extra end-of-session line."""
    from noctis.engine.research import ResearchSummary

    summary = ResearchSummary(iterations=2, stopped_reason="agent_done")

    class _Budgets:
        name = "test-profile"
        max_iterations = 20

    class _Toolbox:
        def session_counters(self):
            return SessionCounters(backtests_run=1)

    class _Session:
        model = "fake/model"
        budgets = _Budgets()
        toolbox = _Toolbox()

        def run(self, *, max_iterations=None, stop_event=None):
            return summary

    monkeypatch.setattr("noctis.bootstrap.build_research_session", lambda **kwargs: _Session())
    result = runner.invoke(app, ["research", "-c", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "undecided" not in result.output.lower()


@pytest.mark.parametrize(
    "author_calls, expected",
    [
        (
            0,
            "Session over (agent_done): 4 tool rounds, 3 backtests, "
            "1 promotion(s), 2 rejection(s).",
        ),
        (
            2,
            "Session over (agent_done): 4 tool rounds, 3 backtests, 2 coder authoring call(s), "
            "1 promotion(s), 2 rejection(s).",
        ),
    ],
)
def test_research_session_over_line_is_filled_from_the_counters_snapshot(
    tmp_path, monkeypatch, author_calls, expected
):
    """#260: the operator's end-of-session grep target, byte-for-byte — and every counter in it
    comes from one ``session_counters()`` snapshot, the verb's only reach into the toolbox."""
    from noctis.engine.research import ResearchSummary

    summary = ResearchSummary(iterations=4, promotions=1, rejections=2, stopped_reason="agent_done")

    class _Budgets:
        name = "test-profile"
        max_iterations = 20

    class _Toolbox:
        def session_counters(self):
            return SessionCounters(backtests_run=3, author_calls=author_calls)

    class _Session:
        model = "fake/model"
        budgets = _Budgets()
        toolbox = _Toolbox()

        def run(self, *, max_iterations=None, stop_event=None):
            return summary

    monkeypatch.setattr("noctis.bootstrap.build_research_session", lambda **kwargs: _Session())
    result = runner.invoke(app, ["research", "-c", _paper_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert expected in result.output


def test_champions_command_runs(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")
    result = runner.invoke(app, ["champions", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "No champions yet" in result.output


def test_backtest_command_runs_on_catalog(tmp_path):
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns

    from ._data_helpers import MockVendor

    lake_dir = tmp_path / "lake"
    md = MarketDataLake(lake_dir, MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    md.ensure_coverage("EQUS.MINI", "ohlcv-1m", ["AAPL"], to_ns("2026-01-01"), to_ns("2026-12-31"))

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"data:\n  lake_dir: {lake_dir}\n  dataset: EQUS.MINI\n"
        f"state_dir: {tmp_path}/state/\nuniverse: [AAPL]\n"
    )
    result = runner.invoke(app, ["backtest", "sma_crossover", "--symbol", "AAPL", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "avg test metric" in result.output


def test_backtest_unknown_strategy_errors(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("universe: [AAPL]\n")
    result = runner.invoke(app, ["backtest", "nope", "-c", str(cfg)])
    assert result.exit_code != 0
    assert "Unknown strategy" in result.output


# --- run auto-backfill (opt-in) -------------------------------------------------------


def test_run_auto_backfill_off_makes_zero_fetches(tmp_path, monkeypatch):
    """Default (auto_backfill unset): an empty lake triggers ZERO vendor calls — no behavior
    change from before the feature. Proves the opt-in is truly off by default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")  # key present, yet nothing is fetched
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _backfill_config(tmp_path)  # auto_backfill left off

    result = runner.invoke(app, ["run", "--config", cfg])

    assert result.exit_code == 0, result.output
    assert vendor.fetch_calls == 0
    assert vendor.cost_calls == 0
    assert "Auto-backfilling" not in result.output
    assert "ingest history first" in result.output


@pytest.mark.parametrize(
    "frozen_now",
    [
        FROZEN_NOW,  # midday: the UTC and ET calendars agree
        # 11 PM EDT on Jul 3 — the UTC date is already Jul 4, the ET trading date is not.
        # A UTC-dated boundary here crosses the vendor's live-license line (403).
        datetime(2026, 7, 4, 3, 0, tzinfo=UTC),
    ],
    ids=["midday", "late-evening-et"],
)
def test_run_auto_backfill_on_fetches_and_enters_loop(tmp_path, monkeypatch, frozen_now):
    """auto_backfill: true on an empty lake fetches the universe, symbols become ready, and
    the run proceeds into the loop (stops immediately via --time-limit-hours 0)."""
    from noctis.data import MarketDataLake

    from ._data_helpers import MockVendor

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")
    monkeypatch.setattr("noctis.cli._utcnow", lambda: frozen_now)
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _backfill_config(tmp_path, auto_backfill=True)

    result = runner.invoke(app, ["run", "--config", cfg, "--time-limit-hours", "0"])

    assert result.exit_code == 0, result.output
    assert "Auto-backfilling 2 symbol(s)" in result.output
    assert vendor.fetch_calls == 2  # one slice per symbol on an empty lake
    # The window ends at the T+1 boundary — UTC midnight of the current *ET* trading date,
    # never wall-clock now and never the UTC date (which rolls over at 8 PM ET, a day past
    # the vendor's license line). Both frozen clocks are Jul 3 in ET, so both must land here:
    from noctis.data.types import day_start_ns

    boundary = day_start_ns(date(2026, 7, 3))
    assert all(fetch_end == boundary for (_, fetch_end) in vendor.fetch_ranges)
    # Coverage was created and both symbols are now ready.
    check = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=1.0, calendar="XNYS")
    assert check.check_symbol_ready("AAPL")
    assert check.check_symbol_ready("MSFT")
    # The run proceeded into the loop rather than bailing on missing data.
    assert "ingest history first" not in result.output
    assert "Stopped (" in result.output


def test_run_auto_backfill_is_fetch_once(tmp_path, monkeypatch):
    """Running twice with auto_backfill on: the second run's backfill is a $0 no-op (zero new
    fetch_bars calls) because coverage already spans the window."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")
    monkeypatch.setattr("noctis.cli._utcnow", lambda: FROZEN_NOW)
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _backfill_config(tmp_path, auto_backfill=True)

    first = runner.invoke(app, ["run", "--config", cfg, "--time-limit-hours", "0"])
    assert first.exit_code == 0, first.output
    assert vendor.fetch_calls == 2

    second = runner.invoke(app, ["run", "--config", cfg, "--time-limit-hours", "0"])
    assert second.exit_code == 0, second.output
    assert vendor.fetch_calls == 2  # unchanged — the window is already covered
    # Both symbols are already ready, so nothing is "missing" and the backfill isn't even
    # re-attempted — an even stronger fetch-once guarantee. The run still enters the loop.
    assert "Auto-backfilling" not in second.output
    assert "Stopped (" in second.output


def test_run_auto_backfill_over_budget_refuses_cleanly(tmp_path, monkeypatch):
    """auto_backfill on with a $0 budget: the preflight refuses, nothing is fetched, state is
    uncorrupted, and the run exits cleanly (surfacing the refusal) rather than crashing."""
    from noctis.data import MarketDataLake

    from ._data_helpers import MockVendor

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")
    monkeypatch.setattr("noctis.cli._utcnow", lambda: FROZEN_NOW)
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _backfill_config(tmp_path, auto_backfill=True, budget=0.0)

    result = runner.invoke(app, ["run", "--config", cfg])

    assert result.exit_code == 0, result.output
    assert "refused" in result.output
    assert vendor.fetch_calls == 0  # priced only; never fetched
    # State uncorrupted: nothing became ready; the run fell through to the no-data path.
    check = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=1.0, calendar="XNYS")
    assert not check.check_symbol_ready("AAPL")
    assert "ingest history first" in result.output


def test_run_auto_backfill_on_without_key_warns_and_skips(tmp_path, monkeypatch):
    """auto_backfill on but no DATABENTO_API_KEY: warn, skip the backfill, exit cleanly, and
    make zero fetches (the read-only vendor is used, the mock is never constructed)."""
    monkeypatch.chdir(tmp_path)
    # conftest clears DATABENTO_API_KEY, so it is absent here.
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _backfill_config(tmp_path, auto_backfill=True)

    result = runner.invoke(app, ["run", "--config", cfg])

    assert result.exit_code == 0, result.output
    assert "no DATABENTO_API_KEY" in result.output
    assert vendor.fetch_calls == 0
    assert "ingest history first" in result.output


def test_data_ingest_prints_per_symbol_progress(tmp_path, monkeypatch):
    """A multi-symbol ingest announces each symbol as it starts ('ingesting AAPL (1/2)…') —
    the non-TTY fallback of the interactive spinner — before the per-symbol result lines.
    Without it a long DataBento backfill is minutes of dead silence."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")
    _install_mock_vendor(monkeypatch)
    cfg = _backfill_config(tmp_path)

    result = runner.invoke(
        app,
        [
            "data",
            "ingest",
            "AAPL,MSFT",
            "--start",
            "2026-01-05",
            "--end",
            "2026-01-09",
            "--config",
            cfg,
        ],
    )

    assert result.exit_code == 0, result.output
    progress = result.output + result.stderr
    assert "ingesting AAPL (1/2)" in progress
    assert "ingesting MSFT (2/2)" in progress
    assert "AAPL: ingested" in result.output
    assert "MSFT: ingested" in result.output


# --- --debug: the QA recorder wiring on run and research (story #45) ------------------


def _debug_run_config(tmp_path, *, keep_last_runs: int | None = None) -> str:
    """A paper config with a ready two-symbol lake and a tmp QA area — so ``run`` enters the loop
    (and, with ``--time-limit-hours 0``, stops immediately) instead of the no-data early return."""
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns

    from ._data_helpers import MockVendor

    lake_dir = tmp_path / "lake"
    md = MarketDataLake(lake_dir, MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    md.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL", "MSFT"], to_ns("2026-01-01"), to_ns("2026-12-31")
    )
    lines = [
        "mode: paper",
        "universe: [AAPL, MSFT]",
        "research_time_budget_minutes: 0",
        "data:",
        f"  lake_dir: {lake_dir}",
        "  dataset: EQUS.MINI",
        f"state_dir: {tmp_path}/state/",
        f"qa_dir: {tmp_path}/qa",
    ]
    if keep_last_runs is not None:
        lines += ["qa:", f"  keep_last_runs: {keep_last_runs}"]
    cfg = tmp_path / "config.yaml"
    cfg.write_text("\n".join(lines) + "\n")
    return str(cfg)


def _one_qa_run(tmp_path) -> Path:
    """The single QA run folder minted under the tmp QA area."""
    from noctis.observability.runid import RUN_ID_RE

    qa = tmp_path / "qa"
    runs = [p for p in qa.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)]
    assert len(runs) == 1, [p.name for p in qa.iterdir()]
    return runs[0]


def _strip_qa_lines(output: str) -> str:
    """Drop the additive ``QA …``/``Run …`` framing lines so the event/console feed can be
    compared. Both carry the minted run id, which differs by construction between invocations."""
    return "".join(
        line + "\n"
        for line in output.splitlines()
        if not line.startswith(("QA ", "Run: ", "Run record: "))
    )


def test_run_debug_creates_report_tree_and_echoes_start_and_stop(tmp_path):
    import json

    cfg = _debug_run_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code == 0, result.output

    run_dir = _one_qa_run(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["stopped"] is not None  # closed cleanly via the finally
    assert manifest["duration_s"] is not None

    run_id = run_dir.name
    # start echo (run id + report path) AND stop echo (again + the funnel one-liner)
    assert result.output.count(run_id) >= 2
    assert result.output.count("QA report:") >= 2
    assert "QA funnel:" in result.output


def test_run_debug_without_v_records_silently(tmp_path):
    """--debug alone records but never turns on the -v console feed: the phase banners the loop
    emits reach the recorder's events.jsonl, not stdout."""
    import json

    cfg = _debug_run_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code == 0, result.output

    # no event feed on stdout (those are the -v console renderings, gated off here)
    assert "# RESEARCH" not in result.output
    assert "# STOPPED" not in result.output

    # but the events WERE recorded: the phase frames are in the run's events.jsonl
    run_dir = _one_qa_run(tmp_path)
    lines = (run_dir / "h00" / "events.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines if line.strip()]
    assert "phase" in kinds


def test_run_debug_v_output_is_byte_identical_to_v_alone(tmp_path):
    """Recording never perturbs the console: -v with --debug renders the same event feed as -v
    alone (the only difference is the additive QA framing lines)."""
    cfg = _debug_run_config(tmp_path)
    plain = runner.invoke(app, ["run", "--config", cfg, "-v", "--time-limit-hours", "0"])
    debug = runner.invoke(app, ["run", "--config", cfg, "-v", "--debug", "--time-limit-hours", "0"])
    assert plain.exit_code == 0 and debug.exit_code == 0, debug.output
    assert "QA report:" in debug.output  # the framing IS present under --debug
    assert "QA report:" not in plain.output
    # ...and stripped of that framing, the two feeds are byte-for-byte the same.
    assert _strip_qa_lines(debug.output) == _strip_qa_lines(plain.output)


def test_run_debug_prunes_qa_area_on_start(tmp_path):
    from noctis.observability.runid import RUN_ID_RE

    qa = tmp_path / "qa"
    qa.mkdir(parents=True)
    older = [f"2026010{i}T000000Z-00000{i}" for i in range(1, 6)]  # five old run folders
    for name in older:
        (qa / name).mkdir()

    cfg = _debug_run_config(tmp_path, keep_last_runs=2)
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code == 0, result.output

    remaining = {p.name for p in qa.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)}
    # the 2 newest old folders survive + this run's folder; the oldest 3 are pruned
    assert older[0] not in remaining and older[2] not in remaining
    assert older[3] in remaining and older[4] in remaining
    assert len(remaining) == 3


def test_run_debug_time_limit_leaves_readable_segment_and_stamped_manifest(tmp_path):
    """The time-limit interruption case: a clean between-phases stop still lands a readable final
    segment and a stamped manifest (the finally reaches the recorder's close)."""
    import json

    cfg = _debug_run_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code == 0, result.output

    run_dir = _one_qa_run(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["stopped"] is not None
    # the open segment was finalized on close → its counts document is on disk and readable
    assert (run_dir / "h00" / "counts.md").read_text().startswith("# QA counts")


def test_run_debug_hard_exception_still_stamps_manifest(tmp_path, monkeypatch):
    """A hard failure inside the run still closes the recorder (try/finally), so the manifest is
    stamped and no run tree is orphaned; recording is secondary, the error still propagates."""
    import json

    cfg = _debug_run_config(tmp_path)

    class _Boom:
        def request_stop(self):  # pragma: no cover - never reached in this path
            pass

        def run(self):
            raise RuntimeError("boom mid-run")

    monkeypatch.setattr("noctis.engine.build_runtime", lambda *a, **k: _Boom())
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code != 0  # the exception propagated

    run_dir = _one_qa_run(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["stopped"] is not None  # closed via the finally despite the crash


def test_run_debug_echoes_the_qa_frame_in_its_fixed_order(tmp_path):
    """The ``--debug`` transcript, pinned (story #250): the run's identity, the QA tree at start,
    the loop's own stop line, then the closing QA report + funnel — which print *after* the work
    on every path, including a crash, because they are what says the recording is complete."""
    cfg = _debug_run_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code == 0, result.output

    framing = [
        line.split(":")[0]
        for line in result.output.splitlines()
        if line.startswith(("Run: ", "Run record: ", "QA ", "Stopped ("))
    ]
    assert framing == [
        "Run",
        "Run record",
        "QA run",
        "QA report",
        "Stopped (time_limit)",
        "QA report",
        "QA funnel",
    ]


def test_a_recorder_that_refuses_to_build_still_releases_the_run_lock(tmp_path, monkeypatch):
    """The ``--debug`` recorder is assembled *inside* the guarded region (story #250): a qa_dir
    that refuses to take a recorder used to leave the run locked until the stale-lock timeout,
    because the builder ran after the lock was taken and outside the try that released it."""
    import json

    from noctis.reporting.run_tree import RUN_LOCK_NAME

    def _refuse(*args, **kwargs):
        raise OSError("qa_dir is unwritable")

    monkeypatch.setattr("noctis.bootstrap.build_recorder", _refuse)
    cfg = _debug_run_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg, "--debug", "--time-limit-hours", "0"])
    assert result.exit_code != 0  # the failure is not swallowed

    (run_dir,) = [p for p in (tmp_path / "workspace" / "runs").iterdir() if p.is_dir()]
    assert not (run_dir / RUN_LOCK_NAME).exists(), "the lock outlived the segment"
    record = json.loads((run_dir / "run.json").read_text())
    # Nothing was measured, so the segment closes at the sentinel — a closed, resumable run.
    assert record["segments"][-1]["stopped_reason"] == "startup"


def test_run_no_debug_writes_no_qa_tree(tmp_path):
    """The default (no --debug) is byte-identical to today: no recorder, no QA writes, no echoes."""
    cfg = _debug_run_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", cfg, "--time-limit-hours", "0"])
    assert result.exit_code == 0, result.output
    assert "QA report:" not in result.output
    qa = tmp_path / "qa"
    assert not qa.exists() or not any(qa.iterdir())


# --- research --debug -----------------------------------------------------------------


class _FakeSession:
    """A stand-in agent session: it emits a couple of events into the wired sink, then reports a
    summary — enough to drive the research command's echoes and the recorder's funnel."""

    def __init__(self, on_event):
        from types import SimpleNamespace

        from noctis.research.pricing import default_table

        self.model = "anthropic/claude-fake"
        self.price_table = default_table()
        self.budgets = SimpleNamespace(name="balanced", max_iterations=5)
        self.toolbox = SimpleNamespace(session_counters=lambda: SessionCounters(backtests_run=1))
        self._on_event = on_event

    def run(self, *, max_iterations=None):
        from noctis.engine import ResearchSummary
        from noctis.observability import Event

        if self._on_event is not None:
            self._on_event(Event("phase", "RESEARCH · cycle 0", meta={"phase": "RESEARCH"}))
            self._on_event(
                Event(
                    "tool",
                    "write_strategy(alpha) -> ok",
                    meta={"ok": True, "tool": "write_strategy", "args": {"name": "alpha"}},
                )
            )
        return ResearchSummary(
            iterations=1, promotions=0, rejections=0, stopped_reason="done", candidates=["alpha"]
        )


def _patch_research_agent(monkeypatch):
    """Make the research command believe an agent session is available and hand it the fake one,
    capturing the wired ``on_event`` so the fake can emit into it."""
    from noctis.research.llm import ClientStatus

    monkeypatch.setattr(
        "noctis.research.client_status",
        lambda settings: ClientStatus(
            ok=True, model="anthropic/claude-fake", provider="anthropic", reason=None
        ),
    )

    def fake_build(**kwargs):
        return _FakeSession(kwargs.get("on_event"))

    monkeypatch.setattr("noctis.bootstrap.build_research_session", fake_build)


def test_research_debug_records_and_echoes(tmp_path, monkeypatch):
    import json

    _patch_research_agent(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nqa_dir: {tmp_path}/qa\n")

    result = runner.invoke(app, ["research", "--config", str(cfg), "--debug"])
    assert result.exit_code == 0, result.output

    run_dir = _one_qa_run(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["stopped"] is not None
    # The session resolved the gate, so the QA manifest carries its verdict rather than a
    # null nobody measured (#247).
    assert manifest["mode"] == "paper"

    run_id = run_dir.name
    assert result.output.count(run_id) >= 2  # echoed at start and again at stop
    assert "QA report:" in result.output
    # the funnel one-liner reflects the recorded write_strategy event
    assert "QA funnel: written=1" in result.output
    # --debug without -v stays silent: the emitted feed events never hit stdout
    assert "write_strategy(alpha)" not in result.output


def test_research_debug_echoes_the_qa_frame_in_its_fixed_order(tmp_path, monkeypatch):
    """The research ``--debug`` transcript, pinned (#251): the run's identity, the QA tree at
    start, the session's own stop line, then the closing QA report + funnel — which print *after*
    the session on every path, including a crash, because they say the recording is complete."""
    _patch_research_agent(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nqa_dir: {tmp_path}/qa\n")

    result = runner.invoke(app, ["research", "--config", str(cfg), "--debug"])
    assert result.exit_code == 0, result.output

    framing = [
        line.split(":")[0]
        for line in result.output.splitlines()
        if line.startswith(("Run: ", "Run record: ", "QA ", "Session over ("))
    ]
    assert framing == [
        "Run",
        "Run record",
        "QA run",
        "QA report",
        "Session over (done)",
        "QA report",
        "QA funnel",
    ]


def test_a_research_recorder_that_refuses_to_build_still_releases_the_run_lock(
    tmp_path, monkeypatch
):
    """The ``--debug`` recorder is assembled inside the band's guarded region (#251), so a qa_dir
    that refuses to take one still closes the segment: nothing was measured, so it closes at the
    sentinel — a closed, resumable run rather than a lock held until the stale-lock timeout."""
    import json

    from noctis.reporting.run_tree import RUN_LOCK_NAME

    def _refuse(*args, **kwargs):
        raise OSError("qa_dir is unwritable")

    _patch_research_agent(monkeypatch)
    monkeypatch.setattr("noctis.bootstrap.build_recorder", _refuse)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nqa_dir: {tmp_path}/qa\n")

    result = runner.invoke(app, ["research", "--config", str(cfg), "--debug"])
    assert result.exit_code != 0  # the failure is not swallowed

    (run_dir,) = [p for p in (tmp_path / "workspace" / "runs").iterdir() if p.is_dir()]
    assert not (run_dir / RUN_LOCK_NAME).exists(), "the lock outlived the segment"
    segment = json.loads((run_dir / "run.json").read_text())["segments"][-1]
    assert segment["stopped_reason"] == "startup"
    assert segment["counters"] == {}  # measured nothing is never measured zero
    assert segment["phase_seconds"] is None


def test_research_says_when_a_reasoning_view_surfaced_no_reasoning(tmp_path, monkeypatch):
    """A reasoning view that came back empty is named as the provider's doing, once — so silence
    reads as "expected here", not "the feature is broken". The command duck-types ``saw_think``
    and ``hint`` off the sink the band built, so the hint survives the move (#251)."""
    _patch_research_agent(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")

    surfaced = runner.invoke(app, ["research", "--config", str(cfg)])
    asked = runner.invoke(app, ["research", "--config", str(cfg), "--show-reasoning"])

    assert asked.exit_code == 0, asked.output
    assert "reasoning not surfaced by anthropic" in asked.output
    assert "narration still shows" in asked.output
    # …and nobody who did not ask for a reasoning view is told about one.
    assert "reasoning not surfaced" not in surfaced.output


def test_the_reasoning_hint_still_reaches_an_operator_through_a_teed_sink(tmp_path, monkeypatch):
    """The command reads ``saw_think`` and calls ``hint`` on whatever sink the band built, with no
    existence check (#337): under ``--debug`` that sink is an ``EventTee``, which proxies both to
    its console primary — so a reasoning view that came back empty is still named once."""
    _patch_research_agent(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nqa_dir: {tmp_path}/qa\n")

    result = runner.invoke(app, ["research", "--config", str(cfg), "--show-reasoning", "--debug"])

    assert result.exit_code == 0, result.output
    assert "reasoning not surfaced by anthropic" in result.output


def test_a_research_minted_run_records_the_gates_verdict(tmp_path, monkeypatch):
    """Rule 1's verdict is measured on every verb that mints a run (#247): a run born from a
    research session says its orders were paper-only, instead of "nobody measured"."""
    import json

    _patch_research_agent(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"mode: paper\nstate_dir: {tmp_path}/state/\n")

    result = runner.invoke(app, ["research", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    (run_dir,) = [p for p in (tmp_path / "workspace" / "runs").iterdir() if p.is_dir()]
    record = json.loads((run_dir / "run.json").read_text())
    assert record["inputs"]["execution_mode"] == "paper"
    assert record["assumptions"]["paper_only"] is True


def _cli_tree():
    """``noctis/cli.py`` parsed — the source these structural tests read."""
    import ast

    import noctis.cli

    source = Path(noctis.cli.__file__)
    return ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def _command_body(name: str):
    """One Typer command's function definition, as written in ``noctis/cli.py``."""
    import ast

    (command,) = [
        node for node in _cli_tree().body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return command


def test_neither_command_opens_its_own_run_store():
    """D3 (#248), finished by #251: both verbs open their run through the band, so
    ``open_run_store`` — and every unpacked ``SessionInputs`` field it used to be handed by hand
    — is named nowhere in ``cli.py``. A new resume-policy tier is a one-file change in
    ``bootstrap.py`` instead of an edit to two hand-walked call sites."""
    import ast

    calls = [
        node
        for node in ast.walk(_cli_tree())
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open_run_store"
    ]

    assert calls == [], "the band opens the run, for `run` and `research` alike"


@pytest.mark.parametrize("verb", ["run", "research"])
def test_neither_command_owns_a_segment_lifecycle_of_its_own(verb):
    """Stories #250 / #251: the open → work → close lifecycle lives in the band, so neither
    command builds a store, a recorder or an event sink, closes anything by hand, or carries a
    ``"startup"`` sentinel — each reports what stopped its segment and lets the band close it."""
    import ast

    command = _command_body(verb)

    built = {
        node.func.id
        for node in ast.walk(command)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not built & {"open_run_store", "build_recorder", "build_event_sink"}

    closed = [
        node
        for node in ast.walk(command)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
    ]
    assert closed == [], "the band closes the recorder and the store, on every exit path"

    said = {
        node.value
        for node in ast.walk(command)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "startup" not in said, "the sentinel is the band's, not the command's"

    finishes = _finish_calls(command)
    assert finishes, f"{verb} reports what stopped the segment"
    assert all(
        isinstance(node.func.value, ast.Name) and node.func.value.id == "seg" for node in finishes
    ), "the segment is finished through the handle the band yielded"


def _finish_calls(command):
    """Every ``…finish(…)`` call written in one command body."""
    import ast

    return [
        node
        for node in ast.walk(command)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "finish"
    ]


@pytest.mark.parametrize("verb", ["run", "research"])
def test_both_commands_hand_the_band_the_session_whole(verb):
    """Stories #250 / #251: ``mode``, ``rebase`` and ``engine_upgrade`` are no command's to
    forward — the band reads them off the session it was handed."""
    import ast

    (call,) = [
        node
        for node in ast.walk(_command_body(verb))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_segment_or_exit"
    ]

    passed = {keyword.arg for keyword in call.keywords}
    assert not passed & {"mode", "rebase", "engine_upgrade", "mandate", "overrides", "resume"}


def test_research_finishes_its_segment_with_the_working_seconds_it_measured_itself():
    """#251: a ``ResearchSummary`` carries no phase timings, so the session's working seconds are
    measured around ``session.run`` and reported explicitly — never derived from the outcome, and
    never a zero standing in for a night nobody timed."""
    reporting = [
        call
        for call in _finish_calls(_command_body("research"))
        if {"outcome", "phase_seconds"} <= {keyword.arg for keyword in call.keywords}
    ]

    assert len(reporting) == 1, "one call reports the session's outcome and the seconds it took"


def test_research_reports_what_the_session_spent_and_calls_the_dollars_an_estimate(
    tmp_path, monkeypatch
):
    """Story #140: an operator sees the bill, and sees that it is priced from a table rather than
    charged — the CLI is held to the same labelling rule as the record."""
    _patch_research_agent(monkeypatch)
    monkeypatch.setattr(
        _FakeSession,
        "run",
        lambda self, *, max_iterations=None: _summary(tokens=1500, usd=0.0123),
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")

    result = runner.invoke(app, ["research", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert "1,500 tokens" in result.output
    assert "$0.0123" in result.output
    assert "estimate" in result.output


def test_research_says_so_when_the_price_table_cannot_bill_the_model(tmp_path, monkeypatch):
    """Never a zero, never a guess: an unpriced model is named as unpriced."""
    _patch_research_agent(monkeypatch)
    monkeypatch.setattr(
        _FakeSession, "run", lambda self, *, max_iterations=None: _summary(tokens=900, usd=None)
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")

    result = runner.invoke(app, ["research", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert "900 tokens" in result.output
    assert "$" not in result.output.split("900 tokens")[1]
    assert "no price" in result.output


def _summary(*, tokens: int, usd: float | None):
    from noctis.engine import ResearchSummary

    return ResearchSummary(
        iterations=1,
        stopped_reason="done",
        tokens_total=tokens,
        usage={
            "input_tokens": tokens,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        usd_estimate=usd,
    )


def test_research_debug_without_session_writes_no_run_tree(tmp_path):
    """When the agent can't run (no key/extra), research exits before any recorder is opened —
    no orphaned half-written QA tree."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nqa_dir: {tmp_path}/qa\n")
    result = runner.invoke(app, ["research", "--config", str(cfg), "--debug"])
    assert result.exit_code != 0
    assert "[llm] extra" in result.output
    qa = tmp_path / "qa"
    assert not qa.exists() or not any(qa.iterdir())


# --- the overlay an operator can actually read: kickoff echo + status (story #122) -----
# The mandate overlay carries the whole run configuration now (model, budgets, data window,
# metric), so the configuration a run assembled from has to be visible where an operator
# already looks: the kickoff log of `run`/`research`, and `noctis status`.

# One mandate that moves knobs in three different sections — the everyday widened overlay.
_WIDE_OVERLAY = (
    "  promotion:\n"
    "    metric: sortino\n"
    "  research:\n"
    "    model: anthropic/claude-mandate-fake\n"
    "  data:\n"
    "    history_days: 45\n"
)
_WIDE_ECHOES = (
    "data.history_days=45",
    "promotion.metric=sortino",
    "research.model=anthropic/claude-mandate-fake",
)


def _mandate_config(tmp_path, profile: str, config_block: str = "") -> str:
    """A paper config whose active mandate is ``profile``; ``config_block`` is its overlay.

    The lake stays empty on purpose: ``run`` echoes its kickoff lines and then exits cleanly
    on the no-data path, so these tests read the echo without entering the loop.
    """
    front = f"config:\n{config_block}" if config_block else ""
    md = tmp_path / "mandate" / "profiles" / f"{profile}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(f"---\nsummary: A steering personality.\n{front}---\nSteer this session.\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "mode: paper",
                "data:",
                f"  lake_dir: {tmp_path}/lake",
                f"state_dir: {tmp_path}/state/",
                f"mandate_dir: {tmp_path}/mandate",
                "research:",
                f"  mandate: {profile}",
            ]
        )
        + "\n"
    )
    return str(cfg)


def test_run_kickoff_echoes_every_applied_override(tmp_path):
    """Every knob the mandate moved is in the log an operator already reads, under one header
    naming the mandate that moved them."""
    cfg = _mandate_config(tmp_path, "homelab", _WIDE_OVERLAY)
    result = runner.invoke(app, ["run", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "Mandate: profile:homelab" in result.output
    assert "mandate profile:homelab overrides:" in result.output
    for echo in _WIDE_ECHOES:
        assert echo in result.output


def test_research_kickoff_echoes_every_applied_override(tmp_path, monkeypatch):
    _patch_research_agent(monkeypatch)
    result = runner.invoke(
        app, ["research", "--config", _mandate_config(tmp_path, "homelab", _WIDE_OVERLAY)]
    )
    assert result.exit_code == 0, result.output
    assert "mandate profile:homelab overrides:" in result.output
    for echo in _WIDE_ECHOES:
        assert echo in result.output


def test_kickoff_is_unchanged_without_a_mandate(tmp_path):
    result = runner.invoke(app, ["run", "--config", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Mandate:" not in result.output
    assert "overrides" not in result.output


def test_kickoff_is_unchanged_when_the_mandate_applies_nothing(tmp_path):
    """A prose-only mandate still names itself and adds not one line more."""
    result = runner.invoke(app, ["run", "--config", _mandate_config(tmp_path, "prose")])
    assert result.exit_code == 0, result.output
    assert "Mandate: profile:prose" in result.output
    assert "overrides" not in result.output


def test_a_mandate_set_research_model_is_visible_at_kickoff(tmp_path):
    """The research-engine line reads post-overlay too: the model a mandate chose is the model
    the run announces, not the one config.yaml would have used."""
    cfg = _mandate_config(
        tmp_path, "homelab", "  research:\n    model: anthropic/claude-mandate-fake\n"
    )
    result = runner.invoke(app, ["run", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "Research engine:" in result.output
    assert "anthropic/claude-mandate-fake" in result.output
    assert "research.model=anthropic/claude-mandate-fake" in result.output


def test_status_reports_post_mandate_values(tmp_path):
    """status resolves the whole session, so its configuration lines show what the run would
    actually use — including a direction-clamped knob the mandate tightened."""
    cfg = _mandate_config(
        tmp_path,
        "homelab",
        "  research_time_budget_minutes: 17\n  data:\n    budget_usd: 3.5\n",
    )
    result = runner.invoke(app, ["status", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "research_budget:   17 min" in result.output
    assert "budget $3.5" in result.output


def test_status_prints_the_mandate_the_overrides_and_the_research_model(tmp_path):
    cfg = _mandate_config(tmp_path, "homelab", _WIDE_OVERLAY)
    result = runner.invoke(app, ["status", "--config", cfg])
    assert result.exit_code == 0, result.output
    assert "mandate:           profile:homelab" in result.output
    assert "research model:    anthropic/claude-mandate-fake" in result.output
    for echo in _WIDE_ECHOES:
        assert echo in result.output


def test_status_without_a_mandate_still_reports(tmp_path):
    """No mandate configured is the default install: status says so and keeps every line."""
    result = runner.invoke(app, ["status", "--config", _paper_config(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "mandate:           none" in result.output
    assert "overrides:         none" in result.output
    assert "research model:    openai/gpt-5.4" in result.output  # the shipped default
    assert "mode (resolved):   paper" in result.output


def test_status_research_model_falls_back_to_the_agent_model(tmp_path):
    """``research.model: null`` is the documented fallback to ``research.agent.model`` — status
    reports the model a session would actually drive, not the empty knob."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\nstate_dir: {tmp_path}/state/\n"
        f"research:\n  model: null\n  agent:\n    model: ollama_chat/local-30b\n"
    )
    result = runner.invoke(app, ["status", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "research model:    ollama_chat/local-30b" in result.output


def test_status_reports_an_unusable_mandate_instead_of_crashing(tmp_path):
    """A mandate `run` refuses to start under is exactly what an operator runs `status` to
    diagnose, so the refusal is reported in place — the remaining lines fall back to the
    pre-overlay settings and say so — while `run` itself still refuses outright."""
    cfg = _mandate_config(tmp_path, "sneaky", "  promotion:\n    max_gap: 5.0\n")

    status_result = runner.invoke(app, ["status", "--config", cfg])
    assert status_result.exit_code == 0, status_result.output
    assert "promotion.max_gap" in status_result.output  # the refusal, named
    assert "pre-overlay" in status_result.output  # ...and what that costs the lines below
    assert "mode (resolved):   paper" in status_result.output  # the diagnosis still prints

    run_result = runner.invoke(app, ["run", "--config", cfg])
    assert run_result.exit_code != 0
    assert "MANDATE" in run_result.output


def test_status_beside_a_legacy_layout_still_warns_rather_than_exits(tmp_path, monkeypatch):
    """The full-session switch keeps the diagnostic contract: status warns where run refuses."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state").mkdir()  # an un-migrated legacy artifact beside the config
    md = tmp_path / "mandate" / "profiles" / "homelab.md"
    md.parent.mkdir(parents=True)
    md.write_text(f"---\nconfig:\n{_WIDE_OVERLAY}---\nSteer this session.\n")
    cfg = tmp_path / "config.yaml"  # state/lake left at their workspace defaults
    cfg.write_text(f"mode: paper\nmandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: homelab\n")

    result = runner.invoke(app, ["status", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "legacy" in result.output.lower()
    assert "mode (resolved):   paper" in result.output
    assert "mandate:           profile:homelab" in result.output


# --- every reader resolves the same chain a run does (epic #292) ----------------------
# ``promotion.metric`` is the one ``promotion.*`` key a mandate may bind, and it was exactly the
# key the read-only verbs read *pre*-overlay, off a raw ``load_settings`` that stopped before the
# mandate — so a run steered onto ``sortino`` was reported on in ``sharpe``'s terms. One row per
# reader that has moved onto the reading band (``bootstrap.open_reading``, through
# ``_reading_or_exit``); a reader that stops at ``load_settings`` again fails its own row. Each
# row brings whatever that verb needs in order to *say* a metric at all.


def _needs_nothing(tmp_path) -> None:
    """This reader shows what it resolved with no state to read."""


def _crown_a_sortino_champion(tmp_path) -> None:
    """The board a run under this mandate produces: crowned under the metric it bound.

    Seeded directly rather than researched — the promotion itself is not what is under test here,
    the label the board is then read back under is.
    """
    from noctis.backtest.scorecard import Scorecard
    from noctis.champions.registry import ChampionEntry, ChampionRegistry

    registry = ChampionRegistry(tmp_path / "state" / "champions.json", 3)
    registry.champions.append(
        ChampionEntry(
            family="steered_winner",
            params={"fast": 3},
            scorecard=Scorecard(family="steered_winner", params={"fast": 3}, metric_name="sortino"),
            crowned_at="2026-01-01",
            rationale="seed",
        )
    )
    registry.save()


def _fill_the_lake(tmp_path) -> None:
    """One symbol of catalog data, so ``backtest`` reaches a scorecard rather than the no-data
    path — the scorecard is where the metric it scored on is printed."""
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns

    from ._data_helpers import MockVendor

    lake = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL"], to_ns("2026-01-01"), to_ns("2026-12-31")
    )


@pytest.mark.parametrize(
    ("setup", "argv", "present", "absent"),
    [
        pytest.param(_needs_nothing, ["report"], (), (), id="report"),
        # The bug this epic is named for: every entry on the board reads ``sortino(stale)``
        # when the label is taken off the pre-overlay metric.
        pytest.param(
            _crown_a_sortino_champion,
            ["champions"],
            ("steered_winner", "sortino"),
            ("sortino(stale)", "sharpe"),
            id="champions",
        ),
        # The scorecard a replay prints must be the one that promoted the champion.
        pytest.param(
            _fill_the_lake,
            ["backtest", "sma_crossover"],
            ("metric:           sortino",),
            ("sharpe",),
            id="backtest",
        ),
        # ``status`` narrates the steering itself: the mandate it resolved, every knob that
        # mandate moved, and the post-overlay model — none of which a raw ``load_settings``
        # would have, so a status that stopped there prints "none (unconstrained)" instead.
        pytest.param(
            _needs_nothing,
            ["status"],
            ("mandate:           profile:homelab", "promotion.metric=sortino"),
            ("sharpe", "mandate:           none"),
            id="status",
        ),
    ],
)
def test_readers_see_the_post_overlay_metric(tmp_path, setup, argv, present, absent):
    cfg = _mandate_config(tmp_path, "homelab", _WIDE_OVERLAY)
    setup(tmp_path)

    result = runner.invoke(app, [*argv, "--config", cfg])

    assert result.exit_code == 0, result.output
    for text in present:
        assert text in result.output
    for text in absent:
        assert text not in result.output


# --- the three lake verbs read the steered lake (story #298) --------------------------
# The lake is SHARED by every run and run-neutral, so ``data status|sync|ingest`` take no
# address — but they are readings all the same, and a verb that stopped at ``load_settings``
# inspected and filled a lake nobody had steered. What a mandate may bind here is narrow by
# design: the acquisition window (``data.history_days``, ``data.auto_backfill``), the seed
# ``universe``, and — towards discipline only — the vendor budget every fetch is priced
# against. ``data.dataset`` and ``data.lake_dir`` are refused by the overlay (vendor identity
# and state redirection are the arena), so the steering that *can* reach these three verbs
# arrives through the lake builder, and the one that cannot is refused out loud.

_THRIFTY_OVERLAY = "  data:\n    budget_usd: 0.0\n"


def _track_one_series(tmp_path) -> None:
    """One tracked series whose coverage stops well short of the T+1 boundary — so ``data sync``
    has a tail to price rather than a noop to report."""
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns

    from ._data_helpers import MockVendor

    lake = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL"], to_ns("2026-01-05"), to_ns("2026-01-09")
    )


def test_data_ingest_prices_against_the_post_overlay_budget(tmp_path, monkeypatch):
    """A mandate may only spend *less* of the operator's money — and the ingest it steers is
    priced against that number, not against the ceiling ``config.yaml`` set. Off a raw
    ``load_settings`` the fetch was made under the unsteered $125 budget."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _mandate_config(tmp_path, "thrifty", _THRIFTY_OVERLAY)

    result = runner.invoke(
        app,
        ["data", "ingest", "AAPL", "--start", "2026-01-05", "--end", "2026-01-09", "-c", cfg],
    )

    assert result.exit_code == 0, result.output
    assert "AAPL: refused" in result.output
    assert "budget $0.00" in result.output
    assert vendor.fetch_calls == 0  # priced only; the steered budget stopped the spend


def test_data_sync_extends_no_series_past_the_post_overlay_budget(tmp_path, monkeypatch):
    """The nightly tail extension is the same spend, so it meets the same steered ceiling."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-key")
    vendor = _install_mock_vendor(monkeypatch)
    cfg = _mandate_config(tmp_path, "thrifty", _THRIFTY_OVERLAY)
    _track_one_series(tmp_path)

    result = runner.invoke(app, ["data", "sync", "-c", cfg])

    assert result.exit_code == 0, result.output
    assert "AAPL: refused" in result.output
    assert vendor.fetch_calls == 0


def test_data_status_refuses_the_steering_it_cannot_apply(tmp_path, monkeypatch):
    """``data.dataset`` is refused by the overlay — which vendor dataset a lake holds is the
    arena, not steering. Read through the band, a mandate that tries to bind it is said out loud
    in the one refusal voice; before, the three lake verbs were the last place an operator's
    steering could be silently ignored, and they answered as if no mandate existed."""
    monkeypatch.chdir(tmp_path)
    cfg = _mandate_config(tmp_path, "wrongway", "  data:\n    dataset: XNAS.ITCH\n")

    result = runner.invoke(app, ["data", "status", "-c", cfg])

    assert result.exit_code == 1
    assert "MANDATE: " in result.stderr
    assert "data.dataset" in result.stderr


# --- sealing a run under the metric the run ran under (bug fix 3, story #297) ----------
# ``--finish`` and ``run-prune`` are the two read-only-ish verbs that *rewrite* the record, and
# the record carries the engine's ``comparable_key`` — whose last component is the election
# metric two runs must match on before their numbers may be pooled. Taken off a raw
# ``load_settings``, a run steered onto ``sortino`` was sealed under ``sharpe``: the bucket its
# numbers were produced in changed the moment an operator published them, and `noctis runs`
# printed the sealed one. Both verbs read it off the reading now — the run's **own** frozen
# metric, which is why the steering is taken away here before either verb runs.


def _minted_run_dir(tmp_path) -> Path:
    (run_dir,) = [p for p in (tmp_path / "workspace" / "runs").iterdir() if p.is_dir()]
    return run_dir


def _printed_comparable_key(cfg: str, address: str) -> str:
    """The run's bucket label as an operator reads it back: through ``noctis run-record``."""
    import json

    result = runner.invoke(app, ["run-record", address, "--config", cfg])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)["engine"]["comparable_key"]


def _run_steered_onto_sortino(tmp_path) -> tuple[str, Path]:
    """Mint one run under the mandate that binds ``promotion.metric: sortino``, then take the
    binding away — so the only thing that still says ``sortino`` is the run's own record."""
    cfg = _mandate_config(tmp_path, "homelab", _WIDE_OVERLAY)
    minted = runner.invoke(app, ["run", "--config", cfg])
    assert minted.exit_code == 0, minted.output
    run_dir = _minted_run_dir(tmp_path)
    _mandate_config(tmp_path, "homelab")  # the same profile, now binding nothing
    return cfg, run_dir


def _forget_the_engine_section(run_dir: Path) -> None:
    """Leave the record with no *readable* engine identity to carry forward.

    The shape a record written before engine epochs has — and the one a hand-edit produces — which
    the run tree tolerates by restamping the run with the identity of whatever engine writes
    next (``run_tree.store._frozen_engine``). It is therefore the one shape in which the election
    metric a sealing verb passes actually reaches ``engine.comparable_key``, which is what makes
    it the honest test of *which* metric that is: on every other record the key is frozen at
    creation and carried forward verbatim, so a wrong metric is latent rather than absent.
    """
    import json

    record = json.loads((run_dir / "run.json").read_text())
    record["engine"] = {}
    (run_dir / "run.json").write_text(json.dumps(record))


def test_finish_seals_a_run_under_the_election_metric_it_ran_under(tmp_path):
    """The key a run was sealed with is the key it ran under: same bucket at the end as at the
    start, whatever ``config.yaml`` says by the time an operator publishes the result."""
    cfg, run_dir = _run_steered_onto_sortino(tmp_path)
    before = _printed_comparable_key(cfg, run_dir.name)
    assert before.endswith("|sortino")

    sealed = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name, "--finish"])

    assert sealed.exit_code == 0, sealed.output
    assert _printed_comparable_key(cfg, run_dir.name) == before


def test_run_prune_keeps_the_election_metric_the_run_ran_under(tmp_path):
    """Retention rewrites the record too, and the record is all a pruned run has left."""
    cfg, run_dir = _run_steered_onto_sortino(tmp_path)
    before = _printed_comparable_key(cfg, run_dir.name)
    assert before.endswith("|sortino")
    sealed = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name, "--finish"])
    assert sealed.exit_code == 0, sealed.output

    pruned = runner.invoke(app, ["run-prune", run_dir.name, "--config", cfg])

    assert pruned.exit_code == 0, pruned.output
    assert _printed_comparable_key(cfg, run_dir.name) == before


@pytest.mark.parametrize(
    ("argv", "seal_first"),
    [
        pytest.param(lambda run_id: ["run", "--resume", run_id, "--finish"], False, id="finish"),
        # Only a completed run may be pruned, so this one is sealed while its record still
        # carries an engine — and it is the prune that restamps.
        pytest.param(lambda run_id: ["run-prune", run_id], True, id="run-prune"),
    ],
)
def test_a_sealing_verb_restamps_a_record_with_the_runs_own_election_metric(
    tmp_path, argv, seal_first
):
    """Where the election metric a sealing verb passes actually reaches the record, it is the
    run's — not whatever ``config.yaml`` resolves to today, and not the pre-overlay value a raw
    ``load_settings`` would have handed over."""
    cfg, run_dir = _run_steered_onto_sortino(tmp_path)
    before = _printed_comparable_key(cfg, run_dir.name)
    assert before.endswith("|sortino")
    if seal_first:
        sealed = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name, "--finish"])
        assert sealed.exit_code == 0, sealed.output
    _forget_the_engine_section(run_dir)

    result = runner.invoke(app, [*argv(run_dir.name), "--config", cfg])

    assert result.exit_code == 0, result.output
    assert _printed_comparable_key(cfg, run_dir.name) == before


def test_debug_manifest_digests_post_overlay_settings(tmp_path):
    """Regression: the QA manifest's config digest is taken over the settings the run actually
    assembled from, so a recorded run's manifest reflects the mandate's overlay — a knob set by
    a mandate digests exactly like the same knob set in config.yaml."""
    import json

    cfg = Path(_debug_run_config(tmp_path))
    profile = tmp_path / "mandate" / "profiles" / "spicy.md"
    profile.parent.mkdir(parents=True)
    profile.write_text("---\nconfig:\n  promotion:\n    metric: sortino\n---\nGo.\n")
    cfg.write_text(cfg.read_text() + f"mandate_dir: {tmp_path}/mandate\n")

    def digest(*args: str) -> str:
        result = runner.invoke(
            app, ["run", "--config", str(cfg), "--debug", "--time-limit-hours", "0", *args]
        )
        assert result.exit_code == 0, result.output
        run_id = next(
            line.split("QA run:")[1].strip()
            for line in result.output.splitlines()
            if line.startswith("QA run:")
        )
        return json.loads((tmp_path / "qa" / run_id / "run.json").read_text())["config_digest"]

    plain = digest()
    steered = digest("--mandate", "spicy")
    assert steered != plain  # the overlay reached the manifest at all...

    cfg.write_text(cfg.read_text() + "promotion:\n  metric: sortino\n")
    configured = digest()
    assert configured == steered  # ...and what it digests is the post-overlay value


# --- `noctis mandate <name>`: the preflight (story #124) -------------------------------
# The mandate is the sole run input, so a dry-run before committing a machine to a multi-day
# loop is what makes the widened surface self-documenting: resolve the mandate, print what it
# would actually do — provenance, declared symbols, references, and the effective settings
# diff from what to what — and start nothing at all.


def _write_profile(
    tmp_path, name: str, front_matter: str = "", body: str = "Steer this.\n"
) -> None:
    """Write ``mandate/profiles/<name>.md`` with the given front matter."""
    md = tmp_path / "mandate" / "profiles" / f"{name}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(f"---\n{front_matter}---\n{body}")


def _preflight_config(tmp_path) -> str:
    """A paper config with a mandate_dir but NO pinned mandate.

    The selector argument does all the work here — the way an operator preflights a
    personality they have not committed to yet. ``promotion.metric`` is pinned so the diff
    has a stated pre-value to move away from, and the lake stays empty so a cross-checking
    ``run`` exits cleanly on the no-data path right after its kickoff echo.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "mode: paper",
                "data:",
                f"  lake_dir: {tmp_path}/lake",
                f"state_dir: {tmp_path}/state/",
                f"mandate_dir: {tmp_path}/mandate",
                "promotion:",
                "  metric: sharpe",
            ]
        )
        + "\n"
    )
    return str(cfg)


def test_mandate_preflight_prints_source_summary_symbols_and_references(tmp_path):
    """The provenance an operator needs before committing a machine to a mandate: which file
    it resolved to, what it says it does, the tickers it declares, and what it pulled in."""
    refs = tmp_path / "mandate" / "references"
    refs.mkdir(parents=True)
    (refs / "watchlist.md").write_text("SMR is a small modular reactor play.\n")
    _write_profile(
        tmp_path,
        "homelab",
        "summary: A small-context homelab personality.\n"
        "symbols: [smr, CCJ, smr]\n"
        "references: [references/watchlist]\n",
    )

    result = runner.invoke(app, ["mandate", "homelab", "-c", _preflight_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "profile:homelab" in result.output
    assert "A small-context homelab personality." in result.output
    assert "SMR, CCJ" in result.output  # normalized and de-duped, as the session sees them
    assert "references/watchlist.md" in result.output


def test_mandate_preflight_prints_the_effective_settings_diff(tmp_path):
    """Every path the overlay binds, from what config resolved to what the run would use."""
    _write_profile(tmp_path, "homelab", f"config:\n{_WIDE_OVERLAY}")

    result = runner.invoke(app, ["mandate", "homelab", "-c", _preflight_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "promotion.metric" in result.output
    assert "sharpe → sortino" in result.output  # the config.yaml value → the mandate's
    assert "data.history_days" in result.output
    assert "365 → 45" in result.output  # the shipped default → the mandate's
    assert "openai/gpt-5.4 → anthropic/claude-mandate-fake" in result.output


def test_mandate_preflight_with_no_config_block_prints_provenance_and_an_empty_diff(tmp_path):
    """A prose-only mandate is a valid mandate: it still says what it is, and says plainly
    that it binds nothing."""
    _write_profile(tmp_path, "prose", "summary: Prose only, no knobs.\n")

    result = runner.invoke(app, ["mandate", "prose", "-c", _preflight_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "profile:prose" in result.output
    assert "Prose only, no knobs." in result.output
    assert "overrides:         none" in result.output


def test_mandate_preflight_of_auto_reports_an_empty_overlay(tmp_path):
    """``auto`` resolves to a selection instruction rather than a profile, so it binds nothing
    — a preflight that showed a profile's knobs here would be lying about the session."""
    _write_profile(tmp_path, "homelab", f"config:\n{_WIDE_OVERLAY}")

    result = runner.invoke(app, ["mandate", "auto", "-c", _preflight_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "mandate:           auto" in result.output
    assert "overrides:         none" in result.output
    assert "sortino" not in result.output  # the profile's overlay never reaches this session


def test_mandate_preflight_of_an_empty_mandate_says_it_steers_nothing(tmp_path):
    """The shipped MANDATE.md is all comments, so the selector resolves to no mandate at all —
    said out loud, and still a clean exit, because unconstrained is a configuration too."""
    (tmp_path / "mandate").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mandate" / "MANDATE.md").write_text("<!-- how-to header only -->\n")

    result = runner.invoke(app, ["mandate", "MANDATE", "-c", _preflight_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "no steering" in result.output
    assert "unconstrained" in result.output


def test_mandate_preflight_exits_nonzero_and_lists_every_problem_at_once(tmp_path):
    """A refused key, a wrong-direction clamp and an unknown key are all named in one pass,
    each with the reason for that path — and they are the same problems startup prints, so a
    cron job can gate on the preflight and get the diagnosis a run would have given it."""
    _write_profile(
        tmp_path,
        "sneaky",
        "config:\n"
        "  promotion:\n"
        "    max_gap: 5.0\n"
        "  research:\n"
        "    min_trials: 2\n"
        "    metrik: sortino\n",
    )
    cfg = _preflight_config(tmp_path)

    result = runner.invoke(app, ["mandate", "sneaky", "-c", cfg])

    assert result.exit_code != 0
    assert "promotion.max_gap" in result.output
    assert "promotion gates" in result.output  # the reason that path is refused
    assert "research.min_trials" in result.output
    assert "may only be raised" in result.output  # the clamp direction
    assert "research.metrik" in result.output
    assert "not a setting" in result.output

    startup = runner.invoke(app, ["run", "--config", cfg, "--mandate", "sneaky"])
    assert startup.exit_code != 0
    for problem in ("promotion.max_gap", "research.min_trials", "research.metrik"):
        assert problem in startup.output


def test_mandate_preflight_exits_nonzero_on_an_invalid_value(tmp_path):
    """A value the owning config section rejects fails the preflight too — the whole point is
    that a gating script never has to start a run to find out."""
    _write_profile(tmp_path, "typo", "config:\n  promotion:\n    metric: not_a_metric\n")

    result = runner.invoke(app, ["mandate", "typo", "-c", _preflight_config(tmp_path)])

    assert result.exit_code != 0
    assert "promotion.metric" in result.output
    assert "not_a_metric" in result.output


def test_mandate_preflight_unresolvable_selector_exits_with_the_not_found_diagnosis(tmp_path):
    """The one not-found message, reused: it names the selector and both paths it looked in."""
    result = runner.invoke(app, ["mandate", "no-such-profile", "-c", _preflight_config(tmp_path)])

    assert result.exit_code != 0
    assert "no-such-profile" in result.output
    assert "not found" in result.output
    assert "profiles/no-such-profile.md" in result.output


def test_mandate_preflight_starts_nothing(tmp_path, monkeypatch):
    """A preflight is a dry run: no research session, no LLM client (not even the probe, which
    drags litellm in), and no runtime that could place an order."""
    started: list[str] = []
    monkeypatch.setattr(
        "noctis.bootstrap.build_research_session", lambda **kwargs: started.append("session")
    )
    monkeypatch.setattr(
        "noctis.research.build_llm_client", lambda *a, **kw: started.append("llm_client")
    )
    monkeypatch.setattr(
        "noctis.research.client_status", lambda *a, **kw: started.append("client_status")
    )
    monkeypatch.setattr("noctis.engine.build_runtime", lambda *a, **kw: started.append("runtime"))
    _write_profile(tmp_path, "homelab", f"config:\n{_WIDE_OVERLAY}")

    result = runner.invoke(app, ["mandate", "homelab", "-c", _preflight_config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert started == []


def test_mandate_preflight_shows_what_a_run_would_actually_get(tmp_path):
    """One precedence chain, two renderings: the preflight's post-value is the value the
    kickoff echo prints for the same mandate — it resolves through the same session path."""
    _write_profile(tmp_path, "homelab", "config:\n  research_time_budget_minutes: 17\n")
    cfg = _preflight_config(tmp_path)

    preflight = runner.invoke(app, ["mandate", "homelab", "-c", cfg])
    assert preflight.exit_code == 0, preflight.output
    assert "research_time_budget_minutes" in preflight.output
    assert "60 → 17" in preflight.output  # the shipped default → the mandate's

    kickoff = runner.invoke(app, ["run", "--config", cfg, "--mandate", "homelab"])
    assert kickoff.exit_code == 0, kickoff.output
    assert "research_time_budget_minutes=17" in kickoff.output


# ── the one startup-error table (D7, story #252) ───────────────────────────────────────────

REFUSAL_PREFIXES = (
    "MANDATE: ",
    "SAFETY GATE: ",
    "RESUME: ",
    "RUN LOCKED: ",
    "FINISH: ",
    "PRUNE: ",
)

NO_SUCH_RUN = "20260101T000000Z-nope00"


def _refusing_argv(prefix: str, tmp_path, monkeypatch) -> list[str]:
    """The shortest invocation that provokes one typed startup refusal, whole with its setup."""
    if prefix == "MANDATE: ":
        return ["run", "--config", _paper_config(tmp_path), "--mandate", "no-such-mandate"]
    if prefix == "SAFETY GATE: ":
        monkeypatch.delenv("ALLOW_LIVE", raising=False)
        return ["run", "--config", _live_config(tmp_path)]
    if prefix == "RESUME: ":
        return ["run", "--config", _paper_config(tmp_path), "--resume", NO_SUCH_RUN]
    if prefix == "FINISH: ":
        return ["run", "--config", _paper_config(tmp_path), "--resume", NO_SUCH_RUN, "--finish"]
    if prefix == "PRUNE: ":
        return ["run-prune", NO_SUCH_RUN, "--config", _paper_config(tmp_path)]
    from noctis.reporting.run_tree.lock import acquire_lock

    cfg = _paper_config(tmp_path)
    assert runner.invoke(app, ["run", "--config", cfg]).exit_code == 0
    (run_dir,) = [p for p in (tmp_path / "workspace" / "runs").iterdir() if p.is_dir()]
    # A live lock held by this very process: never stale, so the next engine must refuse.
    acquire_lock(run_dir, run_id=run_dir.name, now=datetime.now(UTC))
    return ["run", "--config", cfg, "--resume", run_dir.name]


@pytest.mark.parametrize("prefix", REFUSAL_PREFIXES)
def test_every_typed_refusal_still_names_itself_on_stderr(prefix, tmp_path, monkeypatch):
    """One table, six sentences unchanged (#252): each typed startup error still exits 1 with its
    own prefix and its own diagnosis on stderr, whichever verb provoked it."""
    result = runner.invoke(app, _refusing_argv(prefix, tmp_path, monkeypatch))

    assert result.exit_code == 1, result.output
    said = [line for line in result.stderr.splitlines() if line.startswith(prefix)]
    assert len(said) == 1, result.stderr
    assert said[0] != prefix.rstrip(), "the refusal's own diagnosis rides behind the prefix"


@pytest.mark.parametrize("prefix", REFUSAL_PREFIXES)
def test_each_refusal_prefix_is_spelled_exactly_once(prefix):
    """D7 (#252): every startup refusal maps through one module-level table, so a prefix cannot
    drift between the ladder in ``_resolve_session_or_exit`` and the four sites that used to
    re-spell it by hand."""
    import ast

    spellings = [
        node
        for node in ast.walk(_cli_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == prefix
    ]

    assert len(spellings) == 1, f"{prefix!r} is spelled {len(spellings)} times in noctis/cli.py"
