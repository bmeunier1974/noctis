"""Tests for the Typer CLI skeleton."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app

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
    reports = tmp_path / "workspace" / "reports"  # the workspace-derived default
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
    reports = tmp_path / "workspace" / "reports"  # the workspace-derived default
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
        author_calls = 0
        backtests_run = 3

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
        author_calls = 0
        backtests_run = 1

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
    from noctis.observability.debug import RUN_ID_RE

    qa = tmp_path / "qa"
    runs = [p for p in qa.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)]
    assert len(runs) == 1, [p.name for p in qa.iterdir()]
    return runs[0]


def _strip_qa_lines(output: str) -> str:
    """Drop the additive ``QA …`` framing lines so the event/console feed can be compared."""
    return "".join(line + "\n" for line in output.splitlines() if not line.startswith("QA "))


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
    assert _strip_qa_lines(debug.output) == plain.output


def test_run_debug_prunes_qa_area_on_start(tmp_path):
    from noctis.observability.debug import RUN_ID_RE

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

        self.model = "anthropic/claude-fake"
        self.budgets = SimpleNamespace(name="balanced", max_iterations=5)
        self.toolbox = SimpleNamespace(author_calls=0, backtests_run=1)
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

    run_id = run_dir.name
    assert result.output.count(run_id) >= 2  # echoed at start and again at stop
    assert "QA report:" in result.output
    # the funnel one-liner reflects the recorded write_strategy event
    assert "QA funnel: written=1" in result.output
    # --debug without -v stays silent: the emitted feed events never hit stdout
    assert "write_strategy(alpha)" not in result.output


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
