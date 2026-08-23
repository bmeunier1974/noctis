"""CLOSE orchestration order, failure isolation, and reconciliation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from noctis.engine.close import reconcile_bars, run_close
from noctis.reporting import ReportData


def _bars(closes, ts0=0):
    n = len(closes)
    return pd.DataFrame(
        {
            "ts_event": [ts0 + i * 60_000_000_000 for i in range(n)],
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [100] * n,
        }
    )


# --- reconciliation ----------------------------------------------------------------------


def test_reconcile_identical_bars_not_flagged():
    bars = _bars([100.0, 101.0, 102.0])
    rep = reconcile_bars(bars, bars.copy(), threshold=0.005)
    assert rep.n_compared == 3
    assert rep.max_drift == 0.0
    assert rep.flagged is False


def test_reconcile_flags_injected_drift():
    live = _bars([100.0, 101.0, 102.0])
    vendor = _bars([100.0, 101.0, 102.0])
    vendor.loc[1, ["open", "high", "low", "close"]] = [103.0, 103.1, 102.9, 103.0]  # ~2% off
    rep = reconcile_bars(live, vendor, threshold=0.005)
    assert rep.flagged is True
    assert rep.max_drift > 0.005


# --- orchestration -----------------------------------------------------------------------


class _FakeMemory:
    def __init__(self, log):
        self.log = log
        self.reorganized = False

    def reorganize(self, registry=None):
        self.log.append("memory")
        self.reorganized = True


class _FakeLake:
    def __init__(self, log, sync_raises=False):
        self.log = log
        self.sync_raises = sync_raises

    def sync(self):
        if self.sync_raises:
            raise RuntimeError("sync boom")
        self.log.append("sync")
        return {"AAPL": SimpleNamespace(status="noop")}

    def check(self, dataset, schema, symbol):
        self.log.append("integrity")
        return SimpleNamespace(clean=True, gap_count=0, duplicate_count=0)

    def repair(self, report):  # pragma: no cover - not reached when clean
        self.log.append("repair")


def test_close_runs_steps_in_order(tmp_path):
    log: list[str] = []
    memory = _FakeMemory(log)
    lake = _FakeLake(log)

    def reconcile():
        log.append("reconcile")
        return reconcile_bars(_bars([100.0]), _bars([100.0]))

    result = run_close(
        report_data=ReportData(as_of="2026-07-03"),
        reports_dir=str(tmp_path / "reports"),
        memory=memory,
        market_lake=lake,
        registry=None,
        reconcile_fn=reconcile,
        tracked=[("EQUS.MINI", "ohlcv-1m", "AAPL")],
    )
    assert log == ["sync", "integrity", "reconcile", "memory"]
    assert result.report_path and result.report_path.endswith("2026-07-03.md")
    assert result.memory_reorganized is True
    assert not result.errors


def test_close_failing_step_does_not_block_memory(tmp_path):
    log: list[str] = []
    memory = _FakeMemory(log)
    lake = _FakeLake(log, sync_raises=True)

    result = run_close(
        report_data=ReportData(as_of="2026-07-03"),
        reports_dir=str(tmp_path / "reports"),
        memory=memory,
        market_lake=lake,
        tracked=[("EQUS.MINI", "ohlcv-1m", "AAPL")],
    )
    # Sync failed, but integrity, and crucially memory upkeep, still ran.
    assert any("sync" in e for e in result.errors)
    assert memory.reorganized is True
    assert result.memory_reorganized is True


def test_close_reconciliation_drift_is_flagged_without_crashing_on_a_frozen_report(tmp_path):
    """A flagged drift is reported on the result, and composing its event onto the (frozen)
    report never raises. The event still reaches no file: it is composed after both reports
    are on disk — the known ordering bug epic #264 fixes by rendering last."""
    memory = _FakeMemory([])
    report_data = ReportData(as_of="2026-07-03")

    def reconcile():
        live = _bars([100.0, 101.0])
        vendor = _bars([100.0, 101.0])
        vendor.loc[1, ["open", "high", "low", "close"]] = [110.0, 110.0, 110.0, 110.0]
        return reconcile_bars(live, vendor, threshold=0.005)

    result = run_close(
        report_data=report_data,
        reports_dir=str(tmp_path / "reports"),
        memory=memory,
        reconcile_fn=reconcile,
    )
    assert result.reconciliation is not None and result.reconciliation.flagged is True
    assert not result.errors
    assert "Feed drift" not in (tmp_path / "reports" / "2026-07-03.md").read_text()


def test_close_writes_the_overwrite_note_into_both_report_files(tmp_path):
    """The close writes ``.md`` and ``.json`` from one final report value, so a re-run's
    overwrite note reaches both files — it used to ride an in-place mutation of a shared
    (then-mutable) report between the two writes."""
    reports = tmp_path / "reports"
    note = "Overwrote existing report for 2026-07-03 (prior archived)"

    run_close(
        report_data=ReportData(as_of="2026-07-03", end_equity=100_000.0),
        reports_dir=str(reports),
        memory=_FakeMemory([]),
    )
    run_close(
        report_data=ReportData(as_of="2026-07-03", end_equity=105_000.0),  # differs
        reports_dir=str(reports),
        memory=_FakeMemory([]),
    )

    assert note in (reports / "2026-07-03.md").read_text()
    assert json.loads((reports / "2026-07-03.json").read_text())["events"] == [note]


def test_close_distill_step_is_isolated_and_precedes_reorganize(tmp_path):
    log: list[str] = []
    memory = _FakeMemory(log)

    def distill():
        log.append("distill")
        return True

    result = run_close(
        report_data=ReportData(as_of="2026-07-03"),
        reports_dir=str(tmp_path / "reports"),
        memory=memory,
        distill_fn=distill,
    )
    assert log == ["distill", "memory"]  # distillation feeds the reorganize that follows
    assert result.memory_distilled is True and result.memory_reorganized is True

    # A failing distillation is recorded but never blocks memory upkeep.
    log.clear()
    result = run_close(
        report_data=ReportData(as_of="2026-07-03"),
        reports_dir=str(tmp_path / "reports"),
        memory=memory,
        distill_fn=lambda: (_ for _ in ()).throw(RuntimeError("distill boom")),
    )
    assert any("distill" in e for e in result.errors)
    assert result.memory_distilled is False and result.memory_reorganized is True
