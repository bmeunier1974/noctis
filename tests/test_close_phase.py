"""ClosePhase — the close finishes its evidence *before* it renders it (story #267).

The day's report used to be written first and reconciled afterwards, so a flagged feed drift
was composed onto a frozen report that was already on disk and reached no file. These tests
drive the phase end to end and assert on the **written** files: sync → integrity → reconcile →
account → mark → assemble → write → distill → reorganize, with every step isolated so memory
upkeep always runs.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from noctis.broker.paper import PaperBroker
from noctis.broker.persistence import EQUITY_CURVE_NAME, AccountStore, EquityLedger
from noctis.broker.seam import Order, Side
from noctis.champions import ChampionRegistry
from noctis.config import load_settings
from noctis.data.types import empty_bars
from noctis.engine.close import ClosePhase
from noctis.engine.report_assembly import SessionActivity
from noctis.reporting.report import Trade

CLOSE_AT = datetime(2026, 7, 3, 21, 0, tzinfo=UTC)
AS_OF = "2026-07-03"
TRACKED = [("EQUS.MINI", "ohlcv-1m", "AAPL")]


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


class _FakeMemory:
    def __init__(self, log):
        self.log = log
        self.reorganized = False

    def findings(self):
        return []

    def reorganize(self, registry=None):
        self.log.append("memory")
        self.reorganized = True


class _FakeLake:
    """The close's data seam: the tail sync, the integrity check, and the vendor bars the
    reconcile compares the session's live bars against."""

    def __init__(self, log, vendor=None, sync_raises=False):
        self.log = log
        self.vendor = vendor or {}
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

    def get_bars(self, dataset, schema, symbols, start, end):
        self.log.append("reconcile")
        return {s: self.vendor.get(s, empty_bars()) for s in symbols}


def _phase(tmp_path, *, log=None, lake=None, memory=None, distill_fn=None) -> ClosePhase:
    log = [] if log is None else log
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    return ClosePhase(
        settings=load_settings(state_dir=str(state)),
        reports_dir=str(tmp_path / "reports"),
        memory=memory if memory is not None else _FakeMemory(log),
        market_lake=lake,
        registry=ChampionRegistry(state / "champions.json", 3),
        distill_fn=distill_fn,
    )


def _diverging_session() -> tuple[SessionActivity, dict[str, pd.DataFrame]]:
    """A session whose live bars disagree with the vendor catalog by ~10% on one bar."""
    live = _bars([100.0, 101.0])
    vendor = _bars([100.0, 101.0])
    vendor.loc[1, ["open", "high", "low", "close"]] = [110.0, 110.0, 110.0, 110.0]
    return SessionActivity(live_bars={"AAPL": live}), {"AAPL": vendor}


def _events_section(markdown: str) -> str:
    return markdown.split("## Notable events", 1)[1]


def _seed_account(state, *, session=date(2026, 7, 2)) -> None:
    """A paper account with one open position — the account a close reads back and marks."""
    broker = PaperBroker()
    broker.set_price("AAPL", 120.0)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    broker.set_price("AAPL", 130.0)
    AccountStore(state / "paper_account.json").save(broker, session)


# --- the flagged drift reaches both files ------------------------------------------------


def test_a_flagged_drift_reaches_both_report_files(tmp_path):
    """THE test of story #267: the reconcile runs *before* the render, so the drift event the
    close discovers is in the day's ``.md`` and ``.json`` — not composed onto a frozen report
    that is already on disk."""
    log: list[str] = []
    cycle, vendor = _diverging_session()
    phase = _phase(tmp_path, log=log, lake=_FakeLake(log, vendor=vendor))

    result = phase.run(CLOSE_AT, cycle, tracked=TRACKED)

    assert result.reconciliation is not None and result.reconciliation.flagged is True
    assert not result.errors
    markdown = (tmp_path / "reports" / f"{AS_OF}.md").read_text()
    assert "Feed drift" in _events_section(markdown)
    events = json.loads((tmp_path / "reports" / f"{AS_OF}.json").read_text())["events"]
    assert any(e.startswith("Feed drift") for e in events)
    # The event belongs to the cycle that produced it, not to a throwaway report copy.
    assert any(e.startswith("Feed drift") for e in cycle.events)


def test_a_clean_reconciliation_adds_no_event(tmp_path):
    log: list[str] = []
    bars = _bars([100.0, 101.0])
    cycle = SessionActivity(live_bars={"AAPL": bars.copy()})
    phase = _phase(tmp_path, log=log, lake=_FakeLake(log, vendor={"AAPL": bars.copy()}))

    result = phase.run(CLOSE_AT, cycle, tracked=TRACKED)

    assert result.reconciliation is not None and result.reconciliation.flagged is False
    assert cycle.events == []
    assert "Feed drift" not in (tmp_path / "reports" / f"{AS_OF}.md").read_text()


# --- step order and failure isolation -----------------------------------------------------


def test_close_runs_steps_in_order(tmp_path):
    log: list[str] = []
    cycle, vendor = _diverging_session()
    phase = _phase(tmp_path, log=log, lake=_FakeLake(log, vendor=vendor))

    result = phase.run(CLOSE_AT, cycle, tracked=TRACKED)

    assert log == ["sync", "integrity", "reconcile", "memory"]
    assert result.report_path and result.report_path.endswith(f"{AS_OF}.md")
    assert result.memory_reorganized is True
    assert not result.errors


def test_a_failing_sync_still_reconciles_marks_writes_and_reorganizes(tmp_path):
    log: list[str] = []
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    _seed_account(state)
    cycle, vendor = _diverging_session()
    phase = _phase(tmp_path, log=log, lake=_FakeLake(log, vendor=vendor, sync_raises=True))

    result = phase.run(CLOSE_AT, cycle, tracked=TRACKED)

    assert any("sync" in e for e in result.errors)
    assert log == ["integrity", "reconcile", "memory"]
    assert result.reconciliation is not None and result.reconciliation.flagged is True
    assert result.memory_reorganized is True
    assert [m["date"] for m in EquityLedger(state / EQUITY_CURVE_NAME).marks()] == [AS_OF]
    # The report still exists, and it carries the drift the failed sync did not prevent.
    assert "Feed drift" in (tmp_path / "reports" / f"{AS_OF}.md").read_text()


def test_distill_is_isolated_and_precedes_reorganize(tmp_path):
    log: list[str] = []

    def distill():
        log.append("distill")
        return True

    phase = _phase(tmp_path, log=log, distill_fn=distill)
    result = phase.run(CLOSE_AT, SessionActivity())

    assert log == ["distill", "memory"]  # distillation feeds the reorganize that follows
    assert result.memory_distilled is True and result.memory_reorganized is True

    log.clear()
    boom = _phase(
        tmp_path,
        log=log,
        distill_fn=lambda: (_ for _ in ()).throw(RuntimeError("distill boom")),
    )
    result = boom.run(CLOSE_AT, SessionActivity())

    assert any("distill" in e for e in result.errors)
    assert result.memory_distilled is False and result.memory_reorganized is True


# --- the account is read once, and the mark rides it --------------------------------------


def test_the_account_is_parsed_once_per_close(tmp_path, monkeypatch):
    """One ``load()`` of ``paper_account.json`` feeds the equity mark *and* the report, so the
    curve and the report state the same account — they are the same read."""
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    _seed_account(state)
    loads: list[str] = []
    real_load = AccountStore.load

    def counting_load(self, *args, **kwargs):
        loads.append(str(self.path))
        return real_load(self, *args, **kwargs)

    monkeypatch.setattr(AccountStore, "load", counting_load)
    phase = _phase(tmp_path)

    phase.run(CLOSE_AT, SessionActivity())

    assert len(loads) == 1


def test_an_account_file_yields_one_mark_carrying_the_sessions_trades(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    _seed_account(state)
    cycle = SessionActivity(
        start_equity=100_000.0,
        end_equity=100_300.0,
        positions={"AAPL": 10.0},
        trades=[Trade(symbol="AAPL", side="BUY", quantity=10, price=120.0)],
    )

    _phase(tmp_path).run(CLOSE_AT, cycle)

    marks = EquityLedger(state / EQUITY_CURVE_NAME).marks()
    account = AccountStore(state / "paper_account.json").summary()
    assert [m["date"] for m in marks] == [AS_OF]
    assert marks[0]["equity"] == pytest.approx(account.equity)
    assert marks[0]["trades"][0]["symbol"] == "AAPL"


def test_a_close_with_no_account_yet_appends_no_mark(tmp_path):
    _phase(tmp_path).run(CLOSE_AT, SessionActivity())

    assert EquityLedger(tmp_path / "state" / EQUITY_CURVE_NAME).marks() == []


# --- the overwrite note reaches both files ------------------------------------------------


def test_a_differing_prior_report_is_archived_and_the_note_reaches_both_files(tmp_path):
    reports = tmp_path / "reports"
    note = f"Overwrote existing report for {AS_OF} (prior archived)"

    _phase(tmp_path).run(CLOSE_AT, SessionActivity(end_equity=100_000.0))
    _phase(tmp_path).run(CLOSE_AT, SessionActivity(end_equity=105_000.0))  # differs

    assert note in _events_section((reports / f"{AS_OF}.md").read_text())
    assert json.loads((reports / f"{AS_OF}.json").read_text())["events"] == [note]
    assert list((reports / "archive").glob(f"{AS_OF}.*.md")), "the prior report is kept"
