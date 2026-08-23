"""The CLOSE phase orchestration.

On market close, in order: (1) tail-only catalog sync, (2) integrity check + flag-limited
repair, (3) reconcile the session's live bars against vendor T+1 history, (4) one read of the
paper account + forward ledger, (5) the day's equity mark, (6) assemble the report, (7) write
it (Markdown + JSON), (8) periodic memory distillation, (9) reorganize memory.

**The close finishes its evidence before it renders it.** The reconcile has to follow the sync
(it compares against T+1 vendor bars), so the report is written last — a flagged feed drift is
appended to the session's events and reaches both files, where it used to be composed onto a
frozen report already on disk and reach none (epic #264, D1). Every step is isolated: a failure
is logged and recorded but never prevents memory upkeep or the transition back to RESEARCH.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from noctis.data.types import empty_bars
from noctis.engine.report_assembly import SessionActivity, assemble_report, gather_account_forward
from noctis.reporting.report import write_reports

if TYPE_CHECKING:
    from noctis.broker.persistence import AccountSummary
    from noctis.data.seam import MarketData
    from noctis.memory.base import Memory

logger = logging.getLogger("noctis.close")


@dataclass
class ReconciliationReport:
    n_compared: int
    max_drift: float
    mean_drift: float
    threshold: float
    flagged: bool


def reconcile_bars(
    live: pd.DataFrame, vendor: pd.DataFrame, threshold: float = 0.005
) -> ReconciliationReport:
    """Compare live vs vendor bars on matching timestamps; flag drift over ``threshold``.

    Drift per bar is the max relative difference across OHLC. Only timestamps present in
    both frames are compared.
    """
    if len(live) == 0 or len(vendor) == 0:
        return ReconciliationReport(0, 0.0, 0.0, threshold, flagged=False)

    merged = live.merge(vendor, on="ts_event", suffixes=("_live", "_vendor"))
    if len(merged) == 0:
        return ReconciliationReport(0, 0.0, 0.0, threshold, flagged=False)

    drifts = []
    for col in ("open", "high", "low", "close"):
        lv = merged[f"{col}_live"].to_numpy(dtype="float64")
        vd = merged[f"{col}_vendor"].to_numpy(dtype="float64")
        rel = abs(lv - vd) / pd.Series(vd).replace(0, pd.NA).to_numpy(dtype="float64")
        drifts.append(rel)
    per_bar_max = pd.DataFrame(drifts).max(axis=0)
    max_drift = float(per_bar_max.max())
    mean_drift = float(per_bar_max.mean())
    return ReconciliationReport(
        n_compared=len(merged),
        max_drift=max_drift,
        mean_drift=mean_drift,
        threshold=threshold,
        flagged=max_drift > threshold,
    )


@dataclass
class CloseResult:
    report_path: str | None = None
    sync: dict | None = None
    integrity: dict | None = None
    reconciliation: ReconciliationReport | None = None
    memory_distilled: bool = False
    memory_reorganized: bool = False
    errors: list[str] = field(default_factory=list)


class ClosePhase:
    """Run one CLOSE entry: finish the day's evidence, then state it once.

    The runtime assembles this at startup and drives it at each CLOSE with the cycle the day
    accumulated; the phase owns everything between the session's end and the report on disk —
    the sync, the integrity pass, the reconciliation, the equity mark, the report, and memory
    upkeep. What it hands back (:class:`CloseResult`) is the runtime's bookkeeping: the report
    path, the step outcomes and the errors the isolated steps swallowed.
    """

    def __init__(
        self,
        *,
        settings,
        reports_dir: str,
        memory: Memory,
        registry,
        market_lake: MarketData | None = None,
        schema: str = "ohlcv-1m",
        distill_fn: Callable[[], bool] | None = None,
    ):
        self.settings = settings
        self.reports_dir = reports_dir
        self.memory = memory
        self.market_lake = market_lake
        self.registry = registry
        self.schema = schema
        # CLOSE owns memory upkeep (the reorganize below), so the periodic distillation rides
        # the same isolated-step machinery instead of racing a live research session.
        self.distill_fn = distill_fn

    def run(
        self,
        t: datetime,
        cycle: SessionActivity,
        *,
        tracked: list[tuple[str, str, str]] | None = None,
    ) -> CloseResult:
        """Run the close-phase steps in order, isolating failures so upkeep always completes."""
        as_of = t.astimezone(UTC).date().isoformat()
        result = CloseResult()

        # 1) Tail-only incremental sync.
        if self.market_lake is not None:
            try:
                result.sync = {s: r.status for s, r in self.market_lake.sync().items()}
            except Exception as exc:  # noqa: BLE001
                logger.exception("close: sync failed")
                result.errors.append(f"sync: {exc}")

            # 2) Integrity check + flag-limited repair.
            try:
                integrity: dict = {}
                for dataset, schema, symbol in tracked or []:
                    report = self.market_lake.check(dataset, schema, symbol)
                    if not report.clean:
                        self.market_lake.repair(report)
                    integrity[symbol] = {
                        "gap_count": report.gap_count,
                        "duplicate_count": report.duplicate_count,
                        "repaired": not report.clean,
                    }
                result.integrity = integrity
            except Exception as exc:  # noqa: BLE001
                logger.exception("close: integrity failed")
                result.errors.append(f"integrity: {exc}")

        # 3) Reconcile live vs vendor — after the sync, whose T+1 bars it compares against, and
        # before the report, which is why the drift it finds reaches the day's files at all.
        try:
            result.reconciliation = self.reconcile(cycle.live_bars)
            if result.reconciliation.flagged:
                cycle.events.append(
                    f"Feed drift {result.reconciliation.max_drift:.4f} exceeds "
                    f"threshold {result.reconciliation.threshold:.4f}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: reconciliation failed")
            result.errors.append(f"reconcile: {exc}")

        # 4) The paper account + forward track record, read once: the equity mark and the report
        # state the same account because they are the same read.
        forward = None
        try:
            forward = gather_account_forward(self.settings.state_dir, self.registry.list())
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: account read failed")
            result.errors.append(f"account: {exc}")

        # 5) The day's equity mark — before the report, as it has always been: the mark belongs
        # to the session that just closed, and the run record re-derives the whole curve from
        # the ledger at its next write.
        self._mark_equity(as_of, cycle, forward.account if forward is not None else None)

        # 6+7) Assemble the day's evidence into one frozen report and write it — Markdown
        # (human) + JSON (structured, for a frontend) from that one final value, so the pair
        # agrees on everything the write itself discovers (the "Overwrote existing report"
        # note). The report is frozen: it says what it said.
        try:
            data = assemble_report(
                as_of=as_of,
                mode=self.settings.mode,
                registry=self.registry,
                memory=self.memory,
                state_dir=self.settings.state_dir,
                session=cycle,
                account_forward=forward,
            )
            markdown, _ = write_reports(data, self.reports_dir)
            result.report_path = str(markdown)
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: report failed")
            result.errors.append(f"report: {exc}")

        # 8+9) Memory upkeep — ALWAYS runs, even if earlier steps failed. Periodic distillation
        # first (it reads the full findings history), then reorganize (whose size budget sees
        # the final state).
        if self.distill_fn is not None:
            try:
                result.memory_distilled = bool(self.distill_fn())
            except Exception as exc:  # noqa: BLE001
                logger.exception("close: memory distillation failed")
                result.errors.append(f"distill: {exc}")
        try:
            self.memory.reorganize(self.registry)
            result.memory_reorganized = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: memory reorganize failed")
            result.errors.append(f"memory: {exc}")

        return result

    def reconcile(
        self, live_bars: dict[str, pd.DataFrame], threshold: float = 0.005
    ) -> ReconciliationReport:
        """Compare the session's live-built bars against the (T+1 synced) catalog.

        When a live feed ran, each symbol's retained live bars are reconciled against the
        authoritative catalog and the per-symbol results are aggregated (drift over the
        threshold on any symbol flags). Without a live feed — or without a catalog to compare
        against — there is nothing external to check, so this is a no-op that never flags.
        """
        if not live_bars or self.market_lake is None:
            return ReconciliationReport(0, 0.0, 0.0, threshold, flagged=False)
        vendor_bars = self.market_lake.get_bars(
            self.settings.data.dataset, self.schema, list(live_bars), 0, 2**63 - 1
        )
        n = 0
        max_drift = 0.0
        weighted_mean = 0.0
        flagged = False
        for sym, live in live_bars.items():
            rep = reconcile_bars(live, vendor_bars.get(sym, empty_bars()), threshold=threshold)
            n += rep.n_compared
            max_drift = max(max_drift, rep.max_drift)
            weighted_mean += rep.mean_drift * rep.n_compared
            flagged = flagged or rep.flagged
        mean_drift = weighted_mean / n if n else 0.0
        return ReconciliationReport(n, max_drift, mean_drift, threshold, flagged=flagged)

    def _mark_equity(
        self, as_of: str, cycle: SessionActivity, account: AccountSummary | None
    ) -> None:
        """Append this CLOSE's daily equity mark to the run's own account ledger (story #142).

        The mark is the **account's** mark-to-market — the cumulative paper account read back
        for this close — not the session's own end equity, so the curve is the account's and a
        resumed run continues the same line. The session's fills, orders and closing positions
        ride with it, which is what makes the run record's trade log derivable from one durable
        artifact instead of from a report that a later ``noctis report`` could overwrite.

        No account means no mark: a night that never traded has no equity to state, and an
        invented flat 100 000 would be a claim about trading that never happened (epic D10).

        Never fatal, like every other reporting step at CLOSE: a ledger that cannot be written
        costs the record a day, never the run.
        """
        from noctis.broker.persistence import EQUITY_CURVE_NAME, EquityLedger

        if account is None:
            return
        try:
            EquityLedger(Path(self.settings.state_dir) / EQUITY_CURVE_NAME).mark(
                date=as_of,
                equity=account.equity,
                start_equity=cycle.start_equity,
                end_equity=cycle.end_equity,
                realized_pnl=cycle.end_equity - cycle.start_equity,
                orders_submitted=len(cycle.trades),
                positions_end=dict(cycle.positions),
                trades=[trade.as_dict() for trade in cycle.trades],
            )
        except Exception:  # noqa: BLE001 — the record is evidence, never a gate
            logger.exception("close: equity mark failed for %s; continuing", as_of)
