"""The CLOSE phase orchestration.

On market close, in order: (1) write the report, (2) tail-only catalog sync, (3) integrity
check + flag-limited repair, (4) reconcile the session's live bars against vendor T+1
history, (5) reorganize memory. Every step is isolated: a failure is logged and recorded but
never prevents memory upkeep or the transition back to RESEARCH.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from noctis.reporting.report import ReportData, write_report, write_report_json

if TYPE_CHECKING:
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


def run_close(
    *,
    report_data: ReportData,
    reports_dir: str,
    memory: Memory,
    market_lake: MarketData | None = None,
    registry=None,
    reconcile_fn: Callable[[], ReconciliationReport] | None = None,
    tracked: list[tuple[str, str, str]] | None = None,
    distill_fn: Callable[[], bool] | None = None,
) -> CloseResult:
    """Run the close-phase steps in order, isolating failures so upkeep always completes."""
    result = CloseResult()

    # 1) Report — Markdown (human) + JSON (structured, for a frontend). The JSON is
    # best-effort: a serialization hiccup must not lose the Markdown report.
    try:
        result.report_path = str(write_report(report_data, reports_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("close: report failed")
        result.errors.append(f"report: {exc}")
    try:
        write_report_json(report_data, reports_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("close: report JSON failed")
        result.errors.append(f"report_json: {exc}")

    # 2) Tail-only incremental sync.
    if market_lake is not None:
        try:
            result.sync = {s: r.status for s, r in market_lake.sync().items()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: sync failed")
            result.errors.append(f"sync: {exc}")

        # 3) Integrity check + flag-limited repair.
        try:
            integrity: dict = {}
            for dataset, schema, symbol in tracked or []:
                report = market_lake.check(dataset, schema, symbol)
                if not report.clean:
                    market_lake.repair(report)
                integrity[symbol] = {
                    "gap_count": report.gap_count,
                    "duplicate_count": report.duplicate_count,
                    "repaired": not report.clean,
                }
            result.integrity = integrity
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: integrity failed")
            result.errors.append(f"integrity: {exc}")

    # 4) Reconcile live vs vendor.
    if reconcile_fn is not None:
        try:
            result.reconciliation = reconcile_fn()
            if result.reconciliation and result.reconciliation.flagged:
                report_data.events.append(
                    f"Feed drift {result.reconciliation.max_drift:.4f} exceeds "
                    f"threshold {result.reconciliation.threshold:.4f}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: reconciliation failed")
            result.errors.append(f"reconcile: {exc}")

    # 5) Memory upkeep — ALWAYS runs, even if earlier steps failed. Periodic distillation
    # first (it reads the full findings history), then reorganize (whose size budget sees
    # the final state).
    if distill_fn is not None:
        try:
            result.memory_distilled = bool(distill_fn())
        except Exception as exc:  # noqa: BLE001
            logger.exception("close: memory distillation failed")
            result.errors.append(f"distill: {exc}")
    try:
        memory.reorganize(registry)
        result.memory_reorganized = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("close: memory reorganize failed")
        result.errors.append(f"memory: {exc}")

    return result
