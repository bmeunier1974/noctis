"""The ``MarketData`` seam and the catalog-backed lake façade.

``MarketData`` is the swappable interface the rest of the app talks to: read bars **from
the catalog only** (:meth:`get_bars`), trigger ingest when needed (:meth:`ensure_coverage`),
and run coverage upkeep (:meth:`check_symbol_ready`, :meth:`sync`, :meth:`check`,
:meth:`repair` — the close phase syncs and heals the lake). The default provider,
:class:`MarketDataLake`, wires the lake owner, coverage registry, cost preflight, ingest,
updater, and integrity check into one object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from noctis.data.coverage import CoverageRecord, CoverageRegistry
from noctis.data.ingest import Ingestor, IngestResult, ProgressFn
from noctis.data.integrity import IntegrityChecker, IntegrityReport
from noctis.data.lake import Lake
from noctis.data.preflight import CostPreflight
from noctis.data.types import SeriesKey, normalize_bars
from noctis.data.updater import SyncResult, Updater
from noctis.data.vendor import VendorClient


@runtime_checkable
class MarketData(Protocol):
    """The market-data seam: catalog reads + coverage management."""

    def get_bars(
        self, dataset: str, schema: str, symbols: list[str], start: int, end: int
    ) -> dict[str, pd.DataFrame]:
        """Read bars for ``symbols`` in ``[start, end]`` **from the catalog only**."""
        ...

    def ensure_coverage(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: int,
        end: int,
        dry_run: bool = False,
    ) -> dict[str, IngestResult]:
        """Ingest any missing slices so the requested range is covered."""
        ...

    def check_symbol_ready(
        self, symbol: str, dataset: str | None = None, schema: str | None = None
    ) -> bool:
        """Whether the symbol's coverage is complete enough to trade/research on."""
        ...

    def coverage_records(self) -> list[CoverageRecord]:
        """Every tracked series' coverage claim (``noctis data status``)."""
        ...

    def sync(self, **kwargs) -> dict[str, SyncResult]:
        """Incrementally extend coverage of tracked series to the present."""
        ...

    def check(self, dataset: str, schema: str, symbol: str) -> IntegrityReport:
        """Audit one series' catalog data against its coverage claims."""
        ...

    def repair(self, report: IntegrityReport) -> dict[str, object]:
        """Re-ingest the slices an integrity check flagged."""
        ...


class MarketDataLake:
    """Catalog-backed :class:`MarketData` provider — the fetch-once data lake."""

    def __init__(
        self,
        lake_dir: str | Path,
        vendor: VendorClient,
        budget_usd: float,
        calendar: str = "XNYS",
    ):
        self.lake = Lake(lake_dir)
        self.coverage = CoverageRegistry(Path(lake_dir) / "coverage.db")
        self.preflight = CostPreflight(budget_usd)
        self.vendor = vendor
        self.calendar = calendar
        self.ingestor = Ingestor(self.lake, self.coverage, self.preflight, vendor)
        self.updater = Updater(self.lake, self.coverage, self.preflight, vendor)
        self.integrity = IntegrityChecker(
            self.lake, self.coverage, self.preflight, vendor, calendar
        )
        # Recover from a crash mid-ingest: stale 'ingesting' rows become 'error'.
        self.coverage.sweep_stale_ingesting()

    # --- MarketData seam ---
    def get_bars(
        self, dataset: str, schema: str, symbols: list[str], start: int, end: int
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.lake.read(SeriesKey(dataset, schema, symbol))
            mask = (df["ts_event"] >= start) & (df["ts_event"] <= end)
            out[symbol] = normalize_bars(df.loc[mask])
        return out

    def ensure_coverage(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: int,
        end: int,
        dry_run: bool = False,
        on_progress: ProgressFn | None = None,
    ) -> dict[str, IngestResult]:
        return self.ingestor.ingest(
            dataset=dataset,
            schema=schema,
            symbols=symbols,
            start=start,
            end=end,
            dry_run=dry_run,
            on_progress=on_progress,
        )

    def sync(self, **kwargs) -> dict[str, SyncResult]:
        return self.updater.sync(**kwargs)

    def check(self, dataset: str, schema: str, symbol: str) -> IntegrityReport:
        return self.integrity.check(SeriesKey(dataset, schema, symbol))

    def repair(self, report: IntegrityReport) -> dict[str, object]:
        return self.integrity.repair(report)

    def check_symbol_ready(
        self, symbol: str, dataset: str | None = None, schema: str | None = None
    ) -> bool:
        return self.coverage.check_symbol_ready(symbol, dataset, schema)

    def coverage_records(self) -> list[CoverageRecord]:
        return self.coverage.all()
