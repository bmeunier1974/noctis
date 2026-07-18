"""The incremental updater — nightly tail-only sync.

For each tracked series, fetch ``last_ts + 1ns → until`` (the T+1 boundary), append, and
update coverage. History is never re-downloaded; an empty vendor response is a no-op that
leaves the registry untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from noctis.data.coverage import CoverageRecord, CoverageRegistry
from noctis.data.ingest import ProgressFn
from noctis.data.lake import Lake
from noctis.data.preflight import CostPreflight
from noctis.data.types import normalize_bars, t1_boundary_ns
from noctis.data.vendor import VendorClient


@dataclass
class SyncResult:
    symbol: str
    status: str  # updated | noop | skipped | refused | error
    fetch_calls: int = 0
    rows_added: int = 0
    detail: str = ""


def _default_until_ns(now: datetime | None) -> int:
    """T+1 boundary: UTC midnight of the current *ET* trading date (through end of yesterday).

    ET, not UTC — see :func:`t1_boundary_ns`: after 8 PM ET the UTC date has already rolled
    over, and a UTC-dated boundary crosses the vendor's live-license line (403).
    """
    return t1_boundary_ns(now or datetime.now(UTC))


class Updater:
    """Tail-only incremental sync over tracked series."""

    def __init__(
        self,
        lake: Lake,
        coverage: CoverageRegistry,
        preflight: CostPreflight,
        vendor: VendorClient,
    ):
        self.lake = lake
        self.coverage = coverage
        self.preflight = preflight
        self.vendor = vendor

    def sync(
        self,
        *,
        until_ns: int | None = None,
        now: datetime | None = None,
        dataset: str | None = None,
        on_progress: ProgressFn | None = None,
    ) -> dict[str, SyncResult]:
        boundary = until_ns if until_ns is not None else _default_until_ns(now)
        records = [rec for rec in self.coverage.all() if dataset is None or rec.dataset == dataset]
        results: dict[str, SyncResult] = {}
        for i, rec in enumerate(records, start=1):
            if on_progress is not None:
                on_progress(rec.symbol, i, len(records))
            results[rec.symbol] = self._sync_one(rec, boundary)
        return results

    def _sync_one(self, rec: CoverageRecord, boundary: int) -> SyncResult:
        key = rec.key
        if rec.status != "idle":
            return SyncResult(rec.symbol, "skipped", detail=f"status={rec.status}")
        if rec.last_ts is None:
            return SyncResult(rec.symbol, "skipped", detail="no existing coverage")

        start = rec.last_ts + 1
        if start > boundary:
            return SyncResult(rec.symbol, "noop", detail="already at boundary")

        # Budget preflight before any spend: price the tail slice and refuse if it exceeds
        # the budget, leaving coverage untouched (status stays idle, no fetch).
        raw_cost = self.vendor.get_cost(
            dataset=key.dataset, schema=key.schema, symbol=key.symbol, start=start, end=boundary
        )
        decision = self.preflight.decide(raw_cost)
        if not decision.allowed:
            return SyncResult(rec.symbol, "refused", detail=decision.reason)

        self.coverage.set_status(key, "ingesting")
        try:
            df = normalize_bars(
                self.vendor.fetch_bars(
                    dataset=key.dataset,
                    schema=key.schema,
                    symbol=key.symbol,
                    start=start,
                    end=boundary,
                )
            )
            if len(df) == 0:
                # Empty response: no state change beyond restoring idle status.
                self.coverage.set_status(key, "idle")
                return SyncResult(rec.symbol, "noop", fetch_calls=1, detail="empty response")
            entry = self.lake.write(key, df)
            # Coverage now extends through the requested boundary (covered interval), while
            # row_count reflects the actual bars on disk.
            self.coverage.upsert(
                key,
                first_ts=rec.first_ts,
                last_ts=boundary,
                row_count=entry.row_count,
                status="idle",
            )
            return SyncResult(rec.symbol, "updated", fetch_calls=1, rows_added=int(len(df)))
        except Exception as exc:  # noqa: BLE001
            self.coverage.set_status(key, "error", str(exc))
            return SyncResult(rec.symbol, "error", detail=str(exc))
