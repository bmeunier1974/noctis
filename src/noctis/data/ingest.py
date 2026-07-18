"""Coverage-diffed ingest — the fetch-once contract.

A requested range is diffed against the coverage registry; only the **missing** boundary
slices are fetched. A fully covered range triggers **zero** vendor calls and is reported as
``noop``. Interior gaps (a hole inside an otherwise-covered range) are the integrity
check's job, not this module's — ingest only extends coverage at the edges.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from noctis.data.coverage import CoverageRegistry
from noctis.data.lake import Lake
from noctis.data.preflight import CostPreflight
from noctis.data.types import SeriesKey, empty_bars, normalize_bars
from noctis.data.vendor import VendorClient

# Per-symbol liveness callback: (symbol, index_1_based, total). A multi-symbol vendor fetch
# can run for minutes with no other output — this is the caller's only progress signal.
ProgressFn = Callable[[str, int, int], None]


def missing_slices(
    first_ts: int | None, last_ts: int | None, req_start: int, req_end: int
) -> list[tuple[int, int]]:
    """Boundary slices of ``[req_start, req_end]`` not already covered by ``[first, last]``.

    Coverage is treated as a contiguous range (append-only). Returns the below-coverage
    and/or above-coverage tails; an empty list means the request is fully covered.
    """
    if req_end < req_start:
        return []
    if first_ts is None or last_ts is None:
        return [(req_start, req_end)]
    slices: list[tuple[int, int]] = []
    if req_start < first_ts:
        slices.append((req_start, min(req_end, first_ts - 1)))
    if req_end > last_ts:
        slices.append((max(req_start, last_ts + 1), req_end))
    return [(s, e) for (s, e) in slices if s <= e]


@dataclass
class IngestResult:
    symbol: str
    status: str  # noop | ingested | refused | dry_run | error
    fetch_calls: int = 0
    rows_added: int = 0
    raw_cost: float = 0.0
    padded_cost: float = 0.0
    slices: list[tuple[int, int]] = field(default_factory=list)
    detail: str = ""


class Ingestor:
    """Coverage-diffed ingest across a set of symbols."""

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

    def ingest(
        self,
        *,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: int,
        end: int,
        dry_run: bool = False,
        on_progress: ProgressFn | None = None,
    ) -> dict[str, IngestResult]:
        results: dict[str, IngestResult] = {}
        for i, symbol in enumerate(symbols, start=1):
            if on_progress is not None:
                on_progress(symbol, i, len(symbols))
            results[symbol] = self._ingest_one(dataset, schema, symbol, start, end, dry_run)
        return results

    def _ingest_one(
        self, dataset: str, schema: str, symbol: str, start: int, end: int, dry_run: bool
    ) -> IngestResult:
        key = SeriesKey(dataset, schema, symbol)
        rec = self.coverage.get(key)
        first = rec.first_ts if rec else None
        last = rec.last_ts if rec else None
        slices = missing_slices(first, last, start, end)

        if not slices:
            return IngestResult(symbol, "noop", detail="range fully covered")

        # Price the missing slices (metadata only — does not spend or fetch data).
        try:
            raw_cost = sum(
                self.vendor.get_cost(dataset=dataset, schema=schema, symbol=symbol, start=s, end=e)
                for (s, e) in slices
            )
        except Exception as exc:  # noqa: BLE001 - one symbol's vendor error must not abort the rest
            return IngestResult(symbol, "error", slices=slices, detail=str(exc))
        decision = self.preflight.decide(raw_cost)
        if not decision.allowed:
            return IngestResult(
                symbol,
                "refused",
                raw_cost=decision.raw_cost,
                padded_cost=decision.padded_cost,
                slices=slices,
                detail=decision.reason,
            )
        if dry_run:
            return IngestResult(
                symbol,
                "dry_run",
                raw_cost=decision.raw_cost,
                padded_cost=decision.padded_cost,
                slices=slices,
                detail="priced only; no data fetched",
            )

        self.coverage.set_status(key, "ingesting")
        try:
            frames = []
            calls = 0
            for s, e in slices:
                df = self.vendor.fetch_bars(
                    dataset=dataset, schema=schema, symbol=symbol, start=s, end=e
                )
                calls += 1
                frames.append(normalize_bars(df))
            new_bars = (
                normalize_bars(pd.concat(frames, ignore_index=True)) if frames else empty_bars()
            )
            entry = self.lake.write(key, new_bars)
            # Coverage tracks the *covered request interval* (union of ranges asked for),
            # not the min/max bar timestamp — otherwise re-requesting a range whose start
            # precedes the first bar would look like an uncovered gap and refetch. row_count
            # comes from the lake (actual bars on disk).
            covered_start = start if first is None else min(first, start)
            covered_end = end if last is None else max(last, end)
            self.coverage.upsert(
                key,
                first_ts=covered_start,
                last_ts=covered_end,
                row_count=entry.row_count,
                status="idle",
            )
            return IngestResult(
                symbol,
                "ingested",
                fetch_calls=calls,
                rows_added=int(len(new_bars)),
                raw_cost=decision.raw_cost,
                padded_cost=decision.padded_cost,
                slices=slices,
            )
        except Exception as exc:  # noqa: BLE001 - record and surface, don't crash the loop
            self.coverage.set_status(key, "error", str(exc))
            return IngestResult(symbol, "error", slices=slices, detail=str(exc))
