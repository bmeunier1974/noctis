"""The integrity check — decide when a repair fetch is actually needed.

``check()`` reports missing trading days (gaps vs the calendar), duplicate rows, schema
validity, and whether the manifest matches the files on disk. ``repair()`` fixes **exactly**
what the report flags — gap days are re-fetched, duplicates are compacted away — and nothing
more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from noctis.data.coverage import CoverageRegistry
from noctis.data.lake import Lake
from noctis.data.market_calendar import trading_sessions
from noctis.data.preflight import CostPreflight
from noctis.data.types import (
    BAR_COLUMNS,
    SeriesKey,
    day_bounds_ns,
    empty_bars,
    normalize_bars,
    ns_to_date,
)
from noctis.data.vendor import VendorClient


@dataclass
class IntegrityReport:
    key: SeriesKey
    gap_count: int
    duplicate_count: int
    schema_ok: bool
    manifest_ok: bool
    gap_days: list[date] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            self.gap_count == 0
            and self.duplicate_count == 0
            and self.schema_ok
            and self.manifest_ok
        )


class IntegrityChecker:
    """Gap / duplicate / schema / manifest verification and flag-limited repair."""

    def __init__(
        self,
        lake: Lake,
        coverage: CoverageRegistry,
        preflight: CostPreflight,
        vendor: VendorClient,
        calendar: str = "XNYS",
    ):
        self.lake = lake
        self.coverage = coverage
        self.preflight = preflight
        self.vendor = vendor
        self.calendar = calendar

    def check(self, key: SeriesKey) -> IntegrityReport:
        # Read raw (not normalized) so duplicates on disk are visible.
        path = self.lake.path_for(key)
        if not path.is_file():
            return IntegrityReport(key, 0, 0, schema_ok=True, manifest_ok=False)
        import pandas as pd

        raw = pd.read_parquet(path)

        schema_ok = list(raw.columns[: len(BAR_COLUMNS)]) == list(BAR_COLUMNS) or set(
            BAR_COLUMNS
        ).issubset(set(raw.columns))
        duplicate_count = int(raw["ts_event"].duplicated().sum()) if "ts_event" in raw else 0

        gap_days: list[date] = []
        if "ts_event" in raw and len(raw):
            present = {ns_to_date(int(ts)) for ts in raw["ts_event"]}
            lo, hi = min(present), max(present)
            expected = trading_sessions(lo, hi, self.calendar)
            # Exclude days a prior repair confirmed empty (e.g. holidays the weekday-fallback
            # calendar treats as sessions) so they aren't re-flagged forever.
            known_empty = self.coverage.known_empty_days(key)
            gap_days = [d for d in expected if d not in present and d not in known_empty]

        manifest_ok = self.lake.manifest_matches_disk(key, frame=raw)
        return IntegrityReport(
            key=key,
            gap_count=len(gap_days),
            duplicate_count=duplicate_count,
            schema_ok=schema_ok,
            manifest_ok=manifest_ok,
            gap_days=gap_days,
        )

    def repair(self, report: IntegrityReport) -> dict[str, object]:
        """Fetch exactly the flagged gap days and compact duplicates. Returns a summary.

        A budget preflight gates the gap-day fetch: the flagged days are priced first and,
        if the padded estimate exceeds budget, the repair is refused with no fetch and no
        state change (``summary["refused"]`` is ``True``, the reason is in ``detail``).
        """
        key = report.key
        summary: dict[str, Any] = {
            "gap_fetches": 0,
            "rows_added": 0,
            "compacted": 0,
            "empty_days": 0,
            "refused": False,
        }
        rec = self.coverage.get(key)
        # Preserve the covered request interval; repairs never shrink it.
        cov_first = rec.first_ts if rec else None
        cov_last = rec.last_ts if rec else None

        def _refresh_coverage(entry_row_count: int) -> None:
            self.coverage.upsert(
                key,
                first_ts=cov_first,
                last_ts=cov_last,
                row_count=entry_row_count,
                status="idle",
            )

        if report.gap_days:
            bounds = [day_bounds_ns(day) for day in report.gap_days]
            # Price the flagged gap days and refuse the whole repair if over budget — no
            # fetch, state untouched.
            raw_cost = sum(
                self.vendor.get_cost(
                    dataset=key.dataset, schema=key.schema, symbol=key.symbol, start=s, end=e
                )
                for (s, e) in bounds
            )
            decision = self.preflight.decide(raw_cost)
            if not decision.allowed:
                summary["refused"] = True
                summary["detail"] = decision.reason
                return summary

            self.coverage.set_status(key, "ingesting")
            try:
                import pandas as pd

                frames = []
                empties: list[date] = []
                for day, (start, end) in zip(report.gap_days, bounds, strict=True):
                    df = normalize_bars(
                        self.vendor.fetch_bars(
                            dataset=key.dataset,
                            schema=key.schema,
                            symbol=key.symbol,
                            start=start,
                            end=end,
                        )
                    )
                    summary["gap_fetches"] += 1
                    if len(df):
                        frames.append(df)
                    else:
                        empties.append(day)
                if empties:
                    # Confirmed-empty days won't be re-flagged or re-fetched → convergence.
                    self.coverage.mark_empty_days(key, empties)
                    summary["empty_days"] = len(empties)
                new_bars = (
                    normalize_bars(pd.concat(frames, ignore_index=True)) if frames else empty_bars()
                )
                entry = self.lake.write(key, new_bars)
                summary["rows_added"] = int(len(new_bars))
                _refresh_coverage(entry.row_count)
            except Exception as exc:  # noqa: BLE001
                self.coverage.set_status(key, "error", str(exc))
                raise

        if report.duplicate_count > 0:
            # Writing an empty frame forces a dedup + resort + manifest restamp.
            entry = self.lake.write(key, empty_bars())
            summary["compacted"] = report.duplicate_count
            _refresh_coverage(entry.row_count)
        elif not report.manifest_ok and not report.gap_days:
            # Manifest drifted but data is intact: restamp without fetching.
            entry = self.lake.write(key, empty_bars())
            _refresh_coverage(entry.row_count)

        return summary
