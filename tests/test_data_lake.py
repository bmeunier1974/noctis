"""The fetch-once contract: no byte is ever bought twice."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from noctis.data import (
    CostPreflight,
    CoverageRegistry,
    Ingestor,
    IngestResult,
    IntegrityChecker,
    Lake,
    MarketDataLake,
    SeriesKey,
    Updater,
    missing_slices,
)
from noctis.data.types import empty_bars, ns_to_date, to_ns, to_ns_end_inclusive

from ._data_helpers import MockVendor, bars_for_range

_ONE_DAY_NS = 86_400_000_000_000

DATASET = "EQUS.MINI"
SCHEMA = "ohlcv-1m"
SYMBOL = "AAPL"


def _ns(day: str) -> int:
    return to_ns(day)


@pytest.fixture
def lake_dir(tmp_path):
    return tmp_path / "lake"


@pytest.fixture
def stack(lake_dir):
    vendor = MockVendor()
    lake = Lake(lake_dir)
    coverage = CoverageRegistry(lake_dir / "coverage.db")
    preflight = CostPreflight(budget_usd=1000.0)
    ingestor = Ingestor(lake, coverage, preflight, vendor)
    updater = Updater(lake, coverage, preflight, vendor)
    integrity = IntegrityChecker(lake, coverage, preflight, vendor, calendar="XNYS")
    return {
        "vendor": vendor,
        "lake": lake,
        "coverage": coverage,
        "preflight": preflight,
        "ingestor": ingestor,
        "updater": updater,
        "integrity": integrity,
    }


# --- missing_slices unit -----------------------------------------------------------------


def test_missing_slices_no_coverage():
    assert missing_slices(None, None, 10, 20) == [(10, 20)]


def test_missing_slices_fully_covered():
    assert missing_slices(10, 20, 12, 18) == []
    assert missing_slices(10, 20, 10, 20) == []


def test_missing_slices_right_extension():
    assert missing_slices(10, 20, 10, 30) == [(21, 30)]


def test_missing_slices_both_ends():
    assert missing_slices(10, 20, 5, 30) == [(5, 9), (21, 30)]


# --- fetch-once contract -----------------------------------------------------------------


def test_reingest_covered_range_is_zero_call_noop(stack):
    """(1) Ingest A, re-ingest A → zero vendor calls the second time, result 'noop'."""
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    first = stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end
    )
    assert first[SYMBOL].status == "ingested"
    calls_after_first = stack["vendor"].fetch_calls
    cost_calls_after_first = stack["vendor"].cost_calls
    assert calls_after_first >= 1

    second = stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end
    )
    assert second[SYMBOL].status == "noop"
    # No fetch AND no cost estimate the second time — a covered range is fully free.
    assert stack["vendor"].fetch_calls == calls_after_first
    assert stack["vendor"].cost_calls == cost_calls_after_first


def test_ingest_diff_fetches_only_the_new_slice(stack):
    """(2) Ingest A, then request A∪B → vendor fetches B only."""
    a_start, a_end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=a_start, end=a_end
    )
    stack["vendor"].fetch_ranges.clear()
    calls_before = stack["vendor"].fetch_calls

    b_end = _ns("2026-01-16T23:59:59")
    res = stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=a_start, end=b_end
    )
    assert res[SYMBOL].status == "ingested"
    assert stack["vendor"].fetch_calls == calls_before + 1  # exactly one new slice
    (fetched_start, fetched_end) = stack["vendor"].fetch_ranges[0]
    assert fetched_start == a_end + 1  # begins just past prior coverage
    assert fetched_end == b_end


def test_ingest_vendor_pricing_error_is_contained_per_symbol(stack):
    """A vendor error while pricing one symbol (e.g. a 422 for an end past the available
    range) yields an 'error' result for that symbol only — the rest still ingests, and the
    failing symbol's coverage stays untouched so a later attempt starts fresh."""
    real_get_cost = stack["vendor"].get_cost

    def flaky_get_cost(*, dataset, schema, symbol, start, end):
        if symbol == "MSFT":
            raise RuntimeError("422 data_end_after_available_end")
        return real_get_cost(dataset=dataset, schema=schema, symbol=symbol, start=start, end=end)

    stack["vendor"].get_cost = flaky_get_cost
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    res = stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=["MSFT", SYMBOL], start=start, end=end
    )
    assert res["MSFT"].status == "error"
    assert "data_end_after_available_end" in res["MSFT"].detail
    assert res[SYMBOL].status == "ingested"  # the earlier error did not abort the batch
    assert stack["coverage"].get(SeriesKey(DATASET, SCHEMA, "MSFT")) is None


def test_sync_requests_only_the_tail_and_empty_is_noop(stack):
    """(3) sync asks only for last_ts+1 → boundary; an empty response is untouched."""
    a_start, a_end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=a_start, end=a_end
    )
    rec = stack["coverage"].get(SeriesKey(DATASET, SCHEMA, SYMBOL))
    stack["vendor"].fetch_ranges.clear()

    boundary = _ns("2026-01-16")
    results = stack["updater"].sync(until_ns=boundary)
    assert results[SYMBOL].status == "updated"
    (s, e) = stack["vendor"].fetch_ranges[0]
    assert s == rec.last_ts + 1
    assert e == boundary

    # A second sync to the same boundary returns nothing new → noop, registry untouched.
    rec_before = stack["coverage"].get(SeriesKey(DATASET, SCHEMA, SYMBOL))
    calls_before = stack["vendor"].fetch_calls
    results2 = stack["updater"].sync(until_ns=boundary)
    assert results2[SYMBOL].status == "noop"
    assert stack["vendor"].fetch_calls == calls_before  # nothing past the boundary
    rec_after = stack["coverage"].get(SeriesKey(DATASET, SCHEMA, SYMBOL))
    assert (rec_after.first_ts, rec_after.last_ts) == (rec_before.first_ts, rec_before.last_ts)


def test_ingest_and_sync_report_per_symbol_progress(stack):
    """``on_progress`` fires once per symbol, in order, with 1-based (index, total) — the CLI
    spinner's only liveness signal while a long multi-symbol vendor fetch is running."""
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    seen: list[tuple[str, int, int]] = []
    stack["ingestor"].ingest(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=["AAPL", "MSFT"],
        start=start,
        end=end,
        on_progress=lambda sym, i, n: seen.append((sym, i, n)),
    )
    assert seen == [("AAPL", 1, 2), ("MSFT", 2, 2)]

    seen.clear()
    stack["updater"].sync(
        until_ns=_ns("2026-01-16"), on_progress=lambda sym, i, n: seen.append((sym, i, n))
    )
    assert {sym for sym, _, _ in seen} == {"AAPL", "MSFT"}
    assert [(i, n) for _, i, n in seen] == [(1, 2), (2, 2)]


def test_t1_boundary_follows_the_et_trading_date():
    """The default sync/backfill boundary is UTC midnight of the *ET* trading date, not the
    UTC date. At 11 PM EDT the UTC calendar has already rolled to tomorrow, and a UTC-dated
    boundary crosses DataBento's live-license line (403 license_not_found for EQUS.MINI
    past the current session's ET midnight)."""
    from noctis.data.types import day_start_ns, t1_boundary_ns
    from noctis.data.updater import _default_until_ns

    late_evening = datetime(2026, 7, 18, 3, 0, tzinfo=UTC)  # 11 PM EDT, still Jul 17 in ET
    assert t1_boundary_ns(late_evening) == day_start_ns(date(2026, 7, 17))
    assert _default_until_ns(late_evening) == day_start_ns(date(2026, 7, 17))
    # Midday, the calendars agree.
    midday = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
    assert t1_boundary_ns(midday) == day_start_ns(date(2026, 7, 17))
    # Winter (EST, UTC-5): the rollover window widens to 05:00Z.
    winter_evening = datetime(2026, 1, 16, 4, 30, tzinfo=UTC)  # 11:30 PM EST, Jan 15
    assert t1_boundary_ns(winter_evening) == day_start_ns(date(2026, 1, 15))


def test_sync_empty_vendor_response_leaves_registry_untouched(stack, monkeypatch):
    """(3b) An empty vendor response advances nothing."""
    a_start, a_end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(
        dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=a_start, end=a_end
    )
    rec_before = stack["coverage"].get(SeriesKey(DATASET, SCHEMA, SYMBOL))

    from noctis.data.types import empty_bars

    monkeypatch.setattr(stack["vendor"], "fetch_bars", lambda **_kw: empty_bars())
    results = stack["updater"].sync(until_ns=_ns("2026-02-01"))
    assert results[SYMBOL].status == "noop"
    rec_after = stack["coverage"].get(SeriesKey(DATASET, SCHEMA, SYMBOL))
    assert (rec_after.first_ts, rec_after.last_ts, rec_after.row_count) == (
        rec_before.first_ts,
        rec_before.last_ts,
        rec_before.row_count,
    )
    assert rec_after.status == "idle"


# --- integrity ---------------------------------------------------------------------------


def test_integrity_flags_gap_and_repair_fetches_only_that_day(stack, lake_dir):
    """(4a) A deleted trading day is flagged as a gap; repair fetches only that slice."""
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end)

    # Delete Wednesday 2026-01-07 directly on disk (simulate a hole).
    path = stack["lake"].path_for(key)
    df = pd.read_parquet(path)
    deleted_day = to_ns("2026-01-07")
    keep = df[(df["ts_event"] < deleted_day) | (df["ts_event"] >= deleted_day + 86_400_000_000_000)]
    keep.to_parquet(path, index=False)

    report = stack["integrity"].check(key)
    assert report.gap_count == 1
    assert report.gap_days and report.gap_days[0].isoformat() == "2026-01-07"

    stack["vendor"].fetch_ranges.clear()
    calls_before = stack["vendor"].fetch_calls
    summary = stack["integrity"].repair(report)
    assert summary["gap_fetches"] == 1  # only the one flagged day
    assert stack["vendor"].fetch_calls == calls_before + 1
    after = stack["integrity"].check(key)
    assert after.gap_count == 0


def test_integrity_flags_duplicate_and_repair_compacts(stack, lake_dir):
    """(4b) A duplicated row is flagged; repair compacts it away without a fetch."""
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end)

    path = stack["lake"].path_for(key)
    df = pd.read_parquet(path)
    dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dupe.to_parquet(path, index=False)

    report = stack["integrity"].check(key)
    assert report.duplicate_count == 1

    calls_before = stack["vendor"].fetch_calls
    summary = stack["integrity"].repair(report)
    assert summary["compacted"] == 1
    assert stack["vendor"].fetch_calls == calls_before  # dedup needs no vendor call
    after = stack["integrity"].check(key)
    assert after.duplicate_count == 0


def test_manifest_flags_tampered_file(stack):
    """(6) Manifest restamped on write; manifest_ok=False when a file is tampered."""
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end)
    assert stack["lake"].manifest_matches_disk(key) is True

    # Tamper: append a row directly, bypassing the lake owner.
    path = stack["lake"].path_for(key)
    df = pd.read_parquet(path)
    extra = df.iloc[[0]].copy()
    extra["ts_event"] = extra["ts_event"] + 1
    pd.concat([df, extra], ignore_index=True).to_parquet(path, index=False)
    assert stack["lake"].manifest_matches_disk(key) is False
    assert stack["integrity"].check(key).manifest_ok is False


# --- preflight ---------------------------------------------------------------------------


def test_preflight_refuses_over_budget_and_does_not_fetch(lake_dir):
    """(5) Over-budget padded estimate → refusal with the estimate, no data fetched."""
    vendor = MockVendor(cost_per_day=1000.0)
    lake = Lake(lake_dir)
    coverage = CoverageRegistry(lake_dir / "coverage.db")
    ingestor = Ingestor(lake, coverage, CostPreflight(budget_usd=1.0), vendor)
    res: IngestResult = ingestor.ingest(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[SYMBOL],
        start=_ns("2026-01-05"),
        end=_ns("2026-01-09T23:59:59"),
    )[SYMBOL]
    assert res.status == "refused"
    assert res.padded_cost > 1.0
    assert vendor.fetch_calls == 0  # no data purchased on refusal


def test_preflight_dry_run_prices_without_fetching(stack):
    """(5b) dry_run prices the request (cost call ok) but never fetches data."""
    res = stack["ingestor"].ingest(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[SYMBOL],
        start=_ns("2026-01-05"),
        end=_ns("2026-01-09T23:59:59"),
        dry_run=True,
    )[SYMBOL]
    assert res.status == "dry_run"
    assert res.padded_cost > 0
    assert stack["vendor"].fetch_calls == 0  # priced only, nothing bought


# --- readiness ---------------------------------------------------------------------------


def test_check_symbol_ready(stack):
    """(7) check_symbol_ready false for untracked/non-idle; true only for idle+data."""
    cov = stack["coverage"]
    assert cov.check_symbol_ready("AAPL") is False  # untracked
    stack["ingestor"].ingest(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[SYMBOL],
        start=_ns("2026-01-05"),
        end=_ns("2026-01-09T23:59:59"),
    )
    assert cov.check_symbol_ready("AAPL") is True

    cov.set_status(SeriesKey(DATASET, SCHEMA, SYMBOL), "ingesting")
    assert cov.check_symbol_ready("AAPL") is False  # not idle


def test_sweep_stale_ingesting_recovers_crash(stack):
    """A crash leaving status='ingesting' is recovered to 'error' on sweep."""
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    stack["coverage"].set_status(key, "ingesting")
    n = stack["coverage"].sweep_stale_ingesting()
    assert n == 1
    assert stack["coverage"].get(key).status == "error"


# --- seam façade -------------------------------------------------------------------------


def test_marketdatalake_get_bars_reads_from_catalog_only(lake_dir):
    """The seam serves bars from the catalog and ensure_coverage triggers ingest."""
    vendor = MockVendor()
    md = MarketDataLake(lake_dir, vendor, budget_usd=1000.0, calendar="XNYS")
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")

    # Before ingest, the catalog is empty (get_bars never fetches).
    empty = md.get_bars(DATASET, SCHEMA, [SYMBOL], start, end)
    assert len(empty[SYMBOL]) == 0
    assert vendor.fetch_calls == 0

    md.ensure_coverage(DATASET, SCHEMA, [SYMBOL], start, end)
    bars = md.get_bars(DATASET, SCHEMA, [SYMBOL], start, end)
    expected = bars_for_range(SYMBOL, start, end)
    assert len(bars[SYMBOL]) == len(expected)
    assert md.check_symbol_ready(SYMBOL) is True


# --- Finding 1: sync() and repair() are budget-gated ------------------------------------


def _seed_series(lake_dir, budget=1000.0):
    """Ingest a Mon–Fri week under a generous budget; return the shared components."""
    vendor = MockVendor()
    lake = Lake(lake_dir)
    coverage = CoverageRegistry(lake_dir / "coverage.db")
    Ingestor(lake, coverage, CostPreflight(budget), vendor).ingest(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[SYMBOL],
        start=_ns("2026-01-05"),
        end=_ns("2026-01-09T23:59:59"),
    )
    return vendor, lake, coverage


def test_sync_refuses_over_budget_without_fetching(lake_dir):
    """(1a) A zero-budget preflight makes sync() fetch nothing and touch no state."""
    vendor, lake, coverage = _seed_series(lake_dir)
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    rec_before = coverage.get(key)

    updater = Updater(lake, coverage, CostPreflight(budget_usd=0.0), vendor)
    calls_before = vendor.fetch_calls
    results = updater.sync(until_ns=_ns("2026-01-16"))

    assert results[SYMBOL].status == "refused"
    assert vendor.fetch_calls == calls_before  # ZERO fetches
    rec_after = coverage.get(key)
    assert (rec_after.first_ts, rec_after.last_ts, rec_after.row_count, rec_after.status) == (
        rec_before.first_ts,
        rec_before.last_ts,
        rec_before.row_count,
        "idle",
    )


def test_repair_refuses_over_budget_without_fetching(lake_dir):
    """(1b) An over-budget preflight refuses repair: no fetch, gap left flagged."""
    vendor, lake, coverage = _seed_series(lake_dir)
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)

    # Punch a hole so a gap day is flagged.
    path = lake.path_for(key)
    df = pd.read_parquet(path)
    hole = to_ns("2026-01-07")
    keep = df[(df["ts_event"] < hole) | (df["ts_event"] >= hole + _ONE_DAY_NS)]
    keep.to_parquet(path, index=False)

    checker = IntegrityChecker(
        lake, coverage, CostPreflight(budget_usd=0.0), vendor, calendar="XNYS"
    )
    report = checker.check(key)
    assert report.gap_count == 1

    calls_before = vendor.fetch_calls
    summary = checker.repair(report)
    assert summary["refused"] is True
    assert "detail" in summary  # estimate/reason surfaced
    assert vendor.fetch_calls == calls_before  # ZERO fetches
    # State unchanged: the gap is still there on a fresh check.
    assert checker.check(key).gap_count == 1


# --- Finding 3: repair converges on confirmed-empty (holiday) days ----------------------


def test_empty_gap_day_is_marked_known_empty_and_converges(stack):
    """(3) A gap day the vendor returns empty is recorded and never re-flagged/re-fetched."""
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end)

    # Delete Wednesday 2026-01-07 on disk → the checker flags it as a gap.
    path = stack["lake"].path_for(key)
    df = pd.read_parquet(path)
    hole = to_ns("2026-01-07")
    keep = df[(df["ts_event"] < hole) | (df["ts_event"] >= hole + _ONE_DAY_NS)]
    keep.to_parquet(path, index=False)

    report = stack["integrity"].check(key)
    assert report.gap_count == 1
    assert report.gap_days[0] == date(2026, 1, 7)

    # The vendor has no bars for that day (a real holiday the weekday fallback can't know).
    original_fetch = stack["vendor"].fetch_bars
    fetched_days: list[date] = []

    def _empty_for_holiday(**kw):
        fetched_days.append(ns_to_date(kw["start"]))
        return empty_bars()

    stack["vendor"].fetch_bars = _empty_for_holiday
    summary = stack["integrity"].repair(report)
    stack["vendor"].fetch_bars = original_fetch  # restore

    assert summary["gap_fetches"] == 1
    assert summary["empty_days"] == 1
    assert fetched_days == [date(2026, 1, 7)]
    assert stack["coverage"].known_empty_days(key) == {date(2026, 1, 7)}

    # Convergence: the next check no longer flags the day.
    after = stack["integrity"].check(key)
    assert after.gap_count == 0
    assert date(2026, 1, 7) not in after.gap_days

    # A second repair issues no fetch for the now-known-empty day.
    calls_before = stack["vendor"].fetch_calls
    stack["integrity"].repair(after)
    assert stack["vendor"].fetch_calls == calls_before


# --- Finding 4: date-only --end is inclusive of the end day -----------------------------


def test_to_ns_end_inclusive_covers_full_day():
    """(4a) A date-only end maps to end-of-day ns; timed/int values pass through."""
    from noctis.data.types import day_bounds_ns

    assert to_ns_end_inclusive("2024-01-10") == day_bounds_ns(date(2024, 1, 10))[1]
    assert to_ns_end_inclusive("2024-01-10") > to_ns("2024-01-10")
    assert to_ns_end_inclusive(date(2024, 1, 10)) == day_bounds_ns(date(2024, 1, 10))[1]
    # A value carrying an explicit time is unchanged.
    assert to_ns_end_inclusive("2024-01-10T14:30:00") == to_ns("2024-01-10T14:30:00")
    assert to_ns_end_inclusive(123) == 123


def test_date_only_end_includes_end_day_and_is_not_a_gap(lake_dir):
    """(4b) Ingesting with a date-only --end includes that day's bar; no gap flagged."""
    vendor = MockVendor()
    md = MarketDataLake(lake_dir, vendor, budget_usd=1000.0, calendar="XNYS")
    start = to_ns("2026-01-05")  # Monday
    end = to_ns_end_inclusive("2026-01-09")  # Friday, inclusive of the whole day

    md.ensure_coverage(DATASET, SCHEMA, [SYMBOL], start, end)
    bars = md.get_bars(DATASET, SCHEMA, [SYMBOL], start, end)[SYMBOL]
    present = {ns_to_date(int(ts)) for ts in bars["ts_event"]}
    assert date(2026, 1, 9) in present  # the Friday bar (14:00 UTC) is included

    report = md.check(DATASET, SCHEMA, SYMBOL)
    assert date(2026, 1, 9) not in report.gap_days
    assert report.gap_count == 0


# --- Finding 6: Parquet writes are atomic (no torn file on crash) -----------------------


def test_parquet_write_is_atomic_on_crash(stack, monkeypatch):
    """(6) A crash during the file replace leaves the prior Parquet intact and readable."""
    key = SeriesKey(DATASET, SCHEMA, SYMBOL)
    start, end = _ns("2026-01-05"), _ns("2026-01-09T23:59:59")
    stack["ingestor"].ingest(dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], start=start, end=end)

    path = stack["lake"].path_for(key)
    before = pd.read_parquet(path)
    rows_before = len(before)

    def _boom(self, target):
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(Path, "replace", _boom)

    extra = bars_for_range(SYMBOL, _ns("2026-01-12"), _ns("2026-01-16T23:59:59"))
    with pytest.raises(OSError):
        stack["lake"].write(key, extra)

    monkeypatch.undo()  # restore Path.replace for the read below
    after = pd.read_parquet(path)  # must not be torn
    assert len(after) == rows_before
    assert after.equals(before)
