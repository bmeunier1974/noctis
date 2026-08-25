"""The session's two ticker surfaces, in the data layer that owns them (story #344).

``trading_roster`` is what a session *trades*: the config seed, order preserved, plus every
ready symbol the lake already tracks. ``research_focus`` is what it *tells the model about*:
the fit-set/symbol-holdout window over that roster, plus the mandate's declared names, capped
at ``research.focus_size``. The two are deliberately different things (rule 5).

Both read the lake through the ``MarketData`` seam's own ``coverage_records()``. A lake that
cannot answer it is an error, never an invented empty answer — this layer does not probe for
the surface and so cannot fabricate a roster out of a missing attribute.
"""

from __future__ import annotations

import pytest

from noctis.config.settings import Settings
from noctis.data.coverage import CoverageRecord
from noctis.data.universe import research_focus, trading_roster
from noctis.research import Mandate

DATASET = "EQUS.MINI"
SCHEMA = "ohlcv-1m"


def _record(symbol: str, *, status: str = "idle", rows: int = 100) -> CoverageRecord:
    return CoverageRecord(
        dataset=DATASET,
        schema=SCHEMA,
        symbol=symbol,
        first_ts=0,
        last_ts=1,
        row_count=rows,
        status=status,
        last_update=None,
        error_msg=None,
    )


class _Lake:
    """The two ``MarketData`` methods these helpers call, and nothing else."""

    def __init__(self, ready: tuple[str, ...] = (), records: tuple[CoverageRecord, ...] = ()):
        self._ready = {s.upper() for s in ready}
        self._records = list(records)

    def check_symbol_ready(self, symbol: str, dataset=None, schema=None) -> bool:
        return symbol.upper() in self._ready

    def coverage_records(self) -> list[CoverageRecord]:
        return list(self._records)


def _mandate(symbols: list[str]) -> Mandate:
    return Mandate(
        text="x", source="cli", summary="x", references=[], config_overrides={}, symbols=symbols
    )


# ── the trading roster: the growing universe ────────────────────────────────────────────────
def test_a_lake_tracking_nothing_leaves_the_roster_at_the_config_seed():
    settings = Settings(universe=["AAA", "BBB"])

    assert trading_roster(settings, _Lake()) == ["AAA", "BBB"]


def test_ready_discoveries_follow_the_config_seed_in_sorted_order():
    """Config order first, so the fit set stays stable as discoveries accumulate; the lake's
    own tracked names — anything ``ensure_data`` ever fetched — follow, sorted."""
    settings = Settings(universe=["BBB", "AAA"])
    lake = _Lake(records=(_record("ZZZ"), _record("MMM")))

    assert trading_roster(settings, lake) == ["BBB", "AAA", "MMM", "ZZZ"]


def test_a_tracked_symbol_already_on_the_seed_is_not_repeated():
    settings = Settings(universe=["AAA"])
    lake = _Lake(records=(_record("aaa"), _record("ZZZ")))

    assert trading_roster(settings, lake) == ["AAA", "ZZZ"]


def test_a_record_that_is_not_idle_or_holds_no_bars_stays_off_the_roster():
    """Mid-ingest, errored and empty series are tracked but not tradeable."""
    settings = Settings(universe=["AAA"])
    lake = _Lake(
        records=(
            _record("ERR", status="error"),
            _record("BUSY", status="ingesting"),
            _record("NIL", rows=0),
            _record("ZZZ"),
        )
    )

    assert trading_roster(settings, lake) == ["AAA", "ZZZ"]


def test_the_roster_never_shrinks_under_an_unready_seed_symbol():
    """Readiness filters the *consumers* of the roster, never the roster itself — a champion
    trades discovered symbols, so this list must not narrow behind it."""
    settings = Settings(universe=["AAA", "BBB"])
    lake = _Lake(ready=(), records=(_record("ZZZ"),))

    assert trading_roster(settings, lake) == ["AAA", "BBB", "ZZZ"]


def test_a_lake_that_cannot_list_its_coverage_is_an_error_not_an_empty_roster():
    """No probe: a seam without ``coverage_records`` is a broken lake, and saying so beats
    answering "the config seed" for a lake that may hold far more."""

    class _NotALake:
        def check_symbol_ready(self, symbol, dataset=None, schema=None) -> bool:
            return True

    with pytest.raises(AttributeError):
        trading_roster(Settings(universe=["AAA"]), _NotALake())


# ── the research focus: the capped, prompt-facing enumeration ───────────────────────────────
def test_the_focus_is_the_ready_fit_and_holdout_window_of_the_roster():
    settings = Settings(
        universe=["AAA", "BBB", "CCC", "DDD", "EEE"],
        research={"fit_set_size": 2, "symbol_holdout_size": 1, "focus_size": 4},
    )
    lake = _Lake(ready=("AAA", "BBB", "CCC", "DDD"))  # EEE is not ready

    assert research_focus(settings, lake) == ["AAA", "BBB", "CCC"]


def test_the_focus_is_empty_when_no_roster_symbol_is_ready():
    settings = Settings(universe=["AAA", "BBB"])
    lake = _Lake(ready=())

    assert research_focus(settings, lake) == []


def test_a_discovered_ready_symbol_joins_the_focus_window():
    settings = Settings(
        universe=["AAA"],
        research={"fit_set_size": 1, "symbol_holdout_size": 1, "focus_size": 4},
    )
    lake = _Lake(ready=("AAA", "ZZZ"), records=(_record("ZZZ"),))

    assert research_focus(settings, lake) == ["AAA", "ZZZ"]


def test_mandate_symbols_join_after_the_window_deduped():
    """A mandate steers the search (rule 5): its symbols enter the focus set — even unready
    ones, which consumers filter themselves — and never the roster."""
    settings = Settings(
        universe=["AAA", "BBB", "CCC"],
        research={"fit_set_size": 2, "symbol_holdout_size": 1, "focus_size": 5},
    )
    lake = _Lake(ready=("AAA", "BBB", "CCC"))

    focus = research_focus(settings, lake, _mandate(["ccc", "QQQ"]))

    assert focus == ["AAA", "BBB", "CCC", "QQQ"]
    assert trading_roster(settings, lake) == ["AAA", "BBB", "CCC"]


def test_the_cap_is_the_prompt_size_lever():
    """``focus_size`` is what stops every future prompt growing with every discovery."""
    settings = Settings(
        universe=["AAA", "BBB", "CCC"],
        research={"fit_set_size": 2, "symbol_holdout_size": 1, "focus_size": 2},
    )
    lake = _Lake(ready=("AAA", "BBB", "CCC"))
    mandate = _mandate(["QQQ"])

    assert research_focus(settings, lake, mandate) == ["AAA", "BBB"]

    settings.research.focus_size = 4
    assert research_focus(settings, lake, mandate) == ["AAA", "BBB", "CCC", "QQQ"]


def test_no_mandate_stops_the_focus_at_the_window():
    settings = Settings(
        universe=["AAA", "BBB"],
        research={"fit_set_size": 1, "symbol_holdout_size": 0, "focus_size": 4},
    )
    lake = _Lake(ready=("AAA", "BBB"))

    assert research_focus(settings, lake, None) == ["AAA"]
