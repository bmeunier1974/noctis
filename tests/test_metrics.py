"""``reporting/metrics.py`` — the realised record's numbers, hand-checked (story #142, epic #126).

Every metric here is pinned against a **hand-computed** literal: the derivation is written out in
the test (mean, variance, the annualisation factor, the closed form) and the expected value is a
number, never a second call to the formula under test. A test that re-implements the implementation
proves only that two copies of one mistake agree.

The two numbers that carry the most weight — the Probabilistic and the Deflated Sharpe Ratio — are
checked twice over: once against their published closed forms (Bailey & López de Prado), and once
against an **independent** published result that must agree with them (Lo's standard error of the
Sharpe ratio; the exact expected maximum of two standard normals). Sources are cited per test.

The module itself is **pure** — no I/O, no clock, no configuration — and a structural test enforces
that by AST, the way ``tests/test_run_record.py`` does for the record builder. Series and trades
arrive as data, so every number below is computable from a fixture written by hand.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from statistics import NormalDist

import pytest

from noctis.reporting import metrics
from noctis.reporting.metrics import DailySession, TradeFill

METRICS_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/reporting/metrics.py"

# ── the worked series ──────────────────────────────────────────────────────────────────────
#
# Five daily returns, chosen so every moment is exact in decimal arithmetic:
#
#   r        = [+0.02, -0.01, +0.01, -0.02, +0.05]
#   mean     = 0.05 / 5                                     = 0.01
#   d        = r - mean = [+0.01, -0.02, 0.00, -0.03, +0.04]
#   Σd²      = 0.0001 + 0.0004 + 0 + 0.0009 + 0.0016        = 0.0030
#   var(n-1) = 0.0030 / 4                                   = 0.00075
#   std      = √0.00075                                     = 0.027386127875258306
#
# The equity curve below is 100 000 compounded by those returns.
RETURNS = [0.02, -0.01, 0.01, -0.02, 0.05]
EQUITY = [100000.0, 102000.0, 100980.0, 101989.8, 99950.004, 104947.5042]
DATES = [
    "2026-07-27",
    "2026-07-28",
    "2026-07-29",
    "2026-07-30",
    "2026-07-31",
    "2026-08-03",
]

# std × √252 = 0.027386127875258306 × 15.874507866387544
ANNUAL_VOL = 0.4347413023856832
# per-period Sharpe = 0.01 / 0.027386127875258306 = 0.3651483716701107, annualised × √252
PER_PERIOD_SHARPE = 0.3651483716701107
ANNUAL_SHARPE = 5.796550698475775
# downside deviation over the WHOLE sample (Sortino & Price 1994):
#   √((0.01² + 0.02²) / 5) = √(0.0005 / 5) = √0.0001 = 0.01
# Sortino = 0.01 / 0.01 × √252 = √252
ANNUAL_SORTINO = 15.874507866387544
# central moments about the mean, divisor n:
#   m2 = 0.0030 / 5 = 0.0006     m3 = 0.00003 / 5 = 0.000006     m4 = 0.00000354 / 5 = 7.08e-7
#   skew     = m3 / m2^1.5 = 6e-6 / 1.46969385e-5
#   kurtosis = m4 / m2²    = 7.08e-7 / 3.6e-7   (non-excess; the record publishes excess)
SKEW = 0.4082482904638631
KURTOSIS_NON_EXCESS = 1.9666666666666666
EXCESS_KURTOSIS = -1.0333333333333334


def _points(equity=None, dates=None) -> list[tuple[str, float]]:
    return list(zip(dates or DATES, equity or EQUITY, strict=True))


def _sessions(equity=None, dates=None, **overrides) -> list[DailySession]:
    return [
        DailySession(date=date, equity=value, **overrides) for date, value in _points(equity, dates)
    ]


# ── returns, volatility and the risk-adjusted ratios ───────────────────────────────────────


def test_daily_returns_are_read_off_the_equity_marks():
    assert metrics.returns(_points()) == pytest.approx(RETURNS, abs=1e-12)


def test_a_curve_with_one_mark_has_no_returns_and_no_ratios():
    """One night is not a track record. Every ratio over it is ``null``, never zero."""
    one = _points(equity=[100000.0], dates=["2026-07-27"])

    assert metrics.returns(one) == []
    assert metrics.sharpe([]) is None
    assert metrics.sortino([]) is None
    assert metrics.annual_volatility([]) is None
    assert metrics.total_return(one) is None


def test_total_return_is_the_end_over_the_start():
    # 104947.5042 / 100000 - 1
    assert metrics.total_return(_points()) == pytest.approx(0.04947504200000008, rel=1e-12)


def test_cagr_compounds_the_total_return_over_the_curves_calendar_span():
    """365 calendar days between the two marks, so the CAGR *is* the total return."""
    year = _points(equity=[100000.0, 110000.0], dates=["2026-01-01", "2027-01-01"])

    assert metrics.cagr(year) == pytest.approx(0.10, rel=1e-12)


def test_cagr_annualises_a_span_shorter_than_a_year():
    """+10% in 182 days annualises to 1.1^(365/182) − 1 = 21.06338215370842%."""
    half = _points(equity=[100000.0, 110000.0], dates=["2026-01-01", "2026-07-02"])

    assert metrics.cagr(half) == pytest.approx(0.2106338215370842, rel=1e-12)


def test_cagr_of_a_curve_that_spans_no_time_is_null_not_infinite():
    same_day = _points(equity=[100000.0, 110000.0], dates=["2026-01-01", "2026-01-01"])

    assert metrics.cagr(same_day) is None


def test_annualised_volatility_is_the_sample_deviation_times_root_252():
    assert metrics.annual_volatility(RETURNS) == pytest.approx(ANNUAL_VOL, rel=1e-12)


def test_sharpe_is_the_mean_over_the_deviation_annualised():
    assert metrics.sharpe(RETURNS) == pytest.approx(ANNUAL_SHARPE, rel=1e-12)
    assert metrics.sharpe(RETURNS, annualised=False) == pytest.approx(PER_PERIOD_SHARPE, rel=1e-12)


def test_sortino_divides_by_the_full_sample_downside_deviation():
    """Deliberately *not* the negative-only denominator ``backtest/scorecard.py`` ranks gates
    with: this module may never drift into gate math, so it states its own convention."""
    assert metrics.sortino(RETURNS) == pytest.approx(ANNUAL_SORTINO, rel=1e-12)


def test_a_curve_that_never_fell_has_no_sortino_rather_than_an_infinity():
    assert metrics.sortino([0.01, 0.02, 0.03]) is None


def test_a_flat_curve_has_no_sharpe_rather_than_a_zero_division():
    assert metrics.sharpe([0.0, 0.0, 0.0]) is None


def test_skew_and_excess_kurtosis_are_the_third_and_fourth_moments():
    assert metrics.skew(RETURNS) == pytest.approx(SKEW, rel=1e-12)
    assert metrics.kurtosis(RETURNS) == pytest.approx(EXCESS_KURTOSIS, rel=1e-12)


# ── drawdown: depth AND duration ───────────────────────────────────────────────────────────


def test_drawdown_reports_depth_dates_and_duration_and_whether_it_recovered():
    """Peak 120 on the 28th, trough 90 on the 29th, back above the peak on the 31st.

    depth = 90/120 − 1 = −0.25; the drawdown ran from the peak to the recovery, 3 calendar days.
    """
    curve = _points(
        equity=[100.0, 120.0, 90.0, 110.0, 130.0],
        dates=["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"],
    )

    dd = metrics.drawdown(curve)

    assert dd.depth == pytest.approx(-0.25, rel=1e-12)
    assert dd.peak_date == "2026-07-28"
    assert dd.trough_date == "2026-07-29"
    assert dd.recovered is True
    assert dd.days == 3


def test_an_unrecovered_drawdown_is_measured_to_the_last_mark_and_says_so():
    curve = _points(
        equity=[100.0, 120.0, 90.0, 95.0],
        dates=["2026-07-27", "2026-07-28", "2026-07-29", "2026-08-07"],
    )

    dd = metrics.drawdown(curve)

    assert dd.depth == pytest.approx(-0.25, rel=1e-12)
    assert dd.recovered is False
    assert dd.days == 10  # 07-28 → 08-07, still under water at the last mark


def test_a_curve_that_only_rose_has_no_drawdown():
    dd = metrics.drawdown(_points(equity=[100.0, 110.0], dates=["2026-07-27", "2026-07-28"]))

    assert dd.depth == 0.0 and dd.days == 0 and dd.trough_date is None


def test_calmar_and_recovery_factor_divide_growth_by_the_worst_fall():
    """Calmar = CAGR / |max drawdown| = 0.10 / 0.25 = 0.4;
    recovery factor = total return / |max drawdown| = 0.30 / 0.25 = 1.2."""
    assert metrics.calmar(0.10, -0.25) == pytest.approx(0.4, rel=1e-12)
    assert metrics.recovery_factor(0.30, -0.25) == pytest.approx(1.2, rel=1e-12)


def test_calmar_and_recovery_factor_are_null_when_there_was_no_drawdown():
    """A zero denominator is not an infinite ratio — it is a question nobody can answer yet."""
    assert metrics.calmar(0.10, 0.0) is None
    assert metrics.recovery_factor(0.10, 0.0) is None


# ── PSR and DSR: checked against the published closed forms ────────────────────────────────


def test_the_probabilistic_sharpe_ratio_matches_the_published_closed_form():
    """PSR(SR*) = Φ[ (ŜR − SR*)·√(n−1) / √(1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) ]

    Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier", *Journal of Risk*
    15(2), eq. (3) — with ŜR the **per-observation** Sharpe, γ₃ the skew and γ₄ the
    (non-excess) kurtosis of the same series.

    Hand-evaluated on the worked series above:
      ŜR      = 0.3651483716701107,  n = 5,  γ₃ = 0.4082482904638631,  γ₄ = 1.9666666666666666
      ŜR²     = 0.0001 / 0.00075                       = 0.13333333333
      γ₃·ŜR   = 0.4082482904638631 × 0.3651483716701107 = 0.14907119849
      (γ₄−1)/4·ŜR² = 0.24166666667 × 0.13333333333      = 0.03222222222
      radicand = 1 − 0.14907119849 + 0.03222222222      = 0.88315102373
      z        = 0.3651483716701107 × 2 / √0.88315102373 = 0.7771105…
      Φ(z)                                              = 0.781452734425495
    """
    psr = metrics.psr(PER_PERIOD_SHARPE, n_observations=5, skew=SKEW, kurtosis=KURTOSIS_NON_EXCESS)

    assert psr == pytest.approx(0.781452734425495, rel=1e-12)


def test_the_psr_agrees_with_los_standard_error_of_the_sharpe_ratio():
    """The independent cross-check. For i.i.d. **normal** returns (γ₃ = 0, γ₄ = 3), Lo (2002),
    "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4), gives

        SE(ŜR) = √( (1 + ŜR²/2) / n )

    so the probability that the true Sharpe exceeds zero is Φ(ŜR / SE). With ŜR = 0.5 and
    n = 10: SE = √(1.125/10) = 0.33541019662496846, ŜR/SE = 1.4907119849998598,
    Φ = 0.9319814359429281.

    Bailey & López de Prado's PSR is that same statistic with the finite-sample √(n−1) in place of
    Lo's √n: Φ(0.5·3/√1.125) = Φ(1.4142135623730951) = 0.9213503964748575. Two independently
    published derivations agreeing to within the one degree of freedom that separates them — which
    is what tells us the implementation is *the* PSR rather than a formula that resembles it.
    """
    psr = metrics.psr(0.5, n_observations=10, skew=0.0, kurtosis=3.0)
    lo = NormalDist().cdf(0.5 / math.sqrt(1.125 / 10))

    assert psr == pytest.approx(0.9213503964748575, rel=1e-12)
    assert lo == pytest.approx(0.9319814359429281, rel=1e-12)
    assert abs(psr - lo) < 0.011


def test_a_psr_needs_at_least_two_observations():
    assert metrics.psr(0.5, n_observations=1, skew=0.0, kurtosis=3.0) is None


def test_the_expected_maximum_sharpe_matches_the_published_approximation():
    """E[max ŜR] ≈ √V · [ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality", *Journal of Portfolio Management* 40(5), eq. (5),
    with γ the Euler–Mascheroni constant (0.5772156649015329).

    Hand-evaluated at unit variance:
      N = 1000: Z⁻¹(0.999) = 3.090232306167813, Z⁻¹(1 − 1/2718.28…) = 3.375895391061807
                E[max] = 0.4227843350984671×3.090232306167813
                       + 0.5772156649015329×3.375895391061807 = 3.255121513652723
      N = 100 : 2.5306028932016846        N = 10: 1.57459830134575
    """
    assert metrics.expected_max_sharpe(1000, 1.0) == pytest.approx(3.255121513652723, rel=1e-12)
    assert metrics.expected_max_sharpe(100, 1.0) == pytest.approx(2.5306028932016846, rel=1e-12)
    assert metrics.expected_max_sharpe(10, 1.0) == pytest.approx(1.57459830134575, rel=1e-12)


def test_the_expected_maximum_is_close_to_the_exact_value_for_two_trials():
    """The independent cross-check on the *other* half of DSR. The expected maximum of two
    independent standard normals is exactly 1/√π = 0.5641895835477563 (a standard order-statistics
    result). Bailey & López de Prado's approximation gives 0.5197553442805939 — within 8%, which
    is what an approximation of that form should be, and is how we know the implementation is the
    published expression rather than something else."""
    approximate = metrics.expected_max_sharpe(2, 1.0)

    assert approximate == pytest.approx(0.5197553442805939, rel=1e-12)
    assert abs(approximate - 1 / math.sqrt(math.pi)) < 0.05


def test_a_single_trial_deflates_by_nothing():
    """One trial is no selection at all, so the deflation benchmark is zero and DSR = PSR(0)."""
    assert metrics.expected_max_sharpe(1, 1.0) == 0.0


def test_the_deflated_sharpe_is_the_psr_taken_against_the_expected_maximum():
    """DSR = PSR(SR₀) with SR₀ = E[max ŜR] over the run's trial count (Bailey & López de Prado
    2014, eq. (6)). On the worked series, with the estimator's own variance as V:

      V   = (1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) / (n−1) = 0.88315102373 / 4 = 0.22078775593055908
      √V  = 0.4698805762431121
      N = 10   → SR₀ = 0.4698805762431121 × 1.57459830134575 = 0.7398731571877665 → DSR = 0.2125834…
      N = 1000 → SR₀ = 0.4698805762431121 × 3.255121513652723 = 1.5295183725764927
               → DSR = 0.0066058…
    """
    ten = metrics.deflated_sharpe(
        PER_PERIOD_SHARPE, n_observations=5, skew=SKEW, kurtosis=KURTOSIS_NON_EXCESS, n_trials=10
    )
    thousand = metrics.deflated_sharpe(
        PER_PERIOD_SHARPE, n_observations=5, skew=SKEW, kurtosis=KURTOSIS_NON_EXCESS, n_trials=1000
    )

    assert ten == pytest.approx(0.21258342416548337, rel=1e-12)
    assert thousand == pytest.approx(0.006605823487458862, rel=1e-12)


def test_more_trials_always_deflate_further():
    """The property the whole number exists for: the more param sets a run tried, the higher the
    Sharpe it had to clear by luck alone — so a run cannot search its way to a better DSR."""
    deflated = [
        metrics.deflated_sharpe(
            PER_PERIOD_SHARPE,
            n_observations=5,
            skew=SKEW,
            kurtosis=KURTOSIS_NON_EXCESS,
            n_trials=n,
        )
        for n in (1, 2, 10, 100, 1000, 3180)
    ]

    assert deflated == sorted(deflated, reverse=True)
    assert deflated[0] == pytest.approx(0.781452734425495, rel=1e-12)  # N=1 ⇒ PSR(0)


def test_a_deflated_sharpe_without_a_trial_count_is_null():
    """The count is the whole point: a DSR that quietly assumed one trial would overstate every
    run that never journaled its search."""
    assert (
        metrics.deflated_sharpe(
            PER_PERIOD_SHARPE,
            n_observations=5,
            skew=SKEW,
            kurtosis=KURTOSIS_NON_EXCESS,
            n_trials=None,
        )
        is None
    )


# ── round trips: what a "trade" is for the trade statistics ────────────────────────────────


def _fill(symbol, side, qty, price, fees=0.0, ts=None, champion=None) -> TradeFill:
    return TradeFill(
        ts=ts,
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        fees_usd=fees,
        champion=champion,
    )


def test_a_round_trip_is_flat_to_flat_and_its_pnl_is_the_cash_the_account_gained():
    """Bought 10 @ 100 (fee 1.00), sold 10 @ 110 (fee 1.10):
    cash = −1000 − 1.00 + 1100 − 1.10 = +97.90, which is the net P&L including both fees."""
    trips = metrics.round_trips(
        [
            _fill("AAPL", "BUY", 10, 100.0, fees=1.0, ts="2026-07-27T14:31:00.000Z", champion="a"),
            _fill("AAPL", "SELL", 10, 110.0, fees=1.1, ts="2026-07-27T19:55:00.000Z"),
        ]
    )

    assert len(trips) == 1
    trip = trips[0]
    assert trip.pnl_usd == pytest.approx(97.9, rel=1e-12)
    assert trip.fees_usd == pytest.approx(2.1, rel=1e-12)
    assert trip.symbol == "AAPL" and trip.champion == "a"
    assert trip.opened == "2026-07-27T14:31:00.000Z"
    assert trip.closed == "2026-07-27T19:55:00.000Z"


def test_a_short_round_trip_realises_the_fall():
    """Sold 5 @ 100, bought back 5 @ 90, no fees: cash = +500 − 450 = +50."""
    trips = metrics.round_trips([_fill("TSLA", "SELL", 5, 100.0), _fill("TSLA", "BUY", 5, 90.0)])

    assert [t.pnl_usd for t in trips] == pytest.approx([50.0], rel=1e-12)


def test_a_position_still_open_is_not_a_trade_yet():
    """Trade statistics are about *closed* trades — an open position's P&L is unrealised and
    already in the equity curve. Counting it as a win would double-count it."""
    trips = metrics.round_trips([_fill("AAPL", "BUY", 10, 100.0)])

    assert trips == []


def test_a_fill_that_flips_through_zero_closes_one_trip_and_opens_the_next():
    """Long 10 @ 100, then sell 15 @ 110: 10 close the long (+100), 5 open a short.
    The closing leg's fee is split by the quantity it closed — 2.0 of the 3.0 charged."""
    trips = metrics.round_trips(
        [
            _fill("AAPL", "BUY", 10, 100.0),
            _fill("AAPL", "SELL", 15, 110.0, fees=3.0),
            _fill("AAPL", "BUY", 5, 105.0),
        ]
    )

    assert [t.symbol for t in trips] == ["AAPL", "AAPL"]
    assert trips[0].pnl_usd == pytest.approx(100.0 - 2.0, rel=1e-12)
    # the short: sold 5 @ 110 (1.0 of the fee), bought back 5 @ 105 → +25 − 1.0
    assert trips[1].pnl_usd == pytest.approx(24.0, rel=1e-12)


def test_round_trips_of_different_symbols_never_net_against_each_other():
    trips = metrics.round_trips(
        [
            _fill("AAPL", "BUY", 10, 100.0),
            _fill("TSLA", "BUY", 1, 200.0),
            _fill("AAPL", "SELL", 10, 90.0),
            _fill("TSLA", "SELL", 1, 260.0),
        ]
    )

    assert {t.symbol: round(t.pnl_usd, 6) for t in trips} == {"AAPL": -100.0, "TSLA": 60.0}


# ── trade statistics ───────────────────────────────────────────────────────────────────────


def _trips(*pnls: float) -> list[metrics.RoundTrip]:
    fills: list[TradeFill] = []
    for i, pnl in enumerate(pnls):
        symbol = f"S{i}"
        fills.append(_fill(symbol, "BUY", 1, 100.0, champion="momo" if i % 2 else "mean_rev"))
        fills.append(_fill(symbol, "SELL", 1, 100.0 + pnl))
    return metrics.round_trips(fills)


def test_the_trade_statistics_are_the_practitioner_set():
    """Four wins (+60, +40, +20, +80) and two losses (−30, −10) over six closed trades:

    win rate     = 4/6 = 0.6666666666666666      loss rate = 2/6 = 0.3333333333333333
    gross profit = 200      gross loss = 40      profit factor = 5.0
    expectancy   = (200 − 40) / 6 = 26.666666666666668
    avg win      = 50       avg loss = −20       payoff ratio = 2.5
    """
    stats = metrics.trade_stats(_trips(60.0, -30.0, 40.0, 20.0, -10.0, 80.0))

    assert stats["count"] == 6
    assert stats["win_rate"] == pytest.approx(0.6666666666666666, rel=1e-12)
    assert stats["loss_rate"] == pytest.approx(0.3333333333333333, rel=1e-12)
    assert stats["profit_factor"] == pytest.approx(5.0, rel=1e-12)
    assert stats["expectancy_usd"] == pytest.approx(26.666666666666668, rel=1e-12)
    assert stats["avg_win_usd"] == pytest.approx(50.0, rel=1e-12)
    assert stats["avg_loss_usd"] == pytest.approx(-20.0, rel=1e-12)
    assert stats["payoff_ratio"] == pytest.approx(2.5, rel=1e-12)


def test_a_run_that_never_lost_reports_null_ratios_rather_than_infinities():
    stats = metrics.trade_stats(_trips(10.0, 20.0))

    assert stats["profit_factor"] is None
    assert stats["payoff_ratio"] is None
    assert stats["avg_loss_usd"] is None
    assert stats["win_rate"] == pytest.approx(1.0, rel=1e-12)


def test_no_closed_trades_at_all_leaves_every_trade_statistic_null():
    stats = metrics.trade_stats([])

    assert stats["count"] == 0
    assert stats["win_rate"] is None and stats["expectancy_usd"] is None


def test_the_trade_log_is_attributable_to_the_champion_that_opened_each_position():
    """The point of champion attribution: "which of my champions actually made money" is a
    question the record answers without joining anything."""
    stats = metrics.trade_stats(_trips(60.0, -30.0, 40.0, 20.0, -10.0, 80.0))

    assert set(stats["by_champion"]) == {"momo", "mean_rev"}
    # the odd-indexed trips are momo's: −30, +20, +80 → 70 over 3 trades
    assert stats["by_champion"]["momo"]["pnl_usd"] == pytest.approx(70.0, rel=1e-12)
    assert stats["by_champion"]["momo"]["count"] == 3
    assert stats["by_champion"]["mean_rev"]["pnl_usd"] == pytest.approx(90.0, rel=1e-12)


# ── exposure, turnover, monthly returns ────────────────────────────────────────────────────


def test_exposure_is_the_share_of_sessions_the_account_was_in_the_market():
    """Three sessions: one traded, one held a carried position over the close, one flat.
    Exposure = 2/3 = 0.6666666666666666."""
    sessions = [
        DailySession(date="2026-07-27", equity=100.0, fills=(_fill("AAPL", "BUY", 1, 100.0),)),
        DailySession(date="2026-07-28", equity=100.0, positions_end={"AAPL": 1.0}),
        DailySession(date="2026-07-29", equity=100.0),
    ]

    assert metrics.exposure(sessions) == pytest.approx(0.6666666666666666, rel=1e-12)


def test_turnover_is_the_traded_notional_against_the_equity_it_was_traded_on():
    """Two sessions, 1 000 and 500 of notional traded, both marked at 100 000 equity:
    mean daily notional 750 / mean equity 100 000 = 0.0075."""
    sessions = [
        DailySession(date="2026-07-27", equity=100000.0, fills=(_fill("A", "BUY", 10, 100.0),)),
        DailySession(date="2026-07-28", equity=100000.0, fills=(_fill("A", "SELL", 5, 100.0),)),
    ]

    assert metrics.turnover(sessions) == pytest.approx(0.0075, rel=1e-12)


def test_monthly_returns_compound_within_each_calendar_month():
    """July closes at 99950.004 from 100000 (−0.049996%), August at 104947.5042 from 99950.004
    (+5.00%). Each month is measured from the *last mark of the previous month*, so the months
    compound back to the total return."""
    monthly = metrics.monthly_returns(_points())

    assert set(monthly) == {"2026-07", "2026-08"}
    assert monthly["2026-07"] == pytest.approx(-0.00049996, rel=1e-9)
    assert monthly["2026-08"] == pytest.approx(0.05, rel=1e-12)


# ── benchmark-relative statistics ──────────────────────────────────────────────────────────


def test_the_benchmark_statistics_are_the_regression_of_one_series_on_the_other():
    """Strategy [+0.02, −0.01, +0.03, 0.00] against benchmark [+0.01, −0.02, +0.02, +0.01]:

    mean_s = 0.01, mean_b = 0.005
    cov  = Σ(s−s̄)(b−b̄)/(n−1) = 0.0008/3 = 0.0002666666666666666
    var_b = 0.0009/3          = 0.0003
    beta  = 0.8888888888888886
    alpha = (0.01 − beta×0.005) × 252 = 1.4000000000000003 → 140.00000000000003 %
    active = [+0.01, +0.01, +0.01, −0.01], std = 0.01 → TE = 0.01×√252 = 15.874507866387544 %
    IR    = 0.005 / 0.01 × √252 = 7.93725393319377
    corr  = cov / √(var_s·var_b) = 0.8432740427115677
    """
    stats = metrics.benchmark_stats(
        [0.02, -0.01, 0.03, 0.00],
        [0.01, -0.02, 0.02, 0.01],
    )

    assert stats["beta"] == pytest.approx(0.8888888888888886, rel=1e-12)
    assert stats["alpha_pct"] == pytest.approx(140.00000000000003, rel=1e-12)
    assert stats["tracking_error_pct"] == pytest.approx(15.874507866387544, rel=1e-12)
    assert stats["information_ratio"] == pytest.approx(7.93725393319377, rel=1e-12)
    assert stats["correlation"] == pytest.approx(0.8432740427115677, rel=1e-12)


def test_a_benchmark_that_never_moved_has_no_beta_and_no_correlation():
    stats = metrics.benchmark_stats([0.01, -0.01, 0.02], [0.0, 0.0, 0.0])

    assert stats["beta"] is None and stats["correlation"] is None
    assert stats["alpha_pct"] is not None  # the excess return is still a number


def test_benchmark_statistics_need_two_aligned_observations():
    assert metrics.benchmark_stats([0.01], [0.02])["beta"] is None


# ── the assembled performance block ────────────────────────────────────────────────────────


def test_performance_assembles_every_section_from_the_sessions_it_was_handed():
    block = metrics.performance(_sessions(), n_trials=1000)

    assert block["source"] == "paper_account"
    assert block["equity_curve"][0] == {"date": "2026-07-27", "equity": 100000.0}
    assert len(block["equity_curve"]) == 6
    assert block["returns"]["total_return_pct"] == pytest.approx(4.947504200000008, rel=1e-9)
    assert block["returns"]["annual_volatility_pct"] == pytest.approx(43.47413023856832, rel=1e-9)
    assert block["risk_adjusted"]["sharpe"] == pytest.approx(ANNUAL_SHARPE, rel=1e-9)
    assert block["risk_adjusted"]["sortino"] == pytest.approx(ANNUAL_SORTINO, rel=1e-9)
    assert block["risk_adjusted"]["psr"] == pytest.approx(0.781452734425495, rel=1e-9)
    assert block["risk_adjusted"]["deflated_sharpe"] == pytest.approx(
        0.006605823487458862, rel=1e-9
    )
    assert block["risk_adjusted"]["n_trials_used"] == 1000
    assert block["risk_adjusted"]["annualization_basis"] == 252
    assert block["account"]["start_equity"] == 100000.0
    assert block["account"]["end_equity"] == 104947.5042
    assert block["account"]["cumulative_pnl_usd"] == pytest.approx(4947.5042, rel=1e-9)


def test_the_deflated_sharpe_is_published_beside_the_count_that_deflated_it():
    """Auditable by construction: the number and the N it was computed with travel together."""
    block = metrics.performance(_sessions(), n_trials=None)

    assert block["risk_adjusted"]["deflated_sharpe"] is None
    assert block["risk_adjusted"]["n_trials_used"] is None
    assert block["risk_adjusted"]["psr"] is not None  # PSR needs no trial count


def test_performance_over_one_mark_is_a_block_of_explicit_nulls_never_zeros():
    """A run's first night has an account and no track record. Every ratio is ``null``; the
    account facts it *does* know are still stated."""
    block = metrics.performance(_sessions(equity=[100000.0], dates=["2026-07-27"]), n_trials=8)

    assert block["risk_adjusted"]["sharpe"] is None
    assert block["returns"]["cagr_pct"] is None
    assert block["drawdown"]["max_drawdown_pct"] == 0.0
    assert block["account"]["end_equity"] == 100000.0


def test_performance_carries_the_benchmark_it_was_given():
    """Equal-weight buy-and-hold over the names the run traded — computed from bars already in
    the lake, and named so nobody mistakes it for an index."""
    bench = metrics.Benchmark(
        name="equal_weight_universe_bh",
        symbols=("AAPL", "MSFT"),
        points=tuple(zip(DATES, [100.0, 101.0, 100.5, 101.5, 100.0, 103.0], strict=True)),
    )

    block = metrics.performance(_sessions(), n_trials=10, benchmark=bench)

    assert block["benchmark"]["name"] == "equal_weight_universe_bh"
    assert block["benchmark"]["symbols"] == ["AAPL", "MSFT"]
    assert block["benchmark"]["total_return_pct"] == pytest.approx(3.0, rel=1e-9)
    assert block["benchmark"]["beta"] is not None
    assert block["benchmark"]["note"] is None
    assert "points" not in block["benchmark"]  # stats, never the series


def test_a_benchmark_with_no_lake_coverage_is_null_with_a_note():
    """No new vendor spend, ever: a symbol the lake does not hold is simply not benchmarked, and
    the record says so instead of inventing a comparison."""
    bench = metrics.Benchmark(
        name="equal_weight_universe_bh",
        symbols=(),
        points=(),
        note="no lake coverage for AAPL, MSFT over 2026-07-27…2026-08-03",
    )

    block = metrics.performance(_sessions(), n_trials=10, benchmark=bench)

    assert block["benchmark"]["total_return_pct"] is None
    assert block["benchmark"]["beta"] is None
    assert block["benchmark"]["note"].startswith("no lake coverage")


def test_a_run_with_no_benchmark_at_all_still_carries_the_block_as_nulls():
    block = metrics.performance(_sessions(), n_trials=10)

    assert block["benchmark"]["name"] == "equal_weight_universe_bh"
    assert block["benchmark"]["beta"] is None
    assert block["benchmark"]["note"] is not None


def test_performance_is_deterministic():
    assert metrics.performance(_sessions(), n_trials=10) == metrics.performance(
        _sessions(), n_trials=10
    )


# ── purity, structurally ───────────────────────────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_metrics_reaches_no_io_no_clock_and_no_config():
    """The module is a pure function of the data it is handed — the same rule ``run_record`` and
    ``engine_id`` are held to. Series and trades arrive as values, so every number here is
    reproducible from a fixture and nothing can quietly read a file, a clock or the settings.

    The import allowlist is the structural half (no ``os``, no ``pathlib``, no package of our own,
    so nothing here can reach a file or the settings even transitively); the text check is the
    half that catches a clock or a file opened through a name that *is* allowed."""
    assert _imports(METRICS_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "math",
        "statistics",
        "typing",
    }
    text = METRICS_SOURCE.read_text()
    for forbidden in ("datetime.now", "utcnow", "open(", "Path("):
        assert forbidden not in text, forbidden
