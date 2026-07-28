"""The realised record's numbers — a **pure** module, deliberately not in ``scorecard.py``.

``backtest/scorecard.py`` feeds the promotion gates. Nothing added for *reporting* may drift into
gate math (AGENTS.md rule 2), so the practitioner metric set a results page is judged on lives
here, in its own module, on the far side of that line: no gate imports it, nothing here is a
promotion criterion, and a test in ``tests/test_gate_evidence.py`` proves the decision path cannot
reach this package at all.

The separation is not only hygiene — the two answer different questions and are allowed to differ.
The Sortino below divides by the **full-sample** downside deviation (Sortino & Price 1994); the
scorecard's divides by the negative observations only, because it ranks candidates. Neither is
wrong; blending them would be. Each states its own convention where it computes it.

**Everything here is a function of data.** Equity marks, fills and a benchmark series arrive as
values; no file, clock or configuration is reachable, which is what makes every number below
checkable against a hand-computed fixture and makes the record's segmentation equivalence free —
three short segments hand in the same marks one long segment does, so they compute the same run.

Two rules run through all of it:

* **``null`` over zeros.** A ratio whose denominator is zero or unknown is ``None``. A run that
  never lost has *no* profit factor; reporting an infinity (or a zero) answers a question nobody
  can answer yet.
* **Nothing is annualised silently.** The Sharpe published in the record is annualised at
  :data:`PERIODS_PER_YEAR`; the Sharpe the PSR and DSR are computed from is the raw per-observation
  one their published formulas are defined on. Both are stated where they are used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from statistics import NormalDist

__all__ = [
    "BENCHMARK_METHOD",
    "BENCHMARK_NAME",
    "DAYS_PER_YEAR",
    "EULER_MASCHERONI",
    "PERIODS_PER_YEAR",
    "Benchmark",
    "DailySession",
    "Drawdown",
    "RoundTrip",
    "TradeFill",
    "annual_volatility",
    "benchmark_stats",
    "cagr",
    "calmar",
    "deflated_sharpe",
    "drawdown",
    "expected_max_sharpe",
    "exposure",
    "kurtosis",
    "monthly_returns",
    "performance",
    "psr",
    "recovery_factor",
    "returns",
    "round_trips",
    "sharpe",
    "skew",
    "sortino",
    "trade_stats",
    "turnover",
]

# Daily marks: one per session close, so a year of trading is ~252 observations.
PERIODS_PER_YEAR = 252
# Calendar days a compounding span is annualised over (CAGR). Calendar, not trading, days: the
# span between two marks is wall-clock time, and a run that sat out a month really did sit it out.
DAYS_PER_YEAR = 365.0
# Euler–Mascheroni, as it appears in the expected-maximum-Sharpe expression below.
EULER_MASCHERONI = 0.5772156649015329

# The one benchmark this engine computes, named so nobody mistakes it for an index. It answers the
# fair question — did the strategy beat simply holding the names it traded? — from bars the shared
# lake already holds, so publishing it costs no vendor spend.
BENCHMARK_NAME = "equal_weight_universe_bh"
BENCHMARK_METHOD = (
    "equal-weight buy-and-hold over the symbols this run actually traded, priced from bars "
    "already in the shared lake over the run's own session window; weights are set at the first "
    "session mark and never rebalanced"
)

# What "this section is the paper account's realised record" is spelled as, in the record itself.
# Backtest and scorecard numbers live under ``strategies[].scorecard`` and are never blended in —
# stated on the artifact so no consumer can present one as the other.
PERFORMANCE_SOURCE = "paper_account"

# Quantities below this are zero. Fills carry fractional quantities, so a position is "flat" when
# it is flat to within floating-point noise rather than exactly 0.0.
_EPSILON = 1e-9

# Every published number is rounded to this many decimals, so two writes of one run are byte
# identical and a diff between two records is meaningful. Deep enough that a small probability
# (a deflated Sharpe near zero is the normal case) keeps its significant figures.
_DIGITS = 12


@dataclass(frozen=True)
class TradeFill:
    """One paper fill, enriched the way the record carries it (story #142).

    ``champion`` is the attribution that makes the trade log readable per champion: the champion
    the symbol was assigned when the fill happened. ``slippage_bps`` is the cost model the fill was
    charged under — stated per trade, because a results page without its fill assumptions is not
    taken seriously.
    """

    ts: str | None
    symbol: str
    side: str  # BUY | SELL
    quantity: float
    price: float
    fees_usd: float = 0.0
    slippage_bps: float | None = None
    champion: str | None = None
    rationale: str | None = None

    @property
    def signed_quantity(self) -> float:
        return self.quantity if str(self.side).upper() == "BUY" else -self.quantity

    @property
    def notional_usd(self) -> float:
        return abs(self.quantity * self.price)


@dataclass(frozen=True)
class DailySession:
    """One session's realised evidence: the account's mark at its close, and what it traded.

    The atom the whole performance block is derived from, and the shape the run's own daily ledger
    persists. ``equity`` is the *account's* mark-to-market at that close — the curve — while
    ``start_equity``/``end_equity`` are that session's own bounds, which is a different (and
    smaller) number on any day the account carried positions in.
    """

    date: str  # YYYY-MM-DD
    equity: float
    start_equity: float | None = None
    end_equity: float | None = None
    realized_pnl: float | None = None
    orders_submitted: int = 0
    fills: Sequence[TradeFill] = ()
    positions_end: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundTrip:
    """One **closed** trade: flat → position → flat, and the cash the account gained on it.

    Deliberately cash-based rather than average-price-based: a round trip that ends flat has no
    position left to value, so the sum of its fills' cash flows (fees included) *is* its net P&L.
    That definition needs no second copy of the broker's accounting to be exactly right.
    """

    symbol: str
    champion: str | None
    pnl_usd: float
    fees_usd: float
    notional_usd: float
    opened: str | None
    closed: str | None


@dataclass(frozen=True)
class Drawdown:
    """The worst peak-to-trough fall, its dates, its duration, and whether it ever came back."""

    depth: float  # ≤ 0, a fraction
    days: int
    peak_date: str | None
    trough_date: str | None
    recovered: bool


@dataclass(frozen=True)
class Benchmark:
    """An equal-weight buy-and-hold series over the names a run traded, as data.

    ``points`` is ``(date, level)`` — the reading is done where files are (the run store); this
    module only compares. Empty points with a ``note`` is the honest shape of "the lake did not
    hold these bars", which degrades the comparison to ``null`` rather than fetching anything.
    """

    name: str = BENCHMARK_NAME
    symbols: Sequence[str] = ()
    points: Sequence[tuple[str, float]] = ()
    note: str | None = None


# ── the series ─────────────────────────────────────────────────────────────────────────────


def returns(points: Sequence[tuple[str, float]]) -> list[float]:
    """Simple period returns between consecutive marks. A zero mark contributes ``0.0``."""
    values = [value for _date, value in points]
    return [
        (later / earlier - 1.0) if earlier else 0.0
        for earlier, later in zip(values[:-1], values[1:], strict=True)
    ]


def total_return(points: Sequence[tuple[str, float]]) -> float | None:
    """End over start − 1, or ``None`` when there is not yet a span to measure."""
    if len(points) < 2 or points[0][1] == 0:
        return None
    return points[-1][1] / points[0][1] - 1.0


def cagr(points: Sequence[tuple[str, float]]) -> float | None:
    """The compounded annual growth rate over the curve's **calendar** span.

    ``None`` when the curve spans no time (a single night, or two marks on one date): an
    annualised growth rate over zero days is a division, not a number.
    """
    growth = total_return(points)
    if growth is None:
        return None
    span = _days_between(points[0][0], points[-1][0])
    if span is None or span <= 0 or (1.0 + growth) <= 0:
        return None
    return (1.0 + growth) ** (DAYS_PER_YEAR / span) - 1.0


def annual_volatility(
    period_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR
) -> float | None:
    """The sample standard deviation of the returns, annualised by ``√periods_per_year``."""
    deviation = _stdev(period_returns)
    return None if deviation is None else deviation * math.sqrt(periods_per_year)


def sharpe(
    period_returns: Sequence[float],
    periods_per_year: int = PERIODS_PER_YEAR,
    *,
    annualised: bool = True,
) -> float | None:
    """Mean over standard deviation, annualised unless asked for the raw per-period ratio.

    No risk-free rate is subtracted — the record states the excess-over-zero Sharpe, which is what
    every other number here is consistent with. ``annualised=False`` is what PSR and DSR consume:
    their published formulas are defined on the per-observation ratio.
    """
    deviation = _stdev(period_returns)
    if deviation is None or deviation == 0:
        return None
    ratio = _mean(period_returns) / deviation
    return ratio * math.sqrt(periods_per_year) if annualised else ratio


def sortino(
    period_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR
) -> float | None:
    """Mean over the **full-sample** downside deviation (Sortino & Price 1994), annualised.

    ``√(Σ min(r, 0)² / n)`` — the whole sample in the denominator, not just the losing
    observations. ``backtest/scorecard.py`` uses the negative-only denominator for gate ranking and
    keeps it; this module states its own convention rather than sharing one, because a reporting
    number must never become a gate number by accident.
    """
    n = len(period_returns)
    if n < 2:
        return None
    downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in period_returns) / n)
    if downside == 0:
        return None
    return _mean(period_returns) / downside * math.sqrt(periods_per_year)


def skew(period_returns: Sequence[float]) -> float | None:
    """Third standardised central moment (population divisor), or ``None`` on a flat series."""
    moments = _moments(period_returns)
    if moments is None:
        return None
    _m1, m2, m3, _m4 = moments
    return m3 / m2**1.5


def kurtosis(period_returns: Sequence[float]) -> float | None:
    """**Excess** kurtosis — the number a reader expects to see beside a skew (0 = Gaussian).

    PSR and DSR consume the non-excess moment their formulas are written in terms of; the
    conversion happens where they are called, once, rather than in two places that could disagree.
    """
    moments = _moments(period_returns)
    if moments is None:
        return None
    _m1, m2, _m3, m4 = moments
    return m4 / m2**2 - 3.0


def drawdown(points: Sequence[tuple[str, float]]) -> Drawdown:
    """The deepest peak-to-trough fall, **with its duration** — depth alone hides the pain.

    Duration is measured from the peak that preceded the deepest trough to the mark that first
    recovered it, or to the last mark when it never did (``recovered=False``), in calendar days.
    """
    if not points:
        return Drawdown(0.0, 0, None, None, recovered=True)
    peak_value = points[0][1]
    peak_date = points[0][0]
    depth = 0.0
    worst_peak: str | None = None
    worst_trough: str | None = None
    for stamp, value in points:
        if value > peak_value:
            peak_value, peak_date = value, stamp
        if peak_value > 0:
            fall = value / peak_value - 1.0
            if fall < depth:
                depth, worst_peak, worst_trough = fall, peak_date, stamp
    if worst_peak is None or worst_trough is None:
        return Drawdown(0.0, 0, None, None, recovered=True)
    level = next(value for stamp, value in points if stamp == worst_peak)
    recovery = next(
        (stamp for stamp, value in points if stamp > worst_trough and value >= level), None
    )
    ended = recovery if recovery is not None else points[-1][0]
    return Drawdown(
        depth=depth,
        days=_days_between(worst_peak, ended) or 0,
        peak_date=worst_peak,
        trough_date=worst_trough,
        recovered=recovery is not None,
    )


def calmar(growth: float | None, depth: float) -> float | None:
    """CAGR over the worst fall. ``None`` when nothing ever fell — that is not an infinity."""
    return _ratio(growth, abs(depth))


def recovery_factor(growth: float | None, depth: float) -> float | None:
    """Total return over the worst fall: how many drawdowns of pain the run's profit paid for."""
    return _ratio(growth, abs(depth))


def monthly_returns(points: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Each calendar month's compounded return, keyed ``YYYY-MM``.

    A month is measured from the **last mark of the previous month**, so the months multiply back
    to the run's total return instead of silently dropping the gap between them. The first month is
    measured from the run's opening mark, which is where its account began.
    """
    if len(points) < 2:
        return {}
    last_of_month: dict[str, float] = {}
    order: list[str] = []
    for stamp, value in points:
        month = stamp[:7]
        if month not in last_of_month:
            order.append(month)
        last_of_month[month] = value
    out: dict[str, float] = {}
    base = points[0][1]
    for month in order:
        close = last_of_month[month]
        if base:
            out[month] = close / base - 1.0
        base = close
    # The opening month's base is the opening mark itself, so a run whose first month has a single
    # mark reports 0.0 for it — an honest "nothing happened yet", not a missing month.
    return out


def exposure(sessions: Sequence[DailySession]) -> float | None:
    """The share of sessions the account was in the market at all.

    A session counts as exposed when it traded or ended holding a position — which together cover
    every way a position can be open during it, since a day that starts open and never trades ends
    open. Sessions, not bars: the record's granularity is the daily mark.
    """
    if not sessions:
        return None
    exposed = sum(
        1
        for session in sessions
        if session.fills or any(abs(qty) > _EPSILON for qty in session.positions_end.values())
    )
    return exposed / len(sessions)


def turnover(sessions: Sequence[DailySession]) -> float | None:
    """Average daily traded notional as a fraction of the account's equity.

    ``Σ|quantity × price| / (sessions × mean equity)`` — one number an operator can read as "this
    strategy turns over 0.75% of the account a day". ``None`` when there is no equity to divide by.
    """
    if not sessions:
        return None
    marks = [session.equity for session in sessions if session.equity]
    if not marks:
        return None
    notional = sum(fill.notional_usd for session in sessions for fill in session.fills)
    return notional / (len(sessions) * (sum(marks) / len(marks)))


# ── trades ─────────────────────────────────────────────────────────────────────────────────


def round_trips(fills: Sequence[TradeFill]) -> list[RoundTrip]:
    """Group fills into **closed** trades — flat to flat, per symbol, in the order they filled.

    A fill that flips through zero closes the trip it was carrying and opens the next one with the
    remainder, its fee split by the quantity each leg used. A position still open at the end is not
    a trade yet: its P&L is unrealised, it is already in the equity curve, and counting it among
    the wins would be counting it twice.
    """
    carried: dict[str, _OpenTrip] = {}
    closed: list[RoundTrip] = []
    for fill in fills:
        signed = fill.signed_quantity
        if abs(signed) <= _EPSILON:
            continue
        remaining = signed
        trip = carried.get(fill.symbol)
        if trip is not None and (trip.position > 0) != (signed > 0):
            closing = min(abs(signed), abs(trip.position))
            trip.absorb(fill, math.copysign(closing, signed), closing / abs(signed))
            remaining = signed - math.copysign(closing, signed)
            if abs(trip.position) <= _EPSILON:
                closed.append(trip.settle())
                del carried[fill.symbol]
            else:
                remaining = 0.0
        if abs(remaining) > _EPSILON:
            trip = carried.setdefault(fill.symbol, _OpenTrip(fill.symbol, fill.champion, fill.ts))
            trip.absorb(fill, remaining, abs(remaining) / abs(signed))
    return closed


def trade_stats(trips: Sequence[RoundTrip]) -> dict:
    """The practitioner set over closed trades, plus the per-champion split.

    Every ratio is ``null`` where its denominator is: a run that never lost has no profit factor
    and no payoff ratio, which is a statement about a young track record rather than a perfect one.
    """
    wins = [trip.pnl_usd for trip in trips if trip.pnl_usd > 0]
    losses = [trip.pnl_usd for trip in trips if trip.pnl_usd < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    count = len(trips)
    return {
        "count": count,
        "win_rate": _ratio(len(wins), count),
        "loss_rate": _ratio(len(losses), count),
        "profit_factor": _ratio(gross_profit, gross_loss),
        "expectancy_usd": _ratio(gross_profit - gross_loss, count),
        "avg_win_usd": _ratio(gross_profit, len(wins)),
        "avg_loss_usd": _ratio(-gross_loss, len(losses)),
        "payoff_ratio": _ratio(_ratio(gross_profit, len(wins)), _ratio(gross_loss, len(losses))),
        "gross_profit_usd": gross_profit if count else None,
        "gross_loss_usd": -gross_loss if count else None,
        "total_fees_usd": sum(trip.fees_usd for trip in trips) if count else None,
        "by_champion": _by_champion(trips),
    }


def _by_champion(trips: Sequence[RoundTrip]) -> dict:
    """What each champion's own trades did — the question attribution exists to answer."""
    groups: dict[str, list[RoundTrip]] = {}
    for trip in trips:
        groups.setdefault(trip.champion or "unattributed", []).append(trip)
    return {
        name: {
            "count": len(group),
            "pnl_usd": _rounded(sum(trip.pnl_usd for trip in group)),
            "fees_usd": _rounded(sum(trip.fees_usd for trip in group)),
            "win_rate": _rounded(_ratio(sum(1 for t in group if t.pnl_usd > 0), len(group))),
        }
        for name, group in sorted(groups.items())
    }


# ── PSR and DSR (Bailey & López de Prado) ──────────────────────────────────────────────────


def psr(
    sharpe_ratio: float | None,
    *,
    n_observations: int,
    skew: float | None,
    kurtosis: float | None,
    benchmark: float = 0.0,
) -> float | None:
    """The Probabilistic Sharpe Ratio: the confidence the true Sharpe exceeds ``benchmark``.

    ``PSR(SR*) = Φ[ (ŜR − SR*)·√(n−1) / √(1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) ]`` — Bailey & López de Prado
    (2012), *The Sharpe Ratio Efficient Frontier*, Journal of Risk 15(2).

    ``sharpe_ratio``, ``benchmark`` and the moments are all in the **observation** frequency (a
    daily Sharpe for daily marks): the correction is about the sample the Sharpe was estimated on,
    so annualising first would inflate the number it deflates. ``None`` whenever the inputs do not
    exist yet or the radicand is not positive — a probability that cannot be computed is not 0.5.
    """
    if sharpe_ratio is None or skew is None or kurtosis is None or n_observations < 2:
        return None
    variance = _sharpe_variance(sharpe_ratio, skew=skew, kurtosis=kurtosis)
    if variance is None:
        return None
    return NormalDist().cdf(
        (sharpe_ratio - benchmark) * math.sqrt(n_observations - 1) / math.sqrt(variance)
    )


def expected_max_sharpe(n_trials: int, sharpe_stdev: float) -> float:
    """``E[max ŜR] ≈ √V·[(1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e))]``.

    Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management
    40(5): the Sharpe a researcher should expect to reach by luck alone after ``N`` independent
    trials whose Sharpes have standard deviation ``√V``.

    ``N ≤ 1`` is 0.0 — one trial is no selection, so there is nothing to deflate.
    """
    if n_trials <= 1:
        return 0.0
    normal = NormalDist()
    upper = normal.inv_cdf(1.0 - 1.0 / n_trials)
    second = normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_stdev * ((1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * second)


def deflated_sharpe(
    sharpe_ratio: float | None,
    *,
    n_observations: int,
    skew: float | None,
    kurtosis: float | None,
    n_trials: int | None,
    trial_sharpe_stdev: float | None = None,
) -> float | None:
    """The Deflated Sharpe Ratio: PSR taken against the Sharpe selection alone would have found.

    ``DSR = PSR(SR₀)`` with ``SR₀ = E[max ŜR]`` over the run's **cumulative** trial count — the
    same count the exhaustion gate reads off the run's experiment journals, which is exactly the
    multiple-testing number the correction wants and the number this engine uniquely knows.

    ``trial_sharpe_stdev`` is the dispersion of the trials' own Sharpes when a caller has it. When
    it does not — the trials were scored on backtests under the election metric, which is not this
    account's realised Sharpe and must not be substituted for it — the deflation falls back to the
    **null hypothesis** the DSR is derived under: every trial has a true Sharpe of zero, so the
    trial Sharpes are distributed with the estimator's own variance
    ``V̂ = (1 − γ₃·ŜR + (γ₄−1)/4·ŜR²)/(n−1)``. That is the same quantity PSR's denominator carries,
    and the record names it (``deflation_basis``) so the assumption is auditable rather than
    implied.

    ``None`` without a trial count: a DSR that quietly assumed a single trial would flatter every
    run that never journaled its search, which is the opposite of what this number is for.
    """
    if n_trials is None or sharpe_ratio is None or skew is None or kurtosis is None:
        return None
    if trial_sharpe_stdev is None:
        variance = _sharpe_variance(sharpe_ratio, skew=skew, kurtosis=kurtosis)
        if variance is None or n_observations < 2:
            return None
        trial_sharpe_stdev = math.sqrt(variance / (n_observations - 1))
    threshold = expected_max_sharpe(n_trials, trial_sharpe_stdev)
    return psr(
        sharpe_ratio,
        n_observations=n_observations,
        skew=skew,
        kurtosis=kurtosis,
        benchmark=threshold,
    )


def _sharpe_variance(sharpe_ratio: float, *, skew: float, kurtosis: float) -> float | None:
    """``1 − γ₃·ŜR + (γ₄−1)/4·ŜR²`` — the non-normality correction both formulas share.

    ``kurtosis`` is the **non-excess** moment the papers are written in. A non-positive value means
    the moments cannot support the correction; the caller reports ``null`` rather than a root of a
    negative number.
    """
    variance = 1.0 - skew * sharpe_ratio + (kurtosis - 1.0) / 4.0 * sharpe_ratio**2
    return variance if variance > 0 else None


# ── benchmark-relative ─────────────────────────────────────────────────────────────────────


def benchmark_stats(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """Alpha, beta, information ratio, tracking error and correlation over aligned returns.

    Beta and correlation are ``null`` against a benchmark that never moved (a zero variance is not
    a relationship); alpha survives, because an excess return over a flat benchmark is still a
    number. Both series must already be aligned on the same dates — aligning them is the caller's
    job, because only the caller knows which dates both sides actually have.
    """
    n = len(strategy_returns)
    if n < 2 or n != len(benchmark_returns):
        return dict.fromkeys(
            ("alpha_pct", "beta", "information_ratio", "tracking_error_pct", "correlation")
        )
    mean_s = _mean(strategy_returns)
    mean_b = _mean(benchmark_returns)
    covariance = sum(
        (s - mean_s) * (b - mean_b)
        for s, b in zip(strategy_returns, benchmark_returns, strict=True)
    ) / (n - 1)
    variance_b = sum((b - mean_b) ** 2 for b in benchmark_returns) / (n - 1)
    variance_s = sum((s - mean_s) ** 2 for s in strategy_returns) / (n - 1)
    beta = covariance / variance_b if variance_b > 0 else None
    active = [s - b for s, b in zip(strategy_returns, benchmark_returns, strict=True)]
    active_dev = _stdev(active)
    return {
        # Annualised Jensen's alpha in percent: the mean return the strategy earned beyond what its
        # exposure to the benchmark explains. Without a beta it degrades to the plain excess.
        "alpha_pct": (mean_s - (beta if beta is not None else 0.0) * mean_b)
        * periods_per_year
        * 100.0,
        "beta": beta,
        "information_ratio": (
            None
            if active_dev is None or active_dev == 0
            else _mean(active) / active_dev * math.sqrt(periods_per_year)
        ),
        "tracking_error_pct": (
            None if active_dev is None else active_dev * math.sqrt(periods_per_year) * 100.0
        ),
        "correlation": (
            None
            if beta is None or variance_s <= 0
            else covariance / math.sqrt(variance_s * variance_b)
        ),
    }


# ── the assembled block ────────────────────────────────────────────────────────────────────


def performance(
    sessions: Sequence[DailySession],
    *,
    n_trials: int | None,
    benchmark: Benchmark | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """The whole realised-performance block, from one run's daily sessions.

    Named ``paper_account`` in its own ``source`` field: this is what the paper account actually
    did, and a backtest's scorecard — which lives under ``strategies[]`` — is never blended into
    it. A website that renders both is then unable to present one as the other by accident.
    """
    points = [(session.date, session.equity) for session in sessions]
    period_returns = returns(points)
    per_period = sharpe(period_returns, periods_per_year, annualised=False)
    moment_skew = skew(period_returns)
    excess_kurtosis = kurtosis(period_returns)
    # The papers are written in the non-excess fourth moment; the record publishes the excess one a
    # reader expects. Converted here, once.
    moment_kurtosis = None if excess_kurtosis is None else excess_kurtosis + 3.0
    fall = drawdown(points)
    growth = total_return(points)
    trips = round_trips([fill for session in sessions for fill in session.fills])
    return {
        "source": PERFORMANCE_SOURCE,
        "account": {
            "opened": points[0][0] if points else None,
            "start_equity": _rounded(points[0][1]) if points else None,
            "end_equity": _rounded(points[-1][1]) if points else None,
            "cumulative_pnl_usd": _rounded(points[-1][1] - points[0][1]) if points else None,
            "sessions": len(sessions),
        },
        "equity_curve": [{"date": stamp, "equity": _rounded(value)} for stamp, value in points],
        "returns": {
            "total_return_pct": _percent(growth),
            "cagr_pct": _percent(cagr(points)),
            "annual_volatility_pct": _percent(annual_volatility(period_returns, periods_per_year)),
            "best_day_pct": _percent(max(period_returns)) if period_returns else None,
            "worst_day_pct": _percent(min(period_returns)) if period_returns else None,
        },
        "risk_adjusted": {
            "sharpe": _rounded(sharpe(period_returns, periods_per_year)),
            "sortino": _rounded(sortino(period_returns, periods_per_year)),
            "calmar": _rounded(calmar(cagr(points), fall.depth)),
            "psr": _rounded(
                psr(
                    per_period,
                    n_observations=len(period_returns),
                    skew=moment_skew,
                    kurtosis=moment_kurtosis,
                )
            ),
            "deflated_sharpe": _rounded(
                deflated_sharpe(
                    per_period,
                    n_observations=len(period_returns),
                    skew=moment_skew,
                    kurtosis=moment_kurtosis,
                    n_trials=n_trials,
                )
            ),
            # Published beside the deflated Sharpe, always: the deflation is only auditable when
            # the multiple-testing count it used travels with it.
            "n_trials_used": n_trials,
            "deflation_basis": "sharpe_standard_error_under_the_zero_sharpe_null",
            "skew": _rounded(moment_skew),
            "excess_kurtosis": _rounded(excess_kurtosis),
            "annualization_basis": periods_per_year,
        },
        "drawdown": {
            "max_drawdown_pct": _percent(fall.depth),
            "max_drawdown_days": fall.days,
            "peak_date": fall.peak_date,
            "trough_date": fall.trough_date,
            "recovered": fall.recovered,
            "recovery_factor": _rounded(recovery_factor(growth, fall.depth)),
        },
        "trades": {
            **{
                key: _rounded(value) if isinstance(value, float) else value
                for key, value in trade_stats(trips).items()
            },
            # Positions still open at the last mark — the trades that are *not* in the statistics
            # above, because their P&L is unrealised and already sits in the equity curve. Stated
            # so a reader can see the difference rather than wonder about it.
            "open_at_close": sum(
                1
                for session in sessions[-1:]
                for quantity in session.positions_end.values()
                if abs(quantity) > _EPSILON
            ),
            "exposure": _rounded(exposure(sessions)),
            "turnover": _rounded(turnover(sessions)),
        },
        "benchmark": _benchmark_block(points, benchmark, periods_per_year),
        "monthly_returns_pct": {
            month: _percent(value) for month, value in monthly_returns(points).items()
        },
    }


def _benchmark_block(
    points: Sequence[tuple[str, float]], benchmark: Benchmark | None, periods_per_year: int
) -> dict:
    """The benchmark comparison, or the same keys as explicit nulls plus a note saying why not.

    Never a fetch, never a guess: the series either came out of the shared lake or it did not, and
    a run whose names the lake does not hold reports ``null`` with the reason rather than a
    comparison against something it made up.
    """
    name = benchmark.name if benchmark is not None else BENCHMARK_NAME
    symbols = list(benchmark.symbols) if benchmark is not None else []
    note = benchmark.note if benchmark is not None else "no benchmark was computed for this run"
    aligned = _aligned(points, benchmark.points if benchmark is not None else ())
    if len(aligned) < 2:
        stats = benchmark_stats([], [], periods_per_year)
        return {
            "name": name,
            "method": BENCHMARK_METHOD,
            "symbols": symbols,
            "total_return_pct": None,
            "sharpe": None,
            **stats,
            "note": note
            or "the lake held no overlapping bars for these symbols over the run's sessions",
        }
    strategy_returns = returns([(stamp, value) for stamp, value, _level in aligned])
    benchmark_returns = returns([(stamp, level) for stamp, _value, level in aligned])
    stats = benchmark_stats(strategy_returns, benchmark_returns, periods_per_year)
    return {
        "name": name,
        "method": BENCHMARK_METHOD,
        "symbols": symbols,
        "total_return_pct": _percent(
            total_return([(stamp, level) for stamp, _value, level in aligned])
        ),
        "sharpe": _rounded(sharpe(benchmark_returns, periods_per_year)),
        **{key: _rounded(value) for key, value in stats.items()},
        "note": note,
    }


def _aligned(
    points: Sequence[tuple[str, float]], levels: Sequence[tuple[str, float]]
) -> list[tuple[str, float, float]]:
    """The dates both series have, in order.

    Comparing anything else would compare two calendars — a benchmark level from a day the account
    never marked is not a comparison, it is an alignment error with a plausible-looking number.
    """
    by_date = dict(levels)
    return [(stamp, value, by_date[stamp]) for stamp, value in points if stamp in by_date]


# ── arithmetic helpers ─────────────────────────────────────────────────────────────────────


class _OpenTrip:
    """A position being carried, until it closes and becomes a :class:`RoundTrip`."""

    def __init__(self, symbol: str, champion: str | None, opened: str | None) -> None:
        self.symbol = symbol
        self.champion = champion
        self.opened = opened
        self.closed = opened
        self.position = 0.0
        self.cash = 0.0
        self.fees = 0.0
        self.notional = 0.0

    def absorb(self, fill: TradeFill, quantity: float, fee_share: float) -> None:
        """Apply ``quantity`` of ``fill`` (signed) plus its share of the fee to this trip."""
        self.position += quantity
        self.cash += -quantity * fill.price - fill.fees_usd * fee_share
        self.fees += fill.fees_usd * fee_share
        self.notional += abs(quantity) * fill.price
        self.closed = fill.ts
        if self.champion is None:
            self.champion = fill.champion

    def settle(self) -> RoundTrip:
        return RoundTrip(
            symbol=self.symbol,
            champion=self.champion,
            pnl_usd=self.cash,
            fees_usd=self.fees,
            notional_usd=self.notional,
            opened=self.opened,
            closed=self.closed,
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float | None:
    """The sample (n−1) standard deviation, or ``None`` under two observations."""
    n = len(values)
    if n < 2:
        return None
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))


def _moments(values: Sequence[float]) -> tuple[float, float, float, float] | None:
    """The first four central moments (population divisor), or ``None`` on a degenerate series."""
    n = len(values)
    if n < 2:
        return None
    mean = _mean(values)
    deviations = [value - mean for value in values]
    m2 = sum(d**2 for d in deviations) / n
    if m2 <= 0:
        return None
    return mean, m2, sum(d**3 for d in deviations) / n, sum(d**4 for d in deviations) / n


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """One ratio, or ``None`` when either side is unknown or the denominator is zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _percent(value: float | None) -> float | None:
    return None if value is None else _rounded(value * 100.0)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, _DIGITS)


def _days_between(earlier: str, later: str) -> int | None:
    """Calendar days between two ``YYYY-MM-DD`` marks, or ``None`` if either is not one."""
    try:
        return (date.fromisoformat(later) - date.fromisoformat(earlier)).days
    except ValueError:
        return None
