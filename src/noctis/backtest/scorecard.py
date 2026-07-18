"""Performance metrics and the ``Scorecard``.

Metrics are computed in-house (Sharpe, Sortino, max drawdown, win rate, turnover,
exposure) so the scorecard has no heavy runtime dependency; ``quantstats-lumi`` is an
optional tearsheet renderer, not a requirement for the numbers. Ranking is by the
**out-of-sample test metric**; the **train − test gap** is the overfit signal.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace

DEFAULT_PERIODS_PER_YEAR = 252
# Metric-robustness defaults (surfaced as config knobs on PromotionConfig, threaded through the
# pipeline). ``None`` on the raw metric functions means "no cap" so a direct call is unchanged;
# the active defaults live here and are applied via PrefilterConfig/ValidationConfig.
DEFAULT_ANNUALIZATION_CAP = 252  # annualize no finer than daily (see _annualization)
DEFAULT_MAX_PERIOD_RATIO = 1.0  # clamp the per-period risk-adjusted ratio (see _cap_ratio)


def _annualization(periods_per_year: int, annualization_cap: int | None) -> float:
    """``sqrt`` of the annualization periods, optionally capped. Annualizing sub-daily returns by
    ``sqrt(intraday periods)`` assumes intraday returns are i.i.d. (false) and inflates a Sharpe/
    Sortino 20-300x (1m: x313 vs a daily x16); capping at the daily scale keeps every timeframe on
    one comparable annualized footing. ``None`` leaves the classic uncapped annualization."""
    ppy = periods_per_year
    if annualization_cap is not None:
        ppy = min(ppy, annualization_cap)
    return math.sqrt(ppy)


def _cap_ratio(ratio: float, max_period_ratio: float | None) -> float:
    """Clamp a per-period risk-adjusted ratio (mean/std or mean/downside-std) to
    ``[-cap, +cap]``. A per-*bar* Sharpe/Sortino above ~1 is degeneracy — a split with near-zero
    downside, not real edge — and, unclamped, annualizes into the tens of thousands and sits
    unbeatable atop the champion registry. ``None`` leaves the ratio untouched."""
    if max_period_ratio is None:
        return ratio
    return max(-max_period_ratio, min(ratio, max_period_ratio))


def _returns_from_equity(equity: Sequence[float]) -> list[float]:
    out = []
    for prev, cur in zip(equity[:-1], equity[1:], strict=True):
        out.append((cur / prev - 1.0) if prev != 0 else 0.0)
    return out


def total_return(equity: Sequence[float]) -> float:
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return equity[-1] / equity[0] - 1.0


def sharpe(
    returns: Sequence[float],
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    *,
    annualization_cap: int | None = None,
    max_period_ratio: float | None = None,
) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return _cap_ratio(mean / std, max_period_ratio) * _annualization(
        periods_per_year, annualization_cap
    )


def sortino(
    returns: Sequence[float],
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    *,
    annualization_cap: int | None = None,
    max_period_ratio: float | None = None,
) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    dvar = sum(r**2 for r in downside) / len(downside)
    dstd = math.sqrt(dvar)
    if dstd == 0:
        return 0.0
    return _cap_ratio(mean / dstd, max_period_ratio) * _annualization(
        periods_per_year, annualization_cap
    )


def max_drawdown(equity: Sequence[float]) -> float:
    """Most negative peak-to-trough drawdown as a fraction (≤ 0)."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            dd = value / peak - 1.0
            worst = min(worst, dd)
    return worst


def win_rate(returns: Sequence[float]) -> float:
    wins = sum(1 for r in returns if r > 0)
    active = sum(1 for r in returns if r != 0)
    return wins / active if active else 0.0


def turnover(targets: Sequence[int]) -> float:
    """Average per-bar absolute change in target position."""
    if len(targets) < 2:
        return 0.0
    changes = sum(abs(b - a) for a, b in zip(targets[:-1], targets[1:], strict=True))
    return changes / len(targets)


def exposure(targets: Sequence[int]) -> float:
    """Fraction of bars with a nonzero position."""
    if not targets:
        return 0.0
    return sum(1 for t in targets if t != 0) / len(targets)


class Metric(enum.StrEnum):
    """An election metric — the one objective research ranks, gates, and promotes on.

    The single home of the metric contract: :meth:`parse` is the only place an unknown
    name is diagnosed (settings, the CLI flag, the mandate overlay, and the pre-filter all
    route through it — same message everywhere, their refusal *policies* stay their own),
    and :meth:`from_returns` is the coarse per-returns computation the pre-filter ranks
    with. Members are :class:`Metrics` field names, so an election metric always reads off
    a scored record via ``Metrics.get``; being ``str`` subclasses, they persist and compare
    as their plain names.
    """

    SHARPE = "sharpe"
    SORTINO = "sortino"
    TOTAL_RETURN = "total_return"

    @classmethod
    def parse(cls, value: str) -> Metric:
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"unknown metric {value!r} (valid: {valid})") from None

    def from_returns(
        self,
        returns: Sequence[float],
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
        *,
        annualization_cap: int | None = None,
        max_period_ratio: float | None = None,
    ) -> float:
        """The metric over a bare per-bar return series (no equity curve yet)."""
        if self is Metric.SHARPE:
            return sharpe(
                returns,
                periods_per_year,
                annualization_cap=annualization_cap,
                max_period_ratio=max_period_ratio,
            )
        if self is Metric.SORTINO:
            return sortino(
                returns,
                periods_per_year,
                annualization_cap=annualization_cap,
                max_period_ratio=max_period_ratio,
            )
        return math.prod(1.0 + r for r in returns) - 1.0


@dataclass(frozen=True)
class Metrics:
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    turnover: float
    exposure: float

    def get(self, name: str) -> float:
        return float(getattr(self, name))


def compute_metrics(
    equity_curve: Sequence[float],
    targets: Sequence[int],
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    *,
    annualization_cap: int | None = None,
    max_period_ratio: float | None = None,
) -> Metrics:
    rets = _returns_from_equity(equity_curve)
    return Metrics(
        total_return=round(total_return(equity_curve), 10),
        sharpe=round(
            sharpe(
                rets,
                periods_per_year,
                annualization_cap=annualization_cap,
                max_period_ratio=max_period_ratio,
            ),
            10,
        ),
        sortino=round(
            sortino(
                rets,
                periods_per_year,
                annualization_cap=annualization_cap,
                max_period_ratio=max_period_ratio,
            ),
            10,
        ),
        max_drawdown=round(max_drawdown(equity_curve), 10),
        win_rate=round(win_rate(rets), 10),
        turnover=round(turnover(targets), 10),
        exposure=round(exposure(targets), 10),
    )


@dataclass(frozen=True)
class SplitScore:
    split_index: int
    train: Metrics
    test: Metrics


@dataclass(frozen=True)
class SymbolScore:
    """One fit symbol's walk-forward splits + its temporal forward-holdout metric."""

    splits: list[SplitScore]
    holdout_metric: float | None = None


def _split_to_dict(s: SplitScore) -> dict:
    return {"split_index": s.split_index, "train": asdict(s.train), "test": asdict(s.test)}


def _split_from_dict(s: dict) -> SplitScore:
    return SplitScore(
        split_index=s["split_index"],
        train=Metrics(**s["train"]),
        test=Metrics(**s["test"]),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_metrics(metrics: list[Metrics]) -> Metrics:
    """Element-wise mean of a symbol's per-split :class:`Metrics` into one averaged row. Field
    names come from the dataclass, so a newly added metric is folded in automatically."""
    names = [f.name for f in fields(Metrics)]
    return Metrics(**{n: round(_mean([m.get(n) for m in metrics]), 10) for n in names})


# Sentinel symbol keying the splits of a legacy pre-panel card (persisted before the
# always-panel shape). Never binds champion eligibility — the registry filters it out.
LEGACY_SYMBOL = "*"


@dataclass
class Scorecard:
    """Per-split and aggregate metrics for one candidate; ranked by ``avg_test_metric``.

    **A Scorecard is always a panel**: ``symbols`` holds each fit symbol's splits (a single
    symbol is a panel of one), and ``avg_train_metric`` / ``avg_test_metric`` / ``gap`` are
    panel means (mean of the per-symbol values) — one meaning, every caller. An empty
    ``symbols`` means no out-of-sample evidence (pre-filter kill or structurally empty
    panel); promotion rejects such cards. Legacy persisted cards with top-level splits are
    normalized on read into a panel of one under :data:`LEGACY_SYMBOL`.
    """

    family: str
    params: dict
    metric_name: str = "sharpe"
    stage: str = "validated"  # "validated" | "prefilter_rejected"
    prefilter_metric: float | None = None
    # Metric on the forward-holdout window — the most-recent bars the search never touched;
    # the mean of the per-symbol temporal holdout metrics. ``None`` when no holdout was
    # reserved (small dataset).
    holdout_metric: float | None = None
    # The panel: each fit symbol's walk-forward splits + temporal holdout metric.
    symbols: dict[str, SymbolScore] = field(default_factory=dict)
    # Mean metric across the held-out symbols (names never used in tuning/selection);
    # ``None`` when no symbol holdout was reserved.
    symbol_holdout_metric: float | None = None
    # Cross-symbol dispersion (population std of per-symbol test metrics) — a reported
    # diagnostic, never an election penalty.
    panel_dispersion: float | None = None
    # Symbols structurally dropped from the panel (not ready / too short for one split),
    # mapped to the reason. Never populated by PnL — that would be symbol cherry-picking.
    dropped_symbols: dict[str, str] | None = None

    @property
    def fit_symbols(self) -> list[str]:
        """The real fit symbols this card was scored on — the :data:`LEGACY_SYMBOL` sentinel
        keys scores, not symbols, so it never appears here. What champion eligibility binds."""
        return sorted(s for s in self.symbols if s != LEGACY_SYMBOL)

    def symbol_train_metrics(self) -> dict[str, float]:
        if not self.symbols:
            return {}
        return {
            sym: _mean([s.train.get(self.metric_name) for s in ss.splits])
            for sym, ss in self.symbols.items()
        }

    def symbol_test_metrics(self) -> dict[str, float]:
        """Per-symbol mean out-of-sample test metric (empty when the card has no panel)."""
        if not self.symbols:
            return {}
        return {
            sym: _mean([s.test.get(self.metric_name) for s in ss.splits])
            for sym, ss in self.symbols.items()
        }

    @property
    def avg_train_metric(self) -> float:
        return _mean(list(self.symbol_train_metrics().values()))

    def avg_test_named(self, name: str) -> float:
        """Mean out-of-sample test value of an arbitrary metric ``name``.

        Same panel-mean logic as :attr:`avg_test_metric`, but reads the given metric
        ``name`` from each split's persisted ``test`` :class:`Metrics` instead of
        ``self.metric_name``. A computed read over already-persisted numbers — it changes no
        stored shape or value. Used to score every profile's champions on a common Sharpe
        basis regardless of the election metric each was tuned on (§7).
        """
        return _mean([_mean([s.test.get(name) for s in ss.splits]) for ss in self.symbols.values()])

    @property
    def avg_test_metric(self) -> float:
        return self.avg_test_named(self.metric_name)

    @property
    def gap(self) -> float:
        """train − test on the ranking metric. Large positive gap ⇒ overfit."""
        return self.avg_train_metric - self.avg_test_metric

    @property
    def test_activity(self) -> float:
        """Fraction of out-of-sample test splits with any market exposure.

        Pools every fit symbol's splits. Near-zero activity means the aggregate metric
        rests on a handful of trades — noise, not edge — which is what the promotion
        activity floor rejects.
        """
        splits = [s for ss in self.symbols.values() for s in ss.splits]
        if not splits:
            return 0.0
        return sum(1 for s in splits if s.test.get("exposure") > 0.0) / len(splits)

    def compact(self) -> Scorecard:
        """A persistence-light copy: each fit symbol's walk-forward splits collapsed to a single
        mean split (that symbol's element-wise mean train/test :class:`Metrics`).

        Every value a *stored* champion is ever read for — ``avg_train_metric``,
        ``avg_test_metric``, ``avg_test_named(...)``, ``symbol_test_metrics``, ``gap`` — is a
        mean over that symbol's splits, so a single mean split reproduces it exactly; only
        per-split detail (which no persisted champion is read for; the activity floor and the
        holdout gates run on the fresh in-memory challenger) is dropped. Idempotent, and safe on
        an empty/sentinel panel. This is what the registry serializes, so ``champions.json``
        stays a few KB instead of tens of MB — one champion's five thousand 1m splits per symbol
        were the whole bloat.
        """
        symbols = {
            sym: SymbolScore(
                splits=(
                    [
                        SplitScore(
                            0,
                            _mean_metrics([s.train for s in ss.splits]),
                            _mean_metrics([s.test for s in ss.splits]),
                        )
                    ]
                    if ss.splits
                    else []
                ),
                holdout_metric=ss.holdout_metric,
            )
            for sym, ss in self.symbols.items()
        }
        return replace(self, symbols=symbols)

    # --- serialization (champion registry persists these) ---
    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "params": self.params,
            "metric_name": self.metric_name,
            "stage": self.stage,
            "prefilter_metric": self.prefilter_metric,
            "avg_train_metric": round(self.avg_train_metric, 10),
            "avg_test_metric": round(self.avg_test_metric, 10),
            "gap": round(self.gap, 10),
            "holdout_metric": (
                None if self.holdout_metric is None else round(self.holdout_metric, 10)
            ),
            "symbols": {
                sym: {
                    "holdout_metric": ss.holdout_metric,
                    "splits": [_split_to_dict(s) for s in ss.splits],
                }
                for sym, ss in self.symbols.items()
            },
            "symbol_holdout_metric": self.symbol_holdout_metric,
            "panel_dispersion": self.panel_dispersion,
            "dropped_symbols": self.dropped_symbols,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> Scorecard:
        symbols_raw = data.get("symbols")
        if symbols_raw is not None:
            symbols = {
                sym: SymbolScore(
                    splits=[_split_from_dict(s) for s in sd.get("splits", [])],
                    holdout_metric=sd.get("holdout_metric"),
                )
                for sym, sd in symbols_raw.items()
            }
        else:
            # Legacy pre-panel card: top-level splits normalize to a panel of one under
            # the sentinel. Aggregates read identically (a panel of one is a plain mean
            # over its splits), so persisted champions rank unchanged.
            legacy_splits = [_split_from_dict(s) for s in data.get("splits", [])]
            symbols = (
                {
                    LEGACY_SYMBOL: SymbolScore(
                        splits=legacy_splits, holdout_metric=data.get("holdout_metric")
                    )
                }
                if legacy_splits
                else {}
            )
        return cls(
            family=data["family"],
            params=data["params"],
            metric_name=data.get("metric_name", "sharpe"),
            stage=data.get("stage", "validated"),
            prefilter_metric=data.get("prefilter_metric"),
            holdout_metric=data.get("holdout_metric"),
            symbols=symbols,
            symbol_holdout_metric=data.get("symbol_holdout_metric"),
            panel_dispersion=data.get("panel_dispersion"),
            dropped_symbols=data.get("dropped_symbols"),
        )

    @classmethod
    def from_json(cls, text: str) -> Scorecard:
        return cls.from_dict(json.loads(text))
