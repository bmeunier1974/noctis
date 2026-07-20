"""The evaluation pipeline — the single entry point the research loop calls.

``evaluate(candidate, panel, config)`` runs the cheap pre-filter first; only survivors
reach the execution-realistic validation stage and receive a full :class:`Scorecard`. A
candidate killed at the pre-filter never touches validation (that short-circuit is tested).

``panel`` is always a ``dict[symbol, DataFrame]`` — a single symbol is a panel of one.
Per-symbol prefilter (median kill) + walk-forward + temporal holdout aggregate onto one
panel :class:`Scorecard`. ``workers > 1`` parallelizes the per-symbol work across
processes — same math, same order, same scorecard; the median-kill short-circuit and
shared split geometry are preserved because parallelism happens phase-by-phase, never
across the gate.

The pre-filter is **exit-blind** by design: it scores the raw target series and cannot see
engine-enforced protective-exit fills. That is acceptable in its coarse selection-filter
role — the event-driven walk-forward, authoritative for the Scorecard and every promotion
gate, prices exits exactly. Never approximate stops vectorially here.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from statistics import median

import pandas as pd

from noctis.backtest.candidate import Candidate
from noctis.backtest.pool import (
    POOL_TEARDOWN_GRACE_S,
    PoolStalled,
    shutdown_pool,
    wait_or_stall,
)
from noctis.backtest.prefilter import PrefilterConfig, coarse_score
from noctis.backtest.scorecard import (
    DEFAULT_ANNUALIZATION_CAP,
    DEFAULT_MAX_PERIOD_RATIO,
    DEFAULT_PERIODS_PER_YEAR,
    Metric,
    Scorecard,
    SymbolScore,
)
from noctis.backtest.splits import walk_forward
from noctis.backtest.validate import ValidationConfig, score_window, validate_candidate
from noctis.strategies.families import FamilyRegistry

logger = logging.getLogger("noctis.pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol worker tasks (module-level so a process pool can pickle them).
# A FamilyRegistry holding spec/library-minted classes is not picklable, so it
# cannot ride the (pickled) task args; ``evaluate`` stashes it here before the
# pool forks and the tasks read it back — in a forked child by inheritance, in
# the sequential fallback directly. Candidates stay picklable (name + params).
# ─────────────────────────────────────────────────────────────────────────────
_POOL_FAMILIES: FamilyRegistry | None = None


def _coarse_task(args) -> float:
    candidate, search_bars, prefilter_config = args
    return coarse_score(candidate, search_bars, prefilter_config, _POOL_FAMILIES)


def _fit_symbol_task(args):
    """One fit symbol: walk-forward splits + optional temporal-holdout metric."""
    candidate, search_bars, holdout_bars, splits, validation, metric_name = args
    split_scores = validate_candidate(candidate, search_bars, splits, validation, _POOL_FAMILIES)
    holdout_metric = None
    if holdout_bars is not None:
        metrics = score_window(candidate, holdout_bars, validation, _POOL_FAMILIES)
        holdout_metric = round(metrics.get(metric_name), 10)
    return split_scores, holdout_metric


def _held_symbol_task(args) -> float:
    candidate, bars, validation, metric_name = args
    return score_window(candidate, bars, validation, _POOL_FAMILIES).get(metric_name)


class _PanelPool:
    """Order-preserving map over symbols: fork-based process pool with a sequential
    fallback (pool unavailable/broken → identical results, one core)."""

    def __init__(self, workers: int, tasks: int):
        self._pool = None
        if workers > 1 and tasks > 1:
            try:
                # Fork explicitly: Python 3.14 defaults Linux to forkserver, which
                # re-imports __main__ per worker (would re-run a console-script CLI)
                # and would not inherit dynamically registered strategy families.
                self._pool = ProcessPoolExecutor(
                    max_workers=min(workers, tasks),
                    mp_context=multiprocessing.get_context("fork"),
                )
            except (OSError, ValueError) as exc:
                logger.warning("panel pool unavailable (%s); evaluating sequentially", exc)

    def map(self, fn, args_list: list) -> list:
        if self._pool is not None:
            try:
                # submit + wait_or_stall (not pool.map): a stall guard so an OOM-killed worker
                # can't wedge the parent forever when BrokenProcessPool never fires.
                futures = [self._pool.submit(fn, args) for args in args_list]
                wait_or_stall(futures)
                return [f.result() for f in futures]
            except (BrokenExecutor, PoolStalled) as exc:
                logger.warning("panel pool broke (%s); evaluating sequentially", exc)
                shutdown_pool(self._pool, grace_s=0.0)
                self._pool = None
        return [fn(args) for args in args_list]

    def close(self) -> None:
        """Tear down on the clean path — bounded, and never a joining ``shutdown()``.

        Teardown must never join a worker unboundedly, on ANY path. A fully drained pool is
        not a pool whose every worker is joinable: a fork-poisoned worker deadlocks before it
        ever dequeues a task, so its healthy siblings complete every future while it never
        exits (that clean-path join froze a finished session for 142 minutes). The grace here
        is the full one — an ordinary exit gives the workers their whole window to leave on
        their own sentinels, so SIGKILL never becomes the routine path.
        """
        if self._pool is not None:
            shutdown_pool(self._pool, grace_s=POOL_TEARDOWN_GRACE_S)
            self._pool = None

    def abort(self) -> None:
        """The same bounded teardown at zero grace — the wedged/unwind path.

        An exception is already in flight (or the pool is known broken), so there is nothing
        to wait out: kill the workers at once rather than spend a grace window on processes
        the run has given up on. After this, :meth:`close` is a no-op.
        """
        if self._pool is not None:
            shutdown_pool(self._pool, grace_s=0.0)
            self._pool = None


@dataclass(frozen=True)
class PipelineConfig:
    prefilter: PrefilterConfig = PrefilterConfig()
    validation: ValidationConfig = ValidationConfig()
    # Candidates whose median coarse score sits at/below this are killed early; ``None``
    # disables the kill entirely (the coarse score is still computed and recorded) — the
    # explicit "always give me the full scorecard" entry the tools and CLI use.
    prefilter_min_score: float | None = 0.0
    metric_name: Metric = Metric.SHARPE
    train_size: int = 120
    test_size: int = 40
    step: int = 40
    # Bars reserved at the chronological END as a forward holdout — never seen by the
    # prefilter, walk-forward, or optimizer. 0 disables the gate (the ``Scorecard`` then
    # carries no ``holdout_metric``).
    holdout_size: int = 0

    @classmethod
    def auto(
        cls,
        n_bars: int,
        *,
        metric: str = Metric.SHARPE,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
        prefilter_min_score: float | None = 0.0,
        annualization_cap: int | None = DEFAULT_ANNUALIZATION_CAP,
        max_period_ratio: float | None = DEFAULT_MAX_PERIOD_RATIO,
        fee_bps: float = 1.0,
        slippage_bps: float = 1.0,
    ) -> PipelineConfig:
        """The single home of the split-geometry heuristic and metric threading.

        Uniform splits sized from ``n_bars`` (a panel passes its shortest series so every
        symbol gets identical windows); one test-window of the most-recent bars reserved as
        the forward holdout when enough remain to still form a full walk-forward split —
        small data degrades to no gate rather than starving the search. The election
        ``metric`` and ``periods_per_year`` are stated once here and threaded into the
        prefilter and validation stages, so no stage can disagree on units. The per-side
        fill costs (``fee_bps`` / ``slippage_bps``, defaulting to the shipped baseline) ride
        the same path into both stages, so the coarse screen and the execution-realistic
        stage charge one identical cost.
        """
        elected = Metric.parse(metric)
        train = max(40, min(120, n_bars // 3))
        test = max(20, min(40, n_bars // 6))
        holdout = test if n_bars - test >= train + test else 0
        return cls(
            prefilter=PrefilterConfig(
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                metric=elected,
                periods_per_year=periods_per_year,
                annualization_cap=annualization_cap,
                max_period_ratio=max_period_ratio,
            ),
            validation=ValidationConfig(
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                periods_per_year=periods_per_year,
                annualization_cap=annualization_cap,
                max_period_ratio=max_period_ratio,
            ),
            prefilter_min_score=prefilter_min_score,
            metric_name=elected,
            train_size=train,
            test_size=test,
            step=test,
            holdout_size=holdout,
        )

    @classmethod
    def auto_from_settings(
        cls,
        settings,
        n_bars: int,
        *,
        periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
        prefilter_min_score: float | None = 0.0,
    ) -> PipelineConfig:
        """The one settings→pipeline mapping (the ``PromotionRules.from_settings`` of this
        module) — every entrypoint that scores against the promotion election uses it, so a
        new promotion knob is a one-site edit.

        ``settings`` is duck-typed (anything with ``promotion.metric`` /
        ``promotion.annualization_cap`` / ``promotion.max_period_ratio`` /
        ``backtest.fee_bps`` / ``backtest.slippage_bps``) so this module stays config-free.
        Per-call geometry — bar count, timeframe annualization, the prefilter kill — stays
        explicit at the call site.
        """
        promotion = settings.promotion
        backtest = settings.backtest
        return cls.auto(
            n_bars,
            metric=promotion.metric,
            periods_per_year=periods_per_year,
            prefilter_min_score=prefilter_min_score,
            annualization_cap=promotion.annualization_cap,
            max_period_ratio=promotion.max_period_ratio,
            fee_bps=backtest.fee_bps,
            slippage_bps=backtest.slippage_bps,
        )


def evaluate(
    candidate: Candidate,
    panel: dict[str, pd.DataFrame],
    config: PipelineConfig | None = None,
    symbol_holdout: dict[str, pd.DataFrame] | None = None,
    workers: int = 1,
    families: FamilyRegistry | None = None,
) -> Scorecard:
    """Pre-filter → (if survived) validate → one panel Scorecard for one candidate.

    ``panel`` maps each fit symbol to its bars — a single symbol is a panel of one and
    produces the same numbers a lone series always did. Per fit symbol: carve the temporal
    holdout tail (``config.holdout_size`` most-recent bars the search provably never
    influenced), prefilter on the search window, walk-forward validate, score the temporal
    holdout. The prefilter kills the **whole candidate** on the *median* coarse score — it
    never drops individual symbols, because pruning by strategy PnL would be cross-sectional
    cherry-picking. Symbols leave the panel only for structural reasons (too short for one
    full split), logged and recorded on the scorecard. ``symbol_holdout`` — held-out
    symbols never used in tuning/selection — get one causal ``score_window`` pass each.

    ``workers > 1`` runs the per-symbol work through one process pool, phase by phase
    (coarse → gate → validate/holdout → held-out symbols), preserving the median-kill
    short-circuit, the shared split geometry, and the exact sequential numbers.
    """
    config = config or PipelineConfig()
    # Stash the resolver for the (possibly forked) per-symbol tasks — see _POOL_FAMILIES.
    # evaluate() runs sequentially within a process, so one slot is enough.
    global _POOL_FAMILIES  # noqa: PLW0603
    _POOL_FAMILIES = families if families is not None else FamilyRegistry()
    search: dict[str, pd.DataFrame] = {}
    temporal_holdouts: dict[str, pd.DataFrame] = {}
    dropped: dict[str, str] = {}
    min_split_bars = config.train_size + config.test_size
    for sym, bars in panel.items():
        search_bars = bars
        if config.holdout_size > 0 and len(bars) > config.holdout_size:
            cut = len(bars) - config.holdout_size
            search_bars = bars.iloc[:cut]
            temporal_holdouts[sym] = bars.iloc[cut:]
        if len(search_bars) < min_split_bars:
            reason = (
                f"too short for one split ({len(search_bars)} search bars < "
                f"train+test = {min_split_bars})"
            )
            dropped[sym] = reason
            temporal_holdouts.pop(sym, None)
            logger.warning("panel: dropping %s — %s", sym, reason)
            continue
        search[sym] = search_bars

    if not search:
        # Nothing structurally usable: not validated (decide() rejects), not a dead end.
        return Scorecard(
            family=candidate.family,
            params=dict(candidate.params),
            metric_name=config.metric_name,
            stage="validated",
            symbols={},
            dropped_symbols=dropped or None,
        )

    pool = _PanelPool(workers, max(len(search), len(symbol_holdout or {})))
    try:
        # Prefilter: median coarse score across fit symbols kills the whole candidate.
        fit_syms = list(search)
        coarse_vals = pool.map(
            _coarse_task, [(candidate, search[sym], config.prefilter) for sym in fit_syms]
        )
        prefilter_metric = round(float(median(coarse_vals)), 10)
        min_score = config.prefilter_min_score
        if min_score is not None and prefilter_metric <= min_score:
            return Scorecard(
                family=candidate.family,
                params=dict(candidate.params),
                metric_name=config.metric_name,
                stage="prefilter_rejected",
                prefilter_metric=prefilter_metric,
                dropped_symbols=dropped or None,
            )

        # Identical split geometry for every symbol, from the shortest search window, so the
        # per-symbol scores stay comparable (splits are positional, so they fit every symbol).
        n_min = min(len(sb) for sb in search.values())
        splits = walk_forward(n_min, config.train_size, config.test_size, config.step)

        fit_results = pool.map(
            _fit_symbol_task,
            [
                (
                    candidate,
                    search[sym],
                    temporal_holdouts.get(sym),
                    splits,
                    config.validation,
                    config.metric_name,
                )
                for sym in fit_syms
            ],
        )
        symbols: dict[str, SymbolScore] = {
            sym: SymbolScore(splits=split_scores, holdout_metric=sym_holdout_metric)
            for sym, (split_scores, sym_holdout_metric) in zip(fit_syms, fit_results, strict=True)
        }

        temporal = [ss.holdout_metric for ss in symbols.values() if ss.holdout_metric is not None]
        holdout_metric = round(sum(temporal) / len(temporal), 10) if temporal else None

        symbol_holdout_metric = None
        if symbol_holdout:
            held = pool.map(
                _held_symbol_task,
                [
                    (candidate, bars, config.validation, config.metric_name)
                    for bars in symbol_holdout.values()
                ],
            )
            symbol_holdout_metric = round(sum(held) / len(held), 10)
    except BaseException:
        # Unwinding (a raising task, an evaluation timeout, an interrupt): abort() kills the
        # workers at once instead of granting close()'s grace window, because spending it
        # here only delays the exception that is trying to save the run. Both paths are
        # bounded — neither ever joins a worker. abort() leaves close() a no-op below.
        pool.abort()
        raise
    finally:
        pool.close()

    card = Scorecard(
        family=candidate.family,
        params=dict(candidate.params),
        metric_name=config.metric_name,
        stage="validated",
        symbols=symbols,
        prefilter_metric=prefilter_metric,
        holdout_metric=holdout_metric,
        symbol_holdout_metric=symbol_holdout_metric,
        dropped_symbols=dropped or None,
    )
    per_test = list(card.symbol_test_metrics().values())
    mean = sum(per_test) / len(per_test)
    card.panel_dispersion = round(
        math.sqrt(sum((v - mean) ** 2 for v in per_test) / len(per_test)), 10
    )
    return card
