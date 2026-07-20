"""One sweep's search engine: seeded sampling, the fork pool, and the stall guard.

``ResearchToolbox.run_sweep`` owns a sweep's *accounting* — budget, journal, result
ranking — while :class:`SweepRunner` owns its *execution*: propose parameter sets
(Optuna TPE when installed, seeded RNG otherwise), evaluate them on a fork pool when
configured, and survive the pool's real failure modes (a fork-poisoned or OOM-killed
worker) by falling back to sequential evaluation without ever re-running a yielded
trial. The pool's failure paths are injectable on the runner itself (``worker_eval``,
``wait``), so tests drive a wedged or broken pool through this interface instead of
monkeypatching module globals.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from collections.abc import Callable, Iterator
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import Any, cast

import numpy as np
import pandas as pd

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.backtest.pool import (
    EvaluationTimeout,
    PoolStalled,
    scale_workers,
    shutdown_pool,
    wait_or_stall,
)
from noctis.strategies import library
from noctis.strategies.base import ParamSpec
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.proposer import _repair

logger = logging.getLogger("noctis.research.sweep")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep-trial worker (module-level so ProcessPoolExecutor can pickle it).
# Each worker gets the strategy library, the panel bars, and the pipeline
# config ONCE via the initializer; per-trial traffic is just a params dict in
# and an aggregate Scorecard out. The journal stays parent-side (one writer).
# ─────────────────────────────────────────────────────────────────────────────
_WORKER_BARS: dict[str, pd.DataFrame] | None = None
_WORKER_CONFIG: PipelineConfig | None = None
_WORKER_FAMILIES: FamilyRegistry | None = None


def _worker_init(
    strategies_dir: library.LibraryPaths, bars: dict[str, pd.DataFrame], config
) -> None:
    from noctis.strategies.library import load_and_register

    global _WORKER_BARS, _WORKER_CONFIG, _WORKER_FAMILIES  # noqa: PLW0603 — per-process state
    _WORKER_FAMILIES = FamilyRegistry()
    load_and_register(strategies_dir, _WORKER_FAMILIES)
    _WORKER_BARS = bars
    _WORKER_CONFIG = config


def _worker_eval(task: tuple[str, dict]):
    name, params = task
    bars = cast(dict[str, pd.DataFrame], _WORKER_BARS)  # set by _worker_init
    # evaluate() is panel-only: a single symbol is a panel of one, never a bare frame.
    return evaluate(Candidate(name, params), bars, config=_WORKER_CONFIG, families=_WORKER_FAMILIES)


class SweepSampler:
    """Seeded parameter sampler: Optuna TPE when installed, plain RNG otherwise.

    ``ask(k)`` returns a batch of ``(handle, params)`` so trials can be evaluated in
    parallel; ``tell`` feeds results back per handle (batched TPE, standard practice).
    """

    def __init__(self, name: str, space: list[ParamSpec], seed: int = 0):
        self.name = name
        self.space = space
        self._rng = np.random.default_rng(seed)
        self._study: Any = None
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
            )
        except ImportError:
            pass

    def _one(self) -> tuple[object, dict]:
        params: dict = {}
        trial = self._study.ask() if self._study is not None else None
        for s in self.space:
            lo, hi = cast(float, s.low), cast(float, s.high)  # numeric kinds carry bounds
            if trial is not None:
                if s.kind == "categorical":
                    params[s.name] = trial.suggest_categorical(s.name, list(s.choices))
                elif s.kind == "int":
                    params[s.name] = trial.suggest_int(s.name, int(lo), int(hi))
                else:
                    params[s.name] = trial.suggest_float(s.name, float(lo), float(hi))
            else:
                if s.kind == "categorical":
                    params[s.name] = list(s.choices)[int(self._rng.integers(0, len(s.choices)))]
                elif s.kind == "int":
                    params[s.name] = int(self._rng.integers(int(lo), int(hi) + 1))
                else:
                    params[s.name] = float(self._rng.uniform(float(lo), float(hi)))
        return trial, _repair(self.name, params)

    def ask(self, k: int) -> list[tuple[object, dict]]:
        return [self._one() for _ in range(k)]

    def tell(self, handle, objective: float | None) -> None:
        if self._study is not None and handle is not None:
            self._study.tell(handle, objective if objective is not None else float("-inf"))


class SweepRunner:
    """Evaluate one sweep's trials — parallel when configured, sequential on any pool failure.

    Construction binds the session-scoped knobs: the library tier roots (each worker
    re-loads the library once via the pool initializer), the configured worker count and
    the memory bar budget that scales it down, and the parent-side sequential evaluator
    ``evaluate_fn(name, params, bars) -> Scorecard``. ``worker_eval`` and ``wait`` default
    to the real pool implementations; tests inject a wedged worker or an instant stall to
    drive the failure paths through this interface.
    """

    def __init__(
        self,
        *,
        strategies_dir: library.LibraryPaths,
        workers: int,
        bar_budget: int,
        evaluate_fn: Callable[[str, dict, dict[str, pd.DataFrame]], Any],
        worker_eval: Callable[[tuple[str, dict]], Any] = _worker_eval,
        wait: Callable[..., None] = wait_or_stall,
    ):
        self.strategies_dir = strategies_dir
        self.workers = workers
        self.bar_budget = bar_budget
        self.evaluate_fn = evaluate_fn
        self.worker_eval = worker_eval
        self.wait = wait

    def run(
        self,
        name: str,
        space: list[ParamSpec],
        bars: dict[str, pd.DataFrame],
        n: int,
        *,
        config: PipelineConfig,
        seed: int = 0,
    ) -> Iterator[tuple[dict, Any]]:
        """Yield ``(params, card)`` for ``n`` trials; ``card`` is ``None`` for a trial that raised.

        With more than one effective worker a process pool evaluates each asked batch
        concurrently (batched TPE: the sampler learns between batches, not within one).
        A broken/unavailable/stalled pool falls back to sequential evaluation for the
        rest of the sweep — continuing from where the pool died, never re-running a
        yielded trial. Budget spend for errored (``None``) trials is the caller's call.
        """
        sampler = SweepSampler(name, space, seed=seed)
        # Each sweep worker holds a full copy of the panel, so scale the ceiling down by the
        # panel's total bar count (a 1m panel is ~60× a 1h one) before capping by trials/CPU.
        total_bars = sum(len(df) for df in bars.values())
        scaled = scale_workers(self.workers, total_bars, budget=self.bar_budget)
        workers = max(1, min(scaled, n, os.cpu_count() or 1))
        done = 0
        if workers > 1:
            pool = None
            clean = False
            try:
                # Fork explicitly: Python 3.14 defaults Linux to forkserver, which
                # re-imports __main__ in every worker — under a console-script entry
                # point that would re-run the CLI. Fork inherits the loaded process
                # image instead (and is cheaper); platforms without fork fall through
                # to the sequential path via the except below.
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("fork"),
                    initializer=_worker_init,
                    initargs=(self.strategies_dir, dict(bars), config),
                )
                while done < n:
                    batch = sampler.ask(min(workers, n - done))
                    futures = [pool.submit(self.worker_eval, (name, params)) for _, params in batch]
                    # Stall guard: an OOM-killed worker can leave future.result() blocked on
                    # a futex forever (BrokenProcessPool not always raised — a 10h hang was
                    # seen). Bail to sequential if the batch makes no progress in time.
                    self.wait(futures)
                    cards = []
                    for future in futures:
                        try:
                            cards.append(future.result())
                        except BrokenExecutor:
                            raise
                        except Exception as exc:  # noqa: BLE001 — one bad trial ≠ sweep
                            logger.warning("sweep trial failed: %s", exc)
                            cards.append(None)
                    for (handle, params), card in zip(batch, cards, strict=True):
                        sampler.tell(handle, card.avg_test_metric if card is not None else None)
                        yield params, card
                    done += len(batch)
                clean = True
            except (BrokenExecutor, PoolStalled, OSError, ValueError) as exc:
                # ValueError: no "fork" on this platform; PoolStalled: a wedged (OOM'd) worker.
                # Either way, continue from where the pool died — never re-run yielded trials.
                logger.warning("sweep pool unavailable (%s); continuing sequentially", exc)
            finally:
                # Not a `with` block: its __exit__ is shutdown(wait=True), which JOINS the
                # workers — a wedged worker never exits, so that join re-froze the run right
                # after the stall guard raised. Only a fully-drained pool (every batch
                # completed) may be waited on; any other exit tears down without waiting.
                if pool is not None:
                    if clean:
                        pool.shutdown()
                    else:
                        shutdown_pool(pool, grace_s=0.0)

        while done < n:
            ((handle, params),) = sampler.ask(1)
            try:
                card = self.evaluate_fn(name, params, bars)
            except EvaluationTimeout as exc:
                # A hung trial, not a failed one. There is no further fallback below
                # sequential, so spending the remaining trials on a strategy that hangs
                # would burn hours for nothing — journal this trial as errored and end
                # the sweep; the caller reports fewer trials than asked.
                logger.warning(
                    "sweep trial hung (%s); ending the sweep after %d of %d trials",
                    exc,
                    done + 1,
                    n,
                )
                sampler.tell(handle, None)
                yield params, None
                return
            except Exception as exc:  # noqa: BLE001 — one bad trial ≠ sweep
                logger.warning("sweep trial failed: %s", exc)
                card = None
            sampler.tell(handle, card.avg_test_metric if card is not None else None)
            yield params, card
            done += 1
