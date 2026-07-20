"""SweepRunner drives one sweep's execution — sampling, the fork pool, the stall guard —
behind its own interface, so the pool's real failure modes (a wedged worker, a dead
worker, a raising trial) are exercised here directly: no toolbox, no module-global
monkeypatching."""

from __future__ import annotations

import multiprocessing
import os
import time

import pandas as pd

from noctis.backtest.pool import POOL_TEARDOWN_GRACE_S, PoolStalled, shutdown_pool
from noctis.research.sweep import SweepRunner
from noctis.strategies.base import ParamSpec
from noctis.strategies.library import LibraryPaths

_SPACE = [ParamSpec("lookback", "int", 5, 40, 1)]


def _bars() -> dict[str, pd.DataFrame]:
    return {"AAA": pd.DataFrame({"close": [float(i) for i in range(8)]})}


class _Card:
    """The one scorecard field the runner itself touches (the sampler objective)."""

    def __init__(self, test: float = 1.0):
        self.avg_test_metric = test


def _runner(tmp_path, evaluate_fn, *, workers: int = 2, **overrides) -> SweepRunner:
    return SweepRunner(
        # An empty library: the worker initializer loads nothing, which is all the
        # failure-mode tests need — their workers never reach a real evaluation.
        strategies_dir=LibraryPaths.from_single_root(tmp_path / "lib"),
        workers=workers,
        bar_budget=0,  # disable memory scaling — tests pick worker counts explicitly
        evaluate_fn=evaluate_fn,
        **overrides,
    )


# ── worker behaviors (module-level: submitted callables must pickle by reference) ──
def _wedged_eval(_task):
    time.sleep(3600)  # simulates a fork-poisoned worker that never returns


def _raising_eval(task):
    raise ValueError(f"bad trial for {task[0]}")


def _dying_eval(_task):
    os._exit(1)  # simulates an OOM-killed worker: the process vanishes mid-trial


def _healthy_eval(_task):
    return _Card()  # a worker that completes its trial normally — the clean-exit path


def _raise_stall(futures, **kwargs):
    raise PoolStalled("simulated wedge")


def test_sequential_run_yields_every_trial_and_none_on_error(tmp_path):
    """workers=1 never opens a pool: each asked param set is evaluated parent-side,
    and a raising trial yields ``(params, None)`` without ending the sweep."""
    calls: list[dict] = []

    def evaluate_fn(name, params, bars):
        calls.append(params)
        if len(calls) == 2:
            raise ValueError("bad trial")
        return _Card(test=float(len(calls)))

    runner = _runner(tmp_path, evaluate_fn, workers=1)
    out = list(runner.run("probe", _SPACE, _bars(), 4, config=None))
    assert len(out) == 4 and len(calls) == 4
    assert [card is None for _, card in out] == [False, True, False, False]
    assert all(isinstance(params["lookback"], int) for params, _ in out)


def test_pool_trial_exception_yields_none_without_fallback(tmp_path):
    """A raising trial costs only itself: the pool stays healthy, every trial is still
    asked and yielded (as ``None``), and the sequential fallback never engages."""
    seq_calls: list[dict] = []

    def evaluate_fn(name, params, bars):
        seq_calls.append(params)
        return _Card()

    runner = _runner(tmp_path, evaluate_fn, worker_eval=_raising_eval)
    out = list(runner.run("probe", _SPACE, _bars(), 3, config=None))
    assert len(out) == 3
    assert all(card is None for _, card in out)
    assert seq_calls == []  # per-trial errors are not a pool failure


def test_broken_pool_falls_back_sequentially(tmp_path):
    """A worker that dies outright (BrokenProcessPool) hands the rest of the sweep to
    the sequential path — every trial still evaluated exactly once."""
    seq_calls: list[dict] = []

    def evaluate_fn(name, params, bars):
        seq_calls.append(params)
        return _Card()

    runner = _runner(tmp_path, evaluate_fn, worker_eval=_dying_eval)
    out = list(runner.run("probe", _SPACE, _bars(), 3, config=None))
    assert len(out) == 3 and len(seq_calls) == 3
    assert all(card is not None for _, card in out)


def test_stalled_pool_falls_back_without_joining_the_wedged_worker(tmp_path):
    """Regression for the sweep-pool re-freeze: PoolStalled fired, but the `with` block's
    __exit__ ran shutdown(wait=True) and joined the wedged worker forever, so the sequential
    fallback was never reached. The sweep must complete sequentially, promptly, and leave no
    killed worker lingering."""
    seq_calls: list[dict] = []

    def evaluate_fn(name, params, bars):
        seq_calls.append(params)
        return _Card()

    runner = _runner(tmp_path, evaluate_fn, worker_eval=_wedged_eval, wait=_raise_stall)
    start = time.monotonic()
    out = list(runner.run("probe", _SPACE, _bars(), 3, config=None))
    assert time.monotonic() - start < 60  # the old join would still be sleeping on the worker
    assert len(out) == 3 and all(card is not None for _, card in out)
    assert len(seq_calls) == 3  # nothing yielded before the stall — all three ran sequentially
    # The wedged workers were SIGKILLed and reaped, not left futex-blocked at a panel each.
    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not multiprocessing.active_children()


def test_clean_pool_exit_tears_down_with_the_bounded_helper(tmp_path, monkeypatch):
    """Regression for the 2026-07-20 freeze: every trial completed, so the stall guard
    rightly stayed quiet — and the CLEAN teardown then joined a fork-poisoned worker that
    never exits, freezing a finished session for hours. A fully drained pool is not a pool
    whose every worker is joinable, so the clean exit must route through the bounded
    teardown too, at its full grace."""
    graces: list[float] = []

    def recording_shutdown(pool, *, grace_s=POOL_TEARDOWN_GRACE_S):
        graces.append(grace_s)
        shutdown_pool(pool, grace_s=grace_s)

    monkeypatch.setattr("noctis.research.sweep.shutdown_pool", recording_shutdown)

    seq_calls: list[dict] = []

    def evaluate_fn(name, params, bars):
        seq_calls.append(params)
        return _Card()

    runner = _runner(tmp_path, evaluate_fn, worker_eval=_healthy_eval)
    out = list(runner.run("probe", _SPACE, _bars(), 4, config=None))
    assert len(out) == 4 and all(card is not None for _, card in out)
    assert seq_calls == []  # the pool drained every trial — no fallback engaged
    assert graces == [POOL_TEARDOWN_GRACE_S]  # bounded teardown, clean-path grace (5.0s)


def test_sequential_trial_timeout_ends_the_sweep_early(tmp_path):
    """A HUNG trial (EvaluationTimeout from the wall-clock guard) is not a failed trial:
    there is no fallback below sequential, so the runner journals it as errored and ends
    the sweep rather than burning the remaining trials on a strategy that hangs."""
    from noctis.backtest.pool import EvaluationTimeout

    calls: list[dict] = []

    def evaluate_fn(name, params, bars):
        calls.append(params)
        if len(calls) == 2:
            raise EvaluationTimeout("evaluation exceeded 1800s wall-clock")
        return _Card()

    runner = _runner(tmp_path, evaluate_fn, workers=1)
    out = list(runner.run("probe", _SPACE, _bars(), 5, config=None))
    assert len(calls) == 2  # trials 3..5 were never attempted
    assert len(out) == 2
    assert out[0][1] is not None and out[1][1] is None
