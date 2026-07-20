"""Two-stage backtest: no-lookahead, cost accounting, determinism, splits, scorecard."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass

import pandas as pd
import pytest

from noctis.backtest import (
    Candidate,
    PipelineConfig,
    coarse_score,
    compute_metrics,
    evaluate,
    max_drawdown,
    require_symbols_ready,
    sharpe,
    validate_candidate,
    vectorized_returns,
    walk_forward,
)
from noctis.backtest import pipeline as pipeline_mod
from noctis.backtest.prefilter import PrefilterConfig
from noctis.backtest.splits import walk_forward as wf
from noctis.broker import FeeModel, Order, PaperBroker, Side, SlippageModel, simulate
from noctis.strategies import FamilyRegistry, TraderStrategy
from noctis.strategies.base import ExitRules, ParamSpec

from ._data_helpers import make_ohlcv, price_series

# --- helper strategies registered for these tests ----------------------------------------


@dataclass(frozen=True)
class _NoParams:
    pass


class PeekJump(TraderStrategy):
    """Goes long exactly on a bar whose close jumps far above its open (a peeking edge)."""

    name = "peek_jump"
    params_cls = _NoParams

    @classmethod
    def signals(cls, data, params):
        o = data["open"].to_numpy(dtype="float64")
        c = data["close"].to_numpy(dtype="float64")
        return pd.Series([1 if c[i] > o[i] * 1.5 else 0 for i in range(len(c))], dtype=int)

    def on_start(self, ctx):
        pass

    def on_bar(self, ctx, bar):
        ctx.set_target(1 if bar.close > bar.open * 1.5 else 0)

    @classmethod
    def param_space(cls):
        return []


class ConstantLong(TraderStrategy):
    name = "const_long"
    params_cls = _NoParams

    @classmethod
    def signals(cls, data, params):
        return pd.Series([1] * len(data), dtype=int)

    def on_start(self, ctx):
        pass

    def on_bar(self, ctx, bar):
        ctx.set_target(1)

    @classmethod
    def param_space(cls):
        return [ParamSpec("noop", "int", low=0, high=0)]


# The probe families live in this module's registry; seed-family tests use the default.
FAMILIES = FamilyRegistry()
FAMILIES.register(PeekJump)
FAMILIES.register(ConstantLong)


def _jump_bars() -> pd.DataFrame:
    rows = [
        (0, 100.0, 100.0, 100.0, 100.0, 1),
        (1, 100.0, 100.0, 100.0, 100.0, 1),
        (2, 100.0, 200.0, 100.0, 200.0, 1),  # intrabar jump: open 100 → close 200
        (3, 200.0, 200.0, 200.0, 200.0, 1),
        (4, 200.0, 200.0, 200.0, 200.0, 1),
    ]
    return pd.DataFrame(rows, columns=["ts_event", "open", "high", "low", "close", "volume"])


# --- 1. no lookahead (both stages) -------------------------------------------------------


def test_no_lookahead_validation_stage_does_not_profit_from_jump():
    bars = _jump_bars()
    cand = Candidate("peek_jump", {})
    broker = PaperBroker(starting_cash=100_000.0)
    result = simulate(cand.build(FAMILIES), bars, broker, symbol="JMP")
    # The signal fires on the jump bar, but next-bar execution buys after the jump →
    # the +100% move is never captured. Equity does not rise.
    assert result.final_equity <= result.starting_equity + 1e-6


def test_no_lookahead_prefilter_stage_does_not_profit_from_jump():
    bars = _jump_bars()
    cand = Candidate("peek_jump", {})
    targets = PeekJump.signals(bars, _NoParams())
    rets = vectorized_returns(bars, targets, fee_bps=1.0, slippage_bps=1.0)
    # Position held over each bar is the previous bar's target → the jump is not captured.
    assert (1.0 + rets).prod() - 1.0 <= 0.0
    assert coarse_score(cand, bars, families=FAMILIES) <= 0.0


# --- 2. cost accounting ------------------------------------------------------------------


def test_broker_cost_accounting_is_exact():
    """start − equity equals fees_paid + slippage_cost exactly on a flat round-trip."""
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(3.0), slippage_model=SlippageModel(4.0)
    )
    broker.set_price("SPY", 400.0)
    broker.submit_order(Order("SPY", Side.BUY, 7))
    broker.submit_order(Order("SPY", Side.SELL, 7))
    assert broker.position("SPY").quantity == 0
    spent = 100_000.0 - broker.equity()
    # Exact in principle; ~1e-11 float accumulation error is expected.
    assert spent == pytest.approx(broker.fees_paid + broker.slippage_cost, abs=1e-6)
    assert spent > 0.0


def test_costs_bite_in_validation():
    """Fees + slippage strictly reduce equity versus a zero-cost run on the same trades."""
    bars = make_ohlcv(price_series(n=200, seed=5))
    strat = FamilyRegistry().create("sma_crossover", {"fast": 5, "slow": 20})

    free = simulate(
        strat, bars, PaperBroker(fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0))
    )
    costly = simulate(
        FamilyRegistry().create("sma_crossover", {"fast": 5, "slow": 20}),
        bars,
        PaperBroker(fee_model=FeeModel(5.0), slippage_model=SlippageModel(5.0)),
    )
    assert len(costly.fills) > 0
    assert costly.final_equity < free.final_equity


# --- 3. determinism ----------------------------------------------------------------------


def test_evaluate_is_deterministic():
    bars = make_ohlcv(price_series(n=250, seed=9))
    cand = Candidate("sma_crossover", {"fast": 8, "slow": 24})
    config = PipelineConfig(prefilter_min_score=None, train_size=120, test_size=40, step=40)
    first = evaluate(cand, {"AAA": bars}, config=config)
    second = evaluate(cand, {"AAA": bars}, config=config)
    assert first.stage == "validated"
    assert first.to_json() == second.to_json()


# --- 4. split correctness ----------------------------------------------------------------


def test_walk_forward_invariants():
    splits = walk_forward(n=250, train_size=120, test_size=40, step=40)
    assert len(splits) == 3
    prev_start = -1
    for s in splits:
        assert s.train_start < s.train_end == s.test_start < s.test_end  # train then test
        assert s.test_end - s.test_start == 40
        assert s.train_end - s.train_start == 120
        assert s.test_end <= 250
        assert s.train_start > prev_start  # advancing
        prev_start = s.train_start
    assert splits[0].train_start == 0
    assert splits[-1].test_end == 240  # covers to the last full window


def test_walk_forward_rejects_bad_sizes():
    with pytest.raises(ValueError):
        wf(100, 0, 10, 10)


# --- 5. scorecard aggregation ------------------------------------------------------------


def test_exit_fills_raise_turnover_and_are_counted_as_fills():
    """Engine-enforced stop-outs are real activity: the executed stance (latched flat after
    the stop) raises turnover and cuts exposure vs the identical run without exit rules, and
    the exit fill lands in ``fills`` like any other."""

    class _AlwaysLong(TraderStrategy):
        name = "always_long_probe"
        params_cls = _NoParams

        def __init__(self, exits=None):
            super().__init__(_NoParams())
            self._exits = exits

        def on_start(self, ctx):
            pass

        def on_bar(self, ctx, bar):
            ctx.set_target(1, exits=self._exits)

        @classmethod
        def param_space(cls):
            return []

    rows = [
        (0, 100.0, 101.0, 99.0, 100.0, 1),
        (1, 100.0, 101.0, 100.0, 101.0, 1),
        (2, 100.0, 100.0, 88.0, 92.0, 1),  # breaches the 10% stop at 90
        (3, 91.0, 92.0, 90.0, 91.0, 1),
        (4, 91.0, 92.0, 90.0, 91.0, 1),
    ]
    tape = pd.DataFrame(rows, columns=["ts_event", "open", "high", "low", "close", "volume"])

    def _run(exits):
        broker = PaperBroker(
            starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
        )
        return simulate(_AlwaysLong(exits), tape, broker, symbol="TST", alloc=1.0)

    plain = _run(None)
    stopped = _run(ExitRules(stop_pct=0.10))

    assert len(stopped.fills) == len(plain.fills) + 1  # the stop-out is a fill
    assert stopped.fills[-1].reason == "stop"
    assert stopped._extra["exit_count"] == 1 and "exit_count" not in plain._extra

    plain_metrics = compute_metrics(plain.equity_curve, plain.targets)
    stopped_metrics = compute_metrics(stopped.equity_curve, stopped.targets)
    assert stopped_metrics.turnover > plain_metrics.turnover
    assert stopped_metrics.exposure < plain_metrics.exposure


def test_sharpe_hand_computed():
    # returns [0.10, -0.05, 0.10, -0.05], ppy=1: mean=0.025, sample std≈0.086603
    value = sharpe([0.10, -0.05, 0.10, -0.05], periods_per_year=1)
    assert value == pytest.approx(0.288675, abs=1e-5)


def test_metric_caps_bound_degenerate_sortino_but_not_healthy_scores():
    """The scaling fix: a near-zero-downside split annualizes into the tens of thousands on 1m
    bars. Capping annualization at daily (√252) and clamping the per-period ratio to ±1 bounds
    it to a sane ceiling; a healthy strategy with real downside is untouched (not clipped)."""
    import math

    from noctis.backtest.scorecard import sortino

    ppy_1m = 252 * 390
    degenerate = [0.001] * 500 + [-0.00001]  # tiny consistent gains, one tiny loss
    uncapped = sortino(degenerate, ppy_1m)
    capped = sortino(degenerate, ppy_1m, annualization_cap=252, max_period_ratio=1.0)
    assert uncapped > 1000.0  # the pathology this fix targets
    assert capped == pytest.approx(math.sqrt(252), abs=1e-9)  # ratio clamped to 1 × daily anno

    healthy = sortino([0.01, -0.008, 0.012, -0.005, 0.009], periods_per_year=252)
    assert sortino(
        [0.01, -0.008, 0.012, -0.005, 0.009], 252, annualization_cap=252, max_period_ratio=1.0
    ) == pytest.approx(healthy)  # below the cap ⇒ unchanged


def test_pipeline_auto_threads_metric_caps_into_both_stages():
    """The caps ride PipelineConfig.auto into the prefilter AND validation configs, so a coarse
    rank and a validated score share one bounded scale."""
    cfg = PipelineConfig.auto(2000, metric="sortino", annualization_cap=252, max_period_ratio=0.5)
    assert cfg.prefilter.annualization_cap == 252 and cfg.prefilter.max_period_ratio == 0.5
    assert cfg.validation.annualization_cap == 252 and cfg.validation.max_period_ratio == 0.5


def test_pipeline_auto_threads_fill_costs_into_both_stages():
    """The configured fee/slippage ride PipelineConfig.auto into the prefilter AND validation
    configs, so the coarse screen and the execution-realistic stage charge one identical cost."""
    cfg = PipelineConfig.auto(2000, fee_bps=2.5, slippage_bps=3.0)
    assert cfg.prefilter.fee_bps == 2.5 and cfg.prefilter.slippage_bps == 3.0
    assert cfg.validation.fee_bps == 2.5 and cfg.validation.slippage_bps == 3.0


def test_pipeline_auto_defaults_to_the_shipped_fill_costs():
    """Unset costs behave bit-for-bit as today: 1bp fee + 1bp slippage per side."""
    cfg = PipelineConfig.auto(2000)
    assert cfg.prefilter.fee_bps == 1.0 and cfg.prefilter.slippage_bps == 1.0
    assert cfg.validation.fee_bps == 1.0 and cfg.validation.slippage_bps == 1.0


def test_pool_stall_guard_raises_on_no_progress_but_not_on_completion():
    """The hang fix: wait_or_stall raises PoolStalled when the pool makes zero progress within
    the timeout (an OOM-killed worker leaves future.result() futex-blocked forever), but returns
    normally when work completes — so a slow-but-live pool is never falsely tripped."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from noctis.backtest.pool import PoolStalled, wait_or_stall

    release = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as ex:
        # A future that never resolves within the (tiny) timeout → PoolStalled. It waits on an
        # Event we set afterward, so the executor's shutdown doesn't block on it.
        stuck = [ex.submit(release.wait)]
        with pytest.raises(PoolStalled):
            wait_or_stall(stuck, timeout=0.2)
        release.set()  # let the worker finish so shutdown returns promptly

        # Futures that complete are collected without a stall.
        quick = [ex.submit(lambda i=i: i * 2) for i in range(4)]
        wait_or_stall(quick, timeout=5.0)
        assert sorted(f.result() for f in quick) == [0, 2, 4, 6]


def _double(i):
    return i * 2


def _wedge_one_worker(marker_dir):
    """Pool initializer that hangs forever in exactly ONE worker.

    Models the fork-poisoned worker: the first worker to claim the marker (an atomic
    O_EXCL create) deadlocks in its initializer, *before it ever dequeues a task*, and
    records its pid for the test. Every sibling initializes normally and drains the queue.
    """
    import os
    import time

    path = os.path.join(marker_dir, "wedged.pid")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    while True:  # the worker that never exits: only teardown ever meets it
        time.sleep(3600)


def _await_wedged_pid(path, timeout=30.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(path.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"no worker wedged itself within {timeout:.0f}s")


def _teardown_in_daemon_thread(pool, *, grace_s, bound_s=30.0):
    """Run the teardown off the main thread and return whether it finished within ``bound_s``.

    The dev deps carry no test-timeout plugin: a teardown that joins a wedged worker would
    hang the suite forever. Bounding it here turns that regression into a failed assertion.
    """
    import threading

    from noctis.backtest.pool import shutdown_pool

    thread = threading.Thread(
        target=shutdown_pool, args=(pool,), kwargs={"grace_s": grace_s}, daemon=True
    )
    thread.start()
    thread.join(timeout=bound_s)
    return not thread.is_alive()


def _child_state(pid):
    """``'gone' | 'zombie' | 'alive'`` for a child pid, read straight from the OS.

    ``Process.is_alive()`` is not usable here: the executor's own management thread reaps the
    workers too, and whichever thread loses that ``waitpid`` race gets ECHILD — after which the
    Process object reports "alive" forever. ``waitpid`` on the pid answers what the test
    actually cares about: killed (not ``alive``) and reaped by somebody (not ``zombie``).
    """
    import os

    try:
        seen, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return "gone"
    return "zombie" if seen == pid else "alive"


def _kill_leftovers(procs):
    import os
    import signal

    for proc in procs:
        try:
            os.kill(proc.pid, signal.SIGKILL)
            os.waitpid(proc.pid, 0)
        except (ChildProcessError, ProcessLookupError):  # already gone
            pass


_needs_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the fork start method is unavailable on this platform",
)


@_needs_fork
def test_shutdown_pool_is_bounded_and_reaps_a_birth_wedged_worker(tmp_path):
    """The overnight-freeze fix: a worker wedged at birth stalls no batch — its healthy sibling
    drains the queue, every future completes, and the in-flight guards rightly stay quiet — so
    teardown is the only path that meets it. A joining shutdown() waits on it forever; the
    bounded teardown returns within its grace and leaves the wedged pid dead AND reaped."""
    from concurrent.futures import ProcessPoolExecutor, wait

    procs = []
    pool = ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_wedge_one_worker,
        initargs=(str(tmp_path),),
    )
    try:
        futures = [pool.submit(_double, i) for i in range(4)]
        wedged_pid = _await_wedged_pid(tmp_path / "wedged.pid")
        procs = list(pool._processes.values())
        assert len(procs) == 2  # one wedged at birth, one healthy
        done, pending = wait(futures, timeout=60)
        assert not pending  # the healthy sibling drained the whole queue
        assert sorted(f.result() for f in done) == [0, 2, 4, 6]
        assert wedged_pid in {p.pid for p in procs}

        assert _teardown_in_daemon_thread(pool, grace_s=0.5)  # never joins the wedged worker
        # Killed (not "alive" — it would have slept out the hour) AND reaped (not "zombie").
        assert _child_state(wedged_pid) == "gone"
    finally:
        _kill_leftovers(procs)


@_needs_fork
def test_shutdown_pool_grace_lets_healthy_workers_exit_without_a_kill_warning(caplog):
    """A kill is the observable trace of a fork-poisoned worker, so it must stay rare: healthy
    workers exit on their shutdown sentinels inside the grace window, and no warning fires."""
    import logging
    from concurrent.futures import ProcessPoolExecutor

    procs = []
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork"))
    try:
        futures = [pool.submit(_double, i) for i in range(4)]
        assert sorted(f.result() for f in futures) == [0, 2, 4, 6]
        procs = list(pool._processes.values())
        assert procs

        with caplog.at_level(logging.WARNING, logger="noctis.backtest.pool"):
            assert _teardown_in_daemon_thread(pool, grace_s=5.0)
        # They exited on their own shutdown sentinels inside the grace, and were reaped.
        assert [_child_state(p.pid) for p in procs] == ["gone"] * len(procs)
        assert [r.getMessage() for r in caplog.records] == []  # no kill warning on a clean pool
    finally:
        _kill_leftovers(procs)


def test_shutdown_pool_logs_an_empty_process_snapshot_only_when_workers_are_believable():
    """The residual broken-pool race: the executor can clear its process table before teardown
    reads it, putting those pids beyond reach. Log it so the leak is visible — but stay quiet
    for a pool that simply never ran anything, where an empty table is the honest answer."""
    import logging
    from typing import Any, cast

    from noctis.backtest.pool import shutdown_pool

    class _EmptyTablePool:
        def __init__(self, queue_count):
            self._processes: dict = {}
            self._queue_count = queue_count

        def shutdown(self, wait=True, *, cancel_futures=False):
            self._processes = {}

    logger = logging.getLogger("noctis.backtest.pool")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        shutdown_pool(cast(Any, _EmptyTablePool(queue_count=0)))
        assert records == []  # a pool that never submitted work never had workers
        shutdown_pool(cast(Any, _EmptyTablePool(queue_count=3)))
        assert len(records) == 1  # ran work, yet no workers to guard: say so
        assert records[0].levelno == logging.WARNING
    finally:
        logger.removeHandler(handler)


def test_scale_workers_sheds_workers_on_large_1m_panels_but_not_coarse_ones():
    """Memory scales with the panel's total bar count (each worker holds a copy), so the worker
    ceiling is scaled by bars: a ~1.3M-bar 1m panel sheds workers toward sequential, while a ~22k
    -bar 1h panel (≈60× smaller) keeps the full count. This is what prevents the OOM pool hang."""
    from noctis.backtest.pool import scale_workers

    budget = 6_000_000
    bars_1m = 6 * 220_000  # six fit symbols on 1m ≈ 1.32M bars — what OOM'd at 8
    bars_1h = 6 * 3_700  # same span on 1h ≈ 22k bars
    assert scale_workers(8, bars_1m, budget=budget) == 4  # 6M // 1.32M
    assert scale_workers(8, bars_1h, budget=budget) == 8  # tiny panel keeps all workers
    assert scale_workers(8, 0, budget=budget) == 8  # unknown size ⇒ no scaling
    assert scale_workers(8, 10 * budget, budget=budget) == 1  # never below one worker


def test_max_drawdown_hand_computed():
    assert max_drawdown([100, 120, 90, 150]) == pytest.approx(-0.25)


def test_compute_metrics_shapes():
    equity = [100, 110, 105, 120]
    targets = [0, 1, 1, 0]
    m = compute_metrics(equity, targets, periods_per_year=252)
    assert m.total_return == pytest.approx(0.20)
    assert 0.0 <= m.exposure <= 1.0
    assert m.turnover > 0.0


def test_scorecard_json_roundtrip():
    bars = make_ohlcv(price_series(n=250, seed=1))
    cand = Candidate("donchian_breakout", {"channel": 15})
    config = PipelineConfig(prefilter_min_score=None, train_size=120, test_size=40, step=40)
    sc = evaluate(cand, {"AAA": bars}, config=config)
    from noctis.backtest import Scorecard

    restored = Scorecard.from_json(sc.to_json())
    assert restored.to_json() == sc.to_json()
    assert restored.avg_test_metric == pytest.approx(sc.avg_test_metric)
    assert restored.gap == pytest.approx(sc.gap)


def test_forward_holdout_reserved_scored_and_roundtrips():
    bars = make_ohlcv(price_series(n=250, seed=3))
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(
        prefilter_min_score=None, train_size=120, test_size=40, step=40, holdout_size=40
    )
    sc = evaluate(cand, {"AAA": bars}, config=cfg)
    # Walk-forward saw only the first 210 bars (250 − 40 reserved), not the full series.
    assert sc.stage == "validated"
    splits = sc.symbols["AAA"].splits
    assert len(splits) == len(wf(210, 120, 40, 40))
    assert len(splits) < len(wf(250, 120, 40, 40))  # holdout genuinely reduced the search
    assert sc.holdout_metric is not None
    from noctis.backtest import Scorecard

    assert Scorecard.from_json(sc.to_json()).holdout_metric == pytest.approx(sc.holdout_metric)


def test_election_metric_drives_scoring_and_gates():
    """metric_name flows to the Scorecard so score, gap and holdout are all in that metric."""
    bars = make_ohlcv(price_series(n=250, seed=3))
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(
        prefilter_min_score=None,
        train_size=120,
        test_size=40,
        step=40,
        holdout_size=40,
        metric_name="total_return",
    )
    sc = evaluate(cand, {"AAA": bars}, config=cfg)
    assert sc.metric_name == "total_return"
    # avg_test_metric / gap / holdout are read from the total_return field, not sharpe.
    splits = sc.symbols["AAA"].splits
    assert sc.avg_test_metric == pytest.approx(
        sum(s.test.total_return for s in splits) / len(splits)
    )
    assert sc.holdout_metric is not None


def test_avg_test_named_reads_neutral_metric_not_election_metric():
    """avg_test_named('sharpe') reads the Sharpe basis even when the election metric differs."""
    bars = make_ohlcv(price_series(n=250, seed=3))
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(
        prefilter_min_score=None,
        train_size=120,
        test_size=40,
        step=40,
        metric_name="total_return",
    )
    sc = evaluate(cand, {"AAA": bars}, config=cfg)
    # The named read equals the by-hand Sharpe mean across the test splits...
    splits = sc.symbols["AAA"].splits
    expected_sharpe = sum(s.test.sharpe for s in splits) / len(splits)
    assert sc.avg_test_named("sharpe") == pytest.approx(expected_sharpe)
    # ...and avg_test_metric (total_return units) is a genuinely different number, proving
    # the named read is the neutral yardstick, not the election metric.
    assert sc.avg_test_named("sharpe") != pytest.approx(sc.avg_test_metric)
    # avg_test_metric still delegates to the named read on its own metric.
    assert sc.avg_test_metric == pytest.approx(sc.avg_test_named("total_return"))


def test_no_forward_holdout_when_size_zero():
    bars = make_ohlcv(price_series(n=250, seed=3))
    cand = Candidate("donchian_breakout", {"channel": 15})
    sc = evaluate(
        cand, {"AAA": bars}, config=PipelineConfig(prefilter_min_score=None, holdout_size=0)
    )
    assert sc.holdout_metric is None


# --- 6. pipeline short-circuit -----------------------------------------------------------


def test_bad_candidate_killed_at_prefilter_never_validates(monkeypatch):
    """A losing candidate is rejected by the pre-filter; validation is never called."""
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return validate_candidate(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "validate_candidate", _spy)

    # Always-long on a steady downtrend → negative coarse score → killed at prefilter.
    # Long enough for a full split (train+test), so the kill is the prefilter's — not a
    # structural too-short drop.
    downtrend = make_ohlcv([100.0 - 0.1 * i for i in range(200)])
    cand = Candidate("const_long", {})
    sc = evaluate(
        cand,
        {"DDD": downtrend},
        config=PipelineConfig(prefilter_min_score=0.0, train_size=100, test_size=30, step=30),
        families=FAMILIES,
    )
    assert sc.stage == "prefilter_rejected"
    assert sc.prefilter_metric is not None and sc.prefilter_metric <= 0.0
    assert calls["n"] == 0  # validation stage was never reached


def test_prefilter_min_score_none_disables_the_kill_but_still_records_the_score():
    """The explicit no-prefilter entry: the same losing candidate reaches validation, and
    the coarse score is still computed and recorded (a disabled gate hides nothing)."""
    downtrend = make_ohlcv([100.0 - 0.1 * i for i in range(200)])
    sc = evaluate(
        Candidate("const_long", {}),
        {"DDD": downtrend},
        config=PipelineConfig(prefilter_min_score=None, train_size=100, test_size=30, step=30),
        families=FAMILIES,
    )
    assert sc.stage == "validated"
    assert sc.prefilter_metric is not None and sc.prefilter_metric <= 0.0


# --- 7. panel research (multi-symbol evaluate) --------------------------------------------


def test_panel_aggregates_are_per_symbol_means():
    """Panel avg/gap/holdout equal the plain mean of the per-symbol single evaluations."""
    panel = {
        "AAA": make_ohlcv(price_series(n=250, seed=1)),
        "BBB": make_ohlcv(price_series(n=250, seed=2)),
    }
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(
        prefilter_min_score=None, train_size=120, test_size=40, step=40, holdout_size=40
    )
    sc = evaluate(cand, panel, config=cfg)
    assert sc.stage == "validated"
    assert set(sc.symbols) == {"AAA", "BBB"}

    # Panels of one — the single-symbol path IS the panel path, so this comparison is
    # also the panel-of-one equivalence proof.
    singles = {sym: evaluate(cand, {sym: bars}, config=cfg) for sym, bars in panel.items()}
    n = len(singles)
    assert sc.avg_test_metric == pytest.approx(sum(s.avg_test_metric for s in singles.values()) / n)
    assert sc.avg_train_metric == pytest.approx(
        sum(s.avg_train_metric for s in singles.values()) / n
    )
    assert sc.gap == pytest.approx(sum(s.gap for s in singles.values()) / n)
    assert sc.holdout_metric == pytest.approx(sum(s.holdout_metric for s in singles.values()) / n)
    assert sc.panel_dispersion is not None and sc.panel_dispersion >= 0.0
    assert sc.symbol_holdout_metric is None  # no held-out symbols were passed


def test_panel_prefilter_kills_on_median_without_touching_validation(monkeypatch):
    """The MEDIAN coarse score kills the whole candidate — even with one great symbol."""
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return validate_candidate(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "validate_candidate", _spy)

    panel = {
        "DN1": make_ohlcv([100.0 - 0.10 * i for i in range(200)]),
        "DN2": make_ohlcv([100.0 - 0.12 * i for i in range(200)]),
        "UP": make_ohlcv([100.0 + 1.0 * i for i in range(200)]),
    }
    cand = Candidate("const_long", {})
    cfg = PipelineConfig(prefilter_min_score=0.0, train_size=100, test_size=30, step=30)
    sc = evaluate(cand, panel, config=cfg, families=FAMILIES)
    assert sc.stage == "prefilter_rejected"
    assert sc.prefilter_metric is not None and sc.prefilter_metric <= 0.0  # the median
    assert calls["n"] == 0  # no symbol reached validation


def test_panel_never_drops_a_symbol_by_pnl():
    """With a positive median, the losing symbol stays in the panel (no PnL pruning)."""
    panel = {
        "UP1": make_ohlcv([100.0 + 1.0 * i for i in range(200)]),
        "UP2": make_ohlcv([100.0 + 0.8 * i for i in range(200)]),
        "DN": make_ohlcv([300.0 - 0.5 * i for i in range(200)]),
    }
    cand = Candidate("const_long", {})
    cfg = PipelineConfig(prefilter_min_score=0.0, train_size=100, test_size=30, step=30)
    sc = evaluate(cand, panel, config=cfg, families=FAMILIES)
    assert sc.stage == "validated"
    assert set(sc.symbols) == {"UP1", "UP2", "DN"}  # the loser was NOT dropped
    assert not sc.dropped_symbols


def test_panel_drops_structurally_short_symbol_and_records_it():
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(prefilter_min_score=None, train_size=120, test_size=40, step=40)
    panel = {
        "LONG": make_ohlcv(price_series(n=250, seed=1)),
        "SHRT": make_ohlcv(price_series(n=100, seed=2)),  # < train+test = 160
    }
    sc = evaluate(cand, panel, config=cfg)
    assert sc.stage == "validated"
    assert set(sc.symbols) == {"LONG"}
    assert "SHRT" in sc.dropped_symbols
    assert "too short" in sc.dropped_symbols["SHRT"]


def test_panel_parallel_workers_bit_identical_to_sequential():
    """workers=2 must produce the exact same Scorecard as workers=1 (rounding included)."""
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(
        prefilter_min_score=None, train_size=120, test_size=40, step=40, holdout_size=40
    )
    panel = {
        "AAA": make_ohlcv(price_series(n=280, seed=1)),
        "BBB": make_ohlcv(price_series(n=280, seed=2)),
        "CCC": make_ohlcv(price_series(n=280, seed=3)),
    }
    held = {"HHH": make_ohlcv(price_series(n=280, seed=9))}

    sequential = evaluate(cand, panel, config=cfg, symbol_holdout=held, workers=1)
    parallel = evaluate(cand, panel, config=cfg, symbol_holdout=held, workers=2)

    assert sequential.stage == "validated"
    assert parallel.to_dict() == sequential.to_dict()
    assert parallel.panel_dispersion == sequential.panel_dispersion


def test_panel_parallel_preserves_prefilter_short_circuit(monkeypatch):
    """The median kill still short-circuits with workers > 1: validation never runs."""
    calls = {"n": 0}

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return validate_candidate(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "validate_candidate", _spy)
    panel = {
        "DN1": make_ohlcv([100.0 - 0.10 * i for i in range(200)]),
        "DN2": make_ohlcv([100.0 - 0.12 * i for i in range(200)]),
    }
    cand = Candidate("const_long", {})
    cfg = PipelineConfig(prefilter_min_score=0.0, train_size=100, test_size=30, step=30)
    sc = evaluate(cand, panel, config=cfg, workers=2, families=FAMILIES)
    assert sc.stage == "prefilter_rejected"
    assert sc.symbols == {}
    # Parent-side validate never ran; worker processes can't touch the parent's spy either,
    # but the definitive check is the stage + absent per-symbol scores above.
    assert calls["n"] == 0


def test_symbol_holdout_scored_with_one_causal_pass():
    from noctis.backtest.validate import score_window

    panel = {"AAA": make_ohlcv(price_series(n=250, seed=1))}
    held = {
        "HH1": make_ohlcv(price_series(n=250, seed=8)),
        "HH2": make_ohlcv(price_series(n=250, seed=9)),
    }
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(prefilter_min_score=None, train_size=120, test_size=40, step=40)
    sc = evaluate(cand, panel, config=cfg, symbol_holdout=held)
    expected = sum(
        score_window(cand, bars, cfg.validation).get(cfg.metric_name) for bars in held.values()
    ) / len(held)
    assert sc.symbol_holdout_metric == pytest.approx(expected)
    # Held-out symbols never join the fit panel.
    assert set(sc.symbols) == {"AAA"}


def test_panel_scorecard_roundtrip_and_legacy_json_backcompat():
    from noctis.backtest import Scorecard

    panel = {
        "AAA": make_ohlcv(price_series(n=250, seed=1)),
        "BBB": make_ohlcv(price_series(n=250, seed=2)),
    }
    held = {"HH1": make_ohlcv(price_series(n=250, seed=8))}
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig(
        prefilter_min_score=None, train_size=120, test_size=40, step=40, holdout_size=40
    )
    sc = evaluate(cand, panel, config=cfg, symbol_holdout=held)
    restored = Scorecard.from_json(sc.to_json())
    assert restored.to_json() == sc.to_json()
    assert restored.avg_test_metric == pytest.approx(sc.avg_test_metric)
    assert restored.symbol_holdout_metric == pytest.approx(sc.symbol_holdout_metric)
    assert set(restored.symbols) == set(sc.symbols)

    # A pre-panel scorecard JSON (no panel keys at all) still loads; with no splits either
    # it normalizes to an empty panel — no out-of-sample evidence, so decide() rejects it.
    legacy = {
        k: v
        for k, v in sc.to_dict().items()
        if k not in ("symbols", "symbol_holdout_metric", "panel_dispersion", "dropped_symbols")
    }
    old = Scorecard.from_dict(legacy)
    assert old.symbols == {}
    assert old.symbol_holdout_metric is None
    assert old.panel_dispersion is None
    assert old.dropped_symbols is None


def test_legacy_single_scorecard_loads_as_panel_of_one():
    """Old persisted single-symbol cards (top-level splits) read back as a panel of one.

    The sentinel symbol keys the splits; every aggregate keeps the meaning it had when the
    card was written, so legacy champions rank unchanged until displaced.
    """
    from noctis.backtest import Scorecard
    from noctis.backtest.scorecard import LEGACY_SYMBOL

    m = {
        "total_return": 0.1,
        "sharpe": 1.5,
        "sortino": 2.0,
        "max_drawdown": -0.1,
        "win_rate": 0.6,
        "turnover": 0.2,
        "exposure": 0.5,
    }
    legacy = {
        "family": "sma_crossover",
        "params": {"fast": 5, "slow": 20},
        "metric_name": "sharpe",
        "stage": "validated",
        "prefilter_metric": 0.9,
        "holdout_metric": 1.25,
        "avg_train_metric": 1.5,
        "avg_test_metric": 1.5,
        "gap": 0.0,
        "splits": [
            {"split_index": 0, "train": m, "test": m},
            {"split_index": 1, "train": m, "test": m},
        ],
    }
    card = Scorecard.from_dict(legacy)
    assert set(card.symbols) == {LEGACY_SYMBOL}
    assert len(card.symbols[LEGACY_SYMBOL].splits) == 2
    assert card.avg_test_metric == pytest.approx(1.5)
    assert card.avg_train_metric == pytest.approx(1.5)
    assert card.gap == pytest.approx(0.0)
    assert card.holdout_metric == pytest.approx(1.25)
    assert card.test_activity == 1.0  # every split traded (exposure > 0)
    # Round-trips in the always-panel shape.
    restored = Scorecard.from_json(card.to_json())
    assert restored.to_json() == card.to_json()
    assert set(restored.symbols) == {LEGACY_SYMBOL}


# --- config: one home for split geometry + election-metric threading ----------------------


def test_pipeline_config_auto_geometry_matches_heuristic():
    """auto() owns the split-geometry heuristic both research loops used to copy."""
    cfg = PipelineConfig.auto(250)
    assert (cfg.train_size, cfg.test_size, cfg.step) == (83, 40, 40)
    assert cfg.holdout_size == 40  # 250 − 40 ≥ 83 + 40 → one test-window reserved
    # Small data degrades to no holdout rather than starving the search.
    small = PipelineConfig.auto(70)
    assert (small.train_size, small.test_size, small.holdout_size) == (40, 20, 0)
    # Long series cap at the fixed ceiling.
    big = PipelineConfig.auto(10_000)
    assert (big.train_size, big.test_size, big.step, big.holdout_size) == (120, 40, 40, 40)


@pytest.mark.parametrize("metric", ["sharpe", "sortino", "total_return"])
def test_prefilter_scores_every_election_metric(metric):
    """auto() threads the election metric into the coarse screen, so the screen must be
    able to rank on every electable metric (config.yaml elects sortino in production)."""
    bars = make_ohlcv(price_series(n=250, seed=1))
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig.auto(250, metric=metric, prefilter_min_score=None)
    score = coarse_score(cand, bars, cfg.prefilter)
    assert isinstance(score, float) and score == score  # finite, not NaN
    sc = evaluate(cand, {"AAA": bars}, config=cfg)
    assert sc.stage == "validated"
    assert sc.metric_name == metric


def test_pipeline_config_auto_threads_metric_and_periods_once():
    """The election metric and annualization are stated once and flow to every stage."""
    cfg = PipelineConfig.auto(
        250, metric="sortino", periods_per_year=98_280, prefilter_min_score=None
    )
    assert cfg.metric_name == "sortino"
    assert cfg.prefilter.metric == "sortino"  # coarse screen ranks on the elected metric
    assert cfg.prefilter.periods_per_year == 98_280
    assert cfg.validation.periods_per_year == 98_280
    assert cfg.prefilter_min_score is None


def test_unknown_metric_fails_one_way_in_one_place():
    """Metric.parse is the single diagnosis — auto() and the coarse screen (and, at their
    own seams, settings / --metric / the mandate overlay) all surface its one message."""
    from noctis.backtest import Metric

    with pytest.raises(ValueError, match=r"unknown metric 'alpha'") as parse_err:
        Metric.parse("alpha")
    assert "valid: sharpe, sortino, total_return" in str(parse_err.value)
    with pytest.raises(ValueError, match=r"unknown metric 'alpha'"):
        PipelineConfig.auto(250, metric="alpha")
    bars = make_ohlcv(price_series(n=60, seed=3))
    with pytest.raises(ValueError, match=r"unknown metric 'alpha'"):
        # A plain-string config bypasses the typed field; the screen still fails the one way.
        coarse_score(Candidate("const_long", {}), bars, PrefilterConfig(metric="alpha"), FAMILIES)


def test_metric_members_read_off_a_metrics_record():
    """Every election metric is a Metrics field, so a Metric always reads off a scored
    record via Metrics.get — the enum cannot drift from the record shape."""
    from noctis.backtest import Metric

    m = compute_metrics([100.0, 101.0, 102.0], [1, 1, 1])
    for member in Metric:
        assert m.get(member) == m.get(member.value)


# --- readiness guard ---------------------------------------------------------------------


def test_require_symbols_ready(tmp_path):
    from noctis.backtest import SymbolNotReadyError
    from noctis.data import CoverageRegistry, SeriesKey

    reg = CoverageRegistry(tmp_path / "cov.db")
    with pytest.raises(SymbolNotReadyError):
        require_symbols_ready(reg, ["AAPL"])
    reg.upsert(SeriesKey("EQUS.MINI", "ohlcv-1m", "AAPL"), first_ts=0, last_ts=1, row_count=10)
    require_symbols_ready(reg, ["AAPL"])  # now ready, no raise


def test_prefilter_screen_ranks_and_shortlists():
    from noctis.backtest import screen

    bars = make_ohlcv(price_series(n=200, seed=2))
    cands = [
        Candidate("sma_crossover", {"fast": 5, "slow": 20}),
        Candidate("sma_crossover", {"fast": 3, "slow": 50}),
        Candidate("donchian_breakout", {"channel": 20}),
    ]
    top = screen(cands, bars, top_k=2, config=PrefilterConfig())
    assert len(top) == 2
    assert top[0].score >= top[1].score  # ranked descending


# --- in-process evaluation time limit (the sequential sibling of the stall guard) --------


def test_evaluation_time_limit_bounds_a_hung_python_loop_and_restores_the_timer():
    """The sequential-path hang fix: workers=1 evaluations run in-process where no pool guard
    can fire, so a strategy loop that never terminates would wedge the research loop forever.
    The alarm turns that into a bounded EvaluationTimeout — and leaves no timer armed after."""
    import signal
    import time

    from noctis.backtest.pool import EvaluationTimeout, evaluation_time_limit

    before = signal.getsignal(signal.SIGALRM)
    start = time.monotonic()
    with pytest.raises(EvaluationTimeout):
        with evaluation_time_limit(0.2):
            while True:  # the hang: pure-Python, param-independent
                pass
    assert time.monotonic() - start < 30
    # The exit disarmed the timer and restored the previous handler — no stray alarm later.
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is before


def test_evaluation_time_limit_is_a_noop_off_the_main_thread():
    """SIGALRM only works on the main thread; anywhere else the guard must degrade to a clean
    no-op (unguarded, but never an error) so threaded callers and tests keep working."""
    import threading

    from noctis.backtest.pool import evaluation_time_limit

    result: list[str] = []

    def work():
        with evaluation_time_limit(0.01):
            result.append("ran")

    t = threading.Thread(target=work)
    t.start()
    t.join(timeout=10)
    assert result == ["ran"]


def _raise_in_worker(_task):
    raise ValueError("boom in worker")


def _healthy_in_worker(task):
    return task


@_needs_fork
def test_panel_pool_close_tears_down_with_the_bounded_helper(monkeypatch):
    """Regression for the clean-path freeze: every task completed, so the stall guard rightly
    stayed quiet — and close() then joined a fork-poisoned worker that never exits, freezing a
    finished run for hours. A fully drained panel pool is not a pool whose every worker is
    joinable, so the clean close must route through the bounded teardown too, at its full grace.
    """
    import time

    from noctis.backtest.pipeline import _PanelPool
    from noctis.backtest.pool import POOL_TEARDOWN_GRACE_S, shutdown_pool

    graces: list[float] = []

    def recording_shutdown(pool, *, grace_s=POOL_TEARDOWN_GRACE_S):
        graces.append(grace_s)
        shutdown_pool(pool, grace_s=grace_s)

    monkeypatch.setattr("noctis.backtest.pipeline.shutdown_pool", recording_shutdown)

    pool = _PanelPool(workers=2, tasks=2)
    assert pool._pool is not None  # a real fork pool, or the test proves nothing
    assert pool.map(_healthy_in_worker, [1, 2]) == [1, 2]  # the pool drained every task
    procs = list((pool._pool._processes or {}).values())

    start = time.monotonic()
    pool.close()
    assert time.monotonic() - start < 30  # bounded: the old join would still be waiting
    assert graces == [POOL_TEARDOWN_GRACE_S]  # bounded teardown, clean-path grace (5.0s)
    deadline = time.monotonic() + 10
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(p.is_alive() for p in procs)


def test_panel_pool_abort_never_joins_and_close_is_a_noop_after():
    """The unwind fix: a task exception propagates out of _PanelPool.map (only pool failures
    fall back), and evaluate()'s cleanup must then tear down via abort() — shutdown()'s join
    would hang forever if another worker were wedged. abort() returns promptly, kills the
    workers, and leaves close() a no-op."""
    import time

    from noctis.backtest.pipeline import _PanelPool

    pool = _PanelPool(workers=2, tasks=2)
    assert pool._pool is not None  # a real fork pool, or the test proves nothing
    with pytest.raises(ValueError, match="boom in worker"):
        pool.map(_raise_in_worker, [None, None])
    procs = list((pool._pool._processes or {}).values())
    start = time.monotonic()
    pool.abort()
    assert time.monotonic() - start < 30
    pool.close()  # must be a clean no-op after abort
    deadline = time.monotonic() + 10
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(p.is_alive() for p in procs)
