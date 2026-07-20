"""Shared pool guards — the catalogue of known pool hangs, one guard each.

A ``ProcessPoolExecutor`` worker killed by the OOM killer can leave the parent blocked in
``future.result()`` / ``pool.map()`` **forever**: ``BrokenProcessPool`` is not always raised
(observed live — a 10-hour hang with four zombie workers, the parent futex-blocked). Waiting on
the pool with a stall timeout turns that infinite hang into a recoverable signal, so callers fall
back to sequential evaluation — which also uses far less memory and usually completes.

:func:`evaluation_time_limit` is the stall guard's sequential sibling: the fallback (and any
``workers=1`` panel) evaluates **in-process**, where no pool guard can fire — a strategy whose
``on_bar`` loop never terminates on a pathological param set would hang the research loop
forever. The alarm turns that into a bounded :class:`EvaluationTimeout`.

The third failure mode shows no in-flight symptom at all, which is why it needs its own guard.
The pool forks from a parent with many live threads (HTTP clients, an allocator thread, …), so a
worker can inherit a lock some other thread held at the instant of ``fork()`` and deadlock in its
initializer — **before it ever dequeues a task**. Its healthy siblings drain the whole queue, so
every future completes and the stall guard rightly stays quiet; only teardown ever meets that
worker, and a joining ``shutdown()`` waits on it forever (observed live: an overnight session
frozen for hours *after* its sweep had finished its work; SIGKILLing the one wedged pid unfroze
it instantly). :func:`shutdown_pool` is that mode's guard — every teardown path is bounded, and
the kill warning it logs is the observable trace of a fork-poisoned worker.
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing.connection
import signal
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("noctis.backtest.pool")

# Seconds to wait for ANY pending future to make progress before declaring the pool wedged.
# Generous on purpose: a single 1m-bar trial/symbol is seconds to ~a minute, so whole minutes of
# zero progress across every worker is an unambiguous stall (an OOM-killed worker tree), never
# mere slowness. Bounds a hang to minutes instead of the 10 hours seen in the wild.
POOL_STALL_TIMEOUT_S = 600.0


class PoolStalled(RuntimeError):
    """No pending future completed within the stall timeout — the pool is wedged (e.g. a worker
    was OOM-killed without the executor raising ``BrokenProcessPool``)."""


# Wall-clock ceiling on ONE in-process candidate evaluation. Hang insurance, not pacing: an
# order of magnitude above any legitimate full-panel evaluation (per-symbol work is seconds to
# ~a minute), so it only ever fires on a genuine hang — a strategy loop that cannot terminate
# on the params it was handed — and converts it into a recoverable failure.
EVAL_TIME_LIMIT_S = 1800.0


class EvaluationTimeout(RuntimeError):
    """One in-process evaluation exceeded its wall-clock ceiling — treat the trial as hung and
    move on; never let it wedge the research loop."""


@contextmanager
def evaluation_time_limit(seconds: float = EVAL_TIME_LIMIT_S) -> Iterator[None]:
    """Bound one in-process evaluation in wall-clock time; raise :class:`EvaluationTimeout`.

    SIGALRM-based, so it needs the main thread and a POSIX platform — anywhere else (or with a
    non-positive ``seconds``) it is a clean no-op, never an error. The alarm interrupts Python
    bytecode only, which covers the realistic hang (a strategy's ``on_bar`` loop); a hang buried
    inside a C extension is out of reach, but the pool paths already guard those via
    :func:`wait_or_stall`. Not reentrant: never nest it — the inner exit would cancel the outer
    timer. Today it is armed only at the two evaluation seams (the research toolbox's
    ``_evaluate`` and the runtime's legacy ``_evaluate``), which never nest.
    """
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _raise(signum, frame):
        raise EvaluationTimeout(
            f"evaluation exceeded {seconds:.0f}s wall-clock; treating it as hung"
        )

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


# A single sweep/panel worker holds a full copy of the panel bars plus per-trial intermediates,
# so its peak memory scales with the panel's TOTAL bar count — a 1m panel is ~60× a 1h one. A
# fixed worker count therefore OOM-kills workers on fine timeframes (the 10h hang) while
# under-using cores on coarse ones. Callers scale by this budget instead.
DEFAULT_WORKER_BAR_BUDGET = 6_000_000  # ceiling on (workers × total panel bars) held in flight


def scale_workers(
    configured: int, total_bars: int, *, budget: int = DEFAULT_WORKER_BAR_BUDGET
) -> int:
    """Cap ``configured`` workers so ``workers × total_bars <= budget`` (never below 1).

    ``total_bars`` is the panel's summed bar count — the real memory driver — computed at the
    call site from the actual bars, so it reflects timeframe, symbol count, and any ``max_bars``
    cap at once. A large 1m panel scales down toward sequential; a small 1h+ panel keeps the full
    count. A non-positive ``total_bars`` or ``budget`` disables scaling (returns ``configured``).
    """
    if total_bars <= 0 or budget <= 0:
        return max(1, configured)
    return max(1, min(configured, budget // total_bars))


# Grace given to the WHOLE worker snapshot (one shared deadline, not per worker) to exit on
# its shutdown sentinels before the kill. Healthy workers take milliseconds, so this costs a
# clean teardown nothing; it exists so an ordinary exit stays ordinary — SIGKILL never becomes
# the routine path, which is what keeps the kill warning meaningful as a fork-poisoning signal.
POOL_TEARDOWN_GRACE_S = 5.0

# Ceiling on the reap-join of a process we just SIGKILLed. A killed child is reaped in
# microseconds; the bound is here so even this join can never be the thing that hangs.
POOL_REAP_JOIN_S = 5.0


def _reap(proc: multiprocessing.process.BaseProcess, *, timeout: float) -> None:
    """Bounded ``join`` of an exited worker so it leaves no zombie; never raises.

    The executor's management thread joins the same workers, so this join can find the child
    already reaped (``waitpid`` → ECHILD, swallowed by multiprocessing) — either way the pid is
    collected and this returns within ``timeout``.
    """
    try:
        proc.join(timeout=timeout)
    except (ValueError, AssertionError, OSError):  # not started / already closed out
        pass


def shutdown_pool(
    pool: concurrent.futures.ProcessPoolExecutor, *, grace_s: float = POOL_TEARDOWN_GRACE_S
) -> None:
    """Tear down a process pool without ever joining a worker unboundedly.

    The only pool-teardown primitive in the codebase. ``shutdown(wait=True)`` — including the
    ``with`` form's ``__exit__`` — joins the worker processes, and a worker deadlocked on a
    fork-inherited lock never exits, so that join freezes the parent forever (see the module
    docstring's third failure mode). Instead: snapshot the workers, shut down *without* waiting,
    give the snapshot one shared ``grace_s`` window to exit on its sentinels, then SIGKILL every
    survivor and reap it with a bounded join so no zombie — and no futex-blocked child still
    holding its full copy of the panel — is left behind. ``grace_s=0.0`` kills immediately.

    The executor's management thread is deliberately left alone: with the workers dead it
    finishes and exits on its own, whereas joining it is exactly the unbounded wait being fixed.
    """
    # shutdown() drops the process table — snapshot the workers first so they can be killed.
    procs = list((pool._processes or {}).values())
    submitted = getattr(pool, "_queue_count", None)  # work items ever handed to this pool
    if not procs and submitted:
        # The pool ran work, so it HAD workers; an empty table means the executor cleared it
        # first (the residual broken-pool race) and those pids are now beyond our reach. There
        # is nothing left to guard here — log it rather than let the leak be silent.
        logger.warning(
            "pool teardown found no worker processes on a pool that ran work; any surviving "
            "worker is beyond reach (the executor cleared its process table first)"
        )
    pool.shutdown(wait=False, cancel_futures=True)

    # Aliveness is read from each worker's sentinel (readable the moment the process exits),
    # never from ``Process.is_alive()``: the executor's own management thread reaps the workers
    # too, and whichever ``waitpid`` loses that race gets ECHILD — after which the Process object
    # claims "alive" forever. Killing on that lie would signal a recycled pid.
    pending: dict[Any, multiprocessing.process.BaseProcess] = {}
    for proc in procs:
        try:
            pending[proc.sentinel] = proc
        except ValueError:  # already closed out — nothing to wait on
            continue

    # ONE deadline for the whole snapshot: healthy workers exit on their shutdown sentinels in
    # milliseconds, so the grace is free on a clean teardown and never multiplies by worker
    # count on a wedged one. ``grace_s=0.0`` degenerates to a single non-blocking poll.
    deadline = time.monotonic() + grace_s
    while pending:
        remaining = max(0.0, deadline - time.monotonic())
        exited = multiprocessing.connection.wait(list(pending), timeout=remaining)
        for sentinel in exited:
            _reap(pending.pop(sentinel), timeout=0.0)
        if not exited and remaining <= 0:
            break

    killed = []
    for survivor in pending.values():
        try:
            survivor.kill()
        except (ValueError, ProcessLookupError):  # closed out / exited under us
            continue
        killed.append(survivor.pid)
        _reap(survivor, timeout=POOL_REAP_JOIN_S)  # bounded reap: kill, don't leave a zombie
    if killed:
        # Named pids on purpose: with a real grace window, a worker that outlives its shutdown
        # sentinel is a fork-poisoned worker, and this line is the only trace it ever leaves.
        logger.warning(
            "pool teardown killed %d worker(s) still alive %.1fs after a non-waiting "
            "shutdown: pids %s",
            len(killed),
            grace_s,
            ", ".join(str(pid) for pid in killed),
        )


def wait_or_stall(
    futures: Iterable[concurrent.futures.Future],
    *,
    timeout: float = POOL_STALL_TIMEOUT_S,
) -> None:
    """Block until every future is done, raising :class:`PoolStalled` if the whole set goes
    ``timeout`` seconds with no completion.

    Watches ALL pending futures for ANY progress, so a merely-slow task never trips the guard
    while the pool keeps advancing. Reads no results — the caller retrieves each
    ``future.result()`` with its own per-item error handling, and a genuine
    ``BrokenProcessPool`` still surfaces there.
    """
    pending = set(futures)
    while pending:
        done, pending = concurrent.futures.wait(
            pending, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
        )
        if not done:
            raise PoolStalled(
                f"no process-pool result in {timeout:.0f}s; treating the pool as wedged "
                "(likely an OOM-killed worker) and falling back to sequential evaluation"
            )
