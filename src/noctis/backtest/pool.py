"""Shared process-pool result collection with a stall guard.

A ``ProcessPoolExecutor`` worker killed by the OOM killer can leave the parent blocked in
``future.result()`` / ``pool.map()`` **forever**: ``BrokenProcessPool`` is not always raised
(observed live — a 10-hour hang with four zombie workers, the parent futex-blocked). Waiting on
the pool with a stall timeout turns that infinite hang into a recoverable signal, so callers fall
back to sequential evaluation — which also uses far less memory and usually completes.

:func:`evaluation_time_limit` is the stall guard's sequential sibling: the fallback (and any
``workers=1`` panel) evaluates **in-process**, where no pool guard can fire — a strategy whose
``on_bar`` loop never terminates on a pathological param set would hang the research loop
forever. The alarm turns that into a bounded :class:`EvaluationTimeout`.
"""

from __future__ import annotations

import concurrent.futures
import signal
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

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


def shutdown_wedged(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    """Tear down a pool presumed wedged, never waiting on its workers.

    ``shutdown(wait=True)`` — including the ``with`` form's ``__exit__`` — joins the worker
    processes. A worker deadlocked on a fork-inherited lock never exits, so that join freezes
    the parent forever (observed live: the stall guard raised :class:`PoolStalled`, then the
    sweep pool's ``with`` block re-froze the run while unwinding it). Shut down without
    waiting, then SIGKILL any worker still alive so a wedged child doesn't linger
    futex-blocked holding its full copy of the panel.
    """
    # shutdown() drops the process table — snapshot the workers first so they can be killed.
    procs = list((pool._processes or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        if proc.is_alive():
            proc.kill()


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
