"""The bench pool — the runner's execution seam, widened to worker processes.

A bench is embarrassingly parallel and slow for a reason nobody owns: one job is one case under one
configuration on one rep, it shares nothing with the next, and its cost is a network round trip.
So the pool is worth having — and the interesting part of it is not the speedup, it is what happens
when a worker stops answering. This module is the eval layer's answer, and it is deliberately the
engine's answer transposed rather than a second one: the guards in :mod:`noctis.backtest.pool` were
paid for in production hangs (a 10-hour OOM wedge, a 142-minute freeze *after* a sweep had finished
its work), and :class:`~noctis.research.sweep.SweepRunner` already wired them into a pool of this
exact shape. Every one of them is **imported**, never re-implemented — the eval layer may import
the engine, which is what that one-way boundary is for.

**Three failure modes, one guard each.** A worker that dies outright breaks the executor and every
pending future with it; a worker killed by the OOM killer can leave the parent futex-blocked with
no ``BrokenProcessPool`` ever raised, so each batch is waited on through :func:`wait_or_stall` and
a batch that stops making progress is declared wedged; a fork-poisoned worker deadlocks in its
initializer *before* it dequeues anything, so its healthy siblings drain the queue, every future
completes, the stall guard rightly stays quiet — and only teardown ever meets it. That is why
teardown always runs through :func:`shutdown_pool` (shut down without waiting, one bounded grace,
then kill), on the clean path as well as the wedged one.

**Degrading means continuing, not restarting.** Whatever the pool finished is harvested and kept —
including the finished half of a batch its wedged sibling stalled — and only the jobs with no
result are worked in-process afterwards. A completed job is never re-run: results are collected by
job index, so the sequential tail is exactly the complement of what the pool achieved, and it is
also what runs when there was never a pool at all.

**Nothing heavy crosses a pickle boundary.** The work callable and the job list are handed to the
workers by ``fork`` **inheritance** through the pool initializer — with the fork context, a
``Process``'s arguments are inherited rather than pickled — and the call queue carries only a job
index out and one result back. That is what lets a bench parallelise over sites whose renderers are
closures and cases whose payloads are read-only views: neither has to be picklable, and the runner
itself never is. A platform without ``fork`` raises on construction and takes the sequential path,
which is a slower bench and not a broken one.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from noctis.backtest.pool import (
    POOL_TEARDOWN_GRACE_S,
    PoolStalled,
    shutdown_pool,
    wait_or_stall,
)

__all__ = ["PooledExecutor"]

logger = logging.getLogger("noctis.eval.pool")

_Job = TypeVar("_Job")
_Done = TypeVar("_Done")


# ── the worker side ───────────────────────────────────────────────────────────────────────
#
# Per-process state, installed once by the pool initializer and read by every job the worker
# takes. Module-level because the submitted callable must pickle by reference; the *state* it
# reads never travels — under the fork context the initializer's arguments are inherited with the
# rest of the parent's image, so an unpicklable work callable (a bound method of a runner holding
# a site registry, say) is carried across intact.
_WORKER_WORK: Callable[[Any], Any] | None = None
_WORKER_JOBS: Sequence[Any] = ()


def _worker_init(work: Callable[[Any], Any], jobs: Sequence[Any]) -> None:
    global _WORKER_WORK, _WORKER_JOBS  # noqa: PLW0603 — per-process state, set once at fork
    _WORKER_WORK, _WORKER_JOBS = work, jobs


def _worker_job(index: int) -> Any:
    """Work the job at ``index`` — the only thing the call queue ever carries."""
    work = _WORKER_WORK
    if work is None:  # pragma: no cover — a worker always runs its initializer first
        raise RuntimeError("a bench worker was asked for a job before it was initialized")
    return work(_WORKER_JOBS[index])


# ── the executor ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PooledExecutor:
    """Work a bench's jobs on a pool of worker processes, and survive the pool.

    Satisfies the runner's ``JobExecutor`` seam, so a pooled bench and a sequential one differ in
    how they were worked and in nothing a record can see. ``workers`` is stated rather than
    defaulted: a bench's width is a decision about somebody's provider rate limit, and it is
    deliberately **not** capped by the core count the way a sweep's is — a bench job is a network
    round trip, not a panel evaluation, so cores are not what bounds it and no worker here holds a
    copy of a panel to be scaled down by memory.

    The three collaborators below are what make a wedged pool testable through this interface
    rather than through a monkeypatched global — the discipline
    :class:`~noctis.research.sweep.SweepRunner` established:

    * ``worker`` — what each worker applies to its job. ``None`` (the honest default) means the
      work callable the caller handed :meth:`run`; a test injects a worker that wedges or dies.
    * ``wait`` — the waiting policy for one batch, defaulting to the stall guard itself.
    * ``teardown`` — how a pool is disposed of, defaulting to the bounded, never-joining helper.
    """

    workers: int
    worker: Callable[[Any], Any] | None = None
    wait: Callable[..., None] = wait_or_stall
    teardown: Callable[..., None] = shutdown_pool
    # The grace a *clean* exit gives its workers to leave on their own sentinels. A pool already
    # declared wedged or broken gets none: it has proved it is not going to answer.
    grace_s: float = POOL_TEARDOWN_GRACE_S

    def run(self, work: Callable[[_Job], _Done], jobs: Sequence[_Job]) -> tuple[_Done, ...]:
        """Every job worked exactly once, returned in job order however the pool answered.

        The tail loop is both halves of the contract at once: it is the sequential degrade when the
        pool failed part-way, and it is the whole execution when there was no pool to begin with.
        Either way it works only the jobs with no result yet, so nothing already finished is asked
        for twice.
        """
        queued = tuple(jobs)
        done: dict[int, _Done] = {}
        if min(self.workers, len(queued)) > 1:
            self._pooled(work, queued, done)
        for index, job in enumerate(queued):
            if index not in done:
                done[index] = work(job)
        return tuple(done[index] for index in range(len(queued)))

    def _pooled(
        self, work: Callable[[_Job], _Done], jobs: tuple[_Job, ...], done: dict[int, _Done]
    ) -> None:
        """Work as many jobs as the pool will take, recording every one it finishes into ``done``.

        Never raises for a pool failure: a broken, stalled or unavailable pool is a slower bench,
        so it is logged and left to the caller's sequential tail. Batching is per pool width, which
        is what makes the stall guard's blast radius one batch rather than the whole bench.
        """
        width = min(self.workers, len(jobs))
        pool: ProcessPoolExecutor | None = None
        clean = False
        try:
            # Fork explicitly, for the sweep's two reasons: the 3.14 default (forkserver) re-imports
            # __main__ in every worker — under a console-script entry point that re-runs the CLI —
            # and inheriting the parent's image is what carries the work callable and the jobs over
            # without pickling them. A platform without fork raises here and takes the tail.
            pool = ProcessPoolExecutor(
                max_workers=width,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_worker_init,
                initargs=(work if self.worker is None else self.worker, jobs),
            )
            for start in range(0, len(jobs), width):
                futures = {
                    index: pool.submit(_worker_job, index)
                    for index in range(start, min(start + width, len(jobs)))
                }
                try:
                    self.wait(list(futures.values()))
                finally:
                    # Harvested on every path: a batch one wedged worker stalled still holds its
                    # siblings' finished work, and throwing that away would re-ask for answers
                    # somebody already paid a provider for.
                    _harvest(futures, done)
            clean = True
        except (BrokenExecutor, PoolStalled, OSError, ValueError) as exc:
            logger.warning(
                "bench pool unavailable (%s); finishing the remaining %d job(s) in-process",
                exc,
                len(jobs) - len(done),
            )
        finally:
            # Never a `with` block and never a joining shutdown on ANY path: a fork-poisoned worker
            # deadlocks before it dequeues anything, so its siblings complete every future while it
            # never exits — and joining it there is the freeze this helper exists to prevent.
            if pool is not None:
                self.teardown(pool, grace_s=self.grace_s if clean else 0.0)


def _harvest(futures: dict[int, Future[_Done]], done: dict[int, _Done]) -> None:
    """Keep every result the pool actually produced; leave the rest to the sequential tail.

    A future that is not done is never read — reading one is the unbounded wait the stall guard
    just refused. A job that *raised* is left unrecorded on purpose rather than swallowed: the
    sequential tail re-runs it in-process, where it surfaces to the caller exactly as it would have
    under ``SequentialExecutor``. A bench job that raises is a defect in the corpus or the site (a
    prompt that cannot be composed), never a provider falling over — the runner turns that one into
    a failed attempt long before it reaches here — so it must not be quietly demoted to a warning.
    """
    for index, future in futures.items():
        if not future.done():
            continue
        try:
            done[index] = future.result()
        except BrokenExecutor:
            raise
        except Exception as exc:  # noqa: BLE001 — one bad job ≠ a dead bench
            logger.warning(
                "bench job %d failed in a worker (%s); it will be re-run in-process", index, exc
            )


if TYPE_CHECKING:  # the seam this class exists to satisfy, checked rather than asserted in prose
    from noctis.eval.runner import JobExecutor

    _seam: JobExecutor = PooledExecutor(workers=1)
