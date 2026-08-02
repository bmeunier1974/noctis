"""The bench worker pool (#203): the runner's execution seam, widened to worker processes.

A bench is embarrassingly parallel — one case, one configuration, one rep is a job that shares
nothing with the next — and it is also dominated by a round trip nobody controls, so the pool's
real subject is not speed but **survival**: an OOM-killed worker, a fork-poisoned one that never
dequeues anything, a job that falls over. The engine has already paid for those lessons in
``noctis/backtest/pool.py`` and ``noctis/research/sweep.py``, so this suite drives the same three
against a bench.

Every failure mode is driven through the executor's own interface — the worker callable, the
waiting policy and the teardown are injectable on :class:`~noctis.eval.pool.PooledExecutor` — so
nothing here monkeypatches a module global, and a wedged pool is an argument rather than a mock.
The ledger the workers write is what makes "a completed job is never re-run" checkable from
outside: a worker records **its own pid** against the job it finished, so a test can tell the two
halves of a degraded run apart without asking the executor anything.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from noctis.backtest.pool import POOL_TEARDOWN_GRACE_S, PoolStalled, shutdown_pool, wait_or_stall
from noctis.eval.pool import PooledExecutor
from noctis.eval.record import validate
from noctis.eval.runner import replay
from tests.test_eval_runner import StubAttempt, _corpus, _run, _runner

# One unit of work: which job it is, and the ledger a worker marks it in.
Job = tuple[int, Path]


def _jobs(ledger: Path, count: int) -> tuple[Job, ...]:
    return tuple((index, _made(ledger)) for index in range(count))


def _made(ledger: Path) -> Path:
    ledger.mkdir(parents=True, exist_ok=True)
    return ledger


def _worked_by(ledger: Path) -> dict[int, int]:
    """Every job the pool finished, and the pid of the process that finished it."""
    return {
        int(path.name.removeprefix("job-")): int(path.read_text(encoding="utf-8"))
        for path in ledger.iterdir()
    }


# ── worker behaviours (they run in a forked child, so they take the job and nothing else) ──


def _in_the_pool(job: Job) -> int:
    """A worker that finishes its job, and signs the ledger with the pid that did it."""
    index, ledger = job
    (ledger / f"job-{index}").write_text(str(os.getpid()), encoding="utf-8")
    return index


def _finishes_in_reverse(job: Job) -> int:
    """The later the job, the sooner it lands — so completion order is not job order."""
    time.sleep(0.05 * (4 - job[0]))
    return _in_the_pool(job)


def _dies_from_the_third_job_on(job: Job) -> int:
    if job[0] >= 2:
        os._exit(1)  # an OOM-killed worker: the process vanishes mid-batch, mid-job
    return _in_the_pool(job)


def _wedges_forever(job: Any) -> Any:
    time.sleep(3600)  # a fork-poisoned worker: it never answers, and it never dies


def _wedges_after_the_first_job(job: Job) -> int:
    return _in_the_pool(job) if job[0] == 0 else _wedges_forever(job)


# ── the two policies a test injects instead of the real ones ──────────────────────────────


def _stall_at_once(futures: Any, **_kwargs: Any) -> None:
    raise PoolStalled("simulated wedge")


def _stall_on_a_short_fuse(futures: Any, **_kwargs: Any) -> None:
    """The real guard, fused short: a batch that stops making progress is declared wedged."""
    wait_or_stall(futures, timeout=0.5)


@dataclass
class RecordingWork:
    """The parent-side work callable — what the sequential path ran, in the order it ran it."""

    ran: list[int] = field(default_factory=list)

    def __call__(self, job: Job) -> int:
        self.ran.append(job[0])
        return job[0]


@dataclass
class RecordingTeardown:
    """The bounded teardown, wrapped so a test can read the grace the executor gave it."""

    graces: list[float] = field(default_factory=list)

    def __call__(self, pool: Any, *, grace_s: float = POOL_TEARDOWN_GRACE_S) -> None:
        self.graces.append(grace_s)
        shutdown_pool(pool, grace_s=grace_s)


def _no_child_is_left_behind(seconds: float = 10.0) -> bool:
    """Whether every worker this test forked is gone — killed and reaped, not futex-blocked."""
    deadline = time.monotonic() + seconds
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    return not multiprocessing.active_children()


def _without_the_minted_id(record: Mapping[str, Any]) -> dict[str, Any]:
    """One bench record minus the one field that is minted per run rather than measured."""
    return {**record, "bench": {**record["bench"], "bench_id": None}}


def _evidence(directory: Path) -> tuple[tuple[Any, ...], ...]:
    """Every job's retained evidence, minus the directory it happens to live in."""
    return tuple(
        (kept.case_id, kept.config_id, kept.rep, kept.prompt, kept.attempts)
        for kept in replay(directory)
    )


# ── the pool works the jobs ───────────────────────────────────────────────────────────────


def test_the_pooled_executor_states_the_width_it_was_configured_with():
    """The seam declares how wide it is, so a record or a log never has to ask what class it is."""
    assert PooledExecutor(workers=4).workers == 4


def test_every_job_is_worked_in_a_worker_process_and_none_is_worked_twice(tmp_path):
    ledger = tmp_path / "ledger"

    results = PooledExecutor(workers=3).run(_in_the_pool, _jobs(ledger, 6))

    assert results == (0, 1, 2, 3, 4, 5)
    worked = _worked_by(ledger)
    assert sorted(worked) == [0, 1, 2, 3, 4, 5]
    assert os.getpid() not in worked.values()


def test_jobs_that_finish_out_of_order_are_still_returned_in_job_order(tmp_path):
    """Order-independence is the collection's job: the pool may answer in any order it likes."""
    results = PooledExecutor(workers=5).run(_finishes_in_reverse, _jobs(tmp_path / "ledger", 5))

    assert results == (0, 1, 2, 3, 4)


def test_a_pool_one_worker_wide_never_forks_at_all(tmp_path):
    """A width of one is not a pool; paying for a fork to run jobs in order would be theatre."""
    ledger = tmp_path / "ledger"

    results = PooledExecutor(workers=1).run(_in_the_pool, _jobs(ledger, 3))

    assert results == (0, 1, 2)
    assert set(_worked_by(ledger).values()) == {os.getpid()}


# ── the failure modes ─────────────────────────────────────────────────────────────────────


def test_a_broken_pool_finishes_the_rest_in_process_and_re_runs_nothing_already_done(tmp_path):
    """The OOM'd-worker path: degrade to sequential *from where the pool died*, never restart."""
    ledger = tmp_path / "ledger"
    work = RecordingWork()
    executor = PooledExecutor(workers=2, worker=_dies_from_the_third_job_on)

    results = executor.run(work, _jobs(ledger, 6))

    assert results == (0, 1, 2, 3, 4, 5)
    assert sorted(_worked_by(ledger)) == [0, 1]  # the pool finished exactly two jobs
    assert work.ran == [2, 3, 4, 5]  # and the parent finished the other four, once each


def test_the_batch_stall_guard_keeps_what_the_wedged_batch_had_already_finished(tmp_path):
    """A silently-dead worker cannot block the wait forever, and cannot lose its batch-mate's
    finished work either: the guard fires, the batch is recovered, the rest runs in-process."""
    ledger = tmp_path / "ledger"
    work = RecordingWork()
    executor = PooledExecutor(
        workers=2, worker=_wedges_after_the_first_job, wait=_stall_on_a_short_fuse
    )

    results = executor.run(work, _jobs(ledger, 4))

    assert results == (0, 1, 2, 3)
    assert sorted(_worked_by(ledger)) == [0]
    assert work.ran == [1, 2, 3]


def test_a_wedged_pool_is_torn_down_without_ever_joining_the_worker_it_cannot_reach(tmp_path):
    """Bounded teardown: no grace for a pool already declared wedged, and no lingering child."""
    teardown = RecordingTeardown()
    work = RecordingWork()
    executor = PooledExecutor(
        workers=2, worker=_wedges_forever, wait=_stall_at_once, teardown=teardown
    )

    started = time.monotonic()
    results = executor.run(work, _jobs(tmp_path / "ledger", 4))

    assert time.monotonic() - started < 60  # a joining teardown would still be sleeping on it
    assert results == (0, 1, 2, 3)
    assert work.ran == [0, 1, 2, 3]
    assert teardown.graces == [0.0]
    assert _no_child_is_left_behind()


def test_a_pool_that_drained_every_job_still_leaves_through_the_bounded_teardown(tmp_path):
    """A fully drained pool is not a pool whose every worker is joinable — the sweep's 142-minute
    freeze — so the clean exit is bounded too, at its full grace."""
    teardown = RecordingTeardown()
    work = RecordingWork()
    executor = PooledExecutor(workers=2, worker=_in_the_pool, teardown=teardown)

    results = executor.run(work, _jobs(tmp_path / "ledger", 4))

    assert results == (0, 1, 2, 3)
    assert work.ran == []  # the pool finished everything — no fallback engaged
    assert teardown.graces == [POOL_TEARDOWN_GRACE_S]


# ── a whole bench, on the pool ────────────────────────────────────────────────────────────


def test_a_pooled_bench_writes_the_same_record_as_the_sequential_one_over_one_corpus(tmp_path):
    """The seam's whole promise: what a bench *measures* cannot depend on how it was worked."""
    corpus = _corpus("alpha", "beta", "gamma")

    sequential = _run(_runner(tmp_path / "one", StubAttempt()), corpus, reps=2)
    pooled = _run(
        _runner(tmp_path / "two", StubAttempt(), executor=PooledExecutor(workers=3)),
        corpus,
        reps=2,
    )

    assert _without_the_minted_id(pooled.record) == _without_the_minted_id(sequential.record)
    assert pooled.record["comparable_key"] == sequential.record["comparable_key"]


def test_a_pooled_bench_asks_its_cases_from_the_worker_processes(tmp_path):
    """The proof the pool is not decoration: nothing was asked in *this* process, and every case
    was still asked and still kept its evidence."""
    attempt = StubAttempt()
    runner = _runner(tmp_path, attempt, executor=PooledExecutor(workers=3))

    bench = _run(runner, _corpus("alpha", "beta", "gamma"))

    assert attempt.requests == []
    assert len(replay(bench.directory)) == 3


def test_a_pooled_bench_keeps_the_same_per_case_evidence_as_the_sequential_one(tmp_path):
    """Artifacts are written by whichever process finished the job, and they read back the same."""
    corpus = _corpus("alpha", "beta", "gamma")

    sequential = _run(_runner(tmp_path / "one", StubAttempt()), corpus)
    pooled = _run(
        _runner(tmp_path / "two", StubAttempt(), executor=PooledExecutor(workers=3)), corpus
    )

    assert _evidence(pooled.directory) == _evidence(sequential.directory)


def test_one_case_that_falls_over_in_a_worker_fails_itself_and_not_the_bench(tmp_path):
    attempt = StubAttempt(raises=frozenset({"beta"}))
    runner = _runner(tmp_path, attempt, executor=PooledExecutor(workers=3))

    bench = _run(runner, _corpus("alpha", "beta", "gamma"))

    outcomes = {run.result.case_id: run.result.passed for run in bench.runs}
    assert outcomes == {"alpha": True, "beta": False, "gamma": True}
    assert bench.record["bench"]["complete"] is True
    assert validate(bench.record) == []


def test_a_bench_whose_pool_goes_silently_dead_still_completes_and_records(tmp_path):
    """The bench outlives the pool: every case is finished in-process and the record is written."""
    executor = PooledExecutor(workers=2, worker=_wedges_forever, wait=_stall_at_once)
    runner = _runner(tmp_path, StubAttempt(), executor=executor)

    bench = _run(runner, _corpus("alpha", "beta", "gamma"))

    assert bench.record["bench"]["complete"] is True
    assert validate(bench.record) == []
    assert [kept.case_id for kept in replay(bench.directory)] == ["alpha", "beta", "gamma"]
