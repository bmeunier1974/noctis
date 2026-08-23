"""The run tree — the ONE package that touches ``workspace/runs/<run_id>/`` (epic #284).

A run's tree is its ``run.json`` (the record), its ``run.lock`` (liveness) and everything that run
produced; ``runs/index.json`` is the derived roll-up beside them. No other package reads or writes
a byte of it — that boundary is what keeps ``run_record`` and ``schema`` pure.

Today the whole implementation is one module, :mod:`~noctis.reporting.run_tree.store`; it is peeled
into ``record`` / ``address`` / ``index`` / ``lock`` / ``evidence`` over the stories that follow,
all behind this surface. Every public name keeps its spelling, so a caller says
``from noctis.reporting.run_tree import open_run, resolve_run_dir`` and never has to know which
module answers.
"""

from __future__ import annotations

from noctis.reporting.run_tree.store import (
    PRUNED_SUBDIRS,
    RUN_INDEX_KIND,
    RUN_INDEX_NAME,
    RUN_LOCK_NAME,
    RUN_RECORD_NAME,
    RUNS_SUBDIR,
    SHORT_RUN_S,
    STALE_HEARTBEAT_S,
    STRATEGIES_SUBDIR,
    STRATEGY_TIER_SUBDIRS,
    FinishOutcome,
    PruneOutcome,
    RunAmbiguousError,
    RunCompletedError,
    RunLockedError,
    RunNotFoundError,
    RunNotPrunableError,
    RunStore,
    assert_resumable,
    collect,
    finish_run,
    index_entry,
    open_run,
    prune_run_state,
    read_benchmark,
    read_record,
    read_run_record,
    read_sessions,
    read_strategies,
    read_trials,
    rebuild_index,
    resolve_run_dir,
    update_index,
    visible_runs,
    write,
    write_index,
)

__all__ = [
    "PRUNED_SUBDIRS",
    "RUNS_SUBDIR",
    "STRATEGIES_SUBDIR",
    "STRATEGY_TIER_SUBDIRS",
    "RUN_INDEX_KIND",
    "RUN_INDEX_NAME",
    "RUN_LOCK_NAME",
    "RUN_RECORD_NAME",
    "SHORT_RUN_S",
    "STALE_HEARTBEAT_S",
    "FinishOutcome",
    "PruneOutcome",
    "RunAmbiguousError",
    "RunCompletedError",
    "RunLockedError",
    "RunNotFoundError",
    "RunNotPrunableError",
    "RunStore",
    "assert_resumable",
    "collect",
    "finish_run",
    "index_entry",
    "open_run",
    "prune_run_state",
    "read_benchmark",
    "read_record",
    "read_run_record",
    "read_sessions",
    "read_strategies",
    "read_trials",
    "rebuild_index",
    "resolve_run_dir",
    "update_index",
    "visible_runs",
    "write",
    "write_index",
]
