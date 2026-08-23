"""The run tree — the ONE package that touches ``workspace/runs/<run_id>/`` (epic #284).

A run's tree is its ``run.json`` (the record), its ``run.lock`` (liveness) and everything that run
produced; ``runs/index.json`` is the derived roll-up beside them. No other package reads or writes
a byte of it — that boundary is what keeps ``run_record`` and ``schema`` pure.

Four modules are peeled off already, layered ``record ← {address, index, lock} ← store`` and
pinned there by ``tests/test_run_tree_boundary.py``: :mod:`~noctis.reporting.run_tree.record` (the
tree's names, the one narrow read, the one atomic write), :mod:`~noctis.reporting.run_tree.lock`
(the whole liveness protocol — the one fatal failure), :mod:`~noctis.reporting.run_tree.address`
(one operator-typed string → one run dir) and :mod:`~noctis.reporting.run_tree.index` (the derived
listing roll-up). The two readers hold nothing but ``record``, so resolving ``@label`` or deriving
an index entry takes no lock and runs no collector; ``evidence`` follows in the story after. The
rest is still :mod:`~noctis.reporting.run_tree.store`. Every public name keeps its spelling, so a
caller says ``from noctis.reporting.run_tree import open_run, resolve_run_dir`` and never has to
know which module answers.
"""

from __future__ import annotations

from noctis.reporting.run_tree.address import (
    RunAmbiguousError,
    RunNotFoundError,
    read_run_record,
    resolve_run_dir,
)
from noctis.reporting.run_tree.index import (
    RUN_INDEX_KIND,
    RUN_INDEX_NAME,
    SHORT_RUN_S,
    index_entry,
    rebuild_index,
    update_index,
    visible_runs,
    write_index,
)
from noctis.reporting.run_tree.lock import (
    RUN_LOCK_NAME,
    STALE_HEARTBEAT_S,
    RunLockedError,
)
from noctis.reporting.run_tree.record import (
    RUN_RECORD_NAME,
    RUNS_SUBDIR,
    read_record,
    write,
)
from noctis.reporting.run_tree.store import (
    PRUNED_SUBDIRS,
    STRATEGIES_SUBDIR,
    STRATEGY_TIER_SUBDIRS,
    FinishOutcome,
    PruneOutcome,
    RunCompletedError,
    RunNotPrunableError,
    RunStore,
    assert_resumable,
    collect,
    finish_run,
    open_run,
    prune_run_state,
    read_benchmark,
    read_sessions,
    read_strategies,
    read_trials,
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
