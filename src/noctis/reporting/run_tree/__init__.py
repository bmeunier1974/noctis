"""The run tree — the ONE package that touches ``workspace/runs/<run_id>/`` (epic #284).

A run's tree is its ``run.json`` (the record), its ``run.lock`` (liveness) and everything that run
produced; ``runs/index.json`` is the derived roll-up beside them. No other package reads or writes
a byte of it — that boundary is what keeps ``run_record`` and ``schema`` pure.

Five modules are peeled off, layered ``record ← {address, index, lock, evidence} ← store`` and
pinned there by ``tests/test_run_tree_boundary.py``: :mod:`~noctis.reporting.run_tree.record` (the
tree's names, the one narrow read, the one atomic write), :mod:`~noctis.reporting.run_tree.lock`
(the whole liveness protocol — the one fatal failure), :mod:`~noctis.reporting.run_tree.address`
(one operator-typed string → one run dir), :mod:`~noctis.reporting.run_tree.index` (the derived
listing roll-up) and :mod:`~noctis.reporting.run_tree.evidence` (the six reads a record derives its
numbers from, and **every** heavy import in the package). The readers hold nothing but ``record``,
so resolving ``@label`` or deriving an index entry takes no lock and runs no collector. The rest —
the lifecycle verbs and :class:`~noctis.reporting.run_tree.store.RunStore` — is
:mod:`~noctis.reporting.run_tree.store`, the one module that holds the others.

A record is read in **two halves**, and each verb says which of them it needs:
:func:`~noctis.reporting.run_tree.store.read_artifacts` parses the prior record (every verb does
that) and :func:`~noctis.reporting.run_tree.evidence.derive_evidence` takes the six reads **once**,
at write time only — which is why opening a run no longer costs two passes over the journals, the
ledgers and the lake. Every public name keeps its spelling, so a caller says ``from
noctis.reporting.run_tree import open_run, resolve_run_dir`` and never has to know which module
answers.
"""

from __future__ import annotations

from noctis.reporting.run_tree.address import (
    RunAmbiguousError,
    RunNotFoundError,
    read_run_record,
    resolve_run_dir,
)
from noctis.reporting.run_tree.evidence import (
    STRATEGIES_SUBDIR,
    STRATEGY_TIER_SUBDIRS,
    Evidence,
    derive_evidence,
    read_benchmark,
    read_sessions,
    read_strategies,
    read_trials,
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
    FinishOutcome,
    PruneOutcome,
    RunCompletedError,
    RunNotPrunableError,
    RunStore,
    assert_resumable,
    finish_run,
    open_run,
    prune_run_state,
    read_artifacts,
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
    "Evidence",
    "FinishOutcome",
    "PruneOutcome",
    "RunAmbiguousError",
    "RunCompletedError",
    "RunLockedError",
    "RunNotFoundError",
    "RunNotPrunableError",
    "RunStore",
    "assert_resumable",
    "derive_evidence",
    "finish_run",
    "index_entry",
    "open_run",
    "prune_run_state",
    "read_artifacts",
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
