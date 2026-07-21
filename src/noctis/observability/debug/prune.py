"""QA-area retention (epic #36, story #42): prune the debug run tree on start.

The QA area (``workspace/qa/``) holds one folder per debug-recorded run, named by run id. Left
unbounded it grows without limit, so the only retention policy is prune-on-start: keep the newest
``qa.keep_last_runs`` run folders and delete the rest. Recency is *name order* — a run id leads
with a UTC compact timestamp, so a plain descending sort recovers chronological order with no
metadata read (see :mod:`noctis.observability.debug.runid`).

The pruner is deliberately narrow: it recognizes run folders *only* by the exact run-id name
shape (:data:`RUN_ID_RE`) and only when they are directories. Everything else in the area —
notes, dotfiles, near-miss names, a stray file that happens to match the shape — is off-limits
and never touched. This module owns the disk (it calls ``rmtree``); the shape it trusts is the
shared one from ``runid.py``, not a re-derived copy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from noctis.observability.debug.runid import RUN_ID_RE

__all__ = ["prune_qa_dir"]


def prune_qa_dir(qa_dir: Path | str, keep: int) -> list[str]:
    """Prune the QA area to its newest ``keep`` run folders; return the pruned run-id names.

    Parameters
    ----------
    qa_dir:
        The QA-area root (``workspace/qa/``). A missing directory is a no-op — a first-ever run
        has nothing to prune — so callers need not pre-check existence.
    keep:
        How many of the most-recent run folders to retain. ``keep <= 0`` keeps nothing (every
        run folder is pruned); a negative value is clamped to 0 rather than sliced, so a bad
        input can never accidentally delete the *oldest* few instead of all.

    Returns
    -------
    The names (run ids) of the folders removed, so a caller can log exactly what it evicted.
    Only run folders are ever considered: entries whose name does not match the run-id shape,
    and any run-shaped *file* rather than directory, are left untouched.
    """
    root = Path(qa_dir)
    if not root.is_dir():
        return []

    # Recency is name order, so a descending sort puts the most recent first. Only directories
    # whose name is the exact run-id shape are run folders — everything else is out of scope.
    run_folders = sorted(
        (child for child in root.iterdir() if child.is_dir() and RUN_ID_RE.match(child.name)),
        key=lambda child: child.name,
        reverse=True,
    )

    # Clamp negative keep to 0: ``run_folders[max(keep, 0):]`` prunes all when keep <= 0, and
    # never lets a negative index slice off only the oldest few.
    doomed = run_folders[max(keep, 0) :]
    for folder in doomed:
        shutil.rmtree(folder)
    return [folder.name for folder in doomed]
