"""Debug observability: the --debug QA run tree and its helpers (epic #36).

Kept deliberately thin — the run-id helper is stdlib-only so it stays liftable and importing
it can never drag in an optional extra. Later stories add the hour-segmented report writers here.
"""

from __future__ import annotations

from noctis.observability.debug.runid import new_run_id

__all__ = ["new_run_id"]
