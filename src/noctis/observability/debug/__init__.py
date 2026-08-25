"""Debug observability: the --debug QA run tree and its pure core (epic #36).

Kept deliberately thin and stdlib-only where it can be — the funnel ledger + renderers are pure
functions over caller-stamped event lists (no disk, no clock, no I/O), mirroring how champion
promotion is a pure decision over scorecards. The recorder (story #43) owns the clock and the
disk and feeds this module. The run-id minter is *not* one of the package's exports (story
#343): it moved to :mod:`noctis.observability.runid` because the run record and the eval
harness mint with it too, and this package is one of its consumers, not its doorway.
"""

from __future__ import annotations

from noctis.observability.debug.funnel import (
    FunnelCounts,
    Ledger,
    StampedEvent,
    StrategyFate,
    build_ledger,
    phase_durations,
)
from noctis.observability.debug.prune import prune_qa_dir
from noctis.observability.debug.recorder import Recorder
from noctis.observability.debug.render import (
    LEGACY_NOTICE,
    render_counts_json,
    render_counts_markdown,
    render_errors_markdown,
    render_summary_markdown,
)

__all__ = [
    "LEGACY_NOTICE",
    "FunnelCounts",
    "Ledger",
    "Recorder",
    "StampedEvent",
    "StrategyFate",
    "build_ledger",
    "phase_durations",
    "prune_qa_dir",
    "render_counts_json",
    "render_counts_markdown",
    "render_errors_markdown",
    "render_summary_markdown",
]
