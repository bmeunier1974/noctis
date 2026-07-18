"""The paper/live safety gate.

Real-money execution is unreachable unless **two independent gates are both open**:

1. ``mode: live`` in the config, and
2. the environment flag ``ALLOW_LIVE=true``.

``paper`` mode always resolves to paper regardless of ``ALLOW_LIVE`` (config wins toward
safety). ``mode: live`` *without* ``ALLOW_LIVE`` is a hard startup error — the process
refuses to start rather than silently downgrading, so the misconfiguration is visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from noctis.config.settings import Settings


class SafetyGateError(RuntimeError):
    """Raised when execution mode cannot be resolved safely."""


def resolve_execution_mode(settings: Settings) -> Literal["paper", "live"]:
    """Resolve the effective execution mode from both safety gates.

    Truth table:

    ======  ==========  ===========================
    mode    ALLOW_LIVE  result
    ======  ==========  ===========================
    paper   unset       paper
    paper   true        paper (config wins)
    live    unset       SafetyGateError (refuse start)
    live    true        live
    ======  ==========  ===========================
    """
    if settings.mode == "paper":
        return "paper"

    # mode == "live" from here.
    if settings.allow_live:
        return "live"

    raise SafetyGateError(
        "Refusing to start: config mode is 'live' but the ALLOW_LIVE environment gate is "
        "not set to true. Both gates must be open to arm real-money order paths. Set "
        "mode: paper for paper trading, or export ALLOW_LIVE=true to confirm live intent."
    )
