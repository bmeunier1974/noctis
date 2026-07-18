"""The gated live execution adapter — a stub that proves the seam and stays unreachable.

Its existence shows the ``Broker`` seam supports a live adapter; the double gate proves it
cannot be reached in paper mode. Construction fails unless **both** gates are open
(``mode: live`` *and* ``ALLOW_LIVE=true``), and even then it refuses because no real-order
path is implemented. There is deliberately no code here that could move real money.
"""

from __future__ import annotations

from noctis.config.gate import resolve_execution_mode


class LiveBrokerUnavailableError(RuntimeError):
    """Raised whenever the live broker is requested but must not / cannot run."""


class LiveBroker:
    """Placeholder live adapter. Never yields a usable broker."""

    def __init__(self, settings):
        # This re-raises SafetyGateError if mode==live without ALLOW_LIVE.
        mode = resolve_execution_mode(settings)
        if mode != "live":
            raise LiveBrokerUnavailableError(
                "Live broker requires both gates open (mode: live and ALLOW_LIVE=true). "
                f"Resolved mode is '{mode}'."
            )
        # Both gates are open — but there is still no real-order path by design.
        raise LiveBrokerUnavailableError(
            "Live execution adapter is a stub: no real-money order path is implemented. "
            "The paper broker is the only functional execution path."
        )
