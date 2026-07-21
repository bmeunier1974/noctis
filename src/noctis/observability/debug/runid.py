"""The run/session id: ``20260720T144233Z-a3f9c1`` — sortable, collision-free, greppable.

This is the system's first run id, so it earns its keep by being three things at once and
nothing more:

* **Sortable by name** — the leading UTC compact timestamp means a plain lexicographic sort of
  the QA tree recovers chronological order, no metadata read required.
* **Collision-free across concurrent runs** — the 6-hex random suffix (24 bits) separates two
  runs that mint inside the same one-second tick, which is the realistic concurrency here.
* **Greppable** — a fixed, dependency-free shape a human (or a later feature reusing this) can
  eyeball and pattern-match.

It stays **stdlib-only on purpose**: future features may import it, and importing it must never
pull an optional extra, so the helper could be lifted out of noctis wholesale. Keep it that way —
do not reach into config, engine, or any heavy package from this module.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

# ``Z`` is a literal here (UTC marker), not a strftime directive — the timestamp is always UTC.
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def new_run_id(now: datetime | None = None) -> str:
    """Mint a fresh run id of the form ``YYYYMMDDTHHMMSSZ-xxxxxx`` (UTC + 6 lowercase hex).

    Parameters
    ----------
    now:
        An optional timestamp, injected for deterministic tests. An aware datetime is
        normalized to UTC so the trailing ``Z`` stays honest; a naive one is taken as UTC
        as-is. Defaults to :func:`datetime.now` in UTC.
    """
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return f"{moment.strftime(_TIMESTAMP_FORMAT)}-{secrets.token_hex(3)}"
