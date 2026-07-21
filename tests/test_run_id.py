"""The run-id helper (epic #36): a sortable, collision-free, greppable session id.

The behaviors under test are external — the exact string shape, chronological ordering by
name, cross-mint distinctness, and that the helper drags in no heavy optional extra (it must
work on the core install alone, so a future feature can reuse it or lift it out wholesale).
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

from noctis.observability.debug.runid import new_run_id

# ``20260720T144233Z-a3f9c1``: UTC compact timestamp, a ``Z`` literal, dash, 6 lowercase hex.
_SHAPE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")


def test_run_id_matches_the_documented_shape():
    assert _SHAPE.match(new_run_id())


def test_injected_timestamp_lands_in_the_id_verbatim():
    moment = datetime(2026, 7, 20, 14, 42, 33, tzinfo=UTC)
    assert new_run_id(moment).startswith("20260720T144233Z-")


def test_an_aware_timestamp_is_normalized_to_utc():
    """A non-UTC ``now`` is honest: the ``Z`` means UTC, so the id carries UTC wall-clock."""
    eastern = timezone(timedelta(hours=-4))  # UTC-4
    moment = datetime(2026, 7, 20, 10, 42, 33, tzinfo=eastern)  # == 14:42:33Z
    assert new_run_id(moment).startswith("20260720T144233Z-")


def test_run_ids_sort_chronologically_by_name():
    earlier = new_run_id(datetime(2026, 7, 20, 9, 0, 0, tzinfo=UTC))
    later = new_run_id(datetime(2026, 7, 20, 17, 30, 0, tzinfo=UTC))
    assert earlier < later
    # Plain string sort recovers chronological order — the property the QA tree relies on.
    assert sorted([later, earlier]) == [earlier, later]


def test_run_ids_do_not_collide_across_concurrent_mints():
    """Same instant, many runs: the random hex suffix keeps every id distinct."""
    moment = datetime(2026, 7, 20, 14, 42, 33, tzinfo=UTC)
    ids = [new_run_id(moment) for _ in range(100)]
    assert len(set(ids)) == len(ids)


def test_helper_pulls_no_heavy_optional_extras():
    """Dependency-free: importing + minting must not load an optional-extra package."""
    code = (
        "import sys\n"
        "from noctis.observability.debug.runid import new_run_id\n"
        "new_run_id()\n"
        "heavy = {"
        "'nautilus_trader', 'vectorbt', 'optuna', 'quantstats', "
        "'databento', 'exchange_calendars', 'anthropic', 'litellm'"
        "}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
