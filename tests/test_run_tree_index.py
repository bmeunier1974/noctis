"""The derived ``runs/index.json`` roll-up (story #287, epic #284).

The index is **derived, never authoritative**, and that is the whole subject of this file: every
entry comes from one run's record alone, a rebuild from the records on disk reproduces the
incrementally-maintained file byte for byte, and a record nobody can read is *listed* as exactly
that rather than crashing the listing.

So every fixture is a record written through the pure builder (``write_run``) — or a hand-broken
``run.json`` where the test is about one — and the index is maintained by calling
:func:`~noctis.reporting.run_tree.update_index` the way the store does. No store is opened: the
roll-up never needed one. The CLI's use of the listing (``noctis runs``, its noise filter and its
``--all``) stays in ``tests/test_run_tree_store.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from noctis.reporting.run_tree import (
    RUN_INDEX_KIND,
    RUN_INDEX_NAME,
    RUN_RECORD_NAME,
    rebuild_index,
    update_index,
    write_index,
)

from ._run_tree_helpers import CREATED_UTC, stamp, write_run

RUN_ID = "20260727T142233Z-a1b2c3"
SECOND_RUN_ID = "20260727T152233Z-d4e5f6"
THIRD_RUN_ID = "20260727T162233Z-778899"


def _index(runs_dir: Path) -> dict:
    return json.loads((runs_dir / RUN_INDEX_NAME).read_text())


def _listed(runs: Path, run_id: str, **kwargs) -> Path:
    """One written run, folded into the index the way the store folds each write into it."""
    run_dir = write_run(runs, run_id, **kwargs)
    update_index(runs, run_id)
    return run_dir


def test_the_run_tree_carries_a_derived_index_of_every_run(tmp_path):
    runs = tmp_path / "runs"
    _listed(runs, RUN_ID, label="nightly-momo")

    index = _index(runs)

    assert index["schema_version"] == 1
    assert index["kind"] == RUN_INDEX_KIND
    (entry,) = index["runs"]
    assert entry["run_id"] == RUN_ID
    assert entry["label"] == "nightly-momo"
    assert entry["status"] == "stopped"
    assert entry["segments"] == 1
    assert entry["cumulative_runtime_s"] == 3600.0
    assert entry["created_utc"] == CREATED_UTC
    assert entry["readable"] is True


def test_the_index_is_rebuildable_from_the_records_alone_byte_for_byte(tmp_path):
    """The index is DERIVED, never authoritative. A rebuild that read only the records on disk
    must reproduce the incrementally-maintained file exactly — otherwise the roll-up is a second
    source of truth, free to drift from the records it summarizes."""
    runs = tmp_path / "runs"
    for ordinal, (run_id, label) in enumerate(
        ((RUN_ID, "alpha"), (SECOND_RUN_ID, "beta"), (THIRD_RUN_ID, "gamma"))
    ):
        _listed(runs, run_id, label=label, created_utc=stamp(3600 * ordinal))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    write_index(elsewhere, rebuild_index(runs))

    assert (elsewhere / RUN_INDEX_NAME).read_bytes() == (runs / RUN_INDEX_NAME).read_bytes()
    assert [e["label"] for e in _index(runs)["runs"]] == ["gamma", "beta", "alpha"]  # newest first


def test_every_index_entry_and_every_record_carries_the_same_comparable_key(tmp_path):
    """The key a leaderboard partitions on structurally, so nobody has to remember which runs
    are poolable — on the record and on its index entry, never on one alone."""
    runs = tmp_path / "runs"
    run_dir = _listed(runs, RUN_ID, comparable_key="1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sortino")

    record = json.loads((run_dir / RUN_RECORD_NAME).read_text())
    (entry,) = _index(runs)["runs"]

    assert record["engine"]["comparable_key"].endswith("|sortino")
    assert entry["comparable_key"] == record["engine"]["comparable_key"]
    assert entry["engine_version"] == record["engine"]["engine_version"]
    assert entry["mixed_engine"] is False


def test_a_run_with_no_record_or_an_unreadable_one_is_listed_as_such(tmp_path):
    """A broken record is evidence, not a crash: the entry says what is wrong and the rest of
    the listing is unaffected."""
    runs = tmp_path / "runs"
    write_run(runs, RUN_ID)
    (runs / "20260101T000000Z-empty0").mkdir()
    broken = runs / "20260102T000000Z-brokn0"
    broken.mkdir()
    (broken / RUN_RECORD_NAME).write_text('{"schema_version": 1, "run"')

    entries = {entry["run_id"]: entry for entry in rebuild_index(runs)["runs"]}

    assert entries[RUN_ID]["readable"] is True
    assert entries[RUN_ID]["note"] is None
    assert entries["20260101T000000Z-empty0"]["readable"] is False
    assert "no run.json" in entries["20260101T000000Z-empty0"]["note"]
    assert entries["20260102T000000Z-brokn0"]["readable"] is False
    assert "unreadable" in entries["20260102T000000Z-brokn0"]["note"]
    # Explicit nulls, never missing keys: every entry answers every question (schema convention).
    assert set(entries["20260101T000000Z-empty0"]) == set(entries[RUN_ID])
    assert entries["20260101T000000Z-empty0"]["comparable_key"] is None
    assert entries["20260102T000000Z-brokn0"]["status"] is None


def test_a_record_of_a_foreign_shape_is_listed_as_unreadable_too(tmp_path):
    """Valid JSON, foreign shape — hand-edited or another tool's file. Same degradation."""
    runs = tmp_path / "runs"
    run_dir = write_run(runs, RUN_ID)
    (run_dir / RUN_RECORD_NAME).write_text('{"run": 5, "segments": "nope"}')

    (entry,) = rebuild_index(runs)["runs"]

    assert entry["run_id"] == RUN_ID
    assert entry["readable"] is False
    assert "unreadable" in entry["note"]


def test_the_index_is_regenerable_from_scratch_at_any_time(tmp_path):
    runs = tmp_path / "runs"
    _listed(runs, RUN_ID, label="alpha")
    _listed(runs, SECOND_RUN_ID, label="beta", created_utc=stamp(3600))
    before = (runs / RUN_INDEX_NAME).read_bytes()

    (runs / RUN_INDEX_NAME).unlink()
    write_index(runs, rebuild_index(runs))

    assert (runs / RUN_INDEX_NAME).read_bytes() == before
