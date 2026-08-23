"""The record module — the run tree's one narrow read and its one atomic write (story #286).

``record`` is the bottom of the package: the tree's names, ``read_record``, ``write`` and the two
small helpers everything else shares. It imports nothing from ``noctis.reporting.run_tree``, which
is what lets the eval layer read a run record without pulling the store, the lock and the
collectors along.

So every test here is a file on disk and a function call: write a record through the pure builder
(``write_run``), read it back, or break it and read the reason. No store is opened.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from noctis.reporting import schema
from noctis.reporting.run_tree.record import (
    RUN_RECORD_NAME,
    RUNS_SUBDIR,
    optional_str,
    read_record,
    write,
    write_json,
)

from ._run_tree_helpers import CREATED_UTC, LAST_ACTIVE_UTC, write_run

RUN_ID = "20260727T142233Z-a1b2c3"


def test_the_tree_names_are_spelled_once(tmp_path):
    """Both names are the directory layout itself: ``workspace/runs/<run_id>/run.json``."""
    assert RUNS_SUBDIR == "runs"
    assert RUN_RECORD_NAME == "run.json"


# ── write → read_record, the round trip every other module stands on ───────────────────────


def test_a_written_record_reads_back_whole_with_no_reason_to_report(tmp_path):
    run_dir = write_run(tmp_path / RUNS_SUBDIR, RUN_ID, label="nightly-momo")

    record, reason = read_record(run_dir)

    assert reason is None
    assert record is not None
    assert record["run"]["run_id"] == RUN_ID
    assert record["run"]["label"] == "nightly-momo"
    assert record["run"]["status"] == "stopped"
    assert record["run"]["created_utc"] == CREATED_UTC
    assert record["run"]["last_active_utc"] == LAST_ACTIVE_UTC
    assert schema.validate(record) == []  # the fixture writes what a store would


def test_a_completed_run_is_written_sealed_and_reads_back_terminal(tmp_path):
    run_dir = write_run(tmp_path / RUNS_SUBDIR, RUN_ID, status="completed", complete=True)

    record, _ = read_record(run_dir)

    assert record["run"]["status"] == "completed"
    assert record["run"]["completed_utc"] == LAST_ACTIVE_UTC
    assert record["run"]["complete"] is True
    assert schema.validate(record) == []


def test_writing_puts_the_record_at_run_json_inside_the_run_dir(tmp_path):
    run_dir = tmp_path / RUNS_SUBDIR / RUN_ID
    run_dir.mkdir(parents=True)

    write(run_dir, {"schema_version": 1, "kind": "noctis.run"})

    assert [p.name for p in run_dir.iterdir()] == [RUN_RECORD_NAME]
    assert json.loads((run_dir / RUN_RECORD_NAME).read_text())["kind"] == "noctis.run"


# ── a broken record is evidence, not a crash ───────────────────────────────────────────────


def test_a_run_dir_with_no_record_yet_reads_as_a_reason_not_an_exception(tmp_path):
    run_dir = tmp_path / RUNS_SUBDIR / RUN_ID
    run_dir.mkdir(parents=True)

    assert read_record(run_dir) == (None, f"no {RUN_RECORD_NAME} yet")


def test_an_unreadable_record_reads_as_a_reason_naming_what_went_wrong(tmp_path):
    run_dir = write_run(tmp_path / RUNS_SUBDIR, RUN_ID)
    (run_dir / RUN_RECORD_NAME).write_text('{"schema_version": 1, "run"')

    record, reason = read_record(run_dir)

    assert record is None
    assert reason == f"an unreadable {RUN_RECORD_NAME} (JSONDecodeError)"


def test_a_record_that_is_valid_json_but_not_an_object_is_unreadable_too(tmp_path):
    run_dir = write_run(tmp_path / RUNS_SUBDIR, RUN_ID)
    (run_dir / RUN_RECORD_NAME).write_text("[1, 2, 3]")

    assert read_record(run_dir) == (None, f"an unreadable {RUN_RECORD_NAME} (not an object)")


# ── atomic: the record on disk is whole, or it is the previous one ─────────────────────────


def test_a_kill_between_the_temp_write_and_the_replace_leaves_the_previous_record(
    tmp_path, monkeypatch
):
    """``os.replace`` is the atomic step, so a kill before it can only lose the *new* record."""
    run_dir = write_run(tmp_path / RUNS_SUBDIR, RUN_ID)
    good = (run_dir / RUN_RECORD_NAME).read_bytes()

    def killed(*_args, **_kwargs):
        raise KeyboardInterrupt("killed between the tmp write and the replace")

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        write(run_dir, {"schema_version": 1, "kind": "noctis.run", "half": "written"})

    assert (run_dir / RUN_RECORD_NAME).read_bytes() == good
    assert [p.name for p in run_dir.iterdir()] == [RUN_RECORD_NAME]  # no tmp litter


def test_a_first_write_that_never_replaces_leaves_no_record_at_all(tmp_path, monkeypatch):
    """The other half of "whole or not at all": a half-written file is never a record."""
    run_dir = tmp_path / RUNS_SUBDIR / RUN_ID
    run_dir.mkdir(parents=True)

    def killed(*_args, **_kwargs):
        raise KeyboardInterrupt("killed between the tmp write and the replace")

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        write(run_dir, {"schema_version": 1, "kind": "noctis.run"})

    assert list(run_dir.iterdir()) == []


# ── the two helpers that cross a module line ───────────────────────────────────────────────


def test_write_json_writes_any_document_atomically_and_leaves_no_temp_file(tmp_path):
    """The index is written through the same one atomic write the record is — one discipline."""
    target = tmp_path / "index.json"

    write_json(target, {"schema_version": 1, "runs": [{"run_id": RUN_ID}]})

    text = target.read_text()
    assert json.loads(text) == {"schema_version": 1, "runs": [{"run_id": RUN_ID}]}
    assert text.endswith("\n")
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]


def test_write_json_renders_a_value_json_cannot_carry_as_its_string(tmp_path):
    target = tmp_path / "index.json"

    write_json(target, {"where": Path("/tmp/runs")})

    assert json.loads(target.read_text()) == {"where": "/tmp/runs"}


def test_optional_str_keeps_a_missing_value_missing_and_stringifies_the_rest():
    """The record's convention: an absent value is an explicit ``null``, never a ``"None"``."""
    assert optional_str(None) is None
    assert optional_str("stopped") == "stopped"
    assert optional_str(3600) == "3600"
