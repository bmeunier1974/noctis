"""Run addressing — one string an operator typed becomes one run dir (story #287, epic #284).

Four forms in one fixed order — a ``run.json`` path, ``@label``, the reserved word ``latest``, a
run id — and a bare address is *always* the id. Every rule here is answerable from the records on
disk alone, so every fixture is a record written through the pure builder (``write_run``): no lock
is taken, no engine fingerprint is computed and no collector runs to test ``@label``. The store's
own *use* of the resolver (an alias open locks and records under the real id) stays in
``tests/test_run_tree_store.py``.

Everything asserted is external: the directory returned, or the text of the refusal — which is
half the contract, since an address that resolved to nothing must always say how to find the runs
that exist. The last two tests are what an address resolves *to*: a record with no sidecars, and
the derived roll-up beside it, both under the gitignored workspace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from noctis.reporting.run_tree import (
    RUN_INDEX_NAME,
    RUN_RECORD_NAME,
    RunAmbiguousError,
    RunNotFoundError,
    index_entry,
    rebuild_index,
    resolve_run_dir,
)

from ._run_tree_helpers import START, stamp, write_run


def _written(runs: Path, ordinal: int = 0, **kwargs) -> Path:
    """One run written ``ordinal`` hours after the fixture epoch, with an hour of runtime.

    The ordinal is the whole ordering story: it gives the run a distinct id *and* a distinct
    ``last_active_utc``, which is what ``latest`` selects on and what makes a two-run fixture
    total rather than dependent on directory order.
    """
    created = START + timedelta(hours=ordinal)
    return write_run(
        runs,
        f"{created:%Y%m%dT%H%M%S}Z-run{ordinal:03d}",
        created_utc=stamp(3600 * ordinal),
        runtime_s=3600.0,
        **kwargs,
    )


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def test_a_run_is_addressed_by_its_id(tmp_path):
    runs = tmp_path / "runs"
    run_dir = _written(runs)

    assert resolve_run_dir(runs, run_dir.name) == run_dir


def test_an_unknown_address_is_a_clean_lookup_failure_naming_the_run_tree(tmp_path):
    runs = tmp_path / "runs"
    _written(runs)

    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(runs, "20260101T000000Z-nope00")

    assert "20260101T000000Z-nope00" in str(excinfo.value)
    assert str(runs) in str(excinfo.value)


# ── addressing: `latest`, a record path, `@label` (story #133) ─────────────────────────────


def test_latest_addresses_the_most_recently_active_resumable_run(tmp_path):
    """The common case needs no id lookup — and a ``completed`` run is never it, because a
    published result refuses resume anyway."""
    runs = tmp_path / "runs"
    _written(runs, 0)
    newest_resumable = _written(runs, 1)
    _written(runs, 2, status="completed")

    assert resolve_run_dir(runs, "latest") == newest_resumable


def test_latest_reads_the_records_own_stamps_not_the_filesystems_mtimes(tmp_path):
    """*Most recently active* is what the record says it is. An mtime lies after a copy, a
    migration or a `jq` rewrite, and a run addressed by mistake is a run polluted by mistake."""
    runs = tmp_path / "runs"
    older = _written(runs, 0)
    newest = _written(runs, 1)
    future = 2_000_000_000
    os.utime(older / RUN_RECORD_NAME, (future, future))
    os.utime(older, (future, future))

    assert resolve_run_dir(runs, "latest") == newest


def test_latest_with_no_resumable_run_says_so_and_how_to_list_the_runs(tmp_path):
    runs = tmp_path / "runs"
    _written(runs, status="completed")

    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(runs, "latest")

    message = str(excinfo.value)
    assert "completed" in message
    assert str(runs) in message
    assert "noctis runs" in message


def test_latest_in_an_empty_workspace_says_there_is_nothing_to_resume(tmp_path):
    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(tmp_path / "runs", "latest")

    assert "mints one" in str(excinfo.value)


def test_a_run_is_addressed_by_the_path_of_the_record_you_are_looking_at(tmp_path):
    """The file in front of you *is* an address — its own dir, and the dir holding it."""
    runs = tmp_path / "runs"
    run_dir = _written(runs)

    assert resolve_run_dir(runs, str(run_dir / RUN_RECORD_NAME)) == run_dir
    assert resolve_run_dir(runs, str(run_dir)) == run_dir


def test_a_record_path_outside_the_configured_run_tree_still_addresses_its_run(tmp_path):
    """A path is an explicit address, so it is honoured wherever it points — a record copied out
    of a workspace, or a second workspace's tree, is exactly the case the form exists for."""
    runs = tmp_path / "runs"
    run_dir = _written(runs)

    assert resolve_run_dir(tmp_path / "elsewhere", str(run_dir / RUN_RECORD_NAME)) == run_dir


def test_a_path_that_names_no_record_is_a_clean_lookup_failure(tmp_path):
    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(tmp_path / "runs", str(tmp_path / "nowhere" / RUN_RECORD_NAME))

    assert "nowhere" in str(excinfo.value)
    assert "noctis runs" in str(excinfo.value)


def test_a_run_is_addressed_by_its_label_behind_the_at_sigil(tmp_path):
    runs = tmp_path / "runs"
    _written(runs, 0, label="sector-specialist")
    momo = _written(runs, 1, label="nightly-momo")

    assert resolve_run_dir(runs, "@nightly-momo") == momo


def test_a_label_is_stored_in_the_record_and_derived_into_the_index(tmp_path):
    """The alias lives in the record — the source of truth — and reaches ``index.json`` only by
    derivation, so a rebuild from the records alone still carries it."""
    runs = tmp_path / "runs"
    run_dir = _written(runs, label="nightly-momo")

    assert _record(run_dir)["run"]["label"] == "nightly-momo"
    assert [entry["label"] for entry in rebuild_index(runs)["runs"]] == ["nightly-momo"]


def test_an_unknown_label_says_how_to_find_the_runs(tmp_path):
    runs = tmp_path / "runs"
    _written(runs, label="nightly-momo")

    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(runs, "@no-such-label")

    assert "no-such-label" in str(excinfo.value)
    assert "noctis runs" in str(excinfo.value)


def test_a_bare_label_is_not_an_address_and_the_refusal_names_the_sigil(tmp_path):
    """The bare form is *always* the id, so a label typed without its sigil must not silently
    resolve — but the refusal says exactly what to type instead."""
    runs = tmp_path / "runs"
    _written(runs, label="nightly-momo")

    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(runs, "nightly-momo")

    assert "@nightly-momo" in str(excinfo.value)


def test_a_label_may_be_reassigned_and_each_run_keeps_its_own_id_and_record(tmp_path):
    """A label is convenience; the id is the identity. Re-using one on a second run neither
    renames the first nor merges the two — each keeps its own id, record and history."""
    runs = tmp_path / "runs"
    first = _written(runs, 0, label="nightly-momo")
    second = _written(runs, 1, label="nightly-momo")

    assert first.name != second.name
    assert resolve_run_dir(runs, first.name) == first
    assert resolve_run_dir(runs, second.name) == second
    assert _record(first)["run"]["run_id"] == first.name
    assert _record(second)["run"]["run_id"] == second.name
    assert {entry["label"] for entry in rebuild_index(runs)["runs"]} == {"nightly-momo"}


def test_an_ambiguous_label_refuses_and_names_every_candidate_id(tmp_path):
    """Two runs answer one alias, so there is no honest single answer: refuse, and name both, so
    the operator addresses the one they meant by its id."""
    runs = tmp_path / "runs"
    first = _written(runs, 0, label="nightly-momo")
    second = _written(runs, 1, label="nightly-momo")

    with pytest.raises(RunAmbiguousError) as excinfo:
        resolve_run_dir(runs, "@nightly-momo")

    message = str(excinfo.value)
    assert first.name in message and second.name in message
    assert isinstance(excinfo.value, RunNotFoundError)  # every existing caller already handles it


def test_a_label_that_looks_like_a_run_id_and_an_id_typed_with_a_sigil_both_resolve(tmp_path):
    """The two collisions the forms allow, both decided by one rule: bare is the id, ``@`` is the
    label first. So a run labelled with another run's id is reachable, and so is that other run."""
    runs = tmp_path / "runs"
    impostor = _written(runs, 0)
    labelled = _written(runs, 1, label=impostor.name)
    plain = _written(runs, 2)

    assert resolve_run_dir(runs, impostor.name) == impostor
    assert resolve_run_dir(runs, f"@{impostor.name}") == labelled
    assert resolve_run_dir(runs, f"@{plain.name}") == plain  # no label: falls back to id


def test_the_literal_latest_wins_over_a_run_named_or_labelled_latest(tmp_path):
    """``--resume latest`` means the same thing in every workspace. A run *named* ``latest`` is
    addressed by its path, one *labelled* ``latest`` by ``@latest``."""
    runs = tmp_path / "runs"
    named = write_run(runs, "latest", created_utc=stamp(0), runtime_s=3600.0)
    labelled = _written(runs, 1, label="latest")
    newest = _written(runs, 2)

    assert resolve_run_dir(runs, "latest") == newest
    assert resolve_run_dir(runs, "@latest") == labelled
    assert resolve_run_dir(runs, str(named)) == named


# ── what an address resolves to: one file, standing alone ──────────────────────────────────


def test_the_record_has_no_sidecar_files_and_stands_alone(tmp_path):
    """One `fetch()` of one URL returns everything a run page needs: the run dir holds exactly
    one file, and that file alone reproduces the run's whole index entry."""
    runs = tmp_path / "runs"
    run_dir = _written(runs, label="nightly-momo")

    assert [p.name for p in run_dir.iterdir()] == [RUN_RECORD_NAME]
    isolated = tmp_path / "isolated" / run_dir.name
    isolated.mkdir(parents=True)
    shutil.copy(run_dir / RUN_RECORD_NAME, isolated / RUN_RECORD_NAME)
    assert index_entry(isolated) == index_entry(run_dir)


def test_the_index_lands_under_the_gitignored_workspace(tmp_path):
    checked = subprocess.run(
        ["git", "check-ignore", "-q", f"workspace/runs/{RUN_INDEX_NAME}"],
        cwd=Path(__file__).resolve().parents[1],
    )
    assert checked.returncode == 0
