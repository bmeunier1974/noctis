"""The ratchet mechanics, tested once over every policy that runs on them.

A ratchet is a record plus a rule. The *record* — how it is built, loaded, written, compared and
regenerated, what ``--check`` exits with and what ``--write`` refuses — lives once in
:mod:`noctis.observability.ratchet`, so it is tested once here, parametrized over each policy's
:class:`~noctis.observability.ratchet.RatchetSpec`. Each policy's own *rule* (the engine's tier
split, the prompt's declared-change rule) is tested in its own file, where the rule is written.

Every assertion is external behaviour — the status a comparison returns, the names and files it
prints, the exit code the entrypoint gives CI, what the record file holds, whether a file was
written. A scenario is a miniature repo in ``tmp_path``: build it, record it, edit exactly one
file, record it again — the shape ``tests/test_engine_ratchet.py`` and
``tests/test_eval_boundary.py`` use.

Adding a policy here is one entry in :data:`POLICIES`: its spec, a builder for a tree it can
fingerprint, one move it refuses to regenerate over, and one entry a test may leave unidentified.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from noctis.observability import engine_ratchet
from noctis.observability.engine_id import COMPONENT_PATHS
from noctis.observability.ratchet import (
    RatchetSpec,
    build_record,
    check,
    compare_records,
    load_record,
    main,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Policy:
    """One ratchet under test: its spec, a tree it can fingerprint, and two named moves."""

    spec: RatchetSpec
    build_tree: Callable[[Path], Path]
    # A move this policy refuses to record with ``--write`` (undeclared, in its own vocabulary).
    refused_move: str
    # An entry whose inputs a test may delete, so both sides read null.
    null_entry: str
    # What the policy says about an entry it never recorded and cannot identify now (D2).
    null_entry_status: str


def _build_asset_tree(spec: RatchetSpec, root: Path) -> Path:
    """Every path the spec's map names, with distinguishable content."""
    for paths in spec.asset_paths.values():
        for rel in paths:
            target = root.joinpath(*rel.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {rel}\nbody of {rel}\n")
    return root


def _build_engine_tree(root: Path) -> Path:
    return _build_asset_tree(engine_ratchet.SPEC, root)


def _write_unwatched_files(root: Path) -> None:
    """Docs, a test and the README — outside every allowlist, so they move no digest."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "architecture.md").write_text("# architecture\n")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_something.py").write_text("# a test\n")
    (root / "README.md").write_text("# Noctis\n")


ENGINE = Policy(
    spec=engine_ratchet.SPEC,
    build_tree=_build_engine_tree,
    refused_move=COMPONENT_PATHS["gates"][0],
    null_entry="schema",
    null_entry_status="warn",
)

POLICIES = [pytest.param(ENGINE, id="engine")]


@pytest.fixture(params=POLICIES)
def policy(request) -> Policy:
    return request.param


def _tree(policy: Policy, root: Path) -> Path:
    policy.build_tree(root)
    _write_unwatched_files(root)
    return root


def _entries(policy: Policy, record: dict) -> dict:
    return record[policy.spec.entries_key]


def _record_file(policy: Policy, root: Path) -> Path:
    return root / policy.spec.record_path


def _edit(root: Path, rel: str) -> None:
    path = root.joinpath(*rel.split("/"))
    path.write_text(path.read_text() + "# a change\n")


def _drop_inputs(policy: Policy, root: Path, name: str) -> None:
    for rel in policy.spec.asset_paths[name]:
        root.joinpath(*rel.split("/")).unlink()


# ── the record ────────────────────────────────────────────────────────────────────────────


def test_a_record_holds_its_kind_the_command_and_every_entry_with_its_files(policy, tmp_path):
    root = _tree(policy, tmp_path)

    record = build_record(policy.spec, root)

    assert record["kind"] == policy.spec.record_kind
    assert record["regenerate_with"] == policy.spec.regenerate_command
    entries = _entries(policy, record)
    assert set(entries) == set(policy.spec.asset_paths)
    for name, paths in policy.spec.asset_paths.items():
        assert isinstance(entries[name]["digest"], str), name
        # Per-file digests, so a drift report can name the file that moved, not its siblings.
        assert set(entries[name]["files"]) == set(paths), name


def test_an_unchanged_tree_checks_ok(policy, tmp_path):
    root = _tree(policy, tmp_path)
    write_record(policy.spec, root)

    result = check(policy.spec, root)

    assert result.status == "ok"
    assert result.ok
    assert not result.drifts
    assert "matches this tree" in result.report()


def test_editing_docs_a_test_or_the_readme_never_fires_the_check(policy, tmp_path):
    root = _tree(policy, tmp_path)
    committed = build_record(policy.spec, root)
    (root / "docs" / "architecture.md").write_text("# architecture, rewritten\n")
    (root / "tests" / "test_something.py").write_text("# a new test\n")
    (root / "README.md").write_text("# Noctis, rewritten\n")

    result = compare_records(policy.spec, build_record(policy.spec, root), committed)

    assert result.status == "ok"
    assert not result.drifts


def test_a_missing_record_file_fails_with_the_regenerate_message(policy, tmp_path):
    root = _tree(policy, tmp_path)

    result = check(policy.spec, root)

    assert load_record(policy.spec, root) is None
    assert result.status == "fail"
    assert not result.ok
    assert policy.spec.record_path in result.report()
    assert policy.spec.regenerate_command in result.report()


def test_a_malformed_record_file_fails_the_same_way(policy, tmp_path):
    root = _tree(policy, tmp_path)
    _record_file(policy, root).write_text("{ not json at all")

    result = check(policy.spec, root)

    assert load_record(policy.spec, root) is None
    assert result.status == "fail"
    assert policy.spec.regenerate_command in result.report()


def test_a_record_without_its_entries_key_reads_as_no_record_at_all(policy, tmp_path):
    root = _tree(policy, tmp_path)
    _record_file(policy, root).write_text(json.dumps({"kind": policy.spec.record_kind}))

    assert load_record(policy.spec, root) is None
    assert check(policy.spec, root).status == "fail"


# ── entries nobody can identify ───────────────────────────────────────────────────────────


def test_an_entry_null_on_both_sides_is_not_drift(policy, tmp_path):
    root = _tree(policy, tmp_path)
    _drop_inputs(policy, root, policy.null_entry)
    committed = build_record(policy.spec, root)

    result = compare_records(policy.spec, build_record(policy.spec, root), committed)

    assert _entries(policy, committed)[policy.null_entry]["digest"] is None
    assert result.status == "ok"
    assert not result.drifts


def test_an_entry_the_record_never_knew_is_drift_even_when_the_tree_cannot_identify_it(
    policy, tmp_path
):
    """A name appearing in the map is news, and silence on it is what a ratchet exists to end.

    The record predates the name (it was added to the map in this PR) and this checkout cannot
    identify it either — an optional input, a file not landed yet. Present on one side only is
    drift, so the entry is reported as ``null -> null`` with no files under it: the name is the
    news.
    """
    root = _tree(policy, tmp_path)
    _drop_inputs(policy, root, policy.null_entry)
    committed = build_record(policy.spec, root)
    del _entries(policy, committed)[policy.null_entry]

    result = compare_records(policy.spec, build_record(policy.spec, root), committed)

    assert result.status == policy.null_entry_status
    assert [drift.name for drift in result.drifts] == [policy.null_entry]
    assert result.drifts[0].files == ()
    report = result.report()
    assert f"{policy.null_entry} (" in report
    assert "null -> null" in report
    assert not [line for line in report.splitlines() if line.startswith("      ")]


# ── the entrypoint CI runs ────────────────────────────────────────────────────────────────


def test_the_check_is_the_default_action(policy, tmp_path):
    root = _tree(policy, tmp_path)

    assert main(policy.spec, ["--root", str(root)]) == 1  # no record committed yet
    main(policy.spec, ["--write", "--root", str(root)])

    assert main(policy.spec, ["--root", str(root)]) == 0


def test_the_regeneration_command_writes_a_record_the_check_then_accepts(policy, tmp_path, capsys):
    root = _tree(policy, tmp_path)

    assert main(policy.spec, ["--write", "--root", str(root)]) == 0

    written = json.loads(_record_file(policy, root).read_text())
    assert written["kind"] == policy.spec.record_kind
    assert main(policy.spec, ["--check", "--root", str(root)]) == 0
    assert _record_file(policy, root).read_text().endswith("\n")
    assert policy.spec.record_path in capsys.readouterr().out


def test_a_refused_regeneration_writes_nothing_and_leaves_the_check_failing(
    policy, tmp_path, capsys
):
    """``--write`` is the fix every failure recommends, and it rewrites every entry at once — so
    it cannot also be how the move the policy refuses gets recorded."""
    root = _tree(policy, tmp_path)
    main(policy.spec, ["--write", "--root", str(root)])
    before = _record_file(policy, root).read_bytes()
    _edit(root, policy.refused_move)
    capsys.readouterr()

    code = main(policy.spec, ["--write", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 1
    assert _record_file(policy, root).read_bytes() == before  # byte-identical: nothing written
    assert policy.spec.write_refusal in out
    assert policy.refused_move in out
    assert main(policy.spec, ["--check", "--root", str(root)]) == 1


def test_a_missing_record_still_regenerates_as_the_baseline(policy, tmp_path):
    """There is nothing to compare against, and this is how the baseline is created."""
    root = _tree(policy, tmp_path)
    assert load_record(policy.spec, root) is None

    assert main(policy.spec, ["--write", "--root", str(root)]) == 0

    assert check(policy.spec, root).status == "ok"


def test_regenerating_an_unchanged_tree_is_idempotent(policy, tmp_path):
    root = _tree(policy, tmp_path)
    main(policy.spec, ["--write", "--root", str(root)])
    before = _record_file(policy, root).read_bytes()

    assert main(policy.spec, ["--write", "--root", str(root)]) == 0

    assert _record_file(policy, root).read_bytes() == before


def test_the_pure_writer_stays_available_for_creating_a_baseline(policy, tmp_path):
    """``write_record`` states the tree as it is and asks no questions; the refusal is a decision
    in the ``--write`` path, so the comparison itself stays pure."""
    root = _tree(policy, tmp_path)
    write_record(policy.spec, root)
    _edit(root, policy.refused_move)

    assert check(policy.spec, root).status == "fail"
    assert write_record(policy.spec, root) == _record_file(policy, root)
    assert check(policy.spec, root).status == "ok"


# ── this checkout ─────────────────────────────────────────────────────────────────────────


def test_the_committed_record_in_this_checkout_matches_this_tree(policy):
    """The file in the repo is the baseline CI compares against — it must be in sync now."""
    record = load_record(policy.spec, REPO_ROOT)

    assert record is not None
    result = check(policy.spec, REPO_ROOT)
    assert result.ok, result.report()
