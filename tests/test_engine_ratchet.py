"""The CI ratchet (#128): arbiter drift cannot land without an ENGINE_VERSION bump.

Every assertion here is external behaviour — the status a comparison returns, the component and
file names it prints, the exit code the entrypoint gives CI, what the committed record file holds.
The comparison is pure over two records, so a scenario is built by making a miniature repo in
``tmp_path``, recording it, editing exactly one file, and recording it again — the same shape
``tests/test_engine_id.py`` uses.

The tier split is the contract: an edit to ``gates`` or ``backtest`` (the arbiter — what passes,
what a number means) **fails**; an edit to a searcher component **warns and passes**, because a
ratchet that blocks on a docstring edit gets disabled, and a disabled ratchet asserts nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noctis.observability import engine_change, engine_id, engine_ratchet
from noctis.observability.engine_id import COMPONENT_PATHS, ENGINE_VERSION, fingerprint
from noctis.observability.engine_ratchet import (
    RECORD_PATH,
    REGENERATE_COMMAND,
    build_record,
    check,
    compare_records,
    load_record,
    main,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET_SOURCE = REPO_ROOT / "src" / "noctis" / "observability" / "engine_ratchet.py"

GATES_FILE = "src/noctis/champions/promotion.py"
GATES_SIBLING = "src/noctis/backtest/splits.py"
BACKTEST_FILE = "src/noctis/backtest/pipeline.py"
SCHEMA_FILE = COMPONENT_PATHS["schema"][0]

SEARCHER_EDITS = {
    "prompts": "src/noctis/research/prompt.py",
    "profiles": "mandate/profiles/aggressive.md",
    "seeds": "strategies/sma_crossover.py",
    "memory_seed": "MEMORY.seed.md",
    "research": "src/noctis/research/tools.py",
}


def _build_tree(root: Path) -> Path:
    """A miniature repo: every path the component map names, plus docs and a test file."""
    for paths in COMPONENT_PATHS.values():
        for rel in paths:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {rel}\nbody of {rel}\n")
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "architecture.md").write_text("# architecture\n")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_engine_ratchet.py").write_text("# a test\n")
    (root / "README.md").write_text("# Noctis\n")
    return root


def _record(root: Path, *, engine_version: int | None = None) -> dict:
    """The record for ``root``, optionally stamped with another declared version.

    A record carries the version that was declared when it was written, so overriding it is how
    a test expresses "this file was committed before the bump" without patching a constant.
    """
    record = build_record(root)
    if engine_version is not None:
        record["engine_version"] = engine_version
    return record


def _edit(root: Path, rel: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + "# a behavioural change\n")


# ── the committed record ──────────────────────────────────────────────────────────────────


def test_a_record_holds_the_declared_version_and_every_component_digest(tmp_path):
    record = build_record(_build_tree(tmp_path))

    assert record["engine_version"] == ENGINE_VERSION
    assert set(record["components"]) == set(COMPONENT_PATHS)
    assert record["components"]["gates"]["digest"] == engine_id.fingerprint(tmp_path).digest(
        "gates"
    )
    # Per-file digests, so a drift report can name the file that moved and not its siblings.
    assert set(record["components"]["gates"]["files"]) == set(COMPONENT_PATHS["gates"])
    assert record["regenerate_with"] == REGENERATE_COMMAND


def test_the_committed_fingerprint_file_records_this_checkout():
    """The file in the repo is the baseline CI compares against — it must be in sync now."""
    record = load_record(REPO_ROOT)

    assert record is not None
    assert record["engine_version"] == ENGINE_VERSION
    fp = engine_id.fingerprint(REPO_ROOT)
    for name in engine_id.ARBITER_COMPONENTS:
        assert record["components"][name]["digest"] == fp.digest(name), name
    # Searcher drift is allowed to be pending (it only warns), an arbiter mismatch never is.
    assert check(REPO_ROOT).ok, check(REPO_ROOT).report()


def test_an_unchanged_tree_checks_ok(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)

    result = check(root)

    assert result.status == "ok"
    assert result.ok
    assert not result.arbiter_drift and not result.searcher_drift


# ── the arbiter tier: fail ────────────────────────────────────────────────────────────────


def test_a_gates_edit_without_a_version_bump_fails_naming_the_component_and_the_file(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert not result.ok
    assert [drift.component for drift in result.arbiter_drift] == ["gates"]
    assert result.arbiter_drift[0].files == (GATES_FILE,)
    report = result.report()
    assert "gates" in report and GATES_FILE in report
    assert GATES_SIBLING not in report  # only what moved, never its untouched siblings
    assert "ENGINE_VERSION" in report


def test_a_backtest_edit_without_a_version_bump_fails_naming_the_component_and_the_file(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, BACKTEST_FILE)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert [drift.component for drift in result.arbiter_drift] == ["backtest"]
    assert result.arbiter_drift[0].files == (BACKTEST_FILE,)
    assert BACKTEST_FILE in result.report()


def test_bumping_the_version_without_regenerating_the_record_still_fails(tmp_path):
    """A bump is half the deal: the reviewer must also see which surface moved, in the diff."""
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root, engine_version=ENGINE_VERSION), committed)

    assert result.status == "fail"
    assert REGENERATE_COMMAND in result.report()


def test_an_arbiter_edit_with_a_bump_and_a_regenerated_record_passes(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)
    assert compare_records(_record(root, engine_version=ENGINE_VERSION - 1), committed).status == (
        "fail"
    )

    # The PR bumps ENGINE_VERSION and regenerates the committed file in the same commit.
    regenerated = _record(root, engine_version=ENGINE_VERSION)

    assert compare_records(regenerated, regenerated).status == "ok"


# ── the searcher tier: warn and pass ──────────────────────────────────────────────────────


@pytest.mark.parametrize(("component", "rel"), sorted(SEARCHER_EDITS.items()))
def test_a_searcher_edit_without_a_bump_warns_and_passes(tmp_path, component, rel):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, rel)

    result = compare_records(_record(root), committed)

    assert result.status == "warn"
    assert result.ok  # visible, not blocking
    assert [drift.component for drift in result.searcher_drift] == [component]
    assert result.searcher_drift[0].files == (rel,)
    report = result.report()
    assert component in report and rel in report
    assert not result.arbiter_drift


def test_a_searcher_edit_beside_an_arbiter_edit_still_fails(tmp_path):
    """The strict tier wins the verdict; the advisory tier is still named."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)
    _edit(root, SEARCHER_EDITS["prompts"])

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert [drift.component for drift in result.searcher_drift] == ["prompts"]
    assert SEARCHER_EDITS["prompts"] in result.report()


# ── the allowlist ─────────────────────────────────────────────────────────────────────────


def test_editing_docs_a_test_or_the_readme_never_fires_the_check(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root)
    (root / "docs" / "architecture.md").write_text("# architecture, rewritten\n")
    (root / "tests" / "test_engine_ratchet.py").write_text("# a new test\n")
    (root / "README.md").write_text("# Noctis, rewritten\n")

    result = compare_records(_record(root), committed)

    assert result.status == "ok"
    assert not result.arbiter_drift and not result.searcher_drift


# ── null components ───────────────────────────────────────────────────────────────────────


def test_a_component_that_is_null_on_both_sides_is_not_drift(tmp_path):
    root = _build_tree(tmp_path)
    (root / SCHEMA_FILE).unlink()
    committed = _record(root)

    result = compare_records(_record(root), committed)

    assert committed["components"]["schema"]["digest"] is None
    assert result.status == "ok"


def test_a_null_component_gaining_a_digest_is_searcher_tier_drift(tmp_path):
    """When the run-record schema module lands (#143), ``schema`` goes null → digest: a warn."""
    root = _build_tree(tmp_path)
    (root / SCHEMA_FILE).unlink()
    committed = _record(root)
    schema = root / SCHEMA_FILE
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text("SCHEMA_VERSION = 1\n")

    result = compare_records(_record(root), committed)

    assert result.status == "warn"
    assert [drift.component for drift in result.searcher_drift] == ["schema"]
    assert SCHEMA_FILE in result.report()


# ── staleness: the contributor is told to regenerate ──────────────────────────────────────


def test_a_record_that_no_longer_declares_this_version_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert "stale" in result.report().lower()
    assert REGENERATE_COMMAND in result.report()


def test_a_missing_record_file_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)

    result = check(root)

    assert result.status == "fail"
    assert load_record(root) is None
    assert RECORD_PATH in result.report()
    assert REGENERATE_COMMAND in result.report()


def test_a_malformed_record_file_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)
    (root / RECORD_PATH).write_text("{ not json at all")

    result = check(root)

    assert result.status == "fail"
    assert REGENERATE_COMMAND in result.report()


# ── the entrypoint CI runs ────────────────────────────────────────────────────────────────


def test_the_regeneration_command_writes_a_record_the_check_then_accepts(tmp_path, capsys):
    root = _build_tree(tmp_path)

    assert main(["--write", "--root", str(root)]) == 0
    written = json.loads((root / RECORD_PATH).read_text())

    assert written["engine_version"] == ENGINE_VERSION
    assert main(["--check", "--root", str(root)]) == 0
    assert (root / RECORD_PATH).read_text().endswith("\n")
    assert "engine_fingerprint" in capsys.readouterr().out


def test_the_check_exits_non_zero_on_arbiter_drift_naming_the_component_and_file(tmp_path, capsys):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, GATES_FILE)

    code = main(["--check", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 1
    assert "gates" in out and GATES_FILE in out


def test_the_check_exits_zero_on_a_searcher_warning_naming_the_component_and_file(tmp_path, capsys):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, SEARCHER_EDITS["prompts"])

    code = main(["--check", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 0
    assert "prompts" in out and SEARCHER_EDITS["prompts"] in out
    assert "warn" in out.lower()


def test_the_check_is_the_default_action(tmp_path):
    root = _build_tree(tmp_path)

    assert main(["--root", str(root)]) == 1  # no record committed yet
    main(["--write", "--root", str(root)])
    assert main(["--root", str(root)]) == 0


# ── one arbiter line, drawn once ──────────────────────────────────────────────────────────


def test_the_ratchet_and_the_resume_policy_read_one_arbiter_components_constant(tmp_path):
    """Two copies of that set would eventually disagree, and the disagreement would be silent.

    The CI ratchet (a change may not land) and the resume policy (a run may not continue) are the
    two enforcers of the one line, so this binds them **both ways**: they classify through the same
    function object, that function reads the one constant the resume policy also quotes by name,
    and — the part a refactor cannot fake — they return the same tier for every component when the
    whole engine has moved underneath them.
    """
    assert engine_ratchet.tier_of is engine_id.tier_of
    assert engine_change.tier_of is engine_id.tier_of
    assert engine_change.ARBITER_COMPONENTS is engine_id.ARBITER_COMPONENTS

    root = _build_tree(tmp_path)
    frozen = {
        "engine": {"engine_version": ENGINE_VERSION, "fingerprint": fingerprint(root).digests()}
    }
    committed = _record(root)
    for paths in COMPONENT_PATHS.values():
        _edit(root, paths[0])

    result = compare_records(_record(root), committed)
    ratchet = {
        drift.component: drift.tier for drift in (*result.arbiter_drift, *result.searcher_drift)
    }
    policy = {
        change.component: change.tier
        for change in engine_change.engine_change(frozen, fingerprint(root)).components
    }
    assert policy == ratchet
    assert set(policy) == set(COMPONENT_PATHS)

    # No second literal anywhere a consumer (this ratchet, the resume policy) could read.
    sources = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "scripts").rglob("*"))
    literals = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in sources
        if path.suffix == ".py"
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if '"gates"' in line and '"backtest"' in line
    ]
    assert literals == ["src/noctis/observability/engine_id.py:54"], literals


# ── wiring and documentation ──────────────────────────────────────────────────────────────


def test_the_ratchet_runs_in_ci_and_in_pre_commit():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    hooks = (REPO_ROOT / ".pre-commit-config.yaml").read_text()

    assert "scripts/engine_fingerprint.py" in workflow
    assert "--check" in workflow
    assert "scripts/engine_fingerprint.py" in hooks


def test_the_regeneration_command_is_documented():
    development = (REPO_ROOT / "docs" / "development.md").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()

    assert REGENERATE_COMMAND in development
    assert "scripts/engine_fingerprint.py" in agents
    assert RECORD_PATH in development


def test_the_tier_rule_is_documented_in_the_module_docstring():
    """The rule (fail on arbiter, warn on searcher, stale is what you are told to fix) is the
    contract; it is written where the next reader of the code will be."""
    docstring = RATCHET_SOURCE.read_text().split('"""')[1].lower()

    assert "arbiter" in docstring and "searcher" in docstring
    assert "regenerate" in docstring
