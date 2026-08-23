"""The engine ratchet's *rule*: the tier split, and the version it is declared against.

The mechanics — building, loading and writing the record, the four-case check, what ``--write``
does with a verdict, the report — are shared by every ratchet and tested once in
``tests/test_ratchet.py``. What is left here is the policy this module exists to state: which tier
fails, which warns and passes, when ``ENGINE_VERSION`` has to move, and which drift ``--write``
refuses to record.

Every assertion is external behaviour — the status a comparison returns, the component and file
names it prints, the exit code the entrypoint gives CI. The comparison is pure over two records,
so a scenario is built by making a miniature repo in ``tmp_path``, recording it, editing exactly
one file, and recording it again — the same shape ``tests/test_engine_id.py`` uses.

The tier split is the contract: an edit to ``gates`` or ``backtest`` (the arbiter — what passes,
what a number means) **fails**; an edit to a searcher component **warns and passes**, because a
ratchet that blocks on a docstring edit gets disabled, and a disabled ratchet asserts nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from noctis.observability import engine_change, engine_id, engine_ratchet, ratchet
from noctis.observability.engine_id import COMPONENT_PATHS, ENGINE_VERSION, fingerprint
from noctis.observability.engine_ratchet import (
    RECORD_PATH,
    REGENERATE_COMMAND,
    build_record,
    check,
    compare_records,
    main,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET_SOURCE = REPO_ROOT / "src" / "noctis" / "observability" / "engine_ratchet.py"
SCRIPT = REPO_ROOT / "scripts" / "engine_fingerprint.py"

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


def _tagged(result, tag: str) -> list[str]:
    """The names this verdict filed under one tier, in the order the report prints them."""
    return [drift.name for drift in result.drifts if drift.tag == tag]


# ── the policy module holds the rule and borrows the mechanics ────────────────────────────


def test_the_policy_module_runs_on_the_shared_mechanics(tmp_path):
    """The verdict types are the shared ones, and every verdict carries this policy's spec."""
    assert engine_ratchet.RatchetResult is ratchet.RatchetResult
    assert engine_ratchet.WriteOutcome is ratchet.WriteOutcome

    root = _build_tree(tmp_path)
    write_record(root)
    result = check(root)

    assert isinstance(result, ratchet.RatchetResult)
    assert result.spec is engine_ratchet.SPEC
    assert result.status == "ok"


def test_a_record_declares_the_engine_version_of_the_tree_it_states(tmp_path):
    """The declared comparison key is this policy's own field in the shared record."""
    record = build_record(_build_tree(tmp_path))

    assert record["engine_version"] == ENGINE_VERSION
    assert record["components"]["gates"]["digest"] == engine_id.fingerprint(tmp_path).digest(
        "gates"
    )


# ── the arbiter tier: fail ────────────────────────────────────────────────────────────────


def test_a_gates_edit_without_a_version_bump_fails_naming_the_component_and_the_file(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert not result.ok
    assert _tagged(result, "arbiter") == ["gates"]
    assert result.drifts[0].files == (GATES_FILE,)
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
    assert _tagged(result, "arbiter") == ["backtest"]
    assert result.drifts[0].files == (BACKTEST_FILE,)
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
    assert _tagged(result, "searcher") == [component]
    assert result.drifts[0].files == (rel,)
    report = result.report()
    assert component in report and rel in report
    assert not _tagged(result, "arbiter")


def test_a_searcher_edit_beside_an_arbiter_edit_still_fails(tmp_path):
    """The strict tier wins the verdict; the advisory tier is still named, and named second."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)
    _edit(root, SEARCHER_EDITS["prompts"])

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert _tagged(result, "searcher") == ["prompts"]
    assert [drift.name for drift in result.drifts] == ["gates", "prompts"]
    assert SEARCHER_EDITS["prompts"] in result.report()


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
    assert _tagged(result, "searcher") == ["schema"]
    assert SCHEMA_FILE in result.report()


# ── staleness: the contributor is told to regenerate ──────────────────────────────────────


def test_a_record_that_no_longer_declares_this_version_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert "stale" in result.report().lower()
    assert REGENERATE_COMMAND in result.report()


# ── what --write refuses: an arbiter move must arrive declared ────────────────────────────


def test_arbiter_drift_with_the_versions_in_agreement_is_what_write_refuses(tmp_path):
    """The loophole (#145): ``--write`` is the fix the failure itself recommends, and it rewrites
    every component at once — so if it absorbed an arbiter digest too, the ratchet's promise would
    reduce to a contributor noticing the failure before typing the command it printed."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root), committed)

    assert result.refuse_write


def test_searcher_drift_is_never_what_write_refuses(tmp_path):
    """The common case must stay one command, or the ratchet becomes a chore and gets disabled."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, SEARCHER_EDITS["prompts"])

    result = compare_records(_record(root), committed)

    assert result.status == "warn"
    assert not result.refuse_write


def test_an_arbiter_move_whose_bump_is_declared_is_not_refused(tmp_path):
    """The bump is there; the record simply had not been regenerated with it yet."""
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root, engine_version=ENGINE_VERSION), committed)

    assert result.status == "fail"
    assert not result.refuse_write


# ── the exit code each tier gives CI ──────────────────────────────────────────────────────


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
    ratchet_tiers = {drift.name: drift.tag for drift in result.drifts}
    policy = {
        change.component: change.tier
        for change in engine_change.engine_change(frozen, fingerprint(root)).components
    }
    assert policy == ratchet_tiers
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


def test_the_write_refusal_is_documented_where_a_contributor_reads_the_rule():
    """A guard nobody has been told about reads as a broken tool, so the one place the ratchet is
    explained to a contributor has to say that ``--write`` will not record an undeclared move."""
    development = (REPO_ROOT / "docs" / "development.md").read_text()
    script = (REPO_ROOT / "scripts" / "engine_fingerprint.py").read_text()

    assert "refuses to regenerate" in development
    assert "declared" in development
    assert "refuses" in script


def test_the_tier_rule_is_documented_in_the_module_docstring():
    """The rule (fail on arbiter, warn on searcher, stale is what you are told to fix, and
    ``--write`` will not launder an undeclared arbiter move) is the contract; it is written where
    the next reader of the code will be."""
    docstring = RATCHET_SOURCE.read_text().split('"""')[1].lower()

    assert "arbiter" in docstring and "searcher" in docstring
    assert "regenerate" in docstring
    assert "--write" in docstring and "refus" in docstring


def test_the_script_is_a_shim_a_guard_and_one_main_import():
    """``scripts/engine_fingerprint.py`` runs *before* the package is importable, which is the
    only reason it exists: it holds the ``sys.path`` shim, the ``ImportError`` guard and the one
    import of ``main`` — no policy, no argument parsing, nothing a module could hold instead."""
    module = ast.parse(SCRIPT.read_text())
    docstring, *statements = module.body

    assert isinstance(docstring, ast.Expr) and isinstance(docstring.value, ast.Constant)
    assert [type(node).__name__ for node in statements] == [
        "Import",  # sys, for the shim
        "ImportFrom",  # pathlib.Path, for the shim
        "Assign",  # _SRC
        "If",  # the sys.path shim
        "Try",  # the ImportError guard
        "If",  # the __main__ dispatch
    ]

    guard = statements[4]
    assert isinstance(guard, ast.Try)
    assert [ast.unparse(node) for node in guard.body] == [
        "from noctis.observability.engine_ratchet import main"
    ]

    dispatch = statements[5]
    assert isinstance(dispatch, ast.If)
    assert ast.unparse(dispatch.test) == "__name__ == '__main__'"


def test_the_script_docstring_names_what_write_refuses():
    """A guard nobody has been told about reads as a broken tool, and the script is where an
    operator lands first — so its docstring keeps the refusal sentence."""
    docstring = ast.get_docstring(ast.parse(SCRIPT.read_text())) or ""

    assert "refuses" in docstring
    assert "--write" in docstring
    assert "arbiter" in docstring and "ENGINE_VERSION" in docstring
