"""Engine identity (#127): the per-component fingerprint, the comparable key, `noctis engine`.

Every behavior asserted here is external — a digest value, *which* component moved under an
edit, the tuple `comparable_key` returns, what the CLI prints. The sensitivity block is the
contract of the whole module: a component-wise fingerprint exists precisely so "did the thing
that decides what passes move?" is answerable separately from "was a prompt reworded?", so each
row of that table gets its own explicit test.

The root is injectable, so the sensitivity tests build a miniature repo in ``tmp_path``, mutate
exactly one file, and read back the set of components that moved.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

import noctis
from noctis.cli import app
from noctis.observability.engine_id import (
    ARBITER_COMPONENTS,
    COMPONENT_PATHS,
    ENGINE_VERSION,
    comparable_key,
    compare,
    fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ID_SOURCE = REPO_ROOT / "src" / "noctis" / "observability" / "engine_id.py"
runner = CliRunner()


def _build_tree(root: Path) -> Path:
    """A miniature repo: every path the component map names, plus a ``docs/`` page.

    Content is per-path so no two components can share a digest by accident.
    """
    for rel in _listed_paths():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {rel}\nbody of {rel}\n")
    (root / "docs" / "adr").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "architecture.md").write_text("# architecture\n")
    (root / "docs" / "adr" / "0001-example.md").write_text("# ADR 1\n")
    return root


def _listed_paths() -> list[str]:
    return [rel for paths in COMPONENT_PATHS.values() for rel in paths]


def _config(tmp_path: Path, *, metric: str = "sharpe") -> str:
    """A paper config with nothing steering it — ``mandate_dir`` points at an empty tmp tree."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        f"state_dir: {tmp_path}/state/\n"
        f"mandate_dir: {tmp_path}/mandate\n"
        f"promotion:\n  metric: {metric}\n"
    )
    return str(cfg)


# ── the declared version ──────────────────────────────────────────────────────────────────


def test_engine_version_is_a_plain_integer_independent_of_the_package_version():
    """A plain incrementing integer versioning the behavioural contract — not the release."""
    assert type(ENGINE_VERSION) is int
    assert ENGINE_VERSION >= 1
    assert str(ENGINE_VERSION) != noctis.__version__


# ── the fingerprint ───────────────────────────────────────────────────────────────────────


def test_fingerprint_is_a_per_component_mapping_not_one_opaque_hash(tmp_path):
    fp = fingerprint(_build_tree(tmp_path))

    assert set(fp.digests()) == set(COMPONENT_PATHS)
    digests = list(fp.digests().values())
    assert all(digests), fp.digests()
    # Distinct inputs, distinct digests: one component moving cannot drag another with it.
    assert len(set(digests)) == len(digests)


def test_the_real_checkout_fingerprints_from_its_own_default_root():
    """No argument: the module finds this checkout, and every null component carries a note."""
    fp = fingerprint()

    assert fp.engine_version == ENGINE_VERSION
    assert fp.digest("gates") and fp.digest("backtest")
    for name in COMPONENT_PATHS:
        assert fp.digest(name) is not None or fp.note(name), name


def test_every_hashed_path_is_a_committed_file():
    """Only committed scaffold is hashed, so fingerprints stay portable between machines —
    an operator's gitignored mandate or custom profile can never reach the allowlist."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    listed = _listed_paths()
    assert "mandate/MANDATE.md" not in listed
    untracked = [rel for rel in listed if (REPO_ROOT / rel).exists() and rel not in tracked]
    assert not untracked


# ── the comparable key ────────────────────────────────────────────────────────────────────


def test_comparable_key_is_version_gates_backtest_and_election_metric(tmp_path):
    fp = fingerprint(_build_tree(tmp_path))

    assert comparable_key("sharpe", fp) == (
        ENGINE_VERSION,
        fp.digest("gates"),
        fp.digest("backtest"),
        "sharpe",
    )


def test_two_configurations_differing_only_in_election_metric_are_not_comparable(tmp_path):
    fp = fingerprint(_build_tree(tmp_path))

    assert comparable_key("sharpe", fp) != comparable_key("sortino", fp)


def test_a_docstring_edit_leaves_the_comparable_key_unchanged(tmp_path):
    """The fingerprint over-partitions on prose (deliberately); the key does not. A seed
    strategy's docstring moves ``seeds`` and nothing the key is built from."""
    root = _build_tree(tmp_path)
    before_fp = fingerprint(root)
    seed = root / "strategies" / "sma_crossover.py"
    seed.write_text(seed.read_text() + '\n"""A better sentence about the same behaviour."""\n')

    after_fp = fingerprint(root)
    assert compare(before_fp, after_fp) == {"seeds"}
    assert comparable_key("sharpe", after_fp) == comparable_key("sharpe", before_fp)


# ── compare + the arbiter split ───────────────────────────────────────────────────────────


def test_compare_returns_the_names_of_the_components_that_differ(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    (root / "MEMORY.seed.md").write_text("a new starting lesson\n")
    (root / COMPONENT_PATHS["gates"][0]).write_text("max_gap = 0.7\n")

    assert compare(before, fingerprint(root)) == {"gates", "memory_seed"}
    assert compare(before, before) == set()


def test_arbiter_components_is_gates_and_backtest():
    assert ARBITER_COMPONENTS == {"gates", "backtest"}
    assert set(ARBITER_COMPONENTS) <= set(COMPONENT_PATHS)


def test_arbiter_components_is_declared_exactly_once_in_the_source_tree():
    """One line drawn once: two copies of this set would eventually disagree, silently."""
    declarations = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in sorted((REPO_ROOT / "src").rglob("*.py"))
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if line.startswith("ARBITER_COMPONENTS") and "=" in line
    ]
    assert len(declarations) == 1, declarations
    assert declarations[0].startswith("src/noctis/observability/engine_id.py:")


# ── stability ─────────────────────────────────────────────────────────────────────────────


def test_recomputing_over_an_unchanged_tree_is_byte_identical(tmp_path):
    root = _build_tree(tmp_path)

    assert fingerprint(root).digests() == fingerprint(root).digests()
    assert fingerprint(REPO_ROOT).digests() == fingerprint(REPO_ROOT).digests()


def test_crlf_line_endings_do_not_move_a_digest(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    for rel in _listed_paths():
        path = root / rel
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert compare(before, fingerprint(root)) == set()


def test_a_pycache_directory_does_not_move_a_digest(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    cache = root / "src" / "noctis" / "champions" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "promotion.cpython-311.pyc").write_bytes(b"\x00stale bytecode")

    assert compare(before, fingerprint(root)) == set()


# ── sensitivity: the table IS the contract, one test per row ──────────────────────────────


def test_editing_a_shipped_mandate_profile_moves_profiles_and_nothing_else(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    (root / "mandate" / "profiles" / "aggressive.md").write_text("Take more risk than before.\n")

    assert compare(before, fingerprint(root)) == {"profiles"}


def test_editing_a_promotion_threshold_moves_gates(tmp_path):
    root = _build_tree(tmp_path)
    promotion = root / "src" / "noctis" / "champions" / "promotion.py"
    promotion.write_text("MAX_GAP = 0.5\n")
    before = fingerprint(root)
    promotion.write_text("MAX_GAP = 0.7\n")

    assert compare(before, fingerprint(root)) == {"gates"}


def test_the_schema_component_tracks_the_run_records_schema_version(tmp_path):
    """Bumping :data:`SCHEMA_VERSION` moves the ``schema`` digest and **only** it (story #143).

    That mapping is the whole reason ``schema`` is a fingerprint component: a run whose record
    changed shape mid-flight has to be identifiable as such, and a version bump that left every
    digest where it was would make the change invisible to a reader pooling two runs. It moves no
    arbiter component, so a schema change never invalidates a champion comparison.
    """
    root = _build_tree(tmp_path)
    schema_source = root / "src" / "noctis" / "reporting" / "schema.py"
    schema_source.write_text("SCHEMA_VERSION = 1\n")
    before = fingerprint(root)
    schema_source.write_text("SCHEMA_VERSION = 2\n")

    assert COMPONENT_PATHS["schema"] == ("src/noctis/reporting/schema.py",)
    assert compare(before, fingerprint(root)) == {"schema"}
    assert not {"schema"} & ARBITER_COMPONENTS  # a recording change is not an arbiter change


def test_editing_the_memory_seed_moves_memory_seed(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    (root / "MEMORY.seed.md").write_text("# Lessons\n- Start here.\n")

    assert compare(before, fingerprint(root)) == {"memory_seed"}


def test_editing_docs_moves_nothing(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    (root / "docs" / "architecture.md").write_text("# architecture, rewritten\n")
    (root / "docs" / "adr" / "0002-new.md").write_text("# ADR 2\n")
    (root / "README.md").write_text("# Noctis\n")

    assert compare(before, fingerprint(root)) == set()


def test_a_private_operator_mandate_and_custom_profile_move_no_digest(tmp_path):
    """``mandate/MANDATE.md``, ``mandate/<name>.md``, ``profiles/<custom>.md`` and personal
    references are gitignored — hashing them would make fingerprints machine-local."""
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    (root / "mandate" / "MANDATE.md").write_text("My own operator mandate.\n")
    (root / "mandate" / "my-night-shift.md").write_text("A custom personality.\n")
    (root / "mandate" / "profiles" / "mine.md").write_text("My own profile.\n")
    references = root / "mandate" / "references"
    references.mkdir(parents=True, exist_ok=True)
    (references / "my-watchlist.md").write_text("AAPL MSFT\n")

    assert compare(before, fingerprint(root)) == set()


# ── missing inputs ────────────────────────────────────────────────────────────────────────


def test_a_missing_input_yields_a_null_component_with_a_note(tmp_path):
    root = _build_tree(tmp_path)
    shutil.rmtree(root / "mandate" / "profiles")

    fp = fingerprint(root)

    assert fp.digest("profiles") is None
    assert "aggressive.md" in (fp.note("profiles") or "")
    assert fp.digest("gates")  # its neighbours are unaffected
    assert fp.note("gates") is None
    # Deterministic, and a null component is not "different" from itself.
    assert compare(fp, fingerprint(root)) == set()


def test_an_entirely_absent_tree_fingerprints_to_all_null_without_crashing(tmp_path):
    fp = fingerprint(tmp_path / "nothing-here")

    assert set(fp.digests().values()) == {None}
    assert all(fp.note(name) for name in COMPONENT_PATHS)
    assert comparable_key("sharpe", fp) == (ENGINE_VERSION, None, None, "sharpe")


# ── purity ────────────────────────────────────────────────────────────────────────────────


def test_the_module_imports_no_config_no_clock_no_network():
    """Pure by construction: source files in, digests out. Nothing else may be reachable."""
    allowed = {"__future__", "collections", "dataclasses", "hashlib", "pathlib", "typing"}
    tree = ast.parse(ENGINE_ID_SOURCE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= allowed, imported - allowed


def test_fingerprinting_pulls_no_optional_extra_and_reads_no_settings():
    code = (
        "import sys\n"
        "from noctis.observability.engine_id import comparable_key, fingerprint\n"
        "fp = fingerprint()\n"
        "comparable_key('sharpe', fp)\n"
        "assert 'noctis.config' not in sys.modules, 'engine identity read config'\n"
        "heavy = {"
        "'nautilus_trader', 'vectorbt', 'optuna', 'quantstats', "
        "'databento', 'exchange_calendars', 'anthropic', 'litellm'"
        "}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ── the CLI verb ──────────────────────────────────────────────────────────────────────────


def test_engine_command_prints_the_version_every_component_digest_and_the_key(tmp_path):
    result = runner.invoke(app, ["engine", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    fp = fingerprint()
    assert f"engine version:    {ENGINE_VERSION}" in result.output
    for name in COMPONENT_PATHS:
        assert name in result.output
        digest = fp.digest(name)
        assert (digest or "null") in result.output, name
    assert "election metric:   sharpe" in result.output
    assert f"{ENGINE_VERSION}|{fp.digest('gates')}|{fp.digest('backtest')}|sharpe" in result.output


def test_engine_command_reports_the_resolved_election_metric(tmp_path):
    """The key's fourth field is the metric the run would actually elect on."""
    result = runner.invoke(app, ["engine", "--config", _config(tmp_path, metric="sortino")])

    assert result.exit_code == 0, result.output
    assert "election metric:   sortino" in result.output
    assert result.output.rstrip().endswith("|sortino")


def test_engine_command_marks_the_arbiter_components(tmp_path):
    """Which digests bind comparability is visible without reading the source."""
    result = runner.invoke(app, ["engine", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    for line in result.output.splitlines():
        name = line.split()[0] if line.split() else ""
        if name in COMPONENT_PATHS:
            assert ("arbiter" in line) == (name in ARBITER_COMPONENTS), line
