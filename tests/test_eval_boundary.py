"""The one-way eval boundary (#189): the engine never imports the eval layer.

Every assertion here is external behaviour — what the scan returns for a tree, the module names
and file positions it prints, whether the contract is stated where a contributor would break it.
The scan is pure over a directory of sources, so a scenario is a miniature package in ``tmp_path``:
build an engine-side module and an eval layer, change exactly one import, assert the verdict — the
shape ``tests/test_engine_ratchet.py`` and ``tests/test_prompt_ratchet.py`` use.

The real-tree test is the CI enforcement of platform invariant 3 (production never depends on
benchmark infrastructure): it fails hard, naming the offending modules, the moment an engine module
grows an import of the eval layer.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import noctis.eval
from noctis.eval.guard import (
    EVAL_PACKAGE,
    BoundaryViolation,
    default_package_root,
    eval_layer_importers,
    report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "noctis"
GUARD_SOURCE = SOURCE_ROOT / "eval" / "guard.py"


def _tree(root: Path, modules: dict[str, str]) -> Path:
    """A miniature ``noctis`` package: an eval layer, plus the modules a scenario names."""
    package = root / "noctis"
    (package / "eval").mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text('"""a miniature engine."""\n')
    (package / "eval" / "__init__.py").write_text('"""the eval layer."""\n')
    for rel, source in modules.items():
        target = package / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return package


def test_the_real_source_tree_has_no_engine_module_importing_the_eval_layer() -> None:
    violations = eval_layer_importers(SOURCE_ROOT)
    assert violations == (), report(violations)


def test_an_engine_module_that_imports_the_eval_layer_is_named_as_a_violation(
    tmp_path: Path,
) -> None:
    package = _tree(tmp_path, {"research/agent.py": "from noctis.eval import sites\n"})

    violations = eval_layer_importers(package)

    assert [violation.module for violation in violations] == ["noctis.research.agent"]


def test_the_violation_names_the_file_and_line_of_the_offending_import(tmp_path: Path) -> None:
    package = _tree(
        tmp_path, {"research/agent.py": "import json\n\nfrom noctis.eval import sites\n"}
    )

    (violation,) = eval_layer_importers(package)

    assert violation.path == "noctis/research/agent.py"
    assert violation.lineno == 3
    assert violation.imported == "noctis.eval"


def test_a_plain_import_of_an_eval_submodule_is_a_violation(tmp_path: Path) -> None:
    package = _tree(tmp_path, {"engine/runtime.py": "import noctis.eval.harness as harness\n"})

    violations = eval_layer_importers(package)

    assert [violation.module for violation in violations] == ["noctis.engine.runtime"]


def test_the_from_noctis_import_eval_spelling_is_a_violation(tmp_path: Path) -> None:
    package = _tree(tmp_path, {"bootstrap.py": "from noctis import eval\n"})

    violations = eval_layer_importers(package)

    assert [violation.imported for violation in violations] == ["noctis.eval"]


def test_a_relative_import_of_the_eval_layer_is_a_violation(tmp_path: Path) -> None:
    package = _tree(tmp_path, {"research/tools.py": "from ..eval import sites\n"})

    violations = eval_layer_importers(package)

    assert [violation.module for violation in violations] == ["noctis.research.tools"]


def test_a_module_inside_the_eval_layer_may_import_the_eval_layer(tmp_path: Path) -> None:
    package = _tree(tmp_path, {"eval/runner.py": "from noctis.eval import sites\n"})

    assert eval_layer_importers(package) == ()


def test_the_eval_layer_importing_the_engine_is_not_a_violation(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "eval/runner.py": "from noctis.research.episode import EpisodeRunner\n",
            "research/episode.py": "class EpisodeRunner:\n    pass\n",
        },
    )

    assert eval_layer_importers(package) == ()


def test_a_module_whose_name_merely_starts_with_eval_is_not_a_violation(tmp_path: Path) -> None:
    package = _tree(tmp_path, {"research/agent.py": "from noctis.evaluator import score\n"})

    assert eval_layer_importers(package) == ()


def test_the_scan_reports_violations_in_a_deterministic_order(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/tools.py": "from noctis.eval import sites\n",
            "bootstrap.py": "import noctis.eval\n",
            "engine/runtime.py": "from noctis.eval.harness import Harness\n",
        },
    )

    first = eval_layer_importers(package)
    second = eval_layer_importers(package)

    assert [violation.module for violation in first] == [
        "noctis.bootstrap",
        "noctis.engine.runtime",
        "noctis.research.tools",
    ]
    assert first == second


def test_the_report_names_every_offending_module(tmp_path: Path) -> None:
    package = _tree(
        tmp_path,
        {
            "research/tools.py": "from noctis.eval import sites\n",
            "bootstrap.py": "import noctis.eval\n",
        },
    )

    rendered = report(eval_layer_importers(package))

    assert "noctis.research.tools" in rendered
    assert "noctis.bootstrap" in rendered


def test_the_import_isolation_guard_depends_only_on_the_standard_library() -> None:
    tree = ast.parse(GUARD_SOURCE.read_text(encoding="utf-8"), filename=str(GUARD_SOURCE))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])

    assert roots
    assert roots <= sys.stdlib_module_names | {"noctis"}


def test_the_default_package_root_is_the_installed_noctis_package() -> None:
    assert default_package_root().name == EVAL_PACKAGE.split(".")[0]
    assert (default_package_root() / "eval" / "guard.py").is_file()


def test_the_eval_package_docstring_states_the_one_way_boundary() -> None:
    docstring = " ".join((noctis.eval.__doc__ or "").lower().split())

    assert "one-way" in docstring
    assert "the eval layer imports the engine" in docstring
    assert "the engine never imports the eval layer" in docstring


def test_a_violation_renders_as_one_line_naming_module_and_import() -> None:
    violation = BoundaryViolation(
        module="noctis.research.agent",
        path="noctis/research/agent.py",
        lineno=12,
        imported="noctis.eval.sites",
    )

    assert violation.line() == (
        "noctis.research.agent imports noctis.eval.sites (noctis/research/agent.py:12)"
    )
