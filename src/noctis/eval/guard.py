"""The import-isolation guard — the one-way eval boundary, checkable by a machine.

:mod:`noctis.eval` states the contract (the eval layer imports the engine; the engine never
imports the eval layer). This module is the check that the source tree still honours it:
:func:`eval_layer_importers` walks a package's ``*.py`` files and returns every module **outside**
the eval layer that imports it, naming the module, the file, the line and the dotted name it
reached for.

**Source-level, not runtime.** The scan parses each file with :mod:`ast` and reads its import
statements; it never imports the code it is judging. That is what makes the verdict deterministic
and free of optional extras — a tree whose heavy seams are uninstalled scans exactly the same as
one with every extra present, and no import side effect can hide (or invent) a violation. It also
sets the guard's honest limit: an import spelled as a runtime string —
``importlib.import_module("noctis.eval")`` — is not an import statement and is not seen. The
boundary is a rule contributors keep in the open, not a sandbox.

**Every spelling of the same reach counts**, because the boundary is about the dependency, not the
syntax: ``import noctis.eval``, ``import noctis.eval.harness as h``, ``from noctis.eval import
sites``, ``from noctis.eval.harness import Harness``, ``from noctis import eval``, and the relative
forms an engine module could use from inside the package (``from ..eval import sites``), which are
resolved against the importing module's own package before comparison. A name that merely *starts*
with the layer's — ``noctis.evaluator`` — is not a violation; the match is on dotted components.

**Reported, never repaired, and never skipped.** A file that cannot be parsed or decoded raises out
of the scan rather than passing quietly: an unreadable module is a module the guard cannot clear,
and silence there is the one failure mode a boundary check must not have.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "EVAL_PACKAGE",
    "LAYER_NAME",
    "BoundaryViolation",
    "default_package_root",
    "eval_layer_importers",
    "report",
]

# This package: the layer the guard protects, spelled as the engine would import it.
EVAL_PACKAGE = "noctis.eval"

# Its directory inside the package root — the one component that identifies the layer in a tree
# whose top-level package is named something else (a fabricated tree in a test, say).
LAYER_NAME = EVAL_PACKAGE.rsplit(".", 1)[-1]

# Restated in every report, because a violation is most usefully answered by the rule it broke.
_CONTRACT = "the eval layer imports the engine; the engine never imports the eval layer"


@dataclass(frozen=True)
class BoundaryViolation:
    """One engine-side import of the eval layer: who imported what, and exactly where."""

    module: str
    path: str
    lineno: int
    imported: str

    def line(self) -> str:
        """The single line a failure prints — the module first, since that is what must change."""
        return f"{self.module} imports {self.imported} ({self.path}:{self.lineno})"


def default_package_root() -> Path:
    """The ``noctis`` package directory this module was loaded from."""
    return Path(__file__).resolve().parents[1]


def eval_layer_importers(package_root: Path | None = None) -> tuple[BoundaryViolation, ...]:
    """Every module under ``package_root`` outside the eval layer that imports it.

    ``package_root`` is the top-level package directory (``src/noctis`` for this repo, or a
    fabricated one in a test); the layer is its ``eval`` subdirectory, and dotted module names are
    derived from the root's own name so a miniature tree reads exactly like the real one.

    Sorted by module, then line: the same tree always yields the same tuple, so a failure message
    is stable across machines and runs.
    """
    root = default_package_root() if package_root is None else Path(package_root)
    layer_dir = root / LAYER_NAME
    layer = f"{root.name}.{LAYER_NAME}"
    violations: list[BoundaryViolation] = []
    for path in root.rglob("*.py"):
        if path.is_relative_to(layer_dir):
            continue
        for module, lineno, names in _imports(root, path):
            reach = next((name for name in names if _reaches(name, layer)), None)
            if reach is not None:
                violations.append(
                    BoundaryViolation(
                        module=module,
                        path=path.relative_to(root.parent).as_posix(),
                        lineno=lineno,
                        imported=reach,
                    )
                )
    return tuple(sorted(violations, key=lambda violation: (violation.module, violation.lineno)))


def report(violations: Sequence[BoundaryViolation]) -> str:
    """The verdict a failing guard prints: every offending module, then the rule they broke."""
    if not violations:
        return f"{EVAL_PACKAGE} is import-isolated: {_CONTRACT}"
    header = f"the one-way eval boundary is broken by {len(violations)} import(s):"
    return "\n".join([header, *(f"  {v.line()}" for v in violations), f"  {_CONTRACT}"])


def _reaches(name: str, layer: str) -> bool:
    """Whether a dotted name is the eval layer or something inside it. Components, not prefixes."""
    return name == layer or name.startswith(f"{layer}.")


def _imports(root: Path, path: Path) -> Iterator[tuple[str, int, tuple[str, ...]]]:
    """Each import statement in ``path``, as ``(importing module, line, dotted names it reaches)``.

    One entry per statement, carrying its candidate names shallowest first, so a match is reported
    once and named as the source names it. A ``from`` statement offers both halves — the module it
    names *and* each name pulled out of it — because ``from noctis import eval`` is a reach for the
    layer, not for the package that contains it.
    """
    module = _module_name(root, path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield module, node.lineno, (alias.name,)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(package, node) if node.level else node.module or ""
            if not base:
                continue
            names = [alias.name for alias in node.names if alias.name != "*"]
            yield module, node.lineno, (base, *(f"{base}.{name}" for name in names))


def _module_name(root: Path, path: Path) -> str:
    """The dotted name of the module at ``path``, rooted at the package directory's own name."""
    parts = list(path.relative_to(root).parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = path.stem
    return ".".join([root.name, *parts])


def _resolve(package: str, node: ast.ImportFrom) -> str:
    """A relative ``from`` statement as its absolute dotted name, or ``""`` if it escapes the root.

    Level 1 is the importing module's own package, each further level one step up — the resolution
    Python itself performs, done statically so ``from ..eval import sites`` is not a blind spot.
    """
    parts = package.split(".") if package else []
    if node.level - 1 > len(parts):
        return ""
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, *(node.module.split(".") if node.module else [])])
