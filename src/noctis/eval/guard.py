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

**One exemption, and its shape is checked too** (:data:`DEFERRED_EXEMPTIONS`, #211). ``noctis.cli``
may name ``noctis.eval.cli`` — the bodies behind the operator's ``bench`` verb group — and only from
**inside a function body**. Everything about that sentence is load-bearing:

* *Why an exemption exists at all.* The rule protects **production behaviour**: a benchmark measures
  the engine, so the engine must not be able to notice that a benchmark exists, or a bench-only
  ablation becomes reachable from a real research session. A CLI verb group is not production
  behaviour — it is an operator typing a word — and the alternative shapes are worse for the
  invariant, not better (a second console script that no ``noctis --help`` lists, or a plugin lookup
  that spells the import as a runtime string and so leaves the guard nothing at all to check).
* *Why it must be deferred.* A module-level import would make **every** ``noctis run`` load the eval
  layer, which is exactly the coupling the boundary refuses. Inside a function body, nothing is
  imported until an operator types ``bench``, so no production path can reach the layer even by
  accident. The scan therefore reports the *shape*: a top-level import in ``cli.py`` is still a
  violation, and so is a deferred one anywhere else, or a deferred one naming any other eval module.
* *Why the exemption names a module and not the package.* The permitted target is
  ``noctis.eval.cli`` **exactly** — ``from noctis.eval import cli`` reaches the layer first (see the
  spelling rule above) and is refused, so the crack cannot be widened by a rename or an alias.

Anything else is a violation, which is the whole point: the exemption is one line of data here, it
is exercised by the suite from both sides, and it is the only reason a reader of a report ever needs
to ask whether an import was allowed.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "DEFERRED_EXEMPTIONS",
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

#: The only engine-side reach the boundary permits, as ``importer → target``, both spelled relative
#: to the package root so a fabricated tree in a test reads exactly like the real one. The engine's
#: CLI module may name the eval layer's CLI module — and, per :func:`eval_layer_importers`, only
#: from inside a function body. See this module's docstring for why this one earns an exemption and
#: why nothing else does; adding a second entry here is a design decision, not a fix.
DEFERRED_EXEMPTIONS: Mapping[str, str] = MappingProxyType({"cli": f"{LAYER_NAME}.cli"})


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

    The one permitted reach (:data:`DEFERRED_EXEMPTIONS`) is excluded here and *only* in the shape
    it is declared in — the named importer, the named target, from inside a function body — so a
    top-level import of the same module by the same file is still reported.
    """
    root = default_package_root() if package_root is None else Path(package_root)
    layer_dir = root / LAYER_NAME
    layer = f"{root.name}.{LAYER_NAME}"
    violations: list[BoundaryViolation] = []
    for path in root.rglob("*.py"):
        if path.is_relative_to(layer_dir):
            continue
        for module, lineno, names, deferred in _imports(root, path):
            reach = next((name for name in names if _reaches(name, layer)), None)
            if reach is not None and not _exempt(root.name, module, reach, deferred=deferred):
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


def _exempt(root: str, module: str, reach: str, *, deferred: bool) -> bool:
    """Whether this exact reach is the one the boundary permits, in the one shape it permits it.

    Three conditions, all necessary: the importing module is the declared one, the name it reached
    is the declared target *exactly* (never something under it, and never the layer itself), and the
    import sits inside a function body so no production path loads the layer to satisfy it.
    """
    if not deferred:
        return False
    target = DEFERRED_EXEMPTIONS.get(module.removeprefix(f"{root}."))
    return target is not None and reach == f"{root}.{target}"


def _imports(root: Path, path: Path) -> Iterator[tuple[str, int, tuple[str, ...], bool]]:
    """Each import in ``path``, as ``(module, line, dotted names it reaches, deferred)``.

    One entry per statement, carrying its candidate names shallowest first, so a match is reported
    once and named as the source names it. A ``from`` statement offers both halves — the module it
    names *and* each name pulled out of it — because ``from noctis import eval`` is a reach for the
    layer, not for the package that contains it.

    ``deferred`` is true when the statement sits inside a function body: the difference between an
    import every user of the module pays for and one that happens only when a code path runs. It is
    a fact about the source, read statically like everything else here, and it is what
    :data:`DEFERRED_EXEMPTIONS` is checked against.
    """
    module = _module_name(root, path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    deferred = _deferred(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield module, node.lineno, (alias.name,), id(node) in deferred
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(package, node) if node.level else node.module or ""
            if not base:
                continue
            names = [alias.name for alias in node.names if alias.name != "*"]
            reached = (base, *(f"{base}.{name}" for name in names))
            yield module, node.lineno, reached, id(node) in deferred


def _deferred(tree: ast.AST) -> set[int]:
    """The identities of every import statement nested inside a function body in one parsed file."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for nested in ast.walk(node):
                if isinstance(nested, ast.Import | ast.ImportFrom):
                    inside.add(id(nested))
    return inside


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
