"""The run tree's layering, pinned: ``record ← {address, index, lock, evidence} ← store`` (#287).

The package is five modules over one narrow read, and the whole value of that split is that the
narrow modules stay narrow: resolving ``@label`` needs the record and nothing else, deriving an
index entry needs the record and nothing else, and deciding whether a lock may be taken needs not
even that. A dependency added by accident — an index that reaches for the store, an address that
reaches for the lock — would quietly put the collectors back behind every ten-line function, so
the layering is a **test** rather than a convention, exactly like ``tests/test_eval_boundary.py``.

The second clause is the same shape, pointed outwards (story #288): the six collectors need the
research package, the champion registry, the broker's ledgers, the data types, the settings model
and pandas, and **only** ``evidence.py`` may name one. That is what makes "a record write stays as
cheap as the rest of this package" a shape rather than a comment — the subprocess probe in
``tests/test_run_tree_store.py`` measures what an import actually costs, and this says where a
heavy import may ever be *written*, deferred or not.

The scan is pure over source text: it walks every import statement with ``ast`` — module level
**and** function level, since a deferred import is still a dependency — resolves the relative
forms, and returns the set of package modules (or of heavy packages) one module names. So a
scenario is a string, and the real files are scanned the same way the strings are.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = "noctis.reporting.run_tree"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "noctis" / "reporting" / "run_tree"

# What each module of the package may import **from the package**. The layering itself: ``record``
# is the bottom and imports nothing; ``lock`` is a file with a pid and needs no record at all;
# ``address``, ``index`` and ``evidence`` read records and nothing else; ``store`` is the only
# module allowed to hold the rest. ``__init__`` is the public surface — it re-exports every module,
# so it is scanned for nothing here.
ALLOWED_PACKAGE_IMPORTS = {
    "record": frozenset(),
    "lock": frozenset(),
    "address": frozenset({"record"}),
    "index": frozenset({"record"}),
    "evidence": frozenset({"record"}),
    "store": frozenset({"record", "lock", "address", "index", "evidence"}),
}

# The packages a record write must never reach for — the research package and the champion
# registry behind the run's own journals and board, the broker's ledgers behind its equity curve,
# the data types and pandas behind the lake, and the settings model that derives a state dir. Every
# one of them is a *deferred* import in ``evidence.py``, and nowhere else in the package at all.
HEAVY_PACKAGES = (
    "noctis.research",
    "noctis.champions",
    "noctis.broker",
    "noctis.data",
    "noctis.config.settings",
    "pandas",
)

# The one module that may name them — the module that *is* the six reads.
EVIDENCE = "evidence"


def package_imports(source: str, *, module: str) -> set[str]:
    """Every module of :data:`PACKAGE` that this source imports, by its bare name.

    Both spellings of a dependency count, and both levels: ``from noctis.reporting.run_tree.record
    import read_record``, ``from .record import read_record``, ``from . import record``,
    ``import noctis.reporting.run_tree.record`` — inside a function body just as much as at the
    top of the file. The package **itself** (``from noctis.reporting.run_tree import open_run``,
    which would import every module through ``__init__``) is reported as ``""``, so no module can
    reach its siblings by going around through the public surface.
    """
    tree = ast.parse(source, filename=f"{module}.py")
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported |= _named(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = _absolute(node)
            if target is None:
                continue
            if target != PACKAGE:
                imported |= _named(target)
                continue
            names = {alias.name for alias in node.names}
            siblings = names & _modules()
            imported |= siblings
            if names - siblings:  # a public name, i.e. the package's own __init__
                imported |= {""}
    return imported


def heavy_imports(source: str, *, module: str) -> set[str]:
    """Every package of :data:`HEAVY_PACKAGES` this source names in an import.

    Every spelling and every level: ``import pandas as pd``, ``from noctis.research.journal import
    ExperimentJournal``, ``from noctis.config import settings`` — inside a function body just as
    much as at the top of the file, because deferring an import hides its cost at import time, not
    the fact that this module reaches for it.
    """
    tree = ast.parse(source, filename=f"{module}.py")
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                named |= _heavy(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = _absolute(node)
            if target is None:
                continue
            named |= _heavy(target)
            for alias in node.names:  # ``from noctis.config import settings``
                named |= _heavy(f"{target}.{alias.name}")
    return named


def _heavy(dotted: str) -> set[str]:
    """The heavy packages a dotted name belongs to — the package itself, or anything under it."""
    return {
        package
        for package in HEAVY_PACKAGES
        if dotted == package or dotted.startswith(f"{package}.")
    }


def _absolute(node: ast.ImportFrom) -> str | None:
    """The dotted module an ``ImportFrom`` names, with a relative form resolved against PACKAGE."""
    if node.level == 0:
        return node.module
    if node.level > 1:  # ``from .. import x`` leaves the package; never a package import
        return None
    return f"{PACKAGE}.{node.module}" if node.module else PACKAGE


def _named(dotted: str) -> set[str]:
    """``noctis.reporting.run_tree.index`` → ``{"index"}``; the package itself → ``{""}``."""
    if dotted == PACKAGE:
        return {""}
    if dotted.startswith(f"{PACKAGE}."):
        return {dotted[len(PACKAGE) + 1 :].split(".")[0]}
    return set()


def _modules() -> set[str]:
    """Every module of the package as it stands on disk — ``__init__`` is the surface, not one."""
    return {path.stem for path in PACKAGE_ROOT.glob("*.py")} - {"__init__"}


def _module_source(module: str) -> str:
    return (PACKAGE_ROOT / f"{module}.py").read_text(encoding="utf-8")


def _imports_of(module: str) -> set[str]:
    return package_imports(_module_source(module), module=module)


def _heavy_imports_of(module: str) -> set[str]:
    return heavy_imports(_module_source(module), module=module)


def test_the_package_is_exactly_the_modules_the_layering_names() -> None:
    """A module added without a clause here would be unguarded, so the set itself is asserted."""
    assert _modules() == set(ALLOWED_PACKAGE_IMPORTS)


@pytest.mark.parametrize("module", sorted(ALLOWED_PACKAGE_IMPORTS))
def test_every_module_imports_only_what_the_layering_allows_it(module: str) -> None:
    allowed = ALLOWED_PACKAGE_IMPORTS[module]

    assert _imports_of(module) <= allowed, (
        f"{module}.py imports {sorted(_imports_of(module) - allowed)} from {PACKAGE}"
    )


def test_the_record_is_the_bottom_of_the_package() -> None:
    """The one narrow read: the eval layer imports it and gets ten lines of JSON handling."""
    assert _imports_of("record") == set()


def test_the_lock_needs_no_record_at_all() -> None:
    """A lock is a pid, a hashed host and a heartbeat — no addressing, no index, no record."""
    assert _imports_of("lock") == set()


def test_addressing_reads_records_and_nothing_else() -> None:
    """Resolving ``@label`` must never take a lock or run a collector."""
    assert _imports_of("address") == {"record"}


def test_the_derived_index_reads_records_and_nothing_else() -> None:
    """The roll-up is derived from the records on disk — its only input is ``read_record``."""
    assert _imports_of("index") == {"record"}


def test_the_evidence_reads_records_and_nothing_else() -> None:
    """The six collectors need the run tree's *names*, not its store, its lock or its addressing."""
    assert _imports_of("evidence") <= {"record"}


def test_the_store_is_the_one_module_that_holds_the_rest() -> None:
    assert _imports_of("store") == {"record", "lock", "address", "index", "evidence"}


# ── the heavy imports: evidence, and nowhere else (story #288) ──────────────────────────────


@pytest.mark.parametrize(
    "module", sorted((set(ALLOWED_PACKAGE_IMPORTS) | {"__init__"}) - {EVIDENCE})
)
def test_no_module_but_the_evidence_may_name_a_heavy_package(module: str) -> None:
    """A record write must stay as cheap as the rest of the package — at every nesting level."""
    named = _heavy_imports_of(module)

    assert named == set(), f"{module}.py imports {sorted(named)}; only {EVIDENCE}.py may"


def test_the_evidence_is_where_every_heavy_import_lives() -> None:
    """The other side of the clause: the list is not aspirational, it is where these imports are."""
    assert _heavy_imports_of(EVIDENCE) == set(HEAVY_PACKAGES)


# ── the scan itself: every spelling of a dependency, at either level ────────────────────────


def test_an_absolute_import_of_a_sibling_is_seen() -> None:
    source = "from noctis.reporting.run_tree.store import RunStore\n"

    assert package_imports(source, module="index") == {"store"}


def test_a_relative_from_import_is_resolved_against_the_package() -> None:
    source = "from .lock import acquire_lock\n"

    assert package_imports(source, module="address") == {"lock"}


def test_the_from_dot_import_module_spelling_is_seen() -> None:
    """The exact line a maintainer would add to reach a sibling — and the one that must fail."""
    source = "from . import store\n"

    assert package_imports(source, module="index") == {"store"}


def test_a_plain_import_of_a_sibling_module_is_seen() -> None:
    source = "import noctis.reporting.run_tree.evidence as evidence\n"

    assert package_imports(source, module="index") == {"evidence"}


def test_a_deferred_import_inside_a_function_body_counts_as_a_dependency() -> None:
    """Deferring an import hides the cost at import time, not the dependency."""
    source = "def entry() -> None:\n    from .store import read_artifacts\n"

    assert package_imports(source, module="index") == {"store"}


def test_reaching_a_sibling_through_the_public_surface_is_reported_as_the_package() -> None:
    """``from noctis.reporting.run_tree import open_run`` imports *every* module, via __init__."""
    source = "from noctis.reporting.run_tree import open_run\n"

    assert package_imports(source, module="address") == {""}


def test_an_import_of_another_package_is_not_a_package_import() -> None:
    source = "import json\nfrom noctis.reporting.run_record import build\nimport pandas as pd\n"

    assert package_imports(source, module="index") == set()


def test_a_parent_relative_import_leaves_the_package_and_is_not_one_of_its_modules() -> None:
    source = "from ..run_record import build\n"

    assert package_imports(source, module="index") == set()


def test_the_guard_refuses_a_module_that_imports_a_sibling_it_may_not() -> None:
    """The guard's own red: an index reaching for the store fails the clause allowing record."""
    reaching = "from . import store\n"

    imported = package_imports(reaching, module="index")

    assert not imported <= ALLOWED_PACKAGE_IMPORTS["index"]


# ── the heavy scan itself ───────────────────────────────────────────────────────────────────


def test_a_deferred_pandas_import_is_seen_wherever_it_is_written() -> None:
    """The exact line the clause exists to refuse — and it is refused inside a function body."""
    source = "def entry() -> None:\n    import pandas as pd\n"

    assert heavy_imports(source, module="index") == {"pandas"}


def test_a_submodule_of_a_heavy_package_is_the_heavy_package() -> None:
    source = "from noctis.research.journal import ExperimentJournal\n"

    assert heavy_imports(source, module="lock") == {"noctis.research"}


def test_importing_the_settings_model_as_a_name_is_seen() -> None:
    """``from noctis.config import settings`` reaches the settings model by another spelling."""
    source = "from noctis.config import settings\n"

    assert heavy_imports(source, module="store") == {"noctis.config.settings"}


def test_a_lighter_neighbour_of_a_heavy_package_is_not_one() -> None:
    """The clause names ``noctis.config.settings``, not the whole of ``noctis.config``."""
    source = "from noctis.config.gate import resolve_mode\nfrom noctis.reporting import schema\n"

    assert heavy_imports(source, module="store") == set()


def test_one_line_can_name_more_than_one_heavy_package() -> None:
    source = "import pandas\n\ndef entry():\n    from noctis.broker.persistence import Ledger\n"

    assert heavy_imports(source, module="record") == {"pandas", "noctis.broker"}
