"""The pages that say where the run tree's I/O lives are held to the package (story #290).

A split (epic #284) is only half-shipped while the prose still describes one module: a reader told
"the one module that touches the run tree" goes looking for a 2300-line file that no longer exists,
and lands nowhere. The point of the split is that a reader arrives on 200 lines about addressing
rather than on everything at once — which only happens if the pages name the module that answers.

Two checks, each red on a **code** change rather than on a human remembering:

* the glossary defines **Run tree** — the directory, the package, every module the package has on
  disk, and the layering the boundary test pins — so the term a review uses resolves to a file;
* the pages an operator or an agent reads first (``AGENTS.md``, ``docs/run-record.md``,
  ``docs/architecture.md``, ``docs/cli.md``, ``docs/development.md``) name the package, the two
  verbs a caller reaches it by, and the guard that keeps the layering true.

The module list is read off the package, not listed here: a sixth module added tomorrow reddens the
glossary entry that does not mention it. A page still naming a *deleted* module is already refused
elsewhere — ``tests/test_docs_phase_modules.py`` resolves every module path and dotted reference a
page spells out — so this file only asserts what the pages must now say.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "noctis" / "reporting" / "run_tree"

# The pages a reader lands on when asking "where is the run tree written?", and what each owes the
# question: architecture explains the boundary, run-record is the contract, AGENTS is the operating
# contract every agent reads first.
PAGES = ("AGENTS.md", "docs/run-record.md", "docs/architecture.md")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _glossary_entry(title: str) -> str:
    """One ``## <title>`` section of the domain glossary."""
    text = _read("CONTEXT.md")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"CONTEXT.md: the '{title}' entry is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def _modules() -> set[str]:
    """Every module of the package as it stands on disk — ``__init__`` is the surface, not one."""
    return {path.stem for path in PACKAGE_ROOT.glob("*.py")} - {"__init__"}


def test_the_glossary_defines_the_run_tree_next_to_the_run_segment() -> None:
    """The two run terms sit together: one is a process's stretch of work, one is what it writes."""
    text = _read("CONTEXT.md")

    segment = text.find("## Run segment\n")
    tree = text.find("## Run tree\n")

    assert segment >= 0 and tree > segment, "CONTEXT.md has no 'Run tree' entry after 'Run segment'"
    following = text.find("\n## ", segment + 1) + 1
    assert following == tree, "'Run tree' does not directly follow 'Run segment'"


def test_the_glossary_entry_names_the_directory_and_the_one_package_that_touches_it() -> None:
    """A reader of the term reaches both the tree on disk and the code that owns it."""
    entry = _glossary_entry("Run tree")

    assert "workspace/runs/<run_id>/" in entry
    assert "run.json" in entry and "run.lock" in entry
    assert "reporting/run_tree/" in entry


def test_the_glossary_entry_names_every_module_the_package_has() -> None:
    """The split is the entry's whole point, so a module missing from it is a wrong page."""
    entry = _glossary_entry("Run tree")

    missing = sorted(module for module in _modules() if f"`{module}`" not in entry)

    assert not missing, f"CONTEXT.md 'Run tree' never names {missing}"


def test_the_glossary_entry_states_the_layering_and_the_two_halves_of_a_read() -> None:
    """The facts a maintainer needs before editing the package: the direction, and what costs."""
    entry = _glossary_entry("Run tree")

    assert "record ← {address, index, lock, evidence} ← store" in entry
    assert "tests/test_run_tree_boundary.py" in entry
    assert "read_artifacts" in entry and "derive_evidence" in entry


@pytest.mark.parametrize("page", PAGES)
def test_every_page_that_explains_the_run_tree_names_the_package(page: str) -> None:
    assert "reporting/run_tree/" in _read(page)


def test_the_pages_name_the_resolver_and_the_prune_verb_on_the_package() -> None:
    """The two verbs a reader follows out of the prose: one address form, one deletion path."""
    assert "run_tree.resolve_run_dir" in _read("docs/run-record.md")
    assert "run_tree.resolve_run_dir" in _read("docs/cli.md")
    assert "run_tree.resolve_run_dir" in _read("AGENTS.md")
    assert "run_tree.prune_run_state" in _read("AGENTS.md")


@pytest.mark.parametrize("page", ("AGENTS.md", "docs/development.md"))
def test_the_pages_that_list_the_import_guards_name_the_run_tree_boundary(page: str) -> None:
    """The layering is a guard like the eval boundary, so it is listed where guards are listed."""
    assert "tests/test_run_tree_boundary.py" in _read(page)
