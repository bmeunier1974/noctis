"""The pages that say how a read-only command resolves a run are held to `open_reading` (#299).

Epic #292 gave every reader one entry in the composition root. That is only half-shipped while the
prose still describes the old shape: a reader told that a bare `report` reads "not the run that just
finished" is left with no remedy, and one told to bind a run with a helper that no longer exists
lands nowhere. So the pages an operator or an agent reads first get a check that fires on a **code**
change rather than on a human remembering:

* the glossary defines **Reading a run** — the noun, the entry, what it does *not* do (lock, write,
  act) and what a pruned run costs — so the term a review uses resolves to a function;
* the four pages that explain the precedence chain and the verbs name that entry, by the name the
  function actually has: rename it and every page that named it goes red;
* every verb that takes the **shared** address argument is read off the Typer app itself, so a
  seventh addressed reader tomorrow reddens the CLI page that does not document its address;
* no page still names one of the four preludes the epic deleted.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from noctis import cli
from noctis.bootstrap import Reading, RunPrunedError, open_reading

ROOT = Path(__file__).resolve().parents[1]

# The pages a reader lands on when asking "what does this command read, and how was it resolved?":
# AGENTS is the operating contract, architecture explains the seam, configuration owns the
# precedence chain, cli is the verb-by-verb surface.
PAGES = ("AGENTS.md", "docs/architecture.md", "docs/configuration.md", "docs/cli.md")

# Every page an operator or an agent reads — nothing here may name a deleted prelude.
ALL_PAGES: tuple[str, ...] = (
    "AGENTS.md",
    "CONTEXT.md",
    "README.md",
    "strategies/README.md",
) + tuple(sorted(str(path.relative_to(ROOT)) for path in (ROOT / "docs").rglob("*.md")))

# The four preludes epic #292 replaced with one band plus one wrapper. They cannot be derived from
# the code — they are gone — so they are listed once, here, as the names a page may no longer use.
DELETED_PRELUDES = (
    "bind_addressed_run",
    "_bind_reported_run_or_exit",
    "_resolve_status_session",
    "_resolve_mode_or_exit",
)

# The sentence the epic retired: it stated the surprise and offered no remedy.
RETIRED_WARNING = "not the run that just finished"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _glossary_entry(title: str) -> str:
    """One ``## <title>`` section of the domain glossary."""
    text = _read("CONTEXT.md")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"CONTEXT.md: the '{title}' entry is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def _addressed_verbs() -> dict[str, str]:
    """Every CLI verb carrying the one shared address argument, read off the Typer app.

    The shared help string is what makes them one family (five verbs describing four address forms
    in four wordings is how a form grows a fifth meaning), so it is also what identifies them here.
    """
    verbs: dict[str, str] = {}
    for command in cli.app.registered_commands:
        callback = command.callback
        assert callback is not None
        name = command.name or callback.__name__.replace("_", "-")
        for parameter in inspect.signature(callback).parameters.values():
            if getattr(parameter.default, "help", None) == cli._ADDRESS_HELP:
                verbs[name] = parameter.name
    return verbs


def test_the_glossary_defines_reading_a_run_beside_the_other_two_run_terms() -> None:
    """The three run terms sit together: the work, what it writes, and the look that neither."""
    text = _read("CONTEXT.md")

    segment = text.find("## Run segment\n")
    tree = text.find("## Run tree\n")
    reading = text.find("## Reading a run\n")

    assert reading > tree > segment, "CONTEXT.md has no 'Reading a run' entry after 'Run tree'"
    following = text.find("\n## ", tree + 1) + 1
    assert following == reading, "'Reading a run' does not directly follow 'Run tree'"


def test_the_glossary_entry_names_the_one_entry_and_the_value_it_hands_back() -> None:
    """A reader of the term reaches the function, and learns it opened nothing to close."""
    entry = _glossary_entry("Reading a run")

    assert open_reading.__name__ in entry
    assert f"`{Reading.__name__}`" in entry
    assert "context manager" in entry


def test_the_glossary_entry_states_that_a_reading_neither_locks_nor_writes() -> None:
    """The whole claim the band makes about a run's tree, spelled where the term is defined."""
    entry = _glossary_entry("Reading a run")

    assert "no lock is taken" in entry
    assert "no record is written" in entry


def test_the_glossary_entry_says_what_an_address_means_and_what_a_pruned_run_costs() -> None:
    """The two rules an operator meets first: bare reads the reserved run, pruned is refused."""
    entry = _glossary_entry("Reading a run")

    assert "address means the reserved `legacy` run" in entry
    assert RunPrunedError.__name__ in entry


@pytest.mark.parametrize("page", PAGES)
def test_every_page_a_reader_lands_on_names_the_one_reading_entry(page: str) -> None:
    assert open_reading.__name__ in _read(page)


@pytest.mark.parametrize("page", ALL_PAGES)
def test_no_page_names_a_prelude_the_epic_deleted(page: str) -> None:
    text = _read(page)

    named = [prelude for prelude in DELETED_PRELUDES if prelude in text]

    assert not named, f"{page} still names {named}, which no longer exists"


@pytest.mark.parametrize("verb", sorted(_addressed_verbs()))
def test_the_cli_page_documents_an_address_on_every_verb_that_takes_one(verb: str) -> None:
    """A verb whose signature grew an address, and a page that never says so, is a hidden one."""
    lines = [line for line in _read("docs/cli.md").splitlines() if f"noctis {verb}" in line]

    assert lines, f"docs/cli.md never shows `noctis {verb}`"
    assert any("address" in line for line in lines), (
        f"docs/cli.md shows `noctis {verb}` but never with its address argument"
    )


@pytest.mark.parametrize("page", ALL_PAGES)
def test_no_page_states_the_bare_form_surprise_without_the_remedy(page: str) -> None:
    """Naming the surprise is half a sentence; the other half is the address that answers it.

    Read with its whitespace collapsed, because where a sentence happens to wrap is not a fact
    about what it says.
    """
    assert RETIRED_WARNING not in " ".join(_read(page).split())


def test_the_cli_page_offers_the_address_as_the_remedy() -> None:
    assert "pass the address" in _read("docs/cli.md")
