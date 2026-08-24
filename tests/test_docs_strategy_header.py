"""The pages that say where a strategy's research record is read and written are held to the
module that owns it (story #317, epic #311).

The epic gave the header one home: `noctis.strategies.header` owns the value, the parse and both
stamps, and a header value is legal by construction. That is only half-shipped while the prose
still describes the old shape — a reader told the parse lives in `library.py` goes looking through
a 950-line write gate for the four lines that moved, and a reviewer asked to hold "the header is
legal by construction" has never been told the term anywhere.

Three checks, each red on a **code** change rather than on a human remembering:

* the glossary defines **Strategy header** — the record, the module that owns it, the value that
  cannot be illegal, the tolerant parse and the one exception the library wraps — so the term a
  review uses resolves to a file;
* the pages a strategy author lands on (``strategies/README.md``, ``docs/architecture.md``) name
  that module and its write side, by the names the functions actually have: rename one and every
  page that named it goes red;
* the library's own module docstring points at the module rather than at a parser it no longer
  holds.
"""

from __future__ import annotations

from pathlib import Path

from noctis.strategies import library
from noctis.strategies.header import (
    HeaderError,
    StrategyHeader,
    stamp_header,
    write_params,
)

ROOT = Path(__file__).resolve().parents[1]

# The name of the module every page must send a reader to.
MODULE = StrategyHeader.__module__  # noctis.strategies.header

# The row of the architecture module map that answers "where does a strategy live?".
LIBRARY_ROW = "📚 Strategy library"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _glossary_entry(title: str) -> str:
    """One ``## <title>`` section of the domain glossary."""
    text = _read("CONTEXT.md")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"CONTEXT.md: the '{title}' entry is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def test_the_glossary_defines_the_strategy_header_on_the_module_that_owns_it() -> None:
    """A reader of the term reaches the one module, the value, the parse and both stamps."""
    entry = _glossary_entry("Strategy header")

    assert MODULE in entry
    assert StrategyHeader.__name__ in entry
    assert f"`{stamp_header.__name__}`" in entry
    assert f"`{write_params.__name__}`" in entry


def test_the_glossary_entry_states_the_legality_rule_the_term_exists_for() -> None:
    """The header's whole claim: a value cannot be illegal, so the check is asked in one place."""
    entry = _glossary_entry("Strategy header")

    assert "legal by construction" in entry
    assert "VALID_STATUSES" in entry


def test_the_glossary_entry_names_the_one_exception_and_what_the_library_wraps_it_into() -> None:
    """The currency question a maintainer asks first: what does a header error reach me as?"""
    entry = _glossary_entry("Strategy header")

    assert HeaderError.__name__ in entry
    assert library.StrategyValidationError.__name__ in entry


def test_the_author_page_names_the_parser_and_the_stamp() -> None:
    """Someone writing a strategy file learns which code reads the header they are typing."""
    text = _read("strategies/README.md")

    assert MODULE in text
    assert f"{StrategyHeader.__name__}.parse" in text
    assert stamp_header.__name__ in text and write_params.__name__ in text


def test_the_architecture_library_row_names_the_header_module() -> None:
    """The module map's strategy row sends a reader to the file that stamps the record."""
    rows = [line for line in _read("docs/architecture.md").splitlines() if LIBRARY_ROW in line]

    assert rows, f"docs/architecture.md has no '{LIBRARY_ROW}' row"
    assert all("strategies/header.py" in row for row in rows)


def test_the_library_docstring_points_at_the_header_module() -> None:
    """The library re-exports the names; it must not read as though it still parses them."""
    assert MODULE in (library.__doc__ or "")
