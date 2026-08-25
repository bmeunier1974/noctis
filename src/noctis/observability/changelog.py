"""The changelog reader both ratchets declare through — one parser, one grammar, one rule.

A ratchet's record states what the tree *is*; a changelog entry is the human sentence a moved
digest reads back to. Both of Noctis' ratchets declare that way, over their own page and on their
own clock, so *how an entry is read* is written here once — exactly as
:mod:`noctis.observability.ratchet` writes the mechanics once, and for the same reason: a parser
fix cannot then land in one ratchet and be missed in the other.

**What an entry is.** The newest entry is the **first** ``## `` line outside a code fence.
Newest-first is how these pages are written and how a human reads them, so "the newest entry"
needs no date parsed and no ordering trusted: it is the first one. Its ``digest`` covers the whole
entry text, so *amending* the newest entry — the natural way a second change in one PR gets
declared — reads as a new declaration. Fenced blocks are skipped both when finding the entry and
when finding where it ends: a page documents its own heading format in a fence, and a template is
not a declaration.

**The clause grammar**, which is what lets one reader serve two policies. A heading is a list of
clauses separated by `` — ``, each one ``key: value`` with the key lower-cased; a part carrying no
colon — the date every heading opens with — is not a clause. :meth:`ChangelogEntry.names` reads
one clause back as its comma-separated list, ``()`` when the heading has no such clause. So
``## 2026-08-01 — sites: author, ideation`` and
``## 2026-08-25 — components: backtest — behaviour: unchanged`` are one shape of sentence read two
ways, and **which** clauses count is the policy's word, never this module's: nothing here knows
what a site or a component is, or what naming one permits.

**The two halves a policy stores and reads.** :func:`header` is the ``changelog`` block a record
carries — ``{path, heading, digest, declares}`` — and :func:`declared_since` is the reading over a
computed and a committed record that makes "arrived *after* the record was written" checkable at
all: the newest entry's names count only when its digest differs from the one the committed record
was regenerated against. A matching digest means the page gained nothing since, so that entry is
yesterday's standing permission rather than today's declaration; a committed record with no block
at all reads as digest null, so any entry is new against it. :func:`footer` is the line a report
prints beneath its drifts — ``newest <path> entry: <heading|none>`` — because "I wrote one and it
still fails" is the question these tools get asked, and a fenced or mis-headed entry is invisible
without it.

Everything here is pure over text and mappings but :func:`read_entry`, the one read of the page
itself, where a missing page reads as no entry rather than an error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_ENTRY_PREFIX = "## "
_FENCE = "```"

# Clauses are separated by the spaced em dash these pages are written with, so a dash *inside* a
# value is not a clause boundary.
_CLAUSE_SEPARATOR = " — "
_CLAUSE_ASSIGNMENT = ":"

_DIGEST_CHARS = 16


@dataclass(frozen=True)
class ChangelogEntry:
    """One dated changelog entry: its heading, that heading's clauses, and its identity.

    ``digest`` covers the whole entry text, so *amending* the newest entry — the natural way a
    second change in one PR gets declared — reads as a new declaration.
    """

    heading: str
    clauses: Mapping[str, str]
    digest: str

    def names(self, key: str) -> tuple[str, ...]:
        """The comma-separated names on one clause — ``()`` when the heading carries no such one.

        The one reading a policy does of a heading, and the reason a declaration has to live
        there: prose is not machine-readable, and a paragraph mentioning a name is not a
        declaration of it.
        """
        listed = self.clauses.get(key.lower(), "")
        return tuple(name for name in (part.strip() for part in listed.split(",")) if name)


def newest_entry(text: str) -> ChangelogEntry | None:
    """The page's newest entry — the first ``## `` section — or ``None`` when there is none."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings = _entry_headings(lines)
    if not headings:
        return None
    start = headings[0]
    end = next((index for index in headings if index > start), len(lines))
    heading = lines[start][len(_ENTRY_PREFIX) :].strip()
    body = "\n".join(lines[start:end]).strip()
    return ChangelogEntry(
        heading=heading,
        clauses=_clauses(heading),
        digest=hashlib.sha256(body.encode("utf-8")).hexdigest()[:_DIGEST_CHARS],
    )


def read_entry(root: Path, path: str) -> ChangelogEntry | None:
    """The newest entry of the page at ``root/path``, or ``None`` when the page has none.

    A page that does not exist is a tree with nothing declared, not an error: the ratchets run on
    miniature trees and on fresh checkouts, and a missing page there is one more undeclared state.
    """
    page = root.joinpath(*path.split("/"))
    if not page.is_file():
        return None
    return newest_entry(page.read_text(encoding="utf-8"))


def header(path: str, entry: ChangelogEntry | None, declares: Sequence[str] = ()) -> dict[str, Any]:
    """The ``changelog`` block a record is written against: the page head, and what it declares.

    Storing the entry's *digest* is what makes "arrived after the record" checkable at all, and
    storing ``declares`` is what makes the record readable by a human without re-running the
    parser. Which names those are is the policy's word — it hands them in.
    """
    return {
        "changelog": {
            "path": path,
            "heading": None if entry is None else entry.heading,
            "digest": None if entry is None else entry.digest,
            "declares": list(declares),
        }
    }


def declared_since(computed: Mapping[str, Any], committed: Mapping[str, Any]) -> frozenset[str]:
    """The names the page declares *since* the committed record was written.

    Both halves of every declared-change rule live here: the newest entry has to name the thing
    that moved, **and** it has to be a different entry than the one the record was regenerated
    against. An entry that was already the head is a standing permission, not a declaration.
    """
    head = _field(computed, "digest")
    if head is None or head == _field(committed, "digest"):
        return frozenset()
    declares = _block(computed).get("declares")
    if not isinstance(declares, list):
        return frozenset()
    return frozenset(name for name in declares if isinstance(name, str))


def footer(path: str, record: Mapping[str, Any]) -> tuple[str, ...]:
    """Which entry the check actually read, from the record it computed for this tree."""
    heading = _field(record, "heading") or "none"
    return (f"newest {path} entry: {heading}",)


def _entry_headings(lines: Sequence[str]) -> list[int]:
    """The line numbers that open an entry: ``## `` lines outside any code fence."""
    headings: list[int] = []
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
        elif not in_fence and line.startswith(_ENTRY_PREFIX):
            headings.append(index)
    return headings


def _clauses(heading: str) -> dict[str, str]:
    """A heading read as ``key: value`` clauses; a part with no colon (the date) is not one."""
    clauses: dict[str, str] = {}
    for part in heading.split(_CLAUSE_SEPARATOR):
        key, assignment, value = part.partition(_CLAUSE_ASSIGNMENT)
        if assignment:
            clauses[key.strip().lower()] = value.strip()
    return clauses


def _block(record: Mapping[str, Any]) -> Mapping[str, Any]:
    block = record.get("changelog")
    return block if isinstance(block, dict) else {}


def _field(record: Mapping[str, Any], field: str) -> str | None:
    value = _block(record).get(field)
    return value if isinstance(value, str) else None
