"""The strategy file's research record, as text.

A strategy is a file, and the file carries its own research record in the module docstring: a
one-sentence thesis, then the header fields ``status`` / ``style`` / ``symbols`` / ``tuned``
(``strategies/README.md``). This module owns that record — the value (:class:`StrategyHeader`,
frozen, and refusing an illegal ``status`` in its constructor), its vocabulary
(:data:`HEADER_FIELDS`, :data:`VALID_STATUSES`, :data:`FIELD_RE`), the one parser
(:meth:`StrategyHeader.parse`, aliased :func:`parse_header`) and the one writer
(:func:`stamp_header`, typed and keyword-only). The ``Params`` default write-back arrives here in
a later story of the same epic (#311).

Parse is **tolerant of the file and strict on the value**: a file with no docstring, or one that
does not even parse as Python, reads as the default ``draft`` header, but a status the file
*does* declare must be legal or :class:`HeaderError` is raised. The stamp is the mirror image —
**strict on the file and strict on the value**: it needs a docstring to write into, and it routes
its ``status`` through that same constructor before it edits anything, so an illegal status never
reaches a file and the library's write gate needs no second check.

The module edits *strings*: no path, no tier, no gate, no I/O. **It imports the standard library
and nothing else, by rule** — every reader of a strategy file depends on it, so it may drag
nothing behind it; ``tests/test_strategy_header.py`` pins the import closure.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

HEADER_FIELDS = ("status", "style", "symbols", "tuned")
VALID_STATUSES = ("draft", "candidate", "champion", "rejected")
# One header line: ``field: value`` (a trailing ``# comment`` is the value's, and stripped).
FIELD_RE = re.compile(rf"^({'|'.join(HEADER_FIELDS)})\s*:\s*(.*)$")


class HeaderError(ValueError):
    """A header value that is not legal — today, a ``status`` outside :data:`VALID_STATUSES`.

    A ``ValueError``, so callers that only ever knew that keep catching what they caught.
    """


@dataclass(frozen=True)
class StrategyHeader:
    """One strategy file's research record — legal by construction, immutable once parsed."""

    thesis: str = ""
    status: str = "draft"
    style: str = ""
    symbols: list[str] = field(default_factory=list)
    tuned: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise HeaderError(
                f"header status {self.status!r} invalid; want one of {VALID_STATUSES}"
            )

    def to_dict(self) -> dict:
        return {
            "thesis": self.thesis,
            "status": self.status,
            "style": self.style,
            "symbols": list(self.symbols),
            "tuned": self.tuned,
        }

    @staticmethod
    def parse(source: str) -> StrategyHeader:
        """Parse the docstring header (thesis first paragraph + ``field: value`` lines)."""
        try:
            doc = ast.get_docstring(ast.parse(source)) or ""
        except SyntaxError:
            return StrategyHeader()
        thesis_lines: list[str] = []
        status = "draft"
        style = ""
        symbols: list[str] = []
        tuned: str | None = None
        in_thesis = True
        for line in doc.splitlines():
            stripped = line.strip()
            match = FIELD_RE.match(stripped)
            if match:
                in_thesis = False
                value = match.group(2).split("#", 1)[0].strip()
                if match.group(1) == "symbols":
                    symbols = [s.strip().upper() for s in re.split(r"[,\s]+", value) if s.strip()]
                elif match.group(1) == "tuned":
                    tuned = value or None
                elif match.group(1) == "status":
                    status = value
                else:
                    style = value
                continue
            if in_thesis:
                if not stripped and thesis_lines:
                    in_thesis = False
                    continue
                if stripped:
                    thesis_lines.append(stripped)
        return StrategyHeader(
            thesis=" ".join(thesis_lines),
            status=status,
            style=style,
            symbols=symbols,
            tuned=tuned,
        )


parse_header = StrategyHeader.parse


# ─────────────────────────────────────────────────────────────────────────────
# The write side: the stamp
# ─────────────────────────────────────────────────────────────────────────────
def _docstring_span(source: str) -> tuple[int, int]:
    """(start, end) line indexes (0-based, end exclusive) of the module docstring."""
    tree = ast.parse(source)
    node = tree.body[0] if tree.body else None
    if (
        node is None
        or not isinstance(node, ast.Expr)
        or not isinstance(node.value, ast.Constant)
        or not isinstance(node.value.value, str)
    ):
        raise HeaderError("strategy file has no module docstring header")
    return node.lineno - 1, cast(int, node.end_lineno)


def stamp_header(
    source: str,
    *,
    status: str | None = None,
    style: str | None = None,
    symbols: Sequence[str] | None = None,
    tuned: str | None = None,
) -> str:
    """Return ``source`` with the named header fields written into its docstring header.

    A **partial** write on purpose: ``None`` means *leave that line exactly as it is*, byte for
    byte, trailing comment included — the seeds carry ``# draft | candidate | …`` hints on lines
    no stamp is changing, so re-rendering a whole :class:`StrategyHeader` would throw them away.
    A field the file does not declare yet is inserted before the closing quotes in
    :data:`HEADER_FIELDS` order; a one-line docstring is split open to hold it.

    ``status`` is validated *before* any edit, through the one spelling every reader shares
    (:class:`StrategyHeader`'s constructor), so a refused stamp leaves the source untouched and
    an illegal status can never reach a file. Keyword-only and typed: a mistyped field name is a
    ``TypeError`` from Python (and a mypy error before that), never a silently dropped stamp.
    """
    if status is not None:
        StrategyHeader(status=status)  # the one status check: raises HeaderError, writes nothing
    start, end = _docstring_span(source)
    lines = source.splitlines(keepends=True)
    doc_lines = lines[start:end]
    declared: dict[str, str | Sequence[str] | None] = {
        "status": status,
        "style": style,
        "symbols": symbols,
        "tuned": tuned,
    }
    pending: dict[str, str | Sequence[str]] = {k: v for k, v in declared.items() if v is not None}

    def render(name: str, value: str | Sequence[str]) -> str:
        if name == "symbols" and isinstance(value, (list, tuple)):
            value = " ".join(value)
        return f"{name}: {value}\n"

    for i, line in enumerate(doc_lines):
        match = FIELD_RE.match(line.strip())
        if match and match.group(1) in pending:
            name = match.group(1)
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            doc_lines[i] = f"{indent}{render(name, pending.pop(name)).rstrip()}{newline}"

    if pending:
        inserted = [render(name, pending[name]) for name in HEADER_FIELDS if name in pending]
        if len(doc_lines) == 1:
            # Single-line docstring: split it open so the fields live inside it.
            match = re.match(r"^(\s*)(\"\"\"|''')(.*?)(\2)\s*$", doc_lines[0].rstrip("\n"))
            if match is None:
                raise HeaderError("cannot rewrite docstring header (unusual quoting)")
            indent, quote, body = match.group(1), match.group(2), match.group(3)
            doc_lines = [f"{indent}{quote}{body}\n", "\n", *inserted, f"{indent}{quote}\n"]
        else:
            # Insert just before the closing quotes, blank-separated from a thesis
            # paragraph above (but packed together with existing header fields).
            closing = len(doc_lines) - 1
            above = doc_lines[closing - 1].strip()
            if above and not FIELD_RE.match(above):
                inserted = ["\n", *inserted]
            doc_lines = doc_lines[:closing] + inserted + doc_lines[closing:]

    return "".join(lines[:start] + doc_lines + lines[end:])
