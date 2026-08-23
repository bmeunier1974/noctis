"""The strategy file's research record, as text.

A strategy is a file, and the file carries its own research record in the module docstring: a
one-sentence thesis, then the header fields ``status`` / ``style`` / ``symbols`` / ``tuned``
(``strategies/README.md``). This module owns that record — the value (:class:`StrategyHeader`,
frozen, and refusing an illegal ``status`` in its constructor), its vocabulary
(:data:`HEADER_FIELDS`, :data:`VALID_STATUSES`, :data:`FIELD_RE`) and the one parser
(:meth:`StrategyHeader.parse`, aliased :func:`parse_header`). The write side — the header stamp
and the ``Params`` default write-back — arrives here in later stories of the same epic (#311).

Parse is **tolerant of the file and strict on the value**: a file with no docstring, or one that
does not even parse as Python, reads as the default ``draft`` header, but a status the file
*does* declare must be legal or :class:`HeaderError` is raised. That is why the write gate needs
no second status check.

The module edits *strings*: no path, no tier, no gate, no I/O. **It imports the standard library
and nothing else, by rule** — every reader of a strategy file depends on it, so it may drag
nothing behind it; ``tests/test_strategy_header.py`` pins the import closure.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

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
