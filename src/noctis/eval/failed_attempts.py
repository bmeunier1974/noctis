"""A run's rejected coder attempts, read back off disk and broken down by failure class.

:class:`~noctis.research.failed_store.FailedAttemptStore` already writes the raw material: one
file per attempt the write gate refused, under the working tier's ``strategies/__tmp/failed/``,
carrying the strategy name, the attempt number, the (often multi-line) gate error, the fixed
oracle on the spec path, and the source the coder actually emitted. What it does not do — what
nothing did before this module — is *count* them. Fifty files is a scroll; "31 of 50 failures
were the file not importing, and the knob that points at is the contract sheet" is a decision.

**Standalone by design.** This is the eval layer's cheapest useful thing: point it at a folder,
get a taxonomy breakdown. No corpus, no harness, no benchmark record, no site registry — a folder
path is the whole input, so an operator can read yesterday's run without running anything. The
only thing it borrows is the vocabulary itself
(:func:`~noctis.eval.coder_taxonomy.classify_coder_failure` and the knob annotations beside it),
which is the one part that must not be duplicated: a second copy of the classifier would drift
from the gate exactly as silently as a re-implemented matcher would.

**One I/O function over a pure core.** :func:`read_failed_attempts` is the only place a directory
is touched; parsing, counting and rendering are functions of data, so a breakdown is reproducible
from the records alone and testable without a filesystem. The header format is *parsed*, not
guessed at — the markers below are quoted from
:meth:`~noctis.research.failed_store.FailedAttemptStore._render`, and every test in
``tests/test_eval_failed_attempts.py`` reads files the real store wrote, so a reworded header
fails a test here instead of quietly emptying the reader.

**Two honesty rules, chosen and stated.**

* *An empty folder is a fact; a missing folder is a question.* A folder that exists and holds
  nothing means "this run rejected nothing" and reports zero attempts. A folder that does not
  exist (or is not a folder) means the caller named the wrong path, and answering "zero failures"
  to that would be a lie with a comforting shape — so it refuses with
  :class:`MissingFailureFolder`, naming the path.
* *The escape hatch is always on the screen.* ``other``/``unclassified`` is rendered with every
  other class, at zero when nothing missed, and a share **strictly above**
  :data:`ROT_THRESHOLD` (10% — one failure in ten nobody has a class for is enough to make the
  other nine shares untrustworthy) adds an explicit rot-warning line. A file that could not be
  parsed is never binned into a class either: it is skipped, and named in the result's warnings,
  because a mis-binned file is worse than a missing one.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from noctis.eval.coder_taxonomy import CODER_CLASSES, OTHER, classify_coder_failure

__all__ = [
    "ROT_THRESHOLD",
    "Breakdown",
    "ClassTally",
    "FailedAttempt",
    "FailedAttempts",
    "MissingFailureFolder",
    "SkippedFile",
    "failure_breakdown",
    "read_failed_attempts",
    "render_breakdown",
]

#: The escape hatch's share above which the vocabulary is reported as rotting. One failure in ten
#: that no class recognises already makes the other nine shares a partial picture; it is a
#: threshold to act on (grow the taxonomy), not a gate anything passes or fails.
ROT_THRESHOLD = 0.10

# The store's header, quoted from FailedAttemptStore._render. Each marker is exactly the line that
# writer emits; the round-trip tests drive the real store, so a reword breaks here loudly.
_HEADER_RE = re.compile(
    r"^# rejected coder attempt — strategy (?P<name>.+), attempt (?P<attempt>\d+)$"
)
_RECORDED_PREFIX = "# recorded: "
_ERROR_HEAD = "# gate error:"
_ORACLE_HEAD = "# fixed oracle:"
_SOURCE_HEAD = "# --- attempted source below ---"

# Every line of a header section is the original line under a comment marker and three spaces, so
# a blank line in a gate error is "#   " and an indented one keeps its own indentation after it.
_CONTINUATION = "#   "

# The store's zero-padded monotonic sequence: what puts the folder back in write order.
_SEQUENCE_RE = re.compile(r"^(\d+)-")

# Column widths for the one-screen rendering: names, counts, shares, then the knob.
_COUNT_WIDTH = 5
_SHARE_WIDTH = 7


class MissingFailureFolder(FileNotFoundError):
    """The failure folder named is not there — the one thing this reader refuses to answer."""


@dataclass(frozen=True)
class FailedAttempt:
    """One rejected coder attempt, as the store wrote it and this module read it back.

    ``error`` is the gate's own text with its line breaks intact (the diagnostics ride the later
    lines), ``oracle`` is the fixed target the attempt was gated against on the spec path and
    ``None`` everywhere else, and ``source`` is the emitted file byte-for-byte below the header.
    """

    name: str
    attempt: int
    error: str
    source: str
    filename: str
    oracle: str | None = None


@dataclass(frozen=True)
class SkippedFile:
    """A file in the failure folder that is not a readable attempt — named, never counted."""

    filename: str
    reason: str

    def line(self) -> str:
        """``notes.txt: first line is not the store's header`` — one greppable line."""
        return f"{self.filename}: {self.reason}"


@dataclass(frozen=True)
class FailedAttempts:
    """What one folder held: the attempts that parsed, and every file that did not."""

    records: tuple[FailedAttempt, ...] = ()
    warnings: tuple[SkippedFile, ...] = ()


@dataclass(frozen=True)
class ClassTally:
    """One failure class's row of the breakdown: how often, what share, and which knob."""

    name: str
    knob: str
    count: int
    share: float


@dataclass(frozen=True)
class Breakdown:
    """Every declared class counted over one batch of attempts, escape hatch included and last.

    Zero-hit classes are present at zero — a class that stopped happening and a class that
    stopped matching look identical if the row disappears, and telling those apart is the point.
    """

    tallies: tuple[ClassTally, ...]
    total: int

    @property
    def escape_hatch_share(self) -> float:
        """The rot gauge on its own: the share of attempts no class recognised."""
        return next(tally.share for tally in self.tallies if tally.name == OTHER)

    @property
    def rotting(self) -> bool:
        """Whether the escape hatch's share is strictly above :data:`ROT_THRESHOLD`."""
        return self.escape_hatch_share > ROT_THRESHOLD


def read_failed_attempts(folder: Path | str) -> FailedAttempts:
    """Read one ``failed/`` folder: the attempts it holds, and a warning per file that is not one.

    The only function here that touches a directory. Records come back in the store's own write
    order (its zero-padded sequence), files that are not readable attempts are skipped with a
    named reason rather than guessed at, and a folder that does not exist refuses rather than
    reporting an empty run.
    """
    root = Path(folder)
    if not root.is_dir():
        raise MissingFailureFolder(
            f"no failure folder at {root} — an empty folder reports zero rejected attempts, but a "
            f"folder that is not there is a wrong path, not a clean run (a run's failures live in "
            f"<run_dir>/strategies/__tmp/failed/)"
        )
    records: list[FailedAttempt] = []
    warnings: list[SkippedFile] = []
    for entry in sorted(root.iterdir(), key=lambda path: _write_order(path.name)):
        if not entry.is_file():
            warnings.append(SkippedFile(entry.name, "not a file"))
            continue
        try:
            text = entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:  # unreadable or not text at all
            warnings.append(SkippedFile(entry.name, f"unreadable: {exc}"))
            continue
        parsed = _parse_attempt(entry.name, text)
        if isinstance(parsed, SkippedFile):
            warnings.append(parsed)
        else:
            records.append(parsed)
    return FailedAttempts(records=tuple(records), warnings=tuple(warnings))


def failure_breakdown(records: Iterable[FailedAttempt]) -> Breakdown:
    """Count a batch of attempts into the coder vocabulary — every class, in precedence order.

    Pure and order-independent: each attempt's gate error is classified on its own. An empty
    batch is all zeros rather than a division nobody can perform.
    """
    counts = dict.fromkeys((declared.name for declared in CODER_CLASSES), 0)
    total = 0
    for record in records:
        counts[classify_coder_failure(record.error)] += 1
        total += 1
    tallies = tuple(
        ClassTally(
            name=declared.name,
            knob=declared.knob,
            count=counts[declared.name],
            share=(counts[declared.name] / total if total else 0.0),
        )
        for declared in CODER_CLASSES
    )
    return Breakdown(tallies=tallies, total=total)


def render_breakdown(breakdown: Breakdown, *, skipped: Sequence[SkippedFile] = ()) -> str:
    """One screen: a row per class with its count, share and knob, then the escape hatch reading.

    ``skipped`` is rendered under the table when supplied, because a file that could not be read
    is missing from every count above it and a reader should be told so on the same screen.
    """
    name_width = max(len(tally.name) for tally in breakdown.tallies)
    lines = [
        f"Coder failures — {breakdown.total} attempts, {len(breakdown.tallies)} classes",
        "",
    ]
    lines += [
        f"{tally.name:<{name_width}}{tally.count:>{_COUNT_WIDTH}}"
        f"{tally.share:>{_SHARE_WIDTH}.1%}  {tally.knob}"
        for tally in breakdown.tallies
    ]
    lines += [
        "",
        f"escape hatch ({OTHER}): {breakdown.escape_hatch_share:.1%} of {breakdown.total} "
        f"attempts — rot threshold {ROT_THRESHOLD:.0%}",
    ]
    if breakdown.rotting:
        lines.append(
            f"ROT WARNING — {breakdown.escape_hatch_share:.1%} of these failures matched no "
            f"class, above the {ROT_THRESHOLD:.0%} threshold: the vocabulary has stopped covering "
            "the gate, so grow it before trusting the shares above."
        )
    if skipped:
        lines += ["", f"skipped {len(skipped)} file(s) — counted in nothing above:"]
        lines += [f"  {warning.line()}" for warning in skipped]
    return "\n".join(lines)


# ── parsing: pure over (filename, text), so the store's format has exactly one reader ──────
def _parse_attempt(filename: str, text: str) -> FailedAttempt | SkippedFile:
    """One file's text as a record, or the named reason it is not a recorded attempt."""
    header, marker, source = text.partition(_SOURCE_HEAD + "\n")
    if not marker:
        return SkippedFile(filename, f"no {_SOURCE_HEAD!r} marker — not a recorded attempt")
    lines = header.splitlines()
    opening = _HEADER_RE.match(lines[0]) if lines else None
    if opening is None:
        return SkippedFile(
            filename, "first line is not the store's 'rejected coder attempt' header"
        )
    rest = lines[1:]
    if not rest or not rest[0].startswith(_RECORDED_PREFIX):
        return SkippedFile(filename, f"no {_RECORDED_PREFIX.strip()!r} stamp under the header")
    rest = rest[1:]
    if not rest or rest[0] != _ERROR_HEAD:
        return SkippedFile(filename, f"no {_ERROR_HEAD!r} section")
    error, rest = _section(rest[1:])
    oracle: str | None = None
    if rest and rest[0] == _ORACLE_HEAD:
        oracle, rest = _section(rest[1:])
    if rest:
        return SkippedFile(filename, f"unexpected header line {rest[0]!r}")
    return FailedAttempt(
        name=_strategy_name(opening.group("name")),
        attempt=int(opening.group("attempt")),
        error=error,
        source=source,
        filename=filename,
        oracle=oracle,
    )


def _section(lines: Sequence[str]) -> tuple[str, list[str]]:
    """The commented block at the head of ``lines``, un-commented, and what follows it."""
    body: list[str] = []
    index = 0
    while index < len(lines) and lines[index].startswith(_CONTINUATION):
        body.append(lines[index][len(_CONTINUATION) :])
        index += 1
    return "\n".join(body), list(lines[index:])


def _strategy_name(token: str) -> str:
    """The name the store wrote with ``repr``, read back — quoting undone, never evaluated."""
    try:
        parsed = ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return token.strip("'\"")
    return parsed if isinstance(parsed, str) else token


def _write_order(filename: str) -> tuple[int, int, str]:
    """The store's own order: by its monotonic sequence, with anything unsequenced last."""
    sequence = _SEQUENCE_RE.match(filename)
    if sequence is None:
        return (1, 0, filename)
    return (0, int(sequence.group(1)), filename)
