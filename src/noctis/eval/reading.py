"""The eval layer's reading vocabulary: how a figure prints, and how a co-primary pair is declared.

Every reading this layer publishes — the DECIDE approval pair, the coder's pass rates, a paired
comparison, a whole bench report — is the same two things underneath: cells formatted one way, and
rows named once. Before this module they were neither. Four copies of a four-line ``_fmt`` decided
independently what a missing number looks like, and the rows of a co-primary pair were restated in
the type that owns them, in the block builder that publishes them and in the reader that renders
them. Nothing was wrong; everything was duplicated, which is the same thing one release later.

**One spelling of an absence.** :func:`fmt` is the single place a ``None`` becomes the word
:data:`NOT_AVAILABLE`, a rate becomes four decimals and a flag becomes ``yes``/``no``. The flag arm
is first on purpose: a ``bool`` *is* an ``int`` in Python, so a later arm would print ``True`` where
a reader expects a word. :func:`table` lays those cells out in two columns at whatever widths a
caller asks for — the only reason a width is a parameter is that the coder's labels are longer than
DECIDE's, and a report lines up with the block it quotes.

**A co-primary pair is declared once.** :class:`PairManifest` is one pair's whole vocabulary: the
figures it is made of, the order a JSON block publishes them in, the order a report prints them in,
the **flagship** figure that may never appear on its own, and the refusal text for a block that
carries the flagship without the rest. Two manifests are declared here — :data:`APPROVAL_PAIR` and
:data:`PASS_RATES`, together :data:`PAIR_MANIFESTS` — and the dataclasses that own the arithmetic
keep it: :class:`~noctis.eval.decide_scorer.ApprovalPair` and
:class:`~noctis.eval.coder_scorer.PassRates` compute their figures and route their rendering
through their manifest, so a row cannot drift between the three places that used to spell it.

Why the two orders are separate tuples rather than one: a block and a report genuinely disagree.
The approval block publishes ``decided``, which no report row prints, and prints ``promoted`` above
the two counts the block lists it below; the coder's rates block publishes a label that the report
folds *into* the job rate's own row, and prints three counts the enclosing block owns. Declaring
one ordering and deriving the other would have meant re-ordering a published document, which is
exactly what the golden records in ``tests/test_eval_goldens.py`` exist to refuse.

**Pure, and stdlib-only.** No engine import, no other eval module, no file, no clock. Every site's
reading depends on this module, so this module may drag nothing behind it —
``tests/test_eval_reading.py`` holds it to that both statically and as a module closure in a fresh
interpreter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "APPROVAL_PAIR",
    "LABEL_WIDTH",
    "NOT_AVAILABLE",
    "PAIR_MANIFESTS",
    "PASS_RATES",
    "VALUE_WIDTH",
    "PairManifest",
    "PairRow",
    "fmt",
    "table",
]

#: How every unknown figure renders, everywhere in the eval layer. One token, spelled once.
NOT_AVAILABLE = "n/a"

#: The two-column default: DECIDE's label column, and the value column every reading shares.
LABEL_WIDTH = 28
VALUE_WIDTH = 12


def fmt(value: object) -> str:
    """One cell: :data:`NOT_AVAILABLE` for a null, four decimals for a rate, ``yes``/``no`` for a
    flag, the value itself otherwise.

    The flag arm sits above the number arm deliberately — ``isinstance(True, int)`` is ``True``, so
    the obvious ordering prints a Python repr where a reader is owed a word.
    """
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(
    rows: Sequence[tuple[str, object]],
    *,
    label_w: int = LABEL_WIDTH,
    value_w: int = VALUE_WIDTH,
) -> str:
    """Label/value rows in two columns, in the order handed in — every cell through :func:`fmt`."""
    return "\n".join(f"{label:<{label_w}}{fmt(value):>{value_w}}" for label, value in rows)


# ── the co-primary pair, declared once ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairRow:
    """One figure of a pair: what a block calls it, how a report prints it, what it reads.

    ``key`` is the JSON key, and the attribute the figure is read off its pair — unless ``attr``
    names a different one, which is how the coder's ``pass_label`` key reads the ``label`` field.
    ``label`` is the printed row's label, or ``None`` for a figure no report prints a row of its own
    for (a count a block publishes, a qualification another row folds in). A label may carry a
    ``{field}`` slot, filled from the pair itself, so a qualification like *pass@k with feedback*
    travels in the row rather than in a caption.
    """

    key: str
    label: str | None = None
    attr: str | None = None

    @property
    def field(self) -> str:
        """The attribute this figure is read off the pair that owns the arithmetic."""
        return self.attr if self.attr is not None else self.key


@dataclass(frozen=True)
class PairManifest:
    """One pair's whole reading vocabulary — the rows, the keys, the flagship, the refusal.

    ``figures`` declares every figure once, **in the order a report prints them**; the ones with no
    label are skipped when printing. ``keys`` is the order a JSON block publishes them in, naming
    figures from ``figures``. ``flagship`` is the figure that may never be published or printed on
    its own — agreement without the approval rate, a retry-informed pass rate without the opening
    ask's — and ``refusal`` is what a reader prints instead when it finds one that way.

    Constructing a manifest that publishes a key no figure declares, or whose flagship is not one of
    its figures, raises: a half-declared pair would render a half-truth, which is the one thing the
    pairing rule exists to prevent.
    """

    name: str
    figures: tuple[PairRow, ...]
    keys: tuple[str, ...]
    flagship: str
    refusal: str
    label_w: int = LABEL_WIDTH

    def __post_init__(self) -> None:
        declared = {figure.key for figure in self.figures}
        unknown = [key for key in self.keys if key not in declared]
        if unknown:
            raise ValueError(
                f"the {self.name} pair manifest publishes {', '.join(unknown)}, which no figure "
                "it declares carries"
            )
        if self.flagship not in declared:
            raise ValueError(
                f"the {self.name} pair manifest names {self.flagship} as its flagship, which no "
                "figure it declares carries"
            )

    def attributes(self) -> frozenset[str]:
        """Every attribute this manifest reads off its pair — the subset guard's left-hand side."""
        return frozenset(figure.field for figure in self.figures)

    def rows(self, pair: object) -> list[tuple[str, object]]:
        """The pair's printed rows, the flagship first and the counts that qualify it beneath."""
        values = {figure.field: getattr(pair, figure.field) for figure in self.figures}
        return [
            (figure.label.format_map(values), values[figure.field])
            for figure in self.figures
            if figure.label is not None
        ]

    def render(self, pair: object) -> str:
        """The pair, as the block of lines a report prints — never one figure of it."""
        return table(self.rows(pair), label_w=self.label_w)

    def block(self, pair: object) -> dict[str, Any]:
        """The pair as record data, whole: the flagship is never published without its partner."""
        figures = {figure.key: figure for figure in self.figures}
        return {key: getattr(pair, figures[key].field) for key in self.keys}


# ── DECIDE: approval-side agreement, and the approval rate it was bought at ───────────────

_AGREEMENT = "agreement"

APPROVAL_PAIR = PairManifest(
    name="approval",
    figures=(
        PairRow(_AGREEMENT, "Approval-side agreement"),
        PairRow("approval_rate", "Approval rate"),
        # Published as the rates' shared denominator; the report states it in its own header
        # instead, so it prints no row here.
        PairRow("decided"),
        PairRow("approvals", "Approvals"),
        PairRow("promoted", "  gates promoted"),
        PairRow("labeled_approvals", "  gates labeled"),
        PairRow("unlabeled_approvals", "  unlabeled (n/a)"),
    ),
    keys=(
        _AGREEMENT,
        "approval_rate",
        "decided",
        "approvals",
        "labeled_approvals",
        "unlabeled_approvals",
        "promoted",
    ),
    flagship=_AGREEMENT,
    refusal=(
        f"REFUSED — this block carries {_AGREEMENT!r} without the approval rate and counts that "
        "qualify it. Agreement alone is trivially gamed (approve one candidate, score 1.0), so the "
        "co-primary pair renders whole or not at all."
    ),
)


# ── the coder: the opening ask's pass rate, and the whole job's ───────────────────────────

_JOB_PASS_RATE = "job_pass_rate"

PASS_RATES = PairManifest(
    name="pass rates",
    figures=(
        # The qualification the job rate wears. It is published as its own key and printed inside
        # the job rate's label, so the words cannot be separated from the number.
        PairRow("pass_label", attr="label"),
        PairRow("first_attempt_pass_rate", "First-attempt pass rate"),
        PairRow(_JOB_PASS_RATE, "Job pass rate ({label})"),
        PairRow("jobs", "Jobs"),
        PairRow("cases", "  cases"),
        PairRow("first_attempt_passes", "  first ask landed"),
        PairRow("passed_jobs", "  landed in the end"),
        PairRow("unattempted_jobs", "  never asked (n/a)"),
    ),
    keys=(
        "pass_label",
        "first_attempt_pass_rate",
        _JOB_PASS_RATE,
        "first_attempt_passes",
        "passed_jobs",
    ),
    flagship=_JOB_PASS_RATE,
    refusal=(
        f"REFUSED — this block carries {_JOB_PASS_RATE!r} without the first-attempt rate and "
        "counts that qualify it. A retry-informed rate alone is a half-truth (a composition that "
        "fails first and recovers twice is not the product one that lands it immediately is), so "
        "the co-primary pair renders whole or not at all."
    ),
    label_w=36,
)


#: Every co-primary pair the eval layer declares. A reader that renders pairs generically walks
#: this, so a third pair is one declaration here rather than an edit in every reader.
PAIR_MANIFESTS: tuple[PairManifest, ...] = (APPROVAL_PAIR, PASS_RATES)
