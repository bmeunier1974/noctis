"""The DECIDE scorer — approval-side agreement, paired with the throughput it was bought at.

DECIDE is the model judgment that turns a tuned candidate into a promotion attempt or an honest
dead end. Scoring it is not "was the model right" — nothing in this system knows that — but a
narrower, checkable question: **of the candidates the model approved, what share did the promotion
gates then promote?** That is *approval-side* agreement, and the name is worn everywhere the number
appears, because an unqualified "agreement" invites the reading it cannot support (an overall
accuracy over approvals *and* rejections, which would need gate verdicts for candidates the gates
never saw).

**The pairing rule is structural, not a convention.** Agreement alone is trivially gamed: a
configuration that approves one candidate a year, the one it cannot possibly be wrong about, scores
a perfect 1.0 and is useless. So agreement and the approval rate are one frozen co-primary value
(:class:`ApprovalPair`) — there is no function here that returns agreement on its own, no flattened
copy of it on the whole-batch record, and the only rendering of it prints the approval rate in the
same block. A reader cannot quote half of the pair without going out of their way to strip the other
half off, which is exactly the friction the rule is for.

**A revise is a deferral, not a gradeable judgment.** The production emit vocabulary is
``approve | reject | revise`` (:data:`~noctis.research.driver.DECIDE_CONTRACT`, whose final-ask
variant drops ``revise`` once the cap is spent), and a ``revise`` says "ask me again", not "this
should ship". Counting it as a wrong approval would punish a model for declining to guess, and
counting it as a right rejection would reward the same. So it is excluded from agreement's numerator
*and* denominator, and reported as its own :attr:`DecideMetrics.revise_rate` beside
:attr:`DecideMetrics.revise_flip_rate` — of the candidates that earned a re-ask, how often the
verdict actually moved. A high revise rate whose flips are near zero is a site burning a round trip
to arrive where it started.

**Absences are ``n/a``, and they are counted.** The gates only ever labelled the candidates a run
actually put through them; an approval with no gate verdict is *unlabelled*, and it lands in
:attr:`ApprovalPair.unlabeled_approvals` rather than in agreement's denominator, where it would
silently deflate the number. Every ratio is denominator-guarded to ``None`` — a measured zero
(nothing was approved, out of candidates that really were decided) is a finding and stays ``0.0``;
a missing input (no labelled approvals, no revises, an empty batch) is ``None`` and renders ``n/a``.

**Comparing two configurations is not this module's arithmetic.** :func:`promotion_outcomes` turns
a batch into the case-keyed mapping :func:`noctis.eval.stats.compare` already pairs on — 1 promoted,
0 refused, ``None`` for an approval the gates never labelled — and that is the whole comparison
surface. The refusals come out of it for free and mean the right things: two configurations that
approved *different* candidates were never measured on one population (``mismatched_cases``), and an
unlabelled approval refuses rather than quietly shrinking the population (``missing_outcomes``).

**Pure.** Values in, values out: no file, no clock, no configuration, no seeded draw, and no import
of the engine at all — the verdict vocabulary is asserted against the driver's emit contract in the
suite instead, so drift is caught without coupling. The one module it reaches for is
:mod:`noctis.eval.reading`, the eval layer's own stdlib-only reading vocabulary, which owns the
pair's rows (:data:`~noctis.eval.reading.APPROVAL_PAIR`) and computes no figure. Any figure
published from here is reproducible from a fixture with a pencil.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType

from noctis.eval.reading import APPROVAL_PAIR, table

__all__ = [
    "APPROVE",
    "REJECT",
    "REVISE",
    "VERDICTS",
    "ApprovalPair",
    "DecideMetrics",
    "DecideOutcome",
    "GateLabel",
    "promotion_outcomes",
    "score_decide_batch",
]

# The production DECIDE emit vocabulary, spelled here rather than imported so the module stays pure
# and engine-free; the suite asserts the two agree, which is where drift is meant to be caught.
APPROVE = "approve"
REJECT = "reject"
REVISE = "revise"

#: Every terminal verdict a scored DECIDE outcome may carry.
VERDICTS: tuple[str, ...] = (APPROVE, REJECT, REVISE)


class GateLabel(Enum):
    """What the promotion gates said about a candidate the model decided — the label, or its lack.

    ``PROMOTED`` and ``REFUSED`` are the two gate verdicts (:mod:`noctis.champions.promotion` is the
    arbiter that produced them). ``UNLABELED`` is the honest third state, not a default nobody
    noticed: a candidate whose run never reached the gates has no verdict, and pretending it was
    refused would turn missing evidence into evidence.
    """

    PROMOTED = "promoted"
    REFUSED = "refused"
    UNLABELED = "unlabeled"


@dataclass(frozen=True)
class DecideOutcome:
    """One scored DECIDE case: what the model eventually decided, and what the gates said.

    ``verdict`` is the *eventual* verdict — the terminal state of the case, after any re-ask — from
    the production vocabulary. ``revised`` records that a revise happened on the way there (a case
    that *ended* on ``revise`` is one, by construction), and ``revise_flipped`` that the eventual
    verdict differed from the proposal the revise deferred. ``label`` is the gates' verdict, or
    ``UNLABELED`` when this case never reached them.

    The constructor refuses what a scored case cannot mean: a verdict outside the vocabulary, a flip
    with no revise behind it, a terminal ``revise`` that does not record its own revise, and a gate
    label on a deferral — a candidate the model never decided never reached the gates, so a gate
    verdict about it is fabricated evidence.
    """

    case_id: str
    verdict: str
    label: GateLabel = GateLabel.UNLABELED
    revised: bool = False
    revise_flipped: bool = False

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"{self.case_id!r}: verdict must be one of {', '.join(VERDICTS)} — "
                f"got {self.verdict!r}"
            )
        if self.revise_flipped and not self.revised:
            raise ValueError(
                f"{self.case_id!r}: a revise cannot have flipped the verdict without a revise"
            )
        if self.verdict == REVISE and not self.revised:
            raise ValueError(f"{self.case_id!r}: a case that ended on a revise records that revise")
        if self.verdict == REVISE and self.label is not GateLabel.UNLABELED:
            raise ValueError(
                f"{self.case_id!r}: a deferral never reached the gates, so it carries no gate "
                f"label — got {self.label.value}"
            )

    @property
    def approved(self) -> bool:
        """Whether the model's eventual verdict was an approval — the approval-side population."""
        return self.verdict == APPROVE

    @property
    def labeled(self) -> bool:
        """Whether the gates recorded a verdict about this candidate at all."""
        return self.label is not GateLabel.UNLABELED


@dataclass(frozen=True)
class ApprovalPair:
    """The co-primary pair: approval-side agreement **and** the approval rate it was bought at.

    One value, deliberately, because either number alone is a half-truth. Agreement is the share of
    the *labelled* approvals the gates promoted; the approval rate is the share of decided
    candidates the model approved at all. ``unlabeled_approvals`` is the explicit excluded count —
    approvals the gates never judged, kept out of agreement's denominator and named rather than
    implied. Both rates are ``None`` when their denominator is empty.
    """

    agreement: float | None
    approval_rate: float | None
    decided: int
    approvals: int
    labeled_approvals: int
    unlabeled_approvals: int
    promoted: int

    def render(self) -> str:
        """The pair, as the two lines a report prints — never one of them."""
        return APPROVAL_PAIR.render(self)

    def _rows(self) -> list[tuple[str, object]]:
        """The pair's rows, agreement first and the approval rate immediately beneath it.

        The rows themselves are :data:`~noctis.eval.reading.APPROVAL_PAIR`'s, not restated here:
        one declaration serves this rendering, :func:`~noctis.eval.decide_site.pair_block`'s keys
        and the bench report's, so they cannot drift apart.
        """
        return APPROVAL_PAIR.rows(self)


@dataclass(frozen=True)
class DecideMetrics:
    """Everything one scored DECIDE batch says, in one frozen record.

    The co-primary pair lives on :attr:`approval` and *only* there — there is no flattened
    ``agreement`` field here to quote on its own. Beside it sit the deferral figures: the share of
    decided candidates that earned a revise, and the share of those whose verdict then changed.
    """

    approval: ApprovalPair
    revise_rate: float | None
    revise_flip_rate: float | None
    revises: int
    revise_flips: int
    rejections: int

    @property
    def decided(self) -> int:
        """How many candidates this batch decided — the rates' shared denominator, stated once."""
        return self.approval.decided

    def render(self) -> str:
        """One reading of the batch: the pair, then the deferral figures, then the counts."""
        rows = self.approval._rows() + [
            ("Revise rate", self.revise_rate),
            ("Revise flip rate", self.revise_flip_rate),
            ("Revises", self.revises),
            ("  flipped the verdict", self.revise_flips),
            ("Rejections", self.rejections),
        ]
        return "\n".join(
            [
                f"DECIDE approval side — {self.decided} decided candidates",
                "",
                table(rows, label_w=APPROVAL_PAIR.label_w),
            ]
        )


def score_decide_batch(outcomes: Sequence[DecideOutcome]) -> DecideMetrics:
    """Score one batch of DECIDE outcomes into the paired agreement record.

    Total by construction: an empty batch yields a record whose every rate is ``None`` and whose
    counts are honest zeros, because nothing here can divide by nothing.
    """
    decided = len(outcomes)
    approvals = [outcome for outcome in outcomes if outcome.approved]
    labeled = [outcome for outcome in approvals if outcome.labeled]
    promoted = sum(1 for outcome in labeled if outcome.label is GateLabel.PROMOTED)
    revises = sum(1 for outcome in outcomes if outcome.revised)
    flips = sum(1 for outcome in outcomes if outcome.revise_flipped)

    return DecideMetrics(
        approval=ApprovalPair(
            agreement=_share(promoted, len(labeled)),
            approval_rate=_share(len(approvals), decided),
            decided=decided,
            approvals=len(approvals),
            labeled_approvals=len(labeled),
            unlabeled_approvals=len(approvals) - len(labeled),
            promoted=promoted,
        ),
        revise_rate=_share(revises, decided),
        revise_flip_rate=_share(flips, revises),
        revises=revises,
        revise_flips=flips,
        rejections=sum(1 for outcome in outcomes if outcome.verdict == REJECT),
    )


def promotion_outcomes(outcomes: Sequence[DecideOutcome]) -> Mapping[str, int | None]:
    """One batch as the case-keyed outcomes :func:`noctis.eval.stats.compare` pairs two configs on.

    The population is the *approvals*: a candidate the model rejected or deferred has no
    approval-side outcome and is not a case in the comparison at all. Within it, ``1`` is a
    promotion, ``0`` a refusal, and ``None`` an approval the gates never labelled — present, so the
    paired machinery refuses the comparison rather than silently shrinking the population it is
    about.

    Refuses a batch that decides one case twice: a case id with two answers cannot key a paired
    comparison, and picking one of them would be a comparison nobody could reproduce.
    """
    graded: dict[str, int | None] = {}
    for outcome in outcomes:
        if not outcome.approved:
            continue
        if outcome.case_id in graded:
            raise ValueError(
                f"case {outcome.case_id!r} was decided twice in one batch — a paired comparison "
                "is keyed by case id, so one case cannot carry two answers"
            )
        graded[outcome.case_id] = None if not outcome.labeled else int(_was_promoted(outcome))
    return MappingProxyType(graded)


def _was_promoted(outcome: DecideOutcome) -> bool:
    """Whether the gates promoted this labelled approval."""
    return outcome.label is GateLabel.PROMOTED


def _share(part: int, whole: int) -> float | None:
    """A ratio, or ``None`` for an empty denominator — the guard, stated once."""
    return part / whole if whole else None


# Read once at import so the pairing rule is a property of the type rather than a comment: the
# co-primary value must carry both halves, or this module refuses to load at all.
_PAIR_FIELDS = {field.name for field in fields(ApprovalPair)}
if not {"agreement", "approval_rate"} <= _PAIR_FIELDS:  # pragma: no cover — a structural assertion
    raise RuntimeError("the co-primary pair carries agreement and the approval rate, or neither")

# And the manifest that renders it may only name figures the pair really carries: a row reading an
# attribute that moved would fail one rendering at a time, months later, in whichever report ran
# first. It fails here instead, at import, for everybody at once.
if not APPROVAL_PAIR.attributes() <= _PAIR_FIELDS:  # pragma: no cover — a structural assertion
    raise RuntimeError("the approval manifest names a figure the co-primary pair does not carry")
