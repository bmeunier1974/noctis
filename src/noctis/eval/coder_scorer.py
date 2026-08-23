"""The coder site's reading — two co-primary pass rates, and everything that explains their gap.

The eval core scores what every benchmarkable site shares: did the ask emit, in how many *runner*
attempts, at what cost. The coder's interesting arithmetic is one level down. One runner attempt is
one whole authoring **job** (:mod:`noctis.eval.coder_site`), and inside a job the engine may retry
privately, each retry seeing the write gate's own error, before falling back to a paid coder. So the
numbers an operator compares two prompt compositions on are numbers over *internal* attempts, and
they are computed here, from the :class:`~noctis.eval.coder_site.JobRecord` the runner retained.

**Two rates, co-primary, never blended.** ``first_attempt_pass`` (did the composition get a
gate-clean file out of the opening ask?) and ``job_pass`` (did it get there at all, inside the retry
budget?) answer different questions, and either alone is a half-truth: a composition that fails
first and recovers twice is not the same product as one that lands it immediately, and a
composition that never recovers is not the same as one that does. They are therefore one frozen
value (:class:`PassRates`) — there is no function here that returns one of them, no flattened copy
of either on :class:`CoderMetrics`, and no third number that folds them together. **Nothing in this
module is a combined score.** Cost sits beside the pair rather than inside it: the Pareto view is
the numbers, read together.

**"pass@k with feedback", wherever retry-informed passing renders.** A retry here is not an
independent re-draw — it is re-asked *with the gate's rejection in the prompt* — so calling the
job-level number ``pass@k`` would claim a weaker, more familiar thing than what was measured.
:data:`FEEDBACK_LABEL` is carried in every block that reports a figure retries contributed to
(:data:`RETRY_INFORMED_BLOCKS`), so the words travel with the number rather than living in a caption
somebody may not have read.

**Absences are ``n/a``.** Every ratio is denominator-guarded, exactly as
:mod:`noctis.eval.metrics` guards its own: a *measured* zero (nothing passed, out of jobs that
really asked) is a finding and stays ``0.0``; a *missing* input (a job whose brief the engine
refused before spending a completion, a batch with nothing to price, a class share over no failures
at all) is ``None``. A bench that answered nothing publishes no reading rather than empty figures.

**The generic arithmetic is reused, and where it could not be, that is said.** A job's internal
attempts are exactly the shape :class:`~noctis.eval.metrics.CaseResult` takes, so the within-case-
first pass rates, the attempts-to-pass mean, the attempt distribution, the retry yield and the whole
cost family are :mod:`noctis.eval.metrics`' functions applied to *internal* attempts — not
re-derived here. Three figures are coder-specific because the runner-level shapes genuinely do not
fit:

* **wall clock** is recorded per *job*, not per internal attempt (the engine holds one clock around
  the whole authoring run), so latency is read at the job level, where the clock really ran, by
  feeding the generic percentile machinery one attempt per job;
* **escalation** has no runner-level counterpart at all — the fallback to the paid coder happens
  inside a job, so "of the jobs that escalated, how many did the fallback rescue" and "what did a
  rescued file cost" are computed here, over the ``escalated`` flag #225 records per attempt;
* **the failure taxonomy** counts internal attempts rather than jobs, through
  :func:`~noctis.eval.failed_attempts.breakdown_of_errors` — the same counting a run's failure
  census uses, so a bench and a folder never disagree about the same errors.

**The pair again, per difficulty axis.** The axes a coder case is labelled on exist to be split by:
a job pass rate that is one thing on ``bars_only`` briefs and another on ``exits`` ones is two
findings, not one. So the reading carries a :data:`STRATA_KEY` block — axis → level → the same
co-primary pair over just those jobs, computed by the same arithmetic as the whole batch
(:func:`strata_block`), exactly as :func:`noctis.eval.decide_site.strata_block` stratifies DECIDE's.
*Which* axes is the site declaration's answer, not this module's: the seven ride on
:data:`~noctis.eval.coder_distill_sites.CODER_SITE`'s ``difficulty_axes`` and the runner hands them
to :meth:`CoderReadingScorer.read`, while the grouping itself is
:func:`~noctis.eval.reading.strata`'s — the one loop both sites stratify through (#306).
A stratum publishes the **pass breakdown** and not a second copy of the spend and taxonomy blocks:
those repeated across twenty levels are a wall nobody reads, while the pair is what an axis was
invented to split. A level no case in the batch carries is absent rather than present at zero.

**Detector warnings ride beside the scores and provably never inside them.** The degenerate-pass
detectors (:mod:`noctis.eval.coder_detectors`) run at the end of a passing job and are stamped into
its record; this module copies them into warning rows and reads them nowhere else. Every figure
above is computed from attempts, usage and durations — a :class:`DegenerateFinding` carries none of
those, and the same batch with and without findings differs only in its warning rows.

**Pure over records.** No file, no clock, no configuration, no network: values in, a mapping out, so
any published figure is reproducible from the retained job records with a pencil.

**Two imports are deferred to call time, and that is structural.** The coder declaration
(:data:`~noctis.eval.coder_distill_sites.CODER_SITE`) names this module's scorer in its ``scorers``
slot, while the job record and the failure counting both sit on modules that read that same
declaration. The cycle is closed at *call* time (the two imports sit inside the functions that use
them) — exactly as :mod:`noctis.eval.decide_site` closes its own — so importing any of the modules
first works. Everything else here is a top-level import.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from noctis.eval.case import Case
from noctis.eval.coder_case import CODER_AXES
from noctis.eval.metrics import (
    AttemptDistribution,
    AttemptOutcome,
    CaseResult,
    CostMetrics,
    LatencyMetrics,
    attempt_distribution,
    cost_metrics,
    first_attempt_pass_rate,
    job_pass_rate,
    latency_metrics,
    mean_attempts_to_pass,
    retry_yield,
)
from noctis.eval.reading import (
    ANSWERS_FRESH,
    ANSWERS_KEY,
    ATTEMPT_CALLS_KEY,
    NOT_APPLICABLE,
    PASS_RATES,
    STRATA_KEY,
    Pair,
    SiteReading,
    strata,
)
from noctis.eval.reading import table as render_table
from noctis.eval.site import AnsweredCase
from noctis.research.pricing import USAGE_FIELDS, PriceTable, default_table

if TYPE_CHECKING:  # the cycle-closing imports, for annotations only — see the module docstring
    from noctis.eval.coder_site import AttemptRecord, JobRecord
    from noctis.eval.failed_attempts import Breakdown

__all__ = [
    "ANSWERS_FRESH",
    "CODER_DIALS_KEY",
    "CODER_SCORER",
    "FEEDBACK_LABEL",
    "NOT_APPLICABLE",
    "PASS_LABEL_KEY",
    "RATES_KEY",
    "RETRY_INFORMED_BLOCKS",
    "STRATA_KEY",
    "UNATTEMPTED_KEY",
    "UNREADABLE_KEY",
    "CoderMetrics",
    "CoderReadingScorer",
    "EscalationReading",
    "PassRates",
    "coder_block",
    "coder_reading",
    "job_records",
    "score_coder_jobs",
    "strata_block",
]

#: The key the whole coder reading rides under, inside the dials subtree a record quotes verbatim.
CODER_DIALS_KEY = "coder"

# ``ANSWERS_FRESH`` — what ``dials.answers`` says on a bench that really asked a model — is not
# this site's word and not DECIDE's either: it is the eval layer's, spelled once in
# :mod:`noctis.eval.reading` and re-exported here. It used to be spelled in both sites with a suite
# test pinning the copies equal, which is a pin that can only report drift after it happens (#305).

#: What retry-informed passing is called, in full, wherever it renders. Never plain ``pass@k``:
#: the retries were shown the gate's rejection, which is a materially easier question.
FEEDBACK_LABEL = "pass@k with feedback"

#: The key :data:`FEEDBACK_LABEL` is published under inside each block that carries it.
PASS_LABEL_KEY = "pass_label"

#: Every block of the reading whose figures a retry contributed to. Carrying the label in each of
#: them is what keeps the words attached to the number rather than to a caption.
RETRY_INFORMED_BLOCKS: tuple[str, ...] = ("rates", "effort", "escalation", "cost")

#: The two exclusion counts the reading carries beside the pair — the n/a side, named.
UNREADABLE_KEY = "unreadable"
UNATTEMPTED_KEY = "unattempted_jobs"

#: The key the co-primary pair rides under inside the coder's block — the word DECIDE spells
#: ``approval``. It travels with the manifest and the value on one
#: :class:`~noctis.eval.reading.Pair`, so a block builder no longer spells it by hand.
RATES_KEY = "rates"

# ``STRATA_KEY`` is the key the per-axis breakdown rides under, on this site and DECIDE's alike, so
# one generic reader renders both; ``NOT_APPLICABLE`` is what the shared grouping loop keys a
# stratum by when a job's case carries no level on an axis (every validated coder case is labelled
# on all seven, so it is a defensive spelling rather than an expected row). Both are the eval
# layer's words, imported from :mod:`noctis.eval.reading` and re-exported here (#305).


# ── the co-primary pair ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PassRates:
    """The co-primary pair: the opening ask's pass rate **and** the whole job's, with its counts.

    One value, deliberately, because either number alone misdescribes the composition (see the
    module docstring). ``job_pass_rate`` is a *pass@k with feedback* rate and the type says so —
    :attr:`label` is a field, not a docstring, so the qualification cannot be dropped on the way to
    a report. Both rates are ``None`` when no job recorded an internal attempt at all.
    """

    first_attempt_pass_rate: float | None
    job_pass_rate: float | None
    cases: int
    jobs: int
    first_attempt_passes: int
    passed_jobs: int
    unattempted_jobs: int
    label: str = FEEDBACK_LABEL

    @classmethod
    def rate_fields(cls) -> tuple[str, ...]:
        """The two halves that are published together or not at all."""
        return ("first_attempt_pass_rate", "job_pass_rate")

    def render(self) -> str:
        """The pair, as the two lines a report prints — never one of them."""
        return PASS_RATES.render(self)

    def _rows(self) -> list[tuple[str, object]]:
        """The pair's rows, the opening ask first and the retry-informed rate beneath it.

        The rows themselves are :data:`~noctis.eval.reading.PASS_RATES`', not restated here: one
        declaration serves this rendering and :func:`_rates_block`'s keys, and it is what carries
        :data:`FEEDBACK_LABEL` into the job rate's own label rather than into a caption.
        """
        return PASS_RATES.rows(self)


@dataclass(frozen=True)
class EscalationReading:
    """What falling back to the paid coder bought: how often it rescued a job, and at what price.

    ``rescue_rate`` is over the jobs that really escalated — a bench where nothing escalated has no
    rescue rate, not a zero one. ``usd_per_rescued_file_estimate`` charges the *whole* escalated
    job, the local failures included, to the file it eventually bought: that is the number a
    ``max_escalations`` decision is actually argued from, and it inherits the pricing module's
    all-or-nothing posture, so one unpriceable attempt makes it ``None``.
    """

    escalated_jobs: int
    rescued_jobs: int
    rescue_rate: float | None
    usd_per_rescued_file_estimate: float | None
    pricing_table_version: str


@dataclass(frozen=True)
class CoderMetrics:
    """Everything one scored batch of coder jobs says, in one frozen record.

    The co-primary pair lives on :attr:`rates` and *only* there — there is no flattened rate here
    to quote on its own, and no field that folds the two into one figure.
    """

    rates: PassRates
    mean_attempts_to_pass: float | None
    attempt_distribution: AttemptDistribution | None
    retry_yield: Mapping[int, float] | None
    cost: CostMetrics
    latency: LatencyMetrics
    seconds_total: float | None
    escalation: EscalationReading
    failures: Breakdown

    def render(self) -> str:
        """One reading of the batch: the pair, then the effort and spend that explain it."""
        rows = self.rates._rows() + [
            ("Mean attempts to pass", self.mean_attempts_to_pass),
            ("Escalation rescue rate", self.escalation.rescue_rate),
            ("USD per rescued file", self.escalation.usd_per_rescued_file_estimate),
            ("USD per pass (est.)", self.cost.usd_per_pass_estimate),
            ("Tokens per pass", self.cost.tokens_per_pass),
            ("p50 seconds to pass", self.latency.p50_seconds),
        ]
        return "\n".join(
            [
                f"Coder site — {self.rates.jobs} job(s) over {self.rates.cases} case(s); "
                f"job passing is {FEEDBACK_LABEL}",
                "",
                render_table(rows, label_w=PASS_RATES.label_w),
            ]
        )


# ── the arithmetic ────────────────────────────────────────────────────────────────────────


def score_coder_jobs(records: Sequence[JobRecord], table: PriceTable | None = None) -> CoderMetrics:
    """Score one batch of retained authoring jobs into the paired coder record.

    Total by construction: an empty batch yields a record whose every rate is ``None`` and whose
    counts are honest zeros, because nothing here can divide by nothing.
    """
    priced = table or default_table()
    internal = _internal_results(records)
    jobs = _job_results(records)
    return CoderMetrics(
        rates=_pass_rates(records, internal),
        mean_attempts_to_pass=mean_attempts_to_pass(internal),
        attempt_distribution=attempt_distribution(internal),
        retry_yield=retry_yield(internal),
        cost=cost_metrics(internal, priced),
        latency=latency_metrics(jobs),
        seconds_total=_seconds_total(records),
        escalation=_escalation(records, priced),
        failures=_failures(records),
    )


def _pass_rates(records: Sequence[JobRecord], internal: Sequence[CaseResult]) -> PassRates:
    """The co-primary pair, computed once — both halves or neither, by construction."""
    return PassRates(
        first_attempt_pass_rate=first_attempt_pass_rate(internal),
        job_pass_rate=job_pass_rate(internal),
        cases=len({record.case_id for record in records}),
        jobs=len(records),
        first_attempt_passes=sum(1 for record in records if record.first_attempt_passed),
        passed_jobs=sum(1 for result in internal if result.passed),
        unattempted_jobs=sum(1 for record in records if not record.attempts),
    )


def _internal_results(records: Sequence[JobRecord]) -> tuple[CaseResult, ...]:
    """Every job as the eval core's rep, whose *attempts* are the engine's private ones.

    This is the whole reuse: a job's internal attempts already have the shape
    :class:`~noctis.eval.metrics.CaseResult` takes, so the within-case-first rules, the
    distribution and the spend arithmetic apply unchanged, one level lower than the runner sees.
    """
    return tuple(
        CaseResult(
            case_id=record.case_id,
            attempts=tuple(_outcome(attempt) for attempt in record.attempts),
        )
        for record in records
    )


def _outcome(attempt: AttemptRecord) -> AttemptOutcome:
    """One internal attempt as the metrics module's input — an unreported field stays absent.

    ``seconds`` is deliberately never set: the engine times the whole job, not each attempt, so
    claiming a per-attempt duration would be inventing a measurement (see :func:`_job_results`).
    """
    return AttemptOutcome(
        passed=attempt.passed,
        model=attempt.model,
        error=attempt.error,
        **{name: int(attempt.usage[name]) for name in USAGE_FIELDS if name in attempt.usage},
    )


def _job_results(records: Sequence[JobRecord]) -> tuple[CaseResult, ...]:
    """Every job as one attempt — the level the clock really ran at.

    The coder-specific half of the latency reading, and the reason it is coder-specific: the
    engine records one duration per authoring job, so the percentile sample is the passing *jobs*.
    A job that never asked is left out rather than entered as an unrun rep.
    """
    return tuple(
        CaseResult(
            case_id=record.case_id,
            attempts=(
                AttemptOutcome(passed=record.passed, model=record.model, seconds=record.seconds),
            ),
        )
        for record in records
        if record.attempts
    )


def _seconds_total(records: Sequence[JobRecord]) -> float | None:
    """The wall clock the whole batch spent, or ``None`` when any job left it unrecorded.

    All-or-nothing, the same posture a partial bill takes: a total over a silently shortened
    sample is a confidently wrong number.
    """
    timed = [record.seconds for record in records if record.attempts]
    if not timed or any(seconds is None for seconds in timed):
        return None
    return round(sum(seconds for seconds in timed if seconds is not None), 6)


def _escalation(records: Sequence[JobRecord], table: PriceTable) -> EscalationReading:
    """What the fallback rescued, and what a rescued file cost — see :class:`EscalationReading`."""
    escalated = [record for record in records if record.escalated]
    rescued = [record for record in escalated if _was_rescued(record)]
    spent = table.total_usd(
        (attempt.model or "", dict(attempt.usage))
        for record in escalated
        for attempt in record.attempts
    )
    return EscalationReading(
        escalated_jobs=len(escalated),
        rescued_jobs=len(rescued),
        rescue_rate=_share(len(rescued), len(escalated)),
        usd_per_rescued_file_estimate=(
            round(spent / len(rescued), 6) if spent is not None and rescued else None
        ),
        pricing_table_version=table.version,
    )


def _was_rescued(record: JobRecord) -> bool:
    """Whether the *fallback* is what landed this job's file, rather than the local coder."""
    return any(attempt.escalated and attempt.passed for attempt in record.attempts)


def _failures(records: Sequence[JobRecord]) -> Breakdown:
    """Every failed internal attempt, counted into the coder failure vocabulary.

    The counting is the failure census's own (:func:`
    noctis.eval.failed_attempts.breakdown_of_errors`), so a bench reading and a run's folder census
    can never disagree about what a given gate error is.
    """
    from noctis.eval.failed_attempts import breakdown_of_errors

    return breakdown_of_errors(
        attempt.error or ""
        for record in records
        for attempt in record.attempts
        if not attempt.passed
    )


def _share(part: int, whole: int) -> float | None:
    """A ratio, or ``None`` for an empty denominator — the guard, stated once."""
    return part / whole if whole else None


# ── the reading a record quotes ───────────────────────────────────────────────────────────


def coder_reading(
    metrics: CoderMetrics,
    *,
    attempt_calls: int = 0,
    warnings: Sequence[Mapping[str, Any]] = (),
    unreadable: int = 0,
    strata: Mapping[str, Any] | None = None,
) -> SiteReading:
    """One scored batch as the site reading a record quotes — the sections, and where each rides.

    The pair whole, the cost beside it, no blended figure. ``warnings`` are the detector rows,
    carried verbatim and read by nothing above them; ``unreadable`` counts the jobs whose retained
    output was not a job record at all; ``strata`` is the per-axis breakdown of the pair
    (:func:`strata_block`), empty for a caller that computed none — a batch nobody labelled has no
    axes, which is an absence and not a zero.

    What each section *is* belongs to :class:`~noctis.eval.reading.SiteReading`, which publishes
    them (#308); what this function decides is which section each of the coder's figures is filed
    under. Two of those choices are the coder's published order rather than anyone's taste: the
    batch opens with the population its rates are over, above the pair, and its per-axis breakdown
    sits between the blocks that explain the two rates and the warning rows beneath them — so the
    extras name the strata's slot there, and it is filled from the field.
    """
    axes = dict(strata or {})
    return SiteReading(
        dials_key=CODER_DIALS_KEY,
        headline={ANSWERS_KEY: ANSWERS_FRESH, ATTEMPT_CALLS_KEY: attempt_calls},
        pair=Pair(key=RATES_KEY, manifest=PASS_RATES, value=metrics.rates),
        counts={
            PASS_LABEL_KEY: FEEDBACK_LABEL,
            "cases": metrics.rates.cases,
            "jobs": metrics.rates.jobs,
            UNATTEMPTED_KEY: metrics.rates.unattempted_jobs,
            UNREADABLE_KEY: unreadable,
        },
        strata=axes,
        extras={
            "effort": _effort_block(metrics),
            "escalation": _escalation_block(metrics.escalation),
            "cost": _cost_block(metrics),
            "failures": _failures_block(metrics.failures),
            STRATA_KEY: axes,
            "warnings": [dict(row) for row in warnings],
            "warned_jobs": len({(row["case_id"], row["rep"]) for row in warnings}),
        },
    )


def coder_block(
    metrics: CoderMetrics,
    *,
    warnings: Sequence[Mapping[str, Any]] = (),
    unreadable: int = 0,
    strata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One scored batch as record data — the block half of :func:`coder_reading`.

    The headline facts above the block (how the bench came by its answers, how many model calls
    sit behind them) are the *scoring pass*'s to state, so a caller that wants the batch's figures
    alone reads them here.
    """
    return coder_reading(metrics, warnings=warnings, unreadable=unreadable, strata=strata).block()


def strata_block(
    cases: Sequence[Case],
    records: Sequence[JobRecord],
    *,
    axes: Sequence[str] = CODER_AXES,
) -> dict[str, Any]:
    """Each difficulty axis's levels, each carrying the co-primary pair over just those jobs.

    Stratified pass rates are the reason the axes exist: a job pass rate that is one thing on
    ``bars_only`` briefs and another on ``exits`` ones is two findings, not one. The arithmetic is
    the whole batch's own (:func:`_pass_rates` over the same internal attempts), so a level's pair
    and the headline pair can never be computed two different ways — DECIDE's
    :func:`~noctis.eval.decide_site.strata_block` is stratified the same way for the same reason.

    The grouping itself is :func:`~noctis.eval.reading.strata`'s, the one loop both sites stratify
    through: an absent level is :data:`~noctis.eval.reading.NOT_APPLICABLE`, levels come back
    sorted, and a level no case in the batch carries is absent rather than present at zero —
    nothing was measured there, and an empty row would read as a measurement. What this function
    keeps is the coder's own two contributions: how a job finds its level, and what a level scores
    (:func:`_stratum_block`).

    What a stratum publishes is deliberately the **pass breakdown** and not a second copy of the
    whole reading: the spend, effort and taxonomy blocks repeated across twenty levels would be a
    wall of numbers nobody reads, while the pair is exactly what an axis was invented to split. Both
    halves ride together (:func:`_rates_block`), so the pairing rule holds at every depth.

    ``axes`` are the site's declared :attr:`~noctis.eval.site.AgentSite.difficulty_axes`, which a
    bench's runner hands the scoring pass; a caller that names none reads the coder's whole
    vocabulary, because that is what a coder corpus labels its briefs on.
    """
    levels = {case.case_id: dict(case.difficulty) for case in cases}
    return strata(
        axes,
        records,
        level_of=lambda record, axis: levels.get(record.case_id, {}).get(axis),
        block_of=_stratum_block,
    )


def _stratum_block(records: Sequence[JobRecord]) -> dict[str, Any]:
    """One level's jobs, as the pass breakdown the reading lists under its axis."""
    rates = _pass_rates(records, _internal_results(records))
    return {
        "cases": rates.cases,
        "jobs": rates.jobs,
        UNATTEMPTED_KEY: rates.unattempted_jobs,
        "rates": _rates_block(rates),
    }


def _rates_block(rates: PassRates) -> dict[str, Any]:
    """The co-primary value, whole: neither rate is published without the other beside it.

    The keys and their order are :data:`~noctis.eval.reading.PASS_RATES`', not restated here — the
    same declaration the pair renders itself from, so a published document and a printed report can
    never name the figures two different things.
    """
    return PASS_RATES.block(rates)


def _effort_block(metrics: CoderMetrics) -> dict[str, Any]:
    """What the gap between the two rates cost in attempts: the mean, the shape, the margin."""
    distribution = metrics.attempt_distribution
    return {
        PASS_LABEL_KEY: FEEDBACK_LABEL,
        "mean_attempts_to_pass": metrics.mean_attempts_to_pass,
        "attempt_distribution": (
            None
            if distribution is None
            else {
                "passed_on": {str(index): share for index, share in distribution.passed_on.items()},
                "never": distribution.never,
            }
        ),
        "retry_yield": (
            None
            if metrics.retry_yield is None
            else {str(index): share for index, share in metrics.retry_yield.items()}
        ),
    }


def _escalation_block(escalation: EscalationReading) -> dict[str, Any]:
    """What the paid fallback rescued, and what each rescued file is estimated to have cost."""
    return {
        PASS_LABEL_KEY: FEEDBACK_LABEL,
        "escalated_jobs": escalation.escalated_jobs,
        "rescued_jobs": escalation.rescued_jobs,
        "escalation_rescue_rate": escalation.rescue_rate,
        "usd_per_rescued_file_estimate": escalation.usd_per_rescued_file_estimate,
        "pricing_table_version": escalation.pricing_table_version,
    }


def _cost_block(metrics: CoderMetrics) -> dict[str, Any]:
    """Spend beside the rates rather than folded into them — the Pareto view is the numbers."""
    return {
        PASS_LABEL_KEY: FEEDBACK_LABEL,
        "tokens_total": metrics.cost.tokens_total,
        "tokens_per_pass": metrics.cost.tokens_per_pass,
        "usd_total_estimate": metrics.cost.usd_total_estimate,
        "usd_per_pass_estimate": metrics.cost.usd_per_pass_estimate,
        "pricing_table_version": metrics.cost.pricing_table_version,
        "seconds_total": metrics.seconds_total,
        "p50_seconds_to_pass": metrics.latency.p50_seconds,
        "p95_seconds_to_pass": metrics.latency.p95_seconds,
    }


def _failures_block(breakdown: Breakdown) -> dict[str, Any]:
    """Every declared failure class with its knob — the escape hatch on the screen at zero.

    Keyed by class name, in the vocabulary's own precedence order, and *every* class is present
    even at zero: a class that stopped happening and a class that stopped matching look identical
    the moment a row disappears, and telling those apart is what the vocabulary is for. A mapping
    rather than a list of rows because a mapping is what a generic reader renders as a table — the
    knob a share points at is the actionable half, and it should be on the screen beside it.

    A share over *no* failures is ``None`` rather than ``0.0``: nothing failed, so no class holds
    any share of anything, and a screen full of confident zeroes would read as a measurement.
    """
    return {
        "attempts": breakdown.total,
        "classes": {
            tally.name: {
                "count": tally.count,
                "share": tally.share if breakdown.total else None,
                "knob": tally.knob,
            }
            for tally in breakdown.tallies
        },
    }


def job_records(answered: Sequence[AnsweredCase]) -> tuple[tuple[JobRecord, int], ...]:
    """Every answered job's retained record, paired with the rep that produced it.

    The job's settled reply *is* its record — :meth:`~noctis.eval.coder_site.JobRecord.attempt`
    puts the document in the attempt's output — so reading a bench is a parse, never a re-run. A
    reply that is not one is skipped here and counted as unreadable by the caller: a job nobody can
    read decides nothing, and inventing an empty record for it would deflate every rate.
    """
    from noctis.eval.coder_site import job_record

    read: list[tuple[JobRecord, int]] = []
    for one in answered:
        document = _document(one.settled)
        if document is None:
            continue
        read.append((job_record(document), one.rep))
    return tuple(read)


def _document(reply: str | None) -> Mapping[str, Any] | None:
    """One retained reply as the job document it should be, or ``None`` when it is not one."""
    if not reply:
        return None
    try:
        parsed = json.loads(reply)
    except ValueError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _warning_rows(read: Iterable[tuple[JobRecord, int]]) -> tuple[dict[str, Any], ...]:
    """Every detector finding a job drew, as the rows the reading lists them in.

    Copied out of the records and read by nothing else in this module — which is what makes the
    inertness claim checkable rather than promised.
    """
    rows = [
        {
            "case_id": record.case_id,
            "rep": rep,
            "strategy": record.strategy,
            "detector": finding.detector,
            "severity": finding.severity,
            "summary": finding.summary,
        }
        for record, rep in read
        for finding in record.findings
    ]
    return tuple(sorted(rows, key=lambda row: (row["case_id"], row["rep"], row["detector"])))


@dataclass(frozen=True)
class CoderReadingScorer:
    """The coder site's scorer: every retained job record, folded into one reading.

    Stateless and pure over the answered cases — a scored batch is reproducible from the corpus and
    the retained job records alone, which is what makes a published pass rate checkable. The runner
    hands it every answered case and folds the block it returns into the record's dials.
    """

    def read(
        self, answered: Sequence[AnsweredCase], *, axes: tuple[str, ...] = CODER_AXES
    ) -> SiteReading | None:
        """The whole coder reading over one live bench's answers — see the module docstring.

        ``axes`` is what the site's declaration says its cases are labelled on, handed over by the
        runner (:class:`~noctis.eval.site.Scorer`) rather than looked up here: this scorer is the
        singleton that declaration carries, so it cannot import it back.

        ``None`` for a bench that answered nothing: a reading over no answers is not a measured
        zero, it is an absence, and publishing empty figures beside real dials would read as one.

        **The population is jobs, and the reps are not folded — on purpose.** DECIDE folds a case's
        reps into one contribution by strict majority (:func:`~noctis.eval.reading.fold_by_case`),
        because a case has one right verdict and several answers to it. A coder brief has no such
        thing: every rep is a real authoring *job* that really asked, really spent and really landed
        or did not, and the pass rates are rates over those jobs. There is no verdict vocabulary
        over an authored file to take a majority of, so a fold here would blur the spread the two
        rates exist to report rather than settle anything (#305).
        """
        if not answered:
            return None
        read = job_records(answered)
        records = [record for record, _ in read]
        return coder_reading(
            score_coder_jobs(records),
            attempt_calls=sum(len(one.replies) for one in answered),
            warnings=_warning_rows(read),
            unreadable=len(answered) - len(read),
            strata=strata_block([one.case for one in answered], records, axes=axes),
        )


#: The scorer the coder declaration carries in its ``scorers`` slot. One instance: it is stateless,
#: and a per-bench copy would invite per-bench configuration of a thing that has none.
CODER_SCORER = CoderReadingScorer()


# Read once at import so the pairing rule is a property of the type rather than a comment: the
# co-primary value must carry both rates, or this module refuses to load at all.
_RATE_FIELDS = {field.name for field in fields(PassRates)}
if not set(PassRates.rate_fields()) <= _RATE_FIELDS:  # pragma: no cover — a structural assertion
    raise RuntimeError("the co-primary pair carries both pass rates, or neither")

# And no third figure may fold them together: a blended score is exactly the number this site's
# reading refuses to publish, so its absence is asserted on the type rather than reviewed for.
if _RATE_FIELDS & {"score", "blended_score", "combined_score"}:  # pragma: no cover — structural
    raise RuntimeError("the coder reading publishes two co-primary rates and no blended score")

# And the manifest that renders the pair may only name figures the pair really carries: a row
# reading an attribute that moved would fail one rendering at a time, months later, in whichever
# report ran first. It fails here instead, at import, for everybody at once.
if not PASS_RATES.attributes() <= _RATE_FIELDS:  # pragma: no cover — a structural assertion
    raise RuntimeError("the pass-rate manifest names a figure the co-primary pair does not carry")
