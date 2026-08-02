"""The bench metrics — pass rates, effort, spend and wall clock, computed from values alone.

A benchmark result is a pile of attempts; this module is the arithmetic that turns that pile into
the handful of numbers an operator compares two prompt compositions on. It is **pure**: no file, no
clock, no seed, no configuration. Everything it needs arrives as data (:class:`AttemptOutcome`,
:class:`CaseResult`) and everything it returns is a frozen record, which is what lets any published
figure be reproduced from a fixture with a pencil. The runner (a later story) collects the inputs;
the input types are deliberately dumb, so collecting them requires no judgment.

**Within-case-first, because a case is the unit of difficulty.** A bench runs each case some number
of times (*reps*), and each rep may retry (*attempts*). Reps aggregate **inside** a case before
cases average, so every case carries equal weight and a case that happened to be run twenty times
cannot outvote nineteen cases run once. That ordering is the Artificial-Analysis convention and it
is not cosmetic: pooling every rep into one ratio silently re-weights the benchmark toward whatever
was repeated most, which is usually whatever was flaky. Two rates come out of it — the
**first-attempt** pass rate (did the site get it right without help?) and the **job-level** pass
rate (did it get there at all, retries allowed?) — and the gap between them is what the effort
metrics below explain.

**Effort.** :func:`mean_attempts_to_pass` averages, over the cases that ever passed, how many
attempts that took. :func:`attempt_distribution` is the shape behind that mean: the equal-weighted
share of cases settled on attempt 1, 2, … and the share that never settled at all. Shares rather
than counts, because a count over reps would re-introduce exactly the weighting the within-case
rule exists to remove. :func:`retry_yield` reads the same distribution as a decision: what the
*second* attempt bought over the first, the third over the second — the marginal pass rate each
extra attempt is paying for.

**Spend is the pricing module's, verbatim.** Dollars come from
:mod:`noctis.research.pricing` — the one place in the repo that turns tokens into money — and this
module inherits its honesty rules rather than restating them: usage is the same four neutral fields
billed at four different rates, a model the table does not carry prices to ``None`` and **poisons**
any total it enters, "free" is a stated price on an explicit allowlist and never an inference from
an absent one, and the table's version label is stamped onto every figure it produced
(:attr:`CostMetrics.pricing_table_version`). Failed attempts are charged to the pass they bought:
tokens-per-pass counts the whole spend, retries included, which is the number a retry budget is
actually argued from. Unlike the rates, the spend totals are **pooled** rather than equal-weighted —
a sum is an absolute, and running a case twice really did cost twice.

**Wall clock.** :func:`latency_metrics` reports p50 and p95 of the seconds it took to *reach a
pass*, summing the failed attempts that preceded it. Percentiles do not average, so the sample is
the passing reps themselves; and because a percentile computed over a silently shortened sample is
a confidently wrong number, one unrecorded duration leaves both figures ``None`` rather than
quietly dropping its rep.

**Absences are honest.** Every ratio is denominator-guarded and every metric whose inputs are
missing is ``None`` — rendered ``n/a`` by consumers, never a fabricated zero. The distinction worth
holding on to: a *measured* zero (nothing passed, out of attempts that really ran) is a real
finding and stays ``0.0``; a *missing* input (a rep with no attempts recorded, a pass rate with no
cases, a per-pass ratio with no passes) is ``None``. An empty result set therefore yields a
None-carrying record, not an exception.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from noctis.research.pricing import USAGE_FIELDS, PriceTable, default_table

__all__ = [
    "AttemptDistribution",
    "AttemptOutcome",
    "BenchMetrics",
    "CaseResult",
    "CostMetrics",
    "LatencyMetrics",
    "attempt_distribution",
    "compute_metrics",
    "cost_metrics",
    "first_attempt_pass_rate",
    "job_pass_rate",
    "latency_metrics",
    "mean_attempts_to_pass",
    "retry_yield",
]


# ── the inputs: dumb, frozen data the runner collects ─────────────────────────────────────
@dataclass(frozen=True)
class AttemptOutcome:
    """One LLM attempt at one case: did it pass, what did it spend, how long did it take.

    The four token fields are the pricing module's own usage fields, spelled identically so a
    recorded attempt can be handed to a price table without translation. Each is ``None`` when the
    seam did not report it — unknown, never zero — and an attempt missing any of them is
    *unpriceable* rather than billed at a blended guess. ``model`` is per attempt because a retry
    may be re-issued to a different model; an attempt that never recorded one cannot be priced
    either. ``error`` is free text kept for the record's benefit; nothing here reads it.
    """

    passed: bool
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    seconds: float | None = None
    error: str | None = None

    def usage(self) -> Mapping[str, int]:
        """The usage fields that were actually recorded, keyed as the price table expects.

        Absent fields are *omitted* rather than zeroed, which is what makes the price table refuse
        the attempt (:meth:`~noctis.research.pricing.ModelPrice.usd_for` requires all four).
        """
        recorded = {name: getattr(self, name) for name in USAGE_FIELDS}
        return {name: value for name, value in recorded.items() if value is not None}

    @property
    def tokens(self) -> int | None:
        """This attempt's total tokens, or ``None`` when any of the four fields went unrecorded."""
        counted = self.usage()
        if len(counted) < len(USAGE_FIELDS):
            return None
        return sum(counted.values())


@dataclass(frozen=True)
class CaseResult:
    """One **rep**: one run of one case, and the attempts it took (retries in order).

    Several reps of the same case share a ``case_id`` — that is how a bench repeats a case, and why
    every rate in this module aggregates within a case before averaging across cases. ``attempts``
    is empty for a rep that never ran, which is missing input rather than a failure.
    """

    case_id: str
    attempts: tuple[AttemptOutcome, ...] = ()

    @property
    def ran(self) -> bool:
        """Whether this rep recorded any attempt at all — the guard every rate honours."""
        return bool(self.attempts)

    @property
    def first_attempt_passed(self) -> bool | None:
        """Whether the opening attempt passed, or ``None`` for a rep that never ran."""
        return self.attempts[0].passed if self.ran else None

    @property
    def passed(self) -> bool | None:
        """Whether any attempt passed, or ``None`` for a rep that never ran."""
        return any(attempt.passed for attempt in self.attempts) if self.ran else None

    @property
    def attempts_to_pass(self) -> int | None:
        """The 1-based index of the attempt that passed, or ``None`` if none ever did."""
        for index, attempt in enumerate(self.attempts, start=1):
            if attempt.passed:
                return index
        return None

    @property
    def seconds_to_pass(self) -> float | None:
        """Wall clock to reach the pass, failed attempts included — ``None`` if unknown.

        Unknown covers both "never passed" and "an attempt on the way there recorded no duration":
        a partial sum presented as a latency is the same confidently false number a partial bill is.
        """
        reached = self.attempts_to_pass
        if reached is None:
            return None
        spent = 0.0
        for attempt in self.attempts[:reached]:
            if attempt.seconds is None:
                return None
            spent += attempt.seconds
        return spent


# ── the outputs ───────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AttemptDistribution:
    """Where cases settled: the equal-weighted share that passed on each attempt index, and the
    share that never passed. ``passed_on`` carries every attempt index the bench offered — an index
    nobody passed on is a stated ``0.0``, not a missing key — and the shares plus ``never`` sum to
    one."""

    passed_on: Mapping[int, float]
    never: float


@dataclass(frozen=True)
class CostMetrics:
    """What the bench spent, in tokens and in estimated dollars, under one named price table.

    A ``None`` dollar figure means the table could not price something the bench spent, and it
    poisons the total by design (see :meth:`~noctis.research.pricing.PriceTable.total_usd`); a
    ``None`` token figure means an attempt never reported its usage split. ``*_per_pass`` is
    ``None`` when nothing passed — there is no per-pass figure without a pass.
    ``pricing_table_version`` travels with the numbers so a later reader knows which prices were in
    force, exactly as it does in the run record.
    """

    tokens_total: int | None
    tokens_per_pass: float | None
    usd_total_estimate: float | None
    usd_per_pass_estimate: float | None
    pricing_table_version: str


@dataclass(frozen=True)
class LatencyMetrics:
    """Wall clock to reach a pass, as p50 and p95 over the passing reps. Both are ``None`` when no
    rep passed or when any passing rep left a duration unrecorded."""

    p50_seconds: float | None
    p95_seconds: float | None


@dataclass(frozen=True)
class BenchMetrics:
    """Every family in one frozen record — what :func:`compute_metrics` returns.

    ``cases`` counts distinct case ids and ``runs`` counts reps, so a reader can tell at a glance
    how much repetition sits behind the rates.
    """

    cases: int
    runs: int
    first_attempt_pass_rate: float | None
    job_pass_rate: float | None
    mean_attempts_to_pass: float | None
    attempt_distribution: AttemptDistribution | None
    retry_yield: Mapping[int, float] | None
    cost: CostMetrics
    latency: LatencyMetrics


# ── within-case-first aggregation ─────────────────────────────────────────────────────────
# One rep's contribution to one metric, or ``None`` when this rep has nothing to say about it (it
# never ran; it never passed, for a metric only passes define). The ``None`` is what keeps a
# missing input out of an average instead of entering it as a zero.
RepValue = Callable[["CaseResult"], float | None]


def _mean(values: Sequence[float]) -> float | None:
    """The mean, or ``None`` for an empty sample — the denominator guard, stated once."""
    return sum(values) / len(values) if values else None


def _by_case(results: Sequence[CaseResult]) -> Mapping[str, tuple[CaseResult, ...]]:
    """The reps grouped by case id, in first-seen order (the equal-weight unit is a case)."""
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result)
    return {case_id: tuple(reps) for case_id, reps in grouped.items()}


def _per_case_mean(results: Sequence[CaseResult], value_of: RepValue) -> float | None:
    """The within-case-first average: reps average inside each case, then the cases average.

    This one function is the ordering rule the whole module rests on. A case whose reps all decline
    to contribute is skipped rather than counted as zero, and both averages are guarded, so an
    empty bench returns ``None`` instead of dividing.
    """
    per_case = []
    for reps in _by_case(results).values():
        contributed = [value for rep in reps if (value := value_of(rep)) is not None]
        case_mean = _mean(contributed)
        if case_mean is not None:
            per_case.append(case_mean)
    return _mean(per_case)


def _flag(value: bool | None) -> float | None:
    """A yes/no rep contribution as a number, keeping "did not run" out of the average."""
    return None if value is None else float(value)


def first_attempt_pass_rate(results: Sequence[CaseResult]) -> float | None:
    """The share of cases whose opening attempt passed — reps averaged inside each case first.

    ``None`` when no case recorded a single attempt: an unrun bench has no rate, it has no data.
    """
    return _per_case_mean(results, lambda rep: _flag(rep.first_attempt_passed))


def job_pass_rate(results: Sequence[CaseResult]) -> float | None:
    """The share of cases that passed on *some* attempt — reps averaged inside each case first."""
    return _per_case_mean(results, lambda rep: _flag(rep.passed))


def mean_attempts_to_pass(results: Sequence[CaseResult]) -> float | None:
    """Mean attempts a pass took, over the cases that ever passed — cases equally weighted.

    A case that never passed contributes nothing (it has no attempts-to-pass) rather than
    contributing its attempt count as though the attempts had worked. ``None`` when nothing passed.
    """
    return _per_case_mean(
        results,
        lambda rep: None if rep.attempts_to_pass is None else float(rep.attempts_to_pass),
    )


def attempt_distribution(results: Sequence[CaseResult]) -> AttemptDistribution | None:
    """Where the cases settled, equal-weighted: share passing on each attempt index, share never.

    ``None`` when no rep ran. The index range covers every attempt the bench actually offered, so
    an attempt nobody converted reads as a stated zero rather than an absent key.
    """
    ran = [rep for rep in results if rep.ran]
    if not ran:
        return None

    def settled_at(index: int) -> RepValue:
        return lambda rep: _flag(rep.attempts_to_pass == index) if rep.ran else None

    depth = max(len(rep.attempts) for rep in ran)
    shares = {
        index: _per_case_mean(results, settled_at(index)) or 0.0 for index in range(1, depth + 1)
    }
    never = _per_case_mean(
        results, lambda rep: _flag(rep.attempts_to_pass is None) if rep.ran else None
    )
    return AttemptDistribution(passed_on=MappingProxyType(shares), never=never or 0.0)


def retry_yield(results: Sequence[CaseResult]) -> Mapping[int, float] | None:
    """What each retry bought: the marginal pass rate attempt 2, 3, … added over its predecessor.

    Attempt 1's "yield" is the first-attempt pass rate itself, so the mapping starts at 2. It is
    empty when no case was ever attempted twice, and ``None`` when no rep ran at all.
    """
    distribution = attempt_distribution(results)
    if distribution is None:
        return None
    return MappingProxyType(
        {index: share for index, share in distribution.passed_on.items() if index >= 2}
    )


# ── cost, through the pricing module ──────────────────────────────────────────────────────
def cost_metrics(results: Sequence[CaseResult], table: PriceTable | None = None) -> CostMetrics:
    """Tokens and estimated dollars, in total and per pass, under ``table`` (the shipped one by
    default).

    Every rule here is the pricing module's, applied rather than restated: the bill is
    all-or-nothing (one unpriceable attempt makes the total ``None``), an unrecorded usage split
    makes the token total unknown for the same reason, and the table's version label rides along on
    the result. The denominator is the reps that reached a pass — retries are charged to the pass
    they bought — and it is guarded, so a bench that passed nothing reports no per-pass figure
    rather than a division.
    """
    priced = table or default_table()
    attempts = [attempt for result in results for attempt in result.attempts]

    counted = [attempt.tokens for attempt in attempts]
    known = [count for count in counted if count is not None]
    tokens_total = sum(known) if len(known) == len(counted) else None
    usd_total = priced.total_usd((attempt.model or "", attempt.usage()) for attempt in attempts)

    passes = sum(1 for result in results if result.passed)
    return CostMetrics(
        tokens_total=tokens_total,
        tokens_per_pass=tokens_total / passes if tokens_total is not None and passes else None,
        usd_total_estimate=usd_total,
        usd_per_pass_estimate=(
            round(usd_total / passes, 6) if usd_total is not None and passes else None
        ),
        pricing_table_version=priced.version,
    )


# ── wall clock ────────────────────────────────────────────────────────────────────────────
def _percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile by linear interpolation between the two neighbouring ranks.

    Stated explicitly because a percentile is only reproducible if its definition is: the sample is
    sorted, the position is ``q * (n - 1)``, and a fractional position interpolates linearly
    between the ranks it falls between. Empty sample, no percentile.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def latency_metrics(results: Sequence[CaseResult]) -> LatencyMetrics:
    """p50 and p95 of the wall clock it took to reach a pass, over the reps that reached one.

    A rep that never passed has no per-pass latency and is not in the sample. A passing rep whose
    duration went unrecorded makes both figures ``None``: dropping it would publish a percentile
    over a sample the reader has no way to know was shortened.
    """
    reached = [result.seconds_to_pass for result in results if result.passed]
    if any(seconds is None for seconds in reached):
        return LatencyMetrics(None, None)
    sample = [seconds for seconds in reached if seconds is not None]
    return LatencyMetrics(
        p50_seconds=_percentile(sample, 0.50), p95_seconds=_percentile(sample, 0.95)
    )


# ── the whole record ──────────────────────────────────────────────────────────────────────
def compute_metrics(results: Sequence[CaseResult], table: PriceTable | None = None) -> BenchMetrics:
    """Every metric family for one bench run, in one frozen record.

    Total by construction: an empty result set yields a record whose every rate is ``None`` and
    whose counts are honest zeros, because there is nothing here that can divide by nothing.
    """
    return BenchMetrics(
        cases=len(_by_case(results)),
        runs=len(results),
        first_attempt_pass_rate=first_attempt_pass_rate(results),
        job_pass_rate=job_pass_rate(results),
        mean_attempts_to_pass=mean_attempts_to_pass(results),
        attempt_distribution=attempt_distribution(results),
        retry_yield=retry_yield(results),
        cost=cost_metrics(results, table),
        latency=latency_metrics(results),
    )
