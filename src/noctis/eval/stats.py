"""Paired statistics — is the difference between two configurations real, or is it noise?

A benchmark run yields, for each case, whether configuration A passed and whether configuration B
passed. The interesting number is the *difference*, and the only honest question about a difference
is whether it survives the noise floor of an LLM that answers the same prompt differently twice.
This module is the arithmetic for that question, and — as much as the arithmetic — the **refusal**
around it: a comparison it cannot qualify returns a :class:`Refusal` naming the reason, never a
number a reader might quote.

**The inputs are paired, per case, over the same cases.** A configuration's results arrive as a
mapping from case id to *passes out of ``reps``* — ``0``/``1`` for a single-rep run, ``0..k`` when
each case was run ``k`` times. The pairing is the whole point: both configurations face the same
cases, so the per-case delta ``(passes_b - passes_a) / reps`` cancels the case's own difficulty,
which is by far the largest source of variance in an agent benchmark. Two mappings whose case sets
differ are therefore a refusal and never a silent intersection — an intersection would quietly
change the population a p-value is about.

**Which test, and why it is chosen for you.** :func:`compare` selects on ``reps``, and the result
says which it used:

* ``reps == 1`` → **McNemar's exact test** on the discordant pairs. With one binary outcome per case
  the concordant cases carry no information about a difference, and the exact conditional binomial
  over the ``b + c`` discordant pairs is the textbook answer — cheap, deterministic, hand-checkable.
  The variant implemented here is the **two-sided exact conditional binomial**:
  ``min(1, 2 * P(X <= min(b, c)))`` for ``X ~ Binomial(b + c, 1/2)``, where ``b`` counts cases only
  A passed and ``c`` cases only B passed. (Because that binomial is symmetric, doubling the smaller
  tail names exactly the set of outcomes at least as extreme as the observed one; the ``min`` caps
  the doubling when the split is even.) No continuity-corrected chi-square approximation: discordant
  counts in an agent benchmark are small, which is precisely where the approximation is worst.
  A caller holding ``reps > 1`` results may collapse them by majority vote and read McNemar as a
  quick check — :func:`mcnemar_exact_p` takes the two counts directly — but the primary test on
  repeated runs is the one below.

* ``reps > 1`` → the **paired permutation test** on per-case pass proportions. McNemar wants one
  binary outcome per case; reps give a proportion, and throwing that resolution away to satisfy a
  test is the wrong trade. Under the null hypothesis that the two configurations are exchangeable
  on a case, the sign of that case's delta is arbitrary — so the reference distribution is the mean
  delta under every sign flip. Small case sets are enumerated **exhaustively** (all ``2**n`` sign
  vectors, an exact p-value); larger ones are **sampled** with the standard ``(1 + extreme) /
  (1 + permutations)`` estimator, which never reports the impossible p-value of zero.

**Everything random is seeded, explicitly, by the caller.** ``seed`` is a required keyword-only
argument of every entry point that can sample, and the generators are private
:class:`random.Random` instances derived from it by name — the process-global generator is never
touched, so two runs of the same comparison anywhere give the same p-value and the same interval.

**The interval and the noise floor.** :func:`bootstrap_ci` resamples *cases* with replacement
(cases are the sampling unit; resampling reps would measure the wrong variance) and takes a
percentile interval on the delta. :func:`measure_stability_band` is the other half: run one
configuration twice, and the pass-rate gap between those two runs is the observed noise floor. A
comparison whose delta sits inside that band is **unresolvable** — inconclusive, not a win and not a
loss.

**The verdict is tri-state, and it is the parity module's design verbatim** (see
``src/noctis/research/parity.py``, which decides the episodic flip the same way). Each half of the
judgment is ``bool | None``: did B improve on A, is the p-value below ``alpha``, is the delta
outside the noise band. ``None`` means *the input for that half was ``n/a``* — a missing p-value, an
unmeasured stability band — and an ``n/a`` half makes the verdict ``inconclusive`` rather than
deciding it in either direction. Every ratio is denominator-guarded to ``None`` rather than dividing
by zero, and ``None`` becomes the literal ``n/a`` in exactly one place: the formatter in
:func:`render_comparison`.

**Pure.** Values in, values out: no I/O, no clock, no configuration, no import of the engine. The
module is stdlib-only on purpose — the suite asserts it structurally.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "Assessment",
    "Comparison",
    "Refusal",
    "StabilityBand",
    "assess",
    "bootstrap_ci",
    "compare",
    "mcnemar_exact_p",
    "measure_stability_band",
    "paired_permutation",
    "render_comparison",
    "render_result",
]

# The two tests, named on every result so a reader never has to infer which one produced a number.
MCNEMAR = "mcnemar"
PAIRED_PERMUTATION = "paired_permutation"

# The tri-state verdict, following the parity module's flip assessment.
PASS = "pass"
FAIL = "fail"
INCONCLUSIVE = "inconclusive"

# Why a comparison refused. Stable codes: a caller branches on these, a reader reads ``detail``.
NO_CASES = "no_cases"
MISMATCHED_CASES = "mismatched_cases"
MISSING_OUTCOMES = "missing_outcomes"
INVALID_REPS = "invalid_reps"
OUT_OF_RANGE = "out_of_range"
NO_DISCORDANT_PAIRS = "no_discordant_pairs"

DEFAULT_PERMUTATIONS = 10_000
DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05

# Beyond this many cases the exhaustive branch is never even costed (2**n is astronomically past
# any sane permutation budget long before it is expensive to compute).
_MAX_EXHAUSTIVE_CASES = 24

# Sums of floats are not associative, so "at least as extreme as observed" is asked with a slack
# of one part in a trillion — far below any difference a benchmark could mean, far above the
# rounding error of summing a few hundred proportions.
_TOLERANCE = 1e-12

# How many offending case ids a refusal names before it stops listing them.
_MAX_NAMED_CASES = 8

# One configuration's per-case results: case id -> passes out of ``reps``, or ``None`` for a case
# with no honest outcome (an error, a crash, a case that never ran). ``None`` is refused rather
# than dropped: which cases were compared is part of what a comparison means.
CaseOutcomes = Mapping[str, int | None]


@dataclass(frozen=True)
class Refusal:
    """A comparison the module will not qualify: the reason code, and the sentence for a human.

    Returned instead of a :class:`Comparison` — never alongside one, and never carrying a
    statistic, because the entire purpose of a refusal is that there is no number to quote.
    """

    reason: str
    detail: str

    def render(self) -> str:
        """The one line a report prints in place of a comparison table."""
        return f"REFUSED ({self.reason}) — {self.detail}"


@dataclass(frozen=True)
class Comparison:
    """The computed comparison of two configurations over one case set.

    It names its own scope — ``n`` cases, ``reps`` repetitions, the ``test`` that produced
    ``p_value`` — so a number lifted out of it cannot lose the population it is about. Fields that
    a denominator guard can empty are ``float | None`` and render as ``n/a``.
    """

    n: int
    reps: int
    test: str
    rate_a: float
    rate_b: float
    delta: float
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    discordant: int
    permutations: int | None


@dataclass(frozen=True)
class StabilityBand:
    """The run-twice noise floor: the pass-rate gap between two runs of the *same* configuration.

    ``band`` is that gap in absolute value — the smallest delta a comparison of two *different*
    configurations must exceed before it is worth calling a difference at all.
    """

    n: int
    reps: int
    rate_first: float
    rate_second: float
    band: float


@dataclass(frozen=True)
class Assessment:
    """The tri-state judgment of a comparison, and why — the parity module's shape.

    Each half is ``True``/``False``/``None``, where ``None`` is an ``n/a`` input (no p-value, no
    measured band) that makes the verdict ``inconclusive`` instead of deciding it.
    """

    improved: bool | None
    significant: bool | None
    resolvable: bool | None
    alpha: float
    band: float | None
    verdict: str
    summary: str


# ── the statistics, each denominator-guarded ──────────────────────────────────────────────
def mcnemar_exact_p(favouring_a: int, favouring_b: int) -> float | None:
    """The two-sided exact conditional binomial McNemar p-value on discordant counts.

    ``favouring_a`` is ``b`` (cases only A passed), ``favouring_b`` is ``c`` (cases only B passed).
    The p-value is ``min(1, 2 * P(X <= min(b, c)))`` for ``X ~ Binomial(b + c, 1/2)``, computed in
    exact integer arithmetic. ``None`` when there are no discordant pairs at all: the test's
    denominator is empty, and an empty denominator is an ``n/a``, never a 1.0 dressed up as
    agreement.
    """
    if favouring_a < 0 or favouring_b < 0:
        raise ValueError(f"discordant counts cannot be negative (got {favouring_a}, {favouring_b})")
    discordant = favouring_a + favouring_b
    if discordant == 0:
        return None
    smaller = min(favouring_a, favouring_b)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2 * tail / 2**discordant)


def paired_permutation(
    deltas: Sequence[float],
    *,
    seed: int,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> float | None:
    """The two-sided paired (sign-flip) permutation p-value on per-case deltas.

    The statistic is the mean per-case delta; the reference distribution is that mean under sign
    flips, which is what the null hypothesis "the two configurations are exchangeable on each case"
    licenses. Exact by enumeration when ``2**n`` fits inside ``permutations``, otherwise sampled
    from ``random.Random`` seeded off ``seed`` with the ``(1 + extreme) / (1 + permutations)``
    estimator. ``None`` when there is nothing to permute — no cases, or every delta zero.
    """
    values = [float(delta) for delta in deltas]
    n = len(values)
    if n == 0 or not any(value != 0.0 for value in values):
        return None

    observed = abs(math.fsum(values) / n) - _TOLERANCE
    if _is_exhaustive(n, permutations):
        vectors = itertools.product((1.0, -1.0), repeat=n)
        extreme = sum(
            1
            for signs in vectors
            if abs(math.fsum(sign * value for sign, value in zip(signs, values, strict=True)) / n)
            >= observed
        )
        return extreme / 2**n

    generator = random.Random(f"{seed}:paired-permutation")
    extreme = 0
    for _ in range(permutations):
        flipped = math.fsum(value if generator.random() < 0.5 else -value for value in values)
        if abs(flipped / n) >= observed:
            extreme += 1
    return (1 + extreme) / (1 + permutations)


def bootstrap_ci(
    deltas: Sequence[float],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float] | None:
    """A percentile bootstrap interval for the mean per-case delta, resampling **cases**.

    Cases are the sampling unit — the question is how the delta would move on a different draw of
    cases from the same population, not on a different draw of reps within these cases. ``None``
    when there is nothing to resample (no cases, or no resamples asked for).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1 (got {confidence})")
    values = [float(delta) for delta in deltas]
    n = len(values)
    if n == 0 or resamples <= 0:
        return None

    generator = random.Random(f"{seed}:bootstrap")
    means = sorted(math.fsum(generator.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


# ── the two entry points: a comparison, and a noise floor ─────────────────────────────────
def compare(
    a: CaseOutcomes,
    b: CaseOutcomes,
    *,
    seed: int,
    reps: int = 1,
    permutations: int = DEFAULT_PERMUTATIONS,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Comparison | Refusal:
    """Compare configuration ``b`` against configuration ``a`` over one shared case set.

    Returns a :class:`Comparison` naming its ``n``, its ``reps`` and the ``test`` it selected
    (``reps == 1`` → McNemar, ``reps > 1`` → the paired permutation test), or a :class:`Refusal`
    for a comparison that cannot be qualified: a rep count below one, case sets that differ, an
    empty case set, a missing per-case outcome, a pass count outside ``0..reps``, or no discordant
    cases at all — perfect agreement leaves the tests with an empty denominator.
    """
    paired = _pair(a, b, reps)
    if isinstance(paired, Refusal):
        return paired

    n = len(paired.deltas)
    favouring_a = sum(1 for delta in paired.deltas if delta < 0.0)
    favouring_b = sum(1 for delta in paired.deltas if delta > 0.0)
    discordant = favouring_a + favouring_b
    if discordant == 0:
        return Refusal(
            NO_DISCORDANT_PAIRS,
            f"the two configurations agreed on all {n} cases, so there are no discordant pairs "
            "to test — a comparison of identical results is not a result",
        )

    if reps == 1:
        test, p_value, vectors = MCNEMAR, mcnemar_exact_p(favouring_a, favouring_b), None
    else:
        test = PAIRED_PERMUTATION
        p_value = paired_permutation(paired.deltas, seed=seed, permutations=permutations)
        vectors = 2**n if _is_exhaustive(n, permutations) else permutations

    interval = bootstrap_ci(paired.deltas, seed=seed, resamples=resamples, confidence=confidence)
    rate_a = paired.passes_a / (n * reps)
    rate_b = paired.passes_b / (n * reps)
    return Comparison(
        n=n,
        reps=reps,
        test=test,
        rate_a=rate_a,
        rate_b=rate_b,
        delta=rate_b - rate_a,
        p_value=p_value,
        ci_low=None if interval is None else interval[0],
        ci_high=None if interval is None else interval[1],
        confidence=confidence,
        discordant=discordant,
        permutations=vectors,
    )


def measure_stability_band(
    first: CaseOutcomes,
    second: CaseOutcomes,
    *,
    reps: int = 1,
) -> StabilityBand | Refusal:
    """The noise floor from running **one** configuration twice: the gap between the two runs.

    Same qualification as :func:`compare` (the runs must cover the same cases, with an outcome
    each), minus the discordance check — two identical runs are a perfectly honest band of zero,
    not an unqualifiable comparison.
    """
    paired = _pair(first, second, reps)
    if isinstance(paired, Refusal):
        return paired

    n = len(paired.deltas)
    rate_first = paired.passes_a / (n * reps)
    rate_second = paired.passes_b / (n * reps)
    return StabilityBand(
        n=n,
        reps=reps,
        rate_first=rate_first,
        rate_second=rate_second,
        band=abs(rate_second - rate_first),
    )


# ── the tri-state verdict ─────────────────────────────────────────────────────────────────
def assess(
    comparison: Comparison,
    *,
    band: float | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> Assessment:
    """Judge a comparison on three halves: improvement, significance, resolvability.

    ``band`` is a measured noise floor (:attr:`StabilityBand.band`). Leaving it out is not a pass:
    without a noise floor the resolvability half is ``n/a``, and — exactly as an ``n/a``
    tokens/verdict makes the parity flip inconclusive rather than deciding it — the verdict is
    ``inconclusive``. A delta *inside* a measured band is likewise inconclusive: unresolvable noise
    is neither a win nor a loss. Otherwise the verdict is ``pass`` when B improved on A and the
    test called it significant, and ``fail`` when it did not.
    """
    improved = comparison.delta > 0.0
    significant = None if comparison.p_value is None else comparison.p_value < alpha
    resolvable = None if band is None else abs(comparison.delta) > band

    scope = f"{comparison.n} cases x {comparison.reps} reps, {comparison.test}"
    if significant is None or resolvable is None:
        missing = []
        if significant is None:
            missing.append(f"the {comparison.test} p-value is n/a")
        if resolvable is None:
            missing.append("no run-twice stability band was measured, so the noise floor is n/a")
        verdict, summary = (
            INCONCLUSIVE,
            (
                f"INCONCLUSIVE — {'; '.join(missing)} ({scope}). A half nothing qualified decides "
                "nothing: measure it, then re-read this comparison."
            ),
        )
    elif not resolvable:
        verdict, summary = (
            INCONCLUSIVE,
            (
                f"INCONCLUSIVE — the delta ({comparison.delta:+.4f}) sits inside the run-twice "
                f"stability band (+/-{band:.4f}), so it is unresolvable noise ({scope})."
            ),
        )
    elif improved and significant:
        verdict, summary = (
            PASS,
            (
                f"PASS — B beats A by {comparison.delta:+.4f} ({scope}), outside the "
                f"+/-{band:.4f} stability band, at p={comparison.p_value:.4g} < {alpha:g}."
            ),
        )
    else:
        reasons = []
        if not improved:
            reasons.append(f"B did not beat A (delta {comparison.delta:+.4f})")
        if not significant:
            assert comparison.p_value is not None  # significant is False ⇒ a p-value was computed
            reasons.append(f"p={comparison.p_value:.4g} is not below {alpha:g}")
        verdict, summary = FAIL, f"FAIL — {'; '.join(reasons)} ({scope})."

    return Assessment(
        improved=improved,
        significant=significant,
        resolvable=resolvable,
        alpha=alpha,
        band=band,
        verdict=verdict,
        summary=summary,
    )


# ── rendering: the only place a ``None`` becomes a word ───────────────────────────────────
def _fmt(value: object) -> str:
    """A cell: the literal ``n/a`` for ``None``, four decimals for a float, the value otherwise."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_comparison(comparison: Comparison, assessment: Assessment) -> str:
    """A one-page reading of a comparison: its scope on the header, its numbers, its verdict."""
    label_w, value_w = 28, 22
    interval = (
        None
        if comparison.ci_low is None or comparison.ci_high is None
        else f"{comparison.ci_low:.4f} .. {comparison.ci_high:.4f}"
    )
    rows: list[tuple[str, object]] = [
        ("Cases (n)", comparison.n),
        ("Reps per case", comparison.reps),
        ("Discordant cases", comparison.discordant),
        ("Sign vectors", comparison.permutations),
        ("Pass rate A", comparison.rate_a),
        ("Pass rate B", comparison.rate_b),
        ("Delta (B - A)", comparison.delta),
        (f"{comparison.confidence * 100:.0f}% CI (bootstrap)", interval),
        ("p-value", None if comparison.p_value is None else f"{comparison.p_value:.4g}"),
        ("Stability band", assessment.band),
    ]

    lines = [
        f"Paired comparison — n={comparison.n} cases, reps={comparison.reps}, "
        f"test={comparison.test}",
        "",
        f"{'-' * label_w}{'-' * value_w}",
    ]
    lines += [f"{name:<{label_w}}{_fmt(value):>{value_w}}" for name, value in rows]
    lines += ["", assessment.summary]
    return "\n".join(lines)


def render_result(
    result: Comparison | Refusal,
    *,
    band: float | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    """Render whatever :func:`compare` returned — a table for a comparison, one line for a refusal.

    This is where the refusal posture is visible to a reader: an unqualifiable comparison prints
    the reason it was refused, and no statistic at all.
    """
    if isinstance(result, Refusal):
        return result.render()
    return render_comparison(result, assess(result, band=band, alpha=alpha))


# ── qualification: everything a comparison must survive before it is arithmetic ───────────
@dataclass(frozen=True)
class _Paired:
    """Two configurations' results, aligned on sorted case ids: the deltas and the pass totals."""

    deltas: tuple[float, ...]
    passes_a: int
    passes_b: int


def _pair(a: CaseOutcomes, b: CaseOutcomes, reps: int) -> _Paired | Refusal:
    """Align two case-keyed mappings, or refuse. Case ids are sorted, so a mapping's insertion
    order can never change a p-value."""
    if reps < 1:
        return Refusal(INVALID_REPS, f"a comparison needs at least one rep per case (got {reps})")

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        return Refusal(
            MISMATCHED_CASES,
            "the two configurations were measured on different case sets — "
            f"only A has {_names(only_a)}, only B has {_names(only_b)}. A paired test over an "
            "intersection would silently change the population it is about",
        )

    cases = sorted(a)
    if not cases:
        return Refusal(NO_CASES, "no cases were compared, so there is nothing to test")

    missing = [case for case in cases if a[case] is None or b[case] is None]
    if missing:
        return Refusal(
            MISSING_OUTCOMES,
            f"no outcome for {_names(missing)} — a case without a result cannot be dropped "
            "quietly, because which cases were compared is part of what the comparison means",
        )

    out_of_range = [
        case
        for case in cases
        if not (0 <= _passes(a[case]) <= reps and 0 <= _passes(b[case]) <= reps)
    ]
    if out_of_range:
        return Refusal(
            OUT_OF_RANGE,
            f"pass counts outside 0..{reps} for {_names(out_of_range)}",
        )

    return _Paired(
        deltas=tuple((_passes(b[case]) - _passes(a[case])) / reps for case in cases),
        passes_a=sum(_passes(a[case]) for case in cases),
        passes_b=sum(_passes(b[case]) for case in cases),
    )


def _passes(outcome: int | None) -> int:
    """A qualified outcome as an int — reached only after the missing-outcome refusal."""
    assert outcome is not None
    return outcome


def _names(cases: Sequence[str]) -> str:
    """Offending case ids for a refusal sentence, capped so one bad batch stays readable."""
    if not cases:
        return "none"
    shown = ", ".join(cases[:_MAX_NAMED_CASES])
    extra = len(cases) - _MAX_NAMED_CASES
    return shown if extra <= 0 else f"{shown} (+{extra} more)"


def _is_exhaustive(n: int, permutations: int) -> bool:
    """Whether every sign vector fits inside the permutation budget (an exact p-value)."""
    return n <= _MAX_EXHAUSTIVE_CASES and 2**n <= permutations


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """The linearly interpolated quantile of an already-sorted sample."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
