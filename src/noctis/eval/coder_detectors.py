"""Two detectors for a *degenerate* pass — a file that cleared the write gate by collapsing.

The write gate answers one question ("does this file hold together?") and answers it well. It cannot
answer the next one: *how* was the pass won. Two shapes of hollow pass show up often enough to be
worth naming, and both are visible without running a backtest:

* **Param-floor collapse** — the file's declared ``Params`` defaults sit at the very bottom of the
  file's *own* declared :meth:`~noctis.strategies.base.TraderStrategy.param_space`. A lookback of 2
  in a space that offers 2–200 is a search that found the cheapest corner of its own arena rather
  than a thesis; a whole file of those is a shrink-until-it-passes artifact.
* **Trivial-target degeneracy** — the file reaches a directional (long/short) target on a handful of
  bars and is flat everywhere else. It replays, it is deterministic, it honours its scenarios, and
  it decides almost nothing.

**Nothing here is a gate and nothing here is a score.** Both detectors emit ``WARNING`` findings
(:data:`SEVERITY`) that name their evidence and stop there. They are read by a human comparing two
prompt compositions, never by :mod:`noctis.eval.metrics`: no pass rate moves, no bill moves, and no
file is refused because a detector spoke. That inertness is structural rather than promised — the
metrics arithmetic consumes :class:`~noctis.eval.metrics.AttemptOutcome` and
:class:`~noctis.eval.metrics.CaseResult`, neither of which has a field a
:class:`DegenerateFinding` fits in, and this module imports no scoring surface at all.

**The thresholds, and why they are set where they are.** Each is a named constant because a magic
number inside a detector is an opinion nobody argued:

* :data:`FLOOR_BAND` = 10% — a default within 10% of the *bottom of its own declared range* counts
  as sitting on the floor. Measured as a position in the declared range, so it is unit-free and a
  space's own author sets the scale. The number is checked against the shipped seeds: the closest
  any of them comes is ``sma_crossover``'s ``slow=30`` in ``[20, 100]``, at 12.5% — outside the
  band, with room to spare.
* :data:`COLLAPSE_SHARE` = 75% — the share of *ordered tunable dimensions* that must be on the floor
  before a finding is emitted. One knob at its floor is an ordinary design choice (a fast-reacting
  thesis really does want a short window), so a single floored dimension says nothing. A file where
  three quarters of its dimensions bottomed out is a different claim. Near-unanimity is the
  conservative reading, and it keeps the detector quiet on the mixed spaces real strategies declare.
* :data:`MIN_NONFLAT_FRACTION` = 5% — the fraction of *decidable* bars on which the file must reach
  some direction. On the gate's 180-bar fixture that is 9 bars. The shipped seeds land between 34%
  and 58% on that tape, so the floor sits roughly seven times under the least active thing the
  repository ships: it recognises "decides almost nothing", never "is selective".

Only **ordered** dimensions are measured — an ``int``/``float`` row with a real (non-zero-width)
range. A categorical has no bottom to collapse toward and a pinned ``low == high`` row has no
position within itself, so both are left out of the numerator *and* the denominator rather than
counted as clean; a share is only honest over the dimensions it could actually read.

**Warmup is excluded from the activity denominator.** A strategy declares
:meth:`~noctis.strategies.base.TraderStrategy.warmup_bars` and the Tier-1 honesty invariant makes it
keep that promise, so counting those bars as inactivity would punish a long-lookback thesis for
being honest. The measurement is over the bars where a decision was possible.

**The measurement is the gate's own, not a copy of it.** The activity reading replays through
:func:`noctis.strategies.library.fixture_frame` and :func:`noctis.strategies.base.replay_targets` —
the exact tape and the exact replay function ``_validate_file`` uses — because a detector with its
own private notion of what a strategy did on a bar would drift away from the gate in silence.

The core (:func:`read_param_floors`, :func:`read_target_activity`) is **pure**: values in, frozen
records out, no file, no clock, no randomness. :func:`inspect_strategy` is the one thin convenience
that *executes* — it produces the replay the core then reads — and it is kept in its own section
below so the line between the two is impossible to miss.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import Any

from noctis.strategies.base import ParamSpec, TraderStrategy, params_to_dict, replay_targets
from noctis.strategies.library import fixture_frame

__all__ = [
    "COLLAPSE_SHARE",
    "FLOOR_BAND",
    "MIN_NONFLAT_FRACTION",
    "ORDERED_KINDS",
    "PARAM_FLOOR_COLLAPSE",
    "SEVERITY",
    "TRIVIAL_TARGET",
    "DegenerateFinding",
    "DegenerateReport",
    "DimensionReading",
    "ParamFloorReading",
    "TargetActivityReading",
    "inspect_strategy",
    "read_param_floors",
    "read_target_activity",
]

#: A default this close to the bottom of its own declared range counts as sitting on the floor.
FLOOR_BAND: float = 0.10

#: The share of ordered tunable dimensions that must be floored before a finding is emitted.
COLLAPSE_SHARE: float = 0.75

#: The share of decidable bars on which a file must reach *some* direction.
MIN_NONFLAT_FRACTION: float = 0.05

#: Every finding is advisory. There is no other severity, by design: see the module docstring.
SEVERITY: str = "WARNING"

#: The detector names, carried on the findings so a reader never has to infer which one spoke.
PARAM_FLOOR_COLLAPSE: str = "param_floor_collapse"
TRIVIAL_TARGET: str = "trivial_target"

#: The :class:`~noctis.strategies.base.ParamSpec` kinds that have a bottom to collapse toward.
ORDERED_KINDS: frozenset[str] = frozenset({"int", "float"})


# ── findings ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DegenerateFinding:
    """One advisory observation: which detector spoke, at what severity, over what evidence.

    ``summary`` is the whole payload and it is written to be read on its own line — it names the
    measured numbers and the threshold they were judged against, so a finding quoted out of context
    still carries its own argument. There is deliberately no pass field, no score field and no cost
    field here: a finding is evidence, and the moment it carried a number a metric could consume it
    would stop being inert.
    """

    detector: str
    severity: str
    summary: str

    def line(self) -> str:
        """This finding as one screen line, severity first."""
        return f"{self.severity} {self.detector}: {self.summary}"


def _warning(detector: str, summary: str) -> DegenerateFinding:
    return DegenerateFinding(detector=detector, severity=SEVERITY, summary=summary)


# ── param-floor collapse ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DimensionReading:
    """Where one ordered tunable dimension's default sits inside its own declared range.

    ``position`` is 0.0 at the declared floor and 1.0 at the declared ceiling, which is what makes
    a lookback in bars and a threshold in price ratios comparable at all.
    """

    name: str
    value: float
    low: float
    high: float
    position: float

    @property
    def at_floor(self) -> bool:
        """Whether this dimension sits within :data:`FLOOR_BAND` of its declared bottom."""
        return self.position <= FLOOR_BAND

    def line(self) -> str:
        """The evidence for this one dimension: value, position, and the range it was read in."""
        return f"{self.name}={self.value:g} at {self.position:.0%} of [{self.low:g}, {self.high:g}]"


@dataclass(frozen=True)
class ParamFloorReading:
    """Every ordered dimension that could be measured, which of them are floored, and the verdict.

    ``share`` is ``None`` — not ``0.0`` — when the declared space offered nothing ordered to read:
    an unmeasurable space is missing evidence, and reporting it as a clean zero would be a finding
    nobody made.
    """

    dimensions: tuple[DimensionReading, ...]
    floored: tuple[DimensionReading, ...]
    share: float | None
    finding: DegenerateFinding | None


def _defaults(params: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """The declared defaults as a mapping, whether handed a frozen ``Params`` or a plain dict."""
    if isinstance(params, Mapping):
        return params
    if is_dataclass(params) and not isinstance(params, type):
        return params_to_dict(params)
    raise TypeError("params must be a mapping or a dataclass instance of declared defaults")


def _measure(spec: ParamSpec, value: Any) -> DimensionReading | None:
    """One dimension's position, or ``None`` when it has no bottom this detector can read."""
    if spec.kind not in ORDERED_KINDS or spec.low is None or spec.high is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    low, high = float(spec.low), float(spec.high)
    if high <= low:  # a pinned dimension has no position within itself
        return None
    return DimensionReading(
        name=spec.name,
        value=float(value),
        low=low,
        high=high,
        position=(float(value) - low) / (high - low),
    )


def read_param_floors(
    params: Mapping[str, Any] | Any, space: Sequence[ParamSpec]
) -> ParamFloorReading:
    """Read the declared defaults against the declared search space — pure, values only.

    A dimension the space declares but the defaults never name cannot be positioned, so it is left
    out of both halves of the share, exactly as a categorical or a pinned row is.
    """
    declared = _defaults(params)
    dimensions = tuple(
        reading
        for spec in space
        if spec.name in declared and (reading := _measure(spec, declared[spec.name])) is not None
    )
    if not dimensions:
        return ParamFloorReading(dimensions=(), floored=(), share=None, finding=None)

    floored = tuple(reading for reading in dimensions if reading.at_floor)
    share = len(floored) / len(dimensions)
    finding = None
    if share >= COLLAPSE_SHARE:
        evidence = "; ".join(reading.line() for reading in floored)
        finding = _warning(
            PARAM_FLOOR_COLLAPSE,
            f"{len(floored)} of {len(dimensions)} tunable dimensions ({share:.0%}, at or over the "
            f"{COLLAPSE_SHARE:.0%} share) sit within {FLOOR_BAND:.0%} of their declared floor: "
            f"{evidence}",
        )
    return ParamFloorReading(dimensions=dimensions, floored=floored, share=share, finding=finding)


# ── trivial-target degeneracy ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TargetActivityReading:
    """How often a replayed file reached *some* direction over the bars it could decide on.

    ``bars`` counts the decidable bars (the tape minus the declared ``warmup``) and ``fraction`` is
    ``None`` when there were none — an unmeasured tape, never a measured zero.
    """

    bars: int
    warmup: int
    nonflat: int
    fraction: float | None
    finding: DegenerateFinding | None


def read_target_activity(targets: Sequence[int], *, warmup: int = 0) -> TargetActivityReading:
    """Read a replayed target series for trivial-target degeneracy — pure, values only.

    ``targets`` is the per-bar target series the gate's replay produces (+1 long / 0 flat / −1
    short); long and short both count as *deciding something*, since the detector asks whether the
    file took a direction, not which one.
    """
    decidable = list(targets)[max(warmup, 0) :]
    nonflat = sum(1 for target in decidable if int(target) != 0)
    if not decidable:
        return TargetActivityReading(bars=0, warmup=warmup, nonflat=0, fraction=None, finding=None)

    fraction = nonflat / len(decidable)
    finding = None
    if fraction < MIN_NONFLAT_FRACTION:
        warmup_note = f" (after a declared warmup of {warmup} bars)" if warmup else ""
        finding = _warning(
            TRIVIAL_TARGET,
            f"a directional target on {nonflat} of {len(decidable)} decidable bars{warmup_note} — "
            f"{fraction:.1%}, under the {MIN_NONFLAT_FRACTION:.1%} floor",
        )
    return TargetActivityReading(
        bars=len(decidable),
        warmup=warmup,
        nonflat=nonflat,
        fraction=fraction,
        finding=finding,
    )


# ── the report both readings roll up into ─────────────────────────────────────────────────
@dataclass(frozen=True)
class DegenerateReport:
    """One file, both readings, and whatever findings they produced (often none)."""

    strategy: str
    param_floors: ParamFloorReading
    target_activity: TargetActivityReading

    @property
    def findings(self) -> tuple[DegenerateFinding, ...]:
        """Every finding this file drew, in detector order — empty when it drew none."""
        emitted = (self.param_floors.finding, self.target_activity.finding)
        return tuple(finding for finding in emitted if finding is not None)


# ── the one convenience that executes: replay through the gate's own machinery ────────────
def inspect_strategy(
    strategy_cls: type[TraderStrategy], params: Any | None = None
) -> DegenerateReport:
    """Measure a loaded strategy class on both axes, replaying the write gate's own fixture tape.

    This is the only function in the module that *runs* anything, and it runs nothing of its own:
    the tape is :func:`~noctis.strategies.library.fixture_frame` and the replay is
    :func:`~noctis.strategies.base.replay_targets`, the same pair ``write_strategy``'s validator
    uses, so the activity a detector reports is by construction the behaviour the gate graded.
    ``params`` defaults to the file's own declared defaults, which is what the collapse question is
    about in the first place.
    """
    resolved = strategy_cls.params_cls() if params is None else params
    targets = replay_targets(strategy_cls(resolved), fixture_frame())
    return DegenerateReport(
        strategy=strategy_cls.name,
        param_floors=read_param_floors(resolved, strategy_cls.param_space()),
        target_activity=read_target_activity(targets, warmup=strategy_cls.warmup_bars(resolved)),
    )
