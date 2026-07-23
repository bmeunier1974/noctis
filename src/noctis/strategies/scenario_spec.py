"""ScenarioSpec vocabulary and the pure, warmup-parametric scenario compiler (#82).

The known-outcome oracle (:mod:`noctis.strategies.scenarios`) is honest only if the
assertion windows are *reasoned from the thesis*, never *read back from the code's output* —
the self-fulfilling-oracle trap. To make that structural rather than a matter of prompting,
authorship of the tape geometry is inverted: instead of the coder inventing code, tape, and
assertion windows jointly, the driver (FORMULATE, #83) emits a structured :class:`ScenarioSpec`
and this pure compiler derives all bar arithmetic. **The model never writes a bar index.**

This is not a new DSL. A spec reuses the existing scenario vocabulary — the segment builders
(``flat``/``trend``/``selloff``/``recovery``/``chop``/``vol_spike``/``gap``) and the frozen
:class:`~noctis.strategies.scenarios.Scenario`/:class:`~noctis.strategies.scenarios.Segment`/
expectation dataclasses — and only moves *who authors it*. A spec speaks in **legs** (a segment
``kind`` plus its decision-bar length and shape params) and exactly **one behavior tag** per
scenario; :func:`compile_spec` turns that into concrete ``Scenario`` objects.

Warmup-parametric
-----------------
The compiler takes ``warm`` as a parameter (the write gate resolves ``warm =
warmup_bars(default params)`` at validation time — #84). Every scenario is compiled with a
leading flat **setup leg sized ``warm + pad``** (:data:`SETUP_PAD`) so the strategy can warm up
before the interesting legs. Expectation windows are computed from the post-setup leg
boundaries and clamped to begin no earlier than ``warm`` (``max(leg.start, warm)``), so a
directional window can never open during warmup. Because the setup leg is sized *from* ``warm``,
an entry leg always begins after warmup by construction: the "warmup exceeds the entry leg"
conflict the epic flags for the gate (#84) cannot arise at this layer — a ``warm`` so large it
overruns the maximum tape length surfaces instead as a precise out-of-range compile error.

Purity
------
Compilation is a pure, deterministic function of ``(spec, warm)``: no LLM, no I/O, no clock, no
randomness. It lives in the strategy layer and imports nothing from ``noctis.research``. The
same spec compiled at the same ``warm`` yields identical ``Scenario`` objects.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from noctis.strategies.scenarios import (
    MAX_SCENARIO_BARS,
    MAX_SCENARIOS,
    MIN_SCENARIO_BARS,
    MIN_SCENARIOS,
    Expectation,
    Scenario,
    ScenarioError,
    Segment,
    always_flat,
    chop,
    flat,
    flat_by,
    gap,
    holds_long_through,
    holds_short_through,
    long_within,
    recovery,
    selloff,
    short_within,
    trend,
    vol_spike,
)

# The leading setup stretch is ``warm + SETUP_PAD`` decision bars: enough flat bars past the
# warmup itself that a well-behaved strategy has settled before the interesting legs begin.
SETUP_PAD = 20


class SpecError(Exception):
    """A ScenarioSpec violated the vocabulary or shape rules the compiler enforces."""


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary — frozen spec dataclasses (no bar index anywhere)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LegSpec:
    """One leg of a scenario tape: a segment ``kind`` plus its decision-bar length and shape.

    ``bars`` is a *length* (decision bars), never a bar index. ``pct`` feeds
    trend/selloff/recovery/gap; ``amplitude``/``period`` feed chop/vol_spike; unused fields are
    ignored per kind. A ``gap`` leg emits no bars (``bars`` must be 0).
    """

    kind: str
    bars: int
    pct: float = 0.0
    amplitude: float = 0.0
    period: int = 0


class Behavior(StrEnum):
    """The one behavior tag a scenario declares — the only thing the thesis contributes.

    Directional tags reference a leg by index (``leg``); ``NEVER_TRADE`` references none. Long
    and short are explicit variants so direction is never inferred.
    """

    ENTER_LONG = "enter_long_during_leg"
    ENTER_SHORT = "enter_short_during_leg"
    HOLD_LONG = "hold_long_through_leg"
    HOLD_SHORT = "hold_short_through_leg"
    FLAT_BY_END = "flat_by_end_of_leg"
    NEVER_TRADE = "never_trade"


_DIRECTIONAL = frozenset(
    {Behavior.ENTER_LONG, Behavior.ENTER_SHORT, Behavior.HOLD_LONG, Behavior.HOLD_SHORT}
)


@dataclass(frozen=True)
class ScenarioSpec:
    """A named tape (legs) plus ONE behavior tag — the model's whole contribution.

    ``leg`` is a 0-based reference into ``legs`` (the target of an indexed behavior), never a
    bar index; it is ``None`` for ``NEVER_TRADE``.
    """

    name: str
    legs: tuple[LegSpec, ...]
    behavior: Behavior
    leg: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "legs", tuple(self.legs))


@dataclass(frozen=True)
class SpecSuite:
    """A suite of 2–8 scenario specs — the unit FORMULATE (#83) emits and this compiler checks."""

    scenarios: tuple[ScenarioSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", tuple(self.scenarios))


# ─────────────────────────────────────────────────────────────────────────────
# Compilation — pure, deterministic derivation of Scenario objects
# ─────────────────────────────────────────────────────────────────────────────
_BUILDERS = {
    "flat": lambda leg: flat(leg.bars),
    "trend": lambda leg: trend(leg.bars, leg.pct),
    "selloff": lambda leg: selloff(leg.bars, leg.pct),
    "recovery": lambda leg: recovery(leg.bars, leg.pct),
    "chop": lambda leg: chop(leg.bars, leg.amplitude, leg.period or 8),
    "vol_spike": lambda leg: vol_spike(leg.bars, leg.amplitude or 0.05),
    "gap": lambda leg: gap(leg.pct),
}
_KNOWN_KINDS = "flat/trend/selloff/recovery/chop/vol_spike/gap"


def _segment(spec: ScenarioSpec, index: int) -> Segment:
    """Build one authored leg's :class:`Segment`, wrapping builder errors with spec context."""
    leg = spec.legs[index]
    builder = _BUILDERS.get(leg.kind)
    if builder is None:
        raise SpecError(
            f"scenario {spec.name!r} leg {index}: unknown leg kind {leg.kind!r}; "
            f"use one of {_KNOWN_KINDS}"
        )
    try:
        return builder(leg)
    except ScenarioError as exc:
        raise SpecError(f"scenario {spec.name!r} leg {index} ({leg.kind}): {exc}") from exc


def _leg_bounds(setup_bars: int, segments: Sequence[Segment], k: int) -> tuple[int, int]:
    """The ``[start, end)`` bar span of authored leg ``k`` in the compiled tape (after setup)."""
    start = setup_bars + sum(seg.n for seg in segments[:k])
    return start, start + segments[k].n


def _expectations(
    spec: ScenarioSpec, warm: int, setup_bars: int, segments: Sequence[Segment], n_bars: int
) -> tuple[Expectation, ...]:
    """Map the scenario's single behavior tag to its expectation window(s)."""
    if not isinstance(spec.behavior, Behavior):
        raise SpecError(
            f"scenario {spec.name!r}: unknown behavior {spec.behavior!r}; "
            f"use one of {[b.name for b in Behavior]}"
        )
    if spec.behavior is Behavior.NEVER_TRADE:
        return (always_flat(),)

    k = spec.leg
    if k is None:
        raise SpecError(
            f"scenario {spec.name!r}: behavior {spec.behavior.name} requires a target leg index"
        )
    if not 0 <= k < len(spec.legs):
        raise SpecError(
            f"scenario {spec.name!r}: behavior {spec.behavior.name} targets leg {k} "
            f"but the spec has {len(spec.legs)} legs (0..{len(spec.legs) - 1})"
        )
    start, end = _leg_bounds(setup_bars, segments, k)
    if end == start:
        raise SpecError(
            f"scenario {spec.name!r}: behavior {spec.behavior.name} targets leg {k} which emits "
            f"no bars (a gap); target a leg with bars >= 1"
        )

    if spec.behavior is Behavior.FLAT_BY_END:
        if end >= n_bars:
            raise SpecError(
                f"scenario {spec.name!r}: flat_by_end_of_leg targets the final leg {k}; add a "
                f"following leg so the flat-by-exit is observable"
            )
        return (flat_by(end),)

    lo, hi = max(start, warm), end - 1
    builders = {
        Behavior.ENTER_LONG: long_within,
        Behavior.ENTER_SHORT: short_within,
        Behavior.HOLD_LONG: holds_long_through,
        Behavior.HOLD_SHORT: holds_short_through,
    }
    return (builders[spec.behavior](lo, hi),)


def compile_scenario(spec: ScenarioSpec, warm: int) -> Scenario:
    """Compile one :class:`ScenarioSpec` into a warmup-parametric :class:`Scenario`.

    Pure and deterministic: prepend a flat setup leg of ``warm + SETUP_PAD`` bars, build the
    authored legs, then derive the behavior tag's expectation window from the leg boundaries.
    """
    if isinstance(warm, bool) or not isinstance(warm, int) or warm < 0:
        raise SpecError(f"warm must be a non-negative int, got {warm!r}")
    setup_bars = warm + SETUP_PAD
    setup = flat(setup_bars)
    segments = tuple(_segment(spec, i) for i in range(len(spec.legs)))
    n_bars = setup_bars + sum(seg.n for seg in segments)
    if not MIN_SCENARIO_BARS <= n_bars <= MAX_SCENARIO_BARS:
        raise SpecError(
            f"scenario {spec.name!r}: compiles to {n_bars} bars (setup {setup_bars} + legs "
            f"{n_bars - setup_bars}), outside [{MIN_SCENARIO_BARS}, {MAX_SCENARIO_BARS}]; "
            f"adjust the leg lengths or the strategy warmup"
        )
    expect = _expectations(spec, warm, setup_bars, segments, n_bars)
    return Scenario(name=spec.name, segments=(setup, *segments), expect=expect)


def compile_spec(spec: SpecSuite, warm: int) -> tuple[Scenario, ...]:
    """Compile a :class:`SpecSuite` into contract-satisfying :class:`Scenario` objects.

    Enforces the existing declaration shape rules at compile time — 2–8 scenarios, unique names,
    at least one directional entry, at least one no-trade tape, 60–2000 bars each — so any valid
    spec compiles to a suite that passes ``check_scenario_contract``'s declaration checks. Raises
    :class:`SpecError` with a precise message on any violation.
    """
    specs = spec.scenarios
    n = len(specs)
    if not MIN_SCENARIOS <= n <= MAX_SCENARIOS:
        raise SpecError(f"want {MIN_SCENARIOS}-{MAX_SCENARIOS} scenarios, got {n}")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise SpecError(f"scenario names must be unique, got {names}")
    compiled = tuple(compile_scenario(s, warm) for s in specs)
    if not any(s.behavior in _DIRECTIONAL for s in specs):
        raise SpecError(
            "at least one scenario must demand a directional entry (enter/hold long/short)"
        )
    if not any(s.behavior is Behavior.NEVER_TRADE for s in specs):
        raise SpecError("at least one scenario must be a no-trade tape (never_trade)")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# JSON round-trip — the pure carrier that crosses the write-gate subprocess boundary (#84)
# ─────────────────────────────────────────────────────────────────────────────
# The write gate resolves ``warm`` from the *candidate's* declared warmup, so it must carry the
# uncompiled :class:`SpecSuite` — not compiled ``Scenario`` objects — into the fresh interpreter
# it validates in. These two functions are the pure, deterministic (spec ⇄ text) round trip used
# both to hand the spec to the subprocess validator and to embed it in the machine-stamped
# ``scenarios()`` block, so the installed file re-derives the same oracle at runtime.
def spec_to_json(suite: SpecSuite) -> str:
    """Serialize a :class:`SpecSuite` to a deterministic JSON string (pure — no I/O, no clock)."""
    return json.dumps(
        {
            "scenarios": [
                {
                    "name": s.name,
                    "legs": [
                        {
                            "kind": leg.kind,
                            "bars": leg.bars,
                            "pct": leg.pct,
                            "amplitude": leg.amplitude,
                            "period": leg.period,
                        }
                        for leg in s.legs
                    ],
                    "behavior": s.behavior.name,
                    "leg": s.leg,
                }
                for s in suite.scenarios
            ]
        }
    )


def spec_from_json(text: str) -> SpecSuite:
    """Reconstruct a :class:`SpecSuite` from :func:`spec_to_json` output (its pure inverse)."""
    payload = json.loads(text)
    return SpecSuite(
        scenarios=tuple(
            ScenarioSpec(
                name=s["name"],
                legs=tuple(
                    LegSpec(
                        kind=leg["kind"],
                        bars=leg["bars"],
                        pct=leg["pct"],
                        amplitude=leg["amplitude"],
                        period=leg["period"],
                    )
                    for leg in s["legs"]
                ),
                behavior=Behavior[s["behavior"]],
                leg=s["leg"],
            )
            for s in payload["scenarios"]
        )
    )
