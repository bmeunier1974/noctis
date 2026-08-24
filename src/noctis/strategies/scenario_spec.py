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

Three crossings, one vocabulary
------------------------------
Every crossing into the spec vocabulary is owned here, and each leaves as the same frozen
:class:`SpecSuite`, so what a spec *is* is decided in this module rather than at either boundary:

* **Model dialect** — :func:`spec_from_payload` reads the *model's* FORMULATE emit: behavior by its
  wire value, shape params omitted rather than zeroed, an absent ``leg`` — and a precise refusal
  for every other malformed shape, because that sentence becomes the corrective the model is
  re-prompted with. :data:`PARSE_WARM` is the representative warmup that parse compiles at, a
  structural check only; the write gate re-compiles at the candidate's real declared warmup. What
  the model was *asked* for is here too: :data:`SPEC_JSON_SCHEMA` is the JSON Schema the FORMULATE
  emit contract advertises, with its enums read off :data:`LEG_KINDS` and :class:`Behavior`, so the
  offer and the parse are the same vocabulary by construction rather than by two lists agreeing.
* **Carrier** — :func:`spec_to_json` writes the *machine-exact* text and :func:`spec_from_json`
  reads it back: the pure round trip that crosses the write gate's subprocess and rides inside the
  machine-stamped ``scenarios()`` block.
* **Compile** — :func:`compile_spec`, the pure ``(spec, warm)`` function below.

The research layer keeps only the boundary — read the emitted field, hand it over, translate the
refusal into the episode runner's currency.

What is **not** owned here: the suite-shape rules. 2-8 scenarios, unique names, at least one
directional entry, at least one no-trade tape, 60-2000 bars each are the known-outcome contract's
own, written once in :func:`~noctis.strategies.scenarios.check_suite_shape`, which
:func:`compile_spec` runs on the compiled tuple — wrapping the
:class:`~noctis.strategies.scenarios.ScenarioError` into this module's :class:`SpecError` — so a
spec-path refusal and a hand-authored ``scenarios()`` refusal are the same sentence.

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
from typing import Any

from noctis.strategies.scenarios import (
    MAX_SCENARIO_BARS,
    MIN_SCENARIO_BARS,
    Expectation,
    Scenario,
    ScenarioError,
    Segment,
    always_flat,
    check_suite_shape,
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

# The representative warmup FORMULATE compiles a spec at *parse* time — purely a structural
# validity check of the spec's shape (the write gate #84 re-compiles at the strategy's real
# declared warmup). Zero, so a directional entry leg begins right after the setup pad.
PARSE_WARM = 0


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

# The leg kinds, in one place: the builders' own keys. Everything that names the vocabulary reads
# it from here — the enum :data:`SPEC_JSON_SCHEMA` advertises to the model and the unknown-kind
# refusal below — so a builder added to the table is offered and refused by the same edit.
LEG_KINDS: tuple[str, ...] = tuple(_BUILDERS)
_KNOWN_KINDS = "/".join(LEG_KINDS)


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

    The tape-length refusal here is a **diagnosis**, not a second rule: it carries the setup/legs
    arithmetic (and so the advice to adjust the leg lengths or the warmup) that the shared
    arbiter, :func:`~noctis.strategies.scenarios.check_suite_shape`, cannot know. The arbiter of
    the 60–2000 bar range, like every other suite-shape rule's, is that one check.
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


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable rendering — the faithful coder-brief presentation of the fixed oracle (#85)
# ─────────────────────────────────────────────────────────────────────────────
# One readable phrase per indexed behavior tag; NEVER_TRADE renders its own no-trade sentence.
_BEHAVIOR_PHRASES = {
    Behavior.ENTER_LONG: "enter long during",
    Behavior.ENTER_SHORT: "enter short during",
    Behavior.HOLD_LONG: "hold long through",
    Behavior.HOLD_SHORT: "hold short through",
    Behavior.FLAT_BY_END: "be flat by the end of",
}


def _leg_phrase(leg: LegSpec) -> str:
    """One leg as ``kind(bars)`` — the tape shape a coder reads (no bar index)."""
    return f"{leg.kind}({leg.bars})"


def describe_scenario(spec: ScenarioSpec) -> str:
    """One scenario tape rendered as ``name: leg then leg — behavior leg <k>`` (or a no-trade
    sentence for ``never_trade``). Faithful to the spec: leg kinds and decision-bar lengths in
    order, plus the single behavior and its target leg index — never a compiled bar index."""
    tape = " then ".join(_leg_phrase(leg) for leg in spec.legs)
    if spec.behavior is Behavior.NEVER_TRADE:
        return f"{spec.name}: {tape} — stay flat throughout (never trade)"
    return f"{spec.name}: {tape} — {_BEHAVIOR_PHRASES[spec.behavior]} leg {spec.leg}"


def describe_spec(suite: SpecSuite) -> str:
    """A faithful human-readable rendering of the fixed oracle: one line per scenario tape naming
    the ordered legs (kind + decision-bar length) and the single behavior each tape must prove
    with its target leg index.

    Pure and deterministic — the same suite always renders identically. Both the episodic driver's
    author brief and the coder prompt present the fixed oracle from this one rendering, so a coder
    can reason about the tape shape without inventing bar indices (#85). The compiler
    (:func:`compile_spec`) derives every window from the leg lengths and the strategy's declared
    warmup, so the rendering carries no absolute bar position."""
    return "; ".join(describe_scenario(spec) for spec in suite.scenarios)


def compile_spec(spec: SpecSuite, warm: int) -> tuple[Scenario, ...]:
    """Compile a :class:`SpecSuite` into contract-satisfying :class:`Scenario` objects.

    Compiles every :class:`ScenarioSpec` through :func:`compile_scenario`, then runs the
    contract's own :func:`~noctis.strategies.scenarios.check_suite_shape` on the compiled tuple —
    the suite-shape rules (2–8 scenarios, unique names, at least one directional entry, at least
    one no-trade tape, 60–2000 bars each) are written once, in the strategy layer's oracle module,
    so a spec-path refusal and a hand-authored ``scenarios()`` refusal are the same sentence. The
    :class:`~noctis.strategies.scenarios.ScenarioError` is wrapped into this module's one
    exception, exactly as ``_segment`` wraps a builder's.

    Compilation runs **before** the shape check, so an uncompilable tape in an over-long suite
    reports its own compile error rather than the count. Of the five rules, the bar-range one has
    an earlier **diagnosis** in :func:`compile_scenario` (which knows the setup/legs arithmetic);
    its **arbiter** is ``check_suite_shape``, like every other rule's.
    """
    compiled = tuple(compile_scenario(s, warm) for s in spec.scenarios)
    try:
        check_suite_shape(compiled)
    except ScenarioError as exc:
        raise SpecError(str(exc)) from exc
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


# ─────────────────────────────────────────────────────────────────────────────
# Model dialect — the schema FORMULATE asks for, and the tolerant parse of what comes back (#83)
# ─────────────────────────────────────────────────────────────────────────────
# The structured scenario_spec the model emits — a 1:1 mapping onto the #82 vocabulary. The model
# reasons about tape SHAPE (legs) and ONE behavior tag per scenario; it NEVER writes a bar index —
# the compiler derives every window from the leg boundaries and the strategy's declared warmup.
# The schema lives beside the vocabulary it describes and takes its enums from it, so what the
# model is *offered* and what the parse *accepts* are the same list by construction (#322). The
# episodic driver composes this object into its FORMULATE emit contract under the field name.
_LEG_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": list(LEG_KINDS),
            "description": "The segment shape of this leg.",
        },
        "bars": {
            "type": "integer",
            "description": "The leg's LENGTH in decision bars (never a bar index); 0 for a gap.",
        },
        "pct": {
            "type": "number",
            "description": "Signed total move for trend/selloff/recovery/gap (0.05 = +5%).",
        },
        "amplitude": {
            "type": "number",
            "description": "Oscillation amplitude for chop / vol_spike (e.g. 0.03).",
        },
        "period": {"type": "integer", "description": "Wave length in bars for chop (default 8)."},
    },
    "required": ["kind", "bars"],
}

_SCENARIO_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "A unique name for this scenario tape."},
        "legs": {
            "type": "array",
            "items": _LEG_SPEC_SCHEMA,
            "description": "The ordered legs of the tape, in decision-bar lengths.",
        },
        "behavior": {
            "type": "string",
            "enum": [b.value for b in Behavior],
            "description": "The ONE behavior this tape must prove (the thesis's contribution).",
        },
        "leg": {
            "type": "integer",
            "description": "0-based index into 'legs' the behavior targets; omit for never_trade.",
        },
    },
    "required": ["name", "legs", "behavior"],
}

SPEC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": _SCENARIO_ITEM_SCHEMA,
            "description": (
                "2-8 known-outcome tapes: at least one directional entry (enter/hold "
                "long/short) and at least one never_trade tape. You author tape SHAPE and "
                "behavior only — never a bar index."
            ),
        },
    },
    "required": ["scenarios"],
}


# The other dialect (see the module docstring): what a model emitted, read one level at a time —
# leg, then behavior, then scenario. Every refusal here is a sentence the model is re-prompted
# with, so the wording is contract rather than diagnostics, and it is checked as such.
def _leg_from_payload(payload: Any, scenario_name: str, index: int) -> LegSpec:
    """Construct one frozen :class:`LegSpec` from the model's JSON leg; raise on a malformed shape
    (a non-object leg, a missing kind, a non-integer length) so it re-prompts as a schema misfire.
    ``pct``/``amplitude``/``period`` default to 0 and are ignored per kind by the compiler."""
    if not isinstance(payload, dict):
        raise SpecError(f"scenario {scenario_name!r} leg {index}: each leg must be an object")
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise SpecError(f"scenario {scenario_name!r} leg {index}: a leg 'kind' is required")
    bars = payload.get("bars", 0)
    if isinstance(bars, bool) or not isinstance(bars, int):
        raise SpecError(
            f"scenario {scenario_name!r} leg {index}: 'bars' must be an integer length (0 for gap)"
        )
    return LegSpec(
        kind=kind,
        bars=bars,
        pct=float(payload.get("pct", 0.0) or 0.0),
        amplitude=float(payload.get("amplitude", 0.0) or 0.0),
        period=int(payload.get("period", 0) or 0),
    )


def _behavior_from_payload(value: Any, scenario_name: str) -> Behavior:
    """Map the model's behavior string onto the :class:`Behavior` tag; raise with the allowed
    vocabulary on an unknown/missing tag so it re-prompts as a schema misfire."""
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"scenario {scenario_name!r}: a 'behavior' tag is required")
    try:
        return Behavior(value)
    except ValueError:
        allowed = ", ".join(b.value for b in Behavior)
        raise SpecError(
            f"scenario {scenario_name!r}: unknown behavior {value!r}; use one of {allowed}"
        ) from None


def _scenario_from_payload(payload: Any, index: int) -> ScenarioSpec:
    """Construct one frozen :class:`ScenarioSpec` from the model's JSON scenario; raise on any
    malformed shape (missing name/legs, a non-integer leg reference) so it re-prompts as a
    misfire."""
    if not isinstance(payload, dict):
        raise SpecError(f"scenario {index}: each scenario must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError(f"scenario {index}: a non-empty 'name' is required")
    raw_legs = payload.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        raise SpecError(f"scenario {name!r}: a non-empty 'legs' list is required")
    legs = tuple(_leg_from_payload(leg, name, i) for i, leg in enumerate(raw_legs))
    behavior = _behavior_from_payload(payload.get("behavior"), name)
    leg_ref = payload.get("leg")
    if leg_ref is not None and (isinstance(leg_ref, bool) or not isinstance(leg_ref, int)):
        raise SpecError(f"scenario {name!r}: 'leg' must be an integer leg index or omitted")
    return ScenarioSpec(name=name, legs=legs, behavior=behavior, leg=leg_ref)


def spec_from_payload(raw: Any) -> SpecSuite:
    """Parse the model's emitted ``scenario_spec`` object into a frozen :class:`SpecSuite`.

    ``raw`` is the spec object itself (``{"scenarios": [...]}``), not the whole FORMULATE payload:
    reading the field out of the emit is the driver's boundary, deciding what a spec *is* is this
    module's. Pure — no LLM, no I/O — and tolerant only where the model's dialect legitimately
    differs from the JSON carrier's (behavior by value, omitted shape params defaulting to 0, an
    absent ``leg``). Every other malformed shape raises :class:`SpecError` with a precise sentence,
    which the episode runner turns into the corrective the model reads on its re-prompt.

    ``bool`` is refused where an integer belongs (``bars``, ``leg``) even though Python calls it an
    int: a model emitting ``true`` for a count is misfiring, not authoring a one-bar leg.

    The suite is *shape*-checked at compile time, not here — :func:`compile_spec` is the one arbiter
    of the suite rules, so this function is the vocabulary check and nothing more.
    """
    if not isinstance(raw, dict):
        raise SpecError("scenario_spec must be an object with a 'scenarios' list")
    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise SpecError("scenario_spec.scenarios must be a non-empty list of scenarios")
    return SpecSuite(
        scenarios=tuple(_scenario_from_payload(s, i) for i, s in enumerate(raw_scenarios))
    )
