"""ScenarioSpec vocabulary and the pure warmup-parametric compiler (#82).

The spec layer moves every piece of bar arithmetic out of the model and into code: a
:class:`ScenarioSpec` names legs by *kind* and length (decision bars) and carries ONE
behavior tag; :func:`compile_spec` derives the warmup-parametric :class:`Scenario` objects —
the model never writes a bar index, and expectation windows are computed from leg boundaries
and ``warm``, never calibrated from observed behavior.
"""

from __future__ import annotations

import ast
import dataclasses
from collections import deque
from dataclasses import dataclass

import pytest

from noctis.strategies import indicators as ind
from noctis.strategies.base import ParamSpec, TraderStrategy
from noctis.strategies.scenario_spec import (
    SETUP_PAD,
    Behavior,
    LegSpec,
    ScenarioSpec,
    SpecError,
    SpecSuite,
    compile_scenario,
    compile_spec,
    spec_from_json,
    spec_to_json,
)
from noctis.strategies.scenarios import (
    MAX_SCENARIO_BARS,
    MAX_SCENARIOS,
    MIN_SCENARIO_BARS,
    MIN_SCENARIOS,
    AlwaysFlat,
    FlatBy,
    HoldsLongThrough,
    HoldsShortThrough,
    LongWithin,
    Scenario,
    Segment,
    ShortWithin,
)

WARMS = [0, 1, 5, 20, 50]

# Leg kinds that emit bars (everything except gap). Each maps to a valid shape param set.
_KIND_PARAMS = {
    "flat": {},
    "trend": {"pct": 0.10},
    "selloff": {"pct": 0.10},
    "recovery": {"pct": 0.10},
    "chop": {"amplitude": 0.03, "period": 8},
    "vol_spike": {"amplitude": 0.05},
}
BAR_KINDS = list(_KIND_PARAMS)
DIRECTIONAL_TAGS = [
    Behavior.ENTER_LONG,
    Behavior.ENTER_SHORT,
    Behavior.HOLD_LONG,
    Behavior.HOLD_SHORT,
]


def _leg(kind: str, bars: int) -> LegSpec:
    return LegSpec(kind=kind, bars=bars, **_KIND_PARAMS.get(kind, {}))


def _directional_spec(
    kind: str, tag: Behavior, *, name: str = "dir", bars: int = 60
) -> ScenarioSpec:
    """A single-leg directional scenario: enter/hold on leg 0 of the given kind."""
    return ScenarioSpec(name=name, legs=[_leg(kind, bars)], behavior=tag, leg=0)


def _never_trade_spec(*, name: str = "flat_tape", bars: int = 60) -> ScenarioSpec:
    return ScenarioSpec(name=name, legs=[_leg("flat", bars)], behavior=Behavior.NEVER_TRADE)


# ── acceptance: frozen dataclasses, vocabulary limited, no bar index ────────────────────────
def test_spec_dataclasses_are_frozen():
    leg = LegSpec(kind="flat", bars=10)
    scen = ScenarioSpec(name="s", legs=[leg], behavior=Behavior.NEVER_TRADE)
    suite = SpecSuite(scenarios=[scen])
    for obj, field, value in [(leg, "bars", 5), (scen, "name", "x"), (suite, "scenarios", ())]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, value)


def test_behavior_is_a_small_enum_of_exactly_the_declared_tags():
    assert {b.name for b in Behavior} == {
        "ENTER_LONG",
        "ENTER_SHORT",
        "HOLD_LONG",
        "HOLD_SHORT",
        "FLAT_BY_END",
        "NEVER_TRADE",
    }


def test_a_spec_carries_no_bar_index_only_lengths_and_a_leg_reference():
    # The whole point: the model writes leg *lengths* (decision bars) and a *leg* index, never a
    # bar index. The dataclass fields make that structural — there is no window/index field.
    leg_fields = {f.name for f in dataclasses.fields(LegSpec)}
    assert leg_fields == {"kind", "bars", "pct", "amplitude", "period"}
    scen_fields = {f.name for f in dataclasses.fields(ScenarioSpec)}
    assert scen_fields == {"name", "legs", "behavior", "leg"}
    # `leg` is a leg reference (0..n-1), never a bar index into a tape.
    assert ScenarioSpec("s", [LegSpec("flat", 10)], Behavior.NEVER_TRADE).leg is None


# ── acceptance: compiler is pure, deterministic, and free of research-layer deps ─────────────
def test_compiler_module_imports_nothing_from_the_research_layer():
    import noctis.strategies.scenario_spec as mod

    source = ast.parse(open(mod.__file__).read())
    imported: list[str] = []
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [name for name in imported if name.startswith("noctis.research")]
    assert offenders == [], f"scenario_spec must not import the research layer: {offenders}"


def test_compile_is_deterministic_same_spec_same_warm_identical_scenarios():
    suite = SpecSuite([_directional_spec("trend", Behavior.ENTER_LONG), _never_trade_spec()])
    assert compile_spec(suite, 20) == compile_spec(suite, 20)


def test_compile_prepends_a_flat_setup_leg_sized_warm_plus_pad():
    scen = compile_scenario(_directional_spec("trend", Behavior.ENTER_LONG, bars=60), warm=20)
    setup = scen.segments[0]
    assert setup == Segment("flat", 20 + SETUP_PAD)
    # the authored leg follows the setup leg
    assert scen.segments[1] == Segment("drift", 60, 0.10)


# ── acceptance: property — every valid spec compiles to contract-passing scenarios ──────────
def _assert_declaration_ok(scenarios: tuple[Scenario, ...], warm: int) -> None:
    """Mirror ``check_scenario_contract``'s declaration checks, plus the warmup invariants."""
    assert MIN_SCENARIOS <= len(scenarios) <= MAX_SCENARIOS
    assert len({s.name for s in scenarios}) == len(scenarios)
    has_directional = has_flat = False
    for sc in scenarios:
        assert MIN_SCENARIO_BARS <= sc.n_bars <= MAX_SCENARIO_BARS
        for exp in sc.expect:
            # window sits inside the tape
            assert exp.last_index < sc.n_bars
            # window sits strictly after warmup (directional entries only)
            if getattr(exp, "is_directional", False):
                assert exp.lo >= warm
                assert exp.lo >= 0 and exp.lo <= exp.hi
                has_directional = True
            if isinstance(exp, AlwaysFlat):
                has_flat = True
    assert has_directional and has_flat


@pytest.mark.parametrize("kind", BAR_KINDS)
@pytest.mark.parametrize("tag", DIRECTIONAL_TAGS)
@pytest.mark.parametrize("warm", WARMS)
def test_every_valid_directional_spec_compiles_to_a_contract_passing_suite(kind, tag, warm):
    suite = SpecSuite([_directional_spec(kind, tag), _never_trade_spec()])
    _assert_declaration_ok(compile_spec(suite, warm), warm)


@pytest.mark.parametrize("kind", BAR_KINDS)
@pytest.mark.parametrize("warm", WARMS)
def test_flat_by_end_spec_compiles_inside_the_tape(kind, warm):
    flat_by = ScenarioSpec(
        name="exit",
        legs=[_leg(kind, 50), _leg("flat", 30)],
        behavior=Behavior.FLAT_BY_END,
        leg=0,
    )
    suite = SpecSuite(
        [flat_by, _directional_spec("trend", Behavior.ENTER_LONG), _never_trade_spec()]
    )
    compiled = compile_spec(suite, warm)
    exit_scenario = compiled[0]
    (exit_exp,) = exit_scenario.expect
    assert isinstance(exit_exp, FlatBy)
    assert exit_exp.last_index < exit_scenario.n_bars


@pytest.mark.parametrize("warm", WARMS)
def test_gap_leg_is_a_valid_non_target_leg(warm):
    # A gap emits no bars but is a legal waveform between two bar-emitting legs; targeting the
    # trend after it compiles cleanly.
    spec = ScenarioSpec(
        name="gap_then_trend",
        legs=[LegSpec("flat", 30), LegSpec("gap", 0, pct=0.10), LegSpec("trend", 40, pct=0.10)],
        behavior=Behavior.ENTER_LONG,
        leg=2,
    )
    compiled = compile_spec(SpecSuite([spec, _never_trade_spec()]), warm)
    _assert_declaration_ok(compiled, warm)


# ── behavior-tag → expectation mapping ──────────────────────────────────────────────────────
@pytest.mark.parametrize("warm", WARMS)
@pytest.mark.parametrize(
    ("tag", "exp_type"),
    [
        (Behavior.ENTER_LONG, LongWithin),
        (Behavior.ENTER_SHORT, ShortWithin),
        (Behavior.HOLD_LONG, HoldsLongThrough),
        (Behavior.HOLD_SHORT, HoldsShortThrough),
    ],
)
def test_directional_tag_maps_to_its_expectation_over_the_target_leg(tag, exp_type, warm):
    scen = compile_scenario(_directional_spec("trend", tag, bars=60), warm)
    (exp,) = scen.expect
    assert isinstance(exp, exp_type)
    start = warm + SETUP_PAD  # setup leg then leg 0
    assert (exp.lo, exp.hi) == (start, start + 60 - 1)


@pytest.mark.parametrize("warm", WARMS)
def test_flat_by_end_maps_to_flat_by_leg_boundary(warm):
    spec = ScenarioSpec(
        name="x", legs=[_leg("trend", 40), _leg("flat", 30)], behavior=Behavior.FLAT_BY_END, leg=0
    )
    scen = compile_scenario(spec, warm)
    (exp,) = scen.expect
    assert isinstance(exp, FlatBy)
    assert exp.bar == warm + SETUP_PAD + 40  # end of leg 0


@pytest.mark.parametrize("warm", WARMS)
def test_never_trade_maps_to_always_flat(warm):
    scen = compile_scenario(_never_trade_spec(bars=80), warm)
    (exp,) = scen.expect
    assert isinstance(exp, AlwaysFlat)


# ── malformed specs fail compilation with precise messages ──────────────────────────────────
def test_unknown_leg_kind_is_rejected():
    suite = SpecSuite([_directional_spec("mystery", Behavior.ENTER_LONG), _never_trade_spec()])
    with pytest.raises(SpecError, match="unknown leg kind 'mystery'"):
        compile_spec(suite, 5)


def test_unknown_behavior_is_rejected():
    bad = ScenarioSpec(name="b", legs=[_leg("trend", 60)], behavior="bogus", leg=0)  # type: ignore[arg-type]
    with pytest.raises(SpecError, match="unknown behavior"):
        compile_spec(SpecSuite([bad, _never_trade_spec()]), 5)


def test_leg_index_out_of_range_is_rejected():
    bad = ScenarioSpec(name="b", legs=[_leg("trend", 60)], behavior=Behavior.ENTER_LONG, leg=5)
    with pytest.raises(SpecError, match="leg 5"):
        compile_spec(SpecSuite([bad, _never_trade_spec()]), 5)


def test_indexed_behavior_without_a_target_leg_is_rejected():
    bad = ScenarioSpec(name="b", legs=[_leg("trend", 60)], behavior=Behavior.ENTER_LONG, leg=None)
    with pytest.raises(SpecError, match="requires a target leg"):
        compile_spec(SpecSuite([bad, _never_trade_spec()]), 5)


def test_targeting_a_gap_leg_is_rejected():
    bad = ScenarioSpec(
        name="b",
        legs=[_leg("flat", 60), LegSpec("gap", 0, pct=0.1)],
        behavior=Behavior.ENTER_LONG,
        leg=1,
    )
    with pytest.raises(SpecError, match="no bars"):
        compile_spec(SpecSuite([bad, _never_trade_spec()]), 5)


def test_flat_by_end_on_the_final_leg_is_rejected():
    bad = ScenarioSpec(name="b", legs=[_leg("trend", 60)], behavior=Behavior.FLAT_BY_END, leg=0)
    suite = SpecSuite([bad, _directional_spec("trend", Behavior.ENTER_LONG), _never_trade_spec()])
    with pytest.raises(SpecError, match="final leg"):
        compile_spec(suite, 5)


def test_too_short_tape_is_rejected_as_out_of_range():
    tiny = ScenarioSpec(name="tiny", legs=[_leg("trend", 5)], behavior=Behavior.ENTER_LONG, leg=0)
    with pytest.raises(SpecError, match="outside"):
        compile_spec(SpecSuite([tiny, _never_trade_spec()]), 0)


def test_warmup_that_blows_the_tape_past_the_maximum_is_rejected():
    suite = SpecSuite([_directional_spec("trend", Behavior.ENTER_LONG), _never_trade_spec()])
    with pytest.raises(SpecError, match="outside"):
        compile_spec(suite, 3000)


def test_invalid_shape_params_are_rejected_with_the_builder_message():
    bad = ScenarioSpec(
        name="b",
        legs=[LegSpec("chop", 60, amplitude=0.0, period=8)],
        behavior=Behavior.ENTER_LONG,
        leg=0,
    )
    with pytest.raises(SpecError, match="amplitude"):
        compile_spec(SpecSuite([bad, _never_trade_spec()]), 5)


@pytest.mark.parametrize(
    ("scenarios", "match"),
    [
        ([_directional_spec("trend", Behavior.ENTER_LONG)], "2-8"),
        (
            [_directional_spec("trend", Behavior.ENTER_LONG, name=f"s{i}") for i in range(9)]
            + [_never_trade_spec()],
            "2-8",
        ),
        (
            [
                _directional_spec("trend", Behavior.ENTER_LONG, name="dup"),
                _never_trade_spec(name="dup"),
            ],
            "unique",
        ),
        ([_never_trade_spec(name="a"), _never_trade_spec(name="b")], "directional"),
        (
            [
                _directional_spec("trend", Behavior.ENTER_LONG, name="a"),
                _directional_spec("trend", Behavior.ENTER_SHORT, name="b"),
            ],
            "no-trade",
        ),
    ],
)
def test_suite_shape_rules_are_enforced(scenarios, match):
    with pytest.raises(SpecError, match=match):
        compile_spec(SpecSuite(scenarios), 5)


def test_negative_warm_is_rejected():
    suite = SpecSuite([_directional_spec("trend", Behavior.ENTER_LONG), _never_trade_spec()])
    with pytest.raises(SpecError, match="non-negative"):
        compile_spec(suite, -1)


# ── JSON round-trip: the pure carrier across the write-gate subprocess boundary (#84) ────────
def _mixed_suite() -> SpecSuite:
    """A suite exercising every field: shape params, a gap leg, an indexed leg, NEVER_TRADE."""
    return SpecSuite(
        [
            ScenarioSpec(
                "rally",
                [LegSpec("flat", 30), LegSpec("gap", 0, pct=0.1), _leg("chop", 60)],
                Behavior.HOLD_LONG,
                leg=2,
            ),
            ScenarioSpec("dip", [_leg("selloff", 60)], Behavior.ENTER_SHORT, leg=0),
            _never_trade_spec(name="grind", bars=80),
        ]
    )


def test_spec_json_round_trip_reconstructs_the_suite_exactly():
    suite = _mixed_suite()
    assert spec_from_json(spec_to_json(suite)) == suite


def test_spec_to_json_is_deterministic():
    suite = _mixed_suite()
    assert spec_to_json(suite) == spec_to_json(suite)


def test_round_tripped_suite_compiles_identically():
    suite = _mixed_suite()
    restored = spec_from_json(spec_to_json(suite))
    assert compile_spec(restored, 20) == compile_spec(suite, 20)


# ── depth: a compiled suite passes the real scenario-contract end to end ─────────────────────
class _LongShort(TraderStrategy):
    """Long above its SMA, short below — a real thesis to replay compiled tapes through."""

    name = "longshort"

    @dataclass(frozen=True)
    class Params:
        lookback: int = 10

    params_cls = Params

    def on_start(self, ctx):
        self._closes = deque(maxlen=self.params.lookback)

    def on_bar(self, ctx, bar):
        self._closes.append(bar.close)
        mean = ind.sma(self._closes, self.params.lookback)
        if mean is None or bar.close == mean:
            ctx.set_target(0)
        else:
            ctx.set_target(1 if bar.close > mean else -1)

    @classmethod
    def param_space(cls):
        return [ParamSpec("lookback", "int", 5, 40, 1)]


def test_compiled_suite_passes_the_full_scenario_contract_against_a_real_strategy():
    from noctis.strategies.scenarios import check_scenario_contract

    suite = SpecSuite(
        [
            ScenarioSpec("rally", [_leg("trend", 60)], Behavior.ENTER_LONG, leg=0),
            ScenarioSpec("decline", [_leg("selloff", 60)], Behavior.ENTER_SHORT, leg=0),
            _never_trade_spec(name="grind", bars=60),
        ]
    )
    compiled = compile_spec(suite, warm=10)

    class _Probe(_LongShort):
        @classmethod
        def scenarios(cls):
            return list(compiled)

    check_scenario_contract(_Probe)
