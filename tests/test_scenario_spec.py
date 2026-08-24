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
    PARSE_WARM,
    SETUP_PAD,
    Behavior,
    LegSpec,
    ScenarioSpec,
    SpecError,
    SpecSuite,
    compile_scenario,
    compile_spec,
    describe_spec,
    spec_from_json,
    spec_from_payload,
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
    ScenarioError,
    Segment,
    ShortWithin,
    check_suite_shape,
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


# ── describe_spec: a faithful human-readable rendering of the fixed oracle (#85) ────────────
def test_describe_spec_renders_tape_shapes_behaviors_and_target_leg():
    # The coder-facing rendering names every leg (kind + decision-bar length) and the ONE behavior
    # each tape proves with its target leg, so a coder reasons about the tape without a bar index.
    suite = SpecSuite(
        scenarios=[
            ScenarioSpec(
                name="rally",
                legs=[_leg("flat", 20), _leg("trend", 60)],
                behavior=Behavior.ENTER_LONG,
                leg=1,
            ),
            _never_trade_spec(name="grind", bars=60),
        ]
    )

    text = describe_spec(suite)

    # Tape shapes: kind(bars) for each leg, in order.
    assert "rally: flat(20) then trend(60)" in text
    assert "grind: flat(60)" in text
    # Behaviors with the target leg index (never a bar index).
    assert "enter long during leg 1" in text
    assert "never trade" in text
    # No bar index leaks into the rendering.
    assert "bar 0" not in text and "bar 1" not in text


def test_describe_spec_covers_every_behavior_phrase():
    # Each directional/flat-by tag renders a distinct, readable phrase; never_trade its own.
    phrases = {
        Behavior.ENTER_LONG: "enter long during leg 0",
        Behavior.ENTER_SHORT: "enter short during leg 0",
        Behavior.HOLD_LONG: "hold long through leg 0",
        Behavior.HOLD_SHORT: "hold short through leg 0",
    }
    for tag, phrase in phrases.items():
        assert phrase in describe_spec(SpecSuite(scenarios=[_directional_spec("trend", tag)]))


def test_describe_spec_is_pure_and_deterministic():
    suite = SpecSuite(
        scenarios=[_directional_spec("trend", Behavior.ENTER_LONG), _never_trade_spec()]
    )
    assert describe_spec(suite) == describe_spec(suite)


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


def test_the_suite_shape_refusal_has_one_spelling_across_both_dialects():
    # compile_spec compiles, then runs the contract's own check_suite_shape and wraps its
    # ScenarioError — so a spec-path refusal and a DSL-path refusal are the same sentence (#320).
    missing_no_trade = [
        _directional_spec("trend", Behavior.ENTER_LONG, name="a"),
        _directional_spec("trend", Behavior.ENTER_SHORT, name="b"),
    ]
    missing_directional = [_never_trade_spec(name="a"), _never_trade_spec(name="b")]
    for specs in (missing_no_trade, missing_directional):
        compiled = tuple(compile_scenario(s, 5) for s in specs)
        with pytest.raises(ScenarioError) as from_shape:
            check_suite_shape(compiled)
        with pytest.raises(SpecError) as from_spec:
            compile_spec(SpecSuite(specs), 5)
        assert str(from_spec.value) == str(from_shape.value)


def test_an_uncompilable_tape_is_reported_before_the_suite_is_counted():
    # Named ordering change (#320): compile_spec compiles before it counts, so a 9-tape suite
    # with one uncompilable tape reports that tape's compile error, not `want 2-8`.
    tiny = ScenarioSpec(name="tiny", legs=[_leg("trend", 5)], behavior=Behavior.ENTER_LONG, leg=0)
    suite = SpecSuite(
        [tiny]
        + [_directional_spec("trend", Behavior.ENTER_LONG, name=f"s{i}") for i in range(7)]
        + [_never_trade_spec()]
    )
    with pytest.raises(SpecError) as excinfo:
        compile_spec(suite, 0)
    assert "compiles to" in str(excinfo.value)
    assert "2-8" not in str(excinfo.value)


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


# ── the model dialect: spec_from_payload, the tolerant parse of the FORMULATE emit (#83) ─────
# One scenario the model could emit, and its never-trade partner — the two dialect fixtures.
_RALLY_PAYLOAD = {
    "name": "rally",
    "legs": [{"kind": "trend", "bars": 60, "pct": 0.05}],
    "behavior": "enter_long_during_leg",
    "leg": 0,
}
_GRIND_PAYLOAD = {
    "name": "grind",
    "legs": [{"kind": "flat", "bars": 60}],
    "behavior": "never_trade",
}

# Every refusal sentence, copied verbatim from the parse. These messages ride into the FORMULATE
# corrective the model reads on a re-prompt, so the wording is contract, not diagnostics: pinning
# it here is what makes "the parse moved without changing what the model is told" a checked claim.
_REFUSALS = [
    pytest.param(
        {"scenarios": [{"name": "rally", "legs": ["nope"], "behavior": "never_trade"}]},
        "scenario 'rally' leg 0: each leg must be an object",
        id="leg-not-an-object",
    ),
    pytest.param(
        {"scenarios": [{"name": "rally", "legs": [{"bars": 60}], "behavior": "never_trade"}]},
        "scenario 'rally' leg 0: a leg 'kind' is required",
        id="leg-kind-missing",
    ),
    pytest.param(
        {
            "scenarios": [
                {
                    "name": "rally",
                    "legs": [{"kind": "flat", "bars": "60"}],
                    "behavior": "never_trade",
                }
            ]
        },
        "scenario 'rally' leg 0: 'bars' must be an integer length (0 for gap)",
        id="bars-not-an-integer",
    ),
    pytest.param(
        {"scenarios": [{"name": "rally", "legs": [{"kind": "flat", "bars": 60}]}]},
        "scenario 'rally': a 'behavior' tag is required",
        id="behavior-missing",
    ),
    pytest.param(
        {
            "scenarios": [
                {"name": "rally", "legs": [{"kind": "flat", "bars": 60}], "behavior": "mystery"}
            ]
        },
        "scenario 'rally': unknown behavior 'mystery'; use one of enter_long_during_leg, "
        "enter_short_during_leg, hold_long_through_leg, hold_short_through_leg, "
        "flat_by_end_of_leg, never_trade",
        id="behavior-unknown",
    ),
    pytest.param(
        {"scenarios": ["nope"]},
        "scenario 0: each scenario must be an object",
        id="scenario-not-an-object",
    ),
    pytest.param(
        {"scenarios": [{"legs": [{"kind": "flat", "bars": 60}], "behavior": "never_trade"}]},
        "scenario 0: a non-empty 'name' is required",
        id="name-missing",
    ),
    pytest.param(
        {"scenarios": [{"name": "rally", "behavior": "never_trade"}]},
        "scenario 'rally': a non-empty 'legs' list is required",
        id="legs-missing",
    ),
    pytest.param(
        {"scenarios": [{"name": "rally", "legs": [], "behavior": "never_trade"}]},
        "scenario 'rally': a non-empty 'legs' list is required",
        id="legs-empty",
    ),
    pytest.param(
        {
            "scenarios": [
                {
                    "name": "rally",
                    "legs": [{"kind": "flat", "bars": 60}],
                    "behavior": "never_trade",
                    "leg": "0",
                }
            ]
        },
        "scenario 'rally': 'leg' must be an integer leg index or omitted",
        id="leg-reference-not-an-integer",
    ),
    pytest.param(
        ["scenarios"],
        "scenario_spec must be an object with a 'scenarios' list",
        id="spec-not-an-object",
    ),
    pytest.param(
        {},
        "scenario_spec.scenarios must be a non-empty list of scenarios",
        id="scenarios-missing",
    ),
    pytest.param(
        {"scenarios": []},
        "scenario_spec.scenarios must be a non-empty list of scenarios",
        id="scenarios-empty",
    ),
]


def test_parse_warm_is_zero_so_a_parse_time_compile_checks_the_shape_only():
    # FORMULATE compiles at parse time purely to prove the spec is structurally valid; the write
    # gate re-compiles at the candidate's real declared warmup. Zero, so a directional entry leg
    # begins right after the setup pad.
    assert PARSE_WARM == 0


def test_spec_from_payload_parses_the_model_dialect_into_the_frozen_suite():
    suite = spec_from_payload({"scenarios": [_RALLY_PAYLOAD, _GRIND_PAYLOAD]})
    assert suite == SpecSuite(
        [
            ScenarioSpec("rally", [LegSpec("trend", 60, pct=0.05)], Behavior.ENTER_LONG, leg=0),
            ScenarioSpec("grind", [LegSpec("flat", 60)], Behavior.NEVER_TRADE),
        ]
    )


def test_a_parsed_payload_compiles_at_the_parse_warmup():
    # The two halves the FORMULATE boundary puts together: the tolerant parse, then the compile at
    # PARSE_WARM that proves the emitted spec is structurally valid.
    suite = spec_from_payload({"scenarios": [_RALLY_PAYLOAD, _GRIND_PAYLOAD]})
    compiled = compile_spec(suite, PARSE_WARM)
    assert [s.name for s in compiled] == ["rally", "grind"]


def test_spec_from_payload_defaults_the_omitted_shape_params_to_zero():
    # pct/amplitude/period are ignored per kind by the compiler, so a model that omits them is
    # emitting a valid leg, not a malformed one.
    suite = spec_from_payload({"scenarios": [_GRIND_PAYLOAD]})
    leg = suite.scenarios[0].legs[0]
    assert (leg.pct, leg.amplitude, leg.period) == (0.0, 0.0, 0)


def test_spec_from_payload_reads_an_omitted_leg_reference_as_none():
    suite = spec_from_payload({"scenarios": [_GRIND_PAYLOAD]})
    assert suite.scenarios[0].leg is None


@pytest.mark.parametrize("tag", list(Behavior))
def test_spec_from_payload_reads_the_behavior_tag_by_value(tag):
    # The model dialect speaks the tag's *value* ("enter_long_during_leg"), unlike the
    # machine-exact JSON carrier, which speaks its name.
    suite = spec_from_payload(
        {
            "scenarios": [
                {
                    "name": "rally",
                    "legs": [{"kind": "flat", "bars": 60}, {"kind": "flat", "bars": 60}],
                    "behavior": tag.value,
                    "leg": 0,
                }
            ]
        }
    )
    assert suite.scenarios[0].behavior is tag


def test_a_gap_leg_carries_zero_bars_and_parses():
    suite = spec_from_payload(
        {
            "scenarios": [
                {
                    "name": "rally",
                    "legs": [
                        {"kind": "gap", "bars": 0, "pct": 0.05},
                        {"kind": "trend", "bars": 60},
                    ],
                    "behavior": "enter_long_during_leg",
                    "leg": 1,
                }
            ]
        }
    )
    assert suite.scenarios[0].legs[0] == LegSpec("gap", 0, pct=0.05)


@pytest.mark.parametrize(("raw", "message"), _REFUSALS)
def test_a_malformed_payload_is_refused_with_its_exact_sentence(raw, message):
    with pytest.raises(SpecError) as exc:
        spec_from_payload(raw)
    assert str(exc.value) == message


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("bars", "scenario 'rally' leg 0: 'bars' must be an integer length (0 for gap)"),
        ("leg", "scenario 'rally': 'leg' must be an integer leg index or omitted"),
    ],
)
def test_true_is_not_an_integer_length_or_leg_reference(field, message):
    # bool is an int in Python; a model emitting `true` where a count belongs is a schema misfire,
    # not a leg of length 1.
    leg = {"kind": "flat", "bars": 60}
    scenario = {"name": "rally", "legs": [leg], "behavior": "never_trade"}
    if field == "bars":
        leg["bars"] = True
    else:
        scenario["leg"] = True
    with pytest.raises(SpecError) as exc:
        spec_from_payload({"scenarios": [scenario]})
    assert str(exc.value) == message


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
