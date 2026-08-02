"""Degenerate-pass detectors (#224): a file that cleared the write gate by collapsing.

The gate says pass/fail; these two detectors say *how* a pass was won. Every fixture here is a real
:class:`~noctis.strategies.base.TraderStrategy` subclass built to sit on one side of one named
threshold, and every assertion is external behaviour — the finding a reading carries, the evidence
its summary names, the numbers the reading reports. The fixtures come in pairs on purpose: one that
trips a detector and one built to *just* miss it, because a detector nobody has shown the near side
of is a threshold nobody has checked.

The activity measurement is deliberately not re-derived here: the tests replay through
:func:`noctis.strategies.library.fixture_frame` + :func:`noctis.strategies.base.replay_targets` —
the two functions the write gate itself replays with — and assert the detector reports exactly that
series' shape, which is what makes "no second replay implementation" a checked claim.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

import noctis.eval.coder_detectors as detectors_module
from noctis.eval import metrics as metrics_module
from noctis.eval.coder_detectors import (
    COLLAPSE_SHARE,
    FLOOR_BAND,
    MIN_NONFLAT_FRACTION,
    PARAM_FLOOR_COLLAPSE,
    SEVERITY,
    TRIVIAL_TARGET,
    DegenerateFinding,
    inspect_strategy,
    read_param_floors,
    read_target_activity,
)
from noctis.eval.metrics import AttemptOutcome, CaseResult
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy, replay_targets
from noctis.strategies.library import fixture_frame

MODULE_SOURCE = Path(detectors_module.__file__)


# ── fixture strategies: one per side of each threshold ─────────────────────────────────────
class _Counting(TraderStrategy):
    """A strategy whose target schedule is a declared bar arithmetic — the honest way to build a
    fixture that lands on a chosen side of the activity threshold."""

    every: int = 1
    warmup: int = 0

    @dataclass(frozen=True)
    class Params:
        lookback: int = 10

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        self._index = -1

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._index += 1
        if self._index < self.warmup:
            ctx.set_target(0)
            return
        ctx.set_target(1 if self._index % self.every == 0 else 0)

    @classmethod
    def warmup_bars(cls, params) -> int:
        return cls.warmup

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [ParamSpec("lookback", "int", low=0, high=100, step=1)]


class TrivialTarget(_Counting):
    """A directional target on 2 of the 180 fixture bars — 1.1%, far under the floor."""

    name = "trivial_target"
    every = 90


class JustActive(_Counting):
    """A directional target on 10 of the 180 fixture bars — 5.6%, just over the floor."""

    name = "just_active"
    every = 18


class WarmupThenActive(_Counting):
    """Flat for a declared 100-bar warmup, then directional every other decidable bar."""

    name = "warmup_then_active"
    every = 2
    warmup = 100


class _Floored(TraderStrategy):
    """A strategy whose declared defaults sit wherever a fixture needs them in its own space."""

    name = "floored"
    dimensions: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class Params:
        alpha: int = 0
        beta: int = 0
        gamma: int = 0
        delta: int = 0

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        return None

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        ctx.set_target(1)

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return _space(*cls.dimensions)


def _space(*names: str) -> list[ParamSpec]:
    """An ordered [0, 100] dimension per name, so a value reads as its own percent position."""
    return [ParamSpec(name, "int", low=0, high=100, step=1) for name in names]


def _replayed(cls: type[TraderStrategy]) -> Sequence[int]:
    """The gate's own replay of ``cls`` over the gate's own fixture tape."""
    return replay_targets(cls(cls.params_cls()), fixture_frame())


# ── param-floor collapse ──────────────────────────────────────────────────────────────────
def test_defaults_at_the_bottom_of_every_declared_dimension_are_flagged_as_floor_collapse() -> None:
    reading = read_param_floors(
        {"alpha": 0, "beta": 0, "gamma": 0}, _space("alpha", "beta", "gamma")
    )

    assert reading.finding is not None
    assert reading.finding.detector == PARAM_FLOOR_COLLAPSE
    assert reading.share == 1.0


def test_defaults_spread_through_the_middle_of_their_declared_space_are_not_flagged() -> None:
    reading = read_param_floors(
        {"alpha": 27, "beta": 50, "gamma": 66}, _space("alpha", "beta", "gamma")
    )

    assert reading.finding is None
    assert reading.floored == ()
    assert reading.share == 0.0


def test_a_dimension_exactly_on_the_floor_band_counts_as_floored() -> None:
    reading = read_param_floors({"alpha": 10}, _space("alpha"))

    assert [dimension.name for dimension in reading.floored] == ["alpha"]
    assert reading.dimensions[0].position == pytest.approx(FLOOR_BAND)


def test_a_dimension_one_step_above_the_floor_band_does_not_count_as_floored() -> None:
    reading = read_param_floors({"alpha": 11}, _space("alpha"))

    assert reading.floored == ()
    assert reading.finding is None


def test_two_thirds_of_the_dimensions_on_the_floor_just_misses_the_collapse_share() -> None:
    """Two of three floored is 67%, under the 75% share — one low knob is a design choice."""
    reading = read_param_floors(
        {"alpha": 0, "beta": 2, "gamma": 80}, _space("alpha", "beta", "gamma")
    )

    assert reading.share == pytest.approx(2 / 3)
    assert reading.share < COLLAPSE_SHARE
    assert reading.finding is None


def test_three_quarters_of_the_dimensions_on_the_floor_reaches_the_collapse_share() -> None:
    reading = read_param_floors(
        {"alpha": 0, "beta": 2, "gamma": 5, "delta": 80}, _space("alpha", "beta", "gamma", "delta")
    )

    assert reading.share == pytest.approx(COLLAPSE_SHARE)
    assert reading.finding is not None


def test_the_floor_finding_names_every_floored_param_its_value_and_its_declared_range() -> None:
    reading = read_param_floors({"alpha": 0, "beta": 5}, _space("alpha", "beta"))

    assert reading.finding is not None
    summary = reading.finding.summary
    assert "alpha" in summary
    assert "beta" in summary
    assert "[0, 100]" in summary
    assert "2 of 2" in summary


def test_a_categorical_dimension_has_no_floor_and_is_left_out_of_the_measurement() -> None:
    """A choice list has no bottom to collapse to, so it neither trips nor dilutes the share."""
    space = [*_space("alpha"), ParamSpec("mode", "categorical", choices=("fast", "slow"))]

    reading = read_param_floors({"alpha": 0, "mode": "fast"}, space)

    assert [dimension.name for dimension in reading.dimensions] == ["alpha"]
    assert reading.finding is not None


def test_a_zero_width_dimension_is_pinned_not_floored_and_is_left_out_of_the_measurement() -> None:
    space = [*_space("alpha"), ParamSpec("beta", "int", low=7, high=7, step=1)]

    reading = read_param_floors({"alpha": 50, "beta": 7}, space)

    assert [dimension.name for dimension in reading.dimensions] == ["alpha"]
    assert reading.finding is None


def test_a_param_space_with_nothing_ordered_to_measure_reports_no_share_and_no_finding() -> None:
    reading = read_param_floors(
        {"mode": "fast"}, [ParamSpec("mode", "categorical", choices=("a",))]
    )

    assert reading.dimensions == ()
    assert reading.share is None
    assert reading.finding is None


def test_a_declared_dimension_absent_from_the_defaults_is_not_measured() -> None:
    reading = read_param_floors({"alpha": 0}, _space("alpha", "missing"))

    assert [dimension.name for dimension in reading.dimensions] == ["alpha"]


def test_a_frozen_params_dataclass_is_read_the_same_way_a_mapping_is() -> None:
    from_mapping = read_param_floors({"alpha": 0, "beta": 0}, _space("alpha", "beta"))
    from_dataclass = read_param_floors(_Floored.Params(alpha=0, beta=0), _space("alpha", "beta"))

    assert from_dataclass.share == from_mapping.share
    assert from_dataclass.finding == from_mapping.finding


# ── trivial-target degeneracy ─────────────────────────────────────────────────────────────
def test_a_strategy_that_almost_never_takes_a_direction_is_flagged_as_a_trivial_target() -> None:
    report = inspect_strategy(TrivialTarget)

    assert report.target_activity.finding is not None
    assert report.target_activity.finding.detector == TRIVIAL_TARGET
    assert report.target_activity.nonflat == 2


def test_a_strategy_just_over_the_activity_floor_is_not_flagged() -> None:
    report = inspect_strategy(JustActive)

    assert report.target_activity.fraction == pytest.approx(10 / 180)
    assert report.target_activity.fraction > MIN_NONFLAT_FRACTION
    assert report.target_activity.finding is None


def test_an_activity_fraction_exactly_on_the_floor_is_not_flagged() -> None:
    """The threshold is a floor, not a target: measuring exactly at it is clearing it."""
    targets = [1] * 5 + [0] * 95

    reading = read_target_activity(targets)

    assert reading.fraction == pytest.approx(MIN_NONFLAT_FRACTION)
    assert reading.finding is None


def test_an_activity_fraction_one_bar_under_the_floor_is_flagged() -> None:
    targets = [1] * 4 + [0] * 96

    reading = read_target_activity(targets)

    assert reading.finding is not None
    assert reading.finding.detector == TRIVIAL_TARGET


def test_short_targets_count_toward_activity_the_same_way_long_ones_do() -> None:
    """Direction, not side: a short-only file is active, and flagging it would be a bug."""
    reading = read_target_activity([-1] * 50 + [0] * 50)

    assert reading.nonflat == 50
    assert reading.finding is None


def test_the_declared_warmup_bars_are_left_out_of_the_activity_denominator() -> None:
    """A file promises to be flat through its warmup; counting that promise as inactivity would
    punish a long-lookback thesis for keeping it."""
    report = inspect_strategy(WarmupThenActive)

    assert report.target_activity.warmup == 100
    assert report.target_activity.bars == 80
    assert report.target_activity.fraction == pytest.approx(40 / 80)
    assert report.target_activity.finding is None


def test_a_warmup_that_swallows_the_whole_tape_leaves_the_activity_unmeasured() -> None:
    reading = read_target_activity([0] * 10, warmup=10)

    assert reading.bars == 0
    assert reading.fraction is None
    assert reading.finding is None


def test_an_empty_target_series_leaves_the_activity_unmeasured() -> None:
    reading = read_target_activity([])

    assert reading.fraction is None
    assert reading.finding is None


def test_the_trivial_target_finding_names_the_measured_fraction_and_the_threshold() -> None:
    reading = read_target_activity([1] * 2 + [0] * 178)

    assert reading.finding is not None
    summary = reading.finding.summary
    assert "2 of 180" in summary
    assert f"{MIN_NONFLAT_FRACTION:.1%}" in summary


# ── the measurement runs the gate's own replay ────────────────────────────────────────────
def test_the_measured_activity_is_the_gate_fixture_replay_bar_for_bar() -> None:
    """Hand-check against the write gate's own tape and its own replay function: the detector may
    not carry a second opinion about what a strategy did on a bar."""
    targets = _replayed(TrivialTarget)

    report = inspect_strategy(TrivialTarget)

    assert report.target_activity.bars == len(targets)
    assert report.target_activity.nonflat == sum(1 for target in targets if target != 0)


def test_the_convenience_measures_the_class_declared_defaults_and_declared_space() -> None:
    class Collapsed(_Floored):
        name = "collapsed"
        dimensions = ("alpha", "beta")

    report = inspect_strategy(Collapsed)

    assert report.strategy == "collapsed"
    assert [finding.detector for finding in report.findings] == [PARAM_FLOOR_COLLAPSE]


def test_a_report_carries_both_findings_when_a_file_is_degenerate_on_both_axes() -> None:
    class Collapsed(TrivialTarget):
        name = "collapsed_and_idle"

        @classmethod
        def param_space(cls) -> list[ParamSpec]:
            return _space("lookback")

    # lookback defaults to 10 in a [0, 100] space — exactly on the floor band.
    report = inspect_strategy(Collapsed)

    assert sorted(finding.detector for finding in report.findings) == sorted(
        (PARAM_FLOOR_COLLAPSE, TRIVIAL_TARGET)
    )


@pytest.mark.parametrize("seed", ["sma_crossover", "rsi_meanrev", "donchian_breakout"])
def test_a_shipped_seed_strategy_trips_neither_detector(seed: str) -> None:
    """The conservatism claim, checked against every shipped seed rather than asserted in prose."""
    report = inspect_strategy(_seed_class(seed))

    assert report.findings == ()


def _seed_class(name: str) -> type[TraderStrategy]:
    """The shipped seed of that name, loaded from the committed read-only library."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "strategies" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"seed_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, TraderStrategy) and obj is not TraderStrategy
    ]
    return found[0]


# ── findings are warnings, and structurally cannot become a score ─────────────────────────
def test_every_finding_a_detector_emits_carries_the_warning_severity() -> None:
    floors = read_param_floors({"alpha": 0}, _space("alpha"))
    activity = read_target_activity([0] * 100)

    assert floors.finding is not None and floors.finding.severity == SEVERITY
    assert activity.finding is not None and activity.finding.severity == SEVERITY
    assert SEVERITY == "WARNING"


def test_a_finding_carries_evidence_only_and_no_pass_or_cost_field() -> None:
    fields = {field.name for field in dataclasses.fields(DegenerateFinding)}

    assert fields == {"detector", "severity", "summary"}


def test_a_finding_renders_as_one_line_leading_with_its_severity_and_detector() -> None:
    finding = DegenerateFinding(detector=TRIVIAL_TARGET, severity=SEVERITY, summary="2 of 180")

    assert finding.line() == "WARNING trivial_target: 2 of 180"


def test_defaults_that_are_neither_a_mapping_nor_a_declared_params_object_are_refused() -> None:
    """Better a refusal than a silently empty reading, which reads exactly like a clean file."""
    with pytest.raises(TypeError):
        read_param_floors("alpha=0", _space("alpha"))


def test_no_bench_metric_input_accepts_a_finding_as_a_field() -> None:
    """Score-inertness, structurally: the types the metrics arithmetic consumes have no seat for a
    detector finding, so no wiring mistake can slide one into a pass rate or a bill."""
    finding = DegenerateFinding(detector=TRIVIAL_TARGET, severity=SEVERITY, summary="x")

    with pytest.raises(TypeError):
        AttemptOutcome(passed=True, finding=finding)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CaseResult(case_id="c", finding=finding)  # type: ignore[call-arg]


def test_no_type_in_the_metrics_module_names_a_detector_type() -> None:
    exported = set(detectors_module.__all__)
    annotated = {
        str(field.type)
        for name in dir(metrics_module)
        if dataclasses.is_dataclass(declared := getattr(metrics_module, name))
        for field in dataclasses.fields(declared)
    }

    for annotation in annotated:
        assert not (exported & set(annotation.replace("|", " ").split())), annotation


def test_the_metrics_module_does_not_import_the_detectors() -> None:
    imported = _imported_modules(Path(metrics_module.__file__))

    assert detectors_module.__name__ not in imported


def test_the_detectors_import_no_scoring_surface() -> None:
    """The other direction: a module that cannot see a scorer cannot feed one."""
    imported = _imported_modules(MODULE_SOURCE)

    assert not {name for name in imported if name.startswith("noctis.eval")}


# ── purity of the detector core ───────────────────────────────────────────────────────────
def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _referenced_names(source: Path) -> set[str]:
    """Every bare name and attribute the *code* touches — prose in a docstring is not code."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    touched: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            touched.add(node.id)
        elif isinstance(node, ast.Attribute):
            touched.add(node.attr)
    return touched


def test_the_detector_module_reaches_no_io_no_clock_and_no_randomness() -> None:
    """The same structural rule the bench metrics are held to: every number the detectors report is
    a function of the values handed in, so a fixture reproduces it exactly."""
    roots = {name.split(".")[0] for name in _imported_modules(MODULE_SOURCE)}

    assert roots <= {"__future__", "collections", "dataclasses", "typing", "noctis"}
    touched = _referenced_names(MODULE_SOURCE)
    forbidden = {"open", "random", "os", "time", "datetime", "now", "utcnow", "Path", "read_text"}

    assert not (touched & forbidden), sorted(touched & forbidden)


def test_the_only_engine_modules_the_detectors_reach_for_are_the_gates_own_replay() -> None:
    """The measurement path is the write gate's, named in the imports: the tape and the replay."""
    engine = {name for name in _imported_modules(MODULE_SOURCE) if name.startswith("noctis")}

    assert engine == {"noctis.strategies.base", "noctis.strategies.library"}


def test_the_thresholds_are_named_constants_the_module_documents() -> None:
    """A magic number in a detector is an unargued opinion; each threshold states its rationale."""
    docstring = " ".join((detectors_module.__doc__ or "").split())

    assert 0.0 < FLOOR_BAND <= 0.2
    assert 0.5 < COLLAPSE_SHARE <= 1.0
    assert 0.0 < MIN_NONFLAT_FRACTION <= 0.1
    assert f"{FLOOR_BAND:.0%}" in docstring
    assert f"{COLLAPSE_SHARE:.0%}" in docstring
    assert f"{MIN_NONFLAT_FRACTION:.0%}" in docstring
