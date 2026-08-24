"""The coder case schema (#217): a brief-shaped payload, a bucket, and seven difficulty axes.

A coder case is a benchmark job that must be indistinguishable in shape from a real authoring job,
so the assertions here are about *mirroring production*: the payload's fields are the fields
:class:`~noctis.research.author.StrategyBrief` declares (the mirror test below breaks the build if
either side drifts), the optional fixed oracle is parsed by the same code path FORMULATE's emit is,
and nothing anywhere holds an expected output — the write gate is the expectation.

Everything else asserted here is a refusal. A mislabelled or half-written case does not fail loudly
at benchmark time; it silently skews a number, which is why validation is strict, one-pass, and
names the file and every defect at once.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from noctis.eval.case import MalformedCase, Provenance, ProvenanceKind, Split
from noctis.eval.case_provider import YamlCaseProvider
from noctis.eval.coder_case import (
    AXIS_LEVELS,
    BRIEF_KEYS,
    REQUIRED_BRIEF_KEYS,
    Axis,
    Bucket,
    CoderPayload,
    coder_payload,
    parse_coder_case,
)
from noctis.research.author import StrategyBrief
from noctis.strategies.scenario_spec import Behavior, LegSpec, ScenarioSpec, SpecSuite

SCHEMA_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/eval/coder_case.py"

# A run id of the shape the engine mints, so a mined coder case rides the real provenance form.
RUN_ID = "20260720T144233Z-a3f9c1"

# A fixed oracle a real FORMULATE could have emitted: one directional entry tape, one no-trade
# tape, written in the #82 vocabulary as JSON/YAML data (never bar indices).
SPEC_DOCUMENT: dict[str, Any] = {
    "scenarios": [
        {
            "name": "rally",
            "legs": [{"kind": "trend", "bars": 60, "pct": 0.05}],
            "behavior": "enter_long_during_leg",
            "leg": 0,
        },
        {
            "name": "selloff_stays_flat",
            "legs": [{"kind": "selloff", "bars": 60, "pct": 0.05}],
            "behavior": "never_trade",
        },
    ]
}

SPEC_SUITE = SpecSuite(
    scenarios=(
        ScenarioSpec("rally", (LegSpec("trend", 60, pct=0.05),), Behavior.ENTER_LONG, leg=0),
        ScenarioSpec(
            "selloff_stays_flat", (LegSpec("selloff", 60, pct=0.05),), Behavior.NEVER_TRADE
        ),
    )
)

AXES = {
    "composition_mode": "reference",
    "oracle_mode": "fixed_spec",
    "warmup_arithmetic": "single",
    "state_complexity": "rolling",
    "no_trade_tape": "falsified",
    "param_space_breadth": "narrow",
    "api_surface": "indicators",
}


def _payload(**overrides: Any) -> dict[str, Any]:
    """A complete, coherent coder payload; keyword arguments replace one field each."""
    payload: dict[str, Any] = {
        "thesis": "A 20-day breakout re-prices faster than the spread it pays to cross.",
        "entry_exit": "Enter long on a close above the prior 20-day high; exit on a 10-day low.",
        "param_space": "lookback 10-40, exit_lookback 5-20",
        "scenarios": "rally: trend(60) — enter long during leg 0; selloff_stays_flat: never trade",
        "reference": "donchian_breakout",
        "style": "momentum",
        "symbols": ["AAPL", "MSFT"],
        "scenario_spec": SPEC_DOCUMENT,
    }
    payload.update(overrides)
    return {key: value for key, value in payload.items() if value is not _ABSENT}


class _Absent:
    """A sentinel that removes a key from the fixture rather than setting it to ``None``."""


_ABSENT = _Absent()


def _document(**overrides: Any) -> dict[str, Any]:
    """A complete, well-formed coder case document; keyword arguments replace one key each."""
    document: dict[str, Any] = {
        "site_id": "coder",
        "payload": _payload(),
        "provenance": f"mined:{RUN_ID}",
        "tags": ["bucket:field", "breakout"],
        "difficulty": dict(AXES),
        "split": "tuning",
    }
    document.update(overrides)
    return document


def _write(tmp_path: Path, document: Any, name: str = "donchian_reprice") -> Path:
    """Write one coder case into the site directory the provider reads."""
    directory = tmp_path / "coder"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


def _load(tmp_path: Path, **overrides: Any) -> CoderPayload:
    """Round-trip one case document through the real provider into a typed coder payload."""
    path = _write(tmp_path, _document(**overrides))
    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")
    return coder_payload(case, source=str(path))


def _refusal(tmp_path: Path, **overrides: Any) -> tuple[str, Path]:
    """The refusal message a defective case earns, beside the file path it must name."""
    path = _write(tmp_path, _document(**overrides))
    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")
    with pytest.raises(MalformedCase) as refusal:
        coder_payload(case, source=str(path))
    return str(refusal.value), path


# ── what a coder case carries ─────────────────────────────────────────────────────────────


def test_a_coder_case_file_round_trips_through_the_provider_into_a_typed_brief(tmp_path):
    payload = _load(tmp_path)

    assert payload.brief == StrategyBrief(
        thesis="A 20-day breakout re-prices faster than the spread it pays to cross.",
        entry_exit="Enter long on a close above the prior 20-day high; exit on a 10-day low.",
        param_space="lookback 10-40, exit_lookback 5-20",
        scenarios="rally: trend(60) — enter long during leg 0; selloff_stays_flat: never trade",
        reference="donchian_breakout",
        style="momentum",
        symbols=("AAPL", "MSFT"),
    )


def test_the_optional_fixed_oracle_round_trips_into_the_production_spec_suite(tmp_path):
    payload = _load(tmp_path)

    assert payload.scenario_spec == SPEC_SUITE


def test_a_case_carries_its_bucket_its_seven_axes_its_provenance_and_its_split(tmp_path):
    payload = _load(tmp_path)

    assert payload.case_id == "donchian_reprice"
    assert payload.bucket is Bucket.FIELD
    assert dict(payload.difficulty) == AXES
    assert payload.provenance == Provenance(kind=ProvenanceKind.MINED, reference=RUN_ID)
    assert payload.split is Split.TUNING


def test_the_bucket_tag_is_consumed_and_the_remaining_tags_ride_along(tmp_path):
    payload = _load(tmp_path, tags=["bucket:canary", "breakout", "seed"])

    assert (payload.bucket, payload.tags) == (Bucket.CANARY, ("breakout", "seed"))


def test_an_axis_level_reads_back_by_its_typed_axis_name(tmp_path):
    payload = _load(tmp_path)

    assert payload.level(Axis.NO_TRADE_TAPE) == "falsified"


def test_a_case_with_no_split_declared_loads_unassigned_so_a_corpus_can_freeze_one(tmp_path):
    document = _document()
    del document["split"]
    _write(tmp_path, document)

    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    assert coder_payload(case).split is None


def test_a_case_that_declares_no_scenario_spec_leaves_the_coder_owning_its_tapes(tmp_path):
    payload = _load(
        tmp_path,
        payload=_payload(scenario_spec=_ABSENT),
        difficulty={**AXES, "oracle_mode": "authored"},
    )

    assert payload.scenario_spec is None


def test_the_optional_brief_fields_default_the_way_the_production_brief_defaults(tmp_path):
    payload = _load(
        tmp_path,
        payload=_payload(reference=_ABSENT, style=_ABSENT, symbols=_ABSENT),
        difficulty={**AXES, "composition_mode": "scratch"},
    )

    assert (payload.brief.reference, payload.brief.style, payload.brief.symbols) == (None, None, ())


def test_a_coder_payload_refuses_field_assignment(tmp_path):
    payload = _load(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        payload.bucket = Bucket.EDGE


def test_the_axes_of_a_loaded_payload_are_read_only(tmp_path):
    payload = _load(tmp_path)

    with pytest.raises(TypeError):
        payload.difficulty["oracle_mode"] = "authored"


# ── the mirror: the payload is the production brief ───────────────────────────────────────


def test_the_declared_brief_keys_mirror_the_production_strategy_brief_fields():
    """Drift in either direction breaks the build: a benchmark job is shaped like a real one."""
    assert set(BRIEF_KEYS) == {field.name for field in dataclasses.fields(StrategyBrief)}


def test_the_required_brief_keys_are_the_production_brief_fields_that_have_no_default():
    required = {
        field.name
        for field in dataclasses.fields(StrategyBrief)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }

    assert set(REQUIRED_BRIEF_KEYS) == required


def test_a_loaded_brief_is_the_production_brief_type_the_author_engine_consumes(tmp_path):
    assert isinstance(_load(tmp_path).brief, StrategyBrief)


# ── no expected output, ever ──────────────────────────────────────────────────────────────


def test_the_coder_payload_type_declares_no_expected_output_field():
    names = {field.name for field in dataclasses.fields(CoderPayload)}

    assert not [
        name
        for name in names
        for banned in ("expect", "gold", "answer", "solution", "reference_output")
        if banned in name
    ]


@pytest.mark.parametrize(
    "key",
    [
        "expected_output",
        "solution",
        "gold",
        "oracle",
        "expected_source",
        "expected_code",
        "reference_solution",
        "reference_implementation",
        "model_answer",
    ],
)
def test_a_payload_carrying_an_expected_output_key_is_refused_by_that_key_name(tmp_path, key):
    message, path = _refusal(tmp_path, payload=_payload(**{key: "class Foo(TraderStrategy): ..."}))

    assert key in message
    assert "oracle" in message
    assert str(path) in message


def test_a_coder_specific_expected_output_key_at_the_top_level_is_refused_by_name(tmp_path):
    with pytest.raises(MalformedCase) as refusal:
        parse_coder_case(
            _document(expected_source="class Foo: ..."),
            case_id="leaky",
            source="cases/coder/leaky.yaml",
        )

    assert "expected_source" in str(refusal.value)
    assert "oracle" in str(refusal.value)


# ── refusals: the brief ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["thesis", "entry_exit", "param_space", "scenarios"])
def test_a_missing_brief_field_is_refused_naming_the_file_and_the_field(tmp_path, missing):
    message, path = _refusal(tmp_path, payload=_payload(**{missing: _ABSENT}))

    assert missing in message
    assert str(path) in message


@pytest.mark.parametrize("value", ["", "   ", 7, None, ["a list"]])
def test_a_brief_field_that_is_not_a_non_empty_string_is_refused_naming_it(tmp_path, value):
    message, _ = _refusal(tmp_path, payload=_payload(thesis=value))

    assert "thesis" in message


@pytest.mark.parametrize("symbols", ["AAPL", [""], [1], {"a": "b"}])
def test_brief_symbols_that_are_not_a_list_of_tickers_are_refused(tmp_path, symbols):
    message, _ = _refusal(tmp_path, payload=_payload(symbols=symbols))

    assert "symbols" in message


def test_an_unknown_payload_key_is_refused_beside_the_keys_a_coder_payload_declares(tmp_path):
    message, _ = _refusal(tmp_path, payload=_payload(hint="use a 20-day window"))

    assert "hint" in message
    assert "entry_exit" in message


# ── refusals: the bucket ──────────────────────────────────────────────────────────────────


def test_an_unknown_bucket_is_refused_naming_the_value_and_the_four_declared_buckets(tmp_path):
    message, _ = _refusal(tmp_path, tags=["bucket:smoke"])

    assert "smoke" in message
    assert all(bucket.value in message for bucket in Bucket)


def test_a_case_declaring_no_bucket_tag_is_refused_naming_the_four_declared_buckets(tmp_path):
    message, _ = _refusal(tmp_path, tags=["breakout"])

    assert "bucket" in message
    assert all(bucket.value in message for bucket in Bucket)


def test_a_case_declaring_two_bucket_tags_is_refused(tmp_path):
    message, _ = _refusal(tmp_path, tags=["bucket:field", "bucket:canary"])

    assert "bucket:field" in message and "bucket:canary" in message


# ── refusals: the seven axes ──────────────────────────────────────────────────────────────


def test_a_missing_difficulty_axis_is_refused_naming_the_axis(tmp_path):
    axes = dict(AXES)
    del axes["warmup_arithmetic"]

    message, _ = _refusal(tmp_path, difficulty=axes)

    assert "warmup_arithmetic" in message


def test_an_unknown_difficulty_axis_is_refused_naming_it_beside_the_seven(tmp_path):
    message, _ = _refusal(tmp_path, difficulty={**AXES, "vibes": "high"})

    assert "vibes" in message
    assert all(axis.value in message for axis in Axis)


@pytest.mark.parametrize("axis", sorted(axis.value for axis in Axis))
def test_an_unknown_axis_value_is_refused_naming_the_axis_and_its_value_set(tmp_path, axis):
    message, _ = _refusal(tmp_path, difficulty={**AXES, axis: "medium"})

    assert axis in message
    assert "medium" in message
    assert all(level in message for level in AXIS_LEVELS[Axis(axis)])


def test_a_case_that_declares_no_difficulty_axes_at_all_is_refused_naming_all_seven(tmp_path):
    document = _document()
    del document["difficulty"]
    _write(tmp_path, document)
    (case,) = YamlCaseProvider(cases_root=tmp_path).load("coder")

    with pytest.raises(MalformedCase) as refusal:
        coder_payload(case)

    assert all(axis.value in str(refusal.value) for axis in Axis)


# ── refusals: the fixed oracle ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [
        "two tapes, one long one flat",
        {},
        {"scenarios": []},
        {"scenarios": [{"name": "rally", "legs": [{"kind": "moonshot", "bars": 60}]}]},
        {
            "scenarios": [
                {
                    "name": "rally",
                    "legs": [{"kind": "trend", "bars": 60, "pct": 0.05}],
                    "behavior": "go_long_i_guess",
                    "leg": 0,
                }
            ]
        },
        {
            "scenarios": [
                {
                    "name": "rally",
                    "legs": [{"kind": "trend", "bars": 60, "pct": 0.05}],
                    "behavior": "enter_long_during_leg",
                    "leg": 0,
                },
                {
                    "name": "rally_again",
                    "legs": [{"kind": "trend", "bars": 60, "pct": 0.05}],
                    "behavior": "enter_long_during_leg",
                    "leg": 0,
                },
            ]
        },
    ],
)
def test_a_malformed_scenario_spec_is_refused_naming_the_file_and_the_defect(tmp_path, spec):
    message, path = _refusal(tmp_path, payload=_payload(scenario_spec=spec))

    assert "scenario_spec" in message
    assert str(path) in message


def test_a_scenario_spec_that_could_never_compile_is_refused_before_a_benchmark_runs(tmp_path):
    """The oracle takes the strategy layer's own parse and parse-time compile, so a case is a job
    FORMULATE could have emitted — a suite with no no-trade tape is refused here, not at the gate.
    """
    spec = {
        "scenarios": [
            {
                "name": "rally",
                "legs": [{"kind": "trend", "bars": 60, "pct": 0.05}],
                "behavior": "enter_long_during_leg",
                "leg": 0,
            },
            {
                "name": "second_rally",
                "legs": [{"kind": "trend", "bars": 70, "pct": 0.06}],
                "behavior": "hold_long_through_leg",
                "leg": 0,
            },
        ]
    }

    message, _ = _refusal(tmp_path, payload=_payload(scenario_spec=spec))

    assert "never_trade" in message


# ── refusals: a label that disagrees with the payload ─────────────────────────────────────


def test_a_case_labelled_fixed_spec_that_carries_no_scenario_spec_is_refused(tmp_path):
    message, _ = _refusal(tmp_path, payload=_payload(scenario_spec=_ABSENT))

    assert "fixed_spec" in message
    assert "scenario_spec" in message


def test_a_case_labelled_authored_that_carries_a_scenario_spec_is_refused(tmp_path):
    message, _ = _refusal(tmp_path, difficulty={**AXES, "oracle_mode": "authored"})

    assert "authored" in message
    assert "scenario_spec" in message


def test_a_case_labelled_reference_composition_that_names_no_reference_is_refused(tmp_path):
    message, _ = _refusal(tmp_path, payload=_payload(reference=_ABSENT))

    assert "reference" in message


# ── one refusal, every defect ─────────────────────────────────────────────────────────────


def test_every_coder_defect_in_one_file_is_reported_in_a_single_refusal(tmp_path):
    message, _ = _refusal(
        tmp_path,
        payload=_payload(thesis=_ABSENT, scenario_spec="a rally then a selloff"),
        tags=["bucket:smoke"],
        difficulty={**AXES, "state_complexity": "spicy"},
    )

    assert all(
        defect in message for defect in ("thesis", "scenario_spec", "smoke", "state_complexity")
    )


def test_a_generic_defect_and_a_coder_defect_are_reported_in_one_refusal(tmp_path):
    """The two halves of a coder case compose into one read: one edit fixes the file."""
    with pytest.raises(MalformedCase) as refusal:
        parse_coder_case(
            _document(provenance="hand-written", tags=["bucket:smoke"]),
            case_id="messy",
            source="cases/coder/messy.yaml",
        )

    message = str(refusal.value)
    assert "provenance" in message and "smoke" in message
    assert message.startswith("cases/coder/messy.yaml:")


def test_a_valid_document_parses_straight_into_a_typed_payload_without_a_provider():
    payload = parse_coder_case(_document(), case_id="from_memory", source="memory")

    assert (payload.case_id, payload.bucket) == ("from_memory", Bucket.FIELD)


# ── the vocabularies are closed and documented ────────────────────────────────────────────


def test_the_bucket_vocabulary_is_exactly_the_four_declared_names():
    assert [bucket.value for bucket in Bucket] == ["field", "replay", "edge", "canary"]


def test_the_difficulty_vocabulary_is_exactly_the_seven_declared_axes():
    assert [axis.value for axis in Axis] == [
        "composition_mode",
        "oracle_mode",
        "warmup_arithmetic",
        "state_complexity",
        "no_trade_tape",
        "param_space_breadth",
        "api_surface",
    ]


def test_every_axis_declares_a_closed_value_set_of_at_least_two_distinct_levels():
    assert set(AXIS_LEVELS) == set(Axis)
    for axis, levels in AXIS_LEVELS.items():
        assert len(set(levels)) == len(levels) >= 2, axis


def test_every_axis_and_every_level_is_documented_in_the_module():
    """A closed vocabulary nobody can read is a vocabulary curators guess at."""
    text = SCHEMA_SOURCE.read_text(encoding="utf-8")

    for axis, levels in AXIS_LEVELS.items():
        assert axis.value in text
        for level in levels:
            assert level in text


# ── purity, structurally ──────────────────────────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_coder_case_schema_reaches_no_file_and_no_clock():
    """Validation over data: the provider is still the only thing in the layer that reads."""
    assert _imports(SCHEMA_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "types",
        "typing",
        "noctis",
    }
    text = SCHEMA_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("open(", "Path(", "os.", "random", "datetime.now", "utcnow", "today("):
        assert forbidden not in text, forbidden


def test_importing_the_coder_case_schema_never_loads_the_episodic_driver():
    """A case's fixed oracle is validated through the strategy layer, never through the LLM
    driver (#319 D6), so nothing behind this import reaches ``noctis.research.driver``.

    A fresh interpreter, because this process has already imported the driver for other tests:
    the question is what ``import noctis.eval.coder_case`` pulls in on its own.
    """
    probe = "import sys, noctis.eval.coder_case\nsys.exit('noctis.research.driver' in sys.modules)"

    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert completed.returncode == 0, (
        f"importing noctis.eval.coder_case loaded noctis.research.driver{completed.stderr}"
    )
