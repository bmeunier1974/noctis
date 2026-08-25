"""``bench.json`` — the pure builder, the pure validator, the key and the refusal (#201).

Every test here states a :class:`BenchArtifacts` **in memory** and asserts on the dict the builder
returns, exactly as ``tests/test_run_record.py`` does for the run record: the runner collects, the
builder builds, the validator checks, and none of the three reads a file or a clock. The only disk
touch in this file is the committed golden record, which is a fixture rather than an input.

Four properties carry the weight:

* **Derived, never incremented.** Every aggregate is recomputed from the per-case artifacts at
  build time, so rebuilding the same artifacts reproduces the record byte for byte — and the
  record's own ``results`` section is complete enough to recompute the metrics block from.
* **The key is composed verbatim.** Changing any single component moves ``comparable_key``, and
  identical inputs reproduce it exactly (the parametrized sensitivity test below).
* **Different keys refuse a delta.** The side-by-side view carries a loud banner and, structurally,
  no delta fields at all — the paired-stats verb belongs to a later epic; this is the refusal.
* **Absences are honest.** A missing key component is ``null`` (never a blank, never a zero), and a
  figure whose input is unknown stays ``null`` all the way through the comparison.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from noctis.eval.identity import SiteIdentity
from noctis.eval.metrics import AttemptOutcome, CaseResult, compute_metrics
from noctis.eval.record import (
    KEY_COMPONENTS,
    KIND,
    SCHEMA_VERSION,
    BenchArtifacts,
    CaseRun,
    CorpusIdentity,
    EngineStamp,
    ModelConfig,
    build,
    comparable,
    comparable_key,
    side_by_side,
    validate,
)
from noctis.eval.taxonomy import UNCLASSIFIED, FailureClass, FailureTaxonomy
from noctis.research.pricing import ModelPrice, PriceTable

RECORD_SOURCE = Path(__file__).resolve().parents[1] / "src" / "noctis" / "eval" / "record.py"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "bench_record_golden.json"

# One attempt's usage, four fields of a round million each, so a $/Mtok rate reads off the fixture.
MTOK = 1_000_000

# A bench-local price table: one dollar per million tokens on every field, so an attempt of four
# million tokens costs exactly $4 and a reader can check the record with a pencil.
BENCH_TABLE = PriceTable(version="bench-1", prices={"bench/oracle": ModelPrice(1.0, 1.0, 1.0, 1.0)})


def _attempt(
    passed: bool,
    *,
    model: str | None = "bench/oracle",
    seconds: float | None = 2.0,
    tokens: int | None = MTOK,
    error: str | None = None,
) -> AttemptOutcome:
    """One attempt with all four usage fields set to ``tokens`` (or all unrecorded)."""
    usage = dict.fromkeys(
        (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ),
        tokens,
    )
    return AttemptOutcome(passed=passed, model=model, seconds=seconds, error=error, **usage)


SITE = SiteIdentity(
    site_id="coder",
    version="1",
    prompt_asset_groups=("author",),
    prompt_asset_hash="7c1f0d9a4b2e6f83",
)

ENGINE = EngineStamp(
    engine_version=3,
    fingerprint={
        "gates": "f63d47b7b9604ab1",
        "backtest": "3ba3e0bf1c97134f",
        "prompts": "14eb169506a6b5aa",
    },
    noctis_version="0.1.0",
)

CORPUS = CorpusIdentity(version="2026-07", hash="9f2c1d3e4b5a6c7d", case_count=2, split="tuning")

CONFIGS = (ModelConfig(config_id="primary", provider="anthropic", requested_model="claude-opus-4"),)

# The bench: one case that passes cleanly, the same case retried into a pass, and a second case
# that never gets there. Two distinct cases, three reps, five attempts — hand-computable.
RUNS = (
    CaseRun(
        result=CaseResult(case_id="breakout", attempts=(_attempt(True),)),
        config_id="primary",
        served_models=("claude-opus-4-20260501",),
    ),
    CaseRun(
        result=CaseResult(
            case_id="breakout",
            attempts=(_attempt(False, error="strategy did not import"), _attempt(True)),
        ),
        config_id="primary",
        served_models=("claude-opus-4-20260501", "claude-opus-4-20260501"),
    ),
    CaseRun(
        result=CaseResult(
            case_id="meanrev",
            attempts=(
                _attempt(False, error="strategy traded twice in ten years"),
                _attempt(False, error="the model emitted no code block"),
            ),
        ),
        config_id="primary",
        served_models=("claude-opus-4-20260501", "claude-opus-4-20260501"),
    ),
)


def _taxonomy() -> FailureTaxonomy:
    """The coder's toy vocabulary, specific first — registration order is priority order."""
    taxonomy = FailureTaxonomy()
    taxonomy.register(
        "coder",
        (
            FailureClass(name="import_error", matches=lambda error: "import" in error),
            FailureClass(name="inactive", matches=lambda error: "traded twice" in error),
        ),
    )
    return taxonomy


def _artifacts(**overrides) -> BenchArtifacts:
    """The three-rep bench the golden record snapshots."""
    base = dict(
        bench_id="20260801T090000Z-bench01",
        label="coder-baseline",
        site=SITE,
        engine=ENGINE,
        corpus=CORPUS,
        harness_hash="0011223344556677",
        harness_dials={"contract_sheet": True, "worked_example": "as_shipped"},
        runs=RUNS,
        started_utc="2026-08-01T09:00:00.000Z",
        finished_utc="2026-08-01T09:41:12.500Z",
        complete=True,
        price_table=BENCH_TABLE,
        taxonomy=_taxonomy(),
        configs=CONFIGS,
        client_stack={"litellm": "1.63.0", "anthropic": "0.120.0"},
    )
    base.update(overrides)
    return BenchArtifacts(**base)  # type: ignore[arg-type]


# ── the record's declared shape ────────────────────────────────────────────────────────────


def test_a_bench_record_declares_its_schema_version_and_the_kind_noctis_bench():
    record = build(_artifacts())

    assert record["schema_version"] == SCHEMA_VERSION == 1
    assert record["kind"] == KIND == "noctis.bench"


def test_the_record_carries_the_bench_site_corpus_harness_engine_and_result_sections():
    record = build(_artifacts())

    assert record["bench"]["bench_id"] == "20260801T090000Z-bench01"
    assert record["bench"]["label"] == "coder-baseline"
    assert record["site"]["site_id"] == "coder"
    assert record["site"]["site_version"] == "1"
    assert record["corpus"]["hash"] == "9f2c1d3e4b5a6c7d"
    assert record["harness"]["hash"] == "0011223344556677"
    assert record["engine"]["engine_version"] == 3
    assert [entry["case_id"] for entry in record["results"]] == [
        "breakout",
        "breakout",
        "meanrev",
    ]


def test_build_then_validate_round_trips_with_no_problems():
    assert validate(build(_artifacts())) == []


def test_a_record_may_grow_a_whole_new_section_and_still_validate():
    """Additive-only where the vocabulary is open: ``REQUIRED_SECTIONS`` is a floor, not a ceiling,
    so a section a later story added is readable rather than rejected."""
    record = build(_artifacts())
    record["something_a_later_story_added"] = {"figure": 1}

    assert validate(record) == []


def test_a_key_no_section_declares_is_refused():
    """The bench record inherits the run record's two-way keys walker (story #350): a block's
    declared keys are the whole of what it may carry, so an emitter typo is named rather than
    published as a field no reader indexes."""
    record = build(_artifacts())
    record["bench"]["future_counter"] = 7

    problems = validate(record)

    assert [problem.split(":")[0] for problem in problems] == ["bench.future_counter"]
    assert "undeclared key" in problems[0]


def test_the_validator_returns_every_problem_at_once_never_only_the_first():
    """One read of the refusal must be enough to fix the document — the schema module's posture."""
    record = build(_artifacts())
    record["schema_version"] = 99
    record["kind"] = "noctis.something-else"
    del record["metrics"]
    del record["bench"]["runs"]
    record["bench"]["started_utc"] = "2026-08-01 09:00:00"

    problems = validate(record)

    assert any("schema_version" in problem for problem in problems)
    assert any("kind" in problem for problem in problems)
    assert any("metrics" in problem for problem in problems)
    assert any("bench.runs" in problem for problem in problems)
    assert any("started_utc" in problem for problem in problems)


def test_the_validator_names_a_cost_field_that_does_not_call_itself_an_estimate():
    """These dollars come from a versioned list-price table, never from an invoice."""
    record = build(_artifacts())
    record["metrics"]["cost"]["usd_total"] = record["metrics"]["cost"].pop("usd_total_estimate")

    assert any("estimate" in problem for problem in validate(record))


def test_the_validator_names_a_dimensioned_number_that_spells_its_unit_the_long_way():
    record = build(_artifacts())
    record["metrics"]["latency"]["p50_seconds"] = record["metrics"]["latency"].pop("p50_s")

    assert any("p50_seconds" in problem for problem in validate(record))


def test_the_quoted_harness_dials_are_exempt_from_the_naming_conventions():
    """The dials are *quoted* from whatever the harness was configured with, not authored here —
    so an operator's own dial named the long way is not a bench-record schema violation."""
    record = build(_artifacts(harness_dials={"timeout_seconds": 30, "budget_usd": 5.0}))

    assert validate(record) == []


# ── derived, never incremented ─────────────────────────────────────────────────────────────


def test_building_the_same_artifacts_twice_returns_identical_records():
    assert build(_artifacts()) == build(_artifacts())


def test_every_aggregate_is_recomputable_from_the_per_case_artifacts_it_was_built_from():
    """The whole argument for a pure builder: the record carries enough per-case evidence that a
    reader can recompute every published aggregate and get the same numbers."""
    record = build(_artifacts())

    recomputed = compute_metrics(_results_of(record), table=BENCH_TABLE)

    assert record["metrics"]["first_attempt_pass_rate"] == recomputed.first_attempt_pass_rate
    assert record["metrics"]["job_pass_rate"] == recomputed.job_pass_rate
    assert record["metrics"]["mean_attempts_to_pass"] == recomputed.mean_attempts_to_pass
    assert record["metrics"]["cost"]["usd_total_estimate"] == recomputed.cost.usd_total_estimate
    assert record["metrics"]["cost"]["tokens_total"] == recomputed.cost.tokens_total


def _results_of(record: dict) -> tuple[CaseResult, ...]:
    """The record's own ``results`` section, read back as the metrics module's input type."""
    return tuple(
        CaseResult(
            case_id=entry["case_id"],
            attempts=tuple(
                AttemptOutcome(
                    passed=attempt["passed"],
                    model=attempt["model"],
                    seconds=attempt["latency_s"],
                    error=attempt["error"],
                    **attempt["usage"],
                )
                for attempt in entry["attempts"]
            ),
        )
        for entry in record["results"]
    )


def test_the_headline_rates_are_the_numbers_the_metrics_module_computes():
    """Two cases: ``breakout`` passed both reps (first attempt on one of them), ``meanrev``
    neither. First-attempt rate = mean(0.5, 0.0) = 0.25; job rate = mean(1.0, 0.0) = 0.5."""
    record = build(_artifacts())

    assert record["metrics"]["first_attempt_pass_rate"] == pytest.approx(0.25)
    assert record["metrics"]["job_pass_rate"] == pytest.approx(0.5)
    assert record["bench"]["cases"] == 2
    assert record["bench"]["runs"] == 3
    assert record["bench"]["attempts"] == 5


def test_the_bill_is_the_price_tables_and_it_carries_the_tables_version():
    """Five attempts of four million tokens at $1/Mtok on every field — $20, under ``bench-1``."""
    record = build(_artifacts())

    assert record["metrics"]["cost"]["usd_total_estimate"] == pytest.approx(20.0)
    assert record["metrics"]["cost"]["tokens_total"] == 5 * 4 * MTOK
    assert record["metrics"]["cost"]["pricing_table_version"] == "bench-1"
    assert record["provenance"]["pricing_table_version"] == "bench-1"


def test_the_failure_breakdown_is_classified_by_the_sites_own_vocabulary():
    """Three failed attempts: an import error, an inactive strategy, and one nothing recognises —
    and the catch-all is reported like any other class, because its share is the rot gauge."""
    record = build(_artifacts())

    assert record["failures"]["counts"] == {"import_error": 1, "inactive": 1, UNCLASSIFIED: 1}
    assert record["failures"]["total"] == 3
    assert record["failures"]["unclassified_share"] == pytest.approx(1 / 3)


def test_a_bench_with_no_failure_vocabulary_reports_a_null_breakdown_never_zero_counts():
    """A breakdown of zeros would say "nothing failed" about a bench nobody classified."""
    record = build(_artifacts(taxonomy=None))

    assert record["failures"] is None
    assert validate(record) == []


# ── honest absences ────────────────────────────────────────────────────────────────────────


def test_an_empty_bench_reports_null_rates_and_honest_zero_counts():
    record = build(_artifacts(runs=()))

    assert record["bench"]["cases"] == 0
    assert record["bench"]["runs"] == 0
    assert record["metrics"]["first_attempt_pass_rate"] is None
    assert record["metrics"]["job_pass_rate"] is None
    assert record["metrics"]["attempt_distribution"] is None
    assert validate(record) == []


def test_an_unpriceable_attempt_leaves_the_bill_null_never_zero():
    priced = _artifacts(
        runs=(
            CaseRun(
                result=CaseResult(
                    case_id="breakout",
                    attempts=(_attempt(True, model="a-model-the-table-never-heard-of"),),
                )
            ),
        )
    )

    record = build(priced)

    assert record["metrics"]["cost"]["usd_total_estimate"] is None
    assert record["metrics"]["cost"]["usd_per_pass_estimate"] is None


def test_a_known_absent_value_is_an_explicit_null_rather_than_a_missing_key():
    record = build(_artifacts(label=None, client_stack=None, harness_hash=None))

    assert record["bench"]["label"] is None
    assert record["provenance"]["client_stack"] is None
    assert record["harness"]["hash"] is None
    assert validate(record) == []


# ── the comparable key ─────────────────────────────────────────────────────────────────────


# One artifacts override per key component: the smallest honest change to the thing that component
# names. Every one of them must move the key, or two runs of different experiments would pool.
MOVED = {
    "engine_version": {"engine": EngineStamp(engine_version=4, fingerprint=ENGINE.fingerprint)},
    "engine_fingerprint": {
        "engine": EngineStamp(
            engine_version=3, fingerprint={**ENGINE.fingerprint, "prompts": "0000000000000000"}
        )
    },
    "site_id": {
        "site": SiteIdentity(
            site_id="distill",
            version=SITE.version,
            prompt_asset_groups=SITE.prompt_asset_groups,
            prompt_asset_hash=SITE.prompt_asset_hash,
        )
    },
    "site_version": {
        "site": SiteIdentity(
            site_id=SITE.site_id,
            version="2",
            prompt_asset_groups=SITE.prompt_asset_groups,
            prompt_asset_hash=SITE.prompt_asset_hash,
        )
    },
    "prompt_asset_hash": {
        "site": SiteIdentity(
            site_id=SITE.site_id,
            version=SITE.version,
            prompt_asset_groups=SITE.prompt_asset_groups,
            prompt_asset_hash="ffffffffffffffff",
        )
    },
    "corpus_version": {
        "corpus": CorpusIdentity(version="2026-08", hash=CORPUS.hash, case_count=2, split="tuning")
    },
    "corpus_hash": {
        "corpus": CorpusIdentity(
            version=CORPUS.version, hash="aaaabbbbccccdddd", case_count=2, split="tuning"
        )
    },
    "harness_hash": {"harness_hash": "8877665544332211"},
    "pricing_table_version": {
        "price_table": PriceTable(version="bench-2", prices=BENCH_TABLE.prices)
    },
}


def test_the_key_names_every_component_it_is_composed_from():
    key = comparable_key(_artifacts())

    assert KEY_COMPONENTS == (
        "engine_version",
        "engine_fingerprint",
        "site_id",
        "site_version",
        "prompt_asset_hash",
        "corpus_version",
        "corpus_hash",
        "harness_hash",
        "pricing_table_version",
    )
    assert [part.split("=")[0] for part in str(key).split("|")] == list(KEY_COMPONENTS)
    assert "site_id=coder" in str(key)
    assert "pricing_table_version=bench-1" in str(key)


def test_identical_inputs_reproduce_the_key_byte_for_byte():
    assert str(comparable_key(_artifacts())) == str(comparable_key(_artifacts()))
    assert build(_artifacts())["comparable_key"] == str(comparable_key(_artifacts()))


@pytest.mark.parametrize("component", sorted(MOVED))
def test_changing_one_key_component_changes_the_key(component):
    """Every component earns its place: if one of them could move without the key moving, two
    incomparable runs would land in one bucket and nothing downstream could tell."""
    baseline = str(comparable_key(_artifacts()))

    moved = str(comparable_key(_artifacts(**MOVED[component])))

    assert moved != baseline
    assert component in {
        name for name, value in _components(baseline).items() if _components(moved)[name] != value
    }


def _components(label: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in label.split("|"))


def test_a_component_nothing_could_identify_renders_as_an_explicit_null():
    key = comparable_key(_artifacts(harness_hash=None))

    assert "harness_hash=null" in str(key)
    assert key.complete is False


def test_an_engine_with_one_unidentifiable_component_has_no_folded_digest():
    """The fingerprint module's all-or-none rule: a digest over *some* of the engine looks like an
    identity while meaning less than one."""
    engine = EngineStamp(engine_version=3, fingerprint={**ENGINE.fingerprint, "prompts": None})

    record = build(_artifacts(engine=engine))

    assert record["engine"]["fingerprint_digest"] is None
    assert "engine_fingerprint=null" in record["comparable_key"]


# ── the refusal ────────────────────────────────────────────────────────────────────────────


def test_two_records_under_one_key_compare_as_comparable_and_carry_deltas():
    first = build(_artifacts())
    second = build(
        _artifacts(
            bench_id="20260802T090000Z-bench02",
            runs=RUNS[:2],
            started_utc="2026-08-02T09:00:00.000Z",
            finished_utc="2026-08-02T09:20:00.000Z",
        )
    )

    assert comparable(first, second)
    view = side_by_side(first, second)

    assert view.comparable is True
    assert view.banner is None
    assert view.deltas["job_pass_rate"] == pytest.approx(0.5)  # 1.0 (breakout only) − 0.5
    assert view.figures["a"]["job_pass_rate"] == pytest.approx(0.5)


def test_two_records_under_different_keys_refuse_a_delta_and_carry_a_loud_banner():
    first = build(_artifacts())
    second = build(_artifacts(**MOVED["prompt_asset_hash"]))

    assert not comparable(first, second)
    view = side_by_side(first, second)

    assert view.comparable is False
    assert "INCOMPARABLE" in view.banner
    # Named once, with both values, so the operator's next question is already answered.
    moved = f"prompt_asset_hash: {SITE.prompt_asset_hash} vs ffffffffffffffff"
    assert view.differences == (moved,)
    # Structurally, not by convention: the refusal hands back a value with nothing to subtract.
    assert not hasattr(view, "deltas")


def test_the_incomparable_view_still_presents_both_records_figures_side_by_side():
    """Refusing a delta is not refusing to show the numbers — an operator still reads both."""
    view = side_by_side(build(_artifacts()), build(_artifacts(**MOVED["corpus_hash"])))

    assert view.figures["a"]["job_pass_rate"] == pytest.approx(0.5)
    assert view.figures["b"]["job_pass_rate"] == pytest.approx(0.5)


def test_two_records_whose_keys_match_but_carry_an_unknown_component_are_still_incomparable():
    """Two nulls are two absences of evidence, never an agreement — the fail-safe direction."""
    first = build(_artifacts(harness_hash=None))
    second = build(_artifacts(bench_id="other", harness_hash=None))

    assert first["comparable_key"] == second["comparable_key"]
    assert not comparable(first, second)
    view = side_by_side(first, second)

    assert "harness_hash" in view.banner
    assert not hasattr(view, "deltas")


def test_a_delta_against_an_unknown_figure_is_null_never_zero():
    """``n/a`` propagates: a bench whose bill nobody could price has no cost delta to publish."""
    unpriced = _artifacts(
        runs=(
            CaseRun(
                result=CaseResult(
                    case_id="breakout", attempts=(_attempt(True, model="unknown-model"),)
                )
            ),
        )
    )

    view = side_by_side(build(_artifacts()), build(unpriced))

    assert build(unpriced)["metrics"]["cost"]["usd_per_pass_estimate"] is None
    assert view.figures["b"]["usd_per_pass_estimate"] is None
    # The two ran under one key, so a delta is admissible — and the unpriceable one is null, not
    # the difference between a real number and a zero somebody invented.
    assert view.deltas["usd_per_pass_estimate"] is None
    assert view.deltas["job_pass_rate"] is not None


def test_a_delta_is_null_when_either_side_never_measured_the_figure():
    first = build(_artifacts())
    second = build(_artifacts(bench_id="empty", runs=()))

    view = side_by_side(first, second)

    assert view.comparable is True
    assert view.deltas["job_pass_rate"] is None
    assert view.deltas["p50_s"] is None


# ── the provenance tier: recorded, never keyed ─────────────────────────────────────────────


def test_the_served_model_ids_are_recorded_on_every_attempt():
    record = build(_artifacts())

    assert record["results"][0]["attempts"][0]["model"] == "bench/oracle"
    assert record["results"][0]["attempts"][0]["served_model"] == "claude-opus-4-20260501"


def test_a_different_served_model_id_does_not_change_the_key():
    """Model identity is deliberately outside the key: a provider moving a dated snapshot under an
    alias must not fragment every comparison bucket."""
    drifted = tuple(
        CaseRun(
            result=run.result,
            config_id=run.config_id,
            served_models=tuple("claude-opus-4-20260812" for _ in run.served_models),
        )
        for run in RUNS
    )

    drifted_key = build(_artifacts(runs=drifted))["comparable_key"]

    assert drifted_key == build(_artifacts())["comparable_key"]


def test_a_same_alias_different_served_id_run_is_detectable_from_the_record():
    """The other half of that decision: not keyed, but never invisible."""
    drifted = (
        RUNS[0],
        CaseRun(
            result=RUNS[1].result,
            config_id="primary",
            served_models=("claude-opus-4-20260812", "claude-opus-4-20260812"),
        ),
        RUNS[2],
    )

    config = build(_artifacts(runs=drifted))["provenance"]["configs"][0]

    assert config["requested_model"] == "claude-opus-4"
    assert config["served_models"] == ["claude-opus-4-20260501", "claude-opus-4-20260812"]
    assert config["alias_drift"] is True


def test_one_served_id_throughout_reports_no_drift():
    config = build(_artifacts())["provenance"]["configs"][0]

    assert config["served_models"] == ["claude-opus-4-20260501"]
    assert config["alias_drift"] is False
    assert config["attempts"] == 5


def test_a_config_whose_served_ids_were_never_reported_reports_null_drift_not_false():
    """Reporting "no drift" would be a claim about evidence that does not exist."""
    silent = tuple(CaseRun(result=run.result, config_id="primary") for run in RUNS)

    config = build(_artifacts(runs=silent))["provenance"]["configs"][0]

    assert config["served_models"] == []
    assert config["served_unrecorded"] == 5
    assert config["alias_drift"] is None


def test_the_client_stack_versions_are_recorded_and_never_keyed():
    stacked = _artifacts(client_stack={"litellm": "9.9.9", "anthropic": None})

    record = build(stacked)

    assert record["provenance"]["client_stack"] == {"litellm": "9.9.9", "anthropic": None}
    assert record["comparable_key"] == build(_artifacts())["comparable_key"]


# ── the golden record ──────────────────────────────────────────────────────────────────────


def test_the_golden_bench_record_still_matches_the_committed_fixture():
    """Snapshot the whole document so schema drift is visible in review rather than discovered by
    whoever is comparing two benchmark runs. Regenerate deliberately, and explain the diff."""
    assert build(_artifacts()) == json.loads(GOLDEN.read_text())


def test_the_golden_bench_record_is_schema_valid():
    assert validate(json.loads(GOLDEN.read_text())) == []


def test_the_golden_bench_record_is_json_serializable_as_written():
    """The record is a document, not an object graph: every value survives a round trip through
    JSON unchanged, or a reader gets something the writer never meant."""
    record = build(_artifacts())

    assert json.loads(json.dumps(record)) == record


# ── purity, structurally ───────────────────────────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_builder_and_the_validator_reach_no_io_no_clock_and_no_randomness():
    """The load-bearing decision of this story, enforced by AST rather than by convention: the
    runner collects (and may read the tree and the wall clock), the builder only ever renders what
    it was handed, and the validator only ever reads the document."""
    assert _imports(RECORD_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "typing",
        "noctis",
    }
    text = RECORD_SOURCE.read_text()
    for forbidden in ("datetime.now", "utcnow", "time(", "open(", "Path(", "random", "os."):
        assert forbidden not in text, forbidden


def test_the_conventions_are_the_run_records_walkers_themselves_not_a_second_copy():
    """Story #349: the unit, stamp, estimate and key-presence rules have **one** implementation.

    A bench record and a run record read side by side must agree about what ``_s`` means and what
    a dollar figure has to call itself; two copies of the walkers eventually disagree. The import
    runs the legal way only — the eval layer reads the engine, never the reverse.
    """
    tree = ast.parse(RECORD_SOURCE.read_text())

    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    walkers = {"check_keys", "check_units", "check_stamps", "check_stamp", "check_estimate_labels"}
    assert not defined & (walkers | {f"_{name}" for name in walkers})

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "noctis.reporting.schema"
        for alias in node.names
    }
    assert {"check_keys", "check_units", "check_stamps", "check_estimate_labels"} <= imported


def test_the_engine_modules_the_record_reaches_for_are_the_two_pure_ones():
    """Cost is the pricing table's rules and the unit/estimate conventions are the run record's —
    both by identity rather than by a second copy, and both pure."""
    tree = ast.parse(RECORD_SOURCE.read_text())
    engine_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("noctis.")
        and not node.module.startswith("noctis.eval")
    }

    assert engine_modules == {"noctis.reporting.schema", "noctis.research.pricing"}
