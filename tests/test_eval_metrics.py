"""Pure bench metrics (#198): within-case-first rates, effort, cost through pricing, latency.

Every number here is asserted against a hand-computed fixture, because the point of the module is
that an operator can reproduce any figure it publishes with a pencil. Three properties carry the
weight:

* **Within-case-first.** Reps aggregate inside a case before cases average, so a case run twenty
  times cannot outvote nineteen cases run once. The headline fixture is one where the two orderings
  genuinely differ, so a pooled implementation cannot pass by accident.
* **Honest absences.** A metric whose inputs are missing is ``None`` (rendered ``n/a``), never a
  fabricated zero, and every ratio is denominator-guarded — including on an empty result set, which
  produces a None-carrying record rather than an exception.
* **Cost is the pricing module's, verbatim.** Dollars come from
  :mod:`noctis.research.pricing`: an unknown model poisons the total to ``None``, "free" is a
  stated price on the allowlist, and the table version travels with every figure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from noctis.eval.metrics import (
    AttemptOutcome,
    CaseResult,
    attempt_distribution,
    compute_metrics,
    cost_metrics,
    first_attempt_pass_rate,
    job_pass_rate,
    latency_metrics,
    mean_attempts_to_pass,
    retry_yield,
)
from noctis.research.pricing import PRICING_TABLE_VERSION, default_table, table_from_config

METRICS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "noctis" / "eval" / "metrics.py"

# One attempt's usage, as the four neutral fields the LLM seam reports. A round million of each, so
# a $/Mtok rate reads straight off the fixture.
MTOK = 1_000_000


def _attempt(
    passed: bool,
    *,
    model: str | None = None,
    seconds: float | None = None,
    tokens: int | None = None,
    error: str | None = None,
) -> AttemptOutcome:
    """One attempt with the four usage fields all set to ``tokens`` (or all unrecorded)."""
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


def _case(case_id: str, *pattern: bool, **kwargs) -> CaseResult:
    """One run of a case from a pass/fail pattern: ``_case("a", False, True)`` failed then passed.

    ``kwargs`` (model, tokens, seconds) apply to every attempt in the pattern.
    """
    return CaseResult(
        case_id=case_id, attempts=tuple(_attempt(passed, **kwargs) for passed in pattern)
    )


# The fixture the two aggregation orders disagree about: alpha was run three times (one clean pass,
# one pass on retry, one washout), beta once (a clean pass). Within-case-first gives each case one
# vote; pooling over the four runs lets alpha's three runs outvote beta.
DIVERGENT = (
    _case("alpha", True),
    _case("alpha", False, True),
    _case("alpha", False, False),
    _case("beta", True),
)

# A hand-computable effort fixture, one run per case: pass on 1, on 2, on 3, and never.
EFFORT = (
    _case("solo", True),
    _case("retried", False, True),
    _case("stubborn", False, False, True),
    _case("lost", False, False),
)


# ── within-case-first aggregation ─────────────────────────────────────────────────────────
def test_the_first_attempt_pass_rate_averages_within_each_case_before_averaging_across_cases():
    """alpha's runs pass first-try 1-of-3; beta 1-of-1. Equal case weight ⇒ (1/3 + 1) / 2."""
    assert first_attempt_pass_rate(DIVERGENT) == pytest.approx(2 / 3)


def test_the_first_attempt_pass_rate_is_not_the_pooled_rate_over_every_run():
    """The pooled reading (2 first-try passes out of 4 runs = 0.5) is exactly what is refused."""
    assert first_attempt_pass_rate(DIVERGENT) != pytest.approx(0.5)


def test_the_job_pass_rate_averages_within_each_case_before_averaging_across_cases():
    """alpha passes on some attempt in 2 of 3 runs; beta in 1 of 1 ⇒ (2/3 + 1) / 2."""
    assert job_pass_rate(DIVERGENT) == pytest.approx(5 / 6)


def test_the_job_pass_rate_is_not_the_pooled_rate_over_every_run():
    assert job_pass_rate(DIVERGENT) != pytest.approx(0.75)


def test_the_job_pass_rate_counts_a_case_that_needed_every_retry_it_took():
    assert job_pass_rate(EFFORT) == pytest.approx(0.75)
    assert first_attempt_pass_rate(EFFORT) == pytest.approx(0.25)


# ── effort: attempts to pass, the distribution, retry yield ───────────────────────────────
def test_mean_attempts_to_pass_averages_only_the_cases_that_ever_passed():
    """(1 + 2 + 3) / 3 — the case that never passed has no attempts-to-pass to contribute."""
    assert mean_attempts_to_pass(EFFORT) == pytest.approx(2.0)


def test_mean_attempts_to_pass_weights_a_repeated_case_once():
    """alpha passed on attempt 1 and on attempt 2 across its runs (mean 1.5); beta on 1."""
    assert mean_attempts_to_pass(DIVERGENT) == pytest.approx(1.25)


def test_the_attempt_distribution_reports_the_share_of_cases_settled_at_each_attempt():
    distribution = attempt_distribution(EFFORT)

    assert distribution is not None
    assert dict(distribution.passed_on) == pytest.approx({1: 0.25, 2: 0.25, 3: 0.25})
    assert distribution.never == pytest.approx(0.25)


def test_the_attempt_distribution_carries_a_zero_for_an_attempt_index_nobody_passed_on():
    """Three attempts were available and attempt 2 converted nobody — a stated zero, not a gap."""
    settled = (_case("solo", True), _case("stubborn", False, False, True))

    distribution = attempt_distribution(settled)

    assert distribution is not None
    assert dict(distribution.passed_on) == pytest.approx({1: 0.5, 2: 0.0, 3: 0.5})
    assert distribution.never == pytest.approx(0.0)


def test_the_attempt_distribution_sums_to_one_across_the_indices_and_the_never_share():
    distribution = attempt_distribution(DIVERGENT)

    assert distribution is not None
    assert sum(distribution.passed_on.values()) + distribution.never == pytest.approx(1.0)


def test_retry_yield_reports_the_marginal_pass_rate_each_extra_attempt_bought():
    """Attempt 1 is the first-attempt rate; retries bought a quarter of the cases each."""
    assert dict(retry_yield(EFFORT) or {}) == pytest.approx({2: 0.25, 3: 0.25})


def test_retry_yield_is_empty_when_no_case_was_ever_attempted_twice():
    assert dict(retry_yield((_case("solo", True), _case("other", False))) or {}) == {}


# ── cost, through the existing pricing module ─────────────────────────────────────────────
def _priced(model: str | None) -> tuple[CaseResult, ...]:
    """One case: a failed attempt then a passing one, each spending 1 Mtok of every usage field."""
    return (
        CaseResult(
            case_id="priced",
            attempts=(
                _attempt(False, model=model, tokens=MTOK),
                _attempt(True, model=model, tokens=MTOK),
            ),
        ),
    )


BENCH_TABLE = table_from_config(
    {
        "bench/oracle": {
            "input_usd_per_mtok": 1.0,
            "output_usd_per_mtok": 2.0,
            "cache_write_usd_per_mtok": 3.0,
            "cache_read_usd_per_mtok": 4.0,
        }
    }
)


def test_tokens_per_pass_charges_the_failed_attempts_to_the_pass_they_bought():
    cost = cost_metrics(_priced("bench/oracle"), table=BENCH_TABLE)

    assert cost.tokens_total == 8 * MTOK
    assert cost.tokens_per_pass == pytest.approx(8 * MTOK)


def test_dollars_per_pass_is_priced_field_by_field_through_the_pricing_table():
    """Each attempt bills 1 + 2 + 3 + 4 = $10; two attempts bought one pass."""
    cost = cost_metrics(_priced("bench/oracle"), table=BENCH_TABLE)

    assert cost.usd_total_estimate == pytest.approx(20.0)
    assert cost.usd_per_pass_estimate == pytest.approx(20.0)


def test_an_unknown_model_poisons_the_dollar_total_to_none_while_tokens_stay_counted():
    cost = cost_metrics(_priced("acme/never-surveyed"), table=BENCH_TABLE)

    assert cost.usd_total_estimate is None
    assert cost.usd_per_pass_estimate is None
    assert cost.tokens_total == 8 * MTOK


def test_one_unpriceable_attempt_poisons_the_whole_bill_rather_than_understating_it():
    mixed = (
        *_priced("bench/oracle"),
        CaseResult(case_id="odd", attempts=(_attempt(True, model="acme/off-table", tokens=MTOK),)),
    )

    assert cost_metrics(mixed, table=BENCH_TABLE).usd_total_estimate is None


def test_an_attempt_that_never_recorded_a_model_is_unpriceable_rather_than_free():
    cost = cost_metrics(_priced(None), table=BENCH_TABLE)

    assert cost.usd_total_estimate is None


def test_an_attempt_missing_its_usage_split_is_unpriceable_and_leaves_tokens_unknown():
    unsplit = (CaseResult(case_id="thin", attempts=(_attempt(True, model="bench/oracle"),)),)

    cost = cost_metrics(unsplit, table=BENCH_TABLE)

    assert cost.tokens_total is None
    assert cost.tokens_per_pass is None
    assert cost.usd_total_estimate is None


def test_a_free_local_backend_costs_a_stated_zero_because_the_table_says_so():
    cost = cost_metrics(_priced("ollama_chat/qwen3"), table=default_table())

    assert cost.usd_total_estimate == pytest.approx(0.0)
    assert cost.usd_per_pass_estimate == pytest.approx(0.0)


def test_every_cost_figure_carries_the_shipped_pricing_table_version_stamp():
    cost = cost_metrics(_priced("claude-sonnet-4"), table=default_table())

    assert cost.pricing_table_version == PRICING_TABLE_VERSION
    assert cost.usd_total_estimate is not None


def test_an_overridden_table_stamps_its_own_derived_version_on_the_cost_figures():
    cost = cost_metrics(_priced("bench/oracle"), table=BENCH_TABLE)

    assert cost.pricing_table_version.startswith(f"{PRICING_TABLE_VERSION}+custom.")


def test_a_run_where_nothing_passed_reports_no_cost_per_pass_rather_than_zero():
    lost = (
        CaseResult(case_id="lost", attempts=(_attempt(False, model="bench/oracle", tokens=MTOK),)),
    )

    cost = cost_metrics(lost, table=BENCH_TABLE)

    assert cost.usd_total_estimate == pytest.approx(10.0)
    assert cost.usd_per_pass_estimate is None
    assert cost.tokens_per_pass is None


# ── latency ───────────────────────────────────────────────────────────────────────────────
def _timed(case_id: str, *steps: tuple[bool, float]) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        attempts=tuple(_attempt(passed, seconds=seconds) for passed, seconds in steps),
    )


def test_wall_clock_percentiles_match_the_hand_computed_p50_and_p95():
    """Passes at 1, 2, 3 and 4 seconds: p50 interpolates to 2.5, p95 to 3 + 0.85 = 3.85."""
    timed = tuple(_timed(f"case{i}", (True, float(i))) for i in (1, 2, 3, 4))

    latency = latency_metrics(timed)

    assert latency.p50_seconds == pytest.approx(2.5)
    assert latency.p95_seconds == pytest.approx(3.85)


def test_a_pass_latency_includes_the_failed_attempts_that_preceded_it():
    latency = latency_metrics((_timed("retried", (False, 2.0), (True, 3.0)),))

    assert latency.p50_seconds == pytest.approx(5.0)


def test_a_run_that_never_passed_contributes_no_wall_clock_to_the_percentiles():
    latency = latency_metrics((_timed("won", (True, 4.0)), _timed("lost", (False, 99.0))))

    assert latency.p50_seconds == pytest.approx(4.0)
    assert latency.p95_seconds == pytest.approx(4.0)


def test_an_unrecorded_attempt_duration_leaves_the_percentiles_none_rather_than_short():
    partial = (_timed("won", (True, 4.0)), _case("untimed", True))

    latency = latency_metrics(partial)

    assert latency.p50_seconds is None
    assert latency.p95_seconds is None


def test_a_run_with_no_passes_at_all_reports_no_percentiles():
    latency = latency_metrics((_timed("lost", (False, 4.0)),))

    assert latency.p50_seconds is None
    assert latency.p95_seconds is None


# ── honest absences and denominator guards ────────────────────────────────────────────────
def test_a_case_result_with_no_attempts_recorded_yields_none_rather_than_a_zero_rate():
    empty_run = (CaseResult(case_id="unrun"),)

    assert first_attempt_pass_rate(empty_run) is None
    assert job_pass_rate(empty_run) is None
    assert attempt_distribution(empty_run) is None


def test_an_unrun_case_is_skipped_rather_than_dragging_the_rate_of_the_cases_that_ran():
    mixed = (_case("ran", True), CaseResult(case_id="unrun"))

    assert job_pass_rate(mixed) == pytest.approx(1.0)


def test_an_empty_result_set_produces_a_none_carrying_record_rather_than_an_exception():
    metrics = compute_metrics(())

    assert metrics.cases == 0
    assert metrics.first_attempt_pass_rate is None
    assert metrics.job_pass_rate is None
    assert metrics.mean_attempts_to_pass is None
    assert metrics.attempt_distribution is None
    assert metrics.retry_yield is None
    assert metrics.cost.tokens_per_pass is None
    assert metrics.cost.usd_per_pass_estimate is None
    assert metrics.latency.p50_seconds is None


def test_a_run_where_no_case_passed_reports_a_measured_zero_rate_but_no_per_pass_figures():
    """A zero *rate* is a real measurement; a per-pass ratio with no passes is n/a."""
    metrics = compute_metrics((_case("lost", False, False),))

    assert metrics.job_pass_rate == pytest.approx(0.0)
    assert metrics.mean_attempts_to_pass is None
    assert metrics.latency.p50_seconds is None


def test_compute_metrics_bundles_every_family_into_one_record():
    metrics = compute_metrics(DIVERGENT)

    assert metrics.cases == 2
    assert metrics.runs == 4
    assert metrics.first_attempt_pass_rate == pytest.approx(2 / 3)
    assert metrics.job_pass_rate == pytest.approx(5 / 6)
    assert metrics.cost.pricing_table_version == PRICING_TABLE_VERSION
    assert metrics.attempt_distribution is not None


def test_compute_metrics_prices_through_the_table_it_is_handed():
    metrics = compute_metrics(_priced("bench/oracle"), table=BENCH_TABLE)

    assert metrics.cost.usd_per_pass_estimate == pytest.approx(20.0)
    assert metrics.cost.pricing_table_version == BENCH_TABLE.version


# ── purity, structurally ──────────────────────────────────────────────────────────────────
def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_metrics_module_reaches_no_io_no_clock_and_no_randomness():
    """The same structural rule the run record and the pricing table are held to: every number is
    a function of the values handed in, so a fixture reproduces it exactly.

    The import allowlist is the structural half (no ``os``, no ``pathlib``, nothing that could
    reach a file or the settings even transitively); the text check is the half that catches a
    clock, a file or a seed opened through a name that *is* allowed."""
    assert _imports(METRICS_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "types",
        "typing",
        "noctis",
    }
    text = METRICS_SOURCE.read_text()
    for forbidden in ("datetime.now", "utcnow", "time(", "open(", "Path(", "random", "os."):
        assert forbidden not in text, forbidden


def test_the_only_engine_module_the_metrics_reach_for_is_the_pure_pricing_table():
    """Cost is the pricing module's rules, verbatim — and that is the *whole* engine dependency."""
    tree = ast.parse(METRICS_SOURCE.read_text())
    engine_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("noctis")
    }

    assert engine_modules == {"noctis.research.pricing"}
