"""#226: the coder site's own reading — two co-primary pass rates, and everything that explains
the gap between them.

The eval core already scores what every site shares (did the ask emit, in how many *runner*
attempts, at what cost). The coder's interesting arithmetic is one level down, over the authoring
engine's **private** attempts, and #225 retained exactly that: a :class:`JobRecord` per runner
attempt, carrying every internal attempt's verdict, error, model and usage. This module's tests are
the arithmetic over those records, hand-computed from scripted sequences, plus the two integration
properties that matter — a live coder bench's record carries the reading, and ``bench report``
prints it.

Five properties carry the story:

* **Two co-primary rates, never blended.** ``first_attempt_pass`` and ``job_pass`` are one paired
  value; nothing in the reading is a combined score, and the tests assert no such key exists.
* **Retry-informed passing is labelled.** Wherever a figure counts a pass a retry bought, the words
  ``pass@k with feedback`` are in the same block: the retries saw the gate's error, so plain
  ``pass@k`` would overstate what was measured.
* **Every failed internal attempt is classified**, with the knob its share points at.
* **Detector warnings ride beside the numbers and never inside them.** Stripping every finding from
  a batch changes the warning rows and nothing else — asserted key by key.
* **A missing component is ``n/a``, never zero.** An empty batch, a job that never asked, a batch
  with nothing to price: all ``None``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from noctis.cli import app
from noctis.eval.bootstrap import BenchSeams, cases_root
from noctis.eval.cli import run_bench
from noctis.eval.coder_case import Axis
from noctis.eval.coder_detectors import PARAM_FLOOR_COLLAPSE, SEVERITY
from noctis.eval.coder_distill_sites import CODER_SITE
from noctis.eval.coder_scorer import (
    ANSWERS_FRESH,
    CODER_DIALS_KEY,
    CODER_SCORER,
    FEEDBACK_LABEL,
    NOT_APPLICABLE,
    PASS_LABEL_KEY,
    RETRY_INFORMED_BLOCKS,
    STRATA_KEY,
    PassRates,
    coder_block,
    score_coder_jobs,
)
from noctis.eval.coder_site import AttemptRecord, JobRecord, coder_attempt, strategy_name
from noctis.eval.runner import BENCH_RECORD_NAME, bench_root
from noctis.eval.site import AnsweredCase
from tests.test_eval_coder_site import (
    MISNAMED,
    NAME,
    SEEDS,
    ScriptedCoder,
    _case,
    _document,
    _one_shot,
    _run,
    _settings,
    fenced,
)
from tests.test_eval_coder_site import named as renamed

runner = CliRunner()

#: A model the shipped price table carries, so a hand-computed dollar figure is checkable.
MODEL = "anthropic/claude-sonnet-4"

#: Exactly one million input tokens and nothing else — $3.00 per attempt under ``MODEL``.
SPEND: Mapping[str, int] = {
    "input_tokens": 1_000_000,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

#: One dollar figure per attempt at ``SPEND`` under ``MODEL`` — 1 Mtok × $3/Mtok.
USD_PER_ATTEMPT = 3.0

#: A gate error the coder taxonomy recognises as ``name_mismatch``.
MISNAMED_ERROR = "class sets name='probe' but the strategy/file name is 'canary_probe_case'"

#: A gate error the coder taxonomy recognises as ``truncated``.
TRUNCATED_ERROR = "the reply was cut off by the output-token limit"


# ── scripted records ──────────────────────────────────────────────────────────────────────


def _attempt(
    number: int,
    passed: bool,
    *,
    escalated: bool = False,
    error: str | None = None,
    usage: Mapping[str, int] | None = SPEND,
) -> AttemptRecord:
    """One internal coder attempt, as #225's engine wrapper records it."""
    return AttemptRecord(
        attempt=number,
        passed=passed,
        escalated=escalated,
        error=(MISNAMED_ERROR if error is None and not passed else error),
        model=MODEL,
        served_model=f"{MODEL}-20260701",
        usage=dict(usage or {}),
    )


def _job(
    case_id: str,
    outcomes: Sequence[bool],
    *,
    escalated_from: int | None = None,
    seconds: float | None = 4.0,
    findings: Sequence[Any] = (),
    usage: Mapping[str, int] | None = SPEND,
) -> JobRecord:
    """One whole authoring job whose internal attempts landed as ``outcomes`` says.

    ``escalated_from`` is the 1-based index at which the job fell back to the paid coder, which is
    what makes a rescue distinguishable from a local retry that happened to work.
    """
    attempts = tuple(
        _attempt(
            index,
            passed,
            escalated=escalated_from is not None and index >= escalated_from,
            usage=usage,
        )
        for index, passed in enumerate(outcomes, start=1)
    )
    return JobRecord(
        case_id=case_id,
        strategy=case_id.replace("-", "_"),
        passed=any(outcomes),
        attempts=attempts,
        error=None if any(outcomes) else MISNAMED_ERROR,
        model=MODEL,
        seconds=seconds,
        findings=tuple(findings),
    )


def _answered(*records: JobRecord) -> tuple[AnsweredCase, ...]:
    """Every scripted job as the answered case the runner hands a scoring pass."""
    return tuple(
        AnsweredCase(
            case=_case(record.case_id),
            config_id="default",
            rep=1,
            replies=(json.dumps(record.document(), sort_keys=True),),
        )
        for record in records
    )


def _block(*records: JobRecord) -> Mapping[str, Any]:
    """The coder reading over a scripted batch, as a record carries it."""
    reading = CODER_SCORER.read(_answered(*records))
    assert reading is not None
    return reading[CODER_DIALS_KEY]


def _keys(node: Any, seen: set[str] | None = None) -> set[str]:
    """Every key at every depth of a reading — what a "no blended score" claim is checked over."""
    found = set() if seen is None else seen
    if isinstance(node, Mapping):
        for key, value in node.items():
            found.add(str(key))
            _keys(value, found)
    elif isinstance(node, list | tuple):
        for value in node:
            _keys(value, found)
    return found


# ── the co-primary rates, hand-computed ───────────────────────────────────────────────────


def test_the_first_attempt_pass_rate_counts_only_the_opening_internal_attempt_of_each_job():
    """Three cases; only one landed a file on the opening ask, but two landed one in the end."""
    metrics = score_coder_jobs(
        [_job("a", [True]), _job("b", [False, True]), _job("c", [False, False])]
    )

    assert metrics.rates.first_attempt_pass_rate == pytest.approx(1 / 3)
    assert metrics.rates.job_pass_rate == pytest.approx(2 / 3)


def test_the_reps_of_one_case_fold_before_the_cases_average_each_other():
    """Case ``a`` was run twice and disagreed with itself; it still carries one case's weight."""
    metrics = score_coder_jobs([_job("a", [True]), _job("a", [False]), _job("b", [True])])

    assert metrics.rates.first_attempt_pass_rate == pytest.approx(0.75)
    assert metrics.rates.cases == 2
    assert metrics.rates.jobs == 3


def test_the_two_pass_rates_are_one_paired_value_neither_half_is_published_alone():
    metrics = score_coder_jobs([_job("a", [True])])

    assert isinstance(metrics.rates, PassRates)
    assert not hasattr(metrics, "job_pass_rate")
    assert not hasattr(metrics, "first_attempt_pass_rate")


def test_the_attempt_distribution_shares_where_the_cases_settled_over_internal_attempts():
    metrics = score_coder_jobs(
        [_job("a", [True]), _job("b", [False, True]), _job("c", [False, False])]
    )

    distribution = metrics.attempt_distribution
    assert distribution is not None
    assert distribution.passed_on[1] == pytest.approx(1 / 3)
    assert distribution.passed_on[2] == pytest.approx(1 / 3)
    assert distribution.never == pytest.approx(1 / 3)


def test_the_retry_yield_is_the_marginal_pass_rate_each_internal_retry_bought():
    metrics = score_coder_jobs(
        [_job("a", [True]), _job("b", [False, True]), _job("c", [False, False, True])]
    )

    assert dict(metrics.retry_yield or {}) == pytest.approx({2: 1 / 3, 3: 1 / 3})


def test_the_mean_attempts_to_pass_averages_only_the_cases_that_ever_passed():
    metrics = score_coder_jobs(
        [_job("a", [True]), _job("b", [False, False, True]), _job("c", [False])]
    )

    assert metrics.mean_attempts_to_pass == pytest.approx(2.0)


# ── escalation: what the paid fallback rescued, and what a rescued file cost ───────────────


def test_the_escalation_rescue_rate_is_the_share_of_escalated_jobs_the_fallback_saved():
    metrics = score_coder_jobs(
        [
            _job("a", [False, True], escalated_from=2),
            _job("b", [False, True], escalated_from=2),
            _job("c", [False, False], escalated_from=2),
            _job("d", [True]),
        ]
    )

    assert metrics.escalation.escalated_jobs == 3
    assert metrics.escalation.rescued_jobs == 2
    assert metrics.escalation.rescue_rate == pytest.approx(2 / 3)


def test_a_job_that_passed_before_escalating_is_no_rescue_of_the_paid_coder():
    metrics = score_coder_jobs([_job("a", [False, True])])

    assert metrics.escalation.escalated_jobs == 0
    assert metrics.escalation.rescue_rate is None


def test_dollars_per_rescued_file_prices_the_whole_escalated_job_that_bought_it():
    """One rescue, two attempts at $3.00 apiece — the local failure is charged to the rescue."""
    metrics = score_coder_jobs([_job("a", [False, True], escalated_from=2)])

    assert metrics.escalation.usd_per_rescued_file_estimate == pytest.approx(2 * USD_PER_ATTEMPT)


def test_dollars_per_rescued_file_is_n_a_when_the_price_table_cannot_price_the_spend():
    metrics = score_coder_jobs([_job("a", [False, True], escalated_from=2, usage={})])

    assert metrics.escalation.rescue_rate == 1.0
    assert metrics.escalation.usd_per_rescued_file_estimate is None


# ── cost and wall clock ───────────────────────────────────────────────────────────────────


def test_the_cost_per_pass_charges_every_internal_attempt_to_the_pass_it_bought():
    metrics = score_coder_jobs([_job("a", [False, True]), _job("b", [False, False])])

    assert metrics.cost.usd_total_estimate == pytest.approx(4 * USD_PER_ATTEMPT)
    assert metrics.cost.usd_per_pass_estimate == pytest.approx(4 * USD_PER_ATTEMPT)
    assert metrics.cost.tokens_total == 4 * SPEND["input_tokens"]


def test_the_wall_clock_is_read_at_the_job_level_where_the_engine_really_timed_it():
    metrics = score_coder_jobs(
        [_job("a", [False, True], seconds=2.0), _job("b", [True], seconds=6.0)]
    )

    assert metrics.latency.p50_seconds == pytest.approx(4.0)
    assert metrics.seconds_total == pytest.approx(8.0)


def test_a_job_that_never_recorded_its_duration_leaves_the_wall_clock_unmeasured():
    metrics = score_coder_jobs([_job("a", [True], seconds=None), _job("b", [True], seconds=6.0)])

    assert metrics.latency.p50_seconds is None
    assert metrics.seconds_total is None


# ── absences: n/a, never a fabricated zero ────────────────────────────────────────────────


def test_an_empty_batch_reads_as_n_a_everywhere_rather_than_as_a_measured_zero():
    metrics = score_coder_jobs([])

    assert metrics.rates.first_attempt_pass_rate is None
    assert metrics.rates.job_pass_rate is None
    assert metrics.mean_attempts_to_pass is None
    assert metrics.attempt_distribution is None
    assert metrics.retry_yield is None
    assert metrics.escalation.rescue_rate is None
    assert metrics.cost.usd_per_pass_estimate is None
    assert metrics.latency.p50_seconds is None


def test_a_job_whose_brief_was_refused_before_any_ask_is_counted_but_never_rated():
    """The engine refuses an unknown reference without spending a completion — no internal ask."""
    metrics = score_coder_jobs([_job("a", [True]), _job("b", [])])

    assert metrics.rates.first_attempt_pass_rate == 1.0
    assert metrics.rates.unattempted_jobs == 1
    assert metrics.rates.jobs == 2


def test_the_taxonomy_shares_are_n_a_rather_than_zero_when_no_internal_attempt_failed():
    block = _block(_job("a", [True]))

    assert block["failures"]["attempts"] == 0
    assert all(row["share"] is None for row in block["failures"]["classes"].values())


# ── the per-axis breakdown (#227) ─────────────────────────────────────────────────────────


def _labelled(record: JobRecord, **axes: str) -> AnsweredCase:
    """One scripted job as the answered case it came from, labelled on the axes given."""
    return AnsweredCase(
        case=_case(record.case_id, difficulty={**_document()["difficulty"], **axes}),
        config_id="default",
        rep=1,
        replies=(json.dumps(record.document(), sort_keys=True),),
    )


def _strata(*answered: AnsweredCase) -> Mapping[str, Any]:
    """The per-axis breakdown of a reading over cases labelled case by case."""
    reading = CODER_SCORER.read(tuple(answered))
    assert reading is not None
    return reading[CODER_DIALS_KEY][STRATA_KEY]


def test_the_reading_stratifies_its_pass_rates_by_every_difficulty_axis_the_site_declares():
    strata = _strata(_labelled(_job("a", [True])))

    assert set(strata) == {axis.value for axis in Axis}


def test_a_stratum_reports_the_pass_pair_of_only_the_jobs_labelled_at_that_level():
    strata = _strata(
        _labelled(_job("a", [True]), api_surface="bars_only"),
        _labelled(_job("b", [False, True]), api_surface="exits"),
    )

    by_level = strata["api_surface"]
    assert by_level["bars_only"]["rates"]["first_attempt_pass_rate"] == 1.0
    assert by_level["exits"]["rates"]["first_attempt_pass_rate"] == 0.0
    assert by_level["exits"]["rates"]["job_pass_rate"] == 1.0


def test_a_stratum_publishes_the_two_rates_together_exactly_as_the_headline_does():
    """The pairing rule holds at every depth: a level's agreement-free half-truth is still one."""
    stratum = _strata(_labelled(_job("a", [False, True])))["api_surface"]["indicators"]

    assert set(PassRates.rate_fields()) <= set(stratum["rates"])
    assert stratum["rates"][PASS_LABEL_KEY] == FEEDBACK_LABEL


def test_a_level_no_case_in_the_batch_carries_is_absent_rather_than_a_confident_zero():
    strata = _strata(_labelled(_job("a", [True]), api_surface="bars_only"))

    assert set(strata["api_surface"]) == {"bars_only"}


def test_every_axis_stratifies_the_same_jobs_so_each_axis_totals_the_headline():
    answered = (
        _labelled(_job("a", [True]), api_surface="bars_only"),
        _labelled(_job("b", [False, True]), api_surface="exits"),
        _labelled(_job("c", [False, False]), api_surface="exits"),
    )
    reading = CODER_SCORER.read(answered)
    assert reading is not None

    block = reading[CODER_DIALS_KEY]
    for axis, levels in block[STRATA_KEY].items():
        assert sum(level["jobs"] for level in levels.values()) == block["jobs"], axis


# ── the failure taxonomy, with the knob each share points at ──────────────────────────────


def test_every_failed_internal_attempt_is_classified_into_the_coder_vocabulary():
    block = _block(
        _job("a", [False, True]),
        JobRecord(
            case_id="b",
            strategy="b",
            passed=False,
            attempts=(_attempt(1, False, error=TRUNCATED_ERROR),),
            error=TRUNCATED_ERROR,
        ),
    )

    counted = block["failures"]["classes"]
    assert block["failures"]["attempts"] == 2
    assert counted["name_mismatch"]["count"] == 1
    assert counted["truncated"]["count"] == 1


def test_each_failure_class_row_carries_the_knob_its_share_points_at():
    block = _block(_job("a", [False, True]))

    rows = block["failures"]["classes"]
    assert rows["truncated"]["knob"] == "coder max-tokens, thinking allowance, thinking dial"
    assert rows["name_mismatch"]["share"] == 1.0


def test_the_escape_hatch_class_is_on_the_screen_even_when_nothing_reached_it():
    block = _block(_job("a", [False, True]))

    assert block["failures"]["classes"]["unclassified"] == {
        "count": 0,
        "share": 0.0,
        "knob": "grow the taxonomy",
    }


# ── no blended score, and the feedback label wherever retries paid ────────────────────────


def test_the_reading_carries_no_combined_score_key_at_any_depth():
    block = _block(_job("a", [False, True]), _job("b", [True]))

    forbidden = ("score", "blended", "combined", "composite", "overall")
    assert not [key for key in _keys(block) if any(word in key.lower() for word in forbidden)]


def test_both_pass_rates_are_published_together_in_one_block_neither_alone():
    block = _block(_job("a", [False, True]))

    assert set(PassRates.rate_fields()) <= set(block["rates"])


def test_every_block_reporting_retry_informed_passing_carries_the_feedback_label():
    block = _block(_job("a", [False, True], escalated_from=2))

    assert RETRY_INFORMED_BLOCKS
    for name in RETRY_INFORMED_BLOCKS:
        assert FEEDBACK_LABEL in block[name].values(), name


def test_the_feedback_label_says_the_retries_saw_the_gate_error():
    assert FEEDBACK_LABEL == "pass@k with feedback"


def test_the_paired_rates_render_the_job_rate_under_its_feedback_label():
    rendered = score_coder_jobs([_job("a", [False, True])]).rates.render()

    assert "First-attempt pass rate" in rendered
    assert FEEDBACK_LABEL in rendered


# ── detector warnings: rows beside the numbers, never inside them ─────────────────────────


def _finding(detector: str = PARAM_FLOOR_COLLAPSE) -> Any:
    from noctis.eval.coder_detectors import DegenerateFinding

    return DegenerateFinding(detector=detector, severity=SEVERITY, summary="3 of 3 dimensions")


def test_a_detector_finding_on_a_passing_job_rides_as_a_warning_row():
    block = _block(_job("a", [True], findings=[_finding()]))

    (row,) = block["warnings"]
    assert (row["case_id"], row["detector"], row["severity"]) == (
        "a",
        PARAM_FLOOR_COLLAPSE,
        "WARNING",
    )
    assert row["summary"] == "3 of 3 dimensions"
    assert block["warned_jobs"] == 1


def test_detector_findings_change_the_warning_rows_and_no_other_figure_in_the_reading():
    """Inertness, proved key by key: the same batch with and without findings."""
    warned = _block(_job("a", [True], findings=[_finding()]), _job("b", [False, True]))
    silent = _block(_job("a", [True]), _job("b", [False, True]))

    assert warned != silent
    assert {
        key: value for key, value in warned.items() if key not in ("warnings", "warned_jobs")
    } == {key: value for key, value in silent.items() if key not in ("warnings", "warned_jobs")}


def test_a_batch_that_drew_no_finding_carries_no_warning_rows_at_all():
    block = _block(_job("a", [True]))

    assert block["warnings"] == []
    assert block["warned_jobs"] == 0


# ── the scoring pass itself ───────────────────────────────────────────────────────────────


def test_a_bench_that_answered_nothing_publishes_no_reading_rather_than_empty_figures():
    assert CODER_SCORER.read(()) is None


def test_a_reply_that_is_not_a_job_record_is_counted_as_unreadable_and_rated_nowhere():
    answered = (
        AnsweredCase(case=_case("a"), replies=("the provider fell over",)),
        *_answered(_job("b", [True])),
    )

    reading = CODER_SCORER.read(answered)
    assert reading is not None
    assert reading[CODER_DIALS_KEY]["unreadable"] == 1
    assert reading[CODER_DIALS_KEY]["rates"]["job_pass_rate"] == 1.0


def test_the_reading_marks_itself_as_freshly_answered_and_counts_the_calls_behind_it():
    reading = CODER_SCORER.read(_answered(_job("a", [True]), _job("b", [True])))

    assert reading is not None
    assert (reading["answers"], reading["attempt_calls"]) == ("fresh", 2)


def test_the_coder_declaration_carries_the_reading_scorer_in_its_scorers_slot():
    assert CODER_SITE.scorers == (CODER_SCORER,)


def test_a_fresh_answer_is_spelled_the_same_word_the_decide_reading_spells_it():
    """The vocabulary is shared so two records diff; each module states it without importing."""
    from noctis.eval.decide_site import ANSWERS_FRESH as DECIDE_FRESH

    assert ANSWERS_FRESH == DECIDE_FRESH


def test_an_unlabelled_stratum_is_spelled_the_same_word_the_decide_reading_spells_it():
    from noctis.eval.decide_case import NOT_APPLICABLE as DECIDE_NOT_APPLICABLE

    assert NOT_APPLICABLE == DECIDE_NOT_APPLICABLE


def test_the_block_builder_and_the_scoring_pass_publish_the_same_shape():
    records = [_job("a", [False, True])]

    assert set(coder_block(score_coder_jobs(records))) == set(_block(*records))


# ── where the detectors run: inside the job, on the file it just landed ───────────────────


def _floored(name: str) -> str:
    """A gate-clean file whose one tunable default sits at the bottom of its own declared space."""
    return fenced(
        renamed(name).replace(
            'ParamSpec("lookback", "int", 5, 40, 1)', 'ParamSpec("lookback", "int", 12, 200, 1)'
        )
    )


FLOORED = _floored("canary_probe_0")


def test_a_passing_job_stamps_the_detector_findings_its_validated_file_drew(tmp_path, fast_gate):
    """The detectors run where the file is — inside the job — and ride on the record it retains."""
    answer, _ = _run(tmp_path, ScriptedCoder([_floored(NAME)]))

    document = json.loads(answer.output or "{}")
    assert answer.outcome.passed is True
    assert [one["detector"] for one in document["findings"]] == [PARAM_FLOOR_COLLAPSE]
    assert document["findings"][0]["severity"] == SEVERITY


def test_a_detector_finding_leaves_the_gates_own_verdict_exactly_where_it_was(tmp_path, fast_gate):
    clean, _ = _run(tmp_path / "clean", ScriptedCoder([fenced(renamed(NAME))]))
    warned, _ = _run(tmp_path / "warned", ScriptedCoder([_floored(NAME)]))

    assert clean.outcome.passed is warned.outcome.passed is True
    assert json.loads(clean.output or "{}")["findings"] == []


def test_a_job_the_gate_refused_records_no_detector_finding_at_all(tmp_path, fast_gate):
    """There is no validated file to inspect, so there is nothing honest for a detector to say."""
    answer, _ = _run(tmp_path, ScriptedCoder([MISNAMED]), settings=_one_shot(tmp_path))

    assert answer.outcome.passed is False
    assert json.loads(answer.output or "{}")["findings"] == []


# ── end to end: a stubbed live coder bench, and the report over its record ────────────────


def _corpus(tmp_path: Path, *, cases: int = 2) -> Path:
    """A small coder corpus in the workspace's cases root, in the bucket layout the site reads."""
    directory = cases_root(tmp_path / "workspace") / "coder" / "canary"
    directory.mkdir(parents=True)
    for index in range(cases):
        (directory / f"canary-probe-{index}.yaml").write_text(
            yaml.safe_dump(_document(), sort_keys=True), encoding="utf-8"
        )
    return directory


def _bench_record(tmp_path: Path) -> Mapping[str, Any]:
    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    return json.loads((directory / BENCH_RECORD_NAME).read_text(encoding="utf-8"))


def _live_bench(tmp_path: Path, replies: Sequence[str], *, cases: int = 2) -> Mapping[str, Any]:
    """One stubbed live coder bench over ``cases`` canary cases, and the record it wrote."""
    _corpus(tmp_path, cases=cases)
    attempt = coder_attempt(
        _settings(tmp_path), client=ScriptedCoder(replies), model=MODEL, seeds=SEEDS
    )
    run_bench("coder", seams=BenchSeams(attempt=attempt))
    return _bench_record(tmp_path)


def test_a_stubbed_live_coder_bench_publishes_the_coder_reading_in_its_record(tmp_path, fast_gate):
    replies = [fenced(renamed(strategy_name(f"canary-probe-{index}"))) for index in range(2)]

    record = _live_bench(tmp_path, replies)

    block = record["harness"]["dials"][CODER_DIALS_KEY]
    assert block["rates"]["first_attempt_pass_rate"] == 1.0
    assert block["rates"]["job_pass_rate"] == 1.0


def test_a_live_coder_bench_reads_the_internal_retry_that_rescued_a_case(tmp_path, fast_gate):
    replies = [fenced(renamed("wrong_name")), fenced(renamed(strategy_name("canary-probe-0")))]

    record = _live_bench(tmp_path, replies, cases=1)

    block = record["harness"]["dials"][CODER_DIALS_KEY]
    assert block["rates"]["first_attempt_pass_rate"] == 0.0
    assert block["rates"]["job_pass_rate"] == 1.0
    assert block["failures"]["attempts"] == 1


def test_a_live_coder_bench_warns_about_a_passing_file_whose_defaults_sit_on_the_floor(
    tmp_path, fast_gate
):
    record = _live_bench(tmp_path, [FLOORED], cases=1)

    block = record["harness"]["dials"][CODER_DIALS_KEY]
    (row,) = block["warnings"]
    assert row["detector"] == PARAM_FLOOR_COLLAPSE
    assert block["rates"]["job_pass_rate"] == 1.0


def test_bench_report_over_a_live_coder_record_prints_both_rates_and_the_feedback_label(
    tmp_path, fast_gate
):
    replies = [fenced(renamed(strategy_name(f"canary-probe-{index}"))) for index in range(2)]
    _live_bench(tmp_path, replies)
    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())

    result = runner.invoke(app, ["bench", "report", directory.name])

    assert result.exit_code == 0, result.output
    assert "first_attempt_pass_rate" in result.output
    assert "job_pass_rate" in result.output
    assert FEEDBACK_LABEL in result.output
