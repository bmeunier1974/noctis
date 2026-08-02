"""``bench run --tier`` (#227): a declared subset of a corpus, sized so a smoke run is minutes.

A benchmark over a whole corpus is the measurement; a **tier** is the fast question an operator asks
before spending on one — "is the harness alive, and does the coder still land the plain cases?".
The story of this module is that a tier is **data**, not a flag with logic behind it: the coder's
``smoke`` tier is a named list of twelve case ids in the eval layer's own declaration table
(:data:`~noctis.eval.bootstrap.SITE_TIERS`), so what it selects can be read, reviewed and asserted
without running anything.

Four properties carry it:

* **The declaration is the selection.** Twelve case ids, every canary plus six named edge cases, and
  together they touch every level of all seven difficulty axes — asserted by enumeration over the
  committed corpus rather than trusted to a curator's memory.
* **Dealt first, selected second.** A tier never re-deals a split: each case keeps the half the
  whole corpus gave it, so a tuning case cannot become a holdout one by being asked about in a
  smaller group.
* **A tier and a split are two ways to name a population, and naming both is refused.** Filtering a
  declared twelve-case tier down to its holdout would publish the word ``smoke`` over a different
  measurement; the refusal names both flags rather than picking one.
* **The width is derived from the tier, not guessed.** Twelve independent jobs, worked
  :data:`~noctis.eval.bootstrap.TIER_WORKER_CAP` at a time when nobody stated ``--workers`` — and a
  stated width always wins.

Everything here is stubbed: the committed corpus is copied into a throwaway workspace and the model
call is a scripted job record. Nothing touches a network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.eval.bootstrap import (
    CODER_SMOKE_CASES,
    SEQUENTIAL_WORKERS,
    SMOKE_TIER,
    TIER_WORKER_CAP,
    BenchSeams,
    CaseTier,
    Population,
    bench_width,
    cases_root,
    load_corpus,
    select_population,
)
from noctis.eval.cli import run_bench
from noctis.eval.coder_case import AXIS_LEVELS, Axis, Bucket, coder_payload
from noctis.eval.coder_corpus import CODER_SITE_ID, bucket_of
from noctis.eval.coder_scorer import CODER_DIALS_KEY
from noctis.eval.coder_site import AttemptRecord, JobRecord, strategy_name
from noctis.eval.corpus import Corpus
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.record import validate
from noctis.eval.runner import BENCH_RECORD_NAME, Attempt, AttemptRequest, bench_root

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CASES = REPO_ROOT / "cases"

#: What the shipped smoke tier declares, spelled here so a silent edit to the table is a red test.
SMOKE_SIZE = 12


# ── the committed corpus, in a throwaway workspace ────────────────────────────────────────────


def _workspace_corpus(tmp_path: Path) -> Path:
    """The committed coder corpus, copied into the workspace's cases root."""
    target = cases_root(tmp_path / "workspace") / CODER_SITE_ID
    shutil.copytree(COMMITTED_CASES / CODER_SITE_ID, target)
    return target


def _committed() -> Corpus:
    """The whole committed coder corpus, loaded and dealt through the real provider."""
    return load_corpus(CODER_SITE_ID, cases_root=COMMITTED_CASES)


def _smoke() -> Corpus:
    """The smoke tier of the committed corpus."""
    return SMOKE_TIER.select(_committed())


def _line(output: str, prefix: str) -> str:
    """The one printed line starting with ``prefix``, without it."""
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"no line starting with {prefix!r} in:\n{output}")


# ── the declaration ───────────────────────────────────────────────────────────────────────────


def test_the_coder_smoke_tier_declares_exactly_twelve_named_cases():
    assert len(CODER_SMOKE_CASES) == SMOKE_SIZE
    assert len(set(CODER_SMOKE_CASES)) == SMOKE_SIZE


def test_the_smoke_tier_is_every_canary_the_corpus_ships_plus_six_named_edge_cases():
    buckets = [bucket_of(case) for case in _smoke().cases]
    canaries = [case for case in _committed().cases if bucket_of(case) is Bucket.CANARY]

    assert buckets.count(Bucket.CANARY) == len(canaries)
    assert buckets.count(Bucket.EDGE) == SMOKE_SIZE - len(canaries)


def test_the_smoke_tier_touches_every_declared_level_of_every_coder_difficulty_axis():
    """The selection rule, checked rather than believed: twelve cases, twenty levels, no gaps."""
    labelled = [coder_payload(case) for case in _smoke().cases]

    for axis in Axis:
        covered = {payload.level(axis) for payload in labelled}
        assert set(AXIS_LEVELS[axis]) == covered, axis.value


def test_the_smoke_tier_names_why_it_selects_what_it_selects():
    assert SMOKE_TIER.rationale.strip()
    assert SMOKE_TIER.name == "smoke"


# ── selecting a tier out of a corpus ──────────────────────────────────────────────────────────


def test_selecting_a_tier_yields_exactly_the_cases_it_declares_and_nothing_else():
    selected = _smoke()

    assert len(selected) == SMOKE_SIZE
    assert sorted(case.case_id for case in selected.cases) == sorted(CODER_SMOKE_CASES)


def test_a_tier_keeps_the_half_the_whole_corpus_dealt_each_case_rather_than_re_dealing_it():
    """Dealt first, selected second — a tuning case cannot turn holdout in a smaller group."""
    whole = {case.case_id: case.split for case in _committed().cases}

    tiered = {case.case_id: case.split for case in _smoke().cases}

    assert tiered == {case_id: whole[case_id] for case_id in tiered}
    assert set(tiered.values()) == {whole[case_id] for case_id in CODER_SMOKE_CASES}


def test_a_tier_naming_a_case_the_corpus_does_not_hold_is_refused_naming_the_missing_case():
    thinned = Corpus(
        site_id=CODER_SITE_ID,
        cases=tuple(case for case in _committed().cases if case.case_id != CODER_SMOKE_CASES[0]),
    )

    with pytest.raises(ValueError) as refusal:
        SMOKE_TIER.select(thinned)

    assert CODER_SMOKE_CASES[0] in str(refusal.value)
    assert "smoke" in str(refusal.value)


# ── resolving the population a bench measures ─────────────────────────────────────────────────


def test_no_tier_at_all_measures_the_whole_split_exactly_as_before():
    population = select_population(CODER_SITE_ID, split="all", tier=None)

    assert population == Population()
    assert len(population.of(_committed())) == len(_committed())


def test_an_unknown_tier_is_refused_naming_every_tier_that_site_declares():
    with pytest.raises(ValueError) as refusal:
        select_population(CODER_SITE_ID, split="all", tier="quick")

    assert "quick" in str(refusal.value)
    assert "smoke" in str(refusal.value)


def test_a_site_that_declares_no_tiers_at_all_refuses_a_tier_by_name():
    with pytest.raises(ValueError) as refusal:
        select_population("decide", split="all", tier="smoke")

    assert "decide" in str(refusal.value)
    assert "smoke" in str(refusal.value)


def test_naming_a_tier_and_a_half_of_the_corpus_at_once_is_refused_rather_than_intersected():
    """Two ways to name a population; filtering a declared twelve would rename the measurement."""
    with pytest.raises(ValueError) as refusal:
        select_population(CODER_SITE_ID, split="holdout", tier="smoke")

    assert "--tier" in str(refusal.value)
    assert "--split" in str(refusal.value)
    assert "holdout" in str(refusal.value)


def test_a_tier_beside_the_whole_corpus_word_is_accepted_because_that_word_filters_nothing():
    population = select_population(CODER_SITE_ID, split="all", tier="smoke")

    assert population.tier is SMOKE_TIER
    assert population.split is None


# ── how wide the jobs are worked ──────────────────────────────────────────────────────────────


def test_a_twelve_case_tier_derives_a_pool_the_worker_cap_wide_when_nobody_states_one():
    width = bench_width(None, Population(tier=SMOKE_TIER), cases=SMOKE_SIZE)

    assert width == TIER_WORKER_CAP


def test_a_tier_smaller_than_the_cap_opens_only_as_many_workers_as_it_has_cases():
    small = CaseTier(name="tiny", case_ids=("a", "b"), rationale="two cases")

    assert bench_width(None, Population(tier=small), cases=2) == 2


def test_an_untiered_run_that_states_no_width_stays_sequential():
    assert bench_width(None, Population(), cases=40) == SEQUENTIAL_WORKERS


def test_a_stated_width_wins_over_the_tiers_own_derivation():
    assert bench_width(3, Population(tier=SMOKE_TIER), cases=SMOKE_SIZE) == 3


# ── the verb ──────────────────────────────────────────────────────────────────────────────────


def test_the_smoke_tier_plans_only_the_twelve_cases_it_declares(tmp_path, capsys):
    _workspace_corpus(tmp_path)

    run_bench(CODER_SITE_ID, tier="smoke", dry_run=True)

    assert _line(capsys.readouterr().out, "plan:").startswith("coder/all: 12 case(s)")


def test_a_run_with_no_tier_plans_the_whole_corpus(tmp_path, capsys):
    _workspace_corpus(tmp_path)

    run_bench(CODER_SITE_ID, dry_run=True)

    assert _line(capsys.readouterr().out, "plan:").startswith("coder/all: 20 case(s)")


def test_the_verb_names_the_tier_it_measured_and_how_much_of_the_corpus_that_is(tmp_path, capsys):
    _workspace_corpus(tmp_path)

    run_bench(CODER_SITE_ID, tier="smoke", dry_run=True)

    printed = _line(capsys.readouterr().out, "tier:")
    assert printed.startswith("smoke")
    assert "12 of 20" in printed


def test_a_tiered_run_that_states_no_width_opens_the_pool_the_tier_derives(tmp_path, capsys):
    _workspace_corpus(tmp_path)

    run_bench(CODER_SITE_ID, tier="smoke", dry_run=True)

    assert _line(capsys.readouterr().out, "workers:") == str(TIER_WORKER_CAP)


def test_an_untiered_run_that_states_no_width_still_works_the_jobs_one_at_a_time(tmp_path, capsys):
    _workspace_corpus(tmp_path)

    run_bench(CODER_SITE_ID, dry_run=True)

    assert _line(capsys.readouterr().out, "workers:") == str(SEQUENTIAL_WORKERS)


def test_a_stated_worker_count_wins_over_the_tier_derivation_through_the_cli(tmp_path):
    _workspace_corpus(tmp_path)

    result = runner.invoke(
        app,
        ["bench", "run", "--site", "coder", "--tier", "smoke", "--workers", "2", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert _line(result.output, "workers:") == "2"


def test_an_unknown_tier_through_the_cli_is_refused_naming_the_declared_tiers(tmp_path):
    _workspace_corpus(tmp_path)

    result = runner.invoke(app, ["bench", "run", "--site", "coder", "--tier", "quick", "--dry-run"])

    assert result.exit_code == 1
    assert "quick" in result.output
    assert "smoke" in result.output


def test_a_tier_and_a_split_together_through_the_cli_are_refused_naming_both_flags(tmp_path):
    _workspace_corpus(tmp_path)

    result = runner.invoke(
        app,
        ["bench", "run", "--site", "coder", "--tier", "smoke", "--split", "tuning", "--dry-run"],
    )

    assert result.exit_code == 1
    assert "--tier" in result.output and "--split" in result.output


# ── one stubbed smoke run, over the pool ──────────────────────────────────────────────────────


def _landed(request: AttemptRequest) -> Attempt:
    """A stub authoring job that landed its file on the opening ask — a real retained record."""
    record = JobRecord(
        case_id=request.case.case_id,
        strategy=strategy_name(request.case.case_id),
        passed=True,
        attempts=(
            AttemptRecord(
                attempt=1,
                passed=True,
                escalated=False,
                error=None,
                model="bench/oracle",
                served_model="bench/oracle-2026",
                usage={"input_tokens": 100, "output_tokens": 50},
            ),
        ),
        error=None,
        model="bench/oracle",
        seconds=1.5,
    )
    return Attempt(
        outcome=AttemptOutcome(passed=True, model="bench/oracle", seconds=1.5),
        output=json.dumps(record.document(), sort_keys=True),
        served_model="bench/oracle-2026",
    )


def _smoke_record(tmp_path: Path) -> dict:
    """One stubbed smoke bench, and the record it wrote."""
    _workspace_corpus(tmp_path)
    run_bench(CODER_SITE_ID, tier="smoke", seams=BenchSeams(attempt=_landed))
    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    return json.loads((directory / BENCH_RECORD_NAME).read_text(encoding="utf-8"))


def test_a_stubbed_smoke_run_over_the_pool_writes_one_valid_record_for_its_twelve_cases(tmp_path):
    record = _smoke_record(tmp_path)

    assert validate(record) == []
    assert record["bench"]["cases"] == SMOKE_SIZE
    assert record["corpus"]["case_count"] == SMOKE_SIZE
    assert record["bench"]["complete"] is True


def test_a_stubbed_smoke_runs_record_carries_the_coder_reading_over_the_tier_it_measured(tmp_path):
    block = _smoke_record(tmp_path)["harness"]["dials"][CODER_DIALS_KEY]

    assert block["jobs"] == SMOKE_SIZE
    assert block["rates"]["job_pass_rate"] == 1.0
