"""The bench runner (#202): a minted id, an isolated bench area, artifacts, preflight, one record.

The runner is the eval layer's one impure module, so this is the one eval suite with directories in
it. Everything is still driven through stubs — a stub site declaration, a stub attempt callable, a
throwaway workspace — because a benchmark suite that needed a model to run would be a suite nobody
runs.

Four properties carry the weight:

* **A bench run is not a run.** It mints its own id and writes under ``<workspace>/bench/<id>/`` and
  nowhere else: the whole workspace tree is snapshotted before and after, and the runs tree is
  proved untouched.
* **The evidence survives the run.** Every case keeps the prompt it was asked with and every
  attempt's output and error, in its own throwaway area, and :func:`replay` rebuilds all of it from
  disk alone.
* **Nothing is spent before it is stated.** The preflight is pure — zero attempt calls, zero
  writes — and a run refuses to start unless the plan it would execute is the plan somebody
  acknowledged.
* **An unreadable record is refused, not written.** The document is the pure builder's, checked by
  the pure validator, and a bench that cannot produce a valid one leaves no ``bench.json`` behind.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from noctis.eval.case import Case, Provenance, ProvenanceKind, Split
from noctis.eval.corpus import Corpus
from noctis.eval.harness import AS_SHIPPED, HarnessSpec
from noctis.eval.identity import SiteIdentity
from noctis.eval.knobs import SiteKnobs, UnknownKnob
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.record import EngineStamp, ModelConfig, validate
from noctis.eval.registry import UnknownSite, index_sites
from noctis.eval.runner import (
    BENCH_DIRNAME,
    BENCH_INDEX_NAME,
    BENCH_RECORD_NAME,
    Attempt,
    AttemptRequest,
    BenchRunner,
    InvalidBenchRecord,
    SequentialExecutor,
    SpendUnacknowledged,
    UnsafeArtifactName,
    bench_root,
    harness_dials,
    harness_hash,
    new_bench_id,
    replay,
)
from noctis.eval.site import AgentSite
from noctis.observability.debug.runid import RUN_ID_RE
from noctis.research.pricing import ModelPrice, PriceTable

REPO_ROOT = Path(__file__).resolve().parents[1]

# One dollar per million tokens on every field, so an attempt of four million tokens costs exactly
# $4 and a reader can check a preflight with a pencil (the record suite's own fixture table).
MTOK = 1_000_000
BENCH_TABLE = PriceTable(version="bench-1", prices={"bench/oracle": ModelPrice(1.0, 1.0, 1.0, 1.0)})
FOUR_MTOK = dict.fromkeys(
    ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"),
    MTOK,
)

AUTHORED = Provenance(kind=ProvenanceKind.AUTHORED, reference="2026-07-20")

SITE_IDENTITY = SiteIdentity(
    site_id="stub",
    version="1",
    prompt_asset_groups=("stub",),
    prompt_asset_hash="7c1f0d9a4b2e6f83",
)

ENGINE = EngineStamp(
    engine_version=3,
    fingerprint={"gates": "f63d47b7b9604ab1", "backtest": "3ba3e0bf1c97134f"},
    noctis_version="0.1.0",
)

ORACLE = ModelConfig(config_id="primary", provider="bench", requested_model="bench/oracle")


@dataclass(frozen=True)
class StubKnobs(SiteKnobs):
    """A stub site's levers — one of them named so a dial can publish a timestamp."""

    depth: int = 1
    deadline_utc: str | None = None


def _render(payload: Mapping[str, Any], spec: HarnessSpec) -> str:
    """The stub site's renderer: the ask, plus whichever dials the harness composed in."""
    return f"ask={payload['ask']} sheet={spec.contract_sheet} example={spec.worked_example!r}"


STUB_SITE: AgentSite[Any, Any] = AgentSite(
    id="stub", version="1", contract=None, render=_render, knobs=StubKnobs
)
STUB_REGISTRY = index_sites((STUB_SITE,))


class FakeClock:
    """A clock that never surprises a record: one fixed UTC instant, ticking a second per read."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 7, 20, 14, 42, 33, tzinfo=UTC)

    def __call__(self) -> datetime:
        moment = self._now
        self._now = datetime.fromtimestamp(self._now.timestamp() + 1, tz=UTC)
        return moment


@dataclass
class StubAttempt:
    """A stub model call: records every request, answers by a per-case script.

    ``script`` maps a case id to the pass/fail verdict of each attempt in order; a case the script
    does not name passes on its first attempt.
    """

    script: Mapping[str, tuple[bool, ...]] = field(default_factory=dict)
    raises: frozenset[str] = frozenset()
    requests: list[AttemptRequest] = field(default_factory=list)

    def __call__(self, request: AttemptRequest) -> Attempt:
        self.requests.append(request)
        if request.case.case_id in self.raises:
            raise RuntimeError(f"the seam fell over on {request.case.case_id}")
        verdicts = self.script.get(request.case.case_id, (True,))
        passed = verdicts[min(request.attempt, len(verdicts)) - 1]
        return Attempt(
            outcome=AttemptOutcome(
                passed=passed,
                model="bench/oracle",
                seconds=2.0,
                error=None if passed else f"{request.case.case_id} did not settle",
                **FOUR_MTOK,
            ),
            output=f"answer to {request.case.case_id} #{request.attempt}",
            served_model="bench/oracle-20260701",
        )


def _case(case_id: str, **overrides: Any) -> Case:
    fields: dict[str, Any] = {
        "case_id": case_id,
        "site_id": "stub",
        "payload": {"ask": f"do {case_id}"},
        "provenance": AUTHORED,
    }
    fields.update(overrides)
    return Case(**fields)


def _corpus(*case_ids: str) -> Corpus:
    return Corpus(site_id="stub", cases=tuple(_case(case_id) for case_id in case_ids))


def _workspace(tmp_path: Path) -> Path:
    """A workspace that already holds a run and a data lake — neither of which a bench may touch."""
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / "20260719T101010Z-abc123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text('{"kind": "noctis.run"}\n', encoding="utf-8")
    lake = workspace / "data_lake"
    lake.mkdir(parents=True, exist_ok=True)
    (lake / "AAPL.parquet").write_text("bars", encoding="utf-8")
    return workspace


def _runner(tmp_path: Path, attempt: StubAttempt | None = None, **overrides: Any) -> BenchRunner:
    """A runner over a throwaway workspace, wired to stubs from end to end."""
    settings: dict[str, Any] = {
        "bench_root": bench_root(_workspace(tmp_path)),
        "attempt": attempt or StubAttempt(),
        "registry": STUB_REGISTRY,
        "identity": SITE_IDENTITY,
        "engine": ENGINE,
        "table": BENCH_TABLE,
        "clock": FakeClock(),
    }
    settings.update(overrides)
    return BenchRunner(**settings)


def _run(runner: BenchRunner, corpus: Corpus, **kwargs: Any):
    """Preflight, then run what the preflight stated — the two-step every live bench takes."""
    plan = runner.preflight(corpus, **kwargs)
    return runner.run(corpus, acknowledged=plan, **kwargs)


def _tree(root: Path) -> dict[str, str]:
    """Every path under ``root``, files hashed — the snapshot an isolation claim is checked with."""
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "<dir>"
        )
        for path in sorted(root.rglob("*"))
    }


# ── the bench id, and the area it owns ────────────────────────────────────────────────────


def test_a_bench_run_mints_a_fresh_id_in_the_minted_shape(tmp_path):
    """Minted, never derived: the run-id discipline, and the run-id shape."""
    bench = _run(_runner(tmp_path), _corpus("alpha", "beta"))

    assert RUN_ID_RE.match(bench.bench_id)


def test_two_bench_runs_never_share_an_id_or_a_directory(tmp_path):
    first = _run(_runner(tmp_path), _corpus("alpha"))
    second = _run(_runner(tmp_path), _corpus("alpha"))

    assert first.bench_id != second.bench_id
    assert first.directory != second.directory


def test_the_bench_area_hangs_off_the_workspace_root_beside_the_shared_data_lake(tmp_path):
    """Run-neutral, like ``data_lake/`` — never one of the run-scoped derived paths."""
    workspace = tmp_path / "workspace"

    assert bench_root(workspace) == workspace / BENCH_DIRNAME


def test_a_whole_bench_writes_nothing_outside_its_own_bench_directory(tmp_path):
    """The hard boundary: a bench run is not a run, and the workspace proves it."""
    runner = _runner(tmp_path)
    workspace = runner.bench_root.parent
    before = _tree(workspace)

    bench = _run(runner, _corpus("alpha", "beta"))

    after = _tree(workspace)
    assert {path: digest for path, digest in after.items() if path in before} == before
    added = set(after) - set(before)
    assert added
    assert all(path == BENCH_DIRNAME or path.startswith(f"{BENCH_DIRNAME}/") for path in added)
    assert all(
        path.startswith(f"{BENCH_DIRNAME}/{bench.bench_id}/")
        for path in added
        if after[path] != "<dir>"
    )


def test_the_runs_tree_is_untouched_by_a_bench_run(tmp_path):
    """No run lock, no champions registry, no run-board entry — nothing under ``runs/``."""
    runner = _runner(tmp_path)
    runs = runner.bench_root.parent / "runs"
    before = _tree(runs)

    _run(runner, _corpus("alpha", "beta"))

    assert _tree(runs) == before


def test_the_bench_directory_holds_the_record_under_the_declared_name(tmp_path):
    bench = _run(_runner(tmp_path), _corpus("alpha"))

    assert (bench.directory / BENCH_RECORD_NAME).is_file()


def test_the_derived_bench_index_slot_is_named_but_never_written(tmp_path):
    """The listing roll-up is a later story's; this one leaves the slot empty, not guessed."""
    runner = _runner(tmp_path)

    _run(runner, _corpus("alpha"))

    assert BENCH_INDEX_NAME
    assert not (runner.bench_root / BENCH_INDEX_NAME).exists()


def test_the_bench_results_area_is_gitignored():
    """Nothing a bench writes may reach git — the numbers are the operator's, not the repo's."""
    checked = subprocess.run(
        ["git", "check-ignore", "-q", f"workspace/{BENCH_DIRNAME}"], cwd=REPO_ROOT
    )

    assert checked.returncode == 0


def test_a_minted_bench_id_is_a_fresh_one_every_time():
    assert new_bench_id() != new_bench_id()


# ── per-case artifacts, and the throwaway area they live in ───────────────────────────────


def test_every_case_keeps_the_prompt_it_was_asked_with(tmp_path):
    attempt = StubAttempt()
    bench = _run(_runner(tmp_path, attempt), _corpus("alpha", "beta"))

    asked = {request.case.case_id: request.prompt for request in attempt.requests}
    kept = {artifacts.case_id: artifacts.prompt for artifacts in replay(bench.directory)}
    assert kept == asked


def test_every_attempt_keeps_its_output_and_its_error(tmp_path):
    """One case that settles at once, one that needs a second try, one that never gets there."""
    attempt = StubAttempt(script={"beta": (False, True), "gamma": (False, False)})
    runner = _runner(tmp_path, attempt, max_attempts=2)

    bench = _run(runner, _corpus("alpha", "beta", "gamma"))

    kept = {artifacts.case_id: artifacts.attempts for artifacts in replay(bench.directory)}
    assert [a.passed for a in kept["alpha"]] == [True]
    assert [a.passed for a in kept["beta"]] == [False, True]
    assert kept["beta"][0].output == "answer to beta #1"
    assert kept["beta"][0].error == "beta did not settle"
    assert kept["beta"][1].error is None
    assert [a.error for a in kept["gamma"]] == ["gamma did not settle"] * 2


def test_the_case_artifacts_are_replayable_from_the_directory_alone(tmp_path):
    """Replay reads the tree, not the run: a copied bench directory rebuilds the same evidence."""
    bench = _run(_runner(tmp_path), _corpus("alpha", "beta"))

    assert [artifacts.case_id for artifacts in replay(bench.directory)] == ["alpha", "beta"]
    assert all(artifacts.directory.is_dir() for artifacts in replay(bench.directory))


def test_each_case_is_handed_its_own_empty_work_area_under_the_bench_directory(tmp_path):
    """The isolation seam: a fresh area per job, inside the bench directory, and nowhere else."""
    attempt = StubAttempt()

    bench = _run(_runner(tmp_path, attempt), _corpus("alpha", "beta"))

    areas = [request.workdir for request in attempt.requests]
    assert len(set(areas)) == len(areas)
    assert all(area.is_dir() and area.is_relative_to(bench.directory) for area in areas)


def test_a_file_one_case_writes_stays_inside_that_case_s_own_area(tmp_path):
    """Confinement is the point: nothing a case writes is visible to the next one."""

    class Scribbler(StubAttempt):
        def __call__(self, request: AttemptRequest) -> Attempt:
            (request.workdir / "scratch.txt").write_text(request.case.case_id, encoding="utf-8")
            return super().__call__(request)

    attempt = Scribbler()
    bench = _run(_runner(tmp_path, attempt), _corpus("alpha", "beta"))

    written = sorted(path.read_text() for path in bench.directory.rglob("scratch.txt"))
    assert written == ["alpha", "beta"]
    for request in attempt.requests:
        assert [p.name for p in request.workdir.iterdir()] == ["scratch.txt"]


def test_a_completed_case_has_persisted_its_artifacts_before_the_next_one_starts(tmp_path):
    """Artifacts land as each case finishes, so a killed bench keeps what it already learned."""
    seen: list[list[str]] = []
    area: list[Path] = []

    class Watcher(StubAttempt):
        def __call__(self, request: AttemptRequest) -> Attempt:
            bench_dir = next(iter(area[0].iterdir()))
            seen.append([artifacts.case_id for artifacts in replay(bench_dir)])
            return super().__call__(request)

    runner = _runner(tmp_path, Watcher())
    area.append(runner.bench_root)

    _run(runner, _corpus("alpha", "beta", "gamma"))

    assert seen == [[], ["alpha"], ["alpha", "beta"]]


def test_a_case_id_that_would_escape_its_own_directory_is_refused(tmp_path):
    escaping = Corpus(site_id="stub", cases=(_case("../../etc/passwd"),))
    runner = _runner(tmp_path)

    with pytest.raises(UnsafeArtifactName, match="passwd"):
        _run(runner, escaping)


# ── site invocation and the harness spec ──────────────────────────────────────────────────


def test_the_runner_renders_each_case_through_the_site_the_corpus_names(tmp_path):
    attempt = StubAttempt()

    _run(_runner(tmp_path, attempt), _corpus("alpha"))

    assert attempt.requests[0].prompt.startswith("ask=do alpha")


def test_the_declared_renderer_is_handed_the_harness_spec_it_was_asked_for(tmp_path):
    """The ablation reaches the prompt, or the harness dials are decoration."""
    attempt = StubAttempt()
    ablated = HarnessSpec(contract_sheet=False, worked_example=None)

    _run(_runner(tmp_path, attempt, harness=ablated), _corpus("alpha"))

    assert "sheet=False" in attempt.requests[0].prompt
    assert "example=None" in attempt.requests[0].prompt


def test_a_corpus_naming_a_site_nothing_declares_is_refused_before_anything_is_spent(tmp_path):
    attempt = StubAttempt()
    runner = _runner(tmp_path, attempt)
    foreign = Corpus(site_id="nowhere", cases=(_case("alpha", site_id="nowhere"),))

    with pytest.raises(UnknownSite, match="nowhere"):
        _run(runner, foreign)
    assert attempt.requests == []
    assert not runner.bench_root.exists()


def test_a_knob_override_the_site_does_not_declare_is_refused_before_anything_is_spent(tmp_path):
    attempt = StubAttempt()
    runner = _runner(tmp_path, attempt, harness=HarnessSpec(knob_overrides={"temperature": 0.7}))

    with pytest.raises(UnknownKnob, match="temperature"):
        _run(runner, _corpus("alpha"))
    assert attempt.requests == []
    assert not runner.bench_root.exists()


def test_the_harness_spec_hash_lands_in_the_record_beside_the_dials_it_hashed(tmp_path):
    spec = HarnessSpec(template=False, knob_overrides={"depth": 3})
    bench = _run(_runner(tmp_path, harness=spec), _corpus("alpha"))

    assert bench.record["harness"]["hash"] == harness_hash(spec)
    assert bench.record["harness"]["dials"] == harness_dials(spec)
    assert bench.record["harness"]["dials"]["knob_overrides"] == {"depth": 3}


def test_the_same_harness_spec_hashes_the_same_way_every_time():
    assert harness_hash(HarnessSpec()) == harness_hash(HarnessSpec())


@pytest.mark.parametrize(
    "spec",
    [
        HarnessSpec(contract_sheet=False),
        HarnessSpec(template=False),
        HarnessSpec(worked_example=None),
        HarnessSpec(worked_example="donchian_breakout"),
        HarnessSpec(feasibility_rules=False),
        HarnessSpec(retry_hints=False),
        HarnessSpec(knob_overrides={"depth": 2}),
    ],
)
def test_every_dial_that_moves_moves_the_harness_hash(spec):
    assert harness_hash(spec) != harness_hash(HarnessSpec())


def test_a_worked_example_named_after_the_as_shipped_token_is_still_its_own_composition():
    """The rendering is tagged, so a strategy called ``as_shipped`` is not the shipped example."""
    named = HarnessSpec(worked_example=AS_SHIPPED.value)

    assert harness_hash(named) != harness_hash(HarnessSpec(worked_example=AS_SHIPPED))


def test_the_production_composition_says_so_in_the_dials_it_publishes():
    assert harness_dials(HarnessSpec())["is_production"] is True
    assert harness_dials(HarnessSpec(template=False))["is_production"] is False


# ── execution over the jobs ───────────────────────────────────────────────────────────────


def test_every_case_is_attempted_once_per_rep(tmp_path):
    attempt = StubAttempt()

    bench = _run(_runner(tmp_path, attempt), _corpus("alpha", "beta"), reps=3)

    assert len(attempt.requests) == 6
    assert len(bench.runs) == 6
    assert bench.record["bench"]["cases"] == 2
    assert bench.record["bench"]["runs"] == 6


def test_every_case_is_attempted_once_per_configuration(tmp_path):
    attempt = StubAttempt()
    configs = (ORACLE, ModelConfig(config_id="challenger", requested_model="bench/oracle"))

    bench = _run(_runner(tmp_path, attempt), _corpus("alpha"), configs=configs)

    assert sorted(request.config.config_id for request in attempt.requests) == [
        "challenger",
        "primary",
    ]
    assert sorted(run.config_id for run in bench.runs) == ["challenger", "primary"]


def test_a_rep_retries_until_it_passes_within_the_attempt_ceiling(tmp_path):
    attempt = StubAttempt(script={"alpha": (False, True), "beta": (False, False, True)})

    bench = _run(_runner(tmp_path, attempt, max_attempts=2), _corpus("alpha", "beta"))

    settled = {run.result.case_id: run.result.attempts_to_pass for run in bench.runs}
    assert settled == {"alpha": 2, "beta": None}
    assert len(attempt.requests) == 4


def test_a_case_that_passes_first_time_is_never_retried(tmp_path):
    attempt = StubAttempt()

    _run(_runner(tmp_path, attempt, max_attempts=3), _corpus("alpha"))

    assert len(attempt.requests) == 1


def test_an_attempt_callable_that_raises_fails_its_case_rather_than_the_bench(tmp_path):
    attempt = StubAttempt(raises=frozenset({"beta"}))

    bench = _run(_runner(tmp_path, attempt), _corpus("alpha", "beta", "gamma"))

    outcomes = {run.result.case_id: run.result.passed for run in bench.runs}
    assert outcomes == {"alpha": True, "beta": False, "gamma": True}
    errored = next(run for run in bench.runs if run.result.case_id == "beta")
    assert "fell over" in (errored.result.attempts[0].error or "")


def test_the_error_a_raising_seam_left_is_kept_on_disk_with_the_rest(tmp_path):
    bench = _run(_runner(tmp_path, StubAttempt(raises=frozenset({"beta"}))), _corpus("beta"))

    kept = replay(bench.directory)[0]
    assert "fell over" in (kept.attempts[0].error or "")


def test_a_bench_whose_every_job_ran_reports_itself_complete(tmp_path):
    bench = _run(_runner(tmp_path), _corpus("alpha", "beta"))

    assert bench.record["bench"]["complete"] is True


def test_a_bench_of_no_reps_is_refused_rather_than_quietly_run_once(tmp_path):
    """A clamp is a small silent lie: a count below one is somebody's arithmetic gone wrong."""
    runner = _runner(tmp_path)

    with pytest.raises(ValueError, match="reps must be at least 1"):
        runner.preflight(_corpus("alpha"), reps=0)


def test_a_bench_with_no_attempt_ceiling_at_all_is_refused(tmp_path):
    runner = _runner(tmp_path, max_attempts=0)

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        runner.preflight(_corpus("alpha"))


def test_two_configurations_answering_to_one_id_are_refused(tmp_path):
    """Every result is attributed by config id, so two sharing one are unattributable."""
    runner = _runner(tmp_path)
    twins = (ORACLE, ModelConfig(config_id="primary", requested_model="other/model"))

    with pytest.raises(ValueError, match="primary"):
        runner.preflight(_corpus("alpha"), configs=twins)


def test_a_bench_under_no_configuration_at_all_is_refused(tmp_path):
    runner = _runner(tmp_path)

    with pytest.raises(ValueError, match="at least one model configuration"):
        runner.preflight(_corpus("alpha"), configs=())


def test_the_executor_seam_is_sequential_and_states_its_one_worker():
    """#203 widens this seam to a pool; today it is one worker, and it says so."""
    assert SequentialExecutor().workers == 1


def test_the_executor_runs_every_job_it_is_handed_in_order():
    executor = SequentialExecutor()

    assert executor.run(lambda job: job, [1, 2, 3]) == (1, 2, 3)


# ── the split a bench measures over ───────────────────────────────────────────────────────


def test_a_bench_may_be_confined_to_the_corpus_holdout(tmp_path):
    attempt = StubAttempt()
    corpus = _corpus(*[f"case-{index:03d}" for index in range(10)])

    bench = _run(_runner(tmp_path, attempt), corpus, split=Split.HOLDOUT)

    asked = {request.case.case_id for request in attempt.requests}
    assert asked == {case.case_id for case in corpus.holdout}
    assert bench.record["corpus"]["split"] == "holdout"


def test_a_bench_over_the_whole_corpus_says_so_rather_than_leaving_it_unknown(tmp_path):
    bench = _run(_runner(tmp_path), _corpus("alpha", "beta"))

    assert bench.record["corpus"]["split"] == "all"


# ── the spend preflight ───────────────────────────────────────────────────────────────────


def test_the_preflight_states_the_cases_reps_and_configurations_it_would_run(tmp_path):
    runner = _runner(tmp_path, max_attempts=2)

    plan = runner.preflight(_corpus("alpha", "beta"), reps=3)

    assert (plan.cases, plan.reps, plan.configs) == (2, 3, ("default",))
    assert plan.jobs == 6
    assert plan.attempts_ceiling == 12


def test_the_preflight_prices_the_planned_attempts_from_the_table_and_the_estimate(tmp_path):
    """Four million tokens at $1/Mtok is $4 an attempt: two cases, two reps, one attempt each."""
    runner = _runner(tmp_path, tokens_per_attempt=FOUR_MTOK)

    plan = runner.preflight(_corpus("alpha", "beta"), reps=2, configs=(ORACLE,))

    assert plan.usd_ceiling_estimate == pytest.approx(16.0)
    assert plan.tokens_ceiling == 16 * MTOK
    assert plan.pricing_table_version == "bench-1"


def test_a_model_the_table_cannot_price_leaves_the_estimate_unknown_and_names_it(tmp_path):
    """The pricing module's posture: unknown is ``None``, never a confident zero."""
    runner = _runner(tmp_path, tokens_per_attempt=FOUR_MTOK)
    stranger = ModelConfig(config_id="primary", requested_model="acme/mystery-1")

    plan = runner.preflight(_corpus("alpha"), configs=(stranger,))

    assert plan.usd_ceiling_estimate is None
    assert plan.unpriced_models == ("acme/mystery-1",)
    assert "n/a" in plan.summary()


def test_a_preflight_with_no_token_estimate_reports_no_dollar_figure_at_all(tmp_path):
    plan = _runner(tmp_path).preflight(_corpus("alpha"), configs=(ORACLE,))

    assert plan.usd_ceiling_estimate is None
    assert plan.tokens_ceiling is None


def test_the_preflight_calls_no_attempt_and_writes_nothing_at_all(tmp_path):
    """The dry run *is* the preflight — one code path, and it never spends."""
    attempt = StubAttempt()
    runner = _runner(tmp_path, attempt, tokens_per_attempt=FOUR_MTOK)
    workspace = runner.bench_root.parent
    before = _tree(workspace)

    runner.preflight(_corpus("alpha", "beta"), reps=2)

    assert attempt.requests == []
    assert _tree(workspace) == before


def test_a_run_that_acknowledges_no_plan_is_refused_before_it_spends(tmp_path):
    attempt = StubAttempt()
    runner = _runner(tmp_path, attempt)

    with pytest.raises(SpendUnacknowledged, match="preflight"):
        runner.run(_corpus("alpha"))
    assert attempt.requests == []
    assert not runner.bench_root.exists()


def test_a_run_that_acknowledges_a_plan_for_another_shape_is_refused(tmp_path):
    """Acknowledging two cases must not authorize two hundred."""
    attempt = StubAttempt()
    runner = _runner(tmp_path, attempt)
    small = runner.preflight(_corpus("alpha"), reps=1)

    with pytest.raises(SpendUnacknowledged, match="differs"):
        runner.run(_corpus("alpha"), reps=4, acknowledged=small)
    assert attempt.requests == []


def test_a_plan_acknowledged_for_another_corpus_is_refused(tmp_path):
    runner = _runner(tmp_path)
    other = runner.preflight(_corpus("gamma"))

    with pytest.raises(SpendUnacknowledged, match="differs"):
        runner.run(_corpus("alpha"), acknowledged=other)


def test_the_plan_a_run_executed_is_the_plan_it_reports(tmp_path):
    runner = _runner(tmp_path)
    plan = runner.preflight(_corpus("alpha", "beta"), reps=2)

    bench = runner.run(_corpus("alpha", "beta"), reps=2, acknowledged=plan)

    assert bench.plan == plan


# ── the record ────────────────────────────────────────────────────────────────────────────


def test_the_written_record_validates_clean_against_the_bench_schema(tmp_path):
    bench = _run(_runner(tmp_path), _corpus("alpha", "beta"), reps=2)

    document = json.loads((bench.directory / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert validate(document) == []
    assert document == json.loads(json.dumps(bench.record))


def test_the_record_carries_the_identities_the_runner_was_handed(tmp_path):
    bench = _run(_runner(tmp_path, corpus_version="2026-07"), _corpus("alpha"))

    assert bench.record["site"]["site_id"] == "stub"
    assert bench.record["site"]["prompt_asset_hash"] == SITE_IDENTITY.prompt_asset_hash
    assert bench.record["engine"]["engine_version"] == 3
    assert bench.record["corpus"]["version"] == "2026-07"
    assert bench.record["corpus"]["hash"] == _corpus("alpha").digest
    assert bench.record["bench"]["bench_id"] == bench.bench_id


def test_the_record_stamps_the_two_moments_the_runner_observed(tmp_path):
    bench = _run(_runner(tmp_path), _corpus("alpha"))

    assert bench.record["bench"]["started_utc"] == "2026-07-20T14:42:33Z"
    assert bench.record["bench"]["finished_utc"].endswith("Z")


def test_every_attempt_s_served_model_reaches_the_record_s_provenance_tier(tmp_path):
    bench = _run(_runner(tmp_path), _corpus("alpha"), configs=(ORACLE,))

    config = bench.record["provenance"]["configs"][0]
    assert config["served_models"] == ["bench/oracle-20260701"]
    assert config["alias_drift"] is False


def test_a_record_that_would_not_validate_is_refused_and_never_written(tmp_path):
    """A dial publishing a malformed timestamp makes the document unreadable — so it is not one."""
    runner = _runner(tmp_path, harness=HarnessSpec(knob_overrides={"deadline_utc": "yesterday"}))

    with pytest.raises(InvalidBenchRecord, match="deadline_utc"):
        _run(runner, _corpus("alpha"))
    written = list(runner.bench_root.rglob(BENCH_RECORD_NAME))
    assert written == []


def test_a_refused_record_leaves_the_evidence_it_collected_behind(tmp_path):
    """The artifacts are what an operator debugs the refusal with; deleting them would hide it."""
    runner = _runner(tmp_path, harness=HarnessSpec(knob_overrides={"deadline_utc": "yesterday"}))

    with pytest.raises(InvalidBenchRecord):
        _run(runner, _corpus("alpha"))

    assert list(runner.bench_root.rglob("prompt.txt"))
