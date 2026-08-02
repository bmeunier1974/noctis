"""``bench run --site`` (#210): the first bench verb that spends, preflight-gated and site-agnostic.

The verb is a composition, so the assertions are about what an operator sees and what lands on
disk. Four properties carry the story:

* **Nothing is spent before it is stated.** ``--dry-run`` preflights, prints the plan, and leaves
  the workspace byte-identical — no bench directory, no model client, no key needed. A live run
  through the CLI executes *exactly* the plan it printed, because the CLI has no path that skips
  the preflight: the runner's ``SpendUnacknowledged`` is unreachable from here.
* **The verb knows nothing about DECIDE.** How a case becomes a site's renderer input is a lookup
  in the eval layer (:data:`~noctis.eval.bootstrap.SITE_ASKS`), so a second site with no adapter at
  all runs through the same code path on the payload its cases declare.
* **The flags reach the runner.** ``--reps`` asks every case that many times, ``--split`` selects
  the half of the corpus, ``--workers`` states the width the jobs are worked at.
* **The refusals come from the machinery, rendered cleanly.** An undeclared site names the declared
  ones; an absent corpus names the directory it looked in; a live run with no client says why.

Everything is stubbed: a stub site declaration, a stub attempt callable, a fake LLM client, a
throwaway workspace. Nothing here touches a network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from noctis.cli import app
from noctis.config.rehydrate import freeze_inputs
from noctis.config.settings import Settings, load_settings
from noctis.eval.bootstrap import (
    BenchSeams,
    LiveModelUnavailable,
    SiteAsk,
    build_executor,
    cases_root,
    live_attempt,
)
from noctis.eval.case import Case
from noctis.eval.case_provider import YamlCaseProvider
from noctis.eval.cli import run_bench
from noctis.eval.decide_miner import mine_decide_corpus
from noctis.eval.harness import HarnessSpec
from noctis.eval.identity import SiteIdentity
from noctis.eval.knobs import SiteKnobs
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.pool import PooledExecutor
from noctis.eval.record import ModelConfig, validate
from noctis.eval.registry import index_sites
from noctis.eval.runner import (
    BENCH_RECORD_NAME,
    Attempt,
    AttemptRequest,
    SequentialExecutor,
    bench_root,
)
from noctis.eval.site import AgentSite
from noctis.research.episode import EmitContract
from noctis.research.journal import ExperimentJournal
from noctis.research.llm import Capabilities, ToolCall, Turn

from ._promotion_cases import card

runner = CliRunner()

RUN_ID = "20260720T144233Z-a3f9c1"
MIN_TRIALS = 6


# ── a second site, declared entirely outside the shipped registry ─────────────────────────────


@dataclass(frozen=True)
class StubKnobs(SiteKnobs):
    """The stub site's one lever, so a harness has something to validate an override against."""

    depth: int = 1


@dataclass(frozen=True)
class StubOutput:
    """What the stub site's contract parses an emitted payload into."""

    answer: str


def _render(payload: Mapping[str, Any], spec: HarnessSpec) -> str:
    """The stub site's renderer — the ask its case declares, nothing else."""
    return f"ask={payload['ask']}"


STUB_CONTRACT: EmitContract[StubOutput] = EmitContract(
    name="emit_stub",
    description="Emit the stub answer.",
    schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    parse=lambda payload: StubOutput(answer=str(payload["answer"])),
)

STUB_SITE: AgentSite[Any, Any] = AgentSite(
    id="stub", version="1", contract=STUB_CONTRACT, render=_render, knobs=StubKnobs
)
STUB_REGISTRY = index_sites((STUB_SITE,))
STUB_IDENTITY = SiteIdentity(
    site_id="stub",
    version="1",
    prompt_asset_groups=("stub",),
    prompt_asset_hash="7c1f0d9a4b2e6f83",
)


@dataclass
class StubAttempt:
    """A stub model call: records every request it was handed, and passes."""

    requests: list[AttemptRequest] = field(default_factory=list)

    def __call__(self, request: AttemptRequest) -> Attempt:
        self.requests.append(request)
        return Attempt(
            outcome=AttemptOutcome(passed=True, model="bench/oracle", seconds=0.1),
            output=f"answered {request.case.case_id}",
            served_model="bench/oracle-2026",
        )


def _seams(attempt: StubAttempt | None = None) -> BenchSeams:
    """The stub site's whole assembly: its registry, its identity, and a stub model call."""
    return BenchSeams(
        attempt=attempt or StubAttempt(), registry=STUB_REGISTRY, identity=STUB_IDENTITY
    )


def _stub_corpus(tmp_path: Path, ids: tuple[str, ...] = ("a", "b", "c", "d")) -> Path:
    """Four stub cases in one difficulty stratum — a 3/1 tuning/holdout deal."""
    directory = cases_root(tmp_path / "workspace") / "stub"
    directory.mkdir(parents=True)
    for name in ids:
        document = {
            "site_id": "stub",
            "payload": {"ask": f"question-{name}"},
            "provenance": "authored:2026-07-20",
            "difficulty": {"reasoning": "hard"},
        }
        (directory / f"case-{name}.yaml").write_text(
            yaml.safe_dump(document, sort_keys=True), encoding="utf-8"
        )
    return directory


# ── a real DECIDE corpus, mined from a workspace the way #208 mines one ───────────────────────


def _candidate(journal: ExperimentJournal, name: str, *, promoted: bool) -> None:
    """One candidate's whole tuning history, and the verdict the gates then recorded."""
    journal.record_class_tag(name, "volatility_squeeze")
    journal.record_thesis(name, "volatility compresses before it expands")
    for index in range(MIN_TRIALS):
        journal.record_trial(
            name,
            source="sweep",
            symbols=["AAPL", "MSFT"],
            params={"lookback": 10 + index},
            window={"start": "2024-01-02", "end": "2025-01-02"},
            card=card(test=1.0 - index * 0.01, train=1.2),
        )
    journal.record_sweep_complete(name, n_trials=MIN_TRIALS, symbols=["AAPL", "MSFT"])
    journal.record_scorecard(
        name, card(test=1.0, train=1.2, holdout=2.0 if promoted else 0.0, symbol_holdout=2.0)
    )
    journal.record_approval(
        name,
        promoted=promoted,
        rationale="the panel holds up out of sample",
        params={"lookback": 10},
        symbols=["AAPL", "MSFT"],
        holdout_symbols=["NVDA"],
    )


def _decide_corpus(tmp_path: Path) -> Path:
    """A two-case DECIDE corpus mined into the workspace's cases root."""
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / RUN_ID
    (run_dir / "state").mkdir(parents=True)
    settings = Settings()
    settings.research.min_trials = MIN_TRIALS
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run": {"run_id": RUN_ID},
                "inputs": freeze_inputs(
                    settings, frozen_at="2026-07-20T14:00:00Z", execution_mode="paper"
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    journal = ExperimentJournal(run_dir / "state")
    _candidate(journal, "vol_breakout_squeeze", promoted=True)
    _candidate(journal, "gap_fade_open", promoted=False)
    mine_decide_corpus(workspace / "runs", cases_root=cases_root(workspace))
    return cases_root(workspace) / "decide"


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``root`` with its bytes and its mtime — a tree, pinned."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _line(output: str, prefix: str) -> str:
    """The one printed line starting with ``prefix``, without it."""
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"no line starting with {prefix!r} in:\n{output}")


def _only_bench(tmp_path: Path) -> Path:
    """The single bench directory the run under test wrote."""
    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    return directory


# ── the dry run: a plan, and nothing else ────────────────────────────────────────────────────


def test_a_dry_run_prints_the_spend_plan_for_the_corpus_it_would_measure(tmp_path):
    _decide_corpus(tmp_path)

    result = runner.invoke(app, ["bench", "run", "--site", "decide", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert _line(result.output, "plan:").startswith("decide/all: 2 case(s) × 1 rep(s)")
    assert "2 job(s)" in _line(result.output, "plan:")


def test_a_dry_run_names_the_corpus_directory_and_the_digest_it_planned_over(tmp_path):
    directory = _decide_corpus(tmp_path)

    result = runner.invoke(app, ["bench", "run", "--site", "decide", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert str(directory) in _line(result.output, "corpus:")


def test_a_dry_run_writes_nothing_at_all_into_the_workspace(tmp_path):
    _decide_corpus(tmp_path)
    workspace = tmp_path / "workspace"
    before = _snapshot(workspace)

    result = runner.invoke(app, ["bench", "run", "--site", "decide", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert _snapshot(workspace) == before
    assert not bench_root(workspace).exists()


def test_a_dry_run_spends_nothing_because_no_case_is_ever_asked(tmp_path, capsys):
    _stub_corpus(tmp_path)
    asked = StubAttempt()

    run_bench("stub", dry_run=True, seams=_seams(asked))

    assert asked.requests == []
    assert "dry run" in capsys.readouterr().out


def test_a_dry_run_needs_no_model_client_configured_at_all(tmp_path):
    """No key, no ``[llm]`` extra, no client: a plan is arithmetic over a corpus."""
    _decide_corpus(tmp_path)

    result = runner.invoke(
        app, ["bench", "run", "--site", "decide", "--dry-run", "--model", "anthropic/claude"]
    )

    assert result.exit_code == 0, result.output


# ── the flags reach the runner ───────────────────────────────────────────────────────────────


def test_the_holdout_split_plans_only_the_holdout_half_of_the_corpus(tmp_path, capsys):
    _stub_corpus(tmp_path)

    run_bench("stub", split="holdout", dry_run=True, seams=_seams())

    assert _line(capsys.readouterr().out, "plan:").startswith("stub/holdout: 1 case(s)")


def test_the_tuning_split_plans_only_the_tuning_half_of_the_corpus(tmp_path, capsys):
    _stub_corpus(tmp_path)

    run_bench("stub", split="tuning", dry_run=True, seams=_seams())

    assert _line(capsys.readouterr().out, "plan:").startswith("stub/tuning: 3 case(s)")


def test_a_split_word_the_corpus_has_no_half_for_is_refused_naming_the_three_it_accepts(tmp_path):
    _decide_corpus(tmp_path)

    result = runner.invoke(
        app, ["bench", "run", "--site", "decide", "--split", "everything", "--dry-run"]
    )

    assert result.exit_code == 1
    assert "everything" in result.output
    assert "tuning" in result.output and "holdout" in result.output and "all" in result.output


def test_the_reps_flag_reaches_the_runner_so_every_case_is_asked_once_per_rep(tmp_path):
    _stub_corpus(tmp_path)
    asked = StubAttempt()

    run_bench("stub", reps=2, seams=_seams(asked))

    assert len(asked.requests) == 8
    assert sorted({request.rep for request in asked.requests}) == [1, 2]


def test_the_reps_flag_is_carried_into_the_record_as_one_run_per_case_per_rep(tmp_path):
    _stub_corpus(tmp_path)

    run_bench("stub", reps=2, seams=_seams())

    record = json.loads((_only_bench(tmp_path) / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["bench"]["cases"] == 4
    assert record["bench"]["runs"] == 8


def test_a_rep_count_below_one_is_refused_rather_than_quietly_clamped(tmp_path):
    _decide_corpus(tmp_path)

    result = runner.invoke(app, ["bench", "run", "--site", "decide", "--reps", "0", "--dry-run"])

    assert result.exit_code == 1
    assert "reps" in result.output


def test_the_workers_flag_states_the_width_the_jobs_would_be_worked_at(tmp_path, capsys):
    _stub_corpus(tmp_path)

    run_bench("stub", workers=3, dry_run=True, seams=_seams())

    assert _line(capsys.readouterr().out, "workers:") == "3"


def test_more_than_one_worker_builds_a_pool_that_declares_the_width_it_was_given():
    executor = build_executor(4)

    assert isinstance(executor, PooledExecutor)
    assert executor.workers == 4


def test_one_worker_builds_the_sequential_executor_that_cannot_get_ordering_wrong():
    assert build_executor(1) == SequentialExecutor()


# ── the live run ─────────────────────────────────────────────────────────────────────────────


def test_a_stubbed_run_writes_one_valid_bench_record_under_the_bench_area(tmp_path):
    _stub_corpus(tmp_path)

    run_bench("stub", seams=_seams())

    record = json.loads((_only_bench(tmp_path) / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert validate(record) == []
    assert record["bench"]["cases"] == 4


def test_a_stubbed_run_prints_the_bench_id_and_the_directory_it_landed_in(tmp_path, capsys):
    _stub_corpus(tmp_path)

    run_bench("stub", seams=_seams())

    directory = _only_bench(tmp_path)
    printed = _line(capsys.readouterr().out, "bench:")
    assert directory.name in printed
    assert str(directory) in printed


def test_the_run_executes_exactly_the_plan_its_preflight_printed(tmp_path, capsys):
    """The CLI has no path that skips the preflight, so an unacknowledged spend is unreachable.

    A live run re-preflights inside :meth:`~noctis.eval.runner.BenchRunner.run` and refuses unless
    the plan it computes equals the acknowledged one — so a run that completed *is* the proof that
    the plan printed here is the plan that executed. The two summaries are compared to say so.
    """
    _stub_corpus(tmp_path)

    run_bench("stub", reps=2, dry_run=True, seams=_seams())
    planned = _line(capsys.readouterr().out, "plan:")
    run_bench("stub", reps=2, seams=_seams())

    assert _line(capsys.readouterr().out, "plan:") == planned


def test_a_second_site_is_asked_the_payload_its_cases_declare_with_no_site_specific_wiring(
    tmp_path,
):
    _stub_corpus(tmp_path)

    run_bench("stub", seams=_seams())

    prompts = sorted(
        path.read_text(encoding="utf-8") for path in _only_bench(tmp_path).rglob("prompt.txt")
    )
    assert prompts == [f"ask=question-{name}" for name in ("a", "b", "c", "d")]


def test_the_decide_site_is_asked_through_its_own_input_adapter_and_not_the_raw_payload(tmp_path):
    """The lookup, from the other side: decide's cases render a real briefing, not their payload."""
    _decide_corpus(tmp_path)
    asked = StubAttempt()

    run_bench("decide", seams=BenchSeams(attempt=asked))

    prompts = [request.prompt for request in asked.requests]
    assert all("DECIDE TASK" in prompt for prompt in prompts)
    assert any("EVIDENCE FOR gap_fade_open (gate-facing)" in prompt for prompt in prompts)


def test_the_record_a_stubbed_run_wrote_reports_through_the_bench_report_verb(tmp_path):
    _stub_corpus(tmp_path)
    run_bench("stub", seams=_seams())

    result = runner.invoke(app, ["bench", "report", _only_bench(tmp_path).name])

    assert result.exit_code == 0, result.output
    assert "stub" in result.output


def test_a_label_is_carried_onto_the_record_the_run_wrote(tmp_path):
    _stub_corpus(tmp_path)

    run_bench("stub", label="sonnet-vs-haiku", seams=_seams())

    record = json.loads((_only_bench(tmp_path) / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["bench"]["label"] == "sonnet-vs-haiku"


def test_the_requested_model_is_recorded_as_the_configuration_the_cases_ran_under(tmp_path):
    _stub_corpus(tmp_path)

    run_bench("stub", model="bench/oracle", seams=_seams())

    record = json.loads((_only_bench(tmp_path) / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    configs = record["provenance"]["configs"]
    assert [config["requested_model"] for config in configs] == ["bench/oracle"]


# ── the refusals ─────────────────────────────────────────────────────────────────────────────


def test_an_undeclared_site_is_refused_naming_every_site_that_is_declared(tmp_path):
    result = runner.invoke(app, ["bench", "run", "--site", "oracle", "--dry-run"])

    assert result.exit_code == 1
    assert "oracle" in result.output
    for declared in ("coder", "decide", "discover", "distill", "formulate"):
        assert declared in result.output


def test_a_site_with_no_corpus_in_the_workspace_is_refused_naming_where_it_looked(tmp_path):
    result = runner.invoke(app, ["bench", "run", "--site", "decide", "--dry-run"])

    assert result.exit_code == 1
    assert str(cases_root(tmp_path / "workspace") / "decide") in result.output


def test_a_live_run_with_no_model_client_available_refuses_naming_the_model(tmp_path):
    _decide_corpus(tmp_path)

    result = runner.invoke(app, ["bench", "run", "--site", "decide", "--model", "anthropic/claude"])

    assert result.exit_code == 1
    assert "anthropic/claude" in result.output
    assert not bench_root(tmp_path / "workspace").exists()


# ── the live model seam ──────────────────────────────────────────────────────────────────────


class FakeClient:
    """One completion, scripted: the emit tool call a well-behaved backend answers with."""

    capabilities = Capabilities()

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> Turn:
        self.calls.append(kwargs)
        return Turn(
            text="",
            tool_calls=[ToolCall(id="1", name="emit_stub", arguments=self._payload)],
            stop_reason="tool_use",
            usage={"input_tokens": 120, "output_tokens": 30},
            served_model="bench/oracle-2026",
        )


def _request(case: Case) -> AttemptRequest:
    """One attempt request of the shape the runner hands the attempt callable."""
    return AttemptRequest(
        prompt="ask=question-a",
        case=case,
        config=ModelConfig(config_id="default"),
        harness=HarnessSpec(),
        rep=1,
        attempt=1,
        workdir=Path("/nonexistent"),
    )


STUB_ASKS: Mapping[str, SiteAsk] = {
    "stub": SiteAsk(system="You are a stub.", contract=lambda declaration, case: STUB_CONTRACT)
}


def _stub_case(tmp_path: Path) -> Case:
    """One loaded stub case, straight off the corpus the provider reads."""
    _stub_corpus(tmp_path, ids=("a",))
    (case,) = YamlCaseProvider(
        cases_root=cases_root(tmp_path / "workspace"), registry=STUB_REGISTRY
    ).load("stub")
    return case


def test_the_live_attempt_answers_through_the_sites_emit_contract_and_records_what_it_spent(
    tmp_path,
):
    client = FakeClient({"answer": "forty-two"})
    attempt = live_attempt(
        load_settings(),
        model="bench/oracle",
        client=client,
        registry=STUB_REGISTRY,
        asks=STUB_ASKS,
    )

    answer = attempt(_request(_stub_case(tmp_path)))

    assert answer.outcome.passed is True
    assert answer.outcome.input_tokens == 120
    assert answer.served_model == "bench/oracle-2026"
    assert "forty-two" in (answer.output or "")


def test_the_live_attempt_asks_the_site_with_the_prompt_the_runner_rendered(tmp_path):
    client = FakeClient({"answer": "forty-two"})
    attempt = live_attempt(
        load_settings(),
        model="bench/oracle",
        client=client,
        registry=STUB_REGISTRY,
        asks=STUB_ASKS,
    )

    attempt(_request(_stub_case(tmp_path)))

    assert client.calls[0]["messages"] == [{"role": "user", "content": "ask=question-a"}]
    assert client.calls[0]["system"] == "You are a stub."


def test_the_live_attempt_refuses_a_site_that_declares_no_live_ask_rather_than_inventing_one(
    tmp_path,
):
    attempt = live_attempt(
        load_settings(), model="bench/oracle", client=FakeClient({}), registry=STUB_REGISTRY
    )

    with pytest.raises(LiveModelUnavailable) as refusal:
        attempt(_request(_stub_case(tmp_path)))

    assert "stub" in str(refusal.value)


def test_building_a_live_attempt_refuses_when_no_client_can_be_built_naming_the_model():
    with pytest.raises(LiveModelUnavailable) as refusal:
        live_attempt(load_settings(), model="anthropic/claude-sonnet-4")

    assert "anthropic/claude-sonnet-4" in str(refusal.value)
