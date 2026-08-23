"""The retrospective DECIDE miner (#208): a mined corpus, and the zero-spend bench record.

Two behaviours, one story. **Mining** walks a workspace's runs tree as *data* — each run's journal,
its session ledgers and the configuration it froze — and writes one YAML case per decided candidate
into the per-site corpus layout the eval core's provider already reads. **The retrospective record**
scores those cases' *recorded* verdicts and publishes a valid ``bench.json`` without asking a model
anything at all: the answers were spent months ago, so the only honest bench over them is one that
costs nothing.

Everything asserted here is external behaviour — the files that appear, the counts reported, the
document written, what a second mining does — and every fixture is a real run tree built with the
engine's own writers (``ExperimentJournal``, ``SessionLedger``, ``freeze_inputs``), so the miner is
held to the shape a run really leaves behind rather than to a hand-drawn imitation of it.

The two invariants worth naming: the miner **never writes into the runs tree** (asserted by a
byte-and-mtime snapshot taken around it), and the whole path **spends nothing** (asserted by running
it in a fresh interpreter and inspecting what got imported, the technique
``tests/test_run_tree_store.py`` uses on the run tree).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from noctis.config.rehydrate import freeze_inputs
from noctis.config.settings import Settings
from noctis.eval.case import Split
from noctis.eval.case_provider import YamlCaseProvider
from noctis.eval.corpus import Corpus
from noctis.eval.decide_case import (
    AT_FLOOR,
    BINDING_GATE_AXIS,
    COMFORTABLE,
    EVIDENCE_DEPTH_AXIS,
    MARGIN_AXIS,
    NOT_APPLICABLE,
    SITE_ID,
    decide_case_id,
)
from noctis.eval.decide_miner import (
    load_decide_corpus,
    mine_decide_corpus,
    write_retrospective_bench,
)
from noctis.eval.record import side_by_side, validate
from noctis.eval.runner import BENCH_RECORD_NAME
from noctis.research.journal import ExperimentJournal
from noctis.research.ledger import SessionLedger

from ._promotion_cases import card

MODULE_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/eval/decide_miner.py"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Two run ids of exactly the shape the engine mints, so mined provenance faces the real pattern.
RUN_A = "20260720T144233Z-a3f9c1"
RUN_B = "20260721T090000Z-b7d2e4"

STRATEGY = "vol_breakout_squeeze"
OTHER = "gap_fade_open"

# The exhaustion floor the fixture runs froze, and a trial count that sits exactly on it.
MIN_TRIALS = 6

RESEARCH_MODEL = "claude-sonnet-4-5"


# ── fixtures: a real run tree, written by the engine's own writers ───────────────────────────


def _record(run_id: str, *, min_trials: int, model: str | None) -> dict[str, Any]:
    """One run.json carrying the configuration a run really freezes at creation."""
    settings = Settings()
    settings.research.min_trials = min_trials
    settings.research.model = model
    return {
        "run": {"run_id": run_id},
        "inputs": freeze_inputs(settings, frozen_at="2026-07-20T14:00:00Z", execution_mode="paper"),
    }


def _run(
    workspace: Path,
    run_id: str = RUN_A,
    *,
    min_trials: int = MIN_TRIALS,
    model: str | None = RESEARCH_MODEL,
    record: dict[str, Any] | None = None,
) -> Path:
    """A run directory with its frozen record and an empty state tree."""
    run_dir = workspace / "runs" / run_id
    (run_dir / "state").mkdir(parents=True)
    document = _record(run_id, min_trials=min_trials, model=model) if record is None else record
    (run_dir / "run.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    return run_dir


def _journal(run_dir: Path) -> ExperimentJournal:
    return ExperimentJournal(run_dir / "state")


def _ledger(run_dir: Path, session: str = "session-20260720T144233") -> SessionLedger:
    return SessionLedger(run_dir / "state", session)


def _sweep(journal: ExperimentJournal, name: str = STRATEGY, *, trials: int = MIN_TRIALS) -> None:
    """A candidate's tuning history: its class, its thesis, and ``trials`` journaled trials."""
    journal.record_class_tag(name, "volatility_squeeze")
    journal.record_thesis(name, "volatility compresses before it expands")
    for index in range(trials):
        journal.record_trial(
            name,
            source="sweep",
            symbols=["AAPL", "MSFT"],
            params={"lookback": 10 + index},
            window={"start": "2024-01-02", "end": "2025-01-02"},
            card=card(test=1.0 - index * 0.01, train=1.2),
        )
    journal.record_sweep_complete(name, n_trials=trials, symbols=["AAPL", "MSFT"])


def _approve(journal: ExperimentJournal, name: str = STRATEGY, *, promoted: bool) -> None:
    journal.record_approval(
        name,
        promoted=promoted,
        rationale="the panel holds up out of sample",
        params={"lookback": 10},
        symbols=["AAPL", "MSFT"],
        holdout_symbols=["NVDA"],
    )


def _reject(journal: ExperimentJournal, name: str = STRATEGY) -> None:
    journal.record_rejection(name, reason="the class is a dead end", best_params={"lookback": 10})


def _refused_candidate(run_dir: Path, name: str = STRATEGY) -> None:
    """A post-capture episode: journaled scorecard, an approval the gates then refused."""
    journal = _journal(run_dir)
    _sweep(journal, name)
    journal.record_scorecard(name, card(test=1.0, train=1.2, symbol_holdout=-0.5))
    _approve(journal, name, promoted=False)


def _promoted_candidate(run_dir: Path, name: str = STRATEGY) -> None:
    journal = _journal(run_dir)
    _sweep(journal, name)
    journal.record_scorecard(name, card(test=1.0, train=1.2, holdout=2.0, symbol_holdout=2.0))
    _approve(journal, name, promoted=True)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "runs").mkdir(parents=True)
    return workspace


def _mine(workspace: Path):
    return mine_decide_corpus(workspace / "runs", cases_root=workspace / "cases")


def _case_files(workspace: Path) -> list[str]:
    directory = workspace / "cases" / SITE_ID
    return sorted(path.name for path in directory.glob("*.yaml")) if directory.is_dir() else []


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``root`` with its bytes and its mtime — a tree, pinned."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _age(path: Path, seconds: int = 3600) -> int:
    """Push a file's mtime into the past and return it, so "untouched" is checkable."""
    stamp = path.stat().st_mtime_ns - seconds * 1_000_000_000
    os.utime(path, ns=(stamp, stamp))
    return path.stat().st_mtime_ns


# ── mining: the files that appear ────────────────────────────────────────────────────────────


def test_mining_a_workspace_writes_one_case_file_per_decided_candidate(tmp_path):
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    _refused_candidate(run_dir)
    journal = _journal(run_dir)
    _sweep(journal, OTHER)
    _reject(journal, OTHER)

    report = _mine(workspace)

    assert _case_files(workspace) == [
        f"{decide_case_id(OTHER, RUN_A)}.yaml",
        f"{decide_case_id(STRATEGY, RUN_A)}.yaml",
    ]
    assert len(report.mined) == 2
    assert report.unchanged == ()


def test_two_runs_in_one_workspace_are_mined_into_two_separately_provenanced_cases(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, RUN_A))
    _promoted_candidate(_run(workspace, RUN_B))

    report = _mine(workspace)

    assert {str(case.provenance) for case in report.cases} == {
        f"mined:{RUN_A}",
        f"mined:{RUN_B}",
    }


def test_a_candidate_that_never_reached_a_verdict_is_excluded_and_counted(tmp_path):
    """Undecided: there is no outcome to reconstruct, so nothing is written and the gap is named."""
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    _sweep(_journal(run_dir), OTHER)
    _refused_candidate(run_dir)

    report = _mine(workspace)

    assert report.excluded == (decide_case_id(OTHER, RUN_A),)
    assert _case_files(workspace) == [f"{decide_case_id(STRATEGY, RUN_A)}.yaml"]


def test_the_mined_ask_carries_the_exhaustion_floor_the_run_actually_froze(tmp_path):
    """The floor is read off the run's own frozen configuration, never assumed."""
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, min_trials=11))

    (case,) = _mine(workspace).cases

    assert case.payload["evidence"]["min_trials_gate"] == 11


# ── mining: determinism and idempotence ──────────────────────────────────────────────────────


def test_re_mining_the_same_workspace_writes_byte_identical_files(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace))
    path = workspace / "cases" / SITE_ID / f"{decide_case_id(STRATEGY, RUN_A)}.yaml"
    _mine(workspace)
    first = path.read_bytes()
    path.unlink()

    _mine(workspace)

    assert path.read_bytes() == first


def test_a_case_already_on_disk_is_reported_unchanged_and_left_untouched(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace))
    _mine(workspace)
    path = workspace / "cases" / SITE_ID / f"{decide_case_id(STRATEGY, RUN_A)}.yaml"
    aged = _age(path)

    report = _mine(workspace)

    assert report.unchanged == (decide_case_id(STRATEGY, RUN_A),)
    assert report.mined == ()
    assert path.stat().st_mtime_ns == aged


def test_re_mining_never_duplicates_a_case_the_corpus_already_holds(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace))

    _mine(workspace)
    _mine(workspace)
    _mine(workspace)

    assert _case_files(workspace) == [f"{decide_case_id(STRATEGY, RUN_A)}.yaml"]


# ── mining: the honesty tiers ────────────────────────────────────────────────────────────────


def test_a_post_capture_record_yields_a_case_labelled_on_every_difficulty_axis(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace))

    (case,) = _mine(workspace).cases

    assert dict(case.difficulty) == {
        MARGIN_AXIS: COMFORTABLE,
        BINDING_GATE_AXIS: "symbol_holdout",
        EVIDENCE_DEPTH_AXIS: AT_FLOOR,
    }


def test_an_older_record_without_a_journaled_scorecard_degrades_its_gate_axes(tmp_path):
    """Nothing to replay the arbitration over: the gate axes say ``n/a`` rather than a guess."""
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    journal = _journal(run_dir)
    _sweep(journal)
    _approve(journal, promoted=False)

    (case,) = _mine(workspace).cases

    assert case.difficulty[BINDING_GATE_AXIS] == NOT_APPLICABLE
    assert case.difficulty[MARGIN_AXIS] == NOT_APPLICABLE
    assert case.difficulty[EVIDENCE_DEPTH_AXIS] == AT_FLOOR


def test_a_run_that_froze_no_configuration_is_skipped_rather_than_mined_on_guessed_rules(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, record={"run": {"run_id": RUN_A}}))

    report = _mine(workspace)

    assert _case_files(workspace) == []
    assert [skipped.run_id for skipped in report.skipped_runs] == [RUN_A]


def test_a_run_directory_whose_name_is_not_a_minted_run_id_is_skipped(tmp_path):
    """The reserved ``legacy`` tree is a run nobody minted, so no case can be provenanced to it."""
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, "legacy"))

    report = _mine(workspace)

    assert _case_files(workspace) == []
    assert [skipped.run_id for skipped in report.skipped_runs] == ["legacy"]


def test_the_session_ledger_tail_of_the_deciding_session_is_frozen_onto_the_case(tmp_path):
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    ledger = _ledger(run_dir)
    ledger.record_thesis("earlier_idea", "gaps fade by lunch")
    ledger.record_verdict("earlier_idea", verdict="reject", lesson="gap fades are a dead end")
    ledger.record_stage("decide", strategy=STRATEGY)
    _refused_candidate(run_dir)

    (case,) = _mine(workspace).cases

    assert case.payload["ledger_tail"] == (
        {
            "strategy": "earlier_idea",
            "thesis": "gaps fade by lunch",
            "verdict": "reject",
            "lesson": "gap fades are a dead end",
        },
    )


def test_a_mined_case_freezes_nothing_the_verdict_it_is_graded_against_left_behind(tmp_path):
    """The whole trail of one spent verdict — its journal record and the ledger line an instant
    later — post-dates the ask it answered, so re-mining a real run tree yields a case whose ask
    side carries neither (#212)."""
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    ledger = _ledger(run_dir)
    ledger.record_stage("decide", strategy=STRATEGY)
    ledger.record_thesis(STRATEGY, "volatility compresses before it expands")
    _refused_candidate(run_dir)
    ledger.record_verdict(
        STRATEGY, verdict="approve", lesson="the holdout gave way", promoted=False
    )

    (case,) = _mine(workspace).cases

    assert case.payload["evidence"]["verdicts"] == ()
    assert case.payload["ledger_tail"] == (
        {"strategy": STRATEGY, "thesis": "volatility compresses before it expands"},
    )


# ── mining: the boundary ─────────────────────────────────────────────────────────────────────


def test_the_miner_writes_nothing_into_the_runs_tree(tmp_path):
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    _refused_candidate(run_dir)
    _ledger(run_dir).record_stage("decide", strategy=STRATEGY)
    before = _snapshot(workspace / "runs")

    _mine(workspace)

    assert _snapshot(workspace / "runs") == before


# ── the corpus: the mined files load back through the eval core ──────────────────────────────


def test_mined_cases_load_back_through_the_yaml_provider_unchanged(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, RUN_A))
    _promoted_candidate(_run(workspace, RUN_B))

    report = _mine(workspace)

    loaded = YamlCaseProvider(cases_root=workspace / "cases").load(SITE_ID)
    assert loaded == tuple(sorted(report.cases, key=lambda case: case.case_id))


def test_a_mined_corpus_freezes_a_split_over_every_case_it_holds(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, RUN_A))
    _promoted_candidate(_run(workspace, RUN_B))
    _mine(workspace)

    corpus = load_decide_corpus(workspace / "cases")

    assert isinstance(corpus, Corpus)
    assert {case.split for case in corpus.cases} <= {Split.TUNING, Split.HOLDOUT}
    assert len(corpus.tuning) + len(corpus.holdout) == len(corpus)


def test_two_loads_of_one_mined_corpus_report_the_same_digest(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, RUN_A))
    _promoted_candidate(_run(workspace, RUN_B))
    _mine(workspace)

    assert (
        load_decide_corpus(workspace / "cases").digest
        == load_decide_corpus(workspace / "cases").digest
    )


# ── the zero-spend retrospective record ──────────────────────────────────────────────────────


def _bench(tmp_path, *, promoted: bool = True):
    """A mined two-case corpus and the retrospective record built over it."""
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, RUN_A))
    if promoted:
        _promoted_candidate(_run(workspace, RUN_B))
    report = _mine(workspace)
    corpus = load_decide_corpus(workspace / "cases")
    bench = write_retrospective_bench(
        corpus,
        bench_root=workspace / "bench",
        requested_models=report.requested_models,
    )
    return workspace, bench


def test_the_retrospective_record_validates_clean_against_the_bench_schema(tmp_path):
    _, bench = _bench(tmp_path)

    assert validate(bench.record) == []


def test_the_retrospective_record_is_written_into_the_bench_area(tmp_path):
    workspace, bench = _bench(tmp_path)

    written = json.loads((bench.directory / BENCH_RECORD_NAME).read_text(encoding="utf-8"))
    assert bench.directory.parent == workspace / "bench"
    assert written == json.loads(json.dumps(bench.record))


def test_building_the_retrospective_record_writes_nothing_into_the_runs_tree(tmp_path):
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace, RUN_A))
    _mine(workspace)
    before = _snapshot(workspace / "runs")

    write_retrospective_bench(
        load_decide_corpus(workspace / "cases"), bench_root=workspace / "bench"
    )

    assert _snapshot(workspace / "runs") == before


def test_the_retrospective_record_carries_the_co_primary_approval_pair(tmp_path):
    """One promoted approval and one refused: agreement 0.5, bought at an approval rate of 1.0."""
    _, bench = _bench(tmp_path)

    approval = bench.record["harness"]["dials"]["decide"]["approval"]
    assert approval["agreement"] == 0.5
    assert approval["approval_rate"] == 1.0
    assert approval["labeled_approvals"] == 2


def test_the_retrospective_record_carries_per_axis_strata_inputs(tmp_path):
    _, bench = _bench(tmp_path)

    strata = bench.record["harness"]["dials"]["decide"]["strata"]
    assert set(strata) == {MARGIN_AXIS, BINDING_GATE_AXIS, EVIDENCE_DEPTH_AXIS}
    assert strata[BINDING_GATE_AXIS]["symbol_holdout"]["approval"]["agreement"] == 0.0
    assert strata[BINDING_GATE_AXIS]["none"]["approval"]["agreement"] == 1.0


def test_every_mined_case_is_listed_with_the_axes_its_stratum_was_read_from(tmp_path):
    _, bench = _bench(tmp_path)

    listed = {row["case_id"]: row for row in bench.record["harness"]["dials"]["decide"]["cases"]}
    assert set(listed) == {decide_case_id(STRATEGY, RUN_A), decide_case_id(STRATEGY, RUN_B)}
    assert listed[decide_case_id(STRATEGY, RUN_A)]["label"] == "refused"
    assert listed[decide_case_id(STRATEGY, RUN_A)]["difficulty"][BINDING_GATE_AXIS] == (
        "symbol_holdout"
    )


def test_the_headline_pass_rate_is_the_recorded_approval_side_agreement(tmp_path):
    """The scorer's number is the source of truth; the record's headline is derived from it."""
    _, bench = _bench(tmp_path)

    assert bench.record["metrics"]["job_pass_rate"] == bench.metrics.approval.agreement
    assert bench.record["metrics"]["first_attempt_pass_rate"] == bench.metrics.approval.agreement


def test_the_record_grades_only_the_labelled_approvals_and_says_how_many_it_left(tmp_path):
    """A rejection has no gate counterfactual, so it is corpus but never a graded result."""
    workspace = _workspace(tmp_path)
    run_dir = _run(workspace)
    _refused_candidate(run_dir)
    journal = _journal(run_dir)
    _sweep(journal, OTHER)
    _reject(journal, OTHER)
    _mine(workspace)

    bench = write_retrospective_bench(
        load_decide_corpus(workspace / "cases"), bench_root=workspace / "bench"
    )

    assert bench.record["bench"]["cases"] == 1
    assert bench.record["corpus"]["case_count"] == 2
    assert bench.record["harness"]["dials"]["decide"]["rejections"] == 1


def test_the_retrospective_record_spends_zero_tokens(tmp_path):
    """No model was asked, so the spend is a measured zero rather than an unknown."""
    _, bench = _bench(tmp_path)

    assert bench.record["metrics"]["cost"]["tokens_total"] == 0
    assert bench.record["harness"]["dials"]["attempt_calls"] == 0
    assert all(
        attempt["model"] is None
        for result in bench.record["results"]
        for attempt in result["attempts"]
    )


def test_the_retrospective_record_names_the_historical_run_as_its_configuration(tmp_path):
    _, bench = _bench(tmp_path)

    configs = {config["config_id"]: config for config in bench.record["provenance"]["configs"]}
    assert set(configs) == {RUN_A, RUN_B}
    assert configs[RUN_A]["requested_model"] == RESEARCH_MODEL
    assert configs[RUN_A]["served_models"] == []


def test_an_older_record_that_names_no_research_model_is_configured_with_none(tmp_path):
    """An alias is quoted where the record carries one and left absent where it does not."""
    workspace = _workspace(tmp_path)
    older = _record(RUN_A, min_trials=MIN_TRIALS, model=RESEARCH_MODEL)
    del older["inputs"]["models"]
    _refused_candidate(_run(workspace, record=older))
    report = _mine(workspace)

    bench = write_retrospective_bench(
        load_decide_corpus(workspace / "cases"),
        bench_root=workspace / "bench",
        requested_models=report.requested_models,
    )

    (config,) = bench.record["provenance"]["configs"]
    assert config["requested_model"] is None


def test_a_retrospective_record_declares_no_harness_and_is_never_subtractable(tmp_path):
    """The compositions behind its answers are as many as the runs it was mined from."""
    _, bench = _bench(tmp_path)

    assert bench.record["harness"]["hash"] is None
    comparison = side_by_side(bench.record, bench.record)
    assert comparison.comparable is False
    assert "harness_hash" in (comparison.banner or "")


# ── nothing on this path can spend a token ───────────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_the_miner_imports_no_model_client_of_its_own():
    """A retrospective bench asks nobody anything: the client seams are not on this module."""
    forbidden = {"noctis.research.llm", "noctis.research.agent", "anthropic", "litellm", "openai"}

    assert _imports(MODULE_SOURCE) & forbidden == set()


def test_mining_and_scoring_a_workspace_loads_no_llm_client_package(tmp_path):
    """The whole path, in a fresh interpreter: no vendor SDK is ever imported."""
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace))
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from noctis.eval.decide_miner import (\n"
        "    load_decide_corpus, mine_decide_corpus, write_retrospective_bench,\n"
        ")\n"
        f"workspace = Path({str(workspace)!r})\n"
        "mine_decide_corpus(workspace / 'runs', cases_root=workspace / 'cases')\n"
        "corpus = load_decide_corpus(workspace / 'cases')\n"
        "write_retrospective_bench(corpus, bench_root=workspace / 'bench')\n"
        "vendors = {'anthropic', 'litellm', 'openai', 'httpx'}\n"
        "loaded = vendors & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_a_mined_case_file_is_readable_yaml_a_curator_can_review(tmp_path):
    """The interchange format is the point: one small file per case, in the corpus layout."""
    workspace = _workspace(tmp_path)
    _refused_candidate(_run(workspace))
    _mine(workspace)

    path = workspace / "cases" / SITE_ID / f"{decide_case_id(STRATEGY, RUN_A)}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["site_id"] == SITE_ID
    assert document["provenance"] == f"mined:{RUN_A}"
