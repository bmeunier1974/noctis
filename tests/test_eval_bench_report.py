"""``bench report`` (#211): the headline pair, the per-axis breakdown, and a refusal-first render.

Two surfaces, one story. The **renderer** is pure — a bench record in, text out — so almost every
assertion here is a fabricated record rendered and read back row by row, with no directory and no
clock in sight. The **verb** is the thin body around it: it resolves ``bench/<bench_id>/bench.json``
under the workspace, validates it, prints the render, and refuses (naming the bench root) when it
cannot.

The load-bearing property being asserted throughout is that **the verb knows nothing about any
site**. It renders the reading the record carries: a dials block that publishes a whole co-primary
pair gets that pair through the pair's own labels (never one half of it — since #304 the
whole-or-refused rule of #206 is read off
:data:`~noctis.eval.reading.PAIR_MANIFESTS` and governs the coder's pass rates exactly as it
governs DECIDE's agreement), and a record from another site with entirely different axes renders
through exactly the same code path. The record's own ``metrics`` block is *not* a site reading —
its ``job_pass_rate`` is the eval core's figure over attempt outcomes, and it renders as the core's
own rows however a site spells its keys. The one integration test drives #208's zero-spend
retrospective writer and reports the record it produced, so the mined path is held to the same
reader as any other.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from noctis.cli import app
from noctis.config.rehydrate import freeze_inputs
from noctis.config.settings import Settings
from noctis.eval.bench_report import render_bench_report
from noctis.eval.decide_case import BINDING_GATE_AXIS, EVIDENCE_DEPTH_AXIS, MARGIN_AXIS
from noctis.eval.decide_miner import (
    load_decide_corpus,
    mine_decide_corpus,
    write_retrospective_bench,
)
from noctis.eval.identity import SiteIdentity
from noctis.eval.metrics import AttemptOutcome, CaseResult
from noctis.eval.record import BenchArtifacts, CaseRun, CorpusIdentity, EngineStamp, build
from noctis.eval.runner import BENCH_RECORD_NAME, bench_root
from noctis.research.journal import ExperimentJournal

from ._promotion_cases import card

runner = CliRunner()

BENCH_ID = "20260801T101112Z-a1b2c3"

IDENTITY = SiteIdentity(
    site_id="decide",
    version="1",
    prompt_asset_groups=("decide",),
    prompt_asset_hash="c0ffee1234567890",
)
ENGINE = EngineStamp(
    engine_version=1,
    fingerprint={"gates": "f63d47b7b9604ab1", "backtest": "3ba3e0bf1c97134f"},
    noctis_version="0.1.0",
)

# The co-primary pair, as a record's dials carry it: agreement beside the approval rate it cost,
# and the excluded count named rather than implied.
PAIR: dict[str, Any] = {
    "agreement": 0.5,
    "approval_rate": 0.75,
    "decided": 4,
    "approvals": 3,
    "labeled_approvals": 2,
    "unlabeled_approvals": 1,
    "promoted": 1,
}


def _stratum(agreement: float | None, approval_rate: float | None) -> dict[str, Any]:
    """One scored stratum, in the same shape the whole batch is scored into."""
    return {
        "approval": {**PAIR, "agreement": agreement, "approval_rate": approval_rate},
        "revise_rate": 0.0,
        "revise_flip_rate": None,
    }


DECIDE_DIALS: dict[str, Any] = {
    "retrospective": True,
    "answers": "recorded",
    "attempt_calls": 0,
    "decide": {
        "approval": PAIR,
        "revise_rate": 0.25,
        "revise_flip_rate": 1.0,
        "revises": 1,
        "revise_flips": 1,
        "rejections": 1,
        "cases": [{"case_id": "one"}, {"case_id": "two"}],
        "strata": {
            "margin_distance": {
                "comfortable": _stratum(1.0, 0.5),
                "near": _stratum(0.0, 1.0),
            },
            "evidence_depth": {"at_floor": _stratum(0.5, 0.75)},
        },
    },
}


#: The two rows the coder's pass pair prints, spelled as an operator reads them off the report —
#: the qualification rides *inside* the job rate's label, so the words cannot be separated from the
#: number (:data:`~noctis.eval.reading.PASS_RATES`).
FIRST_ATTEMPT_ROW = "First-attempt pass rate"
JOB_PASS_ROW = "Job pass rate (pass@k with feedback)"


def _rates(first: float | None, job: float | None) -> dict[str, Any]:
    """One co-primary pass pair, in the shape the coder reading publishes it."""
    return {
        "pass_label": "pass@k with feedback",
        "first_attempt_pass_rate": first,
        "job_pass_rate": job,
        "first_attempt_passes": 1,
        "passed_jobs": 2,
    }


# The shapes a coder record carries: a pass pair, a per-axis breakdown of that pair, and the
# taxonomy table whose every row names the knob its share points at.
CODER_READING: dict[str, Any] = {
    "cases": 3,
    "jobs": 3,
    "rates": _rates(0.3333, 0.6667),
    "strata": {
        "api_surface": {
            "bars_only": {"cases": 1, "jobs": 1, "rates": _rates(1.0, 1.0)},
            "exits": {"cases": 2, "jobs": 2, "rates": _rates(0.0, 0.5)},
        },
        "oracle_mode": {"authored": {"cases": 3, "jobs": 3, "rates": _rates(0.3333, 0.6667)}},
    },
    "failures": {
        "attempts": 4,
        "classes": {
            "truncated": {"count": 3, "share": 0.75, "knob": "coder max-tokens"},
            "unclassified": {"count": 0, "share": 0.0, "knob": "grow the taxonomy"},
        },
    },
}

CODER_DIALS: dict[str, Any] = {"answers": "fresh", "attempt_calls": 3, "coder": CODER_READING}

# A record from a site that has never heard of an approval pair: different figures, different axes.
STUB_DIALS: dict[str, Any] = {
    "stubby": {
        "usefulness": 0.5,
        "refusal_rate": None,
        "strata": {
            "prompt_length": {
                "long": {"usefulness": 0.25},
                "short": {"usefulness": None},
            }
        },
    }
}


# ── fixtures: real records, built by the record module's own pure builder ────────────────────


def _runs(cases: int = 2, reps: int = 2) -> tuple[CaseRun, ...]:
    """``cases`` distinct cases, each answered ``reps`` times — one passing attempt each."""
    return tuple(
        CaseRun(
            result=CaseResult(
                case_id=f"case-{index:02d}",
                attempts=(AttemptOutcome(passed=True, input_tokens=10, output_tokens=5),),
            ),
            config_id="default",
        )
        for index in range(cases)
        for _ in range(reps)
    )


def _record(
    *,
    dials: Mapping[str, Any] | None = None,
    runs: tuple[CaseRun, ...] | None = None,
    corpus_cases: int | None = 5,
    bench_id: str = BENCH_ID,
) -> dict[str, Any]:
    """One bench record, assembled through the pure builder every bench record goes through."""
    return build(
        BenchArtifacts(
            bench_id=bench_id,
            site=IDENTITY,
            engine=ENGINE,
            corpus=CorpusIdentity(
                version="decide/1", hash="deadbeefdeadbeef", case_count=corpus_cases, split="all"
            ),
            harness_hash="1122334455667788",
            harness_dials=dict(dials) if dials is not None else None,
            runs=_runs() if runs is None else runs,
            label="nightly",
            started_utc="2026-08-01T10:11:12Z",
            finished_utc="2026-08-01T10:12:00Z",
            complete=True,
        )
    )


def _row(rendered: str, label: str) -> str:
    """The value of the one row carrying ``label`` — the report, read as an operator reads it."""
    rows = [line for line in rendered.splitlines() if line.strip().startswith(label)]
    assert rows, f"no row labelled {label!r} in:\n{rendered}"
    return rows[0].strip()[len(label) :].strip()


def _block(rendered: str, heading: str) -> str:
    """Everything under ``heading``, up to the next line indented no further than it."""
    lines = rendered.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == heading]
    assert starts, f"no block headed {heading!r} in:\n{rendered}"
    start = starts[0]
    depth = len(lines[start]) - len(lines[start].lstrip())
    kept: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= depth:
            break
        kept.append(line)
    return "\n".join(kept)


# ── the headline: the co-primary pair, and never one half of it ──────────────────────────────


def test_the_report_renders_agreement_beside_the_approval_rate_it_was_bought_at(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    assert _row(rendered, "Approval-side agreement") == "0.5000"
    assert _row(rendered, "Approval rate") == "0.7500"


def test_the_report_names_the_excluded_count_of_approvals_the_gates_never_labelled(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    assert _row(rendered, "unlabeled (n/a)") == "1"


def test_the_report_renders_the_revise_rate_and_the_revise_flip_rate_beside_the_pair(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    assert _row(rendered, "revise_rate") == "0.2500"
    assert _row(rendered, "revise_flip_rate") == "1.0000"


def test_an_agreement_carried_without_its_approval_rate_is_refused_rather_than_rendered_alone():
    """The pairing rule of #206, enforced by the reader too: half a pair is not a headline."""
    half = {"decide": {"approval": {"agreement": 0.5, "decided": 4}}}

    rendered = render_bench_report(_record(dials=half))

    assert "0.5000" not in rendered
    assert "REFUSED" in rendered
    assert "approval rate" in rendered


# ── the population: n, reps, and the cases nobody scored ─────────────────────────────────────


def test_the_report_names_how_many_cases_and_how_many_reps_the_bench_ran(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    assert _row(rendered, "Cases (n)") == "2"
    assert _row(rendered, "Runs (reps)") == "4"
    assert _row(rendered, "Reps per case") == "2"


def test_uneven_reps_leave_the_per_case_figure_n_a_rather_than_a_rounded_claim(tmp_path):
    uneven = _runs(cases=2, reps=1) + _runs(cases=1, reps=1)

    rendered = render_bench_report(_record(dials=DECIDE_DIALS, runs=uneven))

    assert _row(rendered, "Runs (reps)") == "3"
    assert _row(rendered, "Reps per case") == "n/a"


def test_the_report_names_the_corpus_cases_this_bench_never_scored(tmp_path):
    """The coverage gap between the corpus and the bench, stated rather than left to arithmetic."""
    rendered = render_bench_report(_record(dials=DECIDE_DIALS, corpus_cases=5))

    assert _row(rendered, "Corpus cases") == "5"
    assert _row(rendered, "not scored") == "3"


def test_an_unstated_corpus_size_leaves_the_unscored_count_n_a(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS, corpus_cases=None))

    assert _row(rendered, "Corpus cases") == "n/a"
    assert _row(rendered, "not scored") == "n/a"


# ── refusal-first: an absence never renders as a zero ────────────────────────────────────────


def test_a_null_figure_renders_as_the_literal_n_a_and_never_as_zero(tmp_path):
    nulled = {"decide": {"approval": {**PAIR, "agreement": None}, "revise_flip_rate": None}}

    rendered = render_bench_report(_record(dials=nulled))

    assert _row(rendered, "Approval-side agreement") == "n/a"
    assert _row(rendered, "revise_flip_rate") == "n/a"


def test_an_unmeasured_headline_metric_renders_n_a_rather_than_a_confident_zero(tmp_path):
    """Nothing ran, so there is no pass rate — a bench that reported 0.0000 would be lying."""
    rendered = render_bench_report(_record(dials=DECIDE_DIALS, runs=()))

    assert _row(rendered, "job_pass_rate") == "n/a"
    assert _row(rendered, "p50_s") == "n/a"


# ── the per-axis breakdown, driven by the record ─────────────────────────────────────────────


def test_the_report_renders_every_axis_and_level_the_record_carries(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    strata = _block(rendered, "strata")
    assert "margin_distance" in strata
    assert "evidence_depth" in strata
    assert "comfortable" in strata
    assert "near" in strata


def test_a_stratum_renders_its_agreement_through_the_same_pairing_rule_as_the_headline(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    near = _block(rendered, "near")
    assert _row(near, "Approval-side agreement") == "0.0000"
    assert _row(near, "Approval rate") == "1.0000"


def test_a_record_from_another_site_renders_its_own_axes_through_the_same_code_path(tmp_path):
    """Nothing decide-specific in the verb: the reading appears because the record carries it."""
    rendered = render_bench_report(_record(dials=STUB_DIALS))

    strata = _block(rendered, "strata")
    assert "prompt_length" in strata
    assert _row(_block(strata, "long"), "usefulness") == "0.2500"
    assert _row(_block(strata, "short"), "usefulness") == "n/a"
    assert "agreement" not in rendered.lower()


# ── a coder record's own shapes, rendered by exactly the same reader (#227) ──────────────────


def test_the_report_renders_a_coder_records_per_axis_pass_breakdown(tmp_path):
    rendered = render_bench_report(_record(dials=CODER_DIALS))

    strata = _block(rendered, "strata")
    assert "api_surface" in strata and "oracle_mode" in strata
    assert _row(_block(strata, "bars_only"), JOB_PASS_ROW) == "1.0000"
    assert _row(_block(strata, "exits"), JOB_PASS_ROW) == "0.5000"


def test_a_whole_coder_pass_pair_renders_through_the_pairs_own_labelled_rows(tmp_path):
    """(#304) The report and the type tell one story: the manifest's labels, not the JSON keys."""
    rendered = render_bench_report(_record(dials=CODER_DIALS))

    rates = _block(rendered, "rates")
    assert _row(rates, FIRST_ATTEMPT_ROW) == "0.3333"
    assert _row(rates, JOB_PASS_ROW) == "0.6667"
    assert _row(rates, "first ask landed") == "1"
    assert _row(rates, "landed in the end") == "2"
    assert "job_pass_rate" not in rates
    assert "pass_label" not in rates


def test_a_nested_stratum_renders_its_pass_pair_through_the_same_labels_as_the_headline(tmp_path):
    """The pairing rule is the block's, not the depth's: an axis level is a pair block too."""
    strata = _block(render_bench_report(_record(dials=CODER_DIALS)), "strata")

    exits = _block(strata, "exits")
    assert _row(exits, FIRST_ATTEMPT_ROW) == "0.0000"
    assert _row(exits, JOB_PASS_ROW) == "0.5000"
    assert "first_attempt_pass_rate" not in exits


def test_a_job_pass_rate_carried_without_its_first_attempt_rate_is_refused_rather_than_rendered():
    """(#304) The coder's half-truth is refused where a human reads it, as DECIDE's already is.

    A retry-informed rate alone is the same shape of lie a bare agreement is — a composition that
    fails first and recovers twice is not the product one that lands it immediately is.
    """
    half = {"coder": {"rates": {"job_pass_rate": 0.5, "passed_jobs": 2}}}

    rendered = render_bench_report(_record(dials=half))

    assert "0.5000" not in rendered
    assert "REFUSED" in rendered
    assert "job_pass_rate" in rendered
    assert "first-attempt rate" in rendered


def test_the_records_own_metrics_are_not_a_sites_pair_and_render_as_the_cores_own_rows(tmp_path):
    """The pairing rule governs the reading a record *quotes*, not the core's own arithmetic.

    ``metrics.job_pass_rate`` is the eval core's figure over attempt outcomes on every site, and it
    shares nothing but a spelling with the coder's pair. A reader that refused it would be refusing
    a block that carries both rates already — a refusal that would read as a lie.
    """
    rendered = render_bench_report(_record(dials=CODER_DIALS))

    assert "REFUSED" not in rendered
    assert _row(rendered, "job_pass_rate") == "1.0000"
    assert _row(rendered, "first_attempt_pass_rate") == "1.0000"


def test_the_report_renders_the_failure_taxonomy_share_table_with_the_knob_each_share_points_at(
    tmp_path,
):
    rendered = render_bench_report(_record(dials=CODER_DIALS))

    classes = _block(rendered, "classes")
    truncated = _block(classes, "truncated")
    assert _row(truncated, "share") == "0.7500"
    assert _row(truncated, "knob") == "coder max-tokens"
    assert _row(_block(classes, "unclassified"), "count") == "0"


def test_the_same_shapes_under_another_sites_key_render_identically(tmp_path):
    """No site branching in the renderer: the reading appears because the record carries it."""
    coder = render_bench_report(_record(dials={"answers": "fresh", "coder": CODER_READING}))
    stubby = render_bench_report(_record(dials={"answers": "fresh", "stubby": CODER_READING}))

    assert _block(coder, "dials.coder") == _block(stubby, "dials.stubby")


def test_a_record_carrying_no_dials_at_all_still_renders_its_generic_metrics_block(tmp_path):
    rendered = render_bench_report(_record(dials=None))

    assert _row(rendered, "job_pass_rate") == "1.0000"
    assert _row(rendered, "first_attempt_pass_rate") == "1.0000"


def test_a_fresh_interpreter_that_imports_the_report_module_loads_no_site_module_at_all():
    """(#304) The docstring's "site-generic" claim, made structural rather than prose.

    The reader knows the record's own shape and the eval layer's shared reading vocabulary. It
    knows no site's types: not ``ApprovalPair``, not ``PassRates``, not a taxonomy — which is why
    the whole-or-refused rule below can be one rule over every pair a record carries instead of
    one site's rule hard-wired into a generic renderer.
    """
    probe = (
        "import json, sys\n"
        "import noctis.eval.bench_report\n"
        "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] == 'noctis')))\n"
    )

    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert json.loads(finished.stdout) == [
        "noctis",
        "noctis.eval",
        "noctis.eval.bench_report",
        "noctis.eval.reading",
    ]


def test_the_report_names_the_bench_the_site_and_the_key_its_numbers_are_comparable_under(tmp_path):
    rendered = render_bench_report(_record(dials=DECIDE_DIALS))

    assert BENCH_ID in rendered
    assert _row(rendered, "Site") == "decide"
    assert "prompt_asset_hash=c0ffee1234567890" in rendered


def test_the_report_names_the_file_it_was_rendered_from_when_it_is_given_one(tmp_path):
    path = tmp_path / "bench" / BENCH_ID / BENCH_RECORD_NAME

    rendered = render_bench_report(_record(dials=DECIDE_DIALS), source=path)

    assert str(path) in rendered


# ── the verb ─────────────────────────────────────────────────────────────────────────────────


def _publish(tmp_path, record: Mapping[str, Any] | str, bench_id: str = BENCH_ID) -> Path:
    """One bench directory in the workspace's bench area (conftest pins the workspace)."""
    directory = bench_root(tmp_path / "workspace") / bench_id
    directory.mkdir(parents=True)
    target = directory / BENCH_RECORD_NAME
    body = record if isinstance(record, str) else json.dumps(record, indent=2)
    target.write_text(body, encoding="utf-8")
    return target


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``root`` with its bytes and its mtime — a tree, pinned."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_bench_report_prints_the_headline_pair_for_a_record_in_the_bench_area(tmp_path):
    _publish(tmp_path, _record(dials=DECIDE_DIALS))

    result = runner.invoke(app, ["bench", "report", BENCH_ID])

    assert result.exit_code == 0, result.output
    assert _row(result.output, "Approval-side agreement") == "0.5000"
    assert _row(result.output, "Approval rate") == "0.7500"


def test_bench_report_names_the_bench_root_it_looked_in_when_no_bench_answers_the_id(tmp_path):
    _publish(tmp_path, _record(dials=DECIDE_DIALS))

    result = runner.invoke(app, ["bench", "report", "20990101T000000Z-nobody"])

    assert result.exit_code == 1
    assert str(bench_root(tmp_path / "workspace")) in result.output
    assert "20990101T000000Z-nobody" in result.output


def test_bench_report_refuses_an_invalid_record_with_every_problem_the_validator_names(tmp_path):
    _publish(tmp_path, {"schema_version": 99, "kind": "noctis.run"})

    result = runner.invoke(app, ["bench", "report", BENCH_ID])

    assert result.exit_code == 1
    assert "schema_version" in result.output
    assert "kind" in result.output
    assert "Approval-side agreement" not in result.output


def test_bench_report_refuses_an_unreadable_record_rather_than_printing_half_a_report(tmp_path):
    path = _publish(tmp_path, "{ this is not json")

    result = runner.invoke(app, ["bench", "report", BENCH_ID])

    assert result.exit_code == 1
    assert str(path) in result.output


def test_importing_the_engine_cli_loads_no_benchmark_code_at_all(tmp_path):
    """The deferred wiring, from the runtime side: `noctis run` never pays for the eval layer.

    The guard asserts the *shape* of the one permitted import statically; this asserts what that
    shape buys — a fresh interpreter that has imported the whole CLI holds no ``noctis.eval``
    module, so nothing on a production path can reach benchmark infrastructure by accident.
    """
    probe = "import noctis.cli, sys; print([m for m in sys.modules if m.startswith('noctis.eval')])"

    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert loaded.stdout.strip() == "[]"


def test_reporting_a_bench_writes_nothing_into_the_bench_tree(tmp_path):
    _publish(tmp_path, _record(dials=DECIDE_DIALS))
    root = bench_root(tmp_path / "workspace")
    before = _snapshot(root)

    result = runner.invoke(app, ["bench", "report", BENCH_ID])

    assert result.exit_code == 0, result.output
    assert _snapshot(root) == before


# ── the mined record reads through exactly the same verb ─────────────────────────────────────

RUN_ID = "20260720T144233Z-a3f9c1"
MIN_TRIALS = 6


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
        name,
        card(test=1.0, train=1.2, holdout=2.0 if promoted else 0.0, symbol_holdout=2.0),
    )
    journal.record_approval(
        name,
        promoted=promoted,
        rationale="the panel holds up out of sample",
        params={"lookback": 10},
        symbols=["AAPL", "MSFT"],
        holdout_symbols=["NVDA"],
    )


def _mined_bench(tmp_path) -> str:
    """A mined DECIDE corpus and the zero-spend retrospective record #208 publishes over it."""
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
    mine_decide_corpus(workspace / "runs", cases_root=workspace / "cases")
    corpus = load_decide_corpus(workspace / "cases")
    return write_retrospective_bench(corpus, bench_root=bench_root(workspace)).bench_id


def test_the_mined_retrospective_record_reports_through_the_same_verb_as_any_other(tmp_path):
    bench_id = _mined_bench(tmp_path)

    result = runner.invoke(app, ["bench", "report", bench_id])

    assert result.exit_code == 0, result.output
    assert _row(result.output, "Approval-side agreement") == "0.5000"
    assert _row(result.output, "Approval rate") == "1.0000"


def test_the_mined_records_own_difficulty_axes_drive_its_per_axis_breakdown(tmp_path):
    bench_id = _mined_bench(tmp_path)

    result = runner.invoke(app, ["bench", "report", bench_id])

    strata = _block(result.output, "strata")
    assert MARGIN_AXIS in strata
    assert BINDING_GATE_AXIS in strata
    assert EVIDENCE_DEPTH_AXIS in strata
