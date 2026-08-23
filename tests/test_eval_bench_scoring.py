"""#213: a live bench scores through the site's *declared* scorers, and the reading lands in the
record.

Before this story a live bench published emit-pass rates and nothing else: the DECIDE scorer existed
(#206), the re-run adapter wired it to a frozen case (#209), and the retrospective miner published a
whole agreement reading (#208) — but the runner never called any of it, so a live ``bench.json``
carried no agreement at all and ``bench report`` had two different stories to tell.

Four properties carry the story here, and every one of them is asserted from the outside — a record
on disk, or the text ``bench report`` prints:

* **The pass is site-agnostic.** The runner hands every answered case to whatever scorers the site's
  declaration carries and folds each reading into ``harness.dials``. A stub site's scorer proves the
  path has nothing of DECIDE in it; a stub site with *no* scorer proves the record it wrote before
  this story is the record it still writes.
* **The DECIDE reading is the retrospective one, computed from fresh verdicts.** Same keys, same
  arithmetic, same rendering — so ``bench report`` prints one headline pair whichever path produced
  the record.
* **Reps fold within the case.** A case is the equal-weight unit (#175), so every answer one case
  gave folds into the one outcome it contributes; a case whose answers hold no majority is unsettled
  rather than guessed.
* **An unreadable reply is a failed attempt and an exclusion, never a denominator.** Agreement is a
  number about candidates somebody really decided.

Everything is stubbed: a mined corpus in a throwaway workspace, canned replies, no network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from typer.testing import CliRunner

from noctis.cli import app
from noctis.config.rehydrate import freeze_inputs
from noctis.config.settings import Settings
from noctis.eval import cli as cli_module
from noctis.eval import runner as runner_module
from noctis.eval.bootstrap import BenchSeams, cases_root, load_corpus
from noctis.eval.decide_case import DIFFICULTY_AXES
from noctis.eval.decide_miner import (
    mine_decide_corpus,
    retrospective_dials,
    write_retrospective_bench,
)
from noctis.eval.decide_scorer import score_decide_batch
from noctis.eval.decide_site import DECIDE_SCORER
from noctis.eval.harness import HarnessSpec
from noctis.eval.identity import SiteIdentity
from noctis.eval.knobs import SiteKnobs
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.registry import index_sites
from noctis.eval.runner import (
    BENCH_RECORD_NAME,
    Attempt,
    AttemptRequest,
    ConflictingReading,
    bench_root,
    harness_dials,
)
from noctis.eval.site import AgentSite, AnsweredCase
from noctis.research.episode import EmitContract

from ._promotion_cases import card

runner = CliRunner()

RUN_ID = "20260720T144233Z-a3f9c1"
MIN_TRIALS = 6


# ── a stub site whose scorer records what the pass handed it ──────────────────────────────────


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

STUB_IDENTITY = SiteIdentity(
    site_id="stub",
    version="1",
    prompt_asset_groups=("stub",),
    prompt_asset_hash="7c1f0d9a4b2e6f83",
)


@dataclass
class RecordingScorer:
    """A stub site's scorer: keeps everything the pass handed it, publishes a canned reading."""

    reading: Mapping[str, Any] | None = None
    answered: tuple[AnsweredCase, ...] = ()
    axes: tuple[str, ...] | None = None

    def read(
        self, answered: Sequence[AnsweredCase], *, axes: tuple[str, ...] = ()
    ) -> Mapping[str, Any] | None:
        self.answered = tuple(answered)
        self.axes = axes
        return self.reading


def _stub_site(*scorers: Any, difficulty_axes: tuple[str, ...] = ()) -> AgentSite[Any, Any]:
    """One stub declaration carrying exactly the scorers a test wants exercised."""
    return AgentSite(
        id="stub",
        version="1",
        contract=STUB_CONTRACT,
        render=_render,
        knobs=StubKnobs,
        scorers=tuple(scorers),
        difficulty_axes=difficulty_axes,
    )


@dataclass
class StubAttempt:
    """A stub model call: answers every case with its own id, and passes."""

    def __call__(self, request: AttemptRequest) -> Attempt:
        return Attempt(
            outcome=AttemptOutcome(passed=True, model="bench/oracle", seconds=0.1),
            output=f"answered {request.case.case_id}",
            served_model="bench/oracle-2026",
        )


def _stub_seams(site: AgentSite[Any, Any]) -> BenchSeams:
    """The stub site's whole assembly: its registry, its identity, and a stub model call."""
    return BenchSeams(attempt=StubAttempt(), registry=index_sites((site,)), identity=STUB_IDENTITY)


def _stub_corpus(tmp_path: Path, ids: tuple[str, ...] = ("a", "b")) -> Path:
    """A handful of stub cases in one difficulty stratum."""
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


def _candidate(journal: Any, name: str, *, promoted: bool) -> None:
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


def _decide_corpus(tmp_path: Path, candidates: Mapping[str, bool]) -> Path:
    """A mined DECIDE corpus: one case per named candidate, labelled as the mapping says."""
    from noctis.research.journal import ExperimentJournal

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
    for name, promoted in candidates.items():
        _candidate(journal, name, promoted=promoted)
    mine_decide_corpus(workspace / "runs", cases_root=cases_root(workspace))
    return cases_root(workspace) / "decide"


def _verdict(verdict: str) -> str:
    """One model reply as prose wrapped around the emitted object — the real transport."""
    payload = {
        "verdict": verdict,
        "reason": "the holdout numbers hold up",
        "class_exhausted": False,
        "class_tag": "volatility_squeeze",
    }
    return f"Here is my verdict.\n\n```json\n{json.dumps(payload)}\n```\n"


@dataclass
class CannedReplies:
    """The attempt seam a live DECIDE bench runs on: one canned reply per candidate.

    Production's own posture, in miniature: the reply is whatever the backend said, and an attempt
    passes exactly when the site's emit contract admits it — so an unreadable reply is a failed
    attempt here for the same reason it is one in :func:`noctis.eval.bootstrap.live_attempt`.
    """

    replies: Mapping[str, str]
    per_rep: Mapping[tuple[str, int], str] = field(default_factory=dict)

    def __call__(self, request: AttemptRequest) -> Attempt:
        strategy = request.case.case_id.split("-")[0]
        reply = self.per_rep.get((strategy, request.rep), self.replies[strategy])
        scored = DECIDE_SCORER.score(request.case, reply)
        return Attempt(
            outcome=scored.attempt_outcome(
                AttemptOutcome(passed=False, model="bench/oracle", seconds=2.0)
            ),
            output=reply,
            served_model="bench/oracle-20260701",
        )


def _only_bench(tmp_path: Path) -> Path:
    """The single bench directory the run under test wrote."""
    (directory,) = sorted(bench_root(tmp_path / "workspace").iterdir())
    return directory


def _record(tmp_path: Path) -> Mapping[str, Any]:
    """The record the one bench under test published."""
    return json.loads((_only_bench(tmp_path) / BENCH_RECORD_NAME).read_text(encoding="utf-8"))


def _decide_block(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """The DECIDE reading a record carries under its verbatim dials subtree."""
    return record["harness"]["dials"]["decide"]


def _section(report: str, title: str) -> list[str]:
    """One titled section of a rendered bench report, as its lines."""
    for chunk in report.split("\n\n"):
        if chunk.splitlines()[0] == title:
            return chunk.splitlines()
    raise AssertionError(f"no section {title!r} in:\n{report}")


def _headline(report: str) -> list[str]:
    """The co-primary pair block of a report's DECIDE reading — its title and its six rows."""
    return _section(report, "dials.decide")[1:8]


# ── the seam: a site's declared scorers, and nothing about any one site ───────────────────────


def test_a_sites_declared_scorer_folds_its_reading_into_the_records_harness_dials(tmp_path):
    _stub_corpus(tmp_path)
    scorer = RecordingScorer(reading={"stub_reading": {"answers": 2}})

    cli_module.run_bench("stub", seams=_stub_seams(_stub_site(scorer)))

    assert _record(tmp_path)["harness"]["dials"]["stub_reading"] == {"answers": 2}


def test_a_site_that_declares_no_scorer_publishes_the_composed_harness_dials_and_nothing_else(
    tmp_path,
):
    """The regression that keeps this story additive: today's record, byte for byte."""
    _stub_corpus(tmp_path)

    cli_module.run_bench("stub", seams=_stub_seams(_stub_site()))

    assert _record(tmp_path)["harness"]["dials"] == harness_dials(HarnessSpec())


def test_a_scorer_with_nothing_to_say_leaves_the_dials_exactly_as_the_harness_composed_them(
    tmp_path,
):
    _stub_corpus(tmp_path)

    cli_module.run_bench("stub", seams=_stub_seams(_stub_site(RecordingScorer(reading=None))))

    assert _record(tmp_path)["harness"]["dials"] == harness_dials(HarnessSpec())


def test_the_scoring_pass_is_handed_every_answered_case_with_the_replies_it_gave(tmp_path):
    _stub_corpus(tmp_path)
    scorer = RecordingScorer()

    cli_module.run_bench("stub", reps=2, seams=_stub_seams(_stub_site(scorer)))

    assert [(one.case.case_id, one.rep, one.replies) for one in scorer.answered] == [
        ("case-a", 1, ("answered case-a",)),
        ("case-a", 2, ("answered case-a",)),
        ("case-b", 1, ("answered case-b",)),
        ("case-b", 2, ("answered case-b",)),
    ]


def test_the_scoring_pass_is_handed_the_difficulty_axes_its_site_declares(tmp_path):
    """The declaration is the one place a site's facts live; the runner carries them to the pass."""
    _stub_corpus(tmp_path)
    scorer = RecordingScorer()

    cli_module.run_bench(
        "stub", seams=_stub_seams(_stub_site(scorer, difficulty_axes=("reasoning",)))
    )

    assert scorer.axes == ("reasoning",)


def test_a_site_that_declares_no_axes_hands_its_scorer_none_to_stratify_by(tmp_path):
    _stub_corpus(tmp_path)
    scorer = RecordingScorer()

    cli_module.run_bench("stub", seams=_stub_seams(_stub_site(scorer)))

    assert scorer.axes == ()


def test_every_declared_scorer_of_a_site_contributes_its_own_reading(tmp_path):
    _stub_corpus(tmp_path)
    site = _stub_site(RecordingScorer(reading={"first": 1}), RecordingScorer(reading={"second": 2}))

    cli_module.run_bench("stub", seams=_stub_seams(site))

    dials = _record(tmp_path)["harness"]["dials"]
    assert (dials["first"], dials["second"]) == (1, 2)


def test_a_reading_that_would_overwrite_a_composed_harness_dial_is_refused_naming_the_key(
    tmp_path, capsys
):
    _stub_corpus(tmp_path)
    site = _stub_site(RecordingScorer(reading={"is_production": False}))

    with pytest.raises(typer.Exit):
        cli_module.run_bench("stub", seams=_stub_seams(site))

    assert "is_production" in capsys.readouterr().err


def test_the_conflicting_reading_refusal_is_raised_by_name_from_the_runner_itself(tmp_path):
    """The clean CLI line above is a rendering of this; the refusal itself is the runner's."""
    assert issubclass(ConflictingReading, ValueError)


_DECIDE_MENTION = re.compile(r"decide_\w+|['\"]decide['\"]")


def test_neither_the_runner_nor_the_bench_verbs_carry_any_decide_specific_wiring():
    """Site-agnostic is structural: the two generic modules name no site at all."""
    for module in (runner_module, cli_module):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert not _DECIDE_MENTION.search(source), module.__name__


# ── the DECIDE reading, computed from fresh verdicts against recorded labels ──────────────────


def test_a_live_decide_bench_carries_the_approval_pair_its_fresh_verdicts_earned(tmp_path):
    """Hand-computed: four decided candidates, three approvals, two of them promoted."""
    _decide_corpus(
        tmp_path,
        {"alpha_one": True, "beta_two": False, "gamma_three": True, "delta_four": True},
    )
    canned = CannedReplies(
        {
            "alpha_one": _verdict("approve"),
            "beta_two": _verdict("approve"),
            "gamma_three": _verdict("reject"),
            "delta_four": _verdict("approve"),
        }
    )

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    approval = _decide_block(_record(tmp_path))["approval"]
    assert approval["agreement"] == pytest.approx(2 / 3)
    assert approval["approval_rate"] == 0.75
    assert (approval["decided"], approval["approvals"], approval["promoted"]) == (4, 3, 2)


def test_a_live_decide_bench_carries_the_deferral_figures_its_fresh_verdicts_earned(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": True})
    canned = CannedReplies(
        {"alpha_one": _verdict("approve"), "beta_two": _verdict("revise")},
    )

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    block = _decide_block(_record(tmp_path))
    assert block["revise_rate"] == 0.5
    assert block["revises"] == 1
    # A re-run is one ask, so nothing it answered can have *flipped* a verdict history spent.
    assert block["revise_flip_rate"] == 0.0


def test_a_live_decide_bench_stratifies_its_reading_by_every_declared_difficulty_axis(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": False})
    canned = CannedReplies({"alpha_one": _verdict("approve"), "beta_two": _verdict("approve")})

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    assert tuple(_decide_block(_record(tmp_path))["strata"]) == DIFFICULTY_AXES


def test_a_live_decide_bench_lists_one_row_per_case_with_the_axes_it_was_scored_on(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": False})
    canned = CannedReplies({"alpha_one": _verdict("approve"), "beta_two": _verdict("reject")})

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    rows = _decide_block(_record(tmp_path))["cases"]
    assert [(row["case_id"], row["verdict"]) for row in rows] == [
        (f"alpha_one-{RUN_ID}", "approve"),
        (f"beta_two-{RUN_ID}", "reject"),
    ]
    assert all(set(row["difficulty"]) == set(DIFFICULTY_AXES) for row in rows)


def test_a_live_decide_bench_marks_its_dials_as_freshly_answered_rather_than_recorded(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True})
    canned = CannedReplies({"alpha_one": _verdict("approve")})

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    dials = _record(tmp_path)["harness"]["dials"]
    assert (dials["answers"], dials["retrospective"], dials["attempt_calls"]) == ("fresh", False, 1)


def test_the_live_decide_reading_carries_the_keys_the_retrospective_reading_publishes(tmp_path):
    """Dials parity: the same block, so one report renders both without knowing which it holds."""
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": False})
    canned = CannedReplies({"alpha_one": _verdict("approve"), "beta_two": _verdict("approve")})
    corpus = load_corpus("decide", cases_root=cases_root(tmp_path / "workspace"))

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    recorded = retrospective_dials(corpus.cases, (), score_decide_batch(()))["decide"]
    live = _decide_block(_record(tmp_path))
    assert set(recorded) <= set(live)
    assert set(recorded["approval"]) == set(live["approval"])


# ── the reps fold: a case is the equal-weight unit ────────────────────────────────────────────


def test_every_rep_of_one_case_folds_into_the_single_outcome_that_case_contributes(tmp_path):
    """Two cases, three reps each — six answers, and still two decided candidates."""
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": False})
    canned = CannedReplies(
        {"alpha_one": _verdict("approve"), "beta_two": _verdict("approve")},
        per_rep={("alpha_one", 3): _verdict("reject")},
    )

    cli_module.run_bench("decide", reps=3, seams=BenchSeams(attempt=canned))

    approval = _decide_block(_record(tmp_path))["approval"]
    assert approval["decided"] == 2
    assert approval["approvals"] == 2


def test_a_case_whose_reps_settle_on_no_single_verdict_is_unsettled_rather_than_guessed(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True})
    canned = CannedReplies(
        {"alpha_one": _verdict("approve")}, per_rep={("alpha_one", 2): _verdict("reject")}
    )

    cli_module.run_bench("decide", reps=2, seams=BenchSeams(attempt=canned))

    block = _decide_block(_record(tmp_path))
    assert block["unsettled"] == 1
    assert block["approval"]["decided"] == 0


# ── an unreadable reply: a failed attempt, an exclusion, never a denominator ──────────────────


def test_an_unparseable_reply_is_counted_as_unreadable_and_kept_out_of_the_denominator(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": True, "gamma_three": True})
    canned = CannedReplies(
        {
            "alpha_one": _verdict("approve"),
            "beta_two": _verdict("approve"),
            "gamma_three": "I cannot tell from this evidence.",
        }
    )

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    block = _decide_block(_record(tmp_path))
    assert block["unreadable"] == 1
    assert block["approval"]["decided"] == 2
    assert block["approval"]["agreement"] == 1.0


def test_an_unparseable_reply_stays_a_failed_attempt_in_the_record_it_was_asked_for(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": True})
    canned = CannedReplies(
        {"alpha_one": _verdict("approve"), "beta_two": "I cannot tell from this evidence."}
    )

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    assert _record(tmp_path)["metrics"]["first_attempt_pass_rate"] == 0.5


def test_an_unreadable_case_row_names_why_no_verdict_was_read_from_it(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True})
    canned = CannedReplies({"alpha_one": "I cannot tell from this evidence."})

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))

    (row,) = _decide_block(_record(tmp_path))["cases"]
    assert row["verdict"] is None
    assert row["error"]


# ── the report: one headline pair, whichever path wrote the record ────────────────────────────


def test_the_headline_pair_renders_identically_from_a_live_bench_and_the_retrospective_baseline(
    tmp_path,
):
    """The epic's second deliverable, end to end: run, mine, report both, compare the headline."""
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": False, "gamma_three": True})
    canned = CannedReplies(
        {
            "alpha_one": _verdict("approve"),
            "beta_two": _verdict("approve"),
            "gamma_three": _verdict("approve"),
        }
    )
    workspace = tmp_path / "workspace"
    corpus = load_corpus("decide", cases_root=cases_root(workspace))

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))
    live = _only_bench(tmp_path).name
    baseline = write_retrospective_bench(corpus, bench_root=bench_root(workspace)).bench_id

    live_report = runner.invoke(app, ["bench", "report", live])
    baseline_report = runner.invoke(app, ["bench", "report", baseline])

    assert live_report.exit_code == 0, live_report.output
    assert baseline_report.exit_code == 0, baseline_report.output
    headline = _headline(live_report.output)
    assert headline == _headline(baseline_report.output)
    assert "Approval-side agreement" in headline[1]


def test_the_report_over_a_live_decide_record_prints_the_agreement_the_bench_measured(tmp_path):
    _decide_corpus(tmp_path, {"alpha_one": True, "beta_two": False})
    canned = CannedReplies({"alpha_one": _verdict("approve"), "beta_two": _verdict("approve")})

    cli_module.run_bench("decide", seams=BenchSeams(attempt=canned))
    result = runner.invoke(app, ["bench", "report", _only_bench(tmp_path).name])

    assert result.exit_code == 0, result.output
    agreement = _headline(result.output)[1]
    assert "Approval-side agreement" in agreement
    assert "0.5000" in agreement
