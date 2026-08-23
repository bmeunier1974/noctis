"""The eval reading layer's golden net (#302, the first story of epic #301).

Epic #301 types the ``Scorer`` seam and moves the eval layer's reading vocabulary into one module.
Nothing in that move is meant to change a published number, a key or a key's *position* — so this
file is the net that says so, before any of it happens: five artifacts, derived in-test from
deterministic fixture batches through the code paths that publish them today, and compared **byte
for byte** against a committed golden.

* ``coder_dials_golden.json`` — what :data:`~noctis.eval.coder_scorer.CODER_SCORER` publishes over a
  live coder bench: the co-primary pass pair, the effort/escalation/cost/failure blocks, the
  per-axis strata and the detector warning rows.
* ``decide_dials_golden.json`` — what :data:`~noctis.eval.decide_site.DECIDE_SCORER` publishes over
  a live DECIDE bench: the approval pair, the deferral figures, the per-case rows, the strata, and
  the two exclusion counts a rep-fold produces (an unreadable case and an unsettled one).
* ``decide_retrospective_dials_golden.json`` — the same reading shaped by
  :func:`~noctis.eval.decide_miner.retrospective_dials` over the *recorded* verdicts of the very
  same cases, so the two paths' shared shaping functions are pinned on both sides.
* ``coder_bench_report_golden.txt`` / ``decide_bench_report_golden.txt`` — those two live readings
  quoted into a whole bench record and rendered by
  :func:`~noctis.eval.bench_report.render_bench_report`, so the reader's own rows, indentation and
  ``n/a`` spellings are pinned too.

**Byte for byte, and key order is part of the pin.** The JSON goldens are dumped with
``sort_keys=False``: a refactor that re-orders the keys of a published reading changes the document
a reader diffs, so the order is snapshotted rather than normalised away.

**Deterministic by construction.** Every fixture is stated here in memory — no clock, no filesystem,
no randomness, no machine path — and every timestamp, bench id and hash a record carries is a
literal below. The one figure that comes from outside is the shipped price table's version (the
coder scorer prices a batch through :func:`~noctis.research.pricing.default_table`), which is
exactly the pin a cost figure deserves: a table bump moves the dollars, and a golden that hid that
would be lying about what the reading published.

**Regenerating.** ``NOCTIS_REGEN_GOLDENS=1 uv run pytest tests/test_eval_goldens.py`` rewrites every
golden from the current code paths. Do it deliberately, and explain the diff in the pull request —
the whole value of this file is that a diff here is a decision somebody made on purpose.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from noctis.eval.bench_report import render_bench_report
from noctis.eval.case import Case, parse_case
from noctis.eval.coder_detectors import SEVERITY, DegenerateFinding
from noctis.eval.coder_scorer import CODER_SCORER
from noctis.eval.coder_site import AttemptRecord, JobRecord
from noctis.eval.coder_taxonomy import register_coder_taxonomy
from noctis.eval.decide_case import (
    BINDING_GATE_AXIS,
    EVIDENCE_DEPTH_AXIS,
    MARGIN_AXIS,
    RecordedOutcome,
    decide_case_id,
)
from noctis.eval.decide_miner import decide_outcomes, retrospective_dials
from noctis.eval.decide_scorer import score_decide_batch
from noctis.eval.decide_site import DECIDE_SCORER
from noctis.eval.identity import SiteIdentity
from noctis.eval.metrics import AttemptOutcome, CaseResult
from noctis.eval.record import (
    BenchArtifacts,
    CaseRun,
    CorpusIdentity,
    EngineStamp,
    ModelConfig,
    build,
    validate,
)
from noctis.eval.site import AnsweredCase
from noctis.eval.taxonomy import FailureTaxonomy

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: Set this to ``1`` to rewrite every golden below from the current code paths — the deliberate
#: regeneration affordance, named in the module docstring.
REGEN_ENV = "NOCTIS_REGEN_GOLDENS"

CODER_DIALS_GOLDEN = "coder_dials_golden.json"
DECIDE_DIALS_GOLDEN = "decide_dials_golden.json"
DECIDE_RETROSPECTIVE_DIALS_GOLDEN = "decide_retrospective_dials_golden.json"
CODER_REPORT_GOLDEN = "coder_bench_report_golden.txt"
DECIDE_REPORT_GOLDEN = "decide_bench_report_golden.txt"


def _pinned(name: str, produced: str) -> None:
    """Compare ``produced`` against the committed golden ``name``, byte for byte."""
    golden = FIXTURES / name
    if os.environ.get(REGEN_ENV) == "1":  # pragma: no cover - the regeneration affordance
        golden.write_text(produced, encoding="utf-8")
    assert produced == golden.read_text(encoding="utf-8")


def _as_json(document: Mapping[str, Any]) -> str:
    """One reading as its golden spells it: today's key order, two spaces, one trailing newline."""
    return json.dumps(document, indent=2, sort_keys=False) + "\n"


# ── the identity every fixture record carries, stated rather than resolved ──────────────────

ENGINE = EngineStamp(
    engine_version=3,
    fingerprint={
        "gates": "f63d47b7b9604ab1",
        "backtest": "3ba3e0bf1c97134f",
        "prompts": "14eb169506a6b5aa",
    },
    noctis_version="0.1.0",
)

CONFIG_ID = "default"

CONFIGS = (
    ModelConfig(config_id=CONFIG_ID, provider="anthropic", requested_model="claude-sonnet-4"),
)


# ══ the coder batch ═════════════════════════════════════════════════════════════════════════

#: Two models the shipped price table carries, so every dollar figure is checkable with a pencil.
CODER_MODEL = "anthropic/claude-sonnet-4"
FALLBACK_MODEL = "anthropic/claude-opus-4"

#: Exactly one million input tokens and nothing else on every internal attempt.
SPEND: Mapping[str, int] = {
    "input_tokens": 1_000_000,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

#: Four gate errors the coder taxonomy recognises as four different classes.
MISNAMED_ERROR = "class sets name='probe' but the strategy/file name is 'gap_fade_revision'"
TRUNCATED_ERROR = "the reply was cut off by the output-token limit"
IMPORT_ERROR = "ModuleNotFoundError: No module named 'pandas_ta'"
SCENARIO_ERROR = "scenario 'rally_runs': expected a long position during leg 1, saw flat"

#: The brief the coder cases below are labelled around — inert to every figure, and stated anyway
#: so the fixture is a case a curator would recognise.
CODER_PAYLOAD: Mapping[str, Any] = {
    "thesis": "Long above a short moving average; the drift persists intraday.",
    "entry_exit": "Long when close > SMA(lookback); flat otherwise.",
    "param_space": "lookback int 5..40",
    "scenarios": "A rally pulls long; a steady decline stays flat.",
    "style": "momentum",
    "symbols": ["AAA", "BBB"],
}


def _coder_case(case_id: str, **difficulty: str) -> Case:
    """One coder case, admitted by the eval core exactly as a corpus file is."""
    return parse_case(
        {
            "site_id": "coder",
            "payload": dict(CODER_PAYLOAD),
            "provenance": "authored:2026-08-02",
            "tags": ["bucket:canary"],
            "difficulty": dict(difficulty),
            "split": "tuning",
        },
        case_id=case_id,
        source=f"{case_id}.yaml",
    )


# Three cases labelled far enough apart that every axis splits, and one (``donchian-reference``)
# that declares no ``api_surface`` at all, so the ``n/a`` level is pinned beside the real ones.
CODER_CASES: Mapping[str, Case] = {
    case.case_id: case
    for case in (
        _coder_case(
            "sma-cross-scratch",
            composition_mode="scratch",
            oracle_mode="authored",
            warmup_arithmetic="single",
            state_complexity="rolling",
            no_trade_tape="falsified",
            param_space_breadth="narrow",
            api_surface="indicators",
        ),
        _coder_case(
            "gap-fade-revision",
            composition_mode="revision",
            oracle_mode="fixed_spec",
            warmup_arithmetic="composed",
            state_complexity="latched",
            no_trade_tape="scale_free",
            param_space_breadth="broad",
            api_surface="exits",
        ),
        _coder_case(
            "donchian-reference",
            composition_mode="reference",
            oracle_mode="authored",
            warmup_arithmetic="higher_timeframe",
            state_complexity="stateless",
            no_trade_tape="trivial",
            param_space_breadth="moderate",
        ),
    )
}

#: Two detector findings, on two different jobs, so the warning rows and ``warned_jobs`` are both
#: exercised — and provably read by nothing else in the reading.
FLOOR_COLLAPSE = DegenerateFinding(
    detector="param_floor_collapse",
    severity=SEVERITY,
    summary="every tuned param sits on its space's floor (lookback=5 of 5..40)",
)
ONE_SIDED = DegenerateFinding(
    detector="one_sided_book",
    severity=SEVERITY,
    summary="the file took 41 long positions and never once went flat",
)


def _attempt(
    number: int,
    passed: bool,
    *,
    error: str | None = None,
    escalated: bool = False,
) -> AttemptRecord:
    """One *internal* coder attempt, as the authoring engine's wrapper records it."""
    model = FALLBACK_MODEL if escalated else CODER_MODEL
    return AttemptRecord(
        attempt=number,
        passed=passed,
        escalated=escalated,
        error=error,
        model=model,
        served_model=f"{model}-20260701",
        usage=dict(SPEND),
        artifact=f"attempt-{number}/{'strategy.py' if passed else 'failure.json'}",
    )


def _job(
    case_id: str,
    *attempts: AttemptRecord,
    seconds: float | None,
    findings: tuple[DegenerateFinding, ...] = (),
) -> JobRecord:
    """One whole authoring job: the gate's verdict, and every internal attempt behind it."""
    strategy = case_id.replace("-", "_")
    passed = any(one.passed for one in attempts)
    return JobRecord(
        case_id=case_id,
        strategy=strategy,
        passed=passed,
        attempts=attempts,
        error=None if passed else (attempts[-1].error if attempts else "the brief was refused"),
        path=f"__tmp/{strategy}.py" if passed else None,
        model=CODER_MODEL,
        seconds=seconds,
        findings=findings,
    )


def _coder_rep(case_id: str, rep: int, *jobs: JobRecord) -> tuple[AnsweredCase, CaseRun]:
    """One rep of one coder case: every runner attempt it took, as the bench retained them.

    A runner attempt *is* a whole authoring job, so a rep that was re-asked carries two job
    documents and the reading grades the settled one — which is exactly what the runner hands a
    scoring pass, produced here by the job's own :meth:`~noctis.eval.coder_site.JobRecord.attempt`.
    """
    attempts = tuple(job.attempt() for job in jobs)
    answered = AnsweredCase(
        case=CODER_CASES[case_id],
        config_id=CONFIG_ID,
        rep=rep,
        replies=tuple(one.output for one in attempts),
    )
    run = CaseRun(
        result=CaseResult(case_id=case_id, attempts=tuple(one.outcome for one in attempts)),
        config_id=CONFIG_ID,
        served_models=tuple(one.served_model for one in attempts),
    )
    return answered, run


def _unreadable_coder_rep(case_id: str, rep: int, reply: str) -> tuple[AnsweredCase, CaseRun]:
    """A rep whose retained output was not a job record at all — the ``unreadable`` count."""
    answered = AnsweredCase(
        case=CODER_CASES[case_id], config_id=CONFIG_ID, rep=rep, replies=(reply,)
    )
    run = CaseRun(
        result=CaseResult(
            case_id=case_id,
            attempts=(
                AttemptOutcome(
                    passed=False,
                    model=CODER_MODEL,
                    seconds=3.0,
                    error="the coder retained no job record",
                    **SPEND,
                ),
            ),
        ),
        config_id=CONFIG_ID,
        served_models=(f"{CODER_MODEL}-20260701",),
    )
    return answered, run


# The batch, rep by rep — a first-attempt pass, a retried rep whose settled job passed on its
# second internal attempt, a paid rescue, an escalation that rescued nothing, a job the engine
# refused before spending a completion, and a reply nobody can read.
_CODER_REPS = (
    _coder_rep("sma-cross-scratch", 1, _job("sma-cross-scratch", _attempt(1, True), seconds=4.0)),
    _coder_rep(
        "sma-cross-scratch",
        2,
        _job(
            "sma-cross-scratch",
            _attempt(1, False, error=TRUNCATED_ERROR),
            _attempt(2, False, error=TRUNCATED_ERROR),
            seconds=11.0,
        ),
        _job(
            "sma-cross-scratch",
            _attempt(1, False, error=TRUNCATED_ERROR),
            _attempt(2, True),
            seconds=9.0,
            findings=(FLOOR_COLLAPSE, ONE_SIDED),
        ),
    ),
    _coder_rep(
        "gap-fade-revision",
        1,
        _job(
            "gap-fade-revision",
            _attempt(1, False, error=SCENARIO_ERROR),
            _attempt(2, False, error=IMPORT_ERROR),
            _attempt(1, True, escalated=True),
            seconds=12.5,
            findings=(ONE_SIDED,),
        ),
    ),
    _coder_rep(
        "gap-fade-revision",
        2,
        _job(
            "gap-fade-revision",
            _attempt(1, False, error=MISNAMED_ERROR),
            _attempt(1, False, error=SCENARIO_ERROR, escalated=True),
            seconds=7.0,
        ),
    ),
    _coder_rep("donchian-reference", 1, _job("donchian-reference", seconds=None)),
    _unreadable_coder_rep(
        "donchian-reference", 2, "I would rather describe the file than write it."
    ),
)

CODER_ANSWERED = tuple(answered for answered, _ in _CODER_REPS)
CODER_RUNS = tuple(run for _, run in _CODER_REPS)


def _coder_reading() -> Mapping[str, Any]:
    """The whole coder reading over the batch above, as a record quotes it under ``harness``."""
    reading = CODER_SCORER.read(CODER_ANSWERED)
    assert reading is not None
    return reading.as_dials()


# ══ the DECIDE batch ════════════════════════════════════════════════════════════════════════

#: Two run ids of exactly the shape the engine mints, so mined provenance faces the real pattern.
RUN_A = "20260720T144233Z-a3f9c1"
RUN_B = "20260721T090000Z-b7d2e4"

DECIDE_MODEL = "anthropic/claude-haiku-4"

#: One decide ask's spend — a tenth of the coder's, and reported on every attempt.
DECIDE_SPEND: Mapping[str, int] = {
    "input_tokens": 100_000,
    "output_tokens": 2_000,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

#: The evidence half of a mined case's payload — the ask, inert to every figure below.
DECIDE_EVIDENCE: Mapping[str, Any] = {
    "trials": 8,
    "min_trials": 6,
    "best": {"lookback": 12, "threshold": 1.4},
    "panel": {"train": 1.21, "test": 0.97, "symbol_holdout": 0.88},
}
DECIDE_LEDGER_TAIL: tuple[Mapping[str, Any], ...] = (
    {"event": "stage", "at": "2026-07-20T14:00:00Z", "stage": "decide"},
)


def _decide_case(
    strategy: str,
    *,
    run_id: str,
    recorded: RecordedOutcome,
    margin: str | None = None,
    binding_gate: str | None = None,
    evidence_depth: str | None = None,
) -> Case:
    """One mined DECIDE case, admitted by the eval core exactly as a mined corpus file is."""
    declared = {
        MARGIN_AXIS: margin,
        BINDING_GATE_AXIS: binding_gate,
        EVIDENCE_DEPTH_AXIS: evidence_depth,
    }
    return parse_case(
        {
            "site_id": "decide",
            "payload": {
                "evidence": dict(DECIDE_EVIDENCE),
                "ledger_tail": [dict(row) for row in DECIDE_LEDGER_TAIL],
                "recorded_outcome": recorded.to_document(),
            },
            "provenance": f"mined:{run_id}",
            "tags": [f"run:{run_id}"],
            "difficulty": {axis: level for axis, level in declared.items() if level is not None},
            "split": "tuning",
        },
        case_id=decide_case_id(strategy, run_id),
        source=f"mined:{run_id}/{strategy}",
    )


def _recorded(
    verdict: str,
    *,
    label: str | None = None,
    revises: int = 0,
    revise_flip: bool = False,
    final_ask: bool = False,
) -> RecordedOutcome:
    """What history recorded happening to one candidate, in the case module's own record."""
    sequence = tuple(["revise"] * revises + [verdict])
    return RecordedOutcome(
        eventual_verdict=verdict,
        verdict_sequence=sequence,
        revises=revises,
        revise_flip=revise_flip,
        final_ask=final_ask,
        label=label,
    )


# Eight mined candidates: both gate labels, an approval the gates never judged, a rejection, two
# spent revise caps (so the binary final contract is the one a re-run is read through), and two
# axes levels a case declares nothing on.
DECIDE_CASES: tuple[Case, ...] = (
    _decide_case(
        "vol_breakout_squeeze",
        run_id=RUN_A,
        recorded=_recorded("approve", label="promoted", revises=1, revise_flip=True),
        margin="near",
        binding_gate="none",
        evidence_depth="at-floor",
    ),
    _decide_case(
        "gap_fade_open",
        run_id=RUN_A,
        recorded=_recorded("approve", label="refused"),
        margin="comfortable",
        binding_gate="forward_holdout",
        evidence_depth="above-floor",
    ),
    _decide_case(
        "range_compression",
        run_id=RUN_B,
        recorded=_recorded("approve"),
        margin="comfortable",
        evidence_depth="below-floor",
    ),
    _decide_case(
        "mean_revert_close",
        run_id=RUN_B,
        recorded=_recorded("reject", revises=2, revise_flip=True, final_ask=True),
        margin="near",
        binding_gate="activity_floor",
        evidence_depth="at-floor",
    ),
    _decide_case(
        "momo_pullback",
        run_id=RUN_A,
        recorded=_recorded("reject", revises=1, revise_flip=True, final_ask=True),
        margin="near",
        binding_gate="symbol_holdout",
        evidence_depth="above-floor",
    ),
    _decide_case(
        "orb_fade",
        run_id=RUN_B,
        recorded=_recorded("approve", label="refused"),
        margin="comfortable",
        binding_gate="consistency",
        evidence_depth="below-floor",
    ),
    _decide_case(
        "carry_roll",
        run_id=RUN_A,
        recorded=_recorded("approve", label="promoted"),
        binding_gate="none",
        evidence_depth="at-floor",
    ),
    _decide_case(
        "squeeze_fade",
        run_id=RUN_B,
        recorded=_recorded("approve", label="promoted", revises=1, final_ask=True),
        margin="comfortable",
        binding_gate="forward_holdout",
        evidence_depth="above-floor",
    ),
)

DECIDE_BY_ID: Mapping[str, Case] = {case.case_id: case for case in DECIDE_CASES}

#: A reply nothing can pull an emitted object out of — a failed attempt carrying its error.
PROSE_REPLY = "I would rather talk this candidate through than emit a verdict for it."


def _verdict_reply(verdict: str, **fields: Any) -> str:
    """One model reply as prose wrapped around the emitted object — the JSON-in-text transport."""
    payload = {
        "verdict": verdict,
        "reason": "the panel holds up on both holdouts",
        "class_exhausted": False,
        "class_tag": "volatility_squeeze",
        **fields,
    }
    return f"Here is my verdict.\n\n```json\n{json.dumps(payload, sort_keys=True)}\n```\n"


APPROVE_REPLY = _verdict_reply("approve", holdout_symbols=["NVDA", "AMD"])
REJECT_REPLY = _verdict_reply("reject", reason="the edge is one symbol's, not the panel's")
REVISE_REPLY = _verdict_reply("revise", reason="one more sweep on the exit would settle it")


def _decide_rep(case: Case, rep: int, *replies: str) -> tuple[AnsweredCase, CaseRun]:
    """One rep of one DECIDE case, scored through the production parse the runner uses."""
    spent = AttemptOutcome(passed=False, model=DECIDE_MODEL, seconds=1.5, **DECIDE_SPEND)
    scored = [DECIDE_SCORER.score(case, reply) for reply in replies]
    answered = AnsweredCase(case=case, config_id=CONFIG_ID, rep=rep, replies=replies)
    run = CaseRun(
        result=CaseResult(
            case_id=case.case_id,
            attempts=tuple(one.attempt_outcome(spent) for one in scored),
        ),
        config_id=CONFIG_ID,
        served_models=tuple(f"{DECIDE_MODEL}-20260701" for _ in scored),
    )
    return answered, run


def _case_of(strategy: str, run_id: str) -> Case:
    return DECIDE_BY_ID[decide_case_id(strategy, run_id)]


# The live bench, rep by rep: a three-rep majority, a two-rep agreement, a single approval the
# gates never labelled, two rejections, a rep-fold that ties (unsettled), a case no rep answered
# readably (unreadable), a deferral, and a spent revise cap whose ``revise`` the binary contract
# refuses while its sibling rep still settles the case.
_DECIDE_REPS = (
    _decide_rep(_case_of("vol_breakout_squeeze", RUN_A), 1, PROSE_REPLY, APPROVE_REPLY),
    _decide_rep(_case_of("vol_breakout_squeeze", RUN_A), 2, APPROVE_REPLY),
    _decide_rep(_case_of("vol_breakout_squeeze", RUN_A), 3, REJECT_REPLY),
    _decide_rep(_case_of("gap_fade_open", RUN_A), 1, APPROVE_REPLY),
    _decide_rep(_case_of("gap_fade_open", RUN_A), 2, APPROVE_REPLY),
    _decide_rep(_case_of("range_compression", RUN_B), 1, APPROVE_REPLY),
    _decide_rep(_case_of("mean_revert_close", RUN_B), 1, REJECT_REPLY),
    _decide_rep(_case_of("mean_revert_close", RUN_B), 2, REJECT_REPLY),
    _decide_rep(_case_of("momo_pullback", RUN_A), 1, APPROVE_REPLY),
    _decide_rep(_case_of("momo_pullback", RUN_A), 2, REJECT_REPLY),
    _decide_rep(_case_of("orb_fade", RUN_B), 1, PROSE_REPLY),
    _decide_rep(_case_of("orb_fade", RUN_B), 2, PROSE_REPLY),
    _decide_rep(_case_of("carry_roll", RUN_A), 1, REVISE_REPLY),
    _decide_rep(_case_of("squeeze_fade", RUN_B), 1, REVISE_REPLY),
    _decide_rep(_case_of("squeeze_fade", RUN_B), 2, APPROVE_REPLY),
)

DECIDE_ANSWERED = tuple(answered for answered, _ in _DECIDE_REPS)
DECIDE_RUNS = tuple(run for _, run in _DECIDE_REPS)


def _decide_reading() -> Mapping[str, Any]:
    """The whole live DECIDE reading over the bench above, as a record quotes it."""
    reading = DECIDE_SCORER.read(DECIDE_ANSWERED)
    assert reading is not None
    return reading.as_dials()


def _retrospective_reading() -> Mapping[str, Any]:
    """The same corpus's *recorded* verdicts, read by the retrospective miner's own shaping."""
    outcomes = decide_outcomes(DECIDE_CASES)
    return retrospective_dials(DECIDE_CASES, outcomes, score_decide_batch(outcomes))


# ══ the two whole bench records the report goldens are rendered from ═════════════════════════

CODER_BENCH_ID = "20260801T101112Z-a1b2c3"
DECIDE_BENCH_ID = "20260801T131415Z-d4e5f6"


def _coder_taxonomy() -> FailureTaxonomy:
    """The shipped coder failure vocabulary, so the record's own metrics block classifies too."""
    taxonomy = FailureTaxonomy()
    register_coder_taxonomy(taxonomy)
    return taxonomy


def _coder_record() -> Mapping[str, Any]:
    """One whole coder bench record, built by the record module's own pure builder."""
    return build(
        BenchArtifacts(
            bench_id=CODER_BENCH_ID,
            site=SiteIdentity(
                site_id="coder",
                version="1",
                prompt_asset_groups=("author",),
                prompt_asset_hash="7c1f0d9a4b2e6f83",
            ),
            engine=ENGINE,
            corpus=CorpusIdentity(
                version="coder/2026-07", hash="9f2c1d3e4b5a6c7d", case_count=4, split="tuning"
            ),
            harness_hash="1122334455667788",
            harness_dials=_coder_reading(),
            runs=CODER_RUNS,
            label="golden-net",
            started_utc="2026-08-01T10:11:12Z",
            finished_utc="2026-08-01T10:29:44Z",
            complete=True,
            taxonomy=_coder_taxonomy(),
            configs=CONFIGS,
        )
    )


def _decide_record() -> Mapping[str, Any]:
    """One whole DECIDE bench record, built by that same pure builder."""
    return build(
        BenchArtifacts(
            bench_id=DECIDE_BENCH_ID,
            site=SiteIdentity(
                site_id="decide",
                version="1",
                prompt_asset_groups=("episodic", "briefings"),
                prompt_asset_hash="c0ffee1234567890",
            ),
            engine=ENGINE,
            corpus=CorpusIdentity(
                version="decide/2026-07", hash="deadbeefdeadbeef", case_count=9, split="tuning"
            ),
            harness_hash="8877665544332211",
            harness_dials=_decide_reading(),
            runs=DECIDE_RUNS,
            label="golden-net",
            started_utc="2026-08-01T13:14:15Z",
            finished_utc="2026-08-01T13:21:02Z",
            complete=True,
            configs=CONFIGS,
        )
    )


def _source(bench_id: str) -> str:
    """The file a report names in its header — a stated path, never this machine's."""
    return f"workspace/bench/{bench_id}/bench.json"


# ── the three dials shapes ──────────────────────────────────────────────────────────────────


def test_the_live_coder_reading_still_matches_its_committed_golden():
    """Every figure the coder scorer publishes over one live bench, key order included."""
    _pinned(CODER_DIALS_GOLDEN, _as_json(_coder_reading()))


def test_the_live_decide_reading_still_matches_its_committed_golden():
    """Every figure the DECIDE scorer publishes over one live bench, key order included."""
    _pinned(DECIDE_DIALS_GOLDEN, _as_json(_decide_reading()))


def test_the_retrospective_decide_reading_still_matches_its_committed_golden():
    """The mined corpus's recorded verdicts, read by the zero-spend retrospective path."""
    _pinned(DECIDE_RETROSPECTIVE_DIALS_GOLDEN, _as_json(_retrospective_reading()))


# ── the two rendered reports ────────────────────────────────────────────────────────────────


def test_the_coder_bench_report_still_matches_its_committed_golden():
    """The whole ``bench report`` text of a coder record carrying the live coder reading."""
    _pinned(
        CODER_REPORT_GOLDEN,
        render_bench_report(_coder_record(), source=_source(CODER_BENCH_ID)) + "\n",
    )


def test_the_decide_bench_report_still_matches_its_committed_golden():
    """The whole ``bench report`` text of a DECIDE record carrying the live DECIDE reading."""
    _pinned(
        DECIDE_REPORT_GOLDEN,
        render_bench_report(_decide_record(), source=_source(DECIDE_BENCH_ID)) + "\n",
    )


# ── the fixtures are records a reader could trust ───────────────────────────────────────────


def test_both_golden_records_are_schema_valid_bench_records():
    """A golden rendered from a document the validator would refuse would pin nothing real."""
    assert validate(_coder_record()) == []
    assert validate(_decide_record()) == []
