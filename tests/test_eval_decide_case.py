"""The frozen DECIDE case (#207): the journal slice, the tiered label, the difficulty axes.

A DECIDE case is one strategy's gate-facing evidence, frozen as data, plus what history recorded
happening to it. Everything asserted here is external behaviour — records in, a case out — because
the module under test is pure: it is handed the journal and ledger lines a run already wrote and
never opens a file of its own (the structural test at the bottom holds it to that, in the technique
``tests/test_eval_cases.py`` uses on the case model).

Two properties carry the story. First, **the payload is the briefing's own evidence**: the case is
built to be re-rendered into the DECIDE briefing later, so the evidence block and the ledger tail
are asserted equal to what the production builders in :mod:`noctis.research.briefings` produce from
the same fixtures — a copy that drifted would score a prompt nobody ships. Second, **the label is
recorded, never guessed**: an approval's gate disposition is the label, the binding gate is
recovered by replaying the pure promotion decision over the journaled scorecard, and every
reconstruction that cannot be trusted lands as ``n/a`` rather than a plausible value.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

from noctis.eval.case import MalformedCase
from noctis.eval.case_provider import YamlCaseProvider
from noctis.eval.decide_case import (
    ABOVE_FLOOR,
    AT_FLOOR,
    BELOW_FLOOR,
    BINDING_GATE_AXIS,
    COMFORTABLE,
    EVIDENCE_DEPTH_AXIS,
    LABEL_PROMOTED,
    LABEL_REFUSED,
    MARGIN_AXIS,
    NEAR,
    NO_BINDING_GATE,
    NOT_APPLICABLE,
    RecordedOutcome,
    ask,
    build_decide_case,
    decide_case_document,
    decide_case_id,
    read_outcome,
)
from noctis.research.briefings import _decide_evidence, _ledger_tail
from noctis.research.journal import ExperimentJournal
from noctis.research.ledger import SessionLedger

from ._promotion_cases import card, rules

MODULE_SOURCE = Path(__file__).resolve().parents[1] / "src/noctis/eval/decide_case.py"

# A run id of the shape the engine mints, so mined provenance is exercised against the real pattern.
RUN_ID = "20260720T144233Z-a3f9c1"

STRATEGY = "vol_breakout_squeeze"

# The exhaustion floor the fixtures journal against, and a trial count that clears it exactly.
MIN_TRIALS = 6


# ── fixtures: real journal / ledger writers, so the records are production-shaped ────────────


def _journal(tmp_path: Path) -> ExperimentJournal:
    return ExperimentJournal(tmp_path / "state")


def _ledger(tmp_path: Path) -> SessionLedger:
    return SessionLedger(tmp_path / "state", "session-20260720T144233")


def _sweep(
    journal: ExperimentJournal,
    name: str = STRATEGY,
    *,
    trials: int = MIN_TRIALS,
    symbols: tuple[str, ...] = ("AAPL", "MSFT"),
) -> None:
    """A candidate's tuning history: its class, its thesis, and ``trials`` journaled trials."""
    journal.record_class_tag(name, "volatility_squeeze")
    journal.record_thesis(name, "volatility compresses before it expands")
    for index in range(trials):
        journal.record_trial(
            name,
            source="sweep",
            symbols=list(symbols),
            params={"lookback": 10 + index},
            window={"start": "2024-01-02", "end": "2025-01-02"},
            card=card(test=1.0 - index * 0.01, train=1.2),
        )
    journal.record_sweep_complete(name, n_trials=trials, symbols=list(symbols))


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


def _case(
    journal: ExperimentJournal,
    name: str = STRATEGY,
    *,
    ledger: SessionLedger | None = None,
    min_trials: int = MIN_TRIALS,
    champions: tuple[Any, ...] = (),
    promotion_rules: Any = None,
    run_id: str = RUN_ID,
):
    """The case built from everything the fixtures have journaled so far."""
    return build_decide_case(
        name,
        run_id=run_id,
        journal_records=journal.records(name),
        ledger_records=ledger.records() if ledger is not None else (),
        min_trials=min_trials,
        rules=promotion_rules if promotion_rules is not None else rules(),
        champions=champions,
    )


# ── hand-built records: exact timestamps, for the ordering the verdict sequence turns on ─────


def _verdict_record(at: str, verdict: str, **fields: Any) -> dict[str, Any]:
    """One journal ``verdict`` line as :class:`ExperimentJournal` writes it, at a stated instant."""
    return {"event": "verdict", "at": at, "verdict": verdict, **fields}


def _revise_record(at: str, result: str = "reask") -> dict[str, Any]:
    """One ledger DECIDE ``episode`` line carrying the driver's revise-cap check marker."""
    return {
        "event": "episode",
        "at": at,
        "stage": "decide",
        "model": "haiku",
        "tokens": 900,
        "misfires": 0,
        "outcome": "ok",
        "escalated": False,
        "checks": [{"check": "revise_cap", "result": result}],
    }


def _decide_stage_record(at: str, name: str = STRATEGY) -> dict[str, Any]:
    return {"event": "stage", "at": at, "stage": "decide", "strategy": name}


class _Toolbox:
    """The two attributes the production evidence builder reads off a research toolbox."""

    def __init__(self, journal: ExperimentJournal, min_trials: int) -> None:
        self.journal = journal
        self.min_trials = min_trials


def _outcome(journal_records: list[dict[str, Any]], ledger_records: list[dict[str, Any]]):
    """The recorded outcome of a hand-built pair of record streams."""
    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=journal_records,
        ledger_records=ledger_records,
        min_trials=MIN_TRIALS,
        rules=rules(),
    )
    assert document is not None
    return RecordedOutcome.from_document(document["payload"]["recorded_outcome"])


# ── the label: what the gates recorded, tier by tier ─────────────────────────────────────────


def test_an_approval_the_gates_promoted_is_labelled_promoted(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2))
    _approve(journal, promoted=True)

    case = _case(journal)

    assert read_outcome(case).label == LABEL_PROMOTED


def test_an_approval_the_gates_refused_is_labelled_refused(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2, symbol_holdout=-0.5))
    _approve(journal, promoted=False)

    case = _case(journal)

    assert read_outcome(case).label == LABEL_REFUSED


def test_an_approval_the_gates_refused_records_exactly_the_outcome_history_supports(tmp_path):
    """The whole block, not one field of it: everything a mined case claims about one episode."""
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2, symbol_holdout=-0.5))
    _approve(journal, promoted=False)

    case = _case(journal)

    assert read_outcome(case) == RecordedOutcome(
        eventual_verdict="approve",
        verdict_sequence=("approve",),
        revises=0,
        revise_flip=False,
        final_ask=False,
        label=LABEL_REFUSED,
    )
    assert dict(case.difficulty) == {
        MARGIN_AXIS: COMFORTABLE,
        BINDING_GATE_AXIS: "symbol_holdout",
        EVIDENCE_DEPTH_AXIS: AT_FLOOR,
    }


def test_a_rejection_is_never_labelled(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _reject(journal)

    case = _case(journal)

    outcome = read_outcome(case)
    assert outcome.eventual_verdict == "reject"
    assert outcome.label is None


def test_a_rejected_case_carries_no_label_key_at_all(tmp_path):
    """Never labelled is visible in the file: the key is absent, not a null nobody reads."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _reject(journal)

    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=journal.records(STRATEGY),
        min_trials=MIN_TRIALS,
        rules=rules(),
    )

    assert "label" not in document["payload"]["recorded_outcome"]


def test_an_approval_whose_record_never_carried_the_gate_disposition_is_left_unlabelled():
    """An older record: the outcome is known (approve), the gate's answer is not. Never guessed."""
    records = [_verdict_record("2026-07-20T14:42:33+00:00", "approve", rationale="ship it")]

    assert _outcome(records, []).label is None


def test_a_strategy_that_never_reached_a_verdict_yields_no_case(tmp_path):
    """Undecided: there is no outcome to reconstruct, so the case is excluded outright."""
    journal = _journal(tmp_path)
    _sweep(journal)

    assert _case(journal) is None


# ── the verdict sequence: ordered events, revise ordering, the flip ──────────────────────────


def test_a_single_approval_records_a_one_event_verdict_sequence(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    assert read_outcome(_case(journal)).verdict_sequence == ("approve",)


def test_a_revise_then_approve_records_both_events_in_the_order_they_happened(tmp_path):
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_stage("decide", strategy=STRATEGY)
    ledger.record_episode(
        stage="decide",
        model="haiku",
        outcome="ok",
        checks=[{"check": "revise_cap", "result": "reask"}],
    )
    _approve(journal, promoted=True)

    outcome = read_outcome(_case(journal, ledger=ledger))

    assert outcome.verdict_sequence == ("revise", "approve")
    assert outcome.revises == 1


def test_the_eventual_verdict_is_what_a_revised_case_records_as_its_verdict(tmp_path):
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_stage("decide", strategy=STRATEGY)
    ledger.record_episode(
        stage="decide",
        model="haiku",
        outcome="ok",
        checks=[{"check": "revise_cap", "result": "reask"}],
    )
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2))
    _approve(journal, promoted=True)

    outcome = read_outcome(_case(journal, ledger=ledger))

    assert outcome.eventual_verdict == "approve"
    assert outcome.label == LABEL_PROMOTED


def test_an_exhausted_revise_cap_records_both_revises_and_the_forced_final_ask(tmp_path):
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_stage("decide", strategy=STRATEGY)
    for result in ("reask", "exhausted"):
        ledger.record_episode(
            stage="decide",
            model="haiku",
            outcome="ok",
            checks=[{"check": "revise_cap", "result": result}],
        )
    ledger.record_episode(
        stage="decide",
        model="haiku",
        outcome="ok",
        checks=[{"check": "revise_cap", "result": "final"}],
    )
    _reject(journal)

    outcome = read_outcome(_case(journal, ledger=ledger))

    assert outcome.verdict_sequence == ("revise", "revise", "reject")
    assert outcome.revises == 2
    assert outcome.final_ask is True


def test_a_revise_belonging_to_another_strategy_is_not_counted_against_this_one(tmp_path):
    """Episode lines carry no strategy: the scope is the DECIDE stage line that opened them."""
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_stage("decide", strategy="some_other_candidate")
    ledger.record_episode(
        stage="decide",
        model="haiku",
        outcome="ok",
        checks=[{"check": "revise_cap", "result": "reask"}],
    )
    ledger.record_stage("decide", strategy=STRATEGY)
    _approve(journal, promoted=True)

    outcome = read_outcome(_case(journal, ledger=ledger))

    assert outcome.verdict_sequence == ("approve",)
    assert outcome.revises == 0


def test_a_verdict_reached_without_a_revise_is_not_a_flip(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    assert read_outcome(_case(journal)).revise_flip is False


def test_a_revision_that_left_the_earlier_verdict_standing_is_not_a_flip():
    records = [
        _verdict_record("2026-07-20T14:00:00+00:00", "approve", promoted=False),
        _verdict_record("2026-07-20T14:20:00+00:00", "approve", promoted=True),
    ]
    stage = _decide_stage_record("2026-07-20T13:50:00+00:00")
    revise = _revise_record("2026-07-20T14:10:00+00:00")

    outcome = _outcome(records, [stage, revise])

    assert outcome.verdict_sequence == ("approve", "revise", "approve")
    assert outcome.revise_flip is False


def test_a_revision_that_changed_the_standing_verdict_is_a_flip():
    records = [
        _verdict_record("2026-07-20T14:00:00+00:00", "approve", promoted=False),
        _verdict_record("2026-07-20T14:20:00+00:00", "reject", reason="dead end"),
    ]
    stage = _decide_stage_record("2026-07-20T13:50:00+00:00")
    revise = _revise_record("2026-07-20T14:10:00+00:00")

    outcome = _outcome(records, [stage, revise])

    assert outcome.verdict_sequence == ("approve", "revise", "reject")
    assert outcome.revise_flip is True


def test_a_revise_that_opened_the_sequence_and_ended_in_a_verdict_is_a_flip():
    """The deferral itself is the standing proposal when nothing preceded it."""
    stage = _decide_stage_record("2026-07-20T13:50:00+00:00")
    revise = _revise_record("2026-07-20T14:00:00+00:00")
    verdict = _verdict_record("2026-07-20T14:10:00+00:00", "approve", promoted=True)

    assert _outcome([verdict], [stage, revise]).revise_flip is True


# ── the binding gate: replayed, agreed with, or n/a ──────────────────────────────────────────


def test_the_binding_gate_is_recovered_by_replaying_the_promotion_decision(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2, symbol_holdout=-0.5))
    _approve(journal, promoted=False)

    case = _case(journal)

    assert case.difficulty[BINDING_GATE_AXIS] == "symbol_holdout"


def test_a_refusal_at_the_overfit_guard_names_that_gate(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=2.5))
    _approve(journal, promoted=False)

    case = _case(journal)

    assert case.difficulty[BINDING_GATE_AXIS] == "overfit_gap"


def test_an_approval_that_was_promoted_names_no_binding_gate(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2, holdout=2.0, symbol_holdout=2.0))
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.difficulty[BINDING_GATE_AXIS] == NO_BINDING_GATE


def test_without_a_journaled_scorecard_the_binding_gate_is_not_applicable(tmp_path):
    """The older record: an outcome, but nothing to replay the arbitration over."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=False)

    case = _case(journal)

    assert case.difficulty[BINDING_GATE_AXIS] == NOT_APPLICABLE


def test_a_replay_that_disagrees_with_the_recorded_outcome_yields_no_gate_axes(tmp_path):
    """The board is not always recoverable; a replay that contradicts history is not evidence."""
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=2.5))
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.difficulty[BINDING_GATE_AXIS] == NOT_APPLICABLE
    assert case.difficulty[MARGIN_AXIS] == NOT_APPLICABLE


def test_a_rejection_never_reached_the_gates_so_its_binding_gate_is_not_applicable(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2))
    _reject(journal)

    case = _case(journal)

    assert case.difficulty[BINDING_GATE_AXIS] == NOT_APPLICABLE


# ── margin and evidence depth ────────────────────────────────────────────────────────────────


def test_evidence_sitting_close_to_a_gate_threshold_buckets_as_a_near_margin(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.9, symbol_holdout=-0.5))
    _approve(journal, promoted=False)

    case = _case(journal)

    assert case.difficulty[MARGIN_AXIS] == NEAR


def test_evidence_clear_of_every_gate_threshold_buckets_as_a_comfortable_margin(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.0, holdout=2.0, symbol_holdout=2.0))
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.difficulty[MARGIN_AXIS] == COMFORTABLE


def test_trials_at_the_exhaustion_floor_bucket_as_at_floor_evidence_depth(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal, trials=MIN_TRIALS)
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.difficulty[EVIDENCE_DEPTH_AXIS] == AT_FLOOR


def test_trials_well_past_the_exhaustion_floor_bucket_as_above_floor_evidence_depth(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal, trials=MIN_TRIALS * 2)
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.difficulty[EVIDENCE_DEPTH_AXIS] == ABOVE_FLOOR


def test_trials_short_of_the_exhaustion_floor_bucket_as_below_floor_evidence_depth(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal, trials=MIN_TRIALS - 2)
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.difficulty[EVIDENCE_DEPTH_AXIS] == BELOW_FLOOR


def test_an_unknown_exhaustion_floor_leaves_evidence_depth_not_applicable(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    case = _case(journal, min_trials=0)

    assert case.difficulty[EVIDENCE_DEPTH_AXIS] == NOT_APPLICABLE


# ── the payload IS the briefing's evidence ───────────────────────────────────────────────────


def test_the_payload_carries_exactly_the_evidence_the_decide_briefing_read_at_the_ask(tmp_path):
    """A copy that drifted would benchmark a prompt nobody ships, so it is asserted equal — against
    the production builder's own reading *at the moment of the ask*, taken before the verdict that
    answered it was ever journaled."""
    journal = _journal(tmp_path)
    _sweep(journal, trials=12)
    at_the_ask = _decide_evidence(_Toolbox(journal, MIN_TRIALS), STRATEGY)
    _approve(journal, promoted=True)

    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=journal.records(STRATEGY),
        min_trials=MIN_TRIALS,
        rules=rules(),
    )

    assert document["payload"]["evidence"] == at_the_ask


def test_the_payload_carries_the_session_ledger_tail_the_briefing_folds_in(tmp_path):
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_thesis("earlier_idea", "gaps fade by lunch", pivot_rationale="the gap was small")
    ledger.record_verdict("earlier_idea", verdict="reject", lesson="gap fades are a dead end")
    ledger.record_thesis(STRATEGY, "volatility compresses before it expands")
    _approve(journal, promoted=True)

    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=journal.records(STRATEGY),
        ledger_records=ledger.records(),
        min_trials=MIN_TRIALS,
        rules=rules(),
    )

    assert document["payload"]["ledger_tail"] == _ledger_tail(ledger)


def test_the_ask_a_site_is_re_rendered_from_never_carries_the_recorded_outcome(tmp_path):
    """A re-run must not be handed history's answer; the ask view is the structural guarantee."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    case = _case(journal)

    assert set(ask(case)) == {"evidence", "ledger_tail"}


# ── the ask is cut at the decision point (#212) ──────────────────────────────────────────────


def test_the_frozen_evidence_stops_strictly_before_the_verdict_being_graded(tmp_path):
    """The record of the graded verdict did not exist when the ask was made — it *is* the answer to
    it — so a case that froze it would show a re-run the label."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    case = _case(journal)

    assert case.payload["evidence"]["verdicts"] == ()


def test_a_verdict_from_an_earlier_ask_stays_in_the_evidence_exactly_as_production_showed_it():
    """The cut is at the graded verdict, not at every verdict: the proposal a revise superseded was
    genuinely on the evidence the re-ask was answered from."""
    earlier = _verdict_record(
        "2026-07-20T14:00:00+00:00", "approve", promoted=False, rationale="ship it"
    )
    graded = _verdict_record("2026-07-20T14:20:00+00:00", "reject", reason="dead end")

    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=[earlier, graded],
        ledger_records=[
            _decide_stage_record("2026-07-20T13:50:00+00:00"),
            _revise_record("2026-07-20T14:10:00+00:00"),
        ],
        min_trials=MIN_TRIALS,
        rules=rules(),
    )

    assert document["payload"]["evidence"]["verdicts"] == [earlier]


def test_the_frozen_exhaustion_stats_count_only_the_trials_that_predate_the_verdict(tmp_path):
    """A later session kept tuning this name; the decision could not have counted those trials."""
    journal = _journal(tmp_path)
    _sweep(journal, trials=MIN_TRIALS)
    _approve(journal, promoted=True)
    _sweep(journal, trials=MIN_TRIALS * 2)

    evidence = _case(journal).payload["evidence"]

    assert evidence["n_trials"] == MIN_TRIALS
    assert evidence["n_distinct_params"] == MIN_TRIALS


def test_evidence_depth_is_bucketed_on_the_searching_that_stood_behind_the_verdict(tmp_path):
    """The axis reads the same slice the ask did: trials journaled afterwards deepen nothing."""
    journal = _journal(tmp_path)
    _sweep(journal, trials=MIN_TRIALS)
    _approve(journal, promoted=True)
    _sweep(journal, trials=MIN_TRIALS * 2)

    case = _case(journal)

    assert case.difficulty[EVIDENCE_DEPTH_AXIS] == AT_FLOOR


def test_the_frozen_ledger_tail_stops_at_the_decision_so_the_candidates_own_verdict_is_absent(
    tmp_path,
):
    """The session ledger records the spent verdict too, an instant after the journal does. At the
    moment of the ask the tail carried this candidate's thesis and no outcome beside it."""
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_thesis(STRATEGY, "volatility compresses before it expands")
    _approve(journal, promoted=True)
    ledger.record_verdict(STRATEGY, verdict="approve", lesson="the squeeze holds", promoted=True)

    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=journal.records(STRATEGY),
        ledger_records=ledger.records(),
        min_trials=MIN_TRIALS,
        rules=rules(),
    )

    assert document["payload"]["ledger_tail"] == [
        {"strategy": STRATEGY, "thesis": "volatility compresses before it expands"}
    ]


def test_cutting_the_ask_leaves_the_recorded_outcome_carrying_the_verdict_and_its_disposition(
    tmp_path,
):
    """The label side is untouched by the cut: what the ask may not see, the record still states."""
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_stage("decide", strategy=STRATEGY)
    ledger.record_episode(
        stage="decide",
        model="haiku",
        outcome="ok",
        checks=[{"check": "revise_cap", "result": "reask"}],
    )
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2))
    _approve(journal, promoted=True)

    case = _case(journal, ledger=ledger)

    assert read_outcome(case) == RecordedOutcome(
        eventual_verdict="approve",
        verdict_sequence=("revise", "approve"),
        revises=1,
        revise_flip=True,
        final_ask=False,
        label=LABEL_PROMOTED,
    )


# ── the case as a corpus artifact ────────────────────────────────────────────────────────────


def test_a_mined_case_is_provenanced_to_the_run_it_came_from(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    case = _case(journal)

    assert str(case.provenance) == f"mined:{RUN_ID}"
    assert case.site_id == "decide"
    assert case.case_id == decide_case_id(STRATEGY, RUN_ID)


def test_a_case_mined_under_a_run_id_the_engine_never_mints_is_refused(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)

    with pytest.raises(MalformedCase):
        _case(journal, run_id="yesterday")


def test_a_mined_case_round_trips_through_the_yaml_case_format(tmp_path):
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_thesis(STRATEGY, "volatility compresses before it expands")
    journal.record_scorecard(STRATEGY, card(test=1.0, train=1.2, symbol_holdout=-0.5))
    _approve(journal, promoted=False)
    built = _case(journal, ledger=ledger)
    document = decide_case_document(
        STRATEGY,
        run_id=RUN_ID,
        journal_records=journal.records(STRATEGY),
        ledger_records=ledger.records(),
        min_trials=MIN_TRIALS,
        rules=rules(),
    )
    corpus = tmp_path / "cases" / "decide"
    corpus.mkdir(parents=True)
    (corpus / f"{built.case_id}.yaml").write_text(
        yaml.safe_dump(document, sort_keys=True), encoding="utf-8"
    )

    (loaded,) = YamlCaseProvider(cases_root=tmp_path / "cases").load("decide")

    assert loaded == built
    assert read_outcome(loaded) == read_outcome(built)
    assert loaded.difficulty[BINDING_GATE_AXIS] == "symbol_holdout"


# ── the module is pure ───────────────────────────────────────────────────────────────────────


def _imports(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_decide_case_builder_reaches_no_file_and_no_clock():
    """It is handed records; the miner that reads them is somebody else's module."""
    assert _imports(MODULE_SOURCE) <= {
        "__future__",
        "collections",
        "dataclasses",
        "json",
        "typing",
        "noctis",
    }
    text = MODULE_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("open(", "Path(", "os.", "random", "datetime.now", "utcnow", "today("):
        assert forbidden not in text, forbidden
