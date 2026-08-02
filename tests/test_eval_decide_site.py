"""The DECIDE re-run adapter (#209): briefing parity, emit-vocabulary parsing, the wired scorer.

Three properties carry this story, and each is asserted as external behaviour — records in, bytes
and scored outcomes out.

**Parity.** A frozen case is re-rendered into the *production* DECIDE briefing, and the bytes are
compared against what :func:`noctis.research.briefings.decide_briefing` emits over the very journal
and ledger the case was mined from. Byte-identical or the benchmark is scoring a prompt nobody
ships, which is why the assertion is on the whole string rather than on a field of it.

**Parsing.** A canned reply is read through the driver's own emit contract objects — the primary
three-way ask, or the revise-less final ask (#99) when the case records an exhausted revise cap —
so the bench admits exactly the vocabulary production admits, and a reply neither will parse is a
failed attempt whose error survives on disk.

**Scoring and aggregation.** The scored outcomes go through :mod:`noctis.eval.decide_scorer`'s
batch arithmetic and the runner's own within-case-first metrics; nothing is re-implemented here, so
an end-to-end bench over canned replies publishes numbers a reader can check with a pencil.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from noctis.eval.case import Case
from noctis.eval.corpus import Corpus
from noctis.eval.decide_case import (
    LABEL_PROMOTED,
    LABEL_REFUSED,
    build_decide_case,
)
from noctis.eval.decide_scorer import (
    GateLabel,
    promotion_outcomes,
    score_decide_batch,
)
from noctis.eval.decide_site import (
    DECIDE_SCORER,
    NEUTRAL_SESSION,
    UnreadableReply,
    contract_for,
    decide_input,
    decide_site_input,
    parse_reply,
)
from noctis.eval.episodic_sites import DECIDE_SITE
from noctis.eval.harness import HarnessSpec
from noctis.eval.identity import SiteIdentity
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.record import EngineStamp
from noctis.eval.runner import Attempt, AttemptRequest, BenchRunner, bench_root, replay
from noctis.eval.stats import MISMATCHED_CASES, Refusal, compare
from noctis.research.briefings import decide_briefing
from noctis.research.driver import DECIDE_CONTRACT, DECIDE_FINAL_CONTRACT
from noctis.research.journal import ExperimentJournal
from noctis.research.ledger import SessionLedger

from ._promotion_cases import card, rules

# A run id of the shape the engine mints, so mined provenance is exercised against the real pattern.
RUN_ID = "20260720T144233Z-a3f9c1"

STRATEGY = "vol_breakout_squeeze"

# The exhaustion floor the fixtures journal against, and the trial count that clears it.
MIN_TRIALS = 6

# Generous enough that the briefing's build-time fit assertion never trims a block — the parity
# question is what the builder renders, not what it drops.
WINDOW = 128_000

SITE_IDENTITY = SiteIdentity(
    site_id="decide",
    version="1",
    prompt_asset_groups=("episodic", "briefings"),
    prompt_asset_hash="7c1f0d9a4b2e6f83",
)

ENGINE = EngineStamp(
    engine_version=3,
    fingerprint={"gates": "f63d47b7b9604ab1"},
    noctis_version="0.1.0",
)


# ── fixtures: the real journal / ledger writers, so the records are production-shaped ────────


def _journal(tmp_path: Path) -> ExperimentJournal:
    return ExperimentJournal(tmp_path / "state")


def _ledger(tmp_path: Path) -> SessionLedger:
    return SessionLedger(tmp_path / "state", "session-20260720T144233")


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
            max_bars=2000 if index else None,
        )
    journal.record_sweep_complete(name, n_trials=trials, symbols=["AAPL", "MSFT"])


def _approve(journal: ExperimentJournal, name: str = STRATEGY, *, promoted: bool) -> None:
    journal.record_scorecard(name, card(test=1.0, train=1.2))
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


def _final_ask_records(name: str = STRATEGY) -> list[dict[str, Any]]:
    """A session ledger whose DECIDE episode spent its revise budget — the binary ask (#99)."""
    return [
        {"event": "stage", "at": "2026-07-20T14:00:00Z", "stage": "decide", "strategy": name},
        {
            "event": "episode",
            "at": "2026-07-20T14:00:01Z",
            "stage": "decide",
            "model": "haiku",
            "tokens": 900,
            "misfires": 0,
            "outcome": "ok",
            "escalated": False,
            "checks": [{"check": "revise_cap", "result": "final"}],
        },
    ]


def _case(
    journal: ExperimentJournal,
    name: str = STRATEGY,
    *,
    ledger: SessionLedger | None = None,
    ledger_records: list[dict[str, Any]] | None = None,
) -> Case:
    """The case built from everything the fixtures have journaled so far."""
    session: list[dict[str, Any]] = []
    if ledger is not None:
        session += ledger.records()
    if ledger_records is not None:
        session += ledger_records
    built = build_decide_case(
        name,
        run_id=RUN_ID,
        journal_records=journal.records(name),
        ledger_records=session,
        min_trials=MIN_TRIALS,
        rules=rules(),
    )
    assert built is not None
    return built


def _approved_case(tmp_path: Path, name: str = STRATEGY, *, promoted: bool = True) -> Case:
    """One mined candidate the gates really disposed of — the labelled shape agreement grades."""
    journal = _journal(tmp_path / name)
    _sweep(journal, name)
    _approve(journal, name, promoted=promoted)
    return _case(journal, name)


# ── the session context a case does not freeze ───────────────────────────────────────────────


class _Board:
    """A champion board with nothing crowned on it."""

    def list(self) -> list[Any]:
        return []


class _Memory:
    """The advisory memory tail the briefing folds in."""

    def __init__(self, findings: list[str] | None = None) -> None:
        self._findings = findings or []

    def findings(self) -> list[str]:
        return list(self._findings)

    def rejected_ideas(self) -> list[str]:
        return []


class _Toolbox:
    """Everything the production DECIDE briefing builder reads off a research toolbox."""

    def __init__(
        self,
        journal: ExperimentJournal,
        min_trials: int = MIN_TRIALS,
        *,
        market: dict[str, Any] | None = None,
        memory: _Memory | None = None,
        strategies_dir: Any = None,
    ) -> None:
        self.journal = journal
        self.min_trials = min_trials
        self.registry = _Board()
        self.memory = memory or _Memory()
        self.strategies_dir = strategies_dir if strategies_dir is not None else "/nonexistent/lib"
        self._market = market or {}

    def market_context(self) -> dict[str, Any]:
        return dict(self._market)


MARKET = {
    "round_trip_bps": 4.0,
    "symbols": {"AAPL": {"character": "trending"}},
    "exhausted_classes": ["mean_reversion_1h"],
}


# ── canned replies, in the shapes a backend really answers in ────────────────────────────────


def _reply(verdict: str, **fields: Any) -> str:
    """One model reply as prose wrapped around the emitted object — the JSON-in-text transport."""
    payload = {
        "verdict": verdict,
        "reason": "the holdout numbers hold up",
        "class_exhausted": False,
        "class_tag": "volatility_squeeze",
        **fields,
    }
    return f"Here is my verdict.\n\n```json\n{json.dumps(payload)}\n```\n"


# ── briefing parity: the same records, the same bytes ────────────────────────────────────────


def test_the_briefing_rebuilt_from_a_frozen_case_is_byte_identical_to_the_production_builders(
    tmp_path,
):
    """The property the whole re-run rests on: a case re-renders the very prompt it was mined
    from, through the declared site's own renderer and the production builder behind it."""
    journal = _journal(tmp_path)
    ledger = _ledger(tmp_path)
    _sweep(journal)
    ledger.record_thesis("earlier_idea", "gaps fade by lunch", pivot_rationale="the gap was small")
    ledger.record_verdict("earlier_idea", verdict="reject", lesson="gap fades are a dead end")
    ledger.record_thesis(STRATEGY, "volatility compresses before it expands")
    _approve(journal, promoted=True)
    toolbox = _Toolbox(journal, market=MARKET, memory=_Memory(["momentum decays by lunch"]))
    case = _case(journal, ledger=ledger)

    rebuilt = DECIDE_SITE.render(
        decide_input(case, context_window=WINDOW, context=toolbox), HarnessSpec()
    )

    assert rebuilt == decide_briefing(toolbox, ledger, STRATEGY, context_window=WINDOW)


def test_the_default_session_context_renders_the_blocks_a_case_never_froze_as_honestly_empty(
    tmp_path,
):
    """A case freezes the evidence and the ledger tail; the rest is the bench's to state. Stated
    as nothing by default — an empty market, no board, no memory, no library — never as a
    fabricated one."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)
    case = _case(journal)
    empty = _Toolbox(journal)

    rebuilt = DECIDE_SITE.render(
        decide_input(case, context_window=WINDOW, context=NEUTRAL_SESSION), HarnessSpec()
    )

    assert rebuilt == decide_briefing(
        empty, SessionLedger(tmp_path / "empty", "none"), STRATEGY, context_window=WINDOW
    )


def test_the_rebuilt_briefing_carries_the_candidates_gate_facing_evidence(tmp_path):
    case = _approved_case(tmp_path)

    rebuilt = DECIDE_SITE.render(decide_input(case, context_window=WINDOW), HarnessSpec())

    assert f"EVIDENCE FOR {STRATEGY} (gate-facing)" in rebuilt
    assert '"min_trials_gate": 6' in rebuilt


def test_the_runner_seam_adapts_every_case_in_a_corpus_through_one_declared_adapter(tmp_path):
    adapt = decide_site_input(context_window=WINDOW)
    case = _approved_case(tmp_path)

    site_input = adapt(case)

    assert site_input.strategy == STRATEGY
    assert site_input.context_window == WINDOW


# ── the emit vocabulary: production's contracts, by identity ─────────────────────────────────


def test_a_reply_is_read_through_the_emit_contract_the_driver_itself_emits_through(tmp_path):
    case = _approved_case(tmp_path)

    assert contract_for(case) is DECIDE_CONTRACT


def test_a_case_that_records_an_exhausted_revise_cap_is_read_through_the_final_ask_contract(
    tmp_path,
):
    """That session was asked the binary question, so a re-run offered ``revise`` is a different
    ask — and a different ask is not a re-run at all."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)
    case = _case(journal, ledger_records=_final_ask_records())

    assert contract_for(case) is DECIDE_FINAL_CONTRACT


def test_a_canned_approval_parses_through_the_production_vocabulary_and_scores_against_the_label(
    tmp_path,
):
    case = _approved_case(tmp_path, promoted=True)

    scored = DECIDE_SCORER.score(case, _reply("approve", holdout_symbols=["NVDA"]))

    assert scored.verdict is not None
    assert scored.verdict.verdict == "approve"
    assert scored.verdict.holdout_symbols == ("NVDA",)
    assert scored.outcome == DECIDE_SCORER.outcome(case, scored.verdict)
    assert scored.outcome is not None
    assert scored.outcome.label is GateLabel.PROMOTED
    assert scored.parsed


def test_an_approval_of_a_candidate_the_gates_refused_scores_against_that_refusal(tmp_path):
    case = _approved_case(tmp_path, promoted=False)

    scored = DECIDE_SCORER.score(case, _reply("approve"))

    assert scored.outcome is not None
    assert scored.outcome.label is GateLabel.REFUSED


def test_an_approval_of_a_candidate_the_gates_never_labelled_is_an_unlabeled_approval(tmp_path):
    """A rejected candidate never received a scorecard for the gates to arbitrate, so there is no
    counterfactual to grade a re-run's approval against — and none is invented."""
    journal = _journal(tmp_path)
    _sweep(journal)
    _reject(journal)
    case = _case(journal)

    scored = DECIDE_SCORER.score(case, _reply("approve"))

    assert scored.outcome is not None
    assert scored.outcome.label is GateLabel.UNLABELED


def test_a_revise_reply_scores_as_a_deferral_that_never_reached_the_gates(tmp_path):
    case = _approved_case(tmp_path, promoted=True)

    scored = DECIDE_SCORER.score(case, _reply("revise", new_lever="try a volatility filter"))

    assert scored.outcome is not None
    assert scored.outcome.verdict == "revise"
    assert scored.outcome.revised
    assert scored.outcome.label is GateLabel.UNLABELED


def test_a_revise_is_refused_when_the_case_records_an_exhausted_revise_cap(tmp_path):
    journal = _journal(tmp_path)
    _sweep(journal)
    _approve(journal, promoted=True)
    case = _case(journal, ledger_records=_final_ask_records())

    scored = DECIDE_SCORER.score(case, _reply("revise"))

    assert not scored.parsed
    assert scored.outcome is None
    assert "revise budget is spent" in (scored.error or "")


def test_a_rejection_carries_the_gates_disposition_of_the_candidate_without_being_graded(tmp_path):
    """Agreement is approval-side: a rejected candidate is not in its population, but what the
    gates did with that candidate is recorded fact and rides along on the outcome."""
    case = _approved_case(tmp_path, promoted=True)

    scored = DECIDE_SCORER.score(case, _reply("reject"))

    assert scored.outcome is not None
    assert scored.outcome.label is GateLabel.PROMOTED
    assert promotion_outcomes([scored.outcome]) == {}


# ── an unreadable reply is a failed attempt, never a silent skip ─────────────────────────────


def test_a_reply_carrying_no_emitted_object_is_a_failed_attempt_whose_error_is_retained(tmp_path):
    case = _approved_case(tmp_path)

    scored = DECIDE_SCORER.score(case, "I would rather not decide this one.")

    assert not scored.parsed
    assert scored.outcome is None
    assert scored.error
    assert not scored.attempt_outcome().passed
    assert scored.attempt_outcome().error == scored.error


def test_a_reply_the_emit_contract_refuses_names_what_it_refused(tmp_path):
    case = _approved_case(tmp_path)

    scored = DECIDE_SCORER.score(case, _reply("maybe"))

    assert not scored.parsed
    assert "maybe" in (scored.error or "")


def test_an_unreadable_reply_raises_by_name_when_it_is_parsed_rather_than_scored(tmp_path):
    case = _approved_case(tmp_path)

    with pytest.raises(UnreadableReply):
        parse_reply(case, "no object here")


def test_a_scored_reply_keeps_the_spend_the_seam_measured_beside_its_verdict(tmp_path):
    case = _approved_case(tmp_path)

    scored = DECIDE_SCORER.score(case, _reply("approve"))
    spent = AttemptOutcome(passed=False, model="bench/oracle", seconds=2.0)
    outcome = scored.attempt_outcome(spent)

    assert outcome.passed
    assert outcome.model == "bench/oracle"
    assert outcome.seconds == 2.0


# ── the declaration ──────────────────────────────────────────────────────────────────────────


def test_the_decide_site_declares_the_agreement_scorer_in_its_scorers_slot():
    assert DECIDE_SITE.scorers == (DECIDE_SCORER,)


def test_the_recorded_gate_labels_are_exactly_the_words_the_scorers_gate_label_carries():
    """The two modules meet on one vocabulary; a drift in either is caught here, not in a bench."""
    assert {LABEL_PROMOTED, LABEL_REFUSED} == {
        GateLabel.PROMOTED.value,
        GateLabel.REFUSED.value,
    }


# ── end to end: a corpus, a stubbed seam, one record ─────────────────────────────────────────


class _CannedReplies:
    """The attempt seam a bench runs on here: one canned reply per case, scored on the way out.

    This is the production wiring in miniature — the reply is whatever the backend said, the
    scorer decides whether it parsed (the attempt's pass), and the scored outcomes are what the
    agreement arithmetic is computed over.
    """

    def __init__(
        self, replies: dict[str, str], *, per_rep: dict[tuple[str, int], str] | None = None
    ) -> None:
        self.replies = replies
        self.per_rep = per_rep or {}
        self.scored: list[Any] = []

    def __call__(self, request: AttemptRequest) -> Attempt:
        case_id = request.case.case_id
        reply = self.per_rep.get((case_id, request.rep), self.replies[case_id])
        scored = DECIDE_SCORER.score(request.case, reply)
        self.scored.append(scored)
        return Attempt(
            outcome=scored.attempt_outcome(
                AttemptOutcome(passed=False, model="bench/oracle", seconds=2.0)
            ),
            output=reply,
            served_model="bench/oracle-20260701",
        )


class _FakeClock:
    """One fixed UTC instant, ticking a second per read — a record's stamps are its contract."""

    def __init__(self) -> None:
        self._now = datetime(2026, 7, 20, 14, 42, 33, tzinfo=UTC)

    def __call__(self) -> datetime:
        moment = self._now
        self._now = datetime.fromtimestamp(self._now.timestamp() + 1, tz=UTC)
        return moment


def _runner(tmp_path: Path, attempt: _CannedReplies, **overrides: Any) -> BenchRunner:
    settings: dict[str, Any] = {
        "bench_root": bench_root(tmp_path / "workspace"),
        "attempt": attempt,
        "identity": SITE_IDENTITY,
        "engine": ENGINE,
        "clock": _FakeClock(),
        "site_input": decide_site_input(context_window=WINDOW),
    }
    settings.update(overrides)
    return BenchRunner(**settings)


def _corpus(tmp_path: Path) -> Corpus:
    """Four mined candidates: two the gates promoted, one they refused, one they never labelled."""
    cases = [
        _approved_case(tmp_path, "vol_breakout_squeeze", promoted=True),
        _approved_case(tmp_path, "gap_fade_open", promoted=False),
        _approved_case(tmp_path, "trend_follow_daily", promoted=True),
    ]
    rejected = _journal(tmp_path / "range_scalp")
    _sweep(rejected, "range_scalp")
    _reject(rejected, "range_scalp")
    cases.append(_case(rejected, "range_scalp"))
    return Corpus(site_id="decide", cases=tuple(cases))


def test_a_bench_over_frozen_decide_cases_writes_a_valid_record_and_the_agreement_its_replies_earn(
    tmp_path,
):
    """The whole path: frozen cases in, canned replies through the real runner, one bench.json —
    and an agreement figure a reader recomputes from the four replies by hand."""
    corpus = _corpus(tmp_path)
    seam = _CannedReplies(
        {
            "vol_breakout_squeeze-" + RUN_ID: _reply("approve"),
            "gap_fade_open-" + RUN_ID: _reply("approve"),
            "trend_follow_daily-" + RUN_ID: _reply("reject"),
            "range_scalp-" + RUN_ID: "I cannot tell from this evidence.",
        }
    )
    runner = _runner(tmp_path, seam)

    plan = runner.preflight(corpus)
    run = runner.run(corpus, acknowledged=plan)

    # Three of the four replies carried an emitted verdict; the fourth carried nothing to parse.
    assert run.complete
    assert run.record["metrics"]["first_attempt_pass_rate"] == 0.75
    assert (run.directory / "bench.json").is_file()

    metrics = score_decide_batch([s.outcome for s in seam.scored if s.outcome is not None])
    # Two approvals, both labelled, one of them promoted; one rejection; three decided candidates.
    assert metrics.approval.agreement == 0.5
    assert metrics.approval.approval_rate == pytest.approx(2 / 3)
    assert metrics.approval.decided == 3


def test_an_unreadable_reply_survives_in_the_bench_directory_as_a_failed_attempt_with_its_error(
    tmp_path,
):
    corpus = Corpus(site_id="decide", cases=(_approved_case(tmp_path),))
    seam = _CannedReplies({f"{STRATEGY}-{RUN_ID}": "nothing emitted here"})
    runner = _runner(tmp_path, seam)

    plan = runner.preflight(corpus)
    run = runner.run(corpus, acknowledged=plan)

    (kept,) = replay(run.directory)
    (attempt,) = kept.attempts
    assert not attempt.passed
    assert attempt.error
    assert attempt.output == "nothing emitted here"


def test_reps_of_one_case_aggregate_within_that_case_before_the_cases_average(tmp_path):
    """The eval core's own ordering rule, reached through the decide adapter: a case run twice
    cannot outvote a case run once."""
    steady = _approved_case(tmp_path, "trend_follow_daily", promoted=True)
    flaky = _approved_case(tmp_path, "gap_fade_open", promoted=True)
    corpus = Corpus(site_id="decide", cases=(steady, flaky))
    seam = _CannedReplies(
        {steady.case_id: _reply("approve"), flaky.case_id: _reply("approve")},
        per_rep={(flaky.case_id, 2): "no verdict in this one"},
    )
    runner = _runner(tmp_path, seam)

    plan = runner.preflight(corpus, reps=2)
    run = runner.run(corpus, acknowledged=plan, reps=2)

    # The flaky case answers 1 of 2; the steady one 2 of 2 — 0.75 across two equally weighted cases.
    assert run.record["metrics"]["first_attempt_pass_rate"] == 0.75


def test_two_configurations_that_approved_different_candidates_are_refused_a_comparison(tmp_path):
    """The comparison surface is :mod:`noctis.eval.stats`'s, refusals included: two configurations
    that never answered on one population get a named refusal, not a number."""
    promoted = _approved_case(tmp_path, "trend_follow_daily", promoted=True)
    refused = _approved_case(tmp_path, "gap_fade_open", promoted=False)
    bold = [
        DECIDE_SCORER.score(promoted, _reply("approve")).outcome,
        DECIDE_SCORER.score(refused, _reply("approve")).outcome,
    ]
    cautious = [
        DECIDE_SCORER.score(promoted, _reply("approve")).outcome,
        DECIDE_SCORER.score(refused, _reply("reject")).outcome,
    ]

    verdict = compare(
        promotion_outcomes([outcome for outcome in bold if outcome is not None]),
        promotion_outcomes([outcome for outcome in cautious if outcome is not None]),
        seed=7,
    )

    assert isinstance(verdict, Refusal)
    assert verdict.reason == MISMATCHED_CASES
