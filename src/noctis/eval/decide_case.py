"""The frozen DECIDE case — one strategy's journal slice, what history recorded, and how hard it
was. Pure: records in, a case out; the module never opens the files it reads records from.

**The payload is the briefing's own evidence, not a summary of it.** A DECIDE case exists to be
re-rendered into the very prompt a past session was answered from, and the episodic driver rebuilds
that prompt *fresh from disk* through a pure function of the journal
(:func:`noctis.research.briefings.decide_briefing`). So the case freezes exactly what that builder
reads — the ranked top trials with their train/test/gap/holdout metrics, the exhaustion stats, the
thesis, the class tag, the journaled verdicts, the holdout-taint symbol set and the min-trials floor
— plus the session-ledger tail the briefing folds in beside it. The two blocks are named in
:data:`ASK_KEYS`, and :func:`ask` is the view a re-run renders from.

**The ask is cut at the decision point, on both streams.** A candidate's journal outlives the
verdict that decided it, and the record of that verdict — carrying the gates' own ``promoted`` flag
— sits in the same file as the evidence it answered. Freezing the whole slice would hand a re-run
the label through the front door: production, at the moment it asked, could not have shown a record
that did not exist yet. So the frozen ask stops **strictly before** the graded verdict's event —
journal lines by their position (:func:`_before_index`), the deciding session's lines by their
instant (:func:`_before`), the two streams meeting on the timestamps the verdict sequence is already
merged on. Everything that genuinely predates the decision stays, verdicts included: an earlier
ask's proposal was on the evidence the re-ask really read. The trial counts and the exhaustion
stats are read off that same slice, because "how much searching stood behind this verdict" is a
question about the searching that had happened.

**The label is recorded provenance, not an oracle answer.** A case carries no expected output — the
eval core refuses one by name — and the block this module writes under
:data:`OUTCOME_KEY` is not one: it is *what happened*, the disposition the promotion gates handed a
verdict that was really spent. It sits inside the payload because a case is one reviewable file and
a label in a second file is a label that goes stale in a rename; it is kept out of :func:`ask` so a
re-run is structurally incapable of being shown history's answer.

**Reconstruction is tiered by honesty, and every tier that fails says ``n/a``.**

1. An approval's journaled ``promoted`` flag *is* the agreement label — the gates' own answer,
   recorded at the time (:data:`LABEL_PROMOTED` / :data:`LABEL_REFUSED`).
2. The **binding gate** behind a refusal is recovered by replaying the pure promotion decision
   (:func:`noctis.champions.promotion.decide`) over the compact scorecard the journal records at
   arbitration time (#182). A replay is trusted only when its outcome *agrees* with the recorded
   disposition: the champion board of that moment is not always recoverable, and a replay that
   contradicts history is not evidence of anything.
3. Anything older or thinner than that leaves the axis :data:`NOT_APPLICABLE`. If even the outcome
   is missing — a candidate left undecided — there is no case at all and the builders return
   ``None``. A guessed label would prop up the one number this whole benchmark exists to report.

**Rejections are never labelled.** A rejected candidate never receives a scorecard for the gates to
arbitrate, so there is no counterfactual to grade it against; the case is still built (a rejection
is a real DECIDE outcome, and the denominators of approval rate and revise rate are made of them)
and simply carries no ``label`` key at all.

**Verdict sequences follow the production contract.** DECIDE emits approve | reject | revise;
``revise`` is capped at one corrective re-ask and an exhausted cap forces a final binary ask (#99).
A revise is never journaled as a verdict — it is a deferral, marked on the ledger's DECIDE episode
line by the driver's ``revise_cap`` check — so the sequence is merged from two streams: verdicts
from the strategy's own journal, revises from the session ledger, scoped to the strategy by the
DECIDE stage line that opened them. The **eventual** (latest) verdict is what agreement grades, and
a revision "changed the verdict" when the proposal standing as it fired — the verdict before it, or
the deferral itself when the revise opened the sequence — differs from that eventual verdict.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from noctis.backtest.scorecard import Scorecard
from noctis.champions.promotion import Decision, GateResult, PromotionRules, decide
from noctis.eval.case import Case, parse_case
from noctis.eval.episodic_sites import DECIDE_SITE
from noctis.research.driver import (
    _APPROVE,
    _EXHAUSTED,
    _FINAL,
    _REASK,
    _REJECT,
    _REVISE,
    _REVISE_CHECK,
    DECIDE,
)

# ``TOP_TRIALS`` is imported from the module that owns the evidence block, never re-declared: a
# second copy of the rendered trial cap would drift from it in silence, and the case would then
# freeze a slice the production prompt does not show.
from noctis.research.journal import TOP_TRIALS, JournalStats, Thesis, Trial
from noctis.research.ledger import ThesisLine, Verdict

__all__ = [
    "ABOVE_FLOOR",
    "ASK_KEYS",
    "AT_FLOOR",
    "BELOW_FLOOR",
    "BINDING_GATE_AXIS",
    "COMFORTABLE",
    "DIFFICULTY_AXES",
    "EVIDENCE_DEPTH_AXIS",
    "EVIDENCE_KEY",
    "LABEL_PROMOTED",
    "LABEL_REFUSED",
    "LEDGER_TAIL_KEY",
    "MARGIN_AXIS",
    "NEAR",
    "NEAR_MARGIN",
    "NOT_APPLICABLE",
    "NO_BINDING_GATE",
    "OUTCOME_KEY",
    "RecordedOutcome",
    "SITE_ID",
    "ask",
    "build_decide_case",
    "decide_case_document",
    "decide_case_id",
    "read_outcome",
    "replayed_decision",
]

# The site these cases exercise, taken from the declaration itself rather than spelled again.
SITE_ID = DECIDE_SITE.id

# The payload's blocks. The first two are the ask — exactly what the briefing builder renders from;
# the third is what history recorded happening, which a re-run must never be shown.
EVIDENCE_KEY = "evidence"
LEDGER_TAIL_KEY = "ledger_tail"
OUTCOME_KEY = "recorded_outcome"
ASK_KEYS: tuple[str, ...] = (EVIDENCE_KEY, LEDGER_TAIL_KEY)

# The three difficulty axes a mined corpus stratifies on.
MARGIN_AXIS = "margin"
BINDING_GATE_AXIS = "binding_gate"
EVIDENCE_DEPTH_AXIS = "evidence_depth"
DIFFICULTY_AXES: tuple[str, ...] = (MARGIN_AXIS, BINDING_GATE_AXIS, EVIDENCE_DEPTH_AXIS)

# The one word every axis says when its answer cannot be reconstructed. Never a plausible value.
NOT_APPLICABLE = "n/a"

# The agreement label: the gates' own disposition of an approval that was really spent.
LABEL_PROMOTED = "promoted"
LABEL_REFUSED = "refused"

# Binding-gate levels beside the gate names themselves: nothing bound a promotion.
NO_BINDING_GATE = "none"

# Margin levels, and the rule that separates them. The distance is measured in the gates' own metric
# units — the tightest ``|observed − threshold|`` across every gate that was live and measured — and
# a quarter of a unit is close on both scales the gates use (a Sharpe-shaped metric and the 0–1
# fractions the activity and consistency floors read). The number is declared here rather than
# derived from anything: it is a stratification bucket, never a gate, and no decision reads it.
NEAR, COMFORTABLE = "near", "comfortable"
NEAR_MARGIN = 0.25

# Evidence-depth levels, bucketed against the exhaustion floor itself rather than an absolute count,
# because the floor is what "enough searching" means for a run: short of it, at it, or a multiple
# past it.
BELOW_FLOOR, AT_FLOOR, ABOVE_FLOOR = "below-floor", "at-floor", "above-floor"
DEEP_EVIDENCE_MULTIPLE = 2

# A gate that could not bite says so in its note, in the promotion module's own words. Both of its
# inert forms ("switched off by a threshold of 0", "this scorecard carries no such metric") open
# with this word, and neither is a distance to anything.
_INERT = "inert"


@dataclass(frozen=True)
class RecordedOutcome:
    """What history recorded happening to one candidate — the case's label block.

    ``eventual_verdict`` is the latest verdict spent (what agreement grades); ``verdict_sequence``
    is every ordered event including the deferrals; ``revises`` counts those deferrals and
    ``final_ask`` says whether the exhausted cap forced the binary ask (#99). ``label`` is the
    gates' disposition of an approval — ``None`` for a rejection (never labelled) and for an
    approval whose record never carried the flag (never guessed).
    """

    eventual_verdict: str
    verdict_sequence: tuple[str, ...]
    revises: int
    revise_flip: bool
    final_ask: bool
    label: str | None = None

    def to_document(self) -> dict[str, Any]:
        """The YAML-safe block a case file carries. An absent label is an absent *key*: "this was
        never labelled" is a fact a reader should see spelled out, not a null to interpret."""
        document: dict[str, Any] = {
            "eventual_verdict": self.eventual_verdict,
            "verdict_sequence": list(self.verdict_sequence),
            "revises": self.revises,
            "revise_flip": self.revise_flip,
            "final_ask": self.final_ask,
        }
        if self.label is not None:
            document["label"] = self.label
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> RecordedOutcome:
        """Read the block back off a case — the same value, whichever side of YAML it comes from."""
        label = document.get("label")
        return cls(
            eventual_verdict=str(document.get("eventual_verdict", "")),
            verdict_sequence=tuple(str(kind) for kind in document.get("verdict_sequence") or ()),
            revises=int(document.get("revises") or 0),
            revise_flip=bool(document.get("revise_flip")),
            final_ask=bool(document.get("final_ask")),
            label=None if label is None else str(label),
        )


def decide_case_id(strategy: str, run_id: str) -> str:
    """The corpus-unique name for one mined episode: the candidate, and the run it was mined from.

    A strategy name is unique within a run and a run id is unique everywhere, so the pair is stable
    under re-mining — the same episode mined twice is the same file, never a duplicate case.
    """
    return f"{strategy}-{run_id}"


def build_decide_case(
    strategy: str,
    *,
    run_id: str,
    journal_records: Sequence[Mapping[str, Any]],
    ledger_records: Sequence[Mapping[str, Any]] = (),
    min_trials: int,
    rules: PromotionRules,
    champions: Sequence[Scorecard] = (),
    case_id: str | None = None,
) -> Case | None:
    """One strategy's DECIDE episode as a validated :class:`~noctis.eval.case.Case`, or ``None``.

    ``None`` means the episode has no reconstructable outcome — a candidate left undecided — and is
    excluded rather than labelled by guesswork. Everything else is data the caller read: the
    strategy's journal lines and the session ledger's, the ``min_trials`` floor that session ran
    under, the :class:`~noctis.champions.promotion.PromotionRules` its gates arbitrated with, and
    the champion board they arbitrated against.

    The document goes through the eval core's own :func:`~noctis.eval.case.parse_case`, so a case
    this module builds is admitted on exactly the terms a hand-written one is — including the
    refusal of an unmintable run id.
    """
    document = decide_case_document(
        strategy,
        run_id=run_id,
        journal_records=journal_records,
        ledger_records=ledger_records,
        min_trials=min_trials,
        rules=rules,
        champions=champions,
    )
    if document is None:
        return None
    return parse_case(
        document,
        case_id=case_id or decide_case_id(strategy, run_id),
        source=f"mined:{run_id}/{strategy}",
    )


def decide_case_document(
    strategy: str,
    *,
    run_id: str,
    journal_records: Sequence[Mapping[str, Any]],
    ledger_records: Sequence[Mapping[str, Any]] = (),
    min_trials: int,
    rules: PromotionRules,
    champions: Sequence[Scorecard] = (),
) -> dict[str, Any] | None:
    """The plain-data case document — what a corpus file holds — or ``None`` when there is no
    outcome to record. :func:`build_decide_case` is this, validated.

    The ask blocks are built from the records that existed **before** the graded verdict; the
    outcome block and the replayed arbitration are built from all of them, because the arbitration
    is what happened next and is the label rather than the ask.
    """
    records = [dict(record) for record in journal_records]
    session = [dict(record) for record in ledger_records]
    decided = _decided(strategy, records, session)
    if decided is None:
        return None
    outcome = decided.outcome
    decision = replayed_decision(records, rules=rules, champions=champions, label=outcome.label)
    asked = _before_index(records, decided.index)
    tail = _before(session, decided.at)
    stats = _stats(asked)
    return {
        "site_id": SITE_ID,
        "payload": {
            EVIDENCE_KEY: _evidence(strategy, asked, min_trials=min_trials, stats=stats),
            LEDGER_TAIL_KEY: _ledger_tail(tail),
            OUTCOME_KEY: outcome.to_document(),
        },
        "provenance": f"mined:{run_id}",
        "difficulty": {
            MARGIN_AXIS: _margin(decision),
            BINDING_GATE_AXIS: _binding_gate(decision),
            EVIDENCE_DEPTH_AXIS: _evidence_depth(stats.n_distinct_params, min_trials),
        },
    }


def read_outcome(case: Case) -> RecordedOutcome | None:
    """The recorded outcome carried by a case, or ``None`` for a case that carries no block."""
    block = case.payload.get(OUTCOME_KEY)
    return None if not isinstance(block, Mapping) else RecordedOutcome.from_document(block)


def ask(case: Case) -> dict[str, Any]:
    """Only what the site is asked — the evidence and the ledger tail, never the outcome.

    The seam a re-run renders through. Handing a model the disposition its own verdict earned would
    not measure judgment; keeping the outcome out of the ask makes that impossible rather than
    merely discouraged.
    """
    return {key: case.payload[key] for key in ASK_KEYS if key in case.payload}


def replayed_decision(
    journal_records: Sequence[Mapping[str, Any]],
    *,
    rules: PromotionRules,
    champions: Sequence[Scorecard] = (),
    label: str | None,
) -> Decision | None:
    """The gate arbitration replayed over the journaled compact scorecard — when it can be trusted.

    ``None`` when the journal holds no scorecard record (nothing to replay), when the card cannot be
    read, when there is no label to check the replay against (a rejection was never arbitrated), or
    when the replay's own verdict *disagrees* with what the gates recorded. That last refusal is the
    honest one: the champion board of the moment is not always recoverable, and a replay reaching a
    different answer is measuring a different arbitration, so nothing may be read off it.
    """
    if label is None:
        return None
    card = _scorecard(journal_records)
    if card is None:
        return None
    decision = decide(card, list(champions), rules)
    return decision if decision.promote is (label == LABEL_PROMOTED) else None


# ── the recorded outcome ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Event:
    """One ordered event in a strategy's verdict sequence, and the record behind it."""

    at: str
    rank: int
    index: int
    kind: str
    record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _Decided:
    """One decided candidate: what history recorded, and where the decision sits in that history.

    ``index`` is the graded verdict's position among the journal lines and ``at`` the instant it was
    stamped with — one coordinate per record stream, which is all the cut needs.
    """

    outcome: RecordedOutcome
    index: int
    at: str


def _decided(
    strategy: str,
    journal_records: Sequence[Mapping[str, Any]],
    ledger_records: Sequence[Mapping[str, Any]],
) -> _Decided | None:
    """The verdict sequence, its label and its position, or ``None`` when no verdict was spent."""
    revises, final_ask = _revise_events(strategy, ledger_records)
    events = sorted(
        [*revises, *_verdict_events(journal_records)],
        key=lambda event: (event.at, event.rank, event.index),
    )
    verdicts = [event for event in events if event.kind != _REVISE]
    if not verdicts:
        return None
    eventual = verdicts[-1]
    return _Decided(
        outcome=RecordedOutcome(
            eventual_verdict=eventual.kind,
            verdict_sequence=tuple(event.kind for event in events),
            revises=len(events) - len(verdicts),
            revise_flip=_revise_flip(events, eventual.kind),
            final_ask=final_ask,
            label=_label(eventual),
        ),
        index=eventual.index,
        at=eventual.at,
    )


def _verdict_events(journal_records: Sequence[Mapping[str, Any]]) -> list[_Event]:
    """The strategy's own journaled verdicts — the record of what was actually spent.

    The journal is per strategy and carries the gates' disposition, so it is the verdict source of
    truth; the ledger's verdict lines are the same events told for the whole session.
    """
    return [
        _Event(at=str(record.get("at", "")), rank=1, index=index, kind=kind, record=record)
        for index, record in enumerate(journal_records)
        if record.get("event") == "verdict"
        and (kind := str(record.get("verdict", ""))) in (_APPROVE, _REJECT)
    ]


def _revise_events(
    strategy: str, ledger_records: Sequence[Mapping[str, Any]]
) -> tuple[list[_Event], bool]:
    """This strategy's deferrals, plus whether an exhausted cap forced the final binary ask.

    An episode line names no strategy, so the scope is positional and exactly the driver's own: a
    ``stage`` line binds every episode after it to the strategy it names, until the next one. A
    revise ranks *before* a verdict sharing its instant, because a deferral is what earns the
    re-ask the verdict then lands on.
    """
    events: list[_Event] = []
    final_ask = False
    current: str | None = None
    for index, record in enumerate(ledger_records):
        event = record.get("event")
        if event == "stage":
            current = str(record["strategy"]) if record.get("strategy") else None
        elif event == "episode" and current == strategy and record.get("stage") == DECIDE:
            for check in record.get("checks") or []:
                if not isinstance(check, Mapping) or check.get("check") != _REVISE_CHECK:
                    continue
                result = check.get("result")
                if result in (_REASK, _EXHAUSTED):
                    events.append(
                        _Event(at=str(record.get("at", "")), rank=0, index=index, kind=_REVISE)
                    )
                elif result == _FINAL:
                    final_ask = True
    return events, final_ask


def _revise_flip(events: Sequence[_Event], eventual: str) -> bool:
    """Whether a revision changed the verdict: the proposal standing when the first revise fired,
    compared to the eventual one. With nothing before it, the deferral *is* the standing proposal —
    a session that came back from a revise with a real verdict changed its answer."""
    first = next((position for position, event in enumerate(events) if event.kind == _REVISE), None)
    if first is None:
        return False
    standing = events[first - 1].kind if first > 0 else _REVISE
    return standing != eventual


def _label(eventual: _Event) -> str | None:
    """The agreement label of the eventual verdict, or ``None`` when there honestly is not one.

    Approval-side only: a rejection has no gate counterfactual to be graded against, and an approval
    whose record predates the ``promoted`` flag is unlabelled rather than assumed either way.
    """
    if eventual.kind != _APPROVE or eventual.record is None:
        return None
    promoted = eventual.record.get("promoted")
    if not isinstance(promoted, bool):
        return None
    return LABEL_PROMOTED if promoted else LABEL_REFUSED


# ── the difficulty axes ──────────────────────────────────────────────────────────────────────


def _binding_gate(decision: Decision | None) -> str:
    """The named gate that refused the candidate, :data:`NO_BINDING_GATE` when none did, or
    :data:`NOT_APPLICABLE` when the arbitration could not be replayed faithfully."""
    if decision is None:
        return NOT_APPLICABLE
    failed = next((gate for gate in decision.gates if not gate.passed), None)
    return NO_BINDING_GATE if failed is None else failed.gate


def _margin(decision: Decision | None) -> str:
    """How close the evidence sat to the thresholds it was judged against.

    The tightest distance across every gate that was live *and* measured — a gate switched off by a
    zero threshold, or handed a metric the card never carried, is inert and is no distance to
    anything. :data:`NOT_APPLICABLE` when the arbitration could not be replayed, or when no gate in
    it had two numbers to compare.
    """
    if decision is None:
        return NOT_APPLICABLE
    distances = [
        abs(gate.observed - gate.threshold)
        for gate in decision.gates
        if _measured(gate) and gate.observed is not None and gate.threshold is not None
    ]
    if not distances:
        return NOT_APPLICABLE
    return NEAR if min(distances) <= NEAR_MARGIN else COMFORTABLE


def _measured(gate: GateResult) -> bool:
    """Whether a gate actually compared two numbers, rather than reporting itself inert."""
    if gate.observed is None or gate.threshold is None:
        return False
    return not (gate.note or "").startswith(_INERT)


def _evidence_depth(n_distinct_params: int, min_trials: int) -> str:
    """How much searching stood behind the verdict, relative to the exhaustion floor itself."""
    if min_trials <= 0:
        return NOT_APPLICABLE
    if n_distinct_params < min_trials:
        return BELOW_FLOOR
    if n_distinct_params < min_trials * DEEP_EVIDENCE_MULTIPLE:
        return AT_FLOOR
    return ABOVE_FLOOR


# ── the cut: what each stream held before the verdict was spent ──────────────────────────────


def _before_index(records: Sequence[Mapping[str, Any]], index: int) -> list[Mapping[str, Any]]:
    """The journal lines that stood before the graded verdict's own line.

    Position, not timestamp: a journal is append-only and per strategy, so its order *is* the order
    things happened in — and an older line that never recorded an instant still sits where it was
    written. The verdict's own record is excluded, which is the whole point of the cut.
    """
    return list(records[:index])


def _before(records: Sequence[Mapping[str, Any]], at: str) -> list[Mapping[str, Any]]:
    """The session lines stamped strictly before ``at`` — the ledger's side of the cut.

    Two streams meet only on their timestamps, the same footing the verdict sequence is merged on.
    The comparison is strict, so a line stamped at the very instant of the verdict is left out: it
    cannot be shown to predate the decision, and a tail entry that may post-date it is exactly the
    leak this cut closes. A verdict that recorded no instant at all therefore places nothing before
    itself — an empty tail, rather than one nobody can date.
    """
    return [record for record in records if str(record.get("at", "")) < at]


# ── the frozen evidence: the briefing's own reads, over records instead of files ──────────────


def _evidence(
    strategy: str,
    records: Sequence[Mapping[str, Any]],
    *,
    min_trials: int,
    stats: JournalStats,
) -> dict[str, Any]:
    """Exactly the block :func:`noctis.research.journal.evidence_block` builds, from records.

    The production builder reads a journal object; this reads the lines that journal holds, through
    the journal's own typed views, so the two produce the same mapping for the same history — the
    property ``tests/test_eval_decide_case.py`` asserts rather than assumes. ``records`` is the
    slice as of the ask (see :func:`_before_index`), so the history the two agree on is the one the
    session was really answering from.
    """
    trials = _trials_by_test(records)[:TOP_TRIALS]
    thesis = _thesis(records)
    return {
        "strategy": strategy,
        "thesis": thesis.text if thesis is not None else None,
        "class_tag": _class_tag(records),
        "n_trials": stats.n_trials,
        "n_distinct_params": stats.n_distinct_params,
        "sweep_completed": stats.sweep_completed,
        "min_trials_gate": min_trials,
        "top_trials": [
            {
                "params": trial.params,
                "symbols": trial.symbols,
                "source": trial.source,
                **({"max_bars": trial.max_bars} if trial.max_bars else {}),
                **trial.metrics,
            }
            for trial in trials
        ],
        "verdicts": [dict(record) for record in records if record.get("event") == "verdict"],
        "tuned_off_limits_for_holdout": sorted(
            {symbol for trial in _trials(records) for symbol in trial.symbols}
        ),
    }


def _ledger_tail(ledger_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The "already tried this session" narrative the briefing folds in: each journaled thesis
    paired with the verdict outcome and class-level lesson it earned — the same shape, and the same
    latest-verdict-per-strategy rule, as :func:`noctis.research.briefings._ledger_tail`.

    ``ledger_records`` is the session as of the ask (see :func:`_before`), so "the latest verdict"
    means the latest one *then* — this candidate's own verdict, spent an instant later, is not one
    of them."""
    latest: dict[str, Verdict] = {}
    for record in ledger_records:
        if record.get("event") == "verdict":
            verdict = Verdict.from_record(dict(record))
            latest[verdict.strategy] = verdict
    tail: list[dict[str, Any]] = []
    for record in ledger_records:
        if record.get("event") != "thesis":
            continue
        line = ThesisLine.from_record(dict(record))
        entry: dict[str, Any] = {"strategy": line.strategy, "thesis": line.text}
        if line.pivot_rationale:
            entry["pivot_rationale"] = line.pivot_rationale
        outcome = latest.get(line.strategy)
        if outcome is not None:
            entry["verdict"] = outcome.verdict
            if outcome.lesson:
                entry["lesson"] = outcome.lesson
        tail.append(entry)
    return tail


def _trials(records: Sequence[Mapping[str, Any]]) -> list[Trial]:
    return [Trial.from_record(dict(r)) for r in records if r.get("event") == "trial"]


def _trials_by_test(records: Sequence[Mapping[str, Any]]) -> list[Trial]:
    """Trials ranked best-first by test metric; trials with no test score rank last."""
    return sorted(
        _trials(records),
        key=lambda trial: trial.test if trial.test is not None else float("-inf"),
        reverse=True,
    )


def _stats(records: Sequence[Mapping[str, Any]]) -> JournalStats:
    """What the exhaustion gate reads: trials run, distinct param sets, and a completed sweep.

    The journal's own counting rule, over lines instead of a file — distinct param sets are the
    exhaustion gate's currency, so they are counted here exactly as it counts them.
    """
    trials = [r for r in records if r.get("event") == "trial"]
    distinct = {json.dumps(r.get("params", {}), sort_keys=True) for r in trials}
    return JournalStats(
        n_trials=len(trials),
        n_distinct_params=len(distinct),
        sweep_completed=any(r.get("event") == "sweep_complete" for r in records),
    )


def _thesis(records: Sequence[Mapping[str, Any]]) -> Thesis | None:
    latest: dict[str, Any] | None = None
    for record in records:
        if record.get("event") == "thesis" and record.get("thesis"):
            latest = dict(record)
    return Thesis.from_record(latest) if latest is not None else None


def _class_tag(records: Sequence[Mapping[str, Any]]) -> str | None:
    tag: str | None = None
    for record in records:
        if record.get("event") == "class_tag" and record.get("class_tag"):
            tag = str(record["class_tag"])
    return tag


def _scorecard(records: Sequence[Mapping[str, Any]]) -> Scorecard | None:
    """The last compact scorecard the gates arbitrated on, or ``None`` if none is readable.

    A malformed record reads as absent rather than raising: a card nobody can parse confirms
    nothing, which is the journal's own tolerant-read posture.
    """
    latest: Any = None
    for record in records:
        if record.get("event") == "scorecard" and isinstance(record.get("scorecard"), Mapping):
            latest = record["scorecard"]
    if latest is None:
        return None
    try:
        return Scorecard.from_dict(dict(latest))
    except (KeyError, TypeError, ValueError):
        return None
