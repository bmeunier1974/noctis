"""The DECIDE re-run adapter — a frozen case back into the production ask, a reply back through
the production parse, and the agreement scorer the decide site declares.

A mined DECIDE case (:mod:`noctis.eval.decide_case`) is one candidate's gate-facing evidence plus
the session-ledger tail the briefing folds in beside it. This module is what turns that back into a
*question*: it reconstructs the inputs :func:`noctis.research.briefings.decide_briefing` takes, so
the declared site's own renderer (:data:`~noctis.eval.episodic_sites.DECIDE_SITE`) composes the very
prompt that session was answered from — **byte for byte**, over the same records. Nothing here
renders a briefing of its own; a second builder would drift from production in silence, and the
first prompt change afterwards would be scored against a fork nobody ships.

**Parity is achieved by reconstruction, not by copying.** The production builder reads a journal
object and a session ledger; the case holds the *lines* those objects would have answered with. So
the adapter hands the builder a journal whose reads come off the frozen evidence and a
:class:`~noctis.research.ledger.SessionLedger` whose ``records()`` come off the frozen tail — every
typed read on a ledger funnels through that one method, which is what makes the substitution total.
The builder then computes the evidence block itself, exactly as it does in a session.

**What a case does not freeze, the bench states — as nothing, by default.** The DECIDE briefing
also carries the market cost arithmetic, the champion board, the memory tail and the library index:
session context, not case evidence, and #207 froze none of it. :data:`NEUTRAL_SESSION` renders those
blocks empty, identically for every case, so a comparison between two model configurations is still
a comparison over one prompt. A bench that wants the real thing injects its own context object (any
research-toolbox-shaped stand-in) instead — the adapter forwards to it untouched.

**The reply is read through production's own vocabulary.** :func:`contract_for` returns the driver's
:data:`~noctis.research.driver.DECIDE_CONTRACT` — or the revise-less final ask
(:data:`~noctis.research.driver.DECIDE_FINAL_CONTRACT`, #99) the moment the case records an
exhausted revise cap, because that session was asked the binary question and a re-run offered
``revise`` would be a different ask. The payload is pulled out of the reply text with the episode
runner's own extractor, so the JSON-in-text transport a small backend answers on behaves here
exactly as it does in a session. A reply neither transport nor parse admits is a **failed attempt
carrying its error**, never a case quietly dropped from a denominator.

**Scoring is approval-side agreement, and the arithmetic is elsewhere.**
:class:`DecideAgreementScorer` turns one parsed verdict plus the case's recorded label into a
:class:`~noctis.eval.decide_scorer.DecideOutcome`; the batch figures come from
:func:`~noctis.eval.decide_scorer.score_decide_batch` and the within-case-first rates from
:mod:`noctis.eval.metrics`. This module adds no aggregation of its own. A re-run is **one ask**, so
the only deferral it can record is its own: ``revised`` is true exactly when the reply itself said
``revise``, and a re-run can never record a revise *flip* — history's revise count belongs to the
mined case, not to the attempt being scored.

**The same scorer is the site's declared scoring pass (#213), and its reading is the retrospective
one.** :meth:`DecideAgreementScorer.read` is what the bench runner calls once every job has
answered, and it publishes the block :func:`~noctis.eval.decide_miner.retrospective_dials`
publishes — the co-primary pair, the deferral figures, one row per case and the per-axis strata —
because the two paths share the shaping functions below (:func:`scored_block`, :func:`case_row`,
:func:`strata_block`) rather than agreeing by convention. What differs is stated at the top of the
block and nowhere else: ``answers`` is :data:`ANSWERS_FRESH` here and :data:`ANSWERS_RECORDED`
there.

Three honesty rules govern that fold:

* **One outcome per case, whatever a bench asked it.** The case is the eval core's equal-weight
  unit, so every answer one case gave — each rep, under each configuration — folds into the single
  outcome it contributes, by strict majority of the verdicts that were readable. A bench comparing
  two configurations compares two *records* (:func:`~noctis.eval.record.side_by_side`); it never
  hides two populations inside one reading.
* **A reply nobody can read decides nothing.** It is a failed attempt (the record already says so)
  and lands in :data:`UNREADABLE_KEY`, never in agreement's denominator, exactly as an unlabelled
  approval lands in ``unlabeled_approvals``.
* **A case whose answers hold no majority is unsettled, not guessed.** It is counted under
  :data:`UNSETTLED_KEY` and contributes no verdict, because inventing one would publish an answer no
  rep gave.

**Two imports are deferred to call time, and that is structural.** The DECIDE declaration names this
module's scorer in its ``scorers`` slot, and the frozen case reads its site id off that same
declaration — a declaration that carries its scorer, a scorer that reads its case, a case that names
its site. The cycle is closed at *call* time (the two imports sit inside the functions that use
them), so importing any of the three modules first works. Everything else here is a top-level
import.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from noctis.eval.case import Case
from noctis.eval.decide_scorer import (
    REVISE,
    ApprovalPair,
    DecideMetrics,
    DecideOutcome,
    GateLabel,
    score_decide_batch,
)
from noctis.eval.metrics import AttemptOutcome
from noctis.eval.site import AnsweredCase
from noctis.research.driver import DECIDE_CONTRACT, DECIDE_FINAL_CONTRACT, DecideOutput
from noctis.research.episode import EmitContract, _extract_json_object
from noctis.research.journal import JournalStats, Thesis, Trial
from noctis.research.ledger import SessionLedger
from noctis.research.mandate import Mandate

if TYPE_CHECKING:  # the cycle-closing imports, for annotations only — see the module docstring
    from noctis.eval.decide_case import RecordedOutcome
    from noctis.eval.episodic_sites import DecideSiteInput

__all__ = [
    "ANSWERS_FRESH",
    "ANSWERS_RECORDED",
    "DECIDE_DIALS_KEY",
    "DECIDE_SCORER",
    "NEUTRAL_SESSION",
    "NO_LIBRARY",
    "NO_MAJORITY",
    "NO_VERDICT",
    "UNREADABLE_KEY",
    "UNSETTLED_KEY",
    "DecideAgreementScorer",
    "NeutralSession",
    "ScoredReply",
    "UnreadableReply",
    "case_row",
    "contract_for",
    "decide_input",
    "decide_site_input",
    "pair_block",
    "parse_reply",
    "scored_block",
    "strata_block",
]

#: The key the whole DECIDE reading rides under, inside the dials subtree a record quotes verbatim.
DECIDE_DIALS_KEY = "decide"

#: What ``dials.answers`` says on a bench that really asked a model, and on one that re-read
#: history. One vocabulary, so the two records are diffable rather than merely similar.
ANSWERS_FRESH = "fresh"
ANSWERS_RECORDED = "recorded"

#: The two exclusion counts a live reading carries beside the pair — the n/a side, named.
UNREADABLE_KEY = "unreadable"
UNSETTLED_KEY = "unsettled"

#: Why a case carries no verdict. Spelled once, so the row and its test read the same words.
NO_VERDICT = "no reply this case gave carried a verdict the emit contract admits"
NO_MAJORITY = "the case's answers settled on no single verdict, and none is invented for it"

#: The strategy library a re-run reads when the bench states no session context: a path no checkout
#: has, so the index comes back empty. An *empty stated* library, never somebody else's real one.
NO_LIBRARY = "/nonexistent/noctis-eval/no-strategy-library"

# Where the frozen ledger claims to live. It never opens the path — ``records()`` is overridden —
# but a ledger carries one, and a plainly impossible path is the honest thing to carry.
_NO_SESSION_DIR = "/nonexistent/noctis-eval/frozen-decide-case"
_FROZEN_SESSION = "frozen-decide-case"

# The four keys a frozen top-trial row carries beside its metrics; everything else in the row IS a
# metric, because the production builder spreads the metrics dict flat beside exactly these.
_TRIAL_KEYS = ("params", "symbols", "source", "max_bars")


class UnreadableReply(ValueError):
    """A reply the production emit contract does not admit.

    Either no JSON object could be pulled out of it (a refusal, a truncation, prose) or the payload
    failed the contract's own parse. Carried as the attempt's error rather than raised into the
    bench: an ask that was really made and really answered badly is evidence, not an absence.
    """


# ── the session context a frozen case does not carry ─────────────────────────────────────────


class _NoChampions:
    """A champion board with nothing crowned on it."""

    def list(self) -> list[Any]:
        return []


class _NoMemory:
    """A memory tail with nothing distilled, found or ruled out."""

    def findings(self) -> list[Any]:
        return []

    def distilled(self) -> list[Any]:
        return []

    def rejected_ideas(self) -> list[Any]:
        return []


@dataclass(frozen=True)
class NeutralSession:
    """The session context of a bench that states none: empty, and the same for every case.

    The DECIDE briefing carries four blocks a case does not freeze — the market cost arithmetic,
    the champion board, the advisory memory tail and the library index. A re-run has to render
    *something* there, and the honest something is nothing: an empty market digest, an empty board,
    an empty memory and a library nobody has. Identical across every case in a corpus, so the
    difference between two benched configurations is never the context they happened to be handed.
    """

    registry: Any = _NoChampions()
    memory: Any = _NoMemory()
    strategies_dir: Any = NO_LIBRARY

    def market_context(self) -> dict[str, Any]:
        """No market digest at all — not a plausible one somebody would read as measured."""
        return {}


#: The context :func:`decide_input` renders the un-frozen blocks from unless a bench states one.
NEUTRAL_SESSION = NeutralSession()


# ── the frozen reads: a journal and a ledger answered from the case ──────────────────────────


class _FrozenJournal:
    """The journal reads :func:`noctis.research.briefings._decide_evidence` makes, answered from
    one frozen case.

    A case is one candidate, so the strategy name every read takes is the one this journal was
    frozen for. The trial list is returned in the order the case holds it — that order *is* the
    production builder's ranking, recorded when the case was mined, so re-deriving it here would be
    a second implementation of a decision already made.
    """

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        self._evidence = evidence

    def stats(self, name: str) -> JournalStats:
        return JournalStats(
            n_trials=int(self._evidence.get("n_trials") or 0),
            n_distinct_params=int(self._evidence.get("n_distinct_params") or 0),
            sweep_completed=bool(self._evidence.get("sweep_completed")),
        )

    def trials_by_test(self, name: str) -> list[Trial]:
        return [_trial(row) for row in self._evidence.get("top_trials") or []]

    def thesis(self, name: str) -> Thesis | None:
        text = self._evidence.get("thesis")
        return None if text is None else Thesis.from_record({"thesis": text})

    def class_tag(self, name: str) -> str | None:
        tag = self._evidence.get("class_tag")
        return None if tag is None else str(tag)

    def verdicts(self, name: str) -> list[dict[str, Any]]:
        return [dict(record) for record in self._evidence.get("verdicts") or []]

    def touched_symbols(self, name: str) -> set[str]:
        return {str(symbol) for symbol in self._evidence.get("tuned_off_limits_for_holdout") or []}


def _trial(row: Mapping[str, Any]) -> Trial:
    """One frozen top-trial row back as the journal's own typed view of a ``trial`` line."""
    return Trial.from_record(
        {
            "source": row.get("source", ""),
            "symbols": row.get("symbols") or [],
            "params": row.get("params") or {},
            "metrics": {key: value for key, value in row.items() if key not in _TRIAL_KEYS},
            "max_bars": row.get("max_bars"),
        }
    )


class _FrozenLedger(SessionLedger):
    """The session ledger the briefing folds in, answered from the case's frozen tail.

    Every typed read a ledger offers goes through :meth:`records`, so overriding that one method is
    the whole substitution: :func:`noctis.research.briefings._ledger_tail` walks ``theses()`` and
    ``verdicts()`` exactly as it does over a real session file, and this one opens nothing.
    """

    def __init__(self, tail: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(_NO_SESSION_DIR, _FROZEN_SESSION)
        self._frozen = _tail_records(tail)

    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._frozen]


def _tail_records(tail: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The ledger lines behind one frozen tail: a thesis line per entry, and the verdict it earned.

    The tail is the *derived* narrative — one entry per journaled thesis, carrying the latest
    verdict and lesson that thesis's strategy earned — so this reverses exactly that derivation.
    Order is the tail's own, which is what keeps "the latest verdict per strategy" meaning the same
    thing on the way back as it did on the way in.
    """
    records: list[dict[str, Any]] = []
    for entry in tail:
        thesis: dict[str, Any] = {
            "event": "thesis",
            "strategy": entry.get("strategy", ""),
            "thesis": entry.get("thesis", ""),
        }
        if entry.get("pivot_rationale"):
            thesis["pivot_rationale"] = entry["pivot_rationale"]
        records.append(thesis)
        if entry.get("verdict"):
            records.append(
                {
                    "event": "verdict",
                    "strategy": entry.get("strategy", ""),
                    "verdict": entry["verdict"],
                    "lesson": entry.get("lesson", ""),
                }
            )
    return records


class _CaseToolbox:
    """What the DECIDE briefing builder reads off a toolbox: the case's own journal and exhaustion
    floor, and the session context for every block the case does not carry.

    The delegation is deliberately open-ended. The two case-borne attributes are set here; anything
    else the builder reaches for goes to the context untouched, so a builder that grows a read
    fails loudly against a context that cannot answer it rather than being quietly re-forked here.
    """

    def __init__(self, journal: _FrozenJournal, min_trials: int, context: Any) -> None:
        self._context = context
        self.journal = journal
        self.min_trials = min_trials

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)


# ── the ask: one frozen case as the declared site's input ────────────────────────────────────


def decide_input(
    case: Case,
    *,
    context_window: int,
    context: Any = NEUTRAL_SESSION,
    mandate: Mandate | None = None,
) -> DecideSiteInput:
    """One frozen case as the record the declared DECIDE renderer takes.

    ``context_window`` is stated rather than defaulted: it is one of the site's own declared knobs
    (:class:`~noctis.eval.episodic_sites.DecideKnobs`) and it changes what the briefing builder
    trims, so a bench that did not say which window it measured under measured nothing repeatable.
    """
    # Deferred: the declaration names this module's scorer, and the case module names the
    # declaration — see the module docstring.
    from noctis.eval.decide_case import EVIDENCE_KEY, LEDGER_TAIL_KEY, ask
    from noctis.eval.episodic_sites import DecideSiteInput

    view = ask(case)
    evidence: dict[str, Any] = _thawed(view.get(EVIDENCE_KEY) or {})
    tail: list[dict[str, Any]] = _thawed(view.get(LEDGER_TAIL_KEY) or [])
    return DecideSiteInput(
        toolbox=_CaseToolbox(
            _FrozenJournal(evidence), int(evidence.get("min_trials_gate") or 0), context
        ),
        ledger=_FrozenLedger(tail),
        strategy=str(evidence.get("strategy") or ""),
        context_window=context_window,
        mandate=mandate,
    )


def decide_site_input(
    *,
    context_window: int,
    context: Any = NEUTRAL_SESSION,
    mandate: Mandate | None = None,
) -> Callable[[Case], DecideSiteInput]:
    """The adapter a bench hands :class:`~noctis.eval.runner.BenchRunner` as its ``site_input``.

    One closure over the bench-wide settings, applied per case — the runner's seam is
    ``Callable[[Case], Any]``, and an adapter is injected rather than inferred precisely so a
    benchmark can never start measuring its own guess at how a payload becomes a site's input.
    """

    def adapt(case: Case) -> DecideSiteInput:
        return decide_input(case, context_window=context_window, context=context, mandate=mandate)

    return adapt


def _thawed(value: Any) -> Any:
    """A frozen payload value back as plain data — mappings and sequences, all the way down.

    A case deep-freezes its payload (mappings become read-only proxies, lists become tuples) and
    the briefing serializes with ``default=str``, so a proxy handed to it would render as its own
    repr instead of as an object. Thawing here is what keeps the rendered bytes production's.
    """
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thawed(item) for item in value]
    return value


# ── the answer: production's emit vocabulary, and the agreement it earns ─────────────────────


def contract_for(case: Case) -> EmitContract[DecideOutput]:
    """The emit contract this case was answered through — the driver's own objects, by identity.

    The final-ask variant (#99) the moment the case records an exhausted revise cap: that session's
    emit vocabulary had already dropped ``revise``, and re-running it against the three-way ask
    would be scoring a question nobody was asked.
    """
    recorded = _recorded(case)
    return DECIDE_FINAL_CONTRACT if recorded is not None and recorded.final_ask else DECIDE_CONTRACT


def parse_reply(case: Case, reply: str | None) -> DecideOutput:
    """One model reply as the typed verdict, through production's own extraction and parse.

    The extractor is the episode runner's (:func:`noctis.research.episode._extract_json_object`) by
    identity — the JSON-in-text transport a backend that mishandles a forced tool call answers on —
    and the validation is the contract's, so the bench admits exactly the vocabulary a session
    admits. Anything else raises :class:`UnreadableReply`.
    """
    payload = _extract_json_object(reply or "")
    if payload is None:
        raise UnreadableReply(
            "the reply carries no emitted object to validate — nothing was decided here"
        )
    contract = contract_for(case)
    try:
        return contract.parse(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise UnreadableReply(f"{contract.name}: {exc}") from exc


@dataclass(frozen=True)
class ScoredReply:
    """One re-run reply, judged: the verdict it emitted, the outcome it scored, or why neither.

    ``error`` and ``outcome`` are the two sides of one coin — a reply that parsed carries an
    outcome and no error, one that did not carries an error and no outcome — because a scored
    outcome derived from a reply nobody could read would be a number with no evidence behind it.
    """

    case_id: str
    verdict: DecideOutput | None
    outcome: DecideOutcome | None
    error: str | None = None

    @property
    def parsed(self) -> bool:
        """Whether the reply was readable at all — the bench's pass for this site.

        Passing is *emitting a usable verdict*, never *agreeing with the gates*: agreement is a
        measurement of judgment, and folding it into the pass rate would turn a benchmark of
        prompt composition into a benchmark of how often a model happens to say yes.
        """
        return self.error is None

    def attempt_outcome(self, spent: AttemptOutcome | None = None) -> AttemptOutcome:
        """This reply as the eval core's attempt record, keeping whatever the seam measured.

        ``spent`` is the model, usage and duration the attempt callable observed; only ``passed``
        and ``error`` are this module's to say. The error is retained rather than dropped — the
        runner writes it beside the case, so an unreadable reply is diagnosable months later.
        """
        base = spent if spent is not None else AttemptOutcome(passed=False)
        return replace(base, passed=self.parsed, error=self.error)


# ── the reading: one shape, published by the live pass and the retrospective miner alike ────


def scored_block(metrics: DecideMetrics) -> dict[str, Any]:
    """One scored batch as record data — the pair first, never a bare agreement beside it."""
    return {
        "approval": pair_block(metrics.approval),
        "revise_rate": metrics.revise_rate,
        "revise_flip_rate": metrics.revise_flip_rate,
        "revises": metrics.revises,
        "revise_flips": metrics.revise_flips,
        "rejections": metrics.rejections,
    }


def pair_block(approval: ApprovalPair) -> dict[str, Any]:
    """The co-primary value, whole: agreement is never published without the rate it cost."""
    return {
        "agreement": approval.agreement,
        "approval_rate": approval.approval_rate,
        "decided": approval.decided,
        "approvals": approval.approvals,
        "labeled_approvals": approval.labeled_approvals,
        "unlabeled_approvals": approval.unlabeled_approvals,
        "promoted": approval.promoted,
    }


def case_row(
    case: Case,
    *,
    verdict: str | None,
    label: str | None,
    revises: int,
    revise_flip: bool,
    error: str | None = None,
) -> dict[str, Any]:
    """One case as a DECIDE reading lists it — the same keys whoever computed the verdict.

    ``error`` is an absent *key* when there is none, the way #207's outcome block spells an absent
    label: "this one was readable" is better read off a missing key than off a null to interpret.
    """
    from noctis.eval.decide_case import DIFFICULTY_AXES, NOT_APPLICABLE

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "run_id": case.provenance.mined_from,
        "verdict": verdict,
        "label": label,
        "revises": revises,
        "revise_flip": revise_flip,
        "difficulty": {axis: case.difficulty.get(axis, NOT_APPLICABLE) for axis in DIFFICULTY_AXES},
    }
    if error is not None:
        row["error"] = error
    return row


def strata_block(cases: Sequence[Case], outcomes: Sequence[DecideOutcome]) -> dict[str, Any]:
    """Each difficulty axis's levels, each scored by the same batch scorer as the whole.

    Stratified numbers are the reason the axes exist: an agreement figure that is one thing on
    near-margin cases and another on comfortable ones is two findings, not one.
    """
    from noctis.eval.decide_case import DIFFICULTY_AXES, NOT_APPLICABLE

    levels = {case.case_id: dict(case.difficulty) for case in cases}
    stratified: dict[str, Any] = {}
    for axis in DIFFICULTY_AXES:
        grouped: dict[str, list[DecideOutcome]] = {}
        for outcome in outcomes:
            level = levels.get(outcome.case_id, {}).get(axis, NOT_APPLICABLE)
            grouped.setdefault(level, []).append(outcome)
        stratified[axis] = {
            level: scored_block(score_decide_batch(grouped[level])) for level in sorted(grouped)
        }
    return stratified


@dataclass(frozen=True)
class _CaseReading:
    """One case's whole contribution to a live reading: its folded outcome, or why it has none."""

    case: Case
    outcome: DecideOutcome | None = None
    error: str | None = None

    @property
    def unreadable(self) -> bool:
        """Whether no answer this case gave was a verdict any contract admitted."""
        return self.error == NO_VERDICT

    @property
    def unsettled(self) -> bool:
        """Whether its readable answers held no majority, so no verdict is recorded for it."""
        return self.error == NO_MAJORITY

    def row(self) -> dict[str, Any]:
        """This case as the reading lists it — the retrospective row's own keys."""
        outcome = self.outcome
        return case_row(
            self.case,
            verdict=None if outcome is None else outcome.verdict,
            label=None if outcome is None or not outcome.labeled else outcome.label.value,
            revises=0 if outcome is None else int(outcome.revised),
            revise_flip=False if outcome is None else outcome.revise_flipped,
            error=self.error,
        )


@dataclass(frozen=True)
class DecideAgreementScorer:
    """The decide site's scorer: one reply, read through production's contract and graded against
    what the promotion gates recorded doing with that very candidate.

    Stateless and pure over (case, reply) — a scored batch is reproducible from the corpus and the
    retained replies alone, which is what makes a published agreement figure checkable. The same
    object is the site's declared scoring *pass* (:meth:`read`): the runner hands it every answered
    case and folds the block it returns into the record's dials.
    """

    def read(self, answered: Sequence[AnsweredCase]) -> Mapping[str, Any] | None:
        """The whole DECIDE reading over one live bench's answers — see the module docstring.

        ``None`` for a bench that answered nothing: a reading over no answers is not a measured
        zero, it is an absence, and publishing empty figures beside real dials would read as one.
        """
        if not answered:
            return None
        readings = self._readings(answered)
        outcomes = tuple(one.outcome for one in readings if one.outcome is not None)
        return {
            # The three facts that distinguish this record from a retrospective one, stated up
            # front and in that record's own vocabulary.
            "retrospective": False,
            "answers": ANSWERS_FRESH,
            "attempt_calls": sum(len(one.replies) for one in answered),
            DECIDE_DIALS_KEY: {
                **scored_block(score_decide_batch(outcomes)),
                UNREADABLE_KEY: sum(1 for one in readings if one.unreadable),
                UNSETTLED_KEY: sum(1 for one in readings if one.unsettled),
                "cases": [one.row() for one in readings],
                "strata": strata_block([one.case for one in readings], outcomes),
            },
        }

    def _readings(self, answered: Sequence[AnsweredCase]) -> tuple[_CaseReading, ...]:
        """Every case this bench asked, folded once, in case-id order."""
        grouped: dict[str, list[AnsweredCase]] = {}
        for one in answered:
            grouped.setdefault(one.case.case_id, []).append(one)
        return tuple(self._fold(grouped[case_id]) for case_id in sorted(grouped))

    def _fold(self, jobs: Sequence[AnsweredCase]) -> _CaseReading:
        """One case's answers as the single outcome it contributes — a strict majority, or none.

        Every job's *settled* reply is scored (the runner stops retrying at a pass, so that is the
        answer the job ended on) and the verdicts that parsed are counted. A verdict held by more
        than half of them is this case's; anything else is unreadable or unsettled, and the case
        contributes nothing to a denominator it did not answer.
        """
        case = jobs[0].case
        readable = [
            scored.outcome
            for scored in (self.score(case, job.settled) for job in jobs)
            if scored.outcome is not None
        ]
        if not readable:
            return _CaseReading(case=case, error=NO_VERDICT)
        verdict, held = Counter(one.verdict for one in readable).most_common(1)[0]
        if held * 2 <= len(readable):
            return _CaseReading(case=case, error=NO_MAJORITY)
        return _CaseReading(
            case=case, outcome=next(one for one in readable if one.verdict == verdict)
        )

    def score(self, case: Case, reply: str | None) -> ScoredReply:
        """One reply, parsed and scored — never raising, because a bad reply is a result."""
        try:
            verdict = parse_reply(case, reply)
        except UnreadableReply as exc:
            return ScoredReply(case_id=case.case_id, verdict=None, outcome=None, error=str(exc))
        return ScoredReply(
            case_id=case.case_id, verdict=verdict, outcome=self.outcome(case, verdict)
        )

    def outcome(self, case: Case, verdict: DecideOutput) -> DecideOutcome:
        """One parsed verdict as the scored outcome the batch arithmetic aggregates.

        A ``revise`` is a deferral: it never reached the gates, so it carries no gate label, and
        it records the revise it is. A re-run is one ask, so nothing here can have *flipped* — the
        revises history spent belong to the mined case, not to this attempt.
        """
        deferred = verdict.verdict == REVISE
        return DecideOutcome(
            case_id=case.case_id,
            verdict=verdict.verdict,
            label=GateLabel.UNLABELED if deferred else _gate_label(case),
            revised=deferred,
        )


#: The scorer the DECIDE declaration carries in its ``scorers`` slot. One instance: it is stateless,
#: and a per-bench copy would invite per-bench configuration of a thing that has none.
DECIDE_SCORER = DecideAgreementScorer()


def _gate_label(case: Case) -> GateLabel:
    """What the gates recorded about this candidate, or :attr:`GateLabel.UNLABELED`.

    The case module writes the gates' own two words and the enum carries those same two values (the
    suite pins them against each other), so a label it does not carry raises here rather than being
    read as "unlabelled" — a vocabulary drift must not quietly deflate an agreement denominator.
    """
    recorded = _recorded(case)
    if recorded is None or recorded.label is None:
        return GateLabel.UNLABELED
    return GateLabel(recorded.label)


def _recorded(case: Case) -> RecordedOutcome | None:
    """The label side of a case — deferred import, for the cycle named in the module docstring."""
    from noctis.eval.decide_case import read_outcome

    return read_outcome(case)
