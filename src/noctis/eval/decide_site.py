"""The DECIDE re-run adapter — a frozen case back into the production ask, a reply back through
the production parse, and the agreement scorer the decide site declares.

A mined DECIDE case (:mod:`noctis.eval.decide_case`) is one candidate's gate-facing evidence plus
the session-ledger tail the briefing folds in beside it. This module is what turns that back into a
*question*: it reconstructs the inputs :func:`noctis.research.briefings.decide_briefing` takes, so
the declared site's own renderer (:data:`~noctis.eval.episodic_sites.DECIDE_SITE`) composes the very
prompt that session was answered from — **byte for byte**, over the same records. Nothing here
renders a briefing of its own; a second builder would drift from production in silence, and the
first prompt change afterwards would be scored against a fork nobody ships.

**Parity is achieved by answering the ask, not by copying it.** The production builder asks a
session for *facts* (:class:`~noctis.research.surface.ResearchFacts`) and folds in a session
ledger. The case froze one of those facts — the candidate's evidence block, as
:func:`noctis.research.journal.evidence_block` wrote it at mining time — so the adapter hands that
block straight back, and hands the builder a :class:`~noctis.research.ledger.SessionLedger` whose
``records()`` come off the frozen tail (every typed read on a ledger funnels through that one
method, which is what makes the substitution total). Nothing re-derives the block here: rebuilding
it from its own rows would be a second builder over a decision already made, and a second builder
drifts.

**What a case does not freeze, the bench states — as nothing, by default.** The DECIDE briefing
also carries the market cost arithmetic, the champion board, the memory tail and the library index:
session context, not case evidence, and #207 froze none of it. :data:`NEUTRAL_SESSION` answers those
facts empty, identically for every case, so a comparison between two model configurations is still
a comparison over one prompt. A bench that wants the real thing injects its own ``ResearchFacts``
instead — the adapter forwards every fact but the evidence to it untouched. Both stand-ins are
*facts* and nothing wider: a re-run reads a session, it never drives one.

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
there. The axes those strata split by are declared data — :data:`DIFFICULTY_AXES`, published by
:data:`~noctis.eval.episodic_sites.DECIDE_SITE` and handed to the pass by the runner — and the
grouping loop belongs to no site at all (:func:`~noctis.eval.reading.strata`, #306).

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
from noctis.eval.reading import (
    ANSWERS_FRESH,
    ANSWERS_KEY,
    ANSWERS_RECORDED,
    APPROVAL_PAIR,
    ATTEMPT_CALLS_KEY,
    NOT_APPLICABLE,
    RETROSPECTIVE_KEY,
    STRATA_KEY,
    fold_by_case,
    strata,
    strict_majority,
)
from noctis.eval.site import AnsweredCase
from noctis.research.driver import DECIDE_CONTRACT, DECIDE_FINAL_CONTRACT, DecideOutput
from noctis.research.episode import EmitContract, _extract_json_object
from noctis.research.ledger import SessionLedger
from noctis.research.mandate import Mandate
from noctis.research.surface import ChampionBoard, ResearchFacts, ResearchLimits

if TYPE_CHECKING:  # the cycle-closing imports, for annotations only — see the module docstring
    from noctis.eval.decide_case import RecordedOutcome
    from noctis.eval.episodic_sites import DecideSiteInput

__all__ = [
    "ANSWERS_FRESH",
    "ANSWERS_RECORDED",
    "BINDING_GATE_AXIS",
    "DECIDE_DIALS_KEY",
    "DECIDE_SCORER",
    "DIFFICULTY_AXES",
    "EVIDENCE_DEPTH_AXIS",
    "MARGIN_AXIS",
    "NEUTRAL_SESSION",
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

# ``ANSWERS_FRESH`` / ``ANSWERS_RECORDED`` — what ``dials.answers`` says on a bench that really
# asked a model, and on one that re-read history — are the eval layer's words rather than this
# site's: spelled once in :mod:`noctis.eval.reading`, and re-exported here because this module's
# docstring names them and its readers have always imported them from it (#305).

#: The three difficulty axes a mined DECIDE corpus labels its cases on, in the order every reading
#: publishes them. They live here, one module below the case builder that labels a case
#: (:mod:`noctis.eval.decide_case`) and the declaration that publishes them
#: (:data:`~noctis.eval.episodic_sites.DECIDE_SITE`'s ``difficulty_axes``), because both of those
#: import this module and neither may be imported back: the declaration carries this module's
#: scorer, and the case reads its site id off the declaration. One spelling, imported by all three,
#: is what retired the call-time axis imports this module used to close that cycle with (#306).
MARGIN_AXIS = "margin"
BINDING_GATE_AXIS = "binding_gate"
EVIDENCE_DEPTH_AXIS = "evidence_depth"
DIFFICULTY_AXES: tuple[str, ...] = (MARGIN_AXIS, BINDING_GATE_AXIS, EVIDENCE_DEPTH_AXIS)

#: The two exclusion counts a live reading carries beside the pair — the n/a side, named.
UNREADABLE_KEY = "unreadable"
UNSETTLED_KEY = "unsettled"

#: Why a case carries no verdict. Spelled once, so the row and its test read the same words.
NO_VERDICT = "no reply this case gave carried a verdict the emit contract admits"
NO_MAJORITY = "the case's answers settled on no single verdict, and none is invented for it"

# Where the frozen ledger claims to live. It never opens the path — ``records()`` is overridden —
# but a ledger carries one, and a plainly impossible path is the honest thing to carry.
_NO_SESSION_DIR = "/nonexistent/noctis-eval/frozen-decide-case"
_FROZEN_SESSION = "frozen-decide-case"


class UnreadableReply(ValueError):
    """A reply the production emit contract does not admit.

    Either no JSON object could be pulled out of it (a refusal, a truncation, prose) or the payload
    failed the contract's own parse. Carried as the attempt's error rather than raised into the
    bench: an ask that was really made and really answered badly is evidence, not an absence.
    """


# ── the session context a frozen case does not carry ─────────────────────────────────────────

#: The ceilings a bench that states no session context reports: none, on every dial. No block a
#: briefing renders reads them — the DECIDE prompt quotes the exhaustion floor, and a case freezes
#: its own — so a number here would be one session's budget restated for every case in a corpus.
_NO_LIMITS = ResearchLimits(min_trials=0, max_backtests=0, sweep_trials=0, max_author_calls=0)

# The inventory cap the facts surface declares as its default (noctis.research.surface). Restated
# only so an empty answer keeps the surface's signature; nothing here has a list to cap.
_INVENTORY_LIMIT = 60


@dataclass(frozen=True)
class NeutralSession:
    """The session context of a bench that states none: empty, and the same for every case.

    The DECIDE briefing carries four blocks a case does not freeze — the market cost arithmetic,
    the champion board, the advisory memory tail and the library index. A re-run has to render
    *something* there, and the honest something is nothing: an empty market digest, an empty board,
    an empty memory tail and an empty library. Identical across every case in a corpus, so the
    difference between two benched configurations is never the context they happened to be handed.

    It is a :class:`~noctis.research.surface.ResearchFacts` and nothing wider — every fact
    *answered*, none faked. There is no empty registry, no empty memory store and no library path
    behind these answers: a stand-in collaborator would be a second implementation of a renderer
    that already exists, and every one of them renders exactly this nothing over empty inputs. The
    facts a DECIDE briefing never reads are answered too (that is what the surface promises), which
    is why the same object can state the context of a FORMULATE or DISCOVER re-run.
    """

    # A constant on the class rather than a field: a neutral session has no state to construct,
    # and every instance of it states the same nothing.
    limits = _NO_LIMITS

    def market_context(self) -> dict[str, Any]:
        """No market digest at all — not a plausible one somebody would read as measured."""
        return {}

    def journal_evidence(self, name: str) -> dict[str, Any]:
        """No journaled evidence for anybody: a context states the blocks a case does *not*
        freeze, and a candidate's evidence is never one of them (the case answers that fact
        itself). Empty, so a context can never quietly become the source of a verdict's case."""
        return {}

    def champion_board(self) -> ChampionBoard:
        """An empty board, no crowned family, and no slot count to claim."""
        return ChampionBoard(rows=(), crowned_families=(), capacity=0)

    def library_index(self) -> list[dict]:
        """The index of a library nobody has — empty, never somebody else's real one."""
        return []

    def template_text(self) -> str:
        """No strategy template: the same answer a checkout without the seed tier gives."""
        return "(none)"

    def memory_tail(self, *, prefix_trim: bool = False) -> tuple[list, list]:
        """No findings and no dead ends, at either trim setting."""
        return [], []

    def lake_inventory(self, *, limit: int = _INVENTORY_LIMIT) -> list[str]:
        """No tickers stated as already researchable — a bench states no lake."""
        return []

    def data_budget(self) -> float | None:
        """No data spend to state: the honest ``None`` a lake with no cost preflight answers."""
        return None


#: The context :func:`decide_input` renders the un-frozen blocks from unless a bench states one.
NEUTRAL_SESSION = NeutralSession()


# ── the frozen reads: the evidence and the ledger answered from the case ─────────────────────


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
    """What the DECIDE briefing builder asks a session for: the case's own gate-facing evidence,
    and the session context for every block the case does not carry.

    One fact is the case's. :meth:`journal_evidence` answers the block the case *froze* — which is
    what :func:`noctis.research.journal.evidence_block` wrote when the case was mined — handed back
    as mined rather than rebuilt from its own rows. Re-deriving it would be a second builder over a
    decision already made, and the second one drifts: it would silently drop a field a later miner
    freezes and re-apply a ranking and a cap that were applied once, at mining time.

    Every other fact is the bench's, forwarded to its stated context untouched. The forwarding is
    named method by method rather than caught by a ``__getattr__``, because a facts surface with a
    catch-all is not a surface: a builder that grew a read would be answered by whatever the
    context happened to have, and this class is a :class:`~noctis.research.surface.ResearchFacts`
    exactly to the extent that its answers are enumerated here.
    """

    def __init__(self, evidence: Mapping[str, Any], context: ResearchFacts) -> None:
        self._evidence = dict(evidence)
        self._context = context
        # The bench's ceilings under the exhaustion floor this candidate was really judged against:
        # a re-run spends the bench's budget, but it is answering history's ask, and min_trials is
        # the one limit a rendered briefing quotes. Resolved once — a case cannot change.
        self.limits = replace(
            context.limits, min_trials=int(self._evidence.get("min_trials_gate") or 0)
        )

    def journal_evidence(self, name: str) -> dict[str, Any]:
        """The evidence block this case froze, verbatim.

        ``name`` is not looked up: a case is one candidate, so the name the builder asks with is
        the one the case was frozen for (:func:`decide_input` takes the strategy off this very
        block). A copy per call, so a rendering can never edit the ask.
        """
        return dict(self._evidence)

    def market_context(self) -> dict[str, Any]:
        return self._context.market_context()

    def champion_board(self) -> ChampionBoard:
        return self._context.champion_board()

    def library_index(self) -> list[dict]:
        return self._context.library_index()

    def template_text(self) -> str:
        return self._context.template_text()

    def memory_tail(self, *, prefix_trim: bool = False) -> tuple[list, list]:
        return self._context.memory_tail(prefix_trim=prefix_trim)

    def lake_inventory(self, *, limit: int = _INVENTORY_LIMIT) -> list[str]:
        return self._context.lake_inventory(limit=limit)

    def data_budget(self) -> float | None:
        return self._context.data_budget()


# ── the ask: one frozen case as the declared site's input ────────────────────────────────────


def decide_input(
    case: Case,
    *,
    context_window: int,
    context: ResearchFacts = NEUTRAL_SESSION,
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
        toolbox=_CaseToolbox(evidence, context),
        ledger=_FrozenLedger(tail),
        strategy=str(evidence.get("strategy") or ""),
        context_window=context_window,
        mandate=mandate,
    )


def decide_site_input(
    *,
    context_window: int,
    context: ResearchFacts = NEUTRAL_SESSION,
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
    """The co-primary value, whole: agreement is never published without the rate it cost.

    The keys and their order are :data:`~noctis.eval.reading.APPROVAL_PAIR`'s, not restated here —
    the same declaration the pair renders itself from, so a published document and a printed report
    can never name the figure two different things.
    """
    return APPROVAL_PAIR.block(approval)


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


def strata_block(
    cases: Sequence[Case],
    outcomes: Sequence[DecideOutcome],
    *,
    axes: Sequence[str] = DIFFICULTY_AXES,
) -> dict[str, Any]:
    """Each difficulty axis's levels, each scored by the same batch scorer as the whole.

    Stratified numbers are the reason the axes exist: an agreement figure that is one thing on
    near-margin cases and another on comfortable ones is two findings, not one.

    The grouping is :func:`~noctis.eval.reading.strata`'s, the one loop the coder's reading
    stratifies through as well; what stays here is DECIDE's own two contributions — how an outcome
    finds its level, and that a level is scored by the very batch scorer the headline is scored by,
    so a stratum and the figure it splits can never be computed two different ways.

    ``axes`` are the site's declared :attr:`~noctis.eval.site.AgentSite.difficulty_axes`, which a
    bench's runner hands the scoring pass; a caller that names none — the retrospective miner reads
    history this way — stratifies by the whole vocabulary a mined corpus labels its cases on.
    """
    levels = {case.case_id: dict(case.difficulty) for case in cases}
    return strata(
        axes,
        outcomes,
        level_of=lambda outcome, axis: levels.get(outcome.case_id, {}).get(axis),
        block_of=lambda grouped: scored_block(score_decide_batch(grouped)),
    )


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

    def read(
        self, answered: Sequence[AnsweredCase], *, axes: tuple[str, ...] = DIFFICULTY_AXES
    ) -> Mapping[str, Any] | None:
        """The whole DECIDE reading over one live bench's answers — see the module docstring.

        ``axes`` is what the site's declaration says its cases are labelled on, handed over by the
        runner (:class:`~noctis.eval.site.Scorer`) rather than looked up here: this scorer is the
        singleton that declaration carries, so it cannot import it back.

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
            RETROSPECTIVE_KEY: False,
            ANSWERS_KEY: ANSWERS_FRESH,
            ATTEMPT_CALLS_KEY: sum(len(one.replies) for one in answered),
            DECIDE_DIALS_KEY: {
                **scored_block(score_decide_batch(outcomes)),
                UNREADABLE_KEY: sum(1 for one in readings if one.unreadable),
                UNSETTLED_KEY: sum(1 for one in readings if one.unsettled),
                "cases": [one.row() for one in readings],
                STRATA_KEY: strata_block([one.case for one in readings], outcomes, axes=axes),
            },
        }

    def _readings(self, answered: Sequence[AnsweredCase]) -> tuple[_CaseReading, ...]:
        """Every case this bench asked, folded once, in case-id order.

        The grouping is :func:`~noctis.eval.reading.fold_by_case`'s — arithmetic every site's
        reading is entitled to — and what a folded group *means* is :meth:`_fold`'s, below.
        """
        return tuple(
            self._fold(group) for group in fold_by_case(answered, key=lambda one: one.case.case_id)
        )

    def _fold(self, jobs: Sequence[AnsweredCase]) -> _CaseReading:
        """One case's answers as the single outcome it contributes — a strict majority, or none.

        Every job's *settled* reply is scored (the runner stops retrying at a pass, so that is the
        answer the job ended on) and the verdicts that parsed are counted. A verdict held by more
        than half of them is this case's; anything else is unreadable or unsettled, and the case
        contributes nothing to a denominator it did not answer.

        The counting is :func:`~noctis.eval.reading.strict_majority`'s, because more-than-half is
        arithmetic; :data:`NO_VERDICT` and :data:`NO_MAJORITY` stay here, because what an unsettled
        case is *called* is this site's word and no other site's business.
        """
        case = jobs[0].case
        readable = [
            scored.outcome
            for scored in (self.score(case, job.settled) for job in jobs)
            if scored.outcome is not None
        ]
        if not readable:
            return _CaseReading(case=case, error=NO_VERDICT)
        settled = strict_majority(readable, key=lambda one: one.verdict)
        if settled is None:
            return _CaseReading(case=case, error=NO_MAJORITY)
        return _CaseReading(case=case, outcome=settled)

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
