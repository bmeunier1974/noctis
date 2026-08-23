"""The run record, built — a **pure** function from collected artifacts to one JSON document.

``run_tree.read_artifacts(run_dir)`` parses the record and ``run_tree.derive_evidence(run_dir,
inputs)`` reads the run's own artifacts — between them every read; this module turns the
:class:`RunArtifacts` they produce into the document ``run.json`` holds; ``schema.validate(record)``
checks it. The split is the epic's load-bearing decision, and it is worth stating why: the run
record has to be snapshot-tested against a committed golden, and — once resumption lands — a
three-segment run has to be proved identical to a one-segment run over the same work. Both tests
are trivial when the builder is a function over a value you can write by hand, and both are slow,
flaky integration tests when the builder reads the disk itself.

So **nothing here touches I/O, a clock or the configuration**. Every timestamp arrives as data
inside :class:`RunArtifacts`, already formatted by :func:`utc_iso` (which takes the moment as an
argument and never asks for it). A test enforces that structurally, by AST, rather than trusting
the convention to survive.

Two honesty rules the builder implements rather than documents:

* **``interrupted`` is observed, never guessed.** :func:`build` reports exactly the segment
  statuses it was handed; a segment killed mid-flight only becomes ``interrupted`` when the run is
  next unsealed and :func:`mark_interrupted` says so. A writer that guessed at write time would
  have to guess wrong at least once — at the moment of the crash, when nothing is there to write.
* **Every cap that bites is named.** Bulky lists are bounded by ``schema``'s caps, and a bounded
  list writes ``truncated: {"events": {"kept": N, "total": M}}``. Segments are never capped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TypeVar

from noctis.reporting import schema
from noctis.reporting.assumptions import assumptions
from noctis.reporting.metrics import Benchmark, DailySession, TradeFill, performance
from noctis.reporting.schema import (
    EVENT_CAP,
    KIND,
    PROMOTED_OUTCOME,
    STRATEGY_CAP,
    TRADE_CAP,
)

__all__ = [
    "EMBED_ALL_SOURCES_SETTING",
    "PRUNABLE_STATUSES",
    "RESEARCH_PHASE",
    "RUN_LIMIT_SETTING",
    "TERMINAL_STATUSES",
    "TRADES_COUNTER",
    "TRADING_PHASE",
    "USAGE_FIELDS",
    "Benchmark",
    "DailySession",
    "EngineIdentity",
    "RecordEvent",
    "RunArtifacts",
    "SegmentArtifact",
    "SpendEntry",
    "StrategyArtifact",
    "TradeFill",
    "build",
    "mark_interrupted",
    "prune_refusal",
    "resume_refusal",
    "seal",
    "utc_iso",
]

# The states a run never leaves. Everything else may gain another segment — including ``running``,
# which is the shape a *crash* leaves behind (the kill never got to write a stop stamp) as well as
# the shape a live engine writes. Telling those two apart is the liveness lock's job, not a status
# field's, so refusing ``running`` here would make every crashed run need manual cleanup before it
# could be resumed.
TERMINAL_STATUSES = ("completed",)

# The states whose heavy state may be pruned (story #138) — **the same tuple**, deliberately, not a
# copy that happens to agree today. Retention's whole safety argument is that a run is prunable
# exactly when it can never gain another segment: one constant means a status that stopped being
# terminal would stop being prunable in the same edit, so no window can ever open in which a run is
# both resumable and prunable. That equivalence is what makes "a pruned-then-resumed run is
# impossible" provable from the existing rules rather than guarded by a second flag.
PRUNABLE_STATUSES = TERMINAL_STATUSES

# The two phases whose seconds the run reports separately, spelled the way ``engine.machine.Phase``
# spells them. The record never imports the engine (this module reads no configuration and drives
# nothing), so the names are stated here and a segment simply carries whatever phase keys the
# process that produced it measured.
RESEARCH_PHASE = "RESEARCH"
TRADING_PHASE = "TRADING"

# The per-segment counter that says a process actually placed paper orders — the one piece of
# evidence the record carries today that a run *traded* (story #137). Named here beside the phases
# for the same reason: the record never imports the engine, so the key the engine writes is stated
# once, and ``run.traded`` is derived from it rather than latched by whoever noticed first.
TRADES_COUNTER = "trades"

# The settings knob that holds the run-level compute cap, in hours (story #136). The record does
# **not** keep a second copy of it: the cap is an ordinary frozen setting, pinned into
# ``inputs.settings.resolved`` at creation like everything else that decides what a run's results
# mean, and this section reads it back from there. One source, so the number an operator reads off
# the record is by construction the number the engine rehydrates and enforces — and "the cap is
# frozen" needs no second mechanism to be true.
RUN_LIMIT_SETTING = "run_limit_hours"

# The settings knob that says this run archives **every** candidate's source, not just its
# champions' (story #141, ``noctis run --embed-all-sources``). Frozen at creation for the same
# reason the compute cap is: it says what the run's artifact *is*, and a record whose contents
# depended on how the last segment happened to be invoked would quietly lose the sources an
# earlier one embedded. Read off ``inputs.settings.resolved`` by the store, which is the side that
# can open a file; named here beside the other frozen knobs so the record spells it once.
EMBED_ALL_SOURCES_SETTING = "embed_all_sources"

# The four neutral token fields every provider reports and every provider bills separately (story
# #140). Spelled here as the record's own field names — with their unit, like every other number —
# because the record must stay readable without the research package that produced them.
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

# The key holding a bucket's summed tokens, whatever the split's fate. A bucket always knows its
# total (the ledger journals it) even when the split behind it was never recorded.
TOTAL_TOKENS = "total_tokens"

# Every dollar figure in the record carries this suffix. The prices come from a versioned list-price
# table (``research/pricing.py``); they ignore volume discounts, batch tiers and mid-month changes,
# so a number derived from them is an estimate and says so in its own name.
USD_ESTIMATE = "usd_estimate"


# One bounded list, whatever it holds: the cap rule is the same for events and for candidates.
_T = TypeVar("_T")


def utc_iso(moment: datetime) -> str:
    """``2026-07-27T14:22:33.418Z`` — UTC ISO-8601, milliseconds, an explicit ``Z``.

    The record's one timestamp shape (schema convention). The moment is an **argument**: this
    module never reads a clock, so a caller's injected clock is the only source of time anywhere
    in the record. An aware datetime is normalized to UTC so the ``Z`` stays honest; a naive one
    is taken as UTC as-is.
    """
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class RecordEvent:
    """One notable thing that happened to the run: a lock steal, a degradation, an error.

    Deliberately flat and small — the record is evidence a website renders, not a log. Rich
    per-event detail stays in the ``--debug`` QA tree, which the record references. There is one
    event stream for the whole run and each entry names the ``segment`` it happened in, so a
    reader gets a single timeline without having to stitch per-segment lists together.
    """

    t: str | None
    kind: str  # info | warn | error
    text: str
    segment: int | None = None

    def as_dict(self) -> dict:
        return {"t": self.t, "segment": self.segment, "kind": self.kind, "text": self.text}


@dataclass(frozen=True)
class EngineIdentity:
    """What produced these numbers: the declared version plus the computed per-component digests.

    Both, because neither works alone — see ``observability/engine_id.py``. Carried on the run
    *and* on every segment, since a run resumed after a code change ran two engines: the run's is
    the engine it was **frozen at creation** under (what a resume compares against, and what its
    comparable key names), each segment's is the engine that actually produced that segment.

    ``engine_epoch`` and ``engine_changes`` are the engine's twins of ``inputs.config_epoch`` and
    ``inputs.config_changes``: 1 and empty on every run, moved only by a deliberate
    ``--allow-engine-upgrade`` (story #135), which re-freezes the identity and appends the entry
    naming the components that moved.
    """

    engine_version: int
    fingerprint: Mapping[str, str | None]
    comparable_key: str
    noctis_version: str
    engine_epoch: int = 1
    engine_changes: Sequence[Mapping[str, object]] = ()


@dataclass(frozen=True)
class SegmentArtifact:
    """One process invocation. The stop-each-morning / resume-each-night pattern makes one a night.

    Its own counters, because per-segment throughput is only comparable when it is attributed to
    the process that produced it.
    """

    index: int
    started_utc: str
    engine: EngineIdentity | None = None
    stopped_utc: str | None = None
    stopped_reason: str | None = None
    status: str = "running"
    argv: Sequence[str] = ()
    command: str = "run"
    resumed: bool = False
    counters: Mapping[str, int] = field(default_factory=dict)
    # Seconds this process spent *working* in each phase (``{"RESEARCH": 31200.0, …}``) — the
    # per-segment measurement the run's cumulative research/trading seconds are summed from.
    # ``None`` means "this segment measured nothing", which is what a record written before phase
    # timings existed says, and it is deliberately different from ``{"TRADING": 0.0}`` ("ran, and
    # never traded"). Waiting is not working: the between-phase waits (out a weekend, to a session
    # close) belong to the segment's ``duration_s`` and to no phase.
    phase_seconds: Mapping[str, float] | None = None
    # The machine this process ran on, and the inputs it resolved: hardware, OS, python, git
    # state, lockfile digest, extras present, degraded seams (story #139). Captured by the store
    # through the injected probes of ``observability.environment`` and carried here as data, like
    # every other value in this module. **Per segment, not per run**: a run may migrate machines
    # mid-experiment, and research throughput is CPU-bound — one night's trials-per-hour must
    # never be attributed to another night's hardware. ``None`` means "this segment measured
    # nothing", which is what a record written before environments existed says, and it is
    # deliberately different from a block whose values are individually null (a machine that
    # *was* measured and could not answer).
    environment: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SpendEntry:
    """One journaled model judgment's spend — the atom every spend roll-up is summed from.

    One per ``episode`` line in the run's own session ledgers, read (never counted) by
    ``run_tree.read_spend`` and already priced there, because pricing needs the table and the
    table comes from the run's frozen configuration — which is I/O. What is left here is
    arithmetic: sums by model, by stage and by segment, and the efficiency ratios over them.

    ``at`` is what attributes an entry to a segment: the ledgers and the segments share one UTC
    wall clock, so the segment whose window contains the stamp is the process that spent it.
    ``usage`` is ``None`` when the line journaled no split (a ledger written before #140) —
    unknown, never zero — while ``tokens`` stays known either way. ``usd_estimate`` is ``None``
    for a model the price table does not carry.
    """

    at: str | None
    stage: str
    model: str
    tokens: int
    usage: Mapping[str, int] | None = None
    usd_estimate: float | None = None


@dataclass(frozen=True)
class StrategyArtifact:
    """One candidate this run considered, and how it fared — champion or not (story #141).

    **Every** candidate, because the rejections are the product: "47 of 66 candidates died at the
    symbol-holdout gate" is the sentence that makes this project's results credible, and it is
    computable only when the ones that failed are on the record beside the ones that did not.

    ``gates`` is the structured evidence ``champions.promotion.decide`` produced, carried through
    the champion board that journaled it. A rejection short-circuits, so it is the gates *reached*
    plus the one that failed — an absent gate means "never reached", never "passed".

    ``source`` is the file's full text for a champion and ``None`` for everyone else, with
    ``source_path`` (relative to the run directory) plus ``source_sha256`` always naming the
    artifact. That is the whole size policy: a fortnight's record stays in the hundreds of
    kilobytes instead of the megabytes an all-sources record would weigh, and an experiment worth
    archiving whole says so at creation (:data:`EMBED_ALL_SOURCES_SETTING`). The choice of *which*
    to embed is made where the files are read (``run_tree.read_strategies``); this module renders
    what it was handed.
    """

    name: str
    # promoted | rejected | undecided — the last verdict the run's champion board journaled for
    # this name. ``undecided`` is a real and common state: a candidate authored, maybe tuned, and
    # never carried to a verdict is evidence of the funnel's width, not a gap in the record.
    outcome: str
    tier: str | None = None
    decided_utc: str | None = None
    trials: int | None = None
    gates: Sequence[Mapping[str, object]] = ()
    rationale: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class RunArtifacts:
    """Everything the run tree and this engine know about one run — the builder's only input.

    ``complete`` is false whenever a segment is open, the process was killed, or the writer
    latched off after a failure: a partial record must never pass for a whole one.
    """

    run_id: str
    created_utc: str | None
    last_active_utc: str | None
    # The run's own engine, frozen at creation and carried forward verbatim ever after — the side a
    # resume compares against (story #135), exactly as ``inputs`` is for the configuration.
    engine: EngineIdentity
    # The engine **this process** is: the one the appending segment records, and the one an
    # accepted upgrade re-freezes the run onto. ``None`` on a fresh run, where the two are the same
    # thing, and on any caller that only cares about the run's identity.
    current_engine: EngineIdentity | None = None
    segments: Sequence[SegmentArtifact] = ()
    label: str | None = None
    completed_utc: str | None = None
    complete: bool = False
    events: Sequence[RecordEvent] = ()
    errors: Sequence[RecordEvent] = ()
    # The run's own configuration, frozen at creation and carried verbatim ever after: the
    # resolved settings the frozen tier is restored from, the mandate as resolved *text* plus its
    # applied overlay, and the tier lists that say which is which. Shaped by
    # ``noctis.config.rehydrate.freeze_inputs`` and round-tripped through here untouched — this
    # module stays free of configuration, so the freezing policy lives in exactly one place and
    # the builder cannot quietly reinterpret it. ``None`` for a run that never froze one (an
    # adopted history, story #131).
    inputs: Mapping[str, object] | None = None
    # Trials this run has journaled, counted at read time off its **own** experiment journals
    # (``run_tree.read_trials``) — the very lines the exhaustion gate counts, never a counter
    # incremented in memory. It is therefore cumulative across every segment by construction,
    # including the research-only ones a standalone ``noctis research --resume`` appends (story
    # #137), and it is the multiple-testing count a deflated Sharpe will need. ``None`` when the
    # run journaled nothing at all — a run that has not researched yet has no count to report, and
    # a zero would be a claim about a journal that does not exist.
    trials: int | None = None
    # Whether retention has removed this run's heavy directories (story #138). ``False`` on every
    # run until an operator prunes one deliberately — retention is opt-in, and nothing sets this by
    # accident. Once ``True`` it is carried forward verbatim, like ``inputs``: it is a statement
    # about the tree beside the record ("``state/``, ``strategies/`` and ``reports/`` are gone, so
    # the path-plus-hash references into them no longer resolve"), and re-deriving it from whatever
    # happens to be on disk would let a half-restored directory quietly make it false again.
    state_pruned: bool = False
    # Every judgment episode this run journaled, priced (story #140) — read at write time off the
    # run's own session ledgers (``run_tree.read_spend``), exactly like ``trials`` is read off its
    # experiment journals. ``None`` when the run journaled no ledger at all, which is what a run
    # with no LLM key looks like: it reports a ``null`` bill rather than a $0 one. An **empty**
    # sequence is different and deliberate — a session that ran and spent nothing.
    spend: Sequence[SpendEntry] | None = None
    # Which price table produced the estimates on those entries — the shipped
    # ``<month>[.<revision>]`` label (``2026-07.1``), or a ``<version>+custom.<digest>`` one when an
    # operator overrode it in config. Carried into the record so a later reader knows which prices
    # are behind the numbers; a record keeps the label that was in force when it was written, so it
    # is never migrated when the table gains a price or a prefix.
    pricing_table_version: str | None = None
    # Champions this run currently holds, read off its own champion board at write time — the
    # denominator of the two per-champion efficiency numbers. ``None`` when the board could not be
    # read at all (a pruned or absent state dir), which is not the same as an honest zero.
    champions: int | None = None
    # Every candidate this run considered, read at write time off its own champion board, its
    # experiment journals and its strategy tiers (``run_tree.read_strategies``) — derived like
    # every other cumulative fact, never a list carried across a restart. An empty sequence is a
    # run that has considered nothing yet, which is a statement a young run can honestly make.
    strategies: Sequence[StrategyArtifact] = ()
    # Every session this run's paper account has closed (story #142), read at write time off the
    # run's own account ledger (``run_tree.read_sessions``) — the daily marks the equity curve is
    # **derived** from and the fills the trade log is. Never carried across a restart: a resumed
    # segment re-reads the whole ledger, so three short nights publish exactly the curve one long
    # night would, and a rewrite after a crash cannot double a day.
    sessions: Sequence[DailySession] = ()
    # The equal-weight buy-and-hold comparison over the names this run traded, priced from bars
    # **already in the shared lake** (``run_tree.read_benchmark``) — no vendor call, no new spend.
    # ``None`` for a run nobody computed one for; a benchmark the lake could not price arrives as a
    # :class:`~noctis.reporting.metrics.Benchmark` carrying no points and a note saying why.
    benchmark: Benchmark | None = None


def mark_interrupted(artifacts: RunArtifacts) -> RunArtifacts:
    """Return the artifacts with every unclosed segment marked ``interrupted``.

    Called when a run is **opened**, which is the only moment the observation can honestly be
    made: a segment carrying a start stamp and no stop stamp belongs to a process that is no
    longer here. Pure and idempotent — a cleanly closed segment is untouched.
    """
    segments = tuple(
        segment if segment.stopped_utc is not None else _replace_status(segment, "interrupted")
        for segment in artifacts.segments
    )
    return replace(artifacts, segments=segments)


def seal(artifacts: RunArtifacts, *, at: str) -> RunArtifacts:
    """Return the artifacts marked ``completed`` — terminal — as of ``at`` (``--finish``).

    Pure and **idempotent**: a run that is already sealed keeps the stamp it was sealed with, so
    finishing twice is the documented no-op rather than a rewritten history. Sealing is deliberate
    here; the run-level cap needs no equivalent, because a breach is derived from the segments at
    every write (:func:`_capped_at`).
    """
    if artifacts.completed_utc is not None:
        return artifacts
    return replace(artifacts, completed_utc=at)


def resume_refusal(record: Mapping[str, object]) -> str | None:
    """Why this record may not gain another segment, or ``None`` when it may.

    Pure, and read off the record's own **derived** status rather than a second stored flag, so
    the refusal can never disagree with what the record says about itself. ``completed`` is the
    one terminal state (:data:`TERMINAL_STATUSES`): a run sealed deliberately (or by a run-level
    cap, story #136) is a published result, and a published result that could silently gain
    segments would make every number quoted from it provisional. ``stopped``, ``interrupted`` and
    ``running`` all resume — see :data:`TERMINAL_STATUSES` for why the last one is not this
    function's business.
    """
    section = record.get("run")
    run: Mapping[str, object] = section if isinstance(section, Mapping) else {}
    if run.get("status") not in TERMINAL_STATUSES:
        return None
    return (
        f"this run is completed{_why_completed(run)} — a terminal state, so it can never gain "
        "another segment. Start a new run instead (identity is minted, never derived, so a fresh "
        "run under the same configuration is one command away)."
    )


def prune_refusal(record: Mapping[str, object]) -> str | None:
    """Why this run's heavy state may not be pruned, or ``None`` when it may (story #138).

    Pure, and the **exact complement** of :func:`resume_refusal` by construction: pruning is
    permitted precisely when a resume is refused. That is the whole safety argument — the
    directories retention removes (``state/``, ``strategies/``, ``reports/``) are the ones a resume
    would read, so a run that can still gain a segment must keep them or the record's central
    promise quietly stops being true. A record whose status cannot be read at all is *not* prunable:
    unreadable evidence is not evidence that deleting is safe.

    Nothing here deletes, decides on a lock, or touches a path — the store checks this before it
    touches the disk, and the CLI shows the sentence it returns.
    """
    if resume_refusal(record) is not None:
        return None
    section = record.get("run")
    run: Mapping[str, object] = section if isinstance(section, Mapping) else {}
    status = run.get("status")
    named = str(status) if isinstance(status, str) else "of an unreadable status"
    return (
        f"this run is {named}, so pruning its state/, strategies/ and reports/ directories would "
        "silently destroy its resumability — the one thing the run record promises. Only a "
        "completed run may be pruned (`noctis run --resume <address> --finish` seals one "
        "deliberately, and completed is terminal, so nothing that could still be continued is "
        "ever at risk)."
    )


def _why_completed(run: Mapping[str, object]) -> str:
    """The half-sentence that says *how* a run was sealed, when the record can tell.

    A refusal an operator cannot act on is a wall: "completed" alone leaves them wondering whether
    they typed the wrong id. A run that hit its own compute cap says so, with both numbers, because
    the answer to "why can't I resume it?" is then simply "because you asked for exactly this".
    """
    limit = run.get("run_limit_hours")
    runtime = run.get("cumulative_runtime_s")
    if isinstance(limit, bool) or not isinstance(limit, int | float):
        return ""
    if not isinstance(runtime, int | float) or isinstance(runtime, bool):
        return f" (it was given a {_hours(limit)} run limit)"
    if runtime < limit * 3600.0:
        return ""
    return f" — it reached its {_hours(limit)} run limit, having run for {_hours(runtime / 3600.0)}"


def _hours(value: float) -> str:
    return f"{value:g}h"


def build(artifacts: RunArtifacts) -> dict:
    """Assemble the run record. Pure: same artifacts in, byte-identical document out."""
    events, events_note = _capped(artifacts.events, EVENT_CAP)
    errors, errors_note = _capped(artifacts.errors, EVENT_CAP)
    strategies, strategies_note = _capped(_ordered(artifacts.strategies), STRATEGY_CAP)
    sessions, trades_note = _sessions(artifacts.sessions)
    truncated: dict[str, dict[str, int]] = {}
    if events_note is not None:
        truncated["events"] = events_note
    if errors_note is not None:
        truncated["errors"] = errors_note
    if strategies_note is not None:
        truncated["strategies"] = strategies_note
    if trades_note is not None:
        truncated["trades"] = trades_note

    segments = [_segment(segment) for segment in artifacts.segments]
    runtime_s = _cumulative_runtime_s(segments)
    limit_hours = _run_limit_hours(artifacts.inputs)
    completed_utc = artifacts.completed_utc or _capped_at(segments, limit_hours)
    return {
        # Read off the module rather than bound at import, so a record written by an engine whose
        # schema has moved carries the version that engine actually writes (see
        # ``schema.upgrade``): the upgrade path is exercised end to end instead of only the day
        # someone bumps the constant.
        "schema_version": schema.SCHEMA_VERSION,
        "kind": KIND,
        "run": {
            "run_id": artifacts.run_id,
            "label": artifacts.label,
            "status": _status(artifacts, completed_utc=completed_utc),
            "created_utc": artifacts.created_utc,
            "last_active_utc": artifacts.last_active_utc,
            "completed_utc": completed_utc,
            "run_limit_hours": limit_hours,
            "traded": _traded(segments, artifacts.sessions),
            "cumulative_runtime_s": runtime_s,
            "cumulative_research_s": _cumulative_phase_s(segments, RESEARCH_PHASE),
            "cumulative_trading_s": _cumulative_phase_s(segments, TRADING_PHASE),
            "cumulative_trials": artifacts.trials,
            # Retention's one mark on the record (story #138): the heavy directories beside this
            # file are gone, so a reader knows the run's non-champion source references no longer
            # resolve. The record itself is never pruned — it *is* the long-term history — so this
            # is the only thing a prune changes about it.
            "state_pruned": bool(artifacts.state_pruned),
            "complete": bool(artifacts.complete),
            "truncated": truncated,
        },
        "segments": segments,
        # The newest machine this run has worked on (story #139). Derived from ``segments[]``, so
        # it can never disagree with the per-segment blocks it summarizes — those stay the record
        # of what actually produced each night's numbers.
        "environment_latest": _environment_latest(segments),
        "engine": _engine(artifacts),
        "inputs": dict(artifacts.inputs) if artifacts.inputs is not None else None,
        # Everything considered, champions first (story #141) — the funnel, not the trophy shelf.
        # Its death counts are what a results page is judged on, so the rejections are carried in
        # exactly the same shape as the promotions.
        "strategies": [_strategy(strategy) for strategy in strategies],
        # What this run cost and what that bought (story #140) — derived here from the entries the
        # store read off the run's own ledgers, never from a counter carried across a restart.
        "spend": _spend(artifacts, segments),
        # Every session the paper account closed, with its own trade log (story #142). One entry
        # per session date, uncapped like the segments beside them — they are the run's realised
        # spine — while the *trades* inside them are bounded by ``TRADE_CAP`` across the whole run.
        "sessions": sessions,
        # The realised paper-account record: the equity curve, the practitioner metric set, the
        # deflated Sharpe beside the trial count that deflated it, and the equal-weight benchmark.
        # ``null`` — never zeros — for a run that never traded (epic D10), so a website renders
        # "researching" rather than a flat 0% curve it was handed as a result. Computed from the
        # **whole** session history, not from the capped log above: a truncated trade log must not
        # be able to move a metric.
        "performance": _performance(artifacts, segments),
        # The arena every number above was produced in (story #143): the fill model, the fee and
        # slippage the run was charged, the walk-forward geometry, both holdouts, the exhaustion
        # gate, every promotion threshold, the benchmark's rebalancing convention — and the live
        # gate's own verdict, measured rather than claimed. Derived from the run's frozen inputs,
        # so it states what this run was actually configured with and not what the current
        # ``config.yaml`` happens to say.
        "assumptions": assumptions(artifacts.inputs),
        "events": [event.as_dict() for event in events],
        "errors": [error.as_dict() for error in errors],
    }


def _spend(artifacts: RunArtifacts, segments: Sequence[Mapping[str, object]]) -> dict | None:
    """What the run spent, attributed three ways, plus the ratios it is compared on.

    ``None`` for a run with no research evidence at all — no journaled usage and no journaled
    trials. That is exactly the shape of a run with no LLM key: the legacy loop needs none, writes
    no session ledger, and a block of zeros beside it would read as "this run was free" rather than
    "nobody measured" (epic D10). Once *either* kind of evidence exists the block is present, with
    an explicit ``null`` wherever a number is unknown.

    Everything here is **derived at write time** from the entries the store read back off the run's
    own ledgers, so three short segments total exactly what one long one does and a rewrite after a
    crash cannot double-count (epic D4).
    """
    entries = artifacts.spend
    if entries is None and artifacts.trials is None:
        return None
    # The whole bill, priced once and used twice — the headline number and the denominator of the
    # two cost ratios are the same figure by construction, never two sums that could disagree.
    # ``None`` the moment any entry is unpriceable: a partial sum presented as a total understates
    # the bill while looking complete.
    usd = _summed_usd(entries)
    return {
        "tokens": _tokens_of(entries),
        "by_model": _grouped(entries, lambda entry: entry.model),
        "by_stage": _grouped(entries, lambda entry: entry.stage),
        "by_segment": _by_segment(entries, segments),
        f"llm_{USD_ESTIMATE}": usd,
        "pricing_table_version": artifacts.pricing_table_version,
        "efficiency": _efficiency(artifacts, segments, usd=usd),
    }


def _tokens_of(entries: Sequence[SpendEntry] | None) -> dict | None:
    """One bucket's token counts: the four billed fields, plus the total behind them.

    The total is always known (the ledger journals it per episode); the four fields are ``null``
    when *any* contributing entry never journaled its split, because a sum over the entries that
    did would silently report less spend than happened.
    """
    if entries is None:
        return None
    totals = dict.fromkeys(USAGE_FIELDS, 0)
    split_known = True
    total = 0
    for entry in entries:
        total += int(entry.tokens)
        if entry.usage is None:
            split_known = False
            continue
        for name in USAGE_FIELDS:
            totals[name] += int(entry.usage.get(name, 0) or 0)
    counts: dict[str, int | None] = (
        dict(totals) if split_known else dict.fromkeys(USAGE_FIELDS, None)
    )
    counts[TOTAL_TOKENS] = total
    return counts


def _bucket(entries: Sequence[SpendEntry]) -> dict:
    """One axis bucket: its token counts and what they cost."""
    counts = _tokens_of(entries) or {}
    return {**counts, USD_ESTIMATE: _summed_usd(entries)}


def _grouped(entries: Sequence[SpendEntry] | None, key: Callable[[SpendEntry], str]) -> dict | None:
    """Spend by one attribution axis — the model that was asked, or the stage it was asked in.

    Both axes are read straight off what the ledgers already record per episode; the record
    invents no taxonomy of its own, so what a consumer sees here is what the driver actually did.
    """
    if entries is None:
        return None
    groups: dict[str, list[SpendEntry]] = {}
    for entry in entries:
        groups.setdefault(key(entry), []).append(entry)
    return {name: _bucket(group) for name, group in sorted(groups.items())}


def _by_segment(
    entries: Sequence[SpendEntry] | None, segments: Sequence[Mapping[str, object]]
) -> list | None:
    """Spend per process invocation — one night's bill, comparable to another night's.

    An entry belongs to the segment whose window contains its stamp: the ledgers and the segments
    are written by the same process against the same UTC wall clock, so the join needs no third
    record. An entry that falls in no window (an unparseable stamp, a ledger adopted from before
    this run) is attributed to no segment but still counted in the run's own totals — a number may
    lose its attribution, never its existence.
    """
    if entries is None:
        return None
    buckets: list[list[SpendEntry]] = [[] for _ in segments]
    for entry in entries:
        index = _segment_of(entry, segments)
        if index is not None:
            buckets[index].append(entry)
    return [
        {"index": segment.get("index", position), **_bucket(buckets[position])}
        for position, segment in enumerate(segments)
    ]


def _segment_of(entry: SpendEntry, segments: Sequence[Mapping[str, object]]) -> int | None:
    """The index of the segment that was open when this entry was journaled, or ``None``.

    Searched **newest first**, which is what settles the one ambiguous case: one segment's stop
    stamp and the next one's start stamp can land on the same instant, and the work journaled at
    that instant belongs to the process that had just opened, not to the one that had just left.
    """
    moment = _moment(entry.at)
    if moment is None:
        return None
    for position in reversed(range(len(segments))):
        segment = segments[position]
        started = _moment(segment.get("started_utc"))
        if started is None or moment < started:
            continue
        stopped = _moment(segment.get("stopped_utc"))
        if stopped is None or moment <= stopped:
            return position
    return None


def _moment(stamp: object) -> datetime | None:
    """One record stamp as a moment, or ``None`` for anything that is not one."""
    if not isinstance(stamp, str):
        return None
    try:
        return _parse(stamp)
    except ValueError:
        return None


def _summed_usd(entries: Sequence[SpendEntry] | None) -> float | None:
    """The bill for a set of entries — ``None`` if **any** of them could not be priced.

    Deliberately all-or-nothing, the same rule the price table itself applies: an operator must
    never be shown a confidently false figure, and a total that quietly skipped the models it did
    not know would be exactly that.
    """
    if entries is None:
        return None
    total = 0.0
    for entry in entries:
        if entry.usd_estimate is None:
            return None
        total += float(entry.usd_estimate)
    return round(total, 6)


def _efficiency(
    artifacts: RunArtifacts, segments: Sequence[Mapping[str, object]], *, usd: float | None
) -> dict:
    """The four numbers a run is compared on — cost per result, and results per hour.

    Every one of them is a ratio, and every ratio is ``null`` when its denominator is zero or
    unknown: a run that has crowned no champion yet has no cost per champion, and reporting one
    (or an infinity, or a zero) would be an answer to a question nobody can answer yet. That is
    the normal state of a young run, not an error.
    """
    trials = artifacts.trials
    champions = artifacts.champions
    hours = _hours_of(_cumulative_phase_s(segments, RESEARCH_PHASE))
    return {
        # Both dollar ratios say ``estimate`` in their own names, because both inherit it from the
        # price table that produced their numerator.
        "usd_per_champion_estimate": _ratio(usd, champions),
        "usd_per_trial_estimate": _ratio(usd, trials),
        "trials_per_hour": _ratio(trials, hours, digits=3),
        "research_hours_per_champion": _ratio(hours, champions, digits=3),
    }


def _hours_of(seconds: float | None) -> float | None:
    return None if seconds is None else seconds / 3600.0


def _ratio(numerator: float | None, denominator: float | None, *, digits: int = 6) -> float | None:
    """One efficiency ratio, or ``None`` when either side is unknown or the denominator is zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator, digits)


def _performance(artifacts: RunArtifacts, segments: Sequence[Mapping[str, object]]) -> dict | None:
    """The paper account's realised record, or ``null`` for a run that never traded.

    **The realised record and the backtest record are different sections, deliberately.** What is
    here came out of the paper account: marks the engine wrote at a session close, fills it
    actually placed. Backtest and scorecard numbers live under ``strategies[]`` and are never
    blended in — the block names itself ``paper_account`` so no consumer can present one as the
    other, which is the whole reason the two live apart.

    ``n_trials`` is the run's own cumulative journaled trial count, so the deflated Sharpe is
    deflated by the searching this run actually did (and publishes that count beside itself).

    ``null`` in **two** cases, which are the same case seen from two sides: a run that never traded
    (epic D10 — a website must render "researching", not a flat 0% curve), and a run whose account
    journaled no daily mark at all. The second is what an adopted history looks like — segments
    that counted orders under a Noctis that had no equity ledger — and a block of nulls beside them
    would read as a measurement that came out empty rather than as one nobody took.
    """
    if not _traded(segments, artifacts.sessions) or not artifacts.sessions:
        return None
    return performance(
        list(artifacts.sessions), n_trials=artifacts.trials, benchmark=artifacts.benchmark
    )


def _sessions(
    sessions: Sequence[DailySession],
) -> tuple[list[dict], dict[str, int] | None]:
    """Every session the account closed, and the note if the run's trade log had to be bounded.

    The cap is on **trades across the whole run**, not per session: a fortnight of quiet days
    followed by one frantic one should publish the frantic one, and a per-session cap would hide
    exactly the session worth reading. The earliest trades are kept (the run in order), and the
    note names the total — silent truncation is forbidden.
    """
    kept = 0
    total = sum(len(session.fills) for session in sessions)
    rendered: list[dict] = []
    for session in sessions:
        room = max(TRADE_CAP - kept, 0)
        fills = list(session.fills)[:room]
        kept += len(fills)
        rendered.append(_session(session, fills))
    return rendered, None if total <= TRADE_CAP else {"kept": kept, "total": total}


def _session(session: DailySession, fills: Sequence[TradeFill]) -> dict:
    """One closed session: what the account was worth, what it traded, what it still holds."""
    return {
        "as_of": session.date,
        "equity": session.equity,
        "start_equity": session.start_equity,
        "end_equity": session.end_equity,
        "realized_pnl": session.realized_pnl,
        "orders_submitted": session.orders_submitted,
        "positions_end": dict(session.positions_end),
        "trades": [_trade(fill) for fill in fills],
    }


def _trade(fill: TradeFill) -> dict:
    """One fill, as the record states it — with the champion it is attributed to.

    Every key is present, ``null`` where the fill did not carry the value: the record's convention
    is the opposite of the per-day report's, where an absent enrichment field is *omitted* so an
    operator's existing reports stay byte identical.
    """
    return {
        "ts": fill.ts,
        "symbol": fill.symbol,
        "side": fill.side,
        "quantity": fill.quantity,
        "price": fill.price,
        "fees_usd": fill.fees_usd,
        "slippage_bps": fill.slippage_bps,
        "champion": fill.champion,
        "rationale": fill.rationale,
    }


def _traded(
    segments: Sequence[Mapping[str, object]], sessions: Sequence[DailySession] = ()
) -> bool:
    """Whether this run ever placed a paper order — the flag a ``null`` performance block hangs on.

    A **research-only run is first-class** (epic §5.6): a run may accumulate many research segments
    and never trade — the market was closed, no champion was crowned, or every segment came from a
    standalone ``noctis research --resume``. Then this is ``false`` and ``performance`` is ``null``
    rather than zeros (D10), so a website renders "researching" instead of a fake flat 0% equity
    curve.

    Derived from **every** piece of evidence the record carries, never latched by whichever one
    was written first: the per-segment ``trades`` counter the loop writes at each CLOSE, and (story
    #142) the fills the run's own account ledger journaled per session. Either is enough; neither
    is required of the other, because a segment killed before its counter landed still leaves its
    fills on the ledger, and a record adopted from a history that predates the ledger still has its
    counters. The rule itself has not moved since story #137 — "no trading was ever recorded ⇒
    false" — only the list of places it looks.

    A run whose account exists but never filled an order is **not** traded: an account that opened
    and sat flat has no realised record to show, and rendering its 100k flat line as a result is
    exactly the lie D10 forbids.
    """
    if any(_counted(segment, TRADES_COUNTER) > 0 for segment in segments):
        return True
    return any(session.fills for session in sessions)


def _counted(segment: Mapping[str, object], key: str) -> int:
    counters = segment.get("counters")
    value = counters.get(key) if isinstance(counters, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _status(artifacts: RunArtifacts, *, completed_utc: str | None) -> str:
    """The run's lifecycle status, **derived** from the segments rather than carried beside them.

    Derived, so the two can never disagree: a run is ``running`` while its last segment is open,
    ``interrupted`` while its last segment is one a kill left behind, ``stopped`` (resumable) once
    it closed cleanly, and ``completed`` once something sealed it — a deliberate ``--finish``, or
    the run-level cap the ``completed_utc`` handed in already accounts for (:func:`_capped_at`).
    """
    if completed_utc is not None:
        return "completed"
    if not artifacts.segments:
        return "stopped"
    last = artifacts.segments[-1].status
    return last if last in ("running", "interrupted") else "stopped"


def _environment_of(segment: SegmentArtifact) -> dict | None:
    """One segment's environment block, or ``None`` when that segment measured none.

    An empty mapping is treated as nothing measured rather than as an empty machine: a block with
    no facts in it would satisfy "the key is present" while telling a reader nothing, and the
    record's convention is that a *stated* value is a value.
    """
    environment = segment.environment
    return dict(environment) if environment else None


def _segment(segment: SegmentArtifact) -> dict:
    engine = segment.engine
    phases = segment.phase_seconds
    return {
        "index": segment.index,
        "started_utc": segment.started_utc,
        "stopped_utc": segment.stopped_utc,
        "duration_s": _duration_s(segment.started_utc, segment.stopped_utc),
        "stopped_reason": segment.stopped_reason,
        "status": segment.status,
        "argv": list(segment.argv),
        "command": segment.command,
        "resumed": bool(segment.resumed),
        "counters": dict(segment.counters),
        "phase_seconds": dict(phases) if phases is not None else None,
        "environment": _environment_of(segment),
        "engine_version": engine.engine_version if engine is not None else None,
        "engine_fingerprint": dict(engine.fingerprint) if engine is not None else None,
    }


def _environment_latest(segments: Sequence[Mapping[str, object]]) -> dict | None:
    """The newest environment any segment recorded, or ``None`` when none did.

    **Derived, never stored twice.** A consumer rendering "the machine this run is on" reads one
    key instead of walking ``segments[]`` backwards, and because it is recomputed from the
    segments at every write it can never disagree with them. A segment that measured nothing is
    skipped rather than blanking the answer — a resume by a Noctis too old to capture an
    environment must not erase what the run already knew about itself.
    """
    for segment in reversed(segments):
        environment = segment.get("environment")
        if isinstance(environment, Mapping) and environment:
            return dict(environment)
    return None


def _engine(artifacts: RunArtifacts) -> dict:
    """The run's engine identity — frozen at creation — plus whether it ever ran another engine.

    ``mixed_engine`` is **derived**, from two independent facts that mean the same thing: a segment
    whose digests differ from the run's, and an accepted engine change on the record. Either way
    the run's numbers were produced by more than one engine, and a consumer must know before
    pooling them; a flag stored beside the evidence could disagree with it.
    """
    engine = artifacts.engine
    fingerprints = [
        dict(segment.engine.fingerprint)
        for segment in artifacts.segments
        if segment.engine is not None
    ]
    mixed = bool(engine.engine_changes) or any(
        digests != dict(engine.fingerprint) for digests in fingerprints
    )
    return {
        "engine_version": engine.engine_version,
        "engine_epoch": engine.engine_epoch,
        "noctis_version": engine.noctis_version,
        "fingerprint": dict(engine.fingerprint),
        "comparable_key": engine.comparable_key,
        "mixed_engine": mixed,
        "engine_changes": [dict(change) for change in engine.engine_changes],
    }


def _cumulative_runtime_s(segments: Sequence[Mapping[str, object]]) -> float:
    """Recomputed from the segments at every write, never incremented — so a rewrite after a
    crash is idempotent and three short segments total exactly what one long one would."""
    total = 0.0
    for segment in segments:
        duration = segment.get("duration_s")
        if isinstance(duration, int | float):
            total += float(duration)
    return round(total, 3)


def _cumulative_phase_s(segments: Sequence[Mapping[str, object]], phase: str) -> float | None:
    """Seconds this run spent working in one phase, summed over the segments that measured it.

    Derived at every write from ``segments[]``, exactly like the runtime total beside it, so three
    short nights report what one long night would and a rewrite after a crash cannot double-count.
    ``None`` when **no** segment reports any phase timing at all — a record written before the
    measurement existed knows nothing about it, and ``0.0`` would be a claim it never made. A run
    that did measure and never traded reports an honest ``0.0`` (D10: an absent value is null, a
    zero is a zero).
    """
    total: float | None = None
    for segment in segments:
        phases = segment.get("phase_seconds")
        if not isinstance(phases, Mapping):
            continue
        measured = phases.get(phase)
        total = (total or 0.0) + (float(measured) if isinstance(measured, int | float) else 0.0)
    return None if total is None else round(total, 3)


def _run_limit_hours(inputs: Mapping[str, object] | None) -> float | None:
    """The run-level compute cap, read off the run's own frozen configuration (story #136).

    ``None`` for an uncapped run, for a run that froze no configuration at all (an adopted
    history), and for anything the record carries in that slot that is not a number — a
    hand-edited record must never make a run look capped when the engine would not enforce one.
    """
    settings = inputs.get("settings") if isinstance(inputs, Mapping) else None
    resolved = settings.get("resolved") if isinstance(settings, Mapping) else None
    limit = resolved.get(RUN_LIMIT_SETTING) if isinstance(resolved, Mapping) else None
    if isinstance(limit, bool) or not isinstance(limit, int | float):
        return None
    return float(limit)


def _capped_at(segments: Sequence[Mapping[str, object]], limit_hours: float | None) -> str | None:
    """When this run's accumulated runtime crossed its cap, or ``None`` while it has not.

    The run-level cap seals a run the moment it is breached, and it does so **derivably**: the
    crossing is recomputed from the closed segments at every write rather than latched by whichever
    process happened to notice. A run killed at the instant it crossed is therefore still
    ``completed`` when it is next read, which is the whole promise — a published result cannot
    quietly gain another segment.

    Only *closed* segments count (an open one has no honest duration yet), so the seal always lands
    on a segment boundary: the loop stops between phases, the segment closes, and that stop stamp is
    the moment the run completed.
    """
    if limit_hours is None:
        return None
    cap_s = limit_hours * 3600.0
    total = 0.0
    for segment in segments:
        duration = segment.get("duration_s")
        if not isinstance(duration, int | float):
            continue
        total += float(duration)
        if total >= cap_s:
            stopped = segment.get("stopped_utc")
            return stopped if isinstance(stopped, str) else None
    return None


def _duration_s(started: str | None, stopped: str | None) -> float | None:
    """Seconds between two record stamps, or ``None`` while a segment is still open."""
    if started is None or stopped is None:
        return None
    return round((_parse(stopped) - _parse(started)).total_seconds(), 3)


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _ordered(strategies: Sequence[StrategyArtifact]) -> list[StrategyArtifact]:
    """Champions first, then by name — the order the section is capped in, and read in.

    Two jobs in one rule. A consumer gets the run's *product* at the top without sorting anything,
    and the cap below can never drop a champion: the entries at risk are the tail, and a champion
    is never in it. Everything after the champions is alphabetical, so two writes of the same run
    order identically and a diff between them is meaningful.
    """
    return sorted(strategies, key=lambda s: (s.outcome != PROMOTED_OUTCOME, s.name))


def _strategy(strategy: StrategyArtifact) -> dict:
    """One candidate's entry — the same shape whether it was crowned or killed."""
    return {
        "name": strategy.name,
        "outcome": strategy.outcome,
        "tier": strategy.tier,
        "decided_utc": strategy.decided_utc,
        "trials": strategy.trials,
        "gates": [dict(gate) for gate in strategy.gates],
        "rationale": strategy.rationale,
        "source_path": strategy.source_path,
        "source_sha256": strategy.source_sha256,
        "source": strategy.source,
    }


def _capped(items: Sequence[_T], cap: int) -> tuple[Sequence[_T], dict[str, int] | None]:
    """Bound one list, and say so when the bound bit.

    The **earliest** entries are kept, which each caller has already ordered so that the earliest
    are the ones worth keeping: for events, the ones explaining how a run reached its state (a
    stolen lock, a degraded seam, the first error); for strategies, the champions. The note names
    the total, so a reader always knows what is not here. Silent truncation is forbidden.
    """
    total = len(items)
    if total <= cap:
        return items, None
    return items[:cap], {"kept": cap, "total": total}


def _replace_status(segment: SegmentArtifact, status: str) -> SegmentArtifact:
    return replace(segment, status=status)
