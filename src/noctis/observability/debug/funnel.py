"""The funnel ledger — a pure state machine over the ordered event stream, keyed per strategy.

This is the QA reporting counterpart of :mod:`noctis.champions.promotion`: a *pure* reduction
with no I/O, no clock, and no hidden state. It is fed the same ``Event`` stream a session already
emits (arrival-stamped by the recorder, story #43) and answers one question honestly — *what
happened to each candidate strategy this window?*

**Why key per strategy name, not a bag of counters.** A pile of global counters
(``n_written``, ``n_rejected``, …) cannot answer "was this candidate rejected *before* any
sweep ran?" — the very question that separates an honest early kill from an overfit-then-loosen
death march (AGENTS.md rule 2). Keeping one :class:`StrategyFate` per ``name`` makes that answer
fall straight out: a strategy is *rejected pre-sweep* iff it carries a reject and zero sweeps.
The per-stage funnel counts are then just an aggregation over those fates.

**What drives the ledger.** Only the funnel-relevant tool events do, identified by
``meta["tool"]`` and attributed to a strategy by ``meta["args"]["name"]`` (both shipped in #37):

* ``write_strategy`` → a write *attempt*; a successful one (``meta["ok"]``) also reaches WRITTEN.
  Attempts are counted separately from successes because a write-fail-only stream (the coder
  never produced a valid file) is a real, distinct outcome.
* ``run_backtest`` (ok) → BACKTESTED.
* ``run_sweep`` (ok) → SWEPT, folding in ``meta["n_trials"]`` and ``meta["n_failed"]`` (#38) so a
  sweep's burned budget — trials that errored — is on the record.
* ``evaluate_vs_champion`` (ok) → COMPARED; and CHAMPION when ``meta["promoted"]`` is truthy,
  taking ``meta["rationale"]`` as the reason.
* ``reject_strategy`` (ok) → REJECTED, taking the reason from ``meta["args"]["reason"]``.

Events with no strategy name (phase frames, ``get_market_digest``, reasoning) never touch the
ledger. Fate order is *first-seen* (the order a name first appears in the stream), which is
deterministic for a given input — the property every renderer relies on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from noctis.observability.events import Event

# The tools that move a candidate through the funnel. Everything else (market digests, previews,
# the experiment log) is observation, not a stage transition, so it never touches a fate.
_FUNNEL_TOOLS = frozenset(
    {"write_strategy", "run_backtest", "run_sweep", "evaluate_vs_champion", "reject_strategy"}
)

# Phase buckets the timing accounting reports. RESEARCH/TRADING/CLOSE are the productive phases;
# everything else in the window — the gap before the first phase frame, and any STOPPED tail —
# is the honest catch-all "idle-wait" (no research/trading/close work runs there). These four
# always sum to the full window, so a reader can trust the accounting adds up.
PHASE_KEYS: tuple[str, ...] = ("research", "trading", "close", "idle_wait")
_PRODUCTIVE_PHASES = frozenset({"research", "trading", "close"})


@dataclass(frozen=True)
class StampedEvent:
    """One :class:`Event` plus the UTC instant the recorder observed it.

    ``Event`` carries no timestamp by design (the recorder arrival-stamps it), so the pure module
    takes the stamp injected here rather than reading any clock. ``t`` is a UTC-wall-clock
    ``datetime`` by contract; this module never inspects its tzinfo, only compares stamps.
    """

    t: datetime
    event: Event


@dataclass(frozen=True)
class StrategyFate:
    """The full recorded fate of one candidate strategy — the row behind a per-strategy report.

    ``write_attempts`` counts every ``write_strategy`` call (a fail-only run shows attempts with
    ``writes == 0``); ``sweep_trials``/``sweep_failed`` are the summed sweep budget and the slice
    of it that errored. ``outcome`` is the one honest label a reader scans for.
    """

    name: str
    write_attempts: int
    writes: int
    backtests: int
    sweeps: int
    sweep_trials: int
    sweep_failed: int
    comparisons: int
    promoted: bool
    rejected: bool
    outcome: str
    reason: str


@dataclass(frozen=True)
class FunnelCounts:
    """Per-stage strategy counts — how many *distinct* strategies reached each stage.

    ``write_attempts`` is the exception: a call count (attempts, not strategies), because a
    write-fail-only run has attempts but zero strategies written. ``rejected_pre_sweep`` is the
    subset of ``rejected`` whose candidates never had a sweep — the distinct early-kill count.
    """

    write_attempts: int
    written: int
    backtested: int
    swept: int
    compared: int
    champion: int
    rejected: int
    rejected_pre_sweep: int


@dataclass(frozen=True)
class Ledger:
    """The whole funnel picture for one event stream: aggregate counts + per-strategy fates."""

    counts: FunnelCounts
    fates: tuple[StrategyFate, ...]


@dataclass
class _Fate:
    """Mutable accumulator for one strategy, frozen into a :class:`StrategyFate` at the end."""

    name: str
    write_attempts: int = 0
    writes: int = 0
    backtests: int = 0
    sweeps: int = 0
    sweep_trials: int = 0
    sweep_failed: int = 0
    comparisons: int = 0
    promoted: bool = False
    rejected: bool = False
    reason: str = ""


def _outcome(f: _Fate) -> str:
    """The one honest label for a fate. Champions and rejects are terminal; a run whose only
    write attempts failed is a distinct *write failed*; anything else is still *in progress*."""
    if f.promoted:
        return "champion"
    if f.rejected:
        return "rejected pre-sweep" if f.sweeps == 0 else "rejected"
    if f.write_attempts > 0 and f.writes == 0:
        return "write failed"
    return "in progress"


def _int(value: object) -> int:
    """A defensive coercion for ``n_trials``/``n_failed`` off an untyped ``meta`` dict."""
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def build_ledger(events: Sequence[StampedEvent]) -> Ledger:
    """Reduce a stamped event stream to a :class:`Ledger`. Pure: events in, ledger out.

    Only ``tool`` events naming a funnel-relevant tool and a strategy touch the ledger; the
    walk is a single pass in stream order, so fate order is first-seen and deterministic.
    """
    fates: dict[str, _Fate] = {}

    for stamped in events:
        ev = stamped.event
        if ev.kind != "tool":
            continue
        meta = ev.meta
        tool = meta.get("tool")
        if tool not in _FUNNEL_TOOLS:
            continue
        args = meta.get("args") or {}
        name = args.get("name")
        if not name:
            continue
        ok = bool(meta.get("ok"))
        fate = fates.setdefault(name, _Fate(name=name))

        if tool == "write_strategy":
            fate.write_attempts += 1
            if ok:
                fate.writes += 1
        elif tool == "run_backtest":
            if ok:
                fate.backtests += 1
        elif tool == "run_sweep":
            if ok:
                fate.sweeps += 1
                fate.sweep_trials += _int(meta.get("n_trials"))
                fate.sweep_failed += _int(meta.get("n_failed"))
        elif tool == "evaluate_vs_champion":
            if ok:
                fate.comparisons += 1
                if meta.get("promoted"):
                    fate.promoted = True
                    fate.reason = str(meta.get("rationale") or "")
        elif tool == "reject_strategy":
            if ok:
                fate.rejected = True
                fate.reason = str(args.get("reason") or "")

    frozen = tuple(
        StrategyFate(
            name=f.name,
            write_attempts=f.write_attempts,
            writes=f.writes,
            backtests=f.backtests,
            sweeps=f.sweeps,
            sweep_trials=f.sweep_trials,
            sweep_failed=f.sweep_failed,
            comparisons=f.comparisons,
            promoted=f.promoted,
            rejected=f.rejected,
            outcome=_outcome(f),
            reason=f.reason,
        )
        for f in fates.values()
    )
    counts = FunnelCounts(
        write_attempts=sum(f.write_attempts for f in frozen),
        written=sum(1 for f in frozen if f.writes > 0),
        backtested=sum(1 for f in frozen if f.backtests > 0),
        swept=sum(1 for f in frozen if f.sweeps > 0),
        compared=sum(1 for f in frozen if f.comparisons > 0),
        champion=sum(1 for f in frozen if f.promoted),
        rejected=sum(1 for f in frozen if f.rejected),
        rejected_pre_sweep=sum(1 for f in frozen if f.rejected and f.sweeps == 0),
    )
    return Ledger(counts=counts, fates=frozen)


def phase_durations(
    events: Sequence[StampedEvent], window_start: datetime, window_end: datetime
) -> dict[str, float]:
    """Seconds spent in each phase over ``[window_start, window_end)``. Pure and clock-free.

    A phase runs from its ``phase`` frame's stamp until the next frame's stamp; the last frame
    runs to ``window_end``. Any window time not covered by a RESEARCH/TRADING/CLOSE frame — the
    gap before the first frame, and any STOPPED tail — is accounted as ``idle_wait``, the honest
    catch-all. The four returned keys therefore always sum to the window length, so a reader can
    trust the timing adds up rather than guess at unlabelled gaps.
    """
    buckets: dict[str, float] = dict.fromkeys(PHASE_KEYS, 0.0)
    frames = sorted((se for se in events if se.event.kind == "phase"), key=lambda se: se.t)
    if not frames:
        buckets["idle_wait"] = max(0.0, (window_end - window_start).total_seconds())
        return buckets

    def clamp(t: datetime) -> datetime:
        return min(max(t, window_start), window_end)

    lead = (clamp(frames[0].t) - window_start).total_seconds()
    if lead > 0:
        buckets["idle_wait"] += lead

    for i, frame in enumerate(frames):
        seg_start = clamp(frame.t)
        seg_end = clamp(frames[i + 1].t) if i + 1 < len(frames) else window_end
        seconds = max(0.0, (seg_end - seg_start).total_seconds())
        phase = str(frame.event.meta.get("phase", "")).lower()
        key = phase if phase in _PRODUCTIVE_PHASES else "idle_wait"
        buckets[key] += seconds

    return buckets
