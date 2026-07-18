"""The TRADING loop — champions on live/replayed data emitting paper orders.

Champions from the registry are assigned across the universe and evaluated bar by bar with
next-bar-open execution through the paper broker (the SimulatedExchange — no real orders).
Every order passes the risk manager first; breaches are refused and logged. When the feed is
degraded (delayed quotes), order emission halts but observation continues.

**One driver, one per-bar core.** :func:`run_trading_day` polls any
:class:`~noctis.live.feed.BarFeed` — the live yfinance adapter or a catalog
:class:`~noctis.live.feed.ReplayBarFeed` — and funnels every minute group through
:class:`_TradingSession`, so replay and live cannot diverge. The day is *clock-bounded*
when a sleeper is injected (poll, pace, stop between polls, flush at the close) and
*data-bounded* otherwise (drain the feed at CPU speed until it is exhausted).
:func:`run_trading` is the batch convenience wrapper: a whole static timeline in, one
summary out.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from noctis.broker.exits import ExitState, evaluate, ratchet
from noctis.broker.paper import PaperBroker
from noctis.broker.seam import Broker, FeeModel, SlippageModel
from noctis.champions.assignment import assign_indices
from noctis.data.aggregate import StreamingAggregator
from noctis.live.feed import BarFeed, ReplayBarFeed
from noctis.live.risk import RiskLimits, RiskManager
from noctis.observability import Event
from noctis.strategies.base import Bar, ExitRules, TargetContext
from noctis.strategies.candidate import Candidate
from noctis.strategies.families import FamilyRegistry

logger = logging.getLogger("noctis.trading")


@dataclass
class TradingSummary:
    bars_processed: int = 0
    orders_submitted: int = 0
    orders_refused: int = 0
    halted_for_degraded: int = 0
    final_equity: float = 0.0
    start_equity: float = 0.0
    fills: int = 0
    polls: int = 0
    positions: dict = field(default_factory=dict)
    # Daily-loss latch (live-holdout plan 3): set once the bar the loss floor first trips, so
    # the report can name it and the per-bar refusal INFO collapses to one WARNING.
    halt_latched: bool = False
    halt_equity: float = 0.0
    halt_floor: float = 0.0
    # Symbols whose champion-orphaned position this session flattened. An orphan symbol has
    # no strategy this session, so any fill on it IS the flatten — the report accumulator
    # uses this to label those fills honestly instead of "champion signal".
    orphans_flattened: list[str] = field(default_factory=list)
    # Protective-exit fills this session, counted per reason ("stop" / "take_profit" /
    # "trail") — how the operator learns stops are firing. Empty when none fired.
    exit_fills: dict[str, int] = field(default_factory=dict)
    # What this session tells the close report (plain strings, in occurrence order): feed
    # health transitions while it ran, then the refusal/halt summary at finalize. The report
    # accumulator folds these in verbatim, so no per-event callback threads through the
    # drivers and the report stays byte-identical regardless of console verbosity.
    events: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionConfig:
    """How champions trade a session — the composition every driver shares.

    Everything that is fixed for a whole TRADING phase lives here: the champion roster
    (``candidates`` with their per-champion ``live_symbols`` eligibility sets and election
    ``scores``, all index-aligned), the account (``broker``; ``None`` builds a fresh paper
    broker from ``fee_bps``/``slippage_bps``), the risk ``limits``, the rebalance dead-band
    knobs, and the console sink. What varies per *day* — the feed, the clock bounds, the
    pacing — stays on :func:`run_trading_day`, so a multi-session catch-up builds one config
    and drives it once per session date.
    """

    candidates: list[Candidate]
    families: FamilyRegistry | None = None
    live_symbols: list[set[str] | None] | None = None
    scores: list[float] | None = None
    broker: Broker | None = None
    limits: RiskLimits | None = None
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    min_order_notional: float = 0.0
    rebalance_band_pct: float = 0.0
    on_event: Callable[[Event], None] | None = None
    heartbeat_polls: int = 0
    # Who opened each currently-open position (symbol → champion key), from the forward
    # ledger's holder map. Display-only: it names the champion whose displacement orphaned a
    # position in the flatten event/report line. Empty when unknown (batch wrapper, old
    # ledger files) — the flatten itself never depends on it.
    position_holders: dict[str, str] = field(default_factory=dict)


def _assign(
    candidates: list[Candidate],
    symbols: list[str],
    live_symbols: list[set[str] | None] | None = None,
    scores: list[float] | None = None,
) -> dict[str, Candidate]:
    """Assign each symbol the best-scoring champion eligible to trade it (the RiskLimits
    exposure math is unchanged). Thin wrapper over :func:`assign_indices`."""
    idx = assign_indices(len(candidates), symbols, live_symbols, scores)
    return {sym: candidates[j] for sym, j in idx.items()}


class _TradingSession:
    """The per-bar trading core shared by the batch and streaming drivers.

    One :meth:`step` advances the loop by a single minute across the symbols whose bar
    completed for that minute: mark opens, execute the *previous* bar's risk-checked
    decisions, let champions decide this bar, then mark closes. Both drivers funnel every
    bar through here so replay and live can never diverge.
    """

    def __init__(
        self,
        config: SessionConfig,
        symbols: list[str],
        broker: Broker,
        *,
        record_bars: bool = False,
    ):
        self.broker = broker
        self.symbols = symbols
        self.record_bars = record_bars
        # Rebalance dead-band thresholds (0.0 = off). See :class:`TradingConfig`.
        self.min_order_notional = float(config.min_order_notional)
        self.rebalance_band_pct = float(config.rebalance_band_pct)
        # Inline per-decision feed (verbose-observability P4): a console sink for `trade`/`refuse`
        # events, or ``None`` (a bare run) — every emit is guarded on it, so a quiet session
        # constructs no events on the per-bar hot path. Report accounting is untouched: this only
        # *tees* what already happens (a fill, a refusal), never the report's `orders_refused`
        # total. `refuse` collapses to one event per distinct reason (reusing the quiet-replay
        # spirit — a per-bar flood after the daily-loss latch stays one line, not per-bar).
        self._on_event = config.on_event
        self._refused_reasons: set[str] = set()
        families = config.families if config.families is not None else FamilyRegistry()
        assignment = _assign(config.candidates, symbols, config.live_symbols, config.scores)
        self.strategies = {
            sym: assignment[sym].build(families) for sym in symbols if sym in assignment
        }
        self.ctxs = {sym: TargetContext() for sym in self.strategies}
        # The timeframe proxy: minute bars stream in; each strategy's on_bar sees bars of
        # its declared timeframe, emitted when the bucket completes — mirroring how the
        # backtest scored it on aggregated frames. Native-1m strategies pass through.
        self.aggregators = {
            sym: StreamingAggregator(getattr(strat, "timeframe", "1m"))
            for sym, strat in self.strategies.items()
        }
        for sym, strat in self.strategies.items():
            strat.on_start(self.ctxs[sym])
        limits = config.limits if config.limits is not None else RiskLimits()
        self.risk = RiskManager(limits, broker.equity())
        # A carried position (the continuous account across sessions) keeps its direction
        # until this session's strategy first decides — seeding 0 would force-flatten it at
        # the first bar, injecting turnover the strategy never chose. A fresh broker is
        # flat everywhere, so this seeds 0 exactly as before.
        self.pending: dict[str, int] = {}
        for sym in self.strategies:
            q = broker.position(sym).quantity
            self.pending[sym] = (q > 0) - (q < 0)
        # Protective-exit tracking (the fill-model contract, same engine as the simulator).
        # A carried position anchors at its true entry (the account's avg price); rules stay
        # dormant until the strategy declares them with a target. The latch mirrors the
        # simulator exactly: after an exit fires, the symbol holds flat until the raw target
        # series changes value.
        self.exit_states: dict[str, ExitState | None] = {}
        for sym in self.strategies:
            pos = broker.position(sym)
            self.exit_states[sym] = (
                ExitState(
                    direction=1 if pos.quantity > 0 else -1,
                    entry_price=pos.avg_price,
                    best=pos.avg_price,
                )
                if pos.quantity != 0.0
                else None
            )
        self.pending_exits: dict[str, ExitRules | None] = {sym: None for sym in self.strategies}
        self.exit_latched: dict[str, bool] = {sym: False for sym in self.strategies}
        self.prev_raw_target: dict[str, int] = dict(self.pending)
        # Champion-orphan detection: a carried position whose symbol NO current champion is
        # eligible to trade (its opener was displaced from the board, or its symbol set
        # changed) is unmanaged — no strategy will ever decide it again, so it is flattened
        # at its first bar this session (step 0 below). Eligibility comes from the same
        # resolver the trading assignment and settle attribution use, so the three cannot
        # disagree on what "no champion" means. A symbol REASSIGNED to a *different*
        # champion is NOT an orphan: the new assignee inherits the position (``pending`` is
        # seeded from its sign above) and re-decides at its first completed bar — flattening
        # it would preempt that decision and inject turnover nobody chose, while realized
        # attribution at settle and the unrealized display already follow the inheritor.
        held = sorted(broker.positions())
        owned = assign_indices(len(config.candidates), held, config.live_symbols, config.scores)
        self.orphans: set[str] = {sym for sym in held if sym not in owned}
        self._position_holders = dict(config.position_holders)
        # Fills already on a carried broker belong to earlier sessions' reports.
        self._fills_at_start = len(broker.fills)
        self.summary = TradingSummary(start_equity=broker.equity())
        self._built: dict[str, list[Bar]] = {}

    def step(self, bars: dict[str, Bar], *, degraded: bool = False) -> None:
        """Advance one minute over the symbols present in ``bars``."""
        present = [sym for sym in self.strategies if sym in bars]

        # 0) flatten champion-orphans at their first observable bar. The *decision* was made
        # at session start (no eligible champion — see __init__); the *fill* happens at this
        # bar's open, the same decide-at-t/fill-at-next-open discipline every strategy
        # decision follows — never at a stale carried mark. A degraded feed halts ALL order
        # emission, flattens included (delayed quotes price nothing honestly); the orphan
        # stays queued and retries next bar, or next session — detection re-runs at every
        # session start until the position is flat.
        for sym in [s for s in self.orphans if s in bars]:
            bar = bars[sym]
            self.broker.set_price(sym, bar.open, bar.ts_event)
            if degraded:
                continue
            held_qty = self.broker.position(sym).quantity
            # Through the normal risk path. A flat target is risk-reducing, so the manager
            # never refuses it — including while the daily-loss latch is tripped (risk.py's
            # stated policy: new exposure is refused, "flattening is allowed"). Routing the
            # flatten through ``target`` keeps that policy in one place all the same.
            decision = self.risk.target(
                sym,
                0,
                bar.open,
                self.broker.equity(),
                {s: self.broker.position(s).quantity for s in self.strategies},
                self.broker.marks(),
            )
            if decision.refused:  # unreachable today; honor the risk seam's verdict anyway
                self.summary.orders_refused += 1
                continue
            fill = self.broker.rebalance_to(sym, decision.target_qty)
            self.orphans.discard(sym)
            if fill is None:
                continue
            self.summary.orders_submitted += 1
            self.summary.orphans_flattened.append(sym)
            opener = self._position_holders.get(sym)
            msg = (
                f"Orphaned position flattened: {sym} {held_qty:+.4f} @ {fill.price:.2f} — "
                f"no champion on the board is eligible for it"
                + (f" (opened by {opener})" if opener else "")
            )
            self.summary.events.append(msg)
            if self._on_event is not None:
                self._on_event(
                    Event(
                        "orphan",
                        msg,
                        meta={
                            "symbol": sym,
                            "quantity": held_qty,
                            "price": fill.price,
                            "opened_by": opener or "",
                        },
                        level=1,
                    )
                )

        # 1) mark opens for symbols trading this bar (and retain the live-built bar).
        for sym in present:
            bar = bars[sym]
            self.broker.set_price(sym, bar.open, bar.ts_event)
            if self.record_bars:
                self._built.setdefault(sym, []).append(bar)

        marks = self.broker.marks()

        # 2) execute the previous bar's decisions, risk-checked. Equity and positions are
        # re-read per symbol so earlier fills in this minute count against the gross cap.
        for sym in present:
            self.summary.bars_processed += 1
            if degraded:
                self.summary.halted_for_degraded += 1
                logger.info("trading halted: degraded feed; skipping %s", sym)
                continue
            equity = self.broker.equity()
            # Announce the daily-loss latch exactly once, the bar it first trips, so the
            # per-bar refusal log below collapses to this single WARNING for the rest of the
            # session. The loss LIMIT is unchanged — only how its refusals are logged.
            if self.risk.is_halted(equity) and not self.summary.halt_latched:
                floor = self.risk.start_equity * (1.0 - self.risk.limits.max_daily_loss_pct / 100.0)
                self.summary.halt_latched = True
                self.summary.halt_equity = equity
                self.summary.halt_floor = floor
                logger.warning(
                    "daily-loss halt latched: equity=%.2f floor=%.2f; refusing new "
                    "exposure for the rest of session",
                    equity,
                    floor,
                )
            positions = {s: self.broker.position(s).quantity for s in self.strategies}
            decision = self.risk.target(
                sym, self.pending[sym], marks[sym], equity, positions, marks
            )
            if decision.refused:
                self.summary.orders_refused += 1
                # Once latched, every increase is refused every bar — the one WARNING above
                # already said so. Drop the rest (and any pre-latch "no room" refusal) to DEBUG
                # so a volatile session no longer floods INFO with identical lines. The exact
                # ``orders_refused`` total is unchanged, so the count stays honest.
                logger.debug("order refused for %s: %s", sym, decision.reason)
                # Inline feed: surface the FIRST refusal per distinct reason only (the reason
                # strings are symbol-agnostic — "daily loss limit breached…", "no exposure
                # room"), so a halted session that refuses every bar still shows one `refuse`
                # line, mirroring the collapsed WARNING above. The membership test gates event
                # construction, so a quiet or already-seen refusal costs nothing.
                if self._on_event is not None and decision.reason not in self._refused_reasons:
                    self._refused_reasons.add(decision.reason)
                    self._on_event(
                        Event(
                            "refuse",
                            f"{sym}: {decision.reason}",
                            meta={"symbol": sym, "reason": decision.reason},
                            level=2,
                        )
                    )
                continue
            # Rebalance dead-band: re-true a HELD same-direction position only when the drift is
            # material, so a held champion doesn't emit a sub-share fill nearly every bar as
            # equity/price wobble. Opens (current 0), exits (target 0), and flips (opposite sign)
            # always run; a skip leaves ``self.pending[sym]`` so the next bar re-evaluates and the
            # position simply holds. Both thresholds default 0.0 ⇒ never skips ⇒ fills are
            # byte-identical to the pre-band loop (every fill-count test is a regression guard).
            target_qty = decision.target_qty
            current = positions[sym]
            if current != 0.0 and target_qty != 0.0 and (target_qty > 0) == (current > 0):
                drift = abs(target_qty - current)
                band = self.rebalance_band_pct / 100.0 * abs(target_qty)
                if drift * marks[sym] < self.min_order_notional or drift < band:
                    continue
            fill = self.broker.rebalance_to(sym, target_qty)
            # Exit tracking follows the fills exactly as in the simulator: flat clears the
            # anchor, an open or a flip re-anchors at the true entry (this fill's price).
            new_qty = self.broker.position(sym).quantity
            if new_qty == 0.0:
                self.exit_states[sym] = None
            elif fill is not None and (current == 0.0 or (current > 0.0) != (new_qty > 0.0)):
                self.exit_states[sym] = ExitState(
                    direction=1 if new_qty > 0.0 else -1, entry_price=fill.price, best=fill.price
                )
            if fill is not None:
                self.summary.orders_submitted += 1
                # Inline feed: one `trade` event per ACTUAL fill (flat→long, exit, flip, or a
                # material re-true), mirroring the fills the TRADING phase already folds into
                # the report. The dead-band skip above ``continue``s before this
                # line, so a suppressed sub-share adjustment emits nothing — construction is gated
                # on the sink so a quiet run pays nothing.
                if self._on_event is not None:
                    self._on_event(
                        Event(
                            "trade",
                            f"{fill.symbol} {fill.side.value} {fill.quantity:.4f} @ "
                            f"{fill.price:.2f}",
                            meta={
                                "symbol": fill.symbol,
                                "side": fill.side.value,
                                "qty": fill.quantity,
                                "price": fill.price,
                            },
                            level=2,
                        )
                    )

        # 3) champions decide for this bar. Each minute bar feeds the symbol's timeframe
        # proxy; the strategy only decides when an aggregated bar completes — between
        # completions the pending target holds. When a bucket completes, armed protective
        # exits evaluate FIRST, against that completed strategy-timeframe bar (never a raw
        # sub-timeframe minute), through the same engine the simulator runs. Precedence:
        # the halt latch and orphan flattening above own their paths unchanged — exit
        # evaluation runs only for a live (un-degraded), un-halted, champion-held position.
        for sym in present:
            agg_bar = self.aggregators[sym].add(bars[sym])
            if agg_bar is None:
                continue
            state = self.exit_states.get(sym)
            rules = self.pending_exits.get(sym)
            if (
                state is not None
                and rules is not None
                and not degraded
                and not self.risk.is_halted(self.broker.equity())
            ):
                trigger = evaluate(rules, state, agg_bar)
                if trigger is None:
                    self.exit_states[sym] = ratchet(state, agg_bar)  # after evaluate, never before
                else:
                    # Through the normal risk path like every close (a flat target is
                    # risk-reducing, so the seam never refuses it — honor its verdict anyway).
                    decision = self.risk.target(
                        sym,
                        0,
                        trigger.price,
                        self.broker.equity(),
                        {s: self.broker.position(s).quantity for s in self.strategies},
                        self.broker.marks(),
                    )
                    if decision.refused:  # unreachable today
                        self.summary.orders_refused += 1
                    else:
                        fill = self.broker.rebalance_to(
                            sym, decision.target_qty, price=trigger.price, reason=trigger.reason
                        )
                        self.exit_states[sym] = None
                        self.exit_latched[sym] = True
                        if fill is not None:
                            self.summary.orders_submitted += 1
                            if self._on_event is not None:
                                self._on_event(
                                    Event(
                                        "trade",
                                        f"{fill.symbol} {fill.side.value} {fill.quantity:.4f} @ "
                                        f"{fill.price:.2f} ({fill.reason} exit)",
                                        meta={
                                            "symbol": fill.symbol,
                                            "side": fill.side.value,
                                            "qty": fill.quantity,
                                            "price": fill.price,
                                            "reason": fill.reason,
                                        },
                                        level=2,
                                    )
                                )
            self.strategies[sym].on_bar(self.ctxs[sym], agg_bar)
            raw_target = self.ctxs[sym].target
            if self.exit_latched[sym] and raw_target != self.prev_raw_target[sym]:
                self.exit_latched[sym] = False  # the strategy re-decided; execute normally
            self.prev_raw_target[sym] = raw_target
            self.pending[sym] = 0 if self.exit_latched[sym] else raw_target
            self.pending_exits[sym] = self.ctxs[sym].exits

        # 4) mark closes.
        for sym in present:
            self.broker.set_price(sym, bars[sym].close, bars[sym].ts_event)

    def finalize(self) -> TradingSummary:
        self.summary.final_equity = self.broker.equity()
        self.summary.fills = len(self.broker.fills) - self._fills_at_start
        self.summary.positions = {s: p.quantity for s, p in self.broker.positions().items()}
        exit_fills: dict[str, int] = {}
        for f in self.broker.fills[self._fills_at_start :]:
            if f.reason != "target":
                exit_fills[f.reason] = exit_fills.get(f.reason, 0) + 1
        if exit_fills:
            self.summary.exit_fills = exit_fills
            breakdown = ", ".join(f"{reason} ×{n}" for reason, n in sorted(exit_fills.items()))
            self.summary.events.append(
                f"{sum(exit_fills.values())} protective-exit fill(s): {breakdown}"
            )
        if self.orphans:  # detected but never got a tradable bar (no data / degraded feed)
            self.summary.events.append(
                "Orphaned position(s) still open — no tradable bar this session: "
                f"{', '.join(sorted(self.orphans))} (will retry next session)"
            )
        if self.summary.orders_refused:
            if self.summary.halt_latched:
                self.summary.events.append(
                    f"Daily-loss halt latched at equity {self.summary.halt_equity:.2f} "
                    f"(floor {self.summary.halt_floor:.2f}); "
                    f"{self.summary.orders_refused} orders refused"
                )
            else:
                self.summary.events.append(
                    f"{self.summary.orders_refused} orders refused by risk limits"
                )
        return self.summary

    def built_bars(self) -> dict[str, pd.DataFrame]:
        """The bars actually processed this session, per symbol (for reconciliation)."""
        out: dict[str, pd.DataFrame] = {}
        for sym, bars in self._built.items():
            if not bars:
                continue
            out[sym] = pd.DataFrame(
                {
                    "ts_event": [b.ts_event for b in bars],
                    "open": [b.open for b in bars],
                    "high": [b.high for b in bars],
                    "low": [b.low for b in bars],
                    "close": [b.close for b in bars],
                    "volume": [b.volume for b in bars],
                }
            )
        return out


@dataclass
class TradingDayResult:
    summary: TradingSummary
    live_bars: dict[str, pd.DataFrame]


def run_trading_day(
    config: SessionConfig,
    feed: BarFeed,
    *,
    record_bars: bool = False,
    session_start: datetime | None = None,
    session_end: datetime | None = None,
    now: Callable[[], datetime] | None = None,
    sleeper=None,
    poll_interval_s: float = 2.0,
    stop_event=None,
) -> TradingDayResult:
    """Trade one session of ``config``'s champions over a :class:`~noctis.live.feed.BarFeed`.

    The one TRADING driver: :class:`SessionConfig` is the phase-wide composition, the rest is
    this day's drive. With a ``sleeper`` (plus ``session_start``/``session_end``/``now``)
    the day is **clock-bounded**: wait for the open, pace each poll, break between polls on a
    set ``stop_event``, and flush the feed's held-back tail at a clean close. Without one the
    day is **data-bounded**: drain the feed at CPU speed until it is exhausted (catalog
    replay). Either way every minute group funnels through the same :class:`_TradingSession`,
    so the two ways of ending a day can never diverge on how it is traded. Execution is
    paper-only — orders route through the paper broker.

    What the close report should say comes back *on the summary* (``TradingSummary.events``:
    feed-degraded/recovered transitions, then the refusal/halt line), so the report needs no
    callback and stays byte-identical regardless of verbosity. The one optional sink is
    ``config.on_event`` — the inline console feed (typed
    :class:`~noctis.observability.events.Event`s): the same feed transitions as ``feed``
    events, per-decision ``trade``/``refuse`` from the session, and — on a clock-bounded day,
    every ``heartbeat_polls`` polls (``0`` disables) — a ``heartbeat`` carrying the poll
    count, open-position count, and mark-to-market equity. ``None`` on a bare run, so nothing
    is constructed.

    ``record_bars`` retains every processed bar for close-phase reconciliation — set it when
    the feed is external (live), not when the feed *is* the catalog.
    """
    on_event = config.on_event
    broker = (
        config.broker
        if config.broker is not None
        else PaperBroker(
            fee_model=FeeModel(config.fee_bps), slippage_model=SlippageModel(config.slippage_bps)
        )
    )
    session = _TradingSession(config, sorted(feed.symbols), broker, record_bars=record_bars)

    def emit_feed(msg: str) -> None:
        # A feed-health transition goes to BOTH the report (a plain string on the summary,
        # byte-identical to before) and the console (a level-1 `feed` event, colorized inline).
        session.summary.events.append(msg)
        if on_event is not None:
            on_event(Event("feed", msg, level=1))

    # How the day ends and how polls are paced, resolved once. A clock-bounded day (live)
    # waits for the open, sleeps between polls, and closes with the session; a data-bounded
    # day (replay) drains the feed at CPU speed and only its exhaustion can end it.
    if sleeper is None:

        def session_over() -> bool:
            return False

        def pace() -> None:
            return None

    else:
        if session_start is None or session_end is None or now is None:
            raise ValueError("a clock-bounded day needs session_start, session_end, and now")
        end, now_fn = session_end, now

        def session_over() -> bool:
            return now_fn() >= end

        def pace() -> None:
            sleeper.sleep_until(min(now_fn() + timedelta(seconds=poll_interval_s), end))

        # Wait for the session open (a no-op under a simulated/compressed clock).
        sleeper.sleep_until(session_start)

    was_degraded = False
    while True:
        if stop_event is not None and stop_event.is_set():
            logger.info("trading day: stop requested; breaking between polls")
            break
        if feed.exhausted:  # data-bounded end: the replayed timeline is drained
            break
        if session_over():  # clock-bounded end: the session closed
            break

        completed = feed.poll_once()
        session.summary.polls += 1
        degraded = bool(feed.degraded)
        if degraded and not was_degraded:
            emit_feed("Order emission halted: degraded (delayed) feed")
        elif was_degraded and not degraded:
            emit_feed("Feed recovered: order emission resumed")
        was_degraded = degraded
        # Heartbeat: the "is it alive?" pulse a long unattended session needs — a wall-clock
        # liveness signal, so only a clock-bounded (paced) day emits it; an instant
        # data-bounded replay has no liveness to prove. Level-2 (the -vv feed), every
        # ``heartbeat_polls`` polls; gated so a bare run and a disabled cadence (0)
        # construct nothing.
        if (
            sleeper is not None
            and config.heartbeat_polls > 0
            and on_event is not None
            and (session.summary.polls % config.heartbeat_polls == 0)
        ):
            open_positions = len(broker.positions())  # positions() already drops flats
            equity = broker.equity()
            on_event(
                Event(
                    "heartbeat",
                    f"poll {session.summary.polls}: equity {equity:,.2f}, "
                    f"{open_positions} open position(s)",
                    meta={
                        "polls": session.summary.polls,
                        "equity": equity,
                        "open_positions": open_positions,
                    },
                    level=2,
                )
            )

        if completed:
            session.step(completed, degraded=degraded)

        pace()

    # Flush the feed's held-back tail at a clean session end (not on an abort).
    if stop_event is None or not stop_event.is_set():
        final = feed.flush()
        if final:
            session.step(final, degraded=bool(feed.degraded))

    return TradingDayResult(session.finalize(), session.built_bars())


def run_trading(
    *,
    candidates: list[Candidate],
    bars_by_symbol: dict[str, pd.DataFrame],
    broker: Broker | None = None,
    limits: RiskLimits | None = None,
    is_degraded: Callable[[], bool] = lambda: False,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    live_symbols: list[set[str] | None] | None = None,
    scores: list[float] | None = None,
    min_order_notional: float = 0.0,
    rebalance_band_pct: float = 0.0,
    on_event: Callable[[Event], None] | None = None,
) -> TradingSummary:
    """Evaluate champions over a whole static timeline: the batch convenience wrapper.

    Wraps the timeline in a :class:`~noctis.live.feed.ReplayBarFeed` and drains it through
    :func:`run_trading_day` — one summary out, no pacing, no held-back bars.
    """
    config = SessionConfig(
        candidates=candidates,
        broker=broker,
        limits=limits,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        live_symbols=live_symbols,
        scores=scores,
        min_order_notional=min_order_notional,
        rebalance_band_pct=rebalance_band_pct,
        on_event=on_event,
    )
    return run_trading_day(config, ReplayBarFeed(bars_by_symbol, degraded=is_degraded)).summary
