"""The runtime orchestrator — assembles the full night→day→close loop.

Wires the market clock, state machine, research loop, trading loop, and close phase into one
process. In production a closed market is filled with back-to-back research and the loop
paces to real session boundaries; in simulation there is no real closed time, so it jumps to
the next boundary. Transitions are re-evaluated at each boundary, and a global time limit or a
stop request halts cleanly between phases with all state flushed. Paper-only throughout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from noctis.backtest import PipelineConfig
from noctis.champions.promotion import PromotionRules
from noctis.engine.clock import MarketClock
from noctis.engine.close import ClosePhase
from noctis.engine.machine import Phase, TradingMachine
from noctis.engine.pacing import BoundedWaiter, RealSleeper, StallGuard, StopFlag
from noctis.engine.report_assembly import SessionActivity
from noctis.engine.research_phase import ResearchPanel, ResearchPhase
from noctis.engine.trading_phase import TradingPhase
from noctis.live.risk import RiskLimits
from noctis.observability import Event
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.proposer import CandidateProposer

if TYPE_CHECKING:
    from noctis.data.seam import MarketData
    from noctis.memory.base import Memory

logger = logging.getLogger("noctis.runtime")

# Minimum wall-clock period between back-to-back research sessions while the market is closed.
# Real research sessions (API calls + backtests + journaling) run far longer than this, so it
# never paces them; it is purely a floor that stops a degenerate instant-returning session
# from spinning the CPU / hammering the research API thousands of times a second.
_CLOSED_RESEARCH_MIN_PERIOD = timedelta(seconds=30)


def trading_roster(settings, lake) -> list[str]:
    """The growing trading universe: the config seed plus every lake-tracked ready symbol.

    The config list comes first, order preserved, so the research fit set (the first
    ``fit_set_size`` ready names) stays stable as the agent's discoveries accumulate;
    discovered symbols follow, sorted. The lake IS the persistent store — any symbol the
    research agent ever fetched via ``ensure_data`` is tracked in the coverage registry,
    so it joins the roster with no extra state. Lakes without a coverage registry
    (test fakes) degrade to the config list.

    This feeds the TRADING phase (``_load_bars``) and inventory views. It must never
    shrink under a live champion — champions trade discovered symbols. The *prompt-facing*
    enumeration is the separate, capped :func:`research_focus`.
    """
    seed = list(settings.universe)
    coverage = getattr(lake, "coverage", None)
    if coverage is None:
        return seed
    seen = {s.upper() for s in seed}
    extras = sorted(
        {
            rec.symbol
            for rec in coverage.all()
            if rec.symbol.upper() not in seen and rec.status == "idle" and rec.row_count > 0
        }
    )
    return seed + extras


def research_focus(settings, lake, mandate=None) -> list[str]:
    """What this session *intends* to research: fit set + symbol-holdout names +
    mandate-declared symbols, capped at ``research.focus_size``.

    Feeds the prompt-facing enumerations only (the MARKET REALITY digest and the
    symbol-holdout candidate pool) — never the trading roster. Without a cap, every
    ``ensure_data`` in every session grows every future prompt; discovered-but-unfocused
    symbols stay tradeable (roster) and re-fetchable (``preview_bars``/``list_symbols``).

    The first ``fit_set_size + symbol_holdout_size`` ready roster names come first —
    exactly the runtime's fit-set/holdout window, so the digest describes the symbols
    research actually tunes and gates on. Mandate-declared symbols follow (they may be
    unready — consumers already filter on readiness), then the cap applies.
    """
    ready = [s for s in trading_roster(settings, lake) if lake.check_symbol_ready(s)]
    cfg = settings.research
    focus = ready[: cfg.fit_set_size + cfg.symbol_holdout_size]
    seen = {s.upper() for s in focus}
    for sym in getattr(mandate, "symbols", None) or []:
        if sym.upper() not in seen:
            focus.append(sym)
            seen.add(sym.upper())
    return focus[: cfg.focus_size]


@dataclass
class RuntimeResult:
    history: list[Phase] = field(default_factory=list)
    cycles_completed: int = 0
    research_iterations: int = 0
    research_promotions: int = 0
    trades: int = 0
    reports: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    final_equity: float = 0.0
    # Seconds this process spent **working** in each phase (``{"RESEARCH": …, "TRADING": …}``) —
    # the measurement the run record turns into the run's cumulative research/trading seconds.
    # Waiting is not working: the bounded waits between phases (out a weekend, to a session close)
    # belong to the segment's wall-clock duration and to no phase.
    phase_seconds: dict[str, float] = field(default_factory=dict)


class Runtime:
    """Drives the RESEARCH → TRADING → CLOSE loop until a time limit or stop request.

    It holds one object per phase — :class:`~noctis.engine.research_phase.ResearchPhase`,
    :class:`~noctis.engine.trading_phase.TradingPhase`,
    :class:`~noctis.engine.close.ClosePhase` — and nothing else phase-shaped: what a phase
    *does* is that phase's, and what is left here is the loop itself (the pacing, the stop
    handling, the per-cycle fold and the run's counters). A caller with a stand-in for one
    phase swaps the whole object, so no phase can be driven half through its seam.
    """

    def __init__(
        self,
        *,
        settings,
        clock: MarketClock,
        market_lake: MarketData,
        registry,
        families: FamilyRegistry,
        memory: Memory,
        proposer: CandidateProposer,
        reports_dir: str | None = None,
        research_max_iters: int | None = None,
        schema: str = "ohlcv-1m",
        feed_factory=None,
        sleeper_factory=None,
        ideator=None,
        mandate=None,
        on_event=None,
        on_cycle_close=None,
        prior_runtime_s: float = 0.0,
    ):
        self.settings = settings
        self.clock = clock
        # Observability sink (a Console, or any ``Event | str`` callable). Default ``None`` keeps a
        # bare run byte-identical: the research feed falls back to its own logger, and the phase
        # hooks below emit nothing. The CLI builds this from ``run``'s ``-v``/``--show-reasoning``.
        self._on_event = on_event
        # The run-record checkpoint seam: called once at the end of every CLOSE with the current
        # :class:`RuntimeResult`, so the run's durable record is current after each day-cycle
        # rather than only when the process finally stops. ``None`` on a bare run. The hook is
        # expected not to raise — the run store it is wired to latches its own failures off — so
        # a reporting artifact can never take down a multi-week run.
        self._on_cycle_close = on_cycle_close
        self.market_lake = market_lake
        self.registry = registry
        self.families = families
        self.memory = memory
        # None ⇒ the settings-resolved location (workspace-derived unless overridden).
        self.reports_dir = reports_dir if reports_dir is not None else settings.reports_dir
        self.schema = schema
        self._sleeper_factory = sleeper_factory or (lambda _start: RealSleeper())

        self.mode = settings.mode
        self.rules = PromotionRules.from_settings(settings)
        self.limits = RiskLimits(
            settings.risk.max_position_pct,
            settings.risk.max_gross_exposure_pct,
            settings.risk.max_daily_loss_pct,
        )
        # Wire the machine's phase seam so each RESEARCH→TRADING→CLOSE transition announces itself
        # inline (guarded on ``_on_event`` — a quiet run emits nothing). This frames the interleaved
        # research/trading feeds; entry is the only hook, so each transition is exactly one event.
        # Two ceilings, one stop. ``time_limit_hours`` bounds this process (how long tonight lasts);
        # ``run_limit_hours`` bounds the whole run across every stop/resume, measured against the
        # runtime its earlier segments already spent (``prior_runtime_s``, read off the run record
        # by the composition root). Both stop cleanly between phases through the machine's own
        # terminal move — the run-level cap deliberately adds no second shutdown route.
        self.machine = TradingMachine(
            clock,
            on_enter=self._on_phase_enter,
            time_limit_hours=settings.time_limit_hours,
            run_limit_hours=settings.run_limit_hours,
            prior_runtime_s=prior_runtime_s,
        )
        self._stop = False
        # The event-protocol view (``is_set()``) of ``_stop`` the research/trading loops poll.
        self._stop_event = StopFlag(lambda: self._stop)
        # The RESEARCH dispatch behind its own seam (pick the agent or legacy path, drive it,
        # count the session): assembled once with the collaborators a session needs, driven at
        # each RESEARCH entry with that entry's freshly rebuilt panel — the bars are the
        # argument, so a session can no more be run on a stale panel than a trading day can.
        self.research = ResearchPhase(
            settings=settings,
            market_lake=market_lake,
            registry=registry,
            families=families,
            memory=memory,
            proposer=proposer,
            rules=self.rules,
            mandate=mandate,
            ideator=ideator,
            research_max_iters=research_max_iters,
            on_event=on_event,
            stop_event=self._stop_event,
        )
        # The TRADING dispatch behind its own seam (assemble the session collaborators, pick
        # the live/replay driver, run the catch-up loop, fold): assembled once, driven at
        # each TRADING entry with that entry's freshly loaded catalog bars. ``feed_factory``
        # defaults inside the phase; tests inject fakes so no network is ever touched.
        self.trading = TradingPhase(
            settings=settings,
            clock=clock,
            registry=registry,
            families=families,
            limits=self.limits,
            feed_factory=feed_factory,
            on_event=on_event,
            stop_event=self._stop_event,
        )
        # The CLOSE entry behind its own seam: the day's evidence (sync, integrity, reconcile,
        # account, mark) is finished before the report is rendered, so what the close discovers
        # reaches the files it writes. Assembled once, driven at each CLOSE with that cycle.
        from noctis.research.distill import maybe_distill

        self.close = ClosePhase(
            settings=settings,
            reports_dir=self.reports_dir,
            memory=memory,
            registry=registry,
            market_lake=market_lake,
            schema=schema,
            distill_fn=lambda: maybe_distill(self.settings, self.memory),
        )

        # Load catalog bars for the universe (research + replay share them). Re-run at each
        # RESEARCH and TRADING entry so the CLOSE-phase T+1 sync becomes visible on
        # multi-day runs.
        self._load_bars()

        # Per-cycle accumulators.
        self._reset_cycle()
        self.result = RuntimeResult()

    # --- setup ---
    def _load_bars(self) -> dict[str, pd.DataFrame]:
        universe = trading_roster(self.settings, self.market_lake)
        ready = [s for s in universe if self.market_lake.check_symbol_ready(s)]
        self.trading_bars: dict[str, pd.DataFrame] = {}
        for sym in ready:
            bars = self.market_lake.get_bars(
                self.settings.data.dataset, self.schema, [sym], 0, 2**63 - 1
            )[sym]
            if len(bars) > 0:
                self.trading_bars[sym] = bars
        # Research panel: the first ``fit_set_size`` ready universe symbols are the fit set
        # (tuning + election), the next ``symbol_holdout_size`` are the symbol holdout —
        # scored but never tuned/selected on. Both are deterministic from universe order,
        # fixed for the entire run, and identical for every candidate; a rotating holdout
        # would leak every symbol into selection after a few iterations.
        ordered = list(self.trading_bars)
        fit_n = self.settings.research.fit_set_size
        holdout_n = self.settings.research.symbol_holdout_size
        self.research_panel: dict[str, pd.DataFrame] = {
            s: self.trading_bars[s] for s in ordered[:fit_n]
        }
        self.symbol_holdout: dict[str, pd.DataFrame] = {
            s: self.trading_bars[s] for s in ordered[fit_n : fit_n + holdout_n]
        }
        self.tracked = [(self.settings.data.dataset, self.schema, s) for s in self.trading_bars]
        # Returned so the TRADING entry consumes the same view it just refreshed — the phase
        # cannot be driven on stale bars by construction.
        return self.trading_bars

    def _research_panel(self) -> ResearchPanel:
        """The frozen panel this RESEARCH entry researches on, off a fresh catalog read.

        Rebuilt at entry exactly as the TRADING view is, so a session that follows a CLOSE
        picks up what that close's T+1 sync brought in instead of the bars startup happened
        to see. The geometry has one home (``PipelineConfig.auto_from_settings``) and is sized
        from the fit set's shortest series, so every symbol gets identical windows and
        per-symbol scores stay comparable.
        """
        self._load_bars()
        return ResearchPanel(
            fit=self.research_panel,
            symbol_holdout=self.symbol_holdout,
            config=PipelineConfig.auto_from_settings(
                self.settings,
                min((len(df) for df in self.research_panel.values()), default=0),
            ),
        )

    def has_data(self) -> bool:
        return any(len(df) >= 80 for df in self.research_panel.values())

    # --- lifecycle ---
    def request_stop(self) -> None:
        self._stop = True

    def _on_phase_enter(self, phase: Phase) -> None:
        """Emit a level-1 ``phase`` Event as the machine enters each phase.

        This is the ``run`` command's replacement for the raw ``phase=… | cycle=…`` INFO
        heartbeat as the ``-v`` framing: a clean banner that carries the phase and the cycle it
        opens, so the research (P3) and trading (P4) feeds that follow read as belonging to it.
        A no-op when no console is wired (``on_event=None``), so a bare run stays silent.
        """
        if self._on_event is None:
            return
        cycle = self.result.cycles_completed if hasattr(self, "result") else 0
        self._on_event(
            Event(
                "phase",
                f"{phase.value} · cycle {cycle}",
                meta={"phase": phase.value, "cycle": cycle},
                level=1,
            )
        )

    def _reset_cycle(self) -> None:
        # Everything the day-cycle contributes to the close report, in one accumulator — the
        # live-built bars the close reconciles included.
        self._cycle = SessionActivity()

    # --- main loop ---
    def run(self, start: datetime | None = None, max_cycles: int | None = None) -> RuntimeResult:
        t = start or self.clock.now()
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        # One pacer for the whole run is the single clock the loop advances by. In production
        # (RealSleeper) ``sleep_until`` blocks in wall-clock time between phases, so the loop
        # tracks the real market calendar: it researches back-to-back through the closed
        # market (a night, a weekend) and only trades once the session is genuinely open.
        # Under a SimulatedSleeper (tests, replay) there is no real closed time to fill, so
        # the loop jumps straight to the open and the identical loop runs at CPU speed.
        sleeper = self._sleeper_factory(t)
        self.machine.start(t)
        waiter = self._make_waiter(sleeper)
        guard = StallGuard()

        while self.machine.state is not Phase.STOPPED:
            if guard.stalled(sleeper.now()):
                self.result.stopped_reason = "guard"
                break
            if self._stop:
                self.machine.stop()
                self.result.stopped_reason = "stop_requested"
                break

            phase = self.machine.state
            logger.info(
                "phase=%s | cycle=%d | t=%s",
                phase.value,
                self.result.cycles_completed,
                sleeper.now().isoformat(),
            )
            if phase is Phase.RESEARCH:
                research_start = sleeper.now()
                # The panel is rebuilt here, at the entry, and handed in: the phase holds the
                # collaborators, never the bars, so this session's view is this session's.
                summary = self.research.run(self._research_panel())
                self._cycle.fold_research(summary)
                self.result.research_iterations += summary.iterations
                self.result.research_promotions += summary.promotions
                self._count_phase_time(phase, research_start, sleeper.now())
                # If research overran into an open session, fall through so the machine can
                # trade the remaining hours instead of skipping the day. While the market is
                # still closed:
                #   • real-time pacing — keep the loop in RESEARCH and go straight into the
                #     next session, so the closed stretch (a night, a weekend) is filled with
                #     back-to-back research rather than an idle wait. Real wall-clock time
                #     advances through each session until the open; the floor only guards
                #     against a degenerate instant-returning session busy-spinning.
                #   • simulated clock — research does not advance it, so jump to the next open
                #     or the loop could never reach it.
                if not self.clock.is_open(sleeper.now()):
                    if waiter.wall_clock:
                        waiter.wait_until(research_start + _CLOSED_RESEARCH_MIN_PERIOD)
                    else:
                        waiter.wait_until(self.clock.next_open(sleeper.now()))
            elif phase is Phase.TRADING:
                # The market must be genuinely open to trade. Under real-time pacing we only
                # reach here after sleeping to the open, so this normally holds; the guard is
                # what keeps a start-while-closed (e.g. a Saturday) from ever emitting orders.
                if self.clock.is_open(sleeper.now()):
                    trading_start = sleeper.now()
                    # The catalog view is refreshed here, at the entry, and handed in: the
                    # CLOSE-phase T+1 sync updates the *lake*, but bars were loaded once at
                    # startup — without a reload the newest session would never appear and every
                    # later day would look like "no new data". RESEARCH refreshes the same way
                    # for the same reason; each phase reads the view it just loaded, so neither
                    # can be driven on the other's bars.
                    outcome = self.trading.run(trading_start, sleeper, self._load_bars())
                    self.result.trades += outcome.orders_submitted
                    if outcome.sessions:
                        # The run's headline equity is the LAST settled session's; untouched
                        # when nothing traded, so a skipped day reports the standing number
                        # rather than a fictional zero.
                        self.result.final_equity = outcome.end_equity
                    self._cycle.fold_trading(outcome)
                    self._count_phase_time(phase, trading_start, sleeper.now())
                    # Advance to the session close. The live driver already ran the clock to
                    # the close; the instant replay driver has not, so pace to it here —
                    # bounded, like every between-work wait, so a short time limit stops the
                    # run instead of parking it against the clock for the rest of the session.
                    if self.clock.is_open(sleeper.now()):
                        waiter.wait_until(self.clock.next_close(sleeper.now()))
                else:
                    logger.info("trading skipped: market closed at %s", sleeper.now().isoformat())
                    self._cycle.events.append("Trading phase skipped — market closed")
            elif phase is Phase.CLOSE:
                close_start = sleeper.now()
                closed = self.close.run(close_start, self._cycle, tracked=self.tracked)
                if closed.report_path:
                    self.result.reports.append(closed.report_path)
                self.result.cycles_completed += 1
                self._reset_cycle()
                if self._on_cycle_close is not None:
                    self._on_cycle_close(self.result)
                self._count_phase_time(phase, close_start, sleeper.now())
                if max_cycles is not None and self.result.cycles_completed >= max_cycles:
                    self.machine.stop()
                    self.result.stopped_reason = "max_cycles"
                    break

            self.machine.tick(sleeper.now())

        if not self.result.stopped_reason:
            # Which ceiling ended it — the per-process time limit or the run-level cap — decided in
            # the one place both live, so the reason on the segment always names the real cause.
            self.result.stopped_reason = self.machine.limit_hit(sleeper.now()) or "stopped"
        self.result.history = list(self.machine.history)
        logger.info(
            "runtime stopped: %s after %d cycle(s)",
            self.result.stopped_reason,
            self.result.cycles_completed,
        )
        return self.result

    def _count_phase_time(self, phase: Phase, started, ended) -> None:
        """Add one phase body's wall-clock seconds to this segment's tally.

        Measured on the pacing seam's clock (the loop's only clock) and around the *work* alone —
        the waits that follow a phase are pacing, not research or trading, and a total that folded
        them in would make "100 research hours" mean "100 hours of having been switched on"."""
        elapsed = (ended - started).total_seconds()
        current = self.result.phase_seconds.get(phase.value, 0.0)
        self.result.phase_seconds[phase.value] = round(current + max(0.0, elapsed), 3)

    def _make_waiter(self, sleeper) -> BoundedWaiter:
        """Every between-phase wait goes through one :class:`BoundedWaiter`: it clamps to the
        run's deadline — the earlier of the per-process time limit and the run-level cap — and
        wakes promptly on a stop request, so the loop never parks against the clock (a weekend, a
        session close) past the point it should halt."""
        return BoundedWaiter(sleeper, stop=lambda: self._stop, deadline=self.machine.deadline())


def build_runtime(
    settings,
    *,
    market_lake,
    memory,
    clock: MarketClock | None = None,
    registry=None,
    families: FamilyRegistry | None = None,
    proposer: CandidateProposer | None = None,
    reports_dir: str | None = None,
    research_max_iters: int | None = None,
    seed: int = 0,
    feed_factory=None,
    sleeper_factory=None,
    mandate=None,
    on_event=None,
    on_cycle_close=None,
    prior_runtime_s: float = 0.0,
) -> Runtime:
    """Construct a :class:`Runtime` from settings and the collaborators it needs.

    ``prior_runtime_s`` is the runtime the run's **earlier segments** already spent, which the
    run-level cap (``settings.run_limit_hours``) is measured against. It defaults to 0 — a fresh
    run, and every caller that has no run record in hand.
    """
    from noctis.bootstrap import build_families
    from noctis.champions import build_registry

    # One hydration (seeds → persisted spec-families → library files), ordered inside
    # build_families, so a promoted family's class exists before any champion builds.
    families = families or build_families(settings)
    clock = clock or MarketClock(settings.session.calendar, settings.session.timezone)
    registry = registry or build_registry(settings)
    proposer = proposer or CandidateProposer(families, seed=seed)

    # LLM ideation seam. build_ideator returns a clientless (no-op) Ideator when the [llm]
    # extra is absent or the model's provider has no key, so a bare run mints nothing.
    from noctis.research import build_ideator

    ideator = build_ideator(
        settings=settings,
        registry=registry,
        families=families,
        proposer=proposer,
        memory=memory,
        state_dir=settings.state_dir,
    )
    return Runtime(
        settings=settings,
        clock=clock,
        market_lake=market_lake,
        registry=registry,
        families=families,
        memory=memory,
        proposer=proposer,
        reports_dir=reports_dir,
        research_max_iters=research_max_iters,
        feed_factory=feed_factory,
        sleeper_factory=sleeper_factory,
        ideator=ideator,
        mandate=mandate,
        on_event=on_event,
        on_cycle_close=on_cycle_close,
        prior_runtime_s=prior_runtime_s,
    )
