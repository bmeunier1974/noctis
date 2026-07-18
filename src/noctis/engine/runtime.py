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

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.backtest.pool import evaluation_time_limit
from noctis.champions.promotion import PromotionRules
from noctis.data.types import empty_bars
from noctis.engine.clock import MarketClock
from noctis.engine.close import ReconciliationReport, reconcile_bars, run_close
from noctis.engine.machine import Phase, TradingMachine
from noctis.engine.pacing import BoundedWaiter, RealSleeper, StallGuard, StopFlag
from noctis.engine.report_assembly import SessionActivity, assemble_report
from noctis.engine.research import run_research
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


class Runtime:
    """Drives the RESEARCH → TRADING → CLOSE loop until a time limit or stop request."""

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
    ):
        self.settings = settings
        self.clock = clock
        # Observability sink (a Console, or any ``Event | str`` callable). Default ``None`` keeps a
        # bare run byte-identical: the research feed falls back to its own logger, and the phase
        # hooks below emit nothing. The CLI builds this from ``run``'s ``-v``/``--show-reasoning``.
        self._on_event = on_event
        self.market_lake = market_lake
        self.registry = registry
        self.families = families
        self.memory = memory
        self.proposer = proposer
        # The resolved operator mandate (or None), threaded to each agent research session.
        self.mandate = mandate
        # LLM ideation seam (clientless/no-op when no key or [llm] extra). None → seed-only.
        self.ideator = ideator
        # None ⇒ the settings-resolved location (workspace-derived unless overridden).
        self.reports_dir = reports_dir if reports_dir is not None else settings.reports_dir
        self.research_max_iters = research_max_iters
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
        self.machine = TradingMachine(
            clock,
            on_enter=self._on_phase_enter,
            time_limit_hours=settings.time_limit_hours,
        )
        self._stop = False
        # The event-protocol view (``is_set()``) of ``_stop`` the research/trading loops poll.
        self._stop_event = StopFlag(lambda: self._stop)
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

        # Load catalog bars for the universe (research + replay share them). Re-run at each
        # TRADING entry so the CLOSE-phase T+1 sync becomes visible on multi-day runs.
        self._load_bars()
        self._pipeline_config = self._make_pipeline_config()

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

    def _make_pipeline_config(self) -> PipelineConfig:
        # One geometry/metric home (PipelineConfig.auto), sized from the fit set's shortest
        # series so every symbol gets identical windows (keeps per-symbol scores comparable).
        return PipelineConfig.auto_from_settings(
            self.settings,
            min((len(df) for df in self.research_panel.values()), default=0),
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
        # Everything the day-cycle contributes to the close report, in one accumulator.
        self._cycle = SessionActivity()
        # Bars the live feed actually built this session, retained for close reconciliation.
        self._live_bars: dict[str, pd.DataFrame] = {}

    # --- phases ---
    def _evaluate(self, candidate: Candidate):
        # Same hang insurance as the toolbox's _evaluate: a sequential (workers=1)
        # evaluation has no pool stall guard, so bound it in wall-clock time. The legacy
        # loop absorbs the EvaluationTimeout as a dead end and keeps running.
        with evaluation_time_limit():
            return evaluate(
                candidate,
                self.research_panel,
                config=self._pipeline_config,
                symbol_holdout=self.symbol_holdout,
                workers=self.settings.research.agent.sweep_workers,
                families=self.families,
            )

    def _run_research(self) -> None:
        # The budget is real research time (backtests are wall-clock work even when the
        # session clock jumps), so both loops keep their default wall clock and the
        # wall-clock budget governs. research_max_iters is None in production (unbounded
        # for the legacy loop; the agent loop then uses its config cap); tests pass an
        # explicit cap to bound loops that finish instantly.
        summary = None
        if self.settings.research.mode == "agent":
            summary = self._run_agent_research()
        if summary is None:
            # Legacy proposer/Optuna loop — also the fallback when agent mode has no client.
            summary = run_research(
                proposer=self.proposer,
                evaluate_fn=self._evaluate,
                registry=self.registry,
                rules=self.rules,
                memory=self.memory,
                budget_minutes=self.settings.research_time_budget_minutes,
                stop_event=self._stop_event,
                max_iterations=self.research_max_iters,
                ideate=self.ideator.run if self.ideator is not None else None,
            )
        self._cycle.research_iterations += summary.iterations
        self._cycle.research_promotions += summary.promotions
        self._cycle.research_rejections += summary.rejections
        self._cycle.research_dead_ends += summary.dead_ends
        self._cycle.minted_specs.extend(summary.minted_specs)
        self.result.research_iterations += summary.iterations
        self.result.research_promotions += summary.promotions
        # One completed session toward the periodic memory distillation (runs at CLOSE, not
        # here — a research session's own loop never carries the summarization call).
        from noctis.research.distill import bump_research_session

        bump_research_session(self.settings.state_dir)

    def _run_agent_research(self):
        """One agent-driven session, or ``None`` to fall back to the legacy loop (no key)."""
        from noctis.bootstrap import build_research_session

        # The composition root assembles the same session bundle `noctis research` runs.
        # on_event tees the research feed into the run's console (None ⇒ the loop's own logger
        # sink): `run -v` shows the tool feed, `-vv`/`--show-reasoning` opens think/say — the
        # same streams `noctis research` surfaces, now visible from the day/night loop.
        session = build_research_session(
            settings=self.settings,
            lake=self.market_lake,
            registry=self.registry,
            families=self.families,
            memory=self.memory,
            mandate=self.mandate,
            rules=self.rules,
            on_event=self._on_event,
        )
        if session is None:
            logger.info("research.mode=agent but no research client; using legacy loop")
            return None
        logger.info(
            "agent research session: mandate=%s, metric=%s",
            session.toolbox.mandate_source or "(none)",
            self.settings.promotion.metric,
        )
        return session.run(
            max_iterations=self.research_max_iters,
            stop_event=self._stop_event,
        )

    def _run_trading(self, t: datetime, sleeper) -> None:
        """One TRADING entry: refresh the catalog view, drive the phase, fold its outcome."""
        # Refresh the catalog view first: the CLOSE-phase T+1 sync updates the *lake*, but
        # bars were loaded once at startup — without a reload the newest session would never
        # appear and every later day would look like "no new data". The refresh re-derives
        # the research panel/holdout too; that is safe (deterministic from universe order)
        # and it happens at TRADING entry only, so a research session's view stays frozen
        # for its duration.
        outcome = self.trading.run(t, sleeper, self._load_bars())
        self.result.trades += outcome.orders_submitted
        if outcome.sessions:
            # Equity/positions are the LAST settled session's — the phase already folded a
            # multi-session catch-up that way. Untouched when nothing traded, so a skipped
            # day reports zeros rather than a fictional flat session.
            self._cycle.positions = outcome.positions
            self._cycle.start_equity = outcome.start_equity
            self._cycle.end_equity = outcome.end_equity
            self.result.final_equity = outcome.end_equity
        self._cycle.trades.extend(outcome.trades)
        self._cycle.events.extend(outcome.events)
        self._live_bars = outcome.live_bars

    def _reconcile(self, threshold: float = 0.005):
        """Compare the session's live-built bars against the (T+1 synced) catalog.

        When a live feed ran, each symbol's retained live bars are reconciled against the
        authoritative catalog and the per-symbol results are aggregated (drift over the
        threshold on any symbol flags). Without a live feed there is nothing external to
        compare, so this is a no-op that never flags.
        """
        if not self._live_bars:
            return ReconciliationReport(0, 0.0, 0.0, threshold, flagged=False)
        vendor_bars = self.market_lake.get_bars(
            self.settings.data.dataset, self.schema, list(self._live_bars), 0, 2**63 - 1
        )
        n = 0
        max_drift = 0.0
        weighted_mean = 0.0
        flagged = False
        for sym, live in self._live_bars.items():
            rep = reconcile_bars(live, vendor_bars.get(sym, empty_bars()), threshold=threshold)
            n += rep.n_compared
            max_drift = max(max_drift, rep.max_drift)
            weighted_mean += rep.mean_drift * rep.n_compared
            flagged = flagged or rep.flagged
        mean_drift = weighted_mean / n if n else 0.0
        return ReconciliationReport(n, max_drift, mean_drift, threshold, flagged=flagged)

    def _run_close(self, t: datetime) -> None:
        data = assemble_report(
            as_of=t.astimezone(UTC).date().isoformat(),
            mode=self.mode,
            registry=self.registry,
            memory=self.memory,
            state_dir=self.settings.state_dir,
            session=self._cycle,
        )
        from noctis.research.distill import maybe_distill

        result = run_close(
            report_data=data,
            reports_dir=self.reports_dir,
            memory=self.memory,
            market_lake=self.market_lake,
            registry=self.registry,
            reconcile_fn=self._reconcile,
            tracked=self.tracked,
            # CLOSE owns memory upkeep (reorganize below), so the periodic distillation
            # rides the same isolated-step machinery instead of racing a live session.
            distill_fn=lambda: maybe_distill(self.settings, self.memory),
        )
        if result.report_path:
            self.result.reports.append(result.report_path)
        self.result.cycles_completed += 1
        self._reset_cycle()

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
                self._run_research()
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
                    self._run_trading(sleeper.now(), sleeper)
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
                self._run_close(sleeper.now())
                if max_cycles is not None and self.result.cycles_completed >= max_cycles:
                    self.machine.stop()
                    self.result.stopped_reason = "max_cycles"
                    break

            self.machine.tick(sleeper.now())

        if not self.result.stopped_reason:
            self.result.stopped_reason = (
                "time_limit" if self.machine.time_up(sleeper.now()) else "stopped"
            )
        self.result.history = list(self.machine.history)
        logger.info(
            "runtime stopped: %s after %d cycle(s)",
            self.result.stopped_reason,
            self.result.cycles_completed,
        )
        return self.result

    def _make_waiter(self, sleeper) -> BoundedWaiter:
        """Every between-phase wait goes through one :class:`BoundedWaiter`: it clamps to the
        run's time-limit deadline and wakes promptly on a stop request, so the loop never
        parks against the clock (a weekend, a session close) past the point it should halt."""
        deadline = None
        if self.machine.time_limit_hours is not None and self.machine.start_time is not None:
            deadline = self.machine.start_time + timedelta(hours=self.machine.time_limit_hours)
        return BoundedWaiter(sleeper, stop=lambda: self._stop, deadline=deadline)


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
) -> Runtime:
    """Construct a :class:`Runtime` from settings and the collaborators it needs."""
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
    )
