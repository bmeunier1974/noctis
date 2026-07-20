"""One TRADING entry, behind its own interface.

Dispatching a TRADING phase is one cohesive job: assemble the session collaborators (the
continuous paper account, the forward ledger, one :class:`~noctis.engine.trading_day.TradingDay`
runner), resolve the live or replay bar-feed driver, run the catch-up loop, and fold every
settled session into one :class:`TradingOutcome`. :class:`TradingPhase` owns that job. The
runtime hands ``run`` the freshly loaded catalog bars and copies the outcome into its report
accumulators; tests drive ``run`` directly with fake bars and feeds — the interface is the
test surface. Paper orders only, like everything downstream of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from noctis.broker.persistence import AccountStore
from noctis.broker.seam import FeeModel, Fill, SlippageModel
from noctis.engine.forward_ledger import ForwardLedger
from noctis.engine.sessions import SessionLedger, sessions_present, slice_session, unseen_sessions
from noctis.engine.trading_day import SessionOutcome, TradingDay
from noctis.live.feed import ReplayBarFeed
from noctis.observability import Event
from noctis.reporting.report import Trade

if TYPE_CHECKING:
    from noctis.broker.paper import PaperBroker
    from noctis.engine.clock import MarketClock
    from noctis.live.node import TradingSummary
    from noctis.live.risk import RiskLimits
    from noctis.strategies.families import FamilyRegistry

logger = logging.getLogger("noctis.runtime")


def resolve_trading_driver(settings) -> str:
    """The TRADING driver that will run: ``"live"`` (stream yfinance) or ``"replay"`` (catalog).

    ``trading.execution``: ``auto`` (default) derives from ``data.provider`` — yfinance → live,
    anything else → replay (today's behavior); ``replay``/``live`` force the choice. A forced
    ``live`` the provider can't honor still resolves to ``"live"`` here — the phase attempts
    it and logs a WARNING before falling back to replay, so the unhonored intent is never
    silent. One helper so the TRADING dispatch and ``noctis status`` can never disagree.
    """
    execution = settings.trading.execution
    if execution in ("replay", "live"):
        return execution
    return "live" if settings.data.provider == "yfinance" else "replay"


def _fill_rationale(fill: Fill, orphaned: set[str]) -> str:
    """Why this fill happened, for the report's trade rows.

    An orphan symbol has no strategy this session, so any fill on it IS the flatten; a
    non-``target`` reason is a protective exit the engine enforced; the rest are the
    champion's own decisions.
    """
    if fill.symbol in orphaned:
        return "orphan flatten"
    if fill.reason != "target":
        return f"protective exit ({fill.reason})"
    return "champion signal"


def _default_feed_factory(*, symbols):
    """Build the live yfinance feed for the trading roster (free, delayed; data only)."""
    from noctis.data.yfinance import build_yfinance_feed

    return build_yfinance_feed(symbols=symbols)


@dataclass
class SessionRecord:
    """One settled session, kept on the outcome as per-session evidence.

    ``bars`` is the replay slice the session traded — empty on the live path, whose
    externally built bars land on the outcome's ``live_bars`` instead.
    """

    day: date
    bars: dict[str, pd.DataFrame]
    summary: TradingSummary
    fills: list[Fill]


@dataclass
class TradingOutcome:
    """Everything one TRADING entry hands back for the run/report accumulators.

    ``sessions`` carries the per-session evidence in traded order (a replay catch-up settles
    several); the scalar fields are the phase-level fold the close report reads —
    equity/positions from the *last* session, trades and report events across all of them.
    Empty ``sessions`` means nothing traded (account refusal, no new data, or an empty
    champion board); ``events`` still says why whenever there is a reason worth reporting.
    ``broker`` is the one continuous paper account every session settled on (``None`` only
    when the persisted account refused to load).
    """

    sessions: list[SessionRecord] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    start_equity: float = 0.0
    end_equity: float = 0.0
    orders_submitted: int = 0
    live_bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    broker: PaperBroker | None = None


class TradingPhase:
    """Assemble, drive, and fold one TRADING entry (live or replay, paper orders only)."""

    def __init__(
        self,
        *,
        settings,
        clock: MarketClock,
        registry,
        families: FamilyRegistry,
        limits: RiskLimits,
        feed_factory=None,
        on_event=None,
        stop_event=None,
    ):
        self.settings = settings
        self.clock = clock
        self.registry = registry
        self.families = families
        self.limits = limits
        # Live-feed seam. The default keeps production honest; tests inject fakes so no
        # network is ever touched. A bare (non-yfinance) run never builds a feed.
        self._feed_factory = feed_factory or _default_feed_factory
        self._on_event = on_event
        self._stop_event = stop_event

    def run(self, t: datetime, sleeper, bars: dict[str, pd.DataFrame]) -> TradingOutcome:
        """Dispatch one TRADING entry over ``bars`` (the entry's freshly loaded catalog view)."""
        outcome = TradingOutcome()
        # One continuous paper account (equity AND open positions carried across sessions,
        # persisted in state/). Built exactly once here and threaded into the one TradingDay
        # runner, so the live driver's replay fallback cannot double-load. A corrupt file
        # refuses to trade rather than silently restarting at 100k — that file is the
        # cumulative forward track record.
        store = AccountStore(Path(self.settings.state_dir) / "paper_account.json")
        # Paper fills charge the operator-configured cost (#23) — the same fee/slippage the
        # backtest promoted the champion under, so a live-holdout fill can never be cheaper
        # than the arena the strategy earned its slot in.
        fee_bps = self.settings.backtest.fee_bps
        slippage_bps = self.settings.backtest.slippage_bps
        try:
            broker = store.load(
                fee_model=FeeModel(fee_bps), slippage_model=SlippageModel(slippage_bps)
            )
        except RuntimeError as exc:
            logger.error("trading refused: %s", exc)
            outcome.events.append(f"Trading refused — {exc}")
            return outcome
        outcome.broker = broker
        forward = ForwardLedger(Path(self.settings.state_dir) / "forward_ledger.json")
        forward.load()  # never raises: a corrupt ledger warns and starts empty, never blocks
        day_runner = TradingDay(
            broker=broker,
            store=store,
            ledger=SessionLedger(Path(self.settings.state_dir) / "trading_sessions.json"),
            forward=forward,
            registry=self.registry,
            families=self.families,
            limits=self.limits,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_order_notional=self.settings.trading.min_order_notional,
            rebalance_band_pct=self.settings.trading.rebalance_band_pct,
            # The inline console feed (feed/trade/refuse/heartbeat; None on a bare run so
            # nothing is constructed). Report lines come back on each session's summary.
            on_event=self._on_event,
            heartbeat_polls=self.settings.observability.heartbeat_polls,
        )
        # An entirely empty champion board runs no session at all — so carried positions are
        # not flattened as orphans until the next session with any champion on the board.
        if not day_runner.candidates or not bars:
            return outcome
        # Make the driver choice loud (plan 4): reading `data.provider: databento` gave no
        # signal that TRADING was replaying the catalog rather than streaming — the single most
        # surprising fact in the 2026-07-07 diagnosis. State the resolved driver every entry.
        driver = resolve_trading_driver(self.settings)
        provider = self.settings.data.provider
        if driver == "live":
            if provider != "yfinance":
                logger.warning(
                    "TRADING execution=live but data.provider=%s has no live feed — attempting "
                    "anyway; will fall back to catalog replay if the feed can't be built.",
                    provider,
                )
            else:
                logger.info("TRADING will stream the live yfinance feed.")
            self._run_live(t, sleeper, day_runner, bars, outcome)
        else:
            logger.warning(
                "TRADING will REPLAY the catalog live-holdout — no live feed (data.provider=%s). "
                "Set data.provider=yfinance for a live feed.",
                provider,
            )
            self._run_replay(day_runner, bars, outcome)
        return outcome

    def _run_replay(
        self, day_runner: TradingDay, bars: dict[str, pd.DataFrame], outcome: TradingOutcome
    ) -> None:
        """Replay path: one data-bounded TradingDay per unseen catalog session, oldest first."""
        tz = self.clock.tz
        last_traded = day_runner.ledger.load()
        present = sessions_present(bars, tz)
        to_trade, skipped = unseen_sessions(
            present, last_traded, self.settings.trading.max_catchup_sessions
        )
        if not to_trade:
            newest = present[-1] if present else None
            logger.warning(
                "trading skipped: no unseen session (lake newest=%s, last traded=%s)",
                newest,
                last_traded,
            )
            outcome.events.append(
                f"Trading skipped — no new session data "
                f"(newest in lake {newest}, last traded {last_traded})"
            )
            return
        if skipped:
            outcome.events.append(f"Skipped {skipped} stale sessions older than {to_trade[0]}")
        for day in to_trade:
            session_bars = slice_session(bars, day, tz)
            nsym = sum(1 for df in session_bars.values() if len(df) > 0)
            nbars = sum(len(df) for df in session_bars.values())
            logger.info("TRADING replay: session=%s symbols=%d bars=%d", day, nsym, nbars)
            # A per-session `phase` banner (P4): a catch-up replays several sessions in one
            # TRADING phase, so each announces itself inline instead of the loop emitting one
            # INFO for the batch. Guarded — a bare run stays silent.
            if self._on_event is not None:
                self._on_event(
                    Event(
                        "phase",
                        f"TRADING replay · {day} · {nsym} symbol(s) · {nbars} bars",
                        meta={"session": str(day), "symbols": nsym, "bars": nbars},
                        level=1,
                    )
                )
            # The carried broker is the one continuous account; a fresh _TradingSession/
            # RiskManager per session date keeps the "daily" loss limit daily even during
            # catch-up, anchored to that day's carried starting equity.
            feed = ReplayBarFeed(session_bars)
            settled = day_runner.run(feed=feed, day=day)
            self._fold(outcome, day, session_bars, settled)

    def _run_live(
        self,
        t: datetime,
        sleeper,
        day_runner: TradingDay,
        bars: dict[str, pd.DataFrame],
        outcome: TradingOutcome,
    ) -> None:
        """Live path: one clock-bounded TradingDay off the yfinance feed; paper orders only."""
        try:
            feed = self._feed_factory(symbols=sorted(bars))
        except Exception as exc:  # noqa: BLE001 — never fail the day on a feed misconfig
            logger.exception("live feed unavailable; falling back to catalog replay")
            outcome.events.append(f"Live feed unavailable ({exc}); traded on replay")
            self._run_replay(day_runner, bars, outcome)
            return
        day = t.astimezone(self.clock.tz).date()
        settled = day_runner.run(
            feed=feed,
            day=day,
            record_bars=True,  # external bars are retained for close-phase reconciliation
            session_start=t,
            session_end=self.clock.next_close(t),
            now=sleeper.now,
            sleeper=sleeper,
            poll_interval_s=self.settings.live_feed.poll_interval_s,
            stop_event=self._stop_event,
        )
        outcome.live_bars = settled.live_bars
        self._fold(outcome, day, {}, settled)

    def _fold(
        self,
        outcome: TradingOutcome,
        day: date,
        session_bars: dict[str, pd.DataFrame],
        settled: SessionOutcome,
    ) -> None:
        """Fold one settled session — its OWN fills only — into the phase outcome. The
        carried broker accumulates fills across a catch-up, and re-recording earlier ones
        would double-count trades; ``SessionOutcome.fills`` is already per-session."""
        summary = settled.summary
        outcome.sessions.append(SessionRecord(day, session_bars, summary, settled.fills))
        outcome.orders_submitted += summary.orders_submitted
        outcome.positions = summary.positions
        outcome.start_equity = summary.start_equity
        outcome.end_equity = summary.final_equity
        orphaned = set(summary.orphans_flattened)
        for fill in settled.fills:
            outcome.trades.append(
                Trade(
                    fill.symbol,
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    _fill_rationale(fill, orphaned),
                )
            )
        # The summary carries its own report lines (feed transitions, refusal/halt) in
        # occurrence order — fold them in verbatim.
        outcome.events.extend(summary.events)
