"""One TRADING entry, behind its own interface — traded, settled, and folded.

Dispatching a TRADING phase is one cohesive job, and this is the one module that does it:
assemble the session collaborators (the continuous paper account, the forward ledger, one
:class:`TradingDay` runner), resolve the live or replay bar-feed driver, run the catch-up loop,
and fold every settled session into one :class:`TradingOutcome`. :class:`TradingPhase` owns the
entry; :class:`TradingDay` owns one session end-to-end. The runtime hands ``run`` the freshly
loaded catalog bars and copies the outcome into its report accumulators; tests drive ``run``
directly with fake bars and feeds — the interface is the test surface. Paper orders only, like
everything downstream of it.

The settle order is the crash-safety contract, identical for live and replay days:

1. **Trade** — :func:`~noctis.live.node.run_trading_day` over the session's
   :class:`~noctis.live.feed.BarFeed` (live yfinance or a catalog replay slice).
2. **Attribute** — fold the session's realized P&L into the per-champion forward ledger.
   Derived evidence, never the money state: a ledger hiccup logs and continues.
3. **Account first** — persist the continuous paper account (`state/paper_account.json`).
4. **High-water mark second** — advance the session ledger (`state/trading_sessions.json`).

A crash between 3 and 4 leaves the ledger behind the account, which re-trades that session
(safe — strategies re-decide from carried positions) rather than silently skipping it.
Before the two paths were unified the live path never advanced the high-water mark at all, so a
live-traded day followed by a replay day was re-traded on the carried account; one settle path
for both drivers is what closes that gap.

Then the settle **folds** what the session did straight into the phase's one
:class:`TradingOutcome` and hands back the stamped :class:`~noctis.live.node.TradingSummary`
that is the session's evidence — no per-session wrapper shape in between.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from noctis.backtest import Candidate
from noctis.broker.persistence import AccountStore
from noctis.broker.seam import FeeModel, Fill, SlippageModel
from noctis.champions.assignment import assign_indices, slot_inputs
from noctis.engine.forward_ledger import ForwardLedger, champion_key
from noctis.engine.sessions import SessionLedger, sessions_present, slice_session, unseen_sessions
from noctis.live.feed import BarFeed, ReplayBarFeed
from noctis.live.node import SessionConfig, run_trading_day
from noctis.observability import Event
from noctis.reporting.report import Trade

if TYPE_CHECKING:
    from noctis.broker.paper import PaperBroker
    from noctis.engine.clock import MarketClock
    from noctis.live.node import TradingDayResult, TradingSummary
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


def champion_slots(registry) -> tuple[list[Candidate], list[set[str] | None], list[float], list]:
    """Champions plus their attached live symbol sets, election scores, and the source
    registry entries (all in registry order, so index ``i`` is the same champion across
    every list — the forward attribution keys off the entries).

    ``None`` in the sets list marks a legacy champion (no persisted symbols) that is
    eligible for the whole universe; scores let the driver's assignment give each symbol
    its best-scoring eligible champion.
    """
    entries = registry.list()
    candidates = [Candidate(e.family, e.params) for e in entries]
    sets, scores = slot_inputs(entries)
    return candidates, sets, scores, entries


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


def _fill_stamp(fill: Fill) -> str | None:
    """The fill's event time in the record's one timestamp shape, or ``None`` when it has none.

    ``ts_event`` is UTC nanoseconds — the bar the fill happened on (replay) or the minute the live
    feed closed. A zero is "the broker was never marked with a time", which no real fill carries,
    and it is reported as unknown rather than as the epoch.
    """
    if not fill.ts_event:
        return None
    moment = datetime.fromtimestamp(fill.ts_event / 1_000_000_000, tz=UTC)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond // 1000:03d}Z"


def _default_feed_factory(*, symbols):
    """Build the live yfinance feed for the trading roster (free, delayed; data only)."""
    from noctis.data.yfinance import build_yfinance_feed

    return build_yfinance_feed(symbols=symbols)


@dataclass
class TradingOutcome:
    """Everything one TRADING entry hands back for the run/report accumulators.

    ``sessions`` carries the per-session evidence in traded order (a replay catch-up settles
    several): each entry is the driver's own :class:`~noctis.live.node.TradingSummary`,
    stamped by the settle with the date it speaks for, so a catch-up's sessions are
    inspectable without a wrapper shape around them. The scalar fields are the phase-level
    fold the close report reads — equity/positions from the *last* session, trades and report
    events across all of them. Empty ``sessions`` means nothing traded (account refusal, no
    new data, or an empty champion board); ``events`` still says why whenever there is a
    reason worth reporting.

    What the outcome carries is **facts, never a collaborator**: the paper account the
    sessions settled on is durable (``state/paper_account.json``), and CLOSE reads that file
    once rather than being handed the live broker object.
    """

    sessions: list[TradingSummary] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    start_equity: float = 0.0
    end_equity: float = 0.0
    orders_submitted: int = 0
    live_bars: dict[str, pd.DataFrame] = field(default_factory=dict)


class TradingDay:
    """The one place a TRADING session is traded, settled *and* folded into the outcome.

    Built once per TRADING entry (champion slots resolve once, the forward ledger loads
    once) and run once per session date — a replay catch-up runs it for each unseen day,
    a live day runs it once with pacing. Whatever the feed, the settle order above holds.
    """

    def __init__(
        self,
        *,
        broker: PaperBroker,
        store: AccountStore,
        ledger: SessionLedger,
        forward: ForwardLedger,
        registry,
        families: FamilyRegistry,
        limits: RiskLimits,
        fee_bps: float = 1.0,
        slippage_bps: float = 1.0,
        min_order_notional: float = 0.0,
        rebalance_band_pct: float = 0.0,
        on_event=None,
        heartbeat_polls: int = 0,
    ):
        self.broker = broker
        self.store = store
        self.ledger = ledger
        self.forward = forward
        # The per-side fill costs this day trades under: the carried ``broker`` above already
        # embeds them in its fee/slippage models, and every trade row this day folds out
        # reports them from here — the charged cost and the reported cost are one number.
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.candidates, live_symbols, scores, self.entries = champion_slots(registry)
        # One SessionConfig for the whole TRADING phase — a catch-up drives it once per
        # session date. The attribution below reads the same eligibility/score inputs the
        # trading assignment consumed, so the two cannot drift.
        self.config = SessionConfig(
            candidates=self.candidates,
            families=families,
            live_symbols=live_symbols,
            scores=scores,
            broker=broker,
            limits=limits,
            # The configured per-side fill costs (#23) — here, the fresh-broker fallback's copy
            # of the same one source of truth, so the two can never disagree.
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            min_order_notional=min_order_notional,
            rebalance_band_pct=rebalance_band_pct,
            on_event=on_event,
            heartbeat_polls=heartbeat_polls,
            # Who opened each open position, so an orphan flatten can name the displaced
            # champion in its event. Display-only; empty on pre-holder ledger files.
            position_holders={sym: h["key"] for sym, h in forward.holders.items()},
        )

    def run(
        self,
        feed: BarFeed,
        day: date,
        *,
        outcome: TradingOutcome,
        record_bars: bool = False,
        session_start: datetime | None = None,
        session_end: datetime | None = None,
        now=None,
        sleeper=None,
        poll_interval_s: float = 2.0,
        stop_event=None,
    ) -> TradingSummary:
        """Trade session ``day`` from ``feed``, settle it (attribute → account → mark), and
        fold it into ``outcome``; the stamped summary is the session's evidence, returned."""
        fills_before = len(self.broker.fills)
        realized_before = dict(self.broker.realized_pnl_by_symbol)
        # Who held what *coming into* this session — read before the settle rewrites it, because a
        # flatten drops the holder of the position it just closed and that closing fill still
        # belongs to the champion that opened it.
        holders_before = {sym: entry["family"] for sym, entry in self.forward.holders.items()}
        result = run_trading_day(
            self.config,
            feed,
            record_bars=record_bars,
            session_start=session_start,
            session_end=session_end,
            now=now,
            sleeper=sleeper,
            poll_interval_s=poll_interval_s,
            stop_event=stop_event,
        )
        # Stamp the driver's own summary with the date it settled: the summary IS the
        # per-session evidence the phase hands on, so which session it speaks for travels
        # with it rather than in a wrapper around it. The settle is the one place that
        # knows the date — the drivers only see a feed.
        result.summary.session = day
        # Attribute this session's realized P&L to the champions that earned it (plan 5).
        # Guarded so a ledger failure never blocks the account save below — the forward
        # record is derived evidence, not the money state.
        try:
            self._attribute(feed.symbols, realized_before, day)
            self._update_holders()
            self.forward.save()
        except Exception:  # noqa: BLE001 — evidence upkeep must never fail the session
            logger.exception("forward attribution failed for %s; continuing", day)
        # Persist account first, high-water mark second, both only after the session ran:
        # a crash between the two leaves the ledger behind the account, which re-trades
        # that session (safe) rather than silently skipping it.
        self.store.save(self.broker, day)
        self.ledger.save(day)
        # This session's OWN fills only: the carried broker accumulates fills across a
        # catch-up, and re-recording earlier ones would double-count trades.
        self._fold(
            outcome,
            result,
            self.broker.fills[fills_before:],
            self._assignment(feed.symbols, holders_before),
        )
        return result.summary

    def _fold(
        self,
        outcome: TradingOutcome,
        result: TradingDayResult,
        fills: list[Fill],
        assignment: dict[str, str],
    ) -> None:
        """Fold one settled session into the phase outcome."""
        summary = result.summary
        # The session's evidence is its own stamped summary — appended as it stands, with no
        # wrapper shape to unpack on the way out.
        outcome.sessions.append(summary)
        outcome.orders_submitted += summary.orders_submitted
        outcome.positions = summary.positions
        outcome.start_equity = summary.start_equity
        outcome.end_equity = summary.final_equity
        # External bars, retained for close-phase reconciliation; ``{}`` unless record_bars.
        outcome.live_bars = result.live_bars
        orphaned = set(summary.orphans_flattened)
        for fill in fills:
            outcome.trades.append(
                Trade(
                    fill.symbol,
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    _fill_rationale(fill, orphaned),
                    # The enrichment story #142 needs to make a trade log worth reading: when it
                    # filled, what it cost, and which champion held the symbol. The fee is the
                    # fill's own charge; the slippage is the *modelled* cost it was filled under
                    # (a per-side bps assumption, not a measurement), stated per trade so a
                    # results page can show its fill assumptions beside its returns.
                    ts=_fill_stamp(fill),
                    fees=fill.fee,
                    slippage_bps=self.slippage_bps,
                    champion=assignment.get(fill.symbol),
                )
            )
        # The summary carries its own report lines (feed transitions, refusal/halt) in
        # occurrence order — fold them in verbatim.
        outcome.events.extend(summary.events)

    def _assignment(
        self, traded_symbols: list[str], holders_before: dict[str, str]
    ) -> dict[str, str]:
        """Symbol → the champion family that held it this session — the trade log's attribution.

        The same ``assign_indices`` call the P&L attribution makes, over the same eligibility and
        score inputs, so "which champion made this trade" and "which champion earned this P&L"
        are one answer rather than two that could drift. A symbol no current champion is assigned
        falls back to the holder recorded before the session: its position was orphaned and
        flattened, and that fill is the *displaced* champion's, not nobody's.
        """
        idx = assign_indices(
            len(self.entries), sorted(traded_symbols), self.config.live_symbols, self.config.scores
        )
        attributed = {sym: self.entries[j].family for sym, j in idx.items()}
        for sym, family in holders_before.items():
            attributed.setdefault(sym, family)
        return attributed

    def _attribute(
        self, traded_symbols: list[str], realized_before: dict[str, float], day: date
    ) -> None:
        """Fold one completed session's realized P&L into the forward ledger, per champion.

        Each traded symbol's realized delta (``broker.realized_pnl_by_symbol`` after − before)
        is attributed to the champion assigned that symbol this session, so a multi-session
        catch-up credits each day to whoever held the symbol *that* day. A symbol NO current
        champion is assigned can still realize P&L — its orphaned position was flattened —
        and that closing fill belongs to the champion that opened the position (the ledger's
        recorded holder), not to nobody and not to any current champion.
        """
        after = self.broker.realized_pnl_by_symbol
        idx = assign_indices(
            len(self.entries), sorted(traded_symbols), self.config.live_symbols, self.config.scores
        )
        per_champion: dict[int, dict[str, float]] = {}
        orphan_deltas: dict[str, float] = {}
        for sym in sorted(traded_symbols):
            delta = after.get(sym, 0.0) - realized_before.get(sym, 0.0)
            if sym in idx:
                per_champion.setdefault(idx[sym], {})[sym] = delta
            elif delta != 0.0:
                orphan_deltas[sym] = delta
        for j, by_symbol in per_champion.items():
            entry = self.entries[j]
            self.forward.record(champion_key(entry), entry.family, day, by_symbol)
        for sym, delta in orphan_deltas.items():
            holder = self.forward.holders.get(sym)
            if holder is None:
                # Pre-holder ledger file (or a corrupt one): the money is honest on the
                # account either way; only the per-champion label is lost.
                logger.warning(
                    "realized %.2f on orphaned %s has no recorded holder; unattributed",
                    delta,
                    sym,
                )
                continue
            # count_session=False: the displaced champion's position closed, but it made no
            # decision this session — its money moves, its session count does not.
            self.forward.record(
                holder["key"], holder["family"], day, {sym: delta}, count_session=False
            )

    def _update_holders(self) -> None:
        """Re-derive the ledger's open-position holder map after a settled session.

        Every open position's symbol maps to the champion currently assigned it — on a
        reassignment the inheritor becomes the holder, the same rule the realized/unrealized
        attribution follows. A closed symbol's holder is dropped; an orphaned symbol still
        open (no tradable bar to flatten on) keeps its recorded holder so a later flatten can
        still credit the right champion.
        """
        positions = self.broker.positions()  # open positions only
        idx = assign_indices(
            len(self.entries), sorted(positions), self.config.live_symbols, self.config.scores
        )
        for sym in list(self.forward.holders):
            if sym not in positions:
                del self.forward.holders[sym]
        for sym, j in idx.items():
            entry = self.entries[j]
            self.forward.holders[sym] = {"key": champion_key(entry), "family": entry.family}


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
        # Paper fills charge the operator-configured cost (#23) — the same fee/slippage the
        # backtest promoted the champion under, so a live-holdout fill can never be cheaper
        # than the arena the strategy earned its slot in. Read **once**, here: the account the
        # sessions fill on and the cost a trade row reports are the same number by
        # construction, never two reads of one setting that could drift.
        self.fee_bps = settings.backtest.fee_bps
        self.slippage_bps = settings.backtest.slippage_bps
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
        try:
            broker = store.load(
                fee_model=FeeModel(self.fee_bps), slippage_model=SlippageModel(self.slippage_bps)
            )
        except RuntimeError as exc:
            logger.error("trading refused: %s", exc)
            outcome.events.append(f"Trading refused — {exc}")
            return outcome
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
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
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
            day_runner.run(feed=feed, day=day, outcome=outcome)

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
        day_runner.run(
            feed=feed,
            day=day,
            outcome=outcome,
            record_bars=True,  # external bars are retained for close-phase reconciliation
            session_start=t,
            session_end=self.clock.next_close(t),
            now=sleeper.now,
            sleeper=sleeper,
            poll_interval_s=self.settings.live_feed.poll_interval_s,
            stop_event=self._stop_event,
        )
