"""One TRADING session end-to-end: drive champions over a bar feed, then settle.

The settle order is the crash-safety contract, identical for live and replay days:

1. **Trade** — :func:`~noctis.live.node.run_trading_day` over the session's
   :class:`~noctis.live.feed.BarFeed` (live yfinance or a catalog replay slice).
2. **Attribute** — fold the session's realized P&L into the per-champion forward ledger.
   Derived evidence, never the money state: a ledger hiccup logs and continues.
3. **Account first** — persist the continuous paper account (`state/paper_account.json`).
4. **High-water mark second** — advance the session ledger (`state/trading_sessions.json`).

A crash between 3 and 4 leaves the ledger behind the account, which re-trades that session
(safe — strategies re-decide from carried positions) rather than silently skipping it.
Before this module existed the live path never advanced the high-water mark at all, so a
live-traded day followed by a replay day was re-traded on the carried account; one settle
path for both drivers is what closes that gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from noctis.backtest import Candidate
from noctis.champions.assignment import assign_indices, slot_inputs
from noctis.engine.forward_ledger import ForwardLedger, champion_key
from noctis.engine.sessions import SessionLedger
from noctis.live.feed import BarFeed
from noctis.live.node import SessionConfig, TradingSummary, run_trading_day
from noctis.live.risk import RiskLimits
from noctis.strategies.families import FamilyRegistry

if TYPE_CHECKING:
    from noctis.broker.paper import PaperBroker
    from noctis.broker.persistence import AccountStore
    from noctis.broker.seam import Fill

logger = logging.getLogger("noctis.runtime")


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


@dataclass
class SessionOutcome:
    """What one settled session hands back for the run/report accumulators."""

    summary: TradingSummary
    fills: list[Fill] = field(default_factory=list)  # only this session's, never carried ones
    live_bars: dict[str, pd.DataFrame] = field(default_factory=dict)  # {} unless record_bars


class TradingDay:
    """The one place a TRADING session is traded *and* settled.

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
        min_order_notional: float = 0.0,
        rebalance_band_pct: float = 0.0,
        on_event=None,
        heartbeat_polls: int = 0,
    ):
        self.broker = broker
        self.store = store
        self.ledger = ledger
        self.forward = forward
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
        record_bars: bool = False,
        session_start: datetime | None = None,
        session_end: datetime | None = None,
        now=None,
        sleeper=None,
        poll_interval_s: float = 2.0,
        stop_event=None,
    ) -> SessionOutcome:
        """Trade session ``day`` from ``feed``, then settle (attribute → account → mark)."""
        fills_before = len(self.broker.fills)
        realized_before = dict(self.broker.realized_pnl_by_symbol)
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
        return SessionOutcome(result.summary, self.broker.fills[fills_before:], result.live_bars)

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
