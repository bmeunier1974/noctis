"""Close-of-day report assembly — persisted state + one session's activity → ReportData.

One home for the wiring that gathers everything the report renders: champions and their
decision history (the registry), research findings (memory), minted-spec provenance
(``state/specs.json``), the continuous paper account's curve (``state/paper_account.json``),
and the per-champion forward track record (``state/forward_ledger.json``). The runtime's
CLOSE phase folds in the day's :class:`SessionActivity`; ``noctis report`` assembles from
persisted state alone — so the CLI report and the CLOSE report are the same report, minus
the session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from noctis.broker.persistence import AccountStore, AccountSummary
from noctis.engine.forward_ledger import ForwardLedger, ForwardRecord, forward_records
from noctis.reporting.report import ReportData, Trade
from noctis.strategies.spec import spec_family_names

if TYPE_CHECKING:
    from noctis.champions.registry import ChampionRegistry
    from noctis.memory.base import Memory

logger = logging.getLogger("noctis.report")


@dataclass(frozen=True)
class AccountForward:
    """The persisted paper-account + forward-track view, read once for any renderer
    (``noctis status`` and the close report both show it). ``account`` is ``None`` when no
    account exists yet; ``account_corrupt`` distinguishes an unreadable file (trading
    refuses until reset) from a missing one."""

    account: AccountSummary | None
    account_corrupt: bool
    forward: ForwardLedger
    records: list[ForwardRecord]


def gather_account_forward(state_dir: str | Path, entries) -> AccountForward:
    """One read of ``paper_account.json`` + ``forward_ledger.json`` → account summary and
    per-champion forward records (realized from the ledger + current unrealized on the
    account's open positions; realized-only when the account is unreadable). Evidence-only
    degradation, never an exception."""
    account = None
    broker = None
    corrupt = False
    store = AccountStore(Path(state_dir) / "paper_account.json")
    try:
        account = store.summary()
        broker = store.load() if account is not None else None
    except RuntimeError:
        corrupt = True
    forward = ForwardLedger(Path(state_dir) / "forward_ledger.json")
    forward.load()
    return AccountForward(account, corrupt, forward, forward_records(forward, entries, broker))


@dataclass
class SessionActivity:
    """What one day-cycle contributes to the close report beyond persisted state.

    The runtime's per-cycle accumulator: research folds its summary in, trading records
    equity/trades/positions, and anything noteworthy appends to ``events``. A fresh
    instance is the honest "no session activity" (``noctis report`` outside a run).
    """

    start_equity: float = 0.0
    end_equity: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    research_iterations: int = 0
    research_promotions: int = 0
    research_rejections: int = 0
    research_dead_ends: int = 0
    minted_specs: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def assemble_report(
    *,
    as_of: str,
    mode: str,
    registry: ChampionRegistry,
    memory: Memory,
    state_dir: str | Path,
    session: SessionActivity | None = None,
) -> ReportData:
    """Gather persisted state — plus ``session``, when a day-cycle ran — into a ReportData.

    Evidence-only degradation: an unreadable paper account omits the cumulative curve (the
    TRADING phase already refused and reported the corruption); a corrupt forward ledger
    yields an empty forward section (``ForwardLedger.load`` never raises).
    """
    session = session if session is not None else SessionActivity()
    entries = registry.list()
    champions = [
        {"family": e.family, "params": e.params, "test_metric": e.test_metric, "gap": e.gap}
        for e in entries
    ]
    # Which current champions are minted spec-families (vs the built-in seeds).
    spec_names = set(spec_family_names(state_dir))
    promoted_specs = [e.family for e in entries if e.family in spec_names]
    promotions = [h for h in registry.history[-50:] if h.get("promoted")]
    demotions = registry.demotions()[-10:]
    # The continuous account's cumulative curve + per-champion forward track record.
    af = gather_account_forward(state_dir, entries)
    if af.account_corrupt:
        logger.warning("paper account unreadable; report omits cumulative P&L")
    account = af.account
    forward_data = [r.to_dict() for r in af.records]
    return ReportData(
        as_of=as_of,
        mode=mode,
        start_equity=session.start_equity,
        end_equity=session.end_equity,
        realized_pnl=session.end_equity - session.start_equity,
        cumulative_pnl=account.cumulative_pnl if account else None,
        account_opened=account.opened if account else None,
        forward=forward_data,
        trades=list(session.trades),
        positions=dict(session.positions),
        promotions=promotions,
        demotions=demotions,
        champions=champions,
        research={
            "iterations": session.research_iterations,
            "promotions": session.research_promotions,
            "rejections": session.research_rejections,
            "dead_ends": session.research_dead_ends,
            "findings": memory.findings(),
            "minted": list(session.minted_specs),
            "promoted_specs": promoted_specs,
        },
        events=list(session.events),
    )
