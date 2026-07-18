"""The per-champion forward track record (live-holdout plan 5).

``state/forward_ledger.json`` attributes the rolling live-holdout's realized P&L to the
champion that earned it, session by session. A champion's cumulative forward P&L is its
ledger ``realized_pnl`` **plus** the current unrealized on the positions it holds now.

This is *derived evidence*, not the money state, so its corruption policy is the opposite of
:class:`AccountStore`: a bad file **warns and is omitted from display, never blocks trading**
(blocking the forward test on a display file would be wrong, and the realized history is
reconstructable in principle). Read-only on top of the promotion gates — surfacing the number
never feeds it back into a gate (AGENTS.md rules 2/4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from noctis.champions.assignment import assign_indices, slot_inputs

if TYPE_CHECKING:
    from noctis.broker.seam import Broker

logger = logging.getLogger("noctis.forward")

_FORWARD_LEDGER_VERSION = 1


def champion_key(entry) -> str:
    """Stable identity for a champion *instance*: ``family@crowned_at``.

    Family alone is unstable (a family can be re-promoted with new params); ``crowned_at`` is
    stamped once when the entry enters the board, so it pins one champion instance.
    """
    return f"{entry.family}@{getattr(entry, 'crowned_at', '') or ''}"


@dataclass
class ForwardEntry:
    """One champion instance's accumulated realized forward record."""

    key: str
    family: str
    opened_session: str
    last_session: str
    sessions_traded: int = 0
    realized_pnl: float = 0.0
    symbols: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "family": self.family,
            "opened_session": self.opened_session,
            "last_session": self.last_session,
            "sessions_traded": self.sessions_traded,
            "realized_pnl": self.realized_pnl,
            "symbols": dict(self.symbols),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ForwardEntry:
        return cls(
            key=str(data["key"]),
            family=str(data["family"]),
            opened_session=str(data["opened_session"]),
            last_session=str(data["last_session"]),
            sessions_traded=int(data["sessions_traded"]),
            realized_pnl=float(data["realized_pnl"]),
            symbols={str(s): float(v) for s, v in data.get("symbols", {}).items()},
        )


@dataclass
class ForwardRecord:
    """A champion's forward P&L for display: realized (ledger) + current unrealized."""

    family: str
    key: str
    realized_pnl: float
    unrealized_pnl: float
    sessions_traded: int
    opened_session: str

    @property
    def forward_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "key": self.key,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "forward_pnl": self.forward_pnl,
            "sessions_traded": self.sessions_traded,
            "opened_session": self.opened_session,
        }


class ForwardLedger:
    """Load/record/save of ``state/forward_ledger.json`` (atomic tmp + ``replace``)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, ForwardEntry] = {}
        # Who opened each currently-open position: symbol → {"key", "family"} of the champion
        # assigned it at the settle that saw it open (a reassigned symbol's inheritor takes
        # over — the same rule realized/unrealized attribution follows). This is what lets an
        # orphan flatten's closing fill be credited to the champion that opened the position
        # after that champion has left the board. Derived evidence like the rest of the file:
        # a corrupt/absent map only costs the attribution label, never blocks a flatten.
        self.holders: dict[str, dict[str, str]] = {}
        self.corrupt = False

    def load(self) -> None:
        """Populate ``entries`` from disk. A corrupt/absent file leaves ``entries`` empty and
        sets ``corrupt`` — it never raises, so the trading path can call it freely."""
        self.entries = {}
        self.holders = {}
        self.corrupt = False
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text())
            version = data.get("version")
            if version != _FORWARD_LEDGER_VERSION:
                raise ValueError(f"unsupported forward-ledger version {version!r}")
            self.entries = {
                str(k): ForwardEntry.from_dict(v) for k, v in data.get("champions", {}).items()
            }
            # Absent on pre-holder files → empty map (those positions' openers are unknown).
            self.holders = {
                str(sym): {"key": str(h["key"]), "family": str(h["family"])}
                for sym, h in data.get("holders", {}).items()
            }
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "forward ledger %s unreadable (%r); omitting from display (derived evidence, "
                "not the account — trading is unaffected)",
                self.path,
                exc,
            )
            self.entries = {}
            self.holders = {}
            self.corrupt = True

    def record(
        self,
        key: str,
        family: str,
        session: date,
        realized_by_symbol: dict[str, float],
        *,
        count_session: bool = True,
    ) -> None:
        """Fold one champion's realized P&L for ``session`` into its entry (once per session
        the champion traded — ``sessions_traded`` increments each call).

        ``count_session=False`` folds the P&L without claiming the champion traded the
        session: an orphan flatten closes a displaced champion's position, so the money is
        its, but it made no decision that day.
        """
        s = session.isoformat()
        entry = self.entries.get(key)
        if entry is None:
            entry = ForwardEntry(key=key, family=family, opened_session=s, last_session=s)
            self.entries[key] = entry
        if count_session:
            entry.sessions_traded += 1
        entry.last_session = s
        for sym, pnl in realized_by_symbol.items():
            entry.symbols[sym] = entry.symbols.get(sym, 0.0) + pnl
            entry.realized_pnl += pnl

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _FORWARD_LEDGER_VERSION,
            "champions": {k: e.to_dict() for k, e in self.entries.items()},
            "holders": {sym: dict(h) for sym, h in self.holders.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)  # atomic on POSIX


def _unrealized_by_champion(entries, broker: Broker | None) -> dict[str, float]:
    """Current unrealized P&L per champion key, attributing each open position to the champion
    *currently* assigned that symbol (the known wrinkle: after a reassignment, an open lot's
    unrealized follows the new assignee until it closes — realized history stays correct)."""
    if broker is None:
        return {}
    positions = broker.positions()  # open positions only
    held = sorted(positions)
    if not held or not entries:
        return {}
    live_symbols, scores = slot_inputs(entries)
    idx = assign_indices(len(entries), held, live_symbols, scores)
    out: dict[str, float] = {}
    marks = broker.marks()
    for sym, j in idx.items():
        mark = marks.get(sym)
        if mark is None:
            continue
        pos = positions[sym]
        unrealized = pos.quantity * (mark - pos.avg_price)
        key = champion_key(entries[j])
        out[key] = out.get(key, 0.0) + unrealized
    return out


def forward_records(
    ledger: ForwardLedger, entries, broker: Broker | None = None
) -> list[ForwardRecord]:
    """Per-champion forward records (realized from the ledger + current unrealized), best first.

    One record per champion instance that has traded (a ledger entry). ``broker`` (the current
    account) attributes live unrealized to current champions; pass ``None`` to show realized
    only (e.g. when the account file is unreadable).
    """
    unreal = _unrealized_by_champion(entries or [], broker)
    records = [
        ForwardRecord(
            family=e.family,
            key=key,
            realized_pnl=e.realized_pnl,
            unrealized_pnl=unreal.get(key, 0.0),
            sessions_traded=e.sessions_traded,
            opened_session=e.opened_session,
        )
        for key, e in ledger.entries.items()
    ]
    records.sort(key=lambda r: r.forward_pnl, reverse=True)
    return records
