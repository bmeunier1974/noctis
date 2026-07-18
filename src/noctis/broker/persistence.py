"""Persistence for the continuous paper account (the cumulative forward track record).

``state/paper_account.json`` carries ONE paper account — equity *and* open positions —
across TRADING sessions, so the rolling live-holdout accumulates a genuine multi-day
equity curve instead of a string of disconnected fresh-100k one-day experiments. The
store owns the file format (atomic writes, version stamp, inception provenance); the
broker serialises itself (:meth:`PaperBroker.to_dict` / :meth:`PaperBroker.from_dict`),
keeping file knowledge out of the broker seam. A corrupt file is a hard error, never a
silent reset — silently restarting at 100k would quietly destroy the track record; the
operator resets deliberately via ``noctis account --reset``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from noctis.broker.paper import PaperBroker
from noctis.broker.seam import FeeModel, SlippageModel

_ACCOUNT_VERSION = 1


@dataclass(frozen=True)
class AccountSummary:
    """Read-only account view for ``noctis status`` / ``noctis account`` / the close report."""

    equity: float
    starting_cash: float
    cumulative_pnl: float
    open_positions: int
    opened: str
    last_session: str | None


class AccountStore:
    """Load/save/reset of ``state/paper_account.json``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # Provenance, populated by load() from the file and advanced by save(). ``opened``
        # is stamped on the first save with the first traded session's date (inception).
        self.opened: str | None = None
        self.last_session: str | None = None

    def load(
        self,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> PaperBroker:
        """The carried account; an absent file is account inception (a fresh 100k broker).

        A corrupt/unreadable file raises ``RuntimeError`` — the caller must refuse to
        trade rather than silently reset the cumulative track record.
        """
        if not self.path.is_file():
            return PaperBroker(fee_model=fee_model, slippage_model=slippage_model)
        try:
            data = json.loads(self.path.read_text())
            version = data.get("version")
            if version != _ACCOUNT_VERSION:
                raise ValueError(f"unsupported account version {version!r}")
            broker = PaperBroker.from_dict(data, fee_model=fee_model, slippage_model=slippage_model)
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"corrupt paper account {self.path}: {exc!r} — refusing to silently reset "
                "(this file is the cumulative forward track record); recover deliberately "
                "with `noctis account --reset`"
            ) from exc
        self.opened = data.get("opened")
        self.last_session = data.get("last_session")
        return broker

    def save(self, broker: PaperBroker, session: date) -> None:
        """Persist the account after ``session`` completed (atomic tmp + ``replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.opened is None:
            self.opened = session.isoformat()
        self.last_session = session.isoformat()
        data = broker.to_dict()
        data["version"] = _ACCOUNT_VERSION
        data["opened"] = self.opened
        data["last_session"] = self.last_session
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)  # atomic on POSIX

    def summary(self) -> AccountSummary | None:
        """Account snapshot for display; ``None`` when no account exists yet.

        Raises like :meth:`load` on a corrupt file.
        """
        if not self.path.is_file():
            return None
        broker = self.load()
        equity = broker.equity()
        return AccountSummary(
            equity=equity,
            starting_cash=broker.starting_cash,
            cumulative_pnl=equity - broker.starting_cash,
            open_positions=len(broker.positions()),
            opened=self.opened or "?",
            last_session=self.last_session,
        )

    def reset(self) -> Path | None:
        """Archive the account file so the next session starts fresh at 100k.

        Returns the archive path (``paper_account.<date>.json``), or ``None`` when there
        was nothing to reset. Works on a corrupt file too — this is the documented
        recovery path, and the archive preserves the evidence.
        """
        if not self.path.is_file():
            return None
        stamp = date.today().isoformat()
        archive = self.path.with_name(f"{self.path.stem}.{stamp}.json")
        n = 1
        while archive.exists():  # keep earlier same-day archives
            archive = self.path.with_name(f"{self.path.stem}.{stamp}.{n}.json")
            n += 1
        self.path.replace(archive)
        self.opened = None
        self.last_session = None
        return archive
