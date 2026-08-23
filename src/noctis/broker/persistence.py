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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from noctis.broker.paper import PaperBroker
from noctis.broker.seam import FeeModel, SlippageModel

_ACCOUNT_VERSION = 1

# The account's daily record, beside the account itself: one line per session close carrying the
# equity mark and the session that produced it (story #142). Append-only and dated, so the run
# record's equity curve is **re-derived from it at every write** rather than carried in memory —
# the same "derived, never incremented" rule that makes resume correct everywhere else in the epic.
EQUITY_CURVE_NAME = "equity_curve.jsonl"


@dataclass(frozen=True)
class AccountSummary:
    """Read-only account view for ``noctis status`` / ``noctis account`` / the close report."""

    equity: float
    starting_cash: float
    cumulative_pnl: float
    open_positions: int
    opened: str
    last_session: str | None


class EquityLedger:
    """The paper account's **daily** record: one dated mark per session close, append-only.

    ``AccountStore`` beside it holds the account as it is *now* — cash, positions, marks — which is
    everything trading needs and nothing a curve can be drawn from. This is the missing half: at
    each CLOSE the engine appends the account's equity for that session date, together with the
    session's own fills, orders and closing positions. Nothing derived is stored (no returns, no
    drawdown); the numbers are the marks, and every metric the run record publishes is recomputed
    from them at write time.

    Three properties it is built for:

    * **Append-only.** A night is added, never rewritten, so a crash mid-run loses at most the
      session it was writing.
    * **One mark per date, last write wins** (:meth:`marks`). CLOSE can legitimately run twice for
      one date — a re-run, a catch-up, a resumed segment — and a curve that doubled a day would
      make every derived total a lie. Deduplication happens on *read*, so the file stays a plain
      append and the dedup rule lives in exactly one place.
    * **Never fatal.** A line that cannot be parsed (a kill mid-append) costs that mark and nothing
      else: the record is evidence, and evidence that is partly unreadable is still evidence.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def mark(
        self,
        *,
        date: str,
        equity: float,
        start_equity: float | None = None,
        end_equity: float | None = None,
        realized_pnl: float | None = None,
        orders_submitted: int = 0,
        positions_end: Mapping[str, float] | None = None,
        trades: Sequence[Mapping[str, object]] = (),
    ) -> None:
        """Append one session's mark. ``date`` is the session's own ``YYYY-MM-DD``.

        ``equity`` is the **account's** mark-to-market at that close — the curve — while
        ``start_equity``/``end_equity`` are that session's own bounds, which differ from it
        whenever the account carried positions into the day.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "date": date,
            "equity": float(equity),
            "start_equity": None if start_equity is None else float(start_equity),
            "end_equity": None if end_equity is None else float(end_equity),
            "realized_pnl": None if realized_pnl is None else float(realized_pnl),
            "orders_submitted": int(orders_submitted),
            "positions_end": {str(k): float(v) for k, v in (positions_end or {}).items()},
            "trades": [dict(trade) for trade in trades],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def marks(self) -> list[dict]:
        """Every session's mark, one per date (the last written wins), oldest first."""
        if not self.path.is_file():
            return []
        latest: dict[str, dict] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:  # pragma: no cover - an unreadable ledger is a curve we do not have
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a truncated append costs that mark, never the ledger
            if isinstance(entry, dict) and isinstance(entry.get("date"), str):
                latest[entry["date"]] = entry
        return [latest[key] for key in sorted(latest)]


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

    def summarize(self, broker: PaperBroker) -> AccountSummary:
        """The display view of an **already-loaded** account: pure over ``broker`` plus this
        store's ``opened``/``last_session`` provenance, so a caller that needs both the broker
        and its summary parses the file once (CLOSE does — story #266).
        """
        equity = broker.equity()
        return AccountSummary(
            equity=equity,
            starting_cash=broker.starting_cash,
            cumulative_pnl=equity - broker.starting_cash,
            open_positions=len(broker.positions()),
            opened=self.opened or "?",
            last_session=self.last_session,
        )

    def summary(self) -> AccountSummary | None:
        """Account snapshot for display; ``None`` when no account exists yet.

        Raises like :meth:`load` on a corrupt file.
        """
        if not self.path.is_file():
            return None
        return self.summarize(self.load())

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
