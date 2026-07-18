"""The coverage registry — one SQLite file recording what the lake already holds.

One row per (dataset, schema, symbol): first/last timestamp, row count, status, last
update, error message. Ingest diffs requested ranges against this registry so a covered
range is a zero-cost no-op; backtests refuse symbols that are untracked or not ``idle``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from noctis.data.types import SeriesKey

VALID_STATUSES = ("idle", "ingesting", "error")


@dataclass(frozen=True)
class CoverageRecord:
    dataset: str
    schema: str
    symbol: str
    first_ts: int | None
    last_ts: int | None
    row_count: int
    status: str
    last_update: str | None
    error_msg: str | None

    @property
    def key(self) -> SeriesKey:
        return SeriesKey(self.dataset, self.schema, self.symbol)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CoverageRegistry:
    """Transactional SQLite registry of tracked series."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coverage (
                    dataset      TEXT NOT NULL,
                    schema       TEXT NOT NULL,
                    symbol       TEXT NOT NULL,
                    first_ts     INTEGER,
                    last_ts      INTEGER,
                    row_count    INTEGER NOT NULL DEFAULT 0,
                    status       TEXT NOT NULL DEFAULT 'idle',
                    last_update  TEXT,
                    error_msg    TEXT,
                    PRIMARY KEY (dataset, schema, symbol)
                )
                """
            )
            # Days a vendor fetch confirmed hold no bars (e.g. exchange holidays the
            # weekday-fallback calendar can't know about). Recording them lets the integrity
            # check stop re-flagging and repair stop re-fetching, so repair converges.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS empty_days (
                    dataset TEXT NOT NULL,
                    schema  TEXT NOT NULL,
                    symbol  TEXT NOT NULL,
                    day     TEXT NOT NULL,
                    PRIMARY KEY (dataset, schema, symbol, day)
                )
                """
            )

    # --- reads ---
    def get(self, key: SeriesKey) -> CoverageRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM coverage WHERE dataset=? AND schema=? AND symbol=?",
                (key.dataset, key.schema, key.symbol),
            ).fetchone()
        return self._to_record(row) if row else None

    def all(self) -> list[CoverageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM coverage ORDER BY dataset, schema, symbol"
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def check_symbol_ready(
        self, symbol: str, dataset: str | None = None, schema: str | None = None
    ) -> bool:
        """True iff a matching series is tracked and its status is ``idle``.

        Backtests call this to refuse symbols that are untracked or mid-ingest/errored.
        """
        clauses = ["symbol=?"]
        params: list[object] = [symbol]
        if dataset is not None:
            clauses.append("dataset=?")
            params.append(dataset)
        if schema is not None:
            clauses.append("schema=?")
            params.append(schema)
        query = f"SELECT status, row_count FROM coverage WHERE {' AND '.join(clauses)}"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return False
        return all(r["status"] == "idle" and r["row_count"] > 0 for r in rows)

    # --- writes (transactional) ---
    def upsert(
        self,
        key: SeriesKey,
        *,
        first_ts: int | None,
        last_ts: int | None,
        row_count: int,
        status: str = "idle",
        error_msg: str | None = None,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coverage
                    (dataset, schema, symbol, first_ts, last_ts, row_count,
                     status, last_update, error_msg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, schema, symbol) DO UPDATE SET
                    first_ts=excluded.first_ts,
                    last_ts=excluded.last_ts,
                    row_count=excluded.row_count,
                    status=excluded.status,
                    last_update=excluded.last_update,
                    error_msg=excluded.error_msg
                """,
                (
                    key.dataset,
                    key.schema,
                    key.symbol,
                    first_ts,
                    last_ts,
                    row_count,
                    status,
                    _now(),
                    error_msg,
                ),
            )

    def set_status(self, key: SeriesKey, status: str, error_msg: str | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE coverage SET status=?, error_msg=?, last_update=?
                WHERE dataset=? AND schema=? AND symbol=?
                """,
                (status, error_msg, _now(), key.dataset, key.schema, key.symbol),
            )
            if cur.rowcount == 0:
                # No row yet: create a placeholder so status is never silently lost.
                conn.execute(
                    """
                    INSERT INTO coverage
                        (dataset, schema, symbol, first_ts, last_ts, row_count,
                         status, last_update, error_msg)
                    VALUES (?, ?, ?, NULL, NULL, 0, ?, ?, ?)
                    """,
                    (key.dataset, key.schema, key.symbol, status, _now(), error_msg),
                )

    # --- confirmed-empty days (holiday convergence) ---
    def mark_empty_days(self, key: SeriesKey, days: Iterable[date]) -> None:
        """Record ``days`` as confirmed to hold no bars for this series (idempotent)."""
        rows = [(key.dataset, key.schema, key.symbol, d.isoformat()) for d in days]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO empty_days (dataset, schema, symbol, day) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def known_empty_days(self, key: SeriesKey) -> set[date]:
        """Days previously confirmed empty for this series."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT day FROM empty_days WHERE dataset=? AND schema=? AND symbol=?",
                (key.dataset, key.schema, key.symbol),
            ).fetchall()
        return {date.fromisoformat(r["day"]) for r in rows}

    def sweep_stale_ingesting(self) -> int:
        """Reset any ``ingesting`` rows (left by a crash) to ``error``. Returns count."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE coverage SET status='error', error_msg=?, last_update=? "
                "WHERE status='ingesting'",
                ("stale ingesting reset on startup", _now()),
            )
            return cur.rowcount

    @staticmethod
    def _to_record(row: sqlite3.Row) -> CoverageRecord:
        return CoverageRecord(
            dataset=row["dataset"],
            schema=row["schema"],
            symbol=row["symbol"],
            first_ts=row["first_ts"],
            last_ts=row["last_ts"],
            row_count=row["row_count"],
            status=row["status"],
            last_update=row["last_update"],
            error_msg=row["error_msg"],
        )
