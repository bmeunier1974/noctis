"""The session ledger — the durable narrative of one research session, behind one interface.

``state/sessions/<session_id>.jsonl`` holds one JSON line per session event: the session
opening (``session_start`` — mandate, budgets, models), one motivating idea per formulate
(``thesis``, with the same ``parent_thesis`` / ``pivot_rationale`` lineage the experiment
journal records), each stage transition (``stage``), one line per model judgment
(``episode`` — stage, model, tokens, misfires, outcome, escalated), each spent verdict
(``verdict``, carrying the class-level lesson), and the closing rollup (``session_end``).

The cross-strategy story that today lives in a conversation transcript lives *here* instead,
so a small-context deterministic driver can drive the loop while the LLM is invoked only at
narrow judgment points: the ledger, not a growing chat context, is the ground truth the
formulate briefing tails and the CLOSE report renders. That is why the record schema lives
in this module and nowhere else — every writer is an explicit ``record_*`` method and every
reader gets typed views, so no caller re-parses ``event`` strings by hand.

The schema is *extended, never changed*: one writer per kind, tolerant reads. A malformed
line is skipped and an unknown record kind an older reader never learned is ignored by the
typed views rather than being fatal, so an existing ledger keeps loading as new kinds land.
Records are self-describing and appends are line-atomic — one file, one session, written once
in order — which keeps mid-session resume (out of scope here) a cheap later addition: a
resumed driver replays the append-only tail it already knows how to read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SESSIONS_DIRNAME = "sessions"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _opt_str(value: Any) -> str | None:
    """A tolerant optional-text read: absent/empty stays ``None``, anything else stringifies."""
    return str(value) if value else None


def new_session_id(now: datetime | None = None) -> str:
    """A clock-derived default session id — a pure function of ``now`` (defaults to wall clock).

    The randomness is injectable, never hidden: pass a fixed ``now`` for a deterministic id, or
    supply your own id to :class:`SessionLedger` outright. Production callers (the composition
    root) hand in an explicit id; this is the sensible default when none is given.
    """
    return f"session-{(now or datetime.now(UTC)).strftime('%Y%m%dT%H%M%S')}"


@dataclass(frozen=True)
class SessionStart:
    """A typed view of the ``session_start`` line: what the session was configured to do."""

    at: str
    mandate: str | None
    budgets: dict[str, Any]
    models: dict[str, Any]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SessionStart:
        return cls(
            at=str(record.get("at", "")),
            mandate=_opt_str(record.get("mandate")),
            budgets=dict(record.get("budgets") or {}),
            models=dict(record.get("models") or {}),
        )


@dataclass(frozen=True)
class ThesisLine:
    """One journaled thesis with its lineage — a typed view of a ``thesis`` line.

    ``text`` is the motivating idea in prose; ``parent_thesis`` / ``pivot_rationale`` are the
    optional lineage a pivot chain walks (both ``None`` when the thesis stands on its own).
    """

    at: str
    strategy: str
    text: str
    parent_thesis: str | None = None
    pivot_rationale: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ThesisLine:
        return cls(
            at=str(record.get("at", "")),
            strategy=str(record.get("strategy", "")),
            text=str(record.get("thesis", "")),
            parent_thesis=_opt_str(record.get("parent_thesis")),
            pivot_rationale=_opt_str(record.get("pivot_rationale")),
        )


@dataclass(frozen=True)
class StageTransition:
    """A typed view of a ``stage`` line: the loop moving to a new research stage.

    ``detail`` is an optional structured payload a stage may carry (e.g. the deterministic MATCH
    stage records its screened ``profile``, ``fit`` set, and ``reserved_holdout``); it stays an
    empty dict for the stages that carry none, so a reader never branches on presence."""

    at: str
    stage: str
    strategy: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StageTransition:
        return cls(
            at=str(record.get("at", "")),
            stage=str(record.get("stage", "")),
            strategy=_opt_str(record.get("strategy")),
            detail=dict(record.get("detail") or {}),
        )


@dataclass(frozen=True)
class Episode:
    """A typed view of an ``episode`` line: one narrow model judgment and how it went."""

    at: str
    stage: str
    model: str
    tokens: int
    misfires: int
    outcome: str
    escalated: bool

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Episode:
        return cls(
            at=str(record.get("at", "")),
            stage=str(record.get("stage", "")),
            model=str(record.get("model", "")),
            tokens=int(record.get("tokens") or 0),
            misfires=int(record.get("misfires") or 0),
            outcome=str(record.get("outcome", "")),
            escalated=bool(record.get("escalated")),
        )


@dataclass(frozen=True)
class Verdict:
    """A typed view of a ``verdict`` line: a spent verdict and the class-level lesson it left.

    ``promoted`` is ``None`` on a rejection (and on any verdict that never recorded it).
    """

    at: str
    strategy: str
    verdict: str
    lesson: str
    promoted: bool | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Verdict:
        promoted = record.get("promoted")
        return cls(
            at=str(record.get("at", "")),
            strategy=str(record.get("strategy", "")),
            verdict=str(record.get("verdict", "")),
            lesson=str(record.get("lesson", "")),
            promoted=None if promoted is None else bool(promoted),
        )


@dataclass(frozen=True)
class SessionEnd:
    """A typed view of the ``session_end`` line: the closing rollup for the whole session."""

    at: str
    formulated: int
    promoted: int
    rejected: int
    note: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SessionEnd:
        return cls(
            at=str(record.get("at", "")),
            formulated=int(record.get("formulated") or 0),
            promoted=int(record.get("promoted") or 0),
            rejected=int(record.get("rejected") or 0),
            note=_opt_str(record.get("note")),
        )


class SessionLedger:
    """An append-only per-session ledger under ``<state_dir>/sessions/``.

    One instance is one session, one file. ``state_dir`` is handed in by the composition root
    (the module never reads config); ``session_id`` names the file — supply one for a stable
    name, or let it default to a clock-derived id (see :func:`new_session_id`).
    """

    def __init__(
        self,
        state_dir: str | Path,
        session_id: str | None = None,
        *,
        now: datetime | None = None,
    ) -> None:
        self.session_id = session_id or new_session_id(now)
        self.path = Path(state_dir) / SESSIONS_DIRNAME / f"{self.session_id}.jsonl"

    # ── reads ────────────────────────────────────────────────────────────────
    def records(self) -> list[dict[str, Any]]:
        """Every parseable record, in ledger (append) order. Malformed lines are skipped."""
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _typed(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("event") == event]

    def session_start(self) -> SessionStart | None:
        records = self._typed("session_start")
        return SessionStart.from_record(records[0]) if records else None

    def theses(self) -> list[ThesisLine]:
        return [ThesisLine.from_record(r) for r in self._typed("thesis")]

    def stages(self) -> list[StageTransition]:
        return [StageTransition.from_record(r) for r in self._typed("stage")]

    def episodes(self) -> list[Episode]:
        return [Episode.from_record(r) for r in self._typed("episode")]

    def verdicts(self) -> list[Verdict]:
        return [Verdict.from_record(r) for r in self._typed("verdict")]

    def session_end(self) -> SessionEnd | None:
        records = self._typed("session_end")
        return SessionEnd.from_record(records[-1]) if records else None

    # ── writes ───────────────────────────────────────────────────────────────
    def record_session_start(
        self,
        *,
        mandate: str | None,
        budgets: dict[str, Any],
        models: dict[str, Any],
    ) -> None:
        self._append(
            {
                "event": "session_start",
                "at": _now_iso(),
                "mandate": mandate,
                "budgets": dict(budgets),
                "models": dict(models),
            }
        )

    def record_thesis(
        self,
        strategy: str,
        thesis: str,
        *,
        parent_thesis: str | None = None,
        pivot_rationale: str | None = None,
    ) -> None:
        """One thesis line per formulate, with optional pivot lineage.

        ``parent_thesis`` / ``pivot_rationale`` are the lineage a pivot chain walks; an absent
        field is omitted from the record rather than written as null, so a tolerant read
        distinguishes "no lineage" from a stored empty value.
        """
        record: dict[str, Any] = {
            "event": "thesis",
            "at": _now_iso(),
            "strategy": strategy,
            "thesis": thesis,
        }
        if parent_thesis is not None:
            record["parent_thesis"] = parent_thesis
        if pivot_rationale is not None:
            record["pivot_rationale"] = pivot_rationale
        self._append(record)

    def record_stage(
        self,
        stage: str,
        *,
        strategy: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """One stage-transition line. ``detail`` is an optional structured payload the stage
        carries (e.g. MATCH's screened profile / fit set / reserved holdout); an absent detail is
        omitted from the record rather than written as an empty object, so a tolerant read
        distinguishes "no detail" from a stored empty one."""
        record: dict[str, Any] = {"event": "stage", "at": _now_iso(), "stage": stage}
        if strategy is not None:
            record["strategy"] = strategy
        if detail:
            record["detail"] = dict(detail)
        self._append(record)

    def record_episode(
        self,
        *,
        stage: str,
        model: str,
        outcome: str,
        tokens: int = 0,
        misfires: int = 0,
        escalated: bool = False,
    ) -> None:
        self._append(
            {
                "event": "episode",
                "at": _now_iso(),
                "stage": stage,
                "model": model,
                "tokens": int(tokens),
                "misfires": int(misfires),
                "outcome": outcome,
                "escalated": bool(escalated),
            }
        )

    def record_verdict(
        self,
        strategy: str,
        *,
        verdict: str,
        lesson: str,
        promoted: bool | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "verdict",
            "at": _now_iso(),
            "strategy": strategy,
            "verdict": verdict,
            "lesson": lesson,
        }
        if promoted is not None:
            record["promoted"] = bool(promoted)
        self._append(record)

    def record_session_end(
        self,
        *,
        formulated: int,
        promoted: int,
        rejected: int,
        note: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "session_end",
            "at": _now_iso(),
            "formulated": int(formulated),
            "promoted": int(promoted),
            "rejected": int(rejected),
        }
        if note is not None:
            record["note"] = note
        self._append(record)

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
