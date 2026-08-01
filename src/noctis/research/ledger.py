"""The session ledger — the durable narrative of one research session, behind one interface.

``state/sessions/<session_id>.jsonl`` holds one JSON line per session event: the session
opening (``session_start`` — mandate, budgets, models), one motivating idea per formulate
(``thesis``, with the same ``parent_thesis`` / ``pivot_rationale`` lineage the experiment
journal records), each stage transition (``stage`` — the AUTHOR one also referencing the brief
and compiled spec suite it captured, by content hash), one line per model judgment
(``episode`` — stage, model, tokens, misfires, outcome, escalated, plus per-misfire
diagnostics when any fired, #102), each spent verdict
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
from collections.abc import Mapping
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


def _carried_str(value: Any) -> str | None:
    """A tolerant read that keeps "not carried" and "carried empty" apart: a line that never held
    the field reads ``None``, a line that held an empty one reads ``""``.

    :func:`_opt_str` collapses both to ``None``, which is right for prose a writer may legitimately
    leave blank. It is wrong for provenance: "no served model was recorded" (an older line, or one
    whose provider never reported one) is a different fact from "this line recorded an empty served
    model", and a reader auditing which model answered must be able to tell them apart."""
    return None if value is None else str(value)


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
    empty dict for the stages that carry none, so a reader never branches on presence.

    ``brief_sha256`` / ``spec_sha256`` are the coder site's capture references (#184): the content
    hashes of the brief the authoring job was given and of the compiled spec suite it was gated
    against, whose bodies live as sidecars in the run's ``qa/`` capture area
    (:class:`~noctis.observability.capture.CaptureStore`). Both are ``None`` on a line that carried
    none — an older line, a stage that captures nothing, or a run whose capture store latched off —
    which is a different fact from a line that recorded an empty one."""

    at: str
    stage: str
    strategy: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    brief_sha256: str | None = None
    spec_sha256: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StageTransition:
        return cls(
            at=str(record.get("at", "")),
            stage=str(record.get("stage", "")),
            strategy=_opt_str(record.get("strategy")),
            detail=dict(record.get("detail") or {}),
            brief_sha256=_carried_str(record.get("brief_sha256")),
            spec_sha256=_carried_str(record.get("spec_sha256")),
        )


@dataclass(frozen=True)
class Episode:
    """A typed view of an ``episode`` line: one narrow model judgment and how it went.

    ``checks`` is the optional list of driver-side sanity-check outcomes (story #71) — each a
    ``{"check": <id>, "result": <reask|exhausted>}`` naming a check that fired on this episode's
    output and whether it earned the one corrective re-ask or exhausted it. ``misfire_details``
    is the optional list of per-misfire diagnostics (#102) — each a ``{"note", "raw"}`` pairing
    the classifier note (validation reason included) with a capped excerpt of the rejected
    payload/reply/exception, in attempt order — and ``note`` the episode's last misfire/error
    text. All three stay empty for the episodes that carried none, so a reader never branches
    on presence.

    ``usage`` is the four-field token split behind ``tokens`` (#140), or ``None`` on a line that
    recorded none — *unknown*, never zero. :func:`episode_usage` is the one place that tells an
    unknown split apart from an episode that honestly spent nothing.

    ``model`` is the model this episode *asked* for (a config alias); ``served_model`` is the id
    the provider said actually answered it (#181), or ``None`` on a line that carried none — a
    line written before the field existed, or one whose backend reported no served id. The two
    together are what make a same-alias/different-served-model event visible from the ledger."""

    at: str
    stage: str
    model: str
    tokens: int
    misfires: int
    outcome: str
    escalated: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    misfire_details: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None
    usage: dict[str, int] | None = None
    served_model: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Episode:
        usage = record.get("usage")
        return cls(
            at=str(record.get("at", "")),
            stage=str(record.get("stage", "")),
            model=str(record.get("model", "")),
            tokens=int(record.get("tokens") or 0),
            misfires=int(record.get("misfires") or 0),
            outcome=str(record.get("outcome", "")),
            escalated=bool(record.get("escalated")),
            checks=[dict(c) for c in (record.get("checks") or []) if isinstance(c, dict)],
            misfire_details=[
                dict(d) for d in (record.get("misfire_details") or []) if isinstance(d, dict)
            ],
            note=_opt_str(record.get("note")),
            usage=_read_usage(usage),
            served_model=_carried_str(record.get("served_model")),
        )


# The four neutral usage fields a completion reports, in the order they are billed. Named here
# because the ledger is what *persists* them; the price table (:mod:`noctis.research.pricing`)
# names the same four as the rates it bills at.
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _read_usage(value: Any) -> dict[str, int] | None:
    """One journaled usage split, or ``None`` when the line carries none (or an unreadable one).

    All four fields or nothing: a half-recorded split cannot be priced, and filling the gaps with
    zeros would turn missing evidence into a cheaper bill.
    """
    if not isinstance(value, dict):
        return None
    out: dict[str, int] = {}
    for field_name in USAGE_FIELDS:
        count = value.get(field_name)
        if isinstance(count, bool) or not isinstance(count, int | float):
            return None
        out[field_name] = int(count)
    return out


def episode_usage(episode: Episode) -> dict[str, int] | None:
    """One episode's token split for accounting — ``None`` only when it is genuinely unknown.

    Two things look alike in a ledger and are not: a line that recorded no split, and a line that
    recorded no *tokens* (a coder-escalation marker, an episode that never reached a completion).
    The second is an honest four zeros — it spent nothing, and calling it unknown would poison
    every total it belongs to. Only a line that *spent* tokens without journaling their split is
    unknown, which is exactly what a ledger written before #140 looks like.
    """
    if episode.usage is not None:
        return dict(episode.usage)
    return dict.fromkeys(USAGE_FIELDS, 0) if episode.tokens == 0 else None


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


def _fmt_counts(counts: dict[str, int]) -> str:
    """Render a ``{name: count}`` map as ``a=1 b=2`` (sorted, deterministic) or ``none`` when
    empty — the one formatter the rollup log line and the report renderer share."""
    return " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"


@dataclass(frozen=True)
class SessionRollup:
    """The at-a-glance close of one session, *derived* from the ledger's typed records (never a
    stored line): theses formulated, files authored, validation failures (author stages that
    never reached OPTIMIZE), trials run, verdicts by kind, undecided count, escalations, and
    tokens grouped by stage and by model. This is the rollup the session log and the CLOSE
    report render (story #74) — computed from what the ledger already holds, so no new writer
    is added to the schema."""

    session_id: str
    theses: int
    authored: int
    validation_failures: int
    trials: int
    verdicts: dict[str, int]
    promoted: int
    undecided: int
    escalations: int
    tokens_by_stage: dict[str, int]
    tokens_by_model: dict[str, int]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe view the structured report writes straight to disk."""
        return {
            "session_id": self.session_id,
            "theses": self.theses,
            "authored": self.authored,
            "validation_failures": self.validation_failures,
            "trials": self.trials,
            "verdicts": dict(self.verdicts),
            "promoted": self.promoted,
            "undecided": self.undecided,
            "escalations": self.escalations,
            "tokens_by_stage": dict(self.tokens_by_stage),
            "tokens_by_model": dict(self.tokens_by_model),
            "note": self.note,
        }

    def log_line(self) -> str:
        """One compact line naming every field — the session-log rollup at session end."""
        by_stage = _fmt_counts(self.tokens_by_stage)
        by_model = _fmt_counts(self.tokens_by_model)
        return (
            f"{self.session_id}: {self.theses} theses, {self.authored} authored, "
            f"{self.validation_failures} validation failures, {self.trials} trials, "
            f"verdicts [{_fmt_counts(self.verdicts)}], {self.undecided} undecided, "
            f"{self.escalations} escalations; tokens by stage [{by_stage}]; "
            f"tokens by model [{by_model}]"
        )


@dataclass(frozen=True)
class CandidateTrail:
    """One candidate's formulate → author → optimize → decide trail, derived from the ledger so a
    post-mortem walks structured records instead of prose. ``stages`` is the strategy-scoped stage
    labels it reached in order (its thesis is the FORMULATE step); ``trials`` / ``best_metric``
    come from its OPTIMIZE detail; ``verdict`` / ``promoted`` are ``None`` when it never reached a
    verdict (left undecided); ``oracle`` is the fixed spec's scenario names the AUTHOR stage gated
    it against (#86) — an empty tuple for a spec-less/older author stage that carried none."""

    strategy: str
    thesis: str
    stages: tuple[str, ...]
    trials: int
    best_metric: float | None
    verdict: str | None
    promoted: bool | None
    oracle: tuple[str, ...] = ()

    @property
    def outcome(self) -> str:
        """A display label: ``rejected`` / ``promoted`` / ``not promoted`` / ``undecided``."""
        if self.verdict is None:
            return "undecided"
        if self.verdict == "reject":
            return "rejected"
        if self.verdict == "approve":
            return "promoted" if self.promoted else "not promoted"
        return self.verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "thesis": self.thesis,
            "stages": list(self.stages),
            "trials": self.trials,
            "best_metric": self.best_metric,
            "verdict": self.verdict,
            "promoted": self.promoted,
            "outcome": self.outcome,
            "oracle": list(self.oracle),
        }


def _num(value: Any) -> float | None:
    """A tolerant numeric read: an int/float (not bool) as a float, else ``None``."""
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


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

    @classmethod
    def from_path(cls, path: str | Path) -> SessionLedger:
        """Reopen the ledger that *wrote* ``path`` (``<state_dir>/sessions/<id>.jsonl``) — the
        reader the CLOSE report uses to reach a session's ledger from the summary's stored path,
        without threading the state dir separately."""
        p = Path(path)
        return cls(p.parent.parent, p.stem)

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

    # ── derived views (story #74): the at-a-glance rollup + per-candidate trail ──────────────
    def rollup(self) -> SessionRollup:
        """Derive the :class:`SessionRollup` from this session's typed records — the source of
        truth for the session-log rollup and the CLOSE report. A file authored is one that reached
        OPTIMIZE (passed the write gate); an author stage that never did is a validation failure; a
        strategy authored but never carried to a verdict is undecided. Tolerant of an empty/absent
        ledger (all zeros)."""
        stages = self.stages()
        episodes = self.episodes()
        verdicts = self.verdicts()
        end = self.session_end()

        author_attempts = sum(1 for s in stages if s.stage == "author")
        optimized = [s for s in stages if s.stage == "optimize"]
        trials = sum(int(s.detail.get("trials") or 0) for s in optimized)

        verdict_counts: dict[str, int] = {}
        for v in verdicts:
            verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

        authored_names = {s.strategy for s in optimized if s.strategy}
        decided_names = {v.strategy for v in verdicts if v.strategy}

        tokens_by_stage: dict[str, int] = {}
        tokens_by_model: dict[str, int] = {}
        for e in episodes:
            tokens_by_stage[e.stage] = tokens_by_stage.get(e.stage, 0) + e.tokens
            tokens_by_model[e.model] = tokens_by_model.get(e.model, 0) + e.tokens

        return SessionRollup(
            session_id=self.session_id,
            theses=len(self.theses()),
            authored=len(optimized),
            validation_failures=author_attempts - len(optimized),
            trials=trials,
            verdicts=verdict_counts,
            promoted=sum(1 for v in verdicts if v.promoted),
            undecided=len(authored_names - decided_names),
            escalations=sum(1 for e in episodes if e.stage == "author" and e.escalated),
            tokens_by_stage=tokens_by_stage,
            tokens_by_model=tokens_by_model,
            note=end.note if end else None,
        )

    def usage_totals(self) -> dict[str, int] | None:
        """This session's token split, summed off its own episode lines — or ``None`` (story #140).

        The accounting twin of :meth:`rollup`: derived from what the ledger already holds, never a
        counter the driver carries. ``None`` when the session journaled no episodes at all, and
        when **any** episode that spent tokens journaled no split — a partial sum presented as the
        session's usage would understate it, and the price table would then bill a number nobody
        measured. The totals it *can* report (``tokens_by_stage``/``tokens_by_model`` on the
        rollup) are unaffected: an unknown split is not an unknown total.
        """
        episodes = self.episodes()
        if not episodes:
            return None
        totals = dict.fromkeys(USAGE_FIELDS, 0)
        for episode in episodes:
            usage = episode_usage(episode)
            if usage is None:
                return None
            for name in USAGE_FIELDS:
                totals[name] += usage[name]
        return totals

    def candidate_trails(self) -> list[CandidateTrail]:
        """One :class:`CandidateTrail` per formulated thesis, in ledger (formulate) order — the
        per-candidate stage trail the CLOSE report renders so a post-mortem walks structured
        records instead of a transcript."""
        stages = self.stages()
        verdicts = {v.strategy: v for v in self.verdicts()}
        trails: list[CandidateTrail] = []
        for t in self.theses():
            name = t.strategy
            my_stages = [s for s in stages if s.strategy == name]
            optimize = next((s for s in my_stages if s.stage == "optimize"), None)
            author = next((s for s in my_stages if s.stage == "author"), None)
            verdict = verdicts.get(name)
            oracle = tuple(str(x) for x in (author.detail.get("oracle") or ())) if author else ()
            trails.append(
                CandidateTrail(
                    strategy=name,
                    thesis=t.text,
                    stages=tuple(s.stage for s in my_stages),
                    trials=int(optimize.detail.get("trials") or 0) if optimize else 0,
                    best_metric=_num(optimize.detail.get("best_metric")) if optimize else None,
                    verdict=verdict.verdict if verdict else None,
                    promoted=verdict.promoted if verdict else None,
                    oracle=oracle,
                )
            )
        return trails

    def report_view(self) -> dict[str, Any] | None:
        """The JSON-safe ``{session_id, rollup, candidates}`` the CLOSE report threads into its
        research block, or ``None`` when the ledger holds nothing (absent/empty/all-malformed) so
        the report degrades to its ledgerless rendering. Never raises — a report is evidence, not a
        gate."""
        if not self.records():
            return None
        return {
            "session_id": self.session_id,
            "rollup": self.rollup().to_dict(),
            "candidates": [c.to_dict() for c in self.candidate_trails()],
        }

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
        brief_sha256: str | None = None,
        spec_sha256: str | None = None,
    ) -> None:
        """One stage-transition line. ``detail`` is an optional structured payload the stage
        carries (e.g. MATCH's screened profile / fit set / reserved holdout); an absent detail is
        omitted from the record rather than written as an empty object, so a tolerant read
        distinguishes "no detail" from a stored empty one.

        ``brief_sha256`` / ``spec_sha256`` are the AUTHOR stage's capture references (#184) — the
        content hashes of the captured brief and compiled spec suite, whose bodies sit in the run's
        ``qa/`` capture area. A hash the capture store never returned (nothing captured, or the
        store latched off) is simply omitted, so the ledger never names a body that is not on disk.
        """
        record: dict[str, Any] = {"event": "stage", "at": _now_iso(), "stage": stage}
        if strategy is not None:
            record["strategy"] = strategy
        if detail:
            record["detail"] = dict(detail)
        if brief_sha256:
            record["brief_sha256"] = str(brief_sha256)
        if spec_sha256:
            record["spec_sha256"] = str(spec_sha256)
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
        checks: list[dict[str, Any]] | None = None,
        misfire_details: list[dict[str, Any]] | None = None,
        note: str | None = None,
        usage: Mapping[str, int] | None = None,
        served_model: str | None = None,
    ) -> None:
        """One episode line. ``checks`` is the optional driver-side sanity-check payload (story
        #71) — a list of ``{"check", "result"}`` entries for the checks that fired on this
        episode's output. ``misfire_details`` is the optional per-misfire diagnostic payload
        (#102) — the runner's ``{"note", "raw"}`` entries pairing each rejected attempt's
        classifier note with a capped excerpt of what came back, so an exhausted episode is
        diagnosable from the ledger — and ``note`` the episode's last misfire/error text (the
        API-error reason when no misfire detail exists). ``usage`` is the optional four-field token
        split behind ``tokens`` (#140) — the shape a price table can bill, journaled here so the
        run record derives spend from the ledger instead of from a counter. ``served_model`` is the
        id the provider said actually answered this episode (#181), recorded beside the ``model``
        alias that was requested, so a same-alias/different-served-model event is detectable from
        the ledger alone. Every absent/empty optional is omitted from the record rather than
        written as an empty field, so a tolerant read distinguishes "nothing carried" from a
        stored empty one."""
        record: dict[str, Any] = {
            "event": "episode",
            "at": _now_iso(),
            "stage": stage,
            "model": model,
            "tokens": int(tokens),
            "misfires": int(misfires),
            "outcome": outcome,
            "escalated": bool(escalated),
        }
        if checks:
            record["checks"] = [dict(c) for c in checks]
        if misfire_details:
            record["misfire_details"] = [dict(d) for d in misfire_details]
        if note:
            record["note"] = note
        if served_model:
            record["served_model"] = str(served_model)
        if usage is not None:
            record["usage"] = {name: int(usage.get(name, 0) or 0) for name in USAGE_FIELDS}
        self._append(record)

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
