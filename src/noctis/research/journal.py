"""The experiment journal — the durable record of research evidence, behind one interface.

``state/experiments/<strategy>.jsonl`` holds one JSON line per research event: every
``run_backtest`` call and ``run_sweep`` trial (``trial``), sweep completion
(``sweep_complete``), the class a ``write_strategy`` declared (``class_tag``), the
motivating idea it authored (``thesis`` — the prose plus optional ``parent_thesis`` /
``pivot_rationale`` lineage a later session or report can walk instead of re-parsing the
file), and every verdict spent (``verdict``). The journal — never the agent's context — is
the ground truth the research discipline reads: the exhaustion gate counts distinct
journaled param sets, the symbol-holdout taint check scans journaled trial symbols, and
``reject_strategy`` recovers the best-observed params from journaled trials. That is why the
record schema lives here and nowhere else: every writer is an explicit ``record_*`` method
and every reader gets typed views, so no caller re-parses ``event`` strings by hand.

The schema is *extended, never changed*: one writer per kind, tolerant reads. A malformed
line is skipped (a corrupt record can't confirm anything) and an unknown record kind an
older reader never learned is ignored by the typed views rather than being fatal, so an
existing journal keeps loading as new kinds land. Appends are line-atomic per strategy, and
the toolbox keeps journaling parent-side so there is exactly one writer per session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from noctis.backtest import Scorecard

EXPERIMENTS_DIRNAME = "experiments"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _round(value: Any, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _opt_str(value: Any) -> str | None:
    """A tolerant optional-text read: absent/empty stays ``None``, anything else stringifies."""
    return str(value) if value else None


@dataclass(frozen=True)
class JournalStats:
    """What the exhaustion gate reads: how thoroughly a strategy's space was explored."""

    n_trials: int = 0
    n_distinct_params: int = 0
    sweep_completed: bool = False


@dataclass(frozen=True)
class Trial:
    """One journaled backtest/sweep evaluation — a typed view of a ``trial`` line."""

    at: str
    source: str  # "backtest" | "sweep"
    symbols: list[str]
    params: dict[str, Any]
    window: dict[str, Any]
    metrics: dict[str, Any]  # stage / metric_name / train / test / gap / holdout
    max_bars: int | None  # truncation cap — a truthy value marks exploration fidelity

    @property
    def test(self) -> float | None:
        return self.metrics.get("test")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Trial:
        max_bars = record.get("max_bars")
        return cls(
            at=str(record.get("at", "")),
            source=str(record.get("source", "")),
            symbols=[str(s) for s in record.get("symbols") or []],
            params=dict(record.get("params") or {}),
            window=dict(record.get("window") or {}),
            metrics=dict(record.get("metrics") or {}),
            max_bars=int(max_bars) if max_bars else None,
        )


@dataclass(frozen=True)
class Thesis:
    """One journaled thesis with its lineage — a typed view of a ``thesis`` line.

    ``text`` is the motivating idea in prose; ``parent_thesis`` / ``pivot_rationale`` are the
    optional lineage a pivot chain walks (both ``None`` when the thesis stands on its own).
    """

    at: str
    text: str
    parent_thesis: str | None = None
    pivot_rationale: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Thesis:
        return cls(
            at=str(record.get("at", "")),
            text=str(record.get("thesis", "")),
            parent_thesis=_opt_str(record.get("parent_thesis")),
            pivot_rationale=_opt_str(record.get("pivot_rationale")),
        )


class ExperimentJournal:
    """Append-only per-strategy journals under ``<state_dir>/experiments/``."""

    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir) / EXPERIMENTS_DIRNAME

    def path(self, name: str) -> Path:
        return self.root / f"{name}.jsonl"

    # ── reads ────────────────────────────────────────────────────────────────
    def records(self, name: str) -> list[dict[str, Any]]:
        """Every parseable record, in journal (append) order."""
        path = self.path(name)
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def trials(self, name: str) -> list[Trial]:
        return [Trial.from_record(r) for r in self.records(name) if r.get("event") == "trial"]

    def trials_by_test(self, name: str) -> list[Trial]:
        """Trials ranked best-first by test metric; trials with no test score rank last."""
        return sorted(
            self.trials(name),
            key=lambda t: t.test if t.test is not None else float("-inf"),
            reverse=True,
        )

    def stats(self, name: str) -> JournalStats:
        records = self.records(name)
        trials = [r for r in records if r.get("event") == "trial"]
        distinct = {json.dumps(r.get("params", {}), sort_keys=True) for r in trials}
        return JournalStats(
            n_trials=len(trials),
            n_distinct_params=len(distinct),
            sweep_completed=any(r.get("event") == "sweep_complete" for r in records),
        )

    def verdicts(self, name: str) -> list[dict[str, Any]]:
        """Verdict records as journaled — surfaced verbatim in ``get_experiment_log``."""
        return [r for r in self.records(name) if r.get("event") == "verdict"]

    def class_tag(self, name: str) -> str | None:
        """The most recently journaled ``class_tag`` for ``name`` (or ``None``)."""
        tag = None
        for rec in self.records(name):
            if rec.get("event") == "class_tag" and rec.get("class_tag"):
                tag = str(rec["class_tag"])
        return tag

    def thesis(self, name: str) -> Thesis | None:
        """The most recently journaled ``thesis`` for ``name``, typed with lineage (or ``None``)."""
        latest: dict[str, Any] | None = None
        for rec in self.records(name):
            if rec.get("event") == "thesis" and rec.get("thesis"):
                latest = rec
        return Thesis.from_record(latest) if latest is not None else None

    def touched_symbols(self, name: str) -> set[str]:
        """Every symbol any journaled trial ever tuned on — the holdout taint set."""
        return {s for trial in self.trials(name) for s in trial.symbols}

    # ── writes ───────────────────────────────────────────────────────────────
    def record_trial(
        self,
        name: str,
        *,
        source: str,
        symbols: Iterable[str],
        params: dict[str, Any],
        window: dict[str, Any],
        card: Scorecard,
        max_bars: int | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "trial",
            "at": _now_iso(),
            "source": source,
            "strategy": name,
            "symbols": list(symbols),
            "params": params,
            "window": window,
            "metrics": {
                "stage": card.stage,
                "metric_name": card.metric_name,
                "train": _round(card.avg_train_metric),
                "test": _round(card.avg_test_metric),
                "gap": _round(card.gap),
                "holdout": _round(card.holdout_metric),
            },
        }
        if max_bars is not None:
            record["max_bars"] = int(max_bars)
        self._append(name, record)

    def record_sweep_complete(
        self, name: str, *, n_trials: int, symbols: Iterable[str], max_bars: int | None = None
    ) -> None:
        self._append(
            name,
            {
                "event": "sweep_complete",
                "at": _now_iso(),
                "n_trials": int(n_trials),
                "symbols": list(symbols),
                **({"max_bars": int(max_bars)} if max_bars else {}),
            },
        )

    def record_class_tag(self, name: str, class_tag: str) -> None:
        self._append(name, {"event": "class_tag", "at": _now_iso(), "class_tag": class_tag})

    def record_thesis(
        self,
        name: str,
        thesis: str,
        *,
        parent_thesis: str | None = None,
        pivot_rationale: str | None = None,
    ) -> None:
        """Journal the motivating idea at author time, beside the class-tag record.

        ``parent_thesis`` / ``pivot_rationale`` are the optional lineage a pivot chain walks;
        an absent field is omitted from the record rather than written as null, so a tolerant
        read distinguishes "no lineage" from a stored empty value.
        """
        record: dict[str, Any] = {"event": "thesis", "at": _now_iso(), "thesis": thesis}
        if parent_thesis is not None:
            record["parent_thesis"] = parent_thesis
        if pivot_rationale is not None:
            record["pivot_rationale"] = pivot_rationale
        self._append(name, record)

    def record_approval(
        self,
        name: str,
        *,
        promoted: bool,
        rationale: str,
        params: dict[str, Any],
        symbols: Iterable[str],
        holdout_symbols: Iterable[str],
    ) -> None:
        self._append(
            name,
            {
                "event": "verdict",
                "at": _now_iso(),
                "verdict": "approve",
                "promoted": promoted,
                "rationale": rationale,
                "params": params,
                "symbols": list(symbols),
                "holdout_symbols": list(holdout_symbols),
            },
        )

    def record_rejection(self, name: str, *, reason: str, best_params: dict[str, Any]) -> None:
        self._append(
            name,
            {
                "event": "verdict",
                "at": _now_iso(),
                "verdict": "reject",
                "reason": reason,
                "best_params": best_params,
            },
        )

    def _append(self, name: str, record: dict[str, Any]) -> None:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
