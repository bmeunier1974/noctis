"""The champion registry — persistent, atomic, restart-surviving.

Holds the current champion set (capacity = ``champion_count``) plus a capped decision
history so the close report and memory can explain champion changes. Persisted as JSON with
atomic writes (temp file + rename) so a crash mid-write can never corrupt the file.
"""

from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from noctis.backtest.scorecard import Scorecard
from noctis.champions.promotion import Decision, PromotionRules, decide

_HISTORY_CAP = 200


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ChampionEntry:
    family: str
    params: dict
    scorecard: Scorecard
    crowned_at: str
    rationale: str
    # Symbols the champion was fit/validated on, and the symbols it should trade live.
    # ``None`` (as loaded from an older champions.json) = trade the whole universe — the
    # exact pre-panel behavior. The symbols live here, not on Candidate: the fit set is a
    # property of the family's research run, not of one parameter draw.
    fit_symbols: list[str] | None = None
    live_symbols: list[str] | None = None
    # The active mandate's provenance when this champion was crowned ("profile:aggressive",
    # "mandate/MANDATE.md", "cli", "auto"), or ``None`` for the legacy loop / an older
    # champions.json without the key. Read by the `auto` selection rule to attribute each
    # champion to the profile that produced it.
    mandate_source: str | None = None

    @property
    def test_metric(self) -> float:
        return self.scorecard.avg_test_metric

    @property
    def gap(self) -> float:
        return self.scorecard.gap

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "params": self.params,
            # Compact before persisting: a champion is only ever read back for its aggregate
            # metrics, so collapsing each symbol's (up to thousands of) walk-forward splits to a
            # single mean split keeps champions.json small without changing any promotion number.
            "scorecard": self.scorecard.compact().to_dict(),
            "crowned_at": self.crowned_at,
            "rationale": self.rationale,
            "fit_symbols": self.fit_symbols,
            "live_symbols": self.live_symbols,
            "mandate_source": self.mandate_source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChampionEntry:
        return cls(
            family=data["family"],
            params=data["params"],
            scorecard=Scorecard.from_dict(data["scorecard"]),
            crowned_at=data["crowned_at"],
            rationale=data["rationale"],
            fit_symbols=data.get("fit_symbols"),
            live_symbols=data.get("live_symbols"),
            mandate_source=data.get("mandate_source"),
        )


class ChampionRegistry:
    """JSON-persisted champion set with atomic writes and a decision history."""

    def __init__(self, path: str | Path, capacity: int):
        self.path = Path(path)
        self.capacity = int(capacity)
        self.champions: list[ChampionEntry] = []
        self.history: list[dict] = []
        self.load()

    # --- persistence ---
    def load(self) -> None:
        if not self.path.is_file():
            self.champions = []
            self.history = []
            return
        data = json.loads(self.path.read_text())
        self.champions = [ChampionEntry.from_dict(c) for c in data.get("champions", [])]
        self.history = data.get("history", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "champions": [c.to_dict() for c in self.champions],
            "history": self.history[-_HISTORY_CAP:],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)  # atomic on POSIX

    # --- decisions ---
    def consider(
        self,
        challenger: Scorecard,
        rules: PromotionRules,
        *,
        mandate_source: str | None = None,
    ) -> Decision:
        """Judge a challenger and apply the outcome (promote/demote), persisting the result.

        ``mandate_source`` (keyword-only, defaults ``None``) is stamped onto the crowned
        champion for provenance — the active mandate that produced it. The legacy loop and
        every existing caller omit it, keeping the pre-mandate behavior untouched.
        """
        decision = decide(challenger, [c.scorecard for c in self.champions], rules)
        record = {
            "at": _now(),
            "family": challenger.family,
            "params": challenger.params,
            "test_metric": round(challenger.avg_test_metric, 10),
            "gap": round(challenger.gap, 10),
            "promoted": decision.promote,
            "rationale": decision.rationale,
        }

        if decision.promote:
            if decision.demote_index is not None:
                demoted = self.champions.pop(decision.demote_index)
                record["demoted"] = {
                    "family": demoted.family,
                    "params": demoted.params,
                    "test_metric": round(demoted.test_metric, 10),
                }
            # The scorecard's fit symbols bind to the champion (a panel of one binds its
            # one symbol); live set = fit set until the screener (P3) widens it. A legacy
            # sentinel-only card binds nothing, leaving the pre-panel "trade the whole
            # universe" eligibility (fit_symbols=None).
            fit_symbols = challenger.fit_symbols or None
            self.champions.append(
                ChampionEntry(
                    family=challenger.family,
                    params=dict(challenger.params),
                    scorecard=challenger,
                    crowned_at=_now(),
                    rationale=decision.rationale,
                    fit_symbols=fit_symbols,
                    live_symbols=list(fit_symbols) if fit_symbols else None,
                    mandate_source=mandate_source,
                )
            )
        self.history.append(record)
        self.save()
        return decision

    def reset(self, reason: str) -> int:
        """Drop every champion (history is kept), recording why. Returns how many dropped.

        The lever for regime changes the lazy stale-champion rule cannot express — e.g.
        champions crowned before a gate existed, whose frozen scorecards would never face it.
        """
        dropped = len(self.champions)
        for entry in self.champions:
            self.history.append(
                {
                    "at": _now(),
                    "family": entry.family,
                    "params": entry.params,
                    "test_metric": round(entry.test_metric, 10),
                    "gap": round(entry.gap, 10),
                    "promoted": False,
                    "rationale": f"reset: {reason}",
                    "demoted": {
                        "family": entry.family,
                        "params": entry.params,
                        "test_metric": round(entry.test_metric, 10),
                    },
                }
            )
        self.champions = []
        self.save()
        return dropped

    # --- reads ---
    def list(self) -> list[ChampionEntry]:
        return list(self.champions)

    def is_empty(self) -> bool:
        return not self.champions

    # builtins.list: the sibling ``list`` method shadows the builtin in this class scope.
    def demotions(self) -> builtins.list[dict]:
        return [h for h in self.history if h.get("demoted")]
