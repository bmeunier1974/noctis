"""The cross-session exhausted-class registry — a research-hygiene guard.

When the research agent concludes that a whole *class* of strategy is a dead end (not just
one parameterization), it records that class here via ``reject_strategy`` with
``class_exhausted=True`` and a ``class_tag``. A later ``write_strategy`` tagged with the same
class is refused unless the author names a genuinely new ``new_lever`` the exhaustion
post-mortems did not cover — so the loop cannot bleed budget re-deriving a conclusion a prior
session already reached, across process restarts.

This is a pure *don't-repeat-yourself* guard, keyed entirely on the agent's OWN runtime reject
verdicts — never on any hardcoded class list, and never on the operator mandate (which stays a
research prior, not an engine constraint). It promotes nothing and loosens no gate; deleting
the JSON file simply forgets the accumulated dead ends. Classes match on a normalized tag
(lowercased, whitespace-collapsed), so surfacing the existing tags back to the agent — see
``ResearchToolbox.market_context`` — keeps the wording stable enough to actually collide.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize(tag: str) -> str:
    """Fold a free-text class tag to a stable match key."""
    return " ".join(str(tag).split()).lower()


class ExhaustedClassRegistry:
    """A JSON-backed set of strategy classes the agent has declared exhausted."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def is_exhausted(self, tag: str) -> dict | None:
        """The matching record if ``tag``'s class was declared exhausted, else ``None``."""
        key = _normalize(tag)
        if not key:
            return None
        for rec in self.load():
            if rec.get("class_tag") == key:
                return rec
        return None

    def record(self, tag: str, reason: str, example: str | None = None) -> dict:
        """Mark a class exhausted (upsert): refresh its reason, append the example strategy."""
        key = _normalize(tag)
        if not key:
            raise ValueError("class_tag must be a non-empty string")
        records = self.load()
        for rec in records:
            if rec.get("class_tag") == key:
                rec["reason"] = reason
                rec["at"] = _now_iso()
                examples = rec.setdefault("examples", [])
                if example and example not in examples:
                    examples.append(example)
                self._save(records)
                return rec
        rec = {
            "class_tag": key,
            "label": " ".join(str(tag).split()),
            "reason": reason,
            "examples": [example] if example else [],
            "at": _now_iso(),
        }
        records.append(rec)
        self._save(records)
        return rec

    def summary(self, reason_chars: int = 240) -> list[dict]:
        """Compact view for the session-start digest: label, truncated reason, examples."""
        out: list[dict] = []
        for rec in self.load():
            reason = rec.get("reason", "")
            if len(reason) > reason_chars:
                reason = reason[: reason_chars - 1].rstrip() + "…"
            out.append(
                {
                    "class_tag": rec.get("label") or rec.get("class_tag", ""),
                    "reason": reason,
                    "examples": rec.get("examples", []),
                }
            )
        return out

    def _save(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
