"""The memory seam.

A thin interface the research loop writes findings through and reads known dead ends from,
and the close phase runs upkeep (distillation, reorganize) against. The full human-readable
``MEMORY.md`` store implements this same protocol; an in-memory implementation lives here so
earlier layers and tests do not depend on the full store.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """What the research loop and the close phase need from memory."""

    def append_finding(self, note: str) -> None:
        """Record a noteworthy finding (promotion, dead end, surprising region)."""
        ...

    def record_rejected(self, family: str, params: dict, reason: str = "") -> None:
        """Remember a rejected idea so it is not re-tested."""
        ...

    def rejected_ideas(self) -> list[dict]:
        """Return known dead ends as ``{"family", "params", ...}`` dicts."""
        ...

    def findings(self) -> list[str]:
        """Return the noteworthy findings recorded so far (for the close-of-day report)."""
        ...

    def set_distilled(self, lines: list[str]) -> None:
        """Replace the distilled-lessons view (stage-2 distillation output)."""
        ...

    def distilled(self) -> list[str]:
        """Return the distilled lessons (empty until a distillation has run)."""
        ...

    def reorganize(self, registry=None) -> None:
        """Close-phase upkeep: dedup, refresh the champion view, enforce the size budget."""
        ...


class InMemoryMemory:
    """A non-persistent Memory (handy for tests and dry runs)."""

    def __init__(self) -> None:
        self._findings: list[str] = []
        self._rejected: list[dict] = []
        self._distilled: list[str] = []

    def append_finding(self, note: str) -> None:
        self._findings.append(note)

    def record_rejected(self, family: str, params: dict, reason: str = "") -> None:
        self._rejected.append({"family": family, "params": dict(params), "reason": reason})

    def rejected_ideas(self) -> list[dict]:
        return list(self._rejected)

    def findings(self) -> list[str]:
        return list(self._findings)

    # Distilled-lessons surface, in lockstep with MemoryStore (stage-2 distillation).
    def set_distilled(self, lines: list[str]) -> None:
        self._distilled = [ln for ln in lines if ln.strip()]

    def distilled(self) -> list[str]:
        return list(self._distilled)

    def reorganize(self, registry=None) -> None:
        """Nothing to reorganize — there is no file to dedup or budget."""
