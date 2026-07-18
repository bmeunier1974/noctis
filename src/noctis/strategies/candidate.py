"""A strategy candidate — a family name plus a concrete parameter set.

Deliberately just data (picklable), so candidates can cross process boundaries by name;
``build`` resolves the name against whichever :class:`FamilyRegistry` the caller holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noctis.strategies.base import TraderStrategy
from noctis.strategies.families import FamilyRegistry


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict = field(default_factory=dict)

    def build(self, families: FamilyRegistry) -> TraderStrategy:
        return families.create(self.family, dict(self.params))

    def key(self) -> str:
        items = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.family}({items})"
