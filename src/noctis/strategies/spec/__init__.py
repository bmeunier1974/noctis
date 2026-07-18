"""Agent-first strategy engine: a ``StrategySpec`` compiles to an ordinary ``TraderStrategy``.

Standalone — no grid-mng runtime dependency. grid-mng is a copy-from reference for the spec
shape, the indicator math, and the golden fixtures only.
"""

from __future__ import annotations

from .schema import StrategySpec, to_param_space, validate_spec
from .strategy import (
    SpecStrategy,
    family_class_from_spec,
    load_and_register,
    persisted_spec_json,
    register_spec,
    spec_family_names,
)

__all__ = [
    "StrategySpec",
    "to_param_space",
    "validate_spec",
    "SpecStrategy",
    "family_class_from_spec",
    "register_spec",
    "load_and_register",
    "persisted_spec_json",
    "spec_family_names",
]
