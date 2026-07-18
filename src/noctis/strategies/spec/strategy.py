"""Compile a ``StrategySpec`` into a registerable :class:`~noctis.strategies.base.TraderStrategy`.

Each spec becomes exactly **one family** whose two code paths come from the shared spec
interpreter, so ``signals()`` and ``on_bar()`` agree by construction. From there it flows
through the ordinary proposer/Optuna → prefilter → validate → promotion pipeline: the spec's
``parameters`` become the family's params dataclass and its ``optimizations`` become
``param_space()``.

``register_spec`` also persists the raw spec JSON to ``state/specs.json`` (atomic write,
mirroring the champion registry), and ``load_and_register`` rebuilds + registers every spec at
startup — that is what lets a promoted spec-family survive a restart (``champions.json`` stores
only ``{family, params}``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, make_dataclass
from pathlib import Path

import pandas as pd

from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy
from noctis.strategies.families import FamilyRegistry

from .interpreter import SpecRuntime
from .schema import StrategySpec, to_param_space

_SPECS_FILE = "specs.json"


def _params_dataclass(spec: StrategySpec) -> type:
    """Build a frozen dataclass with one field per ``ParameterSpec`` (name=id, default=value)."""
    fields = [
        (
            p.id,
            int if p.kind == "int" else float,
            (int(p.value) if p.kind == "int" else float(p.value)),
        )
        for p in spec.parameters
    ]
    return make_dataclass(f"{spec.id}_Params", fields, frozen=True)


class SpecStrategy(TraderStrategy):
    """Base for spec-compiled strategies. Concrete families set ``spec`` + ``params_cls`` via
    :func:`family_class_from_spec`."""

    spec: StrategySpec
    name = "spec"

    @classmethod
    def signals(cls, data: pd.DataFrame, params) -> pd.Series:
        runtime = SpecRuntime(cls.spec, asdict(params))
        return runtime.signals(data)

    def on_start(self, ctx: Context) -> None:
        self._eval = SpecRuntime(self.spec, asdict(self.params)).new_incremental()

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        ctx.set_target(self._eval.step(bar))

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return to_param_space(cls.spec)


def family_class_from_spec(spec: StrategySpec) -> type[TraderStrategy]:
    """Build a registerable ``TraderStrategy`` subclass with ``name = spec.id``."""
    return type(
        spec.id,
        (SpecStrategy,),
        {"name": spec.id, "spec": spec, "params_cls": _params_dataclass(spec)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — state/specs.json (atomic, restart-surviving, idempotent)
# ─────────────────────────────────────────────────────────────────────────────
def _specs_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / _SPECS_FILE


def _load_raw(state_dir: str | Path) -> dict:
    path = _specs_path(state_dir)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return data.get("specs", {})


def _write_raw(state_dir: str | Path, specs: dict) -> None:
    path = _specs_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"version": 1, "specs": specs}, indent=2, sort_keys=True))
    tmp.replace(path)  # atomic on POSIX


def register_spec(
    spec: StrategySpec, state_dir: str | Path, families: FamilyRegistry
) -> type[TraderStrategy]:
    """Register the spec as a family and persist its JSON. Idempotent on re-register."""
    cls = family_class_from_spec(spec)
    families.register(cls)
    specs = _load_raw(state_dir)
    specs[spec.id] = spec.model_dump(mode="json", by_alias=True)
    _write_raw(state_dir, specs)
    return cls


def load_and_register(state_dir: str | Path, families: FamilyRegistry) -> list[str]:
    """Rebuild + register every persisted spec at startup. Returns the family names."""
    specs = _load_raw(state_dir)
    names: list[str] = []
    for raw in specs.values():
        spec = StrategySpec.model_validate(raw)
        families.register(family_class_from_spec(spec))
        names.append(spec.id)
    return names


def spec_family_names(state_dir: str | Path) -> list[str]:
    """The ids of every persisted spec-family — used to flag which champions are minted specs."""
    return list(_load_raw(state_dir).keys())


def persisted_spec_json(state_dir: str | Path, spec_id: str) -> dict | None:
    """The raw persisted JSON for one spec-family, or ``None`` if it was never persisted.
    Lets the ideation collision guard tell an identical re-proposal from a real clash."""
    return _load_raw(state_dir).get(spec_id)
