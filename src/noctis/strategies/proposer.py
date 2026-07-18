"""Candidate proposer — samples strategy parameters across families.

Uses Optuna's ``TPESampler`` (seeded) when the ``research`` extra is installed so the search
learns from reported objectives; otherwise falls back to a seeded random sampler that is
fully deterministic. Either way it consults memory's rejected ideas and skips near-duplicate
parameter sets so known dead ends are not re-proposed.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from noctis.strategies.base import ParamSpec
from noctis.strategies.candidate import Candidate
from noctis.strategies.families import SEED_FAMILIES, FamilyRegistry


class Memoryish(Protocol):
    def rejected_ideas(self) -> list[dict]: ...


def _bucket(spec: ParamSpec, value) -> object:
    """Coarsen a value to a dedup bucket so near-duplicates share a signature."""
    if spec.kind == "categorical":
        return value
    step = spec.step or (1 if spec.kind == "int" else 0.0)
    if step:
        return round(float(value) / float(step))
    return round(float(value), 3)


def signature(family: str, param_space: list[ParamSpec], params: dict) -> tuple:
    """A dedup signature for a (family, params) pair based on bucketed values."""
    buckets = tuple((s.name, _bucket(s, params[s.name])) for s in param_space if s.name in params)
    return (family, buckets)


class CandidateProposer:
    """Proposes candidates across families, pruning known dead ends."""

    def __init__(
        self,
        families: FamilyRegistry | None = None,
        rotation: list[str] | None = None,
        seed: int = 0,
        memory: Memoryish | None = None,
        max_dedup_attempts: int = 25,
    ):
        # The registry resolves param spaces; the rotation is which names get proposed.
        self.families = families if families is not None else FamilyRegistry()
        self.rotation = rotation or list(SEED_FAMILIES)
        self.seed = seed
        self.memory = memory
        self.max_dedup_attempts = max_dedup_attempts
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._optuna = _try_build_optuna(self.families, self.rotation, seed)
        self._rejected: set[tuple] = set()
        self._load_rejected()

    def _load_rejected(self) -> None:
        if self.memory is None:
            return
        for idea in self.memory.rejected_ideas():
            fam = idea.get("family")
            params = idea.get("params", {})
            if fam in self.rotation:
                self._rejected.add(signature(fam, self.families.param_space(fam), params))

    def _sample_params(self, family: str) -> dict:
        space = self.families.param_space(family)
        params: dict = {}
        for spec in space:
            if spec.kind == "categorical":
                choices = list(spec.choices)
                params[spec.name] = choices[int(self._rng.integers(0, len(choices)))]
            elif spec.kind == "int":
                lo, hi = int(cast(float, spec.low)), int(cast(float, spec.high))
                params[spec.name] = int(self._rng.integers(lo, hi + 1))
            else:  # float
                lo_f, hi_f = float(cast(float, spec.low)), float(cast(float, spec.high))
                params[spec.name] = float(self._rng.uniform(lo_f, hi_f))
        return _repair(family, params)

    def _is_rejected(self, family: str, params: dict) -> bool:
        sig = signature(family, self.families.param_space(family), params)
        return sig in self._rejected

    def add_family(self, name: str) -> None:
        """Add a newly minted family to the rotation (idempotent). When an Optuna sampler is
        active it also gets a lazily-created study so ``propose`` can tune it immediately."""
        if name in self.rotation:
            return
        self.rotation.append(name)
        if self._optuna is not None:
            self._optuna.add_family(name)

    def propose(self) -> Candidate:
        """Return the next candidate, skipping near-duplicates of rejected ideas."""
        family = ""
        params: dict = {}
        for _ in range(self.max_dedup_attempts):
            family = self.rotation[self._cursor % len(self.rotation)]
            self._cursor += 1
            if self._optuna is not None:
                params = self._optuna.ask(family)
            else:
                params = self._sample_params(family)
            if not self._is_rejected(family, params):
                return Candidate(family, params)
        # Could not dodge the rejected region after many tries — return the last anyway.
        return Candidate(family, params)

    def tell(self, candidate: Candidate, objective: float) -> None:
        """Report the achieved objective so a learning sampler can adapt."""
        if self._optuna is not None:
            self._optuna.tell(candidate.family, candidate.params, objective)

    def reject(self, candidate: Candidate) -> None:
        """Locally mark a candidate's region as a dead end for this proposer instance."""
        space = self.families.param_space(candidate.family)
        self._rejected.add(signature(candidate.family, space, candidate.params))


def _repair(family: str, params: dict) -> dict:
    """Fix parameter constraints that free sampling can violate (e.g. fast < slow)."""
    if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
        params["slow"] = params["fast"] + 1
    if "oversold" in params and "overbought" in params:
        if params["oversold"] >= params["overbought"]:
            params["oversold"], params["overbought"] = 30.0, 70.0
    return params


def _try_build_optuna(families: FamilyRegistry, rotation: list[str], seed: int):
    """Build an Optuna-backed sampler if the library is available, else None."""
    try:
        import optuna
    except ImportError:
        return None
    return _OptunaSampler(optuna, families, rotation, seed)


class _OptunaSampler:  # pragma: no cover - exercised only with the research extra
    """One seeded TPE study per family, driven via ask()/tell()."""

    def __init__(self, optuna, families: FamilyRegistry, rotation: list[str], seed: int):
        self._optuna = optuna
        self._families = families
        self._seed = seed
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._studies = {fam: self._new_study() for fam in rotation}
        self._pending: dict[tuple, object] = {}

    def _new_study(self):
        return self._optuna.create_study(
            direction="maximize", sampler=self._optuna.samplers.TPESampler(seed=self._seed)
        )

    def add_family(self, name: str) -> None:
        """Lazily create the study for a family minted after init (``ask`` would else KeyError)."""
        if name not in self._studies:
            self._studies[name] = self._new_study()

    def _suggest(self, trial, spec: ParamSpec):
        if spec.kind == "categorical":
            return trial.suggest_categorical(spec.name, list(spec.choices))
        lo, hi = cast(float, spec.low), cast(float, spec.high)  # numeric kinds carry bounds
        if spec.kind == "int":
            return trial.suggest_int(spec.name, int(lo), int(hi))
        return trial.suggest_float(spec.name, float(lo), float(hi))

    def ask(self, family: str) -> dict:
        study = self._studies[family]
        trial = study.ask()
        space = self._families.param_space(family)
        params = {s.name: self._suggest(trial, s) for s in space}
        params = _repair(family, params)
        self._pending[(family, tuple(sorted(params.items())))] = trial
        return params

    def tell(self, family: str, params: dict, objective: float) -> None:
        key = (family, tuple(sorted(params.items())))
        trial = self._pending.pop(key, None)
        if trial is not None:
            self._studies[family].tell(trial, objective)
