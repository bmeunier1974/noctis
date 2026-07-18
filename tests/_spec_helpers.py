"""Spec-engine test helpers: the 3 seeds re-expressed as specs, grid-mng fixture families,
and a random valid-spec generator for the generic parity test."""

from __future__ import annotations

import numpy as np

from noctis.strategies.spec.schema import StrategySpec

_SOURCE = {"id": "src", "schema": "ohlcv-1m"}


# ── The three seeds, as specs (must match the hand-coded seeds on the fixtures) ──────────
def seed_sma_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": "spec_sma_crossover",
            "sources": [_SOURCE],
            "parameters": [
                {"id": "fast", "kind": "int", "value": 10},
                {"id": "slow", "kind": "int", "value": 30},
            ],
            "features": [
                {"id": "f_fast", "kind": "sma", "input": "src", "period": "fast"},
                {"id": "f_slow", "kind": "sma", "input": "src", "period": "slow"},
            ],
            "signals": [
                {"id": "enter", "kind": "condition", "op": ">", "a": "f_fast", "b": "f_slow"},
                {"id": "exit", "kind": "condition", "op": "<=", "a": "f_fast", "b": "f_slow"},
            ],
            "entries": [{"id": "e", "enter": "enter", "exit": "exit"}],
            "optimizations": [
                {
                    "id": "opt",
                    "parameters": [
                        {"param": "fast", "type": "int", "min": 3, "max": 30, "step": 1},
                        {"param": "slow", "type": "int", "min": 20, "max": 100, "step": 1},
                    ],
                }
            ],
        }
    )


def seed_rsi_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": "spec_rsi_meanrev",
            "sources": [_SOURCE],
            "parameters": [
                {"id": "period", "kind": "int", "value": 14},
                {"id": "oversold", "kind": "float", "value": 30.0},
                {"id": "overbought", "kind": "float", "value": 70.0},
            ],
            "features": [{"id": "r", "kind": "rsi", "input": "src", "period": "period"}],
            "signals": [
                {"id": "enter", "kind": "condition", "op": "<", "a": "r", "threshold": "oversold"},
                {"id": "exit", "kind": "condition", "op": ">", "a": "r", "threshold": "overbought"},
            ],
            "entries": [{"id": "e", "enter": "enter", "exit": "exit"}],
        }
    )


def seed_donchian_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": "spec_donchian_breakout",
            "sources": [_SOURCE],
            "parameters": [{"id": "channel", "kind": "int", "value": 20}],
            "features": [
                {
                    "id": "hi",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "max",
                    "period": "channel",
                    "field": "high",
                },
                {
                    "id": "lo",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "min",
                    "period": "channel",
                    "field": "low",
                },
            ],
            "signals": [
                {"id": "enter", "kind": "condition", "op": ">", "a": "src", "b": "hi"},
                {"id": "exit", "kind": "condition", "op": "<", "a": "src", "b": "lo"},
            ],
            "entries": [{"id": "e", "enter": "enter", "exit": "exit"}],
        }
    )


# ── grid-mng specFixtures families the vocabulary must also cover ────────────────────────
def classic_breakout_spec() -> StrategySpec:
    """H52 — buy when close crosses above the prior 20-bar high; exit below the 5-bar low."""
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": "spec_classic_breakout",
            "sources": [_SOURCE],
            "features": [
                {
                    "id": "hi20",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "max",
                    "period": 20,
                    "field": "high",
                },
                {
                    "id": "lo5",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "min",
                    "period": 5,
                    "field": "low",
                },
            ],
            "signals": [
                {"id": "brk", "kind": "condition", "op": "cross_above", "a": "src", "b": "hi20"},
                {"id": "stop", "kind": "condition", "op": "cross_below", "a": "src", "b": "lo5"},
            ],
            "entries": [{"id": "e", "enter": "brk", "exit": "stop"}],
        }
    )


def zscore_reversion_spec() -> StrategySpec:
    """H17 — z-score mean-reversion on RSI(14): z < −2 buy, z > +2 exit."""
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": "spec_zscore_reversion",
            "sources": [_SOURCE],
            "features": [
                {"id": "r", "kind": "rsi", "input": "src", "period": 14},
                {
                    "id": "z",
                    "kind": "zScore",
                    "input": "r",
                    "lookback": 20,
                    "upperThreshold": 2.0,
                    "lowerThreshold": -2.0,
                },
            ],
            "signals": [],
            "entries": [{"id": "e", "enter": "z:below", "exit": "z:above"}],
        }
    )


FIXTURE_FAMILIES = [classic_breakout_spec, zscore_reversion_spec]


# ── Random valid-spec generator (structure varies; every spec runs) ─────────────────────
def _p(name, kind, value):
    return {"id": name, "kind": kind, "value": value}


def random_valid_spec(rng: np.random.Generator, idx: int) -> StrategySpec:
    """Pick a template and randomize its periods/thresholds; always schema-valid + runnable."""
    templates = [
        _sma_cross,
        _ema_cross,
        _macd_cross,
        _rsi_meanrev,
        _donchian,
        _zscore_rev,
        _breakout,
        _ensemble_and,
    ]
    return templates[idx % len(templates)](rng, idx)


def _sma_cross(rng, idx):
    fast = int(rng.integers(3, 15))
    slow = int(rng.integers(fast + 5, 40))
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_sma_{idx}",
            "sources": [_SOURCE],
            "features": [
                {"id": "f", "kind": "sma", "input": "src", "period": fast},
                {"id": "s", "kind": "sma", "input": "src", "period": slow},
            ],
            "signals": [
                {"id": "en", "kind": "condition", "op": ">", "a": "f", "b": "s"},
                {"id": "ex", "kind": "condition", "op": "<=", "a": "f", "b": "s"},
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )


def _ema_cross(rng, idx):
    fast = int(rng.integers(3, 15))
    slow = int(rng.integers(fast + 5, 40))
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_ema_{idx}",
            "sources": [_SOURCE],
            "features": [
                {"id": "f", "kind": "ema", "input": "src", "period": fast},
                {"id": "s", "kind": "ema", "input": "src", "period": slow},
            ],
            "signals": [
                {"id": "en", "kind": "condition", "op": "cross_above", "a": "f", "b": "s"},
                {"id": "ex", "kind": "condition", "op": "cross_below", "a": "f", "b": "s"},
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )


def _macd_cross(rng, idx):
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_macd_{idx}",
            "sources": [_SOURCE],
            "features": [
                {
                    "id": "m",
                    "kind": "macd",
                    "input": "src",
                    "fastPeriod": int(rng.integers(6, 12)),
                    "slowPeriod": int(rng.integers(20, 30)),
                    "signalPeriod": int(rng.integers(5, 11)),
                }
            ],
            "signals": [
                {
                    "id": "en",
                    "kind": "condition",
                    "op": "cross_above",
                    "a": "m:macd",
                    "b": "m:signal",
                },
                {
                    "id": "ex",
                    "kind": "condition",
                    "op": "cross_below",
                    "a": "m:macd",
                    "b": "m:signal",
                },
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )


def _rsi_meanrev(rng, idx):
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_rsi_{idx}",
            "sources": [_SOURCE],
            "features": [
                {"id": "r", "kind": "rsi", "input": "src", "period": int(rng.integers(5, 20))}
            ],
            "signals": [
                {
                    "id": "en",
                    "kind": "condition",
                    "op": "<",
                    "a": "r",
                    "threshold": float(rng.integers(20, 40)),
                },
                {
                    "id": "ex",
                    "kind": "condition",
                    "op": ">",
                    "a": "r",
                    "threshold": float(rng.integers(60, 80)),
                },
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )


def _donchian(rng, idx):
    ch = int(rng.integers(5, 30))
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_don_{idx}",
            "sources": [_SOURCE],
            "features": [
                {
                    "id": "hi",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "max",
                    "period": ch,
                    "field": "high",
                },
                {
                    "id": "lo",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "min",
                    "period": ch,
                    "field": "low",
                },
            ],
            "signals": [
                {"id": "en", "kind": "condition", "op": ">", "a": "src", "b": "hi"},
                {"id": "ex", "kind": "condition", "op": "<", "a": "src", "b": "lo"},
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )


def _zscore_rev(rng, idx):
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_z_{idx}",
            "sources": [_SOURCE],
            "features": [
                {"id": "r", "kind": "rsi", "input": "src", "period": int(rng.integers(8, 16))},
                {
                    "id": "z",
                    "kind": "zScore",
                    "input": "r",
                    "lookback": int(rng.integers(10, 25)),
                    "upperThreshold": 1.5,
                    "lowerThreshold": -1.5,
                },
            ],
            "signals": [],
            "entries": [{"id": "e", "enter": "z:below", "exit": "z:above"}],
        }
    )


def _breakout(rng, idx):
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_brk_{idx}",
            "sources": [_SOURCE],
            "features": [
                {
                    "id": "hi",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "max",
                    "period": int(rng.integers(10, 30)),
                    "field": "high",
                },
                {
                    "id": "lo",
                    "kind": "rollingExtreme",
                    "input": "src",
                    "mode": "min",
                    "period": int(rng.integers(3, 10)),
                    "field": "low",
                },
            ],
            "signals": [
                {"id": "en", "kind": "condition", "op": "cross_above", "a": "src", "b": "hi"},
                {"id": "ex", "kind": "condition", "op": "cross_below", "a": "src", "b": "lo"},
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )


def _ensemble_and(rng, idx):
    fast = int(rng.integers(3, 12))
    slow = int(rng.integers(fast + 5, 35))
    return StrategySpec.model_validate(
        {
            "version": 1,
            "id": f"rand_ens_{idx}",
            "sources": [_SOURCE],
            "features": [
                {"id": "f", "kind": "sma", "input": "src", "period": fast},
                {"id": "s", "kind": "sma", "input": "src", "period": slow},
                {"id": "r", "kind": "rsi", "input": "src", "period": int(rng.integers(8, 16))},
            ],
            "signals": [
                {"id": "trend", "kind": "condition", "op": ">", "a": "f", "b": "s"},
                {"id": "nothot", "kind": "condition", "op": "<", "a": "r", "threshold": 70.0},
                {"id": "en", "kind": "ensemble", "method": "and", "inputs": ["trend", "nothot"]},
                {"id": "ex", "kind": "condition", "op": "<=", "a": "f", "b": "s"},
            ],
            "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        }
    )
