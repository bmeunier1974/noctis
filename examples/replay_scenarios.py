#!/usr/bin/env python
"""Replay a shipped strategy's known-outcome scenarios — the write-gate oracle, offline.

This runs on the **core install with no API keys and no market data**. Every strategy file in
``strategies/`` ships 2-8 hand-built "scenario" tapes, each a short synthetic price path with a
*known* outcome (e.g. "goes long within these bars", "never trades"). This script replays them
exactly as ``write_strategy`` does in a subprocess before it lets a file land — so a broken or
mis-tuned strategy can never ship. It is the smallest possible demonstration of the
strategy-as-a-file contract and the seam architecture (the heavy engine/research/data stacks are
never imported).

See ``docs/validation.md`` for how this fits promotion, and ``strategies/README.md`` for the
full strategy-file contract.

Usage::

    uv run python examples/replay_scenarios.py                    # default: sma_crossover
    uv run python examples/replay_scenarios.py donchian_breakout  # any shipped strategy name(s)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from noctis.strategies.base import TraderStrategy
from noctis.strategies.scenarios import run_scenario

REPO = Path(__file__).resolve().parent.parent
STRATEGIES = REPO / "strategies"


def load_strategy(name: str) -> type[TraderStrategy]:
    """Load ``strategies/<name>.py`` and return its TraderStrategy subclass."""
    path = STRATEGIES / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"no such strategy file: {path}")
    spec = importlib.util.spec_from_file_location(f"example_strategy_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, TraderStrategy) and obj is not TraderStrategy:
            return obj
    raise SystemExit(f"{path} declares no TraderStrategy subclass")


def main(argv: list[str]) -> int:
    names = argv or ["sma_crossover"]
    failures = 0
    for name in names:
        cls = load_strategy(name)
        scenarios = cls.scenarios()
        print(f"\n{cls.name}: replaying {len(scenarios)} known-outcome scenario(s)")
        for scenario in scenarios:
            msg = run_scenario(cls, scenario)
            if msg is None:
                print(f"  PASS  {scenario.name}")
            else:
                print(f"  FAIL  {msg}")
                failures += 1
    print()
    if failures:
        print(f"{failures} scenario(s) failed")
        return 1
    print("all scenarios passed — the shipped strategies behave exactly as documented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
