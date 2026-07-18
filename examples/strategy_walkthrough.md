# Walkthrough: reading a strategy file

A Noctis strategy is **one file, and the file is the whole artifact** — thesis, code, tuned
parameters, and its own correctness oracle all live together. This walks through
[`strategies/sma_crossover.py`](../strategies/sma_crossover.py), the simplest shipped example.
The full contract is in [`strategies/README.md`](../strategies/README.md).

## 1. The header is the thesis and the provenance

```python
"""Ride the medium-term trend: hold long while the fast moving average is above the slow.

The classic trend filter: a fast SMA above a slow SMA says recent demand outpaces the
longer-run average, so stay long; below, stay flat. No shorting — the edge claimed is
trend persistence, not symmetry.

status: rejected
style: momentum
"""
```

The docstring states the **falsifiable claim** in plain words, then carries machine-readable
provenance: `status` (`draft` / `candidate` / `champion` / `rejected`), `style`, and — once a
strategy is promoted — the `symbols` it was fit on and the `tuned` date. Promotion writes those
fields back automatically (see [`docs/validation.md`](../docs/validation.md)). This file is
`rejected`: it's kept as a teaching example, not a live champion.

## 2. `Params` — a frozen, tunable parameter set

```python
@dataclass(frozen=True)
class Params:
    fast: int = 10
    slow: int = 30

params_cls = Params
```

Every knob the strategy has is a field here, with a default. On promotion, the *winning* values
are written back as these defaults, so the file on disk always replays exactly what shipped.

## 3. `on_bar` — the decision, one bar at a time

```python
def on_bar(self, ctx: Context, bar: Bar) -> None:
    self._closes.append(bar.close)
    fast = ind.sma(self._closes, self.params.fast)
    slow = ind.sma(self._closes, self.params.slow)
    if fast is None or slow is None:      # still warming up
        ctx.set_target(0)
        return
    ctx.set_target(1 if fast > slow else 0)
```

`on_bar` sets a **target**: `1` = long, `-1` = short, `0` = flat. It must be O(lookback) and do
no I/O, globals, or randomness — those constraints are what make a backtest reproducible. The
engine executes a decision made on bar *t* at bar *t+1*'s open (no lookahead).

## 4. `param_space` — what the optimizer may search

```python
@classmethod
def param_space(cls) -> list[ParamSpec]:
    return [
        ParamSpec("fast", "int", low=3, high=30, step=1),
        ParamSpec("slow", "int", low=20, high=100, step=1),
    ]
```

## 5. `scenarios` — the file's own correctness oracle

```python
@classmethod
def scenarios(cls) -> list[sc.Scenario]:
    warm = cls.params_cls().slow
    return [
        sc.Scenario(
            "trend_ride_then_rollover",
            segments=[sc.flat(warm + 5), sc.trend(40, 0.12), sc.selloff(40, 0.20)],
            expect=[sc.flat_until(warm), sc.long_within(warm + 5, warm + 45), sc.flat_by(warm + 80)],
        ),
        sc.Scenario(
            "steady_decline_never_longs",
            segments=[sc.flat(warm + 5), sc.selloff(60, 0.25)],
            expect=[sc.always_flat()],
        ),
    ]
```

Each scenario is a short **synthetic** price tape with a *known* outcome, built from a small DSL
(`flat` / `trend` / `selloff` / `recovery` / `chop` / `vol_spike` / `gap`) and asserted with
expectations (`long_within` / `holds_long_through` / `short_within` / `flat_by` / `always_flat`,
…). At least one scenario must demand a directional entry and at least one must be a no-trade
tape. `write_strategy` replays these in a fresh subprocess before a file can land, so a broken or
mis-tuned strategy never ships.

Run them yourself — no keys, no market data:

```bash
uv run python examples/replay_scenarios.py sma_crossover
```
