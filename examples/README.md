# Examples

First-contact examples for Noctis. **Every example here runs on the bare core install with no
API keys and no market data** — that's itself a demonstration of the seam architecture (the
heavy engine/research/data stacks stay unimported).

Set up the core environment once:

```bash
uv sync            # core + dev, reproducible from uv.lock
```

## What's here

| File | What it shows |
|------|---------------|
| [`replay_scenarios.py`](replay_scenarios.py) | **Runnable.** Replays a shipped strategy's known-outcome `scenarios()` — the write-gate oracle — with no keys or data. |
| [`strategy_walkthrough.md`](strategy_walkthrough.md) | An annotated read of `strategies/sma_crossover.py`: header/thesis, `Params`, `on_bar`, `param_space`, and `scenarios`. |
| [`example_mandate.md`](example_mandate.md) | A minimal operator mandate you can adapt into `mandate/MANDATE.md`. |

## See the system move in 30 seconds

```bash
# Replay the shipped strategies' known-outcome scenarios (no keys, no data):
uv run python examples/replay_scenarios.py                 # sma_crossover (default)
uv run python examples/replay_scenarios.py donchian_breakout dual_trend_ema
```

Expected output ends with:

```
all scenarios passed — the shipped strategies behave exactly as documented
```

## Next steps

- **Replay a champion on real data.** Once you've built a data lake (see
  [`docs/data.md`](../docs/data.md)), `uv run python -m noctis backtest <name>` replays a library
  strategy on its shipped tuned defaults and prints its scorecard — exactly what earned it a
  slot ([`docs/validation.md`](../docs/validation.md)).
- **Run the loop.** `uv run python -m noctis status` shows the resolved mode and champions;
  `uv run python -m noctis run -v` starts the research/trading/close loop (paper-only).
- **Write your own strategy.** Start from [`strategies/TEMPLATE.py`](../strategies/TEMPLATE.py)
  and the contract in [`strategies/README.md`](../strategies/README.md).
