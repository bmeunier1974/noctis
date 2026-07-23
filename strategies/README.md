# The strategy library

One `.py` file per strategy. The file **is** the artifact: thesis, code, tuned parameters,
and research provenance all live in it, a human can review it like any code, and
`noctis backtest <name>` replays it on the same historical data the research loop used.
The seed files (`sma_crossover`, `rsi_meanrev`, `donchian_breakout`) are **worked examples**
that teach the contract — not a menu; the research agent authors new files below.

## Where files live (three tiers, one directory)

The library is discovered from three tiers, **lowest precedence first** — a later tier
overrides an earlier one for the same name. This directory holds only the committed input
tier; both output tiers live under the workspace:

| Tier | Path | Committed? | Holds |
|---|---|---|---|
| seeds | `strategies/*.py` | **yes** | `TEMPLATE.py` + the three worked examples — the only files in the public repo (read-only input) |
| working | `workspace/strategies/__tmp/` | no (workspace) | the agent's scratch area: drafts, candidates, and rejected files — **local only** |
| champions | `workspace/strategies/champions/` | no (workspace) | locally-promoted champions — never reach the public repo |

So a fresh clone ships only the template and three seeds; `__tmp/` and `champions/` are created at
runtime inside the gitignored `workspace/` and never leave your machine through git. A champion
always wins over a seed of the same name, and a live working copy in `__tmp/` shadows the seed it
derives from. Committed seeds are **never mutated in place**: a mechanical rewrite of a seed (a
status stamp, tuned defaults) is redirected into `__tmp/`, so the public seed files stay pristine.
Corollary: deleting a shadowing copy (`rm workspace/strategies/__tmp/<name>.py`) silently
resurrects the pristine seed underneath it on the next load — that is the shadowing model working
as designed, not a lost verdict (the rejection stays recorded in the journal and memory).

## The contract

Each file defines exactly one `TraderStrategy` subclass (`src/noctis/strategies/base.py`):

| Member | Requirement |
|---|---|
| `name` | Must equal the file name (`rsi_meanrev.py` → `name = "rsi_meanrev"`). |
| `Params` | A `@dataclass(frozen=True)` of tunables; `params_cls = Params`. Defaults are what ships. |
| `on_start(ctx)` | Reset all incremental state (instances are replayed many times). |
| `on_bar(ctx, bar)` | React to one bar; call `ctx.set_target(1)` (long), `ctx.set_target(-1)` (short), or `ctx.set_target(0)` (flat). `bar.ts_event` is the UTC-ns timestamp if the thesis needs a session/clock gate. |
| `param_space()` | `list[ParamSpec]` — the domain `run_sweep`/Optuna explores. |
| `scenarios()` | 2–8 known-outcome `Scenario`s (see below) — the file's own correctness oracle. |
| `warmup_bars(params)` | Optional `int` — decision bars before which the strategy promises to stay flat (derive it from your own lookback; multiply for higher-timeframe filters). Default `0` = undeclared = exempt. |

`signals()` (the vectorised pre-filter path) is **optional**: the base class default replays
`on_bar` over the frame, so both code paths agree by construction. Override it only as a
performance optimization — the write gate rejects an override that disagrees with the replay.

Rules the validation gate enforces on `write_strategy`:
- imports cleanly in a fresh interpreter and runs a smoke replay on a synthetic fixture;
- exactly one strategy class, `name` matches the file, a docstring header exists;
- long/short/flat targets (+1/−1/0), no exceptions, `signals()`/`on_bar` parity;
- the declared known-outcome scenarios replay clean (≥ 1 tape demanding a directional
  (long or short) entry, ≥ 1 `always_flat()` no-trade tape);
- **warmup honesty** — if the file declares `warmup_bars(params)`, no scenario tape may take
  a nonzero position before that bar; a declaration the code's own tapes contradict is rejected,
  naming the offending bar and the declared warmup.

Keep `on_bar` O(lookback): bounded `deque`s + the `ind` tail helpers. No I/O, no globals,
no randomness — a strategy must be a pure function of the bars it has seen.

## The docstring header

```python
"""One-sentence thesis first — the falsifiable claim about market behaviour.

Optional elaboration paragraph.

status: candidate            # draft | candidate | champion | rejected
style: mean-reversion        # momentum | mean-reversion | breakout | ...
symbols: AAPL MSFT NVDA      # the panel it was researched/tuned on
tuned: 2026-07-04            # date the current Params defaults were fitted
"""
```

`status`, `symbols`, and `tuned` are stamped by the research loop: on champion promotion the
winning parameters are written back as the `Params` defaults and the header is updated, so
diffing the file shows exactly what shipped. `reject_strategy` stamps `status: rejected`.

## Known-outcome scenarios (`noctis.strategies.scenarios`)

The scenarios are the file's own oracle: deterministic synthetic tapes plus the behavior
the **thesis** demands on them, declared *before* the code is trusted. The write gate
replays every tape through `on_bar` and rejects the file when the code disagrees with its
own declaration — inverted conditions, dead logic that never trades, and warmup mistakes
all fail here instead of burning backtest budget.

```python
from noctis.strategies import scenarios as sc

@classmethod
def scenarios(cls) -> list[sc.Scenario]:
    warm = cls.params_cls().lookback   # derive windows from the CURRENT defaults
    return [
        sc.Scenario(
            "capitulation_then_recovery",
            segments=[sc.flat(warm + 5), sc.selloff(10, 0.06), sc.recovery(40, 0.11)],
            expect=[sc.flat_until(warm), sc.long_within(warm + 5, warm + 16),
                    sc.flat_by(2 * warm + 20)],
        ),
        sc.Scenario(
            "steady_grind_stays_flat",
            segments=[sc.flat(warm + 5), sc.recovery(60, 0.15)],
            expect=[sc.always_flat()],       # the mandatory no-trade tape
        ),
    ]
```

Segment builders (deterministic, no randomness; legs concatenate continuously):
`flat(n)`, `trend(n, pct)` (signed geometric drift), `selloff(n, pct)`, `recovery(n, pct)`,
`chop(n, amplitude, period=8)`, `vol_spike(n, amplitude=0.05)`, `gap(pct)` (bar-less jump).

Expectations (behavioral windows over the replayed target series, not per-bar values):
`flat_until(i)`, `long_within(lo, hi)`, `holds_long_through(lo, hi)`, `short_within(lo, hi)`,
`holds_short_through(lo, hi)`, `flat_by(i)`, `always_flat()`.

Bounds: 2–8 scenarios, 60–2 000 bars each, ≥ 1 directional-entry expectation (long or
short) and ≥ 1 `always_flat()` tape across the set. Replays run with the current `Params` defaults
(override per scenario via `Scenario(..., params={...})`), so **derive windows from
`cls.params_cls()` defaults** — champion promotion rewrites the defaults, re-runs the gate,
and `evaluate_vs_champion` refuses tuned params that break the declared scenarios.

## Indicator helpers (`noctis.strategies.indicators`)

Tail functions over a list/deque you accumulate (return `None` during warmup and on
degenerate windows — flat range, zero deviation, zero base — so always guard):
`sma(values, p)`, `ema(values, p)`, `rsi(values, p)` (Cutler), `atr(highs, lows, closes, p)`,
`stdev(values, p)` (population σ), `zscore(values, p)`, `bollinger(values, p, mult)` →
`(upper, mid, lower)`, `roc(values, p)` (percent), `wma(values, p)`, `highest(values, p)`,
`lowest(values, p)`, `stoch_k(highs, lows, closes, p)` (raw %K — smooth to %D with `sma`),
`cci(highs, lows, closes, p)`, `bars_since(flags)`, and the crossing predicates
`cross_above(f0, f1, s0, s1)` / `cross_below(f0, f1, s0, s1)`. Each docstring names the
Pine/Wilder variant it matches.

Stateful, golden-tested classes for exact seeded/Wilder/cumulative math: `SmaState`,
`EmaState`, `RsiState`, `AtrState`, `MacdState`, `VwapState`, `AdxState` (with `.plus_di` /
`.minus_di`), `ObvState`, `StochState` (returns `{"k", "d"}`), `SupertrendState` (returns
`{"st", "dir"}`), `ZScoreState`, `RollingExtremeState` — build in `on_start`,
`.update(bar)` per bar (`nan` during warmup). Vector twins for `signals()` overrides:
`adx_vector`, `obv_vector`, `stoch_vector`, `supertrend_vector`.

Deliberately not provided (documented in the module docstring): Ichimoku,
`valuewhen`/pivot-style bar-indexing built-ins, volume-profile indicators.

## Session-clock helpers (`noctis.strategies.session`)

Pure functions of `bar.ts_event` (the UTC-ns int the bar already carries — no `Bar`
schema change) that answer "where am I in the trading day?" for US-equity regular hours
(09:30–16:00 ET), DST-correct via stdlib `zoneinfo`:
`minute_of_session(ts)` (0 at 09:30, 389 at 15:59, `None` outside RTH), `is_rth(ts)`,
`minutes_to_close(ts)`, `session_date(ts)` (the ET trading date), and `new_session(prev_ts,
ts)` (session-boundary edge, for daily resets). The "no entries in the last 15 minutes"
two-liner:

```python
m = session.minutes_to_close(bar.ts_event)
if m is not None and m <= 15:
    ctx.set_target(0); return
```

Limitations (documented, not solved): **no holiday calendar** (a holiday session just has
no bars, so awareness buys nothing in `on_bar`) and **half-day** 1:00 pm ET closes make
`minutes_to_close` overstate on those afternoons.

## Multi-timeframe (`ind.HtfBars`)

Consult a higher timeframe (a 5m strategy reading a 1h trend filter) lookahead-free by
construction. Own an `ind.HtfBars(timeframe)` like any other state: build it in `on_start`,
feed every base bar in `on_bar`, and act on the completed higher-timeframe bar it hands back:

```python
def on_start(self, ctx):
    self.htf = ind.HtfBars("1h")          # must be a MULTIPLE of this strategy's timeframe
    self.htf_ema = ind.EmaState(self.params.trend_period)
    self.trend = float("nan")

def on_bar(self, ctx, bar):
    done = self.htf.add(bar)              # completed 1h bar or None
    if done is not None:
        self.trend = self.htf_ema.update(done.close)
    # ... use self.trend as a gate on your base-timeframe entries ...
```

`on_bar` still receives only your declared base-timeframe bars — no driver or `Context`
change. A bucket is handed back only once the *next* bucket opens, so a higher-timeframe
bar is never visible before it closes. For a vectorised `signals()` override, use
`last_completed_htf(frame, timeframe)` (in `noctis.data.aggregate`) — it gives the same
completed-bucket view per base row.

Two things to know:

- **Warmup multiplies.** A 1h `EmaState(20)` over a 5m strategy needs ~20 completed hours
  ≈ weeks of 5m bars before it says anything. Write your `scenarios()` tapes long enough to
  warm the HTF indicator up, or they will just stay flat.
- **The session-final partial bucket is never emitted** (on purpose — the backtest never
  scored partial buckets either). An HTF filter therefore last updates before the final
  partial hour of the day; backtest and live score that identically.

The higher timeframe must be a multiple of your declared `timeframe` for the bucketing to
be meaningful (a convention the wrapper can't enforce — it can't see your class); `"1m"`
and unsupported timeframes are rejected at construction.

## Protective exits (`ExitRules`)

Attach engine-enforced protection — a fixed stop-loss, a take-profit, a trailing stop — by
declaring rules alongside your target. `ExitRules` imports from `noctis.strategies.base`
beside `Bar`:

```python
from noctis.strategies.base import Bar, Context, ExitRules, ParamSpec, TraderStrategy

def on_bar(self, ctx: Context, bar: Bar) -> None:
    # ... decide target as usual ...
    ctx.set_target(
        target,
        exits=ExitRules(
            stop_pct=self.params.stop_pct,  # exit if adverse move ≥ this fraction of entry
            take_profit_pct=self.params.take_profit_pct,
            trail_pct=self.params.trail_pct,  # exit if give-back from best-since-entry ≥ this
        ),
    )
```

`exits` is keyword-only and defaults to `None` — every existing file is source-compatible,
and a call without it declares no protection. **Re-declare the rules on every `set_target`
call** (stateless from your side): the engine associates them with the open position and
evaluates them intrabar against each completed bar of your declared timeframe, under the
conservative policy locked in the fill-model section of `docs/architecture.md` (a
gap-through fills at the open, never the untouched level; a stop and take-profit both
touched in one bar assume the stop fired; the trail ratchets on the *prior* bar's extreme).

What authors must know:

- **Percentages, not prices.** Each field is a fraction of the entry price (`0.05` = 5%),
  so one tuned param set stays meaningful across a whole panel of symbols. Absolute price
  levels are deliberately not expressible.
- **You will never observe your own stop-outs.** A strategy stays a pure function of the
  bars it has seen: `on_bar` receives no fill, no position, no signal that an exit fired.
  Do not write logic that expects to react to one.
- **The re-arm latch: after an exit fires, the engine holds the symbol flat until your
  target series *changes value* (any transition — `+1→0`, `+1→−1`, …). The first change
  un-latches; the new value then executes normally.** Consequence: a strategy that holds
  `+1` for weeks treats a stop-out as terminal until its signal cycles. That is correct —
  it is what "the thesis was invalidated" means — but you must know it.
- **The scenario oracle is unchanged.** Scenarios verify your *decisions* (the target
  series); exits are engine behavior, tested by the engine's own suites. There is no
  `stopped_out_within(...)` expectation, and none is needed.
- **Exit percentages are ordinary `float` params.** Forward them from `Params` into
  `ExitRules` and declare them in `param_space()` like any other knob — no framework
  change. The promotion gates punish stop-fishing exactly the way they punish any other
  overfit knob.

## Lifecycle

FORMULATE (author the file into `__tmp/`, `write_strategy`) → MATCH (pick symbols that fit the
thesis, `ensure_data`) → OPTIMIZE (`run_backtest` / `run_sweep`; every trial journals to
`workspace/state/experiments/<name>.jsonl`) → DECIDE (`evaluate_vs_champion` or `reject_strategy` —
both refuse until the parameter space has actually been explored).

On promotion the file is **moved** out of `__tmp/` into `champions/`, the tuned params are written
back as the `Params` defaults, and the header is stamped `status: champion`. On rejection the file
stays in `__tmp/` stamped `status: rejected` (kept for local inspection, out of the repo). A
champion file is immutable — to improve one, author a new name; `write_strategy` refuses to
overwrite the crown.

A draft that reaches neither verdict has a third exit: **archive**. A prune-on-start sweep — run
at the top of each research session — **moves** every still-undecided (`draft`/`candidate`)
top-level `__tmp/` file whose age exceeds the TTL (`research.draft_ttl_hours`, default 48h;
`null`/`0` disables) into an `__tmp/archive/` subdirectory. Archiving is housekeeping, not
judgment: the bytes are **moved verbatim — never deleted, never re-stamped** (no `status:
rejected`, no gate, no verdict), so a session never inherits a stale draft it abandoned days ago.
The archive is capped (50 files, oldest sequence evicted first) with the same collision-safe
`{seq}-{name}.py` naming the coder-failure store uses, and the experiment journals under
`workspace/state/experiments/` stay the ground truth for what was tried — archiving a file never
touches them.

Start a new strategy from `TEMPLATE.py` (the loader skips it).
