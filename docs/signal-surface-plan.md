# Signal-surface plan: indicator breadth, session context, multi-timeframe

Closing the useful part of the gap between the Noctis strategy-author surface and a
Pine-Script-class `ta.*` library — **without** touching the order model, the promotion
gates, or the no-lookahead fill contract. Three phases, independent and shippable in
order of increasing risk. Each phase is purely **additive** to the author surface:
no existing strategy, seed, or champion changes behavior.

**Companion plan:** `docs/protective-exits-plan.md` covers the order-type gap
(stop/limit/trailing). It is deliberately separate because it changes the fill
contract; this plan deliberately does not.

---

## Where we are (inventory, 2026-07-15)

- Author-facing indicators: `src/noctis/strategies/indicators.py` — tail functions
  `sma / ema / rsi / atr / highest / lowest / cross_above`, plus re-exported stateful
  classes `SmaState / EmaState / RsiState / AtrState / MacdState / VwapState` from
  `src/noctis/strategies/spec/indicators.py`. `ZScoreState` and `RollingExtremeState`
  exist in the spec module but are **not** re-exported.
- Every stateful primitive has a vectorised twin (`*_vector`) golden-tested against it:
  `tests/test_strategies.py:187` (golden fixture `tests/fixtures/indicator_golden.json`)
  and `tests/test_strategies.py:235` (vector/state parity).
- Per-bar data: `Bar(ts_event, open, high, low, close, volume)` only
  (`src/noctis/strategies/base.py:27`). No session/calendar object, no second timeframe,
  no cross-symbol access. History is self-managed via `deque`s built in `on_start`.
- One declared `timeframe` per strategy (`base.py:68`); the engine aggregates the 1m lake
  on the way in via `aggregate_bars` / `StreamingAggregator`
  (`src/noctis/data/aggregate.py:60,85`).

## Invariants this plan must preserve

1. **No lookahead** — every new helper must be computable from bars ≤ *t* only. The MTF
   phase leans on the already-proven `StreamingAggregator` semantics ("a bucket exists
   only once fully observed").
2. **Core-install clean** — nothing here may add a required dependency. All helpers are
   stdlib + the existing in-house code (`zoneinfo` is stdlib on py311).
   `exchange-calendars` stays behind its seam and out of the strategy path.
3. **Parity by construction** — new stateful primitives get vectorised twins and golden
   tests exactly like the existing six; `signals()` overrides remain parity-gated by the
   write gate (`library.py` validation), which needs no changes.
4. **Seeds are read-only input** — the three shipped examples are not rewritten to use
   new helpers. New capability is documented in `TEMPLATE.py` + `strategies/README.md`,
   which is how the research agent discovers it.

---

## Phase 1 — Indicator breadth (the `ta.*` gap)

**Goal:** take the shipped library from ~12 primitives to the ~25 that cover what
agent-authored strategies actually reach for, so the agent stops hand-rolling (and
mis-rolling) common indicators inside `on_bar`.

### 1a. New tail functions in `src/noctis/strategies/indicators.py`

Same contract as the existing ones: pure, O(period) per call over a `Sequence[float]`
the strategy accumulates, `None` during warmup, `int(period)` coercion, no numpy.

| Function | Signature sketch | Notes |
|---|---|---|
| `cross_below` | `(fast_prev, fast_now, slow_prev, slow_now) -> bool` | mirror of `cross_above:134`; kills the #1 hand-roll |
| `stdev` | `(values, period) -> float \| None` | population σ over the tail window |
| `zscore` | `(values, period) -> float \| None` | `(last − sma) / stdev`; `None` when σ == 0 |
| `bollinger` | `(values, period, mult) -> tuple[float, float, float] \| None` | `(upper, mid, lower)`; built on `sma` + `stdev` |
| `roc` | `(values, period) -> float \| None` | percent rate-of-change; also serves as `ta.change` when `period=1` |
| `wma` | `(values, period) -> float \| None` | linear-weighted MA |
| `stoch_k` | `(highs, lows, closes, period) -> float \| None` | raw %K; authors smooth to %D with `sma` |
| `cci` | `(highs, lows, closes, period) -> float \| None` | typical-price CCI, 0.015 constant |
| `bars_since` | `(flags: Sequence[bool]) -> int \| None` | Pine `ta.barssince` over a self-kept flag deque; `None` if never true |

Each gets a docstring in the existing style (what it is, warmup rule, which Pine/Wilder
variant it matches) and an `__all__` entry.

### 1b. New stateful + vectorised pairs in `src/noctis/strategies/spec/indicators.py`

Only where exact/Wilder/full-history or path-dependent math genuinely needs state —
following the existing `EmaState`/`ema_vector` pattern (state class with
`.update(...) -> float` returning `nan` in warmup, vector twin, golden agreement):

- `AdxState` / `adx_vector` — Wilder ADX with `+DI`/`−DI` accessible
  (`.update(bar) -> adx`, properties `.plus_di` / `.minus_di`). The one indicator
  authors most often get numerically wrong.
- `ObvState` / `obv_vector` — cumulative, so a tail function is impossible.
- `StochState` / `stoch_vector` — smoothed %K/%D (two SMA stages over raw %K).
- `SupertrendState` / `supertrend_vector` — ATR-band trend flip; path-dependent, popular
  with trend styles. **Stretch — cut first if the phase runs long.**
- Re-export the already-existing `ZScoreState` and `RollingExtremeState` through
  `strategies/indicators.py` (one-line change each; they are already golden-tested).

**Deliberately not added** (document the refusal in the module docstring so the agent
doesn't ask): Ichimoku (five lines of config-heavy convention, rarely a thesis),
`valuewhen`/`pivothigh`-style bar-indexing built-ins (Python authors keep their own
state more clearly), volume-profile anything (needs intrabar data the lake doesn't have).

**Spec-engine REGISTRY** (`spec/indicators.py:499`): do **not** register the new
primitives in the legacy spec-graph language in this phase. The registry is the spec
DSL's surface, and widening a legacy DSL is a separate decision. Leave a code comment.

### 1c. Tests and docs

- Extend `tests/fixtures/indicator_golden.json` + `test_indicator_golden_fixture`
  (`tests/test_strategies.py:187`) with hand-checked values for every new primitive
  (source the expected numbers from a reference implementation offline, paste as
  fixture data — same as the existing six).
- Add each new state/vector pair to the `test_indicator_vector_state_parity`
  parametrisation (`tests/test_strategies.py:235`).
- Tail functions: direct unit tests (warmup boundary, `period=1`, constant series,
  σ == 0 for `zscore`/`bollinger`).
- Update the `strategies/indicators.py` module docstring (it is the author's API doc),
  `strategies/README.md` §helpers, and the "helpers you have" comment block in
  `strategies/TEMPLATE.py`. These three files are what the research agent reads —
  updating them **is** the agent rollout; no prompt/tooling change needed.

**Acceptance:** all new primitives golden-tested; `pytest`, `ruff`, `mypy`,
`pre-commit run --all-files` clean; a scratch strategy using `bollinger` + `cross_below`
passes the `write_strategy` gate end-to-end (drive via the `/verify` harness).

**Size:** ~2 sessions. No engine files touched.

---

## Phase 2 — Session/calendar context

**Goal:** let an intraday strategy express "no entries in the last 15 minutes" or
"reset at the session open" in one or two lines, from `bar.ts_event` alone.

### Design: a pure helper module, not a Bar schema change

New module `src/noctis/strategies/session.py`, re-exported alongside `indicators`.
Pure functions over the UTC-ns `int` timestamp using stdlib `zoneinfo`
(`America/New_York`), defaulting to US-equity RTH (09:30–16:00 ET) to match the
assumption already baked into `aggregate.py:26` ("RTH never crosses UTC midnight"):

```python
minute_of_session(ts_event) -> int | None   # 0 at 09:30, 389 at 15:59; None outside RTH
is_rth(ts_event) -> bool
minutes_to_close(ts_event) -> int | None
session_date(ts_event) -> datetime.date     # the ET trading date
new_session(prev_ts, ts) -> bool            # session-boundary edge, for daily resets
```

Why **not** stamping fields onto `Bar`: `Bar` is constructed in at least four places
(simulator, `replay_targets`, `StreamingAggregator`, live feed) and flows through the
scenario builders; widening it ripples everywhere for data that is a pure function of a
field it already carries. A helper keeps strategies pure and the schema frozen.

### Honest limitations (document, don't solve)

- **No holiday calendar.** A holiday session simply has no bars, so holiday-awareness
  buys nothing inside `on_bar`. Say so in the docstring.
- **Half-days** (1:00 pm ET closes): `minutes_to_close` will overstate. Documented
  limitation; strategies that care can watch for the bar stream ending.
- DST is handled correctly by `zoneinfo` — this is the reason the module exists instead
  of a `ts % 86400` hack; test it explicitly.

### Tests and docs

- Unit tests with fixed timestamps spanning a DST spring-forward week and a fall-back
  week (both directions), plus a weekend and the open/close boundary minutes.
- One integration check: `new_session` agrees with `VwapState`'s UTC-day bucketing on
  regular sessions (`spec/indicators.py:376` buckets by UTC day, which equals the ET
  session for RTH — assert that equivalence holds on the test dates).
- Same doc trio as Phase 1: module docstring, `strategies/README.md`, `TEMPLATE.py`.

**Acceptance:** the two-liner works —

```python
m = session.minutes_to_close(bar.ts_event)
if m is not None and m <= 15:
    ctx.set_target(0); return
```

**Size:** ~1 session. Purely additive; no engine files touched.

---

## Phase 3 — Multi-timeframe (HTF) access

**Goal:** a 5m strategy that consults a 1h trend filter — Pine's
`request.security(..., "60", ...)` use-case — lookahead-free by construction.

### Design: author-owned aggregator, not a framework hook

The machinery already exists: `StreamingAggregator` (`src/noctis/data/aggregate.py:85`)
buffers fine bars and emits the completed bucket only when the first bar of the *next*
bucket arrives — exactly the lookahead-free semantics required, already golden-matched
to the vectorised `aggregate_bars`. The plan is to hand it to authors rather than build
a new delivery channel:

1. **Author-facing wrapper `ind.HtfBars(timeframe)`** in `strategies/indicators.py`,
   wrapping `StreamingAggregator`. Owned by the strategy like any other state:

   ```python
   def on_start(self, ctx):
       self.htf = ind.HtfBars("1h")
       self.htf_ema = ind.EmaState(self.params.trend_period)
       self.trend = float("nan")

   def on_bar(self, ctx, bar):
       done = self.htf.add(bar)          # completed 1h bar or None
       if done is not None:
           self.trend = self.htf_ema.update(done.close)
       ...
   ```

   No driver, simulator, or `Context` change — `on_bar` still receives only its declared
   base-timeframe bars, so the walk-forward splitter, the write-gate smoke replay, and
   the live loop all work unchanged.

2. **Alignment guard.** `HtfBars.__init__` validates the HTF is a supported timeframe
   (`validate_timeframe`) and raises if it is not strictly coarser than could make
   sense — concretely: reject `"1m"` (the pass-through special case in
   `StreamingAggregator.add:102` would defeat the wrapper) and document that the HTF
   must be a multiple of the strategy's declared `timeframe` (bucketing by
   `ts // bucket_ns` is only meaningful then). The wrapper cannot see the declaring
   class's `timeframe`, so the multiple-of check is documented convention + a
   `write_strategy`-level lint if it proves to be a real footgun (defer).

3. **Vectorised recipe for `signals()` overrides:** a helper
   `last_completed_htf(data: pd.DataFrame, timeframe: str) -> pd.DataFrame` (lives in
   `noctis/data/aggregate.py` beside its siblings) that returns, per base row, the
   OHLCV of the **last fully completed** HTF bucket — i.e. rows are matched on
   `bucket(base_ts) > bucket(htf_row)`, never `>=`, so the in-progress bucket (which
   `aggregate_bars` includes as a trailing partial) is never visible. Import direction
   is safe: `strategies/indicators.py → data/aggregate.py → strategies/base.py` has no
   cycle (checked — `base.py` imports nothing from either).

4. **The parity test that makes this trustworthy** (the key deliverable): replay a
   multi-week 1m fixture through `HtfBars` bar-by-bar and assert the sequence of
   completed HTF bars — *and the base-bar index at which each becomes visible* — is
   identical to what `last_completed_htf` reports. This extends the existing
   streaming-vs-vector aggregation tests in `tests/test_aggregate.py`.

### Interactions to handle explicitly

- **Warmup budget.** A 1h EMA(20) over a 5m strategy needs ≥ ~2 weeks of bars before it
  says anything. Document in `strategies/README.md` that HTF lookback multiplies the
  warmup, and check the write-gate smoke fixture (`fixture_frame`,
  `src/noctis/strategies/library.py:112`) is long enough that an HTF strategy at least
  *runs* through it (it can legally stay flat during warmup — scenarios are the
  correctness oracle, and the author writes those against tapes long enough to warm up;
  the `scenarios.py` segment builders already support arbitrary lengths).
- **Session-final partial buckets are never emitted** (aggregator behavior, on purpose —
  `aggregate.py:91`). An HTF trend filter therefore updates one last time before the
  final partial hour of the day; that is the same behavior the backtest scores. No
  change; document it.
- **Explicit non-goal: cross-symbol access in `on_bar`.** The evaluation layer scores
  panels per-symbol (`src/noctis/backtest/pipeline.py`); a strategy that reads other
  symbols needs a different simulator, different holdout semantics, and a different
  promotion story. Out of scope for this plan — refuse in-scope-creep, propose a
  separate design if it becomes a real ask.

### Tests and docs

- The streaming-vs-vector visibility parity test above (the load-bearing one).
- A worked micro-example in `TEMPLATE.py`'s comment block + a `strategies/README.md`
  §multi-timeframe section (worked snippet, warmup warning, no-partial-bucket note).
- End-to-end: author a scratch 5m/1h strategy through `write_strategy` via `/verify`,
  confirm the gate passes and a pipeline run produces a sane scorecard.

**Acceptance:** the parity test proves an HTF bar is never visible before its bucket
closes, on both code paths; scratch strategy passes the gate end-to-end.

**Size:** ~2 sessions. Touches `strategies/indicators.py`, `data/aggregate.py` (one new
pure helper), tests, docs. No driver/simulator/broker changes.

---

## Rollout order and checkpoints

Ship phases as three separate PRs, in order (1 → 2 → 3): each is independently useful,
and 3 builds on nothing from 1–2 but is the riskiest to review. After each phase:
`pytest` + `ruff check . && ruff format --check .` + `mypy` + `pre-commit run
--all-files`, plus one `/verify`-harness drive of the write gate with a strategy that
uses the new surface. No config knobs, no migrations, no champion-registry impact —
nothing here rescore or displaces an existing champion.
