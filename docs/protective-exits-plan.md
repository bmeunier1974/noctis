# Protective-exits plan: stop-loss / take-profit / trailing stops

The order-type gap from the Pine comparison — `strategy.exit`-class functionality —
treated as what it is: **a change to the fill contract**, the most invariant-laden seam
in the system. This plan is deliberately conservative in scope and heavy on design
decisions locked *before* code, because the current contract ("decide on bar *t*, fill
at bar *t+1*'s open, nothing else can create a fill") is what makes the no-lookahead
guarantee checkable today (`src/noctis/broker/simulator.py:1-7`).

**Companion plan:** `docs/signal-surface-plan.md` (indicators/session/MTF) is purely
additive and independent; land it first — it needs none of this.

---

## Scope decision (the most important line in this plan)

**In scope: protective exits attached to an open position** — fixed stop-loss,
take-profit, and trailing stop, expressed as *percentages*, declared by the strategy
when it sets a target, evaluated by the **engine** intrabar against subsequent OHLC.

**Out of scope, explicitly refused:**

- **Limit/stop *entries*** and general resting-order management. They change what
  "activity" means, interact with the activity floor and turnover metrics in
  `src/noctis/champions/promotion.py`, and give the research agent a whole new axis to
  overfit (entry-price fishing). Different plan, different risk review, if ever.
- **Strategy-visible fills/position state.** A strategy stays a pure function of the
  bars it has seen (`strategies/README.md` contract). Exits are declarative rules the
  engine enforces; the strategy never learns whether one fired. This keeps
  `signals()`/`on_bar` parity meaningful (both still emit the same *target* series) and
  keeps the write gate's replay semantics intact.
- **Sub-timeframe stop evaluation in live.** Exits evaluate on the strategy's declared
  timeframe bars — identical to what the backtest scores. Evaluating live stops on 1m
  bars under a 1h strategy would make live systematically different from the numbers
  that promoted the champion. Finer-granularity stops are future work with their own
  backtest story.

Why percentages, not absolute prices: panel research tunes one param set across many
symbols (`src/noctis/backtest/pipeline.py`); a `stop_pct` is scale-free and
Optuna-searchable as an ordinary `ParamSpec("float")`, an absolute price is neither.

---

## Phase 0 — Contract design (a doc + decisions, no code)

Deliverable: a `docs/architecture.md` §fill-model subsection (or ADR) that locks the
following before any implementation. Each decision below is the *proposed* resolution;
Phase 0 exists so disagreement happens on paper, not in a half-landed PR.

### 0a. Author API

`Context.set_target` grows one keyword-only, defaulted parameter — source-compatible
with every existing strategy file:

```python
@dataclass(frozen=True)
class ExitRules:                      # noctis/strategies/base.py, beside Bar
    stop_pct: float | None = None     # exit if adverse move ≥ this fraction of entry
    take_profit_pct: float | None = None
    trail_pct: float | None = None    # exit if drawdown from best-since-entry ≥ this

def set_target(self, target: int, exits: ExitRules | None = None) -> None: ...
```

`TargetContext` (`base.py:124`) captures `self.exits` alongside `self.target`. Rules
are re-declared every bar with the target (stateless from the strategy's side); the
engine associates them with the position, not the bar.

### 0b. Execution semantics (the invariant part)

Per bar *t+1*, in this order — chosen so that no step can see later steps' information:

1. **Open:** execute the pending target from bar *t* at the open, exactly as today
   (`simulator.py:72-74`). If the target opens/flips a position, exit tracking
   (re)anchors: `entry_price = fill price`, `best = entry_price`.
2. **Intrabar:** if a position is open and rules are armed, evaluate against *t+1*'s
   high/low with the **conservative policy**:
   - Gap-through: if the *open* is already beyond a level, the exit fills **at the
     open** (never at the untouched level).
   - Stop and take-profit both touched within the same bar: **assume the stop fired**
     (worst case — intrabar path is unknowable from OHLC).
   - Trailing: the high-water mark ratchets on the **prior** bar's extreme; the trigger
     evaluates against the **current** bar's adverse extreme. Ratcheting and triggering
     off the same bar would be intra-bar lookahead (the high that sets the mark may
     occur after the low that hits it).
   - Exit fills route through the normal slippage/fee models
     (`src/noctis/broker/seam.py:53-71`), adverse to the closing side.
3. **Close:** strategy `on_bar` runs (sees the full bar, as today), sets next target;
   equity marks at the close (`simulator.py:81-83`).

### 0c. Re-arm semantics (the hardest decision — decide here, test everywhere)

After an exit fires, the strategy — position-blind by design — may keep emitting the
same nonzero target. If the engine simply honored it, every stop-out would re-enter at
the next open and the stop would be a one-bar speed bump.

**Rule: after an exit fires, the engine latches that symbol flat until the strategy's
target series *changes value* (any transition: `+1→0`, `+1→−1`, …). The first change
un-latches; the new value then executes normally.** Rationale: a target *change* is the
strategy affirmatively re-deciding; a *held* target is stale conviction from before the
stop-out. This mirrors how the existing halt latch already treats "strategy keeps
saying long" during a halted session (`engine` risk path), so operators reason about
one latch shape, not two.

Consequence to document loudly: for a strategy that holds `+1` for weeks, a stop-out is
effectively terminal until its signal cycles. That is *correct* — it is what "the
thesis was invalidated" means — but authors must know it, and `strategies/README.md`
must say it.

### 0d. Honesty about the prefilter

With engine-side exits, realized P&L is **no longer a pure function of the target
series** — the vectorised prefilter (signals → returns) cannot see exit fills. Decision:
**the prefilter stays exit-blind and keeps its existing role as a coarse selection
filter; the event-driven walk-forward (which is authoritative for the Scorecard and
every promotion gate) prices exits exactly.** This matches the system's existing
stance — the prefilter filters, validation arbitrates. Document it in
`docs/validation.md` and in `pipeline.py`'s docstring; do **not** try to approximate
stops vectorially (that is how lookahead bugs are born).

Gate interaction check (rule 2 of AGENTS.md): exits change *candidate behavior*, not
gate thresholds. The activity floor, gap guard, holdouts, and consistency gates apply
to exit-bearing candidates unchanged. No gate is loosened, no new gate is needed.

### 0e. Comparability and rollout posture

- Existing champions were scored under the exit-free fill model. Because exits are
  opt-in per strategy (no rules ⇒ code path identical), old scorecards remain valid —
  **no staleness rule needed** (unlike the cross-metric case in
  `src/noctis/champions/registry.py`).
- Safety net: a golden regression — `noctis backtest` on the three seed strategies
  byte-identical before/after every phase (they declare no exits).
- No config kill-switch: opt-in-by-declaration *is* the switch, and a knob that
  silently ignores declared stops would make backtest and live disagree — the one thing
  this plan must never do.

**Acceptance for Phase 0:** the ADR/doc section merged; the four decisions above
(API, intrabar policy, re-arm latch, exit-blind prefilter) explicitly signed off.

---

## Phase 1 — Value types and the broker seam

Smallest honest widening of `src/noctis/broker/seam.py`:

- `OrderType` gains `STOP` and `LIMIT` members — **provenance labels for fills**, not a
  resting-order book. The paper broker still executes immediately at a caller-supplied
  price; there is no order queue, and the live adapter remains the refusing stub behind
  the double gate (rule 1 — untouched).
- `Fill` gains `reason: str = "target"` (values: `target | stop | take_profit | trail`)
  so session reporting and the forward ledger can distinguish exit fills. Frozen
  dataclass + default ⇒ every existing constructor call site still type-checks.
- `Broker.rebalance_to` gains a keyword `price: float | None = None` — `None` means
  "at the current mark" (today's behavior, and the default everywhere), a value means
  "this fill executes at this price" (the trigger level or the open, per Phase 0b).
  Slippage still applies adversely on top. Update the Protocol
  (`seam.py:88`), `PaperBroker.rebalance_to` (`src/noctis/broker/paper.py:180`), and
  every implementer the type-checker flags.

Tests: extend `tests/test_broker.py` — priced rebalance fills at price±slippage, fee on
notional at fill price, `reason` propagates, default-argument path byte-identical to
current behavior.

**Size:** ~1 session.

---

## Phase 2 — The exit engine (pure, driver-agnostic)

New module `src/noctis/broker/exits.py` — all pure value types and functions, no I/O,
usable identically by the simulator and the live driver (the same "one implementation,
two drivers" shape as `TargetContext` and `StreamingAggregator`):

```python
@dataclass(frozen=True)
class ExitState:        # per open position
    direction: int      # +1 / −1
    entry_price: float
    best: float         # best favorable extreme since entry (prior-bar ratchet)

@dataclass(frozen=True)
class ExitTrigger:
    price: float        # the fill price per the conservative policy
    reason: str         # "stop" | "take_profit" | "trail"

def evaluate(rules: ExitRules, state: ExitState, bar: Bar) -> ExitTrigger | None: ...
def ratchet(state: ExitState, bar: Bar) -> ExitState: ...   # called AFTER evaluate
```

`evaluate` implements Phase 0b exactly: gap-through at open, stop-beats-TP on same-bar
touch, trail measured from `state.best` (which, because `ratchet` runs after
`evaluate`, is by construction the *prior* bar's extreme). Short positions mirror
symmetrically (stop above, TP below, `best` is the low-water mark).

Tests (`tests/test_exits.py`, exhaustive — this table is the contract):
long/short × {stop, tp, trail} × {touched intrabar, gapped through at open, both
levels same bar, exactly-at-level, no rules armed, warmup/no-position}. Plus the
ratchet-ordering case that *fails* if ratchet ran before evaluate (bar whose high sets
a new best AND whose low would breach the trail measured from that new best, but not
from the prior best).

**Size:** ~1 session. Nothing wired yet; pure functions and tests only.

---

## Phase 3 — Simulator integration

`src/noctis/broker/simulator.py::simulate` — the only place backtest fills are born:

- Track `ExitState | None` and the latch flag per run (single-symbol driver, so one
  slot). Loop body becomes: **(1)** execute pending target at open (re-anchor
  `ExitState` on open/flip; clear on flat); **(2)** if position open and rules armed:
  `evaluate` → on trigger, `rebalance_to(symbol, 0.0, price=trigger.price)` with
  `reason`, set the latch; then `ratchet`; **(3)** `on_bar`, capture
  `(target, exits)`; latch logic per Phase 0c: `pending_target` is suppressed to the
  latched flat until the raw target series changes value.
- `SimResult` needs no schema change — exit fills are in `fills` with their `reason`;
  add `_extra["exit_count"]` for observability.
- Walk-forward and the Scorecard flow through automatically (they consume `simulate`'s
  equity/fills). Turnover and the activity floor count exit fills because they *are*
  fills — verify, don't assume: a dedicated test that a stop-heavy run raises turnover.

Tests: extend `tests/test_strategies.py` simulator section + `tests/test_backtest.py` —
a scripted-target stub strategy with known OHLC tape where the expected equity curve is
hand-computable for: stop-out then latch (held target does NOT re-enter), re-entry
after target change, trail ratchet across a trend, TP on a gap-up open, and the
**seeds-unchanged golden regression** (no exits declared ⇒ `SimResult` fields identical
to pre-change, byte-for-byte on the fixture tape).

**Size:** ~1–2 sessions. The riskiest phase; the golden regression gates the merge.

---

## Phase 4 — Live-driver parity

`run_trading_day` (the trading-day driver) grows the identical three-step ordering,
calling the *same* `exits.evaluate`/`exits.ratchet` on the strategy-timeframe bars the
`StreamingAggregator` emits:

- Precedence: the session halt latch and orphan flattening run **first**, unchanged —
  a halted session flattens through the existing risk path regardless of exit rules;
  exit evaluation only runs for a live, un-halted, champion-held position.
- Forward-ledger crediting: an exit fill closes the position through the normal path,
  so the `holders` map credits the opener exactly as any close does — add one test in
  `tests/test_forward_ledger.py` (fill `reason` visible in the ledger row for
  reporting).
- Reporting: CLOSE report and `noctis report` surface exit fills (count + a per-reason
  line in session activity, via `engine/report_assembly.py`). Small, but it is how the
  operator learns stops are firing.

Tests: `tests/test_runtime_trading.py` / `tests/test_trading.py` replay drives with an
exit-declaring stub — assert live and `simulate` produce the same fill sequence on the
same tape (the parity assertion is the point of this phase).

**Size:** ~1 session.

---

## Phase 5 — Author surface, write gate, docs

- `base.py`: `ExitRules`, `Context.set_target(..., exits=None)`, `TargetContext`
  capture; `replay_targets` unchanged (targets only — parity is about decisions, per
  Phase 0). `mypy` will sweep every `set_target` call site; defaulted kwarg means no
  edits outside the framework.
- Write gate (`library.py` validation): no new checks required — an exits-declaring
  strategy passes the same smoke/scenario/parity gates because scenarios and parity are
  target-level. Add one fixture strategy declaring exits to `tests/test_library.py` to
  prove the subprocess validator round-trips it.
- Scenario oracle: **unchanged.** Scenarios verify the strategy's decisions; exits are
  engine behavior verified by Phases 2–4's tests. Note this boundary in
  `strategies/README.md` so authors don't expect `stopped_out_within(...)` expectations.
- Docs: `strategies/README.md` §exits (API, percent semantics, the latch rule in bold,
  "you will not observe your own stop-outs"), `TEMPLATE.py` comment,
  `docs/architecture.md` fill-model section from Phase 0, `docs/validation.md`
  prefilter-is-exit-blind note. `param_space()` needs no framework change — exit
  percentages are ordinary `float` params the strategy forwards into `ExitRules`, so
  the research agent can tune them immediately; say so explicitly in the README (and
  that the promotion gates will punish stop-fishing the same way they punish any
  overfit knob).

**Size:** ~1 session.

---

## Phase 6 — End-to-end verification and merge posture

- `/verify`-harness drives: author an exits-declaring strategy through
  `write_strategy`, run the pipeline, confirm a sane scorecard, run a replay trading
  day and see the stop fire in the session report.
- Golden regression re-run: three seeds byte-identical through `noctis backtest`.
- Full quality bar: `pytest`, `ruff`, `mypy`, `pre-commit run --all-files`.
- Merge as a single PR *stack* in phase order (0 → 6), each phase reviewable alone;
  Phase 3 does not merge without the seeds-unchanged regression green.

## Risk register (what kills this plan if ignored)

1. **Re-arm latch semantics** (Phase 0c) — the difference between a stop and a speed
   bump. Locked in Phase 0, tested in Phases 3 and 4.
2. **Prefilter divergence** (Phase 0d) — accepted and documented, never approximated.
3. **Intrabar ambiguity** — conservative policy only; any "realistic" intrabar path
   model is unfalsifiable from OHLC and will flatter backtests (rule 3 adjacent).
4. **Backtest/live drift** — one `exits.py` consumed by both drivers, plus the Phase 4
   same-tape parity test.
5. **Scope creep toward limit entries / order books** — refused above; re-litigate
   only as a new plan with its own promotion-gate impact analysis.
6. **Champion comparability** — opt-in design + seeds golden regression means no
   registry migration and no staleness rule; if a later change makes exits *default*,
   that day needs its own comparability story.
