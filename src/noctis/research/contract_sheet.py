"""The API contract sheet — the exact surface the write gate grades the coder against.

A live session showed the coder hallucinating helper APIs because its prompt never showed the
real signatures (only ``TEMPLATE.py``, which deliberately elides them). This module closes that
gap: one **data table** of every scenario-DSL builder, expectation, indicator tail function and
State class, and every :class:`~noctis.strategies.base.ExitRules` field — each with its exact
signature and the semantics the validation gate enforces — rendered into a deterministic prompt
block (:data:`CONTRACT_SHEET`) that :class:`~noctis.research.author.StrategyAuthor` folds into the
coder's system prompt.

The table is the **single source of truth**: it is hand-written (not runtime introspection, so the
prompt is deterministic and readable in tests) but shaped as data — ``name → signature → note`` —
so the same rows can later drive retry-error enrichment (epic #13 story #20) without a second copy
of the API surface. A drift-guard test (``tests/test_contract_sheet.py``) walks every row and
asserts its signature against the live modules via :func:`inspect.signature`, so the sheet can
never silently rot away from the code it grades against.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sentinel: a parameter that has no default (drift guard compares this to inspect's `empty`).
REQUIRED: object = object()


@dataclass(frozen=True)
class Param:
    """One positional/keyword parameter of a declared API symbol."""

    name: str
    default: object = REQUIRED

    def render(self) -> str:
        if self.default is REQUIRED:
            return self.name
        return f"{self.name}={self.default!r}"


@dataclass(frozen=True)
class Entry:
    """One callable/class the sheet declares: its name, signature params, and semantics note.

    ``update_arg`` marks an indicator State class and names the first argument of its
    ``.update(...)`` method — ``"bar"`` for the Bar-driven majority, ``"x"`` for the documented
    float-updating exception — so the drift guard can pin the calling convention too.
    """

    name: str
    params: tuple[Param, ...]
    note: str
    update_arg: str | None = None

    def signature(self) -> str:
        return f"{self.name}({', '.join(p.render() for p in self.params)})"

    def render(self) -> str:
        return f"  {self.signature()} — {self.note}"


@dataclass(frozen=True)
class Section:
    """A titled group of entries sharing a live module (the drift guard's resolution root)."""

    title: str
    blurb: str
    module_name: str
    entries: tuple[Entry, ...]

    def render(self) -> str:
        lines = [self.title, self.blurb]
        lines.extend(entry.render() for entry in self.entries)
        return "\n".join(lines)


@dataclass(frozen=True)
class Constant:
    """A numeric tape-shape bound the sheet states verbatim and the drift guard pins to code."""

    label: str
    value: int
    live_name: str


def _p(name: str, default: object = REQUIRED) -> Param:
    return Param(name, default)


# ── the tape-shape numeric bounds (mirrored from scenarios.py, drift-guarded) ──────────────
TAPE_CONSTANTS: tuple[Constant, ...] = (
    Constant("min scenarios", 2, "MIN_SCENARIOS"),
    Constant("max scenarios", 8, "MAX_SCENARIOS"),
    Constant("min bars per tape", 60, "MIN_SCENARIO_BARS"),
    Constant("max bars per tape", 2000, "MAX_SCENARIO_BARS"),
)


_SCENARIO_BUILDERS = Section(
    title="Scenario DSL builders — build every tape's segments ONLY from these (import as "
    "`from noctis.strategies import scenarios as sc`):",
    blurb="  These seven are the entire segment DSL; no other builder exists — do not invent one.",
    module_name="noctis.strategies.scenarios",
    entries=(
        Entry("flat", (_p("n"),), "n bars at a constant price (the no-information leg)."),
        Entry(
            "trend",
            (_p("n"), _p("pct")),
            "n bars drifting geometrically to (1+pct) of the start; pct is the signed total "
            "move as a fraction (0.05 = +5%, -0.05 = -5%).",
        ),
        Entry(
            "selloff",
            (_p("n"), _p("pct")),
            "n bars declining a total of -|pct| (capitulation-shaped).",
        ),
        Entry(
            "recovery",
            (_p("n"), _p("pct")),
            "n bars advancing a total of +|pct| (the bounce after a selloff).",
        ),
        Entry(
            "chop",
            (_p("n"), _p("amplitude"), _p("period", 8)),
            "n bars oscillating +/-amplitude around the level with no net drift; period is the "
            "wave length in bars.",
        ),
        Entry(
            "vol_spike",
            (_p("n"), _p("amplitude", 0.05)),
            "n bars of violent short-period oscillation (a volatility burst).",
        ),
        Entry(
            "gap",
            (_p("pct"),),
            "an instantaneous pct jump between bars; emits NO bars, so it adds 0 to the tape "
            "length.",
        ),
    ),
)


_EXPECTATIONS = Section(
    title="Expectations — attach to each Scenario's expect=(...) (same `sc` import):",
    blurb="  Windows over the replayed target series (+1 long / 0 flat / -1 short), not per-bar "
    "series.",
    module_name="noctis.strategies.scenarios",
    entries=(
        Entry("flat_until", (_p("bar"),), "flat on every bar strictly before `bar`."),
        Entry(
            "long_within",
            (_p("lo"), _p("hi")),
            "long on at least one bar in [lo, hi] (directional).",
        ),
        Entry(
            "holds_long_through",
            (_p("lo"), _p("hi")),
            "long on every bar in [lo, hi] — held, not flickered (directional).",
        ),
        Entry(
            "short_within",
            (_p("lo"), _p("hi")),
            "short on at least one bar in [lo, hi] (directional).",
        ),
        Entry(
            "holds_short_through",
            (_p("lo"), _p("hi")),
            "short on every bar in [lo, hi] — held, not flickered (directional).",
        ),
        Entry(
            "flat_by",
            (_p("bar"),),
            "flat on every bar from `bar` to the end (the exit must have fired).",
        ),
        Entry(
            "always_flat",
            (),
            "flat on every bar — the mandatory no-trade tape; takes zero arguments.",
        ),
    ),
)


_TAIL_FUNCTIONS = Section(
    title="Indicator tail functions — pure functions over a list/deque of floats you keep "
    "yourself (import as `from noctis.strategies import indicators as ind`):",
    blurb="  Each returns None during warmup AND on degenerate windows, so ALWAYS guard with "
    "`if v is not None`.",
    module_name="noctis.strategies.indicators",
    entries=(
        Entry("sma", (_p("values"), _p("period")), "mean of the last `period` values."),
        Entry("ema", (_p("values"), _p("period")), "SMA-seeded EMA over the recent window."),
        Entry("rsi", (_p("values"), _p("period")), "Cutler's RSI, 0-100; needs period+1 values."),
        Entry(
            "atr",
            (_p("highs"), _p("lows"), _p("closes"), _p("period")),
            "simple-average true range; needs period+1 bars of history.",
        ),
        Entry("stdev", (_p("values"), _p("period")), "population standard deviation."),
        Entry(
            "zscore",
            (_p("values"), _p("period")),
            "(last - mean)/sigma; also None when the window deviation is zero (flat tape).",
        ),
        Entry(
            "bollinger",
            (_p("values"), _p("period"), _p("mult", 2.0)),
            "returns (upper, mid, lower).",
        ),
        Entry("roc", (_p("values"), _p("period")), "percent change vs `period` bars ago."),
        Entry("wma", (_p("values"), _p("period")), "linear-weighted mean of the last `period`."),
        Entry("highest", (_p("values"), _p("period")), "max of the last `period` values."),
        Entry("lowest", (_p("values"), _p("period")), "min of the last `period` values."),
        Entry(
            "stoch_k",
            (_p("highs"), _p("lows"), _p("closes"), _p("period")),
            "raw stochastic %K, 0-100; None on a flat window.",
        ),
        Entry(
            "cci",
            (_p("highs"), _p("lows"), _p("closes"), _p("period")),
            "commodity channel index; None on a flat window.",
        ),
        Entry(
            "bars_since",
            (_p("flags"),),
            "bars since the latest True in your flag deque; returns an int (0 if true now), None "
            "if never true.",
        ),
        Entry(
            "cross_above",
            (_p("fast_prev"), _p("fast_now"), _p("slow_prev"), _p("slow_now")),
            "returns bool: fast crossed from at-or-below to above slow.",
        ),
        Entry(
            "cross_below",
            (_p("fast_prev"), _p("fast_now"), _p("slow_prev"), _p("slow_now")),
            "returns bool: fast crossed from at-or-above to below slow.",
        ),
    ),
)


_STATE_CLASSES = Section(
    title="Indicator State classes — construct in on_start, call `.update(bar)` once per Bar in "
    "on_bar (same `ind` import):",
    blurb="  Each returns nan during warmup (guard with `math.isnan`). `.update(bar)` takes one "
    "Bar object — EXCEPT ZScoreState, noted below.",
    module_name="noctis.strategies.indicators",
    entries=(
        Entry("SmaState", (_p("period"),), ".update(bar) -> float.", update_arg="bar"),
        Entry("EmaState", (_p("period"),), ".update(bar) -> float.", update_arg="bar"),
        Entry("RsiState", (_p("period"),), ".update(bar) -> float.", update_arg="bar"),
        Entry("AtrState", (_p("period"),), ".update(bar) -> float.", update_arg="bar"),
        Entry(
            "MacdState",
            (_p("fastPeriod"), _p("slowPeriod"), _p("signalPeriod")),
            ".update(bar) -> {'macd', 'signal', 'histogram'}.",
            update_arg="bar",
        ),
        Entry(
            "VwapState",
            (_p("period", None),),
            ".update(bar) -> float; session VWAP, resets each UTC day (period is unused).",
            update_arg="bar",
        ),
        Entry(
            "AdxState",
            (_p("period"),),
            ".update(bar) -> ADX float; also exposes .plus_di / .minus_di attributes.",
            update_arg="bar",
        ),
        Entry(
            "ObvState",
            (),
            ".update(bar) -> float; no period, no warmup nan (on-balance volume from 0.0).",
            update_arg="bar",
        ),
        Entry(
            "StochState",
            (_p("period"), _p("k_smooth", 3), _p("d_smooth", 3)),
            ".update(bar) -> {'k', 'd'}.",
            update_arg="bar",
        ),
        Entry(
            "SupertrendState",
            (_p("period"), _p("mult", 3.0)),
            ".update(bar) -> {'st', 'dir'}.",
            update_arg="bar",
        ),
        Entry(
            "ZScoreState",
            (_p("lookback"), _p("upperThreshold"), _p("lowerThreshold"), _p("epsilon", 1e-8)),
            "EXCEPTION — .update(x) takes a float, not a Bar; -> "
            "{'zscore', 'mean', 'std', 'above', 'below'}.",
            update_arg="x",
        ),
        Entry(
            "RollingExtremeState",
            (_p("mode"), _p("period"), _p("field", None), _p("excludeCurrent", True)),
            ".update(bar) -> float; mode is 'max' or 'min'.",
            update_arg="bar",
        ),
    ),
)


_EXIT_RULES = Section(
    title="Protective exits — declare alongside the target: "
    "ctx.set_target(target, exits=ExitRules(...)) (from noctis.strategies.base import ExitRules):",
    blurb="  All three fields are optional fractions of the entry price, each armed only when set.",
    module_name="noctis.strategies.base",
    entries=(
        Entry(
            "ExitRules",
            (_p("stop_pct", None), _p("take_profit_pct", None), _p("trail_pct", None)),
            "exactly these three fields; stop_pct/take_profit_pct/trail_pct as fractions of entry "
            "price. Nothing else exists.",
        ),
    ),
)


SECTIONS: tuple[Section, ...] = (
    _SCENARIO_BUILDERS,
    _EXPECTATIONS,
    _TAIL_FUNCTIONS,
    _STATE_CLASSES,
    _EXIT_RULES,
)


def _tape_shape_block() -> str:
    lo_s, hi_s = TAPE_CONSTANTS[0].value, TAPE_CONSTANTS[1].value
    lo_b, hi_b = TAPE_CONSTANTS[2].value, TAPE_CONSTANTS[3].value
    return "\n".join(
        [
            "Tape shape rules the gate enforces:",
            f"  - Declare {lo_s}-{hi_s} Scenario objects from scenarios(), each with a unique "
            "name.",
            f"  - A tape's length = the sum of each segment's n (gap adds none); it must be "
            f"{lo_b}-{hi_b} bars.",
            "  - Every expectation's referenced bar index must fall strictly inside the tape "
            "(last index < tape length).",
            "  - Across the whole set: at least one directional expectation "
            "(long_within/holds_long_through/short_within/holds_short_through) AND at least one "
            "always_flat() no-trade tape.",
        ]
    )


def render_contract_sheet() -> str:
    """Render the data table into the deterministic prompt block folded into the coder's system.

    Pure over the module-level table (no I/O, no introspection), so the prompt is byte-stable and
    directly assertable in tests.
    """
    blocks = [
        "=== NOCTIS AUTHORING API CONTRACT ===",
        "Every symbol below is the exact surface the write gate executes. Use these signatures "
        "verbatim and call NOTHING that is not listed here — an unlisted helper is a "
        "hallucination the gate rejects.",
        _SCENARIO_BUILDERS.render(),
        _EXPECTATIONS.render(),
        _tape_shape_block(),
        _TAIL_FUNCTIONS.render(),
        _STATE_CLASSES.render(),
        _EXIT_RULES.render(),
    ]
    return "\n\n".join(blocks)


CONTRACT_SHEET: str = render_contract_sheet()
