"""Seed-strategy signal correctness and vectorised↔event-driven parity."""

from __future__ import annotations

import pandas as pd
import pytest

from noctis.broker import FeeModel, PaperBroker, SlippageModel, simulate
from noctis.strategies import (
    DonchianBreakout,
    FamilyRegistry,
    RsiMeanReversion,
    SmaCrossover,
)
from noctis.strategies.base import Bar, ExitRules, TargetContext

from ._data_helpers import make_ohlcv, price_series

ALL_SEEDS = [
    (SmaCrossover, {"fast": 5, "slow": 20}),
    (RsiMeanReversion, {"period": 14, "oversold": 30.0, "overbought": 70.0}),
    (DonchianBreakout, {"channel": 10}),
]


# --- signals() correctness on hand-built fixtures ----------------------------------------


def test_sma_signals_expected():
    closes = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2]
    data = make_ohlcv(closes)
    sig = SmaCrossover.signals(data, SmaCrossover.params_cls(fast=2, slow=4))
    assert list(sig) == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]


def test_rsi_signals_expected():
    closes = [10, 10, 10, 9, 8, 7, 8, 9, 10, 11]
    data = make_ohlcv(closes)
    sig = RsiMeanReversion.signals(
        data, RsiMeanReversion.params_cls(period=3, oversold=30.0, overbought=70.0)
    )
    assert list(sig) == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]


def test_donchian_signals_expected():
    closes = [5, 5, 5, 5, 8, 8, 3, 3]
    data = make_ohlcv(closes)
    sig = DonchianBreakout.signals(data, DonchianBreakout.params_cls(channel=3))
    assert list(sig) == [0, 0, 0, 0, 1, 1, 0, 0]


# --- parity: vectorised signals vs event-driven on_bar -----------------------------------


def _event_targets(strategy, bars: pd.DataFrame) -> list[int]:
    """Collect the per-bar target that on_bar sets (the decision, pre-execution)."""

    class _Ctx:
        def __init__(self):
            self.t = 0
            self._pos = 0

        def set_target(self, t):
            self.t = int(t)

        @property
        def position(self):
            return self._pos

    ctx = _Ctx()
    strategy.on_start(ctx)
    out = []
    for _, row in bars.reset_index(drop=True).iterrows():
        bar = Bar(
            int(row["ts_event"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
        )
        strategy.on_bar(ctx, bar)
        out.append(ctx.t)
    return out


@pytest.mark.parametrize(("cls", "params"), ALL_SEEDS)
def test_signal_parity_vectorised_vs_event(cls, params):
    bars = make_ohlcv(price_series(n=250, seed=11))
    strat = cls.create(**params)
    vectorised = list(cls.signals(bars, strat.params))
    event = _event_targets(strat, bars)
    assert event == vectorised, "event-driven on_bar diverged from vectorised signals()"


@pytest.mark.parametrize(("cls", "params"), ALL_SEEDS)
def test_simulate_targets_match_signals(cls, params):
    """The full simulator's recorded targets also equal the vectorised signals."""
    bars = make_ohlcv(price_series(n=250, seed=3))
    strat = cls.create(**params)
    result = simulate(strat, bars, PaperBroker(), symbol="TST")
    assert result.targets == list(cls.signals(bars, strat.params))


# --- no-lookahead: a final-bar decision cannot be executed -------------------------------


def test_next_bar_execution_prevents_acting_on_final_bar():
    """A strategy that only goes long on the last bar produces zero fills."""

    class LastBarLong(SmaCrossover):
        def on_bar(self, ctx, bar):  # override: long only on the very last close
            self._seen = getattr(self, "_seen", 0) + 1
            ctx.set_target(1 if self._seen == self._n else 0)

    bars = make_ohlcv([10, 11, 12, 13, 14])
    strat = LastBarLong.create(fast=2, slow=3)
    strat._n = len(bars)
    result = simulate(strat, bars, PaperBroker(), symbol="TST")
    assert result.targets[-1] == 1
    # The last-bar decision would execute at a (nonexistent) next bar → no fill happened.
    assert len(result.fills) == 0


# --- protective exits: set_target capture + the simulator's intrabar enforcement ----------


def test_set_target_captures_exits_and_stays_source_compatible():
    """Today's one-argument call shape works unchanged; exits are re-declared per call."""
    ctx = TargetContext()
    ctx.set_target(1)
    assert ctx.target == 1
    assert ctx.exits is None

    rules = ExitRules(stop_pct=0.05)
    ctx.set_target(-1, exits=rules)
    assert ctx.target == -1
    assert ctx.exits is rules

    ctx.set_target(1)  # a declaration without exits clears them — stateless re-declaration
    assert ctx.exits is None


class _ScriptedExits(SmaCrossover):
    """Scripted per-bar targets, each re-declared with the same exit rules."""

    def __init__(self, script, exits):
        super().__init__(self.params_cls())
        self._script = list(script)
        self._exits = exits
        self._i = 0

    def on_start(self, ctx):
        self._i = 0

    def on_bar(self, ctx, bar):
        idx = min(self._i, len(self._script) - 1)
        ctx.set_target(self._script[idx], exits=self._exits)
        self._i += 1


def _tape(rows):
    """An OHLCV frame from explicit (open, high, low, close) rows."""
    return pd.DataFrame(
        {
            "ts_event": range(len(rows)),
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )


def _zero_cost_broker():
    return PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )


def test_stop_out_latches_flat_and_a_held_target_does_not_reenter():
    """A stop is a real exit, not a one-bar speed bump: entry at 100 with a 10% stop is
    stopped at 90 intrabar, and the still-held +1 target stays suppressed to flat."""
    strat = _ScriptedExits([1, 1, 1, 1, 1], ExitRules(stop_pct=0.10))
    tape = _tape(
        [
            (100, 101, 99, 100),  # decide +1 at the close
            (100, 101, 100, 101),  # +1 fills at the open: 950 units at 100
            (100, 100, 88, 92),  # low breaches 90 → stop fill at 90, latch on
            (91, 92, 90, 91),  # held +1 does NOT re-enter
            (91, 92, 90, 91),
        ]
    )

    result = simulate(strat, tape, _zero_cost_broker(), symbol="TST")

    assert [f.reason for f in result.fills] == ["target", "stop"]
    stop_fill = result.fills[1]
    assert stop_fill.price == 90.0  # the level, not the low — conservative, not flattering
    assert stop_fill.quantity == 950.0
    assert result.targets == [1, 1, 0, 0, 0]  # the engine's stance: latched flat after the stop
    assert result.equity_curve == [100_000.0, 100_950.0, 90_500.0, 90_500.0, 90_500.0]
    assert result._extra["exit_count"] == 1


def test_target_change_unlatches_and_reenters():
    """The first raw-target change after a stop-out un-latches; the new value executes."""
    strat = _ScriptedExits([1, 1, 1, 0, 1, 1], ExitRules(stop_pct=0.10))
    tape = _tape(
        [
            (100, 101, 99, 100),
            (100, 101, 100, 101),  # +1 fills: 950 units at 100
            (100, 100, 88, 92),  # stop at 90 → latch
            (91, 92, 90, 91),  # raw flips to 0 → un-latch (still flat)
            (90, 91, 89, 90),  # raw back to +1 → decision to re-enter
            (90.5, 91, 90, 91),  # re-entry fills at the open: 950 units at 90.5
        ]
    )

    result = simulate(strat, tape, _zero_cost_broker(), symbol="TST")

    assert [f.reason for f in result.fills] == ["target", "stop", "target"]
    reentry = result.fills[2]
    assert reentry.price == 90.5
    assert reentry.quantity == 950.0  # 0.95 * 90_500 / 90.5
    assert result.targets == [1, 1, 0, 0, 1, 1]
    assert result.equity_curve[-1] == 90_975.0  # 90_500 + 950 * (91 - 90.5)


def test_trail_ratchets_across_a_trend_and_exits_from_the_best():
    """The trail arms from the ratcheted best (130), not entry — alloc=1.0 keeps the
    hold resize-free so the whole curve is hand-computable."""
    strat = _ScriptedExits([1, 1, 1, 1, 1], ExitRules(trail_pct=0.10))
    tape = _tape(
        [
            (100, 101, 99, 100),
            (100, 101, 99.5, 100.5),  # +1 fills: 1000 units at 100; best ratchets to 101
            (101, 130, 101, 128),  # best ratchets to 130
            (128, 129, 116, 120),  # low 116 breaches 130*0.9=117 → trail exit at 117
            (118, 119, 117, 118),  # held +1 stays latched
        ]
    )

    result = simulate(strat, tape, _zero_cost_broker(), symbol="TST", alloc=1.0)

    assert [f.reason for f in result.fills] == ["target", "trail"]
    trail_fill = result.fills[1]
    assert trail_fill.price == 117.0  # armed only by the bar-2 ratchet; entry-based would be 90
    assert result.targets == [1, 1, 1, 0, 0]
    assert result.equity_curve == [100_000.0, 100_500.0, 128_000.0, 117_000.0, 117_000.0]
    assert result._extra["exit_count"] == 1


def test_take_profit_on_a_gap_up_open_fills_at_the_open():
    """A favorable gap through the take-profit banks the better open price."""
    strat = _ScriptedExits([1, 1, 1, 1], ExitRules(take_profit_pct=0.05))
    tape = _tape(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 100),  # +1 fills: 1000 units at 100; TP level is 105
            (108, 110, 107, 109),  # gaps open above 105 → take-profit fills at 108
            (109, 110, 108, 109),
        ]
    )

    result = simulate(strat, tape, _zero_cost_broker(), symbol="TST", alloc=1.0)

    assert [f.reason for f in result.fills] == ["target", "take_profit"]
    assert result.fills[1].price == 108.0  # the open, not the untouched level
    assert result.targets == [1, 1, 0, 0]
    assert result.equity_curve == [100_000.0, 100_000.0, 108_000.0, 108_000.0]


# --- factory round-trip ------------------------------------------------------------------


def test_registry_families_and_roundtrip():
    families = FamilyRegistry()  # a fresh registry always carries the three seeds
    assert {"sma_crossover", "rsi_meanrev", "donchian_breakout"} <= set(families.names())
    strat = families.create("sma_crossover", {"fast": 5, "slow": 20})
    assert isinstance(strat, SmaCrossover)
    assert strat.params_dict() == {"fast": 5, "slow": 20}


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        FamilyRegistry().create("does_not_exist")


# =========================================================================================
# Spec engine (PR1) — a StrategySpec compiles to an ordinary TraderStrategy family.
# =========================================================================================

import json
import math
from pathlib import Path

import numpy as np

from noctis.strategies.candidate import Candidate
from noctis.strategies.spec import indicators as ind
from noctis.strategies.spec import load_and_register, register_spec
from noctis.strategies.spec.schema import HARD_MAX_INDICATORS, StrategySpec, validate_spec
from noctis.strategies.spec.strategy import family_class_from_spec

from . import _spec_helpers as sh

_GOLDEN = Path(__file__).parent / "fixtures" / "indicator_golden.json"


def _golden_frame(series: dict) -> pd.DataFrame:
    n = len(series["close"])
    return pd.DataFrame(
        {
            "ts_event": [i * 60 * 1_000_000_000 for i in range(n)],  # all within one UTC day
            "open": series["open"],
            "high": series["high"],
            "low": series["low"],
            "close": series["close"],
            "volume": series["volume"],
        }
    )


def _close_enough(got, expected, tol) -> bool:
    if expected is None:
        return got is None or (isinstance(got, float) and math.isnan(got))
    if got is None or (isinstance(got, float) and math.isnan(got)):
        return False
    return abs(float(got) - float(expected)) <= tol


# --- Group 1a: indicator parity vs the copied grid-mng golden fixture --------------------


def test_indicator_golden_fixture():
    gold = json.loads(_GOLDEN.read_text())
    tol = gold["meta"]["tolerance"]
    p = gold["meta"]["params"]
    for sname, series in gold["series"].items():
        frame = _golden_frame(series)
        exp = gold["expected"][sname]
        checks = {
            "sma_5": ind.sma_vector(frame, p["sma"]).tolist(),
            "ema_10": ind.ema_vector(frame, p["ema"]).tolist(),
            "rsi_14": ind.rsi_vector(frame, p["rsi"]).tolist(),
            "atr_14": ind.atr_vector(frame, p["atr"]).tolist(),
            "vwap": ind.vwap_vector(frame).tolist(),
        }
        for key, got in checks.items():
            for i, (g, e) in enumerate(zip(got, exp[key], strict=True)):
                assert _close_enough(g, e, tol), f"{sname}.{key}[{i}]: {g} != {e}"
        macd = ind.macd_vector(frame, *p["macd"])
        for port, key in (("macd", "macd"), ("signal", "signal"), ("histogram", "hist")):
            got = macd[port].tolist()
            for i, (g, e) in enumerate(zip(got, exp["macd_12_26_9"][key], strict=True)):
                assert _close_enough(g, e, tol), f"{sname}.macd.{port}[{i}]: {g} != {e}"
        adx = ind.adx_vector(frame, p["adx"])
        for port in ("adx", "plus_di", "minus_di"):
            got = adx[port].tolist()
            for i, (g, e) in enumerate(zip(got, exp[f"adx_{p['adx']}"][port], strict=True)):
                assert _close_enough(g, e, tol), f"{sname}.adx.{port}[{i}]: {g} != {e}"
        obv = ind.obv_vector(frame).tolist()
        for i, (g, e) in enumerate(zip(obv, exp["obv"], strict=True)):
            assert _close_enough(g, e, tol), f"{sname}.obv[{i}]: {g} != {e}"
        stoch = ind.stoch_vector(frame, *p["stoch_kd"])
        skey = "stoch_" + "_".join(str(x) for x in p["stoch_kd"])
        for port in ("k", "d"):
            got = stoch[port].tolist()
            for i, (g, e) in enumerate(zip(got, exp[skey][port], strict=True)):
                assert _close_enough(g, e, tol), f"{sname}.stoch.{port}[{i}]: {g} != {e}"
        st_p, st_mult = p["supertrend"]
        strend = ind.supertrend_vector(frame, st_p, st_mult)
        stkey = f"supertrend_{st_p}_{int(st_mult)}"
        for port in ("st", "dir"):
            got = strend[port].tolist()
            for i, (g, e) in enumerate(zip(got, exp[stkey][port], strict=True)):
                assert _close_enough(g, e, tol), f"{sname}.supertrend.{port}[{i}]: {g} != {e}"


# --- Group 1b: each primitive's vector fn == its incremental State ------------------------

_STATE_PARITY = [
    ("sma", lambda f: ind.sma_vector(f, 5), lambda: ind.SmaState(5), None),
    ("ema", lambda f: ind.ema_vector(f, 10), lambda: ind.EmaState(10), None),
    ("rsi", lambda f: ind.rsi_vector(f, 14), lambda: ind.RsiState(14), None),
    ("atr", lambda f: ind.atr_vector(f, 14), lambda: ind.AtrState(14), None),
    ("vwap", lambda f: ind.vwap_vector(f), lambda: ind.VwapState(), None),
    (
        "macd",
        lambda f: ind.macd_vector(f, 12, 26, 9)["macd"],
        lambda: ind.MacdState(12, 26, 9),
        "macd",
    ),
    (
        "re",
        lambda f: ind.rolling_extreme_vector(f, "max", 15, "high"),
        lambda: ind.RollingExtremeState("max", 15, "high"),
        None,
    ),
    ("adx", lambda f: ind.adx_vector(f, 14)["adx"], lambda: ind.AdxState(14), None),
    ("obv", lambda f: ind.obv_vector(f), lambda: ind.ObvState(), None),
    (
        "stoch_k",
        lambda f: ind.stoch_vector(f, 14, 3, 3)["k"],
        lambda: ind.StochState(14, 3, 3),
        "k",
    ),
    (
        "stoch_d",
        lambda f: ind.stoch_vector(f, 14, 3, 3)["d"],
        lambda: ind.StochState(14, 3, 3),
        "d",
    ),
    (
        "supertrend",
        lambda f: ind.supertrend_vector(f, 10, 3.0)["st"],
        lambda: ind.SupertrendState(10, 3.0),
        "st",
    ),
    (
        "supertrend_dir",
        lambda f: ind.supertrend_vector(f, 10, 3.0)["dir"],
        lambda: ind.SupertrendState(10, 3.0),
        "dir",
    ),
]


def test_adx_vector_state_di_parity():
    """The +DI/−DI ports agree between adx_vector and AdxState's properties."""
    frame = make_ohlcv(price_series(n=300, seed=5))
    vec = ind.adx_vector(frame, 14)
    state = ind.AdxState(14)
    for i, (_, r) in enumerate(frame.iterrows()):
        bar = Bar(
            int(r.ts_event),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.volume),
        )
        state.update(bar)
        for port, got in (("plus_di", state.plus_di), ("minus_di", state.minus_di)):
            want = vec[port].iloc[i]
            assert (math.isnan(want) and math.isnan(got)) or abs(want - got) <= 1e-9, (
                f"adx.{port}[{i}]: {got} != {want}"
            )


@pytest.mark.parametrize(("name", "vec_fn", "state_fn", "port"), _STATE_PARITY)
def test_indicator_vector_state_parity(name, vec_fn, state_fn, port):
    frame = make_ohlcv(price_series(n=300, seed=5))
    vec = vec_fn(frame).tolist()
    state = state_fn()
    got = []
    for _, r in frame.iterrows():
        bar = Bar(
            int(r.ts_event),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.volume),
        )
        v = state.update(bar)
        got.append(v[port] if port else v)
    for i, (a, b) in enumerate(zip(vec, got, strict=True)):
        assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9, f"{name}[{i}]: {a} != {b}"


def test_zscore_vector_state_parity():
    frame = make_ohlcv(price_series(n=300, seed=8))
    rsi = ind.rsi_vector(frame, 14)
    vec = ind.zscore_vector(rsi, 20, 2.0, -2.0)
    state = ind.ZScoreState(20, 2.0, -2.0)
    rows = [state.update(v) for v in rsi.tolist()]
    for port in ("zscore", "mean", "std", "above", "below"):
        vseries = vec[port].tolist()
        for i, (a, b) in enumerate(zip(vseries, [r[port] for r in rows], strict=True)):
            an, bn = math.isnan(a), math.isnan(b)
            assert (an and bn) or (not an and not bn and abs(a - b) <= 1e-9), f"z.{port}[{i}]"


# --- Group 2: vocabulary sufficiency — the 3 seeds as specs match the hand-coded seeds ----

_SEED_SPEC_CASES = [
    (sh.seed_sma_spec, SmaCrossover, {"fast": 2, "slow": 4}, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2]),
    (
        sh.seed_rsi_spec,
        RsiMeanReversion,
        {"period": 3, "oversold": 30.0, "overbought": 70.0},
        [10, 10, 10, 9, 8, 7, 8, 9, 10, 11],
    ),
    (sh.seed_donchian_spec, DonchianBreakout, {"channel": 3}, [5, 5, 5, 5, 8, 8, 3, 3]),
]


@pytest.mark.parametrize(("spec_fn", "seed_cls", "params", "closes"), _SEED_SPEC_CASES)
def test_seed_spec_matches_hand_coded(spec_fn, seed_cls, params, closes):
    data = make_ohlcv(closes)
    spec_cls = family_class_from_spec(spec_fn())
    spec_sig = list(spec_cls.signals(data, spec_cls.params_cls(**params)))
    seed_sig = list(seed_cls.signals(data, seed_cls.params_cls(**params)))
    assert spec_sig == seed_sig


# RSI is intentionally excluded here: the hand-coded seed averages gains/losses with a simple
# rolling mean, while the spec's rsi primitive uses Wilder smoothing (to match the golden
# fixture). The two coincide on the small oversold/overbought fixture above but diverge on a
# long path — so the seed-equivalence-at-scale check only applies to SMA and Donchian, whose
# indicators (rolling mean / rolling extreme) match the seeds by definition.
@pytest.mark.parametrize("spec_fn", [sh.seed_sma_spec, sh.seed_donchian_spec])
def test_seed_spec_matches_hand_coded_random_walk(spec_fn):
    """On a longer random path the spec form still equals the hand-coded seed it mirrors."""
    data = make_ohlcv(price_series(n=250, seed=11))
    spec_cls = family_class_from_spec(spec_fn())
    seed_cls = {"spec_sma_crossover": SmaCrossover, "spec_donchian_breakout": DonchianBreakout}[
        spec_cls.name
    ]
    p = spec_cls.params_cls()  # defaults mirror the seed defaults
    seed_p = seed_cls.params_cls(
        **{f: getattr(p, f) for f in spec_cls.params_cls.__dataclass_fields__}
    )
    assert list(spec_cls.signals(data, p)) == list(seed_cls.signals(data, seed_p))


@pytest.mark.parametrize("spec_fn", sh.FIXTURE_FAMILIES)
def test_fixture_family_specs_run(spec_fn):
    """The grid-mng specFixtures families (classicBreakout, zScoreReversion) are expressible
    and produce a valid long/flat series."""
    data = make_ohlcv(price_series(n=200, seed=4))
    cls = family_class_from_spec(spec_fn())
    sig = list(cls.signals(data, cls.params_cls()))
    assert len(sig) == len(data)
    assert set(sig) <= {0, 1}


# --- Group 3: generic SpecStrategy parity — event on_bar == vectorised signals ------------


@pytest.mark.parametrize("idx", range(24))
def test_generic_specstrategy_parity(idx):
    rng = np.random.default_rng(1000 + idx)
    spec = sh.random_valid_spec(rng, idx)
    cls = family_class_from_spec(spec)
    bars = make_ohlcv(price_series(n=220, seed=idx + 1))
    params = cls.params_cls()
    vectorised = list(cls.signals(bars, params))
    event = _event_targets(cls(params), bars)
    assert event == vectorised, f"{spec.id}: on_bar diverged from signals()"


# --- Group 4: schema validation rejects malformed / cyclic / oversize / dangling ---------


def test_schema_rejects_unknown_kind():
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [{"id": "f", "kind": "no_such_indicator", "input": "src", "period": 5}],
                "entries": [{"id": "e", "enter": "f"}],
            }
        )


def test_schema_rejects_dangling_ref():
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [{"id": "f", "kind": "sma", "input": "nope", "period": 5}],
                "signals": [{"id": "en", "kind": "condition", "op": ">", "a": "f", "threshold": 0}],
                "entries": [{"id": "e", "enter": "en"}],
            }
        )


def test_schema_rejects_cyclic_ref():
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [
                    {"id": "a", "kind": "seriesOp", "op": "add", "a": "b", "scalar": 1.0},
                    {"id": "b", "kind": "seriesOp", "op": "add", "a": "a", "scalar": 1.0},
                ],
                "signals": [{"id": "en", "kind": "condition", "op": ">", "a": "a", "threshold": 0}],
                "entries": [{"id": "e", "enter": "en"}],
            }
        )


def _sized_spec(n_features: int) -> dict:
    features = [
        {"id": f"f{i}", "kind": "sma", "input": "src", "period": i + 2} for i in range(n_features)
    ]
    return {
        "version": 1,
        "id": "sized",
        "sources": [{"id": "src", "schema": "ohlcv-1m"}],
        "features": features,
        "signals": [{"id": "en", "kind": "condition", "op": ">", "a": "f0", "b": "f1"}],
        "entries": [{"id": "e", "enter": "en"}],
    }


def test_schema_rejects_oversize():
    """Parse time enforces only the generous hard ceiling."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(_sized_spec(HARD_MAX_INDICATORS + 1))


def test_validate_spec_enforces_configured_cap():
    """A 13-feature spec parses (hard ceiling), the default admission cap (12) rejects it,
    and a raised configured cap admits it — so ideation.max_indicators > 12 is effective."""
    spec = StrategySpec.model_validate(_sized_spec(13))
    with pytest.raises(Exception):
        validate_spec(spec)
    assert validate_spec(spec, max_indicators=20) is spec


def test_schema_rejects_non_boolean_entry():
    """An entry must reference a boolean signal, not a numeric feature series."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [{"id": "f", "kind": "sma", "input": "src", "period": 5}],
                "entries": [{"id": "e", "enter": "f"}],  # f is a numeric series, not a signal
            }
        )


def test_schema_rejects_source_as_entry():
    """enter: <source> would compile to an always-long pseudo-strategy; reject it."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "entries": [{"id": "e", "enter": "src"}],
            }
        )


def test_schema_rejects_signal_cycle():
    """Mutually-referencing ensembles must fail validation, not recurse at evaluation."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "signals": [
                    {"id": "s1", "kind": "ensemble", "method": "and", "inputs": ["s2"]},
                    {"id": "s2", "kind": "ensemble", "method": "and", "inputs": ["s1"]},
                ],
                "entries": [{"id": "e", "enter": "s1"}],
            }
        )


def test_schema_rejects_unknown_threshold_param():
    """A condition threshold given as a param-id string must name a real parameter."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [{"id": "r", "kind": "rsi", "input": "src", "period": 14}],
                "signals": [
                    {
                        "id": "en",
                        "kind": "condition",
                        "op": "<",
                        "a": "r",
                        "threshold": "no_such_param",
                    }
                ],
                "entries": [{"id": "e", "enter": "en"}],
            }
        )


def test_schema_rejects_non_boolean_ensemble_input():
    """Ensemble inputs must be boolean signals, not numeric feature series."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [{"id": "f", "kind": "sma", "input": "src", "period": 5}],
                "signals": [{"id": "en", "kind": "ensemble", "method": "and", "inputs": ["f"]}],
                "entries": [{"id": "e", "enter": "en"}],
            }
        )


def test_schema_rejects_condition_on_signal():
    """Condition operands are numeric; a ref to another signal would coerce to null → False."""
    with pytest.raises(Exception):
        StrategySpec.model_validate(
            {
                "version": 1,
                "id": "bad",
                "sources": [{"id": "src", "schema": "ohlcv-1m"}],
                "features": [{"id": "f", "kind": "sma", "input": "src", "period": 5}],
                "signals": [
                    {"id": "s1", "kind": "condition", "op": ">", "a": "f", "threshold": 0},
                    {"id": "s2", "kind": "condition", "op": ">", "a": "s1", "threshold": 0},
                ],
                "entries": [{"id": "e", "enter": "s2"}],
            }
        )


# --- Group 5: persistence round-trip — register, reload in a fresh registry, run ----------


def test_spec_persistence_round_trip(tmp_path):
    spec = sh.seed_rsi_spec()
    register_spec(spec, tmp_path, FamilyRegistry())
    assert (tmp_path / "specs.json").is_file()
    on_disk = json.loads((tmp_path / "specs.json").read_text())
    assert spec.id in on_disk["specs"]

    # Simulate a fresh process: a brand-new registry, re-registered from disk only.
    families = FamilyRegistry()
    assert spec.id not in families
    names = load_and_register(tmp_path, families)
    assert spec.id in names
    assert spec.id in families

    # A spec-backed Candidate rebuilds and runs through the ordinary engine.
    bars = make_ohlcv(price_series(n=150, seed=2))
    candidate = Candidate(spec.id, {"period": 10, "oversold": 25.0, "overbought": 75.0})
    result = simulate(candidate.build(families), bars, PaperBroker(), symbol="TST")
    assert len(result.targets) == len(bars)
    assert set(result.targets) <= {0, 1}
