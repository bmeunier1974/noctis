"""The spec interpreter — compiles a resolved ``StrategySpec`` into a +1/0 target series.

Two code paths share **one rule evaluator**:

* the vectorised path (:meth:`SpecRuntime.compute_target_series`) computes every feature as a
  whole ``pd.Series``, then walks the bars applying the scalar signal/entry evaluator;
* the incremental path (:class:`IncrementalEvaluator`) feeds bars to per-feature ``State``
  objects and applies the *same* scalar evaluator.

Because both funnel through :meth:`SpecRuntime._target_step`, the two paths can only diverge if
a primitive's ``vector`` and ``State`` disagree — and that is exactly what the indicator golden
test forbids. So ``signals() == on_bar()`` holds by construction (the ``base.py`` contract).

References mirror grid-mng ``dsl/compileStrategy.ts`` / ``live/evaluateLiveSpec.ts``:
``cross_above`` needs the previous bar, comparisons on a null operand yield ``False``, and the
entry latches long/flat (no short side — out of scope).
"""

from __future__ import annotations

import math

import pandas as pd

from noctis.strategies.base import Bar

from . import indicators as ind
from .schema import (
    ConditionSignal,
    EnsembleSignal,
    FeatureSpec,
    IndicatorFeature,
    RollingExtremeFeature,
    SeriesOpFeature,
    StrategySpec,
    ZScoreFeature,
    parse_ref,
)

NAN = float("nan")
_BAR_KINDS = {"sma", "ema", "rsi", "atr", "vwap", "macd", "rollingExtreme"}


def _isnan(x: float) -> bool:
    return x is None or math.isnan(x)


# ─────────────────────────────────────────────────────────────────────────────
# The shared scalar rule evaluator
# ─────────────────────────────────────────────────────────────────────────────
def _apply_op(op: str, a: float, b: float, pa: float, pb: float) -> bool:
    """Evaluate one comparison on scalars. A null operand → ``False`` (no signal)."""
    if _isnan(a) or _isnan(b):
        return False
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "cross_above":
        if _isnan(pa) or _isnan(pb):
            return False
        return pa <= pb and a > b
    if op == "cross_below":
        if _isnan(pa) or _isnan(pb):
            return False
        return pa >= pb and a < b
    raise ValueError(f"unknown operator {op!r}")


def _latch(pos: int, enter: bool, exit_: bool | None, has_exit: bool) -> int:
    """Long/flat position machine. Stateless (target = enter) when there is no exit."""
    if not has_exit:
        return 1 if enter else 0
    if enter:
        return 1
    if exit_:
        return 0
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# The compiled runtime
# ─────────────────────────────────────────────────────────────────────────────
class SpecRuntime:
    """A spec + a concrete parameter set, ready to produce a target series either way."""

    def __init__(self, spec: StrategySpec, params_values: dict):
        self.spec = spec
        self.pv = dict(params_values)
        self.features = _topo_order(spec)
        self.signals_by_id = {s.id: s for s in spec.signals}
        self.source_ids = {s.id for s in spec.sources}
        self.entry = spec.entries[0]

    # --- parameter resolution ---
    def _num(self, value) -> float:
        """Resolve a ``NumberOrParam`` (literal number or parameter-id ref) to a number."""
        if isinstance(value, str):
            return float(self.pv[value])
        return float(value)

    # --- vectorised feature computation ---
    def _feature_series(self, frame: pd.DataFrame) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        close = frame["close"].astype("float64").to_numpy()
        for sid in self.source_ids:
            out[sid] = close  # a bare source ref resolves to close
        for f in self.features:
            if isinstance(f, IndicatorFeature):  # sma / ema / rsi / atr / vwap
                vfn = ind.REGISTRY[f.kind][0]
                s = vfn(frame, int(self._num(f.period))).to_numpy()
                out[f.id] = out[f"{f.id}:series"] = s
            elif f.kind == "macd":
                d = ind.macd_vector(
                    frame,
                    int(self._num(f.fastPeriod)),
                    int(self._num(f.slowPeriod)),
                    int(self._num(f.signalPeriod)),
                )
                out[f.id] = d["macd"].to_numpy()
                for port in ("macd", "signal", "histogram"):
                    out[f"{f.id}:{port}"] = d[port].to_numpy()
            elif isinstance(f, RollingExtremeFeature):
                s = ind.rolling_extreme_vector(
                    frame, f.mode, int(self._num(f.period)), f.field, f.excludeCurrent
                ).to_numpy()
                out[f.id] = out[f"{f.id}:series"] = s
            elif isinstance(f, ZScoreFeature):
                inp = pd.Series(self._resolve_series(f.input, out, close))
                d = ind.zscore_vector(
                    inp,
                    int(self._num(f.lookback)),
                    self._num(f.upperThreshold),
                    self._num(f.lowerThreshold),
                    f.epsilon,
                )
                out[f.id] = d["zscore"].to_numpy()
                for port in ("zscore", "mean", "std", "above", "below"):
                    out[f"{f.id}:{port}"] = d[port].to_numpy()
            elif isinstance(f, SeriesOpFeature):
                a = pd.Series(self._resolve_series(f.a, out, close))
                b = pd.Series(self._resolve_series(f.b, out, close)) if f.b else None
                s = ind.series_op_vector(a, b, f.scalar, f.op, f.epsilon).to_numpy()
                out[f.id] = out[f"{f.id}:series"] = s
        return out

    @staticmethod
    def _resolve_series(ref: str, out: dict, close) -> list[float]:
        if ref in out:
            return out[ref]
        node_id, _ = parse_ref(ref)
        return out.get(node_id, close)

    def compute_target_series(self, frame: pd.DataFrame) -> list[int]:
        fs = self._feature_series(frame)
        n = len(frame)
        pos = 0
        targets: list[int] = []
        prev: dict[str, float] = {}
        for i in range(n):
            cur = {k: float(v[i]) for k, v in fs.items()}
            pos = self._target_step(pos, cur, prev)
            targets.append(pos)
            prev = cur
        return targets

    def signals(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(self.compute_target_series(frame), dtype=int)

    # --- the shared step (used by both paths) ---
    def _target_step(self, pos: int, cur: dict[str, float], prev: dict[str, float]) -> int:
        enter = self._bool(self.entry.enter, cur, prev)
        exit_ref = self.entry.exit
        exit_ = self._bool(exit_ref, cur, prev) if exit_ref is not None else None
        return _latch(pos, enter, exit_, exit_ref is not None)

    def _bool(self, ref: str, cur: dict[str, float], prev: dict[str, float]) -> bool:
        node_id, _ = parse_ref(ref)
        sig = self.signals_by_id.get(node_id)
        if sig is not None:
            return self._eval_signal(sig, cur, prev)
        val = cur.get(ref, NAN)  # a feature signal port (e.g. zScore ':below')
        return (not _isnan(val)) and bool(val)

    def _eval_signal(self, sig, cur: dict[str, float], prev: dict[str, float]) -> bool:
        if isinstance(sig, ConditionSignal):
            a = cur.get(sig.a, NAN)
            pa = prev.get(sig.a, NAN)
            if sig.b is not None:
                b = cur.get(sig.b, NAN)
                pb = prev.get(sig.b, NAN)
            else:
                b = pb = self._num(sig.threshold)
            return _apply_op(sig.op, a, b, pa, pb)
        # ensemble
        assert isinstance(sig, EnsembleSignal)
        bools = [self._bool(r, cur, prev) for r in sig.inputs]
        return all(bools) if sig.method == "and" else any(bools)

    # --- incremental State construction ---
    def make_state(self, f: FeatureSpec):
        if f.kind == "macd":
            return ind.MacdState(
                int(self._num(f.fastPeriod)),
                int(self._num(f.slowPeriod)),
                int(self._num(f.signalPeriod)),
            )
        if isinstance(f, IndicatorFeature):  # sma / ema / rsi / atr / vwap
            return ind.REGISTRY[f.kind][1](int(self._num(f.period)))
        if isinstance(f, RollingExtremeFeature):
            return ind.RollingExtremeState(
                f.mode, int(self._num(f.period)), f.field, f.excludeCurrent
            )
        if isinstance(f, ZScoreFeature):
            return ind.ZScoreState(
                int(self._num(f.lookback)),
                self._num(f.upperThreshold),
                self._num(f.lowerThreshold),
                f.epsilon,
            )
        if isinstance(f, SeriesOpFeature):
            return ind.SeriesOpState(f.op, f.epsilon)
        raise ValueError(f"no state for feature kind {f.kind!r}")

    def new_incremental(self) -> IncrementalEvaluator:
        return IncrementalEvaluator(self)


class IncrementalEvaluator:
    """Drives the spec bar-by-bar (the ``on_bar`` path), sharing the runtime's rule evaluator."""

    def __init__(self, runtime: SpecRuntime):
        self.rt = runtime
        self.states = {f.id: runtime.make_state(f) for f in runtime.features}
        self.pos = 0
        self.prev: dict[str, float] = {}

    def step(self, bar: Bar) -> int:
        rt = self.rt
        cur: dict[str, float] = {sid: float(bar.close) for sid in rt.source_ids}
        for f in rt.features:
            st = self.states[f.id]
            if f.kind in _BAR_KINDS:
                val = st.update(bar)
                if isinstance(val, dict):  # macd
                    cur[f.id] = val["macd"]
                    for port, v in val.items():
                        cur[f"{f.id}:{port}"] = v
                else:
                    cur[f.id] = cur[f"{f.id}:series"] = val
            elif isinstance(f, ZScoreFeature):
                d = st.update(self._input_scalar(f.input, cur, bar))
                cur[f.id] = d["zscore"]
                for port, v in d.items():
                    cur[f"{f.id}:{port}"] = v
            elif isinstance(f, SeriesOpFeature):
                a = self._input_scalar(f.a, cur, bar)
                b = (
                    self._input_scalar(f.b, cur, bar)
                    if f.b
                    else (f.scalar if f.scalar is not None else NAN)
                )
                cur[f.id] = cur[f"{f.id}:series"] = st.update(a, b)
        self.pos = rt._target_step(self.pos, cur, self.prev)
        self.prev = cur
        return self.pos

    def _input_scalar(self, ref: str, cur: dict[str, float], bar: Bar) -> float:
        if ref in cur:
            return cur[ref]
        node_id, _ = parse_ref(ref)
        if node_id in self.rt.source_ids:
            return float(bar.close)
        return cur.get(node_id, NAN)


# ─────────────────────────────────────────────────────────────────────────────
# Topological ordering of features (seriesOp / zScore consume upstream features)
# ─────────────────────────────────────────────────────────────────────────────
def _feature_deps(f: FeatureSpec, feat_ids: set[str]) -> list[str]:
    if isinstance(f, SeriesOpFeature):
        refs = [f.a] + ([f.b] if f.b else [])
    else:
        refs = [f.input]
    deps = []
    for ref in refs:
        dep, _ = parse_ref(ref)
        if dep in feat_ids:
            deps.append(dep)
    return deps


def _topo_order(spec: StrategySpec) -> list[FeatureSpec]:
    feat_by_id = {f.id: f for f in spec.features}
    feat_ids = set(feat_by_id)
    ordered: list[FeatureSpec] = []
    seen: set[str] = set()

    def visit(fid: str) -> None:
        if fid in seen:
            return
        for dep in _feature_deps(feat_by_id[fid], feat_ids):
            visit(dep)
        seen.add(fid)
        ordered.append(feat_by_id[fid])

    for f in spec.features:
        visit(f.id)
    return ordered
