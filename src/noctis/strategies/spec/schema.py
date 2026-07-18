"""Pydantic mirror of the grid-mng v1 ``StrategySpec`` — the subset Noctis needs.

A ``StrategySpec`` is *strategy as data*: a normalized, JSON-friendly graph of
``sources → features → signals → entries`` that compiles to a
:class:`~noctis.strategies.base.TraderStrategy`. This vendors only the slice needed for a
long/flat (+1 / 0) target series; sizing / risk / slippage / outputs / execution live on
``RiskConfig`` + ``PaperBroker`` and are deliberately out of scope (see the plan).

Wiring is by **reference**, exactly as grid-mng: a ``Ref`` names an upstream node id, optionally
a specific output port (``"<id>:<port>"``). Validators reject unknown kinds (via the
discriminated unions), dangling / cyclic references, a missing entry, and oversize graphs so a
malformed spec can never reach the interpreter.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from noctis.strategies.base import ParamSpec

# ─────────────────────────────────────────────────────────────────────────────
# Size caps (a spec is one family; keep the graph small enough to tune + reason about).
# The DEFAULT_* caps are what :func:`validate_spec` (the ideation admission gate) enforces
# unless configured otherwise; the HARD_* ceilings are enforced at parse time and only bound
# what can ever be parsed or persisted, so a configured cap above the default still bites.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_INDICATORS = 12
DEFAULT_MAX_NODES = 60
HARD_MAX_INDICATORS = 48
HARD_MAX_NODES = 240

ConditionOperator = Literal[">", ">=", "<", "<=", "==", "!=", "cross_above", "cross_below"]

# A number literal *or* a parameter-id reference. Tunable feature/signal fields hold the id of
# a ``ParameterSpec`` (str); constant fields hold the number directly. Resolved at signal-time
# against the params dataclass so Optuna tunes the params and the spec reads them back.
NumberOrParam = float | int | str


# ─────────────────────────────────────────────────────────────────────────────
# Ref helpers
# ─────────────────────────────────────────────────────────────────────────────
Ref = str


def parse_ref(ref: Ref) -> tuple[str, str | None]:
    """Split a ``Ref`` into ``(id, port)``. The first ':' separates id from port."""
    idx = ref.find(":")
    if idx == -1:
        return ref, None
    return ref[:idx], ref[idx + 1 :]


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────
class SourceSpec(BaseModel):
    """The OHLCV frame. Symbol/dates are authoring metadata; the runtime is handed bars."""

    id: str
    symbol: str = "SYM"
    schema_: str = Field("ohlcv-1m", alias="schema")
    start: str = ""
    end: str = ""

    model_config = {"populate_by_name": True}


# ─────────────────────────────────────────────────────────────────────────────
# Features (indicators + transforms). Each carries a `kind` discriminator.
# ─────────────────────────────────────────────────────────────────────────────
IndicatorKind = Literal["sma", "ema", "rsi", "atr", "vwap"]
OhlcvField = Literal["open", "high", "low", "close", "volume"]


class IndicatorFeature(BaseModel):
    """SMA/EMA/RSI/ATR/VWAP over a bar stream. Output: ``series``."""

    id: str
    kind: IndicatorKind
    input: Ref
    period: NumberOrParam = 14


class MacdFeature(BaseModel):
    """MACD. Outputs: ``macd`` (primary), ``signal``, ``histogram``."""

    id: str
    kind: Literal["macd"]
    input: Ref
    fastPeriod: NumberOrParam = 12
    slowPeriod: NumberOrParam = 26
    signalPeriod: NumberOrParam = 9


class ZScoreFeature(BaseModel):
    """Rolling z-score over a numeric series. Outputs: ``zscore`` (primary), ``mean``, ``std``,
    ``above`` (signal), ``below`` (signal)."""

    id: str
    kind: Literal["zScore"]
    input: Ref
    lookback: NumberOrParam = 20
    upperThreshold: NumberOrParam = 2.0
    lowerThreshold: NumberOrParam = -2.0
    epsilon: float = 1e-8


class RollingExtremeFeature(BaseModel):
    """Rolling highest-high / lowest-low over a bar field. Output: ``series``.

    ``excludeCurrent`` (default true) → the window is the PRIOR ``period`` bars, which is what
    makes a breakout cross fire (the current bar is not part of the ceiling it crosses).
    """

    id: str
    kind: Literal["rollingExtreme"]
    input: Ref
    mode: Literal["max", "min"]
    period: NumberOrParam = 20
    field: OhlcvField | None = None
    excludeCurrent: bool = True


class SeriesOpFeature(BaseModel):
    """Elementwise binary arithmetic on two numeric series (or a series + scalar). Output:
    ``series``. Exactly one of ``b`` / ``scalar`` is set (``ratio`` = op ``div``)."""

    id: str
    kind: Literal["seriesOp"]
    op: Literal["add", "sub", "mul", "div"]
    a: Ref
    b: Ref | None = None
    scalar: float | None = None
    epsilon: float = 1e-8


FeatureSpec = Annotated[
    IndicatorFeature | MacdFeature | ZScoreFeature | RollingExtremeFeature | SeriesOpFeature,
    Field(discriminator="kind"),
]

# Which output ports each feature kind exposes (first = primary). Used for ref validation.
FEATURE_PORTS: dict[str, tuple[str, ...]] = {
    "sma": ("series",),
    "ema": ("series",),
    "rsi": ("series",),
    "atr": ("series",),
    "vwap": ("series",),
    "macd": ("macd", "signal", "histogram"),
    "zScore": ("zscore", "mean", "std", "above", "below"),
    "rollingExtreme": ("series",),
    "seriesOp": ("series",),
}
# Ports that carry a boolean signal (may be referenced directly by an entry / ensemble).
SIGNAL_PORTS: set[tuple[str, str]] = {("zScore", "above"), ("zScore", "below")}


# ─────────────────────────────────────────────────────────────────────────────
# Signals (produce boolean streams)
# ─────────────────────────────────────────────────────────────────────────────
class ConditionSignal(BaseModel):
    """Comparison of ``a`` against ``b`` (or a scalar ``threshold``). Output: ``signal``."""

    id: str
    kind: Literal["condition"]
    op: ConditionOperator
    a: Ref
    b: Ref | None = None
    threshold: NumberOrParam | None = None


class EnsembleSignal(BaseModel):
    """Combine 1–4 boolean signals with ``and`` / ``or``. Output: ``signal``."""

    id: str
    kind: Literal["ensemble"]
    inputs: list[Ref]
    method: Literal["and", "or"] = "and"


SignalSpec = Annotated[ConditionSignal | EnsembleSignal, Field(discriminator="kind")]


# ─────────────────────────────────────────────────────────────────────────────
# Entries — map a signal → the long/flat position machine (no short side; out of scope).
# ─────────────────────────────────────────────────────────────────────────────
class EntrySpec(BaseModel):
    """Long/flat entry. ``enter`` fires the long; optional ``exit`` returns to flat.

    With no ``exit`` the target is stateless (``+1`` iff ``enter``); with an ``exit`` it latches
    (``0→1`` on enter, ``1→0`` on exit), mirroring the pos-latch loop in the seed strategies.
    """

    id: str
    enter: Ref
    exit: Ref | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Parameters & optimizations (the Optuna search domain)
# ─────────────────────────────────────────────────────────────────────────────
class ParameterSpec(BaseModel):
    """A named tunable scalar with a default. Becomes a field on the params dataclass."""

    id: str
    kind: Literal["int", "float"] = "int"
    value: float | int = 0
    label: str | None = None

    @field_validator("id")
    @classmethod
    def _id_is_identifier(cls, v: str) -> str:
        # Parameter ids become fields on the params dataclass, so they must be identifiers.
        if not v.isidentifier():
            raise ValueError(f"parameter id {v!r} is not a valid identifier")
        return v


class OptimizationParam(BaseModel):
    """One swept parameter: references a :class:`ParameterSpec` and gives its search range."""

    param: str
    type: Literal["int", "float"] = "int"
    min: float
    max: float
    step: float | None = None


class OptimizationSpec(BaseModel):
    """Which parameters to sweep and over what range (the family's ``param_space``)."""

    id: str = "opt"
    parameters: list[OptimizationParam] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# The StrategySpec
# ─────────────────────────────────────────────────────────────────────────────
class StrategySpec(BaseModel):
    """A whole strategy, as data. ``id`` is the family name it registers under."""

    version: Literal[1] = 1
    id: str
    name: str = ""
    description: str = ""
    sources: list[SourceSpec]
    features: list[FeatureSpec] = Field(default_factory=list)
    signals: list[SignalSpec] = Field(default_factory=list)
    entries: list[EntrySpec]
    parameters: list[ParameterSpec] = Field(default_factory=list)
    optimizations: list[OptimizationSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_graph(self) -> StrategySpec:
        _check_refs_and_cycles(self)
        # Parse-time size checks use the generous hard ceilings only; the configured caps
        # (e.g. ``ideation.max_indicators``) are enforced by validate_spec on admission.
        _check_size(self, HARD_MAX_INDICATORS, HARD_MAX_NODES)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def _node_ports(spec: StrategySpec) -> dict[str, tuple[str, str | None]]:
    """Map each node id → ``(group, kind)`` so refs can be type-checked."""
    out: dict[str, tuple[str, str | None]] = {}
    for s in spec.sources:
        out[s.id] = ("source", None)
    for f in spec.features:
        out[f.id] = ("feature", f.kind)
    for sig in spec.signals:
        out[sig.id] = ("signal", sig.kind)
    return out


def _check_ref(ref: Ref, nodes: dict[str, tuple[str, str | None]], *, where: str) -> None:
    node_id, port = parse_ref(ref)
    if node_id not in nodes:
        raise ValueError(f"dangling ref {ref!r} in {where}: no node with id {node_id!r}")
    group, kind = nodes[node_id]
    if port is not None and group == "feature":
        allowed = FEATURE_PORTS.get(kind or "", ())
        if port not in allowed:
            raise ValueError(
                f"invalid port {port!r} on {ref!r} in {where}; {kind} exposes {allowed}"
            )


def _feature_inputs(feat: FeatureSpec) -> list[Ref]:
    if isinstance(feat, SeriesOpFeature):
        return [feat.a] + ([feat.b] if feat.b else [])
    return [feat.input]


def _is_boolean_ref(ref: Ref, nodes: dict[str, tuple[str, str | None]]) -> bool:
    """True iff ``ref`` resolves to a boolean stream: a signal, or a feature signal port."""
    node_id, port = parse_ref(ref)
    group, kind = nodes[node_id]
    if group == "signal":
        return True
    return group == "feature" and (kind, port) in SIGNAL_PORTS


def _check_acyclic(graph: dict[str, list[str]], what: str) -> None:
    """DFS colouring over ``node id → upstream ids``; raises on the first back-edge."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)

    def visit(node: str) -> None:
        colour[node] = GREY
        for dep in graph[node]:
            if colour.get(dep) == GREY:
                raise ValueError(f"cyclic {what} reference at {node!r} → {dep!r}")
            if colour.get(dep) == WHITE:
                visit(dep)
        colour[node] = BLACK

    for node in graph:
        if colour[node] == WHITE:
            visit(node)


def _check_refs_and_cycles(spec: StrategySpec) -> None:
    # Unique ids across every group.
    ids: list[str] = (
        [s.id for s in spec.sources]
        + [f.id for f in spec.features]
        + [s.id for s in spec.signals]
        + [e.id for e in spec.entries]
    )
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate node ids: {sorted(dupes)}")

    if not spec.sources:
        raise ValueError("spec has no source")
    if not spec.entries:
        raise ValueError("spec has no entry")

    nodes = _node_ports(spec)
    param_ids = {p.id for p in spec.parameters}

    # Feature refs + build the feature dependency graph (feature id → upstream feature ids).
    feat_by_id = {f.id: f for f in spec.features}
    graph: dict[str, list[str]] = {}
    for f in spec.features:
        deps: list[str] = []
        for ref in _feature_inputs(f):
            _check_ref(ref, nodes, where=f"feature {f.id!r}")
            dep_id, _ = parse_ref(ref)
            if dep_id in feat_by_id:
                deps.append(dep_id)
        graph[f.id] = deps

    # Signal refs. Condition operands are numeric (features / sources) — a ref to another
    # signal would silently evaluate as null → False, so it is rejected. Ensemble inputs
    # must be boolean-producing. Ensembles may nest, so their graph is cycle-checked below.
    signal_ids = {s.id for s in spec.signals}
    sig_graph: dict[str, list[str]] = {}
    for sig in spec.signals:
        deps = []
        if isinstance(sig, ConditionSignal):
            for ref in [sig.a] + ([sig.b] if sig.b is not None else []):
                _check_ref(ref, nodes, where=f"signal {sig.id!r}")
                ref_id, _ = parse_ref(ref)
                if ref_id in signal_ids:
                    raise ValueError(
                        f"condition {sig.id!r} operand {ref!r} is a signal; conditions "
                        f"compare numeric series (features / sources)"
                    )
            if isinstance(sig.threshold, str) and sig.threshold not in param_ids:
                raise ValueError(
                    f"condition {sig.id!r} threshold references unknown param {sig.threshold!r}"
                )
        else:  # ensemble
            if not 1 <= len(sig.inputs) <= 4:
                raise ValueError(f"ensemble {sig.id!r} needs 1–4 inputs, got {len(sig.inputs)}")
            for ref in sig.inputs:
                _check_ref(ref, nodes, where=f"ensemble {sig.id!r}")
                if not _is_boolean_ref(ref, nodes):
                    raise ValueError(
                        f"ensemble {sig.id!r} input {ref!r} is not a boolean signal "
                        f"(use a condition/ensemble or a signal port like zScore ':below')"
                    )
                ref_id, _ = parse_ref(ref)
                if ref_id in signal_ids:
                    deps.append(ref_id)
        sig_graph[sig.id] = deps

    # Entry refs must resolve to a boolean-producing node (signal or a feature signal port).
    for e in spec.entries:
        for ref in [e.enter] + ([e.exit] if e.exit else []):
            _check_ref(ref, nodes, where=f"entry {e.id!r}")
            if not _is_boolean_ref(ref, nodes):
                raise ValueError(
                    f"entry {e.id!r} ref {ref!r} is not a boolean signal "
                    f"(use a condition/ensemble or a signal port like zScore ':below')"
                )

    # Parameter references embedded in numeric fields must name a real parameter.
    for f in spec.features:
        for name, val in f.__dict__.items():
            if (
                isinstance(val, str)
                and name not in {"id", "kind", "input", "a", "b", "op", "mode", "field"}
                and val not in param_ids
            ):
                raise ValueError(f"feature {f.id!r} field {name} references unknown param {val!r}")
    for opt in spec.optimizations:
        for op in opt.parameters:
            if op.param not in param_ids:
                raise ValueError(f"optimization sweeps unknown param {op.param!r}")

    # Cycle detection over the feature graph and the signal graph (nested ensembles).
    _check_acyclic(graph, "feature")
    _check_acyclic(sig_graph, "signal")


def _check_size(spec: StrategySpec, max_indicators: int, max_nodes: int) -> None:
    n_features = len(spec.features)
    n_nodes = len(spec.sources) + n_features + len(spec.signals) + len(spec.entries)
    if n_features > max_indicators:
        raise ValueError(f"too many features: {n_features} > max_indicators={max_indicators}")
    if n_nodes > max_nodes:
        raise ValueError(f"spec too large: {n_nodes} nodes > max_nodes={max_nodes}")


def validate_spec(
    spec: StrategySpec,
    *,
    max_indicators: int = DEFAULT_MAX_INDICATORS,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> StrategySpec:
    """Re-run structural validation and enforce the *configured* size caps (the ideation
    admission gate). Parse-time validation only bounds specs by the hard ceilings, so this
    is where ``ideation.max_indicators`` actually bites — above or below the default."""
    _check_refs_and_cycles(spec)
    _check_size(spec, max_indicators, max_nodes)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Search domain
# ─────────────────────────────────────────────────────────────────────────────
def to_param_space(spec: StrategySpec) -> list[ParamSpec]:
    """Build the Optuna search domain (``base.py`` :class:`ParamSpec`) from the spec's
    optimization ranges. A spec with no optimizations is a fixed-parameter family."""
    param_by_id = {p.id: p for p in spec.parameters}
    out: list[ParamSpec] = []
    for opt in spec.optimizations:
        for op in opt.parameters:
            base = param_by_id.get(op.param)
            # NOTE: ``op.type`` defaults to "int", so the inherit-from-parameter fallback
            # is currently dead code (kept for when ``type`` becomes optional).
            kind = op.type or (base.kind if base else "float")  # type: ignore[unreachable]
            out.append(
                ParamSpec(
                    name=op.param,
                    kind=kind,
                    low=int(op.min) if kind == "int" else float(op.min),
                    high=int(op.max) if kind == "int" else float(op.max),
                    step=op.step if op.step is not None else (1 if kind == "int" else None),
                )
            )
    return out
