"""Known-outcome scenarios — the strategy author's own oracle, replayed by the write gate.

A strategy file declares, in a ``scenarios()`` classmethod, deterministic synthetic tapes
built from :class:`Segment` waveforms plus behavioral :class:`Expectation` windows derived
from the thesis (*"on a capitulation-then-recovery tape I must go long around bar 40 and be
flat again by bar 60; on a steady grind I must never trade"*). The gate replays each tape
through ``on_bar`` and rejects the file when the code disagrees with its own declared
behavior — catching inverted conditions, dead logic, and warmup mistakes that a
crash-and-parity smoke test cannot see.

The oracle is honest because the research agent's toolbox cannot execute candidate code
before ``write_strategy``: the expectations are reasoning-derived from the thesis, not
copied from the code's output. Builders contain **no randomness**, so every failure message
is exactly reproducible. Expectations are windows, not per-bar series — off-by-a-bar
indicator arithmetic passes; backwards logic does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from noctis.strategies.base import replay_targets

_NS_PER_MINUTE = 60 * 1_000_000_000
_SPREAD_PCT = 0.002  # high/low bracket relative to close (open == close, repo convention)

MIN_SCENARIOS, MAX_SCENARIOS = 2, 8
MIN_SCENARIO_BARS, MAX_SCENARIO_BARS = 60, 2_000


class ScenarioError(Exception):
    """A scenario declaration or replay violated the known-outcome contract."""


def _one_line(text: object) -> str:
    """Flatten to a single line — the gate subprocess surfaces only the last stderr line."""
    return " ".join(str(text).split())


# ─────────────────────────────────────────────────────────────────────────────
# Segments — deterministic legs of a synthetic close path
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Segment:
    """One deterministic leg of a synthetic close path (use the builder functions)."""

    kind: str  # "flat" | "drift" | "wave" | "gap"
    n: int  # bars emitted (0 for gap)
    a: float = 0.0  # drift/gap: total signed pct move; wave: amplitude pct
    b: float = 0.0  # wave: period in bars

    def __post_init__(self) -> None:
        if self.kind not in ("flat", "drift", "wave", "gap"):
            raise ScenarioError(f"unknown segment kind {self.kind!r}")
        if self.kind == "gap":
            if self.n != 0:
                raise ScenarioError("gap segments emit no bars (n must be 0)")
        elif self.n < 1:
            raise ScenarioError(f"{self.kind} segment needs n >= 1 bars, got {self.n}")
        if self.kind in ("drift", "gap") and self.a <= -1.0:
            raise ScenarioError(f"{self.kind} move must stay above -100%, got {self.a}")
        if self.kind == "wave":
            if not 0.0 < abs(self.a) < 1.0:
                raise ScenarioError(f"wave amplitude must be in (0, 1), got {self.a}")
            if self.b < 2:
                raise ScenarioError(f"wave period must be >= 2 bars, got {self.b}")

    def render(self, start: float) -> tuple[np.ndarray, float]:
        """(closes, end_anchor): closes continue from ``start``; the anchor feeds the next leg."""
        if self.kind == "gap":
            return np.empty(0), start * (1.0 + self.a)
        i = np.arange(1, self.n + 1, dtype="float64")
        if self.kind == "flat":
            closes = np.full(self.n, float(start))
        elif self.kind == "drift":  # geometric path from start to start*(1+a)
            closes = start * (1.0 + self.a) ** (i / self.n)
        else:  # "wave": sinusoid around start
            closes = start * (1.0 + self.a * np.sin(2.0 * math.pi * i / self.b))
        return closes, float(closes[-1])


def flat(n: int) -> Segment:
    """``n`` bars at a constant price — the no-information tape."""
    return Segment("flat", n)


def trend(n: int, pct: float) -> Segment:
    """``n`` bars drifting geometrically to ``(1 + pct)`` of the start (signed)."""
    return Segment("drift", n, pct)


def selloff(n: int, pct: float) -> Segment:
    """``n`` bars of decline totalling ``-|pct|`` — capitulation-shaped."""
    return Segment("drift", n, -abs(pct))


def recovery(n: int, pct: float) -> Segment:
    """``n`` bars of advance totalling ``+|pct|`` — the bounce after a selloff."""
    return Segment("drift", n, abs(pct))


def chop(n: int, amplitude: float, period: int = 8) -> Segment:
    """``n`` bars oscillating ``±amplitude`` around the current level (no net drift)."""
    return Segment("wave", n, amplitude, period)


def vol_spike(n: int, amplitude: float = 0.05) -> Segment:
    """``n`` bars of violent short-period oscillation — a volatility burst."""
    return Segment("wave", n, amplitude, 4)


def gap(pct: float) -> Segment:
    """An instantaneous ``pct`` jump between bars (emits no bars of its own)."""
    return Segment("gap", 0, pct)


# ─────────────────────────────────────────────────────────────────────────────
# Expectations — behavioral windows over the replayed target series
# ─────────────────────────────────────────────────────────────────────────────
class Expectation:
    """A behavioral assertion over the replayed signed (-1/0/+1) target series."""

    is_directional = False  # True when the expectation demands a nonzero (long OR short) entry

    @property
    def last_index(self) -> int:
        """The largest bar index the declaration references (bounds-checked vs the tape)."""
        raise NotImplementedError

    def check(self, targets: list[int]) -> str | None:
        """None on pass; a single-line failure message naming the violating bar."""
        raise NotImplementedError


@dataclass(frozen=True)
class FlatUntil(Expectation):
    bar: int

    def __post_init__(self) -> None:
        if self.bar < 1:
            raise ScenarioError(f"flat_until needs bar >= 1, got {self.bar}")

    @property
    def last_index(self) -> int:
        return self.bar - 1

    def check(self, targets: list[int]) -> str | None:
        for j in range(min(self.bar, len(targets))):
            if targets[j] != 0:
                return f"flat_until({self.bar}) violated: non-flat at bar {j}"
        return None


@dataclass(frozen=True)
class LongWithin(Expectation):
    lo: int
    hi: int
    is_directional = True

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= self.hi:
            raise ScenarioError(f"long_within needs 0 <= lo <= hi, got ({self.lo}, {self.hi})")

    @property
    def last_index(self) -> int:
        return self.hi

    def check(self, targets: list[int]) -> str | None:
        if any(targets[j] == 1 for j in range(self.lo, min(self.hi + 1, len(targets)))):
            return None
        return (
            f"long_within({self.lo},{self.hi}) violated: never long in bars [{self.lo},{self.hi}]"
        )


@dataclass(frozen=True)
class HoldsLongThrough(Expectation):
    lo: int
    hi: int
    is_directional = True

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= self.hi:
            raise ScenarioError(
                f"holds_long_through needs 0 <= lo <= hi, got ({self.lo}, {self.hi})"
            )

    @property
    def last_index(self) -> int:
        return self.hi

    def check(self, targets: list[int]) -> str | None:
        for j in range(self.lo, min(self.hi + 1, len(targets))):
            if targets[j] != 1:
                return f"holds_long_through({self.lo},{self.hi}) violated: flat at bar {j}"
        return None


@dataclass(frozen=True)
class ShortWithin(Expectation):
    lo: int
    hi: int
    is_directional = True

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= self.hi:
            raise ScenarioError(f"short_within needs 0 <= lo <= hi, got ({self.lo}, {self.hi})")

    @property
    def last_index(self) -> int:
        return self.hi

    def check(self, targets: list[int]) -> str | None:
        if any(targets[j] == -1 for j in range(self.lo, min(self.hi + 1, len(targets)))):
            return None
        return (
            f"short_within({self.lo},{self.hi}) violated: never short in bars [{self.lo},{self.hi}]"
        )


@dataclass(frozen=True)
class HoldsShortThrough(Expectation):
    lo: int
    hi: int
    is_directional = True

    def __post_init__(self) -> None:
        if not 0 <= self.lo <= self.hi:
            raise ScenarioError(
                f"holds_short_through needs 0 <= lo <= hi, got ({self.lo}, {self.hi})"
            )

    @property
    def last_index(self) -> int:
        return self.hi

    def check(self, targets: list[int]) -> str | None:
        for j in range(self.lo, min(self.hi + 1, len(targets))):
            if targets[j] != -1:
                return f"holds_short_through({self.lo},{self.hi}) violated: not short at bar {j}"
        return None


@dataclass(frozen=True)
class FlatBy(Expectation):
    bar: int

    def __post_init__(self) -> None:
        if self.bar < 0:
            raise ScenarioError(f"flat_by needs bar >= 0, got {self.bar}")

    @property
    def last_index(self) -> int:
        return self.bar

    def check(self, targets: list[int]) -> str | None:
        for j in range(self.bar, len(targets)):
            if targets[j] != 0:
                return f"flat_by({self.bar}) violated: still in a position at bar {j}"
        return None


@dataclass(frozen=True)
class AlwaysFlat(Expectation):
    @property
    def last_index(self) -> int:
        return 0

    def check(self, targets: list[int]) -> str | None:
        for j, t in enumerate(targets):
            if t != 0:
                return f"always_flat violated: took a position at bar {j}"
        return None


def flat_until(bar: int) -> FlatUntil:
    """Flat on every bar strictly before ``bar`` (setup/warmup must not trade)."""
    return FlatUntil(bar)


def long_within(lo: int, hi: int) -> LongWithin:
    """Long on at least one bar in ``[lo, hi]`` — the thesis demands an entry here."""
    return LongWithin(lo, hi)


def holds_long_through(lo: int, hi: int) -> HoldsLongThrough:
    """Long on every bar in ``[lo, hi]`` — the position must be held, not flickered."""
    return HoldsLongThrough(lo, hi)


def short_within(lo: int, hi: int) -> ShortWithin:
    """Short on at least one bar in ``[lo, hi]`` — the thesis demands a short entry here."""
    return ShortWithin(lo, hi)


def holds_short_through(lo: int, hi: int) -> HoldsShortThrough:
    """Short on every bar in ``[lo, hi]`` — the short must be held, not flickered."""
    return HoldsShortThrough(lo, hi)


def flat_by(bar: int) -> FlatBy:
    """Flat on every bar from ``bar`` to the end — the exit must have fired."""
    return FlatBy(bar)


def always_flat() -> AlwaysFlat:
    """Flat on every bar — the mandatory no-trade tape."""
    return AlwaysFlat()


_HINTS: dict[type, str] = {
    FlatUntil: "early entry during setup/warmup; check warmup guards",
    LongWithin: "dead or late logic: no long entry where the thesis demands one; "
    "check warmup length and the entry condition's direction",
    HoldsLongThrough: "long position not held; check exit/flip conditions inside the window",
    ShortWithin: "dead or late logic: no short entry where the thesis demands one; "
    "check warmup length and the short condition's direction (target must reach -1)",
    HoldsShortThrough: "short not held; check exit/flip conditions inside the window",
    FlatBy: "exit never fires; check the exit condition",
    AlwaysFlat: "code enters where the thesis says stay flat; check condition direction/thresholds",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario — a named tape plus its declared behavior
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Scenario:
    """A deterministic tape (DSL segments only) plus the behavior the thesis demands on it."""

    name: str
    segments: tuple[Segment, ...]
    expect: tuple[Expectation, ...]
    params: dict | None = None  # optional Params overrides for this replay (defaults rule)
    start: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "expect", tuple(self.expect))

    @property
    def n_bars(self) -> int:
        return sum(seg.n for seg in self.segments)

    def frame(self) -> pd.DataFrame:
        """Render the OHLCV tape: legs concatenate continuously from ``start``."""
        parts: list[np.ndarray] = []
        anchor = self.start
        for seg in self.segments:
            closes, anchor = seg.render(anchor)
            if closes.size:
                parts.append(closes)
        close = np.concatenate(parts) if parts else np.empty(0)
        n = close.size
        return pd.DataFrame(
            {
                "ts_event": np.arange(n, dtype="int64") * _NS_PER_MINUTE,
                "open": close,
                "high": close * (1.0 + _SPREAD_PCT),
                "low": close * (1.0 - _SPREAD_PCT),
                "close": close,
                "volume": np.full(n, 1000.0),
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# Observed behavior — execution feedback appended to every failure message
# ─────────────────────────────────────────────────────────────────────────────
def _position_spans(targets: list[int]) -> list[tuple[int, int, int]]:
    """Maximal runs of a constant nonzero target as ``(lo, hi, direction)`` (inclusive)."""
    spans: list[tuple[int, int, int]] = []
    start = 0
    for j in range(1, len(targets) + 1):
        prev = targets[j - 1]
        if j == len(targets) or targets[j] != prev:
            if prev != 0:
                spans.append((start, j - 1, prev))
            start = j
    return spans


def _render_spans(spans: list[tuple[int, int, int]], direction: int) -> str:
    """``"[lo–hi], [lo–hi]"`` for the runs in ``direction``, or ``"none"`` when there are none."""
    rendered = [f"[{lo}–{hi}]" for lo, hi, d in spans if d == direction]
    return ", ".join(rendered) if rendered else "none"


def observed_behavior(targets: list[int]) -> str:
    """A single-line summary of what the code *actually did* on the replayed tape.

    The oracle's failure messages name the window a strategy missed; this names the behavior
    it produced instead — the first nonzero-target bar and its direction, plus the long/short
    position spans it held — so the coder reacts to execution feedback rather than guessing.
    Single-line by construction, so it survives the write gate's last-stderr-line boundary.
    """
    first = next((j for j, t in enumerate(targets) if t != 0), None)
    if first is None:
        return f"observed: never took a position across all {len(targets)} bars"
    direction = "long" if targets[first] == 1 else "short"
    spans = _position_spans(targets)
    return (
        f"observed: first went {direction} at bar {first}; "
        f"long spans {_render_spans(spans, 1)}; short spans {_render_spans(spans, -1)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 invariants — structural honesty checks replayed over every tape
# ─────────────────────────────────────────────────────────────────────────────
# One extensible step in the shared write-gate funnel: each invariant is a pure check over a
# single replayed tape, returning a single-line failure message or None. Warmup honesty is the
# first (and, today, only) member; the rest of the Tier-1 suite lands here later behind the same
# seam, so both validator runners inherit every check with no drift.
def _check_warmup_honesty(cls, scenario: Scenario, params, targets: list[int]) -> str | None:
    """The declared warmup must not lie: no nonzero target before it.

    ``warmup_bars(params)`` is the only model-owned number in the oracle — the author's promise
    to stay flat through indicator warmup. Default 0 is undeclared and exempt, so strategies
    outside the library are untouched. A nonzero target before the declared warmup is a lie the
    gate rejects, naming the offending bar and the declared warmup so the fix is unambiguous.
    """
    try:
        warmup = int(cls.warmup_bars(params))
    except Exception as exc:  # noqa: BLE001 — a broken declaration is a contract failure
        return (
            f"scenario {scenario.name!r}: warmup_bars() raised "
            f"{type(exc).__name__}: {_one_line(exc)}"
        )
    if warmup <= 0:
        return None
    for j in range(min(warmup, len(targets))):
        if targets[j] != 0:
            direction = "long" if targets[j] == 1 else "short"
            return (
                f"scenario {scenario.name!r}: warmup dishonest — took a {direction} position "
                f"at bar {j}, before the declared warmup_bars={warmup} "
                f"(promised flat through bar {warmup - 1}); raise warmup_bars or delay the entry"
            )
    return None


# Ordered — warmup honesty is the first member; later Tier-1 checks (a separate story) append here.
_INVARIANTS = (_check_warmup_honesty,)


def check_invariants(cls, scenario: Scenario, params, targets: list[int]) -> str | None:
    """Run the Tier-1 invariant suite over one replayed tape; None on pass, else one line."""
    for invariant in _INVARIANTS:
        msg = invariant(cls, scenario, params, targets)
        if msg:
            return msg
    return None


def run_scenario(cls, scenario: Scenario) -> str | None:
    """Replay one scenario; None on pass, else a single-line failure message."""
    try:
        params = cls.params_cls(**(scenario.params or {}))
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the author
        return f"scenario {scenario.name!r}: params override rejected: {_one_line(exc)}"
    try:
        targets = replay_targets(cls(params), scenario.frame())
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the author
        return f"scenario {scenario.name!r}: replay raised {type(exc).__name__}: {_one_line(exc)}"
    for j, t in enumerate(targets):
        if t not in (-1, 0, 1):
            return (
                f"scenario {scenario.name!r}: target {t} at bar {j} is outside "
                f"long/short/flat {{-1,0,1}}"
            )
    invariant_msg = check_invariants(cls, scenario, params, targets)
    if invariant_msg:
        return invariant_msg
    for exp in scenario.expect:
        msg = exp.check(targets)
        if msg:
            hint = _HINTS.get(type(exp), "")
            return f"scenario {scenario.name!r}: {msg} — {observed_behavior(targets)} — {hint}"
    return None


def check_scenario_contract(cls, *, require: bool = True) -> None:
    """Enforce the known-outcome contract on ``cls`` (declaration shape, then replay).

    With ``require=False`` (mechanical rewrites of legacy files) an empty declaration
    passes silently, but anything declared is still fully evaluated.
    """
    try:
        declared = cls.scenarios()
    except Exception as exc:  # noqa: BLE001 — a broken declaration is a contract failure
        raise ScenarioError(f"scenarios() raised {type(exc).__name__}: {_one_line(exc)}") from exc
    if not declared:
        if require:
            raise ScenarioError(
                "strategy must declare known-outcome scenarios: a scenarios() classmethod "
                f"returning {MIN_SCENARIOS}-{MAX_SCENARIOS} Scenario objects built from the "
                "noctis.strategies.scenarios DSL"
            )
        return
    if not isinstance(declared, (list, tuple)) or not all(
        isinstance(s, Scenario) for s in declared
    ):
        raise ScenarioError("scenarios() must return a list of Scenario objects")
    if not MIN_SCENARIOS <= len(declared) <= MAX_SCENARIOS:
        raise ScenarioError(f"want {MIN_SCENARIOS}-{MAX_SCENARIOS} scenarios, got {len(declared)}")
    names = [s.name for s in declared]
    if len(set(names)) != len(names):
        raise ScenarioError(f"scenario names must be unique, got {names}")

    has_directional = has_negative = False
    for sc in declared:
        if not sc.segments or not all(isinstance(seg, Segment) for seg in sc.segments):
            raise ScenarioError(
                f"scenario {sc.name!r}: segments must be built from the DSL "
                "(flat/trend/selloff/recovery/chop/vol_spike/gap)"
            )
        if not sc.expect or not all(isinstance(exp, Expectation) for exp in sc.expect):
            raise ScenarioError(
                f"scenario {sc.name!r}: expect must be a non-empty list of expectations "
                "(flat_until/long_within/holds_long_through/short_within/holds_short_through/"
                "flat_by/always_flat)"
            )
        if not MIN_SCENARIO_BARS <= sc.n_bars <= MAX_SCENARIO_BARS:
            raise ScenarioError(
                f"scenario {sc.name!r}: {sc.n_bars} bars is outside "
                f"[{MIN_SCENARIO_BARS}, {MAX_SCENARIO_BARS}]"
            )
        for exp in sc.expect:
            if exp.last_index >= sc.n_bars:
                raise ScenarioError(
                    f"scenario {sc.name!r}: expectation window exceeds the {sc.n_bars}-bar tape"
                )
        has_directional = has_directional or any(exp.is_directional for exp in sc.expect)
        has_negative = has_negative or any(isinstance(exp, AlwaysFlat) for exp in sc.expect)
    if not has_directional:
        raise ScenarioError(
            "at least one scenario must demand a directional entry "
            "(long_within/holds_long_through/short_within/holds_short_through)"
        )
    if not has_negative:
        raise ScenarioError("at least one scenario must be a no-trade tape (always_flat())")

    for sc in declared:
        msg = run_scenario(cls, sc)
        if msg:
            raise ScenarioError(msg)
