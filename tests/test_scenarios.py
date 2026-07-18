"""The known-outcome scenario DSL — builders, expectations, and the contract checker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from noctis.strategies import indicators as ind
from noctis.strategies.base import ParamSpec, TraderStrategy
from noctis.strategies.scenarios import (
    Scenario,
    ScenarioError,
    Segment,
    always_flat,
    check_scenario_contract,
    chop,
    flat,
    flat_by,
    flat_until,
    gap,
    holds_long_through,
    holds_short_through,
    long_within,
    recovery,
    run_scenario,
    selloff,
    short_within,
    trend,
    vol_spike,
)


# ── segment builders ──────────────────────────────────────────────────────────────────────
def _closes(*segments, start=100.0):
    return Scenario("t", segments, (always_flat(),), start=start).frame()["close"].tolist()


def test_segments_concatenate_continuously():
    closes = _closes(flat(10), trend(10, 0.10))
    assert closes[9] == 100.0
    assert closes[10] == pytest.approx(100.0 * 1.10 ** (1 / 10))
    assert closes[19] == pytest.approx(110.0)


def test_gap_shifts_anchor_without_emitting_bars():
    closes = _closes(flat(5), gap(0.10), flat(5))
    assert len(closes) == 10
    assert closes[4] == 100.0
    assert closes[5] == pytest.approx(110.0)


def test_selloff_recovery_signs_and_monotonicity():
    down = _closes(selloff(20, 0.15))
    up = _closes(recovery(20, 0.15))
    assert down == sorted(down, reverse=True) and down[-1] == pytest.approx(85.0)
    assert up == sorted(up) and up[-1] == pytest.approx(115.0)
    # selloff normalizes the sign: a positive pct still means decline.
    assert _closes(selloff(20, -0.15)) == down


def test_chop_and_vol_spike_oscillate_around_start():
    for seg in (chop(64, 0.03), vol_spike(64, 0.05)):
        closes = _closes(seg)
        assert max(closes) > 100.0 > min(closes)
        assert max(closes) <= 100.0 * 1.06 and min(closes) >= 100.0 * 0.94


def test_frames_are_deterministic_and_ohlcv_shaped():
    sc = Scenario("t", (flat(10), trend(30, 0.2), chop(20, 0.02)), (always_flat(),))
    a, b = sc.frame(), sc.frame()
    assert a.equals(b)
    assert sc.n_bars == len(a) == 60
    assert (a["open"] == a["close"]).all()
    assert (a["high"] > a["close"]).all() and (a["low"] < a["close"]).all()
    assert a["ts_event"].is_monotonic_increasing
    assert (a["volume"] == 1000.0).all()


@pytest.mark.parametrize(
    "bad",
    [
        lambda: flat(0),
        lambda: Segment("gap", 3, 0.1),
        lambda: trend(10, -1.5),
        lambda: chop(10, 0.0),
        lambda: chop(10, 1.5),
        lambda: Segment("wave", 10, 0.05, 1),
        lambda: Segment("mystery", 10),
    ],
)
def test_invalid_segments_rejected(bad):
    with pytest.raises(ScenarioError):
        bad()


# ── expectation primitives ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("exp", "passing", "failing", "violating_bar"),
    [
        (flat_until(3), [0, 0, 0, 1, 1], [0, 1, 0, 0, 0], 1),
        (long_within(2, 4), [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], None),
        (holds_long_through(1, 3), [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], 2),
        (flat_by(3), [1, 1, 1, 0, 0], [0, 0, 0, 0, 1], 4),
        (always_flat(), [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], 2),
    ],
)
def test_expectation_pass_fail_and_message_names_the_bar(exp, passing, failing, violating_bar):
    assert exp.check(passing) is None
    msg = exp.check(failing)
    assert msg is not None
    if violating_bar is not None:
        assert f"bar {violating_bar}" in msg
    else:
        assert "[2,4]" in msg  # long_within names the empty window instead


@pytest.mark.parametrize(
    "bad", [lambda: flat_until(0), lambda: long_within(4, 2), lambda: flat_by(-1)]
)
def test_invalid_expectations_rejected(bad):
    with pytest.raises(ScenarioError):
        bad()


# ── the contract checker, run against a real strategy ─────────────────────────────────────
class _Above(TraderStrategy):
    """Long when the close is above its own SMA — the minimal thesis-bearing probe."""

    name = "above"

    @dataclass(frozen=True)
    class Params:
        lookback: int = 10

    params_cls = Params

    def on_start(self, ctx):
        self._closes = deque(maxlen=self.params.lookback)

    def on_bar(self, ctx, bar):
        self._closes.append(bar.close)
        mean = ind.sma(self._closes, self.params.lookback)
        ctx.set_target(0 if mean is None else int(bar.close > mean))

    @classmethod
    def param_space(cls):
        return [ParamSpec("lookback", "int", 5, 40, 1)]


POSITIVE = Scenario(
    "rally_then_fade",
    segments=[flat(15), trend(30, 0.10), selloff(20, 0.15)],
    expect=[flat_until(10), long_within(15, 25), flat_by(55)],
)
NEGATIVE = Scenario(
    "steady_decline_stays_flat",
    segments=[flat(15), selloff(45, 0.20)],
    expect=[always_flat()],
)


def _with_scenarios(*scens):
    class _Probe(_Above):
        @classmethod
        def scenarios(cls):
            return list(scens)

    return _Probe


def test_contract_passes_for_a_correct_strategy():
    check_scenario_contract(_with_scenarios(POSITIVE, NEGATIVE))


def test_base_class_default_is_empty_and_only_blocks_when_required():
    check_scenario_contract(_Above, require=False)
    with pytest.raises(ScenarioError, match="must declare"):
        check_scenario_contract(_Above, require=True)


@pytest.mark.parametrize(
    ("cls_factory", "match"),
    [
        (lambda: _with_scenarios(POSITIVE), "want 2-8"),
        (lambda: _with_scenarios(*([POSITIVE] * 8), NEGATIVE), "want 2-8"),
        (lambda: _with_scenarios(POSITIVE, POSITIVE), "unique"),
        (
            lambda: _with_scenarios(POSITIVE, Scenario("neg", [flat(60)], [flat_until(30)])),
            "no-trade tape",
        ),
        (
            lambda: _with_scenarios(Scenario("a", [flat(60)], [flat_until(30)]), NEGATIVE),
            "directional entry",
        ),
        (
            lambda: _with_scenarios(Scenario("a", [flat(60)], [long_within(50, 99)]), NEGATIVE),
            "window exceeds",
        ),
        (
            lambda: _with_scenarios(Scenario("a", [flat(59)], [long_within(0, 10)]), NEGATIVE),
            "outside",
        ),
        (
            lambda: _with_scenarios(
                Scenario("a", [list(range(60))], [long_within(0, 10)]), NEGATIVE
            ),
            "DSL",
        ),
        (lambda: _with_scenarios(Scenario("a", [flat(60)], []), NEGATIVE), "non-empty"),
    ],
)
def test_contract_rejects_malformed_declarations(cls_factory, match):
    with pytest.raises(ScenarioError, match=match):
        check_scenario_contract(cls_factory())


def test_contract_rejects_non_scenario_returns_and_raising_scenarios():
    class _Wrong(_Above):
        @classmethod
        def scenarios(cls):
            return ["not a scenario", 3]

    class _Boom(_Above):
        @classmethod
        def scenarios(cls):
            raise RuntimeError("kaput")

    with pytest.raises(ScenarioError, match="list of Scenario"):
        check_scenario_contract(_Wrong)
    with pytest.raises(ScenarioError, match="scenarios.. raised"):
        check_scenario_contract(_Boom)


def test_dead_logic_fails_the_positive_scenario():
    class _Dead(_with_scenarios(POSITIVE, NEGATIVE)):
        def on_bar(self, ctx, bar):
            ctx.set_target(0)

    with pytest.raises(ScenarioError, match=r"rally_then_fade.*long_within"):
        check_scenario_contract(_Dead)


def test_inverted_logic_fails_the_no_trade_tape():
    class _Inverted(_with_scenarios(NEGATIVE, POSITIVE)):
        def on_bar(self, ctx, bar):
            self._closes.append(bar.close)
            mean = ind.sma(self._closes, self.params.lookback)
            ctx.set_target(0 if mean is None else int(bar.close < mean))

    with pytest.raises(ScenarioError, match=r"steady_decline.*always_flat.*took a position at bar"):
        check_scenario_contract(_Inverted)


def test_short_targets_are_accepted():
    # Signed targets are first-class: a long/short strategy (long above the SMA, short below
    # it) that declares a short expectation PASSES the contract — shorts are no longer rejected.
    down = Scenario(
        "decline_goes_short",
        segments=[flat(15), selloff(45, 0.20)],
        expect=[flat_until(10), short_within(20, 40)],
    )
    up = Scenario(
        "rally_goes_long",
        segments=[flat(15), trend(45, 0.15)],
        expect=[flat_until(10), long_within(20, 40)],
    )
    no_trade = Scenario("flat_stays_flat", segments=[flat(60)], expect=[always_flat()])

    class _LongShort(_Above):
        @classmethod
        def scenarios(cls):
            return [down, up, no_trade]

        def on_bar(self, ctx, bar):
            self._closes.append(bar.close)
            mean = ind.sma(self._closes, self.params.lookback)
            if mean is None or bar.close == mean:
                ctx.set_target(0)
            else:
                ctx.set_target(1 if bar.close > mean else -1)

    check_scenario_contract(_LongShort)


def test_short_only_scenario_satisfies_the_directional_requirement():
    # A short entry alone satisfies "at least one directional entry" — no long tape needed.
    down = Scenario(
        "short_only",
        segments=[flat(15), selloff(45, 0.20)],
        expect=[flat_until(10), holds_short_through(30, 40)],
    )
    no_trade = Scenario("flat_stays_flat", segments=[flat(60)], expect=[always_flat()])

    class _ShortOnly(_Above):
        @classmethod
        def scenarios(cls):
            return [down, no_trade]

        def on_bar(self, ctx, bar):
            self._closes.append(bar.close)
            mean = ind.sma(self._closes, self.params.lookback)
            ctx.set_target(-1 if (mean is not None and bar.close < mean) else 0)

    check_scenario_contract(_ShortOnly)


def test_per_scenario_params_override_is_honored():
    # With lookback=20 the SMA only exists from bar 19, so entry lands exactly there;
    # the default lookback=10 would enter at bar 15 and violate flat_until(19).
    override = Scenario(
        "slow_warmup",
        segments=[flat(15), trend(45, 0.10)],
        expect=[flat_until(19), long_within(19, 30)],
        params={"lookback": 20},
    )
    check_scenario_contract(_with_scenarios(override, NEGATIVE))
    no_override = Scenario(
        "slow_warmup",
        segments=[flat(15), trend(45, 0.10)],
        expect=[flat_until(19), long_within(19, 30)],
    )
    with pytest.raises(ScenarioError, match="flat_until"):
        check_scenario_contract(_with_scenarios(no_override, NEGATIVE))
    assert run_scenario(_Above, override) is None


def test_bad_params_override_is_reported_not_raised():
    broken = Scenario(
        "bad_override",
        segments=[flat(60)],
        expect=[always_flat()],
        params={"nope": 3},
    )
    msg = run_scenario(_Above, broken)
    assert msg is not None and "params override rejected" in msg
