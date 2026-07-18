"""Pipeline ↔ promotion parity: the gates judge real evaluate() output.

Every other promotion test hand-builds Scorecards; these feed decide()/consider() cards the
pipeline actually produced, proving evaluate() populates — in the units the gates compare —
every field the gates read (activity from real exposure, both holdout metrics, the election
metric). A pipeline aggregation bug can no longer pass the promotion suite unnoticed.
"""

from __future__ import annotations

import pytest

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.champions import ChampionRegistry, PromotionRules, decide

from ._data_helpers import make_ohlcv, price_series


def _lenient(**overrides) -> PromotionRules:
    """Every gate wide open (the fixture card's holdout Sharpe is genuinely negative —
    a fact only a real card can inject); each test then re-arms exactly one gate."""
    base = dict(
        champion_count=3,
        max_gap=1e9,
        min_test_metric=-1e9,
        min_holdout_metric=-1e9,
        min_symbol_holdout_metric=-1e9,
    )
    return PromotionRules(**(base | overrides))


@pytest.fixture(scope="module")
def real_card():
    panel = {
        "AAA": make_ohlcv(price_series(n=250, seed=1)),
        "BBB": make_ohlcv(price_series(n=250, seed=2)),
    }
    held = {"HH1": make_ohlcv(price_series(n=250, seed=8))}
    cand = Candidate("donchian_breakout", {"channel": 15})
    cfg = PipelineConfig.auto(250, metric="sharpe", prefilter_min_score=None)
    return evaluate(cand, panel, config=cfg, symbol_holdout=held)


def test_pipeline_populates_every_field_the_gates_read(real_card):
    assert real_card.stage == "validated"
    assert set(real_card.symbols) == {"AAA", "BBB"}
    assert all(ss.splits for ss in real_card.symbols.values())
    assert real_card.holdout_metric is not None  # forward-holdout gate is live
    assert real_card.symbol_holdout_metric is not None  # symbol-holdout gate is live
    assert 0.0 < real_card.test_activity <= 1.0  # activity floor reads real exposure
    assert real_card.metric_name == "sharpe"


def test_activity_floor_reads_real_pooled_exposure(real_card):
    activity = real_card.test_activity
    below = _lenient(min_test_activity=activity / 2)
    above = _lenient(min_test_activity=min(activity * 1.01, 1.0))
    assert decide(real_card, [], below).promote is True
    d = decide(real_card, [], above)
    if activity < 1.0:  # a floor strictly above the real value must trip the gate
        assert d.promote is False
        assert "activity floor" in d.rationale


def test_holdout_gates_read_real_pipeline_metrics(real_card):
    lenient = _lenient()
    assert decide(real_card, [], lenient).promote is True

    forward_bar = _lenient(min_holdout_metric=real_card.holdout_metric + 1.0)
    d = decide(real_card, [], forward_bar)
    assert d.promote is False
    assert "forward-holdout gate" in d.rationale

    symbol_bar = _lenient(min_symbol_holdout_metric=real_card.symbol_holdout_metric + 1.0)
    d = decide(real_card, [], symbol_bar)
    assert d.promote is False
    assert "symbol-holdout gate" in d.rationale


def test_promoted_real_card_binds_its_fit_panel(tmp_path, real_card):
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    d = reg.consider(real_card, _lenient())
    assert d.promote
    entry = reg.list()[0]
    assert entry.fit_symbols == ["AAA", "BBB"]  # fit panel, not the held-out symbol
    assert entry.live_symbols == ["AAA", "BBB"]
