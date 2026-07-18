"""Champion registry persistence and promotion rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from noctis.backtest.scorecard import Metrics, Scorecard, SplitScore, SymbolScore
from noctis.champions import ChampionRegistry, PromotionRules, decide


def _m(sharpe: float, exposure: float = 0.0) -> Metrics:
    return Metrics(
        total_return=0.0,
        sharpe=sharpe,
        sortino=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        turnover=0.0,
        exposure=exposure,
    )


def make_scorecard(family: str, test_metric: float, train_metric: float, **params) -> Scorecard:
    """Build a validated panel-of-one Scorecard with the given Sharpe train/test aggregates."""
    return Scorecard(
        family=family,
        params=params,
        metric_name="sharpe",
        stage="validated",
        symbols={"FIT": SymbolScore(splits=[SplitScore(0, _m(train_metric), _m(test_metric))])},
    )


def make_panel_scorecard(family: str, per_symbol_test: dict[str, float], **params) -> Scorecard:
    """Build a validated panel Scorecard with the given per-symbol test Sharpes."""
    return Scorecard(
        family=family,
        params=params,
        metric_name="sharpe",
        stage="validated",
        symbols={
            sym: SymbolScore(splits=[SplitScore(0, _m(v), _m(v))])
            for sym, v in per_symbol_test.items()
        },
    )


RULES = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)


# --- pure decide() -----------------------------------------------------------------------


def test_empty_registry_promotes_decent_challenger():
    d = decide(make_scorecard("sma_crossover", test_metric=1.5, train_metric=1.8), [], RULES)
    assert d.promote is True
    assert "free slot" in d.rationale


def test_free_slot_below_bar_rejected():
    d = decide(make_scorecard("rsi_meanrev", test_metric=-0.2, train_metric=0.1), [], RULES)
    assert d.promote is False
    assert "below minimum bar" in d.rationale


def test_gap_guard_rejects_overfit():
    # Excellent test metric but train−test gap of 2.0 > max_gap 1.0.
    d = decide(make_scorecard("donchian_breakout", test_metric=3.0, train_metric=5.0), [], RULES)
    assert d.promote is False
    assert "overfit" in d.rationale


def test_reverse_gap_guard_rejects_degenerate_test_above_train():
    """Part B degeneracy gate: a test metric that wildly EXCEEDS train (large negative gap) is a
    noise signal — the mirror of the overfit guard. Off by default; on via max_reverse_gap."""
    rules = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0, max_reverse_gap=1.0)
    d = decide(make_scorecard("rsi_meanrev", test_metric=5.0, train_metric=1.0), [], rules)
    assert d.promote is False and "degenerate" in d.rationale
    # A modest out-of-sample-better result inside the band still promotes.
    ok = decide(make_scorecard("rsi_meanrev", test_metric=1.5, train_metric=1.0), [], rules)
    assert ok.promote is True


def test_magnitude_cap_rejects_implausible_metric():
    """Part B backstop: an implausibly large test metric (a residual degeneracy) is rejected even
    with a benign gap — the exact shape that froze the live registry (sortino in the tens of k)."""
    rules = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0, max_test_metric=50.0)
    d = decide(make_scorecard("rsi_meanrev", test_metric=74601.0, train_metric=74601.5), [], rules)
    assert d.promote is False and "sane ceiling" in d.rationale


def test_degeneracy_gates_off_by_default():
    """Both knobs default 0.0 (off), so direct-construction tests are unaffected; from_settings
    wires the active config values in production."""
    d = decide(make_scorecard("rsi_meanrev", test_metric=3.0, train_metric=1.0), [], RULES)
    assert d.promote is True  # reverse gap −2.0 ignored while max_reverse_gap == 0


def test_from_settings_maps_degeneracy_knobs():
    from noctis.config.settings import Settings

    rules = PromotionRules.from_settings(
        Settings(promotion={"max_reverse_gap": 1.5, "max_test_metric": 40.0})
    )
    assert rules.max_reverse_gap == 1.5 and rules.max_test_metric == 40.0


def test_holdout_gate_rejects_strong_test_but_weak_holdout():
    # Clean test metric and gap, but negative on the untouched forward slice → rejected.
    sc = make_scorecard("sma_crossover", test_metric=1.5, train_metric=1.6)
    sc.holdout_metric = -0.5
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, 0.0))
    assert d.promote is False
    assert "forward-holdout gate" in d.rationale


def test_holdout_gate_passes_when_holdout_clears_bar():
    sc = make_scorecard("sma_crossover", test_metric=1.5, train_metric=1.6)
    sc.holdout_metric = 0.3
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, 0.0))
    assert d.promote is True


def test_holdout_gate_inert_without_holdout_metric():
    # No holdout reserved (holdout_metric None) → the gate does not apply, even with a high bar.
    rules = PromotionRules(3, 1.0, 0.0, 5.0)
    d = decide(make_scorecard("x", test_metric=1.5, train_metric=1.6), [], rules)
    assert d.promote is True


def test_panel_scorecard_accepted_as_validated():
    # Widened validated-check: a panel scorecard has symbols but no top-level splits.
    sc = make_panel_scorecard("x", {"AAPL": 1.2, "MSFT": 0.8})
    d = decide(sc, [], RULES)
    assert d.promote is True
    assert sc.avg_test_metric == pytest.approx(1.0)  # panel mean drives the decision


def test_symbol_holdout_gate_rejects_weak_holdout():
    # Strong on the fit panel, negative on the never-tuned symbols → rejected.
    sc = make_panel_scorecard("x", {"AAPL": 1.5, "MSFT": 1.4})
    sc.symbol_holdout_metric = -0.2
    rules = PromotionRules(3, 1.0, 0.0, min_symbol_holdout_metric=0.0)
    d = decide(sc, [], rules)
    assert d.promote is False
    assert "symbol-holdout gate" in d.rationale


def test_symbol_holdout_gate_passes_when_clearing_bar():
    sc = make_panel_scorecard("x", {"AAPL": 1.5, "MSFT": 1.4})
    sc.symbol_holdout_metric = 0.4
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_symbol_holdout_metric=0.3))
    assert d.promote is True


def test_symbol_holdout_gate_inert_without_metric():
    # No symbol holdout reserved (metric None) → gate does not apply, even with a high bar.
    sc = make_panel_scorecard("x", {"AAPL": 1.5, "MSFT": 1.4})
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_symbol_holdout_metric=5.0))
    assert d.promote is True


def test_symbol_consistency_gate_optional():
    # 1 of 3 fit symbols positive = 0.33 breadth; bar 0.6 rejects, 0.0 (default) is off.
    sc = make_panel_scorecard("x", {"A": 2.0, "B": -0.1, "C": -0.2})
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_symbol_consistency=0.6))
    assert d.promote is False
    assert "symbol-consistency gate" in d.rationale
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_symbol_consistency=0.0))
    assert d.promote is True


def _rarely_trading_scorecard(n_splits: int = 100, active: int = 1) -> Scorecard:
    """A challenger that trades in ``active`` of ``n_splits`` test splits — the noise-champion
    shape: a tiny positive average metric built on a handful of lucky windows."""
    splits = [
        SplitScore(i, _m(0.0), _m(3.0, exposure=0.5) if i < active else _m(0.0))
        for i in range(n_splits)
    ]
    return Scorecard(
        family="lucky",
        params={},
        metric_name="sharpe",
        symbols={"FIT": SymbolScore(splits=splits)},
    )


def test_activity_floor_rejects_rarely_trading_challenger():
    # Positive average metric, but only 1% of test splits ever traded → noise, rejected.
    sc = _rarely_trading_scorecard(n_splits=100, active=1)
    assert sc.avg_test_metric > 0.0
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_test_activity=0.05))
    assert d.promote is False
    assert "activity floor" in d.rationale


def test_activity_floor_passes_active_strategy():
    sc = _rarely_trading_scorecard(n_splits=100, active=40)
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_test_activity=0.05))
    assert d.promote is True


def test_activity_floor_off_by_default():
    # min_test_activity 0.0 (the default) leaves the gate off — pre-floor behavior.
    sc = _rarely_trading_scorecard(n_splits=100, active=1)
    d = decide(sc, [], RULES)
    assert d.promote is True


def test_activity_floor_pools_panel_splits():
    # Panel path: both symbols flat in every split → activity 0.0 → rejected.
    sc = make_panel_scorecard("x", {"AAPL": 1.5, "MSFT": 1.4})
    assert sc.test_activity == 0.0
    d = decide(sc, [], PromotionRules(3, 1.0, 0.0, min_test_activity=0.05))
    assert d.promote is False
    assert "activity floor" in d.rationale


def test_stale_champion_displaced_on_metric_change():
    # Champions scored on sharpe, the regime moved to total_return: their stored numbers
    # are in different units, so a gate-clearing challenger displaces the stale one.
    champs = [
        make_scorecard("old_a", test_metric=9.0, train_metric=9.0),  # sharpe units
        make_scorecard("old_b", test_metric=8.0, train_metric=8.0),
        make_scorecard("old_c", test_metric=7.0, train_metric=7.0),
    ]
    chal = Scorecard(
        family="fresh",
        params={},
        metric_name="total_return",
        symbols={
            "FIT": SymbolScore(
                splits=[
                    SplitScore(
                        0,
                        Metrics(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                        Metrics(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                    )
                ]
            )
        },
    )
    d = decide(chal, champs, RULES)
    assert d.promote is True
    assert d.demote_index == 0  # the first stale champion loses its slot
    assert "stale champion" in d.rationale


def test_stale_champion_bar_still_applies():
    # A stale slot behaves like a free slot: a challenger below the bar is still rejected.
    champs = [make_scorecard("old", test_metric=9.0, train_metric=9.0)] * 3
    chal = Scorecard(
        family="fresh",
        params={},
        metric_name="total_return",
        symbols={
            "FIT": SymbolScore(
                splits=[
                    SplitScore(
                        0,
                        Metrics(-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                        Metrics(-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                    )
                ]
            )
        },
    )
    d = decide(chal, champs, RULES)
    assert d.promote is False
    assert "stale champion present" in d.rationale


def test_same_metric_champions_not_stale():
    # Metric regimes match → the normal beat-the-weakest comparison, no stale displacement.
    champs = [
        make_scorecard("a", test_metric=1.0, train_metric=1.0),
        make_scorecard("b", test_metric=0.5, train_metric=0.6),
        make_scorecard("c", test_metric=2.0, train_metric=2.1),
    ]
    d = decide(make_scorecard("chal", test_metric=1.2, train_metric=1.3), champs, RULES)
    assert d.promote is True
    assert "stale" not in d.rationale


def test_full_registry_better_metric_demotes_weakest():
    champs = [
        make_scorecard("a", test_metric=1.0, train_metric=1.0),
        make_scorecard("b", test_metric=0.5, train_metric=0.6),  # weakest
        make_scorecard("c", test_metric=2.0, train_metric=2.1),
    ]
    d = decide(make_scorecard("chal", test_metric=1.2, train_metric=1.3), champs, RULES)
    assert d.promote is True
    assert d.demote_index == 1  # 'b' is weakest
    assert "beats weakest" in d.rationale


def test_full_registry_worse_metric_rejected():
    champs = [
        make_scorecard("a", test_metric=1.0, train_metric=1.0),
        make_scorecard("b", test_metric=0.5, train_metric=0.6),
        make_scorecard("c", test_metric=2.0, train_metric=2.1),
    ]
    d = decide(make_scorecard("chal", test_metric=0.4, train_metric=0.4), champs, RULES)
    assert d.promote is False
    assert "does not beat weakest" in d.rationale


def test_unvalidated_challenger_rejected():
    sc = Scorecard(family="x", params={}, stage="prefilter_rejected")
    d = decide(sc, [], RULES)
    assert d.promote is False
    assert "not validated" in d.rationale


# --- registry apply + persistence --------------------------------------------------------


def test_registry_promotes_and_persists(tmp_path):
    path = tmp_path / "champions.json"
    reg = ChampionRegistry(path, capacity=3)
    assert reg.is_empty()
    d = reg.consider(make_scorecard("sma_crossover", 1.5, 1.7, fast=5, slow=20), RULES)
    assert d.promote
    assert len(reg.list()) == 1

    # Fresh process: reload from disk → identical set (restart survival).
    reloaded = ChampionRegistry(path, capacity=3)
    assert len(reloaded.list()) == 1
    entry = reloaded.list()[0]
    assert entry.family == "sma_crossover"
    assert entry.params == {"fast": 5, "slow": 20}
    assert entry.test_metric == pytest.approx(1.5)


def test_registry_full_cycle_demotion_recorded(tmp_path):
    reg = ChampionRegistry(tmp_path / "c.json", capacity=2)
    reg.consider(make_scorecard("a", 1.0, 1.1), RULES)
    reg.consider(make_scorecard("b", 0.5, 0.6), RULES)
    assert len(reg.list()) == 2
    # A stronger challenger displaces the weakest ('b').
    reg.consider(make_scorecard("c", 1.4, 1.5), PromotionRules(2, 1.0, 0.0))
    families = {e.family for e in reg.list()}
    assert families == {"a", "c"}
    assert reg.demotions()  # a demotion was recorded for the report/memory
    assert reg.demotions()[-1]["demoted"]["family"] == "b"


def test_registry_atomic_write_survives_crash(tmp_path, monkeypatch):
    path = tmp_path / "champions.json"
    reg = ChampionRegistry(path, capacity=3)
    reg.consider(make_scorecard("safe", 1.0, 1.0, fast=3, slow=9), RULES)
    assert len(reg.list()) == 1

    # Simulate a crash during the atomic rename step of the next save.
    def boom(self, target):
        raise OSError("simulated crash mid-rename")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        reg.consider(make_scorecard("risky", 2.0, 2.0), RULES)

    # The on-disk registry is still the pre-crash content and loads cleanly.
    fresh = ChampionRegistry(path, capacity=3)
    assert [e.family for e in fresh.list()] == ["safe"]


def test_panel_champion_persists_fit_and_live_symbols(tmp_path):
    path = tmp_path / "champions.json"
    reg = ChampionRegistry(path, capacity=3)
    d = reg.consider(make_panel_scorecard("sma_crossover", {"MSFT": 1.2, "AAPL": 1.0}), RULES)
    assert d.promote

    reloaded = ChampionRegistry(path, capacity=3)
    entry = reloaded.list()[0]
    assert entry.fit_symbols == ["AAPL", "MSFT"]  # sorted, survived the restart
    assert entry.live_symbols == ["AAPL", "MSFT"]  # live = fit until the screener (P3)


def test_panel_of_one_binds_its_fit_symbol(tmp_path):
    """A champion researched on ONE symbol trades only that symbol — never the universe.

    (Pre-unification, single-symbol cards carried no symbols and were crowned universal:
    validated on one name, eligible everywhere. The panel of one closes that hole.)
    """
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    d = reg.consider(make_panel_scorecard("sma_crossover", {"SPY": 1.2}), RULES)
    assert d.promote
    entry = reg.list()[0]
    assert entry.fit_symbols == ["SPY"]
    assert entry.live_symbols == ["SPY"]


def test_legacy_sentinel_never_binds_eligibility(tmp_path):
    """A legacy card (splits under the ``*`` sentinel) keeps universal eligibility.

    The sentinel keys scores, not symbols — binding ``fit_symbols=["*"]`` would quietly
    stop a legacy champion from trading anything.
    """
    from noctis.backtest.scorecard import Scorecard

    split = {"split_index": 0, "train": _m(1.7).__dict__, "test": _m(1.5).__dict__}
    legacy_card = Scorecard.from_dict(
        make_scorecard("sma_crossover", 1.5, 1.7, fast=5, slow=20).to_dict()
        | {"symbols": None, "splits": [split]}
    )
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    d = reg.consider(legacy_card, RULES)
    assert d.promote
    entry = reg.list()[0]
    assert entry.fit_symbols is None  # universal — the sentinel never binds
    assert entry.live_symbols is None


def test_legacy_champion_entry_loads_with_none_symbols(tmp_path):
    """An old champions.json (no symbol keys) loads as None = trade the whole universe."""
    import json

    path = tmp_path / "champions.json"
    reg = ChampionRegistry(path, capacity=3)
    reg.consider(make_scorecard("sma_crossover", 1.5, 1.7, fast=5, slow=20), RULES)

    # Strip the new keys from the persisted file, simulating a pre-panel registry.
    data = json.loads(path.read_text())
    for entry in data["champions"]:
        entry.pop("fit_symbols", None)
        entry.pop("live_symbols", None)
    path.write_text(json.dumps(data))

    entry = ChampionRegistry(path, capacity=3).list()[0]
    assert entry.fit_symbols is None
    assert entry.live_symbols is None


def test_champion_entry_roundtrips_mandate_source():
    from noctis.champions.registry import ChampionEntry

    entry = ChampionEntry(
        family="sma_crossover",
        params={"fast": 5, "slow": 20},
        scorecard=make_scorecard("sma_crossover", 1.5, 1.7),
        crowned_at="2026-07-05T00:00:00+00:00",
        rationale="free slot",
        mandate_source="profile:aggressive",
    )
    restored = ChampionEntry.from_dict(entry.to_dict())
    assert restored.mandate_source == "profile:aggressive"


def test_champion_entry_old_dict_without_mandate_source_loads_none():
    """Hard back-compat: a champions.json entry predating the key loads with None."""
    from noctis.champions.registry import ChampionEntry

    entry = ChampionEntry(
        family="sma_crossover",
        params={"fast": 5, "slow": 20},
        scorecard=make_scorecard("sma_crossover", 1.5, 1.7),
        crowned_at="2026-07-05T00:00:00+00:00",
        rationale="free slot",
    )
    old = entry.to_dict()
    del old["mandate_source"]  # simulate a pre-provenance registry file
    restored = ChampionEntry.from_dict(old)
    assert restored.mandate_source is None


def test_consider_stamps_mandate_source(tmp_path):
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    reg.consider(
        make_scorecard("sma_crossover", 1.5, 1.7),
        RULES,
        mandate_source="profile:aggressive",
    )
    # Survives a restart carrying its provenance.
    reloaded = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    assert reloaded.list()[0].mandate_source == "profile:aggressive"


def test_consider_without_mandate_source_leaves_none(tmp_path):
    reg = ChampionRegistry(tmp_path / "champions.json", capacity=3)
    reg.consider(make_scorecard("sma_crossover", 1.5, 1.7), RULES)
    assert reg.list()[0].mandate_source is None


def test_registry_reset_clears_champions_and_records_history(tmp_path):
    path = tmp_path / "champions.json"
    reg = ChampionRegistry(path, capacity=3)
    reg.consider(make_scorecard("a", 1.0, 1.1), RULES)
    reg.consider(make_scorecard("b", 0.5, 0.6), RULES)

    dropped = reg.reset("metric regime changed")
    assert dropped == 2
    assert reg.is_empty()
    # Each dropped champion is recorded as a demotion with the reset rationale.
    resets = [h for h in reg.demotions() if h["rationale"].startswith("reset:")]
    assert {r["demoted"]["family"] for r in resets} == {"a", "b"}

    # The reset survives a restart — a fresh load sees an empty registry, history intact.
    reloaded = ChampionRegistry(path, capacity=3)
    assert reloaded.is_empty()
    assert len(reloaded.history) == len(reg.history)


# --- CLI ---------------------------------------------------------------------------------


def test_champions_cli_renders(tmp_path):
    from typer.testing import CliRunner

    from noctis.cli import app

    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nchampion_count: 3\n")
    (tmp_path / "state").mkdir()
    reg = ChampionRegistry(tmp_path / "state" / "champions.json", capacity=3)
    reg.consider(make_scorecard("sma_crossover", 1.23, 1.4, fast=5, slow=20), RULES)

    result = CliRunner().invoke(app, ["champions", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "sma_crossover" in result.output
    assert "1.2300" in result.output


def test_champions_cli_reset(tmp_path):
    from typer.testing import CliRunner

    from noctis.cli import app

    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\nchampion_count: 3\n")
    (tmp_path / "state").mkdir()
    reg = ChampionRegistry(tmp_path / "state" / "champions.json", capacity=3)
    reg.consider(make_scorecard("sma_crossover", 1.23, 1.4), RULES)

    result = CliRunner().invoke(app, ["champions", "--config", str(cfg), "--reset"])
    assert result.exit_code == 0, result.output
    assert "Dropped 1 champion(s)" in result.output
    assert ChampionRegistry(tmp_path / "state" / "champions.json", capacity=3).is_empty()


def test_champions_cli_marks_stale_metric(tmp_path):
    from typer.testing import CliRunner

    from noctis.cli import app

    # Champion scored on sharpe, config metric total_return → rendered as stale.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"state_dir: {tmp_path}/state/\nchampion_count: 3\npromotion:\n  metric: total_return\n"
    )
    (tmp_path / "state").mkdir()
    reg = ChampionRegistry(tmp_path / "state" / "champions.json", capacity=3)
    reg.consider(make_scorecard("sma_crossover", 1.23, 1.4), RULES)

    result = CliRunner().invoke(app, ["champions", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "sharpe(stale)" in result.output


def test_champions_cli_empty(tmp_path):
    from typer.testing import CliRunner

    from noctis.cli import app

    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")
    result = CliRunner().invoke(app, ["champions", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "No champions yet" in result.output


# --- champions.json stays small: scorecard compaction ------------------------------------------
def _split(i: int, train: float, test: float, exp: float = 1.0) -> SplitScore:
    return SplitScore(i, _m(train, exposure=exp), _m(test, exposure=exp))


def test_compact_collapses_splits_but_preserves_every_champion_aggregate():
    """A stored champion is read only for mean-of-means aggregates, so compact() — collapsing
    each symbol's walk-forward splits to a single mean split — must leave
    avg_train/avg_test/gap/avg_test_named/symbol_test_metrics unchanged while shrinking the card."""
    card = Scorecard(
        family="fam",
        params={"p": 1},
        metric_name="sharpe",
        stage="validated",
        holdout_metric=0.5,
        symbol_holdout_metric=0.4,
        symbols={
            "AAA": SymbolScore(
                splits=[_split(0, 2.0, 1.0), _split(1, 4.0, 3.0, 0.0), _split(2, 3.0, 2.0)],
                holdout_metric=0.7,
            ),
            "BBB": SymbolScore(
                splits=[_split(0, 1.0, 0.0), _split(1, 3.0, 2.0)], holdout_metric=0.1
            ),
        },
    )
    compact = card.compact()

    # Each symbol collapsed to exactly one split (was 3 + 2 = 5).
    assert [len(ss.splits) for ss in compact.symbols.values()] == [1, 1]

    # Every aggregate a stored champion is read for is preserved.
    assert compact.avg_train_metric == pytest.approx(card.avg_train_metric)
    assert compact.avg_test_metric == pytest.approx(card.avg_test_metric)
    assert compact.gap == pytest.approx(card.gap)
    assert compact.avg_test_named("sharpe") == pytest.approx(card.avg_test_named("sharpe"))
    assert compact.symbol_test_metrics() == pytest.approx(card.symbol_test_metrics())
    # Fit symbols + scalar/per-symbol holdouts survive.
    assert compact.fit_symbols == card.fit_symbols == ["AAA", "BBB"]
    assert compact.holdout_metric == 0.5 and compact.symbol_holdout_metric == 0.4
    assert compact.symbols["AAA"].holdout_metric == 0.7

    # Idempotent: compacting again changes nothing material.
    again = compact.compact()
    assert again.avg_test_metric == pytest.approx(compact.avg_test_metric)
    assert [len(ss.splits) for ss in again.symbols.values()] == [1, 1]


def test_registry_persists_compacted_scorecard(tmp_path):
    """Crowning a champion with a fat multi-split scorecard writes a compact champions.json — one
    split per symbol, not hundreds — while the reloaded champion's metric/gap are unchanged."""
    import json

    fat = Scorecard(
        family="fam",
        params={"p": 1},
        metric_name="sharpe",
        stage="validated",
        symbols={"AAA": SymbolScore(splits=[_split(i, 2.5, 2.0) for i in range(200)])},
    )
    path = tmp_path / "champions.json"
    ChampionRegistry(path, capacity=3).consider(fat, RULES)

    stored = json.loads(path.read_text())["champions"][0]["scorecard"]["symbols"]["AAA"]["splits"]
    assert len(stored) == 1  # 200 splits collapsed to one on disk

    reloaded = ChampionRegistry(path, capacity=3)
    assert reloaded.champions[0].test_metric == pytest.approx(fat.avg_test_metric)
    assert reloaded.champions[0].gap == pytest.approx(fat.gap)
