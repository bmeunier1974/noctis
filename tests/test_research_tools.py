"""Protocol discipline in the research toolbox — journal, exhaustion gate, verdicts,
holdout-capped previews, budget caps, and the new-strategy soft nudge."""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import pandas as pd
import pytest

from noctis.champions import ChampionRegistry, PromotionRules
from noctis.config.settings import Settings
from noctis.data.ingest import IngestResult
from noctis.data.types import empty_bars
from noctis.memory import InMemoryMemory
from noctis.observability import Event
from noctis.research import Capabilities, Turn
from noctis.research.tools import ResearchToolbox
from noctis.strategies import library
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.library import parse_header, strategy_source, write_strategy

_NS_PER_MINUTE = 60 * 1_000_000_000

# Wide-open promotion rules so the approve verdict is deterministic in tests.
LENIENT = PromotionRules(
    champion_count=3,
    max_gap=1e9,
    min_test_metric=-1e9,
    min_holdout_metric=-1e9,
    min_symbol_holdout_metric=-1e9,
)

PROBE = '''"""Toy probe: long above its own moving average.

status: draft
style: momentum
"""
from collections import deque
from dataclasses import dataclass

from noctis.strategies import indicators as ind
from noctis.strategies import scenarios as sc
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class Probe(TraderStrategy):
    name = "probe"

    @dataclass(frozen=True)
    class Params:
        lookback: int = 12
        edge: float = 1.0

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        self._closes = deque(maxlen=self.params.lookback)

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._closes.append(bar.close)
        mean = ind.sma(self._closes, self.params.lookback)
        ctx.set_target(0 if mean is None else int(bar.close > mean * self.params.edge))

    @classmethod
    def param_space(cls):
        return [ParamSpec("lookback", "int", 5, 40, 1)]

    @classmethod
    def scenarios(cls):
        warm = cls.params_cls().lookback
        return [
            sc.Scenario(
                "rally_then_fade",
                segments=[sc.flat(warm + 8), sc.trend(30, 0.10), sc.selloff(20, 0.15)],
                expect=[
                    sc.flat_until(warm),
                    sc.long_within(warm + 8, warm + 33),
                    sc.flat_by(warm + 53),
                ],
            ),
            sc.Scenario(
                "steady_decline_stays_flat",
                segments=[sc.flat(warm + 8), sc.selloff(40, 0.20)],
                expect=[sc.always_flat()],
            ),
        ]
'''


def make_bars(n: int = 320, seed: int = 0, drift: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(drift, 1.0, n)) + 4.0 * np.sin(np.linspace(0, 9, n))
    return pd.DataFrame(
        {
            "ts_event": [i * _NS_PER_MINUTE for i in range(n)],
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1000] * n,
        }
    )


class FakeLake:
    """The four toolbox seams of MarketDataLake, catalog-in-memory."""

    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame]):
        self.bars = bars_by_symbol
        self.ensure_calls: list[tuple] = []

    def check_symbol_ready(self, symbol, dataset=None, schema=None):
        return symbol in self.bars

    def get_bars(self, dataset, schema, symbols, start, end):
        out = {}
        for s in symbols:
            df = self.bars.get(s)
            if df is None:
                out[s] = empty_bars()
            else:
                mask = (df["ts_event"] >= start) & (df["ts_event"] <= end)
                out[s] = df.loc[mask].reset_index(drop=True)
        return out

    def ensure_coverage(self, dataset, schema, symbols, start, end, dry_run=False):
        self.ensure_calls.append((dataset, schema, tuple(symbols), start, end))
        return {s: IngestResult(s, "noop", detail="range fully covered") for s in symbols}


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """This module exercises toolbox protocol, not subprocess isolation — every write
    gate and promotion-plan validation runs through the seam's in-process runner."""


@pytest.fixture
def toolbox(tmp_path):
    return _make_toolbox(tmp_path)


def _make_toolbox(
    tmp_path,
    *,
    min_trials: int = 3,
    max_backtests: int = 50,
    sweep_workers=1,
    coder_client=None,
    coder_model: str | None = None,
    max_author_calls: int | None = None,
    on_event=None,
):
    strategies_dir = tmp_path / "strategies"
    agent = {
        "max_backtests": max_backtests,
        "sweep_trials": 3,
        "sweep_workers": sweep_workers,
    }
    if max_author_calls is not None:
        agent["max_author_calls"] = max_author_calls
    if coder_model is not None:
        agent["coder_model"] = coder_model
    settings = Settings(
        strategies_dir=str(strategies_dir),
        state_dir=str(tmp_path / "state"),
        universe=["AAA", "BBB", "CCC", "DDD"],
        research={
            "min_trials": min_trials,
            "symbol_holdout_size": 1,
            "agent": agent,
        },
    )
    lake = FakeLake(
        {
            "AAA": make_bars(seed=1, drift=0.06),
            "BBB": make_bars(seed=2, drift=0.04),
            "CCC": make_bars(seed=3, drift=0.02),
            "DDD": make_bars(seed=4, drift=0.0),
        }
    )
    registry = ChampionRegistry(tmp_path / "state" / "champions.json", capacity=3)
    memory = InMemoryMemory()
    box = ResearchToolbox(
        settings=settings,
        lake=lake,
        registry=registry,
        families=FamilyRegistry(),
        memory=memory,
        rules=LENIENT,
        coder_client=coder_client,
        on_event=on_event,
    )
    # Author through the toolbox's own tier paths (seeds + workspace tiers), exactly as a
    # session's write_strategy tool would.
    write_strategy(box.strategies_dir, "probe", PROBE, box.families)
    return box


def test_coder_client_defaults_to_none(toolbox):
    """The dedicated coder client is opt-in; unset means driver-authored (today's behavior)."""
    assert toolbox.coder_client is None


def test_coder_client_is_stored_when_supplied(tmp_path):
    """A configured coder client reaches the toolbox as an optional field (inert this story)."""
    from types import SimpleNamespace

    coder = SimpleNamespace(capabilities=Capabilities())
    box = _make_toolbox(tmp_path, coder_client=coder)
    assert box.coder_client is coder


def _journal_lines(box, name):
    return box.journal.records(name)


# ── journal ───────────────────────────────────────────────────────────────────────────────
def test_run_backtest_journals_and_returns_aggregates_only(toolbox):
    out = toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA", "BBB"]})
    assert "error" not in out
    assert out["stage"] == "validated"
    assert out["params"] == {"lookback": 12, "edge": 1.0}  # defaults resolved + journaled
    assert set(out["symbol_test_metrics"]) == {"AAA", "BBB"}
    # Aggregate-only surface: no per-bar or per-trade series anywhere in the result.
    flat = json.dumps(out)
    assert "equity" not in flat and "fills" not in flat and "recent_rows" not in flat

    lines = _journal_lines(toolbox, "probe")
    assert len(lines) == 1 and lines[0]["event"] == "trial"
    assert lines[0]["source"] == "backtest"
    assert lines[0]["params"] == {"lookback": 12, "edge": 1.0}
    assert lines[0]["metrics"]["test"] is not None

    toolbox.dispatch(
        "run_backtest", {"name": "probe", "symbols": ["AAA"], "params": {"lookback": 20}}
    )
    log = toolbox.dispatch("get_experiment_log", {"name": "probe"})
    assert log["n_trials"] == 2
    assert log["n_distinct_params"] == 2
    assert not log["sweep_completed"]
    assert log["top_trials"][0]["test"] >= log["top_trials"][1]["test"]


def test_strategy_timeframe_drives_aggregation(toolbox):
    # A strategy declaring timeframe "5m" is evaluated on 5m bars built from the 1m lake.
    probe5 = PROBE.replace('name = "probe"', 'name = "probe5"\n    timeframe = "5m"')
    out = toolbox.dispatch("write_strategy", {"name": "probe5", "source": probe5})
    assert "error" not in out
    result = toolbox.dispatch("run_backtest", {"name": "probe5", "symbols": ["AAA"]})
    assert "error" not in result
    lines = _journal_lines(toolbox, "probe5")
    assert lines[-1]["window"]["bars"] == 320 // 5  # 1m lake → 5m evaluation frame
    # The native-1m probe still sees the full minute count.
    toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"]})
    assert _journal_lines(toolbox, "probe")[-1]["window"]["bars"] == 320


def test_preview_bars_timeframe(toolbox):
    out = toolbox.dispatch("preview_bars", {"symbol": "AAA", "timeframe": "5m"})
    assert "error" not in out
    assert out["timeframe"] == "5m"
    assert out["training_bars"] <= 320 // 5
    bad = toolbox.dispatch("preview_bars", {"symbol": "AAA", "timeframe": "7m"})
    assert "unsupported timeframe" in bad["error"]


def test_market_context_digest(toolbox):
    ctx = toolbox.market_context()
    assert ctx["bar_schema"] == "ohlcv-1m"
    assert ctx["round_trip_cost_bp"] == pytest.approx(4.0)  # 1bp fee + 1bp slip, both sides
    bars = toolbox.lake.bars["AAA"]
    aaa = ctx["symbols"]["AAA"]
    assert aaa["bars"] == len(bars)
    expected_hold = float(bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0)
    assert aaa["buy_hold_return"] == pytest.approx(expected_hold, abs=1e-4)
    assert aaa["median_abs_bar_move_bp"] > 0
    # Only ready universe symbols appear; missing names are skipped, never errors.
    assert set(ctx["symbols"]) == set(toolbox.settings.universe)
    # The neutral cost-hurdle aggregate: one "k/n" ratio per timeframe, arithmetic only —
    # never a viable/not-viable flag or a symbol recommendation.
    hurdle = ctx["cost_hurdle"]["median_bar_move_clears_round_trip"]
    assert set(hurdle) == set(ctx["supported_timeframes"])
    n = len(ctx["symbols"])
    for ratio in hurdle.values():
        cleared, total = ratio.split("/")
        assert int(total) == n and 0 <= int(cleared) <= n
    assert "viable" not in json.dumps(ctx)


def test_market_context_enumerates_the_capped_focus_set(tmp_path):
    """P2 (context plan): the digest enumerates the capped research focus set, so the prompt
    stops growing with every discovered symbol; unfocused names stay ready in the lake."""
    from noctis.engine.runtime import trading_roster

    box = _make_toolbox(tmp_path)  # universe AAA..DDD, all ready
    box.settings.research.fit_set_size = 2
    box.settings.research.symbol_holdout_size = 1
    box.settings.research.focus_size = 3
    ctx = box.market_context()
    assert list(ctx["symbols"]) == ["AAA", "BBB", "CCC"]  # fit + holdout window, capped
    # DDD is out of the prompt, not out of the system: still on the trading roster.
    assert "DDD" in trading_roster(box.settings, box.lake)

    # A mandate-declared symbol joins the digest once the cap admits it.
    from noctis.research import Mandate

    box.mandate = Mandate(
        text="x", source="cli", summary="x", references=[], config_overrides={}, symbols=["DDD"]
    )
    box.settings.research.focus_size = 4
    assert "DDD" in box.market_context()["symbols"]


def test_backtest_reports_trade_economics_and_hold_benchmark(toolbox):
    out = toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA", "BBB"]})
    econ = out["trade_economics"]
    assert set(econ) == {
        "test_activity",
        "avg_test_exposure",
        "avg_test_turnover",
        "round_trip_cost_bp",
    }
    assert econ["round_trip_cost_bp"] == pytest.approx(4.0)
    assert 0.0 <= econ["test_activity"] <= 1.0
    hold = out["buy_hold_full_window"]
    assert set(hold["per_symbol"]) == {"AAA", "BBB"}
    assert hold["mean"] == pytest.approx(sum(hold["per_symbol"].values()) / 2, abs=1e-3)
    # The surface stays aggregate-only.
    flat = json.dumps(out)
    assert "equity" not in flat and "fills" not in flat


# ── exhaustion gate ──────────────────────────────────────────────────────────────────────
def test_verdicts_refuse_below_min_trials_then_pass_after_sweep(toolbox):
    for tool, args in (
        ("evaluate_vs_champion", {"name": "probe", "symbols": ["AAA"], "params": {}}),
        ("reject_strategy", {"name": "probe", "reason": "premature"}),
    ):
        out = toolbox.dispatch(tool, args)
        assert "error" in out and "exhaustion gate" in out["error"], tool
    assert toolbox.registry.is_empty()
    assert parse_header(strategy_source(toolbox.strategies_dir, "probe")).status == "draft"

    sweep = toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3})
    assert sweep["sweep_completed"] and sweep["n_trials"] == 3
    assert [r.get("event") for r in _journal_lines(toolbox, "probe")].count("trial") == 3
    assert _journal_lines(toolbox, "probe")[-1]["event"] == "sweep_complete"

    out = toolbox.dispatch("reject_strategy", {"name": "probe", "reason": "no edge"})
    assert out.get("ok") is True
    assert parse_header(strategy_source(toolbox.strategies_dir, "probe")).status == "rejected"
    rejected = toolbox.memory.rejected_ideas()
    assert rejected and rejected[0]["family"] == "probe"
    assert rejected[0]["params"] == out["best_params"]
    assert toolbox.rejections == 1


def test_gate_passes_on_distinct_backtests_without_sweep(toolbox):
    for lookback in (8, 16, 24):
        toolbox.dispatch(
            "run_backtest",
            {"name": "probe", "symbols": ["AAA"], "params": {"lookback": lookback}},
        )
    out = toolbox.dispatch("reject_strategy", {"name": "probe", "reason": "flat"})
    assert out.get("ok") is True


def test_repeated_identical_params_do_not_satisfy_gate(toolbox):
    for _ in range(4):
        toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"]})
    out = toolbox.dispatch("reject_strategy", {"name": "probe", "reason": "same params 4x"})
    assert "error" in out and "exhaustion gate" in out["error"]


# ── approval write-back ──────────────────────────────────────────────────────────────────
def test_approval_promotes_and_writes_back_into_file(toolbox):
    toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA", "BBB"], "n_trials": 3})
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 18}},
    )
    assert out.get("promoted") is True, out
    assert out["symbol_holdout_symbols"] == ["CCC"]  # first ready universe name outside fit

    entries = toolbox.registry.list()
    assert len(entries) == 1
    assert entries[0].family == "probe"
    assert entries[0].params == {"lookback": 18, "edge": 1.0}  # resolved, reproducible
    assert entries[0].fit_symbols == ["AAA", "BBB"]

    # The winner moved out of the working area into the champions tier (never the seeds).
    champ = toolbox.strategies_dir.champions / "probe.py"
    assert library.strategy_path(toolbox.strategies_dir, "probe") == champ
    assert not (toolbox.strategies_dir.tmp / "probe.py").exists()

    source = strategy_source(toolbox.strategies_dir, "probe")
    header = parse_header(source)
    assert header.status == "champion"
    assert header.symbols == ["AAA", "BBB"]
    assert header.tuned is not None
    assert "lookback: int = 18" in source  # tuned defaults written back

    assert any("PROMOTED probe" in f for f in toolbox.memory.findings())
    verdicts = [r for r in _journal_lines(toolbox, "probe") if r.get("event") == "verdict"]
    assert verdicts and verdicts[-1]["promoted"] is True
    assert toolbox.promotions == 1
    assert "probe" not in toolbox.undecided


def test_reject_strategy_refuses_a_current_champion(toolbox):
    # Seat "probe" as a champion, then try to reject it: the guard must refuse so the file's
    # status and the registry can't split-brain (the bug that stamped live champions rejected).
    toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA", "BBB"], "n_trials": 3})
    promoted = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 18}},
    )
    assert promoted.get("promoted") is True
    assert any(e.family == "probe" for e in toolbox.registry.list())

    out = toolbox.dispatch("reject_strategy", {"name": "probe", "reason": "second thoughts"})
    assert "error" in out and "current champion" in out["error"]
    # The refusal is total: no re-stamp, no memory record, no counter bump.
    assert parse_header(strategy_source(toolbox.strategies_dir, "probe")).status == "champion"
    assert toolbox.rejections == 0
    assert toolbox.memory.rejected_ideas() == []
    # And the champion is still seated and still tradeable.
    assert any(e.family == "probe" for e in toolbox.registry.list())


# ── soft nudge, preview holdout cap, budget, data ────────────────────────────────────────
def test_new_strategy_while_undecided_warns_but_does_not_block(toolbox):
    toolbox.dispatch("write_strategy", {"name": "probe", "source": PROBE})  # revision: in play
    second = PROBE.replace('name = "probe"', 'name = "probe_two"').replace(
        "class Probe", "class ProbeTwo"
    )
    out = toolbox.dispatch("write_strategy", {"name": "probe_two", "source": second})
    assert out.get("ok") is True  # soft nudge, not a block
    assert "undecided" in out.get("warning", "")

    # Once the first is decided, a third new strategy draws no warning.
    toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3})
    toolbox.dispatch("reject_strategy", {"name": "probe", "reason": "done"})
    third = PROBE.replace('name = "probe"', 'name = "probe_three"').replace(
        "class Probe", "class ProbeThree"
    )
    toolbox.dispatch("run_sweep", {"name": "probe_two", "symbols": ["AAA"], "n_trials": 3})
    out3 = toolbox.dispatch("write_strategy", {"name": "probe_three", "source": third})
    assert out3.get("ok") is True and "warning" not in out3


def test_write_strategy_validation_error_is_tool_error(toolbox):
    bad = PROBE.replace("self._closes.append(bar.close)", "raise RuntimeError('boom')")
    out = toolbox.dispatch(
        "write_strategy",
        {"name": "bad_probe", "source": bad.replace('name = "probe"', 'name = "bad_probe"')},
    )
    assert "error" in out and "validation failed" in out["error"]
    assert "bad_probe" not in toolbox.undecided


def test_write_gate_rejection_steers_toward_repair_not_abandonment(toolbox):
    # A validation failure names the offending source line and instructs a resubmission
    # under the SAME name — a one-line syntax error must not cost the whole thesis (models
    # were observed authoring a brand-new strategy after every rejection).
    bad = PROBE.replace("mean = ind.sma", "mean = = ind.sma")
    out = toolbox.dispatch("write_strategy", {"name": "probe_bad", "source": bad})
    assert "validation failed" in out["error"]
    assert "SAME name" in out["error"]
    assert "offending line: 'mean = = ind.sma(self._closes, self.params.lookback)'" in out["error"]


def test_abandoning_a_failed_draft_draws_a_warning(toolbox):
    bad = PROBE.replace("mean = ind.sma", "mean = = ind.sma")
    out = toolbox.dispatch("write_strategy", {"name": "half_baked", "source": bad})
    assert "error" in out

    # Authoring a DIFFERENT name next, instead of repairing, is called out (soft, no block).
    second = PROBE.replace('name = "probe"', 'name = "probe_two"').replace(
        "class Probe", "class ProbeTwo"
    )
    ok = toolbox.dispatch("write_strategy", {"name": "probe_two", "source": second})
    assert ok.get("ok") is True
    assert "half_baked" in ok.get("warning", "")

    # The nudge fires once: a later write no longer drags the abandoned draft around.
    third = PROBE.replace('name = "probe"', 'name = "probe_three"').replace(
        "class Probe", "class ProbeThree"
    )
    out3 = toolbox.dispatch("write_strategy", {"name": "probe_three", "source": third})
    assert out3.get("ok") is True
    assert "half_baked" not in out3.get("warning", "")


def test_write_gate_fixation_backstop_redirects_to_the_library(toolbox, monkeypatch):
    # Three consecutive write-gate rejections with no backtest yet → the rejection result
    # gains a redirect toward tuning the existing library. A successful write resets the
    # streak. The gate itself never changes: every submission is still rejected.
    real_write = library.write_strategy
    gate = {"fail": True}

    def _gated_write(strategies_dir, name, source, families):
        if gate["fail"]:
            raise library.StrategyValidationError("smoke replay crashed")
        return real_write(strategies_dir, name, source, families)

    monkeypatch.setattr(library, "write_strategy", _gated_write)

    for _ in range(2):
        out = toolbox.dispatch("write_strategy", {"name": "fixated", "source": PROBE})
        assert "error" in out and "consecutive" not in out["error"]
    out = toolbox.dispatch("write_strategy", {"name": "fixated", "source": PROBE})
    assert "consecutive write-gate rejections" in out["error"]
    assert "list_strategies" in out["error"]

    gate["fail"] = False
    ok = toolbox.dispatch("write_strategy", {"name": "probe", "source": PROBE})
    assert ok.get("ok") is True
    gate["fail"] = True
    out = toolbox.dispatch("write_strategy", {"name": "fixated", "source": PROBE})
    assert "error" in out and "consecutive" not in out["error"]  # streak reset


def test_write_gate_backstop_is_silent_once_a_backtest_has_run(toolbox, monkeypatch):
    # The redirect targets the observed fixation signature (authoring without evidence);
    # a session that has already produced a backtest gets the plain rejection only.
    toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"]})

    def _always_fail(strategies_dir, name, source):
        raise library.StrategyValidationError("smoke replay crashed")

    monkeypatch.setattr(library, "write_strategy", _always_fail)
    for _ in range(4):
        out = toolbox.dispatch("write_strategy", {"name": "fixated", "source": PROBE})
    assert "error" in out and "consecutive" not in out["error"]


def test_write_strategy_scenario_violation_is_tool_error(toolbox):
    dead = PROBE.replace(
        "ctx.set_target(0 if mean is None else int(bar.close > mean * self.params.edge))",
        "ctx.set_target(0)",
    ).replace('name = "probe"', 'name = "dead_probe"')
    out = toolbox.dispatch("write_strategy", {"name": "dead_probe", "source": dead})
    # The one surfaced line names the scenario, the expectation, and a hint.
    assert "error" in out and "rally_then_fade" in out["error"]
    assert "long_within" in out["error"]
    assert "dead_probe" not in toolbox.undecided


def test_verdict_plan_blocks_scenario_breaking_params(toolbox):
    toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3})
    before = strategy_source(toolbox.strategies_dir, "probe")
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA"], "params": {"lookback": 12, "edge": 0.9}},
    )
    assert "error" in out and "known-outcome scenarios" in out["error"]
    assert toolbox.registry.is_empty(), "nothing may be crowned on a failed promotion plan"
    assert strategy_source(toolbox.strategies_dir, "probe") == before  # file untouched
    assert not (toolbox.strategies_dir.champions / "probe.py").exists()
    assert toolbox.promotions == 0
    assert not list(toolbox.strategies_dir.tmp.glob(".promote*"))


def test_preview_bars_never_crosses_into_holdout(toolbox):
    bars = toolbox.lake.bars["AAA"]
    out = toolbox.dispatch("preview_bars", {"symbol": "AAA", "rows": 5})
    assert out["holdout_bars_reserved"] > 0
    cut = len(bars) - out["holdout_bars_reserved"]
    last_visible_ns = int(bars["ts_event"].iloc[cut - 1])
    last_returned = out["recent_rows"][-1]["ts"]
    assert last_returned == pd.Timestamp(last_visible_ns, unit="ns", tz="UTC").isoformat()
    assert out["training_bars"] == cut
    # Span end also stays inside the training window.
    last_visible_date = pd.Timestamp(last_visible_ns, unit="ns", tz="UTC").date().isoformat()
    assert out["span"]["end"] <= last_visible_date


def test_backtest_budget_cap(tmp_path):
    box = _make_toolbox(tmp_path, max_backtests=2)
    assert "error" not in box.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"]})
    assert "error" not in box.dispatch(
        "run_backtest", {"name": "probe", "symbols": ["AAA"], "params": {"lookback": 9}}
    )
    out = box.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"]})
    assert "error" in out and "budget" in out["error"]
    out = box.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"]})
    assert "error" in out and "budget" in out["error"]


# ── multi-fidelity sweeps: max_bars cap + full-panel confirmation ────────────────────────
def test_run_sweep_max_bars_truncates_and_journals(toolbox):
    out = toolbox.dispatch(
        "run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3, "max_bars": 250}
    )
    assert out["sweep_completed"] and out["max_bars"] == 250
    assert "confirm" in out["note"]
    trials = [r for r in _journal_lines(toolbox, "probe") if r.get("event") == "trial"]
    assert len(trials) == 3
    for trial in trials:
        assert trial["max_bars"] == 250
        assert trial["window"]["bars"] == 250  # truncated span, not the full 320
        assert trial["metrics"]["test"] is not None

    below_floor = toolbox.dispatch(
        "run_sweep", {"name": "probe", "symbols": ["AAA"], "max_bars": 50}
    )
    assert "error" in below_floor and "max_bars" in below_floor["error"]


def test_verdict_warns_until_confirmed_on_full_panel(toolbox):
    # Exploration fidelity only: subset symbol + truncated window.
    toolbox.dispatch(
        "run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3, "max_bars": 250}
    )
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 18}},
    )
    assert out.get("promoted") is True  # soft nudge — the verdict still lands
    assert "never tested on this full panel" in out.get("warning", "")


def test_verdict_confirmed_on_full_panel_carries_no_warning(tmp_path):
    box = _make_toolbox(tmp_path)
    box.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3, "max_bars": 250})
    # A subset-only confirm does NOT clear the warning for a wider panel…
    box.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"], "params": {"lookback": 18}})
    out = box.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 18}},
    )
    assert "warning" in out
    # …but a full-panel run_backtest with the exact params does (resolved-params match:
    # {"lookback": 18} resolves to include the default edge, same as the verdict call).
    box.dispatch(
        "run_backtest", {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 18}}
    )
    out = box.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 18}},
    )
    assert "warning" not in out


def test_experiment_log_symbols_filter_is_like_for_like(toolbox):
    toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["AAA"]})
    toolbox.dispatch(
        "run_backtest", {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 20}}
    )
    full = toolbox.dispatch("get_experiment_log", {"name": "probe"})
    assert len(full["top_trials"]) == 2
    subset = toolbox.dispatch("get_experiment_log", {"name": "probe", "symbols": ["AAA"]})
    assert subset["n_matching_trials"] == 1
    assert subset["top_trials"][0]["symbols"] == ["AAA"]
    panel = toolbox.dispatch("get_experiment_log", {"name": "probe", "symbols": ["BBB", "AAA"]})
    assert panel["n_matching_trials"] == 1  # order-insensitive exact match
    assert panel["top_trials"][0]["symbols"] == ["AAA", "BBB"]


# ── parallel sweeps ──────────────────────────────────────────────────────────────────────
def test_run_sweep_parallel_workers_journal_and_ranking(tmp_path):
    box = _make_toolbox(tmp_path, sweep_workers=2)
    out = box.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA", "BBB"], "n_trials": 4})
    assert out["sweep_completed"] and out["n_trials"] == 4
    assert box.backtests_run == 4  # budget accounting stays parent-side
    trials = [r for r in _journal_lines(box, "probe") if r.get("event") == "trial"]
    assert len(trials) == 4
    assert all(t["metrics"]["test"] is not None for t in trials)
    tests = [t["test"] for t in out["top_trials"]]
    assert tests == sorted(tests, reverse=True)
    # Parallel trials satisfy the exhaustion gate like sequential ones.
    assert box.dispatch("reject_strategy", {"name": "probe", "reason": "done"})["ok"] is True


def test_run_sweep_parallel_workers_panel_of_one(tmp_path):
    """A single-symbol sweep through the worker pool evaluates every trial.

    Regression: the workers used to unwrap a one-symbol panel to its bare DataFrame —
    a leftover from before evaluate() became panel-only — so panel-only evaluate iterated
    it column-by-column and every trial died with "'Series' object has no attribute
    'itertuples'". A panel of one is not a different mode, just the smallest panel.
    """
    box = _make_toolbox(tmp_path, sweep_workers=2)
    out = box.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3})
    assert out["sweep_completed"] and out["n_trials"] == 3
    trials = [r for r in _journal_lines(box, "probe") if r.get("event") == "trial"]
    assert len(trials) == 3
    assert all(t["metrics"]["test"] is not None for t in trials)


class _StubCard:
    """The scorecard surface the sweep accounting touches (journal + ranking)."""

    stage = "validated"
    metric_name = "sharpe"
    avg_train_metric = 1.0
    gap = 0.1
    holdout_metric = 0.2

    def __init__(self, test):
        self.avg_test_metric = test


class _FakeRunner:
    """A canned SweepRunner: yields prepared (params, card) trials — no sampler, no pool.

    The pool's own failure modes live in tests/test_research_sweep.py against the real
    runner's interface; here the fake isolates the toolbox's accounting side of the seam."""

    def __init__(self, trials):
        self.trials = trials
        self.calls = []

    def run(self, name, space, bars, n, *, config):
        self.calls.append({"name": name, "n": n, "symbols": sorted(bars)})
        yield from self.trials[:n]


def test_run_sweep_fake_runner_budget_journal_and_ranking(tmp_path):
    """The toolbox side of the SweepRunner seam: budget is spent per yielded trial
    (errored ``None`` ones included), only real cards reach the journal, and the
    ranking + exhaustion surface is built from what the runner returned."""
    box = _make_toolbox(tmp_path)
    box.sweep_runner = _FakeRunner(
        [
            ({"lookback": 10}, _StubCard(0.5)),
            ({"lookback": 20}, None),  # an errored trial: budget spent, nothing journaled
            ({"lookback": 30}, _StubCard(1.5)),
        ]
    )
    out = box.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA"], "n_trials": 3})
    assert out["sweep_completed"] and out["n_trials"] == 2
    assert box.backtests_run == 3
    assert out["backtests_remaining"] == 50 - 3
    assert [t["test"] for t in out["top_trials"]] == [1.5, 0.5]
    assert box.sweep_runner.calls == [{"name": "probe", "n": 3, "symbols": ["AAA"]}]
    trials = [r for r in _journal_lines(box, "probe") if r.get("event") == "trial"]
    assert len(trials) == 2 and all(t["source"] == "sweep" for t in trials)
    # A completed sweep satisfies the exhaustion gate no matter how its trials ran.
    assert box.dispatch("reject_strategy", {"name": "probe", "reason": "done"})["ok"] is True


def test_panel_backtest_parallel_matches_sequential(tmp_path):
    seq = _make_toolbox(tmp_path / "a", sweep_workers=1)
    par = _make_toolbox(tmp_path / "b", sweep_workers=2)
    args = {"name": "probe", "symbols": ["AAA", "BBB"], "params": {"lookback": 14}}
    out_seq = seq.dispatch("run_backtest", dict(args))
    out_par = par.dispatch("run_backtest", dict(args))
    for key in (
        "avg_train_metric",
        "avg_test_metric",
        "gap",
        "holdout_metric",
        "symbol_test_metrics",
        "stage",
    ):
        assert out_par[key] == out_seq[key], key


# ── agent-nominated symbol holdout (mandate-driven discovery) ────────────────────────────
def test_nominated_holdout_accepted_and_journaled(toolbox):
    toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA", "BBB"], "n_trials": 3})
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {
            "name": "probe",
            "symbols": ["AAA", "BBB"],
            "params": {"lookback": 18},
            "holdout_symbols": ["DDD"],
        },
    )
    assert out.get("promoted") is True, out
    assert out["symbol_holdout_symbols"] == ["DDD"]
    assert out["symbol_holdout_metric"] is not None
    verdict = [r for r in _journal_lines(toolbox, "probe") if r.get("event") == "verdict"][-1]
    assert verdict["holdout_symbols"] == ["DDD"]


def test_nominated_holdout_refused_when_tainted_or_in_fit(toolbox):
    toolbox.dispatch("run_sweep", {"name": "probe", "symbols": ["AAA", "BBB"], "n_trials": 3})
    # In the fit set → refused.
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {}, "holdout_symbols": ["AAA"]},
    )
    assert "error" in out and "fit set" in out["error"]
    # Outside the fit set but in the journal (used in tuning) → refused.
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["BBB"], "params": {}, "holdout_symbols": ["AAA"]},
    )
    assert "error" in out and "journal" in out["error"]
    # Not in the lake → refused by the readiness check.
    out = toolbox.dispatch(
        "evaluate_vs_champion",
        {"name": "probe", "symbols": ["AAA", "BBB"], "params": {}, "holdout_symbols": ["ZZZ"]},
    )
    assert "error" in out and "not ready" in out["error"]
    assert toolbox.registry.is_empty()  # every refusal happened before any election


# ── the growing universe: trading roster vs research focus ──────────────────────────────
def test_trading_roster_merges_lake_tracked_symbols():
    from types import SimpleNamespace

    from noctis.engine.runtime import trading_roster

    settings = Settings(universe=["AAA", "BBB"])
    lake = FakeLake({"AAA": make_bars(seed=1)})
    # No coverage registry (fakes) → the config seed only.
    assert trading_roster(settings, lake) == ["AAA", "BBB"]

    class _Cov:
        def all(self):
            return [
                SimpleNamespace(symbol="ZZZ", status="idle", row_count=100),  # discovered
                SimpleNamespace(symbol="AAA", status="idle", row_count=100),  # already seeded
                SimpleNamespace(symbol="ERR", status="error", row_count=100),  # not ready
                SimpleNamespace(symbol="NIL", status="idle", row_count=0),  # no bars
            ]

    lake.coverage = _Cov()
    # Config order first (stable fit set), ready discoveries appended sorted.
    assert trading_roster(settings, lake) == ["AAA", "BBB", "ZZZ"]


def test_research_focus_caps_and_orders_the_prompt_enumeration():
    from noctis.engine.runtime import research_focus
    from noctis.research import Mandate

    settings = Settings(
        universe=["AAA", "BBB", "CCC", "DDD", "EEE"],
        research={"fit_set_size": 2, "symbol_holdout_size": 1, "focus_size": 4},
    )
    lake = FakeLake({s: make_bars(seed=i) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])})
    # Fit set (2) + symbol-holdout (1) ready names, in roster order; EEE is not ready.
    assert research_focus(settings, lake) == ["AAA", "BBB", "CCC"]

    # Mandate-declared symbols join after the fit/holdout window, deduped, then the cap.
    mandate = Mandate(
        text="x",
        source="cli",
        summary="x",
        references=[],
        config_overrides={},
        symbols=["DDD", "AAA", "QQQ"],
    )
    assert research_focus(settings, lake, mandate) == ["AAA", "BBB", "CCC", "DDD"]

    # The cap is the prompt-size lever: raising it admits the next mandate symbol (even a
    # not-yet-ready discovery target — consumers filter on readiness themselves).
    settings.research.focus_size = 5
    assert research_focus(settings, lake, mandate) == ["AAA", "BBB", "CCC", "DDD", "QQQ"]


def test_sweep_respects_agent_ranges(toolbox):
    out = toolbox.dispatch(
        "run_sweep",
        {
            "name": "probe",
            "symbols": ["AAA"],
            "n_trials": 3,
            "ranges": {"lookback": {"low": 30, "high": 35}},
        },
    )
    assert out["sweep_completed"]
    for trial in out["top_trials"]:
        assert 30 <= trial["params"]["lookback"] <= 35
    bad = toolbox.dispatch(
        "run_sweep",
        {"name": "probe", "symbols": ["AAA"], "ranges": {"nope": {"low": 1, "high": 2}}},
    )
    assert "error" in bad


def test_ensure_data_and_unready_symbols(toolbox):
    out = toolbox.dispatch(
        "ensure_data", {"symbols": ["AAA"], "start": "2024-01-01", "end": "2024-06-30"}
    )
    assert out["results"]["AAA"]["status"] == "noop"
    assert toolbox.lake.ensure_calls  # went through the budget-gated lake seam

    missing = toolbox.dispatch("run_backtest", {"name": "probe", "symbols": ["ZZZ"]})
    assert "error" in missing and "not ready" in missing["error"]


def test_unknown_strategy_and_unknown_tool(toolbox):
    out = toolbox.dispatch("run_backtest", {"name": "ghost", "symbols": ["AAA"]})
    assert "error" in out and "write_strategy" in out["error"]
    assert "error" in toolbox.dispatch("frobnicate", {})


# ── exhausted-class governor ──────────────────────────────────────────────────────────────
def test_write_strategy_refuses_an_exhausted_class_unless_a_new_lever(toolbox):
    toolbox.exhausted.record("per-symbol long/flat overlay", "forfeits drift", example="old")
    refused = toolbox.dispatch(
        "write_strategy",
        {"name": "probe", "source": PROBE, "class_tag": "Per-Symbol Long/Flat Overlay"},
    )
    assert "error" in refused and "exhausted-class guard" in refused["error"]
    # Naming a genuinely new lever lifts the block.
    allowed = toolbox.dispatch(
        "write_strategy",
        {
            "name": "probe",
            "source": PROBE,
            "class_tag": "Per-Symbol Long/Flat Overlay",
            "new_lever": "adds a short leg",
        },
    )
    assert allowed.get("ok") is True


def test_write_strategy_allows_a_fresh_class_and_journals_the_tag(toolbox):
    out = toolbox.dispatch(
        "write_strategy", {"name": "probe", "source": PROBE, "class_tag": "brand new idea"}
    )
    assert out.get("ok") is True
    assert toolbox.journal.class_tag("probe") == "brand new idea"


def test_market_context_surfaces_exhausted_classes(toolbox):
    toolbox.exhausted.record("dead class", "why it died", example="x")
    ctx = toolbox.market_context()
    assert "dead class" in [c["class_tag"] for c in ctx["exhausted_classes"]]


def test_reject_with_class_exhausted_registers_the_class(toolbox):
    # Satisfy the exhaustion gate (min_trials=3 distinct param sets) before a verdict.
    for lb in (8, 12, 20):
        toolbox.dispatch(
            "run_backtest", {"name": "probe", "symbols": ["AAA"], "params": {"lookback": lb}}
        )
    out = toolbox.dispatch(
        "reject_strategy",
        {
            "name": "probe",
            "reason": "the whole class forfeits drift",
            "class_tag": "per-symbol long/flat overlay",
            "class_exhausted": True,
        },
    )
    assert out.get("status") == "rejected"
    assert out.get("class_exhausted") == "per-symbol long/flat overlay"
    # Registered case-insensitively for future sessions.
    assert toolbox.exhausted.is_exhausted("Per-Symbol Long/Flat Overlay") is not None


def test_reject_class_exhausted_without_tag_warns_and_skips(toolbox):
    for lb in (8, 12, 20):
        toolbox.dispatch(
            "run_backtest", {"name": "probe", "symbols": ["AAA"], "params": {"lookback": lb}}
        )
    out = toolbox.dispatch(
        "reject_strategy",
        {"name": "probe", "reason": "dead", "class_exhausted": True},
    )
    assert out.get("status") == "rejected"
    assert "class_exhausted" not in out and "warning" in out
    assert toolbox.exhausted.load() == []


# ── tool semantics (the declarations the agent loop consumes) ─────────────────────────────
def test_declared_tool_semantics_name_real_tools(toolbox):
    """VERDICT_TOOLS / STRATEGY_HISTORY_TOOLS drive the loop's context budget (what is never
    replaced vs what may collapse to journal pointers); a renamed tool must fail here, not
    silently orphan the semantics."""
    spec_names = {spec["name"] for spec in toolbox.tool_specs()}
    assert ResearchToolbox.VERDICT_TOOLS <= spec_names
    assert ResearchToolbox.STRATEGY_HISTORY_TOOLS <= spec_names
    # Every declared history tool is journal-backed by construction: a dispatched handler
    # for it must exist (the journal itself is exercised by the tests above).
    for name in ResearchToolbox.STRATEGY_HISTORY_TOOLS | ResearchToolbox.VERDICT_TOOLS:
        assert callable(getattr(toolbox, f"tool_{name}"))


def test_result_brief_extracts_the_gate_facing_slice(toolbox):
    """result_brief keeps TOOL_LINE_KEYS order, pulls test_activity up from trade_economics,
    and stays quiet on non-dict / keyless results."""
    result = {
        "n_trials": 7,
        "gap": 0.12,
        "avg_test_metric": 0.9,
        "irrelevant": "noise",
        "trade_economics": {"test_activity": 0.4, "other": 1},
    }
    brief = toolbox.result_brief(result)
    assert list(brief) == ["avg_test_metric", "gap", "n_trials", "test_activity"]
    assert brief["test_activity"] == 0.4
    assert toolbox.result_brief("not a dict") == {}
    assert toolbox.result_brief({"unrelated": 1}) == {}


# ── structural symbol screen (thesis picks the kind, data picks the tickers) ─────────────
def test_screen_symbols_shape_split_and_determinism(toolbox):
    toolbox.settings.research.fit_set_size = 2  # 4-name pool: exercise the split
    out = toolbox.dispatch("screen_symbols", {})
    assert "error" not in out
    assert out["profile"] == {"trend": "any", "volatility": "any", "liquidity": "any"}
    assert out["pool_size"] == 4 and len(out["matched"]) == 4
    strengths = [m["strength"] for m in out["matched"]]
    assert strengths == sorted(strengths, reverse=True)
    for m in out["matched"]:
        assert set(m["bands"]) == {"trend", "volatility", "liquidity"}
        assert {"trend_efficiency", "ann_volatility", "day_dollar_volume_m"} <= set(m["features"])
    ranked = [m["symbol"] for m in out["matched"]]
    # suggested split honors the research config sizes and partitions the ranking in order.
    assert out["suggested_fit"] == ranked[:2]
    assert out["reserved_holdout"] == ranked[2:3]  # symbol_holdout_size=1 in this fixture
    assert "holdout_symbols" in out["note"]  # the keep-it-clean discipline is stated
    assert toolbox.dispatch("screen_symbols", {}) == out  # deterministic


def test_screen_symbols_band_filter_and_rejections(toolbox):
    out = toolbox.dispatch("screen_symbols", {"trend": "high"})
    assert "error" not in out
    matched = {m["symbol"] for m in out["matched"]}
    assert matched and matched | set(out["rejected_bands"]) == {"AAA", "BBB", "CCC", "DDD"}
    for m in out["matched"]:
        assert m["bands"]["trend"] == "high"
    for bands in out["rejected_bands"].values():
        assert bands["trend"] != "high"
    bad = toolbox.dispatch("screen_symbols", {"volatility": "extreme"})
    assert "must be one of" in bad["error"]


def test_screen_symbols_explicit_pool_and_unready_hint(toolbox):
    out = toolbox.dispatch("screen_symbols", {"symbols": ["AAA", "BBB", "ZZZ"]})
    assert out["pool_size"] == 3
    assert {m["symbol"] for m in out["matched"]} == {"AAA", "BBB"}
    assert out["unready"]["symbols"] == ["ZZZ"]
    assert "ensure_data" in out["unready"]["hint"]


def test_market_context_character_matches_the_screen(toolbox):
    """The digest's per-symbol character block IS the screener's read — same keys, same
    numbers — so the prompt and the screen_symbols tool can never tell different stories."""
    digest = toolbox.market_context()
    screened = {
        m["symbol"]: m["features"] for m in toolbox.dispatch("screen_symbols", {})["matched"]
    }
    for sym, entry in digest["symbols"].items():
        character = entry["character"]
        assert set(character) == {"trend_efficiency", "ann_volatility", "day_dollar_volume_m"}
        assert {k: screened[sym][k] for k in character} == character


# ── brief mode: coder-model authoring (schema switch, shared guards, repairable failure) ──
# A brief-carrying source that mismatches its own file name — the write gate rejects it
# deterministically ("class sets name=..."), so a coder that keeps emitting it never lands.
BROKEN = PROBE.replace('name = "probe"', 'name = "mismatch"')

# PROBE with only its thesis line changed — a valid revision whose distinct content proves
# the file was actually replaced (and whose absence in the OLD source proves composition).
REVISED_PROBE = PROBE.replace(
    "Toy probe: long above its own moving average.",
    "Revised probe: long above its own moving average.",
)

BRIEF_ARGS = {
    "thesis": "Long above a short moving average; the drift persists intraday.",
    "entry_exit": "Long when close > SMA(lookback); flat otherwise.",
    "param_space": "lookback int 5..40",
    "scenarios": "A rally pulls long; a steady decline stays flat.",
    "style": "momentum",
    "symbols": ["AAA", "BBB"],
}


def _fenced(source: str) -> str:
    """Wrap strategy source in a python code fence, as a coder reply would."""
    return f"Here is the file:\n```python\n{source}```\n"


def _named(name: str) -> str:
    """PROBE re-pointed at a fresh file name (its `name` attribute must match the file)."""
    return PROBE.replace('name = "probe"', f'name = "{name}"')


class _FakeCoder:
    """A scripted coder client for the toolbox's brief path — no network, no API key.

    Plays fixed text replies through the neutral ``complete()`` seam and records every call,
    so tests can assert both the authored outcome and whether the coder was consulted at all
    (the guards that fire before source exists must spend zero completions)."""

    def __init__(self, replies):
        self._replies = deque(replies)
        self.capabilities = Capabilities()
        self.calls: list[dict] = []

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        self.calls.append({"system": system, "tools": tools, "messages": messages})
        text = self._replies.popleft()
        return Turn(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            usage={},
            assistant_message={"role": "assistant", "content": text},
        )


def _coder_box(
    tmp_path,
    replies,
    *,
    max_author_calls: int | None = None,
    coder_model: str = "fake/coder-1",
    on_event=None,
):
    coder = _FakeCoder(replies)
    return (
        _make_toolbox(
            tmp_path,
            coder_client=coder,
            coder_model=coder_model,
            max_author_calls=max_author_calls,
            on_event=on_event,
        ),
        coder,
    )


def _write_spec(box):
    return next(s for s in box.tool_specs() if s["name"] == "write_strategy")


def test_write_schema_requires_source_without_coder(toolbox):
    """Default (no coder): the driver hand-writes source — schema unchanged, no `brief`."""
    schema = _write_spec(toolbox)["input_schema"]
    assert schema["required"] == ["name", "source"]
    assert "brief" not in schema["properties"]


def test_write_schema_switches_to_brief_with_coder(tmp_path):
    """Coder configured: `brief` becomes required, `source` stays accepted-but-optional —
    exactly one authoring mode is ever visible to the driver."""
    box, _ = _coder_box(tmp_path, [])
    schema = _write_spec(box)["input_schema"]
    assert schema["required"] == ["name", "brief"]
    assert "source" in schema["properties"]  # a capable driver may still hand-write
    brief = schema["properties"]["brief"]
    assert set(brief["required"]) == {"thesis", "entry_exit", "param_space", "scenarios"}
    assert {
        "thesis",
        "entry_exit",
        "param_space",
        "scenarios",
        "reference",
        "style",
        "symbols",
    } <= set(brief["properties"])


def test_brief_scenarios_contract_states_intent_not_tape_dictation(tmp_path):
    """The brief's `scenarios` field is intent (tape shape + expected behavior); the driver must
    not dictate indicator-level tape properties the coder — which owns tape construction — cannot
    honor. This is the driver side of the feasibility fix (the twin of the coder-prompt rules)."""
    box, _ = _coder_box(tmp_path, [])
    schema = _write_spec(box)["input_schema"]
    scenarios = schema["properties"]["brief"]["properties"]["scenarios"]["description"].lower()
    assert "intent" in scenarios
    assert "behavior" in scenarios  # tape shape + expected behavior
    # Do not dictate indicator-level tape properties the coder cannot honor.
    assert "indicator-level" in scenarios
    assert "coder" in scenarios  # tape construction is the coder's job


def test_brief_authors_validated_file_same_shape_as_source(tmp_path):
    """A brief flows driver → toolbox → engine → validated file in the working tier, with
    the same success-result shape (ok/name/path/header) as a source-based write."""
    box, coder = _coder_box(tmp_path, [_fenced(_named("brief_probe"))])
    out = box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})
    assert out.get("ok") is True
    assert out["name"] == "brief_probe"
    assert {"ok", "name", "path", "header"} <= set(out)
    assert (
        library.strategy_path(box.strategies_dir, "brief_probe")
        == box.strategies_dir.tmp / "brief_probe.py"
    )
    # Same post-write bookkeeping as the source path.
    assert "brief_probe" in box.undecided
    assert "brief_probe" in box.strategies_touched
    assert len(coder.calls) == 1  # one completion for a first-try success


def test_source_write_still_works_with_coder_configured(tmp_path):
    """The hand-written revision path: a driver may still submit `source` in coder mode,
    and the coder is never consulted for it."""
    box, coder = _coder_box(tmp_path, [])
    out = box.dispatch("write_strategy", {"name": "hand_written", "source": _named("hand_written")})
    assert out.get("ok") is True
    assert out["name"] == "hand_written"
    assert coder.calls == []


def test_brief_engine_failure_surfaces_repairable_bug_shape(tmp_path):
    """The coder exhausts its private retries → the final validation error reaches the driver
    in the existing repairable-code-bug shape (REPAIR, resubmit the SAME name)."""
    box, coder = _coder_box(tmp_path, [_fenced(BROKEN)] * 3)
    out = box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})
    assert "error" in out
    assert "validation failed" in out["error"]
    assert "SAME name" in out["error"]
    assert "class sets name" in out["error"]  # the gate's real message surfaces
    assert len(coder.calls) == 3  # initial + 2 private retries, invisible to the driver
    assert library.strategy_path(box.strategies_dir, "brief_probe") is None  # nothing landed
    assert "brief_probe" not in box.undecided


def test_brief_mode_exhausted_class_guard_fires_before_coder(tmp_path):
    """The exhausted-class guard fires identically in brief mode — and before any coder
    completion is spent, so delegation cannot reopen a proven dead end."""
    box, coder = _coder_box(tmp_path, [_fenced(_named("brief_probe"))])
    box.exhausted.record("per-symbol long/flat overlay", "forfeits drift", example="old")
    out = box.dispatch(
        "write_strategy",
        {
            "name": "brief_probe",
            "brief": BRIEF_ARGS,
            "class_tag": "Per-Symbol Long/Flat Overlay",
        },
    )
    assert "error" in out and "exhausted-class guard" in out["error"]
    assert coder.calls == []  # blocked before spending a single coder completion
    # Naming a genuinely new lever lifts the block and authors through the coder.
    ok = box.dispatch(
        "write_strategy",
        {
            "name": "brief_probe",
            "brief": BRIEF_ARGS,
            "class_tag": "Per-Symbol Long/Flat Overlay",
            "new_lever": "adds a short leg",
        },
    )
    assert ok.get("ok") is True
    assert len(coder.calls) == 1


def test_brief_mode_undecided_warning_fires(tmp_path):
    """The undecided-strategy soft nudge fires identically in brief mode."""
    box, _ = _coder_box(tmp_path, [_fenced(_named("first_one")), _fenced(_named("second_one"))])
    box.dispatch("write_strategy", {"name": "first_one", "brief": BRIEF_ARGS})
    out = box.dispatch("write_strategy", {"name": "second_one", "brief": BRIEF_ARGS})
    assert out.get("ok") is True  # soft nudge, not a block
    assert "undecided" in out.get("warning", "")


def test_brief_mode_fixation_backstop_fires(tmp_path):
    """The write-fixation backstop fires identically in brief mode: three consecutive
    write-gate rejections with no backtest yet gain the redirect toward the library."""
    box, _ = _coder_box(tmp_path, [_fenced(BROKEN)] * 9)  # 3 completions per failing call
    for _ in range(2):
        out = box.dispatch("write_strategy", {"name": "fixated", "brief": BRIEF_ARGS})
        assert "error" in out and "consecutive" not in out["error"]
    out = box.dispatch("write_strategy", {"name": "fixated", "brief": BRIEF_ARGS})
    assert "consecutive write-gate rejections" in out["error"]
    assert "list_strategies" in out["error"]


# ── brief mode: reference adaptation and revisions (#7) ────────────────────────────────────
def test_brief_reference_composes_library_source_and_lands(tmp_path):
    """A brief naming a library strategy in `reference` includes that strategy's source in the
    coder prompt (translate-don't-invent); the adapted result validates and lands in __tmp."""
    box, coder = _coder_box(tmp_path, [_fenced(_named("adapted"))])
    out = box.dispatch(
        "write_strategy",
        {"name": "adapted", "brief": {**BRIEF_ARGS, "reference": "probe"}},
    )
    assert out.get("ok") is True
    assert out["name"] == "adapted"
    assert (
        library.strategy_path(box.strategies_dir, "adapted")
        == box.strategies_dir.tmp / "adapted.py"
    )
    # probe's full source (the reference) reached the coder for translation.
    assert PROBE in coder.calls[0]["messages"][0]["content"]
    assert len(coder.calls) == 1


def test_brief_unknown_reference_rejected_without_coder_completion(tmp_path):
    """A `reference` naming a strategy that isn't in the library is rejected before any coder
    completion, with the driver-visible repairable error shape."""
    box, coder = _coder_box(tmp_path, [_fenced(_named("adapted"))])
    out = box.dispatch(
        "write_strategy",
        {"name": "adapted", "brief": {**BRIEF_ARGS, "reference": "ghost_strategy"}},
    )
    assert "error" in out
    assert "ghost_strategy" in out["error"]  # names the missing reference
    assert "validation failed" in out["error"] and "SAME name" in out["error"]  # repairable
    assert coder.calls == []  # zero completions spent before the reject
    assert library.strategy_path(box.strategies_dir, "adapted") is None


def test_brief_revision_replaces_existing_file_via_normal_write(tmp_path):
    """A brief whose target name already exists composes the current source as a revision
    request; the validated result replaces the file via the normal write path (__tmp tier)."""
    box, coder = _coder_box(tmp_path, [_fenced(REVISED_PROBE)])
    out = box.dispatch("write_strategy", {"name": "probe", "brief": BRIEF_ARGS})
    assert out.get("ok") is True
    assert out["name"] == "probe"
    # The current version was the change target in the coder prompt.
    assert PROBE in coder.calls[0]["messages"][0]["content"]
    # The validated revision replaced the file in place.
    assert library.strategy_source(box.strategies_dir, "probe") == REVISED_PROBE
    assert library.strategy_path(box.strategies_dir, "probe") == box.strategies_dir.tmp / "probe.py"


def test_brief_failed_revision_leaves_existing_file_untouched(tmp_path):
    """A revision whose validation never passes leaves the previous version untouched — the
    existing library.write_strategy guarantee, proven end-to-end through the brief path."""
    box, coder = _coder_box(tmp_path, [_fenced(BROKEN)] * 3)
    out = box.dispatch("write_strategy", {"name": "probe", "brief": BRIEF_ARGS})
    assert "error" in out and "validation failed" in out["error"]
    assert len(coder.calls) == 3  # initial + 2 private retries
    assert library.strategy_source(box.strategies_dir, "probe") == PROBE  # untouched


# ── brief mode: max_author_calls Class-B budget (#8) ────────────────────────────────────────
def test_author_call_count_starts_at_zero_and_counts_every_completion(tmp_path):
    """Criterion 2 + 4: every coder completion, private retries included, increments the
    session author-call counter the summary surfaces — a first-try success is one, a failing
    job that burns its full retry budget is three."""
    box, _ = _coder_box(tmp_path, [_fenced(_named("ok_one")), *([_fenced(BROKEN)] * 3)])
    assert box.author_calls == 0
    box.dispatch("write_strategy", {"name": "ok_one", "brief": BRIEF_ARGS})
    assert box.author_calls == 1  # one completion for a first-try success
    box.dispatch("write_strategy", {"name": "fails", "brief": BRIEF_ARGS})
    assert box.author_calls == 4  # + initial + 2 private retries on the failing job


def test_source_write_never_touches_the_author_budget(tmp_path):
    """Source-based writes are not coder completions — they never move the author-call count."""
    box, _ = _coder_box(tmp_path, [])
    box.dispatch("write_strategy", {"name": "hand_written", "source": _named("hand_written")})
    assert box.author_calls == 0


def test_exhausted_author_budget_refuses_brief_but_leaves_source_open(tmp_path):
    """Criterion 3: once the author budget is spent, further brief authoring is refused with
    driver guidance (revise by hand or reach a verdict) — a refusal, never a silent failure —
    and a started job may still finish its private retries. Source-based writes stay available."""
    box, coder = _coder_box(
        tmp_path,
        [_fenced(_named("first")), *([_fenced(BROKEN)] * 3)],
        max_author_calls=2,
    )
    # First brief job succeeds on one completion (1/2 spent).
    assert box.dispatch("write_strategy", {"name": "first", "brief": BRIEF_ARGS}).get("ok") is True
    # A started job may overrun the cap with its private retries (2 -> 4): the check refuses to
    # START, it does not abort a running job mid-retry.
    box.dispatch("write_strategy", {"name": "over", "brief": BRIEF_ARGS})
    assert box.author_calls == 4  # 1 + (initial + 2 retries), cap reached mid-job
    spent = len(coder.calls)
    # Now the count has reached the cap: a further brief is refused before any completion.
    refused = box.dispatch("write_strategy", {"name": "late", "brief": BRIEF_ARGS})
    assert "error" in refused
    assert "author" in refused["error"] and "budget" in refused["error"]
    assert "verdict" in refused["error"] and "hand" in refused["error"]
    assert "validation failed" not in refused["error"]  # not a repairable-bug shape
    assert len(coder.calls) == spent  # zero completions spent on the refusal
    assert library.strategy_path(box.strategies_dir, "late") is None
    # The hand-written source path remains open under an exhausted author budget.
    ok = box.dispatch("write_strategy", {"name": "by_hand", "source": _named("by_hand")})
    assert ok.get("ok") is True
    assert len(coder.calls) == spent  # still no coder completion


def test_research_summary_surfaces_the_author_call_count(tmp_path):
    """Criterion 4: the session summary object carries the author-call count (default 0)."""
    from noctis.engine.research import ResearchSummary

    assert ResearchSummary().author_calls == 0


# ── brief mode: authoring observability — one on_event per coder completion (#9) ────────────
def _author_events(events) -> list[Event]:
    return [e for e in events if isinstance(e, Event) and e.kind == "author"]


def test_brief_success_emits_one_author_event_through_on_event(tmp_path):
    """Criterion 1+2+4: a first-try success emits exactly one `author` event through the same
    on_event channel, carrying the coder model, strategy name, attempt number, and outcome."""
    events: list = []
    box, _ = _coder_box(
        tmp_path,
        [_fenced(_named("brief_probe"))],
        coder_model="fake/coder-1",
        on_event=events.append,
    )
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})
    authored = _author_events(events)
    assert len(authored) == 1
    ev = authored[0]
    assert ev.kind == "author"
    assert ev.meta["model"] == "fake/coder-1"
    assert ev.meta["strategy"] == "brief_probe"
    assert ev.meta["attempt"] == 1
    assert ev.meta["ok"] is True
    assert "brief_probe" in ev.text and "fake/coder-1" in ev.text


def test_brief_retry_then_success_emits_one_event_per_completion(tmp_path):
    """Criterion 1: each private retry emits its own event — a failed first attempt (ok False)
    then the completion that lands (ok True), both on the same channel."""
    events: list = []
    box, _ = _coder_box(
        tmp_path,
        [_fenced(BROKEN), _fenced(_named("brief_probe"))],
        on_event=events.append,
    )
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})
    authored = _author_events(events)
    assert [e.meta["attempt"] for e in authored] == [1, 2]
    assert [e.meta["ok"] for e in authored] == [False, True]
    assert "class sets name" in authored[0].meta["outcome"]  # the gate error is the outcome


def test_brief_exhausted_emits_one_failed_author_event_per_completion(tmp_path):
    """Criterion 1: an exhausted job emits one event per completion (initial + 2 retries), each
    carrying a failed outcome — the driver-visible error is separate from the watch feed."""
    events: list = []
    box, _ = _coder_box(tmp_path, [_fenced(BROKEN)] * 3, on_event=events.append)
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})
    authored = _author_events(events)
    assert [e.meta["attempt"] for e in authored] == [1, 2, 3]
    assert all(e.meta["ok"] is False for e in authored)


def test_no_coder_configured_emits_no_author_events(tmp_path):
    """Criterion 3: with no coder configured, a (source-based) write emits no `author` events —
    the event channel is unchanged from today."""
    events: list = []
    box = _make_toolbox(tmp_path, on_event=events.append)
    box.dispatch("write_strategy", {"name": "hand_written", "source": _named("hand_written")})
    assert _author_events(events) == []


# ── brief mode: rejected attempts persist to the capped failed/ area (#18) ──────────────────
# The toolbox is the sole writer of the failed/ store; these tests assert only what lands on
# disk under the working tier's failed/ area — never internal call structure.
def _failed_files(box) -> list:
    root = box.strategies_dir.tmp / "failed"
    return sorted(root.glob("*.py")) if root.is_dir() else []


def test_rejected_brief_attempt_persists_source_and_error_to_failed_area(tmp_path):
    """A gate-rejected coder attempt lands one file under <__tmp>/failed/ carrying BOTH the
    attempted source and the gate error — a bad session inspectable from disk, not scrollback."""
    box, _ = _coder_box(tmp_path, [_fenced(BROKEN), _fenced(_named("brief_probe"))])
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})

    files = _failed_files(box)
    assert len(files) == 1  # only the rejected attempt persisted; the landing one did not
    body = files[0].read_text(encoding="utf-8")
    assert BROKEN in body  # the exact attempted source
    assert "class sets name" in body  # the gate error that rejected it
    assert "brief_probe" in body  # attributed to the strategy


def test_each_rejected_brief_attempt_persists_its_own_file(tmp_path):
    """Every private retry that fails writes its own failure record — three rejections, three
    files, so the whole failing job is on disk."""
    box, _ = _coder_box(tmp_path, [_fenced(BROKEN)] * 3)
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})

    assert len(_failed_files(box)) == 3


def test_successful_brief_persists_no_failure_record(tmp_path):
    """A first-try success writes nothing to the failed/ area (only rejections are persisted)."""
    box, _ = _coder_box(tmp_path, [_fenced(_named("brief_probe"))])
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})

    assert _failed_files(box) == []


def test_non_code_reply_persists_the_raw_reply(tmp_path):
    """A non-code coder reply is a rejected attempt too: its raw text is persisted so a session
    that never produced code is still diagnosable from disk."""
    box, _ = _coder_box(tmp_path, ["no code here, just chatter", _fenced(_named("brief_probe"))])
    box.dispatch("write_strategy", {"name": "brief_probe", "brief": BRIEF_ARGS})

    files = _failed_files(box)
    assert len(files) == 1
    assert "no code here, just chatter" in files[0].read_text(encoding="utf-8")


def test_source_write_persists_no_failure_record(tmp_path):
    """The failed/ area is coder-path only: a rejected hand-written source write (no coder)
    never touches it — the attempt sink is the sole writer."""
    box = _make_toolbox(tmp_path)  # no coder configured
    out = box.dispatch("write_strategy", {"name": "handwritten_fail", "source": BROKEN})

    assert "error" in out
    root = box.strategies_dir.tmp / "failed"
    assert not root.exists() or list(root.glob("*.py")) == []
