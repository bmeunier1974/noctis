"""The strategy library — loader round-trip, the write_strategy gate, header write-back."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from noctis.backtest import Candidate, PipelineConfig, evaluate
from noctis.strategies import library
from noctis.strategies.base import TraderStrategy, replay_targets
from noctis.strategies.families import FamilyRegistry
from noctis.strategies.library import (
    VALID_STATUSES,
    StrategyValidationError,
    _find_strategy_class,
    _load_module,
    _main,
    fixture_frame,
    list_strategies,
    load_and_register,
    parse_header,
    plan_promotion,
    prune_stale_drafts,
    set_header,
    strategy_path,
    strategy_source,
    validate_in_process,
    write_strategy,
)
from noctis.strategies.scenarios import check_scenario_contract

SEED_DIR = "strategies"  # the repo's committed library (read-only in tests)


@pytest.fixture
def families():
    """A per-test registry — whatever a test registers dies with the test."""
    return FamilyRegistry()


GOOD_SOURCE = '''"""Toy probe: long above its own moving average.

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


# ── 1. Library round-trip: seeds register, headers parse, pipeline evaluates ─────────────
def test_seed_library_registers_and_evaluates(families):
    names = load_and_register(SEED_DIR, families)
    # Superset, not equality: the research agent adds files to the live library.
    assert {"sma_crossover", "rsi_meanrev", "donchian_breakout"} <= set(names)

    infos = {i["name"]: i for i in list_strategies(SEED_DIR)}
    # ``status`` is mutable runtime state — the research loop re-stamps it (candidate → champion
    # → rejected) as verdicts land — so assert it parses to a valid value, not a pinned literal.
    # The stable, structural fields below (style/thesis/params/param_space) are the real oracle.
    assert infos["rsi_meanrev"]["status"] in VALID_STATUSES
    assert infos["rsi_meanrev"]["style"] == "mean-reversion"
    assert infos["rsi_meanrev"]["thesis"].startswith("Buy oversold dips")
    assert infos["sma_crossover"]["params"] == {"fast": 10, "slow": 30}
    assert {s["name"] for s in infos["donchian_breakout"]["param_space"]} == {"channel"}

    strategy = families.create("rsi_meanrev", {"period": 7})
    assert strategy.params.period == 7

    card = evaluate(
        Candidate("rsi_meanrev", {}),
        {"AAA": fixture_frame(n=320)},
        config=PipelineConfig(prefilter_min_score=None),
        families=families,
    )
    assert card.stage == "validated"
    assert card.symbols["AAA"].splits


def test_default_signals_equals_hand_computed_series():
    class AboveFirst(TraderStrategy):
        name = "above_first"

        class Params:  # not a dataclass on purpose; signals() never touches it
            pass

        params_cls = Params

        def on_start(self, ctx):
            self._first = None

        def on_bar(self, ctx, bar):
            if self._first is None:
                self._first = bar.close
            ctx.set_target(1 if bar.close > self._first else 0)

        @classmethod
        def param_space(cls):
            return []

    frame = fixture_frame(n=60)
    expected = [1 if c > frame["close"].iloc[0] else 0 for c in frame["close"]]
    assert list(AboveFirst.signals(frame, AboveFirst.Params())) == expected
    assert replay_targets(AboveFirst(AboveFirst.Params()), frame) == expected


# ── 2. The write_strategy gate ────────────────────────────────────────────────────────────
# The two tests below run the REAL subprocess gate — one success, one failure — proving the
# spawn plumbing (argv, exit code, one-line reason across the process boundary). Everything
# else runs the same checks through the seam's in-process runner (the ``fast_gate`` fixture).
def test_write_strategy_good_source_registers(tmp_path, families):
    result = write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    assert result["name"] == "probe"
    assert result["header"]["status"] == "draft"
    # Authored files land in the gitignored working area, not the committed root.
    assert strategy_path(tmp_path, "probe") == tmp_path / "__tmp" / "probe.py"
    assert "probe" in families
    # And the pipeline can evaluate it straight away.
    card = evaluate(
        Candidate("probe", {"lookback": 8}),
        {"AAA": fixture_frame(n=320)},
        config=PipelineConfig(prefilter_min_score=None),
        families=families,
    )
    assert card.stage == "validated"


def test_subprocess_gate_reports_the_reason_across_the_boundary(tmp_path, families):
    broken = GOOD_SOURCE.replace('name = "probe"', 'name = "other"')
    with pytest.raises(StrategyValidationError, match="class sets name"):
        write_strategy(tmp_path, "probe", broken, families)
    assert strategy_path(tmp_path, "probe") is None


def test_scenario_diagnostics_survive_the_subprocess_boundary(tmp_path, families):
    # The execution-feedback diagnostics (#79) are appended to the single-line scenario-failure
    # message, so they survive the fresh-interpreter gate's last-stderr-line boundary — the
    # DEFAULT subprocess validator, not the in-process fast_gate, must still carry them.
    inverted = GOOD_SOURCE.replace("int(bar.close > mean", "int(bar.close < mean")
    with pytest.raises(StrategyValidationError, match="observed: first went long at bar") as exc:
        write_strategy(tmp_path, "probe", inverted, families)
    assert "long spans" in str(exc.value)
    assert strategy_path(tmp_path, "probe") is None


def test_the_gate_seam_defaults_to_the_subprocess_runner():
    assert library.validator is library.validate_in_subprocess


# ── warmup honesty (#80): the shared funnel catches a lying warmup through BOTH runners ────
# A strategy that deterministically enters at bar 5 (ignoring price) yet declares warmup 40 —
# an honest scenario set (directional + no-trade tape) with a dishonest warmup declaration.
LYING_WARMUP_SOURCE = '''"""Toy probe that lies about its warmup: enters at bar 5, declares 40.

status: draft
style: momentum
"""
from dataclasses import dataclass

from noctis.strategies import scenarios as sc
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class LiarProbe(TraderStrategy):
    name = "liar"

    @dataclass(frozen=True)
    class Params:
        enter_at: int = 5

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        self._i = -1

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._i += 1
        ctx.set_target(1 if self._i >= self.params.enter_at else 0)

    @classmethod
    def warmup_bars(cls, params) -> int:
        return 40  # a lie: the code enters at bar `enter_at` = 5

    @classmethod
    def param_space(cls):
        return [ParamSpec("enter_at", "int", 1, 30, 1)]

    @classmethod
    def scenarios(cls):
        return [
            sc.Scenario(
                "enters_early",
                segments=[sc.flat(90)],
                expect=[sc.long_within(5, 89)],
            ),
            sc.Scenario(
                "quiet_tape",
                segments=[sc.flat(90)],
                expect=[sc.always_flat()],
            ),
        ]
'''


def test_lying_warmup_is_caught_through_the_subprocess_gate(tmp_path, families):
    # The DEFAULT fresh-interpreter validator: the actionable warmup message survives the
    # gate's last-stderr-line boundary and nothing lands on disk.
    with pytest.raises(StrategyValidationError, match="warmup_bars=40") as exc:
        write_strategy(tmp_path, "liar", LYING_WARMUP_SOURCE, families)
    assert "bar 5" in str(exc.value)
    assert strategy_path(tmp_path, "liar") is None


def test_lying_warmup_is_caught_through_the_in_process_gate(tmp_path):
    # Same shared funnel, no interpreter spawn — the in-process runner inherits the check.
    with pytest.raises(StrategyValidationError, match="warmup_bars=40") as exc:
        _validate(tmp_path, LYING_WARMUP_SOURCE, name="liar")
    assert "bar 5" in str(exc.value)


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (
            lambda s: s.replace("self._closes.append(bar.close)", "raise RuntimeError('boom')"),
            "raises in on_bar",
        ),
        (lambda s: s.replace('name = "probe"', 'name = "other"'), "name mismatch"),
        (
            lambda s: s.replace("import deque", "import deque  # <<<\nthis is not python"),
            "syntax error",
        ),
        (
            lambda s: s.replace(
                '"""Toy probe: long above its own moving average.\n\n'
                'status: draft\nstyle: momentum\n"""\n',
                "",
            ),
            "no docstring header",
        ),
        (
            lambda s: s.replace("def scenarios(", "def _scenarios("),
            "no known-outcome scenarios declared",
        ),
        (lambda s: s.replace("sc.always_flat()", "sc.flat_until(10)"), "no no-trade scenario"),
        (
            lambda s: s.replace(
                "ctx.set_target(0 if mean is None else int(bar.close > mean * self.params.edge))",
                "ctx.set_target(0)",
            ),
            "dead logic never enters",
        ),
        (
            lambda s: s.replace("bar.close > mean", "bar.close < mean"),
            "inverted logic breaks the declared scenarios",
        ),
        (
            lambda s: s.replace('name = "probe"', 'name = "probe"\n    timeframe = "7m"'),
            "unsupported timeframe",
        ),
    ],
)
def test_write_strategy_rejects_and_leaves_nothing(tmp_path, families, fast_gate, mutate, why):
    with pytest.raises(StrategyValidationError):
        write_strategy(tmp_path, "probe", mutate(GOOD_SOURCE), families)
    assert strategy_path(tmp_path, "probe") is None, why
    assert "probe" not in families, why
    assert not list(tmp_path.rglob(".candidate*.py")), "temp candidate file left behind"


def test_write_strategy_accepts_declared_timeframe(tmp_path, families, fast_gate):
    daily = GOOD_SOURCE.replace('name = "probe"', 'name = "probe"\n    timeframe = "1d"')
    write_strategy(tmp_path, "probe", daily, families)
    assert families.get_class("probe").timeframe == "1d"
    assert list_strategies(tmp_path)[0]["timeframe"] == "1d"


def test_write_strategy_failed_revision_keeps_old_version(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    broken = GOOD_SOURCE.replace("int(bar.close > mean", "int(bar.close / 0 > mean")
    with pytest.raises(StrategyValidationError):
        write_strategy(tmp_path, "probe", broken, families)
    assert strategy_source(tmp_path, "probe") == GOOD_SOURCE  # old artifact untouched
    assert "probe" in families


def test_write_strategy_rejects_parity_violating_signals_override(tmp_path, families, fast_gate):
    cheat = GOOD_SOURCE.replace(
        "    @classmethod\n    def param_space(cls):",
        "    @classmethod\n    def signals(cls, data, params):\n"
        "        import pandas as pd\n"
        "        return pd.Series([1] * len(data), dtype=int)\n\n"
        "    @classmethod\n    def param_space(cls):",
    )
    with pytest.raises(StrategyValidationError, match="parity"):
        write_strategy(tmp_path, "probe", cheat, families)


# ── 3. Approval-time write-back: plan_promotion → commit ─────────────────────────────────
def test_promotion_write_back_roundtrip(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    plan = plan_promotion(
        tmp_path,
        "probe",
        {"lookback": 33, "edge": 1.01},
        symbols=["AAPL", "MSFT"],
        tuned="2026-07-04",
    )
    assert plan.commit(families) == tmp_path / "champions" / "probe.py"

    source = strategy_source(tmp_path, "probe")
    assert "lookback: int = 33" in source
    assert "edge: float = 1.01" in source
    header = parse_header(source)
    assert header.status == "champion"
    assert header.symbols == ["AAPL", "MSFT"]
    assert header.tuned == "2026-07-04"

    # Re-load → the tuned values ARE the defaults now (noctis backtest replays the ship).
    load_and_register(tmp_path, families)
    strategy = families.create("probe")
    assert strategy.params.lookback == 33
    assert strategy.params.edge == 1.01


def test_promotion_plan_that_breaks_scenarios_is_refused(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    # edge=0.9 keeps the probe long on flat/declining tapes — its own declared
    # scenarios (windows derived from the new defaults) must catch the write-back
    # BEFORE anything is crowned or moved: no champion file, no temp residue.
    with pytest.raises(StrategyValidationError, match="violated"):
        plan_promotion(
            tmp_path, "probe", {"lookback": 12, "edge": 0.9}, symbols=["AAA"], tuned="2026-07-04"
        )
    assert "edge: float = 1.0\n" in strategy_source(tmp_path, "probe")  # file untouched
    assert not (tmp_path / "champions" / "probe.py").exists()
    assert not list(tmp_path.rglob(".promote*")), "promotion temp file left behind"


def test_promotion_tolerates_legacy_scenario_less_file(tmp_path, families, fast_gate):
    legacy = GOOD_SOURCE.replace("def scenarios(", "def _scenarios(")
    (tmp_path / "probe.py").write_text(legacy, encoding="utf-8")  # pre-gate artifact
    plan_promotion(tmp_path, "probe", {"lookback": 20}, symbols=["AAA"], tuned="2026-07-04").commit(
        families
    )
    assert parse_header(strategy_source(tmp_path, "probe")).status == "champion"


def test_set_header_tolerates_legacy_scenario_less_file(tmp_path, families, fast_gate):
    legacy = GOOD_SOURCE.replace("def scenarios(", "def _scenarios(")
    (tmp_path / "probe.py").write_text(legacy, encoding="utf-8")  # pre-gate artifact
    set_header(tmp_path, "probe", families=families, status="rejected")
    assert parse_header(strategy_source(tmp_path, "probe")).status == "rejected"


def test_set_header_rejects_bad_status(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    with pytest.raises(ValueError):
        set_header(tmp_path, "probe", families=families, status="shipped")


def _promote(paths, families, name="probe", params=None, symbols=("AAA",), tuned="2026-07-04"):
    """Author-side shorthand: the full plan → commit hand-off with throwaway metadata."""
    plan = plan_promotion(paths, name, params or {}, symbols=list(symbols), tuned=tuned)
    return plan.commit(families)


# ── 3b. Tiered layout: committed seeds pristine, work in __tmp/, champions promoted ───────
def test_write_strategy_authors_into_tmp_not_root(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    assert (tmp_path / "__tmp" / "probe.py").is_file()
    assert not (tmp_path / "probe.py").exists()  # nothing lands in the committed root


def test_promotion_moves_a_working_file(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    dest = _promote(tmp_path, families)
    assert dest == tmp_path / "champions" / "probe.py"
    assert dest.is_file()
    assert not (tmp_path / "__tmp" / "probe.py").exists()  # moved out of the scratch area
    assert strategy_path(tmp_path, "probe") == dest


def test_promotion_copies_a_seed_and_leaves_it_pristine(tmp_path, families, fast_gate):
    (tmp_path / "probe.py").write_text(GOOD_SOURCE, encoding="utf-8")  # a committed seed
    dest = _promote(tmp_path, families)
    assert dest == tmp_path / "champions" / "probe.py"
    assert (tmp_path / "probe.py").read_text() == GOOD_SOURCE  # the seed stays pristine
    assert strategy_path(tmp_path, "probe") == dest  # but the champion wins


def test_re_promotion_updates_the_champion_in_place(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    _promote(tmp_path, families, params={"lookback": 20}, tuned="2026-01-01")
    _promote(
        tmp_path, families, params={"lookback": 33}, symbols=("AAA", "BBB"), tuned="2026-02-02"
    )
    source = strategy_source(tmp_path, "probe")
    assert "lookback: int = 33" in source
    assert parse_header(source).tuned == "2026-02-02"
    assert strategy_path(tmp_path, "probe") == tmp_path / "champions" / "probe.py"


def test_write_strategy_refuses_to_overwrite_a_champion(tmp_path, families, fast_gate):
    write_strategy(tmp_path, "probe", GOOD_SOURCE, families)
    _promote(tmp_path, families)
    with pytest.raises(StrategyValidationError, match="champion"):
        write_strategy(tmp_path, "probe", GOOD_SOURCE, families)


def test_mechanical_rewrite_never_mutates_a_committed_seed(tmp_path, families, fast_gate):
    (tmp_path / "probe.py").write_text(GOOD_SOURCE, encoding="utf-8")  # a committed seed
    set_header(tmp_path, "probe", families=families, status="rejected")
    # the root seed is untouched; the stamped copy lives in the gitignored working area.
    assert parse_header((tmp_path / "probe.py").read_text()).status == "draft"
    assert parse_header((tmp_path / "__tmp" / "probe.py").read_text()).status == "rejected"
    assert parse_header(strategy_source(tmp_path, "probe")).status == "rejected"  # __tmp wins


def test_champion_overrides_seed_of_same_name_on_load(tmp_path, families, fast_gate):
    (tmp_path / "probe.py").write_text(GOOD_SOURCE, encoding="utf-8")  # seed ships edge 1.0
    _promote(tmp_path, families, params={"lookback": 33, "edge": 1.01})
    fresh = FamilyRegistry()
    load_and_register(tmp_path, fresh)
    assert fresh.create("probe").params.edge == 1.01  # the champion, not the 1.0 seed


# ── 4. Known-outcome scenarios: seeds and TEMPLATE are exemplar declarations ─────────────
@pytest.mark.parametrize("name", ["sma_crossover", "rsi_meanrev", "donchian_breakout"])
def test_seed_scenarios_satisfy_the_contract(families, name):
    load_and_register(SEED_DIR, families)
    check_scenario_contract(families.get_class(name))


def test_template_scenarios_satisfy_the_contract():
    cls = _find_strategy_class(_load_module(Path(SEED_DIR) / "TEMPLATE.py"))
    check_scenario_contract(cls)


# Each seed declares an honest warmup derived from its own lookback logic, and the contract check
# (now running the warmup-honesty invariant) still passes on the declaration — the warmup is true.
@pytest.mark.parametrize(
    ("name", "expected_warmup"),
    [("sma_crossover", 30), ("rsi_meanrev", 14), ("donchian_breakout", 20)],
)
def test_seed_declares_an_honest_warmup(families, name, expected_warmup):
    load_and_register(SEED_DIR, families)
    cls = families.get_class(name)
    assert cls.warmup_bars(cls.params_cls()) == expected_warmup
    check_scenario_contract(cls)  # the warmup invariant confirms the declaration is not a lie


def test_template_declares_an_honest_warmup():
    cls = _find_strategy_class(_load_module(Path(SEED_DIR) / "TEMPLATE.py"))
    assert cls.warmup_bars(cls.params_cls()) == 20  # its SMA lookback default
    check_scenario_contract(cls)


# ── 5. LibraryPaths: split tiers (seeds committed, __tmp/champions under the workspace) ──
def test_library_paths_bare_path_coerces_to_the_sibling_layout(tmp_path):
    from noctis.strategies.library import LibraryPaths

    paths = LibraryPaths.coerce(tmp_path)
    assert paths.seeds == tmp_path
    assert paths.tmp == tmp_path / "__tmp"
    assert paths.champions == tmp_path / "champions"
    # An already-built LibraryPaths passes through untouched.
    assert LibraryPaths.coerce(paths) is paths


def test_library_paths_from_settings_splits_seeds_from_workspace_tiers(monkeypatch, tmp_path):
    from noctis.config import load_settings
    from noctis.strategies.library import LibraryPaths

    monkeypatch.delenv("NOCTIS_WORKSPACE", raising=False)
    paths = LibraryPaths.from_settings(load_settings(config_path=tmp_path / "missing.yaml"))
    assert paths.seeds == Path("strategies")
    assert paths.tmp == Path("workspace/strategies/__tmp")
    assert paths.champions == Path("workspace/strategies/champions")


def test_library_paths_pickles_through_pool_initargs(tmp_path):
    import pickle

    from noctis.strategies.library import LibraryPaths

    paths = LibraryPaths.from_single_root(tmp_path)
    assert pickle.loads(pickle.dumps(paths)) == paths


def test_write_strategy_authors_into_the_workspace_tier(tmp_path, families, fast_gate):
    from noctis.strategies.library import LibraryPaths

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    ws = tmp_path / "workspace" / "strategies"
    paths = LibraryPaths(seeds=seeds, tmp=ws / "__tmp", champions=ws / "champions")
    write_strategy(paths, "probe", GOOD_SOURCE, families)
    assert (ws / "__tmp" / "probe.py").is_file()
    assert list(seeds.iterdir()) == []  # committed input stays pristine


def test_promotion_moves_across_the_split_tiers(tmp_path, families, fast_gate):
    from noctis.strategies.library import LibraryPaths

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    ws = tmp_path / "workspace" / "strategies"
    paths = LibraryPaths(seeds=seeds, tmp=ws / "__tmp", champions=ws / "champions")
    write_strategy(paths, "probe", GOOD_SOURCE, families)
    _promote(paths, families)
    assert (ws / "champions" / "probe.py").is_file()
    assert not (ws / "__tmp" / "probe.py").exists()  # moved, not copied
    assert strategy_path(paths, "probe") == ws / "champions" / "probe.py"


# ── protective exits: the gate round-trips a declaring strategy unchanged ────────────────
EXITS_SOURCE = (
    GOOD_SOURCE.replace(
        "from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy",
        "from noctis.strategies.base import Bar, Context, ExitRules, ParamSpec, TraderStrategy",
    )
    .replace(
        "        ctx.set_target(0 if mean is None else int(bar.close > mean * self.params.edge))",
        "        target = 0 if mean is None else int(bar.close > mean * self.params.edge)\n"
        "        ctx.set_target(target, exits=ExitRules(stop_pct=self.params.stop_pct))",
    )
    .replace(
        "        edge: float = 1.0",
        "        edge: float = 1.0\n        stop_pct: float = 0.05",
    )
)


def test_write_strategy_round_trips_an_exits_declaring_strategy(tmp_path, families, fast_gate):
    """No new gate checks exist for exits — scenarios and parity are target-level, so a
    strategy forwarding an ordinary float param into ExitRules passes the same
    fresh-subprocess smoke/scenario/parity validation every strategy does."""
    result = write_strategy(
        tmp_path, "exit_probe", EXITS_SOURCE.replace("probe", "exit_probe"), families
    )
    assert result["name"] == "exit_probe"
    assert "exit_probe" in families
    # The pipeline evaluates it straight away — exits priced by the event-driven stages.
    card = evaluate(
        Candidate("exit_probe", {"lookback": 8}),
        {"AAA": fixture_frame(n=320)},
        config=PipelineConfig(prefilter_min_score=None),
        families=families,
    )
    assert card.stage == "validated"


# ── 6. The gate's checks, hit directly (no subprocess, no write_strategy wrapping) ────────
def _validate(tmp_path, source, name="probe", require_scenarios=True):
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    validate_in_process(path, name, require_scenarios=require_scenarios)


def test_gate_accepts_the_good_source_directly(tmp_path):
    _validate(tmp_path, GOOD_SOURCE)  # the baseline every mutation below breaks


def test_gate_rejects_class_file_name_mismatch(tmp_path):
    bad = GOOD_SOURCE.replace('name = "probe"', 'name = "other"')
    with pytest.raises(StrategyValidationError, match="class sets name='other'"):
        _validate(tmp_path, bad)


def test_gate_requires_a_module_docstring(tmp_path):
    bad = GOOD_SOURCE.replace(
        '"""Toy probe: long above its own moving average.\n\nstatus: draft\nstyle: momentum\n"""\n',
        "",
    )
    with pytest.raises(StrategyValidationError, match="missing module docstring"):
        _validate(tmp_path, bad)


def test_gate_rejects_unsupported_timeframe(tmp_path):
    bad = GOOD_SOURCE.replace('name = "probe"', 'name = "probe"\n    timeframe = "7m"')
    with pytest.raises(StrategyValidationError, match="timeframe '7m' unsupported"):
        _validate(tmp_path, bad)


def test_gate_rejects_invalid_header_status(tmp_path):
    bad = GOOD_SOURCE.replace("status: draft", "status: shipped")
    with pytest.raises(StrategyValidationError, match="header status 'shipped' invalid"):
        _validate(tmp_path, bad)


def test_gate_requires_param_space_to_be_a_list(tmp_path):
    bad = GOOD_SOURCE.replace(
        'return [ParamSpec("lookback", "int", 5, 40, 1)]',
        'return (ParamSpec("lookback", "int", 5, 40, 1),)',
    )
    with pytest.raises(StrategyValidationError, match="param_space"):
        _validate(tmp_path, bad)


def test_gate_rejects_signals_on_bar_parity_violation(tmp_path):
    bad = GOOD_SOURCE.replace(
        "    @classmethod\n    def param_space(cls):",
        "    @classmethod\n    def signals(cls, data, params):\n"
        "        import pandas as pd\n"
        "        return pd.Series([1] * len(data), dtype=int)\n\n"
        "    @classmethod\n    def param_space(cls):",
    )
    with pytest.raises(StrategyValidationError, match="parity"):
        _validate(tmp_path, bad)


def test_gate_requires_scenarios_only_when_asked(tmp_path):
    legacy = GOOD_SOURCE.replace("def scenarios(", "def _scenarios(")
    with pytest.raises(StrategyValidationError, match="scenario"):
        _validate(tmp_path, legacy)
    _validate(tmp_path, legacy, require_scenarios=False)  # legacy files stay stampable


def test_in_process_runner_normalizes_a_crash_to_the_gate_error(tmp_path):
    # Same one-line message contract as the subprocess entry point: a non-gate exception
    # (here: on_bar crashing) surfaces as StrategyValidationError, never raw.
    bad = GOOD_SOURCE.replace("self._closes.append(bar.close)", "raise RuntimeError('boom')")
    with pytest.raises(StrategyValidationError, match="RuntimeError: boom"):
        _validate(tmp_path, bad)


def test_validation_entrypoint_exit_codes_and_one_line_reason(tmp_path, capsys):
    good = tmp_path / "probe.py"
    good.write_text(GOOD_SOURCE, encoding="utf-8")
    assert _main([str(good), "probe", "--require-scenarios"]) == 0
    assert _main([str(good), "other"]) == 1
    assert "StrategyValidationError: class sets name" in capsys.readouterr().err
    assert _main([]) == 2  # usage error


def test_validation_timeout_kills_the_whole_process_tree(tmp_path, monkeypatch):
    """The infinite-hang fix in the write gate: subprocess.run(timeout=...) kills only the
    direct child, then blocks in a second UNBOUNDED communicate() — a grandchild spawned by
    the (agent-authored) file keeps the pipes open and the research loop hangs forever.
    validate_in_subprocess must kill the whole process group, reap promptly, and surface a
    plain StrategyValidationError."""
    import os
    import time

    # Long enough for the child to boot python + import pandas and reach the strategy file
    # (so the grandchild actually spawns), short enough to keep the test quick.
    monkeypatch.setattr(library, "_VALIDATE_TIMEOUT_S", 8)
    pidfile = tmp_path / "grandchild.pid"
    source = (
        "import subprocess, time\n"
        # The grandchild inherits the validation child's stdout/stderr pipes — exactly the
        # shape that wedged run()'s post-kill communicate() forever.
        f'subprocess.Popen(["/bin/sh", "-c", "echo $$ > {pidfile}; exec sleep 60"])\n'
        "time.sleep(60)\n"
    )
    path = tmp_path / "hang_probe.py"
    path.write_text(source, encoding="utf-8")

    start = time.monotonic()
    with pytest.raises(StrategyValidationError, match="timed out"):
        library.validate_in_subprocess(path, "hang_probe")
    assert time.monotonic() - start < 60  # bounded: the old path never returned at all

    # The group kill took the grandchild down with the child — nothing lingers holding pipes.
    deadline = time.monotonic() + 10
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int(pidfile.read_text().strip())
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# ── 7. prune_stale_drafts: sweep undecided drafts out of the working tier into archive/ ───
def _draft_file(directory: Path, name: str, *, status: str = "draft", age_hours: float = 0.0):
    """Author a minimal header-only file in ``directory`` and back-date its mtime.

    prune reads only the docstring header (never imports), so a bare docstring is a full
    fixture here — external filesystem behavior is all these tests assert.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(f'"""Toy {name}.\n\nstatus: {status}\nstyle: momentum\n"""\n', encoding="utf-8")
    if age_hours:
        stamp = time.time() - age_hours * 3600
        os.utime(path, (stamp, stamp))
    return path


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` mapped to its exact bytes — the byte-identical oracle."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_prune_archives_stale_draft_and_spares_fresh(tmp_path):
    work = tmp_path / "__tmp"
    stale = _draft_file(work, "stale_probe", age_hours=3)
    fresh = _draft_file(work, "fresh_probe", age_hours=0)

    archived = prune_stale_drafts(work, ttl_hours=1)

    assert archived == ["stale_probe"]
    assert not stale.exists()  # moved out of the working tier
    assert (work / "archive" / "000001-stale_probe.py").is_file()
    assert fresh.is_file()  # the fresh draft stays put at the working-tier top level


def test_prune_archives_stale_candidate_too(tmp_path):
    work = tmp_path / "__tmp"
    _draft_file(work, "cand_probe", status="candidate", age_hours=5)

    assert prune_stale_drafts(work, ttl_hours=1) == ["cand_probe"]
    assert (work / "archive" / "000001-cand_probe.py").is_file()


def test_prune_spares_rejected_and_champion_status(tmp_path):
    work = tmp_path / "__tmp"
    _draft_file(work, "old_reject", status="rejected", age_hours=9)
    _draft_file(work, "old_champ", status="champion", age_hours=9)

    assert prune_stale_drafts(work, ttl_hours=1) == []
    assert (work / "old_reject.py").is_file()
    assert (work / "old_champ.py").is_file()
    assert not (work / "archive").exists()  # nothing archived ⇒ area never materializes


def test_prune_never_scans_subdirectories(tmp_path):
    work = tmp_path / "__tmp"
    # A stale attempt persisted under failed/ and a pre-existing archived file both sit in
    # subdirectories — discovery globs a tier top-level only, and so must prune.
    failed = _draft_file(work / "failed", "000001-old_attempt", status="draft", age_hours=99)
    prior = _draft_file(work / "archive", "000001-earlier", status="draft", age_hours=99)
    failed_before = failed.read_bytes()
    prior_before = prior.read_bytes()
    _draft_file(work, "live_draft", age_hours=9)

    assert prune_stale_drafts(work, ttl_hours=1) == ["live_draft"]
    assert failed.read_bytes() == failed_before  # failed/ contents untouched
    assert prior.read_bytes() == prior_before  # existing archive/ contents untouched


def test_prune_leaves_other_tiers_and_journals_untouched(tmp_path):
    from noctis.strategies.library import LibraryPaths

    seeds = tmp_path / "seeds"
    ws = tmp_path / "workspace" / "strategies"
    paths = LibraryPaths(seeds=seeds, tmp=ws / "__tmp", champions=ws / "champions")
    _draft_file(seeds, "seeded", status="draft", age_hours=99)  # a committed seed
    _draft_file(paths.champions, "crowned", status="champion", age_hours=99)
    journal = tmp_path / "state" / "experiments" / "stale_probe.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text('{"trial": 1}\n', encoding="utf-8")
    _draft_file(paths.tmp, "stale_probe", age_hours=9)

    seeds_before = _snapshot(seeds)
    champions_before = _snapshot(paths.champions)
    journal_before = journal.read_bytes()

    assert prune_stale_drafts(paths.tmp, ttl_hours=1) == ["stale_probe"]
    assert _snapshot(seeds) == seeds_before  # seeds tier byte-identical
    assert _snapshot(paths.champions) == champions_before  # champions tier byte-identical
    assert journal.read_bytes() == journal_before  # experiment journals byte-identical


def test_archived_file_leaves_discovery_and_seed_unshadows(tmp_path):
    from noctis.strategies.library import LibraryPaths

    seeds = tmp_path / "seeds"
    ws = tmp_path / "workspace" / "strategies"
    paths = LibraryPaths(seeds=seeds, tmp=ws / "__tmp", champions=ws / "champions")
    seed = _draft_file(seeds, "probe", status="draft", age_hours=0)  # committed seed
    _draft_file(paths.tmp, "probe", status="draft", age_hours=9)  # stale working copy shadows it
    assert strategy_path(paths, "probe") == paths.tmp / "probe.py"  # tmp shadows seed

    assert prune_stale_drafts(paths.tmp, ttl_hours=1) == ["probe"]
    assert strategy_path(paths, "probe") == seed  # the seed un-shadows once the draft is archived


def test_archive_collision_sequences_both_files(tmp_path):
    work = tmp_path / "__tmp"
    _draft_file(work, "probe", age_hours=9)
    assert prune_stale_drafts(work, ttl_hours=1) == ["probe"]
    # A fresh draft of the SAME name is authored later and also goes stale.
    _draft_file(work, "probe", age_hours=9)
    assert prune_stale_drafts(work, ttl_hours=1) == ["probe"]

    archived = sorted((work / "archive").glob("*.py"))
    assert [p.name for p in archived] == ["000001-probe.py", "000002-probe.py"]


def test_archive_cap_evicts_the_oldest_sequence(tmp_path):
    work = tmp_path / "__tmp"
    for i in range(5):
        _draft_file(work, f"probe_{i}", age_hours=9)
        prune_stale_drafts(work, ttl_hours=1, cap=3)

    seqs = sorted(int(p.name.split("-", 1)[0]) for p in (work / "archive").glob("*.py"))
    assert seqs == [3, 4, 5]  # sequences 1 and 2 evicted, numbering never reused


def test_ttl_zero_or_none_is_byte_identical_noop(tmp_path):
    work = tmp_path / "__tmp"
    _draft_file(work, "stale_probe", age_hours=99)
    before = _snapshot(work)

    assert prune_stale_drafts(work, ttl_hours=0) == []
    assert _snapshot(work) == before  # byte-identical
    assert prune_stale_drafts(work, ttl_hours=None) == []
    assert _snapshot(work) == before
    assert not (work / "archive").exists()  # the disable path never materializes an archive


def test_prune_never_restamps_header_or_writes_a_verdict(tmp_path):
    work = tmp_path / "__tmp"
    original = _draft_file(work, "stale_probe", age_hours=9).read_bytes()

    prune_stale_drafts(work, ttl_hours=1)

    archived = work / "archive" / "000001-stale_probe.py"
    assert archived.read_bytes() == original  # header intact — status still 'draft', not re-stamped
    # Exactly one file landed in archive/ (the moved draft); no rejection record, no verdict.
    assert [p.name for p in (work / "archive").glob("*")] == ["000001-stale_probe.py"]


def test_prune_on_missing_directory_is_a_noop(tmp_path):
    assert prune_stale_drafts(tmp_path / "does_not_exist", ttl_hours=1) == []
