"""The honesty block, the versioning promise, and ``run-record --validate`` (story #143).

Three things that only mean something together:

* **``assumptions``** — the arena a run's numbers were produced in, machine-readable so a website
  renders it as a table and a diff between two runs is meaningful. Every value in it is *derived*
  from what actually drives behaviour (the run's frozen settings, the constants the pipeline and
  the benchmark implement), never hand-copied — and where a value has to be named here, a test
  in this file keeps it tracking the thing it names.
* **The gate is measured, not claimed.** ``paper_only`` and ``real_orders_reachable`` come off the
  gate's own resolved verdict (AGENTS.md rule 1). Tests below flip the gate and watch them follow.
* **The schema is a promise.** Additive-only, so an unknown key never invalidates a record; and an
  older record resumed by a newer engine is upgraded in place with the upgrade on the record. The
  upgrade path is exercised with a **synthetic** version 2, because a mechanism that only works
  the day someone bumps the real version is a mechanism nobody has tested.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.config import load_settings
from noctis.config.gate import SafetyGateError, resolve_execution_mode
from noctis.config.rehydrate import freeze_inputs
from noctis.reporting import schema
from noctis.reporting.run_record import build

from .test_run_record import _artifacts

runner = CliRunner()

FROZEN_AT = "2026-07-27T14:22:33.418Z"


def _inputs(**overrides) -> dict:
    """One run's frozen configuration, as the composition root freezes it at creation.

    Built from a real :class:`Settings` — the assumptions block's whole point is that it states
    what the engine was actually configured with, so a test that hand-wrote the block's values
    would be testing a transcription instead of a derivation.
    """
    settings = load_settings(**overrides)
    return freeze_inputs(
        settings,
        execution_mode=resolve_execution_mode(settings),
        frozen_at=FROZEN_AT,
    )


def _assumptions(**overrides) -> dict:
    return build(_artifacts(inputs=_inputs(**overrides)))["assumptions"]


# ── the block exists, on every record ──────────────────────────────────────────────────────


def test_every_record_carries_an_assumptions_block():
    record = build(_artifacts())

    assert "assumptions" in schema.REQUIRED_SECTIONS
    assert isinstance(record["assumptions"], dict)
    assert schema.validate(record) == []


def test_the_validator_names_an_assumptions_block_missing_a_key():
    record = build(_artifacts(inputs=_inputs()))
    del record["assumptions"]["fee_bps"]

    assert any("assumptions.fee_bps" in problem for problem in schema.validate(record))


def test_a_record_with_no_assumptions_block_at_all_is_a_schema_violation():
    record = build(_artifacts())
    del record["assumptions"]

    assert any("assumptions" in problem for problem in schema.validate(record))


# ── the fill model and the no-lookahead rule ───────────────────────────────────────────────


def test_the_assumptions_state_the_fill_model_and_the_no_lookahead_rule():
    block = _assumptions()

    assert block["fill_model"] == "next_bar_open"
    assert "t+1" in block["lookahead"]


def test_the_stated_fill_model_is_the_one_the_simulator_actually_implements():
    """The one value here that is a *name* rather than a derived number, kept true by behaviour:
    a decision on bar t fills at bar t+1's open, so the name is checked against a real fill."""
    import pandas as pd

    from noctis.broker import FeeModel, PaperBroker, SlippageModel, simulate
    from noctis.strategies import SmaCrossover

    class LongFromTheStart(SmaCrossover):
        """Long on every bar — so the first fill is the earliest one the engine allows."""

        def on_bar(self, ctx, bar) -> None:
            ctx.set_target(1)

    bars = pd.DataFrame(
        {
            "ts_event": [0, 60_000_000_000, 120_000_000_000],
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [11.0, 21.0, 31.0],
            "volume": [1000, 1000, 1000],
        }
    )
    broker = PaperBroker(fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0))

    result = simulate(LongFromTheStart.create(fast=2, slow=3), bars, broker, symbol="TST")

    # The decision was taken on bar 0 and filled at bar 1's OPEN — not bar 0's close (11.0),
    # not bar 1's close (21.0). That is what ``fill_model: next_bar_open`` names.
    assert result.fills[0].price == 20.0
    assert _assumptions()["fill_model"] == "next_bar_open"


def test_the_no_lookahead_claim_is_the_geometry_the_splitter_produces():
    from noctis.backtest.splits import walk_forward

    block = _assumptions()["walk_forward"]
    splits = walk_forward(400, 120, 40, 40)

    assert splits
    assert all(split.test_start == split.train_end for split in splits)  # embargo_bars: 0
    assert block["embargo_bars"] == 0
    assert block["test_after_train"] is True


# ── the cost model ─────────────────────────────────────────────────────────────────────────


def test_the_assumptions_state_the_fee_and_slippage_the_run_was_charged():
    settings = load_settings()

    block = _assumptions()

    assert block["fee_bps"] == settings.backtest.fee_bps
    assert block["slippage_bps"] == settings.backtest.slippage_bps
    assert block["round_trip_cost_bps"] == 2 * (
        settings.backtest.fee_bps + settings.backtest.slippage_bps
    )


def test_the_cost_model_tracks_the_configuration_rather_than_a_copy_of_it():
    harsher = _assumptions(backtest={"fee_bps": 7.5, "slippage_bps": 3.5})

    assert harsher["fee_bps"] == 7.5
    assert harsher["slippage_bps"] == 3.5
    assert harsher["round_trip_cost_bps"] == 22.0


def test_a_run_that_froze_no_configuration_states_null_costs_rather_than_the_defaults():
    """An adopted history never froze an arena, so it cannot state one — and must not pretend
    the shipped defaults were what it ran under."""
    block = build(_artifacts())["assumptions"]

    assert block["fee_bps"] is None
    assert block["slippage_bps"] is None
    assert block["round_trip_cost_bps"] is None
    # The engine-level facts are still stated: they are the same in every run.
    assert block["fill_model"] == "next_bar_open"


# ── the walk-forward geometry and both holdouts ────────────────────────────────────────────


def test_the_assumptions_state_the_walk_forward_geometry():
    block = _assumptions()["walk_forward"]

    assert set(block) == {
        "sizing",
        "min_train_bars",
        "max_train_bars",
        "min_test_bars",
        "max_test_bars",
        "step_bars",
        "embargo_bars",
        "test_after_train",
    }
    assert block["sizing"] == "auto"
    assert block["step_bars"] is None  # null = one test window; test windows never overlap


def test_the_stated_geometry_tracks_the_pipeline_that_produces_it():
    """The bounds are named here, so a test holds them to the heuristic that actually sizes a
    split — including at both ends, so moving a bound is a red test rather than slack."""
    from noctis.backtest.pipeline import PipelineConfig

    block = _assumptions()["walk_forward"]

    for bars in (60, 120, 240, 600, 5_000, 250_000):
        config = PipelineConfig.auto(bars)
        assert block["min_train_bars"] <= config.train_size <= block["max_train_bars"], bars
        assert block["min_test_bars"] <= config.test_size <= block["max_test_bars"], bars
        assert config.step == config.test_size, bars  # step_bars: null
    assert PipelineConfig.auto(3).train_size == block["min_train_bars"]
    assert PipelineConfig.auto(3).test_size == block["min_test_bars"]
    assert PipelineConfig.auto(1_000_000).train_size == block["max_train_bars"]
    assert PipelineConfig.auto(1_000_000).test_size == block["max_test_bars"]


def test_the_assumptions_state_the_forward_holdout_that_is_reserved():
    from noctis.backtest.pipeline import PipelineConfig

    block = _assumptions()["forward_holdout"]

    assert block["reserved"] is True
    assert block["note"]
    for bars in (60, 240, 5_000):
        config = PipelineConfig.auto(bars)
        assert config.holdout_size in (0, config.test_size), bars
        if config.holdout_size:
            assert block["min_bars"] <= config.holdout_size <= block["max_bars"], bars


def test_the_assumptions_state_the_symbol_holdout_the_run_reserves():
    settings = load_settings()

    block = _assumptions()["symbol_holdout"]

    assert block["size"] == settings.research.symbol_holdout_size
    assert block["fit_set_size"] == settings.research.fit_set_size
    # Which names were held out is a per-session sampling decision, not a run-level fact.
    assert block["symbols"] is None


def test_the_symbol_holdout_tracks_the_configuration():
    block = _assumptions(research={"symbol_holdout_size": 5, "fit_set_size": 9})["symbol_holdout"]

    assert block["size"] == 5
    assert block["fit_set_size"] == 9


# ── the exhaustion gate and every promotion threshold ──────────────────────────────────────


def test_the_assumptions_state_the_exhaustion_gates_min_trials():
    assert _assumptions()["min_trials"] == load_settings().research.min_trials
    assert _assumptions(research={"min_trials": 41})["min_trials"] == 41


def test_the_assumptions_carry_every_promotion_threshold_the_gates_actually_use():
    """Derived from the run's own promotion settings, so a threshold added tomorrow lands here
    without an edit — and this test proves the block is the *rules object*, field for field."""
    from noctis.champions.promotion import PromotionRules

    settings = load_settings()
    rules = PromotionRules.from_settings(settings)

    block = build(_artifacts(inputs=_inputs()))["assumptions"]["promotion_thresholds"]

    for field in dataclasses.fields(PromotionRules):
        assert block[field.name] == getattr(rules, field.name), field.name


def test_the_promotion_thresholds_also_name_the_metric_the_run_elected():
    block = _assumptions(promotion={"metric": "sortino"})["promotion_thresholds"]

    assert block["metric"] == "sortino"


def test_the_promotion_thresholds_track_a_tightened_gate():
    block = _assumptions(promotion={"max_gap": 0.25, "min_holdout_metric": 0.4})

    assert block["promotion_thresholds"]["max_gap"] == 0.25
    assert block["promotion_thresholds"]["min_holdout_metric"] == 0.4


def test_a_run_that_froze_no_configuration_states_null_thresholds():
    block = build(_artifacts())["assumptions"]

    assert block["promotion_thresholds"] is None
    assert block["min_trials"] is None
    assert block["symbol_holdout"]["size"] is None


# ── the benchmark's rebalancing convention ─────────────────────────────────────────────────


def test_the_assumptions_state_the_benchmarks_rebalancing_convention():
    from noctis.reporting.metrics import BENCHMARK_METHOD, BENCHMARK_NAME

    block = _assumptions()["benchmark"]

    assert block["name"] == BENCHMARK_NAME
    assert block["method"] == BENCHMARK_METHOD
    assert "never rebalanced" in block["rebalancing"]
    assert "first session" in block["rebalancing"]


def test_the_stated_rebalancing_convention_is_the_one_the_benchmark_implements():
    """Weights are set at the first session mark and drift thereafter — so a basket whose two
    names diverge lands on the mean of their *ratios*, never on a re-weighted path."""
    from noctis.reporting.run_store import _equal_weight_levels

    closes = {
        "AAA": {"2026-07-01": 100.0, "2026-07-02": 200.0, "2026-07-03": 400.0},
        "BBB": {"2026-07-01": 100.0, "2026-07-02": 50.0, "2026-07-03": 25.0},
    }

    levels = _equal_weight_levels(closes, ["2026-07-01", "2026-07-02", "2026-07-03"])

    # No rebalance: (2.0 + 0.5)/2 then (4.0 + 0.25)/2. A daily-rebalanced basket would have
    # compounded 1.25 × 1.25 = 1.5625 instead.
    assert [level for _day, level in levels] == [1.0, 1.25, 2.125]
    assert "never rebalanced" in _assumptions()["benchmark"]["rebalancing"]


# ── the live gate: measured, never claimed ─────────────────────────────────────────────────


def test_paper_only_is_read_off_the_gates_own_resolved_verdict():
    block = _assumptions()

    assert block["paper_only"] is True
    assert block["live_gate"]["execution_mode"] == "paper"
    assert block["live_gate"]["real_orders_reachable"] is False
    assert block["live_gate"]["re_resolved_each_segment"] is True


def test_the_gate_state_tracks_the_gate_rather_than_being_hardcoded():
    """The proof the criterion asks for: open BOTH live-money gates and the block flips. A
    hardcoded ``paper_only: true`` passes every other test in this file and fails this one."""
    live = _assumptions(mode="live", allow_live=True)
    paper = _assumptions()

    assert live["paper_only"] is False
    assert live["live_gate"]["execution_mode"] == "live"
    assert live["live_gate"]["real_orders_reachable"] is True
    assert paper["paper_only"] is True
    assert paper["live_gate"]["real_orders_reachable"] is False


def test_opening_only_the_environment_gate_keeps_the_run_paper_only():
    """Config wins toward safety: ``ALLOW_LIVE`` alone arms nothing, and the block says so."""
    block = _assumptions(allow_live=True)

    assert block["paper_only"] is True
    assert block["live_gate"]["real_orders_reachable"] is False


def test_a_config_mode_of_live_without_the_environment_gate_never_reaches_a_record():
    """The other half of rule 1: the gate refuses to resolve at all, so no record is written
    claiming a half-open gate."""
    settings = load_settings(mode="live")

    with pytest.raises(SafetyGateError):
        resolve_execution_mode(settings)


def test_a_run_that_froze_no_gate_verdict_says_unknown_rather_than_claiming_paper():
    block = build(_artifacts())["assumptions"]

    assert block["paper_only"] is None
    assert block["live_gate"]["execution_mode"] is None
    assert block["live_gate"]["real_orders_reachable"] is None


def test_the_assumptions_are_never_a_second_source_for_the_two_gates():
    """The block states the gate's *verdict*, never the two settings behind it: the pair must
    have exactly two independent sources, and a record is neither of them."""
    record = build(_artifacts(inputs=_inputs()))

    serialized = json.dumps(record["assumptions"])
    assert '"mode"' not in serialized
    assert '"allow_live"' not in serialized
    assert schema.validate(record) == []


def test_the_validator_refuses_a_block_that_smuggled_a_gate_in():
    """Structurally refused, not merely absent by convention: a record carrying either gate would
    offer a third source for a decision that must have exactly two independent ones."""
    record = build(_artifacts(inputs=_inputs()))
    record["assumptions"]["live_gate"]["allow_live"] = True
    record["assumptions"]["mode"] = "live"

    problems = schema.validate(record)

    assert any("allow_live" in problem for problem in problems)
    assert any("mode" in problem for problem in problems)


# ── the additive-only rule ─────────────────────────────────────────────────────────────────


def test_a_record_carrying_unknown_keys_still_validates():
    """Additive-only, from the reader's side: a record written by a *newer* Noctis carries keys
    this one has never heard of, and must still be readable rather than rejected."""
    record = build(_artifacts(inputs=_inputs()))
    record["some_future_section"] = {"whatever": 1}
    record["run"]["future_total_s"] = 12.0
    record["segments"][0]["future_measurement"] = None
    record["assumptions"]["future_assumption"] = "stated later"

    assert schema.validate(record) == []


# ── schema versioning and the upgrade path ─────────────────────────────────────────────────


def test_the_schema_declares_its_version_and_the_record_carries_it():
    record = build(_artifacts())

    assert record["schema_version"] == schema.SCHEMA_VERSION
    assert isinstance(schema.SCHEMA_VERSION, int)


def test_a_record_already_at_the_current_version_is_left_exactly_alone():
    record = build(_artifacts(inputs=_inputs()))

    outcome = schema.upgrade(record)

    assert outcome.upgraded is False
    assert outcome.record == record
    assert outcome.note() is None


def test_a_version_1_record_read_by_a_version_2_engine_is_upgraded_in_place():
    """The criterion, exercised against a **synthetic** version 2 — the real
    ``SCHEMA_VERSION`` is not bumped to make a test possible."""
    record = build(_artifacts(inputs=_inputs()))
    record["schema_version"] = 1

    def fill_the_new_key(document: dict) -> dict:
        return {**document, "later_section": None}

    outcome = schema.upgrade(record, upgrades={1: fill_the_new_key}, target=2)

    assert outcome.upgraded is True
    assert outcome.from_version == 1
    assert outcome.to_version == 2
    assert outcome.record["schema_version"] == 2
    assert outcome.record["later_section"] is None
    assert record["schema_version"] == 1  # pure: the input document is not mutated


def test_the_upgrade_says_what_it_did_so_the_record_can_record_it():
    record = build(_artifacts())
    record["schema_version"] = 1

    outcome = schema.upgrade(record, upgrades={}, target=3)

    note = outcome.note()
    assert note is not None
    assert "1" in note and "3" in note


def test_an_upgrade_walks_every_intermediate_version_in_order():
    record = build(_artifacts())
    record["schema_version"] = 1
    steps: list[int] = []

    def step(version: int):
        def apply(document: dict) -> dict:
            steps.append(version)
            return document

        return apply

    outcome = schema.upgrade(record, upgrades={1: step(1), 2: step(2)}, target=3)

    assert steps == [1, 2]
    assert outcome.record["schema_version"] == 3


def test_a_record_from_the_future_is_never_downgraded():
    """A reader ignores what it does not know; it never rewrites a newer record to look older."""
    record = build(_artifacts())
    record["schema_version"] = 99

    outcome = schema.upgrade(record, upgrades={}, target=1)

    assert outcome.upgraded is False
    assert outcome.record["schema_version"] == 99


def test_a_record_with_no_version_at_all_is_upgraded_from_the_first_one():
    record = build(_artifacts())
    del record["schema_version"]

    outcome = schema.upgrade(record, upgrades={}, target=2)

    assert outcome.upgraded is True
    assert outcome.from_version is None
    assert outcome.record["schema_version"] == 2


def test_resuming_an_older_record_upgrades_it_in_place_and_records_the_upgrade(
    tmp_path, monkeypatch
):
    """End to end through the store, with the version faked forward: the next segment's write
    leaves a version-2 record on disk carrying an event that says where it came from."""
    from datetime import UTC, datetime, timedelta

    from noctis.reporting.run_store import RUN_RECORD_NAME, open_run

    class Clock:
        def __init__(self) -> None:
            self.now = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)

        def __call__(self) -> datetime:
            return self.now

    clock = Clock()
    runs = tmp_path / "runs"
    first = open_run(runs, clock=clock, argv=["run"], election_metric="sharpe")
    first.close(reason="stop_requested")
    on_disk = json.loads((first.run_dir / RUN_RECORD_NAME).read_text())
    assert on_disk["schema_version"] == schema.SCHEMA_VERSION

    monkeypatch.setattr(schema, "SCHEMA_VERSION", schema.SCHEMA_VERSION + 1)
    clock.now += timedelta(hours=1)
    second = open_run(
        runs, clock=clock, argv=["run"], election_metric="sharpe", run_id=first.run_id, resume=True
    )
    second.close(reason="stop_requested")

    upgraded = json.loads((second.run_dir / RUN_RECORD_NAME).read_text())
    assert upgraded["schema_version"] == schema.SCHEMA_VERSION
    assert any("schema" in event["text"] for event in upgraded["events"]), upgraded["events"]


def test_a_resume_that_changes_no_version_records_no_upgrade(tmp_path):
    from datetime import UTC, datetime, timedelta

    from noctis.reporting.run_store import RUN_RECORD_NAME, open_run

    class Clock:
        def __init__(self) -> None:
            self.now = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)

        def __call__(self) -> datetime:
            return self.now

    clock = Clock()
    runs = tmp_path / "runs"
    first = open_run(runs, clock=clock, argv=["run"], election_metric="sharpe")
    first.close(reason="stop_requested")
    clock.now += timedelta(hours=1)
    second = open_run(
        runs, clock=clock, argv=["run"], election_metric="sharpe", run_id=first.run_id, resume=True
    )
    second.close(reason="stop_requested")

    record = json.loads((second.run_dir / RUN_RECORD_NAME).read_text())
    assert not [event for event in record["events"] if "schema" in event["text"]]


# ── the contract-wide conventions, enforced structurally ───────────────────────────────────


def test_every_dimensioned_field_in_a_full_record_names_its_unit_canonically():
    record = build(_artifacts(inputs=_inputs()))

    assert schema.validate(record) == []


def test_the_validator_names_a_field_that_spells_its_unit_the_long_way():
    record = build(_artifacts())
    record["run"]["cumulative_runtime_seconds"] = 3600.0
    record["assumptions"]["fee_basis_points"] = 1.0

    problems = schema.validate(record)

    assert any("cumulative_runtime_seconds" in problem for problem in problems)
    assert any("fee_basis_points" in problem for problem in problems)


def test_the_unit_rule_leaves_a_plain_count_alone():
    """Only *dimensioned* fields must name a unit: a count of trials is not seconds."""
    record = build(_artifacts())
    record["run"]["cumulative_trials"] = 41

    assert schema.validate(record) == []


def test_every_timestamp_anywhere_in_the_record_is_utc_iso_8601_with_a_z():
    record = build(_artifacts(inputs=_inputs()))
    record["inputs"]["frozen_at_utc"] = "2026-07-27 14:22:33"

    assert any("frozen_at_utc" in problem for problem in schema.validate(record))


def test_a_stamp_in_a_section_the_validator_has_no_rule_for_is_still_checked():
    record = build(_artifacts())
    record["assumptions"]["measured_at_utc"] = "27/07/2026"

    assert any("measured_at_utc" in problem for problem in schema.validate(record))


def test_a_known_absent_assumption_is_an_explicit_null_never_an_omitted_key():
    block = build(_artifacts())["assumptions"]

    for key in ("paper_only", "fee_bps", "slippage_bps", "min_trials", "promotion_thresholds"):
        assert key in block
        assert block[key] is None


# ── the schema module itself ───────────────────────────────────────────────────────────────


def test_the_schema_exposes_a_pure_validate_returning_a_list_of_problems():
    record = build(_artifacts())
    before = json.dumps(record, sort_keys=True)

    problems = schema.validate(record)

    assert problems == []
    assert isinstance(problems, list)
    assert json.dumps(record, sort_keys=True) == before  # validate mutates nothing
    assert "validate" in schema.__all__
    assert all(isinstance(problem, str) for problem in schema.validate({}))


def test_the_schema_module_stays_pure_stdlib():
    import ast

    source = Path(schema.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "collections", "dataclasses", "typing"}


# ── noctis run-record --validate ───────────────────────────────────────────────────────────


def _config(tmp_path) -> str:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n")
    return str(cfg)


def _runs_dir(tmp_path) -> Path:
    return tmp_path / "workspace" / "runs"


def test_run_record_validate_reports_a_schema_valid_record(tmp_path):
    assert runner.invoke(app, ["run", "--config", _config(tmp_path)]).exit_code == 0
    run_dir = next(path for path in _runs_dir(tmp_path).iterdir() if path.is_dir())

    result = runner.invoke(
        app, ["run-record", run_dir.name, "--validate", "--config", _config(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower()


def test_run_record_validate_exits_non_zero_and_names_every_problem(tmp_path):
    assert runner.invoke(app, ["run", "--config", _config(tmp_path)]).exit_code == 0
    run_dir = next(path for path in _runs_dir(tmp_path).iterdir() if path.is_dir())
    record = json.loads((run_dir / "run.json").read_text())
    del record["assumptions"]["fee_bps"]
    record["run"]["created_utc"] = "2026-07-27 14:22:33"
    (run_dir / "run.json").write_text(json.dumps(record))

    result = runner.invoke(
        app, ["run-record", run_dir.name, "--validate", "--config", _config(tmp_path)]
    )

    assert result.exit_code == 1
    assert "assumptions.fee_bps" in result.output
    assert "created_utc" in result.output


def test_run_record_validate_prints_the_verdict_rather_than_the_whole_document(tmp_path):
    assert runner.invoke(app, ["run", "--config", _config(tmp_path)]).exit_code == 0
    run_dir = next(path for path in _runs_dir(tmp_path).iterdir() if path.is_dir())

    result = runner.invoke(
        app, ["run-record", run_dir.name, "--validate", "--config", _config(tmp_path)]
    )

    assert '"schema_version"' not in result.output


def test_run_record_validate_on_an_unreadable_record_exits_non_zero(tmp_path):
    broken = _runs_dir(tmp_path) / "20260102T000000Z-brokn0"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{ not json")

    result = runner.invoke(
        app, ["run-record", broken.name, "--validate", "--config", _config(tmp_path)]
    )

    assert result.exit_code == 1


# ── a real run's record states a real arena ────────────────────────────────────────────────


def test_a_real_run_publishes_the_arena_it_actually_ran_in(tmp_path):
    """End to end: the block on a record a bare ``noctis run`` left behind is populated from the
    settings that run resolved, and the gate state is the one the gate reached."""
    assert runner.invoke(app, ["run", "--config", _config(tmp_path)]).exit_code == 0
    run_dir = next(path for path in _runs_dir(tmp_path).iterdir() if path.is_dir())
    record = json.loads((run_dir / "run.json").read_text())

    block = record["assumptions"]
    assert block["paper_only"] is True
    assert block["live_gate"]["real_orders_reachable"] is False
    assert block["fee_bps"] is not None
    assert block["slippage_bps"] is not None
    assert block["min_trials"] is not None
    assert block["promotion_thresholds"]["max_gap"] is not None
    assert schema.validate(record) == []
