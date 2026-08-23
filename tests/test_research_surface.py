"""The research-toolbox surface (epic #255, story #257): the two-tier Protocol and the derived
facts a briefing, the system prompt, a driver or an eval site reads.

Every assertion here is external behaviour on a **real** toolbox built through the tools tests'
own builder, with state seeded through the collaborators (journal, registry, memory, lake,
library) exactly as a session seeds it — never through a stubbed surface. That is the point of
the story: the seam is only worth declaring if the object that answers it is the production one.

Two kinds of assertion carry the refactor that follows:

* **parity** — each derived fact is compared against the renderer the consumers still call today
  (``digests.champion_digest`` / ``crowned_families`` / ``library_index`` / ``memory_block`` /
  ``lake_inventory``, ``briefings._decide_evidence``, ``briefings._spend_context``). When story
  #258 re-points the consumers at the surface, these are what say nothing moved.
* **leafness** — the surface module imports nothing from the engine, so the eval layer may import
  it without crossing the one-way boundary ``tests/test_eval_boundary.py`` guards.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from noctis.research import digests, surface
from noctis.research.briefings import _TOP_TRIALS, _decide_evidence, _spend_context
from noctis.research.journal import TOP_TRIALS, evidence_block
from noctis.research.surface import (
    ChampionBoard,
    ResearchFacts,
    ResearchLimits,
    SessionCounters,
    Toolbox,
)
from noctis.strategies import library
from tests.test_briefings import _populate
from tests.test_champions import make_scorecard
from tests.test_research_tools import _make_toolbox

# The evidence dict's key set, spelled out once: the DECIDE briefing and get_experiment_log both
# reason on these names, so a renamed key is a silent change to what the model is told.
_EVIDENCE_KEYS = frozenset(
    {
        "strategy",
        "thesis",
        "class_tag",
        "n_trials",
        "n_distinct_params",
        "sweep_completed",
        "min_trials_gate",
        "top_trials",
        "verdicts",
        "tuned_off_limits_for_holdout",
    }
)


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """This module exercises the surface, not subprocess write-gate isolation."""


# ── the seam itself ─────────────────────────────────────────────────────────────────────────
def test_the_real_toolbox_satisfies_both_tiers_of_the_protocol(tmp_path):
    box = _make_toolbox(tmp_path)

    assert isinstance(box, ResearchFacts)
    assert isinstance(box, Toolbox)


def test_the_surface_module_imports_nothing_from_the_engine():
    """The eval layer imports this module, and the engine never imports the eval layer — so the
    surface has to be a leaf: typing and dataclasses, no ``noctis.*`` import at all."""
    tree = ast.parse(Path(surface.__file__).read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert [name for name in sorted(imported) if name.startswith("noctis")] == []


# ── limits ──────────────────────────────────────────────────────────────────────────────────
def test_limits_are_the_four_configured_ceilings(tmp_path):
    box = _make_toolbox(tmp_path, min_trials=4, max_backtests=7, max_author_calls=5)

    assert box.limits == ResearchLimits(
        min_trials=4, max_backtests=7, sweep_trials=3, max_author_calls=5
    )
    # The scalar attributes the toolbox's own gates read stay, and stay in agreement.
    assert box.limits == ResearchLimits(
        min_trials=box.min_trials,
        max_backtests=box.max_backtests,
        sweep_trials=box.default_sweep_trials,
        max_author_calls=box.max_author_calls,
    )


# ── journal evidence: one builder ───────────────────────────────────────────────────────────
def test_journal_evidence_is_the_decide_briefing_evidence_key_for_key(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    evidence = box.journal_evidence("probe")

    assert set(evidence) == _EVIDENCE_KEYS
    assert evidence == _decide_evidence(box, "probe")


def test_journal_evidence_reports_the_seeded_journal(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    evidence = box.journal_evidence("probe")

    assert evidence["strategy"] == "probe"
    assert evidence["class_tag"] == "intraday momentum"
    assert evidence["n_trials"] == 3
    assert evidence["min_trials_gate"] == box.min_trials
    assert evidence["tuned_off_limits_for_holdout"] == ["AAA", "BBB"]
    # Ranked best-first by test metric — the order the verdict episode reads them in.
    assert [row["test"] for row in evidence["top_trials"]] == [1.41, 0.92, 0.55]


def test_evidence_block_is_a_pure_function_of_a_journal(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    assert evidence_block(box.journal, "probe", min_trials=box.min_trials) == box.journal_evidence(
        "probe"
    )


def test_evidence_caps_the_ranked_trials_at_the_top_trials_constant(tmp_path):
    box = _make_toolbox(tmp_path)
    for i in range(TOP_TRIALS + 4):
        box.journal.record_trial(
            "probe",
            source="sweep",
            symbols=["AAA"],
            params={"lookback": 5 + i},
            window={"train": 200, "test": 100},
            card=make_scorecard("probe", test_metric=0.1 * i, train_metric=0.2 * i),
        )

    assert len(box.journal_evidence("probe")["top_trials"]) == TOP_TRIALS


def test_the_briefing_and_the_journal_state_the_same_top_trials_cap():
    """The DECIDE briefing still carries its own copy of the cap (its module is a prompt-ratchet
    asset this story does not touch — story #258 deletes the copy and imports this one). Until it
    does, the two are pinned equal here, so the duplicate cannot quietly drift into showing the
    verdict episode a different depth of leaderboard than ``get_experiment_log`` shows."""
    assert _TOP_TRIALS == TOP_TRIALS


def test_get_experiment_log_rows_are_the_evidence_top_trials_prefix(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    log = box.tool_get_experiment_log("probe", limit=2)

    assert log["top_trials"] == box.journal_evidence("probe")["top_trials"][:2]
    # The log's own keys are untouched by the shared row renderer.
    assert set(log) == {
        "strategy",
        "n_trials",
        "n_distinct_params",
        "sweep_completed",
        "min_trials_gate",
        "top_trials",
        "verdicts",
    }


# ── champion board ──────────────────────────────────────────────────────────────────────────
def test_champion_board_matches_the_digest_renderers_over_the_registry(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    board = box.champion_board()

    assert isinstance(board, ChampionBoard)
    assert list(board.rows) == digests.champion_digest(box.registry)
    assert list(board.crowned_families) == digests.crowned_families(box.registry)
    assert board.capacity == box.registry.capacity == 3
    assert list(board.crowned_families) == ["alpha_mom", "gamma_break"]


# ── library index ───────────────────────────────────────────────────────────────────────────
def test_library_index_collapses_a_rejected_entry_to_a_stub(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    index = box.library_index()

    assert index == digests.library_index(box.strategies_dir)
    entries = {entry["name"]: entry for entry in index}
    assert entries["corpse"] == {"name": "corpse", "status": "rejected"}
    assert "thesis" in entries["probe"]


# ── template text ───────────────────────────────────────────────────────────────────────────
def test_template_text_is_the_seed_template_and_none_when_absent(tmp_path):
    box = _make_toolbox(tmp_path)

    assert box.template_text() == "(none)"

    seeds = box.strategies_dir.seeds
    seeds.mkdir(parents=True, exist_ok=True)
    (seeds / library.TEMPLATE_NAME).write_text("# adapt, do not copy\n", encoding="utf-8")

    assert box.template_text() == "# adapt, do not copy\n"


# ── memory tail ─────────────────────────────────────────────────────────────────────────────
def test_memory_tail_matches_the_memory_block_renderer_and_prefix_trim_caps_it(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)
    for i in range(9):  # ten distinct dead-end families in all, past the trimmed cap of five
        box.memory.record_rejected(f"deadfam{i}", {"lookback": i}, reason="below cost")

    assert box.memory_tail() == digests.memory_block(box.memory)
    assert box.memory_tail(prefix_trim=True) == digests.memory_block(box.memory, prefix_trim=True)

    _full_findings, full_dead_ends = box.memory_tail()
    _trim_findings, trim_dead_ends = box.memory_tail(prefix_trim=True)
    assert len(full_dead_ends) == 10
    assert len(trim_dead_ends) == 5


# ── lake inventory ──────────────────────────────────────────────────────────────────────────
def test_lake_inventory_is_sorted_and_matches_the_shared_builder(tmp_path):
    box = _make_toolbox(tmp_path)

    assert box.lake_inventory() == ["AAA", "BBB", "CCC", "DDD"]
    assert box.lake_inventory() == digests.lake_inventory(box)


def test_lake_inventory_drops_a_symbol_the_lake_is_not_ready_for(tmp_path):
    box = _make_toolbox(tmp_path)
    box.lake.bars.pop("CCC")  # tracked in the universe, no history in the lake

    assert box.lake_inventory() == ["AAA", "BBB", "DDD"]


def test_lake_inventory_stops_at_the_limit(tmp_path):
    box = _make_toolbox(tmp_path)

    assert box.lake_inventory(limit=2) == ["AAA", "BBB"]


def test_lake_inventory_is_empty_when_the_coverage_listing_raises(tmp_path):
    box = _make_toolbox(tmp_path)

    def explode():
        raise RuntimeError("coverage registry unreadable")

    box.lake.coverage = SimpleNamespace(all=explode)

    assert box.lake_inventory() == []


# ── data budget ─────────────────────────────────────────────────────────────────────────────
def test_data_budget_is_none_without_a_cost_preflight(tmp_path):
    box = _make_toolbox(tmp_path)

    assert box.data_budget() is None
    assert "budget_usd" not in _spend_context(box, None)


def test_data_budget_is_the_preflight_budget_as_a_float(tmp_path):
    box = _make_toolbox(tmp_path)
    box.lake.preflight = SimpleNamespace(budget_usd=12)

    assert box.data_budget() == 12.0
    assert isinstance(box.data_budget(), float)
    assert box.data_budget() == _spend_context(box, None)["budget_usd"]


# ── delegating facts ────────────────────────────────────────────────────────────────────────
def test_symbol_ready_delegates_to_the_lake_readiness_check(tmp_path):
    box = _make_toolbox(tmp_path)

    assert box.symbol_ready("AAA") is True
    assert box.symbol_ready("ZZZ") is False


def test_class_exhausted_delegates_to_the_exhausted_class_registry(tmp_path):
    box, _ledger, _mandate = _populate(tmp_path)

    record = box.class_exhausted("Minute RSI Mean Reversion")

    assert record == box.exhausted.is_exhausted("minute rsi mean reversion")
    assert record["examples"] == ["corpse"]
    assert box.class_exhausted("gap fade at the open") is None


def test_market_context_is_the_economics_digest(tmp_path):
    box = _make_toolbox(tmp_path)

    digest = box.market_context()

    costs = box.settings.backtest
    assert digest["round_trip_cost_bp"] == pytest.approx(2.0 * (costs.fee_bps + costs.slippage_bps))
    assert sorted(digest["symbols"]) == ["AAA", "BBB", "CCC", "DDD"]


# ── session counters ────────────────────────────────────────────────────────────────────────
def test_session_counters_is_a_snapshot_a_later_mutation_does_not_change(tmp_path):
    box = _make_toolbox(tmp_path)
    box.backtests_run = 2
    box.promotions = 1
    box.rejections = 1
    box.author_calls = 3
    box.escalations = 1
    box.strategies_touched.append("probe")
    box.undecided.add("probe")

    before = box.session_counters()

    box.backtests_run += 1
    box.strategies_touched.append("corpse")
    box.undecided.add("corpse")
    after = box.session_counters()

    assert before == SessionCounters(
        backtests_run=2,
        promotions=1,
        rejections=1,
        author_calls=3,
        escalations=1,
        strategies_touched=("probe",),
        undecided=frozenset({"probe"}),
    )
    assert after.backtests_run == 3
    assert after.strategies_touched == ("probe", "corpse")
    assert after.undecided == frozenset({"probe", "corpse"})


def test_session_counters_is_frozen(tmp_path):
    counters = _make_toolbox(tmp_path).session_counters()

    with pytest.raises(FrozenInstanceError):
        counters.backtests_run = 99
