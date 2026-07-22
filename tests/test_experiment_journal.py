"""The experiment journal in isolation: typed reads over the exact records the writers
produce, tolerance for corrupt lines, and the ranking/taint/class-tag semantics the
research discipline (exhaustion gate, holdout validation, reject_strategy) depends on."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from noctis.research.journal import ExperimentJournal, Thesis, Trial


def _card(train=1.5, test=1.0, holdout=0.8, stage="validated", metric_name="sharpe"):
    """The Scorecard surface record_trial reads (aggregates only)."""
    return SimpleNamespace(
        stage=stage,
        metric_name=metric_name,
        avg_train_metric=train,
        avg_test_metric=test,
        gap=None if train is None or test is None else train - test,
        holdout_metric=holdout,
    )


@pytest.fixture
def journal(tmp_path):
    return ExperimentJournal(tmp_path)


def test_missing_journal_reads_empty(journal):
    assert journal.records("ghost") == []
    assert journal.trials("ghost") == []
    assert journal.verdicts("ghost") == []
    assert journal.class_tag("ghost") is None
    assert journal.touched_symbols("ghost") == set()
    stats = journal.stats("ghost")
    assert (stats.n_trials, stats.n_distinct_params, stats.sweep_completed) == (0, 0, False)


def test_trial_round_trips_typed(journal):
    journal.record_trial(
        "probe",
        source="backtest",
        symbols=["AAA", "BBB"],
        params={"lookback": 12},
        window={"bars": 320},
        card=_card(train=1.23456789, test=1.0),
    )
    (trial,) = journal.trials("probe")
    assert isinstance(trial, Trial)
    assert trial.source == "backtest"
    assert trial.symbols == ["AAA", "BBB"]
    assert trial.params == {"lookback": 12}
    assert trial.window == {"bars": 320}
    assert trial.max_bars is None
    assert trial.test == 1.0
    assert trial.metrics["train"] == 1.2346  # journal rounds aggregates to 4 digits
    assert trial.metrics["stage"] == "validated" and trial.metrics["metric_name"] == "sharpe"
    # The raw record keeps the on-disk shape older sessions wrote.
    (record,) = journal.records("probe")
    assert record["event"] == "trial" and record["strategy"] == "probe" and record["at"]


def test_stats_counts_distinct_params_and_sweep_completion(journal):
    for params in ({"lookback": 10}, {"lookback": 10}, {"lookback": 25}):
        journal.record_trial(
            "probe", source="sweep", symbols=["AAA"], params=params, window={}, card=_card()
        )
    stats = journal.stats("probe")
    assert (stats.n_trials, stats.n_distinct_params, stats.sweep_completed) == (3, 2, False)

    journal.record_sweep_complete("probe", n_trials=3, symbols=["AAA"])
    assert journal.stats("probe").sweep_completed is True
    assert journal.records("probe")[-1]["event"] == "sweep_complete"


def test_corrupt_lines_are_skipped_not_fatal(journal):
    journal.record_trial(
        "probe", source="backtest", symbols=["AAA"], params={}, window={}, card=_card()
    )
    with journal.path("probe").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    journal.record_class_tag("probe", "mean reversion")
    assert len(journal.records("probe")) == 2
    assert journal.stats("probe").n_trials == 1
    assert journal.class_tag("probe") == "mean reversion"


def test_trials_by_test_ranks_best_first_and_none_last(journal):
    for test in (0.5, None, -0.3, 0.0, 1.2):
        journal.record_trial(
            "probe",
            source="backtest",
            symbols=["AAA"],
            params={"edge": test},
            window={},
            card=_card(train=test, test=test),
        )
    ranked = [t.test for t in journal.trials_by_test("probe")]
    # A 0.0 test metric is a real (bad) score — it outranks negatives; only None sinks.
    assert ranked == [1.2, 0.5, 0.0, -0.3, None]


def test_class_tag_latest_wins(journal):
    journal.record_class_tag("probe", "first idea")
    journal.record_class_tag("probe", "refined idea")
    assert journal.class_tag("probe") == "refined idea"


def test_thesis_round_trips_typed_with_lineage(journal):
    journal.record_thesis(
        "probe",
        "Overnight drift persists into the open.",
        parent_thesis="Momentum survives the gap.",
        pivot_rationale="Widen from close-to-close to gap-through.",
    )
    thesis = journal.thesis("probe")
    assert isinstance(thesis, Thesis)
    assert thesis.text == "Overnight drift persists into the open."
    assert thesis.parent_thesis == "Momentum survives the gap."
    assert thesis.pivot_rationale == "Widen from close-to-close to gap-through."
    assert thesis.at  # every record is timestamped
    # The raw record carries the extended schema beside the existing kinds.
    (record,) = journal.records("probe")
    assert record["event"] == "thesis"
    assert record["thesis"] == "Overnight drift persists into the open."
    assert record["parent_thesis"] == "Momentum survives the gap."
    assert record["pivot_rationale"] == "Widen from close-to-close to gap-through."


def test_thesis_lineage_fields_are_optional(journal):
    journal.record_thesis("probe", "Just an idea, no parent.")
    thesis = journal.thesis("probe")
    assert thesis is not None
    assert thesis.text == "Just an idea, no parent."
    assert thesis.parent_thesis is None
    assert thesis.pivot_rationale is None
    # Absent lineage is omitted from the record, not written as null.
    (record,) = journal.records("probe")
    assert "parent_thesis" not in record
    assert "pivot_rationale" not in record


def test_missing_thesis_reads_none(journal):
    journal.record_class_tag("probe", "some class")
    assert journal.thesis("probe") is None
    assert journal.thesis("ghost") is None


def test_thesis_latest_wins(journal):
    journal.record_thesis("probe", "first idea")
    journal.record_thesis("probe", "refined idea", parent_thesis="first idea")
    latest = journal.thesis("probe")
    assert latest is not None
    assert latest.text == "refined idea"
    assert latest.parent_thesis == "first idea"


def test_unknown_record_kinds_do_not_break_consumers(journal):
    """Tolerant reads: a future record kind an older reader never learned is skipped by the
    typed views, not fatal — the journal schema is extended, never changed."""
    journal.record_thesis("probe", "an idea")
    with journal.path("probe").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "episode", "future_field": 7}) + "\n")
    journal.record_class_tag("probe", "a class")
    # Every consumer keeps loading across the unknown kind.
    assert len(journal.records("probe")) == 3
    assert journal.class_tag("probe") == "a class"
    assert journal.thesis("probe").text == "an idea"
    assert journal.trials("probe") == []
    assert journal.verdicts("probe") == []


def test_touched_symbols_unions_every_journaled_trial(journal):
    journal.record_trial(
        "probe", source="backtest", symbols=["AAA", "BBB"], params={}, window={}, card=_card()
    )
    journal.record_trial(
        "probe", source="sweep", symbols=["CCC"], params={}, window={}, card=_card(), max_bars=500
    )
    assert journal.touched_symbols("probe") == {"AAA", "BBB", "CCC"}


def test_verdicts_surface_verbatim(journal):
    journal.record_approval(
        "probe",
        promoted=True,
        rationale="beats the weakest champion",
        params={"lookback": 18},
        symbols=["AAA"],
        holdout_symbols=["ZZZ"],
    )
    journal.record_rejection("probe", reason="no edge", best_params={"lookback": 10})
    approve, reject = journal.verdicts("probe")
    assert approve["verdict"] == "approve" and approve["promoted"] is True
    assert approve["holdout_symbols"] == ["ZZZ"]
    assert reject["verdict"] == "reject" and reject["best_params"] == {"lookback": 10}


def test_max_bars_marks_exploration_fidelity(journal):
    journal.record_trial(
        "probe", source="sweep", symbols=["AAA"], params={}, window={}, card=_card(), max_bars=500
    )
    journal.record_sweep_complete("probe", n_trials=1, symbols=["AAA"], max_bars=500)
    (trial,) = journal.trials("probe")
    assert trial.max_bars == 500
    assert journal.records("probe")[-1]["max_bars"] == 500


def test_journals_are_per_strategy_files(journal, tmp_path):
    journal.record_class_tag("alpha", "tag-a")
    journal.record_class_tag("beta", "tag-b")
    assert journal.path("alpha") == tmp_path / "experiments" / "alpha.jsonl"
    assert journal.class_tag("alpha") == "tag-a"
    assert journal.class_tag("beta") == "tag-b"


def test_records_written_sorted_and_line_delimited(journal):
    journal.record_rejection("probe", reason="r", best_params={})
    line = journal.path("probe").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)  # one record per line
    assert line == json.dumps(json.loads(line), sort_keys=True)  # stable key order on disk
