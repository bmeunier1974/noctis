"""The session ledger in isolation: one append-only JSONL file per research session under
the state dir, every record kind round-tripping through the public API in append order,
tolerance for unknown kinds a later reader never learned, and append-only semantics (an
append never rewrites an earlier line). The driver stories exercise the ledger in anger;
this suite locks the storage contract #65 lands."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from noctis.research.ledger import (
    Episode,
    SessionEnd,
    SessionLedger,
    SessionStart,
    StageTransition,
    ThesisLine,
    Verdict,
    new_session_id,
)


@pytest.fixture
def ledger(tmp_path):
    return SessionLedger(tmp_path, "sess-1")


def test_missing_ledger_reads_empty(ledger):
    assert ledger.records() == []
    assert ledger.theses() == []
    assert ledger.stages() == []
    assert ledger.episodes() == []
    assert ledger.verdicts() == []
    assert ledger.session_start() is None
    assert ledger.session_end() is None


def test_one_file_per_session_under_state_dir(tmp_path):
    """Each session id maps to its own file under ``<state_dir>/sessions/`` — nothing else."""
    led_a = SessionLedger(tmp_path, "alpha")
    led_b = SessionLedger(tmp_path, "beta")
    led_a.record_stage("formulate")
    led_b.record_stage("match")

    assert led_a.path == tmp_path / "sessions" / "alpha.jsonl"
    assert led_b.path == tmp_path / "sessions" / "beta.jsonl"
    assert led_a.path.is_file() and led_b.path.is_file()
    # The two sessions never share a file.
    assert led_a.records()[0]["stage"] == "formulate"
    assert led_b.records()[0]["stage"] == "match"
    # Everything lands inside the sessions/ subtree of the given state dir.
    written = sorted(p.name for p in (tmp_path / "sessions").iterdir())
    assert written == ["alpha.jsonl", "beta.jsonl"]


def test_default_session_id_is_clock_derived_not_hidden_randomness(tmp_path):
    """No wall-clock randomness baked into the module: the default id is a pure function of an
    injectable clock, and an explicit id always wins."""
    clock = datetime(2026, 7, 22, 13, 30, 5, tzinfo=UTC)
    assert new_session_id(clock) == "session-20260722T133005"
    led = SessionLedger(tmp_path, now=clock)
    assert led.session_id == "session-20260722T133005"
    assert led.path == tmp_path / "sessions" / "session-20260722T133005.jsonl"
    # An explicit id overrides the default entirely.
    assert SessionLedger(tmp_path, "explicit", now=clock).session_id == "explicit"


def test_session_start_round_trips_typed(ledger):
    ledger.record_session_start(
        mandate="momentum-hunter",
        budgets={"max_iterations": 40, "max_backtests": 200},
        models={"agent": "anthropic/claude", "coder": "openai/gpt"},
    )
    start = ledger.session_start()
    assert isinstance(start, SessionStart)
    assert start.mandate == "momentum-hunter"
    assert start.budgets == {"max_iterations": 40, "max_backtests": 200}
    assert start.models == {"agent": "anthropic/claude", "coder": "openai/gpt"}
    assert start.at  # every record is timestamped
    (record,) = ledger.records()
    assert record["event"] == "session_start"


def test_session_start_mandate_is_optional(tmp_path):
    led = SessionLedger(tmp_path, "sess-2")
    led.record_session_start(mandate=None, budgets={}, models={})
    start = led.session_start()
    assert start is not None
    assert start.mandate is None


def test_thesis_line_round_trips_with_lineage(ledger):
    ledger.record_thesis(
        "overnight_drift",
        "Overnight drift persists into the open.",
        parent_thesis="Momentum survives the gap.",
        pivot_rationale="Widen from close-to-close to gap-through.",
    )
    (thesis,) = ledger.theses()
    assert isinstance(thesis, ThesisLine)
    assert thesis.strategy == "overnight_drift"
    assert thesis.text == "Overnight drift persists into the open."
    assert thesis.parent_thesis == "Momentum survives the gap."
    assert thesis.pivot_rationale == "Widen from close-to-close to gap-through."
    assert thesis.at


def test_thesis_lineage_fields_are_optional(ledger):
    ledger.record_thesis("solo", "Just an idea, no parent.")
    (thesis,) = ledger.theses()
    assert thesis.parent_thesis is None
    assert thesis.pivot_rationale is None
    # Absent lineage is omitted from the record, not stored as null.
    (record,) = ledger.records()
    assert "parent_thesis" not in record
    assert "pivot_rationale" not in record


def test_stage_transition_round_trips(ledger):
    ledger.record_stage("optimize", strategy="overnight_drift")
    (stage,) = ledger.stages()
    assert isinstance(stage, StageTransition)
    assert stage.stage == "optimize"
    assert stage.strategy == "overnight_drift"
    assert stage.at


def test_episode_round_trips_typed(ledger):
    ledger.record_episode(
        stage="decide",
        model="anthropic/claude",
        tokens=1234,
        misfires=2,
        outcome="tool_call",
        escalated=True,
    )
    (episode,) = ledger.episodes()
    assert isinstance(episode, Episode)
    assert episode.stage == "decide"
    assert episode.model == "anthropic/claude"
    assert episode.tokens == 1234
    assert episode.misfires == 2
    assert episode.outcome == "tool_call"
    assert episode.escalated is True
    assert episode.checks == []  # absent ⇒ empty, a tolerant default
    assert episode.at


def test_episode_checks_round_trip_and_default_empty(ledger):
    # Driver-side sanity-check outcomes (story #71) ride the episode line as a tolerant extension.
    ledger.record_episode(
        stage="formulate",
        model="local",
        outcome="ok",
        checks=[{"check": "cost_arithmetic", "result": "reask"}],
    )
    ledger.record_episode(stage="decide", model="local", outcome="ok")  # no checks ⇒ absent
    formulate_ep, decide_ep = ledger.episodes()
    assert formulate_ep.checks == [{"check": "cost_arithmetic", "result": "reask"}]
    assert decide_ep.checks == []
    # An empty/None checks list is omitted from the record, never stored as an empty field.
    assert "checks" not in [r for r in ledger.records() if r["stage"] == "decide"][0]


def test_verdict_round_trips_with_class_lesson(ledger):
    ledger.record_verdict(
        "overnight_drift",
        verdict="approve",
        lesson="Gap-through momentum beats close-to-close on liquid names.",
        promoted=True,
    )
    (verdict,) = ledger.verdicts()
    assert isinstance(verdict, Verdict)
    assert verdict.strategy == "overnight_drift"
    assert verdict.verdict == "approve"
    assert verdict.lesson == "Gap-through momentum beats close-to-close on liquid names."
    assert verdict.promoted is True
    assert verdict.at


def test_verdict_promoted_is_optional(ledger):
    ledger.record_verdict("dud", verdict="reject", lesson="No edge after costs.")
    (verdict,) = ledger.verdicts()
    assert verdict.verdict == "reject"
    assert verdict.promoted is None


def test_session_end_rollup_round_trips(ledger):
    ledger.record_session_end(formulated=3, promoted=1, rejected=2, note="one champion landed")
    end = ledger.session_end()
    assert isinstance(end, SessionEnd)
    assert end.formulated == 3
    assert end.promoted == 1
    assert end.rejected == 2
    assert end.note == "one champion landed"
    assert end.at


def test_every_kind_round_trips_in_append_order(ledger):
    """The whole session narrative, appended then read back in order through the public API."""
    ledger.record_session_start(mandate="m", budgets={}, models={})
    ledger.record_thesis("cand", "a thesis")
    ledger.record_stage("match")
    ledger.record_episode(stage="match", model="local", outcome="agent_done")
    ledger.record_verdict("cand", verdict="reject", lesson="thin")
    ledger.record_session_end(formulated=1, promoted=0, rejected=1)

    events = [r["event"] for r in ledger.records()]
    assert events == [
        "session_start",
        "thesis",
        "stage",
        "episode",
        "verdict",
        "session_end",
    ]


def test_unknown_record_kinds_are_tolerated_on_read(ledger):
    """A future record kind an older reader never learned is skipped by the typed views, not
    fatal — the ledger schema is extended, never changed."""
    ledger.record_stage("formulate")
    with ledger.path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "from_the_future", "unknown_field": 7}) + "\n")
    ledger.record_stage("decide")

    # The raw read surfaces the unknown line; every typed view keeps loading across it.
    assert len(ledger.records()) == 3
    assert [s.stage for s in ledger.stages()] == ["formulate", "decide"]
    assert ledger.episodes() == []
    assert ledger.verdicts() == []


def test_corrupt_lines_are_skipped_not_fatal(ledger):
    ledger.record_stage("formulate")
    with ledger.path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    ledger.record_stage("decide")
    assert len(ledger.records()) == 2
    assert [s.stage for s in ledger.stages()] == ["formulate", "decide"]


def test_appends_never_rewrite_earlier_lines(ledger):
    ledger.record_session_start(mandate="m", budgets={}, models={})
    ledger.record_thesis("cand", "a thesis")
    prefix = ledger.path.read_bytes()

    ledger.record_stage("optimize")
    ledger.record_session_end(formulated=1, promoted=0, rejected=1)
    grown = ledger.path.read_bytes()

    # Append-only: the earlier bytes are an unchanged prefix of the grown file.
    assert grown.startswith(prefix)
    assert len(grown) > len(prefix)


def test_records_written_sorted_and_line_delimited(ledger):
    ledger.record_stage("formulate")
    line = ledger.path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)  # exactly one record per line
    assert line == json.dumps(json.loads(line), sort_keys=True)  # stable key order on disk
