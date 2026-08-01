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
    CandidateTrail,
    Episode,
    SessionEnd,
    SessionLedger,
    SessionRollup,
    SessionStart,
    StageTransition,
    ThesisLine,
    Verdict,
    new_session_id,
)


def _full_arc(ledger: SessionLedger) -> None:
    """A complete two-candidate arc: one authored → rejected, one authored (escalated) →
    approved+promoted, plus one thesis whose author failed the write gate (no optimize)."""
    ledger.record_session_start(mandate="m", budgets={}, models={"driver": "d", "coder": "c"})
    # candidate 1: authored locally, rejected.
    ledger.record_thesis("momo_1", "buy strength")
    ledger.record_stage("formulate")
    ledger.record_episode(stage="formulate", model="driver", tokens=12, outcome="ok")
    ledger.record_stage("match", strategy="momo_1")
    ledger.record_stage("author", strategy="momo_1")
    ledger.record_stage("optimize", strategy="momo_1", detail={"trials": 5, "best_metric": 1.2})
    ledger.record_stage("decide", strategy="momo_1")
    ledger.record_episode(stage="decide", model="driver", tokens=8, outcome="ok")
    ledger.record_verdict("momo_1", verdict="reject", lesson="thin", promoted=False)
    # candidate 2: escalated author, approved + promoted.
    ledger.record_thesis("rev_2", "fade the spike")
    ledger.record_stage("formulate")
    ledger.record_episode(stage="formulate", model="driver", tokens=10, outcome="ok")
    ledger.record_stage("match", strategy="rev_2")
    ledger.record_stage("author", strategy="rev_2")
    ledger.record_episode(
        stage="author", model="coder-paid", tokens=40, outcome="ok", escalated=True
    )
    ledger.record_stage("optimize", strategy="rev_2", detail={"trials": 7, "best_metric": 2.5})
    ledger.record_stage("decide", strategy="rev_2")
    ledger.record_episode(stage="decide", model="driver", tokens=6, outcome="ok")
    ledger.record_verdict("rev_2", verdict="approve", lesson="edge holds", promoted=True)
    # candidate 3: author failed the write gate — never optimized, left undecided by no verdict.
    ledger.record_thesis("dud_3", "an idea that will not compile")
    ledger.record_stage("formulate")
    ledger.record_episode(stage="formulate", model="driver", tokens=4, outcome="ok")
    ledger.record_stage("match", strategy="dud_3")
    ledger.record_stage("author", strategy="dud_3")
    ledger.record_session_end(formulated=3, promoted=1, rejected=1, note="max_episodes")


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


def test_author_stage_row_carries_the_captured_brief_and_spec_hashes(ledger):
    """Coder-site capture (#184): the AUTHOR row references the captured brief (and the compiled
    spec suite when one was stamped) by content hash, so the body stays findable in ``qa/`` while
    the ledger line stays small."""
    brief_sha = "a" * 64
    spec_sha = "b" * 64
    ledger.record_stage(
        "author",
        strategy="momo_1",
        detail={"oracle": ["rally", "grind"]},
        brief_sha256=brief_sha,
        spec_sha256=spec_sha,
    )

    (stage,) = ledger.stages()
    assert stage.brief_sha256 == brief_sha
    assert stage.spec_sha256 == spec_sha
    assert stage.detail == {"oracle": ["rally", "grind"]}  # the existing detail is untouched


def test_stage_row_omits_capture_hashes_that_were_never_captured(ledger):
    """A latched-off capture store returns no hash, so the row simply omits the field rather than
    naming a body that was never written."""
    ledger.record_stage("author", strategy="momo_1", detail={"oracle": ["rally"]})

    (stage,) = ledger.stages()
    assert stage.brief_sha256 is None
    assert stage.spec_sha256 is None
    (record,) = ledger.records()
    assert "brief_sha256" not in record and "spec_sha256" not in record


def test_stage_row_without_capture_fields_reads_as_not_carried(ledger):
    """A stage line written before #184 keeps parsing: the fields read as ``None`` — *not
    carried* — which a tolerant read tells apart from a line that carried an empty one."""
    pre_184 = {"event": "stage", "at": "t", "stage": "author", "strategy": "momo_1"}
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text(
        json.dumps(pre_184) + "\n" + json.dumps({**pre_184, "brief_sha256": ""}) + "\n",
        encoding="utf-8",
    )

    old, empty = ledger.stages()
    assert old.brief_sha256 is None  # the field was never carried
    assert empty.brief_sha256 == ""  # the line carried an empty one — a different fact
    assert old.stage == "author" and old.strategy == "momo_1"  # the rest still parses


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


def test_episode_misfire_details_and_note_round_trip_and_default_empty(ledger):
    # Per-misfire diagnostics (#102) ride the episode line as a tolerant extension, like checks:
    # each rejected attempt's classifier note beside a capped excerpt of what came back, plus the
    # episode's last misfire/error note.
    detail = {
        "note": "structured emit invalid (missing thesis)",
        "raw": '{"style": "momentum"}',
    }
    ledger.record_episode(
        stage="formulate",
        model="anthropic/claude-sonnet-5",
        outcome="misfires_exhausted",
        misfires=3,
        misfire_details=[detail],
        note="structured emit invalid (missing thesis)",
    )
    ledger.record_episode(stage="decide", model="local", outcome="ok")  # clean ⇒ absent
    failed, clean = ledger.episodes()
    assert failed.misfire_details == [detail]
    assert failed.note == "structured emit invalid (missing thesis)"
    assert clean.misfire_details == [] and clean.note is None
    # Absent diagnostics are omitted from the record, never stored as empty fields.
    clean_record = [r for r in ledger.records() if r["stage"] == "decide"][0]
    assert "misfire_details" not in clean_record and "note" not in clean_record


def test_episode_usage_split_round_trips_and_is_absent_when_never_measured(ledger):
    """The four-field token split rides the episode line as a tolerant extension (story #140).

    Tokens alone cannot be priced — input, output, cache-write and cache-read bill at four
    different rates — so the split is journaled beside the total and read back typed."""
    split = {
        "input_tokens": 900,
        "output_tokens": 100,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 200,
    }
    ledger.record_episode(
        stage="formulate", model="anthropic/claude", outcome="ok", tokens=1250, usage=split
    )
    ledger.record_episode(stage="decide", model="anthropic/claude", outcome="ok", tokens=8)
    measured, unmeasured = ledger.episodes()

    assert measured.usage == split
    assert unmeasured.usage is None  # spent tokens with no split ⇒ unknown, never zeros
    assert "usage" not in [r for r in ledger.records() if r["stage"] == "decide"][0]


def test_episode_served_model_rides_beside_the_requested_model(ledger):
    """Served-model provenance (#181): the row records what actually answered beside the alias
    that was asked for, so a same-alias/different-served-model event is detectable from the
    ledger alone. Absent/empty is omitted from the record, never stored as an empty field."""
    ledger.record_episode(
        stage="formulate",
        model="openai/gpt-5",
        outcome="ok",
        served_model="gpt-5-2026-04-01",
    )
    ledger.record_episode(
        stage="decide",
        model="openai/gpt-5",  # the same requested alias …
        outcome="ok",
        served_model="gpt-5-2026-06-14",  # … a different model served it
    )
    ledger.record_episode(stage="author", model="local/coder", outcome="ok")  # none reported

    formulate_ep, decide_ep, author_ep = ledger.episodes()
    assert (formulate_ep.model, formulate_ep.served_model) == ("openai/gpt-5", "gpt-5-2026-04-01")
    assert (decide_ep.model, decide_ep.served_model) == ("openai/gpt-5", "gpt-5-2026-06-14")
    # One alias, two served models — visible without any other artifact.
    served = {e.served_model for e in ledger.episodes() if e.model == "openai/gpt-5"}
    assert served == {"gpt-5-2026-04-01", "gpt-5-2026-06-14"}
    assert author_ep.served_model is None
    assert "served_model" not in [r for r in ledger.records() if r["stage"] == "author"][0]


def test_episode_without_a_served_model_field_reads_as_not_carried(ledger):
    """A ledger line written before #181 keeps parsing: the field reads as ``None`` — *not
    carried* — which a tolerant read tells apart from a line that carried an empty one."""
    pre_181 = {"event": "episode", "at": "t", "stage": "decide", "model": "m", "outcome": "ok"}
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text(
        json.dumps(pre_181) + "\n" + json.dumps({**pre_181, "served_model": ""}) + "\n",
        encoding="utf-8",
    )

    old, empty = ledger.episodes()
    assert old.served_model is None  # the field was never carried
    assert empty.served_model == ""  # the line carried an empty one — a different fact
    assert old.model == "m" and old.outcome == "ok"  # the rest of the line still parses


def test_usage_totals_sum_the_episodes_that_recorded_a_split(ledger):
    """What the episodic driver reads back for its summary — summed off the ledger, not counted."""
    ledger.record_episode(
        stage="formulate",
        model="m",
        outcome="ok",
        tokens=30,
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    )
    ledger.record_episode(
        stage="decide",
        model="m",
        outcome="ok",
        tokens=7,
        usage={
            "input_tokens": 5,
            "output_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    )
    # An episode that spent nothing (a coder escalation line) is a real zero, not an unknown.
    ledger.record_episode(stage="author", model="coder", outcome="ok", escalated=True)

    assert ledger.usage_totals() == {
        "input_tokens": 15,
        "output_tokens": 22,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_usage_totals_are_null_when_any_spending_episode_recorded_no_split(ledger):
    """A ledger written before the split existed reports ``None`` — never a partial sum."""
    ledger.record_episode(
        stage="formulate",
        model="m",
        outcome="ok",
        tokens=30,
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    )
    ledger.record_episode(stage="decide", model="m", outcome="ok", tokens=9)

    assert ledger.usage_totals() is None


def test_usage_totals_on_a_ledger_with_no_episodes_are_null(ledger):
    assert ledger.usage_totals() is None


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


# ── derived views: the at-a-glance rollup + per-candidate trail the CLOSE report renders (#74) ─
def test_rollup_derives_every_field_from_the_typed_records(ledger):
    _full_arc(ledger)
    rollup = ledger.rollup()
    assert isinstance(rollup, SessionRollup)
    assert rollup.session_id == "sess-1"
    assert rollup.theses == 3  # one thesis per formulate
    assert rollup.authored == 2  # two files reached OPTIMIZE; the third failed the write gate
    assert rollup.validation_failures == 1  # author stages that never reached optimize
    assert rollup.trials == 12  # summed from each optimize detail (5 + 7)
    assert rollup.verdicts == {"reject": 1, "approve": 1}  # by kind
    assert rollup.promoted == 1
    assert rollup.undecided == 0  # every authored file reached a verdict
    assert rollup.escalations == 1  # one paid AUTHOR episode
    assert rollup.tokens_by_stage == {"formulate": 26, "decide": 14, "author": 40}
    assert rollup.tokens_by_model == {"driver": 40, "coder-paid": 40}
    assert rollup.note == "max_episodes"


def test_rollup_counts_authored_but_unverdicted_files_as_undecided(tmp_path):
    led = SessionLedger(tmp_path, "undec")
    led.record_thesis("draft_1", "an idea")
    led.record_stage("author", strategy="draft_1")
    led.record_stage("optimize", strategy="draft_1", detail={"trials": 3})
    led.record_stage("decide", strategy="draft_1")  # decide ran but left no verdict (undecided)
    assert led.rollup().undecided == 1
    assert led.rollup().verdicts == {}


def test_rollup_on_an_empty_ledger_is_all_zeros(ledger):
    rollup = ledger.rollup()
    assert rollup.theses == 0 and rollup.authored == 0 and rollup.trials == 0
    assert rollup.verdicts == {} and rollup.escalations == 0
    assert rollup.tokens_by_stage == {} and rollup.tokens_by_model == {}


def test_rollup_log_line_names_every_field(ledger):
    _full_arc(ledger)
    line = ledger.rollup().log_line()
    for token in (
        "3 theses",
        "2 authored",
        "1 validation failures",
        "12 trials",
        "approve=1",
        "reject=1",
        "0 undecided",
        "1 escalations",
        "formulate=26",
        "driver=40",
    ):
        assert token in line


def test_candidate_trails_walk_each_candidate_formulate_to_decide(ledger):
    _full_arc(ledger)
    trails = ledger.candidate_trails()
    assert [t.strategy for t in trails] == ["momo_1", "rev_2", "dud_3"]  # thesis (formulate) order
    momo, rev, dud = trails
    assert isinstance(momo, CandidateTrail)
    assert momo.thesis == "buy strength"
    assert momo.stages == ("match", "author", "optimize", "decide")
    assert momo.trials == 5 and momo.best_metric == 1.2
    assert momo.verdict == "reject" and momo.outcome == "rejected"
    assert rev.best_metric == 2.5 and rev.verdict == "approve" and rev.promoted is True
    assert rev.outcome == "promoted"
    # The write-gate failure never reached optimize/decide and left no verdict — an undecided trail.
    assert dud.stages == ("match", "author")
    assert dud.trials == 0 and dud.best_metric is None
    assert dud.verdict is None and dud.outcome == "undecided"


def test_candidate_trail_surfaces_the_author_stage_oracle(tmp_path):
    # The AUTHOR stage's oracle detail (the spec's scenario names, #86) rides the candidate trail,
    # so a post-mortem audits which fixed oracle each candidate was gated against.
    led = SessionLedger(tmp_path, "oracle")
    led.record_thesis("momo_1", "buy strength")
    led.record_stage("author", strategy="momo_1", detail={"oracle": ["rally", "grind"]})
    trail = led.candidate_trails()[0]
    assert trail.oracle == ("rally", "grind")
    assert trail.to_dict()["oracle"] == ["rally", "grind"]


def test_candidate_trail_oracle_defaults_empty_for_a_spec_less_author(tmp_path):
    # An author stage with no oracle detail (a spec-less/older ledger) reads a clean empty oracle,
    # so a reader never branches on presence.
    led = SessionLedger(tmp_path, "no-oracle")
    led.record_thesis("momo_1", "buy strength")
    led.record_stage("author", strategy="momo_1")
    trail = led.candidate_trails()[0]
    assert trail.oracle == ()
    assert trail.to_dict()["oracle"] == []


def test_report_view_is_json_safe_or_none_on_an_empty_ledger(ledger, tmp_path):
    assert ledger.report_view() is None  # nothing written ⇒ nothing to render (graceful)
    _full_arc(ledger)
    view = ledger.report_view()
    assert view is not None
    assert view["session_id"] == "sess-1"
    assert view["rollup"]["theses"] == 3
    assert [c["strategy"] for c in view["candidates"]] == ["momo_1", "rev_2", "dud_3"]
    # JSON-safe end to end (the report writes this straight into the structured report).
    assert json.loads(json.dumps(view)) == view


def test_from_path_reconstructs_a_ledger_at_the_same_file(tmp_path):
    led = SessionLedger(tmp_path, "roundtrip")
    led.record_stage("formulate")
    reopened = SessionLedger.from_path(led.path)
    assert reopened.path == led.path
    assert reopened.session_id == "roundtrip"
    assert [s.stage for s in reopened.stages()] == ["formulate"]
