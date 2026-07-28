"""Stopping and resuming a run — ``noctis run --resume <run_id>`` (story #132, epic #126).

A run, not a process, is the unit progress is tracked on: an operator stops the engine each morning
and resumes it each night, and the same ``run_id`` keeps accumulating into the same record under its
own frozen config. Everything asserted here is external — what the record on disk says, what the
resumed process's settings resolve to, which run tree the state landed in.

The headline is :func:`test_three_one_hour_segments_total_exactly_what_one_three_hour_segment_does`:
the direct proof of "derived, never incremented" (epic D4). Every cumulative number is recomputed at
write time from the durable artifacts plus the append-only ``segments[]`` list, so splitting one
night into three cannot change a total, and a crash mid-write cannot double-count one.
"""

from __future__ import annotations

import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.config.rehydrate import freeze_inputs
from noctis.reporting.run_store import (
    RUN_LOCK_NAME,
    RUN_RECORD_NAME,
    RunCompletedError,
    RunNotFoundError,
    open_run,
)

runner = CliRunner()

START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)
HOUR = 3600.0


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock read reaches the store."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def _settings(tmp_path: Path, body: str = "mode: paper\n"):
    from noctis.config import load_settings

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return load_settings(config_path=path)


def _inputs(tmp_path: Path, **kwargs):
    return freeze_inputs(_settings(tmp_path), frozen_at="2026-07-27T14:22:33.418Z", **kwargs)


def _open(runs_dir: Path, clock: FakeClock, **kwargs):
    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs_dir, clock=clock, **kwargs)


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


# ── the headline: segmentation equivalence ─────────────────────────────────────────────────


def _totals(record: dict) -> dict:
    """Every cumulative number a record carries, plus the work its segments account for.

    The sections that hold the rest of the run's totals — the equity curve, the strategies —
    arrive in stories #141–#142. They slot in *here*, as more keys of this one dict, and the
    equivalence test above keeps passing unchanged: that is the whole point of the epic's
    "derived, never incremented" rule. Nothing is invented in the meantime; what is summed below
    is exactly what the record carries today.

    Per-*segment* breakdowns are summed rather than compared entry by entry — a three-segment run
    obviously has three of them. What must match is the total they add up to, which is exactly the
    claim "attribution is per night, but the bill is the run's".
    """
    counters: dict[str, int] = {}
    for segment in record["segments"]:
        for name, value in segment["counters"].items():
            counters[name] = counters.get(name, 0) + value
    spend = record["spend"] or {}
    per_segment: dict[str, float] = {}
    for bucket in spend.get("by_segment") or []:
        for name, value in bucket.items():
            if name != "index" and isinstance(value, int | float):
                per_segment[name] = per_segment.get(name, 0) + value
    return {
        "status": record["run"]["status"],
        "complete": record["run"]["complete"],
        "cumulative_runtime_s": record["run"]["cumulative_runtime_s"],
        "cumulative_research_s": record["run"]["cumulative_research_s"],
        "cumulative_trials": record["run"]["cumulative_trials"],
        "truncated": record["run"]["truncated"],
        "counters": counters,
        "events": [event["text"] for event in record["events"]],
        "errors": [error["text"] for error in record["errors"]],
        "engine": record["engine"],
        "frozen_inputs": record["inputs"],
        # Story #140: the whole spend roll-up, minus the per-segment attribution it is summed into.
        "spend_tokens": spend.get("tokens"),
        "spend_by_model": spend.get("by_model"),
        "spend_by_stage": spend.get("by_stage"),
        "spend_by_segment_summed": per_segment,
        "llm_usd_estimate": spend.get("llm_usd_estimate"),
        "pricing_table_version": spend.get("pricing_table_version"),
        "efficiency": spend.get("efficiency"),
    }


OPUS = "anthropic/claude-opus-4-8"


def _episode(stage: str, inp: int, out: int) -> dict:
    """One judgment episode's journal line — the ledger evidence spend is derived from."""
    return {
        "stage": stage,
        "model": OPUS,
        "usage": {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _journal(run_dir: Path, session: str, episodes: list[dict]) -> None:
    """Journal a night's episodes and one trial into the run's **own** state — the artifacts the
    record re-derives every total from."""
    from noctis.research.journal import ExperimentJournal
    from noctis.research.ledger import SessionLedger
    from tests.test_champions import make_scorecard

    ledger = SessionLedger(run_dir / "state", session)
    journal = ExperimentJournal(run_dir / "state")
    for episode in episodes:
        ledger.record_episode(
            stage=episode["stage"],
            model=episode["model"],
            outcome="ok",
            tokens=sum(episode["usage"].values()),
            usage=episode["usage"],
        )
        # One trial per episode, its params keyed on the episode itself, so the same work journals
        # the same distinct param sets however it is sliced into processes.
        journal.record_trial(
            "momo_1",
            source="sweep",
            symbols=["AAPL"],
            params={"lookback": episode["usage"]["input_tokens"]},
            window={},
            card=make_scorecard("momo_1", 1.2, 1.4),
        )


def _drive(
    runs_dir: Path,
    tmp_path: Path,
    *,
    segments: list[tuple[float, dict]],
    nights: list[list[dict]] | None = None,
    monkeypatch=None,
) -> dict:
    """Run one experiment as ``segments`` process invocations, and return the record it left.

    ``nights[i]`` is the research evidence segment *i* journals into the run's own ledgers, stamped
    on the same injected clock the store runs on (production shares one wall clock between the
    two), so the record can attribute each episode to the segment that spent it.
    """
    from noctis.research import ledger as ledger_module

    clock = FakeClock()
    if monkeypatch is not None:
        monkeypatch.setattr(ledger_module, "_now_iso", lambda: clock().isoformat())
    inputs = _inputs(tmp_path)
    run_id: str | None = None
    run_dir = runs_dir
    for index, (hours, counters) in enumerate(segments):
        store = _open(
            runs_dir, clock, run_id=run_id, resume=run_id is not None, inputs=inputs, writer=None
        )
        run_id, run_dir = store.run_id, store.run_dir
        if nights:
            _journal(run_dir, f"s{index}", nights[index])
        clock.advance(hours * HOUR)
        store.close(
            reason="time_limit", counters=counters, phase_seconds={"RESEARCH": hours * HOUR}
        )
    return _record(run_dir)


def test_three_one_hour_segments_total_exactly_what_one_three_hour_segment_does(
    tmp_path, monkeypatch
):
    """The epic's most important test (D4). Same injected clock, same work, same fakes — one long
    night and three short ones leave records whose derived totals are identical.

    Story #140 puts spend under the same rule: the token split, the priced estimate, the
    per-model/per-stage attribution and the efficiency ratios are all re-derived from the run's own
    ledgers at write time, so how the work was *sliced into processes* cannot move a single one."""
    work = [{"cycles": 1, "research_iterations": 4, "trades": 2} for _ in range(3)]
    nights = [
        [_episode("formulate", 1000, 200), _episode("decide", 500, 90)],
        [_episode("formulate", 700, 150)],
        [_episode("decide", 300, 40), _episode("formulate", 20, 5)],
    ]

    one = _drive(
        tmp_path / "one",
        tmp_path / "cfg",
        segments=[(3.0, _summed(work))],
        nights=[[episode for night in nights for episode in night]],
        monkeypatch=monkeypatch,
    )
    three = _drive(
        tmp_path / "three",
        tmp_path / "cfg",
        segments=[(1.0, work[i]) for i in range(3)],
        nights=nights,
        monkeypatch=monkeypatch,
    )

    assert _totals(one) == _totals(three)
    assert one["run"]["cumulative_runtime_s"] == 3 * HOUR
    assert len(one["segments"]) == 1 and len(three["segments"]) == 3
    # …and the spend really was measured, so the equality above is not two nulls agreeing.
    assert one["spend"]["tokens"]["total_tokens"] == 3005
    assert one["spend"]["llm_usd_estimate"] > 0
    assert [b["total_tokens"] for b in three["spend"]["by_segment"]] == [1790, 850, 365]


def _summed(counters: list[dict]) -> dict:
    out: dict[str, int] = {}
    for one in counters:
        for name, value in one.items():
            out[name] = out.get(name, 0) + value
    return out


def test_rewriting_a_record_never_double_counts_a_segment(tmp_path):
    """Derived, never incremented: a checkpoint in the middle of a segment rewrites the same
    totals rather than adding to them."""
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock, inputs=_inputs(tmp_path / "cfg"))
    clock.advance(HOUR)
    store.checkpoint(counters={"cycles": 1})
    store.checkpoint(counters={"cycles": 1})
    store.checkpoint(counters={"cycles": 1})
    clock.advance(HOUR)
    store.close(reason="time_limit", counters={"cycles": 2})

    record = _record(store.run_dir)

    assert record["run"]["cumulative_runtime_s"] == 2 * HOUR
    assert record["segments"][0]["counters"] == {"cycles": 2}


# ── resuming: the run, not the process, is the unit ────────────────────────────────────────


def test_resuming_appends_a_segment_to_the_same_run(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, inputs=_inputs(tmp_path / "cfg"))
    clock.advance(HOUR)
    first.close(reason="time_limit", counters={"cycles": 1})

    clock.advance(8 * HOUR)
    second = _open(runs, clock, run_id=first.run_id, resume=True)
    clock.advance(2 * HOUR)
    second.close(reason="stop_requested", counters={"cycles": 3})

    record = _record(first.run_dir)
    assert second.run_id == first.run_id
    assert second.run_dir == first.run_dir
    assert [s["index"] for s in record["segments"]] == [0, 1]
    assert [s["resumed"] for s in record["segments"]] == [False, True]
    assert record["run"]["cumulative_runtime_s"] == 3 * HOUR
    assert record["run"]["created_utc"] == "2026-07-27T14:22:33.418Z"  # the run's own birthday
    assert [p.name for p in runs.iterdir() if p.is_dir()] == [first.run_id]


def test_resuming_an_unknown_run_is_a_clean_lookup_failure(tmp_path):
    with pytest.raises(RunNotFoundError):
        _open(tmp_path / "runs", FakeClock(), run_id="20260101T000000Z-nope00", resume=True)


def test_a_completed_run_refuses_resume(tmp_path):
    """``completed`` is terminal, so a published result can never silently gain segments."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, inputs=_inputs(tmp_path / "cfg"))
    clock.advance(HOUR)
    store.close(reason="time_limit")
    record = _record(store.run_dir)
    record["run"]["completed_utc"] = "2026-07-27T15:22:33.418Z"
    record["run"]["status"] = "completed"
    (store.run_dir / RUN_RECORD_NAME).write_text(json.dumps(record))

    with pytest.raises(RunCompletedError) as excinfo:
        _open(runs, clock, run_id=store.run_id, resume=True)

    assert store.run_id in str(excinfo.value)
    assert "completed" in str(excinfo.value)


def test_a_crash_mid_segment_is_marked_interrupted_on_the_next_open_without_double_counting(
    tmp_path,
):
    """Kill between phases: the next open observes the unclosed segment (it is never guessed at
    write time), the totals stay correct, and a third segment resumes cleanly."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, inputs=_inputs(tmp_path / "cfg"))
    clock.advance(HOUR)
    first.close(reason="time_limit", counters={"cycles": 1})

    killed = _open(runs, clock, run_id=first.run_id, resume=True)
    clock.advance(HOUR)
    killed.checkpoint(counters={"cycles": 1})
    (killed.run_dir / RUN_LOCK_NAME).unlink()  # the process is gone; nothing closed the segment

    third = _open(runs, clock, run_id=first.run_id, resume=True)
    clock.advance(HOUR)
    third.close(reason="stop_requested", counters={"cycles": 1})

    record = _record(first.run_dir)
    assert [s["status"] for s in record["segments"]] == ["stopped", "interrupted", "stopped"]
    assert record["segments"][1]["stopped_utc"] is None
    assert record["segments"][1]["duration_s"] is None
    # The interrupted segment contributes no runtime — an unclosed segment has no honest duration.
    assert record["run"]["cumulative_runtime_s"] == 2 * HOUR
    assert _totals(record)["counters"] == {"cycles": 3}
    assert record["run"]["status"] == "stopped"


# ── the config is frozen at creation, and stays frozen ─────────────────────────────────────


def test_the_config_is_frozen_once_at_creation_and_never_re_frozen(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, inputs=_inputs(tmp_path / "a", execution_mode="paper"))
    first.close(reason="time_limit")

    later = _inputs(tmp_path / "b", execution_mode="paper")
    later["settings"]["resolved"]["promotion"]["metric"] = "sortino"
    second = _open(runs, clock, run_id=first.run_id, resume=True, inputs=later)
    second.close(reason="stopped")

    frozen = _record(first.run_dir)["inputs"]
    assert frozen["settings"]["resolved"]["promotion"]["metric"] == "sharpe"
    assert frozen["frozen_at_utc"] == "2026-07-27T14:22:33.418Z"
    assert frozen["config_epoch"] == 1


def test_a_run_that_never_froze_a_config_freezes_one_on_its_next_open(tmp_path):
    """The adopted-history shape (story #131): a record with no frozen inputs is not a broken
    record, and the first process that actually works the run pins its configuration."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    store.close(reason="time_limit")
    assert _record(store.run_dir)["inputs"] is None

    resumed = _open(runs, clock, run_id=store.run_id, resume=True, inputs=_inputs(tmp_path / "c"))
    resumed.close(reason="stopped")

    assert _record(store.run_dir)["inputs"]["settings"]["digest"]


# ── the CLI ────────────────────────────────────────────────────────────────────────────────


def _config(tmp_path: Path, body: str = "") -> str:
    path = tmp_path / "config.yaml"
    path.write_text(f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n{textwrap.dedent(body)}")
    return str(path)


def _runs_dir(tmp_path: Path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _run_dirs(tmp_path: Path) -> list[Path]:
    return sorted(p for p in _runs_dir(tmp_path).iterdir() if p.is_dir())


def test_noctis_run_resume_continues_the_same_run_instead_of_minting_one(tmp_path):
    cfg = _config(tmp_path)
    first = runner.invoke(app, ["run", "--config", cfg])
    assert first.exit_code == 0, first.output
    run_id = _run_dirs(tmp_path)[0].name

    second = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id])

    assert second.exit_code == 0, second.output
    assert [p.name for p in _run_dirs(tmp_path)] == [run_id]
    record = _record(_runs_dir(tmp_path) / run_id)
    assert len(record["segments"]) == 2
    assert record["segments"][1]["resumed"] is True
    assert run_id in second.output


def test_noctis_run_resume_on_an_unknown_run_exits_nonzero_naming_the_run_tree(tmp_path):
    result = runner.invoke(
        app, ["run", "--config", _config(tmp_path), "--resume", "20260101T000000Z-nope00"]
    )

    assert result.exit_code == 1
    assert "20260101T000000Z-nope00" in result.output


def test_noctis_run_resume_refuses_a_completed_run(tmp_path):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    record = _record(run_dir)
    record["run"]["completed_utc"] = "2026-07-27T15:22:33.418Z"
    record["run"]["status"] = "completed"
    (run_dir / RUN_RECORD_NAME).write_text(json.dumps(record))

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 1
    assert "completed" in result.output


def test_editing_config_between_segments_does_not_move_the_frozen_keys(tmp_path):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\nchampion_count: 5\n")
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    digest_before = _record(run_dir)["inputs"]["settings"]["digest"]

    Path(cfg).write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n"
        "promotion:\n  metric: total_return\nchampion_count: 1\n"
    )
    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 0, result.output
    frozen = _record(run_dir)["inputs"]
    assert frozen["settings"]["resolved"]["promotion"]["metric"] == "sortino"
    assert frozen["settings"]["resolved"]["champion_count"] == 5
    assert frozen["settings"]["digest"] == digest_before


def test_editing_the_mandate_profile_between_segments_does_not_change_the_frozen_text(tmp_path):
    profiles = tmp_path / "mandate" / "profiles"
    profiles.mkdir(parents=True)
    profile = profiles / "aggressive.md"
    profile.write_text("---\nsummary: hunt volatility\n---\n\nTrade the most volatile names.\n")
    cfg = _config(tmp_path, f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n")

    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    profile.write_text("---\nsummary: something else entirely\n---\n\nBuy and hold index funds.\n")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 0, result.output
    frozen = _record(run_dir)["inputs"]["mandate"]
    assert frozen["source"] == "profile:aggressive"
    assert frozen["text"] == "Trade the most volatile names."
    assert frozen["summary"] == "hunt volatility"
    assert "index funds" in profile.read_text()  # the edit really did land on disk
    assert "index funds" not in result.output


def test_the_frozen_mandates_overlay_still_shapes_the_resumed_session(tmp_path):
    """The mandate is frozen with the overlay it applied, and the frozen settings are already
    post-overlay — so a resumed session is steered exactly as the first one was, and rewriting the
    profile's ``config:`` block cannot re-steer it."""
    profiles = tmp_path / "mandate" / "profiles"
    profiles.mkdir(parents=True)
    profile = profiles / "aggressive.md"
    profile.write_text("---\nconfig:\n  promotion:\n    metric: sortino\n---\n\nHunt volatility.\n")
    cfg = _config(tmp_path, f"mandate_dir: {tmp_path}/mandate\nresearch:\n  mandate: aggressive\n")
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    profile.write_text(
        "---\nconfig:\n  promotion:\n    metric: total_return\n---\n\nBuy the index.\n"
    )

    resumed = _resume_inputs(cfg, run_dir.name)

    assert resumed.settings.promotion.metric == "sortino"
    assert resumed.overrides == ["promotion.metric=sortino"]
    assert resumed.mandate is not None and resumed.mandate.source == "profile:aggressive"
    assert _record(run_dir)["inputs"]["mandate"]["config_overrides"] == {
        "promotion.metric": "sortino"
    }


def test_the_safety_gate_re_resolves_on_resume_exactly_as_at_first_start(tmp_path):
    """``mode: live`` without ``ALLOW_LIVE`` is a hard startup error, on a resume as on a first
    start — the gate is re-resolved from the current process, never restored from a record."""
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    Path(cfg).write_text(f"mode: live\ndata:\n  lake_dir: {tmp_path}/lake\n")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 1
    assert "SAFETY GATE" in result.output
    assert "ALLOW_LIVE" in result.output


def test_a_paper_runs_results_cannot_acquire_live_segments(tmp_path, monkeypatch):
    """Both gates open on the second night is a legal *new* run — it is not a legal continuation
    of a paper run's evidence."""
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    Path(cfg).write_text(f"mode: live\ndata:\n  lake_dir: {tmp_path}/lake\n")
    monkeypatch.setenv("ALLOW_LIVE", "true")

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 1
    assert "paper" in result.output and "live" in result.output
    assert len(_record(run_dir)["segments"]) == 1  # the refusal happened before a segment opened


def test_the_frozen_mandate_wins_over_a_mandate_flag_rather_than_being_silently_ignored(tmp_path):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_dir.name, "--directive", "do something else"]
    )

    assert result.exit_code == 1
    assert "frozen" in result.output


def test_no_secret_reaches_a_resumed_runs_record(tmp_path, monkeypatch):
    from noctis.config.settings import SECRET_FIELDS

    secrets = {field: f"sk-{field}-do-not-leak" for field in SECRET_FIELDS}
    for field, value in secrets.items():
        monkeypatch.setenv(field.upper(), value)
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_dir.name])

    assert result.exit_code == 0, result.output
    text = (run_dir / RUN_RECORD_NAME).read_text()
    for value in secrets.values():
        assert value not in text


# ── addressing a resume: latest, a record path, @label, --label (story #133) ───────────────


def _echoed_run_id(result) -> str:
    """The id the kickoff banner names — ``Run:`` on a fresh run, ``Resumed run:`` on a resume.

    Read from the output on purpose: an operator addressing a run by an alias is told which run
    they actually reached, and that echo is the contract the tests below hold to.
    """
    (line,) = [
        text
        for text in result.output.splitlines()
        if text.startswith("Run: ") or text.startswith("Resumed run: ")
    ]
    return line.split(": ", 1)[1].strip()


def _seal(run_dir: Path) -> None:
    """Seal a run the way ``--finish`` will (story #136): ``completed``, so it refuses resume."""
    record = _record(run_dir)
    record["run"]["status"] = "completed"
    record["run"]["completed_utc"] = record["run"]["last_active_utc"]
    (run_dir / RUN_RECORD_NAME).write_text(json.dumps(record))


def test_noctis_run_label_stores_the_alias_in_the_record_and_the_index_and_runs_shows_it(tmp_path):
    cfg = _config(tmp_path)

    result = runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"])

    assert result.exit_code == 0, result.output
    run_dir = _run_dirs(tmp_path)[0]
    assert _record(run_dir)["run"]["label"] == "nightly-momo"
    index = json.loads((_runs_dir(tmp_path) / "index.json").read_text())
    assert [entry["label"] for entry in index["runs"]] == ["nightly-momo"]
    listed = runner.invoke(app, ["runs", "--all", "--config", cfg])
    assert listed.exit_code == 0, listed.output
    assert "nightly-momo" in listed.output


def test_noctis_run_resume_latest_continues_the_most_recent_run_and_names_its_id(tmp_path):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    newest = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg]))

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "latest"])

    assert result.exit_code == 0, result.output
    assert _echoed_run_id(result) == newest
    assert len(_record(_runs_dir(tmp_path) / newest)["segments"]) == 2
    assert len(_run_dirs(tmp_path)) == 2  # nothing new was minted


def test_noctis_run_resume_latest_skips_a_completed_run(tmp_path):
    cfg = _config(tmp_path)
    first = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg]))
    second = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg]))
    _seal(_runs_dir(tmp_path) / second)

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "latest"])

    assert result.exit_code == 0, result.output
    assert _echoed_run_id(result) == first


def test_noctis_run_resume_latest_with_no_resumable_run_exits_nonzero_saying_why(tmp_path):
    cfg = _config(tmp_path)
    _seal(_runs_dir(tmp_path) / _echoed_run_id(runner.invoke(app, ["run", "--config", cfg])))

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "latest"])

    assert result.exit_code == 1
    assert "RESUME" in result.output
    assert "completed" in result.output
    assert "noctis runs" in result.output
    assert len(_run_dirs(tmp_path)) == 1  # a refused resume never mints a run instead


def test_noctis_run_resume_by_the_path_of_the_record_you_are_looking_at(tmp_path):
    cfg = _config(tmp_path)
    run_id = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg]))
    record_path = _runs_dir(tmp_path) / run_id / RUN_RECORD_NAME

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", str(record_path)])

    assert result.exit_code == 0, result.output
    assert _echoed_run_id(result) == run_id
    assert len(_record(record_path.parent)["segments"]) == 2


def test_noctis_run_resume_by_label(tmp_path):
    cfg = _config(tmp_path)
    momo = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"]))
    runner.invoke(app, ["run", "--config", cfg, "--label", "sector-specialist"])

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "@nightly-momo"])

    assert result.exit_code == 0, result.output
    assert _echoed_run_id(result) == momo
    assert len(_record(_runs_dir(tmp_path) / momo)["segments"]) == 2


def test_noctis_run_resume_on_an_ambiguous_label_refuses_and_names_the_candidates(tmp_path):
    cfg = _config(tmp_path)
    first = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"]))
    second = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"]))

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "@nightly-momo"])

    assert result.exit_code == 1
    assert first in result.output and second in result.output
    assert len(_run_dirs(tmp_path)) == 2  # neither continued, and nothing new was minted


def test_noctis_run_resume_on_an_unknown_label_exits_nonzero_saying_how_to_find_runs(tmp_path):
    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"])

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "@no-such-label"])

    assert result.exit_code == 1
    assert "no-such-label" in result.output
    assert "noctis runs" in result.output


def test_a_label_may_be_attached_or_changed_on_a_resume_without_moving_the_id(tmp_path):
    """A label decides nothing — it is a nickname — so fixing a typo'd one must not cost a run.
    The id is untouched, which is what keeps the alias convenience rather than identity."""
    cfg = _config(tmp_path)
    run_id = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"]))

    result = runner.invoke(
        app, ["run", "--config", cfg, "--resume", run_id, "--label", "nightly-momentum"]
    )

    assert result.exit_code == 0, result.output
    assert _echoed_run_id(result) == run_id
    record = _record(_runs_dir(tmp_path) / run_id)
    assert record["run"]["run_id"] == run_id
    assert record["run"]["label"] == "nightly-momentum"


def test_resuming_without_a_label_keeps_the_one_the_run_already_carries(tmp_path):
    cfg = _config(tmp_path)
    run_id = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"]))

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", run_id])

    assert result.exit_code == 0, result.output
    assert _record(_runs_dir(tmp_path) / run_id)["run"]["label"] == "nightly-momo"


def test_a_completed_run_addressed_by_label_refuses_by_its_id_not_by_the_alias(tmp_path):
    cfg = _config(tmp_path)
    run_id = _echoed_run_id(runner.invoke(app, ["run", "--config", cfg, "--label", "nightly-momo"]))
    _seal(_runs_dir(tmp_path) / run_id)

    result = runner.invoke(app, ["run", "--config", cfg, "--resume", "@nightly-momo"])

    assert result.exit_code == 1
    assert run_id in result.output
    assert "completed" in result.output


# ── the resumed process: rehydrated settings and run-scoped state ──────────────────────────


def _resume_inputs(cfg: str, run_id: str):
    from noctis.bootstrap import resolve_session

    return resolve_session(cfg, require_gate=True, resume=run_id)


def test_a_resumed_session_reads_the_frozen_config_and_the_live_paths(tmp_path, monkeypatch):
    cfg = _config(tmp_path, "promotion:\n  metric: sortino\nresearch:\n  min_trials: 20\n")
    runner.invoke(app, ["run", "--config", cfg])
    run_id = _run_dirs(tmp_path)[0].name
    Path(cfg).write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n"
        "promotion:\n  metric: sharpe\nresearch:\n  min_trials: 2\ntime_limit_hours: 9.0\n"
    )
    monkeypatch.setenv("DATABENTO_API_KEY", "sk-from-the-live-env")

    resumed = _resume_inputs(cfg, run_id)

    assert resumed.settings.promotion.metric == "sortino"  # frozen
    assert resumed.settings.research.min_trials == 20  # frozen
    assert resumed.settings.time_limit_hours == 9.0  # live (per-process budget)
    assert resumed.settings.databento_api_key == "sk-from-the-live-env"  # live (secret)
    assert resumed.mode == "paper"


def test_a_resumed_run_picks_up_its_own_run_scoped_state(tmp_path):
    """Champions, account, memory and the strategy tiers belong to the run that produced them —
    a resumed segment reads that run's tree, not the workspace's."""
    from noctis.bootstrap import open_run_store

    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_dir = _run_dirs(tmp_path)[0]
    (run_dir / "state").mkdir(exist_ok=True)
    (run_dir / "state" / "marker.json").write_text('{"from": "segment 0"}')

    resumed = _resume_inputs(cfg, run_dir.name)
    store = open_run_store(resumed.settings, argv=["run"], run_id=run_dir.name, resume=True)
    try:
        settings = resumed.settings
        assert Path(settings.run_dir) == run_dir
        assert Path(settings.state_dir) == run_dir / "state"
        assert Path(settings.memory_path) == run_dir / "memory" / "MEMORY.md"
        assert Path(settings.reports_dir) == run_dir / "reports"
        assert (Path(settings.state_dir) / "marker.json").is_file()
        # …while the shared lake stays workspace-level, never following a run.
        assert Path(settings.data.lake_dir) == tmp_path / "lake"
    finally:
        store.close(reason="stopped")


def test_a_resumed_session_still_fails_informatively_without_the_vendor_key(tmp_path):
    """Secrets come from the live ``.env``; without them a resumed run fails exactly where a
    fresh one does, through the existing missing-key path."""
    from noctis.bootstrap import MissingVendorKey, build_lake

    cfg = _config(tmp_path)
    runner.invoke(app, ["run", "--config", cfg])
    run_id = _run_dirs(tmp_path)[0].name

    resumed = _resume_inputs(cfg, run_id)

    with pytest.raises(MissingVendorKey) as excinfo:
        build_lake(resumed.settings, require_vendor=True)
    assert "DATABENTO_API_KEY" in str(excinfo.value)
