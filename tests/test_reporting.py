"""Close-of-day report generation and retrieval."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from noctis.cli import app
from noctis.reporting import (
    ReportData,
    Trade,
    latest_report,
    render_report,
    sweep_stale_reports,
    write_report,
    write_report_json,
)

SECTIONS = [
    "## Summary",
    "## Trades",
    "## Open positions",
    "## Champion changes",
    "## Current champions",
    "## Forward track record (live-holdout)",
    "## Research",
    "## Notable events",
]


def test_empty_day_report_has_all_sections():
    text = render_report(ReportData(as_of="2026-07-03"))
    assert text.startswith("# Close-of-day report — 2026-07-03")
    for header in SECTIONS:
        assert header in text
    assert "_No trades this session._" in text
    assert "_No open positions._" in text
    assert "_No forward record yet._" in text  # empty forward ledger renders gracefully
    assert "Cumulative P&L" not in text  # no continuous account yet → no cumulative line


def test_report_shows_the_forward_track_record():
    data = ReportData(
        as_of="2026-07-07",
        forward=[
            {
                "family": "sma_crossover",
                "forward_pnl": 180.0,
                "realized_pnl": 30.0,
                "unrealized_pnl": 150.0,
                "sessions_traded": 2,
                "opened_session": "2026-07-05",
            }
        ],
    )
    text = render_report(data)
    assert "## Forward track record (live-holdout)" in text
    assert "sma_crossover" in text
    assert "+180.00" in text  # cumulative forward P&L
    assert "2026-07-05" in text  # since opened_session
    assert "_No forward record yet._" not in text


def test_report_shows_the_continuous_account_curve():
    """With a carried paper account, the summary shows the cumulative P&L since inception
    — the single continuous curve, not just the day's delta."""
    data = ReportData(
        as_of="2026-07-07",
        start_equity=101_500.0,  # the carried value, not a fresh 100k
        end_equity=102_000.0,
        cumulative_pnl=2_000.0,
        account_opened="2026-07-01",
    )
    text = render_report(data)
    assert "- Cumulative P&L since 2026-07-01: +2,000.00" in text


def test_write_report_json_round_trips(tmp_path):
    """The structured JSON report recurses into nested Trade dataclasses and preserves the
    research block (minted / promoted specs, findings) for a frontend to consume."""
    data = ReportData(
        as_of="2026-07-03",
        mode="paper",
        start_equity=100_000.0,
        end_equity=101_000.0,
        realized_pnl=1_000.0,
        trades=[Trade("AAPL", "buy", 10, 190.5, "spec_minted_1 breakout")],
        positions={"AAPL": 10},
        champions=[{"family": "spec_minted_1", "params": {"fast": 8, "slow": 21}}],
        research={
            "iterations": 4,
            "promotions": 1,
            "minted": ["spec_minted_1"],
            "promoted_specs": ["spec_minted_1"],
            "findings": ["MINTED spec family spec_minted_1 — breakout"],
        },
    )
    path = write_report_json(data, tmp_path)
    assert path == tmp_path / "2026-07-03.json"

    loaded = json.loads(path.read_text())
    assert loaded["as_of"] == "2026-07-03"
    assert loaded["realized_pnl"] == 1_000.0
    assert loaded["trades"][0] == {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "price": 190.5,
        "rationale": "spec_minted_1 breakout",
    }
    assert loaded["research"]["minted"] == ["spec_minted_1"]
    assert loaded["research"]["promoted_specs"] == ["spec_minted_1"]
    assert loaded["research"]["findings"][0].startswith("MINTED spec family")


def test_report_includes_trades_promotions_events():
    data = ReportData(
        as_of="2026-07-03",
        start_equity=100_000.0,
        end_equity=101_500.0,
        realized_pnl=1_500.0,
        trades=[Trade("AAPL", "BUY", 10, 190.5, "SMA crossover long")],
        positions={"AAPL": 10},
        promotions=[
            {"family": "sma_crossover", "params": {"fast": 5}, "rationale": "beat weakest"}
        ],
        research={
            "iterations": 12,
            "promotions": 1,
            "rejections": 8,
            "dead_ends": 3,
            "findings": ["momentum works in the morning"],
        },
        events=["Risk halt: daily loss limit reached on TSLA"],
    )
    text = render_report(data)
    assert "AAPL" in text and "SMA crossover long" in text
    assert "1.50%" in text  # session return
    assert "PROMOTED sma_crossover" in text
    assert "momentum works in the morning" in text
    assert "Risk halt" in text


def test_report_shows_undecided_strategies():
    """The research section lists strategies a session authored but left unresolved, so the
    close report and QA rollups surface the honesty check beside the counters."""
    data = ReportData(
        as_of="2026-07-03",
        research={
            "iterations": 5,
            "promotions": 0,
            "rejections": 1,
            "dead_ends": 0,
            "undecided": ["draft_a", "draft_b"],
        },
    )
    text = render_report(data)
    assert "Undecided" in text
    assert "draft_a" in text and "draft_b" in text


def test_report_omits_empty_undecided():
    """A session with nothing undecided renders no undecided line — an empty entry is silent,
    not noise."""
    data = ReportData(
        as_of="2026-07-03",
        research={"iterations": 5, "undecided": []},
    )
    assert "Undecided" not in render_report(data)


# ── session rollup + per-candidate trail from the ledger (story #74) ─────────────────────────
def _ledgered_research(escalations: int = 1) -> dict:
    """A research block carrying one ledgered session's rollup + candidate trail — the shape
    ``assemble_report`` threads in from a session ledger's ``report_view``."""
    return {
        "iterations": 3,
        "promotions": 1,
        "rejections": 1,
        "sessions": [
            {
                "session_id": "session-1",
                "rollup": {
                    "session_id": "session-1",
                    "theses": 3,
                    "authored": 2,
                    "validation_failures": 1,
                    "trials": 12,
                    "verdicts": {"approve": 1, "reject": 1},
                    "promoted": 1,
                    "undecided": 0,
                    "escalations": escalations,
                    "tokens_by_stage": {"formulate": 26, "decide": 14, "author": 40},
                    "tokens_by_model": {"driver": 40, "coder-paid": 40},
                    "note": "max_episodes",
                },
                "candidates": [
                    {
                        "strategy": "momo_1",
                        "thesis": "buy strength above the average",
                        "stages": ["match", "author", "optimize", "decide"],
                        "trials": 5,
                        "best_metric": 1.2,
                        "verdict": "reject",
                        "promoted": False,
                        "outcome": "rejected",
                        "oracle": ["rally", "grind"],
                    },
                    {
                        "strategy": "rev_2",
                        "thesis": "fade the spike",
                        "stages": ["match", "author", "optimize", "decide"],
                        "trials": 7,
                        "best_metric": 2.5,
                        "verdict": "approve",
                        "promoted": True,
                        "outcome": "promoted",
                        "oracle": ["spike_fade", "calm"],
                    },
                ],
            }
        ],
    }


def test_report_renders_the_session_rollup_and_candidate_trail():
    text = render_report(ReportData(as_of="2026-07-03", research=_ledgered_research()))
    # The session rollup names every field of the epic's list.
    assert "session-1" in text
    assert "Theses formulated: 3" in text
    assert "Files authored: 2" in text
    assert "Validation failures: 1" in text
    assert "Trials run: 12" in text
    assert "approve=1" in text and "reject=1" in text
    assert "Undecided: 0" in text
    assert "Escalations: 1" in text
    assert "formulate=26" in text and "author=40" in text  # tokens by stage
    assert "driver=40" in text and "coder-paid=40" in text  # tokens by model
    # The per-candidate trail: each candidate's formulate → decide walk.
    assert "momo_1" in text and "rejected" in text
    assert "rev_2" in text and "promoted" in text
    assert "fade the spike" in text  # the thesis (the FORMULATE step) frames the trail
    # The fixed oracle each candidate was gated against (#86) — auditable per candidate.
    assert "rally, grind" in text
    assert "spike_fade, calm" in text


def test_report_renders_escalations_as_zero_when_none_occurred():
    text = render_report(ReportData(as_of="2026-07-03", research=_ledgered_research(escalations=0)))
    assert "Escalations: 0" in text


def test_report_surfaces_the_session_end_note(tmp_path):
    # Story #94: the session rollup surfaces its session_end note (rendered generically), so an
    # operator distinguishes "ran out of coder budget" (author_budget_exhausted) from "ran out of
    # ideas" (formulate_failed) or a plain budget stop at a glance.
    research = _ledgered_research()
    research["sessions"][0]["rollup"]["note"] = "author_budget_exhausted"
    text = render_report(ReportData(as_of="2026-07-03", research=research))
    assert "author_budget_exhausted" in text


def test_ledgerless_report_is_byte_for_byte_unchanged():
    """A session without a ledger carries no ``sessions`` key; its render must be byte-identical
    to the pre-#74 output (the graceful-degradation acceptance criterion)."""
    data = ReportData(
        as_of="2026-07-03",
        start_equity=100_000.0,
        end_equity=101_500.0,
        realized_pnl=1_500.0,
        trades=[Trade("AAPL", "BUY", 10, 190.5, "SMA crossover long")],
        positions={"AAPL": 10},
        promotions=[
            {"family": "sma_crossover", "params": {"fast": 5}, "rationale": "beat weakest"}
        ],
        research={
            "iterations": 12,
            "promotions": 1,
            "rejections": 8,
            "dead_ends": 3,
            "undecided": ["draft_a"],
            "findings": ["momentum works in the morning"],
            "minted": ["spec_x"],
            "promoted_specs": ["spec_x"],
        },
        events=["Risk halt: daily loss limit reached on TSLA"],
    )
    expected = (
        "# Close-of-day report — 2026-07-03\n\n**Mode:** paper\n\n## Summary\n\n"
        "- Start equity: 100,000.00\n- End equity: 101,500.00\n- Session return: 1.50%\n"
        "- Realized P&L: 1,500.00\n\n## Trades\n\n| Symbol | Side | Qty | Price | Rationale |\n"
        "|---|---|---:|---:|---|\n| AAPL | BUY | 10 | 190.5000 | SMA crossover long |\n\n"
        "## Open positions\n\n- AAPL: 10\n\n## Champion changes\n\n"
        "- PROMOTED sma_crossover {'fast': 5} — beat weakest\n\n## Current champions\n\n"
        "_No champions yet._\n\n## Forward track record (live-holdout)\n\n"
        "_No forward record yet._\n\n"
        "## Research\n\n- Candidates tried: 12\n- Promotions: 1\n- Rejections: 8\n- Dead ends: 3\n"
        "- Undecided (authored, no verdict):\n  - draft_a\n- Notable findings:\n"
        "  - momentum works in the morning\n\n## Notable events\n\n"
        "- Risk halt: daily loss limit reached on TSLA\n"
    )
    assert render_report(data) == expected


def test_write_and_retrieve_report(tmp_path):
    path = write_report(ReportData(as_of="2026-07-03"), tmp_path / "reports")
    assert path.is_file()
    assert latest_report(tmp_path / "reports") == path


def test_report_cli_generates_and_prints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"state_dir: {tmp_path}/state/\n")
    result = CliRunner().invoke(app, ["report", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Close-of-day report" in result.output


# ── overwrite protection (plan 4) ──────────────────────────────────────────────────────────
def test_write_report_archives_a_differing_prior(tmp_path):
    reports = tmp_path / "reports"
    first = write_report(ReportData(as_of="2026-07-03", end_equity=100_000.0), reports)
    data_b = ReportData(as_of="2026-07-03", end_equity=105_000.0)  # differs
    second = write_report(data_b, reports)

    assert second == first  # canonical path stays reports/2026-07-03.md
    archives = list((reports / "archive").glob("2026-07-03.*.md"))
    assert len(archives) == 1  # the prior was moved, not clobbered
    # The overwrite is visible in the day's own report and its events.
    assert "Overwrote existing report for 2026-07-03 (prior archived)" in data_b.events
    assert "Overwrote existing report for 2026-07-03 (prior archived)" in second.read_text()


def test_write_report_json_archives_a_differing_prior(tmp_path):
    reports = tmp_path / "reports"
    write_report_json(ReportData(as_of="2026-07-03", end_equity=1.0), reports)
    write_report_json(ReportData(as_of="2026-07-03", end_equity=2.0), reports)
    assert len(list((reports / "archive").glob("2026-07-03.*.json"))) == 1


def test_identical_report_rewrite_is_a_no_op(tmp_path):
    reports = tmp_path / "reports"
    data = ReportData(as_of="2026-07-03", end_equity=101_000.0)
    write_report(data, reports)
    write_report(data, reports)  # byte-identical → no archive, no overwrite note
    assert not (reports / "archive").exists()
    assert data.events == []


# ── stale future-dated sweep (plan 4) ──────────────────────────────────────────────────────
def test_sweep_stale_reports_dry_run_lists_without_moving(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2099-01-01.md").write_text("future")
    (reports / "2020-01-01.md").write_text("past")
    stale = sweep_stale_reports(reports, apply=False)
    assert [p.name for p in stale] == ["2099-01-01.md"]  # only future-dated
    assert (reports / "2099-01-01.md").is_file()  # dry run left it in place
    assert not (reports / "archive").exists()


def test_sweep_stale_reports_apply_moves_future_only(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2099-01-01.md").write_text("future md")
    (reports / "2099-01-01.json").write_text("{}")
    (reports / "2020-01-01.md").write_text("past")
    (reports / "notes.md").write_text("not a dated report")  # never swept
    moved = sweep_stale_reports(reports, apply=True)
    assert {p.name for p in moved} == {"2099-01-01.md", "2099-01-01.json"}
    assert (reports / "archive" / "2099-01-01.md").is_file()
    assert (reports / "archive" / "2099-01-01.json").is_file()
    assert not (reports / "2099-01-01.md").exists()
    assert (reports / "2020-01-01.md").is_file()  # past untouched
    assert (reports / "notes.md").is_file()  # non-date file untouched
