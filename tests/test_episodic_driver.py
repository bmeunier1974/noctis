"""The episodic research driver (epic #62 / story #68) — the deterministic session machine that
owns the FORMULATE → MATCH → AUTHOR → OPTIMIZE → DECIDE protocol and calls the model only at the
two judgment episodes.

Two harnesses, one contract:

* **Deterministic** tests drive the stage machine with plain fake episodes and a fake toolbox —
  zero LLM — locking the stage order, the ledger persistence, the budget stops (max_episodes,
  wall-clock, stop_event), and each per-stage failed-episode policy (formulate → end, author →
  skip, decide → re-ask once then undecided).
* **End to end**, a real :class:`~noctis.research.episode.EpisodeRunner` fed scripted ``Turn``s by
  a fake client drives a real :class:`~noctis.research.tools.ResearchToolbox` (with a fake coder)
  through a whole session that authors, optimizes, and reaches a GATED verdict with a complete
  ledger — and, with the sweep withheld, that the real exhaustion gate refuses the verdict exactly
  as today and the driver surfaces the refusal honestly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from noctis.research import Capabilities
from noctis.research.driver import (
    _CHEAP_MAX_BARS,
    DECIDE_CONTRACT,
    FORMULATE_CONTRACT,
    DecideOutput,
    FormulateOutput,
    character_to_profile,
    make_episodes,
    run_episodic_research,
)
from noctis.research.episode import (
    API_ERROR,
    MISFIRES_EXHAUSTED,
    OK,
    EpisodeResult,
    EpisodeRunner,
)
from noctis.research.ledger import SessionLedger
from noctis.research.llm import ToolCall, Turn
from tests.test_research_tools import PROBE, _make_toolbox

_FORMULATE_TOOL = FORMULATE_CONTRACT.name
_DECIDE_TOOL = DECIDE_CONTRACT.name

# The real exhaustion-gate refusal shape a below-floor verdict comes back as today.
_EXHAUSTION_REFUSAL = (
    "exhaustion gate: 'x' has only 0 distinct parameter set(s) in its journal and no completed "
    "sweep. Explore the parameter space first"
)


# ── typed episode-output builders (what a formulate/decide episode emits) ───────────────────
def formulate_ok(**over) -> EpisodeResult[FormulateOutput]:
    fields = {
        "thesis": "Buy strength above the moving average while the up-move clears cost.",
        "style": "momentum",
        "class_tag": "intraday momentum",
        "timeframe": "1m",
        "cost_arithmetic": "median 1m move ~8bp vs the 4bp round trip",
        "symbol_character": "liquid trending names",
        "scenario_intent": "one directional long tape and one no-trade selloff tape",
        "param_space_sketch": "lookback 5-40",
    }
    fields.update(over)
    return EpisodeResult(OK, FormulateOutput(**fields), "fake/model", tokens=12, misfires=0)


def formulate_fail() -> EpisodeResult[FormulateOutput]:
    return EpisodeResult(MISFIRES_EXHAUSTED, None, "fake/model", tokens=4, misfires=3, note="junk")


def decide_ok(verdict: str = "reject", **over) -> EpisodeResult[DecideOutput]:
    fields = {
        "verdict": verdict,
        "reason": "gross edge below cost on the fit panel",
        "class_exhausted": False,
        "class_tag": "intraday momentum",
    }
    fields.update(over)
    return EpisodeResult(OK, DecideOutput(**fields), "fake/model", tokens=8, misfires=0)


def decide_fail() -> EpisodeResult[DecideOutput]:
    return EpisodeResult(API_ERROR, None, "fake/model", tokens=2, misfires=0, note="backend down")


# ── fake episodes (a completions counter mirrors the episode runner's budget tally) ─────────
class Episodes:
    def __init__(self, formulate_script, decide_script):
        self._f = list(formulate_script)
        self._d = list(decide_script)
        self.completions = 0
        self.formulate_calls = 0
        self.formulate_correctives: list[str | None] = []
        self.decide_calls: list[tuple[str, str | None]] = []

    def formulate(self, *, corrective=None):
        self.completions += 1
        self.formulate_calls += 1
        self.formulate_correctives.append(corrective)
        return self._f.pop(0)

    def decide(self, strategy, *, corrective=None):
        self.completions += 1
        self.decide_calls.append((strategy, corrective))
        return self._d.pop(0)


# ── recipe-return builders: what a backtest/sweep hands the OPTIMIZE recipe (story #70) ──────
def bt(metric=None, **extra):
    """A ``tool_run_backtest`` return; ``metric`` is the panel ``avg_test_metric`` the recipe's
    improvement checks read (absent ⇒ the recipe treats the round as un-scored)."""
    out = {"ok": True}
    if metric is not None:
        out["avg_test_metric"] = metric
    out.update(extra)
    return out


def sw(params=None, test=None, n_trials=3, **extra):
    """A ``tool_run_sweep`` return; ``top_trials`` carries the best params the recipe confirms and
    narrows around, and ``n_trials`` is what the journal counts toward the exhaustion floor."""
    out = {"ok": True, "sweep_completed": True, "n_trials": n_trials}
    if params is not None:
        out["top_trials"] = [{"params": params, "test": test}]
    out.update(extra)
    return out


# ── a fake exhausted-class registry mirroring ExhaustedClassRegistry.is_exhausted ───────────
class FakeExhausted:
    def __init__(self, tags=()):
        self._tags = {" ".join(str(t).split()).lower() for t in tags}

    def is_exhausted(self, tag):
        key = " ".join(str(tag).split()).lower()
        return {"reason": "a prior session ruled it out"} if key in self._tags else None


# The market digest the driver's cost-arithmetic check compares against; the numbers here (4, 2,
# 0, 8) overlap the default ``formulate_ok`` cost arithmetic (1m, 8bp, 4bp) so the check passes.
_FAKE_DIGEST = {
    "round_trip_cost_bp": 4.0,
    "fee_bps_per_side": 2.0,
    "slippage_bps_per_side": 0.0,
    "median_abs_bar_move_bp": 8.0,
}


# ── a fake toolbox: only the gated surface the driver drives, with honest bookkeeping ───────
class FakeToolbox:
    def __init__(
        self,
        *,
        write_result=None,
        verdict_results=None,
        log=None,
        screen_result=None,
        backtest_results=None,
        sweep_results=None,
        exhausted=(),
    ):
        self.exhausted = FakeExhausted(exhausted)
        self.promotions = 0
        self.rejections = 0
        self.author_calls = 0
        self.escalations = 0
        self.strategies_touched: list[str] = []
        self.undecided: set[str] = set()
        self.writes: list[dict] = []
        self.screens: list[dict] = []
        self.backtests: list[dict] = []
        self.sweeps: list[dict] = []
        self.rejects: list[dict] = []
        self.evaluations: list[dict] = []
        self._write_result = write_result or {"ok": True}
        self._verdicts = list(verdict_results or [{"ok": True, "status": "rejected"}])
        self._log = log or {"top_trials": [{"params": {"lookback": 20}}]}
        # Per-call recipe scripts (OPTIMIZE, #70): each call pops the next result, the last one
        # repeating. ``None`` ⇒ the pre-#70 constant return (a bare ``ok`` / a completed sweep).
        self._backtest_results = list(backtest_results) if backtest_results else None
        self._sweep_results = list(sweep_results) if sweep_results else None
        # Default: an empty structural screen (no lake match) — the driver then falls back to
        # the composition-root fit panel exactly as the pre-#69 passthrough did.
        self._screen_result = (
            screen_result
            if screen_result is not None
            else {"suggested_fit": [], "reserved_holdout": []}
        )

    @staticmethod
    def _next(script, default):
        if script is None:
            return dict(default)
        return dict(script.pop(0) if len(script) > 1 else script[0])

    def market_context(self):
        return dict(_FAKE_DIGEST)

    def tool_screen_symbols(self, trend="any", volatility="any", liquidity="any", symbols=None):
        self.screens.append({"trend": trend, "volatility": volatility, "liquidity": liquidity})
        return dict(self._screen_result)

    def tool_write_strategy(self, **kwargs):
        self.writes.append(kwargs)
        # Mirror the real toolbox: an escalated write bumps the session escalation counter (the
        # summary surfaces it) whether the paid model then authors the file or also fails.
        if self._write_result.get("escalated"):
            self.escalations += 1
        if "error" in self._write_result:
            return dict(self._write_result)
        name = kwargs["name"]
        self.strategies_touched.append(name)
        self.undecided.add(name)
        self.author_calls += 1
        return dict(self._write_result)

    def tool_run_backtest(self, **kwargs):
        self.backtests.append(kwargs)
        return self._next(self._backtest_results, {"ok": True})

    def tool_run_sweep(self, **kwargs):
        self.sweeps.append(kwargs)
        return self._next(self._sweep_results, {"ok": True, "sweep_completed": True})

    def tool_get_experiment_log(self, **kwargs):
        return dict(self._log)

    def tool_reject_strategy(self, **kwargs):
        self.rejects.append(kwargs)
        result = self._next_verdict()
        if "error" not in result:
            self.rejections += 1
            self.undecided.discard(kwargs["name"])
        return result

    def tool_evaluate_vs_champion(self, **kwargs):
        self.evaluations.append(kwargs)
        result = self._next_verdict()
        if "error" not in result and result.get("promoted"):
            self.promotions += 1
            self.undecided.discard(kwargs["name"])
        return result

    def _next_verdict(self):
        if len(self._verdicts) > 1:
            return dict(self._verdicts.pop(0))
        return dict(self._verdicts[0])


def _drive(episodes, toolbox, *, max_episodes, ledger, **over):
    kwargs = dict(
        toolbox=toolbox,
        ledger=ledger,
        formulate=episodes.formulate,
        decide=episodes.decide,
        fit_symbols=["AAA", "BBB", "CCC"],
        budget_minutes=60.0,
        max_episodes=max_episodes,
        completions=lambda: episodes.completions,
    )
    kwargs.update(over)
    return run_episodic_research(**kwargs)


def _drive_optimize(tmp_path, session, *, backtest_results, sweep_results, sweep_trials=4, **over):
    """Drive one full cycle (a reject verdict) with the OPTIMIZE recipe fed per-call backtest/sweep
    scripts; returns the fake toolbox, its ledger, and the episode script for branch assertions."""
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox(backtest_results=backtest_results, sweep_results=sweep_results)
    ledger = SessionLedger(tmp_path, session)
    _drive(episodes, box, max_episodes=2, ledger=ledger, sweep_trials=sweep_trials, **over)
    return box, ledger, episodes


# ── 1. a full cycle: verdict through the gated method, complete ledger ──────────────────────
def test_full_cycle_reaches_a_verdict_and_a_complete_ledger(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "s1")

    summary = _drive(episodes, box, max_episodes=2, ledger=ledger)

    # A verdict went through the gated reject method; the summary reflects the toolbox counters.
    assert box.rejects and box.rejects[0]["name"] == "intraday_momentum_1"
    assert summary.rejections == 1
    assert summary.iterations == 1
    assert summary.undecided == []
    assert summary.stopped_reason == "max_episodes"

    # A complete ledger: start, thesis, every stage in protocol order, both episodes, the
    # verdict, the session-end rollup.
    assert ledger.session_start() is not None
    assert [t.strategy for t in ledger.theses()] == ["intraday_momentum_1"]
    assert [s.stage for s in ledger.stages()] == [
        "formulate",
        "match",
        "author",
        "optimize",
        "decide",
    ]
    assert [e.stage for e in ledger.episodes()] == ["formulate", "decide"]
    verdicts = ledger.verdicts()
    assert len(verdicts) == 1 and verdicts[0].verdict == "reject"
    end = ledger.session_end()
    assert end is not None and end.formulated == 1 and end.rejected == 1


def test_optimize_recipe_runs_baseline_then_cheap_subset_sweep_then_full_panel_confirm(tmp_path):
    # The v1 recipe (story #70): a full-panel baseline, a CHEAP sweep on a subset of the fit
    # panel at reduced bar-fidelity, then a full-panel confirm of that sweep's best params.
    box, ledger, ep = _drive_optimize(
        tmp_path,
        "opt-shape",
        backtest_results=[bt(1.0), bt(1.0)],  # baseline 1.0, confirm 1.0 (no gain ⇒ no re-tune)
        sweep_results=[sw(params={"lookback": 12}, test=1.0)],
    )
    assert box.backtests[0]["symbols"] == ["AAA", "BBB", "CCC"]  # baseline: full fit panel
    assert box.backtests[0].get("params") is None  # baseline runs the shipped defaults
    assert set(box.sweeps[0]["symbols"]) < {"AAA", "BBB", "CCC"}  # cheap: a strict subset
    assert box.sweeps[0]["max_bars"] == _CHEAP_MAX_BARS  # cheap: reduced bar-fidelity
    assert box.backtests[1]["symbols"] == ["AAA", "BBB", "CCC"]  # confirm: full fit panel
    assert box.backtests[1]["params"] == {"lookback": 12}  # confirm the sweep's best params
    assert len(box.sweeps) == 1  # a stalled confirm earns no re-tune
    # Zero LLM in the recipe — only the two judgment episodes ran.
    assert ep.formulate_calls == 1 and len(ep.decide_calls) == 1


def test_optimize_promising_baseline_sweeps_the_full_cheap_budget(tmp_path):
    box, _, _ = _drive_optimize(
        tmp_path,
        "opt-promising",
        backtest_results=[bt(1.0), bt(1.0)],  # positive baseline ⇒ promising branch
        sweep_results=[sw(params={"lookback": 12}, test=1.0)],
        sweep_trials=4,
    )
    assert box.sweeps[0]["n_trials"] == 4  # the full cheap trial budget


def test_optimize_weak_baseline_still_sweeps_but_sizes_down(tmp_path):
    # A flat/negative baseline is exactly what tuning is for — still sweep (the floor needs
    # trials), but the cheap sweep is sized down (the branch point a later interpret slots into).
    box, ledger, _ = _drive_optimize(
        tmp_path,
        "opt-weak",
        backtest_results=[bt(-0.2), bt(-0.2)],  # weak baseline
        sweep_results=[sw(params={"lookback": 12}, test=-0.2)],
        sweep_trials=4,
    )
    assert len(box.sweeps) == 1  # a weak baseline still sweeps to feed the exhaustion floor
    assert box.sweeps[0]["n_trials"] == 2  # sized down (half the cheap budget)
    opt = next(s for s in ledger.stages() if s.stage == "optimize")
    assert opt.detail["weak_baseline"] is True


def test_optimize_retune_improvement_runs_a_second_narrowed_sweep(tmp_path):
    # The confirm beats the baseline meaningfully ⇒ one narrowed re-tune round on the full panel.
    box, ledger, _ = _drive_optimize(
        tmp_path,
        "opt-improve",
        backtest_results=[bt(1.0), bt(2.0), bt(2.0)],  # baseline 1.0, confirm 2.0, re-tune 2.0
        sweep_results=[
            sw(params={"lookback": 12}, test=2.0),
            sw(params={"lookback": 14}, test=2.0),
        ],
    )
    assert len(box.sweeps) == 2  # cheap sweep + exactly one narrowed re-tune
    assert box.sweeps[1].get("ranges")  # the re-tune narrowed the space around the best params
    assert box.sweeps[1]["symbols"] == ["AAA", "BBB", "CCC"]  # re-tune confirms on the full panel
    assert box.sweeps[1].get("max_bars") is None  # full-fidelity refinement (no bar truncation)
    opt = next(s for s in ledger.stages() if s.stage == "optimize")
    assert opt.detail["retune_rounds"] == 1
    assert opt.detail["best_metric"] == 2.0


def test_optimize_retune_stall_stops_without_a_second_sweep(tmp_path):
    # The confirm does not improve on the baseline ⇒ stop, keep the best-so-far, on to DECIDE.
    box, ledger, _ = _drive_optimize(
        tmp_path,
        "opt-stall",
        backtest_results=[bt(1.0), bt(1.0)],  # confirm does not beat the baseline
        sweep_results=[sw(params={"lookback": 12}, test=1.0)],
    )
    assert len(box.sweeps) == 1  # no re-tune
    opt = next(s for s in ledger.stages() if s.stage == "optimize")
    assert opt.detail["retune_rounds"] == 0
    assert opt.detail["stopped"] == "stall"


def test_optimize_retune_is_hard_capped_at_two_rounds(tmp_path):
    # An always-improving sequence still stops at two re-tunes — the cap, not a stall, ends it.
    box, ledger, _ = _drive_optimize(
        tmp_path,
        "opt-cap",
        backtest_results=[bt(1.0), bt(2.0), bt(3.0), bt(4.0), bt(5.0)],
        sweep_results=[
            sw(params={"lookback": 12}, test=2.0),
            sw(params={"lookback": 14}, test=3.0),
            sw(params={"lookback": 16}, test=4.0),
            sw(params={"lookback": 18}, test=5.0),
        ],
    )
    assert len(box.sweeps) == 3  # cheap sweep + exactly two re-tunes, never a third
    opt = next(s for s in ledger.stages() if s.stage == "optimize")
    assert opt.detail["retune_rounds"] == 2
    assert opt.detail["stopped"] == "hard_cap"


def test_optimize_budget_refusal_mid_recipe_proceeds_to_decide(tmp_path):
    # A budget-refused sweep stops the recipe honestly — no further tuning — and DECIDE runs on
    # whatever evidence exists (the gates dispose).
    box, ledger, _ = _drive_optimize(
        tmp_path,
        "opt-budget",
        backtest_results=[bt(1.0)],  # baseline succeeds
        sweep_results=[{"error": "backtest budget exhausted"}],  # cheap sweep is refused
    )
    assert len(box.backtests) == 1  # baseline only — no confirm attempted after the refusal
    assert len(box.sweeps) == 1  # the one refused cheap sweep
    assert box.rejects  # the recipe still proceeded to DECIDE
    opt = next(s for s in ledger.stages() if s.stage == "optimize")
    assert opt.detail["stopped"] == "budget"


def test_optimize_recipe_journals_trials_and_clears_the_floor_on_the_real_toolbox(tmp_path):
    # Against the REAL toolbox + journal: the recipe's trials land in the experiments journal and
    # a completed sweep clears the min-trials exhaustion floor — zero LLM.
    from noctis.research.driver import _optimize_stage

    box = _make_toolbox(tmp_path)
    detail = _optimize_stage(box, "probe", ["AAA", "BBB", "CCC"], sweep_trials=3)

    stats = box.journal.stats("probe")
    assert stats.sweep_completed  # a completed sweep clears the exhaustion floor
    assert stats.n_trials >= 2  # baseline + the cheap sweep's trials journaled
    assert detail["sweeps"] >= 1 and detail["backtests"] >= 2
    assert box._exhaustion_block("probe") is None  # the gate is satisfied by the recipe alone


def test_author_stage_passes_the_thesis_and_lineage_onto_the_brief(tmp_path):
    episodes = Episodes(
        [formulate_ok(parent_thesis="older idea", pivot_rationale="cost too high before")],
        [decide_ok("reject")],
    )
    box = FakeToolbox()
    _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "s3"))

    write = box.writes[0]
    assert write["class_tag"] == "intraday momentum"
    assert write["thesis"].startswith("Buy strength")
    assert write["parent_thesis"] == "older idea"
    assert write["pivot_rationale"] == "cost too high before"
    # The formulate output is mapped onto a StrategyBrief the author engine translates.
    brief = write["brief"]
    assert brief["thesis"].startswith("Buy strength")
    assert brief["param_space"] == "lookback 5-40"
    assert brief["scenarios"] == "one directional long tape and one no-trade selloff tape"
    assert "1m" in brief["entry_exit"]


# ── 2. per-stage failed-episode policies ────────────────────────────────────────────────────
def test_formulate_failure_ends_the_session(tmp_path):
    episodes = Episodes([formulate_fail()], [])
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "s4")

    summary = _drive(episodes, box, max_episodes=10, ledger=ledger)

    assert summary.stopped_reason == "formulate_failed"
    assert box.writes == []  # nothing authored
    assert summary.iterations == 0
    assert [e.stage for e in ledger.episodes()] == ["formulate"]  # the failed episode is ledgered
    assert ledger.session_end() is not None  # the rollup still lands


def test_author_failure_skips_the_strategy(tmp_path):
    episodes = Episodes([formulate_ok()], [])
    box = FakeToolbox(write_result={"error": "validation failed: bad scenario window"})
    ledger = SessionLedger(tmp_path, "s5")

    summary = _drive(episodes, box, max_episodes=1, ledger=ledger)

    assert box.backtests == [] and box.sweeps == []  # skipped before optimize
    assert box.rejects == [] and box.evaluations == []  # no verdict for a strategy that never was
    assert summary.undecided == []  # a refused draft never entered the undecided set
    assert summary.stopped_reason == "max_episodes"
    assert "decide" not in [s.stage for s in ledger.stages()]


# ── 2d. coder-fallback escalation surfaced on the ledger + summary (story #72) ──────────────
def test_escalated_author_records_a_paid_episode_line_and_counts_in_the_summary(tmp_path):
    # A write the paid fallback authored comes back with escalated=True + the fallback model; the
    # driver ledgers a paid AUTHOR episode line and the summary counts the escalation.
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox(
        write_result={"ok": True, "escalated": True, "author_model": "fake/coder-paid"}
    )
    ledger = SessionLedger(tmp_path, "esc1")

    summary = _drive(episodes, box, max_episodes=2, ledger=ledger)

    assert summary.escalations == 1
    author_eps = [e for e in ledger.episodes() if e.stage == "author"]
    assert len(author_eps) == 1
    assert author_eps[0].escalated is True
    assert author_eps[0].model == "fake/coder-paid"  # the paid fallback model, named on the line
    assert author_eps[0].outcome == "ok"
    # A valid file was authored, so the strategy still flows to optimize + a verdict.
    assert box.rejects and summary.rejections == 1


def test_escalated_author_that_also_fails_is_counted_and_skips_the_strategy(tmp_path):
    # The paid fallback also failed: escalated=True rides an error write. The escalation still
    # counts, an escalated author episode line records the failure, and the strategy is skipped.
    episodes = Episodes([formulate_ok()], [])
    box = FakeToolbox(
        write_result={
            "error": "validation failed: bad scenario window",
            "escalated": True,
            "author_model": "fake/coder-paid",
        }
    )
    ledger = SessionLedger(tmp_path, "esc2")

    summary = _drive(episodes, box, max_episodes=1, ledger=ledger)

    assert summary.escalations == 1  # the escalation is metered even when the paid model fails
    author_eps = [e for e in ledger.episodes() if e.stage == "author"]
    assert len(author_eps) == 1 and author_eps[0].escalated is True
    assert author_eps[0].outcome != "ok"  # the paid attempt also failed the gate
    # Skipped exactly like any author failure — no optimize, no verdict.
    assert box.backtests == [] and box.rejects == []
    assert "decide" not in [s.stage for s in ledger.stages()]
    assert summary.undecided == []


def test_non_escalated_author_records_no_paid_episode_line(tmp_path):
    # A local (non-escalated) author success writes NO author episode line — the ledger's episode
    # stream is unchanged from before this story when nothing escalated.
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "esc3")

    summary = _drive(episodes, box, max_episodes=2, ledger=ledger)

    assert summary.escalations == 0
    assert [e.stage for e in ledger.episodes()] == ["formulate", "decide"]  # no author episode


def test_decide_refusal_reasks_once_with_the_refusal_then_leaves_undecided(tmp_path):
    # The gated method refuses a below-floor verdict exactly as today; the driver re-asks once
    # with the refusal as corrective context, then leaves the strategy undecided.
    episodes = Episodes([formulate_ok()], [decide_ok("reject"), decide_ok("reject")])
    box = FakeToolbox(verdict_results=[{"error": _EXHAUSTION_REFUSAL}])
    ledger = SessionLedger(tmp_path, "s6")

    summary = _drive(episodes, box, max_episodes=3, ledger=ledger)

    assert len(box.rejects) == 2  # proposed, refused, re-asked, refused again
    assert episodes.decide_calls[0][1] is None  # first ask has no corrective
    assert episodes.decide_calls[1][1] == _EXHAUSTION_REFUSAL  # re-ask carries the refusal
    assert summary.rejections == 0  # no verdict was actually spent
    assert summary.undecided == ["intraday_momentum_1"]  # left undecided, honestly
    assert ledger.verdicts() == []  # nothing recorded as a spent verdict


def test_decide_refusal_then_success_records_the_verdict(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("reject"), decide_ok("reject")])
    box = FakeToolbox(
        verdict_results=[{"error": _EXHAUSTION_REFUSAL}, {"ok": True, "status": "rejected"}]
    )
    summary = _drive(episodes, box, max_episodes=3, ledger=SessionLedger(tmp_path, "s7"))

    assert len(box.rejects) == 2
    assert summary.rejections == 1
    assert summary.undecided == []


def test_decide_episode_failure_reasks_then_undecided(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_fail(), decide_fail()])
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "s8")

    summary = _drive(episodes, box, max_episodes=3, ledger=ledger)

    assert box.rejects == [] and box.evaluations == []  # never proposed a verdict
    assert len(episodes.decide_calls) == 2  # initial + one re-ask
    assert episodes.decide_calls[1][1] is not None  # the re-ask carries a corrective note
    assert summary.undecided == ["intraday_momentum_1"]
    assert [e.outcome for e in ledger.episodes() if e.stage == "decide"] == [API_ERROR, API_ERROR]


def test_revise_cap_reasks_once_then_leaves_undecided(tmp_path):
    # A `revise` is capped: the first one earns the single corrective re-ask (naming the cap); a
    # second `revise` for the same strategy applies the DECIDE failure policy (left undecided).
    episodes = Episodes(
        [formulate_ok()],
        [decide_ok("revise", new_lever="add a short leg"), decide_ok("revise", new_lever="again")],
    )
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "s9")
    summary = _drive(episodes, box, max_episodes=3, ledger=ledger)

    assert box.rejects == [] and box.evaluations == []  # revise is never a terminal verdict
    assert len(episodes.decide_calls) == 2  # initial + exactly one capped re-ask
    assert episodes.decide_calls[0][1] is None  # first ask has no corrective
    assert episodes.decide_calls[1][1] is not None and "revise" in episodes.decide_calls[1][1]
    assert summary.undecided == ["intraday_momentum_1"]
    dchecks = [e.checks for e in ledger.episodes() if e.stage == "decide"]
    assert dchecks == [
        [{"check": "revise_cap", "result": "reask"}],
        [{"check": "revise_cap", "result": "exhausted"}],
    ]


def test_revise_cap_reask_can_recover_to_a_terminal_verdict(tmp_path):
    # The capped re-ask lands a real verdict: the strategy is disposed, not left undecided.
    episodes = Episodes([formulate_ok()], [decide_ok("revise", new_lever="x"), decide_ok("reject")])
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "s9b")
    summary = _drive(episodes, box, max_episodes=3, ledger=ledger)

    assert episodes.decide_calls[1][1] is not None  # the re-ask carried the cap corrective
    assert len(box.rejects) == 1 and summary.rejections == 1  # recovered to a terminal reject
    dchecks = [e.checks for e in ledger.episodes() if e.stage == "decide"]
    assert dchecks == [[{"check": "revise_cap", "result": "reask"}], []]


# ── 2c. driver-side sanity checks on episode outputs (story #71) ────────────────────────────
def test_cost_arithmetic_check_reasks_then_proceeds_on_a_clean_second_thesis(tmp_path):
    # A number-free cost_arithmetic cites nothing from the digest → the check fires one corrective
    # re-ask; the clean re-ask proceeds to author/optimize/decide as usual.
    episodes = Episodes(
        [formulate_ok(cost_arithmetic="the move clears the round trip nicely"), formulate_ok()],
        [decide_ok("reject")],
    )
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "sc1")
    summary = _drive(episodes, box, max_episodes=3, ledger=ledger)

    assert episodes.formulate_calls == 2  # one corrective re-ask
    assert episodes.formulate_correctives[0] is None
    assert "cost_arithmetic" in episodes.formulate_correctives[1]
    assert box.rejects and summary.iterations == 1  # the clean re-ask reached a verdict
    fchecks = [e.checks for e in ledger.episodes() if e.stage == "formulate"]
    assert fchecks == [[{"check": "cost_arithmetic", "result": "reask"}], []]


def test_cost_arithmetic_check_failing_twice_ends_the_session(tmp_path):
    # The re-ask also cites no digest number → the FORMULATE failure policy applies (end session),
    # never an author call.
    episodes = Episodes(
        [
            formulate_ok(cost_arithmetic="edge beats cost, trust me"),
            formulate_ok(cost_arithmetic="still no numbers"),
        ],
        [],
    )
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "sc2")
    summary = _drive(episodes, box, max_episodes=5, ledger=ledger)

    assert summary.stopped_reason == "formulate_failed"
    assert box.writes == []  # nothing authored — the check caught it before the authoring call
    assert episodes.formulate_calls == 2
    fchecks = [e.checks for e in ledger.episodes() if e.stage == "formulate"]
    assert fchecks == [
        [{"check": "cost_arithmetic", "result": "reask"}],
        [{"check": "cost_arithmetic", "result": "exhausted"}],
    ]


def test_class_tag_exhausted_check_reasks_then_proceeds_on_a_fresh_class(tmp_path):
    # The proposed class is already a declared dead end → one corrective re-ask; a genuinely
    # different class on the re-ask proceeds.
    episodes = Episodes(
        [formulate_ok(class_tag="dead class"), formulate_ok(class_tag="fresh class")],
        [decide_ok("reject")],
    )
    box = FakeToolbox(exhausted=["dead class"])
    ledger = SessionLedger(tmp_path, "sc3")
    summary = _drive(episodes, box, max_episodes=3, ledger=ledger)

    assert episodes.formulate_calls == 2
    assert "dead class" in episodes.formulate_correctives[1]
    assert box.rejects and box.rejects[0]["name"] == "fresh_class_1"  # authored the fresh class
    assert summary.iterations == 1
    fchecks = [e.checks for e in ledger.episodes() if e.stage == "formulate"]
    assert fchecks == [[{"check": "class_tag_exhausted", "result": "reask"}], []]


def test_class_tag_exhausted_check_failing_twice_ends_the_session(tmp_path):
    # The re-ask re-proposes the same exhausted class → the FORMULATE failure policy applies.
    episodes = Episodes(
        [formulate_ok(class_tag="dead class"), formulate_ok(class_tag="dead class")],
        [],
    )
    box = FakeToolbox(exhausted=["dead class"])
    ledger = SessionLedger(tmp_path, "sc4")
    summary = _drive(episodes, box, max_episodes=5, ledger=ledger)

    assert summary.stopped_reason == "formulate_failed"
    assert box.writes == []
    fchecks = [e.checks for e in ledger.episodes() if e.stage == "formulate"]
    assert fchecks == [
        [{"check": "class_tag_exhausted", "result": "reask"}],
        [{"check": "class_tag_exhausted", "result": "exhausted"}],
    ]


def test_approve_routes_through_evaluate_vs_champion_with_best_params_and_holdout(tmp_path):
    # The model nominates DDD, but the MATCH reservation (D) is the structural holdout the
    # driver submits — a code reservation the model proposal never overwrites.
    episodes = Episodes([formulate_ok()], [decide_ok("approve", holdout_symbols=("DDD",))])
    box = FakeToolbox(
        screen_result={"suggested_fit": ["A", "B"], "reserved_holdout": ["D"]},
        verdict_results=[{"ok": True, "promoted": True}],
        log={"top_trials": [{"params": {"lookback": 20}}]},
    )
    ledger = SessionLedger(tmp_path, "s10")

    summary = _drive(episodes, box, max_episodes=2, ledger=ledger)

    assert len(box.evaluations) == 1
    ev = box.evaluations[0]
    assert ev["params"] == {"lookback": 20}  # the best-observed params from the journal
    assert ev["symbols"] == ["A", "B"]  # the MATCH fit set, not the fallback panel
    assert ev["holdout_symbols"] == ["D"]  # the MATCH reservation, not the model's DDD
    assert summary.promotions == 1
    verdicts = ledger.verdicts()
    assert verdicts[0].verdict == "approve" and verdicts[0].promoted is True


# ── 2b. deterministic MATCH: screening, holdout reservation, fallback (story #69) ───────────
@pytest.mark.parametrize(
    "character, expected",
    [
        ("liquid trending names", {"trend": "high", "volatility": "any", "liquidity": "high"}),
        (
            "illiquid small-caps that mean-revert",
            {"trend": "low", "volatility": "any", "liquidity": "low"},
        ),
        (
            "calm mega-caps drifting sideways",
            {"trend": "low", "volatility": "low", "liquidity": "high"},
        ),
        (
            "volatile breakout candidates",
            {"trend": "high", "volatility": "high", "liquidity": "any"},
        ),
        ("anything at all", {"trend": "any", "volatility": "any", "liquidity": "any"}),
        ("", {"trend": "any", "volatility": "any", "liquidity": "any"}),
    ],
)
def test_character_to_profile_is_a_deterministic_keyword_map(character, expected):
    # Low/negative markers win over high ones per dimension: "illiquid" reads low (not high on
    # the "liquid" substring); an unmentioned dimension stays "any".
    assert character_to_profile(character) == expected


def test_match_screens_with_the_profile_mapped_from_symbol_character(tmp_path):
    # "liquid trending names" maps to trend=high, liquidity=high (volatility unmentioned → any),
    # and the screen runs in driver code — no episode beyond formulate + decide is consumed.
    episodes = Episodes(
        [formulate_ok(symbol_character="liquid trending names")], [decide_ok("reject")]
    )
    box = FakeToolbox()
    _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "sm1"))

    assert box.screens == [{"trend": "high", "volatility": "any", "liquidity": "high"}]
    assert episodes.formulate_calls == 1 and len(episodes.decide_calls) == 1  # zero extra LLM


def test_reserved_holdout_never_reaches_tuning_and_reaches_decide(tmp_path):
    # Screening returns A..E: fit = A,B,C; reserved holdout = D,E. The reserved names must never
    # appear in any tuning call (write/backtest/sweep) and must reach DECIDE as the holdout.
    episodes = Episodes([formulate_ok()], [decide_ok("approve")])
    box = FakeToolbox(
        screen_result={"suggested_fit": ["A", "B", "C"], "reserved_holdout": ["D", "E"]},
        verdict_results=[{"ok": True, "promoted": True}],
    )
    _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "sm2"))

    fit = ["A", "B", "C"]
    reserved = {"D", "E"}
    assert box.writes[0]["brief"]["symbols"] == fit
    assert box.backtests[0]["symbols"] == fit  # the baseline runs the full fit panel
    assert set(box.sweeps[0]["symbols"]).issubset(set(fit))  # the cheap sweep stays within the fit
    for call in box.writes:
        assert reserved.isdisjoint(call["brief"]["symbols"])
    for call in box.backtests + box.sweeps:
        assert reserved.isdisjoint(call["symbols"])
    ev = box.evaluations[0]
    assert ev["symbols"] == fit
    assert ev["holdout_symbols"] == ["D", "E"]  # the reserved names reach DECIDE, never tuned


def test_empty_screen_falls_back_to_the_composition_root_panel(tmp_path):
    # No lake match (default empty screen) → the driver falls back to the composition-root fit
    # panel exactly as the passthrough did, and reserves nothing (holdout defers to the toolbox).
    episodes = Episodes([formulate_ok()], [decide_ok("approve", holdout_symbols=("DDD",))])
    box = FakeToolbox(verdict_results=[{"ok": True, "promoted": True}])
    _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "sm3"))

    assert box.backtests[0]["symbols"] == ["AAA", "BBB", "CCC"]  # the fallback panel
    ev = box.evaluations[0]
    assert ev["holdout_symbols"] is None  # no code reservation → the toolbox picks the fallback


def test_match_stage_ledgers_the_profile_fit_and_reservation(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox(
        screen_result={"suggested_fit": ["A", "B", "C"], "reserved_holdout": ["D", "E"]}
    )
    ledger = SessionLedger(tmp_path, "sm4")
    _drive(episodes, box, max_episodes=2, ledger=ledger)

    match = next(s for s in ledger.stages() if s.stage == "match")
    assert match.detail["profile"] == {"trend": "high", "volatility": "any", "liquidity": "high"}
    assert match.detail["fit"] == ["A", "B", "C"]
    assert match.detail["reserved_holdout"] == ["D", "E"]
    assert match.detail.get("fallback") is None


def test_match_fallback_is_ledgered_when_the_screen_is_empty(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    ledger = SessionLedger(tmp_path, "sm5")
    _drive(episodes, FakeToolbox(), max_episodes=2, ledger=ledger)

    match = next(s for s in ledger.stages() if s.stage == "match")
    assert match.detail["fit"] == ["AAA", "BBB", "CCC"]  # the composition-root fallback panel
    assert match.detail["reserved_holdout"] == []
    assert match.detail["fallback"]  # a non-empty reason string records why it fell back


# ── 3. budgets: every episode ledgered before acting; three stop conditions ─────────────────
def test_max_episodes_stops_at_the_next_stage_boundary(tmp_path):
    episodes = Episodes([formulate_ok(), formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox()
    summary = _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "s11"))
    assert summary.stopped_reason == "max_episodes"
    assert episodes.formulate_calls == 1  # the second cycle never started (budget was spent)


def test_wall_clock_budget_stops_before_any_work(tmp_path):
    from datetime import UTC, datetime

    base = datetime(2024, 1, 1, tzinfo=UTC)
    later = datetime(2024, 1, 1, 2, tzinfo=UTC)  # +2h against a 1-minute budget
    calls = {"n": 0}

    def now():
        calls["n"] += 1
        return base if calls["n"] == 1 else later

    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox()
    ledger = SessionLedger(tmp_path, "s12")
    summary = _drive(episodes, box, max_episodes=10, ledger=ledger, budget_minutes=1.0, now=now)
    assert summary.stopped_reason == "time_budget"
    assert episodes.formulate_calls == 0
    assert ledger.session_start() is not None and ledger.session_end() is not None


def test_stop_event_stops_between_stages(tmp_path):
    class _Stop:
        def is_set(self):
            return True

    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox()
    summary = _drive(
        episodes, box, max_episodes=10, ledger=SessionLedger(tmp_path, "s13"), stop_event=_Stop()
    )
    assert summary.stopped_reason == "stop_event"
    assert episodes.formulate_calls == 0


# ── 4. the driver imports no LLM code (structural) ──────────────────────────────────────────
def test_driver_imports_no_llm_client_or_provider_sdk():
    import inspect

    import noctis.research.driver as driver_mod

    src = Path(driver_mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import litellm",
        "import anthropic",
        "from noctis.research.llm",
        "import noctis.research.llm",
    ):
        assert forbidden not in src, f"driver directly imports LLM code: {forbidden!r}"
    # The client is never handed to the protocol — only injected episode callables + a counter.
    params = inspect.signature(driver_mod.run_episodic_research).parameters
    assert "client" not in params


# ── 5. end to end: real EpisodeRunner + real toolbox → a GATED verdict + complete ledger ────
@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """The e2e exercises the driver end to end, not subprocess write-gate isolation."""


class FakeEpisodeClient:
    """Replays scripted ``Turn``s through the neutral ``complete()`` seam (the episode client)."""

    def __init__(self, script):
        self._script = list(script)
        self.model = "fake/model"
        self.capabilities = Capabilities()
        self.calls: list[list[dict]] = []

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        self.calls.append(messages)
        return self._script.pop(0)


class FakeCoder:
    """A coder client: reads the requested name off the prompt and returns a valid renamed PROBE."""

    def __init__(self):
        self.model = "fake/coder"
        self.capabilities = Capabilities()

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        name = re.search(r"name:\s*(\S+)", messages[-1]["content"]).group(1)
        source = PROBE.replace('name = "probe"', f'name = "{name}"')
        block = f"```python\n{source}\n```"
        return Turn(text=block, tool_calls=[], stop_reason="end_turn", usage={})


class FakeBrokenCoder:
    """A local coder that always emits a name-mismatched file — it never passes the write gate,
    so it exhausts its validator-retry budget and triggers escalation to the paid fallback."""

    def __init__(self):
        self.model = "fake/coder-local"
        self.capabilities = Capabilities()
        self.calls = 0

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        self.calls += 1
        source = PROBE.replace('name = "probe"', 'name = "mismatch"')
        block = f"```python\n{source}\n```"
        return Turn(text=block, tool_calls=[], stop_reason="end_turn", usage={})


def _emit(name: str, payload: dict) -> Turn:
    call = ToolCall(id="c", name=name, arguments=payload)
    return Turn(
        text="",
        tool_calls=[call],
        stop_reason="tool_use",
        usage={"input_tokens": 6, "output_tokens": 4},
    )


_FORMULATE_PAYLOAD = {
    "thesis": "Buy strength above the moving average while the up-move clears cost.",
    "style": "momentum",
    "class_tag": "intraday momentum",
    "timeframe": "1m",
    "cost_arithmetic": "median 1m move ~8bp vs the 4bp round trip",
    "symbol_character": "liquid trending names",
    "scenario_intent": "one directional long tape and one no-trade selloff tape",
    "param_space_sketch": "lookback 5-40",
}
_REJECT_PAYLOAD = {
    "verdict": "reject",
    "reason": "gross edge below cost on the fit panel",
    "class_exhausted": False,
    "class_tag": "intraday momentum",
    "holdout_symbols": [],
}


def test_end_to_end_episodic_session_produces_a_gated_verdict_and_a_complete_ledger(tmp_path):
    box = _make_toolbox(tmp_path, coder_client=FakeCoder())
    ledger = SessionLedger(box.state_dir, session_id="ep-e2e")
    client = FakeEpisodeClient(
        [_emit(_FORMULATE_TOOL, _FORMULATE_PAYLOAD), _emit(_DECIDE_TOOL, _REJECT_PAYLOAD)]
    )
    runner = EpisodeRunner(client=client, retries=2)
    formulate, decide = make_episodes(
        runner=runner, toolbox=box, ledger=ledger, mandate=None, context_window=10_000_000
    )

    summary = run_episodic_research(
        toolbox=box,
        ledger=ledger,
        formulate=formulate,
        decide=decide,
        fit_symbols=["AAA", "BBB", "CCC"],
        budget_minutes=60.0,
        max_episodes=2,
        completions=lambda: runner.completions,
        sweep_trials=3,
    )

    name = "intraday_momentum_1"
    # The strategy was really authored, optimized, and its verdict CLEARED the real gate.
    assert box.journal.stats(name).sweep_completed  # a completed sweep cleared the exhaustion floor
    assert summary.rejections == 1 and summary.undecided == []
    assert summary.author_calls == 1  # the coder engine authored one file
    assert summary.candidates == [name]

    # A complete ledger the CLOSE report can render.
    assert ledger.session_start() is not None
    assert [t.strategy for t in ledger.theses()] == [name]
    assert [s.stage for s in ledger.stages()] == [
        "formulate",
        "match",
        "author",
        "optimize",
        "decide",
    ]
    assert [e.stage for e in ledger.episodes()] == ["formulate", "decide"]
    verdicts = ledger.verdicts()
    assert len(verdicts) == 1 and verdicts[0].verdict == "reject"
    assert ledger.session_end() is not None


def test_end_to_end_escalation_authors_via_the_paid_fallback_and_ledgers_it(tmp_path):
    # Real toolbox + real StrategyAuthor engines: the local coder always fails the write gate, so
    # after its full validator-retry budget the SAME brief escalates to the paid fallback, which
    # authors a valid file. The whole session then optimizes and reaches a gated verdict, and the
    # driver ledgers a paid AUTHOR episode line + surfaces the escalation on the summary (#72).
    local, fallback = FakeBrokenCoder(), FakeCoder()
    box = _make_toolbox(
        tmp_path,
        coder_client=local,
        coder_model="fake/coder-local",
        coder_fallback_client=fallback,
        coder_fallback_model="fake/coder-paid",
        max_escalations=1,
    )
    ledger = SessionLedger(box.state_dir, session_id="ep-escalate")
    client = FakeEpisodeClient(
        [_emit(_FORMULATE_TOOL, _FORMULATE_PAYLOAD), _emit(_DECIDE_TOOL, _REJECT_PAYLOAD)]
    )
    runner = EpisodeRunner(client=client, retries=2)
    formulate, decide = make_episodes(
        runner=runner, toolbox=box, ledger=ledger, mandate=None, context_window=10_000_000
    )

    summary = run_episodic_research(
        toolbox=box,
        ledger=ledger,
        formulate=formulate,
        decide=decide,
        fit_symbols=["AAA", "BBB", "CCC"],
        budget_minutes=60.0,
        max_episodes=2,
        completions=lambda: runner.completions,
        sweep_trials=3,
    )

    name = "intraday_momentum_1"
    assert summary.escalations == 1 and box.escalations == 1
    assert local.calls == 3  # the local coder burned its full validator-retry budget first
    assert summary.author_calls == 4  # 3 local + 1 paid fallback completion
    # The driver ledgered a paid AUTHOR episode line naming the fallback model.
    author_eps = [e for e in ledger.episodes() if e.stage == "author"]
    assert len(author_eps) == 1
    assert author_eps[0].escalated is True and author_eps[0].model == "fake/coder-paid"
    assert author_eps[0].outcome == "ok"
    # The escalated file was real: it authored, optimized (sweep cleared the floor), got a verdict.
    assert box.journal.stats(name).sweep_completed
    assert summary.rejections == 1 and summary.undecided == []
    assert [e.stage for e in ledger.episodes()] == ["formulate", "author", "decide"]


def test_end_to_end_below_floor_verdict_is_refused_by_the_real_gate(tmp_path):
    # With the backtest budget capped to 1 (min_trials=3), the baseline backtest spends the whole
    # budget and the sweep can run no trials, so the journal stays below the exhaustion floor and
    # the REAL gate refuses the reject verdict exactly as today; the driver re-asks once (refusal
    # folded in) then leaves the strategy undecided.
    box = _make_toolbox(tmp_path, coder_client=FakeCoder(), min_trials=3, max_backtests=1)
    ledger = SessionLedger(box.state_dir, session_id="ep-refuse")
    client = FakeEpisodeClient(
        [
            _emit(_FORMULATE_TOOL, _FORMULATE_PAYLOAD),
            _emit(_DECIDE_TOOL, _REJECT_PAYLOAD),
            _emit(_DECIDE_TOOL, _REJECT_PAYLOAD),
        ]
    )
    runner = EpisodeRunner(client=client, retries=2)
    formulate, decide = make_episodes(
        runner=runner, toolbox=box, ledger=ledger, mandate=None, context_window=10_000_000
    )

    summary = run_episodic_research(
        toolbox=box,
        ledger=ledger,
        formulate=formulate,
        decide=decide,
        fit_symbols=["AAA", "BBB", "CCC"],
        budget_minutes=60.0,
        max_episodes=3,
        completions=lambda: runner.completions,
    )

    name = "intraday_momentum_1"
    assert not box.journal.stats(name).sweep_completed
    assert summary.rejections == 0  # the gate refused the verdict
    assert summary.undecided == [name]  # left undecided, honestly
    assert ledger.verdicts() == []
    # The re-asked decide episode carried the real gate refusal as corrective context.
    assert "exhaustion gate" in client.calls[2][0]["content"]
    assert [e.stage for e in ledger.episodes()] == ["formulate", "decide", "decide"]
