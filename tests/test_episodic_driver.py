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
    DECIDE_CONTRACT,
    FORMULATE_CONTRACT,
    DecideOutput,
    FormulateOutput,
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
        self.decide_calls: list[tuple[str, str | None]] = []

    def formulate(self):
        self.completions += 1
        self.formulate_calls += 1
        return self._f.pop(0)

    def decide(self, strategy, *, corrective=None):
        self.completions += 1
        self.decide_calls.append((strategy, corrective))
        return self._d.pop(0)


# ── a fake toolbox: only the gated surface the driver drives, with honest bookkeeping ───────
class FakeToolbox:
    def __init__(self, *, write_result=None, verdict_results=None, log=None):
        self.promotions = 0
        self.rejections = 0
        self.author_calls = 0
        self.strategies_touched: list[str] = []
        self.undecided: set[str] = set()
        self.writes: list[dict] = []
        self.backtests: list[dict] = []
        self.sweeps: list[dict] = []
        self.rejects: list[dict] = []
        self.evaluations: list[dict] = []
        self._write_result = write_result or {"ok": True}
        self._verdicts = list(verdict_results or [{"ok": True, "status": "rejected"}])
        self._log = log or {"top_trials": [{"params": {"lookback": 20}}]}

    def tool_write_strategy(self, **kwargs):
        self.writes.append(kwargs)
        if "error" in self._write_result:
            return dict(self._write_result)
        name = kwargs["name"]
        self.strategies_touched.append(name)
        self.undecided.add(name)
        self.author_calls += 1
        return dict(self._write_result)

    def tool_run_backtest(self, **kwargs):
        self.backtests.append(kwargs)
        return {"ok": True}

    def tool_run_sweep(self, **kwargs):
        self.sweeps.append(kwargs)
        return {"ok": True, "sweep_completed": True}

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


def test_optimize_runs_a_baseline_backtest_and_a_sweep_before_decide(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("reject")])
    box = FakeToolbox()
    _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "s2"))
    assert len(box.backtests) == 1 and box.backtests[0]["symbols"] == ["AAA", "BBB", "CCC"]
    assert len(box.sweeps) == 1  # a sweep gives DECIDE a real journal / clears the floor


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


def test_revise_verdict_leaves_the_strategy_undecided(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("revise", new_lever="add a short leg")])
    box = FakeToolbox()
    summary = _drive(episodes, box, max_episodes=2, ledger=SessionLedger(tmp_path, "s9"))

    assert box.rejects == [] and box.evaluations == []  # revise is not a terminal verdict here
    assert summary.undecided == ["intraday_momentum_1"]


def test_approve_routes_through_evaluate_vs_champion_with_best_params_and_holdout(tmp_path):
    episodes = Episodes([formulate_ok()], [decide_ok("approve", holdout_symbols=("DDD",))])
    box = FakeToolbox(
        verdict_results=[{"ok": True, "promoted": True}],
        log={"top_trials": [{"params": {"lookback": 20}}]},
    )
    ledger = SessionLedger(tmp_path, "s10")

    summary = _drive(episodes, box, max_episodes=2, ledger=ledger)

    assert len(box.evaluations) == 1
    ev = box.evaluations[0]
    assert ev["params"] == {"lookback": 20}  # the best-observed params from the journal
    assert ev["symbols"] == ["AAA", "BBB", "CCC"]
    assert ev["holdout_symbols"] == ["DDD"]
    assert summary.promotions == 1
    verdicts = ledger.verdicts()
    assert verdicts[0].verdict == "approve" and verdicts[0].promoted is True


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
