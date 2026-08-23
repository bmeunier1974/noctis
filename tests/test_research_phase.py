"""The RESEARCH entry behind its own seam (#271): ``ResearchPhase.run(panel)``.

RESEARCH is a phase object of the same shape as TRADING and CLOSE: the runtime assembles it
once and drives it at each entry with a frozen :class:`~noctis.engine.ResearchPanel` — the fit
bars, the symbol holdout and the split geometry both are scored under — rebuilt from a fresh
catalog read, exactly as TRADING rebuilds its bars, so no session is ever driven on a stale
panel.

These tests pin the phase's behavior: both paths (the agent session, and the legacy
proposer/Optuna loop that is also its no-key fallback) hand back one ``ResearchSummary``, the
completed session is counted toward the periodic memory distillation, the panel's fit set and
symbol holdout are what the legacy loop actually scores, and the mandate/metric line the agent
path logs stays byte-identical to the one the runtime used to emit.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from noctis.backtest import PipelineConfig
from noctis.bootstrap import build_families
from noctis.champions import build_registry
from noctis.champions.promotion import PromotionRules
from noctis.config import load_settings
from noctis.engine import CloseResult, ResearchPanel, ResearchPhase, SimulatedSleeper, build_runtime
from noctis.engine.research import ResearchSummary
from noctis.memory import InMemoryMemory, MemoryStore
from noctis.strategies.proposer import CandidateProposer

from ._session_helpers import _bars_local, _FakeLake, _FakeRegistry

ET = ZoneInfo("America/New_York")


def _wave(n: int = 400) -> list[float]:
    """A deterministic, tradeable series: a drifting sine, so a crossover actually turns over."""
    return [100.0 + 10.0 * math.sin(i / 7.0) + i * 0.05 for i in range(n)]


def _settings(tmp_path, body: str = ""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL, MSFT]\n"
        f"state_dir: {tmp_path}/state/\n"
        # an empty strategies_dir → in-package seed families only
        f"strategies_dir: {tmp_path}/strategies/\n" + body
    )
    return load_settings(config_path=cfg)


def _phase(settings, **kwargs) -> ResearchPhase:
    families = build_families(settings)
    kwargs.setdefault("registry", build_registry(settings))
    kwargs.setdefault("proposer", CandidateProposer(families, seed=0))
    return ResearchPhase(
        settings=settings,
        market_lake=_FakeLake({}),
        families=families,
        memory=InMemoryMemory(),
        rules=PromotionRules.from_settings(settings),
        **kwargs,
    )


def _panel(*, prefilter_min_score: float | None = None, n: int = 400) -> ResearchPanel:
    day = date(2026, 3, 9)
    return ResearchPanel(
        fit={"AAPL": _bars_local(day, _wave(n))},
        symbol_holdout={"MSFT": _bars_local(day, _wave(n))},
        config=PipelineConfig.auto(n, prefilter_min_score=prefilter_min_score),
    )


def _sessions_counted(settings) -> int:
    """How many completed research sessions are banked toward the next distillation."""
    path = Path(settings.state_dir) / "memory_distill.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text())["sessions_since_distill"])


class _RecordingRegistry:
    """A champion board that promotes nothing and keeps every scorecard it was shown."""

    def __init__(self):
        self.scorecards: list = []

    def consider(self, scorecard, rules):
        self.scorecards.append(scorecard)
        return type("_Decision", (), {"promote": False, "rationale": "recorded"})()

    def list(self):
        return []


class _ExplodingProposer:
    """Proposing at all is the failure: the agent path must not fall through to the legacy loop."""

    def propose(self):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("the legacy loop ran under a live agent session")


# ── the legacy path: the panel is what gets scored ────────────────────────────────────────────
def test_the_legacy_loop_scores_the_panel_and_returns_its_summary(tmp_path):
    """The fit bars are the panel a candidate is tuned and elected on, the symbol holdout is
    scored but never selected on, and both ride in on the argument — the phase holds neither."""
    settings = _settings(tmp_path, "research:\n  mode: legacy\n")
    registry = _RecordingRegistry()
    phase = _phase(settings, registry=registry, research_max_iters=2)

    summary = phase.run(_panel())

    assert isinstance(summary, ResearchSummary)
    assert summary.iterations == 2
    assert summary.stopped_reason == "max_iterations"
    assert len(registry.scorecards) == 2
    for card in registry.scorecards:
        assert card.stage == "validated"  # nothing killed early: the whole funnel ran
        assert set(card.symbols) == {"AAPL"}  # the fit set, and only the fit set
        assert card.symbol_holdout_metric is not None  # MSFT was scored, one causal pass


def test_run_counts_the_completed_session_toward_distillation(tmp_path):
    """Every completed session — whichever path ran it — banks one toward the memory
    distillation that fires at CLOSE."""
    settings = _settings(tmp_path, "research:\n  mode: legacy\n")
    phase = _phase(settings, research_max_iters=1)
    assert _sessions_counted(settings) == 0

    phase.run(_panel(prefilter_min_score=0.0))
    assert _sessions_counted(settings) == 1

    phase.run(_panel(prefilter_min_score=0.0))
    assert _sessions_counted(settings) == 2


# ── the agent path ────────────────────────────────────────────────────────────────────────────
class _Session:
    """The session bundle the composition root hands back: its own mandate, and a run()."""

    def __init__(self, mandate, summary):
        self.mandate = mandate
        self.summary = summary
        self.calls: list[dict] = []

    def run(self, *, max_iterations=None, stop_event=None):
        self.calls.append({"max_iterations": max_iterations, "stop_event": stop_event})
        return self.summary


@pytest.mark.parametrize(
    "source, expected",
    [("profile:spicy", "profile:spicy"), (None, "(none)")],
)
def test_run_agent_session_logs_the_mandate_the_session_carries(
    tmp_path, monkeypatch, caplog, source, expected
):
    """The provenance line moves with the code, byte for byte: it names the session's OWN
    resolved mandate and the metric the run elects on."""
    from noctis.research.mandate import Mandate

    mandate = (
        None
        if source is None
        else Mandate(
            text="Hunt intraday reversals.",
            source=source,
            summary="intraday reversals",
            references=[],
            config_overrides={},
        )
    )
    session = _Session(mandate, ResearchSummary())
    monkeypatch.setattr("noctis.bootstrap.build_research_session", lambda **kwargs: session)
    phase = _phase(_settings(tmp_path))

    with caplog.at_level(logging.INFO, logger="noctis.runtime"):
        assert phase.run_agent_session() is session.summary

    lines = [r.getMessage() for r in caplog.records if "agent research session:" in r.getMessage()]
    assert lines == [f"agent research session: mandate={expected}, metric=sharpe"]


def test_run_in_agent_mode_returns_the_sessions_summary(tmp_path, monkeypatch):
    """A live agent session IS the night's work: its summary comes back untouched and the
    legacy loop never runs."""
    session = _Session(None, ResearchSummary(iterations=3, promotions=1, undecided=["draft_a"]))
    monkeypatch.setattr("noctis.bootstrap.build_research_session", lambda **kwargs: session)
    settings = _settings(tmp_path)
    phase = _phase(settings, proposer=_ExplodingProposer(), research_max_iters=7)

    summary = phase.run(_panel())

    assert summary is session.summary
    assert session.calls == [{"max_iterations": 7, "stop_event": phase.stop_event}]
    assert _sessions_counted(settings) == 1


def test_run_falls_back_to_the_legacy_loop_without_a_research_client(tmp_path, caplog):
    """Agent mode with no key degrades to the legacy loop over the same library — the
    fallback the two-paths-one-contract seam exists for."""
    settings = _settings(tmp_path)  # research.mode defaults to agent; conftest clears the keys
    phase = _phase(settings, research_max_iters=1)

    with caplog.at_level(logging.INFO, logger="noctis.runtime"):
        assert phase.run_agent_session() is None
        summary = phase.run(_panel(prefilter_min_score=0.0))

    assert summary.iterations == 1  # the legacy loop ran
    assert any(
        r.getMessage() == "research.mode=agent but no research client; using legacy loop"
        for r in caplog.records
    )


# ── the runtime's RESEARCH entry ──────────────────────────────────────────────────────────────
class _RecordingPhase:
    """Stands in for the assembled phase: records every panel the runtime drove it with."""

    def __init__(self, summary: ResearchSummary):
        self.summary = summary
        self.panels: list[ResearchPanel] = []

    def run(self, panel: ResearchPanel) -> ResearchSummary:
        self.panels.append(panel)
        return self.summary


def _runtime(tmp_path, lake):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL, MSFT]\n"
        "research:\n  fit_set_size: 1\n  symbol_holdout_size: 1\n"
        f"state_dir: {tmp_path}/state/\n"
        f"strategies_dir: {tmp_path}/strategies/\n"
    )
    return build_runtime(
        load_settings(config_path=cfg),
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=_FakeRegistry(),
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=1,
        sleeper_factory=lambda start: SimulatedSleeper(start),
    )


def test_the_research_entry_drives_the_phase_with_a_panel_of_fresh_bars(tmp_path):
    """The panel is rebuilt at each RESEARCH entry, so a session that follows a CLOSE-phase
    T+1 sync researches the data that sync brought in — never the bars startup happened to
    see. The fit/holdout split stays deterministic from universe order."""
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _wave(120))})
    runtime = _runtime(tmp_path, lake)
    lake.bars["MSFT"] = _bars_local(date(2026, 3, 9), _wave(120))  # the lake grew since startup

    phase = _RecordingPhase(ResearchSummary(iterations=4, promotions=2, undecided=["draft_a"]))
    runtime.research = phase
    runtime._run_trading = lambda t, sleeper: None
    seen: dict = {}

    def _close(t, cycle, *, tracked=None):
        seen["undecided"] = list(cycle.research_undecided)
        return CloseResult()

    runtime.close.run = _close

    result = runtime.run(start=datetime(2027, 1, 4, 6, 0, tzinfo=ET), max_cycles=1)

    panel = phase.panels[0]
    assert list(panel.fit) == ["AAPL"]
    assert list(panel.symbol_holdout) == ["MSFT"]  # the freshly-appeared symbol reached the panel
    assert panel.config.metric_name == "sharpe"
    # The summary folds into the run's counters and into the cycle the CLOSE renders.
    assert (result.research_iterations, result.research_promotions) == (4, 2)
    assert seen["undecided"] == ["draft_a"]
