"""P3 — the ``run`` command's observability wiring.

The day/night loop now narrates each research session the same way ``noctis research`` does:
the runtime carries an ``on_event`` console, tees the research feed into it, and announces each
phase transition inline through the machine's ``on_enter`` seam. A bare run (``on_event=None``)
stays byte-identical to today — no feed, no banners. The two commands also resolve one shared
verbosity ladder.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest

import noctis.research as research_mod
from noctis.cli import _logging_level
from noctis.config import load_settings
from noctis.engine import Phase, build_runtime
from noctis.engine.research import ResearchSummary
from noctis.memory import MemoryStore
from noctis.observability import Console, Event

from ._session_helpers import _bars_local, _FakeLake, _FakeRegistry, _uptrend


def _runtime(tmp_path, *, on_event=None):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        "universe: [AAPL]\n"
        f"state_dir: {tmp_path}/state/\n"
        f"strategies_dir: {tmp_path}/strategies/\n"
    )
    settings = load_settings(config_path=cfg)
    lake = _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    return build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=_FakeRegistry(),
        reports_dir=str(tmp_path / "reports"),
        on_event=on_event,
    )


# ── run wiring: the research feed reaches the injected console ────────────────────────────────
def test_run_agent_research_tees_the_feed_into_the_console(tmp_path, monkeypatch):
    """The wiring P3 adds: the runtime hands its own console to ``run_agent_research`` as
    ``on_event``, so a ``-vv`` day/night run surfaces the model's reasoning inline — exactly
    what ``noctis research -vv`` shows, now visible from the loop."""
    out: list[str] = []
    console = Console(2, sink=out.append, color=False)  # -vv opens think/say
    runtime = _runtime(tmp_path, on_event=console)

    seen: dict = {}
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())

    def fake_loop(*, on_event, **kwargs):
        # Stand in for the real loop: record the sink the runtime passed, then emit through it.
        seen["on_event"] = on_event
        on_event(Event("think", "weighing the mean-reversion thesis", level=2))
        return ResearchSummary()

    monkeypatch.setattr(research_mod, "run_agent_research", fake_loop)

    summary = runtime._run_agent_research()

    assert summary is not None  # a real summary → no legacy fallback
    assert seen["on_event"] is console  # the runtime threaded its own console through
    assert any("weighing the mean-reversion thesis" in line for line in out)


def test_run_agent_research_without_a_console_passes_none(tmp_path, monkeypatch):
    """A bare run (no ``-v``) carries ``on_event=None``, so the research loop falls back to its
    own logger sink — the day/night loop stays silent by default."""
    runtime = _runtime(tmp_path, on_event=None)

    seen: dict = {}
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())

    def fake_loop(*, on_event, **kwargs):
        seen["on_event"] = on_event
        return ResearchSummary()

    monkeypatch.setattr(research_mod, "run_agent_research", fake_loop)

    runtime._run_agent_research()
    assert seen["on_event"] is None


# ── phase transitions: one banner per transition, silent by default ──────────────────────────
def test_phase_banner_emitted_once_per_transition(tmp_path):
    """The runtime wires the machine's ``on_enter`` seam so each phase entry emits one
    level-1 ``phase`` Event that names the phase and the cycle it opens."""
    events: list = []
    runtime = _runtime(tmp_path, on_event=events.append)

    # Fire the seam exactly as the machine does on entering each phase of one cycle.
    for phase in (Phase.RESEARCH, Phase.TRADING, Phase.CLOSE):
        runtime.machine.on_enter(phase)

    banners = [e for e in events if not isinstance(e, str) and e.kind == "phase"]
    assert [e.meta["phase"] for e in banners] == ["RESEARCH", "TRADING", "CLOSE"]
    assert all(e.level == 1 for e in banners)  # shows at -v
    assert all(e.meta["cycle"] == 0 for e in banners)  # first cycle
    assert "RESEARCH" in banners[0].text and "cycle 0" in banners[0].text


def test_phase_banner_stays_silent_on_a_bare_run(tmp_path):
    """``on_event=None`` (a bare ``python -m noctis run``) emits nothing. Without the None-guard
    the hook would call ``None`` and raise ``TypeError`` — so a clean run proves the guard."""
    runtime = _runtime(tmp_path, on_event=None)
    assert runtime._on_event is None
    for phase in (Phase.RESEARCH, Phase.TRADING, Phase.CLOSE):
        runtime.machine.on_enter(phase)  # no-op, no output, no raise


# ── one ladder: run and research map -v identically ──────────────────────────────────────────
@pytest.mark.parametrize(
    "verbose,expected",
    [
        (0, logging.WARNING),  # quiet
        (1, logging.WARNING),  # -v: the level-1 feed rides the Console, not raw INFO logs
        (2, logging.DEBUG),  # -vv: open DEBUG
        (3, logging.DEBUG),
    ],
)
def test_verbosity_ladder_is_shared_and_stable(verbose, expected):
    """Both ``run`` and ``research`` resolve their stdlib logging level through this one helper,
    so the two commands can no longer disagree (they used to: ``run`` -v→INFO, ``research``
    -v→WARNING)."""
    assert _logging_level(verbose) == expected
