"""The composition root — one module resolves a session and builds its collaborators.

``resolve_session`` is the single home of the precedence chain that used to span four
files (``load_settings`` → safety gate → ``resolve_mandate`` → ``apply_overrides`` →
explicit CLI flags), and the builders here are the one copy of assembly the CLI and the
runtime used to duplicate (lake vendor selection, the MEMORY.md store, PromotionRules
from settings, the agent research session bundle).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import noctis.research as research_mod
from noctis.bootstrap import (
    MissingVendorKey,
    UsageError,
    build_lake,
    build_research_session,
    resolve_session,
)
from noctis.champions.promotion import PromotionRules
from noctis.config import SafetyGateError, load_settings
from noctis.engine.research import ResearchSummary
from noctis.research import Capabilities, MandateError


def _fake_coder():
    """A stand-in coder LLM client: only needs the ``capabilities`` the author engine reads."""
    return SimpleNamespace(capabilities=Capabilities())


def _config(tmp_path, lines: list[str], name: str = "config.yaml") -> str:
    cfg = tmp_path / name
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(cfg)


def _mandate_dir(tmp_path, profile: str, body: str) -> Path:
    path = tmp_path / "mandate" / "profiles" / f"{profile}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return tmp_path / "mandate"


# ── resolve_session: the one precedence chain ─────────────────────────────────────────────
def test_metric_precedence_config_then_overlay_then_flag(tmp_path):
    """config.yaml < mandate overlay < --metric flag — the §5 ordering, in one place."""
    mandate_dir = _mandate_dir(
        tmp_path, "spicy", "---\nconfig:\n  promotion:\n    metric: sortino\n---\nGo fast.\n"
    )
    cfg = _config(
        tmp_path,
        [
            "promotion:",
            "  metric: sharpe",
            f"mandate_dir: {mandate_dir}",
            "research:",
            "  mandate: spicy",
        ],
    )

    # config.yaml alone (no mandate resolved): the file's metric stands.
    bare = resolve_session(_config(tmp_path, ["promotion:", "  metric: sharpe"], "bare.yaml"))
    assert bare.settings.promotion.metric == "sharpe"
    assert bare.mandate is None and bare.overrides == []

    # The mandate overlay beats the file...
    overlaid = resolve_session(cfg)
    assert overlaid.settings.promotion.metric == "sortino"
    assert overlaid.overrides == ["promotion.metric=sortino"]
    assert overlaid.mandate is not None and overlaid.mandate.source == "profile:spicy"

    # ...and an explicit --metric flag beats the overlay.
    flagged = resolve_session(cfg, metric="total_return")
    assert flagged.settings.promotion.metric == "total_return"
    assert flagged.overrides == ["promotion.metric=sortino"]  # the echo still records the overlay


def test_directive_and_mandate_are_mutually_exclusive(tmp_path):
    with pytest.raises(UsageError, match="either --directive or --mandate"):
        resolve_session(_config(tmp_path, ["mode: paper"]), directive="go", mandate="spicy")


def test_unknown_metric_refused_before_settings_load(tmp_path):
    with pytest.raises(UsageError, match="nonsense"):
        resolve_session(str(tmp_path / "does-not-exist.yaml"), metric="nonsense")


def test_unresolvable_mandate_selector_is_fatal(tmp_path):
    with pytest.raises(MandateError):
        resolve_session(_config(tmp_path, ["mode: paper"]), mandate="no-such-profile")


def test_gate_resolves_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE", raising=False)
    cfg = _config(tmp_path, ["mode: live"])
    # Entrypoints that never place orders skip the gate: mode stays unresolved.
    assert resolve_session(cfg).mode is None
    # The trading loop arms it — mode: live without ALLOW_LIVE refuses to start.
    with pytest.raises(SafetyGateError):
        resolve_session(cfg, require_gate=True)
    assert resolve_session(_config(tmp_path, ["mode: paper"]), require_gate=True).mode == "paper"


def test_time_limit_flag_overrides_config(tmp_path):
    cfg = _config(tmp_path, ["time_limit_hours: 24"])
    assert resolve_session(cfg).settings.time_limit_hours == 24
    assert resolve_session(cfg, time_limit_hours=0.5).settings.time_limit_hours == 0.5


# ── PromotionRules.from_settings: the one config→rules mapping ────────────────────────────
def test_promotion_rules_from_settings_maps_every_field(tmp_path):
    settings = load_settings(
        config_path=_config(
            tmp_path,
            [
                "champion_count: 5",
                "promotion:",
                "  max_gap: 0.7",
                "  min_test_metric: 0.1",
                "  min_holdout_metric: 0.2",
                "  min_symbol_holdout_metric: 0.3",
                "  min_symbol_consistency: 0.4",
                "  min_test_activity: 0.5",
                "  max_reverse_gap: 0.6",
                "  max_test_metric: 60.0",
            ],
        )
    )
    assert PromotionRules.from_settings(settings) == PromotionRules(
        champion_count=5,
        max_gap=0.7,
        min_test_metric=0.1,
        min_holdout_metric=0.2,
        min_symbol_holdout_metric=0.3,
        min_symbol_consistency=0.4,
        min_test_activity=0.5,
        max_reverse_gap=0.6,
        max_test_metric=60.0,
    )


# ── PipelineConfig.auto_from_settings: the one config→pipeline mapping ────────────────────
def test_pipeline_config_auto_from_settings_threads_promotion_knobs(tmp_path):
    """Pure delegation to ``auto`` with the promotion knobs pulled from settings — every
    entrypoint (CLI backtest, research tools, runtime) shares this one mapping."""
    from noctis.backtest import PipelineConfig

    settings = load_settings(
        config_path=_config(
            tmp_path,
            [
                "promotion:",
                "  metric: sortino",
                "  annualization_cap: 123",
                "  max_period_ratio: 2.5",
            ],
        )
    )
    built = PipelineConfig.auto_from_settings(
        settings, 400, periods_per_year=98_280, prefilter_min_score=None
    )
    assert built == PipelineConfig.auto(
        400,
        metric="sortino",
        periods_per_year=98_280,
        prefilter_min_score=None,
        annualization_cap=123,
        max_period_ratio=2.5,
    )
    # The settings knobs actually landed (guards against the delegation dropping one).
    assert built.metric_name == "sortino"
    assert built.prefilter.annualization_cap == 123
    assert built.validation.annualization_cap == 123
    assert built.prefilter.max_period_ratio == 2.5
    assert built.validation.max_period_ratio == 2.5


def test_pipeline_config_auto_from_settings_threads_fill_costs(tmp_path):
    """The one config→pipeline mapping pulls backtest.fee_bps/slippage_bps from settings into
    BOTH stages, so prefilter and validation charge exactly the operator-configured cost."""
    from noctis.backtest import PipelineConfig

    settings = load_settings(
        config_path=_config(
            tmp_path,
            ["backtest:", "  fee_bps: 2.5", "  slippage_bps: 3.0"],
        )
    )
    built = PipelineConfig.auto_from_settings(settings, 400)
    assert built.prefilter.fee_bps == 2.5 and built.prefilter.slippage_bps == 3.0
    assert built.validation.fee_bps == 2.5 and built.validation.slippage_bps == 3.0


def test_pipeline_config_auto_from_settings_defaults_to_shipped_costs(tmp_path):
    """Unset config threads the shipped baseline — default-equivalence with today."""
    from noctis.backtest import PipelineConfig

    settings = load_settings(config_path=_config(tmp_path, ["mode: paper"]))
    built = PipelineConfig.auto_from_settings(settings, 400)
    assert built.prefilter.fee_bps == 1.0 and built.prefilter.slippage_bps == 1.0
    assert built.validation.fee_bps == 1.0 and built.validation.slippage_bps == 1.0


# ── build_lake: vendor selection from credentials ─────────────────────────────────────────
def test_build_lake_without_key_is_read_only(tmp_path):
    settings = load_settings(config_path=_config(tmp_path, [f"data:\n  lake_dir: {tmp_path}/lake"]))
    lake = build_lake(settings)
    with pytest.raises(RuntimeError, match="read-only"):
        lake.vendor.fetch_bars()


def test_build_lake_requiring_vendor_without_key_raises(tmp_path):
    settings = load_settings(config_path=_config(tmp_path, [f"data:\n  lake_dir: {tmp_path}/lake"]))
    with pytest.raises(MissingVendorKey, match="DATABENTO_API_KEY"):
        build_lake(settings, require_vendor=True)


def test_build_lake_with_key_uses_the_vendor_client(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "noctis.data.databento_provider.DataBentoVendorClient", lambda key: sentinel
    )
    monkeypatch.setenv("DATABENTO_API_KEY", "db-test-key")
    settings = load_settings(config_path=_config(tmp_path, [f"data:\n  lake_dir: {tmp_path}/lake"]))
    assert build_lake(settings).vendor is sentinel


# ── build_research_session: the one bundle both entrypoints run ───────────────────────────
def _session_settings(tmp_path, *, coder_model: str | None = None):
    lines = [
        "research_time_budget_minutes: 42",
        f"state_dir: {tmp_path}/state/",
        f"strategies_dir: {tmp_path}/strategies/",
    ]
    if coder_model is not None:
        lines += ["research:", "  agent:", f"    coder_model: {coder_model}"]
    return load_settings(config_path=_config(tmp_path, lines))


def test_build_research_session_none_without_client(tmp_path, monkeypatch):
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: None)
    session = build_research_session(
        settings=_session_settings(tmp_path),
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert session is None


def test_research_session_runs_the_same_loop_kwargs_as_the_cli_did(tmp_path, monkeypatch):
    """The bundle threads client, budgets, mandate, and sinks into ``run_agent_research`` —
    the kwargs the CLI and the runtime used to wire independently."""
    client = object()
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: client)
    seen: dict = {}

    def fake_loop(**kwargs):
        seen.update(kwargs)
        return ResearchSummary()

    monkeypatch.setattr(research_mod, "run_agent_research", fake_loop)

    settings = _session_settings(tmp_path)
    sink = [].append
    stop = object()
    session = build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
        on_event=sink,
    )
    assert session is not None
    assert session.client is client
    # No explicit cap → the cost-profile budget governs, exactly as both call sites did.
    session.run(stop_event=stop)
    assert seen["client"] is client
    assert seen["budget_minutes"] == 42
    assert seen["max_iterations"] == session.budgets.max_iterations
    assert seen["stop_event"] is stop
    assert seen["on_event"] is sink
    assert seen["toolbox"] is session.toolbox
    # An explicit cap wins over the budget.
    session.run(max_iterations=3)
    assert seen["max_iterations"] == 3


def test_research_session_derives_rules_and_mandate_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    settings = _session_settings(tmp_path)
    mandate = research_mod.Mandate(
        text="Go.", source="profile:spicy", summary="Go.", references=[], config_overrides={}
    )
    session = build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
        mandate=mandate,
    )
    assert session is not None
    assert session.toolbox.rules == PromotionRules.from_settings(settings)
    assert session.toolbox.mandate_source == "profile:spicy"
    assert session.mandate is mandate


# ── the coder-model knob (#4): a dedicated authoring client, threaded here or None ─────────
def test_coder_client_not_built_when_knob_unset(tmp_path, monkeypatch, caplog):
    """Knob unset ⇒ no coder client built, no attempt, no new warning (today's behavior)."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    calls: list = []
    monkeypatch.setattr(research_mod, "client_for", lambda *a, **k: calls.append((a, k)))
    with caplog.at_level(logging.WARNING):
        session = build_research_session(
            settings=_session_settings(tmp_path),
            lake=object(),
            registry=object(),
            families=object(),
            memory=object(),
        )
    assert session is not None
    assert session.toolbox.coder_client is None
    assert calls == []  # the coder builder is never even consulted
    assert not any("coder" in r.getMessage().lower() for r in caplog.records)


def test_coder_client_built_when_configured(tmp_path, monkeypatch):
    """Knob set + provider available ⇒ a stateless coder client reaches the toolbox, built with
    thinking ON — authoring is the reasoning-heavy sub-task (#17). It is a *deliberate*, budgeted
    thinking decision, so ``deliberate=True`` overrides the Sonnet cheap-path pin for the coder."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    coder = _fake_coder()
    seen: dict = {}

    def fake_client_for(settings, model, **kwargs):
        seen["model"] = model
        seen["kwargs"] = kwargs
        return coder

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _session_settings(tmp_path, coder_model="anthropic/claude-sonnet-5")
    session = build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert session is not None
    assert session.toolbox.coder_client is coder
    assert seen["model"] == "anthropic/claude-sonnet-5"
    # Thinking flips ON at the composition root (default coder_thinking), deliberately — so even a
    # Sonnet coder reasons through the scenario/warmup arithmetic instead of repeating an error.
    assert seen["kwargs"].get("thinking") == "on"
    assert seen["kwargs"].get("deliberate") is True


def test_coder_thinking_setting_off_pins_the_coder_client_off(tmp_path, monkeypatch):
    """``research.agent.coder_thinking: off`` is the operator's opt-out: the coder client is then
    built thinking off (still a deliberate decision — the driver dial is a separate knob)."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    seen: dict = {}

    def fake_client_for(settings, model, **kwargs):
        seen["kwargs"] = kwargs
        return _fake_coder()

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _session_settings(tmp_path, coder_model="anthropic/claude-sonnet-5")
    settings.research.agent.coder_thinking = "off"
    build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert seen["kwargs"].get("thinking") == "off"
    assert seen["kwargs"].get("deliberate") is True


def test_coder_thinking_defaults_on(tmp_path):
    """The coder-thinking knob defaults ON (authoring is reasoning-heavy); the driver watch dial
    (``research.agent.thinking``) stays independently OFF by default (untouched by this story)."""
    settings = _session_settings(tmp_path, coder_model="anthropic/claude-sonnet-5")
    assert settings.research.agent.coder_thinking == "on"
    assert settings.research.agent.thinking == "off"


def test_coder_client_missing_key_degrades_loudly(tmp_path, monkeypatch, caplog):
    """Knob set but provider key/extra missing ⇒ a loud warning, session still assembles in
    driver-authored mode (coder client None) — never a mid-session failure."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    monkeypatch.setattr(research_mod, "client_for", lambda *a, **k: None)  # missing key/extra
    settings = _session_settings(tmp_path, coder_model="anthropic/claude-sonnet-5")
    with caplog.at_level(logging.WARNING):
        session = build_research_session(
            settings=settings,
            lake=object(),
            registry=object(),
            families=object(),
            memory=object(),
        )
    assert session is not None
    assert session.toolbox.coder_client is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("coder" in msg.lower() for msg in warnings)
    assert any("claude-sonnet-5" in msg for msg in warnings)
