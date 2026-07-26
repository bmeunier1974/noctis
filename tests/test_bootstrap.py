"""The composition root — one module resolves a session and builds its collaborators.

``resolve_session`` is the single home of the precedence chain that used to span four
files (``load_settings`` → safety gate → ``resolve_mandate`` → ``apply_overrides`` →
explicit CLI flags), and the builders here are the one copy of assembly the CLI and the
runtime used to duplicate (lake vendor selection, the MEMORY.md store, PromotionRules
from settings, the agent research session bundle).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import noctis.research as research_mod
from noctis.bootstrap import (
    _EPISODIC_WINDOW_MAX,
    MissingVendorKey,
    ResearchSession,
    UsageError,
    build_lake,
    build_recorder,
    build_research_session,
    effective_memory_distill_every,
    resolve_research_loop,
    resolve_session,
)
from noctis.champions.promotion import PromotionRules
from noctis.config import SafetyGateError, load_settings
from noctis.engine.research import ResearchSummary
from noctis.research import Capabilities, MandateError
from noctis.strategies.families import FamilyRegistry


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


def test_time_limit_flag_overrides_config_and_a_mandate_overlay(tmp_path):
    """``time_limit_hours`` is run-shaping, so a mandate may overlay it (#117) — and the
    ``--time-limit-hours`` flag is still applied last, so a one-off run bound still wins."""
    cfg = _config(tmp_path, ["time_limit_hours: 24"])
    assert resolve_session(cfg).settings.time_limit_hours == 24
    assert resolve_session(cfg, time_limit_hours=0.5).settings.time_limit_hours == 0.5

    mandate_dir = _mandate_dir(
        tmp_path, "brief", "---\nconfig:\n  time_limit_hours: 6\n---\nShort runs.\n"
    )
    overlaid_cfg = _config(
        tmp_path,
        [
            "time_limit_hours: 24",
            f"mandate_dir: {mandate_dir}",
            "research:",
            "  mandate: brief",
        ],
        "overlaid.yaml",
    )
    overlaid = resolve_session(overlaid_cfg)
    assert overlaid.settings.time_limit_hours == 6  # the mandate beats config.yaml
    assert overlaid.overrides == ["time_limit_hours=6.0"]

    flagged = resolve_session(overlaid_cfg, time_limit_hours=0.5)
    assert flagged.settings.time_limit_hours == 0.5  # ...and the flag beats the mandate
    assert flagged.overrides == ["time_limit_hours=6.0"]  # the echo still records the overlay


def test_effective_precedence_is_flag_over_mandate_over_env_over_dotenv_over_yaml(
    tmp_path, monkeypatch
):
    """The whole chain on one knob: CLI flag > mandate overlay > environment > ``.env`` >
    ``config.yaml`` > defaults.

    The mandate slot is the interesting one — it **inverts** the usual env-over-YAML rule.
    ``config.yaml`` never beats an environment variable, but the mandate is applied after
    settings are fully resolved, so an operator's steering file wins over the shell.
    """
    from noctis.config.settings import Settings

    mandate_dir = _mandate_dir(
        tmp_path, "bounded", "---\nconfig:\n  time_limit_hours: 3\n---\nBounded runs.\n"
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text("TIME_LIMIT_HOURS=12\n", encoding="utf-8")

    # 1. defaults — no file, no env, no mandate.
    assert resolve_session(str(tmp_path / "absent.yaml")).settings.time_limit_hours is None

    # 2. config.yaml beats the default.
    yaml_only = _config(tmp_path, ["time_limit_hours: 24"], "yaml-only.yaml")
    assert resolve_session(yaml_only).settings.time_limit_hours == 24

    # 3. .env beats config.yaml.
    monkeypatch.setitem(Settings.model_config, "env_file", str(dotenv))
    assert resolve_session(yaml_only).settings.time_limit_hours == 12

    # 4. the environment beats .env.
    monkeypatch.setenv("TIME_LIMIT_HOURS", "8")
    assert resolve_session(yaml_only).settings.time_limit_hours == 8

    # 5. the mandate overlay beats the environment — the inversion.
    steered = _config(
        tmp_path,
        ["time_limit_hours: 24", f"mandate_dir: {mandate_dir}", "research:", "  mandate: bounded"],
        "steered.yaml",
    )
    assert resolve_session(steered).settings.time_limit_hours == 3

    # 6. the CLI flag beats everything.
    assert resolve_session(steered, time_limit_hours=0.25).settings.time_limit_hours == 0.25


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


# ── build_recorder: the one --debug recorder assembly (story #45) ─────────────────────────
def _qa_settings(tmp_path, *, keep_last_runs: int | None = None):
    lines = [f"qa_dir: {tmp_path}/qa"]
    if keep_last_runs is not None:
        lines += ["qa:", f"  keep_last_runs: {keep_last_runs}"]
    return load_settings(config_path=_config(tmp_path, lines))


def test_build_recorder_mints_run_tree_and_stamps_the_manifest(tmp_path):
    """A recorder built through the composition root files its report tree and a manifest carrying
    the injected argv/mode plus a config digest and the noctis/python versions."""
    import json
    import platform

    settings = _qa_settings(tmp_path)
    rec = build_recorder(settings, argv=["run", "--debug"], mode="paper")

    assert rec.run_dir.is_dir()
    assert rec.run_dir == Path(f"{tmp_path}/qa") / rec.run_id
    manifest = json.loads((rec.run_dir / "run.json").read_text())
    assert manifest["run_id"] == rec.run_id
    assert manifest["argv"] == ["run", "--debug"]
    assert manifest["mode"] == "paper"
    assert isinstance(manifest["config_digest"], str) and manifest["config_digest"]
    assert manifest["versions"]["python"] == platform.python_version()
    assert manifest["versions"]["noctis"]  # populated (installed version or __version__ fallback)
    assert manifest["stopped"] is None  # not closed yet


def test_build_recorder_prunes_the_qa_area_to_keep_last_runs(tmp_path):
    """Prune-on-start: building a recorder first evicts all but the newest ``keep_last_runs``
    existing run folders, then adds this run."""
    from noctis.observability.debug import RUN_ID_RE

    qa = tmp_path / "qa"
    qa.mkdir(parents=True)
    older = [f"2026010{i}T000000Z-00000{i}" for i in range(1, 6)]  # 5 sortable run-id folders
    for name in older:
        (qa / name).mkdir()

    settings = _qa_settings(tmp_path, keep_last_runs=2)
    rec = build_recorder(settings, argv=["run", "--debug"], mode="paper")

    remaining = sorted(p.name for p in qa.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name))
    # the two newest pre-existing folders survive, the oldest three are pruned, plus this new run
    assert older[-2:] == ["20260104T000000Z-000004", "20260105T000000Z-000005"]
    assert set(remaining) == {*older[-2:], rec.run_id}
    assert older[0] not in remaining


def test_build_recorder_config_digest_excludes_secrets(tmp_path):
    """The manifest digest is over the resolved settings with API keys excluded (AGENTS.md rule 6):
    two configs that differ only by a secret produce the identical digest."""
    import json

    settings = _qa_settings(tmp_path)
    base = json.loads(
        (build_recorder(settings, argv=[], mode=None).run_dir / "run.json").read_text()
    )["config_digest"]

    settings.anthropic_api_key = "sk-super-secret"
    settings.openai_api_key = "sk-other-secret"
    with_secret = json.loads(
        (build_recorder(settings, argv=[], mode=None).run_dir / "run.json").read_text()
    )["config_digest"]

    assert base == with_secret  # a secret never perturbs the digest


# ── build_research_session: the one bundle both entrypoints run ───────────────────────────
def _session_settings(
    tmp_path,
    *,
    coder_model: str | None = None,
    coder_fallback_model: str | None = None,
    max_escalations: int | None = None,
):
    lines = [
        "research_time_budget_minutes: 42",
        f"state_dir: {tmp_path}/state/",
        f"strategies_dir: {tmp_path}/strategies/",
    ]
    agent: list[str] = []
    if coder_model is not None:
        agent.append(f"    coder_model: {coder_model}")
    if coder_fallback_model is not None:
        agent.append(f"    coder_fallback_model: {coder_fallback_model}")
    if max_escalations is not None:
        agent.append(f"    max_escalations: {max_escalations}")
    if agent:
        lines += ["research:", "  agent:", *agent]
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


def test_a_mandate_model_override_reaches_the_built_research_session(tmp_path, monkeypatch):
    """The model seam is run-shaping (#117): a mandate that declares its own model changes
    which model the composition root builds the session's client for, and which model the
    session reports it drives — not just a settings value nobody reads."""
    mandate_dir = _mandate_dir(
        tmp_path,
        "local",
        "---\nconfig:\n  research:\n    model: ollama/qwen3-coder-30b\n---\nRun local.\n",
    )
    cfg = _config(
        tmp_path,
        [
            f"mandate_dir: {mandate_dir}",
            f"state_dir: {tmp_path}/state/",
            f"strategies_dir: {tmp_path}/strategies/",
            "research:",
            "  model: openai/gpt-5.4",
            "  mandate: local",
        ],
    )
    resolved = resolve_session(cfg)
    assert resolved.settings.research.model == "ollama/qwen3-coder-30b"
    assert "research.model=ollama/qwen3-coder-30b" in resolved.overrides

    seen: dict = {}

    def fake_build_llm_client(settings):
        seen["model"] = settings.research.model
        return object()

    monkeypatch.setattr(research_mod, "build_llm_client", fake_build_llm_client)
    session = build_research_session(
        settings=resolved.settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
        mandate=resolved.mandate,
    )
    assert session is not None
    assert seen["model"] == "ollama/qwen3-coder-30b"  # the client is built for the mandate's model
    assert session.model == "ollama/qwen3-coder-30b"  # and the session drives it


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


# ── the coder-fallback knob (#72): a paid escalation client, threaded here or None ─────────
def test_coder_fallback_client_not_built_when_knob_unset(tmp_path, monkeypatch):
    """No ``coder_fallback_model`` ⇒ no fallback client on the toolbox (today's behavior)."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    coder = _fake_coder()
    monkeypatch.setattr(research_mod, "client_for", lambda *a, **k: coder)
    session = build_research_session(
        settings=_session_settings(tmp_path, coder_model="anthropic/claude-sonnet-5"),
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert session is not None
    assert session.toolbox.coder_fallback_client is None


def test_coder_fallback_client_built_when_configured(tmp_path, monkeypatch):
    """``coder_fallback_model`` set + provider available ⇒ a stateless paid client reaches the
    toolbox, built on its OWN thinking dial defaulting OFF (#98): the strong fallback spends its
    whole output ceiling on the file, so the dial tuned for weak local coders can't sink it."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    local, fallback = _fake_coder(), _fake_coder()
    seen: list[tuple] = []

    def fake_client_for(settings, model, **kwargs):
        seen.append((model, kwargs))
        return fallback if model == "anthropic/claude-opus-4-8" else local

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _session_settings(
        tmp_path,
        coder_model="anthropic/claude-sonnet-5",
        coder_fallback_model="anthropic/claude-opus-4-8",
        max_escalations=2,
    )
    assert settings.research.agent.coder_fallback_thinking == "off"  # the #98 default
    session = build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert session is not None
    assert session.toolbox.coder_client is local
    assert session.toolbox.coder_fallback_client is fallback
    assert session.toolbox.max_escalations == 2
    # The fallback client was built for the configured fallback model, thinking OFF by default
    # (#98) — while the LOCAL coder keeps its own dial's default (ON).
    fb = next(kw for model, kw in seen if model == "anthropic/claude-opus-4-8")
    assert fb.get("thinking") == "off"
    assert fb.get("deliberate") is True
    lc = next(kw for model, kw in seen if model == "anthropic/claude-sonnet-5")
    assert lc.get("thinking") == "on"


def test_coder_fallback_thinking_opt_in(tmp_path, monkeypatch):
    """``coder_fallback_thinking: on`` opts the escalated coder back into deliberate thinking —
    the operator's budgeted decision, independent of the local coder's dial (#98)."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    seen: list[tuple] = []

    def fake_client_for(settings, model, **kwargs):
        seen.append((model, kwargs))
        return _fake_coder()

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _session_settings(
        tmp_path,
        coder_model="anthropic/claude-sonnet-5",
        coder_fallback_model="anthropic/claude-opus-4-8",
    )
    settings.research.agent.coder_fallback_thinking = "on"
    session = build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert session is not None
    fb = next(kw for model, kw in seen if model == "anthropic/claude-opus-4-8")
    assert fb.get("thinking") == "on"
    assert fb.get("deliberate") is True


def test_coder_fallback_client_missing_key_degrades_loudly(tmp_path, monkeypatch, caplog):
    """``coder_fallback_model`` set but its provider key/extra missing ⇒ a loud warning, session
    still assembles with no fallback (escalation simply unavailable) — never a mid-session error."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    coder = _fake_coder()

    def fake_client_for(settings, model, **kwargs):
        # local coder resolves; the paid fallback provider has no key
        return None if model == "anthropic/claude-opus-4-8" else coder

    monkeypatch.setattr(research_mod, "client_for", fake_client_for)
    settings = _session_settings(
        tmp_path,
        coder_model="anthropic/claude-sonnet-5",
        coder_fallback_model="anthropic/claude-opus-4-8",
    )
    with caplog.at_level(logging.WARNING):
        session = build_research_session(
            settings=settings,
            lake=object(),
            registry=object(),
            families=object(),
            memory=object(),
        )
    assert session is not None
    assert session.toolbox.coder_client is coder
    assert session.toolbox.coder_fallback_client is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("fallback" in msg.lower() for msg in warnings)
    assert any("claude-opus-4-8" in msg for msg in warnings)


def test_coder_fallback_not_built_without_a_local_coder(tmp_path, monkeypatch):
    """Escalation is a fallback FROM local authoring: a ``coder_fallback_model`` with no
    ``coder_model`` builds no fallback client (there is nothing to escalate from)."""
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    calls: list = []
    monkeypatch.setattr(research_mod, "client_for", lambda *a, **k: calls.append((a, k)))
    settings = _session_settings(tmp_path, coder_fallback_model="anthropic/claude-opus-4-8")
    session = build_research_session(
        settings=settings,
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    assert session is not None
    assert session.toolbox.coder_fallback_client is None
    assert calls == []  # no local coder ⇒ neither builder is consulted


# ── prune-on-start: sweep stale working-tier drafts before the toolbox loads the library (#56) ─
def _prune_settings(tmp_path, *, draft_ttl_hours: float | str | None = None):
    """Session settings with a fully-owned workspace, so the working tier is a clean tmp path."""
    lines = [
        f"workspace_dir: {tmp_path}/workspace",
        f"state_dir: {tmp_path}/state/",
        f"strategies_dir: {tmp_path}/strategies/",
    ]
    if draft_ttl_hours is not None:
        lines += ["research:", f"  draft_ttl_hours: {draft_ttl_hours}"]
    return load_settings(config_path=_config(tmp_path, lines))


def _corpse(work: Path, name: str, *, status: str = "draft", age_hours: float = 99.0) -> Path:
    """A minimal header-only working-tier draft with a back-dated mtime (prune reads only the
    docstring header, never imports — the same fixture the library sweep tests use)."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{name}.py"
    path.write_text(f'"""Toy {name}.\n\nstatus: {status}\nstyle: momentum\n"""\n', encoding="utf-8")
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))
    return path


def test_build_research_session_prunes_stale_drafts_before_toolbox_constructs(
    tmp_path, monkeypatch, caplog
):
    """Prune-on-start (story #56): a stale, still-undecided working-tier draft is swept into
    __tmp/archive/ before the toolbox loads the library, so no session observes the corpse. The
    archived names and count are logged at INFO."""
    from noctis.strategies import library

    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    settings = _prune_settings(tmp_path)
    work = library.LibraryPaths.from_settings(settings).tmp
    corpse = _corpse(work, "stale_probe")

    with caplog.at_level(logging.INFO, logger="noctis.bootstrap"):
        session = build_research_session(
            settings=settings,
            lake=object(),
            registry=object(),
            families=FamilyRegistry(),
            memory=object(),
        )

    assert session is not None
    assert not corpse.exists()  # swept out of the working tier before the library loaded
    assert (work / "archive" / "000001-stale_probe.py").is_file()
    names = {entry["name"] for entry in library.list_strategies(session.toolbox.strategies_dir)}
    assert "stale_probe" not in names  # absent from the built session's library view
    prune_logs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("stale_probe" in msg and "1" in msg for msg in prune_logs)


def test_build_research_session_draft_ttl_none_disables_prune(tmp_path, monkeypatch, caplog):
    """``research.draft_ttl_hours: null`` is the disable path: the stale corpse survives, the
    archive never materializes, and no prune log is emitted (nothing archived ⇒ no noise)."""
    from noctis.strategies import library

    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: object())
    settings = _prune_settings(tmp_path, draft_ttl_hours="null")
    work = library.LibraryPaths.from_settings(settings).tmp
    corpse = _corpse(work, "stale_probe")

    with caplog.at_level(logging.INFO, logger="noctis.bootstrap"):
        session = build_research_session(
            settings=settings,
            lake=object(),
            registry=object(),
            families=FamilyRegistry(),
            memory=object(),
        )

    assert session is not None
    assert corpse.exists()  # disabled sweep leaves the working tier byte-identical
    assert not (work / "archive").exists()
    assert not any("prune" in r.getMessage().lower() for r in caplog.records)


# ── the research-loop knob (#68): conversation vs the episodic driver, selected here ──────────
def _loop_settings(
    tmp_path, *, loop: str | None = None, distill: int | None = None, window: int | None = None
):
    lines = [
        "research_time_budget_minutes: 42",
        f"state_dir: {tmp_path}/state/",
        f"strategies_dir: {tmp_path}/strategies/",
    ]
    research = []
    if distill is not None:
        research.append(f"  memory_distill_every: {distill}")
    agent = []
    if loop is not None:
        agent.append(f"    loop: {loop}")
    if window is not None:
        agent.append(f"    context_window: {window}")
    if agent:
        research += ["  agent:", *agent]
    if research:
        lines = [*lines, "research:", *research]
    name = f"loop-{loop}-{distill}-{window}.yaml"
    return load_settings(config_path=_config(tmp_path, lines, name))


def test_loop_config_knob_defaults_to_auto_and_accepts_the_three_values(tmp_path):
    from noctis.config.settings import AgentResearchConfig

    assert AgentResearchConfig().loop == "auto"
    for value in ("auto", "conversation", "episodic"):
        assert AgentResearchConfig(loop=value).loop == value
    with pytest.raises(Exception):
        AgentResearchConfig(loop="nonsense")


def test_resolve_research_loop_explicit_picks_always_win(tmp_path):
    small = _EPISODIC_WINDOW_MAX
    assert resolve_research_loop(_loop_settings(tmp_path, loop="episodic")) == "episodic"
    conv = _loop_settings(tmp_path, loop="conversation", window=small)
    assert resolve_research_loop(conv) == "conversation"  # explicit beats a small window


def test_resolve_research_loop_auto_flips_on_small_context_window(tmp_path):
    """The evidence-gated flip (#76): ``auto`` selects episodic when the declared context window
    is at or below the documented threshold, conversation for larger or unset windows."""
    assert _EPISODIC_WINDOW_MAX == 32_768  # the documented constant, not folklore
    assert resolve_research_loop(_loop_settings(tmp_path)) == "conversation"  # unset window
    assert resolve_research_loop(_loop_settings(tmp_path, loop="auto")) == "conversation"
    at = _loop_settings(tmp_path, loop="auto", window=_EPISODIC_WINDOW_MAX)
    assert resolve_research_loop(at) == "episodic"  # inclusive: the canonical 32k local box
    assert resolve_research_loop(_loop_settings(tmp_path, loop="auto", window=8_192)) == "episodic"
    above = _loop_settings(tmp_path, loop="auto", window=_EPISODIC_WINDOW_MAX + 1)
    assert resolve_research_loop(above) == "conversation"
    implicit = _loop_settings(tmp_path, window=16_384)  # loop unset = auto
    assert resolve_research_loop(implicit) == "episodic"


def _session(tmp_path, monkeypatch, *, loop=None, distill=None, window=None):
    client = SimpleNamespace(model="fake/model", capabilities=Capabilities())
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: client)
    return build_research_session(
        settings=_loop_settings(tmp_path, loop=loop, distill=distill, window=window),
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )


def test_loop_episodic_selects_the_episodic_driver(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch, loop="episodic")
    seen: dict = {}
    monkeypatch.setattr(
        ResearchSession,
        "_run_episodic",
        lambda self, **k: seen.setdefault("episodic", k) or ResearchSummary(),
    )
    monkeypatch.setattr(
        ResearchSession,
        "_run_conversation",
        lambda self, **k: seen.setdefault("conversation", k) or ResearchSummary(),
    )
    session.run(max_iterations=3)
    assert "episodic" in seen and "conversation" not in seen
    assert seen["episodic"]["max_iterations"] == 3


def _which_loop_ran(session, monkeypatch) -> set[str]:
    seen: set[str] = set()
    monkeypatch.setattr(
        ResearchSession, "_run_conversation", lambda self, **k: seen.add("conversation")
    )
    monkeypatch.setattr(ResearchSession, "_run_episodic", lambda self, **k: seen.add("episodic"))
    session.run()
    return seen


@pytest.mark.parametrize("loop", [None, "auto"])
def test_loop_auto_without_a_declared_window_selects_the_conversation_loop(
    tmp_path, monkeypatch, loop
):
    session = _session(tmp_path, monkeypatch, loop=loop)
    assert _which_loop_ran(session, monkeypatch) == {"conversation"}


@pytest.mark.parametrize("loop", [None, "auto"])
def test_loop_auto_with_a_small_window_selects_the_episodic_driver(tmp_path, monkeypatch, loop):
    session = _session(tmp_path, monkeypatch, loop=loop, window=_EPISODIC_WINDOW_MAX)
    assert _which_loop_ran(session, monkeypatch) == {"episodic"}


def test_conversation_loop_kwargs_unchanged_under_auto(tmp_path, monkeypatch):
    """The loop knob must not perturb the conversation path — ``auto`` threads exactly the kwargs
    the CLI/runtime always wired into ``run_agent_research``."""
    client = SimpleNamespace(model="fake/model", capabilities=Capabilities())
    monkeypatch.setattr(research_mod, "build_llm_client", lambda settings: client)
    seen: dict = {}
    monkeypatch.setattr(
        research_mod, "run_agent_research", lambda **k: seen.update(k) or ResearchSummary()
    )
    session = build_research_session(
        settings=_loop_settings(tmp_path),  # unset ⇒ auto ⇒ conversation
        lake=object(),
        registry=object(),
        families=object(),
        memory=object(),
    )
    session.run(max_iterations=5)
    assert seen["client"] is client
    assert seen["budget_minutes"] == 42
    assert seen["max_iterations"] == 5


# ── the episodic MATCH fallback panel: a lazy source built here (story #110) ──────────────────
class _PanelLake:
    """A lake fake for the fallback-panel source: readiness is a mutable set (so a mid-session
    ingest can be simulated) and the coverage registry reports one discovered symbol, which the
    trading roster appends after the config seed."""

    def __init__(self, ready, discovered=()):
        self.ready = set(ready)
        self.coverage = SimpleNamespace(
            all=lambda: [
                SimpleNamespace(symbol=s, status="idle", row_count=100) for s in discovered
            ]
        )

    def check_symbol_ready(self, symbol) -> bool:
        return symbol in self.ready


def _episodic_session(tmp_path, monkeypatch, lake, *, mandate=None, history_days=None):
    monkeypatch.setattr(
        research_mod,
        "build_llm_client",
        lambda settings: SimpleNamespace(model="fake/model", capabilities=Capabilities()),
    )
    data = [] if history_days is None else ["data:", f"  history_days: {history_days}"]
    settings = load_settings(
        config_path=_config(
            tmp_path,
            [
                f"state_dir: {tmp_path}/state/",
                f"strategies_dir: {tmp_path}/strategies/",
                "universe: [AAA, BBB, CCC]",
                *data,
                "research:",
                "  fit_set_size: 2",
                "  agent:",
                "    loop: episodic",
            ],
            "panel.yaml",
        )
    )
    return build_research_session(
        settings=settings,
        lake=lake,
        registry=object(),
        families=FamilyRegistry(),
        memory=object(),
        mandate=mandate,
    )


def test_composition_root_builds_the_fallback_panel_source_over_roster_and_readiness(
    tmp_path, monkeypatch
):
    """The composition root hands the episodic session a zero-arg SOURCE, not a precomputed list —
    and it resolves to exactly the panel the precomputed list held: the ready trading-roster names
    (config seed, then lake discoveries) capped at ``research.fit_set_size``."""
    from noctis.research import driver as driver_mod

    lake = _PanelLake(ready={"CCC", "ZZZ"}, discovered=["ZZZ"])
    session = _episodic_session(tmp_path, monkeypatch, lake)
    seen: dict = {}
    monkeypatch.setattr(
        driver_mod, "run_episodic_research", lambda **k: seen.update(k) or ResearchSummary()
    )

    session.run(max_iterations=1)

    source = seen["fallback_panel_source"]
    # AAA/BBB are seeded but not ready; CCC is; ZZZ joined via the coverage registry — capped at 2.
    assert source() == ["CCC", "ZZZ"]

    # A symbol that becomes lake-ready mid-session is visible to the NEXT call — the whole point of
    # the source: the panel is no longer frozen at assembly time.
    lake.ready.add("AAA")
    assert source() == ["AAA", "CCC"]


def test_composition_root_passes_no_precomputed_fit_panel(tmp_path, monkeypatch):
    """The frozen-list parameter is gone from the session entry, so the composition root must not
    hand one over — a stale panel can never be smuggled past the rename."""
    from noctis.research import driver as driver_mod

    session = _episodic_session(tmp_path, monkeypatch, _PanelLake(ready={"AAA"}))
    seen: dict = {}
    monkeypatch.setattr(
        driver_mod, "run_episodic_research", lambda **k: seen.update(k) or ResearchSummary()
    )

    session.run(max_iterations=1)

    assert "fit_symbols" not in seen
    assert callable(seen["fallback_panel_source"])


# ── the mandate-symbol data preflight is threaded from here (story #111) ──────────────────────
def _episodic_kwargs(tmp_path, monkeypatch, *, mandate=None, history_days=None) -> dict:
    """Run one episodic session with the driver entry stubbed; returns the kwargs it was handed."""
    from noctis.research import driver as driver_mod

    session = _episodic_session(
        tmp_path,
        monkeypatch,
        _PanelLake(ready={"AAA"}),
        mandate=mandate,
        history_days=history_days,
    )
    seen: dict = {}
    monkeypatch.setattr(
        driver_mod, "run_episodic_research", lambda **k: seen.update(k) or ResearchSummary()
    )
    session.run(max_iterations=1)
    return seen


def _mandate(symbols):
    """A resolved mandate declaring ``symbols`` — the only field the preflight threading reads."""
    from noctis.research.mandate import Mandate

    return Mandate(
        text="hunt liquid momentum",
        source="cli",
        summary="momentum",
        references=[],
        config_overrides={},
        symbols=list(symbols),
    )


def test_composition_root_threads_the_mandate_symbols_and_history_window(tmp_path, monkeypatch):
    """The driver imports no settings, so the two preflight inputs arrive as parameters: the
    resolved mandate's declared symbols and ``data.history_days`` (an existing knob, unchanged)."""
    seen = _episodic_kwargs(
        tmp_path, monkeypatch, mandate=_mandate(["QQQ", "IWM"]), history_days=90
    )

    assert list(seen["mandate_symbols"]) == ["QQQ", "IWM"]
    assert seen["history_days"] == 90


def test_composition_root_declares_no_preflight_symbols_without_a_mandate(tmp_path, monkeypatch):
    """No mandate ⇒ no declared symbols, so the preflight is a no-op and the session unchanged."""
    seen = _episodic_kwargs(tmp_path, monkeypatch, history_days=90)

    assert list(seen["mandate_symbols"]) == []
    assert seen["history_days"] == 90


def test_composition_root_threads_the_default_history_window(tmp_path, monkeypatch):
    """An operator who never touched ``data.history_days`` still gets its shipped default."""
    from noctis.config.settings import DataConfig

    seen = _episodic_kwargs(tmp_path, monkeypatch, mandate=_mandate(["QQQ"]))

    assert seen["history_days"] == DataConfig().history_days


# ── the DISCOVER episode is assembled here too (story #112) ───────────────────────────────────
def test_composition_root_wires_the_discover_episode(tmp_path, monkeypatch):
    """Every episodic session gets the third judgment episode wired, so a no-lake-match MATCH can
    spend one on candidate tickers instead of silently falling back to the default panel. It rides
    the same runner (one completions counter, one episode budget) as formulate/decide."""
    seen = _episodic_kwargs(tmp_path, monkeypatch, mandate=_mandate(["QQQ"]), history_days=90)

    assert callable(seen["discover"])
    assert callable(seen["formulate"]) and callable(seen["decide"])
    # The discover fetch covers the same window the mandate preflight fetches over.
    assert seen["history_days"] == 90


# ── memory_distill_every defaults ON in episodic mode; conversation stays bit-identical ──────
def test_effective_memory_distill_every_defaults_on_only_for_episodic(tmp_path):
    # Episodic + operator left it at 0 ⇒ defaults on (a modest cadence).
    assert effective_memory_distill_every(_loop_settings(tmp_path, loop="episodic")) == 1
    # Conversation / auto / unset ⇒ off, exactly as today.
    assert effective_memory_distill_every(_loop_settings(tmp_path)) == 0
    assert effective_memory_distill_every(_loop_settings(tmp_path, loop="conversation")) == 0
    # An explicit operator value always wins, in either loop.
    assert effective_memory_distill_every(_loop_settings(tmp_path, loop="episodic", distill=4)) == 4
    assert effective_memory_distill_every(_loop_settings(tmp_path, distill=7)) == 7


def test_running_the_episodic_loop_applies_the_distill_default_to_settings(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch, loop="episodic")
    assert session.settings.research.memory_distill_every == 0  # before the run
    monkeypatch.setattr(ResearchSession, "_run_episodic", lambda self, **k: ResearchSummary())
    session.run()
    # CLOSE reads the shared settings instance, so the episodic default now governs distillation.
    assert session.settings.research.memory_distill_every == 1


def test_running_the_conversation_loop_leaves_distill_untouched(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch, loop="conversation")
    monkeypatch.setattr(research_mod, "run_agent_research", lambda **k: ResearchSummary())
    session.run()
    assert session.settings.research.memory_distill_every == 0  # bit-identical to today
