"""Tests for the layered settings loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from noctis.config import DataConfig, Settings, load_settings

REPO_EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"


def _write_yaml(path, body: str):
    path.write_text(textwrap.dedent(body))
    return path


def test_defaults_when_no_config(monkeypatch, tmp_path):
    """With no YAML and no env, defaults apply and the app is paper by default."""
    settings = load_settings(config_path=tmp_path / "missing.yaml")
    assert settings.mode == "paper"
    assert settings.allow_live is False
    assert "AAPL" in settings.universe
    assert settings.champion_count == 3
    assert isinstance(settings.data, DataConfig)
    assert settings.data.dataset == "EQUS.MINI"


def test_yaml_knobs_are_loaded(tmp_path):
    """Knobs (including nested ones) come from the YAML file."""
    cfg = _write_yaml(
        tmp_path / "config.yaml",
        """
        mode: paper
        universe: [SPY, QQQ]
        champion_count: 5
        research_time_budget_minutes: 15
        risk:
          max_daily_loss_pct: 1.5
        data:
          budget_usd: 42.0
          dataset: XNAS.ITCH
        """,
    )
    settings = load_settings(config_path=cfg)
    assert settings.universe == ["SPY", "QQQ"]
    assert settings.champion_count == 5
    assert settings.research_time_budget_minutes == 15
    assert settings.risk.max_daily_loss_pct == 1.5
    assert settings.data.budget_usd == 42.0
    assert settings.data.dataset == "XNAS.ITCH"


def test_env_overrides_yaml_for_secrets(monkeypatch, tmp_path):
    """Environment variables win over the YAML file (secrets live only in env)."""
    cfg = _write_yaml(tmp_path / "config.yaml", "mode: paper\n")
    monkeypatch.setenv("DATABENTO_API_KEY", "db-secret-123")
    settings = load_settings(config_path=cfg)
    assert settings.databento_api_key == "db-secret-123"


def test_env_overrides_yaml_for_knobs(monkeypatch, tmp_path):
    """A knob set in both env and YAML resolves to the env value."""
    cfg = _write_yaml(tmp_path / "config.yaml", "champion_count: 3\n")
    monkeypatch.setenv("CHAMPION_COUNT", "9")
    settings = load_settings(config_path=cfg)
    assert settings.champion_count == 9


def test_allow_live_reads_ALLOW_LIVE_env(monkeypatch, tmp_path):
    """The allow_live field is sourced from the ALLOW_LIVE environment variable."""
    cfg = _write_yaml(tmp_path / "config.yaml", "mode: paper\n")
    monkeypatch.setenv("ALLOW_LIVE", "true")
    settings = load_settings(config_path=cfg)
    assert settings.allow_live is True


def test_constructor_overrides_win(tmp_path):
    """Explicit constructor overrides beat everything (useful for tests)."""
    cfg = _write_yaml(tmp_path / "config.yaml", "champion_count: 3\n")
    settings = load_settings(config_path=cfg, champion_count=7)
    assert settings.champion_count == 7


def test_election_metric_defaults_to_sharpe(tmp_path):
    settings = load_settings(config_path=tmp_path / "missing.yaml")
    assert settings.promotion.metric == "sharpe"


def test_election_metric_and_gates_load_from_yaml(tmp_path):
    """The risk-appetite metric and its gate thresholds all come from the promotion block."""
    cfg = _write_yaml(
        tmp_path / "config.yaml",
        """
        promotion:
          metric: total_return
          max_gap: 0.15
          min_test_metric: 0.05
          min_holdout_metric: 0.02
        """,
    )
    settings = load_settings(config_path=cfg)
    assert settings.promotion.metric == "total_return"
    assert settings.promotion.max_gap == 0.15
    assert settings.promotion.min_test_metric == 0.05
    assert settings.promotion.min_holdout_metric == 0.02


def test_unknown_election_metric_refused_at_load(tmp_path):
    """The validator routes through Metric.parse — the one unknown-metric diagnosis."""
    cfg = _write_yaml(tmp_path / "config.yaml", "promotion:\n  metric: alpha\n")
    with pytest.raises(ValidationError, match="unknown metric 'alpha'"):
        load_settings(config_path=cfg)


def test_research_panel_defaults(tmp_path):
    """Panel research + symbol-holdout knobs default on (6 fit / 2 held out, gates at 0)."""
    settings = load_settings(config_path=tmp_path / "missing.yaml")
    assert settings.research.fit_set_size == 6
    assert settings.research.symbol_holdout_size == 2
    assert settings.research.tuning_dispersion_penalty == 0.0
    assert settings.promotion.min_symbol_holdout_metric == 0.0
    assert settings.promotion.min_symbol_consistency == 0.0
    assert settings.promotion.min_test_activity == 0.0


def test_research_panel_loads_from_yaml(tmp_path):
    cfg = _write_yaml(
        tmp_path / "config.yaml",
        """
        research:
          fit_set_size: 4
          symbol_holdout_size: 1
        promotion:
          min_symbol_holdout_metric: 0.1
          min_symbol_consistency: 0.6
          min_test_activity: 0.05
        """,
    )
    settings = load_settings(config_path=cfg)
    assert settings.research.fit_set_size == 4
    assert settings.research.symbol_holdout_size == 1
    assert settings.promotion.min_symbol_holdout_metric == 0.1
    assert settings.promotion.min_symbol_consistency == 0.6
    assert settings.promotion.min_test_activity == 0.05


def test_unknown_election_metric_rejected(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", "promotion:\n  metric: profitz\n")
    with pytest.raises(ValidationError):
        load_settings(config_path=cfg)


def test_coder_model_defaults_to_none(tmp_path):
    """The dedicated authoring model is off by default — the driver writes full source."""
    settings = load_settings(config_path=tmp_path / "missing.yaml")
    assert settings.research.agent.coder_model is None


def test_coder_model_loads_from_yaml(tmp_path):
    """The cheap-driver + hosted-coder pairing comes from the agent block."""
    cfg = _write_yaml(
        tmp_path / "config.yaml",
        """
        research:
          model: ollama_chat/noctis-qwen3:14b
          agent:
            coder_model: anthropic/claude-sonnet-5
        """,
    )
    settings = load_settings(config_path=cfg)
    assert settings.research.model == "ollama_chat/noctis-qwen3:14b"
    assert settings.research.agent.coder_model == "anthropic/claude-sonnet-5"


def test_coder_model_env_overrides_yaml(monkeypatch, tmp_path):
    """Env wins over YAML for the coder knob, like the other agent-research knobs."""
    cfg = _write_yaml(
        tmp_path / "config.yaml",
        "research:\n  agent:\n    coder_model: anthropic/claude-sonnet-5\n",
    )
    monkeypatch.setenv("RESEARCH__AGENT__CODER_MODEL", "anthropic/claude-opus-4-8")
    settings = load_settings(config_path=cfg)
    assert settings.research.agent.coder_model == "anthropic/claude-opus-4-8"


def test_example_config_ships_the_driver_coder_pairing():
    """The example config carries the commented local-driver + hosted-coder pairing (#4) —
    the whole point of the knob — under research.agent, still fully commented out."""
    lines = REPO_EXAMPLE_CONFIG.read_text(encoding="utf-8").splitlines()
    coder_lines = [ln for ln in lines if "coder_model" in ln]
    assert coder_lines, "example config should mention coder_model"
    assert all(ln.lstrip().startswith("#") for ln in coder_lines)  # inert example, not active
    assert any("anthropic/claude-sonnet-5" in ln for ln in coder_lines)
    assert any("ollama_chat/noctis-qwen3:14b" in ln for ln in lines)


def test_settings_is_the_public_type():
    """load_settings returns a Settings instance."""
    assert isinstance(load_settings(), Settings)
