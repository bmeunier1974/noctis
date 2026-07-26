"""The deny-by-default settings overlay — classifier + validating applier.

``noctis.config.overlay`` is a statement about *settings*: it classifies every leaf dotted
path in :class:`~noctis.config.settings.Settings` exactly once (allowed / clamped /
refused-with-a-reason) and applies a flattened patch by re-validating the owning section
through pydantic. It owns no mandate and no LLM concepts, so everything here is exercised
without a mandate file.

The two **ratchets** come first and are the load-bearing tests: the classification is total
over the live model (a newly added config field fails the suite until someone classifies
it), and the sample-value table the apply tests draw from is total over the allowed set
(widening the surface without testing it is impossible).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from noctis.config import load_settings, overlay
from noctis.config.overlay import (
    ALLOWED,
    CLAMPED,
    REFUSED,
    OverlayError,
    apply_patch,
    assert_gates_unmoved,
    classify,
    gate_snapshot,
)
from noctis.config.settings import Settings


# ─────────────────────────────────────────────────────────────────────────────
# The live model's leaves — walked here, independently of the module under test,
# so a bug in the module's own walk can never make the ratchet pass vacuously.
# ─────────────────────────────────────────────────────────────────────────────
def _leaf_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    leaves: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            leaves |= _leaf_paths(annotation, f"{prefix}{name}.")
        else:
            leaves.add(f"{prefix}{name}")
    return leaves


SETTINGS_LEAVES = _leaf_paths(Settings)

# ─────────────────────────────────────────────────────────────────────────────
# The tier-A run-shaping surface (#117), written out here group by group so this
# suite states the contract independently of the module it checks.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_SEAM = {
    "research.model",
    "research.base_url",
    "research.agent.model",
    "research.agent.coder_model",
    "research.agent.coder_fallback_model",
    "research.agent.thinking",
    "research.agent.coder_thinking",
    "research.agent.coder_fallback_thinking",
    "research.agent.loop",
}
SPEND_CEILINGS = {
    "research.cost_profile",
    "research.agent.max_iterations",
    "research.agent.max_backtests",
    "research.agent.sweep_trials",
    "research.agent.max_author_calls",
    "research.agent.max_escalations",
    "research.agent.max_tokens",
    "research.agent.coder_max_tokens",
    "research.agent.context_window",
    "research.agent.episode_retries",
    "research.agent.web_search",
    "research.agent.max_web_searches",
    "research.agent.sweep_workers",
    "research.agent.worker_bar_budget",
    "research_time_budget_minutes",
    "time_limit_hours",
}
SEARCH_SHAPE = {
    "promotion.metric",
    "research.focus_size",
    "research.tuning_dispersion_penalty",
    "research.draft_ttl_hours",
    "research.memory_distill_every",
}
DATA_ACQUISITION = {"data.history_days", "data.auto_backfill"}
HOUSEKEEPING = {"observability.heartbeat_polls", "qa.keep_last_runs"}

RUN_SHAPING = MODEL_SEAM | SPEND_CEILINGS | SEARCH_SHAPE | DATA_ACQUISITION | HOUSEKEEPING

# The per-path sample values the apply tests draw from. Ratcheted below to stay total over
# ALLOWED, so a widened surface cannot ship untested.
SAMPLE_VALUES: dict[str, object] = {
    # Model / provider seam.
    "research.model": "ollama/qwen3-coder-30b",
    "research.base_url": "http://localhost:11434/v1",
    "research.agent.model": "claude-sonnet-5",
    "research.agent.coder_model": "ollama/qwen2.5-coder",
    "research.agent.coder_fallback_model": "anthropic/claude-sonnet-5",
    "research.agent.thinking": "on",
    "research.agent.coder_thinking": "off",
    "research.agent.coder_fallback_thinking": "on",
    "research.agent.loop": "episodic",
    # Spend + compatibility ceilings.
    "research.cost_profile": "economy",
    "research.agent.max_iterations": 12,
    "research.agent.max_backtests": 30,
    "research.agent.sweep_trials": 24,
    "research.agent.max_author_calls": 9,
    "research.agent.max_escalations": 2,
    "research.agent.max_tokens": 4096,
    "research.agent.coder_max_tokens": 6000,
    "research.agent.context_window": 32768,
    "research.agent.episode_retries": 3,
    "research.agent.web_search": True,
    "research.agent.max_web_searches": 3,
    "research.agent.sweep_workers": 4,
    "research.agent.worker_bar_budget": 3000000,
    "research_time_budget_minutes": 45,
    "time_limit_hours": 12.0,
    # Search shape (prompt-facing).
    "promotion.metric": "sortino",
    "research.focus_size": 8,
    "research.tuning_dispersion_penalty": 0.25,
    "research.draft_ttl_hours": 24.0,
    "research.memory_distill_every": 3,
    # Data acquisition for the mandate's own symbols.
    "data.history_days": 180,
    "data.auto_backfill": True,
    # Display / housekeeping.
    "observability.heartbeat_polls": 30,
    "qa.keep_last_runs": 5,
}


def _settings(tmp_path):
    """Settings from pure defaults (the repo config.yaml is bypassed by a missing path)."""
    return load_settings(config_path=tmp_path / "missing.yaml")


# ── ratchet 1: the classification is total over the live settings model ──────────────────
def test_classification_is_total_over_the_settings_model():
    """Every leaf is classified, and nothing is classified that the model doesn't have.

    This is the completeness property the old flat allowlist lacked: a field added to
    Settings tomorrow is refused *by an explicit decision*, never by accident of omission.
    """
    classified = set(ALLOWED) | set(CLAMPED) | set(REFUSED)
    unclassified = sorted(SETTINGS_LEAVES - classified)
    assert not unclassified, (
        f"{len(unclassified)} settings leaf/leaves are unclassified — add each to ALLOWED, "
        f"CLAMPED, or REFUSED in noctis.config.overlay: {unclassified}"
    )
    stale = sorted(classified - SETTINGS_LEAVES)
    assert not stale, f"classified paths the settings model no longer has: {stale}"


def test_no_leaf_is_classified_twice():
    assert not set(ALLOWED) & set(CLAMPED)
    assert not set(ALLOWED) & set(REFUSED)
    assert not set(CLAMPED) & set(REFUSED)


def test_every_refusal_carries_a_reason():
    blank = sorted(path for path, reason in REFUSED.items() if not str(reason).strip())
    assert not blank, f"refused paths with no reason: {blank}"


# ── ratchet 2: the sample-value table is total over the allowed surface ──────────────────
def test_sample_value_table_covers_the_whole_allowed_surface():
    """Widening ALLOWED without adding a sample value fails here, before the apply tests."""
    missing = sorted(set(ALLOWED) - set(SAMPLE_VALUES))
    assert not missing, f"allowed paths with no sample value in this suite: {missing}"
    stale = sorted(set(SAMPLE_VALUES) - set(ALLOWED))
    assert not stale, f"sample values for paths that are not allowed: {stale}"


def test_allowed_surface_is_the_run_shaping_tier():
    """A mandate configures the whole run — model, budgets, prompt shape, scoring metric —
    and nothing else (#117). ``config.yaml`` keeps the arena."""
    assert set(ALLOWED) == RUN_SHAPING
    assert CLAMPED == {}  # the direction clamps are their own stage


@pytest.mark.parametrize("path", ["universe", "research.min_trials", "data.budget_usd"])
def test_the_paths_that_need_a_guard_first_are_still_refused(path):
    """``universe`` carries the symbol-holdout starvation guard, and the two clamped-tier
    knobs carry a direction — each lands with its guard, not before."""
    verdict = classify(path)
    assert verdict.tier == "refused"
    assert "not yet in the overlay surface" in verdict.reason


# ── classify: one verdict per path, and a verdict for a path the model lacks ─────────────
@pytest.mark.parametrize("path", sorted(SETTINGS_LEAVES))
def test_classify_returns_the_tier_the_tables_declare(path):
    verdict = classify(path)
    expected = "allowed" if path in ALLOWED else "clamped" if path in CLAMPED else "refused"
    assert verdict.tier == expected
    if expected == "refused":
        assert verdict.reason == REFUSED[path]


def test_classify_reports_an_unknown_path():
    verdict = classify("promotion.no_such_knob")
    assert verdict.tier == "unknown"
    assert verdict.reason


def test_classify_honors_a_clamped_entry_over_a_refusal(monkeypatch):
    """CLAMPED is empty today, but it is the tier the direction clamps land in: an entry
    there must win over the same path's refusal, and carry the legal direction."""
    monkeypatch.setattr(overlay, "CLAMPED", {"research.min_trials": "raise_only"})
    verdict = classify("research.min_trials")
    assert verdict.tier == "clamped"
    assert verdict.direction == "raise_only"
    assert "raised" in verdict.reason


# ── apply: the allowed surface ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", sorted(ALLOWED))
def test_allowed_path_applies_and_echoes(tmp_path, path):
    settings = _settings(tmp_path)
    before = gate_snapshot(settings)
    lines = apply_patch(settings, {path: SAMPLE_VALUES[path]})
    assert lines == [f"{path}={SAMPLE_VALUES[path]}"]
    assert _read(settings, path) == SAMPLE_VALUES[path]
    assert_gates_unmoved(before, gate_snapshot(settings))  # no gate moved


def test_the_whole_allowed_surface_applies_in_one_patch(tmp_path):
    """The maximal legal overlay: every run-shaping knob at once, every value landed, and the
    refused subtree byte-identical afterwards. This is the property the whole tier rests on —
    a mandate may reconfigure the run and still cannot move the arena."""
    settings = _settings(tmp_path)
    before = json.dumps(gate_snapshot(settings), sort_keys=True, default=str)

    lines = apply_patch(settings, dict(SAMPLE_VALUES))

    assert lines == sorted(f"{path}={value}" for path, value in SAMPLE_VALUES.items())
    for path, value in SAMPLE_VALUES.items():
        assert _read(settings, path) == value
    after = json.dumps(gate_snapshot(settings), sort_keys=True, default=str)
    assert after == before
    assert_gates_unmoved(gate_snapshot(settings), gate_snapshot(_settings(tmp_path)))


def test_empty_patch_is_a_noop(tmp_path):
    settings = _settings(tmp_path)
    assert apply_patch(settings, {}) == []
    assert settings.model_dump() == _settings(tmp_path).model_dump()


def _read(settings, path: str):
    node = settings
    for part in path.split("."):
        node = getattr(node, part)
    return node


# ── apply: the refused surface is fatal, and leaves nothing behind ───────────────────────
@pytest.mark.parametrize("path", sorted(REFUSED))
def test_refused_path_raises_and_changes_nothing(tmp_path, path):
    settings = _settings(tmp_path)
    baseline = _settings(tmp_path).model_dump_json()
    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {path: "anything-at-all"})
    assert path in str(exc.value)
    assert REFUSED[path] in str(exc.value)
    assert settings.model_dump_json() == baseline


def test_unknown_path_is_fatal(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(OverlayError, match="promotion.metrik"):
        apply_patch(settings, {"promotion.metrik": "sortino"})
    assert settings.promotion.metric == "sharpe"


def test_an_allowed_key_beside_a_refused_one_applies_nothing(tmp_path):
    """A patch is all-or-nothing: the operator fixes the mandate, then the whole thing runs."""
    settings = _settings(tmp_path)
    with pytest.raises(OverlayError):
        apply_patch(settings, {"promotion.metric": "sortino", "mode": "live"})
    assert settings.promotion.metric == "sharpe"
    assert settings.mode == "paper"


# The refusals today's suite names explicitly — the ones a reader must be able to point at.
@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("backtest.fee_bps", 0.1),  # the fill costs: the enforced arena difficulty
        ("backtest.slippage_bps", 0.1),
        ("promotion.max_gap", 5.0),  # every promotion threshold
        ("promotion.min_test_metric", -5.0),
        ("promotion.min_holdout_metric", -5.0),
        ("promotion.min_symbol_holdout_metric", -5.0),
        ("promotion.min_symbol_consistency", 0.0),
        ("promotion.min_test_activity", 0.0),
        ("promotion.annualization_cap", 10_000),
        ("promotion.max_period_ratio", 99.0),
        ("promotion.max_reverse_gap", 0.0),
        ("promotion.max_test_metric", 0.0),
        ("databento_api_key", "db-secret"),  # each secret
        ("anthropic_api_key", "sk-secret"),
        ("openai_api_key", "sk-secret"),
        ("state_dir", "/tmp/fresh-state"),  # the state + strategies dirs
        ("strategies_dir", "/tmp/other-strategies"),
        ("research.mandate", "aggressive"),  # no mandate chaining
        ("mode", "live"),  # both live-money gates
        ("allow_live", True),
    ],
)
def test_named_refusals_stay_refused(tmp_path, path, value):
    settings = _settings(tmp_path)
    with pytest.raises(OverlayError, match=path.replace(".", r"\.")):
        apply_patch(settings, {path: value})


def test_one_error_lists_every_problem_with_its_reason(tmp_path):
    """Fixing a bad overlay is one pass, not a fix-one-rerun loop — and each line says why."""
    settings = _settings(tmp_path)
    patch = {
        "mode": "live",
        "backtest.fee_bps": 0.1,
        "promotion.max_gap": 5.0,
        "state_dir": "/tmp/fresh-state",
        "research.symbol_holdout_size": 0,
        "promotion.metrik": "sortino",  # a typo, not a knob
    }
    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, patch)
    message = str(exc.value)
    for path in patch:
        assert path in message
    for path in ("mode", "backtest.fee_bps", "promotion.max_gap", "state_dir"):
        assert REFUSED[path] in message
    assert "6" in message.splitlines()[0]  # the count leads, so nothing hides below the fold


# ── values are validated by exactly the validators config.yaml gets ──────────────────────
def _assert_bad_value_fails_identically(tmp_path, path, value, yaml_body, diagnosis):
    """Same bad value, two sources: the YAML file and the overlay must reject it with the
    same diagnosis, and the overlay must leave no half-applied patch behind."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml_body, encoding="utf-8")
    with pytest.raises(ValidationError) as from_yaml:
        load_settings(config_path=cfg)
    assert diagnosis in str(from_yaml.value)

    settings = _settings(tmp_path)
    baseline = settings.model_dump_json()
    with pytest.raises(OverlayError) as from_overlay:
        apply_patch(settings, {path: value})
    assert diagnosis in str(from_overlay.value)
    assert path in str(from_overlay.value)
    assert settings.model_dump_json() == baseline  # a rejected patch leaves no half-apply


@pytest.mark.parametrize(
    ("path", "value", "yaml_body", "diagnosis"),
    [
        (
            "promotion.metric",
            "alpha",
            "promotion:\n  metric: alpha\n",
            "unknown metric 'alpha'",
        ),
        (
            "research.agent.loop",
            "turbo",
            "research:\n  agent:\n    loop: turbo\n",
            "'auto', 'conversation' or 'episodic'",
        ),
        (
            "research.agent.max_tokens",
            "lots",
            "research:\n  agent:\n    max_tokens: lots\n",
            "valid integer",
        ),
        (
            "research.cost_profile",
            "lavish",
            "research:\n  cost_profile: lavish\n",
            "'full', 'balanced' or 'economy'",
        ),
    ],
)
def test_a_bad_value_fails_identically_from_either_source(
    tmp_path, path, value, yaml_body, diagnosis
):
    """The applier re-validates the owning section, so the metric parser, ``Literal``
    membership, and type coercion all fire exactly as they do for the YAML file — surfacing
    as the one overlay exception type. Every path here is genuinely allowed."""
    _assert_bad_value_fails_identically(tmp_path, path, value, yaml_body, diagnosis)


@pytest.mark.parametrize(
    ("path", "value", "yaml_body", "diagnosis"),
    [
        (
            "backtest.fee_bps",
            0.1,
            "backtest:\n  fee_bps: 0.1\n",
            "below the enforced minimum",
        ),
        (
            "champion_count",
            "three",
            "champion_count: three\n",
            "valid integer",
        ),
    ],
)
def test_validation_would_hold_for_a_path_a_later_stage_allows(
    tmp_path, monkeypatch, path, value, yaml_body, diagnosis
):
    """The property under test is "whatever the surface holds is validated", so it is checked
    against refused paths too — the section rebuild and the top-level ``TypeAdapter`` path —
    by putting one in ALLOWED for the duration of the test. These two stay refused in
    production (the cost floor is the arena, the board size is a gate)."""
    monkeypatch.setattr(overlay, "ALLOWED", frozenset(set(ALLOWED) | {path}))
    _assert_bad_value_fails_identically(tmp_path, path, value, yaml_body, diagnosis)


def test_apply_never_rereads_env_dotenv_or_yaml(tmp_path, monkeypatch):
    """The applier rebuilds only the sections it touches, from their own dumps.

    Proven by moving every other source under a loaded settings object's feet — the YAML
    file's contents and the environment — and showing that applying a patch to the
    ``promotion`` section neither re-reads its siblings nor re-reads the section itself.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text("champion_count: 5\npromotion:\n  max_gap: 0.25\n", encoding="utf-8")
    settings = load_settings(config_path=cfg)
    assert (settings.champion_count, settings.promotion.max_gap) == (5, 0.25)

    cfg.write_text("champion_count: 99\npromotion:\n  max_gap: 9.0\n", encoding="utf-8")
    monkeypatch.setenv("CHAMPION_COUNT", "77")
    monkeypatch.setenv("PROMOTION__MAX_GAP", "7.0")
    # The sources really did move: a fresh load sees them.
    assert load_settings(config_path=cfg).champion_count == 77

    assert apply_patch(settings, {"promotion.metric": "sortino"}) == ["promotion.metric=sortino"]
    assert settings.promotion.metric == "sortino"
    assert settings.promotion.max_gap == 0.25  # the rebuilt section kept its loaded values
    assert settings.champion_count == 5  # and no sibling was re-resolved


# ── the gate snapshot + the gates-unmoved assertion ──────────────────────────────────────
def test_gate_snapshot_is_the_refused_subtree(tmp_path):
    snapshot = gate_snapshot(_settings(tmp_path))
    assert set(snapshot) == set(REFUSED)
    assert snapshot["mode"] == "paper"
    assert snapshot["backtest.fee_bps"] == 1.0
    assert snapshot["promotion.max_gap"] == 1.0


def test_gate_snapshot_survives_an_overlay(tmp_path):
    settings = _settings(tmp_path)
    before = gate_snapshot(settings)
    apply_patch(settings, {"promotion.metric": "total_return"})
    assert gate_snapshot(settings) == before
    assert_gates_unmoved(before, gate_snapshot(settings))


def test_gate_snapshot_does_not_alias_mutable_values(tmp_path):
    settings = _settings(tmp_path)
    before = gate_snapshot(settings)
    settings.universe.append("SMR")  # an in-place mutation of a refused list
    with pytest.raises(OverlayError, match="universe"):
        assert_gates_unmoved(before, gate_snapshot(settings))


def test_assert_gates_unmoved_names_the_moved_gate_without_its_value(tmp_path):
    settings = _settings(tmp_path)
    before = gate_snapshot(settings)
    settings.promotion.max_gap = 99.0
    settings.databento_api_key = "db-secret-123"
    with pytest.raises(OverlayError) as exc:
        assert_gates_unmoved(before, gate_snapshot(settings))
    message = str(exc.value)
    assert "promotion.max_gap" in message
    assert "databento_api_key" in message
    assert "db-secret-123" not in message  # a diagnostic is no place for a credential
    assert "99.0" not in message
