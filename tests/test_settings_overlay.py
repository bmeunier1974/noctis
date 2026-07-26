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
    patch_snapshot,
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
SEED_UNIVERSE = {"universe"}

RUN_SHAPING = (
    MODEL_SEAM | SPEND_CEILINGS | SEARCH_SHAPE | DATA_ACQUISITION | HOUSEKEEPING | SEED_UNIVERSE
)

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
    # The seed trading roster — already normalized (upper-case, no repeats) and long enough to
    # fill the fit set + the symbol holdout under the shipped geometry, so this sample clears
    # the starvation guard the path ships with.
    "universe": ["SMR", "CCJ", "LEU", "URA", "NNE", "OKLO", "BWXT", "VST"],
}

# The tier-B direction clamps (#118), stated here as the contract: the legal direction, the
# value pure defaults resolve for the path, one value that adds discipline and one that
# removes it. Ratcheted below to stay total over CLAMPED, so a widened clamp tier cannot ship
# without an apply test.
CLAMP_SAMPLES: dict[str, dict[str, object]] = {
    # The exhaustion floor: a count, so demanding more evidence per verdict is legitimate
    # steering and demanding less is the loosening AGENTS.md rule 5 forbids.
    "research.min_trials": {
        "direction": "raise_only",
        "configured": 8,
        "adds_discipline": 20,
        "removes_discipline": 4,
    },
    # The vendor spend cap: a mandate may spend less of the operator's money, never more.
    "data.budget_usd": {
        "direction": "lower_only",
        "configured": 125.0,
        "adds_discipline": 40.0,
        "removes_discipline": 250.0,
    },
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


def test_clamp_sample_table_covers_the_whole_clamped_surface():
    """Widening CLAMPED without stating its direction and its two sample values fails here."""
    missing = sorted(set(CLAMPED) - set(CLAMP_SAMPLES))
    assert not missing, f"clamped paths with no sample values in this suite: {missing}"
    stale = sorted(set(CLAMP_SAMPLES) - set(CLAMPED))
    assert not stale, f"sample values for paths that are not clamped: {stale}"
    for path, sample in CLAMP_SAMPLES.items():
        assert CLAMPED[path] == sample["direction"]


def test_allowed_surface_is_the_run_shaping_tier():
    """A mandate configures the whole run — model, budgets, prompt shape, scoring metric —
    and nothing else (#117). ``config.yaml`` keeps the arena."""
    assert set(ALLOWED) == RUN_SHAPING


def test_clamped_surface_is_the_two_direction_clamped_knobs():
    """Tier B (#118): the exhaustion floor may only go up, the vendor budget only down."""
    assert CLAMPED == {"research.min_trials": "raise_only", "data.budget_usd": "lower_only"}
    assert not set(CLAMPED) & set(REFUSED)  # they left the refused tier in the same change


def test_universe_is_allowed_now_that_it_ships_with_its_guard():
    """``universe`` was held out of tier A until its starvation guard existed (#117), because a
    knob without its guard is a hole. It lands allowed *with* the guard (#121), so no path is
    left carrying the "not yet in the overlay surface" placeholder reason."""
    assert classify("universe").tier == "allowed"
    placeheld = sorted(path for path, reason in REFUSED.items() if "not yet" in reason)
    assert not placeheld


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


@pytest.mark.parametrize(
    ("path", "direction", "wording"),
    [("research.min_trials", "raise_only", "raised"), ("data.budget_usd", "lower_only", "lowered")],
)
def test_classify_carries_the_legal_direction_for_a_clamped_path(path, direction, wording):
    """A clamped verdict says which way the knob may move, in the operator's words."""
    verdict = classify(path)
    assert verdict.tier == "clamped"
    assert verdict.direction == direction
    assert wording in verdict.reason


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


# ── apply: the seed universe — normalization + the starvation guard (#121) ───────────────
def _panel_size(settings) -> int:
    """The ready names the two-axis panel needs: the fit set plus the symbol holdout."""
    return settings.research.fit_set_size + settings.research.symbol_holdout_size


def _roster(count: int, first: str = "AAA") -> list[str]:
    """``count`` distinct, already-normalized tickers, the first one named."""
    return [first, *(f"SYM{i}" for i in range(count - 1))]


def test_an_overlaid_universe_is_upper_cased_and_deduped(tmp_path):
    """A mandate-set roster is normalized exactly as the mandate's own ``symbols:`` list is:
    upper-cased, stripped, de-duped, first-mention order preserved."""
    settings = _settings(tmp_path)

    lines = apply_patch(
        settings,
        {"universe": [" smr ", "CCJ", "smr", "leu", "ura", "nne", "oklo", "bwxt", "vst"]},
    )

    assert settings.universe == ["SMR", "CCJ", "LEU", "URA", "NNE", "OKLO", "BWXT", "VST"]
    # The echo is read back off the settings object, so an operator sees the normalized roster.
    assert lines == [f"universe={settings.universe}"]


def test_a_universe_exactly_at_the_panel_size_is_accepted(tmp_path):
    """The boundary is inclusive: a roster that exactly fills the fit set + the symbol holdout
    keeps both out-of-sample axes live, so it is legitimate steering."""
    settings = _settings(tmp_path)
    exact = _roster(_panel_size(settings))

    assert apply_patch(settings, {"universe": exact}) == [f"universe={exact}"]
    assert settings.universe == exact


def test_a_universe_below_the_panel_size_is_fatal_and_names_the_gate_it_would_disable(tmp_path):
    """A gate can be defeated by starving it as well as by lowering it: too few symbols and no
    scorecard ever carries a symbol-holdout metric, so the gate goes inert while every refused
    knob sits untouched. That is fatal, and the message says which gate."""
    settings = _settings(tmp_path)
    baseline = _settings(tmp_path).model_dump_json()
    starved = _roster(_panel_size(settings) - 1)

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {"universe": starved})

    message = str(exc.value)
    assert "universe" in message
    assert "symbol-holdout gate" in message  # the operator-meaningful "what breaks"
    assert settings.model_dump_json() == baseline  # nothing half-applied


def test_a_repeated_ticker_cannot_pad_the_universe_past_the_guard(tmp_path):
    """The guard counts after de-duplication — otherwise the normalization it ships beside
    would be the hole: ``[AAA] * 8`` is a one-name universe however it is spelled."""
    settings = _settings(tmp_path)
    padded = [*_roster(_panel_size(settings) - 1), "aaa"]  # AAA repeated, lower-cased
    assert len(padded) == _panel_size(settings)

    with pytest.raises(OverlayError, match="symbol-holdout gate"):
        apply_patch(settings, {"universe": padded})
    assert settings.universe == _settings(tmp_path).universe


def test_the_guard_measures_against_the_geometry_config_resolved(tmp_path):
    """The threshold is read off the settings being patched — and the fit-set/holdout sizes are
    refused, so the config-resolved geometry is the only bound there is. A config that runs a
    smaller panel legitimately accepts a smaller mandate roster."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("research:\n  fit_set_size: 3\n  symbol_holdout_size: 1\n", encoding="utf-8")
    settings = load_settings(config_path=cfg)
    four = _roster(4)

    assert apply_patch(settings, {"universe": four}) == [f"universe={four}"]
    assert settings.universe == four

    # The same four names under the shipped 6+2 geometry starve the holdout.
    default_settings = _settings(tmp_path)
    with pytest.raises(OverlayError, match="symbol-holdout gate"):
        apply_patch(default_settings, {"universe": four})


def test_the_holdout_geometry_cannot_be_shrunk_in_the_same_patch_to_clear_the_guard(tmp_path):
    """The obvious way around the guard is to shrink the panel it measures against — which is
    exactly what the refused holdout-geometry paths already forbid, so the attempt dies in the
    classification pass with the geometry refusal, and the roster never lands."""
    settings = _settings(tmp_path)

    with pytest.raises(OverlayError) as exc:
        apply_patch(
            settings,
            {
                "universe": ["AAA", "BBB"],
                "research.fit_set_size": 1,
                "research.symbol_holdout_size": 1,
            },
        )

    message = str(exc.value)
    assert REFUSED["research.fit_set_size"] in message
    assert settings.research.fit_set_size == 6
    assert settings.research.symbol_holdout_size == 2
    assert settings.universe == _settings(tmp_path).universe


def test_a_starvation_violation_is_raised_after_the_classification_pass(tmp_path):
    """Ordering, stated: the starvation guard is a **post-apply consistency check** — it reads
    the normalized, already-validated roster — so it cannot join the collect-all-then-raise-once
    refusal list, which is decided before any value is built. A patch carrying a refusal fails on
    the refusal and never reaches the guard."""
    settings = _settings(tmp_path)

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {"universe": ["AAA"], "promotion.max_gap": 5.0})

    message = str(exc.value)
    assert REFUSED["promotion.max_gap"] in message
    assert "1 config override refused" in message.splitlines()[0]
    assert "symbol-holdout gate" not in message


def test_config_yaml_may_set_a_universe_below_the_panel_size(tmp_path):
    """The guard constrains the overlay and nothing else (like the tier-B clamps): an operator
    running a deliberately tiny universe from ``config.yaml`` — a two-name pilot, say — is
    making that choice with their own file open, and widening the guard to config is a separate
    change."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("universe: [AAA, BBB]\n", encoding="utf-8")

    settings = load_settings(config_path=cfg)

    assert settings.universe == ["AAA", "BBB"]


# ── apply: the tier-B direction clamps (#118) ────────────────────────────────────────────
@pytest.mark.parametrize("path", sorted(CLAMP_SAMPLES))
def test_clamped_path_applies_in_the_direction_that_adds_discipline(tmp_path, path):
    """A clamped knob that passes its direction check is applied exactly like an allowed one:
    same section re-validation, same echo line, same untouched refused subtree."""
    settings = _settings(tmp_path)
    value = CLAMP_SAMPLES[path]["adds_discipline"]
    before = gate_snapshot(settings)

    lines = apply_patch(settings, {path: value})

    assert lines == [f"{path}={value}"]
    assert _read(settings, path) == value
    assert_gates_unmoved(before, gate_snapshot(settings))


@pytest.mark.parametrize("path", sorted(CLAMP_SAMPLES))
def test_clamped_path_is_fatal_in_the_direction_that_removes_discipline(tmp_path, path):
    settings = _settings(tmp_path)
    baseline = _settings(tmp_path).model_dump_json()
    sample = CLAMP_SAMPLES[path]

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {path: sample["removes_discipline"]})

    message = str(exc.value)
    assert path in message
    assert str(sample["configured"]) in message  # the config value it tried to cross
    assert settings.model_dump_json() == baseline


@pytest.mark.parametrize("path", sorted(CLAMP_SAMPLES))
def test_clamped_path_accepts_the_configured_value(tmp_path, path):
    """The boundary is inclusive: a mandate restating what config resolved is not a violation."""
    settings = _settings(tmp_path)
    configured = _read(settings, path)
    assert configured == CLAMP_SAMPLES[path]["configured"]

    assert apply_patch(settings, {path: configured}) == [f"{path}={configured}"]
    assert _read(settings, path) == configured


def test_raising_the_exhaustion_floor_applies_and_lowering_it_is_fatal(tmp_path):
    """The exhaustion floor is a count, not a metric: a tune-first personality demanding 20
    distinct param sets before any verdict is more discipline, and that is the only direction."""
    settings = _settings(tmp_path)
    assert settings.research.min_trials == 8

    assert apply_patch(settings, {"research.min_trials": 20}) == ["research.min_trials=20"]
    assert settings.research.min_trials == 20

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {"research.min_trials": 12})
    assert "20" in str(exc.value)  # the raised floor is what the second overlay must clear
    assert settings.research.min_trials == 20


def test_lowering_the_data_budget_applies_and_raising_it_is_fatal(tmp_path):
    """A mandate may spend less of the operator's vendor budget, never more than the ceiling
    config set."""
    settings = _settings(tmp_path)
    assert settings.data.budget_usd == 125.0

    assert apply_patch(settings, {"data.budget_usd": 40.0}) == ["data.budget_usd=40.0"]
    assert settings.data.budget_usd == 40.0

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {"data.budget_usd": 100.0})
    assert "40.0" in str(exc.value)  # already-lowered, so 100 is now a raise
    assert settings.data.budget_usd == 40.0


def test_a_wrong_direction_error_names_the_config_value_it_crossed(tmp_path):
    """The clamp compares against the value *config resolved*, not against the default — and
    the message names it, so the fix is obvious from the error alone."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("research:\n  min_trials: 17\ndata:\n  budget_usd: 33.5\n", encoding="utf-8")
    settings = load_settings(config_path=cfg)

    with pytest.raises(OverlayError) as exc:
        # 9 clears the default floor of 8 and 99.0 sits under the default cap of 125.0 —
        # both are still wrong-direction against what this config resolved.
        apply_patch(settings, {"research.min_trials": 9, "data.budget_usd": 99.0})

    message = str(exc.value)
    assert "17" in message
    assert "33.5" in message
    assert settings.research.min_trials == 17
    assert settings.data.budget_usd == 33.5


def test_config_yaml_may_set_either_clamped_knob_freely(tmp_path):
    """The clamp constrains the overlay and nothing else: ``config.yaml`` semantics are
    untouched, so an operator can still set a low floor and a large vendor budget there."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("research:\n  min_trials: 2\ndata:\n  budget_usd: 9999.0\n", encoding="utf-8")

    settings = load_settings(config_path=cfg)

    assert settings.research.min_trials == 2
    assert settings.data.budget_usd == 9999.0


def test_a_clamp_violation_and_a_refusal_are_reported_together(tmp_path):
    """One pass, not a fix-one-rerun loop: clamp violations join the same collect-all-then-
    raise-once path as refusals, and the legal key travelling with them applies nothing."""
    settings = _settings(tmp_path)
    patch = {
        "research.min_trials": 2,  # a wrong-direction clamp
        "promotion.max_gap": 5.0,  # a refusal
        "promotion.metric": "sortino",  # the one legal key
    }

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, patch)

    message = str(exc.value)
    assert "research.min_trials" in message
    assert "the configured 8" in message
    assert REFUSED["promotion.max_gap"] in message
    assert "2 config overrides refused" in message.splitlines()[0]
    assert settings.research.min_trials == 8
    assert settings.promotion.max_gap == 1.0
    assert settings.promotion.metric == "sharpe"


def test_the_clamped_tier_applies_beside_the_whole_allowed_surface(tmp_path):
    """The maximal legal overlay, tiers A and B at once: every value lands and the refused
    subtree stays byte-identical."""
    settings = _settings(tmp_path)
    before = json.dumps(gate_snapshot(settings), sort_keys=True, default=str)
    clamped = {path: sample["adds_discipline"] for path, sample in CLAMP_SAMPLES.items()}

    lines = apply_patch(settings, {**SAMPLE_VALUES, **clamped})

    assert len(lines) == len(SAMPLE_VALUES) + len(clamped)
    for path, value in {**SAMPLE_VALUES, **clamped}.items():
        assert _read(settings, path) == value
    assert json.dumps(gate_snapshot(settings), sort_keys=True, default=str) == before


def test_a_clamped_knob_may_not_be_unbounded_by_an_overlay(tmp_path):
    """``null`` means "no bound" — an unlimited vendor budget, no exhaustion floor at all —
    which is the least-disciplined end of either scale, so an overlay may never move there."""
    settings = _settings(tmp_path)

    with pytest.raises(OverlayError) as exc:
        apply_patch(settings, {"data.budget_usd": None, "research.min_trials": None})

    message = str(exc.value)
    assert "the configured 125.0" in message
    assert "the configured 8" in message
    assert settings.data.budget_usd == 125.0
    assert settings.research.min_trials == 8


def test_an_overlay_may_bound_a_budget_config_left_unbounded(tmp_path):
    """The mirror image: putting a number on an unbounded budget *adds* discipline, so it
    applies. (Raw attribute assignment is unvalidated — that is how a ``None`` gets in here
    while ``data.budget_usd`` is still typed ``float``; the clamp is written for the semantic,
    not for today's annotation.)"""
    settings = _settings(tmp_path)
    settings.data.budget_usd = None

    assert apply_patch(settings, {"data.budget_usd": 40.0}) == ["data.budget_usd=40.0"]
    assert settings.data.budget_usd == 40.0


def test_a_quoted_number_cannot_walk_past_the_clamp(tmp_path):
    """Front matter is YAML, so a value can arrive as a string that pydantic then coerces to a
    number on the way into the section. The clamp reads it the same way, or it would be a hole
    a quote character wide."""
    settings = _settings(tmp_path)

    with pytest.raises(OverlayError, match="research.min_trials"):
        apply_patch(settings, {"research.min_trials": "2"})
    assert settings.research.min_trials == 8

    assert apply_patch(settings, {"research.min_trials": "20"}) == ["research.min_trials=20"]
    assert settings.research.min_trials == 20


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
        (
            # A clamped path is validated like any other: a value the clamp cannot read as a
            # number is not a direction question, so pydantic delivers the honest diagnosis.
            "research.min_trials",
            "many",
            "research:\n  min_trials: many\n",
            "valid integer",
        ),
    ],
)
def test_a_bad_value_fails_identically_from_either_source(
    tmp_path, path, value, yaml_body, diagnosis
):
    """The applier re-validates the owning section, so the metric parser, ``Literal``
    membership, and type coercion all fire exactly as they do for the YAML file — surfacing
    as the one overlay exception type. Every path here is genuinely settable (allowed, or
    clamped and moving the legal way)."""
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


# ── the patch snapshot: the pre-value side of an overlay diff (#124) ─────────────────────
def test_patch_snapshot_reads_the_live_value_of_every_path_a_patch_names(tmp_path):
    """Taken either side of an apply, the two snapshots are the change, from what to what."""
    settings = _settings(tmp_path)
    patch = {"promotion.metric": "sortino", "research_time_budget_minutes": 17}

    before = patch_snapshot(settings, patch)
    apply_patch(settings, patch)
    after = patch_snapshot(settings, patch)

    assert before == {"promotion.metric": "sharpe", "research_time_budget_minutes": 60}
    assert after == {"promotion.metric": "sortino", "research_time_budget_minutes": 17}


def test_patch_snapshot_skips_a_key_that_names_no_setting(tmp_path):
    """A snapshot never raises first: an unknown key belongs to ``apply_patch``, which refuses
    it with the reason, and a snapshot that blew up would replace that diagnosis with a worse
    one."""
    settings = _settings(tmp_path)

    assert patch_snapshot(settings, {"promotion.metrik": "sortino"}) == {}
    with pytest.raises(OverlayError, match="not a setting"):
        apply_patch(settings, {"promotion.metrik": "sortino"})


def test_patch_snapshot_does_not_alias_mutable_values(tmp_path):
    settings = _settings(tmp_path)
    before = patch_snapshot(settings, {"universe": ["SMR"]})
    settings.universe.append("SMR")
    assert before["universe"] != settings.universe


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


def test_gate_snapshot_does_not_alias_mutable_values(tmp_path, monkeypatch):
    """A snapshot is deep-copied, so an in-place mutation of a list-valued setting can never
    make a snapshot agree with itself by aliasing.

    Driven against a path that is refused *for the duration of this test*: since #121 moved
    ``universe`` into tier A, no production refusal holds a mutable value — but the refusal
    table is meant to grow, so the property is checked, not assumed."""
    monkeypatch.setattr(
        overlay, "REFUSED", {**REFUSED, "universe": "refused for the duration of this test"}
    )
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
