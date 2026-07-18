"""The ``cost_profile`` knob (#12): the profile table is the single source of the Class-B
budgets, ``balanced`` reproduces today's ceilings, ``full`` only ever raises (never touches the
exhaustion floor or a gate), a free/local provider auto-selects ``full``, and an explicit
per-knob ``research.agent`` value pins its own budget. All pure — no lake, no client."""

from __future__ import annotations

import dataclasses

from noctis.config.settings import Settings
from noctis.research.cost import (
    BALANCED,
    ECONOMY,
    FULL,
    CostProfile,
    is_free_local,
    resolve_budgets,
    resolve_cost_profile,
)

# Class-B numeric/bool budgets (effort/prefix_trim handled separately — different types).
_NUMERIC = ("max_iterations", "max_backtests", "sweep_trials", "max_web_searches")


def _research(**kw):
    """A ResearchConfig with ``kw`` applied over the loaded config."""
    return Settings(research=kw).research


def test_balanced_is_default_and_reproduces_todays_ceilings():
    """Criterion 2: the shipped default is ``balanced`` == exactly today's ceilings (40 rounds /
    200 backtests / 20 sweep-trials / web search on, 8 / high effort / no prefix trim)."""
    b = resolve_budgets(Settings().research)
    assert b.name == "balanced"
    assert (b.max_iterations, b.max_backtests, b.sweep_trials) == (40, 200, 20)
    assert b.web_search is True and b.max_web_searches == 8
    assert b.effort == "high" and b.prefix_trim is False


def test_named_profiles_select_reduced_or_max_ceilings():
    """Criteria 1 + 5: switching the single ``cost_profile`` line picks the profile's ceilings."""
    econ = resolve_budgets(_research(model="openai/gpt-5.4", cost_profile="economy"))
    assert econ.name == "economy"
    assert (econ.max_iterations, econ.max_backtests, econ.sweep_trials) == (20, 80, 10)
    assert econ.max_web_searches == 4 and econ.effort == "medium" and econ.prefix_trim is True

    full = resolve_budgets(_research(model="openai/gpt-5.4", cost_profile="full"))
    assert full.name == "full"
    assert (full.max_iterations, full.max_backtests) == (40, 200)


def test_full_only_raises_never_lowers_a_floor_or_gate():
    """Criterion 3: ``full`` never sits below ``balanced``/``economy`` on any budget, and the
    profile table carries **no** quality lever — no ``min_trials``, no promotion gate — so it
    physically cannot lower the exhaustion floor or move a gate."""
    for field in _NUMERIC:
        assert getattr(FULL, field) >= getattr(BALANCED, field)
        assert getattr(FULL, field) >= getattr(ECONOMY, field)

    profile_fields = {f.name for f in dataclasses.fields(CostProfile)}
    forbidden = {
        "min_trials",
        "max_gap",
        "min_test_metric",
        "min_holdout_metric",
        "min_symbol_holdout_metric",
        "min_symbol_consistency",
        "min_test_activity",
    }
    assert profile_fields.isdisjoint(forbidden)

    # The exhaustion floor is read from research config, never from the profile — full leaves it.
    s = Settings(research={"cost_profile": "full"})
    assert s.research.min_trials == 8


def test_free_local_provider_auto_defaults_to_full():
    """Criterion 4: a $0/local provider resolves to ``full`` (the default balanced upgrades — a
    free model has no reason to run the paid ceilings); a paid provider keeps ``balanced``, and an
    explicit throttle on a free model is still honored."""
    assert resolve_cost_profile(_research(model="ollama/llama3")) is FULL  # auto-upgrade
    assert resolve_cost_profile(_research(model="openai/gpt-5.4")) is BALANCED  # paid default
    # An explicit throttle wins, even on a free model.
    econ = _research(model="ollama/llama3", cost_profile="economy")
    assert resolve_cost_profile(econ) is ECONOMY


def test_is_free_local_signal():
    assert is_free_local("ollama") and is_free_local("vllm") and is_free_local("lm_studio")
    assert not is_free_local("openai") and not is_free_local("anthropic")


def test_explicit_per_knob_override_pins_one_budget():
    """A pinned ``research.agent`` budget wins over its profile value; the rest still come from
    the profile — so an operator can tighten one knob without leaving the profile."""
    r = _research(model="openai/gpt-5.4", cost_profile="economy", agent={"max_backtests": 999})
    b = resolve_budgets(r)
    assert b.max_backtests == 999  # pinned override
    assert b.max_iterations == 20  # still the economy profile value
    assert b.name == "economy"
