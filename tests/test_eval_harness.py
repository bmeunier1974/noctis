"""``HarnessSpec`` (#190): the eval-only prompt-composition overlay, and its unreachability.

Two things are asserted here. First the type itself: the five prompt-composition ablations
(contract sheet, template, worked example, feasibility rules, retry hints) plus typed knob
overrides, whose defaults are the *production* composition — nobody ships with the contract sheet
off, so the unablated spec is what you get for free.

Second, and load-bearing: a harness spec is unreachable from production configuration. It is not a
field of any settings model (walked here from :class:`~noctis.config.settings.Settings`, the same
independent walk the overlay suite uses so a bug in the module under test cannot make the check
pass vacuously), and the mandate overlay — whose classification is total over the settings model
and refuses anything unknown — never classifies a harness path as anything a mandate may bind.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from noctis.config import load_settings
from noctis.config.overlay import ALLOWED, CLAMPED, REFUSED, OverlayError, apply_patch, classify
from noctis.config.settings import Settings
from noctis.eval.harness import AS_SHIPPED, HarnessSpec
from noctis.eval.knobs import SiteKnobs, UnknownKnob

ABLATIONS = ("contract_sheet", "template", "feasibility_rules", "retry_hints")


@dataclass(frozen=True)
class _ExampleKnobs(SiteKnobs):
    model: str = "ollama/qwen3"
    max_tokens: int = 2048


def _settings(tmp_path):
    """Settings from pure defaults (the repo config.yaml is bypassed by a missing path)."""
    return load_settings(config_path=tmp_path / "missing.yaml")


def _annotations(annotation: Any) -> list[Any]:
    """An annotation and every annotation nested inside it (``list[X]``, ``X | None``, …)."""
    return [annotation, *(nested for arg in get_args(annotation) for nested in _annotations(arg))]


def _settings_models(model: type[BaseModel]) -> set[type[BaseModel]]:
    """Every pydantic model reachable from ``model``, itself included."""
    found = {model}
    for field in model.model_fields.values():
        for annotation in _annotations(field.annotation):
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if annotation not in found:
                    found |= _settings_models(annotation)
    return found


# ── the production composition is the default ────────────────────────────────────────────
def test_the_default_harness_spec_is_the_production_prompt_composition() -> None:
    """Everything on, the worked example as production ships it, and no knob override."""
    spec = HarnessSpec()

    assert spec.contract_sheet is True
    assert spec.template is True
    assert spec.feasibility_rules is True
    assert spec.retry_hints is True
    assert spec.worked_example is AS_SHIPPED
    assert dict(spec.knob_overrides) == {}


def test_the_default_harness_spec_reports_itself_as_the_production_composition() -> None:
    assert HarnessSpec().is_production is True


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_a_spec_that_turns_one_prompt_component_off_is_not_the_production_composition(
    ablation: str,
) -> None:
    spec = HarnessSpec(**{ablation: False})

    assert getattr(spec, ablation) is False
    assert spec.is_production is False


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_each_prompt_component_can_be_ablated_without_disturbing_the_others(
    ablation: str,
) -> None:
    spec = HarnessSpec(**{ablation: False})

    still_on = [name for name in ABLATIONS if name != ablation]
    assert all(getattr(spec, name) is True for name in still_on)


def test_a_spec_may_select_a_named_worked_example_instead_of_the_shipped_one() -> None:
    spec = HarnessSpec(worked_example="donchian_breakout")

    assert spec.worked_example == "donchian_breakout"
    assert spec.is_production is False


def test_a_spec_may_drop_the_worked_example_entirely() -> None:
    spec = HarnessSpec(worked_example=None)

    assert spec.worked_example is None
    assert spec.is_production is False


# ── knob overrides ───────────────────────────────────────────────────────────────────────
def test_a_spec_carries_knob_overrides_for_the_site_it_targets() -> None:
    spec = HarnessSpec(knob_overrides={"max_tokens": 4096})

    assert dict(spec.knob_overrides) == {"max_tokens": 4096}
    assert spec.is_production is False


def test_knob_overrides_validate_against_the_knob_set_of_the_site_they_target() -> None:
    spec = HarnessSpec(knob_overrides={"model": "claude-sonnet-5"})

    assert spec.validate_knobs(_ExampleKnobs) is None


def test_an_override_of_a_knob_the_target_site_does_not_declare_is_refused() -> None:
    spec = HarnessSpec(knob_overrides={"temperature": 0.7})

    with pytest.raises(UnknownKnob) as error:
        spec.validate_knobs(_ExampleKnobs)

    assert "temperature" in str(error.value)


def test_a_harness_spec_cannot_be_reassigned_after_construction() -> None:
    spec = HarnessSpec()

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.contract_sheet = False  # type: ignore[misc]


def test_knob_overrides_cannot_be_mutated_through_the_mapping_the_caller_passed() -> None:
    overrides = {"max_tokens": 4096}
    spec = HarnessSpec(knob_overrides=overrides)

    overrides["max_tokens"] = 8

    assert dict(spec.knob_overrides) == {"max_tokens": 4096}


def test_knob_overrides_cannot_be_mutated_through_the_spec() -> None:
    spec = HarnessSpec(knob_overrides={"max_tokens": 4096})

    with pytest.raises(TypeError):
        spec.knob_overrides["max_tokens"] = 8  # type: ignore[index]


# ── unreachable from production configuration ────────────────────────────────────────────
def test_the_settings_walk_reaches_the_nested_config_sections() -> None:
    """Non-vacuity: the checks below only mean something if the walk reaches the real models."""
    names = {model.__name__ for model in _settings_models(Settings)}

    assert {"Settings", "ResearchConfig", "AgentResearchConfig", "PromotionConfig"} <= names


def test_no_settings_model_carries_a_harness_spec_field() -> None:
    """A benchmark ablation must not be settable in config: no settings model may hold one."""
    carriers = sorted(
        f"{model.__name__}.{name}"
        for model in _settings_models(Settings)
        for name, field in model.model_fields.items()
        if HarnessSpec in _annotations(field.annotation)
    )

    assert not carriers, f"settings fields typed as HarnessSpec: {carriers}"


def test_no_settings_leaf_is_a_harness_path() -> None:
    leaves = sorted(
        f"{model.__name__}.{name}"
        for model in _settings_models(Settings)
        for name in model.model_fields
        if "harness" in name
    )

    assert not leaves, f"settings fields naming a harness: {leaves}"


@pytest.mark.parametrize(
    "path", ["harness", "research.harness", "research.agent.harness_spec", "research.agent.harness"]
)
def test_the_mandate_overlay_classifies_a_harness_path_as_unknown(path: str) -> None:
    """Deny-by-default: a path the settings model does not have is unknown, and unknown refuses."""
    verdict = classify(path)

    assert verdict.tier == "unknown"
    assert verdict.reason


def test_no_classified_overlay_path_names_a_harness() -> None:
    classified = set(ALLOWED) | set(CLAMPED) | set(REFUSED)

    assert not sorted(path for path in classified if "harness" in path)


def test_a_mandate_that_tries_to_overlay_a_harness_path_is_refused(tmp_path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(OverlayError) as error:
        apply_patch(settings, {"research.agent.harness": {"contract_sheet": False}})

    assert "research.agent.harness" in str(error.value)
